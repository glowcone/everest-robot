"""Thin I/O wrappers around the RGB cameras and the maker-arm SDK.

Everything here talks to physical devices; keep logic out of this module so the adapter
can be tested with fakes implementing the same protocols.

The MakerMod arm is driven through the `maker-arm` SDK
(https://github.com/makermods-robotics/maker-arm-sdk): seven motors on CAN where motor 1
is the base and motor 7 is the gripper. `Arm.set_joint_targets` is non-blocking — a
200 Hz control loop approaches targets under a velocity limit — so commands here poll
feedback until the joints settle. The gripper is a position-controlled joint whose grip
force is kp x position error, so "close" commands a target past the carabiner and waits a
fixed dwell instead of expecting convergence.
"""

import math
import os
import time
from dataclasses import dataclass
from typing import Any, Protocol

import cv2
import numpy as np

from everest_robot.kinematics import JointAngles

N_MOTORS = 7
GRIPPER_INDEX = 6


class FrameSource(Protocol):
    def capture(self) -> np.ndarray: ...


class Arm(Protocol):
    def move_to_joints(self, joints: JointAngles) -> None: ...

    def open_gripper(self) -> None: ...

    def close_gripper(self) -> None: ...


class Camera:
    """An OpenCV camera that returns a fresh frame per capture."""

    def __init__(self, device_index: int) -> None:
        self._device_index = device_index
        self._capture: cv2.VideoCapture | None = None

    def capture(self) -> np.ndarray:
        if self._capture is None:
            self._capture = cv2.VideoCapture(self._device_index)
            if not self._capture.isOpened():
                raise RuntimeError(f"camera {self._device_index} failed to open")
        # Drain buffered frames so the image reflects the current scene.
        for _ in range(3):
            self._capture.grab()
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"camera {self._device_index} failed to return a frame")
        return frame


def _env_floats(name: str, default: str) -> tuple[float, ...]:
    return tuple(float(part) for part in os.getenv(name, default).split(","))


def _env_ints(name: str, default: str) -> tuple[int, ...]:
    return tuple(int(part) for part in os.getenv(name, default).split(","))


@dataclass(frozen=True)
class MakerArmMapping:
    """How IK joint angles map onto the seven maker-arm motors.

    For each IK joint (base_yaw, shoulder, elbow, wrist) the SDK target is
    `zero + sign * angle` at the mapped motor index. Unmapped motors hold the neutral
    pose. All values are bring-up calibration data.
    """

    ik_motor_indices: tuple[int, int, int, int]
    ik_signs: tuple[float, float, float, float]
    ik_zeros: tuple[float, float, float, float]
    neutral_joints: tuple[float, ...]
    gripper_open: float
    gripper_closed: float
    settle_tolerance: float
    settle_timeout: float
    gripper_dwell: float

    @classmethod
    def from_env(cls) -> "MakerArmMapping":
        mapping = cls(
            ik_motor_indices=_env_ints("ROBOT_IK_MOTOR_INDICES", "0,1,2,4"),
            ik_signs=_env_floats("ROBOT_IK_SIGNS", "1,1,1,1"),
            ik_zeros=_env_floats("ROBOT_IK_ZEROS", "0,0,5.9,2.1"),
            neutral_joints=_env_floats(
                "ROBOT_ARM_NEUTRAL_JOINTS", "0,0,5.9,0,2.1,3.6,-1.0"
            ),
            gripper_open=float(os.getenv("ROBOT_GRIPPER_OPEN_RAD", "-2.0")),
            gripper_closed=float(os.getenv("ROBOT_GRIPPER_CLOSED_RAD", "-0.1")),
            settle_tolerance=float(os.getenv("ROBOT_ARM_SETTLE_TOL_RAD", "0.05")),
            settle_timeout=float(os.getenv("ROBOT_ARM_SETTLE_TIMEOUT_S", "10")),
            gripper_dwell=float(os.getenv("ROBOT_GRIPPER_DWELL_S", "1.5")),
        )
        if len(mapping.neutral_joints) != N_MOTORS:
            raise ValueError("ROBOT_ARM_NEUTRAL_JOINTS must list all 7 joint values")
        if not all(0 <= i < GRIPPER_INDEX for i in mapping.ik_motor_indices):
            raise ValueError("ROBOT_IK_MOTOR_INDICES must map onto arm motors 0-5")
        return mapping


class MakerModArm:
    """Blocking joint/gripper commands on top of the maker-arm SDK `Arm`."""

    def __init__(self, sdk_arm: Any, mapping: MakerArmMapping) -> None:
        self._arm = sdk_arm
        self._mapping = mapping

    def _ensure_enabled(self) -> None:
        state = self._arm.state.value
        if state == "disconnected":
            self._arm.connect()
            state = self._arm.state.value
        if state == "connected":
            self._arm.enable()

    def _current_targets(self) -> list[float]:
        commanded = self._arm.get_commanded_positions()
        if all(math.isfinite(value) for value in commanded):
            return list(commanded)
        return list(self._mapping.neutral_joints)

    def _command(self, targets: list[float]) -> None:
        if not self._arm.set_joint_targets(targets):
            raise RuntimeError(f"arm rejected joint targets (state={self._arm.state.value})")

    def _wait_settled(self, targets: list[float], indices: list[int]) -> None:
        deadline = time.monotonic() + self._mapping.settle_timeout
        while time.monotonic() < deadline:
            positions = self._arm.get_joint_positions()
            if all(
                abs(positions[i] - targets[i]) <= self._mapping.settle_tolerance
                for i in indices
            ):
                return
            time.sleep(0.02)
        raise RuntimeError(
            f"arm did not settle within {self._mapping.settle_timeout}s "
            f"(targets={targets}, positions={self._arm.get_joint_positions()})"
        )

    def move_to_joints(self, joints: JointAngles) -> None:
        self._ensure_enabled()
        mapping = self._mapping
        targets = self._current_targets()
        angles = (joints.base_yaw, joints.shoulder, joints.elbow, joints.wrist)
        for motor_index, sign, zero, angle in zip(
            mapping.ik_motor_indices, mapping.ik_signs, mapping.ik_zeros, angles, strict=True
        ):
            targets[motor_index] = zero + sign * angle
        self._command(targets)
        self._wait_settled(targets, list(mapping.ik_motor_indices))

    def open_gripper(self) -> None:
        self._ensure_enabled()
        targets = self._current_targets()
        targets[GRIPPER_INDEX] = self._mapping.gripper_open
        self._command(targets)
        self._wait_settled(targets, [GRIPPER_INDEX])

    def close_gripper(self) -> None:
        self._ensure_enabled()
        targets = self._current_targets()
        targets[GRIPPER_INDEX] = self._mapping.gripper_closed
        self._command(targets)
        # The gripper stalls on the carabiner by design (compliant force limiting),
        # so wait a fixed dwell instead of expecting position convergence.
        time.sleep(self._mapping.gripper_dwell)


def connect_maker_mod_arm() -> MakerModArm:
    """Build a MakerModArm from environment configuration.

    ROBOT_ARM_BACKEND selects `slcan` (macOS, needs ROBOT_ARM_PORT) or `socketcan`
    (Linux, needs ROBOT_ARM_CHANNEL). ROBOT_ARM_PROFILE overrides the SDK's shipped
    maker_arm_v1 profile.
    """

    from maker_arm import Arm as SdkArm
    from maker_arm.profiles import DEFAULT_ARM_CONFIG

    backend = os.getenv("ROBOT_ARM_BACKEND", "slcan")
    backend_kwargs: dict[str, Any] = {}
    if backend == "slcan":
        port = os.getenv("ROBOT_ARM_PORT")
        if not port:
            raise ValueError("ROBOT_ARM_PORT is required for the slcan backend")
        backend_kwargs["port"] = port
    elif backend == "socketcan":
        backend_kwargs["channel"] = os.getenv("ROBOT_ARM_CHANNEL", "can0")
    profile = os.getenv("ROBOT_ARM_PROFILE", str(DEFAULT_ARM_CONFIG))
    sdk_arm = SdkArm.from_yaml(profile, backend=backend, **backend_kwargs)
    return MakerModArm(sdk_arm, MakerArmMapping.from_env())
