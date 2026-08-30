""":class:`~everest_robot.robot.ports.ArmPort` over LeRobot's ``RobstrideMotorsBus``.

This is the **MIT-protocol** driver, for arms whose RobStride motors are provisioned in MIT
mode (the makermodslab teleop convention). It exists so the lease-local calibration
teleoperation monitor (``robot-monitor`` / ``just monitor``) can run against such an arm
without the operator first switching every motor back to the private protocol.

**Scope.** docs/adr/0001-production-motor-protocol.md rejected ``RobstrideMotorsBus`` as
the production safety boundary, and that decision stands (see the ADR's addendum). This
port is qualified for the calibration monitor only; replay and the durable workflow remain
on :class:`~everest_robot.robot.maker_arm_port.MakerArmPort`. Known, accepted gaps versus
maker-arm-sdk, per the ADR:

* no motor-side CAN watchdog (a dead host leaves the last MIT command in force),
* no host-side feedback-age watchdog,
* no fault-hold (a fault surfaces as an exception; torque is not actively held).

The teleoperation loop mitigates these at its level: it calls
:meth:`hold_current_position` on any failure and torque is released on disconnect.

**Frame.** The bus speaks degrees in the LeRobot zero pose; the port contract is radians in
maker-arm calibrated coordinates. Conversion goes through the
:class:`~everest_robot.robot.lerobot_bridge.JointFrame` built from the parameters file's
``lerobot_frame`` offsets. Those offsets are the most consequential values in this path and
are not yet hardware-verified -- see docs/lerobot-frame-reconciliation.md. An identity
frame is refused outright: it would command a wrong pose.

``lerobot`` is imported lazily so the rest of the runtime installs and tests without the
``hardware`` extra.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

from everest_robot.robot.contracts import (
    ArmLifecycle,
    JointLimit,
    JointState,
    RobotIdentity,
)
from everest_robot.robot.lerobot_bridge import JointFrame
from everest_robot.robot.maker_arm_port import HardwareUnavailableError
from everest_robot.robot.ports import clip_to_limits

# Mirror MakerFollower's per-session full-turn correction: a RobStride motor can come back
# from a power cycle reporting its angle a whole turn off. Values match maker_follower.py's
# _FULL_TURN_DEG / _WRAP_GRACE_DEG (duplicated here so the hardware-free tests need no
# lerobot import; from_deployment() reuses the fork's tables for everything else).
FULL_TURN_DEG = 360.0
WRAP_GRACE_DEG = 20.0


def load_lerobot_mit() -> tuple[Any, Any, Any, Any, Any]:
    """Import the lerobot fork's MIT stack on demand, with an actionable message."""

    try:
        from lerobot.motors import Motor, MotorNormMode
        from lerobot.motors.robstride import RobstrideMotorsBus
        from lerobot.robots.maker_follower.config_maker_follower import MakerFollowerConfigBase
        from lerobot.robots.maker_follower.maker_follower import MOTOR_MODELS
    except ImportError as error:  # pragma: no cover - exercised only without the extra
        raise HardwareUnavailableError(
            "lerobot is not installed; install the hardware extra "
            "(`uv sync --extra hardware`) to talk to a real arm"
        ) from error
    return Motor, MotorNormMode, RobstrideMotorsBus, MakerFollowerConfigBase, MOTOR_MODELS


class RobstrideMitPort:
    """Adapts ``RobstrideMotorsBus`` to the Everest port.

    Takes an already-constructed bus so tests can supply a stub; use
    :meth:`from_deployment` for the ordinary deployment path. Unlike ``MakerFollower``,
    the port drives the bus itself so that CONNECTED (torque off, read-only monitoring
    works) and ENABLED (torque on) stay distinct states: ``MakerFollower.connect()``
    enables torque as a side effect, which the monitor's ``--read-only`` mode must never
    do.
    """

    def __init__(
        self,
        bus: Any,
        identity: RobotIdentity,
        frame: JointFrame,
        *,
        limits_deg: Mapping[str, tuple[float, float]],
        gains: Mapping[str, tuple[float, float]],
    ) -> None:
        if frame.joint_names != identity.joint_names:
            raise ValueError("the joint frame must describe the arm's joints, in the same order")
        missing = [
            name
            for name in identity.joint_names
            if name not in limits_deg or name not in gains
        ]
        if missing:
            raise ValueError(
                f"no MIT limits/gains for joint(s) {', '.join(missing)}; the parameters "
                "file and the MakerFollower tables are describing different arms"
            )
        self._bus = bus
        self._identity = identity
        self._frame = frame
        self._limits_deg = {name: limits_deg[name] for name in identity.joint_names}
        self._gains = {name: gains[name] for name in identity.joint_names}
        self._lifecycle = ArmLifecycle.DISCONNECTED
        self._fault_reason: str | None = None
        self._turn_offsets_deg = dict.fromkeys(identity.joint_names, 0.0)
        self._sequence = dict.fromkeys(identity.joint_names, 0)
        self._feedback_seen: dict[str, float | None] = dict.fromkeys(identity.joint_names)
        self._last_positions_rad: tuple[float, ...] | None = None
        # The last validated command, kept so a caller can see what was clipped.
        self.last_command = None

    @classmethod
    def from_deployment(
        cls,
        identity: RobotIdentity,
        frame: JointFrame,
        *,
        port: str,
        backend: str,
        bitrate: int = 1_000_000,
    ) -> RobstrideMitPort:
        """Build a port over a real CAN bus, reusing the fork's motor tables.

        Refuses an identity frame before touching any hardware dependency: without the
        parameters file's ``lerobot_frame`` offsets, MIT degrees and calibrated radians
        would be conflated and the first command would drive a wrong pose.
        """

        if frame.is_identity:
            raise ValueError(
                "the MIT driver needs the parameters file's lerobot_frame offsets: an "
                "identity frame asserts the MIT zero pose and the maker-arm calibrated "
                "zero pose coincide, which they do not on this arm. Add/restore the "
                "lerobot_frame section (see docs/lerobot-frame-reconciliation.md)"
            )
        if backend not in ("socketcan", "slcan"):
            raise ValueError(
                f"unsupported CAN backend {backend!r} for the MIT driver "
                "(expected socketcan or slcan)"
            )

        motor_class, norm_mode, bus_class, config_class, motor_models = load_lerobot_mit()
        config = config_class()
        unknown = [
            name
            for name in identity.joint_names
            if name not in config.motor_can_ids or name not in motor_models
        ]
        if unknown:
            raise ValueError(
                f"joint(s) {', '.join(unknown)} are not in MakerFollower's motor tables; "
                "the parameters file is describing a different arm"
            )

        motors: dict[str, Any] = {}
        for name in identity.joint_names:
            can_id = config.motor_can_ids[name]
            motor = motor_class(can_id, motor_models[name], norm_mode.DEGREES)
            # RobStride MIT feedback carries the motor id in payload byte 0.
            motor.recv_id = can_id
            motor.motor_type_str = motor_models[name]
            motors[name] = motor

        bus = bus_class(
            port=port,
            motors=motors,
            calibration=None,
            can_interface=backend,
            use_can_fd=False,
            bitrate=bitrate,
            data_bitrate=None,
        )
        return cls(
            bus,
            identity,
            frame,
            limits_deg={name: config.joint_limits[name] for name in identity.joint_names},
            gains={name: config.gains[name] for name in identity.joint_names},
        )

    # ── identity and limits ────────────────────────────────────────────────────────
    @property
    def identity(self) -> RobotIdentity:
        return self._identity

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self._identity.joint_names

    @property
    def lifecycle(self) -> ArmLifecycle:
        return self._lifecycle

    def limits(self) -> tuple[JointLimit, ...]:
        """MakerFollower's degree limits, expressed in calibrated joint radians."""

        lows = self._frame.to_radians(
            [self._limits_deg[name][0] for name in self.joint_names]
        )
        highs = self._frame.to_radians(
            [self._limits_deg[name][1] for name in self.joint_names]
        )
        return tuple(
            JointLimit(name=name, lower_rad=low, upper_rad=high)
            for name, low, high in zip(self.joint_names, lows, highs, strict=True)
        )

    # ── lifecycle ──────────────────────────────────────────────────────────────────
    def connect(self, timeout: float = 2.0) -> None:
        # The bus's handshake carries its own per-motor timeout; ``timeout`` is part of
        # the port contract but has no finer-grained hook here.
        self._bus.connect(handshake=True)
        self._lifecycle = ArmLifecycle.CONNECTED
        self._fault_reason = None

    def disconnect(self) -> None:
        if self._lifecycle is ArmLifecycle.DISCONNECTED:
            return
        try:
            self._bus.disconnect(disable_torque=True)
        finally:
            self._lifecycle = ArmLifecycle.DISCONNECTED
            self._turn_offsets_deg = dict.fromkeys(self.joint_names, 0.0)
            self._last_positions_rad = None

    def enable(self) -> None:
        """Write gains, reconcile whole-turn encoder wraps, gate on limits, arm.

        Mirrors ``MakerFollower._detect_full_turn_offsets`` semantics, except that a
        joint outside its limits by other than a whole turn refuses to enable rather
        than deferring the refusal to the first command: for a calibration monitor an
        arm that must not move should never be put under torque at all.
        """

        if self._lifecycle is not ArmLifecycle.CONNECTED:
            raise RuntimeError(f"cannot enable from {self._lifecycle}")

        # Gains ride along in every MIT frame; the bus default kp=10 is far too soft to
        # hold the arm against gravity, so they must be stored before the first command.
        self._bus.sync_write("Kp", {name: kp for name, (kp, _) in self._gains.items()})
        self._bus.sync_write("Kd", {name: kd for name, (_, kd) in self._gains.items()})

        problems: list[str] = []
        offsets = dict.fromkeys(self.joint_names, 0.0)
        for name in self.joint_names:
            raw = float(self._bus.read("Present_Position", name))
            low, high = self._limits_deg[name]
            low, high = low - WRAP_GRACE_DEG, high + WRAP_GRACE_DEG
            if low <= raw <= high:
                continue
            for shift in (-FULL_TURN_DEG, FULL_TURN_DEG):
                if low <= raw + shift <= high:
                    offsets[name] = shift
                    break
            else:
                problems.append(
                    f"{name} reads {raw:+.1f} deg, outside its soft limits "
                    f"{self._limits_deg[name]} and not by a whole turn"
                )
        if problems:
            raise RuntimeError(
                "refusing to enable: " + "; ".join(problems) + ". The motor zero no "
                "longer matches the calibration pose; re-zero the arm before enabling."
            )

        self._turn_offsets_deg = offsets
        self._bus.enable_torque()
        self._lifecycle = ArmLifecycle.ENABLED

    def disable(self) -> None:
        if self._lifecycle is ArmLifecycle.ENABLED:
            self._bus.disable_torque()
            self._lifecycle = ArmLifecycle.CONNECTED

    def estop(self) -> None:
        if self._lifecycle is ArmLifecycle.DISCONNECTED:
            return
        try:
            self._bus.disable_torque()
        finally:
            if self._lifecycle is ArmLifecycle.ENABLED:
                self._lifecycle = ArmLifecycle.CONNECTED

    def clear_faults(self) -> None:
        if self._lifecycle is not ArmLifecycle.FAULT:
            return
        # update_motor_state re-queries via CLEAR_FAULT and raises if a motor still
        # reports a fault, so reaching the end of the loop is the all-clear.
        for name in self.joint_names:
            self._bus.update_motor_state(name)
        self._fault_reason = None
        self._lifecycle = ArmLifecycle.CONNECTED

    def hold_current_position(self) -> bool:
        if self._lifecycle is not ArmLifecycle.ENABLED:
            return False
        positions = self._last_positions_rad
        if positions is None or not all(math.isfinite(value) for value in positions):
            return False
        return self.send_targets(positions)

    # ── feedback and commands ──────────────────────────────────────────────────────
    def read_state(self) -> JointState:
        """One feedback sample.

        The bus caches per-motor state for ~20 ms and refreshes lazily on read, so this
        is one request/reply per stale motor. A motor that reports a fault moves the
        port to ``FAULT``; a motor that does not reply yields ``nan`` for this sample,
        which is the contract's "no feedback" value, not a fault.
        """

        degrees: list[float] = []
        velocities: list[float] = []
        torques: list[float] = []
        temperatures: list[float] = []
        fault_bits: list[int] = []
        for name in self.joint_names:
            try:
                position = float(self._bus.read("Present_Position", name))
                velocity = float(self._bus.read("Present_Velocity", name))
                torque = float(self._bus.read("Present_Torque", name))
                temperature = float(self._bus.read("Temperature_MOS", name))
                fault = 0
            except RuntimeError as error:
                self._lifecycle = ArmLifecycle.FAULT
                self._fault_reason = f"{name}: {error}"
                position = velocity = torque = temperature = math.nan
                fault = 1
            except ConnectionError:
                position = velocity = torque = temperature = math.nan
                fault = 0
            degrees.append(position + self._turn_offsets_deg[name])
            velocities.append(math.radians(velocity))
            torques.append(torque)
            temperatures.append(temperature)
            fault_bits.append(fault)

        sequence: list[int] = []
        for name in self.joint_names:
            fed_back = self._bus.last_feedback_time.get(name)
            if fed_back is not None and fed_back != self._feedback_seen[name]:
                self._sequence[name] += 1
                self._feedback_seen[name] = fed_back
            sequence.append(self._sequence[name])

        positions = self._frame.to_radians(degrees)
        self._last_positions_rad = positions
        return JointState(
            names=self.joint_names,
            positions=positions,
            velocities=tuple(velocities),
            torques=tuple(torques),
            temperatures=tuple(temperatures),
            fault_bits=tuple(fault_bits),
            sequence=tuple(sequence),
            monotonic_s=time.monotonic(),
            lifecycle=self._lifecycle,
            fault_reason=self._fault_reason,
        )

    def send_targets(self, targets: Sequence[float]) -> bool:
        if self._lifecycle is not ArmLifecycle.ENABLED:
            return False
        values = [float(target) for target in targets]
        if len(values) != len(self.joint_names) or not all(
            math.isfinite(value) for value in values
        ):
            return False
        command = clip_to_limits(values, self.limits())
        self.last_command = command
        degrees = self._frame.to_degrees(command.targets)
        self._bus.sync_write(
            "Goal_Position",
            {
                name: value - self._turn_offsets_deg[name]
                for name, value in zip(self.joint_names, degrees, strict=True)
            },
        )
        return True
