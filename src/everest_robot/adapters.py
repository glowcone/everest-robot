"""Replaceable hardware and perception boundaries.

`ScaffoldRobot` keeps deterministic placeholders for the stages that are not yet
implemented. `MakerModRobot` implements carabiner localization and pickup for the real
hardware: RGB-only detection on the overhead camera, homography projection onto the table
plane, an analytic-IK top-down grasp, and a wrist-camera grasp check.
"""

import os
from dataclasses import dataclass, field
from typing import Any

from everest_robot.calibration import load_homography, pixel_to_table
from everest_robot.domain import (
    AttachmentResult,
    CarabinerPickupResult,
    PositionResult,
    RecoveryTarget,
    VerificationResult,
)
from everest_robot.kinematics import ArmGeometry, solve_ik
from everest_robot.vision import (
    GripperCheckConfig,
    HsvDetectionConfig,
    carabiner_in_gripper,
    detect_carabiner,
)


@dataclass
class ScaffoldRobot:
    """A deterministic stand-in for the future robot integrations."""

    verification_failures: int = 0
    verification_calls: int = 0

    def localize_and_pick_up_carabiner(
        self,
        detector: str,
        grasp_planner: str,
    ) -> CarabinerPickupResult:
        return CarabinerPickupResult(
            secured=True,
            frame="robot_base",
            x=0.42,
            y=0.0,
            z=0.18,
            detector=detector,
            grasp_planner=grasp_planner,
        )

    def go_to_known_position(self, position_name: str) -> PositionResult:
        return PositionResult(reached=True, position_name=position_name)

    def attach_clip(self, controller: str) -> AttachmentResult:
        return AttachmentResult(
            motion_completed=True,
            force_newtons=8.0,
            controller=controller,
        )

    def verify_attachment(self, attachment: AttachmentResult) -> VerificationResult:
        del attachment
        self.verification_calls += 1
        if self.verification_calls <= self.verification_failures:
            return VerificationResult(
                secure=False,
                confidence=0.4,
                recovery_target=RecoveryTarget.ATTACH,
                reason="scaffolded verification failure",
            )
        return VerificationResult(secure=True, confidence=0.99)


def _env_floats(name: str, default: str) -> tuple[float, ...]:
    return tuple(float(part) for part in os.getenv(name, default).split(","))


def _env_ints(name: str, default: str) -> tuple[int, ...]:
    return tuple(int(part) for part in os.getenv(name, default).split(","))


@dataclass(frozen=True)
class MakerModConfig:
    """Environment-backed tuning for the MakerMod pickup pipeline."""

    homography_path: str
    table_z: float
    grasp_clearance: float
    detection: HsvDetectionConfig
    gripper_check: GripperCheckConfig
    arm_geometry: ArmGeometry

    @classmethod
    def from_env(cls) -> "MakerModConfig":
        hsv_lower = _env_ints("ROBOT_HSV_LOWER", "35,80,80")
        hsv_upper = _env_ints("ROBOT_HSV_UPPER", "85,255,255")
        return cls(
            homography_path=os.getenv("ROBOT_HOMOGRAPHY_PATH", "config/homography.json"),
            table_z=float(os.getenv("ROBOT_TABLE_Z_M", "0.02")),
            grasp_clearance=float(os.getenv("ROBOT_GRASP_CLEARANCE_M", "0.08")),
            detection=HsvDetectionConfig(
                hsv_lower=hsv_lower,
                hsv_upper=hsv_upper,
                min_contour_area=float(os.getenv("ROBOT_MIN_CONTOUR_AREA", "400")),
            ),
            gripper_check=GripperCheckConfig(
                hsv_lower=hsv_lower,
                hsv_upper=hsv_upper,
                roi=_env_floats("ROBOT_GRIPPER_ROI", "0.3,0.4,0.4,0.4"),
                min_mask_pixels=int(os.getenv("ROBOT_GRIPPER_MASK_MIN_PIXELS", "200")),
            ),
            arm_geometry=ArmGeometry(
                base_height=float(os.getenv("ROBOT_ARM_BASE_HEIGHT_M", "0.10")),
                upper_link=float(os.getenv("ROBOT_ARM_UPPER_LINK_M", "0.12")),
                forearm_link=float(os.getenv("ROBOT_ARM_FOREARM_LINK_M", "0.12")),
                gripper_length=float(os.getenv("ROBOT_ARM_GRIPPER_LENGTH_M", "0.09")),
            ),
        )


@dataclass
class MakerModRobot:
    """RGB-only carabiner pickup on the MakerMod arm.

    Stages 2-4 still fall back to the scaffold behaviour until their real
    implementations land; only localization/pickup is hardware-backed.
    """

    overhead_camera: Any
    wrist_camera: Any
    arm: Any
    config: MakerModConfig
    verification_failures: int = 0
    _scaffold: ScaffoldRobot = field(init=False)

    def __post_init__(self) -> None:
        self._scaffold = ScaffoldRobot(verification_failures=self.verification_failures)

    def localize_and_pick_up_carabiner(
        self,
        detector: str,
        grasp_planner: str,
    ) -> CarabinerPickupResult:
        del detector, grasp_planner  # this adapter fixes the pipeline it reports

        def result(secured: bool, x: float = 0.0, y: float = 0.0) -> CarabinerPickupResult:
            return CarabinerPickupResult(
                secured=secured,
                frame="robot_base",
                x=x,
                y=y,
                z=self.config.table_z,
                detector="hsv-contour",
                grasp_planner="ik-topdown",
            )

        # Idempotency guard: a retried checkpoint must not re-grasp a held carabiner.
        if carabiner_in_gripper(self.wrist_camera.capture(), self.config.gripper_check):
            return result(secured=True)

        detection = detect_carabiner(self.overhead_camera.capture(), self.config.detection)
        if detection is None:
            return result(secured=False)

        homography = load_homography(self.config.homography_path)
        x, y = pixel_to_table(homography, detection.pixel_x, detection.pixel_y)
        geometry = self.config.arm_geometry
        grasp_z = self.config.table_z
        clear_z = grasp_z + self.config.grasp_clearance

        self.arm.move_to_joints(solve_ik(x, y, clear_z, geometry))
        self.arm.open_gripper()
        self.arm.move_to_joints(solve_ik(x, y, grasp_z, geometry))
        self.arm.close_gripper()
        self.arm.move_to_joints(solve_ik(x, y, clear_z, geometry))

        secured = carabiner_in_gripper(self.wrist_camera.capture(), self.config.gripper_check)
        return result(secured=secured, x=x, y=y)

    def go_to_known_position(self, position_name: str) -> PositionResult:
        return self._scaffold.go_to_known_position(position_name)

    def attach_clip(self, controller: str) -> AttachmentResult:
        return self._scaffold.attach_clip(controller)

    def verify_attachment(self, attachment: AttachmentResult) -> VerificationResult:
        return self._scaffold.verify_attachment(attachment)


def create_robot(params: dict[str, Any]) -> ScaffoldRobot | MakerModRobot:
    """Build the robot adapter selected by the task's `robot` parameter."""

    robot_kind = str(params.get("robot", "scaffold"))
    verification_failures = int(params.get("verification_failures", 0))
    if robot_kind == "scaffold":
        return ScaffoldRobot(verification_failures=verification_failures)
    if robot_kind == "makermod":
        from everest_robot.hardware import Camera, connect_maker_mod_arm

        config = MakerModConfig.from_env()
        return MakerModRobot(
            overhead_camera=Camera(int(os.getenv("ROBOT_OVERHEAD_CAMERA_INDEX", "0"))),
            wrist_camera=Camera(int(os.getenv("ROBOT_WRIST_CAMERA_INDEX", "1"))),
            arm=connect_maker_mod_arm(),
            config=config,
            verification_failures=verification_failures,
        )
    raise ValueError(f"unknown robot adapter: {robot_kind!r}")
