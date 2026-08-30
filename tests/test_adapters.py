import math

import cv2
import numpy as np
import pytest

from everest_robot.adapters import MakerModConfig, MakerModRobot, ScaffoldRobot, create_robot
from everest_robot.calibration import save_homography
from everest_robot.domain import CarabinerPickupResult, RecoveryTarget
from everest_robot.kinematics import ArmGeometry, forward_kinematics
from everest_robot.vision import GripperCheckConfig, HsvDetectionConfig


def test_scaffold_robot_is_deterministic() -> None:
    robot = ScaffoldRobot()
    pickup = robot.localize_and_pick_up_carabiner("deterministic-cv", "graspnet")
    position = robot.go_to_known_position("clip-attachment-ready")
    attachment = robot.attach_clip("vla")
    verification = robot.verify_attachment(attachment)

    assert pickup == CarabinerPickupResult(
        True,
        "robot_base",
        0.42,
        0.0,
        0.18,
        "deterministic-cv",
        "graspnet",
    )
    assert position.reached
    assert attachment.motion_completed
    assert attachment.controller == "vla"
    assert verification.secure


def test_verification_can_request_attachment_recovery() -> None:
    robot = ScaffoldRobot(verification_failures=1)
    attachment = robot.attach_clip("rl-policy")

    first = robot.verify_attachment(attachment)
    second = robot.verify_attachment(attachment)

    assert first.recovery_target is RecoveryTarget.ATTACH
    assert not first.secure
    assert second.secure


GREEN_BGR = (60, 200, 60)
HSV_LOWER = (35, 80, 80)
HSV_UPPER = (85, 255, 255)
# Maps overhead pixels to table meters at 0.5 mm/px, so pixel (320, 240) -> (0.16, 0.12).
PIXEL_SCALE = 0.0005


def frame_with_blob(center: tuple[float, float] | None) -> np.ndarray:
    frame = np.full((480, 640, 3), (255, 255, 255), dtype=np.uint8)
    if center is not None:
        box = cv2.boxPoints((center, (120.0, 40.0), 0.0))
        cv2.fillConvexPoly(frame, box.astype(np.int32), GREEN_BGR)
    return frame


class FakeCamera:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = list(frames)
        self.captures = 0

    def capture(self) -> np.ndarray:
        self.captures += 1
        return self.frames.pop(0)


class FakeArm:
    def __init__(self) -> None:
        self.commands: list = []

    def move_to_joints(self, joints) -> None:
        self.commands.append(("move", joints))

    def open_gripper(self) -> None:
        self.commands.append(("open", None))

    def close_gripper(self) -> None:
        self.commands.append(("close", None))


@pytest.fixture
def makermod_config(tmp_path) -> MakerModConfig:
    homography = np.array(
        [[PIXEL_SCALE, 0.0, 0.0], [0.0, PIXEL_SCALE, 0.0], [0.0, 0.0, 1.0]]
    )
    path = tmp_path / "homography.json"
    save_homography(homography, path)
    return MakerModConfig(
        homography_path=str(path),
        table_z=0.02,
        grasp_clearance=0.08,
        detection=HsvDetectionConfig(
            hsv_lower=HSV_LOWER, hsv_upper=HSV_UPPER, min_contour_area=400
        ),
        gripper_check=GripperCheckConfig(
            hsv_lower=HSV_LOWER,
            hsv_upper=HSV_UPPER,
            roi=(0.25, 0.25, 0.5, 0.5),
            min_mask_pixels=200,
        ),
        arm_geometry=ArmGeometry(
            base_height=0.10, upper_link=0.12, forearm_link=0.12, gripper_length=0.09
        ),
    )


def make_robot(
    config: MakerModConfig,
    wrist_frames: list[np.ndarray],
    overhead_frames: list[np.ndarray],
) -> tuple[MakerModRobot, FakeArm]:
    arm = FakeArm()
    robot = MakerModRobot(
        overhead_camera=FakeCamera(overhead_frames),
        wrist_camera=FakeCamera(wrist_frames),
        arm=arm,
        config=config,
    )
    return robot, arm


def test_makermod_pickup_happy_path(makermod_config) -> None:
    robot, arm = make_robot(
        makermod_config,
        wrist_frames=[frame_with_blob(None), frame_with_blob((320.0, 240.0))],
        overhead_frames=[frame_with_blob((320.0, 240.0))],
    )

    result = robot.localize_and_pick_up_carabiner("hsv-contour", "ik-topdown")

    assert result.secured
    assert result.frame == "robot_base"
    assert abs(result.x - 0.16) < 0.005
    assert abs(result.y - 0.12) < 0.005
    assert result.z == makermod_config.table_z
    assert result.detector == "hsv-contour"
    assert result.grasp_planner == "ik-topdown"

    assert [kind for kind, _ in arm.commands] == ["move", "open", "move", "close", "move"]
    geometry = makermod_config.arm_geometry
    clear_z = makermod_config.table_z + makermod_config.grasp_clearance
    for command_index, expected_z in ((0, clear_z), (2, makermod_config.table_z), (4, clear_z)):
        x, y, z = forward_kinematics(arm.commands[command_index][1], geometry)
        assert math.hypot(x - result.x, y - result.y) < 1e-3
        assert abs(z - expected_z) < 1e-3


def test_makermod_pickup_is_idempotent_when_already_holding(makermod_config) -> None:
    robot, arm = make_robot(
        makermod_config,
        wrist_frames=[frame_with_blob((320.0, 240.0))],
        overhead_frames=[],
    )

    result = robot.localize_and_pick_up_carabiner("hsv-contour", "ik-topdown")

    assert result.secured
    assert arm.commands == []


def test_makermod_pickup_reports_failure_when_nothing_detected(makermod_config) -> None:
    robot, arm = make_robot(
        makermod_config,
        wrist_frames=[frame_with_blob(None)],
        overhead_frames=[frame_with_blob(None)],
    )

    result = robot.localize_and_pick_up_carabiner("hsv-contour", "ik-topdown")

    assert not result.secured
    assert arm.commands == []


def test_makermod_pickup_reports_failed_grasp(makermod_config) -> None:
    robot, arm = make_robot(
        makermod_config,
        wrist_frames=[frame_with_blob(None), frame_with_blob(None)],
        overhead_frames=[frame_with_blob((320.0, 240.0))],
    )

    result = robot.localize_and_pick_up_carabiner("hsv-contour", "ik-topdown")

    assert not result.secured
    assert [kind for kind, _ in arm.commands] == ["move", "open", "move", "close", "move"]


def test_create_robot_defaults_to_scaffold() -> None:
    robot = create_robot({"verification_failures": 2})
    assert isinstance(robot, ScaffoldRobot)
    assert robot.verification_failures == 2


def test_create_robot_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown robot adapter"):
        create_robot({"robot": "mystery"})
