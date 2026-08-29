import json
import math

import numpy as np
import pytest

from everest_robot.robot.cameras import CameraRuntime, load_camera_specs
from everest_robot.robot.clock import ManualClock
from everest_robot.robot.contracts import ArmLifecycle, JointLimit, RobotIdentity
from everest_robot.robot.fake_arm import FakeArm
from everest_robot.robot.lerobot_bridge import JointFrame, RobotBridgeCore

JOINTS = ("shoulder_pan", "shoulder_lift", "gripper")
LIMITS = (
    JointLimit("shoulder_pan", -1.0, 1.0),
    JointLimit("shoulder_lift", -2.0, 0.5),
    JointLimit("gripper", -2.0, 0.0),
)
IDENTITY = RobotIdentity("maker-arm-02", "maker-arm-v1", "cal-2026-08-20", JOINTS)
CAMERAS = [
    {"name": "wrist", "kind": "fake", "index_or_path": "0", "width": 8, "height": 4, "fps": 30}
]


def make_core(*, with_cameras: bool = False, frame: JointFrame | None = None) -> RobotBridgeCore:
    arm = FakeArm(
        identity=IDENTITY,
        joint_limits=LIMITS,
        clock=ManualClock(),
        positions=[0.0, -1.0, -0.1],
    )
    cameras = (
        CameraRuntime.from_specs(load_camera_specs(json.dumps(CAMERAS)))
        if with_cameras
        else None
    )
    return RobotBridgeCore(arm, cameras=cameras, frame=frame)


def connected_core(**kwargs: object) -> RobotBridgeCore:
    core = make_core(**kwargs)  # type: ignore[arg-type]
    core.connect()
    core.port.enable()
    return core


def test_feature_names_match_the_maker_follower_convention() -> None:
    core = make_core(with_cameras=True)

    assert core.action_features == {
        "shoulder_pan.pos": float,
        "shoulder_lift.pos": float,
        "gripper.pos": float,
    }
    assert core.observation_features == {
        **core.action_features,
        "wrist": (4, 8, 3),
    }


def test_feature_order_follows_the_configured_joint_order() -> None:
    assert tuple(make_core().action_features) == tuple(f"{name}.pos" for name in JOINTS)


def test_observations_are_reported_in_degrees() -> None:
    core = connected_core()

    observation = core.get_observation()

    assert observation["shoulder_lift.pos"] == pytest.approx(math.degrees(-1.0))
    assert observation["gripper.pos"] == pytest.approx(math.degrees(-0.1))


def test_observations_include_camera_frames() -> None:
    core = connected_core(with_cameras=True)

    observation = core.get_observation()

    assert observation["wrist"].shape == (4, 8, 3)
    assert observation["wrist"].dtype == np.uint8


def test_actions_are_converted_back_to_radians_before_they_reach_the_driver() -> None:
    core = connected_core()

    sent = core.send_action(
        {
            "shoulder_pan.pos": math.degrees(0.5),
            "shoulder_lift.pos": math.degrees(-0.5),
            "gripper.pos": math.degrees(-0.2),
        }
    )

    assert core.port.sent_commands[-1] == pytest.approx((0.5, -0.5, -0.2))
    assert sent["shoulder_pan.pos"] == pytest.approx(math.degrees(0.5))


def test_an_action_past_a_soft_limit_is_clipped_and_the_clip_is_reported() -> None:
    core = connected_core()

    sent = core.send_action(
        {
            "shoulder_pan.pos": math.degrees(5.0),
            "shoulder_lift.pos": math.degrees(-0.5),
            "gripper.pos": math.degrees(-0.2),
        }
    )

    # LeRobot's contract: send_action returns what was actually sent.
    assert sent["shoulder_pan.pos"] == pytest.approx(math.degrees(1.0))
    assert core.port.sent_commands[-1][0] == pytest.approx(1.0)
    assert core.clipped_joints == {"shoulder_pan"}


def test_a_mismatched_action_space_is_refused() -> None:
    core = connected_core()
    good = {f"{name}.pos": 0.0 for name in JOINTS}

    with pytest.raises(KeyError, match="missing"):
        core.parse_action({"shoulder_pan.pos": 0.0})
    with pytest.raises(KeyError, match="unexpected"):
        core.parse_action({**good, "elbow_flex.pos": 0.0})
    with pytest.raises(ValueError, match="finite"):
        core.parse_action({**good, "gripper.pos": math.nan})


def test_sending_to_a_disabled_arm_fails_loudly() -> None:
    core = make_core()
    core.connect()

    with pytest.raises(RuntimeError, match="refused"):
        core.send_action({f"{name}.pos": 0.0 for name in JOINTS})


def test_a_camera_failure_releases_the_can_bus() -> None:
    core = make_core(with_cameras=True)

    def explode() -> None:
        raise RuntimeError("camera missing")

    core.cameras.connect = explode  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="camera missing"):
        core.connect()

    assert core.port.lifecycle is ArmLifecycle.DISCONNECTED


def test_a_non_identity_frame_shifts_the_zero_pose_both_ways() -> None:
    frame = JointFrame(JOINTS, offsets_deg=(0.0, 90.0, 0.0))
    core = connected_core(frame=frame)

    assert not frame.is_identity
    assert JointFrame(JOINTS).is_identity

    observation = core.get_observation()
    assert observation["shoulder_lift.pos"] == pytest.approx(math.degrees(-1.0) + 90.0)

    core.send_action({**{f"{name}.pos": 0.0 for name in JOINTS}, "shoulder_lift.pos": 90.0})
    assert core.port.sent_commands[-1][1] == pytest.approx(0.0)


def test_a_frame_must_describe_the_ports_joints() -> None:
    with pytest.raises(ValueError, match="same order"):
        make_core(frame=JointFrame(("gripper", "shoulder_pan", "shoulder_lift")))


def test_the_real_lerobot_robot_contract_is_satisfied() -> None:
    pytest.importorskip("lerobot")
    from lerobot.robots.robot import Robot

    from everest_robot.robot.lerobot_bridge import make_lerobot_robot

    robot = make_lerobot_robot(make_core(with_cameras=True))

    assert isinstance(robot, Robot)
    assert robot.action_features == {f"{name}.pos": float for name in JOINTS}
    robot.connect()
    try:
        assert robot.is_connected
        assert set(robot.get_observation()) == set(robot.observation_features)
    finally:
        robot.disconnect()
