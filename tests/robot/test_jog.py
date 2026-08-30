import math

import pytest

from everest_robot.jog import (
    DEFAULT_JOINT,
    MAX_DELTA_RAD,
    JogRefused,
    plan_jog,
    raise_arm,
)
from everest_robot.robot.clock import ManualClock
from everest_robot.robot.contracts import JointLimit, RobotIdentity
from everest_robot.robot.fake_arm import FakeArm
from everest_robot.robot.parameters import RobotParameters
from everest_robot.robot.ports import ArmPort
from everest_robot.robot.session import RobotSession

JOINTS = ["shoulder_pan", "shoulder_lift", "gripper"]
CALIBRATION = "maker-arm-02-2026-08-20"
# shoulder_lift's real span on this arm, from docs/lerobot-frame-reconciliation.md.
LIMITS = (
    JointLimit("shoulder_pan", -0.668, 4.818),
    JointLimit("shoulder_lift", -2.024, 0.979),
    JointLimit("gripper", -2.092, -0.039),
)


def parameters() -> RobotParameters:
    return RobotParameters.from_mapping(
        {
            "schema_version": 1,
            "robot": {
                "id": "maker-arm-02",
                "model": "maker-arm-v1",
                "calibration_id": CALIBRATION,
                "joint_order": list(JOINTS),
                "units": "radians",
            },
            "motion_defaults": {
                "max_velocity_rad_s": 0.5,
                "max_acceleration_rad_s2": 2.0,
                "tolerance_rad": 0.02,
                "settle_time_s": 0.2,
                "timeout_s": 10.0,
                "control_rate_hz": 50,
            },
            "named_positions": {},
            "named_transitions": {},
            "policy": {"default_controller": "vla", "fps": 30, "max_duration_s": 30},
            "replay": {
                "require_matching_robot_id": True,
                "require_matching_calibration_id": True,
                "safe_start_position": None,
                "max_speed_scale": 1.0,
            },
        },
        config_digest="sha256:test",
        source="test.yaml",
    )


def identity() -> RobotIdentity:
    return RobotIdentity(
        robot_id="maker-arm-02",
        model="maker-arm-v1",
        calibration_id=CALIBRATION,
        joint_names=tuple(JOINTS),
    )


def session(positions: list[float]) -> tuple[RobotSession, FakeArm, ManualClock]:
    """An open-able session over FakeArm, which already satisfies ArmPort."""

    clock = ManualClock()
    arm = FakeArm(
        identity=identity(), joint_limits=LIMITS, clock=clock, positions=list(positions)
    )
    assert isinstance(arm, ArmPort)
    return RobotSession(arm, parameters(), clock=clock), arm, clock


# ── plan_jog ───────────────────────────────────────────────────────────────────────


def test_plan_jog_moves_only_the_named_joint():
    target = plan_jog([0.1, 0.0, -1.0], JOINTS, LIMITS, delta_rad=0.10)
    assert target == pytest.approx((0.1, 0.10, -1.0))


def test_plan_jog_is_relative_to_the_measured_pose_not_to_zero():
    target = plan_jog([0.1, -0.55, -1.0], JOINTS, LIMITS, delta_rad=0.10)
    assert target[1] == pytest.approx(-0.45)


def test_plan_jog_raises_toward_the_upper_limit():
    """Increasing shoulder_lift must be the direction with headroom when lowered."""
    lowered = plan_jog([0.1, -1.5, -1.0], JOINTS, LIMITS, delta_rad=0.10)
    assert lowered[1] > -1.5


def test_plan_jog_refuses_past_the_soft_limit_rather_than_clamping():
    # The recorded episodes all start at shoulder_lift's upper limit, so this is the
    # outcome to expect if the arm is parked where the dataset begins.
    with pytest.raises(JogRefused, match="outside the driver's soft limits"):
        plan_jog([0.1, 0.979, -1.0], JOINTS, LIMITS, delta_rad=0.10)


def test_plan_jog_reports_remaining_headroom():
    with pytest.raises(JogRefused, match=r"Only \+0.0290 rad of travel remains"):
        plan_jog([0.1, 0.95, -1.0], JOINTS, LIMITS, delta_rad=0.10)


def test_plan_jog_refuses_an_unknown_joint():
    with pytest.raises(JogRefused, match="unknown joint"):
        plan_jog([0.1, 0.0, -1.0], JOINTS, LIMITS, joint="elbow_flex")


def test_plan_jog_refuses_an_oversized_delta():
    with pytest.raises(JogRefused, match="ceiling"):
        plan_jog([0.1, 0.0, -1.0], JOINTS, LIMITS, delta_rad=MAX_DELTA_RAD + 0.01)


def test_plan_jog_refuses_a_zero_or_nonfinite_delta():
    for bad in (0.0, math.nan, math.inf):
        with pytest.raises(JogRefused, match="non-zero finite"):
            plan_jog([0.1, 0.0, -1.0], JOINTS, LIMITS, delta_rad=bad)


def test_plan_jog_refuses_without_feedback_for_the_joint():
    with pytest.raises(JogRefused, match="no joint feedback"):
        plan_jog([0.1, math.nan, -1.0], JOINTS, LIMITS)


def test_plan_jog_down_moves_toward_the_lower_limit():
    target = plan_jog([0.1, 0.0, -1.0], JOINTS, LIMITS, delta_rad=-0.10)
    assert target[1] == pytest.approx(-0.10)


# ── raise_arm against the fake arm ─────────────────────────────────────────────────


def test_raise_arm_reaches_the_target():
    sess, arm, clock = session([0.1, -0.50, -1.0])
    with sess:
        result, start, target = raise_arm(sess, delta_rad=0.10, speed_scale=1.0)
    assert result.reached, result.failure_detail
    assert start[1] == pytest.approx(-0.50)
    assert target[1] == pytest.approx(-0.40)
    assert result.final_joints[1] == pytest.approx(-0.40, abs=0.02)


def test_raise_arm_leaves_the_other_joints_alone():
    sess, arm, clock = session([0.1, -0.50, -1.0])
    with sess:
        result, _, _ = raise_arm(sess, delta_rad=0.10, speed_scale=1.0)
    assert result.final_joints[0] == pytest.approx(0.1, abs=0.02)
    assert result.final_joints[2] == pytest.approx(-1.0, abs=0.02)


def test_raise_arm_dry_run_sends_no_commands():
    sess, arm, clock = session([0.1, -0.50, -1.0])
    with sess:
        result, _, _ = raise_arm(sess, delta_rad=0.10, dry_run=True)
    assert result.dry_run
    assert not result.reached
    assert result.commands_sent == 0
    assert arm.sent_commands == []
    assert arm.positions[1] == pytest.approx(-0.50)


def test_raise_arm_at_the_upper_limit_refuses_before_energizing():
    sess, arm, clock = session([0.1, 0.979, -1.0])
    with sess, pytest.raises(JogRefused):
        raise_arm(sess, delta_rad=0.10)
    assert arm.sent_commands == []


def test_default_joint_is_shoulder_lift():
    assert DEFAULT_JOINT == "shoulder_lift"
