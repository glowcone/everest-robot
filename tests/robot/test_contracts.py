import json
import math

import pytest

from everest_robot.robot.clock import ManualClock
from everest_robot.robot.contracts import (
    ArmLifecycle,
    FailureReason,
    JointCommand,
    JointLimit,
    JointState,
    MotionProfile,
    MotionResult,
    RobotIdentity,
    TerminationReason,
)

JOINTS = ("shoulder_pan", "gripper")


def make_state(positions: tuple[float, ...], **overrides: object) -> JointState:
    defaults = dict(
        names=JOINTS,
        positions=positions,
        velocities=(0.0, 0.0),
        torques=(0.0, 0.0),
        temperatures=(30.0, 30.0),
        fault_bits=(0, 0),
        sequence=(1, 1),
        monotonic_s=0.0,
        lifecycle=ArmLifecycle.ENABLED,
    )
    defaults.update(overrides)
    return JointState(**defaults)  # type: ignore[arg-type]


def test_durable_results_survive_strict_json() -> None:
    result = MotionResult(
        position_name="clip_attachment_ready",
        reached=False,
        joint_names=JOINTS,
        final_joints=(0.4, math.nan),
        max_tracking_error_rad=math.inf,
        elapsed_s=1.5,
        commands_sent=45,
        robot_id="maker-arm-02",
        calibration_id="cal-2026-08-20",
        config_digest="sha256:abc",
        failure_reason=FailureReason.STALE_FEEDBACK,
    )

    encoded = json.dumps(result.to_json(), allow_nan=False)
    decoded = json.loads(encoded)

    # nan/inf are how the hardware reports missing feedback; they must not reach storage.
    assert decoded["final_joints"] == [0.4, None]
    assert decoded["max_tracking_error_rad"] is None
    assert decoded["failure_reason"] == "stale_feedback"
    assert decoded["joint_names"] == list(JOINTS)


def test_termination_reason_is_a_plain_string_when_stored() -> None:
    encoded = json.dumps({"termination": TerminationReason.CANCELLED.value})
    assert encoded == '{"termination": "cancelled"}'


def test_tracking_error_treats_missing_feedback_as_infinite() -> None:
    state = make_state((0.5, math.nan))

    errors = state.tracking_errors((0.4, 0.0))

    assert errors[0] == pytest.approx(0.1)
    assert errors[1] == math.inf
    assert state.max_tracking_error((0.4, 0.0)) == math.inf
    assert not state.all_finite


def test_fault_bits_alone_mark_the_state_faulted() -> None:
    assert make_state((0.0, 0.0), fault_bits=(0, 2)).has_fault
    assert make_state((0.0, 0.0), lifecycle=ArmLifecycle.FAULT).has_fault
    assert not make_state((0.0, 0.0)).has_fault


def test_identity_mismatch_names_the_offending_field() -> None:
    configured = RobotIdentity("maker-arm-02", "maker-arm-v1", "cal-a", JOINTS)
    hardware = RobotIdentity("maker-arm-02", "maker-arm-v1", "cal-b", JOINTS)

    assert not configured.matches(hardware)
    assert "calibration_id" in str(configured.mismatch_detail(hardware))
    assert configured.mismatch_detail(configured) is None


def test_joint_limit_clamping_respects_the_margin() -> None:
    limit = JointLimit("shoulder_pan", -1.0, 1.0)

    assert limit.clamp(2.0) == 1.0
    assert limit.clamp(2.0, margin=0.1) == pytest.approx(0.9)
    assert not limit.contains(0.95, margin=0.1)


def test_speed_scale_may_not_exceed_the_approved_profile() -> None:
    profile = MotionProfile(0.5, 1.0, 0.03, 0.25, 10.0, 30.0)

    assert profile.scaled(0.5).max_velocity_rad_s == pytest.approx(0.25)
    # Tolerance and timing are not speeds and must not scale with the request.
    assert profile.scaled(0.5).tolerance_rad == pytest.approx(0.03)
    with pytest.raises(ValueError):
        profile.scaled(1.5)


def test_command_reports_clipping() -> None:
    assert JointCommand(JOINTS, (0.0, 0.0)).was_clipped is False
    assert JointCommand(JOINTS, (0.0, 0.0), ("gripper",)).was_clipped is True


def test_manual_clock_advances_only_when_asked() -> None:
    clock = ManualClock()

    clock.sleep(0.5)
    clock.sleep(-1.0)
    clock.advance(0.25)

    assert clock.monotonic() == pytest.approx(0.75)
