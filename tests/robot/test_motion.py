import math

import pytest

from everest_robot.robot.clock import ManualClock
from everest_robot.robot.contracts import (
    ArmLifecycle,
    FailureReason,
    JointLimit,
    MotionProfile,
    RobotIdentity,
)
from everest_robot.robot.fake_arm import FakeArm
from everest_robot.robot.motion import JointMotionController, TrapezoidPath
from everest_robot.robot.parameters import RobotParameters

JOINTS = ["shoulder_pan", "shoulder_lift", "gripper"]
CALIBRATION = "maker-arm-02-2026-08-20"
LIMITS = (
    JointLimit("shoulder_pan", -1.0, 1.0),
    JointLimit("shoulder_lift", -2.0, 0.5),
    JointLimit("gripper", -2.0, 0.0),
)


def parameters(**overrides: object) -> RobotParameters:
    document = {
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
        "named_positions": {
            "ready": {
                "joints": [0.5, -0.5, -0.1],
                "calibration_id": CALIBRATION,
                "approved_by": "operator",
                "captured_at": "2026-08-21",
            },
            "clearance": {
                "joints": [0.0, -1.0, -0.1],
                "calibration_id": CALIBRATION,
                "approved_by": "operator",
                "captured_at": "2026-08-21",
            },
            "reachable_but_slow": {
                "joints": [0.9, -0.5, -0.1],
                "calibration_id": CALIBRATION,
                "approved_by": "operator",
                "captured_at": "2026-08-21",
                "max_velocity_rad_s": 0.1,
                "timeout_s": 0.5,
            },
        },
        "named_transitions": {
            "pickup_to_ready": {"waypoints": ["clearance", "ready"]},
        },
        "policy": {"default_controller": "vla", "fps": 30, "max_duration_s": 30},
        "replay": {
            "require_matching_robot_id": True,
            "require_matching_calibration_id": True,
            "safe_start_position": "ready",
            "max_speed_scale": 1.0,
        },
    }
    document.update(overrides)  # type: ignore[arg-type]
    return RobotParameters.from_mapping(document, config_digest="sha256:test", source="test.yaml")


def make_controller(
    *, arm: FakeArm | None = None, clock: ManualClock | None = None, **kwargs: object
) -> tuple[JointMotionController, FakeArm]:
    clock = clock or ManualClock()
    if arm is None:
        arm = FakeArm(
            identity=RobotIdentity("maker-arm-02", "maker-arm-v1", CALIBRATION, tuple(JOINTS)),
            joint_limits=LIMITS,
            clock=clock,
            positions=[0.0, -1.0, -0.1],
            max_velocity_rad_s=2.0,
        )
    arm.connect()
    controller = JointMotionController(arm, parameters(), clock=clock, **kwargs)  # type: ignore[arg-type]
    return controller, arm


# ── trajectory planning ────────────────────────────────────────────────────────────
def test_trapezoid_starts_at_zero_ends_at_one_and_never_reverses() -> None:
    profile = MotionProfile(0.5, 1.0, 0.02, 0.2, 10.0, 50.0)
    path = TrapezoidPath.plan(1.0, profile)

    samples = [path.at(step / 200.0 * path.duration_s) for step in range(201)]

    assert samples[0] == 0.0
    assert samples[-1] == pytest.approx(1.0)
    assert all(later >= earlier for earlier, later in zip(samples, samples[1:], strict=False))


def test_trapezoid_respects_the_velocity_bound() -> None:
    profile = MotionProfile(0.5, 1.0, 0.02, 0.2, 30.0, 50.0)
    displacement = 2.0
    path = TrapezoidPath.plan(displacement, profile)

    dt = path.duration_s / 500.0
    speeds = [
        abs(path.at(index * dt) - path.at((index - 1) * dt)) * displacement / dt
        for index in range(1, 501)
    ]

    assert max(speeds) <= profile.max_velocity_rad_s * 1.01


def test_a_short_move_is_triangular_and_still_completes() -> None:
    profile = MotionProfile(10.0, 1.0, 0.02, 0.2, 10.0, 50.0)
    path = TrapezoidPath.plan(0.01, profile)

    # Too short to ever reach the velocity bound.
    assert path.duration_s == pytest.approx(2 * math.sqrt(0.01 / 1.0))
    assert path.at(path.duration_s) == 1.0


# ── happy path ─────────────────────────────────────────────────────────────────────
def test_reaching_a_named_position_settles_and_holds() -> None:
    controller, arm = make_controller()

    result = controller.go_to_known_position("ready")

    assert result.reached
    assert result.failure_reason is None
    assert result.position_name == "ready"
    assert result.commands_sent > 0
    assert result.max_tracking_error_rad < 0.05
    assert arm.lifecycle is ArmLifecycle.ENABLED
    for measured, target in zip(result.final_joints, (0.5, -0.5, -0.1), strict=True):
        assert measured == pytest.approx(target, abs=0.02)
    assert result.to_json()["config_digest"] == "sha256:test"


def test_being_at_the_target_already_returns_without_commanding() -> None:
    controller, arm = make_controller()
    controller.go_to_known_position("ready")
    commands_after_first = len(arm.sent_commands)

    result = controller.go_to_known_position("ready")

    assert result.reached
    assert result.already_at_target
    assert result.commands_sent == 0
    assert len(arm.sent_commands) == commands_after_first


def test_a_slow_preset_uses_its_own_velocity_override() -> None:
    controller, _ = make_controller()

    fast = controller.go_to_known_position("ready")
    controller.parameters.position("reachable_but_slow")
    slow = controller.go_to_known_position("reachable_but_slow")

    # 0.4 rad at 0.1 rad/s cannot beat 0.5 rad at 0.5 rad/s.
    assert slow.failure_reason is FailureReason.TIMEOUT
    assert fast.reached


def test_speed_scale_lengthens_the_planned_motion() -> None:
    full, _ = make_controller()
    half, _ = make_controller()

    planned_full = full.go_to_known_position("ready", dry_run=True).planned_duration_s
    half_result = half.go_to_known_position("ready", speed_scale=0.5, dry_run=True)
    planned_half = half_result.planned_duration_s

    assert planned_half > planned_full
    with pytest.raises(ValueError):
        full.go_to_known_position("ready", speed_scale=2.0)


def test_dry_run_validates_and_plans_without_moving() -> None:
    controller, arm = make_controller()

    result = controller.go_to_known_position("ready", dry_run=True)

    assert result.dry_run
    assert not result.reached
    assert result.failure_reason is None
    assert result.planned_duration_s > 0
    assert arm.sent_commands == []
    assert arm.lifecycle is ArmLifecycle.CONNECTED


# ── refusals decided before any motion ─────────────────────────────────────────────
def test_an_unknown_position_is_refused_without_moving() -> None:
    controller, arm = make_controller()

    result = controller.go_to_known_position("nowhere")

    assert result.failure_reason is FailureReason.UNKNOWN_POSITION
    assert arm.sent_commands == []
    assert "known:" in str(result.failure_detail)


def test_a_preset_outside_the_active_hardware_limits_is_refused_not_clamped() -> None:
    narrow = (
        JointLimit("shoulder_pan", -0.2, 0.2),
        JointLimit("shoulder_lift", -2.0, 0.5),
        JointLimit("gripper", -2.0, 0.0),
    )
    clock = ManualClock()
    arm = FakeArm(
        identity=RobotIdentity("maker-arm-02", "maker-arm-v1", CALIBRATION, tuple(JOINTS)),
        joint_limits=narrow,
        clock=clock,
        positions=[0.0, -1.0, -0.1],
    )
    controller, _ = make_controller(arm=arm, clock=clock)

    result = controller.go_to_known_position("ready")

    assert result.failure_reason is FailureReason.LIMIT_VIOLATION
    assert "shoulder_pan" in str(result.failure_detail)
    assert arm.sent_commands == []


def test_a_different_calibration_refuses_every_move() -> None:
    clock = ManualClock()
    arm = FakeArm(
        identity=RobotIdentity(
            "maker-arm-02", "maker-arm-v1", "some-other-calibration", tuple(JOINTS)
        ),
        joint_limits=LIMITS,
        clock=clock,
        positions=[0.0, -1.0, -0.1],
    )
    controller, _ = make_controller(arm=arm, clock=clock)

    result = controller.go_to_known_position("ready")

    assert result.failure_reason is FailureReason.IDENTITY_MISMATCH
    assert arm.sent_commands == []


# ── failures during motion ─────────────────────────────────────────────────────────
def test_a_motor_fault_stops_the_move_and_leaves_the_arm_held() -> None:
    clock = ManualClock()
    arm = FakeArm(
        identity=RobotIdentity("maker-arm-02", "maker-arm-v1", CALIBRATION, tuple(JOINTS)),
        joint_limits=LIMITS,
        clock=clock,
        positions=[0.0, -1.0, -0.1],
        fault_after_commands=3,
    )
    controller, _ = make_controller(arm=arm, clock=clock)

    result = controller.go_to_known_position("ready")

    assert result.failure_reason is FailureReason.MOTOR_FAULT
    assert not result.reached
    # The driver holds under its own fault policy; the controller must not release torque.
    assert arm.lifecycle is ArmLifecycle.FAULT


def test_stale_feedback_stops_the_move() -> None:
    clock = ManualClock()
    arm = FakeArm(
        identity=RobotIdentity("maker-arm-02", "maker-arm-v1", CALIBRATION, tuple(JOINTS)),
        joint_limits=LIMITS,
        clock=clock,
        positions=[0.0, -1.0, -0.1],
        stale_after_commands=3,
    )
    controller, _ = make_controller(arm=arm, clock=clock)

    result = controller.go_to_known_position("ready")

    assert result.failure_reason is FailureReason.STALE_FEEDBACK
    assert "fresh feedback" in str(result.failure_detail)


def test_persistent_tracking_error_stops_the_move() -> None:
    clock = ManualClock()
    arm = FakeArm(
        identity=RobotIdentity("maker-arm-02", "maker-arm-v1", CALIBRATION, tuple(JOINTS)),
        joint_limits=LIMITS,
        clock=clock,
        positions=[0.0, -1.0, -0.1],
        tracking_offset_rad=0.5,
    )
    controller, _ = make_controller(arm=arm, clock=clock)

    result = controller.go_to_known_position("ready")

    assert result.failure_reason is FailureReason.TRACKING_ERROR
    assert result.max_tracking_error_rad >= 0.16


def test_a_refused_command_reports_not_enabled() -> None:
    clock = ManualClock()
    arm = FakeArm(
        identity=RobotIdentity("maker-arm-02", "maker-arm-v1", CALIBRATION, tuple(JOINTS)),
        joint_limits=LIMITS,
        clock=clock,
        positions=[0.0, -1.0, -0.1],
        refuse_targets=True,
    )
    controller, _ = make_controller(arm=arm, clock=clock)

    result = controller.go_to_known_position("ready")

    assert result.failure_reason is FailureReason.NOT_ENABLED


def test_cancellation_stops_promptly_and_holds() -> None:
    clock = ManualClock()
    controller, arm = make_controller(clock=clock)
    ticks = {"count": 0}

    def cancel() -> bool:
        ticks["count"] += 1
        return ticks["count"] > 3

    controller.cancel = cancel

    result = controller.go_to_known_position("ready")

    assert result.failure_reason is FailureReason.CANCELLED
    assert result.commands_sent == 3
    assert arm.lifecycle is ArmLifecycle.ENABLED


def test_estop_on_failure_is_opt_in() -> None:
    clock = ManualClock()
    controller, arm = make_controller(clock=clock, estop_on_failure=True)
    controller.cancel = lambda: True

    controller.go_to_known_position("ready")

    assert arm.lifecycle is ArmLifecycle.CONNECTED


def test_heartbeats_fire_while_a_long_move_runs() -> None:
    clock = ManualClock()
    beats = {"count": 0}

    def heartbeat() -> None:
        beats["count"] += 1

    controller, _ = make_controller(clock=clock, heartbeat=heartbeat, heartbeat_interval_s=0.5)

    result = controller.go_to_known_position("ready")

    assert result.reached
    assert beats["count"] >= 2


# ── waypoint transitions ───────────────────────────────────────────────────────────
def test_a_transition_visits_every_waypoint_in_order() -> None:
    controller, arm = make_controller()

    result = controller.follow_transition("pickup_to_ready")

    assert result.reached
    assert result.waypoints == ("clearance", "ready")
    assert result.position_name == "ready"
    # The clearance pose must actually be passed through, not skipped by interpolation.
    assert any(command[1] == pytest.approx(-1.0, abs=0.02) for command in arm.sent_commands)
    assert result.final_joints[0] == pytest.approx(0.5, abs=0.02)


def test_a_transition_stops_at_the_first_failing_leg() -> None:
    clock = ManualClock()
    arm = FakeArm(
        identity=RobotIdentity("maker-arm-02", "maker-arm-v1", CALIBRATION, tuple(JOINTS)),
        joint_limits=LIMITS,
        clock=clock,
        positions=[0.0, -1.0, -0.1],
        fault_after_commands=2,
    )
    controller, _ = make_controller(arm=arm, clock=clock)

    result = controller.follow_transition("pickup_to_ready")

    assert result.failure_reason is FailureReason.MOTOR_FAULT
    assert not result.reached
    assert result.waypoints == ("clearance", "ready")


def test_an_unknown_transition_is_refused() -> None:
    controller, arm = make_controller()

    result = controller.follow_transition("nowhere_to_nothing")

    assert result.failure_reason is FailureReason.UNKNOWN_POSITION
    assert arm.sent_commands == []


def test_a_raising_heartbeat_still_leaves_the_arm_held() -> None:
    """Absurd signals cancellation by raising from ctx.heartbeat()."""

    class Cancelled(BaseException):
        pass

    clock = ManualClock()
    controller, arm = make_controller(clock=clock, heartbeat_interval_s=0.0)
    beats = {"count": 0}

    def heartbeat() -> None:
        beats["count"] += 1
        if beats["count"] > 2:
            raise Cancelled("run cancelled")

    controller.heartbeat = heartbeat

    with pytest.raises(Cancelled):
        controller.go_to_known_position("ready")

    assert arm.lifecycle is ArmLifecycle.ENABLED
    # Held where it stopped: the commanded target is the measured pose, not the goal.
    held = arm.read_state().positions
    clock.advance(5.0)
    assert arm.read_state().positions == pytest.approx(held)
