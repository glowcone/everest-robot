import math

import pytest

from everest_robot.robot.clock import ManualClock
from everest_robot.robot.contracts import ArmLifecycle, JointLimit, RobotIdentity
from everest_robot.robot.fake_arm import FakeArm, FakeArmError
from everest_robot.robot.ports import ArmPort, clip_to_limits, violations

JOINTS = ("shoulder_pan", "gripper")
LIMITS = (JointLimit("shoulder_pan", -1.0, 1.0), JointLimit("gripper", -2.0, 0.0))
IDENTITY = RobotIdentity("maker-arm-02", "maker-arm-v1", "cal-2026-08-20", JOINTS)


def make_arm(**overrides: object) -> FakeArm:
    defaults: dict[str, object] = {
        "identity": IDENTITY,
        "joint_limits": LIMITS,
        "clock": ManualClock(),
        "positions": [0.0, -1.0],
        "max_velocity_rad_s": 1.0,
    }
    defaults.update(overrides)
    return FakeArm(**defaults)  # type: ignore[arg-type]


def enabled_arm(**overrides: object) -> FakeArm:
    arm = make_arm(**overrides)
    arm.connect()
    arm.enable()
    return arm


def test_fake_arm_satisfies_the_port_protocol() -> None:
    assert isinstance(make_arm(), ArmPort)


def test_commands_are_refused_outside_the_enabled_state() -> None:
    arm = make_arm()

    assert arm.send_targets([0.5, -1.0]) is False
    arm.connect()
    assert arm.send_targets([0.5, -1.0]) is False
    assert arm.hold_current_position() is False

    arm.enable()
    assert arm.send_targets([0.5, -1.0]) is True


def test_lifecycle_transitions_match_the_driver_rules() -> None:
    arm = make_arm()

    with pytest.raises(FakeArmError, match="enable"):
        arm.enable()
    arm.connect()
    with pytest.raises(FakeArmError, match="connect"):
        arm.connect()

    arm.enable()
    assert arm.lifecycle is ArmLifecycle.ENABLED
    arm.estop()
    assert arm.lifecycle is ArmLifecycle.CONNECTED


def test_enable_is_refused_from_a_pose_outside_the_soft_limits() -> None:
    arm = make_arm(positions=[1.5, -1.0])
    arm.connect()

    with pytest.raises(FakeArmError, match="outside soft limits"):
        arm.enable()


def test_the_arm_approaches_its_target_at_the_velocity_bound() -> None:
    clock = ManualClock()
    arm = enabled_arm(clock=clock, max_velocity_rad_s=1.0)

    arm.send_targets([1.0, -1.0])
    clock.advance(0.25)

    assert arm.read_state().positions[0] == pytest.approx(0.25)

    clock.advance(10.0)
    assert arm.read_state().positions[0] == pytest.approx(1.0)


def test_targets_are_clamped_by_the_driver_not_the_caller() -> None:
    clock = ManualClock()
    arm = enabled_arm(clock=clock)

    assert arm.send_targets([5.0, -1.0]) is True
    clock.advance(10.0)

    assert arm.read_state().positions[0] == pytest.approx(1.0)


def test_non_finite_and_mis_sized_targets_are_rejected() -> None:
    arm = enabled_arm()

    assert arm.send_targets([math.nan, -1.0]) is False
    assert arm.send_targets([0.1]) is False


def test_feedback_counters_stop_advancing_when_staleness_is_injected() -> None:
    clock = ManualClock()
    arm = enabled_arm(clock=clock, stale_after_commands=1)

    arm.send_targets([0.5, -1.0])
    clock.advance(0.1)
    first = arm.read_state().sequence
    clock.advance(0.1)
    second = arm.read_state().sequence

    assert first == second


def test_an_injected_fault_holds_the_current_pose() -> None:
    clock = ManualClock()
    arm = enabled_arm(clock=clock, fault_after_commands=1)

    arm.send_targets([1.0, -1.0])
    clock.advance(0.1)
    held = arm.read_state()
    clock.advance(5.0)
    later = arm.read_state()

    assert held.lifecycle is ArmLifecycle.FAULT
    assert held.has_fault
    assert later.positions == held.positions
    assert "injected" in str(later.fault_reason)

    arm.clear_faults()
    assert arm.lifecycle is ArmLifecycle.CONNECTED


def test_tracking_offset_shows_up_as_measurement_error() -> None:
    clock = ManualClock()
    arm = enabled_arm(clock=clock, tracking_offset_rad=0.05)

    arm.send_targets([0.5, -1.0])
    clock.advance(10.0)
    state = arm.read_state()

    assert state.max_tracking_error((0.5, -1.0)) == pytest.approx(0.05)


def test_clipping_reports_which_joints_were_clamped() -> None:
    command = clip_to_limits([2.0, -1.0], LIMITS)

    assert command.targets == (1.0, -1.0)
    assert command.clipped_joints == ("shoulder_pan",)
    assert clip_to_limits([0.5, -1.0], LIMITS).was_clipped is False

    with pytest.raises(ValueError, match="finite"):
        clip_to_limits([math.nan, -1.0], LIMITS)
    with pytest.raises(ValueError, match="expected 2 targets"):
        clip_to_limits([0.5], LIMITS)


def test_violations_report_without_modifying_the_command() -> None:
    assert violations([0.5, -1.0], LIMITS) == ()
    assert violations([2.0, -3.0], LIMITS) == ("shoulder_pan", "gripper")
    assert violations([0.95, -1.0], LIMITS, margin=0.1) == ("shoulder_pan",)
    assert violations([math.nan, -1.0], LIMITS) == ("shoulder_pan",)
