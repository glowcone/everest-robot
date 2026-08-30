import math

import pytest

from everest_robot.monitor import _header_row, _row
from everest_robot.robot.clock import ManualClock
from everest_robot.robot.contracts import ArmLifecycle, JointLimit, RobotIdentity
from everest_robot.robot.fake_arm import FakeArm
from everest_robot.robot.monitor import JointMonitor, format_table

JOINTS = ("shoulder_pan", "shoulder_lift", "gripper")
LIMITS = (
    JointLimit("shoulder_pan", -1.0, 1.0),
    JointLimit("shoulder_lift", -2.0, 0.5),
    JointLimit("gripper", -2.0, 0.0),
)
IDENTITY = RobotIdentity("maker-arm-02", "maker-arm-v1", "cal-2026-08-20", JOINTS)


def connected_arm(clock: ManualClock, **kwargs) -> FakeArm:
    """An arm that is connected but never enabled -- the hand-teaching case."""

    arm = FakeArm(IDENTITY, LIMITS, clock=clock, positions=[0.5, -1.0, -0.5], **kwargs)
    arm.connect()
    return arm


def test_every_joint_is_reported_in_both_units_and_placed_in_its_soft_limits() -> None:
    clock = ManualClock()
    monitor = JointMonitor(connected_arm(clock), clock=clock)

    sample = monitor.sample()

    assert tuple(reading.name for reading in sample.readings) == JOINTS
    pan, lift, gripper = sample.readings
    assert pan.position_rad == pytest.approx(0.5)
    assert pan.position_deg == pytest.approx(math.degrees(0.5))
    # 0.5 sits three quarters of the way along a [-1, 1] span.
    assert pan.span_fraction == pytest.approx(0.75)
    assert lift.span_fraction == pytest.approx(0.4)
    assert gripper.span_fraction == pytest.approx(0.75)
    assert all(reading.within_limits for reading in sample.readings)
    assert sample.out_of_limits == ()
    assert sample.lifecycle is ArmLifecycle.CONNECTED


def test_watching_the_arm_never_enables_it_or_commands_it() -> None:
    """The safety property the whole program rests on."""

    clock = ManualClock()
    arm = connected_arm(clock)
    monitor = JointMonitor(arm, clock=clock)

    for _ in monitor.stream(poll_hz=20.0, limit=50):
        pass

    assert arm.sent_commands == []
    assert arm.lifecycle is ArmLifecycle.CONNECTED
    assert arm.positions == [0.5, -1.0, -0.5]
    assert monitor.samples == 50


def test_a_connected_arm_keeps_reporting_fresh_feedback() -> None:
    """The driver polls the bus while merely connected, so nothing goes quiet."""

    clock = ManualClock()
    monitor = JointMonitor(connected_arm(clock), clock=clock, stale_after_s=1.0)

    samples = list(monitor.stream(poll_hz=10.0, limit=100))

    assert samples[-1].stale_joints == ()


def test_a_motor_that_stops_advancing_its_counter_is_named_as_quiet() -> None:
    clock = ManualClock()
    # stale_after_commands=0 freezes the feedback counters from the first read.
    monitor = JointMonitor(
        connected_arm(clock, stale_after_commands=0), clock=clock, stale_after_s=1.0
    )

    first = monitor.sample()
    assert first.stale_joints == ()  # nothing to compare the first sample against

    clock.advance(0.5)
    assert monitor.sample().stale_joints == ()

    clock.advance(1.0)
    late = monitor.sample()
    assert late.stale_joints == JOINTS
    assert late.readings[0].quiet_for_s == pytest.approx(1.5)


def test_deltas_are_measured_from_the_marked_reference_pose() -> None:
    clock = ManualClock()
    arm = connected_arm(clock)
    monitor = JointMonitor(arm, clock=clock)

    assert monitor.sample().readings[0].delta_deg is None  # no reference yet
    monitor.mark_reference()
    # The operator moves the arm by hand; the motors are not driving it.
    arm.positions[0] = 0.6
    clock.advance(0.1)

    moved = monitor.sample()
    assert moved.readings[0].delta_rad == pytest.approx(0.1)
    assert moved.readings[0].delta_deg == pytest.approx(math.degrees(0.1))
    assert moved.readings[1].delta_rad == pytest.approx(0.0)

    monitor.clear_reference()
    assert monitor.sample().readings[0].delta_deg is None


def test_a_joint_outside_its_soft_limits_is_reported_rather_than_hidden() -> None:
    clock = ManualClock()
    arm = connected_arm(clock)
    arm.positions[1] = -2.5  # below the shoulder_lift lower limit
    monitor = JointMonitor(arm, clock=clock)

    sample = monitor.sample()

    assert sample.out_of_limits == ("shoulder_lift",)
    lift = sample.readings[1]
    assert not lift.within_limits
    # Clamped for the bar, so an out-of-range joint draws at the end rather than off it.
    assert lift.span_fraction == pytest.approx(0.0)


def test_a_joint_with_no_feedback_is_not_formatted_as_a_number() -> None:
    clock = ManualClock()
    arm = connected_arm(clock)
    arm.positions[2] = math.nan
    monitor = JointMonitor(arm, clock=clock)
    monitor.mark_reference()

    sample = monitor.sample()
    gripper = sample.readings[2]

    assert not gripper.has_feedback
    assert gripper.delta_deg is None
    assert gripper.span_fraction is None
    assert not gripper.within_limits
    assert sample.missing_feedback == ("gripper",)
    assert "NO FEEDBACK" in format_table(sample)[-1]


def test_a_faulted_motor_is_surfaced_with_its_reason() -> None:
    clock = ManualClock()
    arm = connected_arm(clock)
    arm.inject_fault("injected motor fault")
    monitor = JointMonitor(arm, clock=clock)

    sample = monitor.sample()

    assert sample.has_fault
    assert sample.fault_reason == "injected motor fault"
    assert all(reading.has_fault for reading in sample.readings)


def test_streaming_paces_itself_on_the_injected_clock() -> None:
    clock = ManualClock()
    monitor = JointMonitor(connected_arm(clock), clock=clock)

    samples = list(monitor.stream(poll_hz=4.0, limit=5))

    assert len(samples) == 5
    # Four gaps of 250 ms between five samples; the last one does not wait.
    assert clock.monotonic() == pytest.approx(1.0)


def test_a_port_whose_limits_do_not_cover_its_joints_is_refused() -> None:
    clock = ManualClock()
    arm = connected_arm(clock)
    arm.joint_limits = LIMITS[:2]

    with pytest.raises(ValueError, match="describing a different arm"):
        JointMonitor(arm, clock=clock)


def test_the_plain_table_has_one_row_per_joint() -> None:
    clock = ManualClock()
    monitor = JointMonitor(connected_arm(clock), clock=clock)

    lines = format_table(monitor.sample())

    assert len(lines) == len(JOINTS) + 2  # header plus rule
    assert all(name in lines[index + 2] for index, name in enumerate(JOINTS))


@pytest.mark.parametrize("width", [60, 80, 100, 140])
def test_narrow_terminals_drop_columns_but_never_the_joint_angles(width: int) -> None:
    clock = ManualClock()
    reading = JointMonitor(connected_arm(clock), clock=clock).sample().readings[0]

    row = _row(reading, width)

    assert len(row) <= width
    assert len(_header_row(width)) <= width
    assert "shoulder_pan" in row
    assert f"{reading.position_deg:.2f}" in row
