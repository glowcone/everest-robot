import pytest

from everest_robot.robot.clock import ManualClock
from everest_robot.robot.contracts import ArmLifecycle, JointLimit, RobotIdentity
from everest_robot.robot.fake_arm import FakeArm
from everest_robot.robot.visual_tracking import TrackerStopped, VisualTracker

JOINTS = ("shoulder_pan", "shoulder_lift", "gripper")
LIMITS = (
    JointLimit("shoulder_pan", -1.0, 1.0),
    JointLimit("shoulder_lift", -2.0, 0.5),
    JointLimit("gripper", -2.0, 0.0),
)


def arm(clock: ManualClock, **overrides: object) -> FakeArm:
    hardware = FakeArm(
        RobotIdentity("maker-arm-02", "maker-arm-v1", "cal-1", JOINTS),
        LIMITS,
        clock=clock,
        positions=[0.0, 0.0, -1.0],
        **overrides,
    )
    hardware.connect()
    return hardware


def tracker(hardware: FakeArm, **overrides: object) -> VisualTracker:
    defaults: dict[str, object] = {
        "rate_hz": 10.0,
        "max_velocity_rad_s": 0.5,
        "lock_frames": 3,
    }
    return VisualTracker(hardware, **{**defaults, **overrides})


TARGET = (0.5, 0.3, -1.0)


def test_start_seeds_the_command_from_measured_feedback_and_enables():
    hardware = arm(ManualClock())
    seed = tracker(hardware).start()
    assert seed == (0.0, 0.0, -1.0)
    assert hardware.lifecycle is ArmLifecycle.ENABLED


def test_a_missing_detection_holds_and_commands_nothing():
    hardware = arm(ManualClock())
    running = tracker(hardware)
    running.start()

    tick = running.tick(None)
    assert not tick.moved
    assert tick.reason == "no detection"
    assert hardware.sent_commands == []


def test_motion_waits_for_a_lock_before_the_first_step():
    hardware = arm(ManualClock())
    running = tracker(hardware, lock_frames=3)
    running.start()

    first, second, third = (running.tick(TARGET) for _ in range(3))
    assert (first.moved, second.moved, third.moved) == (False, False, True)
    assert (first.reason, second.reason) == ("locking on", "locking on")
    assert len(hardware.sent_commands) == 1


def test_the_speed_lock_bounds_every_step():
    clock = ManualClock()
    hardware = arm(clock)
    running = tracker(hardware, rate_hz=10.0, max_velocity_rad_s=0.5, lock_frames=1)
    running.start()

    assert running.max_step_rad == pytest.approx(0.05)
    previous = (0.0, 0.0, -1.0)
    for _ in range(4):
        clock.advance(0.1)
        tick = running.tick(TARGET)
        assert tick.moved
        assert tick.command is not None
        step = max(abs(a - b) for a, b in zip(tick.command, previous, strict=True))
        assert step <= running.max_step_rad + 1e-12
        previous = tick.command
    # Four ticks of 0.05 rad, not the 0.5 rad the target asked for.
    assert previous[0] == pytest.approx(0.2)


def test_a_miss_drops_the_lock_so_motion_must_re_lock():
    hardware = arm(ManualClock())
    running = tracker(hardware, lock_frames=2)
    running.start()

    running.tick(TARGET)
    assert running.tick(TARGET).moved
    running.tick(None)
    assert not running.locked

    assert not running.tick(TARGET).moved
    assert running.tick(TARGET).moved


def test_a_hold_re_seeds_the_command_so_it_cannot_coast_from_a_stale_belief():
    clock = ManualClock()
    hardware = arm(clock)
    running = tracker(hardware, lock_frames=1)
    running.start()

    clock.advance(0.1)
    running.tick(TARGET)
    running.tick(None)
    clock.advance(0.1)
    resumed = running.tick(TARGET)

    assert resumed.command is not None
    measured = hardware.read_state().positions
    step = max(abs(a - b) for a, b in zip(resumed.command, measured, strict=True))
    assert step <= running.max_step_rad + 1e-9


def test_a_target_outside_the_soft_limits_holds_rather_than_clamping():
    hardware = arm(ManualClock())
    running = tracker(hardware, lock_frames=1)
    running.start()

    tick = running.tick((0.5, 9.0, -1.0))
    assert not tick.moved
    assert tick.reason == "target outside soft limits: shoulder_lift"
    assert hardware.sent_commands == []


def test_a_refused_command_stops_the_tracker_and_holds():
    hardware = arm(ManualClock(), refuse_targets=True)
    running = tracker(hardware, lock_frames=1)
    running.start()

    tick = running.tick(TARGET)
    assert not tick.moved
    assert running.stopped_reason is not None
    assert hardware.lifecycle is ArmLifecycle.ENABLED
    with pytest.raises(TrackerStopped):
        running.tick(TARGET)


def test_a_fault_stops_the_tracker():
    hardware = arm(ManualClock(), fault_after_commands=1)
    running = tracker(hardware, lock_frames=1)
    running.start()

    running.tick(TARGET)
    tick = running.tick(TARGET)
    assert not tick.moved
    assert tick.reason == "arm fault"
    assert running.stopped_reason == "arm fault"


def test_a_dry_run_never_energizes_and_never_commands():
    hardware = arm(ManualClock())
    running = tracker(hardware, lock_frames=1, dry_run=True)
    running.start()

    tick = running.tick(TARGET)
    assert hardware.lifecycle is ArmLifecycle.CONNECTED
    assert hardware.sent_commands == []
    assert not tick.moved
    assert tick.reason == "dry run"
    assert tick.command is not None


def test_a_target_of_the_wrong_width_is_a_programming_error_not_a_hold():
    hardware = arm(ManualClock())
    running = tracker(hardware, lock_frames=1)
    running.start()
    with pytest.raises(ValueError, match="expected 3 joints"):
        running.tick((0.1, 0.2))


def test_stop_holds_the_arm_and_is_safe_to_repeat():
    hardware = arm(ManualClock())
    running = tracker(hardware, lock_frames=1)
    running.start()
    running.tick(TARGET)

    running.stop()
    running.stop()
    assert hardware.lifecycle is ArmLifecycle.ENABLED
    assert not running.locked


@pytest.mark.parametrize(
    ("field", "value"),
    [("rate_hz", 0.0), ("max_velocity_rad_s", -1.0), ("lock_frames", 0)],
)
def test_nonsensical_bounds_are_refused_at_construction(field, value):
    with pytest.raises(ValueError):
        VisualTracker(arm(ManualClock()), **{field: value})


def test_stopping_without_a_hold_commands_nothing_on_the_way_out():
    """The session teardown that follows cuts torque, so a parting hold is only a lurch."""

    hardware = arm(ManualClock())
    running = tracker(hardware, lock_frames=1)
    running.start()
    running.tick(TARGET)

    commands = len(hardware.sent_commands)
    running.stop(hold=False)
    assert len(hardware.sent_commands) == commands
    assert not running.locked
