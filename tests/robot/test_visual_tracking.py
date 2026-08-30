import pytest

from everest_robot.robot.clock import ManualClock
from everest_robot.robot.contracts import ArmLifecycle, JointLimit, RobotIdentity
from everest_robot.robot.fake_arm import FakeArm
from everest_robot.robot.visual_tracking import ArrivalGate, TrackerStopped, VisualTracker

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


# ── arrival ────────────────────────────────────────────────────────────────────────
def arrival_ticks(gate, clock, running, count, target=TARGET):
    """Servo `count` ticks of the tracker's period and report the gate's verdict."""

    arrived = False
    for _ in range(count):
        arrived = gate.update(running.tick(target))
        clock.advance(running.period_s)
    return arrived


def test_arrival_needs_the_measured_pose_to_settle_not_just_the_command():
    clock = ManualClock()
    hardware = arm(clock, max_velocity_rad_s=0.0)  # commanded, but never gets there
    running = tracker(hardware, lock_frames=1)
    running.start()
    gate = ArrivalGate(tolerance_rad=0.01, ticks=3)

    assert not arrival_ticks(gate, clock, running, 10)
    assert gate.consecutive == 0


def test_arrival_is_reported_once_the_arm_holds_the_target():
    clock = ManualClock()
    hardware = arm(clock)
    running = tracker(hardware, lock_frames=1)
    running.start()
    gate = ArrivalGate(tolerance_rad=0.01, ticks=3)

    # The FakeArm reaches what it is commanded, so the arm closes the gap within the speed
    # lock and the gate then needs its consecutive settled ticks.
    assert arrival_ticks(gate, clock, running, 40)
    assert gate.arrived


def test_a_miss_resets_the_arrival_count():
    clock = ManualClock()
    hardware = arm(clock)
    running = tracker(hardware, lock_frames=1)
    running.start()
    gate = ArrivalGate(tolerance_rad=0.01, ticks=3)
    arrival_ticks(gate, clock, running, 40)

    # The object may have been picked up or moved; being above where it *was* is not an
    # arrival, so the count starts again from the next good detection.
    assert not gate.update(running.tick(None))
    assert gate.consecutive == 0


def test_a_held_tick_never_counts_as_an_arrival():
    hardware = arm(ManualClock())
    running = tracker(hardware, lock_frames=5)
    running.start()
    gate = ArrivalGate(tolerance_rad=10.0, ticks=1)

    # Locking on: the target is accepted but nothing has been commanded yet.
    assert not gate.update(running.tick(TARGET))


@pytest.mark.parametrize(("field", "value"), [("tolerance_rad", 0.0), ("ticks", 0)])
def test_a_nonsensical_arrival_gate_is_refused_at_construction(field, value):
    with pytest.raises(ValueError):
        ArrivalGate(**{field: value})
