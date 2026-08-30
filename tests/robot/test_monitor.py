import inspect
import math
import re

import pytest

from everest_robot.monitor import (
    _COLUMN_GUIDE,
    _HELP,
    _KEY_BINDINGS,
    _KEY_GUIDE,
    _STATE_GUIDE,
    MonitorContext,
    _header_row,
    _prompt_save,
    _row,
    _state_of,
    _unsaveable,
    help_lines,
    run_tui,
)
from everest_robot.robot.clock import ManualClock
from everest_robot.robot.contracts import ArmLifecycle, JointLimit, RobotIdentity
from everest_robot.robot.fake_arm import FakeArm
from everest_robot.robot.monitor import JointMonitor, format_table
from everest_robot.robot.parameters import RobotParameters

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


# ── saving a captured pose ─────────────────────────────────────────────────────────


CAPTURE_FILE = """schema_version: 1

robot:
  id: maker-arm-02
  model: maker-arm-v1
  calibration_id: cal-2026-08-20
  joint_order: [shoulder_pan, shoulder_lift, gripper]
  units: radians

motion_defaults:
  max_velocity_rad_s: 0.5
  max_acceleration_rad_s2: 2.0
  tolerance_rad: 0.02
  settle_time_s: 0.2
  timeout_s: 10.0
  control_rate_hz: 30

# Presets are captured, never invented.
named_positions: {}

named_transitions: {}

policy:
  default_controller: vla
  fps: 30
  max_duration_s: 30

replay:
  require_matching_robot_id: true
  require_matching_calibration_id: true
  safe_start_position: null
  max_speed_scale: 1.0
"""


def capture_context(**kwargs) -> MonitorContext:
    defaults = {
        "robot_id": "maker-arm-02",
        "model": "maker-arm-v1",
        "calibration_id": "cal-2026-08-20",
        "config_digest": "sha256:test",
        "poll_hz": 10.0,
        "fake": False,
        "powered": True,
    }
    return MonitorContext(**(defaults | kwargs))


def sample_of(arm: FakeArm, clock: ManualClock):
    return JointMonitor(arm, clock=clock).sample()


def capture_setup(tmp_path, monkeypatch, answers: list[str]):
    """A loadable parameters file plus a scripted operator at the keyboard."""

    path = tmp_path / "maker_arm_v1.yaml"
    path.write_text(CAPTURE_FILE)
    monkeypatch.setenv("EVEREST_ROBOT_PARAMETERS", str(path))
    replies = iter(answers)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(replies))
    return path, RobotParameters.from_yaml(path)


def test_a_measured_pose_can_be_saved_and_loads_back_as_a_named_position(
    tmp_path, monkeypatch
) -> None:
    clock = ManualClock()
    arm = connected_arm(clock)
    path, parameters = capture_setup(
        tmp_path, monkeypatch, ["stage", "operator", "at station 2"]
    )

    _prompt_save(sample_of(arm, clock), parameters)

    saved = RobotParameters.from_yaml(path).named_positions["stage"]
    assert saved.joints == pytest.approx((0.5, -1.0, -0.5))
    assert saved.approved_by == "operator"
    assert saved.notes == "at station 2"
    assert saved.calibration_id == "cal-2026-08-20"


def test_a_blank_name_saves_nothing(tmp_path, monkeypatch) -> None:
    clock = ManualClock()
    path, parameters = capture_setup(tmp_path, monkeypatch, [""])
    before = path.read_text()

    _prompt_save(sample_of(connected_arm(clock), clock), parameters)

    assert path.read_text() == before


def test_provenance_cannot_be_left_blank(tmp_path, monkeypatch) -> None:
    clock = ManualClock()
    path, parameters = capture_setup(tmp_path, monkeypatch, ["stage", "", ""])
    before = path.read_text()

    _prompt_save(sample_of(connected_arm(clock), clock), parameters)

    assert path.read_text() == before


def test_overwriting_an_existing_preset_needs_the_word(tmp_path, monkeypatch) -> None:
    clock = ManualClock()
    arm = connected_arm(clock)
    path, parameters = capture_setup(
        tmp_path, monkeypatch, ["stage", "operator", ""]
    )
    _prompt_save(sample_of(arm, clock), parameters)
    reloaded = RobotParameters.from_yaml(path)
    before = path.read_text()

    arm.positions[0] = 0.9
    replies = iter(["stage", "no"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(replies))
    _prompt_save(sample_of(arm, clock), reloaded)
    assert path.read_text() == before

    replies = iter(["stage", "REPLACE", "operator", ""])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(replies))
    _prompt_save(sample_of(arm, clock), reloaded)
    updated = RobotParameters.from_yaml(path).named_positions["stage"]
    assert updated.joints[0] == pytest.approx(0.9)


# ── which poses may not be saved at all ────────────────────────────────────────────


def test_a_good_sample_from_a_real_arm_is_saveable() -> None:
    clock = ManualClock()

    assert _unsaveable(sample_of(connected_arm(clock), clock), capture_context()) is None


def test_fake_numbers_are_never_offered_for_saving() -> None:
    clock = ManualClock()
    sample = sample_of(connected_arm(clock), clock)

    reason = _unsaveable(sample, capture_context(fake=True))
    assert reason == "--fake numbers describe no physical arm"


def test_a_joint_with_no_feedback_is_not_a_pose() -> None:
    clock = ManualClock()
    arm = connected_arm(clock)
    arm.positions[2] = math.nan

    reason = _unsaveable(sample_of(arm, clock), capture_context())

    assert reason is not None and "no feedback from gripper" in reason


def test_a_pose_outside_the_soft_limits_is_not_offered() -> None:
    clock = ManualClock()
    arm = connected_arm(clock)
    arm.positions[1] = -2.5

    reason = _unsaveable(sample_of(arm, clock), capture_context())

    assert reason is not None and "shoulder_lift outside" in reason


# ── the on-screen guide ────────────────────────────────────────────────────────────


def test_the_footer_and_the_guide_describe_the_same_keys() -> None:
    """One table drives both, so a new binding cannot be documented in only one place."""

    assert [key for key, _ in _KEY_BINDINGS] == [key for key, _ in _KEY_GUIDE]
    for key, _label in _KEY_BINDINGS:
        assert f"{key} " in _HELP


def test_every_key_the_loop_handles_is_in_the_guide() -> None:
    source = inspect.getsource(run_tui)
    handled = {match.group(1) for match in re.finditer(r'ord\("(.)"\)', source)}
    documented = {key for key, _ in _KEY_GUIDE} | {" "}

    # 'h' is an alias for '?', and 'Q' for 'q'; neither needs its own guide entry.
    assert handled - documented <= {"h", "Q"}


def test_the_guide_explains_every_column_the_table_can_draw() -> None:
    headers = set(re.split(r"\s{2,}", _header_row(200).strip()))
    explained = {name for name, _ in _COLUMN_GUIDE}

    assert headers - {"joint"} <= explained


def test_the_guide_explains_every_state_a_joint_can_report() -> None:
    clock = ManualClock()
    faulted = connected_arm(clock)
    faulted.inject_fault("injected motor fault")
    stale = connected_arm(clock, stale_after_commands=0)
    quiet_monitor = JointMonitor(stale, clock=clock)
    quiet_monitor.sample()
    clock.advance(5.0)
    missing = connected_arm(clock)
    missing.positions[0] = math.nan
    outside = connected_arm(clock)
    outside.positions[0] = -9.0

    reported = set()
    for sample in (
        sample_of(connected_arm(clock), clock),
        JointMonitor(faulted, clock=clock).sample(),
        quiet_monitor.sample(),
        sample_of(missing, clock),
        sample_of(outside, clock),
    ):
        reported.update(_state_of(reading) for reading in sample.readings)

    explained = {name for name, _ in _STATE_GUIDE}
    for state in reported:
        assert any(state.startswith(name.split()[0]) for name in explained), state


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"powered": True}, "POWERED"),
        ({"powered": False}, "READ ONLY"),
        ({"fake": True, "powered": False}, "FAKE ARM"),
    ],
)
def test_the_guide_leads_with_what_this_mode_does_to_the_arm(kwargs, expected) -> None:
    lines = help_lines(capture_context(**kwargs))

    body = "\n".join(lines)
    assert expected in body
    # Before anything else a reader might skim past.
    assert lines.index("WHAT THIS SESSION IS DOING") < lines.index("KEYS")


def test_the_fake_guide_says_a_pose_cannot_be_saved_from_it() -> None:
    body = "\n".join(help_lines(capture_context(fake=True, powered=False)))

    assert "cannot be saved" in body


def test_the_guide_points_at_the_validation_that_makes_a_preset_safe() -> None:
    body = "\n".join(help_lines(capture_context()))

    assert "just goto" in body
    assert "named-position-capture.md" in body


def test_the_guide_fits_a_standard_terminal() -> None:
    for context in (capture_context(), capture_context(fake=True, powered=False)):
        for line in help_lines(context):
            assert len(line) <= 78, line
