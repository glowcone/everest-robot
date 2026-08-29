"""Replay tests, driven by the real dataset fixture against fake hardware."""

from __future__ import annotations

from pathlib import Path

import pytest

from everest_robot.domain import LimitPolicy, ReplayRequest, ReplayResult, json_dict
from everest_robot.robot.clock import ManualClock
from everest_robot.robot.contracts import ArmLifecycle, JointLimit, RobotIdentity
from everest_robot.robot.datasets import DatasetSnapshot, LeRobotV3Reader
from everest_robot.robot.errors import (
    CalibrationMismatchError,
    DatasetCompatibilityError,
    ReplayLimitError,
)
from everest_robot.robot.fake_arm import FakeArm
from everest_robot.robot.lease import InMemoryLease
from everest_robot.robot.lerobot_bridge import JointFrame
from everest_robot.robot.parameters import RobotParameters
from everest_robot.robot.replay import ReplayControl, ReplayRunner, limits_in_degrees

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "lerobot_v3_mini"
REPO = "h8i76dfsd9/test1_20260829_130743"
REVISION = "55e561161026be306d06354a8941c4431e8e805f"
CALIBRATION = "maker-arm-02-2026-08-20"

JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
    "gripper",
)
# maker-arm-sdk's shipped soft limits, in calibrated joint radians.
SDK_LIMITS = (
    JointLimit("shoulder_pan", -0.668, 4.818),
    JointLimit("shoulder_lift", -2.024, 0.979),
    JointLimit("elbow_flex", 3.882, 7.955),
    JointLimit("wrist_flex", -0.832, 2.122),
    JointLimit("wrist_yaw", 0.577, 3.641),
    JointLimit("wrist_roll", 0.966, 6.292),
    JointLimit("gripper", -2.092, -0.039),
)
# The shipped reconciliation between those radians and MakerFollower's degrees.
OFFSETS = (-119.94, -59.31, -219.66, -17.56, -130.64, -208.53, -0.25)
IDENTITY = RobotIdentity("maker-arm-02", "maker-arm-v1", CALIBRATION, JOINTS)


def parameters(**replay_overrides: object) -> RobotParameters:
    replay = {
        "require_matching_robot_id": True,
        "require_matching_calibration_id": True,
        "safe_start_position": None,
        "max_speed_scale": 1.0,
        "heartbeat_interval_s": 0.2,
        "initial_pose_tolerance_deg": 1.0,
        "settle_time_s": 0.1,
        "tracking_error_limit_deg": 5.0,
        "max_consecutive_missed_deadlines": 5,
    }
    replay.update(replay_overrides)
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
            "max_velocity_rad_s": 2.0,
            "max_acceleration_rad_s2": 8.0,
            "tolerance_rad": 0.02,
            "settle_time_s": 0.1,
            "timeout_s": 30.0,
            "control_rate_hz": 50,
        },
        "named_positions": {},
        "named_transitions": {},
        "policy": {"default_controller": "vla", "fps": 30, "max_duration_s": 30},
        "replay": replay,
        "lerobot_frame": {
            "approved_by": "test",
            "captured_at": "2026-08-29",
            "joints": dict(zip(JOINTS, OFFSETS, strict=True)),
        },
        "approved_replays": {
            REPO: {
                "revision": REVISION,
                "robot_id": "maker-arm-02",
                "calibration_id": CALIBRATION,
                "episodes": [0, 1],
                "limit_policy": "clamp_within_tolerance",
                "max_limit_deviation_deg": 2.0,
                "approved_by": "operator",
            }
        },
    }
    return RobotParameters.from_mapping(document, config_digest="sha256:test", source="test.yaml")


def request(**overrides: object) -> ReplayRequest:
    values: dict[str, object] = {
        "repo_id": REPO,
        "revision": REVISION,
        "episode": 0,
        "robot_id": "maker-arm-02",
        "calibration_id": CALIBRATION,
        "limit_policy": LimitPolicy.CLAMP_WITHIN_TOLERANCE,
        "max_limit_deviation_deg": 2.0,
    }
    values.update(overrides)
    return ReplayRequest(**values)  # type: ignore[arg-type]


class FixtureResolver:
    """Stands in for the Hub: the fixture is already an immutable local snapshot."""

    def __init__(self, root: Path = FIXTURE) -> None:
        self.root = root
        self.calls: list[tuple[str, str]] = []

    def resolve(self, repo_id: str, revision: str) -> DatasetSnapshot:
        self.calls.append((repo_id, revision))
        return DatasetSnapshot(repo_id, revision, self.root)


def away_pose_rad() -> list[float]:
    """A legal pose well away from the episode's start, for exercising alignment."""

    return [(limit.lower_rad + limit.upper_rad) / 2 for limit in SDK_LIMITS]


def start_pose_rad() -> list[float]:
    """The pose replay aligns to: the recorded initial state, clamped into the limits.

    The recording's own start pose sits 1.7 deg past this driver's elbow_flex limit, so the
    reachable target is the clamped one.
    """

    episode = LeRobotV3Reader(DatasetSnapshot(REPO, REVISION, FIXTURE)).read_episode(0)
    frame = JointFrame(JOINTS, offsets_deg=OFFSETS)
    return [
        limit.clamp(value)
        for value, limit in zip(frame.to_radians(episode.states[0]), SDK_LIMITS, strict=True)
    ]


def make_runner(
    *,
    positions: list[float] | None = None,
    params: RobotParameters | None = None,
    clock: ManualClock | None = None,
    **arm_kwargs: object,
) -> tuple[ReplayRunner, FakeArm, ManualClock]:
    clock = clock or ManualClock()
    arm = FakeArm(
        identity=IDENTITY,
        joint_limits=SDK_LIMITS,
        clock=clock,
        positions=positions if positions is not None else start_pose_rad(),
        max_velocity_rad_s=20.0,
        **arm_kwargs,  # type: ignore[arg-type]
    )
    runner = ReplayRunner(
        arm,
        params or parameters(),
        resolver=FixtureResolver(),
        lease=InMemoryLease("maker-arm-02"),
        clock=clock,
    )
    return runner, arm, clock


# ── frame conversion ───────────────────────────────────────────────────────────────
def test_the_frame_maps_driver_limits_onto_the_datasets_degrees() -> None:
    lower, upper = limits_in_degrees(SDK_LIMITS, JointFrame(JOINTS, offsets_deg=OFFSETS))

    # These are MakerFollower's published limits, reached from the SDK's radians.
    assert lower[1] == pytest.approx(-175.3, abs=0.05)
    assert upper[1] == pytest.approx(-3.2, abs=0.05)
    assert lower[2] == pytest.approx(2.8, abs=0.05)
    assert upper[6] == pytest.approx(-2.5, abs=0.05)


def test_a_direction_flipping_frame_is_refused() -> None:
    class Flipped(JointFrame):
        def to_degrees(self, radians):  # type: ignore[override]
            return tuple(-v for v in radians)

    with pytest.raises(DatasetCompatibilityError, match="inverts"):
        limits_in_degrees(SDK_LIMITS, Flipped(JOINTS))


def test_without_the_frame_the_dataset_lands_outside_every_limit() -> None:
    """The offsets are not cosmetic: identity conversion rejects the whole episode."""

    runner, arm, _ = make_runner()
    runner.frame = JointFrame(JOINTS)

    with pytest.raises(ReplayLimitError):
        runner.preflight(request())

    assert arm.lifecycle is ArmLifecycle.DISCONNECTED


# ── approval gating ────────────────────────────────────────────────────────────────
def test_an_unapproved_dataset_is_refused_before_anything_is_resolved() -> None:
    runner, arm, _ = make_runner()

    with pytest.raises(CalibrationMismatchError, match="not in approved_replays"):
        runner.preflight(request(repo_id="someone/else"))

    assert runner.resolver.calls == []  # type: ignore[union-attr]
    assert arm.lifecycle is ArmLifecycle.DISCONNECTED


def test_an_unapproved_revision_or_episode_is_refused() -> None:
    runner, _, _ = make_runner()

    with pytest.raises(CalibrationMismatchError, match="is not approved"):
        runner.preflight(request(revision="b" * 40))
    with pytest.raises(CalibrationMismatchError, match="is not approved"):
        runner.preflight(request(episode=4))


def test_a_request_naming_another_arm_is_refused() -> None:
    runner, _, _ = make_runner()

    with pytest.raises(CalibrationMismatchError, match="maker-arm-02"):
        runner.preflight(request(robot_id="maker-arm-99"))
    with pytest.raises(CalibrationMismatchError, match=CALIBRATION):
        runner.preflight(request(calibration_id="older-calibration"))


def test_the_deployment_identity_is_checked_even_without_an_approval() -> None:
    runner, _, _ = make_runner(params=parameters(require_approved_dataset=False))

    with pytest.raises(CalibrationMismatchError, match="this deployment is"):
        runner.preflight(request(repo_id="someone/else", robot_id="maker-arm-99"))


def test_a_request_may_be_stricter_than_the_approval_but_never_laxer() -> None:
    runner, _, _ = make_runner()

    # Stricter is allowed by the gate; this one then fails on the data itself, which is
    # the point of asking for it.
    with pytest.raises(ReplayLimitError, match="limit policy is 'reject'"):
        runner.preflight(request(limit_policy=LimitPolicy.REJECT, max_limit_deviation_deg=0.0))

    with pytest.raises(ReplayLimitError, match="exceeds the approved"):
        runner.preflight(request(max_limit_deviation_deg=5.0))
    with pytest.raises(ReplayLimitError, match="unbounded clamping is not approved"):
        runner.preflight(request(limit_policy=LimitPolicy.CLAMP, max_limit_deviation_deg=1.0))


# ── preflight ──────────────────────────────────────────────────────────────────────
def test_preflight_reports_what_an_operator_needs_to_decide() -> None:
    runner, arm, _ = make_runner()

    plan, _ = runner.preflight(request())
    report = plan.report.to_json()

    assert report["frames_planned"] == 12
    assert report["fps"] == 30
    assert report["robot_type"] == "maker_follower"
    assert report["task"] == "pick up the carabiner and hook into rope"
    assert report["limit_policy"] == "clamp_within_tolerance"
    assert report["frame_offsets_deg"] == [round(v, 3) for v in OFFSETS]
    assert report["max_step_deg"] >= 0
    assert len(report["action_min_deg"]) == 7
    # Preflight is a paper exercise: nothing is claimed and nothing is energized.
    assert arm.lifecycle is ArmLifecycle.DISCONNECTED
    assert arm.sent_commands == []


def test_the_reject_policy_refuses_this_datasets_known_endpoints() -> None:
    runner, _, _ = make_runner()

    with pytest.raises(ReplayLimitError, match="limit policy is 'reject'"):
        runner.preflight(request(limit_policy=LimitPolicy.REJECT, max_limit_deviation_deg=0.0))


def test_a_tolerant_clamp_accepts_them_and_reports_the_deviation() -> None:
    runner, _, _ = make_runner()

    plan, _ = runner.preflight(request())

    # The known endpoint differences are fractions of a degree.
    assert plan.report.clipped_frames > 0
    assert 0 < plan.report.max_clipping_deg <= 1.0
    assert "elbow_flex.pos" in plan.report.clipped_joints
    # The recorded start pose sits further out than any action does.
    assert plan.report.initial_state_clipping_deg == pytest.approx(1.70, abs=0.05)


def test_a_tolerance_smaller_than_the_deviation_fails() -> None:
    runner, _, _ = make_runner()

    with pytest.raises(ReplayLimitError, match="beyond the 0.1 deg clamping tolerance"):
        runner.preflight(request(max_limit_deviation_deg=0.1))


def test_an_out_of_range_frame_selection_is_refused() -> None:
    runner, _, _ = make_runner()

    with pytest.raises(DatasetCompatibilityError, match="outside episode"):
        runner.preflight(request(start_frame=50))
    with pytest.raises(DatasetCompatibilityError, match="empty or outside"):
        runner.preflight(request(start_frame=5, end_frame=2))


def test_speed_is_bounded_to_the_recorded_rate() -> None:
    runner, _, _ = make_runner()

    for speed in (0.0, -1.0, 1.5):
        with pytest.raises(DatasetCompatibilityError, match="speed"):
            runner.preflight(request(speed=speed))


def test_an_fps_above_the_ceiling_is_refused() -> None:
    runner, _, _ = make_runner(params=parameters(max_fps=10))

    with pytest.raises(DatasetCompatibilityError, match="exceeds the configured ceiling"):
        runner.preflight(request())


def test_a_configured_step_bound_rejects_a_jumpy_episode() -> None:
    runner, _, _ = make_runner(params=parameters(max_step_deg=0.25))

    # The plan's proposed 0.25 deg is far below what this dataset contains.
    with pytest.raises(ReplayLimitError, match="max_step_deg"):
        runner.preflight(request())


def test_a_mismatched_joint_set_is_refused() -> None:
    """A robot whose joints differ from the recording's cannot replay it."""

    six = JOINTS[:6]
    document = {
        "schema_version": 1,
        "robot": {
            "id": "maker-arm-02",
            "model": "maker-arm-v1",
            "calibration_id": CALIBRATION,
            "joint_order": list(six),
            "units": "radians",
        },
        "motion_defaults": {
            "max_velocity_rad_s": 2.0,
            "max_acceleration_rad_s2": 8.0,
            "tolerance_rad": 0.02,
            "settle_time_s": 0.1,
            "timeout_s": 30.0,
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
            "require_approved_dataset": False,
        },
        "lerobot_frame": {
            "approved_by": "test",
            "captured_at": "2026-08-29",
            "joints": dict(zip(six, OFFSETS[:6], strict=True)),
        },
    }
    clock = ManualClock()
    arm = FakeArm(
        identity=RobotIdentity("maker-arm-02", "maker-arm-v1", CALIBRATION, six),
        joint_limits=SDK_LIMITS[:6],
        clock=clock,
    )
    runner = ReplayRunner(
        arm,
        RobotParameters.from_mapping(document, config_digest="sha256:test", source="test.yaml"),
        resolver=FixtureResolver(),
        lease=InMemoryLease("maker-arm-02"),
        clock=clock,
    )

    with pytest.raises(DatasetCompatibilityError, match="do not match this robot"):
        runner.preflight(request())


# ── dry run ────────────────────────────────────────────────────────────────────────
def test_a_dry_run_validates_everything_and_touches_nothing() -> None:
    runner, arm, _ = make_runner()
    lease = runner.lease

    result = runner.run(request(dry_run=True))

    assert result.completed
    assert result.dry_run
    assert result.frames_sent == 0
    assert result.frames_planned == 12
    assert arm.lifecycle is ArmLifecycle.DISCONNECTED
    assert arm.sent_commands == []
    assert not lease.held  # type: ignore[union-attr]


# ── the replay itself ──────────────────────────────────────────────────────────────
def test_a_replay_sends_every_frame_in_order_at_the_recorded_cadence() -> None:
    runner, arm, clock = make_runner()
    episode = LeRobotV3Reader(DatasetSnapshot(REPO, REVISION, FIXTURE)).read_episode(0)

    started = clock.monotonic()
    result = runner.run(request())

    assert result.completed
    assert result.frames_sent == 12
    assert result.last_frame_sent == 11
    assert result.stopped_reason is None

    # 12 frames at 30 fps, plus the alignment that ran first.
    assert result.elapsed_s == pytest.approx(12 / 30, abs=1e-6)
    assert clock.monotonic() > started
    assert result.effective_fps == pytest.approx(30, abs=0.5)

    # Every commanded pose is the recorded action, converted into this arm's frame.
    frame = JointFrame(JOINTS, offsets_deg=OFFSETS)
    replayed = arm.sent_commands[-12:]
    for index, command in enumerate(replayed):
        expected = frame.to_radians(episode.actions[index])
        assert command == pytest.approx(expected, abs=0.02)


def test_the_arm_is_aligned_to_the_recorded_start_before_the_first_action() -> None:
    # Park the arm well away from where the episode began.
    runner, arm, _ = make_runner(positions=away_pose_rad())

    result = runner.run(request())

    assert result.completed
    # Alignment commands come first, and by the time replay starts the arm is there.
    assert len(arm.sent_commands) > 12
    aligned = arm.sent_commands[-13]
    assert aligned == pytest.approx(start_pose_rad(), abs=0.05)


def test_alignment_failure_stops_before_any_frame_is_replayed() -> None:
    runner, arm, _ = make_runner(positions=away_pose_rad(), fault_after_commands=2)

    result = runner.run(request())

    assert not result.completed
    assert result.frames_sent == 0
    assert result.stopped_reason == "initial_alignment"


def test_an_unreachable_start_pose_is_reported_not_forced() -> None:
    # Away from the start, and the driver refuses every command it is given.
    runner, arm, _ = make_runner(positions=away_pose_rad(), refuse_targets=True)

    result = runner.run(request())

    assert not result.completed
    assert result.frames_sent == 0
    assert result.stopped_reason == "initial_alignment"


def test_replay_speed_scales_the_control_period() -> None:
    runner, _, clock = make_runner()

    result = runner.run(request(speed=0.5))

    assert result.completed
    assert result.elapsed_s == pytest.approx(12 / 15, abs=1e-6)


def test_a_frame_range_replays_only_the_selected_frames() -> None:
    runner, arm, _ = make_runner()

    result = runner.run(request(start_frame=3, end_frame=6))

    assert result.frames_planned == 4
    assert result.frames_sent == 4
    assert result.first_frame == 3
    assert result.last_frame_sent == 6


def test_cancellation_stops_promptly_and_reports_progress() -> None:
    runner, arm, _ = make_runner()
    seen = {"count": 0}

    def cancelled() -> bool:
        seen["count"] += 1
        return seen["count"] > 4

    result = runner.run(request(), ReplayControl(cancelled=cancelled))

    assert not result.completed
    assert result.stopped_reason == "replay_cancelled"
    # Partial progress survives the stop; an operator needs to know how far it got.
    assert result.frames_sent == 4
    assert result.last_frame_sent == 3
    assert result.elapsed_s > 0


def test_a_motor_fault_stops_the_replay_and_holds() -> None:
    runner, arm, _ = make_runner(fault_after_commands=3)

    result = runner.run(request())

    assert not result.completed
    assert result.stopped_reason == "robot_fault"
    assert result.frames_sent < 12


def test_heartbeats_report_progress_on_elapsed_time() -> None:
    beats: list[object] = []
    runner, _, _ = make_runner()

    result = runner.run(request(), ReplayControl(heartbeat=beats.append))

    assert result.completed
    # 12 frames at 30 fps is 0.4 s; with a 0.2 s interval that is at least one beat.
    assert beats
    assert max(getattr(beat, "frames_sent", 0) for beat in beats) > 0


def test_clipping_is_measured_against_the_datasets_own_values() -> None:
    runner, _, _ = make_runner()

    result = runner.run(request())

    # The same endpoint deviations preflight found, now attributed to sent frames.
    assert result.clipped_frames > 0
    assert 0 < result.max_clipping_deg <= 1.0


def test_the_lease_is_released_and_the_arm_left_safe_afterwards() -> None:
    runner, arm, _ = make_runner()
    lease = runner.lease

    runner.run(request())

    assert not lease.held  # type: ignore[union-attr]
    assert arm.lifecycle is ArmLifecycle.DISCONNECTED


def test_a_busy_robot_cannot_be_replayed_twice_at_once() -> None:
    from everest_robot.robot.lease import RobotBusy

    runner, _, _ = make_runner()
    holder = InMemoryLease("maker-arm-02")
    holder.acquire()

    try:
        with pytest.raises(RobotBusy):
            runner.run(request())
    finally:
        holder.release()


def test_the_result_round_trips_through_json() -> None:
    runner, _, _ = make_runner()

    result = runner.run(request())
    encoded = json_dict(result)

    assert ReplayResult(**encoded) == result
    assert encoded["config_digest"] == "sha256:test"
    assert encoded["repo_id"] == REPO


def test_replay_never_sends_the_recorded_observation_instead_of_the_action() -> None:
    """The action column is what the robot was commanded; state is where it ended up."""

    runner, arm, _ = make_runner()
    episode = LeRobotV3Reader(DatasetSnapshot(REPO, REVISION, FIXTURE)).read_episode(0)
    frame = JointFrame(JOINTS, offsets_deg=OFFSETS)

    runner.run(request())

    first = arm.sent_commands[-12]
    assert first == pytest.approx(frame.to_radians(episode.actions[0]), abs=0.02)
    assert first != pytest.approx(frame.to_radians(episode.states[0]), abs=1e-6)


def test_a_slow_tick_is_absorbed_rather_than_burst_afterwards() -> None:
    """Replay commands absolute positions: losing time is safe, catching up is not."""

    runner, arm, clock = make_runner()
    stalls = {"count": 0}
    real_read = arm.read_state

    def slow_read():
        stalls["count"] += 1
        if stalls["count"] == 6:
            clock.advance(0.5)  # one tick takes far longer than its 33 ms period
        return real_read()

    arm.read_state = slow_read  # type: ignore[method-assign]

    started = clock.monotonic()
    result = runner.run(request())
    total = clock.monotonic() - started

    assert result.completed
    # The stall is absorbed into the total, not clawed back by rushing later frames.
    assert total >= 0.5 + 11 / 30 - 1e-9
    assert result.elapsed_s >= 0.5


def _fixture_copy(tmp_path: Path) -> Path:
    import shutil

    root = tmp_path / "dataset"
    shutil.copytree(FIXTURE, root)
    return root


def test_a_non_finite_action_fails_before_the_arm_is_touched(tmp_path: Path) -> None:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = _fixture_copy(tmp_path)
    path = root / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    actions = np.array(table.column("action").to_pylist())
    actions[4, 2] = np.nan
    table = table.set_column(
        table.schema.get_field_index("action"),
        "action",
        pa.array([list(row) for row in actions], type=pa.list_(pa.float32(), 7)),
    )
    pq.write_table(table, path)

    runner, arm, _ = make_runner()
    runner.resolver = FixtureResolver(root)

    with pytest.raises(DatasetCompatibilityError, match="non-finite value at frame 4"):
        runner.preflight(request())

    assert arm.lifecycle is ArmLifecycle.DISCONNECTED
    assert arm.sent_commands == []


def test_a_non_positive_dataset_fps_is_refused(tmp_path: Path) -> None:
    import json

    for fps in (0, -30):
        root = _fixture_copy(tmp_path / str(fps))
        info = json.loads((root / "meta/info.json").read_text())
        info["fps"] = fps
        (root / "meta/info.json").write_text(json.dumps(info))

        runner, _, _ = make_runner()
        runner.resolver = FixtureResolver(root)

        with pytest.raises(DatasetCompatibilityError, match="fps must be positive"):
            runner.preflight(request())


def test_clipping_metrics_come_from_what_the_driver_actually_accepted(monkeypatch) -> None:
    """The metric is sourced from the action the driver returned, not from preflight."""

    import everest_robot.robot.replay as replay_module

    class ShavingPlayer(replay_module.SessionPlayer):
        """A driver that quietly shaves half a degree off one joint on every frame."""

        def play(self, plan, request, outcome):
            accepted = self.session.bridge.send_action

            def shave(action):
                sent = accepted(action)
                return {**sent, "shoulder_pan.pos": sent["shoulder_pan.pos"] - 0.5}

            self.session.bridge.send_action = shave  # type: ignore[method-assign]
            return super().play(plan, request, outcome)

    monkeypatch.setattr(replay_module, "SessionPlayer", ShavingPlayer)
    runner, _, _ = make_runner()

    result = runner.run(request())

    assert result.completed
    assert result.clipped_frames == result.frames_sent
    assert result.max_clipping_deg >= 0.5
