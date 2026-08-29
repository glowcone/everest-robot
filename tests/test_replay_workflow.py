"""Workflow-level replay behaviour: one checkpoint, no blind retries, exclusive claim."""

from collections.abc import Callable
from typing import Any

import pytest

from everest_robot.domain import LimitPolicy, ReplayRequest, ReplayResult, json_dict
from everest_robot.workflow import run_replay_session

PARAMS = {
    "repo_id": "h8i76dfsd9/test1_20260829_130743",
    "revision": "55e561161026be306d06354a8941c4431e8e805f",
    "episode": 0,
    "robot_id": "maker-arm-02",
    "calibration_id": "maker-arm-02-2026-08-20",
    "limit_policy": "clamp_within_tolerance",
    "max_limit_deviation_deg": 2.0,
}


def a_result(**overrides: Any) -> ReplayResult:
    values: dict[str, Any] = {
        "repo_id": PARAMS["repo_id"],
        "revision": PARAMS["revision"],
        "episode": 0,
        "robot_id": "maker-arm-02",
        "calibration_id": "maker-arm-02-2026-08-20",
        "completed": True,
        "frames_planned": 12,
        "frames_sent": 12,
        "first_frame": 0,
        "last_frame_sent": 11,
        "elapsed_s": 0.4,
        "effective_fps": 30.0,
        "clipped_frames": 2,
        "max_clipping_deg": 0.8,
    }
    values.update(overrides)
    return ReplayResult(**values)


class RecordingContext:
    """Executes every step, recording its name. Stands in for a first attempt."""

    def __init__(self) -> None:
        self.steps: list[str] = []
        self.heartbeats = 0

    def step(self, name: str, operation: Callable[[], Any]) -> Any:
        self.steps.append(name)
        return operation()

    def heartbeat(self, seconds: int | None = None) -> None:
        self.heartbeats += 1


class ReplayingContext(RecordingContext):
    """Returns a stored checkpoint instead of executing, as Absurd does on a retry."""

    def __init__(self, stored: dict[str, Any]) -> None:
        super().__init__()
        self.stored = stored

    def step(self, name: str, operation: Callable[[], Any]) -> Any:
        self.steps.append(name)
        return self.stored


def test_a_replay_is_one_checkpoint_not_one_per_frame(monkeypatch) -> None:
    calls: list[ReplayRequest] = []

    def fake(request, control=None, **kwargs):
        calls.append(request)
        return a_result()

    monkeypatch.setattr("everest_robot.workflow.replay_session", fake)
    context = RecordingContext()

    result = run_replay_session(dict(PARAMS), context)

    assert context.steps == ["01-replay-session"]
    assert result["frames_sent"] == 12
    assert calls[0].episode == 0
    assert calls[0].limit_policy is LimitPolicy.CLAMP_WITHIN_TOLERANCE


def test_a_completed_replay_is_not_physically_repeated(monkeypatch) -> None:
    def explode(request, control=None, **kwargs):
        raise AssertionError("a stored checkpoint must not re-execute physical motion")

    monkeypatch.setattr("everest_robot.workflow.replay_session", explode)
    stored = json_dict(a_result())
    context = ReplayingContext(stored)

    result = run_replay_session(dict(PARAMS), context)

    assert result == stored
    assert context.steps == ["01-replay-session"]


def test_the_workflow_heartbeat_is_wired_to_replay_progress(monkeypatch) -> None:
    from everest_robot.robot.replay import ReplayProgress

    def fake(request, control=None, **kwargs):
        control.beat(ReplayProgress(3, 12, 4, 0.2, 0))
        control.beat(ReplayProgress(6, 12, 7, 0.4, 0))
        return a_result()

    monkeypatch.setattr("everest_robot.workflow.replay_session", fake)
    context = RecordingContext()

    run_replay_session(dict(PARAMS), context)

    assert context.heartbeats == 2


def test_cancellation_raised_from_the_heartbeat_propagates(monkeypatch) -> None:
    """Absurd signals cancellation by raising out of ctx.heartbeat()."""

    from everest_robot.robot.replay import ReplayProgress

    class Cancelled(BaseException):
        pass

    class CancellingContext(RecordingContext):
        def heartbeat(self, seconds: int | None = None) -> None:
            raise Cancelled("run cancelled")

    def fake(request, control=None, **kwargs):
        control.beat(ReplayProgress(3, 12, 4, 0.2, 0))
        raise AssertionError("the heartbeat should have stopped this")

    monkeypatch.setattr("everest_robot.workflow.replay_session", fake)

    with pytest.raises(Cancelled):
        run_replay_session(dict(PARAMS), CancellingContext())


def test_an_unrecognized_parameter_fails_before_anything_runs(monkeypatch) -> None:
    def explode(request, control=None, **kwargs):
        raise AssertionError("must not reach the adapter")

    monkeypatch.setattr("everest_robot.workflow.replay_session", explode)

    with pytest.raises(ValueError, match="unknown replay parameter"):
        run_replay_session({**PARAMS, "dryrun": True}, RecordingContext())


def test_the_replay_task_is_registered_without_blind_retries() -> None:
    registered: dict[str, dict[str, Any]] = {}

    class FakeApp:
        def __init__(self, queue_name: str) -> None:
            self.queue_name = queue_name

        def register_task(self, name: str, **kwargs: Any):
            registered[name] = kwargs
            return lambda fn: fn

    import everest_robot.workflow as workflow

    original = workflow.Absurd
    workflow.Absurd = FakeApp  # type: ignore[assignment]
    try:
        workflow.create_app()
    finally:
        workflow.Absurd = original  # type: ignore[assignment]

    # A replay interrupted mid-episode leaves the arm in an unknown pose; restarting from
    # frame zero is a different physical motion, not a retry.
    assert registered["replay-session"]["default_max_attempts"] == 1
    assert registered["attach-carabiner"]["default_max_attempts"] == 10
