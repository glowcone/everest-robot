"""Recording is still interface-only; these tests pin that contract.

Replay is implemented -- see test_replay.py.
"""

import pytest

from everest_robot.robot.contracts import RobotIdentity
from everest_robot.robot.recording import (
    InMemorySessionRecorder,
    LeRobotDatasetRecorder,
    NullSessionRecorder,
    SessionRecorder,
)

IDENTITY = RobotIdentity("maker-arm-02", "maker-arm-v1", "cal-2026-08-20", ("a", "b"))


def test_in_memory_recorder_satisfies_the_protocol() -> None:
    assert isinstance(InMemorySessionRecorder(IDENTITY), SessionRecorder)
    assert isinstance(NullSessionRecorder(IDENTITY), SessionRecorder)


def test_a_recorded_frame_keeps_both_the_requested_and_the_sent_action() -> None:
    recorder = InMemorySessionRecorder(IDENTITY, fps=30.0, config_digest="sha256:test")
    recorder.start_episode(task="attach the clip")

    recorder.record_frame({"a.pos": 1.0}, {"a.pos": 90.0}, {"a.pos": 45.0}, timestamp_s=0.0)
    result = recorder.finish_episode()

    frame = recorder.frames[0]
    # Replaying a requested action would replay a command the robot never executed.
    assert frame.requested_action == {"a.pos": 90.0}
    assert frame.sent_action == {"a.pos": 45.0}
    assert result.frames == 1
    assert result.task == "attach the clip"
    assert result.to_json()["config_digest"] == "sha256:test"


def test_recording_outside_an_episode_is_an_error() -> None:
    recorder = InMemorySessionRecorder(IDENTITY)

    with pytest.raises(RuntimeError, match="outside an episode"):
        recorder.record_frame({}, {}, {}, 0.0)


def test_the_null_recorder_counts_frames_without_keeping_them() -> None:
    recorder = NullSessionRecorder(IDENTITY)
    recorder.start_episode()

    recorder.record_frame({"a.pos": 1.0}, {"a.pos": 1.0}, {"a.pos": 1.0}, 0.0)
    result = recorder.finish_episode()

    assert result.frames == 1
    assert recorder.frames == []


def test_dataset_recording_refuses_with_an_actionable_message() -> None:
    with pytest.raises(NotImplementedError, match="format is still being decided"):
        LeRobotDatasetRecorder()
