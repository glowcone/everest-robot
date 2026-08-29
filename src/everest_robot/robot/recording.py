"""Session recording.

**Status: interface only.** The canonical stored-session format is not settled yet, so
:class:`LeRobotDatasetRecorder` deliberately refuses rather than writing a format we would
have to migrate. The protocol and the in-memory recorder are real, because the policy
runner needs somewhere to put frames today.

What a recorded frame has to carry, whatever the eventual container:

* the observation actually used for the decision (joints and camera frames);
* the action the policy or teleoperator *requested*;
* the action actually *sent* after clipping and safety limiting -- replaying requested
  actions would replay commands the robot never executed;
* the timestamp and nominal FPS;
* the task/language input; and
* robot identity, calibration, parameters digest and software revisions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

from everest_robot.robot.contracts import RecordingResult, RobotIdentity


@dataclass(frozen=True, slots=True)
class RecordedFrame:
    """One synchronized decision: what was seen, what was asked for, what was sent."""

    index: int
    timestamp_s: float
    observation: Mapping[str, Any]
    requested_action: Mapping[str, float]
    sent_action: Mapping[str, float]


@runtime_checkable
class SessionRecorder(Protocol):
    """Where a rollout's frames go. Implementations must tolerate an aborted episode."""

    def start_episode(self, task: str | None = None) -> None: ...

    def record_frame(
        self,
        observation: Mapping[str, Any],
        requested_action: Mapping[str, float],
        sent_action: Mapping[str, float],
        timestamp_s: float,
    ) -> None: ...

    def finish_episode(self) -> RecordingResult: ...

    def abort(self) -> None: ...


@dataclass
class InMemorySessionRecorder:
    """Keeps frames in memory. Used by tests and by hardware-free workflow runs.

    Not a storage format: nothing here survives the process.
    """

    identity: RobotIdentity
    fps: float = 30.0
    config_digest: str = ""
    session_id: str = "in-memory"
    revisions: dict[str, str] = field(default_factory=dict)
    frames: list[RecordedFrame] = field(default_factory=list)
    task: str | None = None
    episode_index: int = -1
    _open: bool = False

    def start_episode(self, task: str | None = None) -> None:
        self.task = task
        self.frames = []
        self.episode_index += 1
        self._open = True

    def record_frame(
        self,
        observation: Mapping[str, Any],
        requested_action: Mapping[str, float],
        sent_action: Mapping[str, float],
        timestamp_s: float,
    ) -> None:
        if not self._open:
            raise RuntimeError("record_frame() called outside an episode")
        self.frames.append(
            RecordedFrame(
                index=len(self.frames),
                timestamp_s=timestamp_s,
                observation=dict(observation),
                requested_action=dict(requested_action),
                sent_action=dict(sent_action),
            )
        )

    def finish_episode(self) -> RecordingResult:
        self._open = False
        return RecordingResult(
            session_id=self.session_id,
            dataset_path="",
            episode_index=max(self.episode_index, 0),
            frames=len(self.frames),
            fps=self.fps,
            task=self.task,
            robot_id=self.identity.robot_id,
            calibration_id=self.identity.calibration_id,
            config_digest=self.config_digest,
            revisions=dict(self.revisions),
        )

    def abort(self) -> None:
        self._open = False
        self.frames = []


class NullSessionRecorder(InMemorySessionRecorder):
    """Counts frames and keeps none. The default when a rollout is not being recorded."""

    def record_frame(
        self,
        observation: Mapping[str, Any],
        requested_action: Mapping[str, float],
        sent_action: Mapping[str, float],
        timestamp_s: float,
    ) -> None:
        if not self._open:
            raise RuntimeError("record_frame() called outside an episode")
        self._counted = getattr(self, "_counted", 0) + 1

    def finish_episode(self) -> RecordingResult:
        counted = getattr(self, "_counted", 0)
        result = super().finish_episode()
        self._counted = 0
        return replace(result, frames=counted)


class LeRobotDatasetRecorder:
    """Placeholder for the LeRobotDataset-backed recorder.

    Refuses rather than guessing: the episode schema, the video encoding and the
    normalization metadata a policy will later read all depend on the dataset decision
    still being made, and a half-chosen format on disk is worse than none.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "LeRobot dataset recording is not implemented yet: the stored-session format "
            "is still being decided. Use InMemorySessionRecorder for hardware-free runs, "
            "and see docs/adr/ for the format decision when it lands."
        )
