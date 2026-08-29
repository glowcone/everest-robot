"""Guarded session replay.

**Status: interface only.** Replay is blocked on the same stored-session format decision
as :mod:`everest_robot.robot.recording`, so :meth:`SessionPlayer.play` refuses. The
contract is written down here because the safety requirements are already known and are
not negotiable once the format lands:

* validate schema, robot identity, joint names and units *before* anything is enabled;
* validate that every frame in the requested range is finite and inside the active
  hardware limits -- the whole range, not frame by frame as it plays;
* support a dry run that does all of the above and commands nothing;
* interpolate safely from the current pose to the episode's first frame, under the same
  bounded motion rules as any other move, instead of jumping to it;
* pace frames from a monotonic clock, never by accumulating sleeps;
* honour frame-range and speed controls, cancellation, timeout, hold and e-stop; and
* replay the actions that were actually *sent*, not raw leader values or unprocessed
  policy outputs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from everest_robot.robot.clock import Clock, SystemClock
from everest_robot.robot.contracts import CancelCheck, Heartbeat, ReplayResult
from everest_robot.robot.lerobot_bridge import RobotBridgeCore
from everest_robot.robot.parameters import RobotParameters


@dataclass(frozen=True, slots=True)
class SessionMetadata:
    """What a stored session must state about itself before it may be replayed."""

    session_id: str
    robot_id: str
    calibration_id: str
    joint_names: tuple[str, ...]
    units: str
    fps: float
    frames: int
    task: str | None = None
    config_digest: str = ""
    revisions: Mapping[str, str] = field(default_factory=dict)


class SessionPlayer:
    """Replays a stored session onto the arm.

    Construction is real so callers and the workflow can be wired up now; playback
    refuses until the session format is decided.
    """

    def __init__(
        self,
        bridge: RobotBridgeCore,
        parameters: RobotParameters,
        *,
        clock: Clock | None = None,
        heartbeat: Heartbeat | None = None,
        cancel: CancelCheck | None = None,
    ) -> None:
        self.bridge = bridge
        self.parameters = parameters
        self.clock = clock or SystemClock()
        self.heartbeat = heartbeat
        self.cancel = cancel

    def play(
        self,
        session: Any,
        *,
        frame_range: Sequence[int] | None = None,
        speed_scale: float = 1.0,
        dry_run: bool = False,
    ) -> ReplayResult:
        raise NotImplementedError(
            "session replay is not implemented yet: the stored-session format is still "
            "being decided. The validation and pacing requirements are documented in "
            "everest_robot.robot.replay."
        )
