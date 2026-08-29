"""Guarded replay of a stored LeRobot episode.

The shape of the operation is: resolve an immutable dataset revision, prove the episode is
compatible and in range *before* any hardware is claimed, align the arm to the pose the
episode actually started from, then send the recorded actions at the recorded cadence
while watching for faults, drift and cancellation.

Two rules drive most of the design:

* **Nothing physical happens until the dataset has been fully checked.** Preflight reads
  every frame of the selected range and validates it against the active hardware limits.
  Discovering a bad episode after the arm is claimed and energized is avoidable.
* **A late frame is never made up.** Replay commands absolute positions, so slipping in
  time is harmless -- the arm simply moves slower. Commanding the backlog faster is not:
  it would drive the arm through the recorded path at a speed nobody validated. This is
  the opposite of the policy runner's absolute-deadline pacing, deliberately.

What is replayed is the dataset's ``action`` column -- the commands the recording robot was
actually given -- not ``observation.state``, which is where it happened to end up.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from everest_robot.domain import LimitPolicy, ReplayRequest, ReplayResult
from everest_robot.robot.clock import Clock, SystemClock
from everest_robot.robot.contracts import ArmLifecycle, JointLimit, MotionProfile
from everest_robot.robot.datasets import (
    Episode,
    HuggingFaceDatasetResolver,
    LeRobotV3Reader,
)
from everest_robot.robot.errors import (
    CalibrationMismatchError,
    DatasetCompatibilityError,
    InitialAlignmentError,
    ReplayCancelled,
    ReplayLimitError,
    ReplayTimingError,
    RobotFaultError,
)
from everest_robot.robot.lease import InMemoryLease, RobotLease
from everest_robot.robot.lerobot_bridge import POSITION_SUFFIX, JointFrame
from everest_robot.robot.parameters import ApprovedReplay, RobotParameters
from everest_robot.robot.ports import ArmPort
from everest_robot.robot.session import RobotSession

# Robot types whose datasets share this arm's joint conventions. Anything else has to be
# named explicitly by the caller rather than assumed compatible.
DEFAULT_COMPATIBLE_ROBOT_TYPES = ("maker_follower", "everest_maker_arm")

_RAD_TO_DEG = 180.0 / math.pi


@dataclass(frozen=True, slots=True)
class ReplayProgress:
    """What a heartbeat reports. Process-local; never persisted."""

    frame: int
    frames_planned: int
    frames_sent: int
    elapsed_s: float
    clipped_frames: int


@dataclass
class ReplayControl:
    """Process-local callbacks a workflow supplies. Not serializable, by nature."""

    cancelled: Any = None
    heartbeat: Any = None

    def is_cancelled(self) -> bool:
        return bool(self.cancelled()) if callable(self.cancelled) else False

    def beat(self, progress: ReplayProgress) -> None:
        if callable(self.heartbeat):
            self.heartbeat(progress)


@dataclass(frozen=True, slots=True)
class PreflightReport:
    """Everything an operator needs to decide whether to run this replay.

    Produced by a dry run as well as a real one, and JSON-serializable so it can be logged
    or attached to an approval request.
    """

    repo_id: str
    revision: str
    episode: int
    task: str | None
    robot_type: str | None
    fps: float
    frames_planned: int
    first_frame: int
    last_frame: int
    duration_s: float
    joint_names: tuple[str, ...]
    action_min_deg: tuple[float, ...]
    action_max_deg: tuple[float, ...]
    limit_lower_deg: tuple[float, ...]
    limit_upper_deg: tuple[float, ...]
    max_step_deg: float
    max_step_joint: str
    limit_policy: LimitPolicy
    clipped_frames: int
    max_clipping_deg: float
    clipped_joints: tuple[str, ...]
    initial_state_deg: tuple[float, ...]
    initial_target_deg: tuple[float, ...]
    initial_state_clipping_deg: float
    first_action_deg: tuple[float, ...]
    frame_offsets_deg: tuple[float, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "episode": self.episode,
            "task": self.task,
            "robot_type": self.robot_type,
            "fps": self.fps,
            "frames_planned": self.frames_planned,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "duration_s": self.duration_s,
            "joint_names": list(self.joint_names),
            "action_min_deg": [round(v, 3) for v in self.action_min_deg],
            "action_max_deg": [round(v, 3) for v in self.action_max_deg],
            "limit_lower_deg": [round(v, 3) for v in self.limit_lower_deg],
            "limit_upper_deg": [round(v, 3) for v in self.limit_upper_deg],
            "max_step_deg": round(self.max_step_deg, 3),
            "max_step_joint": self.max_step_joint,
            "limit_policy": str(self.limit_policy),
            "clipped_frames": self.clipped_frames,
            "max_clipping_deg": round(self.max_clipping_deg, 4),
            "clipped_joints": list(self.clipped_joints),
            "initial_state_deg": [round(v, 3) for v in self.initial_state_deg],
            "initial_target_deg": [round(v, 3) for v in self.initial_target_deg],
            "initial_state_clipping_deg": round(self.initial_state_clipping_deg, 4),
            "first_action_deg": [round(v, 3) for v in self.first_action_deg],
            "frame_offsets_deg": [round(v, 3) for v in self.frame_offsets_deg],
        }


@dataclass(frozen=True, slots=True)
class ReplayPlan:
    """A validated, ready-to-send frame range.

    ``actions_deg`` is post-clamp: whatever the limit policy decided has already been
    applied, so the loop sends exactly these values and does not re-decide per frame.
    ``requested_deg`` keeps the dataset's originals so clipping is measured against what
    was recorded, not against what preflight already adjusted.
    """

    report: PreflightReport
    episode: Episode
    joint_names: tuple[str, ...]
    frame_indices: np.ndarray
    actions_deg: np.ndarray
    requested_deg: np.ndarray
    #: The pose alignment actually drives to: the recorded initial state after limit
    #: clamping. Commanding the raw value would be clamped by the driver anyway, so the
    #: arm would silently start somewhere other than where the caller was told.
    initial_state_deg: np.ndarray

    def __len__(self) -> int:
        return int(self.actions_deg.shape[0])


def limits_in_degrees(
    limits: Sequence[JointLimit], frame: JointFrame
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Express the driver's radian soft limits in the dataset's degree frame.

    The conversion must stay order-preserving. A frame that inverted a joint would turn a
    lower bound into an upper one, and every subsequent range check would be backwards, so
    that is refused rather than silently sorted.
    """

    lower = frame.to_degrees([limit.lower_rad for limit in limits])
    upper = frame.to_degrees([limit.upper_rad for limit in limits])
    inverted = [
        limit.name for limit, lo, hi in zip(limits, lower, upper, strict=True) if lo >= hi
    ]
    if inverted:
        raise DatasetCompatibilityError(
            f"the LeRobot joint frame inverts {', '.join(inverted)}; a direction-flipping "
            "frame is not supported"
        )
    return lower, upper


class ReplayPreflight:
    """Validates an episode against this robot before anything is claimed or energized."""

    def __init__(
        self,
        parameters: RobotParameters,
        *,
        frame: JointFrame,
        limits: Sequence[JointLimit],
        compatible_robot_types: Sequence[str] = DEFAULT_COMPATIBLE_ROBOT_TYPES,
    ) -> None:
        self.parameters = parameters
        self.frame = frame
        self.limits = tuple(limits)
        self.compatible_robot_types = tuple(compatible_robot_types)

    def check(
        self,
        episode: Episode,
        request: ReplayRequest,
        approval: ApprovedReplay | None,
    ) -> ReplayPlan:
        settings = self.parameters.replay
        metadata = episode.metadata

        self._check_metadata(episode)
        first, last = self._resolve_range(episode, request)
        policy, tolerance = self._resolve_policy(request, approval)

        actions = episode.actions[first : last + 1]
        states = episode.states[first : last + 1]
        frames = episode.frame_indices[first : last + 1]
        self._check_finite(actions, "action", first)
        self._check_finite(states[:1], "observation.state", first)

        lower, upper = limits_in_degrees(self.limits, self.frame)
        clamped, clipped_frames, max_clipping, clipped_joints = self._apply_limits(
            actions, lower, upper, policy, tolerance, episode.joint_names, first
        )
        # The recorded start pose gets the same treatment as the actions. A recording made
        # on a driver with slightly wider limits can start just outside this one's, and
        # since the driver would clamp the command anyway, an unclamped target would mean
        # the arm quietly begins somewhere other than the caller was told.
        initial, _, initial_clipping, _ = self._apply_limits(
            states[:1], lower, upper, policy, tolerance, episode.joint_names, first,
            what="the recorded initial state, frame",
        )

        steps = np.abs(np.diff(actions, axis=0)) if len(actions) > 1 else np.zeros_like(actions)
        max_step = float(steps.max()) if steps.size else 0.0
        step_joint = episode.joint_names[int(steps.max(axis=0).argmax())] if steps.size else ""
        if settings.max_step_deg is not None and max_step > settings.max_step_deg:
            raise ReplayLimitError(
                f"episode {request.episode}: a frame-to-frame action step of {max_step:.2f} deg "
                f"on {step_joint} exceeds the configured max_step_deg "
                f"({settings.max_step_deg}); the arm would be asked to jump"
            )

        report = PreflightReport(
            repo_id=metadata.repo_id,
            revision=metadata.revision,
            episode=metadata.episode,
            task=metadata.task,
            robot_type=metadata.robot_type,
            fps=metadata.fps,
            frames_planned=int(actions.shape[0]),
            first_frame=first,
            last_frame=last,
            duration_s=float(actions.shape[0]) / metadata.fps / max(request.speed, 1e-9),
            joint_names=episode.joint_names,
            action_min_deg=tuple(float(v) for v in actions.min(axis=0)),
            action_max_deg=tuple(float(v) for v in actions.max(axis=0)),
            limit_lower_deg=lower,
            limit_upper_deg=upper,
            max_step_deg=max_step,
            max_step_joint=step_joint,
            limit_policy=policy,
            clipped_frames=clipped_frames,
            max_clipping_deg=max_clipping,
            clipped_joints=clipped_joints,
            initial_state_deg=tuple(float(v) for v in states[0]),
            initial_target_deg=tuple(float(v) for v in initial[0]),
            initial_state_clipping_deg=initial_clipping,
            first_action_deg=tuple(float(v) for v in actions[0]),
            frame_offsets_deg=self.frame.offsets_deg,
        )
        return ReplayPlan(
            report=report,
            episode=episode,
            joint_names=episode.joint_names,
            frame_indices=frames,
            actions_deg=clamped,
            requested_deg=actions,
            initial_state_deg=initial[0],
        )

    # ── individual checks ──────────────────────────────────────────────────────────
    def _check_metadata(self, episode: Episode) -> None:
        metadata = episode.metadata
        settings = self.parameters.replay

        if metadata.robot_type not in self.compatible_robot_types:
            raise DatasetCompatibilityError(
                f"dataset robot_type {metadata.robot_type!r} is not one of "
                f"{', '.join(self.compatible_robot_types)}; its joint conventions cannot be "
                "assumed to match this arm"
            )
        if not metadata.fps > 0:
            raise DatasetCompatibilityError(f"dataset fps must be positive, got {metadata.fps}")
        if metadata.fps > settings.max_fps:
            raise DatasetCompatibilityError(
                f"dataset fps {metadata.fps} exceeds the configured ceiling {settings.max_fps}"
            )

        expected = tuple(f"{name}{POSITION_SUFFIX}" for name in self.parameters.joint_names)
        if episode.joint_names != expected:
            raise DatasetCompatibilityError(
                f"dataset joints {list(episode.joint_names)} do not match this robot's "
                f"{list(expected)}; order and membership must agree exactly"
            )

    def _resolve_range(self, episode: Episode, request: ReplayRequest) -> tuple[int, int]:
        length = len(episode)
        first = request.start_frame
        last = length - 1 if request.end_frame is None else request.end_frame
        if first < 0 or first >= length:
            raise DatasetCompatibilityError(
                f"start_frame {first} is outside episode {request.episode} (0-{length - 1})"
            )
        if last < first or last >= length:
            raise DatasetCompatibilityError(
                f"frame range {first}-{last} is empty or outside episode {request.episode} "
                f"(0-{length - 1})"
            )
        if not 0 < request.speed <= self.parameters.replay.max_speed_scale:
            raise DatasetCompatibilityError(
                f"speed {request.speed} must be in (0, {self.parameters.replay.max_speed_scale}]; "
                "faster-than-recorded replay needs its own hardware acceptance"
            )
        return first, last

    def _resolve_policy(
        self, request: ReplayRequest, approval: ApprovedReplay | None
    ) -> tuple[LimitPolicy, float]:
        """The approved entry caps the request: a caller may be stricter, never laxer."""

        policy, tolerance = request.limit_policy, request.max_limit_deviation_deg
        if approval is None:
            return policy, tolerance
        if tolerance > approval.max_limit_deviation_deg:
            raise ReplayLimitError(
                f"requested clamping tolerance {tolerance} deg exceeds the approved "
                f"{approval.max_limit_deviation_deg} deg for {approval.repo_id}"
            )
        if policy is LimitPolicy.CLAMP and approval.limit_policy is not LimitPolicy.CLAMP:
            raise ReplayLimitError(
                f"unbounded clamping is not approved for {approval.repo_id} "
                f"(approved policy: {approval.limit_policy})"
            )
        return policy, tolerance

    @staticmethod
    def _check_finite(values: np.ndarray, name: str, offset: int) -> None:
        if not np.isfinite(values).all():
            bad = int(np.argmax(~np.isfinite(values).all(axis=1)))
            raise DatasetCompatibilityError(
                f"{name} contains a non-finite value at frame {offset + bad}"
            )

    def _apply_limits(
        self,
        actions: np.ndarray,
        lower: Sequence[float],
        upper: Sequence[float],
        policy: LimitPolicy,
        tolerance: float,
        joint_names: Sequence[str],
        offset: int,
        what: str = "frame",
    ) -> tuple[np.ndarray, int, float, tuple[str, ...]]:
        low = np.asarray(lower, dtype=np.float64)
        high = np.asarray(upper, dtype=np.float64)
        clamped = np.clip(actions, low, high)
        deviation = np.abs(actions - clamped)
        worst = float(deviation.max()) if deviation.size else 0.0
        offending = tuple(
            name
            for index, name in enumerate(joint_names)
            if deviation[:, index].max(initial=0.0) > 0
        )
        frames = int((deviation > 0).any(axis=1).sum())

        if worst == 0.0:
            return clamped, 0, 0.0, ()

        if policy is LimitPolicy.REJECT:
            row = int(np.argmax((deviation > 0).any(axis=1)))
            raise ReplayLimitError(
                f"{what} {offset + row}: {', '.join(offending)} lies outside the active limits "
                f"by up to {worst:.3f} deg and the limit policy is 'reject'"
            )
        if policy is LimitPolicy.CLAMP_WITHIN_TOLERANCE and worst > tolerance:
            row = int(np.argmax(deviation.max(axis=1) > tolerance))
            raise ReplayLimitError(
                f"{what} {offset + row}: {', '.join(offending)} lies {worst:.3f} deg outside the "
                f"active limits, beyond the {tolerance} deg clamping tolerance"
            )
        return clamped, frames, worst, offending


class SessionPlayer:
    """Aligns the arm to an episode's start and replays its actions at the recorded rate.

    Operates on an already-open :class:`~everest_robot.robot.session.RobotSession`, so the
    lease, connection and identity check are somebody else's job -- :class:`ReplayRunner`'s.
    """

    def __init__(
        self,
        session: RobotSession,
        *,
        clock: Clock | None = None,
        control: ReplayControl | None = None,
    ) -> None:
        self.session = session
        self.parameters = session.parameters
        self.clock = clock or session.clock
        self.control = control or ReplayControl()

    # ── alignment ──────────────────────────────────────────────────────────────────
    def align(self, plan: ReplayPlan, approval: ApprovedReplay | None) -> None:
        """Bring the arm to the pose the episode actually started from.

        Sending frame zero from wherever the arm happens to be standing is the mistake this
        exists to prevent: the first recorded action assumes the recorded initial state,
        and the gap between the two is commanded in a single step.
        """

        settings = self.parameters.replay
        frame = self.session.bridge.frame
        targets_rad = frame.to_radians(plan.initial_state_deg)
        tolerance_rad = settings.initial_pose_tolerance_deg / _RAD_TO_DEG

        state = self.session.port.read_state()
        if not state.all_finite:
            raise InitialAlignmentError("joint feedback is missing; cannot align")
        if state.max_tracking_error(targets_rad) <= tolerance_rad:
            return

        motion = self.session.motion
        # A straight line from here to the recorded pose is not automatically safe. Where
        # the operator has approved a staging path, it is taken first and the direct
        # interpolation only covers the last, short leg.
        if approval is not None and approval.initial_transition:
            self._require(motion.follow_transition(approval.initial_transition), "staging")
        elif approval is not None and approval.initial_position:
            self._require(motion.go_to_known_position(approval.initial_position), "staging")

        profile = _alignment_profile(self.parameters, settings)
        self._require(
            motion.go_to_joint_target(
                f"replay-initial-state:{plan.report.repo_id}@{plan.report.episode}",
                targets_rad,
                profile=profile,
            ),
            "initial state",
        )

    @staticmethod
    def _require(result: Any, what: str) -> None:
        if not result.reached:
            raise InitialAlignmentError(
                f"could not reach the {what} pose: {result.failure_reason} "
                f"({result.failure_detail})"
            )

    # ── the timed loop ─────────────────────────────────────────────────────────────
    def play(
        self, plan: ReplayPlan, request: ReplayRequest, outcome: _PlayOutcome
    ) -> _PlayOutcome:
        """Send the planned frames. Fills ``outcome`` as it goes.

        The caller owns the outcome so that a cancellation or fault mid-episode still
        reports how far the arm actually got -- which is the number an operator needs
        before deciding how to recover.
        """

        settings = self.parameters.replay
        bridge = self.session.bridge
        joint_names = plan.joint_names
        period = 1.0 / (plan.report.fps * request.speed)
        tracking_limit = settings.tracking_error_limit_deg

        started_s = self.clock.monotonic()
        last_beat_s = started_s
        missed_in_a_row = 0
        over_bound_in_a_row = 0
        previous_sent: tuple[float, ...] | None = None

        if self.session.port.lifecycle is ArmLifecycle.CONNECTED:
            self.session.port.enable()

        try:
            for offset in range(len(plan)):
                tick_started_s = self.clock.monotonic()
                frame_index = int(plan.frame_indices[offset])

                if self.control.is_cancelled():
                    raise ReplayCancelled(
                        f"cancelled after {outcome.frames_sent} of {len(plan)} frames "
                        f"(frame {frame_index})"
                    )

                # One read per tick, which is both the health check and the observation: a
                # replay session configures no cameras, so there is nothing else in it.
                state = self.session.port.read_state()
                if state.has_fault:
                    raise RobotFaultError(
                        f"frame {frame_index}: {state.fault_reason or 'motor fault'}"
                    )
                if not state.all_finite:
                    raise RobotFaultError(f"frame {frame_index}: joint feedback is missing")
                measured_deg = bridge.frame.to_degrees(state.positions)

                if previous_sent is not None:
                    error = max(
                        abs(m - s) for m, s in zip(measured_deg, previous_sent, strict=True)
                    )
                    outcome.max_tracking_error_deg = max(outcome.max_tracking_error_deg, error)
                    if error > tracking_limit:
                        # A single sample over the bound is lag behind a fast step; a run of
                        # them is a joint that is stuck or fighting the command.
                        over_bound_in_a_row += 1
                        if over_bound_in_a_row > settings.max_consecutive_missed_deadlines:
                            raise RobotFaultError(
                                f"frame {frame_index}: tracking error {error:.2f} deg exceeded "
                                f"{tracking_limit} deg for {over_bound_in_a_row} consecutive frames"
                            )
                    else:
                        over_bound_in_a_row = 0

                action = {
                    f"{name}": float(value)
                    for name, value in zip(joint_names, plan.actions_deg[offset], strict=True)
                }
                sent = bridge.send_action(action)
                sent_values = tuple(sent[name] for name in joint_names)
                previous_sent = sent_values

                # Clipping is measured against the dataset's own values, so a clamp applied in
                # preflight counts the same as one applied by the driver.
                requested = plan.requested_deg[offset]
                deviation = max(
                    abs(float(r) - s) for r, s in zip(requested, sent_values, strict=True)
                )
                if deviation > 0:
                    outcome.clipped_frames += 1
                    outcome.max_clipping_deg = max(outcome.max_clipping_deg, deviation)

                outcome.frames_sent += 1
                outcome.last_frame_sent = frame_index

                now = self.clock.monotonic()
                if now - last_beat_s >= settings.heartbeat_interval_s:
                    last_beat_s = now
                    self.control.beat(
                        ReplayProgress(
                            frame=frame_index,
                            frames_planned=len(plan),
                            frames_sent=outcome.frames_sent,
                            elapsed_s=now - started_s,
                            clipped_frames=outcome.clipped_frames,
                        )
                    )

                remaining = period - (now - tick_started_s)
                if remaining > 0:
                    self.clock.sleep(remaining)
                    missed_in_a_row = 0
                else:
                    # Deliberately no catch-up: the lost time is absorbed, never recovered by
                    # sending the next frames early.
                    outcome.missed_deadlines += 1
                    missed_in_a_row += 1
                    if missed_in_a_row > settings.max_consecutive_missed_deadlines:
                        raise ReplayTimingError(
                            f"frame {frame_index}: missed the {period * 1000:.1f} ms control "
                            f"deadline {missed_in_a_row} times in a row"
                        )

        finally:
            # Elapsed time is recorded even when the loop raises: a partial replay's
            # duration is part of what happened.
            outcome.elapsed_s = self.clock.monotonic() - started_s
        return outcome


def _alignment_profile(parameters: RobotParameters, settings: Any) -> MotionProfile:
    """Motion defaults, but with replay's own tolerance and settle time.

    "Close enough to start replaying" is a replay decision, not a general motion one.
    """

    return replace(
        parameters.motion_defaults,
        tolerance_rad=settings.initial_pose_tolerance_deg / _RAD_TO_DEG,
        settle_time_s=settings.settle_time_s,
    )


@dataclass
class _PlayOutcome:
    frames_planned: int
    frames_sent: int = 0
    last_frame_sent: int | None = None
    elapsed_s: float = 0.0
    clipped_frames: int = 0
    max_clipping_deg: float = 0.0
    missed_deadlines: int = 0
    max_tracking_error_deg: float = 0.0


class ReplayRunner:
    """End-to-end replay: resolve, preflight, claim, align, play, release.

    Everything that can be decided without hardware is decided first, and a failure there
    raises: the request itself is wrong and there is no physical attempt to report. Once
    the arm is claimed, anticipated failures come back as a :class:`ReplayResult` with
    ``completed=False`` and a machine-readable ``stopped_reason`` instead, because a
    physical attempt did happen and the workflow has to record it.
    """

    def __init__(
        self,
        port: ArmPort,
        parameters: RobotParameters,
        *,
        resolver: HuggingFaceDatasetResolver | None = None,
        reader_factory: Any = None,
        lease: RobotLease | None = None,
        frame: JointFrame | None = None,
        clock: Clock | None = None,
        compatible_robot_types: Sequence[str] = DEFAULT_COMPATIBLE_ROBOT_TYPES,
    ) -> None:
        self.port = port
        self.parameters = parameters
        self.resolver = resolver or HuggingFaceDatasetResolver(
            require_full_revision=parameters.replay.require_full_revision
        )
        self.reader_factory = reader_factory or LeRobotV3Reader
        self.lease = lease or InMemoryLease(parameters.identity.robot_id)
        self.frame = frame or _frame_from(parameters, port)
        self.clock = clock or SystemClock()
        self.compatible_robot_types = tuple(compatible_robot_types)

    # ── phase one: no hardware ─────────────────────────────────────────────────────
    def approval_for(self, request: ReplayRequest) -> ApprovedReplay | None:
        """The operator-approved mapping for this request, if replay requires one."""

        approval = self.parameters.approved_replay(request.repo_id)
        settings = self.parameters.replay

        if approval is None:
            if settings.require_approved_dataset:
                raise CalibrationMismatchError(
                    f"{request.repo_id} is not in approved_replays for "
                    f"{self.parameters.identity.robot_id}; a dataset carries no record of which "
                    "arm produced it, so replaying it needs an operator-approved mapping"
                )
            return None

        if not approval.allows(request.revision, request.episode):
            raise CalibrationMismatchError(
                f"{request.repo_id}: revision {request.revision[:12]} episode {request.episode} "
                f"is not approved (approved revision {approval.revision[:12]}, episodes "
                f"{list(approval.episodes)})"
            )
        if settings.require_matching_robot_id and request.robot_id != approval.robot_id:
            raise CalibrationMismatchError(
                f"request names robot {request.robot_id!r}, approval names {approval.robot_id!r}"
            )
        if (
            settings.require_matching_calibration_id
            and request.calibration_id != approval.calibration_id
        ):
            raise CalibrationMismatchError(
                f"request names calibration {request.calibration_id!r}, approval names "
                f"{approval.calibration_id!r}"
            )
        return approval

    def preflight(self, request: ReplayRequest) -> tuple[ReplayPlan, ApprovedReplay | None]:
        """Resolve, load and validate. Claims nothing and energizes nothing."""

        approval = self.approval_for(request)
        self._check_request_identity(request)

        snapshot = self.resolver.resolve(request.repo_id, request.revision)
        episode = self.reader_factory(snapshot).read_episode(request.episode)
        preflight = ReplayPreflight(
            self.parameters,
            frame=self.frame,
            limits=self.port.limits(),
            compatible_robot_types=self.compatible_robot_types,
        )
        return preflight.check(episode, request, approval), approval

    def _check_request_identity(self, request: ReplayRequest) -> None:
        identity = self.parameters.identity
        settings = self.parameters.replay
        if settings.require_matching_robot_id and request.robot_id != identity.robot_id:
            raise CalibrationMismatchError(
                f"request names robot {request.robot_id!r}, this deployment is "
                f"{identity.robot_id!r}"
            )
        if (
            settings.require_matching_calibration_id
            and request.calibration_id != identity.calibration_id
        ):
            raise CalibrationMismatchError(
                f"request names calibration {request.calibration_id!r}, this deployment is "
                f"{identity.calibration_id!r}"
            )

    # ── phase two: hardware ────────────────────────────────────────────────────────
    def run(self, request: ReplayRequest, control: ReplayControl | None = None) -> ReplayResult:
        control = control or ReplayControl()
        plan, approval = self.preflight(request)

        if request.dry_run:
            # Validated everything, touched nothing: no lease, no connection, no motion.
            return self._result(request, plan, _PlayOutcome(frames_planned=len(plan)),
                                completed=True, stopped_reason=None)

        outcome = _PlayOutcome(frames_planned=len(plan))
        session = RobotSession(
            self.port,
            self.parameters,
            lease=self.lease,
            frame=self.frame,
            clock=self.clock,
            # During alignment there are no frames yet; the heartbeat still has to fire so
            # a slow approach does not lose the worker's claim.
            heartbeat=(lambda: control.beat(_progress(plan, outcome, 0.0)))
            if control.heartbeat
            else None,
            cancel=control.cancelled,
        )
        player = SessionPlayer(session, clock=self.clock, control=control)

        with session:
            try:
                player.align(plan, approval)
                player.play(plan, request, outcome)
            except (
                ReplayCancelled,
                RobotFaultError,
                ReplayTimingError,
                InitialAlignmentError,
            ) as error:
                # A physical attempt happened; the workflow has to see it. The session's
                # teardown holds and releases on the way out.
                self._hold(failed=True)
                return self._result(
                    request, plan, outcome, completed=False, stopped_reason=_reason(error)
                )
            self._hold(failed=False)

        return self._result(request, plan, outcome, completed=True, stopped_reason=None)

    def _hold(self, *, failed: bool) -> None:
        settings = self.parameters.replay
        hold = settings.hold_on_failure if failed else settings.hold_on_completion
        if hold and self.port.lifecycle is ArmLifecycle.ENABLED:
            self.port.hold_current_position()

    def _result(
        self,
        request: ReplayRequest,
        plan: ReplayPlan,
        outcome: _PlayOutcome,
        *,
        completed: bool,
        stopped_reason: str | None,
    ) -> ReplayResult:
        elapsed = outcome.elapsed_s
        return ReplayResult(
            repo_id=request.repo_id,
            revision=request.revision,
            episode=request.episode,
            robot_id=self.parameters.identity.robot_id,
            calibration_id=self.parameters.identity.calibration_id,
            completed=completed,
            frames_planned=outcome.frames_planned,
            frames_sent=outcome.frames_sent,
            first_frame=plan.report.first_frame,
            last_frame_sent=outcome.last_frame_sent,
            elapsed_s=elapsed,
            effective_fps=(outcome.frames_sent / elapsed) if elapsed > 0 else 0.0,
            clipped_frames=outcome.clipped_frames,
            max_clipping_deg=outcome.max_clipping_deg,
            dry_run=request.dry_run,
            config_digest=self.parameters.config_digest,
            stopped_reason=stopped_reason,
        )


def _progress(plan: ReplayPlan, outcome: _PlayOutcome, elapsed: float) -> ReplayProgress:
    return ReplayProgress(
        frame=outcome.last_frame_sent or plan.report.first_frame,
        frames_planned=outcome.frames_planned,
        frames_sent=outcome.frames_sent,
        elapsed_s=elapsed,
        clipped_frames=outcome.clipped_frames,
    )


def _reason(error: Exception) -> str:
    """A stable, machine-readable token for a durable result."""

    name = type(error).__name__
    snake = "".join(f"_{c.lower()}" if c.isupper() else c for c in name).lstrip("_")
    return snake.removesuffix("_error")


def _frame_from(parameters: RobotParameters, port: ArmPort) -> JointFrame:
    spec = parameters.lerobot_frame
    return JointFrame(
        port.joint_names,
        offsets_deg=spec.offsets_deg if spec is not None else (),
    )


__all__ = [
    "PreflightReport",
    "ReplayControl",
    "ReplayPlan",
    "ReplayPreflight",
    "ReplayProgress",
    "ReplayRunner",
    "SessionPlayer",
    "limits_in_degrees",
]
