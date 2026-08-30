"""Replaceable hardware and perception boundaries.

Two implementations live here:

* :class:`ScaffoldRobot` -- deterministic placeholders, so the durable orchestration can
  be exercised with no hardware at all.
* :class:`EverestRobot` -- the real thing, driving an open
  :class:`~everest_robot.robot.session.RobotSession`. The stages that are implemented
  (named-position motion, policy rollout) go through the robot runtime; the stages that
  are not (perception, attachment verification) refuse rather than pretend.

:func:`robot_session` picks between them from deployment configuration, so the workflow
never names a backend.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from everest_robot.attachment_fsm import (
    AttachmentAbort,
    AttachmentState,
    ClipRLStep,
    InitialObservation,
    SearchCVStep,
    SearchRLStep,
)
from everest_robot.domain import (
    AttachmentResult,
    CarabinerPickupResult,
    PositionResult,
    RecoveryTarget,
    VerificationResult,
)
from everest_robot.robot.contracts import ArmLifecycle, TerminationReason

if TYPE_CHECKING:  # pragma: no cover - typing only
    from everest_robot.domain import ReplayRequest, ReplayResult
    from everest_robot.pixel_map import PixelJointMap
    from everest_robot.robot.carabiner_follower import CarabinerFollower
    from everest_robot.robot.policy import PolicyHandle, PolicySession, PolicyStep
    from everest_robot.robot.replay import ReplayControl
    from everest_robot.robot.session import RobotSession


@dataclass
class ScaffoldRobot:
    """A deterministic stand-in for the future robot integrations."""

    verification_failures: int = 0
    verification_calls: int = 0

    def localize_and_pick_up_carabiner(
        self,
        detector: str,
        grasp_planner: str,
    ) -> CarabinerPickupResult:
        return CarabinerPickupResult(
            secured=True,
            frame="robot_base",
            x=0.42,
            y=0.0,
            z=0.18,
            detector=detector,
            grasp_planner=grasp_planner,
        )

    def go_to_known_position(self, position_name: str) -> PositionResult:
        return PositionResult(reached=True, position_name=position_name)

    def attach_clip(self, controller: str) -> AttachmentResult:
        return AttachmentResult(
            motion_completed=True,
            force_newtons=8.0,
            controller=controller,
        )

    def verify_attachment(self, attachment: AttachmentResult) -> VerificationResult:
        del attachment
        self.verification_calls += 1
        if self.verification_calls <= self.verification_failures:
            return VerificationResult(
                secure=False,
                confidence=0.4,
                recovery_target=RecoveryTarget.ATTACH,
                reason="scaffolded verification failure",
            )
        return VerificationResult(secure=True, confidence=0.99)


@dataclass
class ScaffoldAttachmentFSMHandlers:
    """Deterministic no-hardware implementation for the standalone FSM entrypoint."""

    initial_detection: bool = False
    entered: list[AttachmentState] = field(default_factory=list)
    hold_reasons: list[str] = field(default_factory=list)

    def enter_state(
        self, state: AttachmentState, previous: AttachmentState | None
    ) -> None:
        del previous
        self.entered.append(state)

    def observe_initial(self) -> InitialObservation:
        confidence = 0.99 if self.initial_detection else None
        return InitialObservation(False, self.initial_detection, confidence)

    def search_rl_step(self) -> SearchRLStep:
        return SearchRLStep(carabiner_detected=True, confidence=0.99)

    def search_cv_step(self) -> SearchCVStep:
        return SearchCVStep(
            target_visible=True,
            followed=True,
            confidence=0.99,
            pixel_error=0.0,
        )

    def clip_rl_step(self) -> ClipRLStep:
        return ClipRLStep(
            attachment_verified=True,
            carabiner_visible=False,
            carabiner_grasped=True,
            confidence=0.99,
        )

    def hold(self, reason: str) -> None:
        self.hold_reasons.append(reason)


@runtime_checkable
class AttachmentPerception(Protocol):
    """The fresh observations ADR-0003's gates are decided from.

    Separate from the policy handlers because it fails separately: the learned half of
    ``SEARCH_RL`` and ``CLIP_RL`` is a loaded checkpoint and a one-action session, while
    every signal below is perception -- detection, attachment verification, grasp and
    alignment. Keeping them apart means the policy path can be brought up and rehearsed on
    a real arm while the perception fusion is still being built, and it makes the refusal
    name exactly one missing subsystem.

    Each method is called immediately after one physical action, and must report what is
    true *now*. The FSM, not this protocol, decides what any of it means.
    """

    def preflight(self) -> None:
        """Raise unless every gate below can be served. Called before the robot is claimed."""

    def initial_observation(self) -> InitialObservation: ...

    def carabiner_detection(self) -> SearchRLStep: ...

    def clip_observations(self) -> ClipRLStep: ...


@dataclass(frozen=True)
class UnavailableAttachmentPerception:
    """The default: refuses, and says which subsystem is missing.

    Refusing in :meth:`preflight` is what keeps the refusal cheap. A gate that cannot be
    read is discovered before the lease is taken and before a motor is energized, never
    after a learned action has already moved the arm.
    """

    def preflight(self) -> None:
        raise NotImplementedError(
            "no attachment perception is configured: SEARCH_RL needs a carabiner detector "
            "and CLIP_RL needs attachment, grasp, visibility and alignment checks. The "
            "learned half of both states is implemented -- pass a perception "
            "implementation to attachment_fsm_handlers() to run them."
        )

    def carabiner_detection(self) -> SearchRLStep:
        self.preflight()
        raise AssertionError("unreachable")

    def initial_observation(self) -> InitialObservation:
        self.preflight()
        raise AssertionError("unreachable")

    def clip_observations(self) -> ClipRLStep:
        self.preflight()
        raise AssertionError("unreachable")


@dataclass
class EverestAttachmentFSMHandlers:
    """ADR-0003's learned states, over an open robot session.

    ``SEARCH_RL`` and ``CLIP_RL`` are each one persistent
    :class:`~everest_robot.robot.policy.PolicySession`. They stay separate even when both
    are loaded from the same checkpoint, because classical vision physically intervenes
    between them: an action chunk cached before CV moved the arm describes a pose the arm
    has left. Entering either state re-seeds its own session and leaves the other alone.

    ``SEARCH_CV`` is that intervening vision: it drives :class:`CarabinerFollower`, which
    is the fixed camera, the two-tape detector, the calibrated pixel map and a speed-locked
    ``VisualTracker``. Its ``calibration`` is optional only so the learned states can be
    brought up and rehearsed without a camera; the hardware backend always supplies one,
    and entering ``SEARCH_CV`` without it refuses rather than servos blind.

    Every step method performs at most one physical action, then reads the gate signals
    fresh. The transition decision itself belongs to
    :class:`~everest_robot.attachment_fsm.AttachmentFSM` and is deliberately not duplicated
    here -- these methods report, they do not choose.
    """

    session: RobotSession
    search_policy: PolicyHandle
    clip_policy: PolicyHandle
    calibration: PixelJointMap | None = None
    perception: AttachmentPerception = field(default_factory=UnavailableAttachmentPerception)
    task: str | None = None
    fps: float | None = None
    rate_hz: float = 15.0
    max_velocity_rad_s: float = 0.15
    lock_frames: int = 3
    neutral_position: tuple[float, ...] | None = None
    neutral_tolerance_rad: float = 0.05
    neutral_velocity_rad_s: float = 0.05
    last_readiness: Any = field(default=None, init=False)

    def __post_init__(self) -> None:
        from everest_robot.pixel_map import PixelMapError, RobotStamp

        self._capture: Any = None
        self._follower: CarabinerFollower | None = None
        if self.calibration is not None:
            identity = self.session.port.identity
            # A fit only means anything for the arm and the zeroing it was taught on, and
            # SEARCH_CV is reachable straight out of INITIAL. Refuse now, not three states
            # in with the arm already moving.
            self.calibration.robot.verify(
                RobotStamp(identity.robot_id, identity.calibration_id)
            )
            if self.calibration.joint_names != tuple(self.session.port.joint_names):
                raise PixelMapError(
                    f"calibration joints {', '.join(self.calibration.joint_names)} do not "
                    f"match this arm's {', '.join(self.session.port.joint_names)}"
                )
        overrides: dict[str, object] = {"task": self.task}
        if self.fps is not None:
            overrides["fps"] = self.fps
        self._sessions = {
            AttachmentState.SEARCH_RL: self.session.policy_session(
                self.search_policy, **overrides
            ),
            AttachmentState.CLIP_RL: self.session.policy_session(self.clip_policy, **overrides),
        }

    def policy_session_for(self, state: AttachmentState) -> PolicySession:
        """The persistent session backing one learned state. For inspection and tests."""

        return self._sessions[state]

    # ── state entry ────────────────────────────────────────────────────────────────
    def enter_state(
        self, state: AttachmentState, previous: AttachmentState | None
    ) -> None:
        """Re-seed the learned state being entered, from where the arm is standing now.

        Nothing is commanded here, and the state being *left* is not held: on the paths
        that leave a learned state, classical vision is about to command the arm, and an
        interposed hold would fight it. The FSM holds on every terminal outcome and on any
        exception, which is where a hold actually belongs.
        """

        del previous
        # The follower owns lock-on state and a tracker command seeded from feedback, and
        # both are stale the moment a policy moves the arm. Tear it down on the way out of
        # SEARCH_CV and build a fresh one on the way back in, rather than resuming it.
        if state is not AttachmentState.SEARCH_CV:
            self._release_follower()
        elif self._follower is None and self.calibration is not None:
            self._follower = self._build_follower()
            self._follower.start()
        rollout = self._sessions.get(state)
        if rollout is not None:
            rollout.seed()

    # ── the learned states ─────────────────────────────────────────────────────────
    def observe_initial(self) -> InitialObservation:
        """Prove passive hardware readiness, then take one coherent scene observation."""

        self.last_readiness = self.session.initial_readiness(
            neutral_position=self.neutral_position,
            neutral_tolerance_rad=self.neutral_tolerance_rad,
        )
        if not self.last_readiness.ready:
            raise AttachmentAbort(
                "INITIAL passive readiness refused: "
                + "; ".join(self.last_readiness.problems)
            )
        return self.perception.initial_observation()

    def search_rl_step(self) -> SearchRLStep:
        """One learned search action, then a fresh look for the carabiner."""

        step = self._advance(AttachmentState.SEARCH_RL)
        detection = self.perception.carabiner_detection()
        if step.termination is TerminationReason.COMPLETED and not detection.carabiner_detected:
            # The search policy considers itself finished but nothing was found. Continuing
            # would call a spent rollout until the budget ran out, which is a livelock
            # dressed up as progress.
            raise AttachmentAbort(
                "the search policy signalled completion without detecting the carabiner"
            )
        return detection

    def search_cv_step(self) -> SearchCVStep:
        """One detector/map/servo tick, and the follower's own visibility verdict."""

        from everest_robot.robot.visual_tracking import TrackerStopped

        if self._follower is None:
            if self.calibration is None:
                from everest_robot.pixel_map import PixelMapError

                raise PixelMapError(
                    "SEARCH_CV needs a calibrated pixel map; these handlers were built "
                    "without one (`robot-pixel-map fit`, then EVEREST_PIXEL_MAP)"
                )
            raise RuntimeError(
                "search_cv_step() was called outside SEARCH_CV; the follower is built on "
                "entry to that state"
            )
        try:
            tick = self._follower.step()
        except TrackerStopped as error:
            # The tracker holds the arm before it stops. A fault, lost feedback or a
            # refused command is a safety stop, not a step result to keep arbitrating on.
            raise AttachmentAbort(f"visual follower stopped: {error}") from error
        return SearchCVStep(
            target_visible=tick.target_visible,
            followed=tick.followed,
            # The two-tape segmentation is a hard threshold with no score. Same reason
            # `attach_clip` reports no force: a number here would be fiction. What the
            # follower does measure is the servo error, in the map's own pixels.
            confidence=None,
            pixel_error=tick.pixel_error_px,
        )

    def clip_rl_step(self) -> ClipRLStep:
        """One learned grasp/clip action, then fresh attachment and recovery checks.

        UNVERIFIED ASSUMPTION -- confirm before running a trained checkpoint. This reads
        "the policy returned to neutral" off :meth:`PolicyHandle.select_action` returning
        ``None``, which the policy protocol already defines as "the policy considers the
        task finished". The mapping was chosen so the FSM needs no second completion
        channel, but it has not been confirmed against a real checkpoint on hardware, and a
        scripted policy cannot confirm it: a fake exhibits whatever mapping its test
        asserts. Completion is only a candidate; fresh stationary feedback must also match
        the operator-captured neutral pose. If the checkpoint signals neutral some other
        way or returns to neutral silently, this is the line to change. See ADR-0003,
        "Assumption pending verification".
        """

        step = self._advance(AttachmentState.CLIP_RL)
        if step.termination is TerminationReason.COMPLETED:
            self._require_measured_neutral()
            # ADR-0003 gives the neutral signal precedence over any same-step verification.
            # The FSM goes back to INITIAL, which takes its own motion-free observation, so
            # spending a perception call here would answer a question about to be asked
            # again.
            return ClipRLStep(attachment_verified=False, returned_to_neutral=True)
        return self.perception.clip_observations()

    def hold(self, reason: str) -> None:
        del reason
        if self.session.port.lifecycle is ArmLifecycle.ENABLED:
            self.session.port.hold_current_position()

    def close(self) -> None:
        """Stop following and release the fixed camera. The session is closed by its owner."""

        self._release_follower()
        capture, self._capture = self._capture, None
        if capture is not None:
            capture.release()

    # ── helpers ────────────────────────────────────────────────────────────────────
    def _advance(self, state: AttachmentState) -> PolicyStep:
        """Take exactly one action, and turn a stopped rollout into an abort.

        Cancellation and a motor fault are both ADR-0003 aborts rather than failures: the
        attempt stopped for a reason outside the state machine's decisions, and the arm
        needs a person to look at it before another attempt is authorized.
        """

        step = self._sessions[state].step()
        if step.termination is TerminationReason.CANCELLED:
            raise AttachmentAbort(f"{state.value} cancelled")
        if step.termination is TerminationReason.FAILED:
            detail = step.failure_detail or "no detail"
            raise AttachmentAbort(f"{state.value} stopped: {step.failure_reason} -- {detail}")
        return step

    # ── the CV subsystem ───────────────────────────────────────────────────────────
    def _build_follower(self) -> CarabinerFollower:
        from everest_robot.pixel_map import PixelMapError
        from everest_robot.robot.carabiner_follower import (
            CarabinerFollower,
            detect_carabiner,
            read_frame,
        )
        from everest_robot.robot.visual_tracking import VisualTracker

        if self.calibration is None:
            raise PixelMapError(
                "SEARCH_CV needs a calibrated pixel map; build the handlers with one "
                "(`robot-pixel-map fit`, then EVEREST_PIXEL_MAP)"
            )
        camera = self.calibration.camera
        roi = self.calibration.detector.roi_xywh
        if roi is None:
            raise PixelMapError(
                "this calibration stored no detector ROI; re-run "
                "`robot-pixel-map collect --roi X Y W H`"
            )
        capture = self._open_capture(camera)
        return CarabinerFollower(
            calibration=self.calibration,
            tracker=VisualTracker(
                self.session.port,
                rate_hz=self.rate_hz,
                max_velocity_rad_s=self.max_velocity_rad_s,
                lock_frames=self.lock_frames,
                clock=self.session.clock,
            ),
            frames=lambda: read_frame(capture, camera),
            detect=lambda frame: detect_carabiner(frame, roi),
            clock=self.session.clock,
        )

    def _open_capture(self, camera: Any) -> Any:
        """Open the fixed camera once and keep it for the attempt.

        Re-opening per SEARCH_CV entry would cost a capture-session spin-up every time the
        clip policy handed control back, with the arm claimed and energized throughout.
        """

        if self._capture is None:
            from everest_robot.robot.carabiner_follower import open_capture

            self._capture = open_capture(camera)
        return self._capture

    def _release_follower(self) -> None:
        follower, self._follower = self._follower, None
        if follower is not None:
            # Hold: the arm is about to be handed to a policy that seeds from feedback.
            follower.stop()

    def _require_measured_neutral(self) -> None:
        if self.neutral_position is None:
            raise AttachmentAbort(
                "clip policy completed but no operator-captured neutral pose is configured"
            )
        state = self.session.snapshot()
        if state.has_fault or not state.all_finite:
            raise AttachmentAbort("clip policy completed without valid neutral feedback")
        max_velocity = max((abs(value) for value in state.velocities), default=0.0)
        if not math.isfinite(max_velocity) or max_velocity > self.neutral_velocity_rad_s:
            raise AttachmentAbort(
                f"clip policy completed while arm was moving at {max_velocity:.3f} rad/s"
            )
        if len(state.positions) != len(self.neutral_position):
            raise AttachmentAbort("configured neutral pose has the wrong joint count")
        error = max(
            abs(actual - target)
            for actual, target in zip(
                state.positions, self.neutral_position, strict=True
            )
        )
        if error > self.neutral_tolerance_rad:
            raise AttachmentAbort(
                f"clip policy completed {error:.3f} rad from neutral; "
                f"tolerance is {self.neutral_tolerance_rad:.3f} rad"
            )

@contextmanager
def attachment_fsm_handlers(
    params: dict[str, Any],
    *,
    perception: AttachmentPerception | None = None,
) -> Iterator[Any]:
    """Build the standalone FSM handlers under the same deployment and lease boundary.

    Ordering is the point, and it is the same ordering replay preflight uses: the policy
    files are resolved, the perception gates are checked and the pixel map is loaded
    *before* the robot is claimed or energized. A missing checkpoint, a malformed scripted
    policy, an unavailable detector or an uncalibrated camera is a mistake anyone can make,
    and none of them should cost an energized arm to discover.
    """

    import os

    backend = str(params.get("backend") or os.getenv("EVEREST_ROBOT_BACKEND", "scaffold"))
    if backend == "scaffold":
        yield ScaffoldAttachmentFSMHandlers(
            initial_detection=bool(params.get("initial_detection", False))
        )
        return
    if backend != "hardware":
        raise ValueError(f"unknown robot backend {backend!r} (expected scaffold or hardware)")

    from everest_robot.robot.deployment import load_parameters
    from everest_robot.robot.policy import load_policy

    search_path = params.get("search_policy")
    clip_path = params.get("clip_policy")
    missing = [
        name
        for name, value in (("search_policy", search_path), ("clip_policy", clip_path))
        if not value
    ]
    if missing:
        raise ValueError(
            f"the hardware backend needs a policy file for each learned state; missing: "
            f"{', '.join(missing)}"
        )

    # Search and clip are separate sessions even when this resolves to the same file
    # twice: ADR-0003 requires separate carried state, not a shared handle.
    search_policy = load_policy(search_path)
    clip_policy = load_policy(clip_path)
    checks = perception if perception is not None else UnavailableAttachmentPerception()
    checks.preflight()

    parameters = load_parameters()
    neutral_name = str(params.get("neutral_position") or "neutral")
    neutral = parameters.named_positions.get(neutral_name)
    if neutral is None:
        known = ", ".join(sorted(parameters.named_positions)) or "none captured"
        raise ValueError(
            f"the hardware FSM requires an operator-captured neutral named position "
            f"{neutral_name!r} before it can verify policy reset; available: {known}"
        )

    from everest_robot.robot.deployment import load_pixel_map, open_session

    # The pixel map is read and refused here for the same reason the policies are: a stale
    # or missing calibration is not worth an energized arm to discover.
    calibration = load_pixel_map()

    session = open_session()
    try:
        handlers = EverestAttachmentFSMHandlers(
            session,
            search_policy,
            clip_policy,
            calibration=calibration,
            perception=checks,
            task=params.get("attachment_task"),
            fps=None if params.get("policy_fps") is None else float(params["policy_fps"]),
            neutral_position=neutral.joints,
            neutral_tolerance_rad=neutral.profile.tolerance_rad,
        )
        try:
            yield handlers
        finally:
            handlers.close()
    finally:
        session.close()


@dataclass
class EverestRobot:
    """The workflow's view of a real arm, over an open robot session.

    Motion and policy results are translated into the workflow's durable domain types,
    keeping the measurable detail (tracking error, timing, termination, config digest) so
    a stage can be traced back to the configuration and hardware that produced it.
    """

    session: RobotSession
    policy_factory: Callable[[str], PolicyHandle] | None = None
    task: str | None = None
    speed_scale: float = 1.0
    transitions: dict[str, str] = field(default_factory=dict)

    def localize_and_pick_up_carabiner(
        self,
        detector: str,
        grasp_planner: str,
    ) -> CarabinerPickupResult:
        raise NotImplementedError(
            "carabiner localization and grasp planning are not part of the robot SDK "
            "layer: they need the CV detector and GraspNet integration. Run the workflow "
            "with the scaffold backend until that lands."
        )

    def go_to_known_position(self, position_name: str) -> PositionResult:
        """Move to an approved preset, using its approved waypoint path when one exists."""

        motion = self.session.motion
        transition = self.transitions.get(position_name)
        result = (
            motion.follow_transition(transition, speed_scale=self.speed_scale)
            if transition
            else motion.go_to_known_position(position_name, speed_scale=self.speed_scale)
        )
        return PositionResult(
            reached=result.reached,
            position_name=result.position_name,
            already_at_target=result.already_at_target,
            max_tracking_error_rad=result.max_tracking_error_rad,
            elapsed_s=result.elapsed_s,
            config_digest=result.config_digest,
            failure_reason=str(result.failure_reason) if result.failure_reason else None,
            failure_detail=result.failure_detail,
        )

    def attach_clip(self, controller: str) -> AttachmentResult:
        """Run the attachment policy and report what the rollout actually did."""

        from everest_robot.robot.contracts import TerminationReason

        if self.policy_factory is None:
            raise NotImplementedError(
                "no policy factory configured: attach_clip needs a loaded checkpoint. "
                "LeRobot checkpoint loading is still blocked on the dataset decision "
                "(see everest_robot.robot.policy.LeRobotPolicyHandle)."
            )
        handle = self.policy_factory(controller)
        result = self.session.policy.run(handle, task=self.task)
        return AttachmentResult(
            # Running out of step or duration budget is not a completed maneuver.
            motion_completed=result.termination is TerminationReason.COMPLETED,
            controller=controller,
            # The Maker Arm has no force sensor; reporting a number here would be fiction.
            force_newtons=None,
            checkpoint=result.checkpoint,
            steps=result.steps,
            missed_deadlines=result.missed_deadlines,
            termination=str(result.termination),
            config_digest=result.config_digest,
            failure_reason=str(result.failure_reason) if result.failure_reason else None,
            failure_detail=result.failure_detail,
        )

    def verify_attachment(self, attachment: AttachmentResult) -> VerificationResult:
        raise NotImplementedError(
            "attachment verification is not part of the robot SDK layer: it needs the "
            "sensor/CV/VLM fusion. Run the workflow with the scaffold backend until that "
            "lands."
        )


@contextmanager
def robot_session(
    params: dict[str, Any],
    *,
    heartbeat: Callable[[], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> Iterator[Any]:
    """Yield the robot a workflow stage should use, and clean up afterwards.

    The backend comes from ``params['backend']`` or ``EVEREST_ROBOT_BACKEND`` and defaults
    to the scaffold, so nothing touches hardware by accident. On the hardware backend the
    session claims the robot exclusively for the whole stage and releases it on the way
    out, including when the stage raises.
    """

    import os

    backend = str(params.get("backend") or os.getenv("EVEREST_ROBOT_BACKEND", "scaffold"))
    if backend == "scaffold":
        yield ScaffoldRobot(
            verification_failures=int(params.get("verification_failures", 0))
        )
        return
    if backend != "hardware":
        raise ValueError(f"unknown robot backend {backend!r} (expected scaffold or hardware)")

    from everest_robot.robot.deployment import open_session

    session = open_session(heartbeat=heartbeat, cancel=cancel)
    try:
        yield EverestRobot(
            session,
            task=params.get("attachment_task"),
            speed_scale=float(params.get("speed_scale", 1.0)),
            transitions=dict(params.get("named_transitions") or {}),
        )
    finally:
        session.close()


def replay_session(
    request: ReplayRequest,
    control: ReplayControl | None = None,
    *,
    runner: Any = None,
    environ: Any = None,
) -> ReplayResult:
    """Replay a stored dataset episode. The adapter boundary for replay.

    A workflow passes a request and gets a result; datasets, tensors, motors and leases all
    stay on this side of the line.

    Replay claims the robot itself rather than running inside an existing
    :func:`robot_session`, and that ordering is the point: every dataset and configuration
    check has to finish *before* anything is claimed or energized, so discovering a broken
    episode never costs an energized arm.
    """

    if runner is None:
        from everest_robot.robot.deployment import build_replay_runner

        runner = build_replay_runner(environ=environ)
    return runner.run(request, control)
