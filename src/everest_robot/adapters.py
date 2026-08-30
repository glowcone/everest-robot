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

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from everest_robot.attachment_fsm import (
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
from everest_robot.robot.contracts import ArmLifecycle

if TYPE_CHECKING:  # pragma: no cover - typing only
    from everest_robot.domain import ReplayRequest, ReplayResult
    from everest_robot.robot.policy import PolicyHandle
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


@dataclass
class EverestAttachmentFSMHandlers:
    """Production integration surface for ADR-0003.

    Every method deliberately refuses until its owning subsystem is integrated. Keeping
    these stubs on the hardware adapter makes it impossible for orchestration to import a
    camera, policy implementation, or motor driver directly.
    """

    session: RobotSession

    def enter_state(
        self, state: AttachmentState, previous: AttachmentState | None
    ) -> None:
        del previous
        if state in {AttachmentState.SEARCH_RL, AttachmentState.CLIP_RL}:
            raise NotImplementedError(
                f"{state.value} entry needs a persistent one-action policy session that "
                "resets from the current post-transition observation; do not emulate it "
                "with PolicyRunner.run(max_steps=1)"
            )

    def observe_initial(self) -> InitialObservation:
        raise NotImplementedError(
            "INITIAL needs one coherent camera observation plus non-moving attachment "
            "verification; integrate the detector and verifier here"
        )

    def search_rl_step(self) -> SearchRLStep:
        raise NotImplementedError(
            "SEARCH_RL needs exactly one action from a persistent policy session followed "
            "by a fresh carabiner detection; integrate the checkpoint loader and detector"
        )

    def search_cv_step(self) -> SearchCVStep:
        raise NotImplementedError(
            "SEARCH_CV needs one bounded VisualTracker tick driven by the calibrated pixel "
            "map and must expose the CV follower's target_visible and followed signals"
        )

    def clip_rl_step(self) -> ClipRLStep:
        raise NotImplementedError(
            "CLIP_RL needs one action from a freshly seeded persistent policy session, "
            "then neutral-return, attachment, grasp, visibility, and alignment checks"
        )

    def hold(self, reason: str) -> None:
        del reason
        if self.session.port.lifecycle is ArmLifecycle.ENABLED:
            self.session.port.hold_current_position()


@contextmanager
def attachment_fsm_handlers(params: dict[str, Any]) -> Iterator[Any]:
    """Build the standalone FSM handlers under the same deployment and lease boundary."""

    import os

    backend = str(params.get("backend") or os.getenv("EVEREST_ROBOT_BACKEND", "scaffold"))
    if backend == "scaffold":
        yield ScaffoldAttachmentFSMHandlers(
            initial_detection=bool(params.get("initial_detection", False))
        )
        return
    if backend != "hardware":
        raise ValueError(f"unknown robot backend {backend!r} (expected scaffold or hardware)")

    from everest_robot.robot.deployment import open_session

    session = open_session()
    try:
        yield EverestAttachmentFSMHandlers(session)
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
