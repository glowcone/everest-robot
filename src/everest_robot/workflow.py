"""Absurd task definitions: carabiner attachment, and stored-session replay."""

import os
from typing import Any

from absurd_sdk import Absurd

from everest_robot.adapters import replay_session, robot_session
from everest_robot.domain import (
    AttachmentResult,
    RecoveryTarget,
    ReplayRequest,
    VerificationResult,
    json_dict,
)
from everest_robot.robot.replay import ReplayControl

QUEUE_NAME = os.getenv("ROBOT_QUEUE", "robot")


def _verification_from_json(value: dict[str, Any]) -> VerificationResult:
    target = value.get("recovery_target")
    return VerificationResult(
        secure=bool(value["secure"]),
        confidence=float(value["confidence"]),
        recovery_target=RecoveryTarget(target) if target else None,
        reason=value.get("reason"),
    )


def _heartbeat_for(ctx: Any) -> Any:
    """The worker-claim heartbeat, if this context has one.

    Long physical calls hold the arm for far longer than a claim timeout, so the runtime
    beats from inside its control loops. Absurd also signals cancellation by raising from
    here, which the motion and rollout loops treat as a stop-and-hold.
    """

    heartbeat = getattr(ctx, "heartbeat", None)
    return (lambda: heartbeat()) if callable(heartbeat) else None


def run_attach_carabiner(params: dict[str, Any], ctx: Any) -> dict[str, Any]:
    """Localize, pick up, position, attach, verify, and durably recover."""

    with robot_session(params, heartbeat=_heartbeat_for(ctx)) as robot:
        return _attach_carabiner(params, ctx, robot)


def _attach_carabiner(params: dict[str, Any], ctx: Any, robot: Any) -> dict[str, Any]:
    """The durable state machine. The robot is already claimed and connected."""

    detector = str(params.get("carabiner_detector", "deterministic-cv"))
    grasp_planner = str(params.get("grasp_planner", "graspnet"))
    position_name = str(params.get("attachment_position", "clip-attachment-ready"))
    attachment_controller = str(params.get("attachment_controller", "vla"))
    max_cycles = int(params.get("max_recovery_cycles", 5))

    recovery_target = RecoveryTarget.LOCALIZE_AND_PICK_UP
    pickup: dict[str, Any] | None = None
    position: dict[str, Any] | None = None
    for cycle in range(max_cycles):
        suffix = f"cycle-{cycle:02d}"
        if recovery_target is RecoveryTarget.LOCALIZE_AND_PICK_UP:
            pickup = ctx.step(
                f"01-localize-and-pick-up-carabiner-{suffix}",
                lambda: json_dict(
                    robot.localize_and_pick_up_carabiner(detector, grasp_planner)
                ),
            )
            if not pickup["secured"]:
                raise RuntimeError("carabiner localization or GraspNet pickup failed")

        if recovery_target in {
            RecoveryTarget.LOCALIZE_AND_PICK_UP,
            RecoveryTarget.GO_TO_KNOWN_POSITION,
        }:
            position = ctx.step(
                f"02-go-to-known-position-{suffix}",
                lambda: json_dict(robot.go_to_known_position(position_name)),
            )
            if not position["reached"]:
                raise RuntimeError("robot failed to reach the known attachment position")

        if pickup is None or position is None:
            raise RuntimeError("attachment prerequisites are unavailable")
        attachment = ctx.step(
            f"03-rl-vla-attach-clip-{suffix}",
            lambda: json_dict(robot.attach_clip(attachment_controller)),
        )
        verification_json = ctx.step(
            f"04-verify-attachment-{suffix}",
            lambda attachment=attachment: json_dict(
                robot.verify_attachment(AttachmentResult(**attachment))
            ),
        )
        verification = _verification_from_json(verification_json)

        if verification.secure:
            return {
                "status": "complete",
                "cycles": cycle + 1,
                "pickup": pickup,
                "position": position,
                "attachment": attachment,
                "verification": verification_json,
            }
        if verification.recovery_target is None:
            raise RuntimeError("verification failed without a recovery target")
        recovery_target = verification.recovery_target

    raise RuntimeError(f"attachment was not secure after {max_cycles} recovery cycles")


def run_replay_session(params: dict[str, Any], ctx: Any) -> dict[str, Any]:
    """Replay one stored dataset episode as a single durable stage.

    One checkpoint covers the whole invocation. Frame-level checkpoints would be both
    enormous and useless: the value of a checkpoint is that its effect need not be
    repeated, and there is no way to resume a physical replay mid-episode from a stored
    frame number without knowing where the arm actually is.
    """

    request = ReplayRequest.from_json(params)
    heartbeat = _heartbeat_for(ctx)
    control = ReplayControl(
        # Absurd surfaces cancellation by raising out of the heartbeat, so there is no
        # separate flag to poll. The raise unwinds through the replay session, which holds
        # the arm and releases the lease on its way out.
        heartbeat=(lambda progress: heartbeat()) if heartbeat else None,
    )
    return ctx.step(
        "01-replay-session",
        lambda: json_dict(replay_session(request, control)),
    )


def create_app() -> Absurd:
    """Create the database-backed app and register workflow tasks."""

    app = Absurd(queue_name=QUEUE_NAME)
    app.register_task("attach-carabiner", default_max_attempts=10)(run_attach_carabiner)
    # Deliberately one attempt. A replay interrupted after 200 frames cannot be safely
    # restarted from frame zero: the arm is in an unknown intermediate pose, and repeating
    # the sequence from there is a different physical motion. Recovery is an operator
    # decision -- inspect, realign through an approved path, and authorize a new attempt.
    app.register_task("replay-session", default_max_attempts=1)(run_replay_session)
    return app
