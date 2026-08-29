"""Absurd task definition for the carabiner attachment workflow."""

import os
from typing import Any

from absurd_sdk import Absurd

from everest_robot.adapters import robot_session
from everest_robot.domain import (
    AttachmentResult,
    RecoveryTarget,
    VerificationResult,
    json_dict,
)

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


def create_app() -> Absurd:
    """Create the database-backed app and register workflow tasks."""

    app = Absurd(queue_name=QUEUE_NAME)
    app.register_task("attach-carabiner", default_max_attempts=10)(run_attach_carabiner)
    return app
