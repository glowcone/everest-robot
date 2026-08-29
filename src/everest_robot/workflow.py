"""Absurd task definition for the carabiner attachment workflow."""

import os
from typing import Any

from absurd_sdk import Absurd

from everest_robot.adapters import ScaffoldRobot
from everest_robot.domain import (
    AttachmentResult,
    RecoveryTarget,
    RopePose,
    VerificationResult,
    json_dict,
)

QUEUE_NAME = os.getenv("ROBOT_QUEUE", "robot")
app = Absurd(queue_name=QUEUE_NAME)


def _verification_from_json(value: dict[str, Any]) -> VerificationResult:
    target = value.get("recovery_target")
    return VerificationResult(
        secure=bool(value["secure"]),
        confidence=float(value["confidence"]),
        recovery_target=RecoveryTarget(target) if target else None,
        reason=value.get("reason"),
    )


@app.register_task("attach-carabiner", default_max_attempts=10)
def attach_carabiner(params: dict[str, Any], ctx: Any) -> dict[str, Any]:
    """Pick up, locate, attach, verify, and durably recover when needed."""

    robot = ScaffoldRobot(verification_failures=int(params.get("verification_failures", 0)))
    controller = str(params.get("pickup_controller", "rl-policy"))
    detector = str(params.get("rope_detector", "deterministic-cv"))
    max_cycles = int(params.get("max_recovery_cycles", 5))

    pickup = ctx.step(
        "01-pick-up-carabiner",
        lambda: json_dict(robot.pick_up_carabiner(controller)),
    )
    if not pickup["secured"]:
        raise RuntimeError("carabiner pickup failed; retrying from the durable checkpoint")

    recovery_target = RecoveryTarget.LOCATE_ROPE
    rope_json: dict[str, Any] | None = None
    for cycle in range(max_cycles):
        suffix = f"cycle-{cycle:02d}"
        if recovery_target is RecoveryTarget.PICK_UP:
            pickup = ctx.step(
                f"01-pick-up-carabiner-{suffix}",
                lambda: json_dict(robot.pick_up_carabiner(controller)),
            )
            if not pickup["secured"]:
                raise RuntimeError("carabiner re-pickup failed")

        if recovery_target in {RecoveryTarget.PICK_UP, RecoveryTarget.LOCATE_ROPE}:
            rope_json = ctx.step(
                f"02-locate-rope-{suffix}",
                lambda: json_dict(robot.locate_rope(detector)),
            )
        if rope_json is None:
            raise RuntimeError("attachment cannot start without a rope pose")
        rope = RopePose(**rope_json)
        attachment = ctx.step(
            f"03-attach-carabiner-{suffix}",
            lambda rope=rope: json_dict(robot.attach_carabiner(rope)),
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
                "rope": rope_json,
                "attachment": attachment,
                "verification": verification_json,
            }
        if verification.recovery_target is None:
            raise RuntimeError("verification failed without a recovery target")
        recovery_target = verification.recovery_target

    raise RuntimeError(f"attachment was not secure after {max_cycles} recovery cycles")
