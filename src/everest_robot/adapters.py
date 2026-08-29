"""Replaceable hardware and perception boundaries.

The implementations are deterministic placeholders so the durable orchestration can be
developed before an RL/VLA policy, camera pipeline, and robot controller are selected.
"""

from dataclasses import dataclass

from everest_robot.domain import (
    AttachmentResult,
    PickupResult,
    RecoveryTarget,
    RopePose,
    VerificationResult,
)


@dataclass
class ScaffoldRobot:
    """A deterministic stand-in for the future robot integrations."""

    verification_failures: int = 0
    verification_calls: int = 0

    def pick_up_carabiner(self, controller: str) -> PickupResult:
        return PickupResult(secured=True, controller=controller)

    def locate_rope(self, detector: str) -> RopePose:
        return RopePose(frame="robot_base", x=0.42, y=0.0, z=0.18, detector=detector)

    def attach_carabiner(self, rope: RopePose) -> AttachmentResult:
        del rope
        return AttachmentResult(motion_completed=True, force_newtons=8.0)

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

