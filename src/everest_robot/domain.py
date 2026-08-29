"""Domain types shared by the workflow and future robot integrations."""

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class RecoveryTarget(StrEnum):
    """The stage verification says must be attempted again."""

    LOCATE_ROPE = "locate_rope"
    ATTACH = "attach"
    PICK_UP = "pick_up"


@dataclass(frozen=True)
class PickupResult:
    secured: bool
    controller: str


@dataclass(frozen=True)
class RopePose:
    frame: str
    x: float
    y: float
    z: float
    detector: str


@dataclass(frozen=True)
class AttachmentResult:
    motion_completed: bool
    force_newtons: float


@dataclass(frozen=True)
class VerificationResult:
    secure: bool
    confidence: float
    recovery_target: RecoveryTarget | None = None
    reason: str | None = None


def json_dict(value: object) -> dict[str, Any]:
    """Convert a domain result to an Absurd-compatible JSON mapping."""

    return asdict(value)

