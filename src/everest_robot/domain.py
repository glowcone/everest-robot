"""Domain types shared by the workflow and future robot integrations."""

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class RecoveryTarget(StrEnum):
    """The stage verification says must be attempted again."""

    LOCALIZE_AND_PICK_UP = "localize_and_pick_up"
    GO_TO_KNOWN_POSITION = "go_to_known_position"
    ATTACH = "attach"


@dataclass(frozen=True)
class CarabinerPickupResult:
    secured: bool
    frame: str
    x: float
    y: float
    z: float
    detector: str
    grasp_planner: str


@dataclass(frozen=True)
class PositionResult:
    reached: bool
    position_name: str


@dataclass(frozen=True)
class AttachmentResult:
    motion_completed: bool
    force_newtons: float
    controller: str


@dataclass(frozen=True)
class VerificationResult:
    secure: bool
    confidence: float
    recovery_target: RecoveryTarget | None = None
    reason: str | None = None


def json_dict(value: object) -> dict[str, Any]:
    """Convert a domain result to an Absurd-compatible JSON mapping."""

    return asdict(value)
