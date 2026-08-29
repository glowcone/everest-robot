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
    """The workflow-visible outcome of a named-position move.

    The optional fields carry the measurable detail the robot runtime produces
    (:class:`everest_robot.robot.contracts.MotionResult`). They default so results
    persisted by an older revision still load.
    """

    reached: bool
    position_name: str
    already_at_target: bool = False
    max_tracking_error_rad: float | None = None
    elapsed_s: float | None = None
    config_digest: str | None = None
    failure_reason: str | None = None
    failure_detail: str | None = None


@dataclass(frozen=True)
class AttachmentResult:
    """The workflow-visible outcome of the attachment maneuver.

    ``force_newtons`` is optional because the Maker Arm has no force sensor: a controller
    that cannot measure force reports ``None`` rather than a fabricated number.
    """

    motion_completed: bool
    controller: str
    force_newtons: float | None = None
    checkpoint: str | None = None
    steps: int | None = None
    missed_deadlines: int | None = None
    termination: str | None = None
    config_digest: str | None = None
    failure_reason: str | None = None
    failure_detail: str | None = None


@dataclass(frozen=True)
class VerificationResult:
    secure: bool
    confidence: float
    recovery_target: RecoveryTarget | None = None
    reason: str | None = None


def json_dict(value: object) -> dict[str, Any]:
    """Convert a domain result to an Absurd-compatible JSON mapping."""

    return asdict(value)
