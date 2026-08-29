"""Domain types shared by the workflow and the robot integrations."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
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


class LimitPolicy(StrEnum):
    """What preflight does with a stored action that falls outside the active limits.

    There is no implicit option: silent clipping is how a replay ends up executing
    something other than what was recorded, so the choice is always explicit.
    """

    #: Any out-of-range action fails preflight.
    REJECT = "reject"
    #: Deviations up to ``max_limit_deviation_deg`` are clamped and reported; larger fail.
    CLAMP_WITHIN_TOLERANCE = "clamp_within_tolerance"
    #: Every out-of-range value is clamped, and the affected frames are reported.
    CLAMP = "clamp"


@dataclass(frozen=True)
class ReplayRequest:
    """What to replay. JSON-serializable; this is what a workflow task receives.

    ``robot_id`` and ``calibration_id`` are required because the dataset does not identify
    the physical arm that recorded it. They are checked against the connected hardware and
    against the operator-approved mapping in the robot parameters file.
    """

    repo_id: str
    revision: str
    episode: int
    robot_id: str
    calibration_id: str
    speed: float = 1.0
    start_frame: int = 0
    end_frame: int | None = None
    limit_policy: LimitPolicy = LimitPolicy.REJECT
    max_limit_deviation_deg: float = 0.0
    dry_run: bool = False

    @classmethod
    def from_json(cls, params: Mapping[str, Any]) -> "ReplayRequest":
        """Build a request from task parameters, rejecting anything unrecognized.

        A misspelled ``dry_run`` that silently defaults to False would run a physical
        replay someone asked to simulate.
        """

        known = {field.name for field in fields(cls)}
        unknown = sorted(set(params) - known)
        if unknown:
            raise ValueError(f"unknown replay parameter(s): {', '.join(unknown)}")
        required = {"repo_id", "revision", "episode", "robot_id", "calibration_id"}
        missing = sorted(required - set(params))
        if missing:
            raise ValueError(f"missing replay parameter(s): {', '.join(missing)}")

        values = dict(params)
        return cls(
            repo_id=str(values["repo_id"]),
            revision=str(values["revision"]),
            episode=int(values["episode"]),
            robot_id=str(values["robot_id"]),
            calibration_id=str(values["calibration_id"]),
            speed=float(values.get("speed", 1.0)),
            start_frame=int(values.get("start_frame", 0)),
            end_frame=None if values.get("end_frame") is None else int(values["end_frame"]),
            limit_policy=LimitPolicy(values.get("limit_policy", LimitPolicy.REJECT)),
            max_limit_deviation_deg=float(values.get("max_limit_deviation_deg", 0.0)),
            dry_run=bool(values.get("dry_run", False)),
        )


@dataclass(frozen=True)
class ReplayResult:
    """The durable outcome of one replay attempt.

    Per-frame actions are deliberately absent: they already exist in the pinned dataset
    revision, and copying thousands of them into a workflow record would bloat it for no
    recoverable information.
    """

    repo_id: str
    revision: str
    episode: int
    robot_id: str
    calibration_id: str
    completed: bool
    frames_planned: int
    frames_sent: int
    first_frame: int
    last_frame_sent: int | None
    elapsed_s: float
    effective_fps: float
    clipped_frames: int
    max_clipping_deg: float
    dry_run: bool = False
    config_digest: str | None = None
    stopped_reason: str | None = None
