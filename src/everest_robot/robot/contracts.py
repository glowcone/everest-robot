"""Typed contracts shared by the robot runtime.

Two families live here and they are deliberately different:

* **Runtime structures** (:class:`JointState`, :class:`JointCommand`,
  :class:`MotionProfile`) are read and written every control tick. They hold tuples of
  raw floats, are never persisted, and may contain ``nan`` where hardware reported no
  feedback.
* **Durable results** (:class:`MotionResult`, :class:`PolicyRunResult`,
  :class:`ReplayResult`, :class:`RecordingResult`) are what a workflow step stores. They
  are JSON-serializable through :meth:`JsonResult.to_json`, which is the only
  representation Absurd ever sees. Field names in those classes are a durable interface:
  workflows started on an older revision replay stored values by name.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

# A cancellation check polled by every long-running physical loop. Returning ``True``
# means "stop as soon as it is safe to", not "stop immediately".
CancelCheck = Callable[[], bool]

# Called during long physical operations so the worker keeps its claim on the task.
Heartbeat = Callable[[], None]


class FailureReason(StrEnum):
    """Why a physical operation stopped short.

    Persisted inside durable results, so values are a stable interface. The workflow
    routes recovery on these, so keep them coarse enough to act on.
    """

    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    TRACKING_ERROR = "tracking_error"
    MOTOR_FAULT = "motor_fault"
    STALE_FEEDBACK = "stale_feedback"
    LIMIT_VIOLATION = "limit_violation"
    NOT_ENABLED = "not_enabled"
    IDENTITY_MISMATCH = "identity_mismatch"
    UNKNOWN_POSITION = "unknown_position"
    SCHEMA_MISMATCH = "schema_mismatch"
    DEADLINE_MISSED = "deadline_missed"
    POLICY_ERROR = "policy_error"
    NOT_IMPLEMENTED = "not_implemented"


class ArmLifecycle(StrEnum):
    """Mirrors ``maker_arm.ArmState``; commands are only accepted in ``ENABLED``."""

    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    ENABLED = "enabled"
    FAULT = "fault"


def _jsonify(value: Any) -> Any:
    """Convert a dataclass field value to something ``json.dumps`` accepts.

    Tuples become lists and enums become their string values. Non-finite floats become
    ``None``: ``nan`` is how the hardware layer reports "no feedback for this joint", and
    emitting a bare ``NaN`` token would produce a stored result that strict JSON readers
    reject.
    """

    if isinstance(value, StrEnum):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonify(item) for item in value]
    return value


class JsonResult:
    """Mixin giving durable dataclasses an Absurd-compatible mapping."""

    def to_json(self) -> dict[str, Any]:
        return {key: _jsonify(value) for key, value in asdict(self).items()}  # type: ignore[call-overload]


@dataclass(frozen=True, slots=True)
class RobotIdentity:
    """Who the arm is, as agreed between configuration and hardware.

    ``calibration_id`` is the safety-relevant field: joint presets captured under one
    calibration are meaningless under another, so every preset and every stored session
    carries the identity that produced it.
    """

    robot_id: str
    model: str
    calibration_id: str
    joint_names: tuple[str, ...]
    units: str = "radians"

    def matches(self, other: RobotIdentity) -> bool:
        return (
            self.model == other.model
            and self.calibration_id == other.calibration_id
            and self.joint_names == other.joint_names
            and self.units == other.units
        )

    def mismatch_detail(self, other: RobotIdentity) -> str | None:
        """A human-readable description of the first mismatching field."""

        for name in ("model", "calibration_id", "joint_names", "units"):
            mine, theirs = getattr(self, name), getattr(other, name)
            if mine != theirs:
                return f"{name}: configured {mine!r}, hardware {theirs!r}"
        return None


@dataclass(frozen=True, slots=True)
class JointLimit:
    """An absolute soft limit in calibrated joint coordinates, owned by the driver."""

    name: str
    lower_rad: float
    upper_rad: float

    def contains(self, value: float, margin: float = 0.0) -> bool:
        return self.lower_rad + margin <= value <= self.upper_rad - margin

    def clamp(self, value: float, margin: float = 0.0) -> float:
        return min(max(value, self.lower_rad + margin), self.upper_rad - margin)


@dataclass(frozen=True, slots=True)
class JointState:
    """One synchronized feedback sample. Hot-loop structure, never persisted as-is.

    ``sequence`` carries each joint's monotonic feedback counter. Two consecutive reads
    with an unchanged counter mean the value is cached, not fresh, which is a different
    failure from a value that is fresh but wrong.
    """

    names: tuple[str, ...]
    positions: tuple[float, ...]
    velocities: tuple[float, ...]
    torques: tuple[float, ...]
    temperatures: tuple[float, ...]
    fault_bits: tuple[int, ...]
    sequence: tuple[int, ...]
    monotonic_s: float
    lifecycle: ArmLifecycle
    fault_reason: str | None = None

    @property
    def has_fault(self) -> bool:
        return self.lifecycle is ArmLifecycle.FAULT or any(self.fault_bits)

    @property
    def all_finite(self) -> bool:
        return all(math.isfinite(position) for position in self.positions)

    def position_of(self, name: str) -> float:
        return self.positions[self.names.index(name)]

    def tracking_errors(self, targets: Sequence[float]) -> tuple[float, ...]:
        """Per-joint |measured - commanded|; ``inf`` where feedback is missing."""

        return tuple(
            abs(position - target) if math.isfinite(position) else math.inf
            for position, target in zip(self.positions, targets, strict=True)
        )

    def max_tracking_error(self, targets: Sequence[float]) -> float:
        return max(self.tracking_errors(targets), default=0.0)


@dataclass(frozen=True, slots=True)
class JointCommand:
    """A validated joint-space command, after limit clipping."""

    names: tuple[str, ...]
    targets: tuple[float, ...]
    clipped_joints: tuple[str, ...] = ()

    @property
    def was_clipped(self) -> bool:
        return bool(self.clipped_joints)


@dataclass(frozen=True, slots=True)
class MotionProfile:
    """Bounds for one commanded motion, after per-preset overrides are applied."""

    max_velocity_rad_s: float
    max_acceleration_rad_s2: float
    tolerance_rad: float
    settle_time_s: float
    timeout_s: float
    control_rate_hz: float

    def scaled(self, speed_scale: float) -> MotionProfile:
        """Return the same profile at a fraction of its speed.

        Scaling above 1.0 is refused: a preset's recorded velocity is an approved
        hardware bound, not a default to be exceeded by a caller.
        """

        if not 0.0 < speed_scale <= 1.0:
            raise ValueError(f"speed_scale must be in (0, 1], got {speed_scale}")
        return MotionProfile(
            max_velocity_rad_s=self.max_velocity_rad_s * speed_scale,
            max_acceleration_rad_s2=self.max_acceleration_rad_s2 * speed_scale,
            tolerance_rad=self.tolerance_rad,
            settle_time_s=self.settle_time_s,
            timeout_s=self.timeout_s,
            control_rate_hz=self.control_rate_hz,
        )


@dataclass(frozen=True, slots=True)
class MotionResult(JsonResult):
    """The durable outcome of one named-position move."""

    position_name: str
    reached: bool
    joint_names: tuple[str, ...]
    final_joints: tuple[float, ...]
    max_tracking_error_rad: float
    elapsed_s: float
    commands_sent: int
    robot_id: str
    calibration_id: str
    config_digest: str
    already_at_target: bool = False
    dry_run: bool = False
    # The duration the bounded trajectory was planned to take. Reported for a dry run,
    # where nothing is commanded and ``elapsed_s`` says nothing about the move.
    planned_duration_s: float = 0.0
    clipped_joints: tuple[str, ...] = ()
    waypoints: tuple[str, ...] = ()
    failure_reason: FailureReason | None = None
    failure_detail: str | None = None


class TerminationReason(StrEnum):
    """Why a rollout or replay loop ended. Persisted."""

    COMPLETED = "completed"
    MAX_STEPS = "max_steps"
    MAX_DURATION = "max_duration"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class PolicyRunResult(JsonResult):
    """The durable outcome of one policy or VLA rollout."""

    controller: str
    checkpoint: str
    task: str | None
    fps: float
    steps: int
    elapsed_s: float
    missed_deadlines: int
    max_step_latency_s: float
    termination: TerminationReason
    joint_names: tuple[str, ...]
    final_joints: tuple[float, ...]
    robot_id: str
    calibration_id: str
    config_digest: str
    clipped_joints: tuple[str, ...] = ()
    episode_id: str | None = None
    failure_reason: FailureReason | None = None
    failure_detail: str | None = None


@dataclass(frozen=True, slots=True)
class ReplayResult(JsonResult):
    """The durable outcome of one stored-session replay."""

    session_id: str
    episode_index: int
    frames_requested: int
    frames_sent: int
    elapsed_s: float
    speed_scale: float
    dry_run: bool
    termination: TerminationReason
    robot_id: str
    calibration_id: str
    config_digest: str
    clipped_joints: tuple[str, ...] = ()
    failure_reason: FailureReason | None = None
    failure_detail: str | None = None


@dataclass(frozen=True, slots=True)
class RecordingResult(JsonResult):
    """The durable outcome of one recorded session."""

    session_id: str
    dataset_path: str
    episode_index: int
    frames: int
    fps: float
    task: str | None
    robot_id: str
    calibration_id: str
    config_digest: str
    revisions: dict[str, str] = field(default_factory=dict)
    failure_reason: FailureReason | None = None
    failure_detail: str | None = None
