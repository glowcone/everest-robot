"""Versioned robot parameters owned by ``everest-robot``.

This file is the boundary between operator-approved robot configuration and the runtime.
It deliberately does *not* duplicate motor directions, offsets, gains or mechanical limits:
those belong to ``maker-arm-sdk``'s hardware profile and are validated against the
connected arm at startup instead of being restated here.

Everything is validated eagerly and unknown fields are rejected. A typo in a preset is a
physical hazard, not a shrug: silently ignoring ``max_velocity_rad`` (missing ``_s``) would
run an approved-looking preset at the default speed.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from everest_robot.robot.contracts import MotionProfile, RobotIdentity

SCHEMA_VERSION = 1

# Presets are stored in calibrated joint coordinates. Radians is the only unit the
# maker-arm driver speaks, so accepting anything else here would mean a silent conversion.
SUPPORTED_UNITS = ("radians",)

_MOTION_FIELDS = (
    "max_velocity_rad_s",
    "max_acceleration_rad_s2",
    "tolerance_rad",
    "settle_time_s",
    "timeout_s",
    "control_rate_hz",
)


class ParameterError(ValueError):
    """The parameters file is unusable. Never raised for a recoverable condition."""


class IdentityMismatch(ParameterError):
    """Configuration describes a different arm or calibration than the one connected."""


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ParameterError(f"{where}: expected a mapping, got {type(value).__name__}")
    for key in value:
        if not isinstance(key, str):
            raise ParameterError(f"{where}: keys must be strings, got {key!r}")
    return value


def _reject_unknown(value: Mapping[str, Any], allowed: Sequence[str], where: str) -> None:
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise ParameterError(
            f"{where}: unknown field(s) {', '.join(unknown)}; allowed: {', '.join(sorted(allowed))}"
        )


def _require(value: Mapping[str, Any], required: Sequence[str], where: str) -> None:
    missing = sorted(set(required) - set(value))
    if missing:
        raise ParameterError(f"{where}: missing required field(s) {', '.join(missing)}")


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ParameterError(f"{where}: expected a non-empty string, got {value!r}")
    return value


def _number(value: Any, where: str, *, positive: bool = True, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ParameterError(f"{where}: expected a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ParameterError(f"{where}: expected a finite number, got {value!r}")
    if positive and (number < 0 or (number == 0 and not allow_zero)):
        raise ParameterError(f"{where}: expected a positive number, got {value!r}")
    return number


def _flag(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise ParameterError(f"{where}: expected true or false, got {value!r}")
    return value


def _motion_profile(
    value: Mapping[str, Any], where: str, base: MotionProfile | None
) -> MotionProfile:
    """Build a profile, optionally overriding only the fields a preset restates."""

    _reject_unknown(value, _MOTION_FIELDS, where)
    if base is None:
        _require(value, _MOTION_FIELDS, where)
    fields: dict[str, float] = {}
    for name in _MOTION_FIELDS:
        if name in value:
            # settle_time_s may legitimately be zero; every other bound may not.
            fields[name] = _number(
                value[name], f"{where}.{name}", allow_zero=name == "settle_time_s"
            )
        else:
            fields[name] = getattr(base, name)  # type: ignore[union-attr]
    return MotionProfile(**fields)


@dataclass(frozen=True, slots=True)
class NamedPosition:
    """One operator-approved staging pose in calibrated joint coordinates.

    ``calibration_id``, ``approved_by`` and ``captured_at`` are mandatory and carry the
    provenance of the measurement. A preset whose ``calibration_id`` does not match the
    robot's is rejected at load time rather than at motion time.
    """

    name: str
    joints: tuple[float, ...]
    calibration_id: str
    approved_by: str
    captured_at: date
    profile: MotionProfile
    notes: str | None = None


@dataclass(frozen=True, slots=True)
class NamedTransition:
    """An approved waypoint sequence between two poses.

    Exists because direct joint-space interpolation between two safe poses is not itself
    guaranteed safe. When a transition is defined, the runtime must use it.
    """

    name: str
    waypoints: tuple[str, ...]
    notes: str | None = None


@dataclass(frozen=True, slots=True)
class PolicySettings:
    default_controller: str
    fps: float
    max_duration_s: float


@dataclass(frozen=True, slots=True)
class ReplaySettings:
    require_matching_robot_id: bool
    require_matching_calibration_id: bool
    safe_start_position: str | None
    max_speed_scale: float


@dataclass(frozen=True, slots=True)
class RobotParameters:
    """The whole validated parameters file."""

    schema_version: int
    identity: RobotIdentity
    motion_defaults: MotionProfile
    named_positions: Mapping[str, NamedPosition]
    named_transitions: Mapping[str, NamedTransition]
    policy: PolicySettings
    replay: ReplaySettings
    config_digest: str
    source: str

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self.identity.joint_names

    def position(self, name: str) -> NamedPosition:
        try:
            return self.named_positions[name]
        except KeyError:
            known = ", ".join(sorted(self.named_positions)) or "none defined"
            raise ParameterError(f"unknown named position {name!r}; known: {known}") from None

    def transition(self, name: str) -> NamedTransition:
        try:
            return self.named_transitions[name]
        except KeyError:
            known = ", ".join(sorted(self.named_transitions)) or "none defined"
            raise ParameterError(f"unknown named transition {name!r}; known: {known}") from None

    def verify_identity(self, hardware: RobotIdentity) -> None:
        """Raise unless the connected arm is the one these parameters describe.

        Called once per session, before anything is enabled. Presets captured under a
        different calibration describe a different physical pose.
        """

        detail = self.identity.mismatch_detail(hardware)
        if detail is not None:
            raise IdentityMismatch(
                f"{self.source}: parameters do not match the connected arm ({detail})"
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> RobotParameters:
        source = Path(path)
        raw = source.read_bytes()
        digest = "sha256:" + hashlib.sha256(raw).hexdigest()
        document = yaml.safe_load(raw.decode("utf-8"))
        return cls.from_mapping(document, config_digest=digest, source=str(source))

    @classmethod
    def from_mapping(
        cls,
        document: Any,
        *,
        config_digest: str,
        source: str = "<mapping>",
    ) -> RobotParameters:
        root = _mapping(document, source)
        _reject_unknown(
            root,
            ("schema_version", "robot", "motion_defaults", "named_positions",
             "named_transitions", "policy", "replay"),
            source,
        )
        _require(root, ("schema_version", "robot", "motion_defaults", "policy", "replay"), source)

        version = root["schema_version"]
        if version != SCHEMA_VERSION:
            raise ParameterError(
                f"{source}: schema_version {version!r} is not supported (expected {SCHEMA_VERSION})"
            )

        identity = _parse_robot(_mapping(root["robot"], f"{source}.robot"), f"{source}.robot")
        defaults = _motion_profile(
            _mapping(root["motion_defaults"], f"{source}.motion_defaults"),
            f"{source}.motion_defaults",
            base=None,
        )
        positions = _parse_positions(
            _mapping(root.get("named_positions") or {}, f"{source}.named_positions"),
            f"{source}.named_positions",
            identity=identity,
            defaults=defaults,
        )
        transitions = _parse_transitions(
            _mapping(root.get("named_transitions") or {}, f"{source}.named_transitions"),
            f"{source}.named_transitions",
            positions=positions,
        )
        policy = _parse_policy(_mapping(root["policy"], f"{source}.policy"), f"{source}.policy")
        replay = _parse_replay(
            _mapping(root["replay"], f"{source}.replay"), f"{source}.replay", positions=positions
        )

        return cls(
            schema_version=SCHEMA_VERSION,
            identity=identity,
            motion_defaults=defaults,
            named_positions=positions,
            named_transitions=transitions,
            policy=policy,
            replay=replay,
            config_digest=config_digest,
            source=source,
        )


def _parse_robot(value: Mapping[str, Any], where: str) -> RobotIdentity:
    _reject_unknown(value, ("id", "model", "calibration_id", "joint_order", "units"), where)
    _require(value, ("id", "model", "calibration_id", "joint_order", "units"), where)

    order = value["joint_order"]
    if not isinstance(order, list) or not order:
        raise ParameterError(f"{where}.joint_order: expected a non-empty list")
    names = tuple(_text(name, f"{where}.joint_order[{index}]") for index, name in enumerate(order))
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ParameterError(f"{where}.joint_order: duplicate joint(s) {', '.join(duplicates)}")

    units = _text(value["units"], f"{where}.units")
    if units not in SUPPORTED_UNITS:
        raise ParameterError(f"{where}.units: {units!r} is not supported (expected 'radians')")

    return RobotIdentity(
        robot_id=_text(value["id"], f"{where}.id"),
        model=_text(value["model"], f"{where}.model"),
        calibration_id=_text(value["calibration_id"], f"{where}.calibration_id"),
        joint_names=names,
        units=units,
    )


def _parse_date(value: Any, where: str) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as error:
            raise ParameterError(
                f"{where}: expected an ISO date (YYYY-MM-DD), got {value!r}"
            ) from error
    raise ParameterError(f"{where}: expected an ISO date (YYYY-MM-DD), got {value!r}")


def _parse_positions(
    value: Mapping[str, Any],
    where: str,
    *,
    identity: RobotIdentity,
    defaults: MotionProfile,
) -> dict[str, NamedPosition]:
    positions: dict[str, NamedPosition] = {}
    for name, raw in value.items():
        scope = f"{where}.{name}"
        entry = _mapping(raw, scope)
        provenance = ("joints", "calibration_id", "approved_by", "captured_at", "notes")
        _reject_unknown(entry, provenance + _MOTION_FIELDS, scope)
        _require(entry, ("joints", "calibration_id", "approved_by", "captured_at"), scope)

        joints = entry["joints"]
        if not isinstance(joints, list):
            raise ParameterError(
                f"{scope}.joints: expected a list of {len(identity.joint_names)} values"
            )
        if len(joints) != len(identity.joint_names):
            raise ParameterError(
                f"{scope}.joints: expected one value per joint in joint_order "
                f"({len(identity.joint_names)}), got {len(joints)}"
            )
        values = tuple(
            _number(item, f"{scope}.joints[{index}]", positive=False)
            for index, item in enumerate(joints)
        )

        calibration_id = _text(entry["calibration_id"], f"{scope}.calibration_id")
        if calibration_id != identity.calibration_id:
            # A pose measured under another calibration is a different physical pose.
            raise ParameterError(
                f"{scope}.calibration_id: preset was captured under {calibration_id!r} but this "
                f"file describes {identity.calibration_id!r}; recapture the preset or remove it"
            )

        overrides = {key: entry[key] for key in _MOTION_FIELDS if key in entry}
        positions[name] = NamedPosition(
            name=name,
            joints=values,
            calibration_id=calibration_id,
            approved_by=_text(entry["approved_by"], f"{scope}.approved_by"),
            captured_at=_parse_date(entry["captured_at"], f"{scope}.captured_at"),
            profile=_motion_profile(overrides, scope, base=defaults),
            notes=entry.get("notes"),
        )
    return positions


def _parse_transitions(
    value: Mapping[str, Any],
    where: str,
    *,
    positions: Mapping[str, NamedPosition],
) -> dict[str, NamedTransition]:
    transitions: dict[str, NamedTransition] = {}
    for name, raw in value.items():
        scope = f"{where}.{name}"
        entry = _mapping(raw, scope)
        _reject_unknown(entry, ("waypoints", "notes"), scope)
        _require(entry, ("waypoints",), scope)

        waypoints = entry["waypoints"]
        if not isinstance(waypoints, list) or len(waypoints) < 2:
            raise ParameterError(f"{scope}.waypoints: expected at least two named positions")
        names = tuple(
            _text(item, f"{scope}.waypoints[{index}]") for index, item in enumerate(waypoints)
        )
        for index, waypoint in enumerate(names):
            if waypoint not in positions:
                raise ParameterError(
                    f"{scope}.waypoints[{index}]: {waypoint!r} is not a named position"
                )
            if index and waypoint == names[index - 1]:
                raise ParameterError(
                    f"{scope}.waypoints[{index}]: {waypoint!r} repeats the previous waypoint"
                )
        transitions[name] = NamedTransition(name=name, waypoints=names, notes=entry.get("notes"))
    return transitions


def _parse_policy(value: Mapping[str, Any], where: str) -> PolicySettings:
    _reject_unknown(value, ("default_controller", "fps", "max_duration_s"), where)
    _require(value, ("default_controller", "fps", "max_duration_s"), where)
    return PolicySettings(
        default_controller=_text(value["default_controller"], f"{where}.default_controller"),
        fps=_number(value["fps"], f"{where}.fps"),
        max_duration_s=_number(value["max_duration_s"], f"{where}.max_duration_s"),
    )


def _parse_replay(
    value: Mapping[str, Any],
    where: str,
    *,
    positions: Mapping[str, NamedPosition],
) -> ReplaySettings:
    allowed = (
        "require_matching_robot_id",
        "require_matching_calibration_id",
        "safe_start_position",
        "max_speed_scale",
    )
    _reject_unknown(value, allowed, where)
    _require(value, allowed, where)

    start = value["safe_start_position"]
    if start is not None:
        start = _text(start, f"{where}.safe_start_position")
        if start not in positions:
            raise ParameterError(
                f"{where}.safe_start_position: {start!r} is not a named position"
            )

    scale = _number(value["max_speed_scale"], f"{where}.max_speed_scale")
    if scale > 1.0:
        raise ParameterError(
            f"{where}.max_speed_scale: {scale} exceeds the recorded session speed"
        )

    return ReplaySettings(
        require_matching_robot_id=_flag(
            value["require_matching_robot_id"], f"{where}.require_matching_robot_id"
        ),
        require_matching_calibration_id=_flag(
            value["require_matching_calibration_id"], f"{where}.require_matching_calibration_id"
        ),
        safe_start_position=start,
        max_speed_scale=scale,
    )
