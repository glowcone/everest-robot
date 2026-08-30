"""The wrist camera's calibration: an image Jacobian measured on the arm itself.

This is the alternative to :mod:`everest_robot.pixel_map` for the FSM's ``SEARCH_CV``, and
it exists because the two cameras are different kinds of instrument. The fixed camera is
bolted to the bench, so a pixel names a place in the world and can be mapped once and for
all to the joint vector that grasps there. The wrist camera moves *with* the arm, so the
same pixel means a different place at every pose and no such map exists. What is stable
instead is the derivative: how the carabiner's image moves when a joint moves. That is the
image Jacobian, and measuring it is the calibration this file stores.

The detector is :mod:`everest_robot.carabiner_detect` -- the classical Petzl Spirit
segmentation that produces the aperture, spine and insertion point from a wrist-camera
frame. The features servoed on are four numbers taken from one of its detections:

* ``u``, ``v`` -- the insertion point, in pixels. Where the fingertip should go, not the
  aperture centre: it is already biased toward the spine by the detector.
* ``scale_px`` -- the square root of the frame mask's area. Nothing here measures range, and
  a monocular camera cannot; apparent size is the proxy, and taking the square root puts it
  in pixels so it moves linearly with the others under approach.
* ``spine_deg`` -- the spine tangent. Its difference is wrapped into ``[-90, 90)`` because
  the detector's angle is undirected: a spine at +89 degrees and one at -89 are two degrees
  apart, not 178.

Only the last of those is angular, and only ``spine_deg`` gets the wrap. That asymmetry is
why the error is computed here rather than left to a caller subtracting two tuples.

**The Jacobian is measured at the goal pose and nowhere else.** Bumping a joint from a
different pose would measure a different derivative -- the map from joints to image is not
linear, only locally so. That has two consequences kept deliberately visible:

* The calibration is only exactly right where it was taught. Away from the goal it gives a
  direction rather than a step, which is enough for a servo whose step size is clamped by
  :class:`~everest_robot.robot.visual_tracking.VisualTracker` anyway.
* A solve asking for more than :attr:`WristServoCalibration.max_delta_rad` on any joint is
  refused rather than clamped. A clamp would quietly turn an answer the linearization
  cannot support into a confident-looking small move in a possibly wrong direction; a
  refusal holds the arm and says so.

The camera is named, not indexed. Unlike the fixed camera -- whose device id is part of the
calibration because moving it voids every sample -- the wrist camera *is* a policy
observation, configured in ``EVEREST_CAMERAS`` and already open in the session's
:class:`~everest_robot.robot.cameras.CameraRuntime` while a policy runs. Naming it is what
lets the follower share that one open device instead of taking a second one, which on macOS
and V4L2 alike either fails or starves the first.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from everest_robot.pixel_map import RobotStamp, now_stamp

SCHEMA_VERSION = 1

#: The features servoed on, in order. ``spine_deg`` is the only angular one.
FEATURE_NAMES = ("u", "v", "scale_px", "spine_deg")
ANGULAR_FEATURES = frozenset({"spine_deg"})

#: Millimetres per pixel in the wrist view, measured at the taught goal pose. Monocular, so
#: this is a ratio at one range and not a property of the camera: at the top of the descent
#: the same pixel spans more. It exists so tolerances can be reasoned about in the units the
#: part is specified in -- a Petzl Spirit is 100 x 60 mm -- and it is used for nothing else.
#: Re-measure it if the lens, the resolution or the goal pose changes.
MM_PER_PX = 1.44

#: How close each feature must be to its goal for the approach to count as arrived, in the
#: feature's own units: pixels, pixels, pixels, degrees. These are also the weights the
#: solve normalizes by, so a loose tolerance is a feature the servo cares less about.
#:
#: The three pixel tolerances are stated in millimetres at :data:`MM_PER_PX` and converted,
#: because that is the form a person can check against the part: 8 mm of lateral placement
#: on a 60 mm-wide frame, and 10 mm of standoff. The spine tolerance is already angular.
DEFAULT_TOLERANCE = (8.0 / MM_PER_PX, 8.0 / MM_PER_PX, 10.0 / MM_PER_PX, 6.0)

#: Fraction of the solved joint step actually commanded. Below one because the Jacobian is
#: an approximation everywhere except the goal pose, and a proportional servo on an
#: approximate derivative overshoots at unit gain.
DEFAULT_GAIN = 0.6

#: Levenberg damping, as a fraction of the mean normalized curvature. Scale-free on
#: purpose: the raw magnitudes depend on the lens and on the tolerances.
DEFAULT_DAMPING = 0.02

#: Largest per-joint step a single solve may ask for. Beyond this the answer is refused.
DEFAULT_MAX_DELTA_RAD = 0.35

#: A tolerance that means "this feature is not being servoed". Large and finite rather than
#: infinite, because the solve divides by it: a feature whose error is scaled to a
#: thousandth of the others contributes nothing to the step and can never fail the arrival
#: test, which is exactly the intent, and no special case is needed anywhere to express it.
IGNORED_TOLERANCE = 1.0e4


class WristServoError(RuntimeError):
    """The wrist calibration is missing, malformed, or does not fit this arm."""


class UnsupportedSolve(WristServoError):
    """The solve produced an answer the linearization cannot support. Hold, do not move."""


@runtime_checkable
class WristDetection(Protocol):
    """What this calibration needs from one ``carabiner.detect`` detection."""

    @property
    def insert(self) -> tuple[float, float]: ...

    @property
    def spine(self) -> tuple[float, float]: ...

    @property
    def aperture(self) -> tuple[float, float]: ...

    @property
    def area(self) -> float: ...

    @property
    def spine_angle(self) -> float: ...


#: Which point of the detection ``u, v`` track. ``insert`` is the detector's own choice for
#: where the fingertip goes -- the aperture centroid biased toward the spine -- and is what
#: a handover to the clip policy should aim at. ``spine`` is the midpoint of the bar the
#: gripper closes on, which is the easier one for a person to check by eye against an
#: overlay. ``aperture`` is the raw hole centroid.
FEATURE_POINTS = ("insert", "spine", "aperture")


# ── features ───────────────────────────────────────────────────────────────────────
def features_of(detection: WristDetection, point: str = "insert") -> tuple[float, ...]:
    """One detection reduced to the servo's feature vector, in :data:`FEATURE_NAMES` order.

    Which point drives ``u, v`` is a choice, not a detail: all three are points on the same
    rigid object, so they share the object's translation but not its rotation about itself.
    """

    if point not in FEATURE_POINTS:
        raise WristServoError(
            f"unknown feature point {point!r} (expected {', '.join(FEATURE_POINTS)})"
        )
    area = float(detection.area)
    if not math.isfinite(area) or area <= 0.0:
        raise WristServoError("the detection reported no mask area; nothing to scale from")
    where = getattr(detection, point)
    return (
        float(where[0]),
        float(where[1]),
        math.sqrt(area),
        float(detection.spine_angle),
    )


def wrap_deg(value: float) -> float:
    """An undirected angle difference, folded into ``[-90, 90)``."""

    return (value + 90.0) % 180.0 - 90.0


def feature_error(
    current: Sequence[float],
    goal: Sequence[float],
    names: Sequence[str] = FEATURE_NAMES,
) -> tuple[float, ...]:
    """``current - goal`` per feature, with the angular one wrapped."""

    if len(current) != len(names) or len(goal) != len(names):
        raise WristServoError(
            f"expected {len(names)} features ({', '.join(names)}), got "
            f"{len(current)} current and {len(goal)} goal"
        )
    return tuple(
        wrap_deg(float(a) - float(b)) if name in ANGULAR_FEATURES else float(a) - float(b)
        for name, a, b in zip(names, current, goal, strict=True)
    )


# ── teaching ───────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class BumpTrial:
    """One joint moved by a known amount, with the image measured either side of it."""

    joint: str
    delta_rad: float
    before: tuple[float, ...]
    after: tuple[float, ...]

    def feature_delta(self, names: Sequence[str] = FEATURE_NAMES) -> tuple[float, ...]:
        return feature_error(self.after, self.before, names)

    def to_json(self) -> dict[str, Any]:
        return {
            "joint": self.joint,
            "delta_rad": self.delta_rad,
            "before": list(self.before),
            "after": list(self.after),
        }

    @classmethod
    def from_json(cls, raw: Mapping[str, Any], where: str) -> BumpTrial:
        try:
            return cls(
                joint=str(raw["joint"]),
                delta_rad=float(raw["delta_rad"]),
                before=tuple(float(value) for value in raw["before"]),
                after=tuple(float(value) for value in raw["after"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise WristServoError(f"{where}: malformed bump trial ({error})") from None


def fit_jacobian(
    trials: Sequence[BumpTrial],
    servo_joints: Sequence[str],
    names: Sequence[str] = FEATURE_NAMES,
) -> tuple[np.ndarray, dict[str, Any]]:
    """One column per joint, fitted through the origin over that joint's bumps.

    Column-at-a-time rather than one joint least-squares over everything, because that is
    how the evidence was gathered: each trial moved exactly one joint, so each column is
    identified on its own and a bad joint cannot contaminate its neighbours. Through the
    origin because a bump of zero must produce an image displacement of zero -- an
    intercept here would be fitting the detector's noise and calling it a derivative.

    The reported residual is what tells an operator whether a column is trustworthy: a
    joint whose bumps disagree with each other has either moved something else (backlash,
    the carabiner nudged) or barely moves the image at all.
    """

    if not trials:
        raise WristServoError("no bump trials; nothing to fit a Jacobian from")
    jacobian = np.zeros((len(names), len(servo_joints)), dtype=float)
    report: dict[str, Any] = {}
    for column, joint in enumerate(servo_joints):
        rows = [trial for trial in trials if trial.joint == joint]
        if not rows:
            raise WristServoError(
                f"joint {joint!r} has no bump trials; every servo joint needs at least one"
            )
        steps = np.asarray([trial.delta_rad for trial in rows], dtype=float)
        deltas = np.asarray([trial.feature_delta(names) for trial in rows], dtype=float)
        denominator = float(steps @ steps)
        if denominator <= 0.0:
            raise WristServoError(f"joint {joint!r} was bumped by zero; nothing to divide by")
        slope = (steps @ deltas) / denominator
        jacobian[:, column] = slope
        residual = deltas - np.outer(steps, slope)
        report[joint] = {
            "trials": len(rows),
            "px_per_rad": [float(value) for value in slope],
            "residual_rms": [float(value) for value in np.sqrt((residual**2).mean(axis=0))],
        }
    return jacobian, report


# ── the calibration file ───────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class WristSolve:
    """What one servo step asks for, and how far from arrived it thinks it is."""

    delta_rad: dict[str, float]
    error: tuple[float, ...]
    #: The error in tolerance units. One means "exactly at tolerance on every feature".
    normalized_error: float
    settled: bool


@dataclass(frozen=True, slots=True)
class WristServoCalibration:
    """The taught goal image, the measured Jacobian, and the arm both belong to."""

    robot: RobotStamp
    camera_name: str
    joint_names: tuple[str, ...]
    servo_joints: tuple[str, ...]
    goal: tuple[float, ...]
    jacobian: np.ndarray
    feature_names: tuple[str, ...] = FEATURE_NAMES
    #: Which detected point ``u, v`` track. See :data:`FEATURE_POINTS`.
    point: str = "insert"
    tolerance: tuple[float, ...] = DEFAULT_TOLERANCE
    gain: float = DEFAULT_GAIN
    damping: float = DEFAULT_DAMPING
    max_delta_rad: float = DEFAULT_MAX_DELTA_RAD
    #: LeRobot cameras hand back RGB; ``carabiner.detect`` works in Lab derived from BGR.
    #: Getting it backwards swaps a* and b*, the axis the whole segmentation is measured on.
    color_mode: str = "rgb"
    trials: tuple[BumpTrial, ...] = ()
    validation: dict[str, Any] | None = None
    approved_by: str = ""
    created_at: str = ""

    def __post_init__(self) -> None:
        if self.color_mode not in ("rgb", "bgr"):
            raise WristServoError("color_mode must be 'rgb' or 'bgr'")
        if self.point not in FEATURE_POINTS:
            raise WristServoError(f"point must be one of {', '.join(FEATURE_POINTS)}")
        if not self.servo_joints:
            raise WristServoError("a wrist servo calibration needs at least one servo joint")
        unknown = [name for name in self.servo_joints if name not in self.joint_names]
        if unknown:
            raise WristServoError(
                f"unknown servo joint(s) {', '.join(unknown)}; this arm has "
                f"{', '.join(self.joint_names)}"
            )
        if len(set(self.servo_joints)) != len(self.servo_joints):
            raise WristServoError("servo_joints contains a duplicate")
        rows, columns = np.shape(self.jacobian)
        if (rows, columns) != (len(self.feature_names), len(self.servo_joints)):
            raise WristServoError(
                f"the Jacobian is {rows}x{columns} but {len(self.feature_names)} features and "
                f"{len(self.servo_joints)} servo joints were declared"
            )
        if len(self.goal) != len(self.feature_names):
            raise WristServoError(
                f"the goal has {len(self.goal)} values but {len(self.feature_names)} features "
                "were declared"
            )
        if len(self.tolerance) != len(self.feature_names):
            raise WristServoError(
                f"the tolerance has {len(self.tolerance)} values but "
                f"{len(self.feature_names)} features were declared"
            )
        if any(not math.isfinite(value) or value <= 0.0 for value in self.tolerance):
            raise WristServoError("every feature tolerance must be finite and positive")
        if not math.isfinite(self.gain) or not 0.0 < self.gain <= 1.0:
            raise WristServoError("gain must be finite and in (0, 1]")
        if not math.isfinite(self.damping) or self.damping < 0.0:
            raise WristServoError("damping must be finite and non-negative")
        if not math.isfinite(self.max_delta_rad) or self.max_delta_rad <= 0.0:
            raise WristServoError("max_delta_rad must be finite and positive")
        if not np.isfinite(np.asarray(self.jacobian, dtype=float)).all():
            raise WristServoError("the Jacobian has non-finite entries; re-teach it")

    # ── using it ───────────────────────────────────────────────────────────────────
    def verify(self, other: RobotStamp) -> None:
        """Refuse a calibration taught on a different arm or a different zeroing."""

        if (self.robot.robot_id, self.robot.calibration_id) != (
            other.robot_id,
            other.calibration_id,
        ):
            raise WristServoError(
                f"this wrist servo calibration was taught on {self.robot.robot_id}/"
                f"{self.robot.calibration_id} but the connected arm is {other.robot_id}/"
                f"{other.calibration_id}. Re-teach it with `robot-wrist-servo teach`."
            )

    def features(self, detection: WristDetection) -> tuple[float, ...]:
        """This calibration's feature vector for one detection, using its own point."""

        return features_of(detection, self.point)

    def retargeted(
        self,
        *,
        goal: Sequence[float] | None = None,
        tolerance: Sequence[float] | None = None,
        point: str | None = None,
    ) -> WristServoCalibration:
        """The same measured Jacobian, aimed at a different image. For debugging loops.

        The Jacobian is evidence about how the arm moves the image and is untouched; only
        what counts as *arrived* changes. That separation is what makes it legitimate to
        point a taught calibration at, say, the frame centre instead of the taught goal.

        Switching ``point`` is the one part that is an approximation, and it is only sound
        when the two ignored rows are given loose tolerances: the three candidate points sit
        on one rigid object, so the ``u, v`` rows transfer between them under the small
        camera translations this servo makes, while the ``scale_px`` and ``spine_deg`` rows
        do not. Do not use a retargeted point to drive range or rotation.
        """

        from dataclasses import replace

        return replace(
            self,
            goal=self.goal if goal is None else tuple(float(value) for value in goal),
            tolerance=(
                self.tolerance
                if tolerance is None
                else tuple(float(value) for value in tolerance)
            ),
            point=self.point if point is None else point,
        )

    def settled(self, features: Sequence[float]) -> bool:
        """Whether every feature is inside its own tolerance."""

        return all(
            abs(value) <= limit
            for value, limit in zip(
                feature_error(features, self.goal, self.feature_names),
                self.tolerance,
                strict=True,
            )
        )

    def solve(self, features: Sequence[float]) -> WristSolve:
        """The joint step that would put this detection on the goal image.

        Damped least squares in *tolerance-normalized* feature space. Normalizing is not
        cosmetic: the raw features are pixels and degrees, and an unweighted solve would
        silently decide that one degree of spine matters as much as one pixel of position.
        Dividing each row by its tolerance makes the residual dimensionless and makes the
        stopping test and the objective agree -- "settled" is exactly the unit cube.

        Damped because the arm has more servo joints than the image has features, so the
        system is underdetermined and an undamped pseudo-inverse is free to answer with a
        large motion in the null space that changes nothing in the image. The damping is
        set as a fraction of the mean curvature so it does not have to be re-tuned for a
        different lens or a different set of tolerances.
        """

        error = feature_error(features, self.goal, self.feature_names)
        weights = np.asarray(self.tolerance, dtype=float)
        normalized = np.asarray(error, dtype=float) / weights
        settled = bool(np.all(np.abs(normalized) <= 1.0))

        matrix = np.asarray(self.jacobian, dtype=float) / weights[:, None]
        normal = matrix.T @ matrix
        lam = self.damping * float(np.trace(normal)) / max(1, normal.shape[0])
        # Solve for the step that *removes* the error, hence the negation.
        step = np.linalg.solve(normal + lam * np.eye(normal.shape[0]), matrix.T @ -normalized)
        step = self.gain * step

        if not np.isfinite(step).all():
            raise UnsupportedSolve("the servo solve was not finite; the Jacobian is degenerate")
        worst = float(np.max(np.abs(step)))
        if worst > self.max_delta_rad:
            raise UnsupportedSolve(
                f"the servo asked for {worst:.3f} rad on one joint, past the "
                f"{self.max_delta_rad:.3f} rad this Jacobian is trusted for"
            )
        return WristSolve(
            delta_rad={
                joint: float(value) for joint, value in zip(self.servo_joints, step, strict=True)
            },
            error=error,
            normalized_error=float(np.linalg.norm(normalized)),
            settled=settled,
        )

    def joint_target(
        self, solve: WristSolve, measured: Sequence[float]
    ) -> tuple[float, ...]:
        """The solved step applied over the measured pose; untaught joints hold still.

        Absolute joint values would be meaningless here -- the calibration measures a
        derivative, not a pose -- so every target is measured feedback plus a delta. A
        joint outside ``servo_joints`` is one the servo has no opinion about and holds
        exactly where it is, the same rule ``robot-raise`` and the pixel map both follow.
        """

        if len(measured) != len(self.joint_names):
            raise WristServoError(
                f"expected {len(self.joint_names)} measured joints, got {len(measured)}"
            )
        return tuple(
            float(value) + solve.delta_rad.get(name, 0.0)
            for name, value in zip(self.joint_names, measured, strict=True)
        )

    # ── the file ───────────────────────────────────────────────────────────────────
    def with_fit(self, trials: Sequence[BumpTrial]) -> WristServoCalibration:
        """Refit the Jacobian from these trials, keeping everything else."""

        jacobian, report = fit_jacobian(trials, self.servo_joints, self.feature_names)
        return WristServoCalibration(
            robot=self.robot,
            camera_name=self.camera_name,
            joint_names=self.joint_names,
            servo_joints=self.servo_joints,
            goal=self.goal,
            jacobian=jacobian,
            feature_names=self.feature_names,
            point=self.point,
            tolerance=self.tolerance,
            gain=self.gain,
            damping=self.damping,
            max_delta_rad=self.max_delta_rad,
            color_mode=self.color_mode,
            trials=tuple(trials),
            validation={"columns": report},
            approved_by=self.approved_by,
            created_at=self.created_at or now_stamp(),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "created_at": self.created_at or now_stamp(),
            "approved_by": self.approved_by,
            "robot": self.robot.to_json(),
            "camera_name": self.camera_name,
            "color_mode": self.color_mode,
            "joint_names": list(self.joint_names),
            "servo_joints": list(self.servo_joints),
            "feature_names": list(self.feature_names),
            "point": self.point,
            "goal": list(self.goal),
            "tolerance": list(self.tolerance),
            "gain": self.gain,
            "damping": self.damping,
            "max_delta_rad": self.max_delta_rad,
            "jacobian": np.asarray(self.jacobian, dtype=float).tolist(),
            "trials": [trial.to_json() for trial in self.trials],
            "validation": self.validation,
        }

    def save(self, path: str | Path) -> Path:
        """Write the calibration, leaving the previous file intact until the write lands."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_json(), indent=2) + "\n")
        temporary.replace(target)
        return target

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> WristServoCalibration:
        version = int(raw.get("schema_version", 0))
        if version != SCHEMA_VERSION:
            raise WristServoError(
                f"wrist servo schema_version {version} is not {SCHEMA_VERSION}; re-teach it"
            )
        try:
            return cls(
                robot=RobotStamp.from_json(raw["robot"]),
                camera_name=str(raw["camera_name"]),
                joint_names=tuple(str(name) for name in raw["joint_names"]),
                servo_joints=tuple(str(name) for name in raw["servo_joints"]),
                goal=tuple(float(value) for value in raw["goal"]),
                jacobian=np.asarray(raw["jacobian"], dtype=float),
                feature_names=tuple(
                    str(name) for name in raw.get("feature_names", FEATURE_NAMES)
                ),
                point=str(raw.get("point", "insert")),
                tolerance=tuple(
                    float(value) for value in raw.get("tolerance", DEFAULT_TOLERANCE)
                ),
                gain=float(raw.get("gain", DEFAULT_GAIN)),
                damping=float(raw.get("damping", DEFAULT_DAMPING)),
                max_delta_rad=float(raw.get("max_delta_rad", DEFAULT_MAX_DELTA_RAD)),
                color_mode=str(raw.get("color_mode", "rgb")),
                trials=tuple(
                    BumpTrial.from_json(entry, f"trials[{index}]")
                    for index, entry in enumerate(raw.get("trials", []))
                ),
                validation=raw.get("validation"),
                approved_by=str(raw.get("approved_by", "")),
                created_at=str(raw.get("created_at", "")),
            )
        except (KeyError, TypeError) as error:
            raise WristServoError(f"malformed wrist servo calibration: {error}") from None

    @classmethod
    def load(cls, path: str | Path) -> WristServoCalibration:
        source = Path(path)
        try:
            raw = json.loads(source.read_text())
        except FileNotFoundError as error:
            raise WristServoError(
                f"no wrist servo calibration at {source}; teach one with "
                "`robot-wrist-servo teach`"
            ) from error
        except json.JSONDecodeError as error:
            raise WristServoError(f"{source}: invalid JSON ({error})") from error
        if not isinstance(raw, Mapping):
            raise WristServoError(f"{source}: expected a calibration object")
        return cls.from_json(raw)


@dataclass
class WristServoDraft:
    """A calibration being taught: the goal is recorded first, then the bumps accumulate."""

    robot: RobotStamp
    camera_name: str
    joint_names: tuple[str, ...]
    servo_joints: tuple[str, ...]
    goal: tuple[float, ...]
    feature_names: tuple[str, ...] = FEATURE_NAMES
    point: str = "insert"
    tolerance: tuple[float, ...] = DEFAULT_TOLERANCE
    color_mode: str = "rgb"
    trials: list[BumpTrial] = field(default_factory=list)
    #: How far the goal image had drifted by the end of the teach, once the arm was back
    #: at the pose the goal was recorded at. The teaching command refuses a large one; it
    #: is kept in the file because "the scene held still" is the assumption every column
    #: rests on, and a reader deserves to see how well it held.
    return_error: tuple[float, ...] | None = None

    def record(
        self, joint: str, delta_rad: float, before: Sequence[float], after: Sequence[float]
    ) -> None:
        if joint not in self.servo_joints:
            raise WristServoError(f"{joint!r} is not one of the servo joints")
        self.trials.append(
            BumpTrial(
                joint=joint,
                delta_rad=float(delta_rad),
                before=tuple(float(value) for value in before),
                after=tuple(float(value) for value in after),
            )
        )

    def fitted(self, **overrides: Any) -> WristServoCalibration:
        """Fit the columns and freeze the draft. ``None`` overrides keep the defaults.

        Accepting ``None`` matters because the overrides come from optional command-line
        flags: an unpassed ``--gain`` should mean "the default", not "no gain", and
        filtering here keeps every caller from having to build the dict conditionally.
        """

        jacobian, report = fit_jacobian(self.trials, self.servo_joints, self.feature_names)
        validation: dict[str, Any] = {"columns": report}
        if self.return_error is not None:
            validation["return_error"] = [float(value) for value in self.return_error]
        return WristServoCalibration(
            robot=self.robot,
            camera_name=self.camera_name,
            joint_names=self.joint_names,
            servo_joints=self.servo_joints,
            goal=self.goal,
            jacobian=jacobian,
            feature_names=self.feature_names,
            point=self.point,
            tolerance=self.tolerance,
            color_mode=self.color_mode,
            trials=tuple(self.trials),
            validation=validation,
            created_at=now_stamp(),
            **{name: value for name, value in overrides.items() if value is not None},
        )
