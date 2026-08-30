"""Fixed-camera pixel to joint-space calibration.

Answers one question: given the object's centroid in a bolted-down camera's pixels, what
joint vector puts the gripper at its pre-grasp pose? It answers it by interpolating
between poses an operator actually teleoperated to, not by modelling the camera, the
table plane, or the arm's kinematics.

Why an interpolator and not a homography plus IK. The pixel of an object lying on the
table and the pixel of the gripper hovering 50 mm above it are different pixels whenever
the camera is oblique. Pairing the *object's* pixel with the joint vector that grasped
that object folds the parallax, the lens distortion, the table plane and the arm's
kinematics into a single fit. None of them is modelled, so none of them can be modelled
wrong. The cost is that the fit describes exactly one camera pose on exactly one arm
calibration: move the camera or re-zero the arm and every sample in the file is void,
which is why :class:`RobotStamp` is recorded and checked.

Two limits are structural, not conservatism:

* **Never extrapolate.** Both fits behave arbitrarily outside the sampled region. A
  prediction is refused unless the pixel is inside the convex hull of the calibration
  points, because a plausible-looking wrong answer is worse than a refusal.
* **Approach from one direction.** Feetech servos have backlash, so the pose reached by a
  commanded joint vector depends on which way the joint last moved. ``approach_pose()``
  offsets the target so the final motion of every joint has the same sign it had during
  capture. The offsets are worth several millimetres and are stored with the fit.

Wrist roll is not a function of position, so it is kept out of the position fit and
modelled separately by :class:`RollModel`.

This module is deliberately free of OpenCV and of hardware: it is the arithmetic and the
file format. :mod:`everest_robot.calibrate_pixel_map` is the operator's CLI around it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1

# A thin-plate spline with a linear tail needs at least three non-collinear points to
# solve at all. Eight is the smallest number that leaves anything to hold out.
MIN_SAMPLES = 8

# The procedure's q = [q1, q2, q3, q4]. Wrist roll is fitted separately and the gripper is
# not a function of where the object is.
DEFAULT_FIT_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")
DEFAULT_ROLL_JOINT = "wrist_roll"
DEFAULT_BASE_JOINT = "shoulder_pan"

THIN_PLATE_SPLINE = "thin_plate_spline"
QUADRATIC = "quadratic"


class PixelMapError(RuntimeError):
    """The calibration file, or a request against it, is not usable."""


class OutsideCalibratedRegion(PixelMapError):
    """The detection lies outside the region the arm was actually taught."""


# ── samples ────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class Sample:
    """One taught correspondence: where the object was, and the pose that grasped it."""

    pixel: tuple[float, float]
    joints: tuple[float, ...]
    spine_rad: float | None = None
    note: str = ""
    captured_at: str = ""

    def to_json(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "pixel": [float(self.pixel[0]), float(self.pixel[1])],
            "joints": [float(joint) for joint in self.joints],
        }
        if self.spine_rad is not None:
            value["spine_deg"] = math.degrees(self.spine_rad)
        if self.note:
            value["note"] = self.note
        if self.captured_at:
            value["captured_at"] = self.captured_at
        return value

    @classmethod
    def from_json(cls, raw: Mapping[str, Any], where: str) -> Sample:
        try:
            pixel = (float(raw["pixel"][0]), float(raw["pixel"][1]))
            joints = tuple(float(joint) for joint in raw["joints"])
        except (KeyError, TypeError, ValueError, IndexError) as error:
            raise PixelMapError(f"{where}: needs 'pixel' [u, v] and 'joints' [...]") from error
        spine = raw.get("spine_deg")
        return cls(
            pixel=pixel,
            joints=joints,
            spine_rad=None if spine is None else math.radians(float(spine)),
            note=str(raw.get("note", "")),
            captured_at=str(raw.get("captured_at", "")),
        )


def now_stamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def sample_pixels(samples: Sequence[Sample]) -> np.ndarray:
    return np.asarray([sample.pixel for sample in samples], dtype=float).reshape(-1, 2)


def sample_joints(samples: Sequence[Sample]) -> np.ndarray:
    return np.asarray([sample.joints for sample in samples], dtype=float).reshape(len(samples), -1)


# ── the fit ────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class Normalizer:
    """Centres and scales pixels before the fit.

    Thin-plate splines are not scale invariant: with raw pixel coordinates the kernel
    values run to tens of thousands while the joint values are radians, and ``smoothing``
    stops meaning anything an operator can reason about. Normalizing makes the smoothing
    and ridge terms dimensionless, so their defaults transfer between cameras.
    """

    center: tuple[float, float]
    scale: float

    @classmethod
    def of(cls, pixels: np.ndarray) -> Normalizer:
        center = pixels.mean(axis=0)
        spread = float(np.sqrt(((pixels - center) ** 2).sum(axis=1).mean()))
        return cls((float(center[0]), float(center[1])), spread if spread > 1e-9 else 1.0)

    def apply(self, pixels: np.ndarray) -> np.ndarray:
        points = np.asarray(pixels, dtype=float).reshape(-1, 2)
        return (points - np.asarray(self.center)) / self.scale


def _thin_plate_kernel(radii: np.ndarray) -> np.ndarray:
    """r^2 log r, with the removable singularity at r = 0 filled in."""

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(radii > 0.0, radii**2 * np.log(np.where(radii > 0.0, radii, 1.0)), 0.0)


def _distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)


@dataclass(frozen=True, slots=True)
class JointModel:
    """A fitted pixel -> joint-vector map, with everything needed to evaluate it."""

    kind: str
    joints: tuple[str, ...]
    normalizer: Normalizer
    weights: np.ndarray
    tail: np.ndarray | None
    nodes: np.ndarray | None
    smoothing: float
    ridge: float

    def predict(self, pixels: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        points = self.normalizer.apply(np.asarray(pixels, dtype=float))
        if self.kind == THIN_PLATE_SPLINE:
            if self.nodes is None or self.tail is None:
                raise PixelMapError("thin-plate-spline model is missing its nodes or linear tail")
            basis = _thin_plate_kernel(_distances(points, self.nodes))
            linear = np.hstack([np.ones((len(points), 1)), points])
            return basis @ self.weights + linear @ self.tail
        return _quadratic_features(points) @ self.weights


def _quadratic_features(points: np.ndarray) -> np.ndarray:
    u = points[:, 0]
    v = points[:, 1]
    return np.column_stack([np.ones_like(u), u, v, u * u, u * v, v * v])


def fit_thin_plate_spline(
    samples: Sequence[Sample],
    joints: Sequence[str],
    *,
    smoothing: float = 0.0,
) -> JointModel:
    """Interpolate the taught poses exactly (or nearly, with ``smoothing`` > 0).

    ``smoothing`` is in normalized-pixel units, so 0.0 reproduces every sample and small
    positive values trade sample fidelity for a flatter surface between them.
    """

    pixels = sample_pixels(samples)
    values = sample_joints(samples)
    count = len(samples)
    if count < 3:
        raise PixelMapError(f"a thin-plate spline needs at least 3 samples, got {count}")

    normalizer = Normalizer.of(pixels)
    points = normalizer.apply(pixels)
    basis = _thin_plate_kernel(_distances(points, points))
    linear = np.hstack([np.ones((count, 1)), points])

    system = np.zeros((count + 3, count + 3), dtype=float)
    system[:count, :count] = basis + smoothing * np.eye(count)
    system[:count, count:] = linear
    system[count:, :count] = linear.T
    right = np.zeros((count + 3, values.shape[1]), dtype=float)
    right[:count] = values

    try:
        solution = np.linalg.solve(system, right)
    except np.linalg.LinAlgError as error:
        raise PixelMapError(
            "the thin-plate-spline system is singular: samples are collinear or duplicated. "
            "Spread the grid out, drop the duplicate pixels, or raise --smoothing."
        ) from error

    return JointModel(
        kind=THIN_PLATE_SPLINE,
        joints=tuple(joints),
        normalizer=normalizer,
        weights=solution[:count],
        tail=solution[count:],
        nodes=points,
        smoothing=float(smoothing),
        ridge=0.0,
    )


def fit_quadratic(
    samples: Sequence[Sample],
    joints: Sequence[str],
    *,
    ridge: float = 1e-3,
) -> JointModel:
    """Fit each joint against a ridge-regularized quadratic in (u, v).

    Less accurate at the samples than the spline and better behaved just outside them.
    The intercept is left unpenalized so the ridge term cannot pull the whole surface
    toward zero radians, which is not a pose.
    """

    pixels = sample_pixels(samples)
    values = sample_joints(samples)
    if len(samples) < 6:
        raise PixelMapError(f"a quadratic in (u, v) needs at least 6 samples, got {len(samples)}")

    normalizer = Normalizer.of(pixels)
    features = _quadratic_features(normalizer.apply(pixels))
    penalty = np.diag([0.0, 1.0, 1.0, 1.0, 1.0, 1.0]) * ridge
    try:
        weights = np.linalg.solve(features.T @ features + penalty, features.T @ values)
    except np.linalg.LinAlgError as error:
        raise PixelMapError(
            "the quadratic normal equations are singular; raise --ridge or spread the samples out"
        ) from error

    return JointModel(
        kind=QUADRATIC,
        joints=tuple(joints),
        normalizer=normalizer,
        weights=weights,
        tail=None,
        nodes=None,
        smoothing=0.0,
        ridge=float(ridge),
    )


def fit_model(
    samples: Sequence[Sample],
    joints: Sequence[str],
    *,
    kind: str = THIN_PLATE_SPLINE,
    smoothing: float = 0.0,
    ridge: float = 1e-3,
) -> JointModel:
    if kind == THIN_PLATE_SPLINE:
        return fit_thin_plate_spline(samples, joints, smoothing=smoothing)
    if kind == QUADRATIC:
        return fit_quadratic(samples, joints, ridge=ridge)
    raise PixelMapError(f"unknown model {kind!r} (expected {THIN_PLATE_SPLINE} or {QUADRATIC})")


# ── where the fit is allowed to be believed ────────────────────────────────────────
def convex_hull(pixels: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    """Counter-clockwise convex hull of the calibration pixels (monotone chain)."""

    points = np.unique(np.asarray(pixels, dtype=float).reshape(-1, 2), axis=0)
    order = np.lexsort((points[:, 1], points[:, 0]))
    points = points[order]
    if len(points) < 3:
        return points

    def half(sequence: np.ndarray) -> list[np.ndarray]:
        chain: list[np.ndarray] = []
        for point in sequence:
            while len(chain) >= 2 and _cross(chain[-2], chain[-1], point) <= 0.0:
                chain.pop()
            chain.append(point)
        return chain

    lower = half(points)
    upper = half(points[::-1])
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def _cross(origin: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float(
        (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])
    )


def hull_margin_px(pixel: Sequence[float], hull: np.ndarray) -> float:
    """Distance from ``pixel`` to the nearest hull edge: positive inside, negative outside.

    Exact for a convex polygon, which is the only shape this is ever called with.
    """

    if len(hull) < 3:
        raise PixelMapError("the calibration hull is degenerate; collect samples that span an area")
    point = np.asarray(pixel, dtype=float)
    margins = []
    for index, start in enumerate(hull):
        end = hull[(index + 1) % len(hull)]
        edge = end - start
        length = float(np.linalg.norm(edge))
        if length < 1e-9:
            continue
        # Hull vertices are counter-clockwise, so an interior point is to the left of
        # every directed edge and this signed distance is positive.
        offset = point - start
        margins.append(float(edge[0] * offset[1] - edge[1] * offset[0]) / length)
    return min(margins)


# ── wrist roll ─────────────────────────────────────────────────────────────────────
def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _circular_mean(angles: Sequence[float]) -> float:
    values = np.asarray(angles, dtype=float)
    return math.atan2(float(np.sin(values).mean()), float(np.cos(values).mean()))


@dataclass(frozen=True, slots=True)
class RollModel:
    """``q_roll = sign * theta_spine - q_base + offset``.

    Roll is not a function of table position, so it is fitted from the object's measured
    spine angle rather than from its pixel. The gripper's roll axis rotates with the base
    joint, which is why ``q_base`` is subtracted. ``sign`` exists because image
    coordinates put +v downward: the resulting angle is left-handed relative to the arm's
    frame, and which of the two conventions applies depends on the camera's mounting
    rather than on anything worth asking an operator to reason about. It is chosen as
    whichever fits the samples, and ``residual_std_deg`` says whether the model held.
    """

    joint: str
    base_joint: str
    sign: int
    offset_rad: float
    residual_std_deg: float
    samples_used: int

    def predict(self, spine_rad: float, base_rad: float) -> float:
        return _wrap(self.sign * spine_rad - base_rad + self.offset_rad)


def fit_roll_model(
    samples: Sequence[Sample],
    joint_names: Sequence[str],
    *,
    roll_joint: str = DEFAULT_ROLL_JOINT,
    base_joint: str = DEFAULT_BASE_JOINT,
) -> RollModel | None:
    """Solve for the constant offset from every sample that recorded a spine angle.

    Returns ``None`` when nothing recorded one, which is the normal outcome for a
    detector that reports only a centroid.
    """

    for name in (roll_joint, base_joint):
        if name not in joint_names:
            raise PixelMapError(f"unknown joint {name!r}; this arm has {', '.join(joint_names)}")
    roll_index = list(joint_names).index(roll_joint)
    base_index = list(joint_names).index(base_joint)

    usable = [sample for sample in samples if sample.spine_rad is not None]
    if len(usable) < 3:
        return None

    best: RollModel | None = None
    for sign in (1, -1):
        residuals = [
            sample.joints[roll_index] + sample.joints[base_index] - sign * float(sample.spine_rad)
            for sample in usable
            if sample.spine_rad is not None
        ]
        offset = _circular_mean(residuals)
        spread = math.degrees(
            float(np.sqrt(np.mean([_wrap(value - offset) ** 2 for value in residuals])))
        )
        candidate = RollModel(
            joint=roll_joint,
            base_joint=base_joint,
            sign=sign,
            offset_rad=offset,
            residual_std_deg=spread,
            samples_used=len(usable),
        )
        if best is None or candidate.residual_std_deg < best.residual_std_deg:
            best = candidate
    return best


# ── the calibration file ───────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class CameraSource:
    """Which camera produced the pixels. Bolted down, and not moved after sample one."""

    index_or_path: str
    backend: str = "auto"
    width: int | None = None
    height: int | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "index_or_path": self.index_or_path,
            "backend": self.backend,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> CameraSource:
        return cls(
            index_or_path=str(raw["index_or_path"]),
            backend=str(raw.get("backend", "auto")),
            width=None if raw.get("width") is None else int(raw["width"]),
            height=None if raw.get("height") is None else int(raw["height"]),
        )


@dataclass(frozen=True, slots=True)
class DetectorSpec:
    """Which segmentation produced the centroid, and over which crop."""

    kind: str
    roi_xywh: tuple[int, int, int, int] | None = None

    def to_json(self) -> dict[str, Any]:
        return {"kind": self.kind, "roi_xywh": list(self.roi_xywh) if self.roi_xywh else None}

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> DetectorSpec:
        roi = raw.get("roi_xywh")
        return cls(
            kind=str(raw["kind"]),
            roi_xywh=None if roi is None else (int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])),
        )


@dataclass(frozen=True, slots=True)
class RobotStamp:
    """The arm and calibration the samples were taught on.

    A fit is only meaningful against the machine that produced it in the zeroing it was
    produced under, so this is checked before a prediction is allowed to drive anything.
    """

    robot_id: str
    calibration_id: str
    config_digest: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "calibration_id": self.calibration_id,
            "config_digest": self.config_digest,
        }

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> RobotStamp:
        return cls(
            robot_id=str(raw["robot_id"]),
            calibration_id=str(raw["calibration_id"]),
            config_digest=str(raw.get("config_digest", "")),
        )

    def verify(self, other: RobotStamp) -> None:
        if self.robot_id != other.robot_id or self.calibration_id != other.calibration_id:
            raise PixelMapError(
                f"this calibration was taught on {self.robot_id}/{self.calibration_id} but the "
                f"connected arm is {other.robot_id}/{other.calibration_id}. Recapture the samples."
            )


@dataclass(frozen=True, slots=True)
class Prediction:
    """What the map says, and how far inside its evidence the request was."""

    pixel: tuple[float, float]
    joints: dict[str, float]
    hull_margin_px: float
    roll_rad: float | None = None


@dataclass(frozen=True, slots=True)
class PixelJointMap:
    """The whole calibration: what was taught, what was fitted, and where it is valid."""

    camera: CameraSource
    detector: DetectorSpec
    robot: RobotStamp
    joint_names: tuple[str, ...]
    samples: tuple[Sample, ...]
    model: JointModel | None = None
    hull: np.ndarray | None = None
    required_margin_px: float = 0.0
    roll: RollModel | None = None
    approach_offset_rad: tuple[float, ...] = ()
    validation: dict[str, Any] | None = None
    created_at: str = ""

    # ── using it ───────────────────────────────────────────────────────────────────
    def predict(self, pixel: Sequence[float], spine_rad: float | None = None) -> Prediction:
        """The joint vector for one detection, or a refusal.

        Refuses rather than extrapolates. A detection outside the taught region means the
        object is somewhere nobody demonstrated, and the honest answer is to say so.
        """

        if self.model is None or self.hull is None:
            raise PixelMapError("this calibration has no fit yet; run `robot-pixel-map fit` first")
        margin = hull_margin_px(pixel, self.hull)
        if margin < self.required_margin_px:
            raise OutsideCalibratedRegion(
                f"pixel ({pixel[0]:.1f}, {pixel[1]:.1f}) is {-margin:.1f} px outside the "
                f"calibrated region (needs {self.required_margin_px:.1f} px of margin). "
                "Move the object inside the sampled area or teach that corner."
            )
        values = self.model.predict([list(pixel)])[0]
        joints = dict(zip(self.model.joints, (float(value) for value in values), strict=True))
        roll = None
        if self.roll is not None and spine_rad is not None:
            base = joints.get(self.roll.base_joint)
            if base is None:
                raise PixelMapError(
                    f"the roll model needs {self.roll.base_joint!r}, which this fit does not cover"
                )
            roll = self.roll.predict(spine_rad, base)
            joints[self.roll.joint] = roll
        return Prediction(
            pixel=(float(pixel[0]), float(pixel[1])),
            joints=joints,
            hull_margin_px=margin,
            roll_rad=roll,
        )

    def full_target(self, prediction: Prediction, measured: Sequence[float]) -> tuple[float, ...]:
        """The predicted joints over the measured pose: unfitted joints hold where they are.

        Same reasoning as ``robot-raise``: a joint nobody taught is a joint this map has
        no opinion about, and holding measured feedback beats inventing an absolute value
        for a frame whose zero is not shared between drivers.
        """

        if len(measured) != len(self.joint_names):
            raise PixelMapError(
                f"expected {len(self.joint_names)} measured joints, got {len(measured)}"
            )
        return tuple(
            prediction.joints.get(name, float(value))
            for name, value in zip(self.joint_names, measured, strict=True)
        )

    def pixel_error_px(self, prediction: Prediction, measured: Sequence[float]) -> float:
        """How far, in pixels, the measured pose is from the pose this pixel asks for.

        Nothing detects the *gripper*: the map pairs the object's pixel with the pose that
        grasps it, so there is no second image feature to difference against and no
        pixel-space error to measure directly. What there is, is the fit. Solving
        ``J dp = q_measured - q_target`` for the pixel displacement ``dp``, with ``J`` the
        fit's Jacobian at this pixel, expresses the joint-space servo error back in the
        map's own units: *the gripper is standing where a carabiner this many pixels away
        would have put it*. That is comparable to the hull margin and the jump gate, which
        radians are not.

        Least squares because ``J`` is (fitted joints) x 2: four joints move for two pixel
        degrees of freedom, so a joint error with a component off the taught surface has no
        exact pixel explanation and gets the closest one. This is diagnostics -- it is
        derived from the fit, so it inherits every one of the fit's errors and is not
        independent evidence about where the gripper physically is.
        """

        if self.model is None:
            raise PixelMapError("this calibration has no fit yet; run `robot-pixel-map fit` first")
        if len(measured) != len(self.joint_names):
            raise PixelMapError(
                f"expected {len(self.joint_names)} measured joints, got {len(measured)}"
            )
        position = {name: index for index, name in enumerate(self.joint_names)}
        target = np.asarray(
            [prediction.joints[name] for name in self.model.joints], dtype=float
        )
        current = np.asarray(
            [float(measured[position[name]]) for name in self.model.joints], dtype=float
        )
        # One pixel each way. The fit is smooth at that scale, and a central difference is
        # first-order exact without a second model to keep in step with this one.
        u, v = prediction.pixel
        probes = self.model.predict(
            [[u - 1.0, v], [u + 1.0, v], [u, v - 1.0], [u, v + 1.0]]
        )
        jacobian = np.column_stack(
            [(probes[1] - probes[0]) / 2.0, (probes[3] - probes[2]) / 2.0]
        )
        offset, *_ = np.linalg.lstsq(jacobian, current - target, rcond=None)
        return float(np.linalg.norm(offset))

    def approach_pose(self, target: Sequence[float]) -> tuple[float, ...]:
        """The pose to pass through so every joint arrives from the taught direction.

        Backlash makes the reached pose depend on the sign of the last motion. Commanding
        ``target - offset`` and then ``target`` makes that sign the same on every pick and
        during every capture, which is the single cheapest millimetre in this procedure.
        """

        if not self.approach_offset_rad:
            return tuple(float(value) for value in target)
        if len(self.approach_offset_rad) != len(target):
            raise PixelMapError(
                f"approach offsets cover {len(self.approach_offset_rad)} joints, "
                f"target has {len(target)}"
            )
        return tuple(
            float(value) - float(offset)
            for value, offset in zip(target, self.approach_offset_rad, strict=True)
        )

    # ── editing it ─────────────────────────────────────────────────────────────────
    def with_samples(self, samples: Sequence[Sample]) -> PixelJointMap:
        return PixelJointMap(
            camera=self.camera,
            detector=self.detector,
            robot=self.robot,
            joint_names=self.joint_names,
            samples=tuple(samples),
            model=self.model,
            hull=self.hull,
            required_margin_px=self.required_margin_px,
            roll=self.roll,
            approach_offset_rad=self.approach_offset_rad,
            validation=self.validation,
            created_at=self.created_at or now_stamp(),
        )

    def fitted(
        self,
        *,
        fit_joints: Sequence[str] = DEFAULT_FIT_JOINTS,
        kind: str = THIN_PLATE_SPLINE,
        smoothing: float = 0.0,
        ridge: float = 1e-3,
        roll_joint: str | None = DEFAULT_ROLL_JOINT,
        base_joint: str = DEFAULT_BASE_JOINT,
        holdout: int = 6,
    ) -> PixelJointMap:
        """Fit this map's samples and return the calibration with the fit stored in it."""

        if len(self.samples) < MIN_SAMPLES:
            raise PixelMapError(
                f"{len(self.samples)} samples is not enough to fit; collect at least {MIN_SAMPLES} "
                "(the procedure asks for about 30, a 6x5 grid plus the corners)"
            )
        indices = [self.joint_names.index(name) for name in _known(fit_joints, self.joint_names)]
        reduced = tuple(
            Sample(
                pixel=sample.pixel,
                joints=tuple(sample.joints[index] for index in indices),
                spine_rad=sample.spine_rad,
                note=sample.note,
                captured_at=sample.captured_at,
            )
            for sample in self.samples
        )
        model = fit_model(reduced, fit_joints, kind=kind, smoothing=smoothing, ridge=ridge)
        roll = (
            None
            if roll_joint is None
            else fit_roll_model(
                self.samples, self.joint_names, roll_joint=roll_joint, base_joint=base_joint
            )
        )
        return PixelJointMap(
            camera=self.camera,
            detector=self.detector,
            robot=self.robot,
            joint_names=self.joint_names,
            samples=self.samples,
            model=model,
            hull=convex_hull(sample_pixels(self.samples)),
            required_margin_px=self.required_margin_px,
            roll=roll,
            approach_offset_rad=self.approach_offset_rad,
            validation=holdout_report(
                reduced, fit_joints, kind=kind, smoothing=smoothing, ridge=ridge, holdout=holdout
            ),
            created_at=self.created_at or now_stamp(),
        )

    # ── the file ───────────────────────────────────────────────────────────────────
    def to_json(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "created_at": self.created_at or now_stamp(),
            "camera": self.camera.to_json(),
            "detector": self.detector.to_json(),
            "robot": self.robot.to_json(),
            "joint_names": list(self.joint_names),
            "required_margin_px": self.required_margin_px,
            "approach_offset_rad": list(self.approach_offset_rad),
            "samples": [sample.to_json() for sample in self.samples],
        }
        if self.model is not None:
            document["model"] = {
                "kind": self.model.kind,
                "joints": list(self.model.joints),
                "center_px": list(self.model.normalizer.center),
                "scale_px": self.model.normalizer.scale,
                "smoothing": self.model.smoothing,
                "ridge": self.model.ridge,
                "weights": self.model.weights.tolist(),
                "tail": None if self.model.tail is None else self.model.tail.tolist(),
                "nodes": None if self.model.nodes is None else self.model.nodes.tolist(),
            }
        if self.hull is not None:
            document["hull_px"] = self.hull.tolist()
        if self.roll is not None:
            document["roll"] = {
                "joint": self.roll.joint,
                "base_joint": self.roll.base_joint,
                "sign": self.roll.sign,
                "offset_deg": math.degrees(self.roll.offset_rad),
                "residual_std_deg": self.roll.residual_std_deg,
                "samples_used": self.roll.samples_used,
            }
        if self.validation is not None:
            document["validation"] = self.validation
        return document

    def save(self, path: str | Path) -> Path:
        """Write the calibration, leaving the previous file intact until the write lands."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_json(), indent=2) + "\n")
        temporary.replace(target)
        return target

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> PixelJointMap:
        version = int(raw.get("schema_version", 0))
        if version != SCHEMA_VERSION:
            raise PixelMapError(
                f"calibration schema_version {version} is not {SCHEMA_VERSION}; recapture it"
            )
        samples = tuple(
            Sample.from_json(entry, f"samples[{index}]")
            for index, entry in enumerate(raw.get("samples", []))
        )
        model = None
        if (spec := raw.get("model")) is not None:
            nodes = spec.get("nodes")
            tail = spec.get("tail")
            model = JointModel(
                kind=str(spec["kind"]),
                joints=tuple(str(name) for name in spec["joints"]),
                normalizer=Normalizer(
                    (float(spec["center_px"][0]), float(spec["center_px"][1])),
                    float(spec["scale_px"]),
                ),
                weights=np.asarray(spec["weights"], dtype=float),
                tail=None if tail is None else np.asarray(tail, dtype=float),
                nodes=None if nodes is None else np.asarray(nodes, dtype=float),
                smoothing=float(spec.get("smoothing", 0.0)),
                ridge=float(spec.get("ridge", 0.0)),
            )
        roll = None
        if (spec := raw.get("roll")) is not None:
            roll = RollModel(
                joint=str(spec["joint"]),
                base_joint=str(spec["base_joint"]),
                sign=int(spec["sign"]),
                offset_rad=math.radians(float(spec["offset_deg"])),
                residual_std_deg=float(spec.get("residual_std_deg", float("nan"))),
                samples_used=int(spec.get("samples_used", 0)),
            )
        hull = raw.get("hull_px")
        return cls(
            camera=CameraSource.from_json(raw["camera"]),
            detector=DetectorSpec.from_json(raw["detector"]),
            robot=RobotStamp.from_json(raw["robot"]),
            joint_names=tuple(str(name) for name in raw["joint_names"]),
            samples=samples,
            model=model,
            hull=None if hull is None else np.asarray(hull, dtype=float),
            required_margin_px=float(raw.get("required_margin_px", 0.0)),
            roll=roll,
            approach_offset_rad=tuple(
                float(value) for value in raw.get("approach_offset_rad", [])
            ),
            validation=raw.get("validation"),
            created_at=str(raw.get("created_at", "")),
        )

    @classmethod
    def load(cls, path: str | Path) -> PixelJointMap:
        source = Path(path)
        try:
            raw = json.loads(source.read_text())
        except FileNotFoundError as error:
            raise PixelMapError(f"no calibration at {source}") from error
        except json.JSONDecodeError as error:
            raise PixelMapError(f"{source}: invalid JSON ({error})") from error
        if not isinstance(raw, Mapping):
            raise PixelMapError(f"{source}: expected a calibration object")
        return cls.from_json(raw)


def _known(names: Sequence[str], joint_names: Sequence[str]) -> tuple[str, ...]:
    unknown = [name for name in names if name not in joint_names]
    if unknown:
        raise PixelMapError(
            f"unknown joint(s) {', '.join(unknown)}; this arm has {', '.join(joint_names)}"
        )
    if not names:
        raise PixelMapError("no joints to fit")
    return tuple(names)


# ── validation ─────────────────────────────────────────────────────────────────────
def holdout_report(
    samples: Sequence[Sample],
    joints: Sequence[str],
    *,
    kind: str = THIN_PLATE_SPLINE,
    smoothing: float = 0.0,
    ridge: float = 1e-3,
    holdout: int = 6,
) -> dict[str, Any]:
    """Refit without an evenly spaced holdout set and report the joint-space error.

    Joint-space degrees are not millimetres. This says whether the fit generalizes, not
    whether the gripper lands on the carabiner -- only a ruler answers that, which is why
    the procedure asks for one. Treat a bad number here as disqualifying and a good one as
    permission to go measure.
    """

    count = len(samples)
    if holdout <= 0 or count - holdout < 3:
        return {"held_out": 0, "note": "too few samples to hold any out"}

    step = count / holdout
    held = sorted({min(count - 1, int(index * step + step / 2)) for index in range(holdout)})
    train = [sample for index, sample in enumerate(samples) if index not in set(held)]
    model = fit_model(train, joints, kind=kind, smoothing=smoothing, ridge=ridge)

    test = [samples[index] for index in held]
    predicted = model.predict(sample_pixels(test))
    errors = np.degrees(predicted - sample_joints(test))
    worst = int(np.argmax(np.abs(errors).max(axis=1)))
    return {
        "held_out": len(held),
        "trained_on": len(train),
        "rms_error_deg": {
            name: float(np.sqrt((errors[:, index] ** 2).mean()))
            for index, name in enumerate(joints)
        },
        "max_abs_error_deg": {
            name: float(np.abs(errors[:, index]).max()) for index, name in enumerate(joints)
        },
        "worst_sample_pixel": list(test[worst].pixel),
    }
