"""The CV subsystem that the attachment FSM's ``SEARCH_CV`` state is made of.

One tick is: read the fixed camera, detect the carabiner, put its centroid through the
calibrated pixel map, and hand the resulting joint vector to exactly one
:meth:`~everest_robot.robot.visual_tracking.VisualTracker.tick`. Nothing here decides which
state comes next. It reports whether the target is still visible and whether the approach
has finished; :mod:`everest_robot.attachment_fsm` does the arbitrating.

Why the map and not a servo on pixel error: :mod:`everest_robot.pixel_map` is taught on
*pre-grasp* poses, so "directly above the carabiner" is what every sample already encodes.
Tracking to the map's prediction tracks to directly-above by construction, with no model of
the table plane, the camera's obliquity or the arm's kinematics to get wrong.

Three decisions live here rather than in the FSM or in the handler, because nothing else
is in a position to answer them:

* **When the target is lost.** A single dropped frame is the normal behaviour of a
  threshold segmentation, not a lost carabiner. Only ``lost_after_misses`` consecutive
  frames without a detection report ``target_visible=False``, so one flicker cannot bounce
  the FSM back into a learned search.
* **What "followed" means.** The approach is finished when the arm has *arrived*: the
  measured pose within ``settle_tolerance_rad`` of the map's target for ``settle_ticks``
  consecutive ticks. A tick that reported motion means the arm is on its way, which is not
  the same thing, and handing a half-finished approach to the clip policy is exactly the
  failure this state exists to prevent.
* **Pacing.** The tracker's speed lock is a per-tick clamp of
  ``max_velocity_rad_s / rate_hz``, so it only means radians per second if ticks happen at
  ``rate_hz``. The FSM calls one step at a time with its own work in between, so the
  follower waits out the remainder of the period itself rather than trusting its caller.

A detection the map refuses -- outside the taught convex hull, or too far from the previous
frame to be the same blob -- holds the arm but is still *visible*. Handing those back to a
learned search would be pointless: the camera is bolted down, so no arm motion changes
where the carabiner falls in the frame, and the search policy would only hand back the
detection this follower already has. Holding still until the state budget runs out is the
outcome an operator can act on.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from everest_robot.pixel_map import (
    CameraSource,
    OutsideCalibratedRegion,
    PixelJointMap,
    PixelMapError,
    Prediction,
)
from everest_robot.robot.clock import Clock, SystemClock
from everest_robot.robot.visual_tracking import VisualTracker

# A detection that jumps further than this between frames is a different blob, not the same
# carabiner moving. Treated as a hold so the arm does not lunge at it.
DEFAULT_MAX_JUMP_PX = 150.0

# Why a target was not passed to the tracker. Strings rather than an enum for the same
# reason as in ``visual_tracking``: they are shown to an operator, not branched on.
NO_DETECTION = "no detection"
OUTSIDE_REGION = "outside the calibrated region"
NO_FEEDBACK = "no joint feedback"
TRACKING = "tracking"


# ── camera ─────────────────────────────────────────────────────────────────────────
def load_cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as error:
        raise PixelMapError(
            "OpenCV is required for the camera. Install with: uv add opencv-python"
        ) from error
    return cv2


def _capture_api(cv2: Any, backend: str) -> int:
    if backend == "auto":
        return cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    apis = {
        "avfoundation": cv2.CAP_AVFOUNDATION,
        "v4l2": cv2.CAP_V4L2,
        "any": cv2.CAP_ANY,
    }
    if backend not in apis:
        raise PixelMapError(f"unknown camera backend {backend!r} (expected {', '.join(apis)})")
    return apis[backend]


def open_capture(camera: CameraSource) -> Any:
    """Open the fixed camera, or say plainly which id failed."""

    cv2 = load_cv2()
    try:
        index_or_path: int | str = int(camera.index_or_path)
    except ValueError:
        index_or_path = camera.index_or_path
    capture = cv2.VideoCapture(index_or_path, _capture_api(cv2, camera.backend))
    if not capture.isOpened():
        raise PixelMapError(
            f"could not open camera {camera.index_or_path!r} "
            f"(backend {camera.backend}); check the id and that nothing else holds it"
        )
    if camera.width:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera.width)
    if camera.height:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera.height)
    return capture


def read_frame(capture: Any, camera: CameraSource, *, attempts: int = 20) -> Any:
    """One frame, retrying briefly before giving up.

    A single failed ``read()`` means very little. AVFoundation returns false for the first
    few reads while the capture session spins up, and a momentary miss mid-session should
    not end a 25-minute teaching run. Only a sustained failure is real -- and on macOS its
    usual cause is another process already holding the device, because ``VideoCapture``
    still reports ``isOpened()`` in that case and simply never yields a frame.
    """

    import time

    for attempt in range(attempts):
        ok, frame = capture.read()
        if ok:
            return frame
        if attempt + 1 < attempts:
            time.sleep(0.05)
    raise PixelMapError(
        f"camera {camera.index_or_path!r} opened but produced no frame in {attempts} reads. "
        "Another process is probably holding it -- close any running "
        "robot-two-white-black-live, tools/preview.py or robot-pixel-map window. "
        "If nothing else is running, check the camera permission for your terminal."
    )


# ── detection ──────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class CarabinerPixel:
    """One detection, reduced to what the map consumes."""

    centroid: tuple[float, float]
    spine_rad: float
    white_a: tuple[float, float]
    white_b: tuple[float, float]
    gate: tuple[float, float]


def detect_carabiner(frame: Any, roi_xywh: Sequence[int]) -> CarabinerPixel:
    """The two-white-tape plus black-gate detection, reduced to a centroid and an angle.

    The centroid is the midpoint of the two white tapes -- the *object's* pixel, never the
    gripper's. Pairing that pixel with the joint vector that grasped it is what absorbs
    the camera's parallax into the fit instead of leaving it to be modelled.

    The spine angle is measured in image coordinates, where +v runs downward, so it is
    left-handed with respect to the arm. ``RollModel`` picks up the sign; nothing here
    needs to know how the camera is mounted.
    """

    from everest_robot.vision import detect_two_white_tapes_and_black_gate_bgr

    detection = detect_two_white_tapes_and_black_gate_bgr(
        frame, roi_xywh=(int(roi_xywh[0]), int(roi_xywh[1]), int(roi_xywh[2]), int(roi_xywh[3]))
    )
    a, b = detection.white_points_px
    gate = detection.black_gate_px
    midpoint = ((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)

    axis = (b.x - a.x, b.y - a.y)
    length = math.hypot(*axis)
    if length < 1e-6:
        raise PixelMapError("the two white tapes landed on the same pixel")
    axis = (axis[0] / length, axis[1] / length)
    # Two identical tapes leave the axis sign ambiguous; the gate resolves it, exactly as
    # `pickup.carabiner_pose_from_axis_points_and_side_point` does in the robot frame.
    left = (-axis[1], axis[0])
    if left[0] * (gate.x - midpoint[0]) + left[1] * (gate.y - midpoint[1]) < 0.0:
        axis = (-axis[0], -axis[1])

    return CarabinerPixel(
        centroid=midpoint,
        spine_rad=math.atan2(axis[1], axis[0]),
        white_a=(a.x, a.y),
        white_b=(b.x, b.y),
        gate=(gate.x, gate.y),
    )


# ── the follower ───────────────────────────────────────────────────────────────────
class CarabinerDetection(Protocol):
    """What the follower needs from a detection: where the object is, and how it lies."""

    @property
    def centroid(self) -> tuple[float, float]: ...

    @property
    def spine_rad(self) -> float: ...


@dataclass(frozen=True, slots=True)
class FollowTick:
    """What one CV tick saw, what it did about it, and what it concluded."""

    index: int
    target_visible: bool
    followed: bool
    reason: str
    moved: bool
    pixel: tuple[float, float] | None = None
    pixel_error_px: float | None = None
    approach_error_rad: float | None = None
    hull_margin_px: float | None = None
    misses: int = 0
    settled_ticks: int = 0


@dataclass
class CarabinerFollower:
    """Detector, calibrated map and speed-locked tracker, stepped one action at a time."""

    calibration: PixelJointMap
    tracker: VisualTracker
    frames: Callable[[], Any]
    detect: Callable[[Any], CarabinerDetection]
    clock: Clock = field(default_factory=SystemClock)
    max_jump_px: float = DEFAULT_MAX_JUMP_PX
    lost_after_misses: int = 5
    settle_tolerance_rad: float = 0.05
    settle_ticks: int = 3
    track_roll: bool = True

    def __post_init__(self) -> None:
        if self.calibration.model is None:
            raise PixelMapError(
                "this calibration has no fit yet; run `robot-pixel-map fit` first"
            )
        if self.calibration.joint_names != tuple(self.tracker.port.joint_names):
            raise PixelMapError(
                f"calibration joints {', '.join(self.calibration.joint_names)} do not match "
                f"this arm's {', '.join(self.tracker.port.joint_names)}"
            )
        if self.lost_after_misses < 1:
            raise ValueError("lost_after_misses must be at least 1")
        if self.settle_ticks < 1:
            raise ValueError("settle_ticks must be at least 1")
        if not math.isfinite(self.settle_tolerance_rad) or self.settle_tolerance_rad <= 0.0:
            raise ValueError("settle_tolerance_rad must be finite and positive")
        self._index = 0
        self._misses = 0
        self._settled = 0
        self._previous: tuple[float, float] | None = None
        self._started_at: float | None = None

    def start(self) -> None:
        """Seed the tracker from measured feedback and energize.

        Re-callable, and the caller is expected to re-call it on every entry into
        ``SEARCH_CV``: a policy has moved the arm since the last one, so the tracker's
        seeded command and its lock-on count are both stale.
        """

        self._misses = 0
        self._settled = 0
        self._previous = None
        self._started_at = None
        self.tracker.start()

    def stop(self, *, hold: bool = True) -> None:
        """Stop following. Safe from any state, including twice."""

        self.tracker.stop(hold=hold)

    def step(self) -> FollowTick:
        """One detector/map/servo tick, paced to the tracker's rate.

        Raises :class:`~everest_robot.robot.visual_tracking.TrackerStopped` only for a
        genuine stop -- an arm fault, lost feedback, a refused command. An ordinary missed
        detection is the case this loop exists to survive and is reported in the tick.
        """

        self._pace()
        self._index += 1

        target, detection, prediction, reason = self._target_from_frame(self.frames())
        tick = self.tracker.tick(target, reason)

        if detection is None:
            self._misses += 1
        else:
            self._misses = 0

        approach_error: float | None = None
        if tick.target is not None and tick.measured:
            approach_error = max(
                abs(measured - wanted)
                for measured, wanted in zip(tick.measured, tick.target, strict=True)
            )
        if approach_error is not None and approach_error <= self.settle_tolerance_rad:
            self._settled += 1
        else:
            self._settled = 0

        return FollowTick(
            index=self._index,
            target_visible=self._misses < self.lost_after_misses,
            followed=self._settled >= self.settle_ticks,
            reason=tick.reason,
            moved=tick.moved,
            pixel=None if detection is None else self._previous,
            pixel_error_px=self._pixel_error(prediction, tick.measured),
            approach_error_rad=approach_error,
            hull_margin_px=None if prediction is None else prediction.hull_margin_px,
            misses=self._misses,
            settled_ticks=self._settled,
        )

    # ── internals ──────────────────────────────────────────────────────────────────
    def _pace(self) -> None:
        """Wait out the remainder of the tracker's period since the previous step began.

        The speed lock is a per-tick clamp, so ticks arriving faster than ``rate_hz`` would
        move the arm faster than ``max_velocity_rad_s`` without any of the numbers changing.
        """

        if self._started_at is not None:
            self.clock.sleep(self.tracker.period_s - (self.clock.monotonic() - self._started_at))
        self._started_at = self.clock.monotonic()

    def _target_from_frame(
        self, frame: Any
    ) -> tuple[tuple[float, ...] | None, CarabinerDetection | None, Prediction | None, str]:
        """One frame to a full joint target, or to a reason for holding still."""

        try:
            detection = self.detect(frame)
        except (RuntimeError, ValueError) as error:
            self._previous = None
            return None, None, None, f"{NO_DETECTION} ({str(error).split('.')[0][:50]})"

        centroid = (float(detection.centroid[0]), float(detection.centroid[1]))
        jump = None if self._previous is None else math.dist(self._previous, centroid)
        self._previous = centroid
        if jump is not None and jump > self.max_jump_px:
            return None, detection, None, f"detection jumped {jump:.0f} px"

        try:
            prediction = self.calibration.predict(
                centroid, detection.spine_rad if self.track_roll else None
            )
        except OutsideCalibratedRegion:
            return None, detection, None, OUTSIDE_REGION

        state = self.tracker.port.read_state()
        if not state.all_finite:
            return None, detection, prediction, NO_FEEDBACK
        return (
            self.calibration.full_target(prediction, state.positions),
            detection,
            prediction,
            TRACKING,
        )

    def _pixel_error(
        self, prediction: Prediction | None, measured: Sequence[float]
    ) -> float | None:
        if prediction is None or not measured:
            return None
        if not all(math.isfinite(value) for value in measured):
            return None
        return self.calibration.pixel_error_px(prediction, measured)
