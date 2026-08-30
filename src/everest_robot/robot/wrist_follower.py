"""The wrist-camera alternative to :mod:`everest_robot.robot.carabiner_follower`.

Same shape, same three judgements, same :class:`~everest_robot.robot.carabiner_follower.FollowTick`
-- so the FSM's ``SEARCH_CV`` handler cannot tell them apart -- but a different instrument
and therefore a different way of turning a frame into a joint target.

The fixed-camera follower asks a taught map "which pose grasps at this pixel?" and servos
to the answer. That question has no meaning for a wrist camera: it moves with the arm, so
a pixel names a direction relative to the gripper rather than a place on the bench, and the
pose that would put the gripper there is the thing being solved for, not a thing looked up.
So this follower closes the loop in the image instead. Each tick it measures the
carabiner's feature vector, differences it against the goal image taught by
``robot-wrist-servo teach``, and asks :meth:`WristServoCalibration.solve` for the joint step
that would remove the difference. Image-based visual servoing, with a constant Jacobian.

What that buys is the reason to have it at all: **nothing has to be calibrated against the
bench.** The fixed camera has to stay bolted where it was taught and the carabiner has to
land inside the convex hull of the poses somebody demonstrated, or the map refuses. Here the
evidence is a derivative measured on the arm, so the carabiner can be anywhere the wrist
camera can see it, including somewhere no sample was ever taken. What it costs is that
"visible" now depends on where the arm is pointing, which is exactly why losing the target
here *does* hand back to the learned search -- see below.

The three judgements this file owns, matching the fixed-camera follower's, and where they
differ:

* **When the target is lost.** Unchanged: only ``lost_after_misses`` consecutive frames
  without a detection report ``target_visible=False``. One dropped frame is the normal
  behaviour of a threshold segmentation.

  Handing a genuine loss back to ``SEARCH_RL`` means something here that it does not mean
  for the fixed camera. That camera is bolted down, so no arm motion changes what it sees
  and a learned search could only hand back the detection the follower already had. The
  wrist camera is the opposite: it lost the carabiner *because of where the arm is*, and
  moving the arm is precisely the remedy.

* **What "followed" means.** The measured image is inside tolerance on every feature for
  ``settle_ticks`` consecutive ticks. This is a stronger statement than the fixed camera's
  version rather than a weaker one: there, arrival is the arm reaching a pose the map
  predicted, and whether that pose is actually above the carabiner is the fit's problem.
  Here the servo error *is* an image measurement, so arrival is directly observed.

* **Pacing.** Unchanged, and for the same reason: the tracker's speed lock is a per-tick
  clamp of ``max_velocity_rad_s / rate_hz``, which only means radians per second if ticks
  happen at ``rate_hz``. The FSM steps one action at a time with its own work in between,
  so the follower waits out the remainder of the period itself.

A solve the calibration refuses -- :class:`~everest_robot.robot.wrist_servo.UnsupportedSolve`,
raised when the step asked for is past what a Jacobian measured at one pose can support --
holds the arm and is still *visible*, exactly as an out-of-hull detection is for the fixed
camera. The carabiner is there and the servo simply has no trustworthy answer about it;
lunging is the one response that is certainly wrong.

Frames come from the session's :class:`~everest_robot.robot.cameras.CameraRuntime`, never
from a second :class:`cv2.VideoCapture`. The wrist camera is a policy observation and is
already open while ``SEARCH_RL`` and ``CLIP_RL`` run; a second open on the same device
either fails or starves the first.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from everest_robot.robot.carabiner_follower import (
    DEFAULT_MAX_JUMP_PX,
    NO_DETECTION,
    NO_FEEDBACK,
    TRACKING,
    FollowTick,
)
from everest_robot.robot.clock import Clock, SystemClock
from everest_robot.robot.visual_tracking import VisualTracker
from everest_robot.robot.wrist_servo import (
    UnsupportedSolve,
    WristServoCalibration,
    WristServoError,
)

#: Why a tick did not pass a target to the tracker. Strings, not an enum, for the same
#: reason as everywhere else in this loop: an operator reads them, nothing branches on them.
AT_GOAL = "at the goal image"
UNSUPPORTED = "servo step outside the taught range"


def wrist_frames(cameras: Any, camera_name: str, color_mode: str) -> Callable[[], Any]:
    """A source of BGR frames from one named camera of an open ``CameraRuntime``.

    LeRobot cameras hand back RGB. ``carabiner_detect`` works in Lab derived from BGR, and
    a* / b* is the exact axis its teal score is measured on, so a swapped conversion does
    not degrade the detector, it inverts it. The conversion is a copy rather than a
    reversed view because OpenCV rejects negative strides.
    """

    import numpy as np

    def read() -> Any:
        frame = cameras.observation()[camera_name]
        if color_mode == "rgb":
            return np.ascontiguousarray(frame[:, :, ::-1])
        return frame

    return read


def detect_carabiner_wrist(frame: Any) -> Any:
    """One BGR wrist frame through the carabiner detector. A miss raises."""

    from everest_robot.carabiner_detect import detect

    return detect(frame)


@dataclass
class WristCarabinerFollower:
    """Detector, image Jacobian and speed-locked tracker, stepped one action at a time."""

    calibration: WristServoCalibration
    tracker: VisualTracker
    frames: Callable[[], Any]
    detect: Callable[[Any], Any]
    clock: Clock = field(default_factory=SystemClock)
    max_jump_px: float = DEFAULT_MAX_JUMP_PX
    lost_after_misses: int = 5
    settle_ticks: int = 3

    def __post_init__(self) -> None:
        if self.calibration.joint_names != tuple(self.tracker.port.joint_names):
            raise WristServoError(
                f"calibration joints {', '.join(self.calibration.joint_names)} do not match "
                f"this arm's {', '.join(self.tracker.port.joint_names)}"
            )
        if self.lost_after_misses < 1:
            raise ValueError("lost_after_misses must be at least 1")
        if self.settle_ticks < 1:
            raise ValueError("settle_ticks must be at least 1")
        self._index = 0
        self._misses = 0
        self._settled = 0
        self._previous: tuple[float, float] | None = None
        self._started_at: float | None = None

    def start(self) -> None:
        """Seed the tracker from measured feedback and energize.

        Re-callable, and the caller is expected to re-call it on every entry into
        ``SEARCH_CV``: a policy has moved the arm since the last one, so the tracker's
        seeded command, the lock-on count and the jump gate's previous pixel are all stale.
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
        """One detect/solve/servo tick, paced to the tracker's rate.

        Raises :class:`~everest_robot.robot.visual_tracking.TrackerStopped` only for a
        genuine stop -- an arm fault, lost feedback, a refused command. A missed detection
        and a refused solve are both ordinary outcomes and are reported in the tick.
        """

        self._pace()
        self._index += 1

        target, detected, solve, reason = self._target_from_frame(self.frames())
        tick = self.tracker.tick(target, reason)

        if detected:
            self._misses = 0
        else:
            self._misses += 1

        # Settling is judged on the image, not on the arm: a solve inside tolerance is the
        # measurement that the carabiner is where the clip policy was taught to find it.
        # A tick with no detection at all cannot support that claim and resets the count.
        if solve is not None and solve.settled:
            self._settled += 1
        else:
            self._settled = 0

        return FollowTick(
            index=self._index,
            target_visible=self._misses < self.lost_after_misses,
            followed=self._settled >= self.settle_ticks,
            reason=tick.reason,
            moved=tick.moved,
            pixel=self._previous if detected else None,
            pixel_error_px=None if solve is None else math.hypot(solve.error[0], solve.error[1]),
            approach_error_rad=(
                None
                if solve is None or not solve.delta_rad
                else max(abs(value) for value in solve.delta_rad.values())
            ),
            # No hull: the wrist calibration has no sampled region to be inside of. Its
            # equivalent guard is the refused solve, which shows up as a hold reason.
            hull_margin_px=None,
            misses=self._misses,
            settled_ticks=self._settled,
        )

    # ── internals ──────────────────────────────────────────────────────────────────
    def _pace(self) -> None:
        """Wait out the remainder of the tracker's period since the previous step began."""

        if self._started_at is not None:
            self.clock.sleep(self.tracker.period_s - (self.clock.monotonic() - self._started_at))
        self._started_at = self.clock.monotonic()

    def _target_from_frame(
        self, frame: Any
    ) -> tuple[tuple[float, ...] | None, bool, Any, str]:
        """One frame to a full joint target, or to a reason for holding still.

        Returns whether anything was *detected* separately from whether a target came out
        of it, because the two answer different questions: the first decides whether the
        carabiner is still visible, and the second decides whether the arm moves.
        """

        from everest_robot.carabiner_detect import NotFound

        try:
            detection = self.detect(frame)
        except (NotFound, RuntimeError, ValueError) as error:
            self._previous = None
            return None, False, None, f"{NO_DETECTION} ({str(error).split('.')[0][:50]})"

        try:
            features = self.calibration.features(detection)
        except WristServoError as error:
            self._previous = None
            return None, False, None, f"{NO_DETECTION} ({error})"

        pixel = (features[0], features[1])
        jump = None if self._previous is None else math.dist(self._previous, pixel)
        self._previous = pixel
        if jump is not None and jump > self.max_jump_px:
            # A different blob, not the same carabiner moving. Still visible; just not
            # something to lunge at.
            return None, True, None, f"detection jumped {jump:.0f} px"

        try:
            solve = self.calibration.solve(features)
        except UnsupportedSolve as error:
            return None, True, None, f"{UNSUPPORTED}: {error}"

        if solve.settled:
            # Arrived. Holding is the correct command: the clip policy takes over from
            # here, and nudging at the noise floor would only walk the pose around.
            return None, True, solve, AT_GOAL

        state = self.tracker.port.read_state()
        if not state.all_finite:
            return None, True, solve, NO_FEEDBACK
        return (
            self.calibration.joint_target(solve, state.positions),
            True,
            solve,
            TRACKING,
        )
