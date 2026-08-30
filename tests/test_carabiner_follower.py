"""The decisions SEARCH_CV makes: when the target is lost, and when the approach is done.

The detector-to-map-to-tracker plumbing is covered on synthetic frames in
``test_pixel_map_tracking``. Here the detector is scripted, because what is under test is
the follower's own judgement -- a flicker is not a loss, motion is not arrival, and one
step is one tracker period no matter how fast the caller comes back.
"""

from dataclasses import dataclass

import pytest
from test_pixel_map import FIT_JOINTS, calibration

from everest_robot.pixel_map import PixelMapError
from everest_robot.robot.carabiner_follower import CarabinerFollower
from everest_robot.robot.clock import ManualClock
from everest_robot.robot.contracts import JointLimit, RobotIdentity
from everest_robot.robot.fake_arm import FakeArm
from everest_robot.robot.visual_tracking import TrackerStopped, VisualTracker

RATE_HZ = 10.0


@dataclass(frozen=True)
class Detection:
    centroid: tuple[float, float]
    spine_rad: float = 0.0


class ScriptedCamera:
    """Yields one detection per step; ``None`` is a frame the detector rejected."""

    def __init__(self, script):
        self.script = list(script)
        self.reads = 0

    def frame(self) -> object:
        self.reads += 1
        return self.script[min(self.reads, len(self.script)) - 1]

    def detect(self, frame) -> Detection:
        if frame is None:
            raise RuntimeError("Could not detect both markers. white=None")
        return Detection(centroid=frame)


def build(script, **kwargs) -> tuple[CarabinerFollower, FakeArm, ManualClock]:
    fitted = calibration().fitted(fit_joints=FIT_JOINTS)
    clock = ManualClock()
    arm = FakeArm(
        RobotIdentity(
            "maker-arm-02", "maker-arm-v1", "maker-arm-02-2026-08-20", fitted.joint_names
        ),
        tuple(JointLimit(name, -2.0, 2.0) for name in fitted.joint_names),
        clock=clock,
        positions=[0.0] * len(fitted.joint_names),
    )
    arm.connect()
    camera = ScriptedCamera(script)
    follower = CarabinerFollower(
        calibration=fitted,
        tracker=VisualTracker(arm, rate_hz=RATE_HZ, max_velocity_rad_s=0.5, lock_frames=2),
        frames=camera.frame,
        detect=camera.detect,
        clock=clock,
        **kwargs,
    )
    follower.start()
    return follower, arm, clock


def run(follower, count: int):
    return [follower.step() for _ in range(count)]


def test_one_dropped_frame_is_not_a_lost_carabiner():
    """A threshold segmentation flickers; bouncing the FSM back to RL search on that
    would trade a held arm for a search policy that has nothing new to find."""

    follower, _, _ = build([(300, 200), None, (300, 200)], lost_after_misses=3)

    visible = [tick.target_visible for tick in run(follower, 3)]

    assert visible == [True, True, True]


def test_a_sustained_miss_reports_the_target_lost():
    follower, _, _ = build([(300, 200), None, None, None], lost_after_misses=3)

    ticks = run(follower, 4)

    assert [tick.target_visible for tick in ticks] == [True, True, True, False]
    assert ticks[-1].misses == 3
    assert ticks[-1].reason.startswith("no detection")


def test_followed_means_arrived_not_merely_moving():
    """`followed` is the authoritative transition into the clip policy, so it must not
    fire while the arm is still crossing the table toward the pre-grasp pose."""

    follower, arm, _ = build([(300, 200)] * 40, settle_ticks=3)

    ticks = run(follower, 40)
    moving = [tick for tick in ticks if tick.moved and not tick.followed]

    assert moving, "the arm should have spent ticks approaching before it settled"
    assert not any(tick.followed for tick in ticks[:3])
    followed = next(index for index, tick in enumerate(ticks) if tick.followed)
    # Three consecutive settled ticks, and not one more than that.
    assert ticks[followed].settled_ticks == 3
    assert ticks[followed].approach_error_rad <= follower.settle_tolerance_rad
    assert arm.sent_commands


def test_losing_the_target_mid_approach_discards_the_settle_run():
    follower, _, _ = build([(300, 200)] * 30 + [None], settle_ticks=3)

    approaching = run(follower, 30)
    lost = follower.step()

    assert approaching[-1].settled_ticks > 0
    assert lost.settled_ticks == 0
    assert not lost.followed


def test_a_detection_outside_the_taught_region_holds_but_stays_visible():
    """The camera is bolted down, so no arm motion moves the carabiner back inside the
    hull. Handing this to the search policy would only get the same detection back."""

    # Close enough to the previous frame to clear the jump gate, but above the taught grid.
    follower, _, _ = build([(300, 200), (300, 60)])

    follower.step()
    outside = follower.step()

    assert outside.target_visible
    assert not outside.followed
    assert not outside.moved
    assert outside.reason == "outside the calibrated region"
    assert outside.hull_margin_px is None


def test_a_jump_between_frames_is_treated_as_a_different_blob():
    follower, _, _ = build([(300, 200), (300, 200), (460, 330)], max_jump_px=50.0)

    run(follower, 2)
    jumped = follower.step()

    assert jumped.target_visible
    assert not jumped.moved
    assert jumped.reason.startswith("detection jumped")


def test_each_step_costs_one_tracker_period_however_fast_the_caller_returns():
    """The speed lock is a per-tick clamp: unpaced ticks would move the arm faster than
    max_velocity_rad_s without a single number changing."""

    follower, _, clock = build([(300, 200)] * 5)

    run(follower, 5)

    assert clock.monotonic() == pytest.approx(4.0 / RATE_HZ)


def test_the_pixel_error_shrinks_as_the_arm_closes_on_the_target():
    follower, _, _ = build([(300, 200)] * 30)

    ticks = run(follower, 30)
    errors = [tick.pixel_error_px for tick in ticks]

    assert all(error is not None for error in errors)
    assert errors[0] > errors[-1]
    assert errors[-1] == pytest.approx(0.0, abs=1.0)


def test_an_arm_fault_stops_the_follower_rather_than_returning_a_tick():
    follower, arm, _ = build([(300, 200)] * 10)

    run(follower, 3)
    arm.inject_fault("injected motor fault")
    follower.step()  # the tracker holds and latches its stop reason

    with pytest.raises(TrackerStopped, match="arm fault"):
        follower.step()


def test_a_calibration_for_another_arm_is_refused_before_anything_moves():
    fitted = calibration().fitted(fit_joints=FIT_JOINTS)
    clock = ManualClock()
    names = fitted.joint_names[:-1]
    arm = FakeArm(
        RobotIdentity("maker-arm-02", "maker-arm-v1", "maker-arm-02-2026-08-20", names),
        tuple(JointLimit(name, -2.0, 2.0) for name in names),
        clock=clock,
    )
    arm.connect()

    with pytest.raises(PixelMapError, match="do not match this arm"):
        CarabinerFollower(
            calibration=fitted,
            tracker=VisualTracker(arm, rate_hz=RATE_HZ),
            frames=lambda: None,
            detect=lambda frame: Detection((300.0, 200.0)),
            clock=clock,
        )


def test_an_unfitted_calibration_cannot_drive_the_tracker():
    clock = ManualClock()
    arm = FakeArm(
        RobotIdentity(
            "maker-arm-02", "maker-arm-v1", "maker-arm-02-2026-08-20", calibration().joint_names
        ),
        tuple(JointLimit(name, -2.0, 2.0) for name in calibration().joint_names),
        clock=clock,
    )
    arm.connect()

    with pytest.raises(PixelMapError, match="no fit yet"):
        CarabinerFollower(
            calibration=calibration(),
            tracker=VisualTracker(arm, rate_hz=RATE_HZ),
            frames=lambda: None,
            detect=lambda frame: Detection((300.0, 200.0)),
            clock=clock,
        )
