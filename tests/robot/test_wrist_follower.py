"""The decisions the wrist-camera ``SEARCH_CV`` makes, and how they differ from the fixed one.

The detector is scripted here, as it is in ``test_carabiner_follower``, because what is
under test is the follower's own judgement rather than the segmentation: a flicker is not a
loss, an in-tolerance tick is not arrival, a refused solve is a hold and not a
disappearance, and one step is one tracker period however fast the caller comes back.

The one judgement that genuinely differs from the fixed-camera follower is what "followed"
means. There, arrival is the *arm* reaching a pose the map predicted. Here it is the
*image* being on the goal, which is a direct measurement rather than a fit's prediction, so
the tests assert on what the camera saw and never on where the arm ended up.
"""

import numpy as np
import pytest

from everest_robot.robot.clock import ManualClock
from everest_robot.robot.contracts import JointLimit, RobotIdentity
from everest_robot.robot.fake_arm import FakeArm
from everest_robot.robot.visual_tracking import VisualTracker
from everest_robot.robot.wrist_follower import WristCarabinerFollower
from everest_robot.robot.wrist_servo import WristServoError

from .test_wrist_servo import GOAL, JACOBIAN, JOINTS, SERVO, calibration

RATE_HZ = 10.0


class Detection:
    def __init__(self, insert, area, spine_angle):
        self.insert = insert
        self.area = area
        self.spine_angle = spine_angle


def seen(u, v, scale=GOAL[2], spine=GOAL[3]) -> Detection:
    """A detection stated in feature units, so a test reads as what the camera saw."""

    return Detection((u, v), scale**2, spine)


AT_GOAL = seen(GOAL[0], GOAL[1])


class ScriptedCamera:
    """Yields one detection per step; ``None`` is a frame the detector rejected."""

    def __init__(self, script):
        self.script = list(script)
        self.reads = 0

    def frame(self):
        self.reads += 1
        return self.script[min(self.reads, len(self.script)) - 1]

    @staticmethod
    def detect(frame):
        if frame is None:
            raise ValueError("no component in the expected size/shape range")
        return frame


def build(script, *, calibration_overrides=None, **kwargs):
    fitted = calibration(**(calibration_overrides or {}))
    clock = ManualClock()
    arm = FakeArm(
        RobotIdentity("maker-arm-02", "maker-arm-v1", "maker-arm-02-2026-08-20", JOINTS),
        tuple(JointLimit(name, -2.0, 2.0) for name in JOINTS),
        clock=clock,
        positions=[0.0] * len(JOINTS),
    )
    arm.connect()
    camera = ScriptedCamera(script)
    follower = WristCarabinerFollower(
        calibration=fitted,
        tracker=VisualTracker(arm, rate_hz=RATE_HZ, max_velocity_rad_s=0.5, lock_frames=2),
        frames=camera.frame,
        detect=camera.detect,
        clock=clock,
        **kwargs,
    )
    follower.start()
    return follower, arm, clock


def run(follower, count):
    return [follower.step() for _ in range(count)]


# ── losing the target ──────────────────────────────────────────────────────────────
def test_one_dropped_frame_is_not_a_lost_carabiner():
    """A threshold segmentation flickers. Bouncing the FSM back into a learned search on
    one bad frame trades a converging servo for a policy that has nothing new to find."""

    follower, _, _ = build([seen(400, 300), None, seen(400, 300)], lost_after_misses=3)

    assert [tick.target_visible for tick in run(follower, 3)] == [True, True, True]


def test_a_sustained_miss_reports_the_target_lost():
    """Unlike the fixed camera, this one lost the carabiner *because of where the arm is*,
    so handing back to SEARCH_RL is a remedy rather than a round trip to nowhere."""

    follower, _, _ = build([seen(400, 300), None, None, None], lost_after_misses=3)

    assert [tick.target_visible for tick in run(follower, 4)] == [True, True, True, False]


def test_a_miss_holds_the_arm_rather_than_coasting_toward_the_last_target():
    follower, arm, _ = build([None, None], lost_after_misses=5)

    ticks = run(follower, 2)

    assert not any(tick.moved for tick in ticks)
    assert all(tick.reason.startswith("no detection") for tick in ticks)


# ── arriving ───────────────────────────────────────────────────────────────────────
def test_followed_needs_the_image_on_the_goal_for_consecutive_ticks():
    """One in-tolerance frame is a measurement; three in a row is an arrival. Handing a
    half-converged approach to the clip policy is the failure this state exists to stop."""

    follower, _, _ = build([AT_GOAL] * 4, settle_ticks=3)

    assert [tick.followed for tick in run(follower, 4)] == [False, False, True, True]


def test_a_dropped_frame_restarts_the_settle_count():
    """A frame with no detection cannot support the claim that the carabiner is on the
    goal, so it resets rather than being skipped over."""

    follower, _, _ = build([AT_GOAL, AT_GOAL, None, AT_GOAL, AT_GOAL, AT_GOAL], settle_ticks=3)

    assert [tick.followed for tick in run(follower, 6)] == [False] * 5 + [True]


def test_arriving_commands_no_further_motion():
    """At the goal the correct command is a hold. Nudging at the detector's noise floor
    would walk the handed-over pose around while reporting that it had arrived."""

    follower, _, _ = build([AT_GOAL] * 3, settle_ticks=1)

    ticks = run(follower, 3)

    assert all(tick.followed for tick in ticks)
    assert not any(tick.moved for tick in ticks)
    assert ticks[-1].reason == "at the goal image"


def test_an_off_goal_detection_moves_the_arm_toward_it():
    follower, arm, _ = build([seen(400, 300)] * 4)

    ticks = run(follower, 4)

    assert ticks[-1].moved
    assert not any(tick.followed for tick in ticks)
    assert arm.read_state().positions[JOINTS.index("shoulder_pan")] != pytest.approx(0.0)


def test_the_reported_pixel_error_is_measured_rather_than_derived():
    """The fixed-camera follower has to solve its own fit backwards to state an error in
    pixels. Here the error is what the camera saw, so it is simply the distance."""

    follower, _, _ = build([seen(GOAL[0] + 30.0, GOAL[1] + 40.0)])

    assert run(follower, 1)[0].pixel_error_px == pytest.approx(50.0)


# ── holding without disappearing ───────────────────────────────────────────────────
def test_a_solve_the_jacobian_cannot_support_holds_but_stays_visible():
    """The carabiner is right there; only the step is untrustworthy. Reporting it lost
    would send a learned search after something already in frame."""

    follower, _, _ = build(
        [seen(GOAL[0] + 600.0, GOAL[1])] * 3, calibration_overrides={"max_delta_rad": 0.05}
    )

    ticks = run(follower, 3)

    assert all(tick.target_visible for tick in ticks)
    assert not any(tick.moved for tick in ticks)
    assert ticks[0].reason.startswith("servo step outside the taught range")


def test_a_detection_that_jumps_is_a_different_blob_and_is_not_chased():
    follower, _, _ = build([seen(400, 300), seen(400 + 400, 300)], max_jump_px=150.0)

    ticks = run(follower, 2)

    assert ticks[1].target_visible
    assert not ticks[1].moved
    assert "jumped" in ticks[1].reason


# ── pacing and construction ────────────────────────────────────────────────────────
def test_one_step_is_one_tracker_period_however_fast_the_caller_returns():
    """The speed lock is a per-tick clamp, so ticks arriving faster than `rate_hz` would
    move the arm faster than `max_velocity_rad_s` with none of the numbers changing."""

    follower, _, clock = build([seen(400, 300)] * 3)
    started = clock.monotonic()

    run(follower, 3)

    assert clock.monotonic() - started == pytest.approx(2.0 / RATE_HZ)


def test_a_calibration_for_a_different_arm_is_refused_before_any_tick():
    with pytest.raises(WristServoError, match="do not match this arm"):
        build([AT_GOAL], calibration_overrides={"joint_names": JOINTS[:-1],
                                                "servo_joints": SERVO})


def test_restarting_clears_the_lock_the_settle_count_and_the_jump_gate():
    """A policy has moved the arm since the last entry into SEARCH_CV, so every piece of
    carried state is about a pose the arm has left."""

    follower, _, _ = build([AT_GOAL] * 6, settle_ticks=3)
    run(follower, 3)

    follower.start()
    ticks = run(follower, 2)

    assert [tick.followed for tick in ticks] == [False, False]
    assert ticks[0].misses == 0


def test_the_follower_reports_no_hull_margin_because_it_has_no_sampled_region():
    """The wrist calibration is a derivative, not a set of samples, so there is nothing to
    be inside of. Its equivalent guard is the refused solve."""

    follower, _, _ = build([seen(400, 300)])

    assert run(follower, 1)[0].hull_margin_px is None


def test_the_image_converges_when_the_arm_is_driven_through_the_taught_jacobian():
    """End to end against a synthetic camera that answers joint motion the way the taught
    Jacobian says it should: the loop has to actually close, not merely step plausibly."""

    fitted = calibration()
    clock = ManualClock()
    arm = FakeArm(
        RobotIdentity("maker-arm-02", "maker-arm-v1", "maker-arm-02-2026-08-20", JOINTS),
        tuple(JointLimit(name, -2.0, 2.0) for name in JOINTS),
        clock=clock,
        positions=[0.0] * len(JOINTS),
    )
    arm.connect()

    def look(_frame=None):
        positions = np.asarray(
            [arm.read_state().positions[JOINTS.index(name)] for name in SERVO]
        )
        features = np.asarray([360.0, 265.0, 130.0, 30.0]) + JACOBIAN @ positions
        return Detection((features[0], features[1]), features[2] ** 2, features[3])

    follower = WristCarabinerFollower(
        calibration=fitted,
        tracker=VisualTracker(arm, rate_hz=RATE_HZ, max_velocity_rad_s=5.0, lock_frames=1),
        frames=lambda: None,
        detect=look,
        clock=clock,
        settle_ticks=1,
    )
    follower.start()

    for _ in range(40):
        if follower.step().followed:
            break
    else:
        pytest.fail("the servo never brought the image onto the goal")
