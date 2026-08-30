"""The whole chain on synthetic frames: detector -> map -> tracker -> arm.

Every other test covers one link. This one draws a carabiner the real segmentation
accepts and pushes it through :func:`detect_carabiner`, the fitted map, and
:class:`VisualTracker`, because the interesting failures are at the seams -- a centroid
that is not the object's, a spine angle with the wrong handedness, a target the tracker
walks toward at the wrong speed.
"""

import math

import numpy as np
import pytest
from test_pixel_map import FIT_JOINTS, calibration

from everest_robot.calibrate_pixel_map import detect_carabiner
from everest_robot.robot.clock import ManualClock
from everest_robot.robot.contracts import JointLimit, RobotIdentity
from everest_robot.robot.fake_arm import FakeArm
from everest_robot.robot.visual_tracking import VisualTracker

ROI = (100, 80, 400, 380)


def frame_with_carabiner(cx: float, cy: float, angle_deg: float = 20.0) -> np.ndarray:
    """A green body with two white tapes and a black gate, on a mid-grey table.

    The table is deliberately lighter than the detector's black threshold: a dark table
    is itself a black blob, which is the first thing that goes wrong in a real cell.
    """

    import cv2

    frame = np.full((480, 640, 3), 150, dtype=np.uint8)
    direction = np.array([math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))])
    a = np.array([cx, cy]) - 40 * direction
    b = np.array([cx, cy]) + 40 * direction
    cv2.line(frame, tuple(a.astype(int)), tuple(b.astype(int)), (0, 180, 0), 26)
    for endpoint in (a, b):
        cv2.circle(frame, tuple(endpoint.astype(int)), 7, (245, 245, 245), -1)
    gate = np.array([cx, cy]) + 16 * np.array([-direction[1], direction[0]])
    cv2.circle(frame, tuple(gate.astype(int)), 11, (0, 180, 0), -1)
    cv2.circle(frame, tuple(gate.astype(int)), 8, (20, 20, 20), -1)
    return frame


def fake_arm(joint_names, clock):
    hardware = FakeArm(
        RobotIdentity("maker-arm-02", "maker-arm-v1", "maker-arm-02-2026-08-20", joint_names),
        tuple(JointLimit(name, -2.0, 2.0) for name in joint_names),
        clock=clock,
        positions=[0.0] * len(joint_names),
    )
    hardware.connect()
    return hardware


def test_the_detector_reports_the_object_centroid_and_its_spine_angle():
    detection = detect_carabiner(frame_with_carabiner(300, 200, angle_deg=20.0), ROI)
    assert detection.centroid == pytest.approx((300.0, 200.0), abs=2.0)
    assert math.degrees(detection.spine_rad) == pytest.approx(20.0, abs=3.0)


def test_the_gate_resolves_the_axis_sign_rather_than_leaving_it_ambiguous():
    """Two identical tapes give an axis with no sign; flipping the gate must flip it."""

    import cv2

    frame = frame_with_carabiner(300, 200, angle_deg=20.0)
    mirrored = cv2.flip(frame, 0)
    forward = detect_carabiner(frame, ROI)
    backward = detect_carabiner(mirrored, ROI)
    difference = abs(math.degrees(forward.spine_rad - backward.spine_rad))
    assert difference > 30.0


def test_tracking_walks_toward_the_detection_at_the_locked_speed_and_holds_on_a_miss():
    fitted = calibration().fitted(fit_joints=FIT_JOINTS)
    clock = ManualClock()
    arm = fake_arm(fitted.joint_names, clock)
    tracker = VisualTracker(arm, rate_hz=10.0, max_velocity_rad_s=0.5, lock_frames=2)
    tracker.start()

    def step(pixel: tuple[float, float] | None):
        clock.advance(0.1)
        if pixel is None:
            return tracker.tick(None)
        detection = detect_carabiner(frame_with_carabiner(*pixel), ROI)
        prediction = fitted.predict(detection.centroid, detection.spine_rad)
        return tracker.tick(fitted.full_target(prediction, arm.read_state().positions))

    assert not step((300, 200)).moved  # still locking on
    moving = step((300, 200))
    assert moving.moved
    assert moving.command is not None
    assert max(abs(value) for value in moving.command) <= tracker.max_step_rad + 1e-9

    held = step(None)
    assert not held.moved
    assert held.reason == "no detection"
    assert not tracker.locked

    # A new object position re-locks and is then followed, still one step at a time.
    assert not step((360, 260)).moved
    resumed = step((360, 260))
    assert resumed.moved
    assert resumed.target is not None
    assert resumed.target != moving.target


def test_the_tracked_target_matches_what_the_map_predicts_for_that_pixel():
    fitted = calibration().fitted(fit_joints=FIT_JOINTS)
    detection = detect_carabiner(frame_with_carabiner(300, 200), ROI)
    prediction = fitted.predict(detection.centroid, detection.spine_rad)

    arm = fake_arm(fitted.joint_names, ManualClock())
    target = fitted.full_target(prediction, arm.read_state().positions)
    assert target[:4] == pytest.approx(
        tuple(prediction.joints[name] for name in FIT_JOINTS)
    )
    assert target[4] == pytest.approx(prediction.roll_rad)
    assert target[5] == 0.0  # gripper was never fitted; it holds where it is


# ── the handoff ────────────────────────────────────────────────────────────────────
class StubCapture:
    """A camera that always sees the carabiner at one pixel, and drives simulated time.

    Advancing the arm's clock once per frame is what makes the loop converge in a test:
    the tracker commands a step per tick, and the arm integrates one frame's worth of
    motion toward it.
    """

    def __init__(self, clock, pixel=(300, 200), frames=400):
        self.clock = clock
        self.pixel = pixel
        self.remaining = frames

    def read(self):
        if self.remaining <= 0:
            raise KeyboardInterrupt  # the loop treats this as the operator quitting
        self.remaining -= 1
        self.clock.advance(0.1)
        return True, frame_with_carabiner(*self.pixel)

    def release(self):
        pass


def track_arguments(**overrides):
    import argparse

    defaults = dict(
        no_window=True,
        rate=1000.0,  # the loop's own pacing; simulated arm time comes from the capture
        max_velocity=50.0,
        max_jump_px=150.0,
        no_roll_tracking=False,
        policy=None,
    )
    return argparse.Namespace(**{**defaults, **overrides})


def run_track_loop(**overrides):
    from types import SimpleNamespace

    from everest_robot.calibrate_pixel_map import _track_loop
    from everest_robot.pixel_map import CameraSource
    from everest_robot.robot.visual_tracking import ArrivalGate

    fitted = calibration().fitted(fit_joints=FIT_JOINTS)
    clock = ManualClock()
    arm = fake_arm(fitted.joint_names, clock)
    tracker = VisualTracker(arm, rate_hz=1000.0, max_velocity_rad_s=50.0, lock_frames=2)
    tracker.start()
    args = track_arguments(**overrides)
    capture = StubCapture(clock, frames=200)
    reasons, arrived = _track_loop(
        SimpleNamespace(port=arm),
        fitted,
        tracker,
        capture,
        CameraSource(index_or_path="0"),
        ROI,
        args,
        None,
        ArrivalGate(tolerance_rad=0.02, ticks=3),
    )
    return reasons, arrived, capture


def test_the_loop_hands_over_once_the_arm_has_settled_on_the_tracked_pose():
    reasons, arrived, capture = run_track_loop(policy="checkpoints/act")

    assert arrived
    # It stopped at the arrival rather than running the capture out, which is what leaves
    # the arm held on target for the policy instead of still servoing at it.
    assert capture.remaining > 0
    assert reasons["tracking"] > 0


def test_without_a_policy_arriving_does_not_end_the_loop():
    """Tracking with nothing to hand over to is a calibration check; it keeps tracking."""

    reasons, arrived, capture = run_track_loop(policy=None)

    assert not arrived
    assert capture.remaining == 0
    assert reasons["tracking"] > 3
