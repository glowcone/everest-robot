"""The attachment gates the wrist-camera detector serves, and the ones it refuses to invent."""

from __future__ import annotations

import numpy as np
import pytest

from everest_robot.attachment_fsm import AttachmentState
from everest_robot.robot.cameras import CameraRuntime, CameraSpec, FakeCamera
from everest_robot.robot.carabiner_perception import (
    CarabinerVisionPerception,
    PerceptionUnavailable,
    UnverifiableAttachment,
)
from everest_robot.robot.clock import ManualClock
from everest_robot.robot.fake_arm import FakeArm
from robot.test_policy_session import IDENTITY, LIMITS


class StubTarget:
    """What `carabiner.detect` returns, reduced to the field perception reads."""

    def __init__(self, insert) -> None:
        self.insert = insert


def make_perception(detections=(), **overrides):
    """A bound perception whose detector is scripted rather than run over pixels."""

    spec = CameraSpec("wrist", "fake", "0", 64, 48, 30)
    cameras = CameraRuntime({"wrist": FakeCamera(spec)}, [spec])
    cameras.connect()
    arm = FakeArm(identity=IDENTITY, joint_limits=LIMITS, clock=ManualClock())
    arm.connect()

    settings = {
        "verifier": UnverifiableAttachment(acknowledged=True),
        "configured_cameras": ("wrist",),
    }
    settings.update(overrides)
    perception = CarabinerVisionPerception(**settings)
    perception.preflight()
    perception.bind(cameras, arm)

    queue = list(detections)
    perception._detect = lambda: queue.pop(0) if queue else None  # noqa: SLF001
    return perception, arm


# ── what it serves ─────────────────────────────────────────────────────────────────
def test_a_detection_is_reported_without_a_fabricated_confidence() -> None:
    perception, _ = make_perception([StubTarget((100.0, 100.0))])

    step = perception.carabiner_detection()

    assert step.carabiner_detected is True
    # The detector is a threshold with shape validation; there is no score behind it.
    assert step.confidence is None


def test_a_miss_is_reported_as_no_detection() -> None:
    perception, _ = make_perception([])

    assert perception.carabiner_detection().carabiner_detected is False


def test_the_first_clip_observation_becomes_the_alignment_baseline() -> None:
    """SEARCH_CV settled to get here, so the hand-over pose is what aligned looks like."""

    perception, _ = make_perception(
        [StubTarget((100.0, 100.0)), StubTarget((110.0, 100.0))],
        alignment_tolerance_px=50.0,
    )

    assert perception.clip_observations().alignment_degraded is False
    assert perception.clip_observations().alignment_degraded is False


def test_drift_beyond_the_tolerance_is_a_degraded_alignment() -> None:
    perception, _ = make_perception(
        [StubTarget((100.0, 100.0)), StubTarget((200.0, 100.0))],
        alignment_tolerance_px=50.0,
    )

    perception.clip_observations()
    step = perception.clip_observations()

    assert step.carabiner_visible is True
    assert step.alignment_degraded is True


def test_re_entering_clip_drops_the_baseline() -> None:
    """A new approach has its own hand-over pose; carrying the old one over compares poses
    the arm reached on different cycles."""

    perception, _ = make_perception(
        [StubTarget((100.0, 100.0)), StubTarget((200.0, 100.0))],
        alignment_tolerance_px=50.0,
    )

    perception.clip_observations()
    perception.enter_state(AttachmentState.CLIP_RL, AttachmentState.SEARCH_CV)
    step = perception.clip_observations()

    assert step.alignment_degraded is False


def test_a_lost_carabiner_is_not_reported_as_misaligned() -> None:
    """Nothing to measure against is not the same as measurably drifted."""

    perception, _ = make_perception([None])

    step = perception.clip_observations()

    assert step.carabiner_visible is False
    assert step.alignment_degraded is False


# ── what it refuses to invent ──────────────────────────────────────────────────────
def test_attachment_is_never_asserted_by_the_detector() -> None:
    """A detector that can see a carabiner cannot see whether it is through something."""

    perception, _ = make_perception([StubTarget((100.0, 100.0))])

    assert perception.clip_observations().attachment_verified is False
    assert perception.initial_observation().already_attached is False


def test_an_unacknowledged_run_without_a_verifier_refuses_before_the_claim() -> None:
    perception = CarabinerVisionPerception(configured_cameras=("wrist",))

    with pytest.raises(PerceptionUnavailable, match="no attachment verifier"):
        perception.preflight()


def test_an_unconfigured_camera_refuses_before_the_claim() -> None:
    perception = CarabinerVisionPerception(
        verifier=UnverifiableAttachment(acknowledged=True), configured_cameras=("front",)
    )

    with pytest.raises(PerceptionUnavailable, match="not configured"):
        perception.preflight()


def test_grasp_is_not_asserted_until_it_has_been_measured() -> None:
    perception, _ = make_perception([StubTarget((100.0, 100.0))])

    assert perception.clip_observations().carabiner_grasped is False


def test_a_measured_gripper_threshold_reports_a_grasp() -> None:
    perception, arm = make_perception(
        [StubTarget((100.0, 100.0)), StubTarget((100.0, 100.0))],
        grasp_gripper_below_rad=0.2,
    )
    gripper = arm.joint_names.index("gripper")

    arm.positions = tuple(
        0.5 if index == gripper else value for index, value in enumerate(arm.positions)
    )
    assert perception.clip_observations().carabiner_grasped is False

    arm.positions = tuple(
        0.1 if index == gripper else value for index, value in enumerate(arm.positions)
    )
    assert perception.clip_observations().carabiner_grasped is True


def test_a_gripper_joint_this_arm_does_not_have_is_refused_on_binding() -> None:
    spec = CameraSpec("wrist", "fake", "0", 64, 48, 30)
    cameras = CameraRuntime({"wrist": FakeCamera(spec)}, [spec])
    arm = FakeArm(identity=IDENTITY, joint_limits=LIMITS, clock=ManualClock())
    perception = CarabinerVisionPerception(
        verifier=UnverifiableAttachment(acknowledged=True),
        configured_cameras=("wrist",),
        grasp_gripper_below_rad=0.2,
        gripper_joint="claw",
    )
    perception.preflight()

    with pytest.raises(PerceptionUnavailable, match="does not have"):
        perception.bind(cameras, arm)


# ── the colour convention ──────────────────────────────────────────────────────────
def test_rgb_frames_are_handed_to_the_detector_as_bgr() -> None:
    """LeRobot cameras produce RGB; the detector's Lab thresholds are built from BGR."""

    spec = CameraSpec("wrist", "fake", "0", 4, 4, 30)
    camera = FakeCamera(spec)
    cameras = CameraRuntime({"wrist": camera}, [spec])
    cameras.connect()
    arm = FakeArm(identity=IDENTITY, joint_limits=LIMITS, clock=ManualClock())
    seen: list[np.ndarray] = []

    def capture(frame):
        seen.append(frame)
        raise LookupError("stop here; the conversion is what is under test")

    perception = CarabinerVisionPerception(
        verifier=UnverifiableAttachment(acknowledged=True), configured_cameras=("wrist",)
    )
    perception.preflight()
    perception.bind(cameras, arm)

    import everest_robot.carabiner_detect as detector

    original, detector.detect = detector.detect, capture
    try:
        with pytest.raises(LookupError):
            perception.carabiner_detection()
    finally:
        detector.detect = original

    rgb = np.full(spec.frame_shape, camera.frames_read % 256, dtype=np.uint8)
    assert np.array_equal(seen[0], rgb[:, :, ::-1])
    # OpenCV rejects a reversed view; the conversion must produce a real array.
    assert seen[0].flags["C_CONTIGUOUS"]
