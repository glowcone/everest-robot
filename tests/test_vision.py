import cv2
import numpy as np

from everest_robot.vision import (
    Detection,
    GripperCheckConfig,
    HsvDetectionConfig,
    carabiner_in_gripper,
    detect_carabiner,
)

GREEN_BGR = (60, 200, 60)

DETECTION_CONFIG = HsvDetectionConfig(
    hsv_lower=(35, 80, 80),
    hsv_upper=(85, 255, 255),
    min_contour_area=400,
)

GRIPPER_CONFIG = GripperCheckConfig(
    hsv_lower=(35, 80, 80),
    hsv_upper=(85, 255, 255),
    roi=(0.25, 0.25, 0.5, 0.5),
    min_mask_pixels=200,
)


def blank_frame(width: int = 640, height: int = 480) -> np.ndarray:
    return np.full((height, width, 3), (255, 255, 255), dtype=np.uint8)


def draw_rotated_rect(
    frame: np.ndarray,
    center: tuple[float, float],
    size: tuple[float, float],
    angle_deg: float,
) -> None:
    box = cv2.boxPoints((center, size, angle_deg))
    cv2.fillConvexPoly(frame, box.astype(np.int32), GREEN_BGR)


def test_detects_centroid_and_long_axis_angle() -> None:
    frame = blank_frame()
    draw_rotated_rect(frame, center=(320.0, 240.0), size=(120.0, 40.0), angle_deg=30.0)

    detection = detect_carabiner(frame, DETECTION_CONFIG)

    assert isinstance(detection, Detection)
    assert abs(detection.pixel_x - 320.0) <= 2.0
    assert abs(detection.pixel_y - 240.0) <= 2.0
    assert abs(detection.angle_deg - 30.0) <= 3.0
    assert detection.area >= 120.0 * 40.0 * 0.9


def test_empty_scene_returns_none() -> None:
    assert detect_carabiner(blank_frame(), DETECTION_CONFIG) is None


def test_blob_below_min_area_is_ignored() -> None:
    frame = blank_frame()
    draw_rotated_rect(frame, center=(100.0, 100.0), size=(15.0, 10.0), angle_deg=0.0)

    assert detect_carabiner(frame, DETECTION_CONFIG) is None


def test_largest_blob_wins() -> None:
    frame = blank_frame()
    draw_rotated_rect(frame, center=(120.0, 120.0), size=(50.0, 30.0), angle_deg=0.0)
    draw_rotated_rect(frame, center=(450.0, 350.0), size=(150.0, 60.0), angle_deg=0.0)

    detection = detect_carabiner(frame, DETECTION_CONFIG)

    assert detection is not None
    assert abs(detection.pixel_x - 450.0) <= 2.0
    assert abs(detection.pixel_y - 350.0) <= 2.0


def test_gripper_check_true_when_carabiner_in_roi() -> None:
    frame = blank_frame()
    draw_rotated_rect(frame, center=(320.0, 240.0), size=(100.0, 40.0), angle_deg=0.0)

    assert carabiner_in_gripper(frame, GRIPPER_CONFIG)


def test_gripper_check_false_for_empty_roi() -> None:
    frame = blank_frame()
    # Blob outside the centered ROI.
    draw_rotated_rect(frame, center=(50.0, 50.0), size=(100.0, 40.0), angle_deg=0.0)

    assert not carabiner_in_gripper(frame, GRIPPER_CONFIG)
