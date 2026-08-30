"""RGB-only carabiner perception built on OpenCV color/contour analysis.

Pure functions over in-memory images so the pipeline is deterministic and unit-testable
without cameras. Frame capture lives in `everest_robot.hardware`.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class HsvDetectionConfig:
    """HSV threshold and blob-size settings for carabiner detection."""

    hsv_lower: tuple[int, int, int]
    hsv_upper: tuple[int, int, int]
    min_contour_area: float


@dataclass(frozen=True)
class GripperCheckConfig:
    """Wrist-camera region-of-interest settings for the grasp check.

    The ROI is expressed as fractions of the frame: (x, y, width, height).
    """

    hsv_lower: tuple[int, int, int]
    hsv_upper: tuple[int, int, int]
    roi: tuple[float, float, float, float]
    min_mask_pixels: int


@dataclass(frozen=True)
class Detection:
    """A carabiner detection in image space."""

    pixel_x: float
    pixel_y: float
    angle_deg: float
    area: float


def _hsv_mask(
    bgr_image: np.ndarray,
    lower: tuple[int, int, int],
    upper: tuple[int, int, int],
) -> np.ndarray:
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def detect_carabiner(bgr_image: np.ndarray, config: HsvDetectionConfig) -> Detection | None:
    """Find the carabiner as the largest color blob; None when nothing qualifies."""

    mask = _hsv_mask(bgr_image, config.hsv_lower, config.hsv_upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest))
    if area < config.min_contour_area:
        return None
    (center_x, center_y), (width, height), angle = cv2.minAreaRect(largest)
    # Normalize so the angle describes the long axis of the carabiner.
    if width < height:
        angle += 90.0
    return Detection(
        pixel_x=float(center_x),
        pixel_y=float(center_y),
        angle_deg=float(angle % 180.0),
        area=area,
    )


def carabiner_in_gripper(bgr_image: np.ndarray, config: GripperCheckConfig) -> bool:
    """Check whether enough carabiner-colored pixels appear in the gripper ROI."""

    frame_height, frame_width = bgr_image.shape[:2]
    roi_x, roi_y, roi_w, roi_h = config.roi
    x0 = int(roi_x * frame_width)
    y0 = int(roi_y * frame_height)
    x1 = min(frame_width, x0 + int(roi_w * frame_width))
    y1 = min(frame_height, y0 + int(roi_h * frame_height))
    roi = bgr_image[y0:y1, x0:x1]
    if roi.size == 0:
        return False
    mask = _hsv_mask(roi, config.hsv_lower, config.hsv_upper)
    return int(cv2.countNonZero(mask)) >= config.min_mask_pixels
