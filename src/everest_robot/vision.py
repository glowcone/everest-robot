"""Simple colored-marker perception for hackathon carabiner pickup."""

from __future__ import annotations

import math
from dataclasses import dataclass

from everest_robot.pickup import Point2, Pose2D, normalize_angle


@dataclass(frozen=True)
class MarkerDetection:
    center_px: Point2
    area_px: float


@dataclass(frozen=True)
class BarDetection:
    center_px: Point2
    yaw_px: float
    length_px: float
    endpoints_px: tuple[Point2, Point2]


@dataclass(frozen=True)
class TwoWhiteBlackDetection:
    white_points_px: tuple[Point2, Point2]
    black_gate_px: Point2


def _require_cv2():
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for live marker detection. Install with: "
            "uv add opencv-python"
        ) from exc
    return cv2, np


def _largest_blob_center(mask, min_area_px: float) -> MarkerDetection | None:
    cv2, _ = _require_cv2()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < min_area_px:
        return None
    moments = cv2.moments(contour)
    if abs(moments["m00"]) < 1e-9:
        return None
    return MarkerDetection(
        center_px=Point2(
            float(moments["m10"] / moments["m00"]),
            float(moments["m01"] / moments["m00"]),
        ),
        area_px=area,
    )


def detect_colored_markers_bgr(
    frame_bgr,
    min_area_px: float = 80.0,
) -> tuple[MarkerDetection, MarkerDetection]:
    """Detect red and blue tape markers in a BGR frame from cv2.VideoCapture."""

    cv2, np = _require_cv2()
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    red_mask = cv2.inRange(hsv, np.array([0, 80, 60]), np.array([12, 255, 255]))
    red_mask |= cv2.inRange(hsv, np.array([170, 80, 60]), np.array([179, 255, 255]))
    blue_mask = cv2.inRange(hsv, np.array([95, 70, 50]), np.array([135, 255, 255]))

    kernel = np.ones((5, 5), dtype=np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

    red = _largest_blob_center(red_mask, min_area_px)
    blue = _largest_blob_center(blue_mask, min_area_px)
    if red is None or blue is None:
        raise RuntimeError(f"Could not detect both markers. red={red}, blue={blue}")
    return red, blue


def _component_candidates(mask, min_area_px: float, max_area_px: float):
    cv2, _ = _require_cv2()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[MarkerDetection] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area_px or area > max_area_px:
            continue
        moments = cv2.moments(contour)
        if abs(moments["m00"]) < 1e-9:
            continue
        candidates.append(
            MarkerDetection(
                center_px=Point2(
                    float(moments["m10"] / moments["m00"]),
                    float(moments["m01"] / moments["m00"]),
                ),
                area_px=area,
            )
        )
    return candidates


def _add_roi_offset(detection: MarkerDetection, x0: int, y0: int) -> MarkerDetection:
    return MarkerDetection(
        center_px=Point2(detection.center_px.x + x0, detection.center_px.y + y0),
        area_px=detection.area_px,
    )


def detect_white_black_markers_bgr(
    frame_bgr,
    *,
    roi_xywh: tuple[int, int, int, int],
    min_area_px: float = 40.0,
    max_area_px: float = 3000.0,
) -> tuple[MarkerDetection, MarkerDetection]:
    """Detect white-top and black-bottom tape markers inside a tight ROI.

    White tape on white paper is ambiguous. Use this over the wooden table or a darker
    mat, not a white sheet. The detector still restricts marker masks to pixels near
    the green carabiner body.
    It returns `(white_marker, black_marker)`.
    """

    cv2, np = _require_cv2()
    x0, y0, width, height = roi_xywh
    roi = frame_bgr[y0 : y0 + height, x0 : x0 + width]
    if roi.size == 0:
        raise ValueError(f"Invalid empty ROI: {roi_xywh}")

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    green = cv2.inRange(hsv, np.array([35, 35, 25]), np.array([100, 255, 230]))
    green = cv2.morphologyEx(
        green,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    carabiner_neighborhood = cv2.dilate(
        green,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51)),
    )
    if int(cv2.countNonZero(carabiner_neighborhood)) == 0:
        raise RuntimeError("Could not find green carabiner body inside ROI.")

    white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([179, 115, 255]))
    black_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([179, 255, 95]))
    white_mask &= carabiner_neighborhood
    black_mask &= carabiner_neighborhood

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)

    white_candidates = _component_candidates(white_mask, min_area_px, max_area_px)
    black_candidates = _component_candidates(black_mask, min_area_px, max_area_px)
    if not white_candidates or not black_candidates:
        raise RuntimeError(
            "Could not detect both white and black markers. "
            f"white_candidates={white_candidates}, black_candidates={black_candidates}"
        )

    white = max(white_candidates, key=lambda candidate: candidate.area_px)
    black = max(black_candidates, key=lambda candidate: candidate.area_px)
    return _add_roi_offset(white, x0, y0), _add_roi_offset(black, x0, y0)


def detect_two_white_tapes_and_black_gate_bgr(
    frame_bgr,
    *,
    roi_xywh: tuple[int, int, int, int],
    min_white_area_px: float = 40.0,
    max_white_area_px: float = 2500.0,
    min_black_area_px: float = 40.0,
    max_black_area_px: float = 4000.0,
) -> TwoWhiteBlackDetection:
    """Detect two white tape blobs and one black gate blob inside a tight ROI."""

    cv2, np = _require_cv2()
    x0, y0, width, height = roi_xywh
    roi = frame_bgr[y0 : y0 + height, x0 : x0 + width]
    if roi.size == 0:
        raise ValueError(f"Invalid empty ROI: {roi_xywh}")

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    green = cv2.inRange(hsv, np.array([35, 35, 25]), np.array([100, 255, 230]))
    green = cv2.morphologyEx(
        green,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    carabiner_neighborhood = cv2.dilate(
        green,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51)),
    )
    if int(cv2.countNonZero(carabiner_neighborhood)) == 0:
        raise RuntimeError("Could not find green carabiner body inside ROI.")

    white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([179, 115, 255]))
    black_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([179, 255, 95]))
    white_mask &= carabiner_neighborhood
    black_mask &= carabiner_neighborhood

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)

    white_candidates = _component_candidates(
        white_mask,
        min_white_area_px,
        max_white_area_px,
    )
    black_candidates = _component_candidates(
        black_mask,
        min_black_area_px,
        max_black_area_px,
    )
    if len(white_candidates) < 2 or not black_candidates:
        raise RuntimeError(
            "Could not detect two white tapes and one black gate. "
            f"white_candidates={white_candidates}, black_candidates={black_candidates}"
        )

    whites = sorted(white_candidates, key=lambda candidate: candidate.area_px, reverse=True)[:2]
    black = max(black_candidates, key=lambda candidate: candidate.area_px)
    return TwoWhiteBlackDetection(
        white_points_px=(
            _add_roi_offset(whites[0], x0, y0).center_px,
            _add_roi_offset(whites[1], x0, y0).center_px,
        ),
        black_gate_px=_add_roi_offset(black, x0, y0).center_px,
    )


def detect_straight_bar_bgr(
    frame_bgr,
    *,
    roi_xywh: tuple[int, int, int, int] | None = None,
    min_length_px: float = 60.0,
    border_margin_px: int = 20,
) -> BarDetection:
    """Detect the strongest interior straight segment in the pickup zone.

    This is intended for the carabiner's straight gate/spine. It requires a tight pickup-zone
    crop; otherwise table, paper, rope, or laptop edges can be stronger than the carabiner.
    """

    cv2, np = _require_cv2()
    if roi_xywh is None:
        x0, y0 = 0, 0
        roi = frame_bgr
    else:
        x0, y0, width, height = roi_xywh
        roi = frame_bgr[y0 : y0 + height, x0 : x0 + width]
        if roi.size == 0:
            raise ValueError(f"Invalid empty ROI: {roi_xywh}")

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=45,
        minLineLength=int(min_length_px),
        maxLineGap=10,
    )
    if lines is None:
        raise RuntimeError("No straight bar candidates found.")

    roi_height, roi_width = gray.shape
    candidates: list[tuple[float, BarDetection]] = []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        length = math.hypot(x2 - x1, y2 - y1)
        if length < min_length_px:
            continue
        if min(x1, x2) < border_margin_px or max(x1, x2) > roi_width - border_margin_px:
            continue
        if min(y1, y2) < border_margin_px or max(y1, y2) > roi_height - border_margin_px:
            continue

        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.line(mask, (x1, y1), (x2, y2), 255, 5)
        mean_darkness = 255.0 - float(cv2.mean(gray, mask=mask)[0])
        score = length * max(mean_darkness, 1.0)
        yaw = math.atan2(y2 - y1, x2 - x1)
        detection = BarDetection(
            center_px=Point2(float((x1 + x2) / 2 + x0), float((y1 + y2) / 2 + y0)),
            yaw_px=normalize_angle(yaw),
            length_px=float(length),
            endpoints_px=(
                Point2(float(x1 + x0), float(y1 + y0)),
                Point2(float(x2 + x0), float(y2 + y0)),
            ),
        )
        candidates.append((score, detection))

    if not candidates:
        raise RuntimeError("No interior straight bar candidates survived filtering.")
    return max(candidates, key=lambda item: item[0])[1]


def bar_pose_from_detection(detection: BarDetection) -> Pose2D:
    return Pose2D(detection.center_px.x, detection.center_px.y, detection.yaw_px)
