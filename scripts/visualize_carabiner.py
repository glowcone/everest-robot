"""Draw the detected carabiner as a square on an image.

Usage:
    uv run python scripts/visualize_carabiner.py <image> [output]

Defaults match the HSV bounds in .env.example; override with --lower/--upper/--min-area.
If no output path is given, the result is shown in a window.
"""

import argparse
import sys

import cv2
import numpy as np

from everest_robot.vision import Detection, HsvDetectionConfig, detect_carabiner


def _parse_hsv(value: str) -> tuple[int, int, int]:
    h, s, v = (int(part) for part in value.split(","))
    return (h, s, v)


# Triangle covering the robot arm silhouette at bottom-center of the overhead frame,
# as (x, y) fractions of the frame: apex above the arm, base spanning the bottom edge.
ARM_MASK_TRIANGLE = ((0.48, 0.55), (0.30, 1.0), (0.68, 1.0))


def _arm_triangle_px(frame_width: int, frame_height: int) -> np.ndarray:
    return np.array(
        [(int(x * frame_width), int(y * frame_height)) for x, y in ARM_MASK_TRIANGLE],
        dtype=np.int32,
    )


def mask_robot_arm(bgr_image: np.ndarray) -> np.ndarray:
    """Black out the robot arm region so it cannot be detected as the carabiner."""
    masked = bgr_image.copy()
    height, width = masked.shape[:2]
    cv2.fillPoly(masked, [_arm_triangle_px(width, height)], color=(0, 0, 0))
    return masked


def draw_detection(bgr_image: np.ndarray, detection: Detection) -> np.ndarray:
    """Overlay a rotated square (side = sqrt(area)) centered on the detection."""
    side = float(np.sqrt(detection.area))
    rect = ((detection.pixel_x, detection.pixel_y), (side, side), detection.angle_deg)
    box = cv2.boxPoints(rect).astype(np.int32)
    annotated = bgr_image.copy()
    cv2.polylines(annotated, [box], isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.drawMarker(
        annotated,
        (int(detection.pixel_x), int(detection.pixel_y)),
        color=(0, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=10,
        thickness=2,
    )
    cv2.putText(
        annotated,
        f"{detection.angle_deg:.1f} deg, area {detection.area:.0f}",
        (int(detection.pixel_x) + 10, int(detection.pixel_y) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
    )
    return annotated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("output", nargs="?", help="Optional path to save the annotated image")
    parser.add_argument(
        "--lower", type=_parse_hsv, default=(35, 80, 80), help="HSV lower bound h,s,v"
    )
    parser.add_argument(
        "--upper", type=_parse_hsv, default=(85, 255, 255), help="HSV upper bound h,s,v"
    )
    parser.add_argument("--min-area", type=float, default=400.0, help="Minimum blob area in pixels")
    parser.add_argument(
        "--no-arm-mask",
        action="store_true",
        help="Do not black out the robot arm region at bottom-center before detection",
    )
    args = parser.parse_args()

    bgr_image = cv2.imread(args.image)
    if bgr_image is None:
        print(f"error: could not read image {args.image}", file=sys.stderr)
        return 1

    config = HsvDetectionConfig(
        hsv_lower=args.lower,
        hsv_upper=args.upper,
        min_contour_area=args.min_area,
    )
    detect_image = bgr_image if args.no_arm_mask else mask_robot_arm(bgr_image)
    detection = detect_carabiner(detect_image, config)
    if detection is None:
        print("no carabiner detected", file=sys.stderr)
        return 1

    annotated = draw_detection(bgr_image, detection)
    if not args.no_arm_mask:
        height, width = annotated.shape[:2]
        cv2.polylines(
            annotated,
            [_arm_triangle_px(width, height)],
            isClosed=True,
            color=(255, 200, 0),
            thickness=2,
        )
    if args.output:
        cv2.imwrite(args.output, annotated)
        print(f"saved {args.output}")
    else:
        cv2.imshow("carabiner", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
