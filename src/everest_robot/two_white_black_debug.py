"""Capture one frame and detect two white tapes plus the black carabiner gate."""

from __future__ import annotations

import argparse

from everest_robot.vision import detect_two_white_tapes_and_black_gate_bgr


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug two-white-tape plus black-gate detection from a camera frame."
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"), required=True)
    parser.add_argument("--image-out", default="/tmp/carabiner_two_white_black_debug.jpg")
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=15,
        help="Frames to discard while camera exposure and white balance settle (default 15).",
    )
    args = parser.parse_args()

    import cv2  # type: ignore[import-not-found]

    cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")
    try:
        for _ in range(max(0, args.warmup_frames)):
            cap.grab()
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise RuntimeError(f"Could not read from camera index {args.camera}")

    roi = tuple(args.roi)
    detection = detect_two_white_tapes_and_black_gate_bgr(frame, roi_xywh=roi)
    white_a, white_b = detection.white_points_px
    black = detection.black_gate_px

    cv2.rectangle(frame, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (255, 0, 0), 2)
    for white in detection.white_points_px:
        cv2.circle(frame, (int(white.x), int(white.y)), 9, (255, 255, 255), -1)
        cv2.circle(frame, (int(white.x), int(white.y)), 9, (0, 0, 0), 2)
    cv2.circle(frame, (int(black.x), int(black.y)), 9, (0, 0, 0), -1)
    cv2.line(
        frame,
        (int(white_a.x), int(white_a.y)),
        (int(white_b.x), int(white_b.y)),
        (255, 255, 255),
        3,
    )
    cv2.line(
        frame,
        (int((white_a.x + white_b.x) / 2), int((white_a.y + white_b.y) / 2)),
        (int(black.x), int(black.y)),
        (0, 0, 255),
        3,
    )
    cv2.imwrite(args.image_out, frame)

    print(
        "TWO_WHITE_BLACK",
        f"white_a=({white_a.x:.1f},{white_a.y:.1f})",
        f"white_b=({white_b.x:.1f},{white_b.y:.1f})",
        f"black_gate=({black.x:.1f},{black.y:.1f})",
    )
    print(f"Wrote {args.image_out}")


if __name__ == "__main__":
    main()
