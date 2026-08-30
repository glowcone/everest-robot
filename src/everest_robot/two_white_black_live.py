"""Live two-white-tape plus black-gate detection viewer."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

from everest_robot.pickup import MarkerPickupPlanner, load_pickup_config
from everest_robot.vision import detect_two_white_tapes_and_black_gate_bgr


def _draw_detection(frame, roi, detection, cv2) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continuously detect two white tapes plus the black gate."
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"), required=True)
    parser.add_argument(
        "--config",
        help="Optional pickup_config.json for live robot pregrasp printout.",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Print detections without opening a window.",
    )
    parser.add_argument("--print-every-s", type=float, default=0.5)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Exit after this many frames; useful for automated camera checks.",
    )
    parser.add_argument(
        "--image-out",
        help="Optional path to keep updating with the latest annotated frame.",
    )
    args = parser.parse_args()

    import cv2  # type: ignore[import-not-found]

    planner = None
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            planner = MarkerPickupPlanner(load_pickup_config(config_path))
        else:
            raise FileNotFoundError(config_path)

    roi = tuple(args.roi)
    cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    last_print = 0.0
    frames_read = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Could not read from camera index {args.camera}")
            frames_read += 1

            try:
                detection = detect_two_white_tapes_and_black_gate_bgr(frame, roi_xywh=roi)
                _draw_detection(frame, roi, detection, cv2)
                white_a, white_b = detection.white_points_px
                now = time.monotonic()
                if now - last_print >= args.print_every_s:
                    last_print = now
                    print(
                        "POINTS",
                        f"white_a=({white_a.x:.1f},{white_a.y:.1f})",
                        f"white_b=({white_b.x:.1f},{white_b.y:.1f})",
                        f"black_gate=({detection.black_gate_px.x:.1f},"
                        f"{detection.black_gate_px.y:.1f})",
                    )
                    if planner is not None:
                        plan = planner.plan_from_axis_points_and_side_point(
                            axis_a_px=white_a,
                            axis_b_px=white_b,
                            side_px=detection.black_gate_px,
                        )
                        print(
                            "PREGRASP",
                            f"x={plan.pregrasp.x:.4f}",
                            f"y={plan.pregrasp.y:.4f}",
                            f"z={plan.pregrasp.z:.4f}",
                            f"yaw_deg={math.degrees(plan.pregrasp.yaw):.1f}",
                        )
            except RuntimeError as exc:
                cv2.rectangle(
                    frame,
                    (roi[0], roi[1]),
                    (roi[0] + roi[2], roi[1] + roi[3]),
                    (0, 0, 255),
                    2,
                )
                now = time.monotonic()
                if now - last_print >= args.print_every_s:
                    last_print = now
                    print(f"MISS {exc}")

            if args.image_out:
                cv2.imwrite(args.image_out, frame)

            if not args.no_window:
                cv2.imshow("two-white-black detection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in {ord("q"), 27}:
                    break
            if args.max_frames is not None and frames_read >= args.max_frames:
                break
    finally:
        cap.release()
        if not args.no_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
