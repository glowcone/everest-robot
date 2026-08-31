#!/usr/bin/env python3
"""Live camera preview with the carabiner detection overlaid.

Read-only: opens a camera, runs the detector per frame, draws the result.
Nothing here commands the robot.

    ./tools/preview.py                 # wrist camera
    ./tools/preview.py -c side
    ./tools/preview.py -c 2            # raw index
    ./tools/preview.py --video FILE    # replay a recording instead

    ./tools/preview.py --roi 0 0 990 900   # only look at the bench

Keys:  q/Esc quit   m cycle view (overlay / mask / score)   s save a snapshot
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from everest_robot.carabiner_detect import NotFound, chroma_mask, detect, draw, in_roi, teal_score

# Camera indices live in the recording app's robot config, so the names here
# stay in sync with whatever the arm was recorded with.
ROBOT_CONFIG = Path.home() / ".cache/huggingface/lerobot/robots/hack2.json"
FALLBACK = {"wrist": 1, "side": 0}


def resolve_camera(name: str) -> tuple[int, str]:
    """Map a camera name to a device index, preferring the robot config."""
    if name.isdigit():
        return int(name), f"index {name}"
    try:
        cams = json.loads(ROBOT_CONFIG.read_text())["cameras"]
        for c in cams:
            if c["name"] == name:
                return int(c["camera_index"]), f"{name} (index {c['camera_index']}, from config)"
        known = ", ".join(c["name"] for c in cams)
        sys.exit(f"no camera named {name!r} in {ROBOT_CONFIG}. known: {known}")
    except FileNotFoundError:
        if name not in FALLBACK:
            sys.exit(f"no config at {ROBOT_CONFIG} and no fallback for {name!r}")
        return FALLBACK[name], f"{name} (index {FALLBACK[name]}, fallback)"


def open_source(args) -> tuple[cv2.VideoCapture, str]:
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            sys.exit(f"cannot open video {args.video}")
        return cap, Path(args.video).name

    idx, label = resolve_camera(args.camera)
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        sys.exit(
            f"cannot open camera {label}.\n"
            "On macOS the terminal app needs Camera permission in\n"
            "System Settings > Privacy & Security > Camera."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap, label


VIEWS = ("overlay", "mask", "score")


def render(frame: np.ndarray, view: str, roi=None) -> tuple[np.ndarray, str]:
    """Draw the requested view and return it with a one-line status."""
    if view == "score":
        d = in_roi(teal_score(frame), roi)
        vis = cv2.applyColorMap(
            np.clip(d / 8.0 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO
        )
        return vis, "score (sigma, 0-8)"

    if view == "mask":
        masked = chroma_mask(frame, score=in_roi(teal_score(frame), roi))
        return cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR), "mask"

    try:
        t = detect(frame, roi)
    except NotFound as e:
        vis = frame.copy()
        cv2.putText(vis, f"no detection: {e}", (8, 44), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 1, cv2.LINE_AA)
        return vis, "no detection"

    vis = draw(frame, t)
    cx, cy = t.aperture
    cv2.drawMarker(vis, (int(cx), int(cy)), (0, 255, 255), cv2.MARKER_CROSS, 18, 1)
    return vis, f"centre ({cx:.0f}, {cy:.0f})  area {t.area:.0f}px"


def draw_roi(vis: np.ndarray, roi) -> None:
    """Outline the search region, so its numbers can be checked against the scene."""
    if roi is None:
        return
    x, y, w, h = roi
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 255), 1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", "--camera", default="wrist",
                   help="camera name (wrist, side) or a raw device index")
    p.add_argument("--video", help="replay a video file instead of a live camera")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--view", default="overlay", choices=VIEWS)
    p.add_argument(
        "--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
        help="only look inside this rectangle. Use it when the bench is not the only thing "
             "in view: the detector's thresholds are relative to the frame's own background, "
             "so a screen or a plant across the room can look like a small carabiner. The "
             "numbers you settle on go in EVEREST_WRIST_ROI",
    )
    args = p.parse_args()

    cap, label = open_source(args)
    view = args.view
    win = f"carabiner - {label}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print(f"{label}   keys: q quit, m cycle view, s snapshot")

    shots = Path("out/snapshots")
    smoothed_ms = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if args.video:  # loop the file
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                print("camera read failed", file=sys.stderr)
                break

            t0 = time.perf_counter()
            roi = tuple(args.roi) if args.roi else None
            vis, status = render(frame, view, roi)
            draw_roi(vis, roi)
            ms = (time.perf_counter() - t0) * 1000
            smoothed_ms = ms if smoothed_ms is None else 0.9 * smoothed_ms + 0.1 * ms

            cv2.rectangle(vis, (0, 0), (vis.shape[1], 26), (0, 0, 0), -1)
            cv2.putText(vis, f"[{view}] {status}   {smoothed_ms:.0f} ms", (8, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow(win, vis)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("m"):
                view = VIEWS[(VIEWS.index(view) + 1) % len(VIEWS)]
            if k == ord("s"):
                shots.mkdir(parents=True, exist_ok=True)
                path = shots / f"{int(time.time())}.png"
                cv2.imwrite(str(path), vis)
                print(f"saved {path}")
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
