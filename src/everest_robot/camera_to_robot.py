"""Convert fixed-camera pixel coordinates to robot-table X/Y coordinates."""

from __future__ import annotations

import argparse
from pathlib import Path

from everest_robot.pickup import MarkerPickupPlanner, Point2, load_pickup_config, pixel_to_robot


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve camera pixels into robot-base X/Y using pickup calibration."
    )
    parser.add_argument("--config", default="pickup_config.json")
    parser.add_argument(
        "--point",
        nargs=2,
        type=float,
        action="append",
        required=True,
        metavar=("U", "V"),
        help="Camera pixel to convert; repeat --point to convert more than one.",
    )
    parser.add_argument(
        "--show-matrix",
        action="store_true",
        help="Print the solved 3x3 camera-to-robot homography first.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    homography = MarkerPickupPlanner(load_pickup_config(config_path)).homography
    if args.show_matrix:
        print("CAMERA_TO_ROBOT_MATRIX")
        for row in homography:
            print(" ".join(f"{value:.12g}" for value in row))

    for u, v in args.point:
        robot = pixel_to_robot(homography, Point2(u, v))
        print(f"CAMERA u={u:.3f} v={v:.3f} -> ROBOT x={robot.x:.6f} y={robot.y:.6f}")


if __name__ == "__main__":
    main()
