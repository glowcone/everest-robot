"""Overhead-camera pixel to robot-base table-plane mapping via homography.

The carabiner lies flat on the table, so a single planar homography maps overhead-camera
pixels to (x, y) in the robot base frame. Correspondences are measured once during
hardware bring-up and stored as JSON:

    {"points": [{"pixel": [px, py], "table": [x, y]}, ...]}

Compute and save the homography with:

    uv run python -m everest_robot.calibration correspondences.json homography.json
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np


def compute_homography(
    pixel_points: list[tuple[float, float]],
    table_points: list[tuple[float, float]],
) -> np.ndarray:
    """Fit a pixel->table homography from at least four correspondences."""

    if len(pixel_points) < 4 or len(pixel_points) != len(table_points):
        raise ValueError("homography needs at least 4 matched pixel/table points")
    homography, _ = cv2.findHomography(
        np.array(pixel_points, dtype=np.float64),
        np.array(table_points, dtype=np.float64),
    )
    if homography is None:
        raise ValueError("homography estimation failed; check the correspondences")
    return homography


def pixel_to_table(homography: np.ndarray, pixel_x: float, pixel_y: float) -> tuple[float, float]:
    """Project an overhead-camera pixel onto the table plane in robot-base meters."""

    point = np.array([pixel_x, pixel_y, 1.0], dtype=np.float64)
    x, y, w = homography @ point
    if abs(w) < 1e-9:
        raise ValueError("degenerate homography projection")
    return float(x / w), float(y / w)


def save_homography(homography: np.ndarray, path: str | Path) -> None:
    Path(path).write_text(json.dumps({"homography": homography.tolist()}, indent=2))


def load_homography(path: str | Path) -> np.ndarray:
    data = json.loads(Path(path).read_text())
    homography = np.array(data["homography"], dtype=np.float64)
    if homography.shape != (3, 3):
        raise ValueError(f"expected a 3x3 homography in {path}")
    return homography


def _reprojection_error(
    homography: np.ndarray,
    pixel_points: list[tuple[float, float]],
    table_points: list[tuple[float, float]],
) -> float:
    errors = [
        float(np.hypot(*(np.array(pixel_to_table(homography, px, py)) - np.array(table))))
        for (px, py), table in zip(pixel_points, table_points, strict=True)
    ]
    return max(errors)


def main() -> None:
    if len(sys.argv) != 3:
        print(
            "usage: python -m everest_robot.calibration <correspondences.json> <homography.json>",
            file=sys.stderr,
        )
        raise SystemExit(2)
    correspondences = json.loads(Path(sys.argv[1]).read_text())
    pixel_points = [tuple(p["pixel"]) for p in correspondences["points"]]
    table_points = [tuple(p["table"]) for p in correspondences["points"]]
    homography = compute_homography(pixel_points, table_points)
    save_homography(homography, sys.argv[2])
    error = _reprojection_error(homography, pixel_points, table_points)
    print(f"saved homography to {sys.argv[2]} (max reprojection error {error:.4f} m)")


if __name__ == "__main__":
    main()
