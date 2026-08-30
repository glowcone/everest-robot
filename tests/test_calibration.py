import math

import numpy as np
import pytest

from everest_robot.calibration import (
    compute_homography,
    load_homography,
    pixel_to_table,
    save_homography,
)

# A known similarity transform: rotate 10 degrees, scale 0.001 m/px, translate.
ANGLE = math.radians(10.0)
SCALE = 0.001
TX, TY = 0.15, -0.05


def known_transform(px: float, py: float) -> tuple[float, float]:
    x = SCALE * (math.cos(ANGLE) * px - math.sin(ANGLE) * py) + TX
    y = SCALE * (math.sin(ANGLE) * px + math.cos(ANGLE) * py) + TY
    return x, y


PIXELS = [(50.0, 40.0), (600.0, 60.0), (580.0, 420.0), (70.0, 400.0), (320.0, 240.0)]
TABLE = [known_transform(px, py) for px, py in PIXELS]


def test_homography_round_trips_known_transform() -> None:
    homography = compute_homography(PIXELS, TABLE)

    for (px, py), (expected_x, expected_y) in zip(PIXELS, TABLE, strict=True):
        x, y = pixel_to_table(homography, px, py)
        assert abs(x - expected_x) < 1e-6
        assert abs(y - expected_y) < 1e-6

    # A point that was not part of the fit.
    x, y = pixel_to_table(homography, 200.0, 150.0)
    expected_x, expected_y = known_transform(200.0, 150.0)
    assert abs(x - expected_x) < 1e-6
    assert abs(y - expected_y) < 1e-6


def test_requires_four_correspondences() -> None:
    with pytest.raises(ValueError):
        compute_homography(PIXELS[:3], TABLE[:3])


def test_save_and_load_homography(tmp_path) -> None:
    homography = compute_homography(PIXELS, TABLE)
    path = tmp_path / "homography.json"

    save_homography(homography, path)
    loaded = load_homography(path)

    assert np.allclose(loaded, homography)


def test_load_rejects_wrong_shape(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"homography": [[1, 0], [0, 1]]}')

    with pytest.raises(ValueError):
        load_homography(path)
