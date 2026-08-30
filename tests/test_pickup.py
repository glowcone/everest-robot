import math

import cv2
import numpy as np

from everest_robot.pickup import (
    MarkerPickupPlanner,
    PickupConfig,
    Point2,
    Pose3D,
    carabiner_pose_from_axis_points_and_side_point,
    carabiner_pose_from_bar_endpoints,
    compute_grasp_pose,
    find_homography,
    pixel_to_robot,
)
from everest_robot.vision import (
    detect_straight_bar_bgr,
    detect_two_white_tapes_and_black_gate_bgr,
    detect_white_black_markers_bgr,
)


def test_homography_maps_calibration_corners() -> None:
    image_points = (
        Point2(120, 100),
        Point2(520, 100),
        Point2(520, 420),
        Point2(120, 420),
    )
    robot_points = (
        Point2(0.30, -0.10),
        Point2(0.30, 0.10),
        Point2(0.50, 0.10),
        Point2(0.50, -0.10),
    )

    h = find_homography(image_points, robot_points)

    for image_point, robot_point in zip(image_points, robot_points, strict=True):
        mapped = pixel_to_robot(h, image_point)
        assert mapped.x == pytest_approx(robot_point.x)
        assert mapped.y == pytest_approx(robot_point.y)


def test_homography_least_squares_maps_more_than_four_points() -> None:
    expected = np.array(
        [
            [0.0004, 0.00002, 0.12],
            [-0.00001, 0.0003, -0.18],
            [0.0000005, -0.0000003, 1.0],
        ]
    )
    image_points = tuple(
        Point2(u, v)
        for u, v in (
            (100, 100),
            (600, 100),
            (600, 500),
            (100, 500),
            (350, 180),
            (260, 390),
        )
    )
    robot_points = tuple(pixel_to_robot(expected, point) for point in image_points)

    solved = find_homography(image_points, robot_points)

    mapped = pixel_to_robot(solved, Point2(420, 310))
    expected_point = pixel_to_robot(expected, Point2(420, 310))
    assert mapped.x == pytest_approx(expected_point.x)
    assert mapped.y == pytest_approx(expected_point.y)


def test_homography_rejects_mismatched_or_collinear_calibration() -> None:
    import pytest

    with pytest.raises(ValueError, match="counts must match"):
        find_homography(
            [Point2(0, 0), Point2(1, 0), Point2(1, 1), Point2(0, 1)],
            [Point2(0, 0), Point2(1, 0), Point2(1, 1)],
        )

    with pytest.raises(ValueError, match="cannot all be collinear"):
        find_homography(
            [Point2(0, 0), Point2(1, 0), Point2(2, 0), Point2(3, 0)],
            [Point2(0, 0), Point2(1, 0), Point2(1, 1), Point2(0, 1)],
        )


def test_grasp_offset_rotates_with_carabiner_yaw() -> None:
    grasp = compute_grasp_pose(
        carabiner_pose=type("Pose", (), {"x": 0.40, "y": 0.0, "yaw": math.pi / 2})(),
        grasp_offset_m=Point2(0.0, -0.03),
        grasp_yaw_offset_rad=math.pi / 2,
    )

    assert grasp.x == pytest_approx(0.43)
    assert grasp.y == pytest_approx(0.0)
    assert abs(abs(grasp.yaw) - math.pi) < 1e-9


def test_marker_pickup_planner_returns_pregrasp_grasp_lift() -> None:
    config = PickupConfig(
        image_points_px=(
            Point2(120, 100),
            Point2(520, 100),
            Point2(520, 420),
            Point2(120, 420),
        ),
        robot_points_m=(
            Point2(0.30, -0.10),
            Point2(0.30, 0.10),
            Point2(0.50, 0.10),
            Point2(0.50, -0.10),
        ),
        carabiner_center_from_marker_midpoint_m=Point2(0.0, 0.0),
        grasp_offset_from_carabiner_m=Point2(0.0, -0.03),
        grasp_yaw_offset_rad=math.pi / 2,
        z_pregrasp_m=0.12,
        z_grasp_m=0.025,
        z_lift_m=0.14,
        canonical_pose=Pose3D(0.40, 0.0, 0.14, 0.0),
    )

    plan = MarkerPickupPlanner(config).plan_from_marker_pixels(
        red_pixel=Point2(300, 220),
        blue_pixel=Point2(340, 220),
    )

    assert plan.carabiner.x == pytest_approx(0.375)
    assert plan.carabiner.y == pytest_approx(0.0)
    assert math.degrees(plan.carabiner.yaw) == pytest_approx(90.0)
    assert plan.pregrasp.x == pytest_approx(0.405)
    assert plan.pregrasp.z == pytest_approx(0.12)
    assert plan.grasp.z == pytest_approx(0.025)
    assert plan.lift.z == pytest_approx(0.14)
    assert plan.canonical == Pose3D(0.40, 0.0, 0.14, 0.0)


def test_carabiner_pose_from_bar_endpoints_uses_robot_space_yaw() -> None:
    image_points = (
        Point2(120, 100),
        Point2(520, 100),
        Point2(520, 420),
        Point2(120, 420),
    )
    robot_points = (
        Point2(0.30, -0.10),
        Point2(0.30, 0.10),
        Point2(0.50, 0.10),
        Point2(0.50, -0.10),
    )
    h = find_homography(image_points, robot_points)

    pose = carabiner_pose_from_bar_endpoints(
        h,
        endpoint_a_px=Point2(300, 220),
        endpoint_b_px=Point2(340, 220),
        center_from_bar_midpoint_m=Point2(0.0, 0.0),
    )

    assert pose.x == pytest_approx(0.375)
    assert pose.y == pytest_approx(0.0)
    assert math.degrees(pose.yaw) == pytest_approx(90.0)


def test_axis_points_and_side_point_disambiguate_axis_direction() -> None:
    image_points = (
        Point2(120, 100),
        Point2(520, 100),
        Point2(520, 420),
        Point2(120, 420),
    )
    robot_points = (
        Point2(0.30, -0.10),
        Point2(0.30, 0.10),
        Point2(0.50, 0.10),
        Point2(0.50, -0.10),
    )
    h = find_homography(image_points, robot_points)

    pose = carabiner_pose_from_axis_points_and_side_point(
        h,
        axis_a_px=Point2(300, 220),
        axis_b_px=Point2(340, 220),
        side_px=Point2(320, 260),
        center_from_axis_midpoint_m=Point2(0.0, 0.0),
    )

    assert pose.x == pytest_approx(0.375)
    assert pose.y == pytest_approx(0.0)
    assert math.degrees(pose.yaw) == pytest_approx(-90.0)


def test_detect_straight_bar_finds_synthetic_interior_line() -> None:
    image = np.full((240, 320, 3), 255, dtype=np.uint8)
    cv2.line(image, (120, 180), (180, 60), (20, 20, 20), 8)

    detection = detect_straight_bar_bgr(
        image,
        roi_xywh=(80, 30, 160, 180),
        min_length_px=60,
    )

    assert detection.center_px.x == pytest_approx(150.0, abs=5.0)
    assert detection.center_px.y == pytest_approx(120.0, abs=5.0)
    assert detection.length_px > 120
    assert abs(abs(math.degrees(detection.yaw_px)) - 63.4) < 3.0


def test_detect_white_black_markers_finds_synthetic_tape_near_green_body() -> None:
    image = np.full((240, 320, 3), (95, 130, 160), dtype=np.uint8)
    cv2.ellipse(image, (160, 120), (45, 70), 20, 0, 360, (40, 140, 120), 10)
    cv2.rectangle(image, (175, 58), (205, 78), (245, 245, 245), -1)
    cv2.rectangle(image, (118, 165), (150, 185), (15, 15, 15), -1)

    white, black = detect_white_black_markers_bgr(
        image,
        roi_xywh=(80, 30, 160, 180),
        min_area_px=100,
    )

    assert white.center_px.x == pytest_approx(190.0, abs=4.0)
    assert white.center_px.y == pytest_approx(68.0, abs=4.0)
    assert black.center_px.x == pytest_approx(134.0, abs=4.0)
    assert black.center_px.y == pytest_approx(175.0, abs=4.0)


def test_detect_two_white_tapes_and_black_gate_finds_synthetic_points() -> None:
    image = np.full((240, 320, 3), (95, 130, 160), dtype=np.uint8)
    cv2.ellipse(image, (160, 120), (45, 70), 20, 0, 360, (40, 140, 120), 10)
    cv2.rectangle(image, (175, 58), (205, 78), (245, 245, 245), -1)
    cv2.rectangle(image, (115, 165), (145, 185), (245, 245, 245), -1)
    cv2.rectangle(image, (118, 105), (150, 125), (15, 15, 15), -1)

    detection = detect_two_white_tapes_and_black_gate_bgr(
        image,
        roi_xywh=(80, 30, 160, 180),
        min_white_area_px=100,
        min_black_area_px=100,
    )

    white_a, white_b = sorted(detection.white_points_px, key=lambda point: point.y)
    assert white_a.x == pytest_approx(190.0, abs=4.0)
    assert white_a.y == pytest_approx(68.0, abs=4.0)
    assert white_b.x == pytest_approx(130.0, abs=4.0)
    assert white_b.y == pytest_approx(175.0, abs=4.0)
    assert detection.black_gate_px.x == pytest_approx(134.0, abs=4.0)
    assert detection.black_gate_px.y == pytest_approx(115.0, abs=4.0)


def pytest_approx(value: float, *, abs: float = 1e-9):
    import pytest

    return pytest.approx(value, abs=abs)
