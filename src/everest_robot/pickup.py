"""Marker-based carabiner pickup planning.

This module intentionally stops at target generation. The hardware adapter should execute
the returned pregrasp, grasp, lift, and canonical poses using the robot's Cartesian API.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Point2:
    x: float
    y: float


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class Pose3D:
    x: float
    y: float
    z: float
    yaw: float


@dataclass(frozen=True)
class PickupPlan:
    carabiner: Pose2D
    pregrasp: Pose3D
    grasp: Pose3D
    lift: Pose3D
    canonical: Pose3D | None


@dataclass(frozen=True)
class PickupConfig:
    image_points_px: tuple[Point2, Point2, Point2, Point2]
    robot_points_m: tuple[Point2, Point2, Point2, Point2]
    carabiner_center_from_marker_midpoint_m: Point2
    grasp_offset_from_carabiner_m: Point2
    grasp_yaw_offset_rad: float
    z_pregrasp_m: float
    z_grasp_m: float
    z_lift_m: float
    canonical_pose: Pose3D | None


def normalize_angle(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def rotation2(theta_rad: float) -> np.ndarray:
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)
    return np.array([[c, -s], [s, c]], dtype=float)


def find_homography(image_points: Iterable[Point2], robot_points: Iterable[Point2]) -> np.ndarray:
    """Return H such that [x, y, 1] ~= H @ [u, v, 1]."""

    src = list(image_points)
    dst = list(robot_points)
    if len(src) != 4 or len(dst) != 4:
        raise ValueError("Planar homography calibration requires exactly four point pairs.")

    rows: list[list[float]] = []
    for image_point, robot_point in zip(src, dst, strict=True):
        u, v = image_point.x, image_point.y
        x, y = robot_point.x, robot_point.y
        rows.append([-u, -v, -1.0, 0.0, 0.0, 0.0, x * u, x * v, x])
        rows.append([0.0, 0.0, 0.0, -u, -v, -1.0, y * u, y * v, y])

    _, _, vh = np.linalg.svd(np.array(rows, dtype=float))
    h = vh[-1].reshape(3, 3)
    return h / h[2, 2]


def pixel_to_robot(homography: np.ndarray, pixel: Point2) -> Point2:
    q = homography @ np.array([pixel.x, pixel.y, 1.0], dtype=float)
    if abs(q[2]) < 1e-9:
        raise ValueError("Homography produced an invalid point with near-zero scale.")
    q = q / q[2]
    return Point2(float(q[0]), float(q[1]))


def carabiner_pose_from_markers(
    homography: np.ndarray,
    red_pixel: Point2,
    blue_pixel: Point2,
    center_from_marker_midpoint_m: Point2,
) -> Pose2D:
    red_robot = pixel_to_robot(homography, red_pixel)
    blue_robot = pixel_to_robot(homography, blue_pixel)

    midpoint = np.array(
        [
            (red_robot.x + blue_robot.x) / 2.0,
            (red_robot.y + blue_robot.y) / 2.0,
        ],
        dtype=float,
    )
    theta = math.atan2(blue_robot.y - red_robot.y, blue_robot.x - red_robot.x)
    center_offset = rotation2(theta) @ np.array(
        [center_from_marker_midpoint_m.x, center_from_marker_midpoint_m.y],
        dtype=float,
    )
    center = midpoint + center_offset

    return Pose2D(float(center[0]), float(center[1]), normalize_angle(theta))


def carabiner_pose_from_bar_endpoints(
    homography: np.ndarray,
    endpoint_a_px: Point2,
    endpoint_b_px: Point2,
    center_from_bar_midpoint_m: Point2,
) -> Pose2D:
    endpoint_a_robot = pixel_to_robot(homography, endpoint_a_px)
    endpoint_b_robot = pixel_to_robot(homography, endpoint_b_px)

    midpoint = np.array(
        [
            (endpoint_a_robot.x + endpoint_b_robot.x) / 2.0,
            (endpoint_a_robot.y + endpoint_b_robot.y) / 2.0,
        ],
        dtype=float,
    )
    theta = math.atan2(
        endpoint_b_robot.y - endpoint_a_robot.y,
        endpoint_b_robot.x - endpoint_a_robot.x,
    )
    center_offset = rotation2(theta) @ np.array(
        [center_from_bar_midpoint_m.x, center_from_bar_midpoint_m.y],
        dtype=float,
    )
    center = midpoint + center_offset

    return Pose2D(float(center[0]), float(center[1]), normalize_angle(theta))


def carabiner_pose_from_axis_points_and_side_point(
    homography: np.ndarray,
    axis_a_px: Point2,
    axis_b_px: Point2,
    side_px: Point2,
    center_from_axis_midpoint_m: Point2,
) -> Pose2D:
    """Build a carabiner pose from two same-color axis points plus one side point.

    The two axis points define the local x-axis but not its sign. The side point resolves
    that ambiguity by requiring it to land on the positive local-y side of the directed
    x-axis. This is useful for two identical white tapes plus the black gate.
    """

    axis_a_robot = pixel_to_robot(homography, axis_a_px)
    axis_b_robot = pixel_to_robot(homography, axis_b_px)
    side_robot = pixel_to_robot(homography, side_px)

    a = np.array([axis_a_robot.x, axis_a_robot.y], dtype=float)
    b = np.array([axis_b_robot.x, axis_b_robot.y], dtype=float)
    side = np.array([side_robot.x, side_robot.y], dtype=float)
    midpoint = (a + b) / 2.0

    axis = b - a
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        raise ValueError("Axis points are too close together to define orientation.")
    axis = axis / norm

    left_normal = np.array([-axis[1], axis[0]], dtype=float)
    if float(np.dot(left_normal, side - midpoint)) < 0.0:
        axis = -axis

    theta = math.atan2(axis[1], axis[0])
    center_offset = rotation2(theta) @ np.array(
        [center_from_axis_midpoint_m.x, center_from_axis_midpoint_m.y],
        dtype=float,
    )
    center = midpoint + center_offset

    return Pose2D(float(center[0]), float(center[1]), normalize_angle(theta))


def compute_grasp_pose(
    carabiner_pose: Pose2D,
    grasp_offset_m: Point2,
    grasp_yaw_offset_rad: float,
) -> Pose2D:
    offset = rotation2(carabiner_pose.yaw) @ np.array(
        [grasp_offset_m.x, grasp_offset_m.y],
        dtype=float,
    )
    return Pose2D(
        x=float(carabiner_pose.x + offset[0]),
        y=float(carabiner_pose.y + offset[1]),
        yaw=normalize_angle(carabiner_pose.yaw + grasp_yaw_offset_rad),
    )


@dataclass(frozen=True)
class MarkerPickupPlanner:
    config: PickupConfig

    @property
    def homography(self) -> np.ndarray:
        return find_homography(self.config.image_points_px, self.config.robot_points_m)

    def plan_from_marker_pixels(self, red_pixel: Point2, blue_pixel: Point2) -> PickupPlan:
        carabiner = carabiner_pose_from_markers(
            self.homography,
            red_pixel,
            blue_pixel,
            self.config.carabiner_center_from_marker_midpoint_m,
        )
        grasp_2d = compute_grasp_pose(
            carabiner,
            self.config.grasp_offset_from_carabiner_m,
            self.config.grasp_yaw_offset_rad,
        )
        return PickupPlan(
            carabiner=carabiner,
            pregrasp=Pose3D(grasp_2d.x, grasp_2d.y, self.config.z_pregrasp_m, grasp_2d.yaw),
            grasp=Pose3D(grasp_2d.x, grasp_2d.y, self.config.z_grasp_m, grasp_2d.yaw),
            lift=Pose3D(grasp_2d.x, grasp_2d.y, self.config.z_lift_m, grasp_2d.yaw),
            canonical=self.config.canonical_pose,
        )

    def plan_from_bar_endpoints(
        self,
        endpoint_a_px: Point2,
        endpoint_b_px: Point2,
    ) -> PickupPlan:
        carabiner = carabiner_pose_from_bar_endpoints(
            self.homography,
            endpoint_a_px,
            endpoint_b_px,
            self.config.carabiner_center_from_marker_midpoint_m,
        )
        grasp_2d = compute_grasp_pose(
            carabiner,
            self.config.grasp_offset_from_carabiner_m,
            self.config.grasp_yaw_offset_rad,
        )
        return PickupPlan(
            carabiner=carabiner,
            pregrasp=Pose3D(grasp_2d.x, grasp_2d.y, self.config.z_pregrasp_m, grasp_2d.yaw),
            grasp=Pose3D(grasp_2d.x, grasp_2d.y, self.config.z_grasp_m, grasp_2d.yaw),
            lift=Pose3D(grasp_2d.x, grasp_2d.y, self.config.z_lift_m, grasp_2d.yaw),
            canonical=self.config.canonical_pose,
        )

    def plan_from_axis_points_and_side_point(
        self,
        axis_a_px: Point2,
        axis_b_px: Point2,
        side_px: Point2,
    ) -> PickupPlan:
        carabiner = carabiner_pose_from_axis_points_and_side_point(
            self.homography,
            axis_a_px,
            axis_b_px,
            side_px,
            self.config.carabiner_center_from_marker_midpoint_m,
        )
        grasp_2d = compute_grasp_pose(
            carabiner,
            self.config.grasp_offset_from_carabiner_m,
            self.config.grasp_yaw_offset_rad,
        )
        return PickupPlan(
            carabiner=carabiner,
            pregrasp=Pose3D(grasp_2d.x, grasp_2d.y, self.config.z_pregrasp_m, grasp_2d.yaw),
            grasp=Pose3D(grasp_2d.x, grasp_2d.y, self.config.z_grasp_m, grasp_2d.yaw),
            lift=Pose3D(grasp_2d.x, grasp_2d.y, self.config.z_lift_m, grasp_2d.yaw),
            canonical=self.config.canonical_pose,
        )


def _point_from_json(value: list[float]) -> Point2:
    return Point2(float(value[0]), float(value[1]))


def _pose3d_from_json(value: dict[str, float] | None) -> Pose3D | None:
    if value is None:
        return None
    return Pose3D(
        x=float(value["x"]),
        y=float(value["y"]),
        z=float(value["z"]),
        yaw=math.radians(float(value["yaw_deg"])),
    )


def load_pickup_config(path: str | Path) -> PickupConfig:
    raw = json.loads(Path(path).read_text())
    return PickupConfig(
        image_points_px=tuple(_point_from_json(p) for p in raw["image_points_px"]),
        robot_points_m=tuple(_point_from_json(p) for p in raw["robot_points_m"]),
        carabiner_center_from_marker_midpoint_m=_point_from_json(
            raw.get("carabiner_center_from_marker_midpoint_m", [0.0, 0.0])
        ),
        grasp_offset_from_carabiner_m=_point_from_json(raw["grasp_offset_from_carabiner_m"]),
        grasp_yaw_offset_rad=math.radians(float(raw["grasp_yaw_offset_deg"])),
        z_pregrasp_m=float(raw["z_pregrasp_m"]),
        z_grasp_m=float(raw["z_grasp_m"]),
        z_lift_m=float(raw["z_lift_m"]),
        canonical_pose=_pose3d_from_json(raw.get("canonical_pose")),
    )
