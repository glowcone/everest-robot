import math

import pytest

from everest_robot.kinematics import ArmGeometry, forward_kinematics, solve_ik

GEOMETRY = ArmGeometry(
    base_height=0.10,
    upper_link=0.12,
    forearm_link=0.12,
    gripper_length=0.09,
)


def test_ik_fk_round_trip_over_workspace_grid() -> None:
    for radial in (0.08, 0.12, 0.16, 0.20):
        for yaw_deg in (-90, -30, 0, 45, 120):
            for z in (0.01, 0.05, 0.10):
                x = radial * math.cos(math.radians(yaw_deg))
                y = radial * math.sin(math.radians(yaw_deg))
                try:
                    joints = solve_ik(x, y, z, GEOMETRY)
                except ValueError:
                    continue  # not every grid point is reachable
                fx, fy, fz = forward_kinematics(joints, GEOMETRY)
                assert math.hypot(fx - x, fy - y) < 1e-3
                assert abs(fz - z) < 1e-3


def test_base_yaw_points_at_target() -> None:
    joints = solve_ik(0.10, 0.10, 0.02, GEOMETRY)
    assert abs(joints.base_yaw - math.pi / 4) < 1e-9


def test_unreachable_target_raises() -> None:
    with pytest.raises(ValueError, match="out of reach"):
        solve_ik(0.5, 0.0, 0.02, GEOMETRY)


def test_target_on_base_axis_raises() -> None:
    with pytest.raises(ValueError, match="base axis"):
        solve_ik(0.0, 0.0, 0.05, GEOMETRY)


def test_joint_limit_violation_raises() -> None:
    limited = ArmGeometry(
        base_height=GEOMETRY.base_height,
        upper_link=GEOMETRY.upper_link,
        forearm_link=GEOMETRY.forearm_link,
        gripper_length=GEOMETRY.gripper_length,
        limits=type(GEOMETRY.limits)(base_yaw=(-0.1, 0.1)),
    )
    with pytest.raises(ValueError, match="base_yaw"):
        solve_ik(0.0, 0.15, 0.02, limited)
