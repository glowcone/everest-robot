"""Conversion helpers between LeRobot dataset joints and Maker Arm Cartesian poses."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from everest_robot.robot.lerobot_bridge import JointFrame


def dataset_joints_to_tool_transform(
    kinematics: Any,
    frame: JointFrame,
    dataset_joints_deg: Sequence[float],
) -> np.ndarray:
    """Convert one named LeRobot joint frame into a base-to-tool transform."""

    if len(dataset_joints_deg) != len(frame.joint_names):
        raise ValueError(
            f"expected {len(frame.joint_names)} dataset joints, got {len(dataset_joints_deg)}"
        )
    maker_joints_rad = frame.to_radians(dataset_joints_deg)
    arm_joint_count = len(frame.joint_names) - 1
    if tuple(kinematics.joint_names) != tuple(frame.joint_names[:arm_joint_count]):
        raise ValueError("kinematic model joint order does not match the dataset frame")
    transform = np.asarray(kinematics.forward(maker_joints_rad[:arm_joint_count]), dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("forward kinematics returned an invalid transform")
    return transform
