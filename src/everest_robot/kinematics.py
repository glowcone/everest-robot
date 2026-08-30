"""Analytic inverse kinematics for the MakerMod arm.

Assumed structure (typical 4-DOF hobby arm): a base yaw joint, shoulder and elbow pitch
joints forming a planar 2-link chain, and a wrist pitch joint. The wrist is constrained so
the gripper always points straight down, which is what a top-down table grasp needs and
keeps the IK closed-form.

Angle conventions (radians):
- base_yaw: 0 along +x, positive counter-clockwise (atan2(y, x))
- shoulder: 0 horizontal, positive raises the upper arm
- elbow: 0 straight (aligned with the upper arm), positive bends downward
- wrist: pitch relative to the forearm chosen so the gripper is vertical

The gripper tip pose (x, y, z) is in the robot base frame with z up and the shoulder
joint at height `base_height`.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class JointLimits:
    """Symmetric limits per joint in radians (min, max)."""

    base_yaw: tuple[float, float] = (-math.pi, math.pi)
    shoulder: tuple[float, float] = (0.0, math.pi)
    elbow: tuple[float, float] = (-math.pi * 0.75, math.pi * 0.75)
    wrist: tuple[float, float] = (-math.pi, math.pi)


@dataclass(frozen=True)
class ArmGeometry:
    """MakerMod link dimensions in meters."""

    base_height: float
    upper_link: float
    forearm_link: float
    gripper_length: float
    limits: JointLimits = JointLimits()


@dataclass(frozen=True)
class JointAngles:
    """A joint-space target for the MakerMod SDK, radians."""

    base_yaw: float
    shoulder: float
    elbow: float
    wrist: float


def solve_ik(x: float, y: float, z: float, geometry: ArmGeometry) -> JointAngles:
    """Solve joints so the downward-pointing gripper tip reaches (x, y, z).

    Raises ValueError when the target is out of reach or violates joint limits.
    """

    radial = math.hypot(x, y)
    if radial < 1e-9:
        raise ValueError("target is on the base axis; base yaw is undefined")
    base_yaw = math.atan2(y, x)

    # With a vertical gripper, the wrist joint sits directly above the tip.
    wrist_radial = radial
    wrist_height = z + geometry.gripper_length - geometry.base_height

    reach = math.hypot(wrist_radial, wrist_height)
    max_reach = geometry.upper_link + geometry.forearm_link
    min_reach = abs(geometry.upper_link - geometry.forearm_link)
    if reach > max_reach or reach < min_reach:
        raise ValueError(
            f"target ({x:.3f}, {y:.3f}, {z:.3f}) is out of reach: "
            f"wrist distance {reach:.3f} m outside [{min_reach:.3f}, {max_reach:.3f}]"
        )

    # Planar 2-link solution, elbow-up.
    cos_elbow_interior = (
        (reach**2 - geometry.upper_link**2 - geometry.forearm_link**2)
        / (2.0 * geometry.upper_link * geometry.forearm_link)
    )
    cos_elbow_interior = max(-1.0, min(1.0, cos_elbow_interior))
    elbow = math.acos(cos_elbow_interior)

    shoulder = math.atan2(wrist_height, wrist_radial) + math.atan2(
        geometry.forearm_link * math.sin(elbow),
        geometry.upper_link + geometry.forearm_link * math.cos(elbow),
    )
    # Wrist pitch that keeps the gripper pointing straight down.
    wrist = -math.pi / 2.0 - (shoulder - elbow)

    joints = JointAngles(base_yaw=base_yaw, shoulder=shoulder, elbow=elbow, wrist=wrist)
    _validate_limits(joints, geometry.limits)
    return joints


def forward_kinematics(joints: JointAngles, geometry: ArmGeometry) -> tuple[float, float, float]:
    """Gripper tip position for a joint-space pose (assumes the vertical-wrist convention)."""

    wrist_radial = geometry.upper_link * math.cos(joints.shoulder) + geometry.forearm_link * (
        math.cos(joints.shoulder - joints.elbow)
    )
    wrist_height = geometry.upper_link * math.sin(joints.shoulder) + geometry.forearm_link * (
        math.sin(joints.shoulder - joints.elbow)
    )
    # The gripper hangs straight down from the wrist by construction.
    tip_z = geometry.base_height + wrist_height - geometry.gripper_length
    return (
        wrist_radial * math.cos(joints.base_yaw),
        wrist_radial * math.sin(joints.base_yaw),
        tip_z,
    )


def _validate_limits(joints: JointAngles, limits: JointLimits) -> None:
    for name, value, (low, high) in (
        ("base_yaw", joints.base_yaw, limits.base_yaw),
        ("shoulder", joints.shoulder, limits.shoulder),
        ("elbow", joints.elbow, limits.elbow),
        ("wrist", joints.wrist, limits.wrist),
    ):
        if not low <= value <= high:
            raise ValueError(
                f"{name} solution {value:.3f} rad violates limits [{low:.3f}, {high:.3f}]"
            )
