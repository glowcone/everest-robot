"""Inspect Maker Arm Cartesian kinematics without commanding motion."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping, Sequence

import numpy as np

from everest_robot.robot.cartesian import dataset_joints_to_tool_transform
from everest_robot.robot.deployment import (
    build_kinematics,
    build_lease,
    build_port,
    joint_frame,
    load_parameters,
)
from everest_robot.robot.session import RobotSession


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert Maker Arm joints and tool poses using the configured URDF."
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)

    fk = subparsers.add_parser(
        "dataset-fk",
        help="convert one seven-joint LeRobot dataset frame into robot-base XYZ",
    )
    fk.add_argument("--joints-deg", nargs=7, type=float, required=True)

    ik = subparsers.add_parser(
        "ik", help="solve a Cartesian pose from a nearby seven-joint seed"
    )
    seed = ik.add_mutually_exclusive_group(required=True)
    seed.add_argument("--seed-rad", nargs=7, type=float)
    seed.add_argument("--seed-dataset-deg", nargs=7, type=float)
    ik.add_argument(
        "--xyz", nargs=3, type=float, required=True, metavar=("X", "Y", "Z")
    )
    ik.add_argument(
        "--rpy-deg",
        nargs=3,
        type=float,
        required=True,
        metavar=("ROLL", "PITCH", "YAW"),
    )
    ik.add_argument("--position-only", action="store_true")

    capture = subparsers.add_parser(
        "capture",
        help="read current encoders and emit one camera/robot calibration pair",
    )
    capture.add_argument("--camera-u", type=float, required=True)
    capture.add_argument("--camera-v", type=float, required=True)
    return parser


def _transform_json(transform: np.ndarray) -> dict[str, object]:
    return {
        "tool_xyz_m": [float(value) for value in transform[:3, 3]],
        "tool_transform": [
            [float(value) for value in row] for row in transform
        ],
    }


def _target_pose(xyz: Sequence[float], rpy_deg: Sequence[float]) -> np.ndarray:
    try:
        from maker_arm.kinematics import pose_matrix
    except ImportError as error:
        raise RuntimeError(
            "the installed maker-arm SDK does not provide kinematics; install the local "
            "SDK checkout containing SerialChainKinematics"
        ) from error
    return pose_matrix(xyz, [math.radians(value) for value in rpy_deg])


def main(environ: Mapping[str, str] | None = None) -> None:
    args = _parser().parse_args()
    environ = os.environ if environ is None else environ
    parameters = load_parameters(environ)
    frame = joint_frame(parameters)

    if args.operation == "dataset-fk":
        model, _limits = build_kinematics(parameters, environ)
        transform = dataset_joints_to_tool_transform(model, frame, args.joints_deg)
        document = {
            "dataset_joint_names": list(frame.joint_names),
            "dataset_joints_deg": args.joints_deg,
            "maker_joints_rad": list(frame.to_radians(args.joints_deg)),
            **_transform_json(transform),
        }
        print(json.dumps(document, indent=2))
        return

    if args.operation == "ik":
        model, limits = build_kinematics(parameters, environ)
        seed = (
            tuple(args.seed_rad)
            if args.seed_rad is not None
            else frame.to_radians(args.seed_dataset_deg)
        )
        result = model.inverse(
            _target_pose(args.xyz, args.rpy_deg),
            seed[:-1],
            [lower for lower, _upper in limits],
            [upper for _lower, upper in limits],
            orientation_weight=0.0 if args.position_only else 0.1,
        )
        solved_full = (*result.joints_rad, seed[-1])
        document = {
            "converged": result.converged,
            "iterations": result.iterations,
            "maker_joints_rad": list(solved_full),
            "dataset_joints_deg": list(frame.to_degrees(solved_full)),
            "position_error_m": result.position_error_m,
            "orientation_error_rad": result.orientation_error_rad,
        }
        print(json.dumps(document, indent=2))
        if not result.converged:
            raise SystemExit(2)
        return

    port = build_port(parameters, environ)
    with RobotSession(
        port,
        parameters,
        lease=build_lease(parameters, environ),
    ):
        state = port.read_fresh_state()
        transform = np.asarray(port.forward_kinematics(state.positions), dtype=float)
    document = {
        "image_point_px": [args.camera_u, args.camera_v],
        "robot_point_m": [float(transform[0, 3]), float(transform[1, 3])],
        "joint_names": list(state.names),
        "maker_joints_rad": list(state.positions),
        **_transform_json(transform),
    }
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
