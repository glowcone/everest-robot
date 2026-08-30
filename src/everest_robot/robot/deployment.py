"""Building a robot session from deployment configuration.

Everything environment-specific is resolved here: which parameters file, which CAN
interface, which lease backend, which cameras. Keeping it in one place is what lets the
workflow, the adapter and the tests stay free of deployment detail, and it is why none of
these values live in ``config/maker_arm_v1.yaml``.

Environment:

* ``EVEREST_ROBOT_PARAMETERS`` -- path to the robot parameters YAML
  (default ``config/maker_arm_v1.yaml``).
* ``EVEREST_CAN_BACKEND`` -- ``socketcan`` (default) or ``slcan``.
* ``EVEREST_CAN_PORT`` -- interface name (``can0``) or serial port (``/dev/ttyACM0``).
* ``EVEREST_ARM_PROFILE`` -- maker-arm hardware profile path; defaults to the one shipped
  with the installed SDK, which is what its limits and gains were captured against.
* ``EVEREST_ARM_URDF`` -- physically validated six-axis Maker Arm URDF. Required only for
  Cartesian FK/IK and camera calibration capture.
* ``EVEREST_ARM_BASE_LINK`` / ``EVEREST_ARM_TOOL_LINK`` -- URDF chain endpoints; default
  to ``base_link`` and ``gripper_frame_link``.
* ``EVEREST_LEASE_BACKEND`` -- ``postgres`` (default when ``ABSURD_DATABASE_URL`` is set)
  or ``file``.
* ``EVEREST_CAMERAS`` / ``EVEREST_CAMERAS_FILE`` -- see
  :mod:`everest_robot.robot.cameras`.
* ``HF_TOKEN`` -- read by ``huggingface_hub`` itself for a private dataset. It is never
  passed through workflow parameters and never appears in a result or an error.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from everest_robot.robot.cameras import CameraRuntime
from everest_robot.robot.contracts import CancelCheck, Heartbeat
from everest_robot.robot.datasets import HuggingFaceDatasetResolver
from everest_robot.robot.lease import FileLease, PostgresAdvisoryLease, RobotLease
from everest_robot.robot.lerobot_bridge import JointFrame
from everest_robot.robot.maker_arm_port import MakerArmPort, load_maker_arm
from everest_robot.robot.parameters import RobotParameters
from everest_robot.robot.ports import ArmPort
from everest_robot.robot.replay import ReplayRunner
from everest_robot.robot.session import RobotSession

DEFAULT_PARAMETERS_PATH = "config/maker_arm_v1.yaml"


def load_parameters(environ: Mapping[str, str] | None = None) -> RobotParameters:
    environ = os.environ if environ is None else environ
    return RobotParameters.from_yaml(
        Path(environ.get("EVEREST_ROBOT_PARAMETERS", DEFAULT_PARAMETERS_PATH))
    )


def build_lease(
    parameters: RobotParameters, environ: Mapping[str, str] | None = None
) -> RobotLease:
    """Pick a lease backend.

    Postgres by default when the Absurd database is configured: workers can run on
    different hosts, and a host-local file lock would not see the other one.
    """

    environ = os.environ if environ is None else environ
    database_url = environ.get("ABSURD_DATABASE_URL")
    backend = environ.get("EVEREST_LEASE_BACKEND", "postgres" if database_url else "file")
    robot_id = parameters.identity.robot_id

    if backend == "postgres":
        if not database_url:
            raise ValueError(
                "EVEREST_LEASE_BACKEND=postgres needs ABSURD_DATABASE_URL to be set"
            )
        return PostgresAdvisoryLease(robot_id, database_url)
    if backend == "file":
        return FileLease(robot_id)
    raise ValueError(f"unsupported EVEREST_LEASE_BACKEND {backend!r} (expected postgres or file)")


def build_port(
    parameters: RobotParameters, environ: Mapping[str, str] | None = None
) -> ArmPort:
    """Build the maker-arm port for the configured CAN interface."""

    environ = os.environ if environ is None else environ
    port = environ.get("EVEREST_CAN_PORT")
    if not port:
        raise ValueError(
            "EVEREST_CAN_PORT is required: the socketcan interface name (e.g. 'can0') or "
            "the USB-CAN adapter's serial port (e.g. '/dev/ttyACM0') with EVEREST_CAN_BACKEND=slcan"
        )
    backend = environ.get("EVEREST_CAN_BACKEND", "socketcan")
    profile = environ.get("EVEREST_ARM_PROFILE")
    urdf = environ.get("EVEREST_ARM_URDF")

    backend_kwargs = {"channel": port} if backend == "socketcan" else {"port": port}
    return MakerArmPort.from_profile(
        parameters.identity,
        config_path=profile,
        backend=backend,
        urdf_path=urdf,
        base_link=environ.get("EVEREST_ARM_BASE_LINK", "base_link"),
        tool_link=environ.get("EVEREST_ARM_TOOL_LINK", "gripper_frame_link"),
        **backend_kwargs,
    )


def build_kinematics(
    parameters: RobotParameters, environ: Mapping[str, str] | None = None
) -> tuple[Any, tuple[tuple[float, float], ...]]:
    """Load Cartesian geometry and joint limits without constructing a CAN backend."""

    environ = os.environ if environ is None else environ
    urdf = environ.get("EVEREST_ARM_URDF")
    if not urdf:
        raise ValueError(
            "EVEREST_ARM_URDF is required for Cartesian FK/IK; point it at the "
            "physically validated Maker Arm URDF"
        )
    maker_arm = load_maker_arm()
    profile = environ.get("EVEREST_ARM_PROFILE")
    if profile is None:
        from maker_arm.profiles import DEFAULT_ARM_CONFIG

        profile = str(DEFAULT_ARM_CONFIG)
    config = maker_arm.ArmConfig.from_yaml(profile)
    if len(config.joints) != len(parameters.identity.joint_names):
        raise ValueError(
            f"hardware profile has {len(config.joints)} joints but parameters declare "
            f"{len(parameters.identity.joint_names)}"
        )
    solver_class = getattr(maker_arm, "SerialChainKinematics", None)
    if solver_class is None:
        raise RuntimeError(
            "the installed maker-arm SDK has no kinematics support; install the SDK "
            "revision containing SerialChainKinematics"
        )
    model = solver_class.from_urdf(
        urdf,
        base_link=environ.get("EVEREST_ARM_BASE_LINK", "base_link"),
        tool_link=environ.get("EVEREST_ARM_TOOL_LINK", "gripper_frame_link"),
        expected_joint_names=parameters.identity.joint_names[:-1],
    )
    limits = tuple((float(joint.lo), float(joint.hi)) for joint in config.joints[:-1])
    return model, limits


def open_session(
    *,
    heartbeat: Heartbeat | None = None,
    cancel: CancelCheck | None = None,
    environ: Mapping[str, str] | None = None,
) -> RobotSession:
    """Claim, connect and identity-check the deployed robot. Caller closes it."""

    environ = os.environ if environ is None else environ
    parameters = load_parameters(environ)
    session = RobotSession(
        build_port(parameters, environ),
        parameters,
        lease=build_lease(parameters, environ),
        cameras=CameraRuntime.from_env(environ),
        heartbeat=heartbeat,
        cancel=cancel,
    )
    return session.open()


def joint_frame(parameters: RobotParameters) -> JointFrame:
    """The reconciliation between this arm's radians and LeRobot's degrees.

    An absent ``lerobot_frame`` yields the identity frame, which asserts the two drivers
    share a zero pose. They do not on this arm, so replay of a MakerFollower recording will
    then fail preflight against the soft limits rather than command a wrong pose.
    """

    spec = parameters.lerobot_frame
    return JointFrame(
        parameters.identity.joint_names,
        offsets_deg=spec.offsets_deg if spec is not None else (),
    )


def build_replay_runner(
    *,
    environ: Mapping[str, str] | None = None,
    clock: Any = None,
) -> ReplayRunner:
    """Build a replay runner from deployment configuration.

    The arm port is constructed but not connected: preflight has to be able to read the
    driver's soft limits without claiming or energizing anything.
    """

    environ = os.environ if environ is None else environ
    parameters = load_parameters(environ)
    return ReplayRunner(
        build_port(parameters, environ),
        parameters,
        resolver=HuggingFaceDatasetResolver(
            require_full_revision=parameters.replay.require_full_revision
        ),
        lease=build_lease(parameters, environ),
        frame=joint_frame(parameters),
        clock=clock,
    )
