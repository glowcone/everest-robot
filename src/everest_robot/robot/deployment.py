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
* ``EVEREST_ARM_DRIVER`` -- ``maker-arm`` (default; private protocol, fails closed on
  this arm) or ``mit`` for motors provisioned in RobStride's MIT protocol, which is what
  maker-arm-02 runs per docs/adr/0002-mit-protocol-motor-operation.md. The ``mit``
  driver is qualified for the lease-local calibration monitor only; replay and the
  workflow stay unqualified until that ADR's checklist is done.
* ``EVEREST_ARM_PROFILE`` -- maker-arm hardware profile path; defaults to the one shipped
  with the installed SDK, which is what its limits and gains were captured against.
* ``EVEREST_LEASE_BACKEND`` -- ``postgres`` (default when ``ABSURD_DATABASE_URL`` is set)
  or ``file``.
* ``EVEREST_CAMERAS`` / ``EVEREST_CAMERAS_FILE`` -- see
  :mod:`everest_robot.robot.cameras`.
* ``EVEREST_PIXEL_MAP`` -- path to the fixed camera's pixel-to-joint calibration
  (default ``config/pixel_map.json``). It is not a policy observation and deliberately
  does not go through ``EVEREST_CAMERAS``: the camera id it uses is part of the
  calibration, because moving that camera voids every sample in the file.
* ``HF_TOKEN`` -- read by ``huggingface_hub`` itself for a private dataset. It is never
  passed through workflow parameters and never appears in a result or an error.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from everest_robot.pixel_map import PixelJointMap, PixelMapError
from everest_robot.robot.cameras import CameraRuntime
from everest_robot.robot.contracts import CancelCheck, Heartbeat
from everest_robot.robot.datasets import HuggingFaceDatasetResolver
from everest_robot.robot.lease import FileLease, PostgresAdvisoryLease, RobotLease
from everest_robot.robot.lerobot_bridge import JointFrame
from everest_robot.robot.maker_arm_port import MakerArmPort
from everest_robot.robot.parameters import RobotParameters
from everest_robot.robot.ports import ArmPort
from everest_robot.robot.replay import ReplayRunner
from everest_robot.robot.robstride_mit_port import RobstrideMitPort
from everest_robot.robot.session import RobotSession

DEFAULT_PARAMETERS_PATH = "config/maker_arm_v1.yaml"
DEFAULT_PIXEL_MAP_PATH = "config/pixel_map.json"


def parameters_path(environ: Mapping[str, str] | None = None) -> Path:
    """Which parameters file this deployment reads, and writes captured presets back to."""

    environ = os.environ if environ is None else environ
    return Path(environ.get("EVEREST_ROBOT_PARAMETERS", DEFAULT_PARAMETERS_PATH))


def load_parameters(environ: Mapping[str, str] | None = None) -> RobotParameters:
    return RobotParameters.from_yaml(parameters_path(environ))


def pixel_map_path(environ: Mapping[str, str] | None = None) -> Path:
    """Which fixed-camera calibration the visual follower uses."""

    environ = os.environ if environ is None else environ
    return Path(environ.get("EVEREST_PIXEL_MAP", DEFAULT_PIXEL_MAP_PATH))


def load_pixel_map(environ: Mapping[str, str] | None = None) -> PixelJointMap:
    """Load the calibration and refuse one that cannot drive anything.

    A file with samples but no fit, or with no ROI, is a teaching session someone stopped
    halfway through. Both refusals belong here rather than at the first servo tick, where
    the arm is already claimed and the operator is already waiting.
    """

    path = pixel_map_path(environ)
    calibration = PixelJointMap.load(path)
    if calibration.model is None:
        raise PixelMapError(f"{path} has no fit yet; run `robot-pixel-map fit` first")
    if calibration.detector.roi_xywh is None:
        raise PixelMapError(
            f"{path} stored no detector ROI; re-run `robot-pixel-map collect --roi X Y W H`"
        )
    return calibration


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
    """Build the arm port for the configured CAN interface and driver."""

    environ = os.environ if environ is None else environ
    port = environ.get("EVEREST_CAN_PORT")
    if not port:
        raise ValueError(
            "EVEREST_CAN_PORT is required: the socketcan interface name (e.g. 'can0') or "
            "the USB-CAN adapter's serial port (e.g. '/dev/ttyACM0') with EVEREST_CAN_BACKEND=slcan"
        )
    backend = environ.get("EVEREST_CAN_BACKEND", "socketcan")
    driver = environ.get("EVEREST_ARM_DRIVER", "maker-arm")

    # SocketCAN is a Linux kernel facility. python-can reaches for socket.AF_CAN, which
    # only exists there, so on any other host this fails deep inside the driver with an
    # AttributeError that says nothing about the actual misconfiguration.
    if backend == "socketcan" and sys.platform != "linux":
        raise ValueError(
            f"EVEREST_CAN_BACKEND=socketcan needs Linux; this host is {sys.platform}. "
            f"Use a USB-CAN adapter with EVEREST_CAN_BACKEND=slcan and EVEREST_CAN_PORT "
            f"set to its serial port, or run on the robot's Linux host."
        )
    if backend == "slcan" and not Path(port).exists():
        raise ValueError(
            f"EVEREST_CAN_PORT={port!r} does not exist: the USB-CAN adapter is not "
            f"plugged in, or it enumerates under a different name."
        )

    if driver == "mit":
        return RobstrideMitPort.from_deployment(
            parameters.identity,
            joint_frame(parameters),
            port=port,
            backend=backend,
        )
    if driver != "maker-arm":
        raise ValueError(
            f"unsupported EVEREST_ARM_DRIVER {driver!r} (expected maker-arm or mit)"
        )

    profile = environ.get("EVEREST_ARM_PROFILE")
    backend_kwargs = {"channel": port} if backend == "socketcan" else {"port": port}
    return MakerArmPort.from_profile(
        parameters.identity,
        config_path=profile,
        backend=backend,
        **backend_kwargs,
    )


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
