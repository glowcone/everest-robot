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
* ``EVEREST_LEASE_BACKEND`` -- ``postgres`` (default when ``ABSURD_DATABASE_URL`` is set)
  or ``file``.
* ``EVEREST_CAMERAS`` / ``EVEREST_CAMERAS_FILE`` -- see
  :mod:`everest_robot.robot.cameras`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from everest_robot.robot.cameras import CameraRuntime
from everest_robot.robot.contracts import CancelCheck, Heartbeat
from everest_robot.robot.lease import FileLease, PostgresAdvisoryLease, RobotLease
from everest_robot.robot.maker_arm_port import MakerArmPort
from everest_robot.robot.parameters import RobotParameters
from everest_robot.robot.ports import ArmPort
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
