"""The CAN backend guards in deployment configuration.

``build_port`` is the last place a misconfigured interface can be caught with a message
that names the actual problem. Past it, the failure surfaces from inside python-can as an
``AttributeError`` on ``socket.AF_CAN`` that says nothing about the environment.
"""

import pytest

from everest_robot.robot.deployment import build_port, load_parameters
from everest_robot.robot.parameters import RobotParameters


def parameters() -> RobotParameters:
    """The shipped parameters. None of these guards reach the driver, so they suffice."""

    return load_parameters({})


def test_a_missing_can_port_is_refused() -> None:
    with pytest.raises(ValueError, match="EVEREST_CAN_PORT is required"):
        build_port(parameters(), {})


def test_socketcan_is_refused_off_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("everest_robot.robot.deployment.sys.platform", "darwin")

    with pytest.raises(ValueError, match="needs Linux; this host is darwin"):
        build_port(parameters(), {"EVEREST_CAN_PORT": "can0", "EVEREST_CAN_BACKEND": "socketcan"})


def test_an_absent_slcan_serial_port_is_refused(tmp_path) -> None:
    absent = tmp_path / "tty.usbmodem1101"

    with pytest.raises(ValueError, match="does not exist"):
        build_port(
            parameters(),
            {"EVEREST_CAN_PORT": str(absent), "EVEREST_CAN_BACKEND": "slcan"},
        )
