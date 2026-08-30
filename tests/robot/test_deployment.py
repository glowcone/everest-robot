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


# ── which camera SEARCH_CV closes its loop on ──────────────────────────────────────
def test_the_search_cv_backend_defaults_to_the_fixed_camera():
    from everest_robot.robot.deployment import search_cv_backend

    assert search_cv_backend({}) == "fixed"


def test_an_unknown_search_cv_backend_names_both_choices():
    from everest_robot.robot.deployment import search_cv_backend

    with pytest.raises(ValueError, match="pixel map or 'wrist'"):
        search_cv_backend({"EVEREST_SEARCH_CV": "both"})


def test_a_wrist_calibration_with_no_bump_trials_is_refused_before_the_claim(tmp_path):
    """A draft saved before any joint was bumped has no measured Jacobian, so it has no
    opinion about which way to move. Better to say so than to servo on zeros."""

    import numpy as np

    from everest_robot.pixel_map import RobotStamp
    from everest_robot.robot.deployment import load_wrist_servo
    from everest_robot.robot.wrist_servo import WristServoCalibration, WristServoError

    path = tmp_path / "wrist_servo.json"
    WristServoCalibration(
        robot=RobotStamp("maker-arm-02", "cal-1"),
        camera_name="wrist",
        joint_names=("shoulder_pan", "shoulder_lift"),
        servo_joints=("shoulder_pan",),
        goal=(320.0, 240.0, 150.0, 10.0),
        jacobian=np.array([[-220.0], [8.0], [1.0], [0.5]]),
    ).save(path)

    with pytest.raises(WristServoError, match="no bump trials"):
        load_wrist_servo({"EVEREST_WRIST_SERVO": str(path)})
