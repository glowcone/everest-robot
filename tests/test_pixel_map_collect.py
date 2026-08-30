"""What ``robot-pixel-map collect`` does to the arm on the way out.

Teaching walks the arm a long way from where it started, and the session releases torque
unconditionally when it closes. For a long time this path had no park at all, which does
not leave the arm where it is -- it drops it. Because ``start_pose`` is measured with the
torque already off, gravity delivers the arm to almost exactly the pose a park would have
driven it to, so a missing park is indistinguishable from an instant one. That is why it
went unnoticed, and why it is pinned here rather than left to inspection.
"""

import argparse
import types

import pytest

from everest_robot import calibrate_pixel_map
from everest_robot.calibrate_pixel_map import cmd_collect
from everest_robot.robot.clock import ManualClock
from everest_robot.robot.contracts import ArmLifecycle, JointLimit, RobotIdentity
from everest_robot.robot.fake_arm import FakeArm
from everest_robot.robot.parameters import RobotParameters
from everest_robot.robot.session import RobotSession

JOINTS = ("shoulder_pan", "shoulder_lift", "gripper")
LIMITS = (
    JointLimit("shoulder_pan", -1.0, 1.0),
    JointLimit("shoulder_lift", -2.0, 0.5),
    JointLimit("gripper", -2.0, 0.0),
)
IDENTITY = RobotIdentity("maker-arm-02", "maker-arm-v1", "cal", JOINTS)
START_POSE = (0.5, -1.0, -0.5)
TAUGHT_POSE = (-0.5, 0.0, -1.5)


def parameters() -> RobotParameters:
    return RobotParameters.from_mapping(
        {
            "schema_version": 1,
            "robot": {
                "id": "maker-arm-02",
                "model": "maker-arm-v1",
                "calibration_id": "cal",
                "joint_order": list(JOINTS),
                "units": "radians",
            },
            "motion_defaults": {
                "max_velocity_rad_s": 0.5,
                "max_acceleration_rad_s2": 2.0,
                "tolerance_rad": 0.02,
                "settle_time_s": 0.05,
                "timeout_s": 10.0,
                "control_rate_hz": 50,
            },
            "named_positions": {},
            "named_transitions": {},
            "policy": {"default_controller": "vla", "fps": 10, "max_duration_s": 5.0},
            "replay": {
                "require_matching_robot_id": True,
                "require_matching_calibration_id": True,
                "safe_start_position": None,
                "max_speed_scale": 1.0,
            },
        },
        config_digest="sha256:test",
        source="test.yaml",
    )


class StubController:
    """The teleoperation controller as ``cmd_collect``'s teardown sees it."""

    def __init__(self, arm: FakeArm) -> None:
        self.arm = arm
        self.start_pose = START_POSE
        self.max_velocity_rad_s = 0.25
        self.error = None
        self.closed_holding: bool | None = None

    def close(self, *, hold: bool = True) -> None:
        self.closed_holding = hold


@pytest.fixture
def collecting(tmp_path, monkeypatch):
    """``cmd_collect`` with the camera, the window and the leader replaced.

    Everything that is not the teardown is stubbed: the point is what happens to the arm
    after the capture loop ends, not how a frame becomes a sample.
    """

    clock = ManualClock()
    arm = FakeArm(IDENTITY, LIMITS, clock=clock, positions=list(START_POSE))
    session = RobotSession(arm, parameters(), clock=clock, cameras=None).open()
    controller = StubController(arm)

    def start_teleoperation(_session, _args):
        # What the real one does that matters here: the arm ends up enabled, following,
        # and somewhere else entirely.
        arm.enable()
        arm.send_targets(TAUGHT_POSE)
        clock.advance(10.0)
        return controller

    monkeypatch.setattr(calibrate_pixel_map, "_open_session", lambda _args: session)
    monkeypatch.setattr(calibrate_pixel_map, "_start_teleoperation", start_teleoperation)
    monkeypatch.setattr(calibrate_pixel_map, "load_cv2", lambda: types.SimpleNamespace())
    monkeypatch.setattr(calibrate_pixel_map, "open_capture", lambda _camera: _FakeCapture())
    monkeypatch.setattr(calibrate_pixel_map, "read_frame", lambda _capture, _camera: object())
    monkeypatch.setattr(calibrate_pixel_map, "detect_carabiner", _no_detection)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "q")

    args = argparse.Namespace(
        fake=False,
        config=str(tmp_path / "pixel_map.json"),
        camera="0",
        camera_backend="auto",
        width=None,
        height=None,
        roi=[0, 0, 100, 100],
        margin_px=None,
        approach_rad=None,
        no_teleop=False,
        no_fit=False,
        no_window=True,
        joints=list(JOINTS[:2]),
        model="thin_plate_spline",
        smoothing=0.0,
        ridge=1e-3,
        roll_joint="gripper",
        base_joint="shoulder_pan",
        no_roll=True,
        holdout=0,
    )
    try:
        yield args, arm, controller
    finally:
        if session.is_open:
            session.close()


class _FakeCapture:
    def release(self) -> None:
        pass


def _no_detection(_frame, _roi):
    raise RuntimeError("no detection in this test")


def test_teaching_parks_the_arm_instead_of_dropping_it(collecting) -> None:
    """The bug this file exists for: the arm used to be released wherever teaching ended.

    Asserted on the last *command*, not on where the arm ends up. A ``FakeArm`` does not
    fall when torque comes off, so it sits at the taught pose either way and the missing
    park is invisible in its final position -- which is the same reason the bug survived
    on the bench, where gravity happened to deliver the arm to roughly the right place.
    """

    args, arm, _controller = collecting

    assert cmd_collect(args) == 0

    assert arm.sent_commands[-1] == pytest.approx(START_POSE, abs=0.02)
    assert arm.read_state().positions == pytest.approx(START_POSE, abs=0.02)
    # And only then released -- parking under no power is not parking.
    assert arm.lifecycle is ArmLifecycle.DISCONNECTED


def test_the_park_is_a_bounded_trajectory_not_a_jump(collecting) -> None:
    """It must be driven back, at the speed it was taught at, not commanded there at once."""

    args, arm, _controller = collecting

    cmd_collect(args)

    commands = [command for command in arm.sent_commands if command != TAUGHT_POSE]
    steps = [
        abs(later - earlier)
        for before, after in zip(commands, commands[1:], strict=False)
        for earlier, later in zip(before, after, strict=True)
    ]
    assert steps, "the park sent no commands"
    # A quarter of the file's 0.5 rad/s, over one 50 Hz control period.
    assert max(steps) <= 0.125 / 50 + 1e-9


def test_following_is_stopped_without_a_parting_hold(collecting) -> None:
    """A hold before the park only snaps the arm toward its last leader target."""

    args, _arm, controller = collecting

    cmd_collect(args)

    assert controller.closed_holding is False
