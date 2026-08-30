import time

import pytest

from everest_robot.robot.contracts import ArmLifecycle, JointLimit, RobotIdentity
from everest_robot.robot.fake_arm import FakeArm
from everest_robot.robot.teleoperation import TeleoperationController

JOINTS = ("shoulder_pan", "shoulder_lift", "gripper")
LIMITS = (
    JointLimit("shoulder_pan", -1.0, 1.0),
    JointLimit("shoulder_lift", -2.0, 0.5),
    JointLimit("gripper", -2.0, 0.0),
)
IDENTITY = RobotIdentity("maker-arm-02", "maker-arm-v1", "cal", JOINTS)


class FakeLeader:
    servo_ids = (0, 1, 2)

    def __init__(self, readings=None) -> None:
        self.readings = dict(readings or {0: 0.1, 1: -1.1, 2: -0.6})
        self.connected = False

    def connect(self) -> None:
        self.connected = True

    def read_positions(self):
        return dict(self.readings)

    def disconnect(self) -> None:
        self.connected = False


class IdentityMapper:
    def __init__(self) -> None:
        self.last = {0: 0.0, 1: -1.0, 2: -0.5}

    def map(self, readings):
        self.last.update(readings)
        return [self.last[index] for index in range(3)]


def connected_arm() -> FakeArm:
    arm = FakeArm(IDENTITY, LIMITS, positions=[0.0, -1.0, -0.5])
    arm.connect()
    return arm


def test_initial_snapshot_is_complete_and_reports_pose_difference() -> None:
    arm = connected_arm()
    leader = FakeLeader()
    controller = TeleoperationController(arm, leader, IdentityMapper())

    difference = controller.connect_and_measure()

    assert difference == pytest.approx(0.1)
    assert leader.connected
    assert arm.lifecycle is ArmLifecycle.CONNECTED
    controller.close()


def test_following_enables_sends_bounded_steps_and_holds_on_stop() -> None:
    arm = connected_arm()
    leader = FakeLeader({0: 0.5, 1: -1.5, 2: -1.0})
    controller = TeleoperationController(
        arm, leader, IdentityMapper(), rate_hz=100.0, max_velocity_rad_s=0.2
    )
    controller.connect_and_measure()

    controller.start()
    time.sleep(0.04)
    controller.stop()

    assert arm.lifecycle is ArmLifecycle.ENABLED
    assert arm.sent_commands
    assert abs(arm.sent_commands[0][0]) <= 0.002 + 1e-9
    assert controller.error is None
    controller.close()


def test_pause_holds_and_resume_continues_following() -> None:
    arm = connected_arm()
    controller = TeleoperationController(
        arm, FakeLeader(), IdentityMapper(), rate_hz=100.0, max_velocity_rad_s=0.2
    )
    controller.connect_and_measure()
    controller.start()
    time.sleep(0.02)

    assert controller.toggle_pause()
    count = len(arm.sent_commands)
    time.sleep(0.03)
    assert len(arm.sent_commands) == count
    assert not controller.toggle_pause()
    time.sleep(0.02)
    assert len(arm.sent_commands) > count
    controller.close()


def test_mapping_outside_follower_limits_is_rejected_before_enable() -> None:
    arm = connected_arm()
    leader = FakeLeader({0: 4.0, 1: -1.0, 2: -0.5})
    controller = TeleoperationController(arm, leader, IdentityMapper())

    with pytest.raises(RuntimeError, match="outside follower limits"):
        controller.connect_and_measure()

    assert arm.lifecycle is ArmLifecycle.CONNECTED
    assert not leader.connected


def test_gripper_mapping_past_its_limit_is_clamped_not_refused() -> None:
    """Closing a MakerMod gripper commands past the object on purpose (stall grip).

    The Star gripper mapping rests exactly on the follower's soft limit, so squeezing
    the leader always maps past it; that must clamp, while an arm joint outside its
    limits still refuses (see test above).
    """

    arm = connected_arm()
    leader = FakeLeader({0: 0.0, 1: -1.0, 2: 0.5})
    controller = TeleoperationController(
        arm, leader, IdentityMapper(), rate_hz=100.0, max_velocity_rad_s=0.2
    )

    difference = controller.connect_and_measure()
    assert difference == pytest.approx(0.5)  # gripper clamped from +0.5 to its 0.0 limit

    controller.start()
    time.sleep(0.04)
    controller.stop()

    assert controller.error is None
    assert arm.sent_commands
    # Every gripper command stays at or inside the clamped limit.
    assert all(command[2] <= 0.0 + 1e-9 for command in arm.sent_commands)
    controller.close()


def test_persistent_leader_loss_stops_and_holds() -> None:
    arm = connected_arm()
    leader = FakeLeader()
    controller = TeleoperationController(
        arm,
        leader,
        IdentityMapper(),
        rate_hz=200.0,
        max_velocity_rad_s=0.2,
        leader_loss_timeout_s=0.01,
    )
    controller.connect_and_measure()
    leader.readings = {}
    controller.start()

    deadline = time.monotonic() + 0.5
    while controller.running and time.monotonic() < deadline:
        time.sleep(0.005)

    assert controller.error is not None
    assert "readings lost" in controller.error
    assert arm.lifecycle is ArmLifecycle.ENABLED
    controller.close()
