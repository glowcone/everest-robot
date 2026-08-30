import math
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


def test_a_momentary_excursion_is_clamped_and_following_continues() -> None:
    """A leader has reach the follower does not; walking past the edge is not a crash.

    The follower is held at the soft limit and the joint is named, so an operator can
    tell "that joint is at the end of its travel" from "the arm has stopped".
    """

    arm = connected_arm()
    leader = FakeLeader()
    controller = TeleoperationController(
        arm,
        leader,
        IdentityMapper(),
        rate_hz=200.0,
        max_velocity_rad_s=2.0,
        out_of_range_timeout_s=0.5,
    )
    controller.connect_and_measure()
    controller.start()

    leader.readings = {0: 4.0, 1: -1.0, 2: -0.5}
    deadline = time.monotonic() + 0.3
    while not controller.clamped_joints and time.monotonic() < deadline:
        time.sleep(0.005)
    assert controller.clamped_joints == ("shoulder_pan",)
    assert controller.running
    assert all(command[0] <= 1.0 + 1e-9 for command in arm.sent_commands)

    leader.readings = {0: 0.5, 1: -1.0, 2: -0.5}
    deadline = time.monotonic() + 0.3
    while controller.clamped_joints and time.monotonic() < deadline:
        time.sleep(0.005)

    assert controller.clamped_joints == ()
    assert controller.error is None
    assert controller.running
    controller.close()


def test_a_sustained_excursion_stops_and_holds() -> None:
    """Out of range and staying there is the mapping, not the operator."""

    arm = connected_arm()
    leader = FakeLeader()
    controller = TeleoperationController(
        arm,
        leader,
        IdentityMapper(),
        rate_hz=200.0,
        max_velocity_rad_s=0.2,
        out_of_range_timeout_s=0.02,
    )
    controller.connect_and_measure()
    controller.start()
    leader.readings = {0: 4.0, 1: -1.0, 2: -0.5}

    deadline = time.monotonic() + 0.5
    while controller.running and time.monotonic() < deadline:
        time.sleep(0.005)

    assert controller.error is not None
    assert "stayed outside follower limits" in controller.error
    assert "shoulder_pan" in controller.error
    assert arm.lifecycle is ArmLifecycle.ENABLED
    assert controller.clamped_joints == ()
    controller.close()


def test_a_non_finite_mapping_stops_instead_of_being_clamped() -> None:
    """NaN is not an excursion: there is no pose to hold the follower at."""

    arm = connected_arm()
    leader = FakeLeader({0: math.nan, 1: -1.0, 2: -0.5})
    controller = TeleoperationController(arm, leader, IdentityMapper())

    with pytest.raises(RuntimeError, match="non-finite target"):
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
    # The routine gripper clamp is not an excursion and must not be flagged as one.
    assert controller.clamped_joints == ()
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


def test_stopping_without_a_hold_commands_nothing_on_the_way_out() -> None:
    """The teardown that follows cuts torque, so a parting hold is only a lurch."""

    arm = connected_arm()
    controller = TeleoperationController(arm, FakeLeader(), IdentityMapper())
    controller.connect_and_measure()
    controller.start()
    time.sleep(0.05)

    commands = len(arm.sent_commands)
    controller.close(hold=False)
    assert len(arm.sent_commands) == commands


def test_stopping_with_a_hold_still_freezes_the_follower() -> None:
    arm = connected_arm()
    controller = TeleoperationController(arm, FakeLeader(), IdentityMapper())
    controller.connect_and_measure()
    controller.start()
    time.sleep(0.05)

    commands = len(arm.sent_commands)
    controller.close()
    assert len(arm.sent_commands) >= commands
    assert arm.lifecycle is ArmLifecycle.ENABLED


def test_the_pose_the_arm_started_from_is_recorded_and_never_moves() -> None:
    """It is where the operator parked the arm, so it is where the arm can be dropped."""

    arm = connected_arm()
    controller = TeleoperationController(arm, FakeLeader(), IdentityMapper())

    assert controller.start_pose == ()

    controller.connect_and_measure()
    assert controller.start_pose == (0.0, -1.0, -0.5)

    controller.start()
    time.sleep(0.05)
    controller.close()

    # Following moved the follower; the pose it is to be parked at is not dragged along.
    assert arm.sent_commands
    assert controller.start_pose == (0.0, -1.0, -0.5)


def test_a_leader_that_never_measured_records_no_pose_to_return_to() -> None:
    """Nothing was enabled, so there is nothing to undo -- and no pose to invent."""

    arm = connected_arm()
    leader = FakeLeader({0: math.nan, 1: -1.0, 2: -0.5})
    controller = TeleoperationController(arm, leader, IdentityMapper())

    with pytest.raises(RuntimeError):
        controller.connect_and_measure()

    assert controller.start_pose == ()
