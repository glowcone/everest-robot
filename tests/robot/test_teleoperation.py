import math
import time

import pytest

from everest_robot.robot.clock import ManualClock
from everest_robot.robot.contracts import ArmLifecycle, JointLimit, RobotIdentity
from everest_robot.robot.fake_arm import FakeArm
from everest_robot.robot.parameters import RobotParameters
from everest_robot.robot.session import RobotSession
from everest_robot.robot.teleoperation import (
    PARK_LIMIT_TOLERANCE_RAD,
    TeleoperationController,
    park_at_start_pose,
)

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


def test_a_sustained_excursion_is_named_but_never_stops_following() -> None:
    """The mapping does not give every joint headroom, so this is not evidence of a fault.

    wrist_flex's mapped zero *is* the follower's upper limit, so half its leader travel is
    out of range by construction. Stopping the session for it cost an operator a whole
    calibration run over a pose the arm could hold. It is promoted from CLAMPED to a
    sustained excursion, recorded for the end-of-session report, and following continues.
    """

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
    while not controller.sustained_excursions and time.monotonic() < deadline:
        time.sleep(0.005)

    assert controller.sustained_excursions == ("shoulder_pan",)
    assert controller.excursion_joints == ("shoulder_pan",)
    assert controller.error is None
    assert controller.running
    assert arm.lifecycle is ArmLifecycle.ENABLED
    assert all(command[0] <= 1.0 + 1e-9 for command in arm.sent_commands)

    # Coming back clears the live warning; the session's record of it does not.
    leader.readings = {0: 0.5, 1: -1.0, 2: -0.5}
    deadline = time.monotonic() + 0.5
    while controller.sustained_excursions and time.monotonic() < deadline:
        time.sleep(0.005)

    assert controller.sustained_excursions == ()
    assert controller.excursion_joints == ("shoulder_pan",)
    assert controller.running
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


# ── parking the arm before torque comes off ────────────────────────────────────────

START_POSE = (0.5, -1.0, -0.5)


class StubController:
    """Only what ``park_at_start_pose`` reads.

    Where the arm was before it was enabled, and the speed it was followed at -- which is
    a ceiling on the speed it is parked at.
    """

    def __init__(
        self, start_pose: tuple[float, ...] = START_POSE, max_velocity_rad_s: float = 0.25
    ) -> None:
        self.start_pose = start_pose
        self.max_velocity_rad_s = max_velocity_rad_s


def park_parameters() -> RobotParameters:
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
                # The deployed value. At a quarter speed a park of much over a radian takes
                # longer than this, which is the whole reason _park_profile derives its
                # own budget rather than inheriting this one.
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


@pytest.fixture
def teleoperated():
    """A claimed arm parked at START_POSE, as a teleoperation caller has when following begins.

    Yields the session and arm and always closes, so one failing assertion cannot leave
    the in-process lease held for every test after it.
    """

    clock = ManualClock()
    arm = FakeArm(IDENTITY, LIMITS, clock=clock, positions=list(START_POSE))
    session = RobotSession(arm, park_parameters(), clock=clock, cameras=None).open()
    try:
        yield session, arm, clock
    finally:
        session.close()


def walk_away(arm: FakeArm, clock: ManualClock, pose: tuple[float, ...]) -> None:
    """Leave the arm where the leader put it: enabled, and nowhere near where it started."""

    arm.enable()
    arm.send_targets(pose)
    clock.advance(10.0)
    assert arm.read_state().positions == pytest.approx(pose, abs=1e-6)


def test_the_arm_is_driven_back_to_where_it_started_before_torque_comes_off(
    teleoperated,
) -> None:
    """The whole point: it goes limp where the operator parked it, not where it stopped."""

    session, arm, clock = teleoperated
    walk_away(arm, clock, (-0.5, 0.0, -1.5))

    park_at_start_pose(session, StubController())  # type: ignore[arg-type]

    assert arm.read_state().positions == pytest.approx(START_POSE, abs=0.02)
    # Still under power: releasing it is RobotSession.close()'s job, and only after this.
    assert arm.lifecycle is ArmLifecycle.ENABLED


def test_the_park_never_commands_a_step_faster_than_it_was_followed_at(teleoperated) -> None:
    """The park is a bounded trajectory, not a jump at whatever speed the arm can manage.

    Checked on the commands rather than the reached positions: the arm is the thing that
    is not supposed to be asked to move fast, and a stand-in that tracks perfectly would
    hide a jump that a real one would take at full speed.
    """

    session, arm, clock = teleoperated
    walk_away(arm, clock, (-0.5, 0.0, -1.5))
    arm.sent_commands.clear()

    park_at_start_pose(session, StubController(max_velocity_rad_s=0.1))

    # The controller's cap is below a quarter of the file's 0.5 rad/s, so it is the one
    # that binds. One control period of travel is all a single command may ask for.
    per_tick = 0.1 / session.parameters.motion_defaults.control_rate_hz
    steps = [
        abs(later - earlier)
        for before, after in zip(arm.sent_commands, arm.sent_commands[1:], strict=False)
        for earlier, later in zip(before, after, strict=True)
    ]
    assert steps, "the park sent no commands"
    assert max(steps) <= per_tick + 1e-9
    assert arm.read_state().positions == pytest.approx(START_POSE, abs=0.02)


def test_a_park_too_slow_to_finish_in_the_default_budget_still_finishes(teleoperated) -> None:
    """Speed and completion must not trade against each other; the budget follows the plan.

    1.4 rad at a quarter of 0.5 rad/s takes about 11.5s, past the file's 10s timeout. That
    used to end as a TIMEOUT with the arm held partway and then disabled there.
    """

    session, arm, clock = teleoperated
    walk_away(arm, clock, (-0.5, 0.0, -1.9))
    assert session.parameters.motion_defaults.timeout_s == 10.0

    park_at_start_pose(session, StubController())

    assert arm.read_state().positions == pytest.approx(START_POSE, abs=0.02)


def test_the_park_runs_after_a_teleoperation_failure_too(teleoperated, capsys) -> None:
    """An out-of-range stop is exactly when the arm is somewhere it must not be dropped.

    Nothing in the park consults ``controller.error``, which is what makes it run on
    the failure paths as well as on ``q``; this pins that down.
    """

    session, arm, clock = teleoperated
    walk_away(arm, clock, (-0.9, 0.4, -1.9))

    park_at_start_pose(session, StubController())  # type: ignore[arg-type]

    assert arm.read_state().positions == pytest.approx(START_POSE, abs=0.02)
    assert "back at the starting pose" in capsys.readouterr().err


def test_an_arm_already_at_its_starting_pose_is_not_moved(teleoperated, capsys) -> None:
    session, arm, clock = teleoperated
    arm.enable()

    park_at_start_pose(session, StubController())  # type: ignore[arg-type]

    assert arm.sent_commands == []
    assert "already at the pose it started from" in capsys.readouterr().err


def test_an_arm_that_was_never_enabled_is_neither_commanded_nor_complained_about(
    teleoperated, capsys
) -> None:
    """No start pose means the leader was never measured, so nothing was ever energized."""

    session, arm, _ = teleoperated

    park_at_start_pose(session, StubController(()))  # type: ignore[arg-type]

    assert arm.sent_commands == []
    assert capsys.readouterr().err == ""


def test_an_arm_that_is_no_longer_enabled_is_not_commanded_but_is_reported(
    teleoperated, capsys
) -> None:
    """It is wherever it stopped, under the driver's policy; a silent skip would hide that."""

    session, arm, _ = teleoperated

    park_at_start_pose(session, StubController())  # type: ignore[arg-type]

    assert arm.sent_commands == []
    assert "not returning to the starting pose" in capsys.readouterr().err


def test_a_park_that_cannot_run_is_reported_rather_than_raised(teleoperated, capsys) -> None:
    """The teleoperation error is the one the operator needs; this must not replace it."""

    session, arm, clock = teleoperated
    walk_away(arm, clock, (-0.5, 0.0, -1.5))

    # Further outside the limits than any arm droops: refused, not clamped.
    park_at_start_pose(session, StubController((5.0, -1.0, -0.5)))  # type: ignore[arg-type]

    error = capsys.readouterr().err
    assert "FAILED" in error
    assert "limits are not the ones it started under" in error
    assert "support it" in error
    assert arm.lifecycle is ArmLifecycle.ENABLED


def test_a_start_pose_drooped_past_a_limit_is_parked_at_the_limit(teleoperated, capsys) -> None:
    """The gravity case: an arm that enabled from outside its limits still gets a park.

    `enable()` admits a joint drooped past a soft limit and the motion controller refuses
    to command one, so without this the ordinary resting arm ends every powered session
    with no park at all.
    """

    session, arm, clock = teleoperated
    drooped = (1.1, -1.0, -0.5)  # shoulder_pan 0.1 rad past its upper limit of 1.0
    walk_away(arm, clock, (-0.5, 0.0, -1.5))

    park_at_start_pose(session, StubController(drooped))  # type: ignore[arg-type]

    assert arm.read_state().positions == pytest.approx((1.0, -1.0, -0.5), abs=0.02)
    error = capsys.readouterr().err
    assert "nearest in-limit pose" in error
    assert "shoulder_pan" in error
    assert "FAILED" not in error
    assert arm.lifecycle is ArmLifecycle.ENABLED


def test_the_droop_a_park_will_absorb_stops_at_the_drivers_wrap_grace(
    teleoperated, capsys
) -> None:
    """The tolerance is the drivers' own disagreement, not an adjustable convenience."""

    session, arm, clock = teleoperated
    walk_away(arm, clock, (-0.5, 0.0, -1.5))
    just_past = 1.0 + PARK_LIMIT_TOLERANCE_RAD + 0.01

    park_at_start_pose(session, StubController((just_past, -1.0, -0.5)))  # type: ignore[arg-type]

    error = capsys.readouterr().err
    assert "nearest in-limit pose" not in error
    assert "FAILED" in error
