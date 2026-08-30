import pytest

from everest_robot.goto import (
    DEFAULT_SPEED_SCALE,
    GotoRefused,
    go_to,
    resolve_route,
    transitions_ending_at,
    widest_displacement,
)
from everest_robot.robot.clock import ManualClock
from everest_robot.robot.contracts import FailureReason, JointLimit, RobotIdentity
from everest_robot.robot.fake_arm import FakeArm
from everest_robot.robot.parameters import RobotParameters
from everest_robot.robot.session import RobotSession

JOINTS = ["shoulder_pan", "shoulder_lift", "gripper"]
CALIBRATION = "maker-arm-02-2026-08-20"
LIMITS = (
    JointLimit("shoulder_pan", -0.668, 4.818),
    JointLimit("shoulder_lift", -2.024, 0.979),
    JointLimit("gripper", -2.092, -0.039),
)

PARKED = [0.0, -0.5, -1.0]


def _preset(joints: list[float], **extra: object) -> dict[str, object]:
    return {
        "joints": joints,
        "calibration_id": CALIBRATION,
        "approved_by": "operator",
        "captured_at": "2026-08-21",
        **extra,
    }


def parameters(
    positions: dict[str, object] | None = None,
    transitions: dict[str, object] | None = None,
) -> RobotParameters:
    return RobotParameters.from_mapping(
        {
            "schema_version": 1,
            "robot": {
                "id": "maker-arm-02",
                "model": "maker-arm-v1",
                "calibration_id": CALIBRATION,
                "joint_order": list(JOINTS),
                "units": "radians",
            },
            "motion_defaults": {
                "max_velocity_rad_s": 0.5,
                "max_acceleration_rad_s2": 2.0,
                "tolerance_rad": 0.02,
                "settle_time_s": 0.2,
                "timeout_s": 10.0,
                "control_rate_hz": 50,
            },
            "named_positions": positions or {},
            "named_transitions": transitions or {},
            "policy": {"default_controller": "vla", "fps": 30, "max_duration_s": 30},
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


def identity() -> RobotIdentity:
    return RobotIdentity(
        robot_id="maker-arm-02",
        model="maker-arm-v1",
        calibration_id=CALIBRATION,
        joint_names=tuple(JOINTS),
    )


def session(
    params: RobotParameters, positions: list[float]
) -> tuple[RobotSession, FakeArm, ManualClock]:
    clock = ManualClock()
    arm = FakeArm(
        identity=identity(), joint_limits=LIMITS, clock=clock, positions=list(positions)
    )
    return RobotSession(arm, params, clock=clock), arm, clock


# ── route resolution, which happens before anything is claimed ─────────────────────


def test_a_position_with_no_transition_is_reached_directly():
    params = parameters({"stage": _preset([0.2, -0.4, -1.0])})

    route = resolve_route(params, "stage")

    assert route.transition is None
    assert route.waypoints == ("stage",)
    assert route.target.joints == pytest.approx((0.2, -0.4, -1.0))
    assert "direct to stage" in route.describe()


def test_an_unknown_destination_is_refused_with_the_approved_list():
    params = parameters({"stage": _preset([0.2, -0.4, -1.0])})

    with pytest.raises(GotoRefused, match="unknown named position 'stagg'.*stage"):
        resolve_route(params, "stagg")


def test_no_positions_at_all_still_refuses_rather_than_inventing_one():
    with pytest.raises(GotoRefused, match="none defined"):
        resolve_route(parameters(), "stage")


def test_an_approved_transition_is_used_instead_of_the_direct_line():
    """The transition exists because the straight line was not shown to be safe."""

    params = parameters(
        {"lift": _preset([0.0, 0.5, -1.0]), "stage": _preset([0.2, -0.4, -1.0])},
        {"over-the-fixture": {"waypoints": ["lift", "stage"]}},
    )

    route = resolve_route(params, "stage")

    assert route.transition == "over-the-fixture"
    assert route.waypoints == ("lift", "stage")
    assert route.target.name == "stage"


def test_two_transitions_to_one_pose_is_the_operator_s_choice_not_ours():
    params = parameters(
        {"lift": _preset([0.0, 0.5, -1.0]), "stage": _preset([0.2, -0.4, -1.0])},
        {
            "over-the-fixture": {"waypoints": ["lift", "stage"]},
            "around-the-post": {"waypoints": ["lift", "stage"]},
        },
    )

    with pytest.raises(GotoRefused, match="2 approved transitions.*--transition"):
        resolve_route(params, "stage")

    route = resolve_route(params, "stage", transition="around-the-post")
    assert route.transition == "around-the-post"


def test_a_transition_that_ends_somewhere_else_is_not_a_way_to_get_there():
    params = parameters(
        {"lift": _preset([0.0, 0.5, -1.0]), "stage": _preset([0.2, -0.4, -1.0])},
        {"to-lift": {"waypoints": ["stage", "lift"]}},
    )

    with pytest.raises(GotoRefused, match="ends at 'lift', not at 'stage'"):
        resolve_route(params, "stage", transition="to-lift")


def test_an_unknown_transition_is_refused():
    params = parameters({"stage": _preset([0.2, -0.4, -1.0])})

    with pytest.raises(GotoRefused, match="unknown named transition"):
        resolve_route(params, "stage", transition="nope")


def test_transitions_ending_at_ignores_ones_that_merely_pass_through():
    params = parameters(
        {
            "lift": _preset([0.0, 0.5, -1.0]),
            "stage": _preset([0.2, -0.4, -1.0]),
            "park": _preset([0.0, -0.5, -1.0]),
        },
        {"long-way": {"waypoints": ["lift", "stage", "park"]}},
    )

    assert transitions_ending_at(params, "park") == ("long-way",)
    assert transitions_ending_at(params, "stage") == ()


# ── how much the arm is about to move ──────────────────────────────────────────────


def test_widest_displacement_names_the_joint_that_moves_most():
    params = parameters({"stage": _preset([0.9, -0.4, -1.0])})

    widest, joint = widest_displacement(PARKED, resolve_route(params, "stage"), JOINTS)

    assert joint == "shoulder_pan"
    assert widest == pytest.approx(0.9)


def test_widest_displacement_measures_each_leg_not_the_straight_line():
    """A transition's whole point is that it does not go straight there."""

    params = parameters(
        {"lift": _preset([0.0, 0.9, -1.0]), "stage": _preset([0.05, -0.5, -1.0])},
        {"over-the-fixture": {"waypoints": ["lift", "stage"]}},
    )

    widest, joint = widest_displacement(PARKED, resolve_route(params, "stage"), JOINTS)

    # Start to finish, shoulder_lift barely moves; the detour swings it 1.4 rad each way.
    assert joint == "shoulder_lift"
    assert widest == pytest.approx(1.4)


# ── against the fake arm ───────────────────────────────────────────────────────────


def test_go_to_drives_the_arm_to_the_preset():
    params = parameters({"stage": _preset([0.2, -0.4, -1.0])})
    open_session, arm, clock = session(params, PARKED)

    with open_session as live:
        result = go_to(live, resolve_route(params, "stage"), speed_scale=DEFAULT_SPEED_SCALE)

    assert result.reached
    assert result.position_name == "stage"
    # A single-leg move records no waypoint path: MotionResult keeps that field for
    # transitions, where the route is not implied by the destination.
    assert result.waypoints == ()
    assert result.final_joints == pytest.approx((0.2, -0.4, -1.0), abs=0.02)
    assert arm.positions == pytest.approx([0.2, -0.4, -1.0], abs=0.02)
    assert clock.monotonic() > 0.0


def test_go_to_walks_every_waypoint_of_a_transition():
    params = parameters(
        {"lift": _preset([0.0, 0.5, -1.0]), "stage": _preset([0.2, -0.4, -1.0])},
        {"over-the-fixture": {"waypoints": ["lift", "stage"]}},
    )
    open_session, arm, _ = session(params, PARKED)

    with open_session as live:
        result = go_to(live, resolve_route(params, "stage"))

    assert result.reached
    assert result.waypoints == ("lift", "stage")
    assert arm.positions == pytest.approx([0.2, -0.4, -1.0], abs=0.02)


def test_a_dry_run_reports_a_duration_and_leaves_the_arm_where_it_was():
    params = parameters({"stage": _preset([0.2, -0.4, -1.0])})
    open_session, arm, _ = session(params, PARKED)

    with open_session as live:
        result = go_to(live, resolve_route(params, "stage"), dry_run=True)

    assert result.dry_run
    assert not result.reached
    assert result.planned_duration_s > 0.0
    assert result.commands_sent == 0
    assert arm.positions == pytest.approx(PARKED)


def test_a_dry_run_of_a_transition_plans_every_leg():
    params = parameters(
        {"lift": _preset([0.0, 0.5, -1.0]), "stage": _preset([0.2, -0.4, -1.0])},
        {"over-the-fixture": {"waypoints": ["lift", "stage"]}},
    )
    open_session, arm, _ = session(params, PARKED)

    with open_session as live:
        direct = go_to(live, resolve_route(params, "stage", transition=None), dry_run=True)

    assert direct.waypoints == ("lift", "stage")
    assert direct.planned_duration_s > 0.0
    assert arm.positions == pytest.approx(PARKED)


def test_a_preset_outside_the_active_hardware_limits_is_refused_not_clamped():
    """The parameters file and the driver's profile can disagree; the driver wins."""

    params = parameters({"stage": _preset([0.2, -0.4, -1.0])})
    open_session, arm, _ = session(params, PARKED)
    # Narrow the driver's soft limits under the preset, as a re-profiled arm would.
    arm.joint_limits = (
        JointLimit("shoulder_pan", -0.1, 0.1),
        LIMITS[1],
        LIMITS[2],
    )

    with open_session as live:
        result = go_to(live, resolve_route(params, "stage"))

    assert not result.reached
    assert result.failure_reason is FailureReason.LIMIT_VIOLATION
    assert arm.positions == pytest.approx(PARKED)


def test_a_preset_the_arm_is_already_at_commands_nothing():
    params = parameters({"park": _preset(list(PARKED))})
    open_session, arm, _ = session(params, PARKED)

    with open_session as live:
        result = go_to(live, resolve_route(params, "park"))

    assert result.reached
    assert result.already_at_target
    assert result.commands_sent == 0
