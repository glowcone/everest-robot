import pytest

from everest_robot.adapters import EverestRobot, ScaffoldRobot, robot_session
from everest_robot.domain import AttachmentResult, json_dict
from everest_robot.robot.clock import ManualClock
from everest_robot.robot.contracts import JointLimit, RobotIdentity, TerminationReason
from everest_robot.robot.fake_arm import FakeArm
from everest_robot.robot.lease import InMemoryLease
from everest_robot.robot.parameters import RobotParameters
from everest_robot.robot.policy import ScriptedPolicy
from everest_robot.robot.session import RobotSession

JOINTS = ("shoulder_pan", "shoulder_lift", "gripper")
LIMITS = (
    JointLimit("shoulder_pan", -1.0, 1.0),
    JointLimit("shoulder_lift", -2.0, 0.5),
    JointLimit("gripper", -2.0, 0.0),
)
CALIBRATION = "cal-2026-08-20"
IDENTITY = RobotIdentity("maker-arm-02", "maker-arm-v1", CALIBRATION, JOINTS)
ACTION_KEYS = tuple(f"{name}.pos" for name in JOINTS)


def parameters() -> RobotParameters:
    preset = {
        "calibration_id": CALIBRATION,
        "approved_by": "operator",
        "captured_at": "2026-08-21",
    }
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
            "named_positions": {
                "clip_ready": {"joints": [0.5, -0.5, -0.1], **preset},
                "clearance": {"joints": [0.0, -1.0, -0.1], **preset},
                # Past shoulder_lift's upper soft limit of 0.5 rad.
                "unreachable": {"joints": [0.5, 0.9, -0.1], **preset},
            },
            "named_transitions": {"to_clip_ready": {"waypoints": ["clearance", "clip_ready"]}},
            "policy": {"default_controller": "vla", "fps": 10, "max_duration_s": 5.0},
            "replay": {
                "require_matching_robot_id": True,
                "require_matching_calibration_id": True,
                "safe_start_position": "clip_ready",
                "max_speed_scale": 1.0,
            },
        },
        config_digest="sha256:test",
        source="test.yaml",
    )


def make_session(**arm_kwargs) -> RobotSession:
    """An unopened session; the tests open it with `with`."""

    clock = arm_kwargs.pop("clock", ManualClock())
    arm = FakeArm(
        identity=IDENTITY,
        joint_limits=LIMITS,
        clock=clock,
        positions=[0.0, -1.0, -0.1],
        max_velocity_rad_s=5.0,
        **arm_kwargs,
    )
    return RobotSession(arm, parameters(), lease=InMemoryLease("maker-arm-02"), clock=clock)


def policy_factory(controller: str) -> ScriptedPolicy:
    return ScriptedPolicy(
        controller=controller,
        checkpoint="local/checkpoint",
        fps=10.0,
        input_features={key: () for key in ACTION_KEYS},
        action_features=ACTION_KEYS,
        actions=[{key: 0.0 for key in ACTION_KEYS} for _ in range(2)],
    )


def test_a_reached_preset_becomes_a_durable_position_result() -> None:
    with make_session() as session:
        robot = EverestRobot(session)

        result = robot.go_to_known_position("clip_ready")

    assert result.reached
    assert result.position_name == "clip_ready"
    assert result.config_digest == "sha256:test"
    assert result.failure_reason is None
    # The workflow persists this verbatim.
    assert json_dict(result)["max_tracking_error_rad"] is not None


def test_a_failed_move_carries_its_reason_into_the_workflow() -> None:
    with make_session() as session:
        robot = EverestRobot(session)

        result = robot.go_to_known_position("unreachable")

    assert not result.reached
    assert result.failure_reason == "limit_violation"
    assert "shoulder_lift" in str(result.failure_detail)


def test_a_configured_transition_is_used_instead_of_a_direct_move() -> None:
    with make_session() as session:
        robot = EverestRobot(session, transitions={"clip_ready": "to_clip_ready"})

        result = robot.go_to_known_position("clip_ready")

        assert result.reached
        assert any(
            command[1] == pytest.approx(-1.0, abs=0.02) for command in session.port.sent_commands
        )


def test_attachment_reports_the_rollout_outcome_and_no_fabricated_force() -> None:
    with make_session() as session:
        robot = EverestRobot(session, policy_factory=policy_factory, task="attach the clip")
        session.port.enable()

        result = robot.attach_clip("vla")

    assert result.motion_completed
    assert result.controller == "vla"
    # The Maker Arm has no force sensor.
    assert result.force_newtons is None
    assert result.steps == 2
    assert result.termination == str(TerminationReason.COMPLETED)
    assert AttachmentResult(**json_dict(result)) == result


def test_attachment_without_a_policy_refuses_rather_than_pretending() -> None:
    with make_session() as session:
        robot = EverestRobot(session)

        with pytest.raises(NotImplementedError, match="dataset decision"):
            robot.attach_clip("vla")


def test_the_unimplemented_perception_stages_refuse_clearly() -> None:
    with make_session() as session:
        robot = EverestRobot(session)

        with pytest.raises(NotImplementedError, match="scaffold backend"):
            robot.localize_and_pick_up_carabiner("cv", "graspnet")
        with pytest.raises(NotImplementedError, match="scaffold backend"):
            robot.verify_attachment(AttachmentResult(True, "vla"))


def test_the_default_backend_never_touches_hardware() -> None:
    with robot_session({}) as robot:
        assert isinstance(robot, ScaffoldRobot)

    with robot_session({"backend": "scaffold", "verification_failures": 2}) as robot:
        assert robot.verification_failures == 2


def test_an_unknown_backend_is_refused() -> None:
    with (
        pytest.raises(ValueError, match="unknown robot backend"),
        robot_session({"backend": "simulator"}),
    ):
        pass
