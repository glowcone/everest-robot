"""The one-action policy primitive ADR-0003's learned states are built on."""

import json
import math

import pytest

from everest_robot.robot.clock import ManualClock
from everest_robot.robot.contracts import (
    ArmLifecycle,
    FailureReason,
    JointLimit,
    RobotIdentity,
    TerminationReason,
)
from everest_robot.robot.fake_arm import FakeArm
from everest_robot.robot.lerobot_bridge import RobotBridgeCore
from everest_robot.robot.parameters import RobotParameters
from everest_robot.robot.policy import (
    PolicyLoadError,
    PolicySession,
    PolicySessionError,
    ScriptedPolicy,
    load_policy,
)

JOINTS = ("shoulder_pan", "shoulder_lift", "gripper")
LIMITS = (
    JointLimit("shoulder_pan", -1.0, 1.0),
    JointLimit("shoulder_lift", -2.0, 0.5),
    JointLimit("gripper", -2.0, 0.0),
)
IDENTITY = RobotIdentity("maker-arm-02", "maker-arm-v1", "cal-2026-08-20", JOINTS)
ACTION_KEYS = tuple(f"{name}.pos" for name in JOINTS)


def parameters() -> RobotParameters:
    document = {
        "schema_version": 1,
        "robot": {
            "id": "maker-arm-02",
            "model": "maker-arm-v1",
            "calibration_id": "cal-2026-08-20",
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
        "named_positions": {},
        "named_transitions": {},
        "policy": {"default_controller": "vla", "fps": 10, "max_duration_s": 5.0},
        "replay": {
            "require_matching_robot_id": True,
            "require_matching_calibration_id": True,
            "safe_start_position": None,
            "max_speed_scale": 1.0,
        },
    }
    return RobotParameters.from_mapping(document, config_digest="sha256:test", source="test.yaml")


def make_bridge(*, clock: ManualClock, **arm_kwargs):
    arm = FakeArm(
        identity=IDENTITY,
        joint_limits=LIMITS,
        clock=clock,
        positions=[0.0, -1.0, -0.1],
        max_velocity_rad_s=5.0,
        **arm_kwargs,
    )
    bridge = RobotBridgeCore(arm)
    bridge.connect()
    arm.enable()
    return bridge, arm


def policy(actions: int | list = 3, **overrides) -> ScriptedPolicy:
    """Build a scripted policy from either a step count or an explicit action list."""

    if isinstance(actions, int):
        actions = [{key: 0.0 for key in ACTION_KEYS} for _ in range(actions)]
    defaults = dict(
        controller="vla",
        checkpoint="local/checkpoint",
        fps=10.0,
        input_features={key: () for key in ACTION_KEYS},
        action_features=ACTION_KEYS,
        actions=actions,
    )
    defaults.update(overrides)
    return ScriptedPolicy(**defaults)  # type: ignore[arg-type]


def make_session(
    handle=None, *, clock=None, **overrides
) -> tuple[PolicySession, object, ManualClock]:
    clock = clock or ManualClock()
    bridge, arm = make_bridge(clock=clock)
    session = PolicySession(
        bridge,
        parameters(),
        handle if handle is not None else policy(),
        clock=clock,
        **overrides,
    )
    return session, arm, clock


def act(value: float) -> dict[str, float]:
    return {key: value for key in ACTION_KEYS}


def commanded_degrees(arm) -> list[float]:
    """The first joint of every command the arm received, back in the policy's units."""

    return [math.degrees(command[0]) for command in arm.sent_commands]


def count_holds(arm) -> dict[str, int]:
    """Count hold_current_position() calls; FakeArm keeps no counter of its own."""

    seen = {"count": 0}
    original = arm.hold_current_position

    def counting() -> bool:
        seen["count"] += 1
        return original()

    arm.hold_current_position = counting
    return seen


# ── seeding ────────────────────────────────────────────────────────────────────────
def test_seeding_resets_the_policy_and_enables_the_arm() -> None:
    handle = policy(2)
    handle.step = 1
    session, arm, _ = make_session(handle)
    arm.disable()

    session.seed()

    assert handle.step == 0
    assert session.seeded and session.seeds == 1
    assert arm.lifecycle is ArmLifecycle.ENABLED


def test_an_incompatible_policy_is_refused_before_any_motion() -> None:
    session, arm, _ = make_session(policy(action_features=tuple(reversed(ACTION_KEYS))))

    with pytest.raises(PolicySessionError, match="action space mismatch"):
        session.seed()

    assert arm.sent_commands == []
    assert not session.seeded


def test_a_faulted_arm_is_refused_before_any_motion() -> None:
    session, arm, _ = make_session()
    arm.inject_fault("bus off")

    with pytest.raises(PolicySessionError, match="bus off"):
        session.seed()

    assert arm.sent_commands == []


def test_stepping_before_seeding_is_a_misuse_not_a_rollout() -> None:
    session, arm, _ = make_session()

    with pytest.raises(PolicySessionError, match="not seeded"):
        session.step()

    assert arm.sent_commands == []


# ── one action at a time ───────────────────────────────────────────────────────────
def test_each_step_commands_exactly_one_action() -> None:
    session, arm, _ = make_session(policy([act(0.1), act(0.2)]))
    session.seed()

    first = session.step()

    assert commanded_degrees(arm) == pytest.approx([0.1])
    assert first.commanded and first.index == 0 and first.termination is None
    # sent_action reports what the safety boundary actually commanded, not what the policy
    # asked for: the gripper's soft limit is 0.0 rad, so 0.1 deg is clipped on the way out.
    assert first.sent_action["shoulder_pan.pos"] == pytest.approx(0.1)
    assert first.sent_action["gripper.pos"] == pytest.approx(0.0)
    assert session.clipped_joints == ("gripper",)

    second = session.step()

    assert len(arm.sent_commands) == 2
    assert second.index == 1
    assert session.steps == 2


def test_carried_policy_state_survives_between_steps() -> None:
    """The property that makes this a session rather than repeated one-step rollouts.

    A fresh ``PolicyRunner.run(max_steps=1)`` would reset the handle every call and replay
    the first action forever, which is exactly what ADR-0003 forbids.
    """

    handle = policy([act(0.1), act(0.2), act(0.3)])
    session, arm, _ = make_session(handle)
    session.seed()

    session.step()
    session.step()
    session.step()

    assert commanded_degrees(arm) == pytest.approx([0.1, 0.2, 0.3])
    assert handle.step == 3


def test_re_seeding_restarts_the_policy_without_re_enabling_the_arm() -> None:
    handle = policy([act(0.1), act(0.2)])
    session, arm, _ = make_session(handle)
    session.seed()
    session.step()

    session.seed()
    session.step()

    assert commanded_degrees(arm) == pytest.approx([0.1, 0.1])
    assert session.seeds == 2
    assert arm.lifecycle is ArmLifecycle.ENABLED


# ── terminal steps ─────────────────────────────────────────────────────────────────
def test_a_finished_policy_terminates_without_commanding() -> None:
    session, arm, _ = make_session(policy([act(0.1)]))
    session.seed()
    session.step()

    done = session.step()

    assert done.termination is TerminationReason.COMPLETED
    assert not done.commanded
    assert len(arm.sent_commands) == 1
    assert session.finished and not session.seeded


def test_cancellation_terminates_before_the_next_action() -> None:
    cancelled = False
    session, arm, _ = make_session(cancel=lambda: cancelled)
    session.seed()
    session.step()
    cancelled = True

    step = session.step()

    assert step.termination is TerminationReason.CANCELLED
    assert len(arm.sent_commands) == 1


def test_a_motor_fault_during_a_step_is_reported_not_raised() -> None:
    clock = ManualClock()
    bridge, arm = make_bridge(clock=clock, fault_after_commands=1)
    session = PolicySession(bridge, parameters(), policy(3), clock=clock)
    session.seed()

    step = session.step()

    assert step.termination is TerminationReason.FAILED
    assert step.failure_reason is FailureReason.MOTOR_FAULT


def test_an_unusable_action_ends_the_session_with_the_offending_step() -> None:
    session, _, _ = make_session(policy([{"nope.pos": 0.0}]))
    session.seed()

    step = session.step()

    assert step.termination is TerminationReason.FAILED
    assert step.failure_reason is FailureReason.POLICY_ERROR
    assert "unusable action at step 0" in step.failure_detail


def test_an_inference_failure_ends_the_session() -> None:
    class Exploding:
        controller, checkpoint, fps = "vla", "local/checkpoint", 10.0
        input_features = {key: () for key in ACTION_KEYS}
        action_features = ACTION_KEYS

        def reset(self) -> None: ...

        def select_action(self, observation, task=None):
            raise RuntimeError("no weights")

    session, arm, _ = make_session(Exploding())
    session.seed()

    step = session.step()

    assert step.failure_reason is FailureReason.POLICY_ERROR
    assert "no weights" in step.failure_detail
    assert arm.sent_commands == []


# ── pacing ─────────────────────────────────────────────────────────────────────────
def test_actions_are_paced_at_the_policy_rate() -> None:
    session, _, clock = make_session(policy(2))
    session.seed()

    session.step()
    started = clock.monotonic()
    session.step()

    # 10 fps in the shared test parameters: the session waits out the period itself.
    assert clock.monotonic() - started == pytest.approx(0.1)
    assert session.missed_deadlines == 0


def test_a_late_caller_is_absorbed_and_never_made_up() -> None:
    """A caller that spends a period deciding is reported late, not sped up.

    The FSM does a detection between actions, so lateness is expected rather than drift.
    Commanding a backlog faster would drive the arm at a rate nothing validated.
    """

    session, _, clock = make_session(policy(3))
    session.seed()
    session.step()

    clock.advance(5.0)
    late = session.step()
    started = clock.monotonic()
    session.step()

    assert late.missed_deadline and session.missed_deadlines == 1
    # The step after a late one waits a full period from when the late action went out,
    # rather than firing immediately to catch a fixed schedule back up.
    assert clock.monotonic() - started == pytest.approx(0.1)


# ── teardown ───────────────────────────────────────────────────────────────────────
def test_closing_holds_the_arm_where_it_is() -> None:
    session, arm, _ = make_session()
    session.seed()
    session.step()
    holds = count_holds(arm)

    session.close()

    assert holds["count"] == 1
    assert not session.seeded


def test_the_context_manager_holds_even_when_the_body_raises() -> None:
    session, arm, _ = make_session()
    holds = count_holds(arm)

    with pytest.raises(ZeroDivisionError), session:
        session.step()
        raise ZeroDivisionError

    assert holds["count"] == 1


# ── loading a policy from a file ───────────────────────────────────────────────────
def scripted_document(**overrides) -> dict:
    document = {
        "controller": "search-v0",
        "fps": 10.0,
        "action_features": list(ACTION_KEYS),
        "actions": [act(0.1), act(0.2)],
    }
    document.update(overrides)
    return document


def write(tmp_path, name="search.json", **overrides):
    path = tmp_path / name
    path.write_text(json.dumps(scripted_document(**overrides)))
    return path


def test_a_scripted_policy_file_loads_into_a_usable_handle(tmp_path) -> None:
    handle = load_policy(write(tmp_path))

    assert handle.controller == "search-v0"
    assert handle.action_features == ACTION_KEYS
    assert handle.select_action({}) == act(0.1)


def test_a_loaded_scripted_policy_drives_a_session(tmp_path) -> None:
    session, arm, _ = make_session(load_policy(write(tmp_path)))
    session.seed()
    session.step()
    session.step()

    assert commanded_degrees(arm) == pytest.approx([0.1, 0.2])


def test_a_misspelled_field_is_rejected_rather_than_defaulted(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({**scripted_document(), "actionz": []}))

    with pytest.raises(PolicyLoadError, match="unknown field"):
        load_policy(path)


def test_an_action_that_does_not_match_the_action_space_fails_at_load(tmp_path) -> None:
    path = write(tmp_path, actions=[{ACTION_KEYS[0]: 0.0}])

    with pytest.raises(PolicyLoadError, match="action 0 does not match action_features"):
        load_policy(path)


def test_a_non_finite_action_fails_at_load(tmp_path) -> None:
    path = tmp_path / "nan.json"
    # json.dumps writes a bare NaN token and json.loads reads it back, so a file like this
    # reaches the loader intact and has to be refused there.
    path.write_text(json.dumps(scripted_document(actions=[act(float("nan"))])))

    with pytest.raises(PolicyLoadError, match="must be finite"):
        load_policy(path)


def test_a_missing_policy_file_fails_before_anything_is_claimed(tmp_path) -> None:
    with pytest.raises(PolicyLoadError, match="does not exist"):
        load_policy(tmp_path / "absent.json")


def test_an_unrecognized_extension_says_what_is_supported(tmp_path) -> None:
    path = tmp_path / "policy.yaml"
    path.write_text("{}")

    with pytest.raises(PolicyLoadError, match="cannot tell what kind of policy"):
        load_policy(path)


@pytest.mark.parametrize("name", ["checkpoint.safetensors", "checkpoint.pt"])
def test_a_weights_file_points_at_the_directory_that_holds_it(tmp_path, name) -> None:
    """A checkpoint is a directory: the config and the processor pipelines are part of it."""

    path = tmp_path / name
    path.write_bytes(b"")

    with pytest.raises(PolicyLoadError, match="loaded from the directory"):
        load_policy(path)


def test_a_directory_without_a_config_is_not_a_checkpoint(tmp_path) -> None:
    with pytest.raises(PolicyLoadError, match="missing config.json"):
        load_policy(tmp_path)


def test_an_unknown_reference_is_not_mistaken_for_a_repo_id(tmp_path) -> None:
    """'namespace/name' is a hub reference; a plain missing path says so instead."""

    with pytest.raises(PolicyLoadError, match="does not exist"):
        load_policy(tmp_path / "no-such-checkpoint")
