"""ADR-0003's learned states, driven by the real FSM over a fake arm.

These exercise the whole path the hardware backend takes -- a policy loaded from a file, a
persistent one-action session per learned state, and the FSM's gates -- with only
perception scripted, because perception is the one part that is still a stub.
"""

import json

import pytest

from everest_robot.adapters import (
    AttachmentPerception,
    EverestAttachmentFSMHandlers,
    UnavailableAttachmentPerception,
    attachment_fsm_handlers,
)
from everest_robot.attachment_fsm import (
    AttachmentAbort,
    AttachmentFSM,
    AttachmentFSMConfig,
    AttachmentState,
    ClipRLStep,
    InitialObservation,
    SearchCVStep,
    SearchRLStep,
)
from everest_robot.robot.clock import ManualClock
from everest_robot.robot.contracts import ArmLifecycle
from everest_robot.robot.fake_arm import FakeArm
from everest_robot.robot.lease import InMemoryLease
from everest_robot.robot.policy import PolicyLoadError, PolicySessionError, load_policy
from everest_robot.robot.session import RobotSession
from robot.test_policy_session import ACTION_KEYS, IDENTITY, LIMITS, act, parameters


class ScriptedPerception:
    """The gate signals, queued. Stands in for the detector and the attachment checks."""

    def __init__(self, detections=(), clips=(), initial=None) -> None:
        self.detections = list(detections)
        self.clips = list(clips)
        self.detection_calls = 0
        self.clip_calls = 0
        self.initial = initial or InitialObservation(False, False)

    def preflight(self) -> None: ...

    def carabiner_detection(self) -> SearchRLStep:
        self.detection_calls += 1
        return self.detections.pop(0) if self.detections else SearchRLStep(False)

    def initial_observation(self) -> InitialObservation:
        return self.initial

    def clip_observations(self) -> ClipRLStep:
        self.clip_calls += 1
        return self.clips.pop(0) if self.clips else ClipRLStep(False, carabiner_grasped=True)


def write_policy(tmp_path, name, steps=8, value=0.05):
    path = tmp_path / name
    path.write_text(
        json.dumps(
            {
                "controller": name.removesuffix(".json"),
                "fps": 10.0,
                "action_features": list(ACTION_KEYS),
                "actions": [act(value) for _ in range(steps)],
            }
        )
    )
    return path


def make_handlers(tmp_path, perception=None, **overrides):
    clock = ManualClock()
    arm = FakeArm(
        identity=IDENTITY,
        joint_limits=LIMITS,
        clock=clock,
        positions=[0.0, -1.0, -0.1],
        max_velocity_rad_s=5.0,
    )
    session = RobotSession(
        arm,
        parameters(),
        lease=InMemoryLease(IDENTITY.robot_id),
        clock=clock,
    ).open()
    neutral_position = overrides.pop("neutral_position", tuple(arm.positions))
    handlers = EverestAttachmentFSMHandlers(
        session,
        load_policy(write_policy(tmp_path, "search.json")),
        load_policy(write_policy(tmp_path, "clip.json")),
        perception=perception or ScriptedPerception(),
        neutral_position=neutral_position,
        **overrides,
    )
    return handlers, arm, session, clock


# ── the learned states, one action at a time ───────────────────────────────────────
def test_entering_a_learned_state_seeds_only_that_state(tmp_path) -> None:
    handlers, _, session, _ = make_handlers(tmp_path)

    handlers.enter_state(AttachmentState.SEARCH_RL, AttachmentState.INITIAL)

    assert handlers.policy_session_for(AttachmentState.SEARCH_RL).seeded
    assert not handlers.policy_session_for(AttachmentState.CLIP_RL).seeded
    session.close()


def test_initial_is_passive_and_runs_readiness_before_perception(tmp_path) -> None:
    perception = ScriptedPerception(initial=InitialObservation(False, True, 0.9))
    handlers, arm, session, _ = make_handlers(tmp_path, perception)

    observation = handlers.observe_initial()

    assert observation == InitialObservation(False, True, 0.9)
    assert handlers.last_readiness.ready
    assert arm.lifecycle is ArmLifecycle.CONNECTED
    assert arm.sent_commands == []
    session.close()


def test_search_and_clip_keep_separate_carried_state(tmp_path) -> None:
    """ADR-0003: CV physically intervenes between them, so they are separate sessions."""

    handlers, _, session, _ = make_handlers(
        tmp_path, ScriptedPerception(detections=[SearchRLStep(False), SearchRLStep(True)])
    )
    handlers.enter_state(AttachmentState.SEARCH_RL, None)
    handlers.search_rl_step()
    handlers.search_rl_step()
    handlers.enter_state(AttachmentState.CLIP_RL, AttachmentState.SEARCH_CV)
    handlers.clip_rl_step()

    assert handlers.policy_session_for(AttachmentState.SEARCH_RL).steps == 2
    assert handlers.policy_session_for(AttachmentState.CLIP_RL).steps == 1
    session.close()


def test_one_search_step_is_one_action_then_one_detection(tmp_path) -> None:
    perception = ScriptedPerception(detections=[SearchRLStep(True, confidence=0.9)])
    handlers, arm, session, _ = make_handlers(tmp_path, perception)
    handlers.enter_state(AttachmentState.SEARCH_RL, None)

    step = handlers.search_rl_step()

    assert len(arm.sent_commands) == 1
    assert perception.detection_calls == 1
    assert step == SearchRLStep(True, confidence=0.9)
    session.close()


def test_re_entering_a_learned_state_reseeds_it(tmp_path) -> None:
    handlers, _, session, _ = make_handlers(tmp_path)
    rollout = handlers.policy_session_for(AttachmentState.SEARCH_RL)
    handlers.enter_state(AttachmentState.SEARCH_RL, None)
    handlers.search_rl_step()

    handlers.enter_state(AttachmentState.SEARCH_RL, AttachmentState.SEARCH_CV)

    assert rollout.seeds == 2
    assert handlers.search_policy.step == 0
    session.close()


def test_a_policy_returning_to_neutral_is_reported_without_asking_perception(tmp_path) -> None:
    """ADR-0003 gives the neutral signal precedence over a same-step verification.

    This asserts the *mapping* -- a finished policy is reported as a return to neutral --
    and not that a real checkpoint behaves that way. It cannot: a scripted policy exhibits
    whatever this test asserts, which is exactly why ADR-0003 marks the assumption as
    needing hardware verification.
    """

    perception = ScriptedPerception()
    handlers, _, session, _ = make_handlers(
        tmp_path,
        perception,
        neutral_position=tuple(act(0.05).values()),
    )
    handlers.enter_state(AttachmentState.CLIP_RL, None)
    for _ in range(8):
        handlers.clip_rl_step()

    step = handlers.clip_rl_step()

    assert step == ClipRLStep(attachment_verified=False, returned_to_neutral=True)
    assert perception.clip_calls == 8
    session.close()


def test_a_search_policy_that_finishes_without_a_detection_aborts(tmp_path) -> None:
    handlers, _, session, _ = make_handlers(tmp_path)
    handlers.enter_state(AttachmentState.SEARCH_RL, None)
    for _ in range(8):
        handlers.search_rl_step()

    with pytest.raises(AttachmentAbort, match="without detecting the carabiner"):
        handlers.search_rl_step()
    session.close()


def test_a_motor_fault_in_a_learned_state_aborts_the_attempt(tmp_path) -> None:
    handlers, arm, session, _ = make_handlers(tmp_path)
    handlers.enter_state(AttachmentState.CLIP_RL, None)
    arm.fault_after_commands = 1

    with pytest.raises(AttachmentAbort, match="motor_fault"):
        handlers.clip_rl_step()
    session.close()


def test_an_incompatible_policy_is_refused_on_state_entry(tmp_path) -> None:
    path = tmp_path / "wrong.json"
    path.write_text(
        json.dumps(
            {
                "controller": "wrong",
                "fps": 10.0,
                "action_features": list(reversed(ACTION_KEYS)),
                "actions": [act(0.0)],
            }
        )
    )
    handlers, arm, session, _ = make_handlers(tmp_path)
    handlers.clip_policy = load_policy(path)
    handlers.__post_init__()

    with pytest.raises(PolicySessionError, match="action space mismatch"):
        handlers.enter_state(AttachmentState.CLIP_RL, None)

    assert arm.sent_commands == []
    session.close()


# ── the whole state machine over a fake arm ────────────────────────────────────────
def test_the_fsm_drives_both_learned_states_to_success(tmp_path) -> None:
    perception = ScriptedPerception(
        detections=[SearchRLStep(False), SearchRLStep(True, confidence=0.9)],
        clips=[
            ClipRLStep(False, carabiner_grasped=True),
            ClipRLStep(True, carabiner_grasped=True, confidence=0.95),
        ],
    )
    handlers, arm, session, _ = make_handlers(tmp_path, perception)
    # INITIAL and SEARCH_CV are the two states this change does not implement; scripting
    # them here is what lets the learned states be exercised end to end.
    handlers.observe_initial = lambda: InitialObservation(False, False)
    handlers.search_cv_step = lambda: SearchCVStep(True, True)

    result = AttachmentFSM(handlers, AttachmentFSMConfig(max_total_actions=20)).run()

    assert result.final_state is AttachmentState.SUCCESS
    assert [transition.destination for transition in result.transitions] == [
        AttachmentState.SEARCH_RL,
        AttachmentState.SEARCH_CV,
        AttachmentState.CLIP_RL,
        AttachmentState.SUCCESS,
    ]
    # Two search actions and two clip actions physically happened; CV is scripted here.
    assert len(arm.sent_commands) == 4
    assert result.state_actions["search_rl"] == 2
    assert result.state_actions["clip_rl"] == 2
    session.close()


# ── the refusing default, and where it refuses ─────────────────────────────────────
def test_the_default_perception_names_the_missing_subsystem() -> None:
    with pytest.raises(NotImplementedError, match="carabiner detector"):
        UnavailableAttachmentPerception().preflight()


def test_unavailable_perception_satisfies_the_protocol() -> None:
    assert isinstance(UnavailableAttachmentPerception(), AttachmentPerception)


def test_the_hardware_backend_refuses_a_missing_policy_file_before_claiming() -> None:
    with pytest.raises(ValueError, match="search_policy, clip_policy"), attachment_fsm_handlers(
        {"backend": "hardware"}
    ):
        pass


def test_perception_is_checked_before_the_robot_is_claimed(tmp_path, monkeypatch) -> None:
    """A gate that cannot be read costs no lease and no energized arm to discover."""

    def explode(*args, **kwargs):
        raise AssertionError("the robot must not be claimed before the gates are checked")

    monkeypatch.setattr("everest_robot.robot.deployment.open_session", explode)
    params = {
        "backend": "hardware",
        "search_policy": str(write_policy(tmp_path, "search.json")),
        "clip_policy": str(write_policy(tmp_path, "clip.json")),
    }

    with pytest.raises(
        NotImplementedError, match="attachment perception"
    ), attachment_fsm_handlers(params):
        pass


def test_missing_named_neutral_is_refused_before_robot_claim(tmp_path, monkeypatch) -> None:
    def explode(*args, **kwargs):
        raise AssertionError("the robot must not be claimed without a measured neutral pose")

    monkeypatch.setattr("everest_robot.robot.deployment.open_session", explode)
    params = {
        "backend": "hardware",
        "search_policy": str(write_policy(tmp_path, "search.json")),
        "clip_policy": str(write_policy(tmp_path, "clip.json")),
    }

    with pytest.raises(ValueError, match="operator-captured neutral"), attachment_fsm_handlers(
        params, perception=ScriptedPerception()
    ):
        pass


def test_a_bad_policy_file_costs_no_claim(tmp_path, monkeypatch) -> None:
    def explode(*args, **kwargs):
        raise AssertionError("the robot must not be claimed before the policies load")

    monkeypatch.setattr("everest_robot.robot.deployment.open_session", explode)
    params = {
        "backend": "hardware",
        "search_policy": str(tmp_path / "absent.json"),
        "clip_policy": str(write_policy(tmp_path, "clip.json")),
    }

    with pytest.raises(PolicyLoadError, match="does not exist"), attachment_fsm_handlers(params):
        pass


def test_holding_leaves_the_arm_enabled_and_stationary(tmp_path) -> None:
    handlers, arm, session, _ = make_handlers(tmp_path)
    handlers.enter_state(AttachmentState.CLIP_RL, None)
    handlers.clip_rl_step()

    handlers.hold("budget exhausted")

    assert arm.lifecycle is ArmLifecycle.ENABLED
    session.close()
