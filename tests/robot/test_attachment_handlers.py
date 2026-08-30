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
from everest_robot.robot.carabiner_perception import PerceptionUnavailable
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
        self.initial = initial or InitialObservation(False, False)
        self.detection_calls = 0
        self.clip_calls = 0
        self.entered: list[AttachmentState] = []

    def preflight(self) -> None: ...

    def enter_state(self, state: AttachmentState, previous) -> None:
        del previous
        self.entered.append(state)

    def initial_observation(self) -> InitialObservation:
        return self.initial

    def carabiner_detection(self) -> SearchRLStep:
        self.detection_calls += 1
        return self.detections.pop(0) if self.detections else SearchRLStep(False)

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
    # SEARCH_CV needs a camera and a calibrated map; scripting it here is what lets the
    # learned states be exercised end to end. INITIAL comes from the perception.
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


def test_initial_is_the_perceptions_motion_free_observation(tmp_path) -> None:
    """INITIAL commands nothing: it is one look, and the FSM branches on it."""

    perception = ScriptedPerception(initial=InitialObservation(False, True, confidence=None))
    handlers, arm, session, _ = make_handlers(tmp_path, perception)

    observation = handlers.observe_initial()

    assert observation.carabiner_detected is True
    assert observation.already_attached is False
    assert arm.sent_commands == []
    session.close()


def test_an_already_attached_carabiner_is_reported_from_initial(tmp_path) -> None:
    perception = ScriptedPerception(initial=InitialObservation(True, False))
    handlers, arm, session, _ = make_handlers(tmp_path, perception)

    result = AttachmentFSM(handlers, AttachmentFSMConfig(max_total_actions=5)).run()

    assert result.final_state is AttachmentState.SUCCESS
    assert arm.sent_commands == []
    session.close()


def test_state_entry_reaches_the_perception_so_it_can_drop_per_cycle_state(tmp_path) -> None:
    """The alignment baseline belongs to one approach; the FSM's entry is what ends it."""

    perception = ScriptedPerception()
    handlers, _, session, _ = make_handlers(tmp_path, perception)

    handlers.enter_state(AttachmentState.CLIP_RL, AttachmentState.SEARCH_CV)

    assert perception.entered == [AttachmentState.CLIP_RL]
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
    monkeypatch.delenv("EVEREST_CAMERAS", raising=False)
    monkeypatch.delenv("EVEREST_CAMERAS_FILE", raising=False)
    params = {
        "backend": "hardware",
        "search_policy": str(write_policy(tmp_path, "search.json")),
        "clip_policy": str(write_policy(tmp_path, "clip.json")),
    }

    with pytest.raises(
        PerceptionUnavailable, match="not configured"
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


def test_skipping_cv_demands_no_pixel_map(tmp_path, monkeypatch) -> None:
    """No state servos on the fit, so requiring one would refuse over nothing."""

    import dataclasses
    import datetime

    from everest_robot.robot.parameters import NamedPosition

    def explode(*args, **kwargs):
        raise AssertionError("the pixel map must not be read when SEARCH_CV is routed around")

    class Claimed(RuntimeError):
        pass

    def claim(*args, **kwargs):
        raise Claimed("preflight got as far as the robot claim")

    base = parameters()
    neutral = NamedPosition(
        name="neutral",
        joints=(0.0, -1.0, -0.1),
        calibration_id=base.identity.calibration_id,
        approved_by="test",
        captured_at=datetime.date(2026, 8, 30),
        profile=base.motion_defaults,
    )
    monkeypatch.setattr(
        "everest_robot.robot.deployment.load_parameters",
        lambda *a, **k: dataclasses.replace(base, named_positions={"neutral": neutral}),
    )
    monkeypatch.setattr("everest_robot.robot.deployment.load_pixel_map", explode)
    monkeypatch.setattr("everest_robot.robot.deployment.open_session", claim)
    params = {
        "backend": "hardware",
        "skip_cv": True,
        "search_policy": str(write_policy(tmp_path, "search.json")),
        "clip_policy": str(write_policy(tmp_path, "clip.json")),
    }

    with pytest.raises(Claimed), attachment_fsm_handlers(
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


# ── choosing which camera SEARCH_CV closes its loop on ─────────────────────────────
def wrist_calibration(**overrides):
    """A wrist servo calibration for this arm, stamped to match ``IDENTITY``."""

    import numpy as np

    from everest_robot.pixel_map import RobotStamp
    from everest_robot.robot.wrist_servo import BumpTrial, WristServoCalibration

    servo = ("shoulder_pan", "shoulder_lift")
    goal = (320.0, 240.0, 150.0, 10.0)
    settings = dict(
        robot=RobotStamp(IDENTITY.robot_id, IDENTITY.calibration_id),
        camera_name="wrist",
        joint_names=tuple(IDENTITY.joint_names),
        servo_joints=servo,
        goal=goal,
        jacobian=np.array([[-220.0, 10.0], [8.0, 180.0], [1.0, 40.0], [0.5, 0.0]]),
        trials=(BumpTrial("shoulder_pan", 0.08, goal, goal),),
    )
    settings.update(overrides)
    return WristServoCalibration(**settings)


def refused(tmp_path, **overrides):
    """Build handlers that are expected to refuse, releasing the claim either way.

    ``make_handlers`` leaves the session open for the test to drive. These cases raise
    inside the constructor, after the session is open, so the lease has to be released
    here or the next test in the process finds the robot busy.
    """

    clock = ManualClock()
    arm = FakeArm(
        identity=IDENTITY, joint_limits=LIMITS, clock=clock, positions=[0.0, -1.0, -0.1]
    )
    session = RobotSession(
        arm, parameters(), lease=InMemoryLease(IDENTITY.robot_id), clock=clock
    ).open()
    try:
        return EverestAttachmentFSMHandlers(
            session,
            load_policy(write_policy(tmp_path, "search.json")),
            load_policy(write_policy(tmp_path, "clip.json")),
            perception=ScriptedPerception(),
            neutral_position=tuple(arm.positions),
            **overrides,
        )
    finally:
        session.close()


def test_supplying_both_calibrations_is_refused_rather_than_silently_ranked(tmp_path) -> None:
    """Two calibrations commanding one arm with nothing arbitrating between them is not a
    configuration to pick a winner from; it is one nobody meant to be in."""

    from everest_robot.pixel_map import PixelMapError

    with pytest.raises(PixelMapError, match="They are alternatives"):
        refused(tmp_path, calibration=object(), wrist_servo=wrist_calibration())


def test_a_wrist_calibration_taught_on_another_arm_is_refused_before_anything_moves(
    tmp_path,
) -> None:
    from everest_robot.pixel_map import RobotStamp
    from everest_robot.robot.wrist_servo import WristServoError

    with pytest.raises(WristServoError, match="Re-teach it"):
        refused(tmp_path, wrist_servo=wrist_calibration(robot=RobotStamp("maker-arm-99", "x")))


def test_a_wrist_calibration_naming_an_unconfigured_camera_is_refused(tmp_path) -> None:
    """The camera check needs the session -- the runtime only exists once the robot is
    claimed -- but it still lands before anything is energized."""

    from everest_robot.robot.wrist_servo import WristServoError

    with pytest.raises(WristServoError, match="not configured"):
        refused(tmp_path, wrist_servo=wrist_calibration(camera_name="wrist"))


def test_search_cv_without_any_calibration_names_both_ways_to_get_one(tmp_path) -> None:
    from everest_robot.pixel_map import PixelMapError

    handlers, _, session, _ = make_handlers(tmp_path)

    with pytest.raises(PixelMapError, match="robot-wrist-servo teach"):
        handlers.search_cv_step()
    session.close()
