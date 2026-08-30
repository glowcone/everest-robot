import json
import math

import pytest

from everest_robot.robot.cameras import CameraRuntime, load_camera_specs
from everest_robot.robot.clock import ManualClock
from everest_robot.robot.contracts import (
    ArmLifecycle,
    FailureReason,
    JointLimit,
    RobotIdentity,
    TerminationReason,
)
from everest_robot.robot.fake_arm import FakeArm
from everest_robot.robot.lerobot_bridge import JointFrame, RobotBridgeCore
from everest_robot.robot.parameters import RobotParameters
from everest_robot.robot.policy import (
    PolicyHandle,
    PolicyRunner,
    ScriptedPolicy,
    compatibility_problems,
)
from everest_robot.robot.recording import InMemorySessionRecorder

JOINTS = ("shoulder_pan", "shoulder_lift", "gripper")
LIMITS = (
    JointLimit("shoulder_pan", -1.0, 1.0),
    JointLimit("shoulder_lift", -2.0, 0.5),
    JointLimit("gripper", -2.0, 0.0),
)
IDENTITY = RobotIdentity("maker-arm-02", "maker-arm-v1", "cal-2026-08-20", JOINTS)
ACTION_KEYS = tuple(f"{name}.pos" for name in JOINTS)
CAMERAS = [
    {"name": "wrist", "kind": "fake", "index_or_path": "0", "width": 8, "height": 4, "fps": 30}
]


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


def make_bridge(*, with_cameras: bool = False, identity: RobotIdentity = IDENTITY, **arm_kwargs):
    clock = arm_kwargs.pop("clock", ManualClock())
    arm = FakeArm(
        identity=identity,
        joint_limits=LIMITS,
        clock=clock,
        positions=[0.0, -1.0, -0.1],
        max_velocity_rad_s=5.0,
        **arm_kwargs,
    )
    cameras = (
        CameraRuntime.from_specs(load_camera_specs(json.dumps(CAMERAS))) if with_cameras else None
    )
    bridge = RobotBridgeCore(arm, cameras=cameras)
    bridge.connect()
    arm.enable()
    return bridge, arm, clock


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


def test_scripted_policy_satisfies_the_handle_protocol() -> None:
    assert isinstance(policy(), PolicyHandle)


# ── compatibility validation ───────────────────────────────────────────────────────
def test_a_compatible_policy_reports_no_problems() -> None:
    bridge, _, _ = make_bridge(with_cameras=True)
    handle = policy(input_features={**{key: () for key in ACTION_KEYS}, "wrist": (4, 8, 3)})

    assert compatibility_problems(handle, bridge, fps=10.0) == ()


def test_a_missing_camera_or_wrong_image_size_is_caught_before_motion() -> None:
    bridge, arm, _ = make_bridge(with_cameras=True)

    missing = policy(input_features={**{key: () for key in ACTION_KEYS}, "overhead": (4, 8, 3)})
    wrong_size = policy(input_features={**{key: () for key in ACTION_KEYS}, "wrist": (224, 224, 3)})

    assert "does not produce" in compatibility_problems(missing, bridge, fps=10.0)[0]
    assert "policy expects (224, 224, 3)" in compatibility_problems(wrong_size, bridge, fps=10.0)[0]
    assert arm.sent_commands == []


def test_an_action_space_mismatch_is_never_adapted_at_runtime() -> None:
    bridge, _, _ = make_bridge()

    reordered = policy(action_features=tuple(reversed(ACTION_KEYS)))
    partial = policy(action_features=ACTION_KEYS[:2])

    assert "action space mismatch" in compatibility_problems(reordered, bridge, fps=10.0)[0]
    assert "action space mismatch" in compatibility_problems(partial, bridge, fps=10.0)[0]


def test_running_a_policy_off_its_training_rate_is_refused() -> None:
    bridge, _, _ = make_bridge()

    problems = compatibility_problems(policy(fps=30.0), bridge, fps=10.0)

    assert "expects 30.0 fps" in problems[0]


def test_a_non_identity_joint_frame_must_be_acknowledged() -> None:
    bridge, arm, _ = make_bridge()
    bridge.frame = JointFrame(JOINTS, offsets_deg=(0.0, 90.0, 0.0))

    assert "joint frame has non-zero offsets" in compatibility_problems(
        policy(), bridge, fps=10.0
    )[0]
    assert compatibility_problems(
        policy(), bridge, fps=10.0, allow_non_identity_frame=True
    ) == ()


def test_an_incompatible_policy_fails_the_run_without_moving() -> None:
    bridge, arm, clock = make_bridge()
    runner = PolicyRunner(bridge, parameters(), clock=clock)

    result = runner.run(policy(fps=30.0))

    assert result.termination is TerminationReason.FAILED
    assert result.failure_reason is FailureReason.SCHEMA_MISMATCH
    assert result.steps == 0
    assert arm.sent_commands == []


def test_a_different_calibration_fails_the_run_without_moving() -> None:
    other = RobotIdentity("maker-arm-02", "maker-arm-v1", "some-other-calibration", JOINTS)
    bridge, arm, clock = make_bridge(identity=other)
    runner = PolicyRunner(bridge, parameters(), clock=clock)

    result = runner.run(policy())

    assert result.failure_reason is FailureReason.IDENTITY_MISMATCH
    assert arm.sent_commands == []


# ── rollout ────────────────────────────────────────────────────────────────────────
def test_a_rollout_sends_every_action_and_records_both_sides() -> None:
    bridge, arm, clock = make_bridge()
    recorder = InMemorySessionRecorder(IDENTITY, fps=10.0, config_digest="sha256:test")
    runner = PolicyRunner(bridge, parameters(), clock=clock, recorder=recorder)
    handle = policy(actions=3)

    result = runner.run(handle, task="attach the clip")

    assert result.termination is TerminationReason.COMPLETED
    assert result.steps == 3
    assert result.failure_reason is None
    assert len(arm.sent_commands) == 3
    assert len(recorder.frames) == 3
    assert recorder.frames[0].requested_action == {key: 0.0 for key in ACTION_KEYS}
    assert recorder.frames[0].sent_action == {key: 0.0 for key in ACTION_KEYS}
    assert result.to_json()["task"] == "attach the clip"


def test_the_rollout_paces_itself_at_the_configured_rate() -> None:
    bridge, _, clock = make_bridge()
    runner = PolicyRunner(bridge, parameters(), clock=clock)

    started = clock.monotonic()
    result = runner.run(policy(actions=5), fps=10.0)

    # Five steps at 10 Hz, and nothing overran its period.
    assert clock.monotonic() - started == pytest.approx(0.5, abs=1e-9)
    assert result.missed_deadlines == 0


def test_a_step_limit_stops_the_rollout() -> None:
    bridge, _, clock = make_bridge()
    runner = PolicyRunner(bridge, parameters(), clock=clock)

    result = runner.run(policy(actions=100), max_steps=4)

    assert result.termination is TerminationReason.MAX_STEPS
    assert result.steps == 4


def test_a_duration_limit_stops_the_rollout() -> None:
    bridge, _, clock = make_bridge()
    runner = PolicyRunner(bridge, parameters(), clock=clock)

    result = runner.run(policy(actions=1000), fps=10.0, max_duration_s=1.0)

    assert result.termination is TerminationReason.MAX_DURATION
    assert result.steps == 10


def test_cancellation_stops_the_rollout_and_holds() -> None:
    bridge, arm, clock = make_bridge()
    seen = {"count": 0}

    def cancel() -> bool:
        seen["count"] += 1
        return seen["count"] > 2

    runner = PolicyRunner(bridge, parameters(), clock=clock, cancel=cancel)

    result = runner.run(policy(actions=100))

    assert result.termination is TerminationReason.CANCELLED
    assert result.steps == 2
    assert arm.lifecycle is ArmLifecycle.ENABLED


def test_actions_past_a_soft_limit_are_clipped_and_reported() -> None:
    bridge, arm, clock = make_bridge()
    runner = PolicyRunner(bridge, parameters(), clock=clock)
    beyond = {
        "shoulder_pan.pos": math.degrees(5.0),
        "shoulder_lift.pos": math.degrees(-0.5),
        "gripper.pos": math.degrees(-0.2),
    }

    result = runner.run(policy(actions=[beyond]))

    assert result.clipped_joints == ("shoulder_pan",)
    assert arm.sent_commands[-1][0] == pytest.approx(1.0)


def test_an_inference_failure_ends_the_rollout_as_a_policy_error() -> None:
    bridge, _, clock = make_bridge()
    runner = PolicyRunner(bridge, parameters(), clock=clock)

    result = runner.run(_exploding())

    assert result.termination is TerminationReason.FAILED
    assert result.failure_reason is FailureReason.POLICY_ERROR
    assert "checkpoint is corrupt" in str(result.failure_detail)


def _exploding() -> PolicyHandle:
    class Exploding:
        controller = "vla"
        checkpoint = "local/checkpoint"
        fps = 10.0
        input_features = {key: () for key in ACTION_KEYS}
        action_features = ACTION_KEYS

        def reset(self) -> None:
            return None

        def select_action(self, observation, task=None):
            raise RuntimeError("checkpoint is corrupt")

    return Exploding()  # type: ignore[return-value]


def test_a_malformed_action_ends_the_rollout_as_a_policy_error() -> None:
    bridge, _, clock = make_bridge()
    runner = PolicyRunner(bridge, parameters(), clock=clock)

    result = runner.run(policy(actions=[{"shoulder_pan.pos": 0.0}]))

    assert result.failure_reason is FailureReason.POLICY_ERROR
    assert "unusable action" in str(result.failure_detail)


def test_a_motor_fault_during_the_rollout_stops_it() -> None:
    bridge, arm, clock = make_bridge(fault_after_commands=2)
    runner = PolicyRunner(bridge, parameters(), clock=clock)

    result = runner.run(policy(actions=10))

    assert result.termination is TerminationReason.FAILED
    assert result.failure_reason is FailureReason.MOTOR_FAULT
    assert arm.lifecycle is ArmLifecycle.FAULT


def test_a_dry_run_validates_and_commands_nothing() -> None:
    bridge, arm, clock = make_bridge()
    runner = PolicyRunner(bridge, parameters(), clock=clock)

    result = runner.run(policy(), dry_run=True)

    assert result.termination is TerminationReason.COMPLETED
    assert result.steps == 0
    assert arm.sent_commands == []


def test_heartbeats_fire_during_a_long_rollout() -> None:
    bridge, _, clock = make_bridge()
    beats = {"count": 0}
    runner = PolicyRunner(
        bridge,
        parameters(),
        clock=clock,
        heartbeat=lambda: beats.__setitem__("count", beats["count"] + 1),
        heartbeat_interval_s=0.5,
    )

    runner.run(policy(actions=30), fps=10.0)

    assert beats["count"] >= 3


def test_a_raising_heartbeat_holds_the_arm_and_abandons_the_episode() -> None:
    """Absurd signals cancellation by raising from ctx.heartbeat()."""

    class Cancelled(BaseException):
        pass

    bridge, arm, clock = make_bridge()
    recorder = InMemorySessionRecorder(IDENTITY)
    beats = {"count": 0}

    def heartbeat() -> None:
        beats["count"] += 1
        if beats["count"] > 2:
            raise Cancelled("run cancelled")

    runner = PolicyRunner(
        bridge,
        parameters(),
        clock=clock,
        recorder=recorder,
        heartbeat=heartbeat,
        heartbeat_interval_s=0.0,
    )

    with pytest.raises(Cancelled):
        runner.run(policy(actions=100))

    assert arm.lifecycle is ArmLifecycle.ENABLED
    held = arm.read_state().positions
    clock.advance(5.0)
    assert arm.read_state().positions == pytest.approx(held)
    assert recorder.frames == []
