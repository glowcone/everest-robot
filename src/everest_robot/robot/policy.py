"""Policy and VLA rollout.

The runner owns everything around inference: compatibility validation before anything is
enabled, control timing and missed-deadline accounting, clipping through the hardware
safety boundary, cancellation and duration/step limits, recording, heartbeats, and a
durable result.

Inference itself sits behind :class:`PolicyHandle`, so the loop is exercised against a
deterministic fake. :class:`LeRobotPolicyHandle` -- the real checkpoint loader -- is a
guarded stub: LeRobot builds a model-ready observation through ``build_inference_frame``,
which needs the dataset feature metadata that the deferred recording/dataset decision will
settle. Guessing that mapping would produce glue that fails silently against a real
checkpoint, which is worse than refusing.

Joint-action policies only. A Cartesian-action policy would need an approved IK processor
and would still have to pass through this same boundary; IK is explicitly out of scope.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from everest_robot.robot.clock import Clock, SystemClock
from everest_robot.robot.contracts import (
    ArmLifecycle,
    CancelCheck,
    FailureReason,
    Heartbeat,
    PolicyRunResult,
    TerminationReason,
)
from everest_robot.robot.lerobot_bridge import RobotBridgeCore
from everest_robot.robot.parameters import RobotParameters
from everest_robot.robot.recording import NullSessionRecorder, SessionRecorder


@runtime_checkable
class PolicyHandle(Protocol):
    """A loaded policy, already wrapped in its pre/post-processing.

    ``select_action`` returns an action in the robot's action-feature space (the same
    ``{joint}.pos`` keys, in degrees), or ``None`` to signal the policy considers the task
    finished.
    """

    @property
    def controller(self) -> str: ...

    @property
    def checkpoint(self) -> str: ...

    @property
    def fps(self) -> float | None: ...

    @property
    def input_features(self) -> Mapping[str, tuple[int, ...]]: ...

    @property
    def action_features(self) -> tuple[str, ...]: ...

    def reset(self) -> None: ...

    def select_action(
        self, observation: Mapping[str, Any], task: str | None = None
    ) -> Mapping[str, float] | None: ...


def compatibility_problems(
    handle: PolicyHandle,
    bridge: RobotBridgeCore,
    *,
    fps: float,
    allow_non_identity_frame: bool = False,
) -> tuple[str, ...]:
    """Everything that would make this policy the wrong one for this robot.

    Checked before the arm is enabled. A policy whose action space does not match the
    robot's exactly is not adapted at runtime: silently dropping or reordering joints is
    how a rollout ends up commanding the wrong axis.
    """

    problems: list[str] = []
    observation_features = bridge.observation_features

    for name, shape in handle.input_features.items():
        if name not in observation_features:
            problems.append(f"policy needs observation {name!r}, which the robot does not produce")
            continue
        expected = observation_features[name]
        if isinstance(expected, tuple) and tuple(shape) != tuple(expected):
            problems.append(
                f"observation {name!r} is {tuple(expected)} on this robot, policy expects "
                f"{tuple(shape)}"
            )

    robot_actions = tuple(bridge.action_features)
    if tuple(handle.action_features) != robot_actions:
        problems.append(
            f"action space mismatch: policy produces {list(handle.action_features)}, robot "
            f"accepts {list(robot_actions)}"
        )

    if handle.fps is not None and abs(handle.fps - fps) > 1e-6:
        # Running a policy off its training rate changes the dynamics it learned.
        problems.append(f"policy expects {handle.fps} fps, this rollout is configured for {fps}")

    if not bridge.frame.is_identity and not allow_non_identity_frame:
        problems.append(
            "the LeRobot joint frame has non-zero offsets; confirm the checkpoint was "
            "trained in the same frame and pass allow_non_identity_frame=True"
        )

    return tuple(problems)


@dataclass
class _Telemetry:
    steps: int = 0
    missed_deadlines: int = 0
    max_step_latency_s: float = 0.0


class PolicyRunner:
    """Runs one synchronous rollout and reports what physically happened.

    Assumes it already owns a connected arm parked at a safe start pose -- getting there is
    :class:`~everest_robot.robot.motion.JointMotionController`'s job, and the workflow
    sequences the two.
    """

    def __init__(
        self,
        bridge: RobotBridgeCore,
        parameters: RobotParameters,
        *,
        clock: Clock | None = None,
        heartbeat: Heartbeat | None = None,
        cancel: CancelCheck | None = None,
        recorder: SessionRecorder | None = None,
        heartbeat_interval_s: float = 1.0,
        allow_non_identity_frame: bool = False,
        estop_on_failure: bool = False,
    ) -> None:
        self.bridge = bridge
        self.parameters = parameters
        self.clock = clock or SystemClock()
        self.heartbeat = heartbeat
        self.cancel = cancel
        self.recorder = recorder or NullSessionRecorder(parameters.identity)
        self.heartbeat_interval_s = heartbeat_interval_s
        self.allow_non_identity_frame = allow_non_identity_frame
        self.estop_on_failure = estop_on_failure
        self._last_heartbeat_s = -math.inf

    def run(
        self,
        handle: PolicyHandle,
        *,
        task: str | None = None,
        fps: float | None = None,
        max_steps: int | None = None,
        max_duration_s: float | None = None,
        dry_run: bool = False,
    ) -> PolicyRunResult:
        rate = fps if fps is not None else self.parameters.policy.fps
        duration_limit = (
            max_duration_s if max_duration_s is not None else self.parameters.policy.max_duration_s
        )
        if rate <= 0:
            raise ValueError(f"fps must be positive, got {rate}")

        telemetry = _Telemetry()
        started_s = self.clock.monotonic()

        detail = self.parameters.identity.mismatch_detail(self.bridge.port.identity)
        if detail is not None:
            return self._result(
                handle, task, rate, telemetry, 0.0,
                TerminationReason.FAILED, FailureReason.IDENTITY_MISMATCH, detail,
            )

        problems = compatibility_problems(
            handle,
            self.bridge,
            fps=rate,
            allow_non_identity_frame=self.allow_non_identity_frame,
        )
        if problems:
            return self._result(
                handle, task, rate, telemetry, 0.0,
                TerminationReason.FAILED, FailureReason.SCHEMA_MISMATCH, "; ".join(problems),
            )
        if dry_run:
            return self._result(
                handle, task, rate, telemetry, 0.0, TerminationReason.COMPLETED, None, None
            )

        state = self.bridge.port.read_state()
        if state.lifecycle is ArmLifecycle.FAULT:
            return self._result(
                handle, task, rate, telemetry, 0.0, TerminationReason.FAILED,
                FailureReason.MOTOR_FAULT, f"arm is in fault: {state.fault_reason}",
            )
        if state.lifecycle is ArmLifecycle.DISCONNECTED:
            return self._result(
                handle, task, rate, telemetry, 0.0, TerminationReason.FAILED,
                FailureReason.NOT_ENABLED, "arm is disconnected; open a session first",
            )
        if state.lifecycle is ArmLifecycle.CONNECTED:
            self.bridge.port.enable()

        # Every rollout starts from a clean policy and processor state; carrying an action
        # chunk over from a previous rollout would command the arm from stale context.
        handle.reset()
        self.recorder.start_episode(task)

        period = 1.0 / rate
        try:
            termination, failure, failure_detail = self._rollout(
                handle, task, period, duration_limit, max_steps, telemetry, started_s
            )
        except BaseException:
            # A heartbeat raises when the workflow run is cancelled, and an interrupt can
            # arrive anywhere. Neither may leave a moving arm behind, and a half-written
            # episode is not a recording.
            self._safe_stop(failed=True)
            self.recorder.abort()
            raise

        self._safe_stop(failed=failure is not None)

        episode = self.recorder.finish_episode()
        return self._result(
            handle, task, rate, telemetry, self.clock.monotonic() - started_s,
            termination, failure, failure_detail, episode_id=episode.session_id,
        )

    # ── the loop ───────────────────────────────────────────────────────────────────
    def _rollout(
        self,
        handle: PolicyHandle,
        task: str | None,
        period: float,
        duration_limit: float,
        max_steps: int | None,
        telemetry: _Telemetry,
        started_s: float,
    ) -> tuple[TerminationReason, FailureReason | None, str | None]:
        while True:
            step_started_s = self.clock.monotonic()
            elapsed = step_started_s - started_s
            self._beat(step_started_s)

            if self.cancel is not None and self.cancel():
                return TerminationReason.CANCELLED, None, None
            if elapsed >= duration_limit:
                return TerminationReason.MAX_DURATION, None, None
            if max_steps is not None and telemetry.steps >= max_steps:
                return TerminationReason.MAX_STEPS, None, None

            observation = self.bridge.get_observation()
            try:
                action = handle.select_action(observation, task)
            except Exception as error:  # noqa: BLE001 - any inference failure ends the rollout
                return (
                    TerminationReason.FAILED,
                    FailureReason.POLICY_ERROR,
                    f"select_action failed at step {telemetry.steps}: {error}",
                )
            if action is None:
                return TerminationReason.COMPLETED, None, None

            try:
                sent = self.bridge.send_action(action)
            except (KeyError, ValueError) as error:
                return (
                    TerminationReason.FAILED,
                    FailureReason.POLICY_ERROR,
                    f"policy produced an unusable action at step {telemetry.steps}: {error}",
                )
            except RuntimeError as error:
                return TerminationReason.FAILED, FailureReason.NOT_ENABLED, str(error)

            self.recorder.record_frame(observation, dict(action), sent, step_started_s)
            telemetry.steps += 1

            state = self.bridge.port.read_state()
            if state.has_fault:
                return (
                    TerminationReason.FAILED,
                    FailureReason.MOTOR_FAULT,
                    state.fault_reason or "motor fault during rollout",
                )

            now = self.clock.monotonic()
            telemetry.max_step_latency_s = max(telemetry.max_step_latency_s, now - step_started_s)
            # Pace against an absolute deadline rather than sleeping a delta: sleeping
            # `period - latency` each step accumulates drift, so a long rollout silently
            # runs slower than the rate the policy was trained at.
            deadline = started_s + telemetry.steps * period
            if now > deadline:
                # Recorded, not fatal: a policy that occasionally overruns still produces a
                # usable rollout, but a run full of misses was not executed at its rate.
                telemetry.missed_deadlines += 1
            else:
                self.clock.sleep(deadline - now)

    # ── helpers ────────────────────────────────────────────────────────────────────
    def _safe_stop(self, *, failed: bool) -> None:
        """Hold the arm, or e-stop after a failure if the deployment asked for that."""

        if self.bridge.port.lifecycle is not ArmLifecycle.ENABLED:
            return
        if failed and self.estop_on_failure:
            self.bridge.port.estop()
        else:
            self.bridge.port.hold_current_position()

    def _beat(self, now: float) -> None:
        if self.heartbeat is None:
            return
        if now - self._last_heartbeat_s >= self.heartbeat_interval_s:
            self._last_heartbeat_s = now
            self.heartbeat()

    def _result(
        self,
        handle: PolicyHandle,
        task: str | None,
        fps: float,
        telemetry: _Telemetry,
        elapsed_s: float,
        termination: TerminationReason,
        failure_reason: FailureReason | None,
        failure_detail: str | None,
        episode_id: str | None = None,
    ) -> PolicyRunResult:
        state = self.bridge.port.read_state()
        return PolicyRunResult(
            controller=handle.controller,
            checkpoint=handle.checkpoint,
            task=task,
            fps=fps,
            steps=telemetry.steps,
            elapsed_s=elapsed_s,
            missed_deadlines=telemetry.missed_deadlines,
            max_step_latency_s=telemetry.max_step_latency_s,
            termination=termination,
            joint_names=self.bridge.port.joint_names,
            final_joints=state.positions,
            robot_id=self.parameters.identity.robot_id,
            calibration_id=self.parameters.identity.calibration_id,
            config_digest=self.parameters.config_digest,
            clipped_joints=tuple(sorted(self.bridge.clipped_joints)),
            episode_id=episode_id,
            failure_reason=failure_reason,
            failure_detail=failure_detail,
        )


@dataclass
class ScriptedPolicy:
    """A policy that replays a fixed action sequence, then reports it is done.

    For tests and for exercising the rollout path without a checkpoint.
    """

    controller: str
    checkpoint: str
    fps: float | None
    input_features: Mapping[str, tuple[int, ...]]
    action_features: tuple[str, ...]
    actions: Sequence[Mapping[str, float]]
    step: int = 0

    def reset(self) -> None:
        self.step = 0

    def select_action(
        self, observation: Mapping[str, Any], task: str | None = None
    ) -> Mapping[str, float] | None:
        if self.step >= len(self.actions):
            return None
        action = self.actions[self.step]
        self.step += 1
        return action


class LeRobotPolicyHandle:
    """Placeholder for the LeRobot checkpoint loader.

    What it will do, once the dataset decision lands: resolve the config with
    ``PreTrainedConfig.from_pretrained``, build the policy through
    ``get_policy_class(cfg.type).from_pretrained``, load the checkpoint's own processor
    pipelines with ``make_pre_post_processors(cfg, pretrained_path=...)``, and run each
    step through ``predict_action``.

    What it cannot do yet: build the model-ready observation. LeRobot assembles that with
    ``build_inference_frame``, which needs the dataset feature metadata describing how the
    robot's ``{joint}.pos`` scalars and camera frames map into the policy's input tensors.
    That metadata comes from the stored-session format still being decided, and an
    invented mapping would mis-order joints against a real checkpoint without failing.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "LeRobot checkpoint loading is not implemented yet: it needs the dataset "
            "feature metadata that the deferred recording/dataset work will define. "
            "PolicyRunner itself is complete and runs against any PolicyHandle."
        )
