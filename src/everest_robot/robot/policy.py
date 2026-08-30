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

:class:`PolicyRunner` owns an uninterrupted rollout. :class:`PolicySession` owns the other
shape ADR-0003 needs: a rollout a state machine advances one action at a time, keeping the
policy's carried state across the decisions it makes in between. Both go through the same
compatibility validation and the same hardware safety boundary.

:func:`load_policy` turns a file into a handle. It is the only place a path becomes a
policy, so the guarded checkpoint loader is reachable from a command line without any layer
above learning what a checkpoint is.

Joint-action policies only. A Cartesian-action policy would need an approved IK processor
and would still have to pass through this same boundary; IK is explicitly out of scope.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
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


class PolicyLoadError(RuntimeError):
    """A policy file could not be turned into a :class:`PolicyHandle`."""


class PolicySessionError(RuntimeError):
    """A policy session was misused, or is not usable against this robot."""


@dataclass(frozen=True, slots=True)
class PolicyStep:
    """What one action from a persistent session did.

    Process-local; never persisted. The durable record of a rollout is
    :class:`~everest_robot.robot.contracts.PolicyRunResult`, and an FSM attempt's durable
    record is its transition trace.

    ``termination`` is ``None`` while the rollout is still running. Any other value means
    this step produced no motion and the session is finished.
    """

    index: int
    sent_action: Mapping[str, float] | None = None
    latency_s: float = 0.0
    missed_deadline: bool = False
    termination: TerminationReason | None = None
    failure_reason: FailureReason | None = None
    failure_detail: str | None = None

    @property
    def commanded(self) -> bool:
        """Whether this step actually sent a joint command."""

        return self.sent_action is not None


class PolicySession:
    """A persistent rollout advanced one action at a time.

    ADR-0003's ``SEARCH_RL`` and ``CLIP_RL`` arbitrate after individual actions, so they
    need a rollout that survives between decisions. :meth:`PolicyRunner.run` is the wrong
    primitive for that: every call resets the policy and its processors, starts a recording
    episode, and holds the arm on return, so calling it with ``max_steps=1`` would discard
    recurrent state and cached action chunks on every action and would not be equivalent to
    a continuous rollout.

    This class deliberately does *not* replace :class:`PolicyRunner`. The two pace
    differently, and the difference is the point:

    * ``PolicyRunner`` paces against an absolute schedule (``start + n / fps``) because it
      owns an uninterrupted rollout, where drift accumulates into running slower than the
      rate the policy was trained at.
    * A session's caller does other physical work between actions -- a detection, a bounded
      CV follow -- so lateness is expected rather than drift. Each deadline is therefore set
      from the moment the previous action went out. A late step is absorbed and reported,
      never made up, for the same reason replay never makes up a late frame: commanding a
      backlog faster drives the arm at a rate nothing validated.

    The session owns compatibility validation, enabling, seeding, per-action timing,
    clipping through the safety boundary, fault detection, heartbeats and cancellation. It
    does not decide anything: the caller reads :class:`PolicyStep` and chooses.
    """

    def __init__(
        self,
        bridge: RobotBridgeCore,
        parameters: RobotParameters,
        handle: PolicyHandle,
        *,
        task: str | None = None,
        fps: float | None = None,
        clock: Clock | None = None,
        heartbeat: Heartbeat | None = None,
        cancel: CancelCheck | None = None,
        heartbeat_interval_s: float = 1.0,
        allow_non_identity_frame: bool = False,
        estop_on_failure: bool = False,
    ) -> None:
        self.bridge = bridge
        self.parameters = parameters
        self.handle = handle
        self.task = task
        self.fps = fps if fps is not None else parameters.policy.fps
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        self.clock = clock or SystemClock()
        self.heartbeat = heartbeat
        self.cancel = cancel
        self.heartbeat_interval_s = heartbeat_interval_s
        self.allow_non_identity_frame = allow_non_identity_frame
        self.estop_on_failure = estop_on_failure

        self.steps = 0
        self.seeds = 0
        self.missed_deadlines = 0
        self.max_step_latency_s = 0.0
        self._seeded = False
        self._finished = False
        self._next_deadline_s: float | None = None
        self._last_heartbeat_s = -math.inf

    @property
    def seeded(self) -> bool:
        return self._seeded

    @property
    def finished(self) -> bool:
        """Whether a terminal step has been returned; :meth:`seed` starts a new rollout."""

        return self._finished

    @property
    def clipped_joints(self) -> tuple[str, ...]:
        return tuple(sorted(self.bridge.clipped_joints))

    # ── lifecycle ──────────────────────────────────────────────────────────────────
    def seed(self) -> None:
        """Validate, enable, and reset the policy for a rollout from where the arm is now.

        Called on every entry to the owning state, not only the first. Classical vision
        physically intervenes between an FSM's visits to a learned state, so cached action
        chunks and recurrent state from before that intervention describe a pose the arm has
        left. Re-seeding is what makes the next action a response to the present.

        :class:`PolicyHandle` has no priming hook, and needs none: :meth:`PolicyHandle.reset`
        drops the carried state, and the next :meth:`step` reads a fresh observation before
        it asks for an action.
        """

        detail = self.parameters.identity.mismatch_detail(self.bridge.port.identity)
        if detail is not None:
            raise PolicySessionError(detail)

        problems = compatibility_problems(
            self.handle,
            self.bridge,
            fps=self.fps,
            allow_non_identity_frame=self.allow_non_identity_frame,
        )
        if problems:
            raise PolicySessionError("; ".join(problems))

        state = self.bridge.port.read_state()
        if state.lifecycle is ArmLifecycle.FAULT:
            raise PolicySessionError(f"arm is in fault: {state.fault_reason}")
        if state.lifecycle is ArmLifecycle.DISCONNECTED:
            raise PolicySessionError("arm is disconnected; open a session first")
        if state.lifecycle is ArmLifecycle.CONNECTED:
            self.bridge.port.enable()

        self.handle.reset()
        self.seeds += 1
        self._seeded = True
        self._finished = False
        # A fresh rollout owes nothing to the previous one's schedule: the first action
        # goes out as soon as it is asked for.
        self._next_deadline_s = None

    def close(self, *, failed: bool = False) -> None:
        """Stop commanding and leave the arm held. Idempotent."""

        self._seeded = False
        self._next_deadline_s = None
        if self.bridge.port.lifecycle is not ArmLifecycle.ENABLED:
            return
        if failed and self.estop_on_failure:
            self.bridge.port.estop()
        else:
            self.bridge.port.hold_current_position()

    def __enter__(self) -> PolicySession:
        self.seed()
        return self

    def __exit__(self, exc_type: object, *_: object) -> None:
        self.close(failed=exc_type is not None)

    # ── one action ─────────────────────────────────────────────────────────────────
    def step(self) -> PolicyStep:
        """Advance the rollout by exactly one action.

        Never raises for a runtime outcome: cancellation, a policy that considers itself
        finished, a motor fault and an unusable action all come back as a terminal
        :class:`PolicyStep`, because the caller has physical state to unwind and is the only
        layer that knows what a given failure means for its state machine.
        """

        if not self._seeded:
            raise PolicySessionError("policy session is not seeded; call seed() first")

        now = self.clock.monotonic()
        self._beat(now)

        if self.cancel is not None and self.cancel():
            return self._terminal(TerminationReason.CANCELLED)

        missed = False
        if self._next_deadline_s is not None:
            if now < self._next_deadline_s:
                self.clock.sleep(self._next_deadline_s - now)
                now = self.clock.monotonic()
            else:
                # The caller spent longer than one control period deciding. Recorded, not
                # fatal, and deliberately not compensated for.
                missed = True
                self.missed_deadlines += 1

        started_s = now
        observation = self.bridge.get_observation()
        try:
            action = self.handle.select_action(observation, self.task)
        except Exception as error:  # noqa: BLE001 - any inference failure ends the rollout
            return self._terminal(
                TerminationReason.FAILED,
                FailureReason.POLICY_ERROR,
                f"select_action failed at step {self.steps}: {error}",
                missed_deadline=missed,
            )

        if action is None:
            return self._terminal(TerminationReason.COMPLETED, missed_deadline=missed)

        try:
            sent = self.bridge.send_action(action)
        except (KeyError, ValueError) as error:
            return self._terminal(
                TerminationReason.FAILED,
                FailureReason.POLICY_ERROR,
                f"policy produced an unusable action at step {self.steps}: {error}",
                missed_deadline=missed,
            )
        except RuntimeError as error:
            return self._terminal(
                TerminationReason.FAILED,
                FailureReason.NOT_ENABLED,
                str(error),
                missed_deadline=missed,
            )

        index = self.steps
        self.steps += 1

        state = self.bridge.port.read_state()
        if state.has_fault:
            return self._terminal(
                TerminationReason.FAILED,
                FailureReason.MOTOR_FAULT,
                state.fault_reason or "motor fault during rollout",
                missed_deadline=missed,
            )

        completed_s = self.clock.monotonic()
        latency = completed_s - started_s
        self.max_step_latency_s = max(self.max_step_latency_s, latency)
        # Absorbed, never made up: the next deadline is one period after this action went
        # out, not one period after where a fixed schedule says it should have.
        self._next_deadline_s = completed_s + 1.0 / self.fps
        return PolicyStep(
            index=index,
            sent_action=sent,
            latency_s=latency,
            missed_deadline=missed,
        )

    # ── helpers ────────────────────────────────────────────────────────────────────
    def _terminal(
        self,
        termination: TerminationReason,
        failure_reason: FailureReason | None = None,
        failure_detail: str | None = None,
        *,
        missed_deadline: bool = False,
    ) -> PolicyStep:
        self._finished = True
        self._seeded = False
        return PolicyStep(
            index=self.steps,
            missed_deadline=missed_deadline,
            termination=termination,
            failure_reason=failure_reason,
            failure_detail=failure_detail,
        )

    def _beat(self, now: float) -> None:
        if self.heartbeat is None:
            return
        if now - self._last_heartbeat_s >= self.heartbeat_interval_s:
            self._last_heartbeat_s = now
            self.heartbeat()


# ── loading a policy from a file ───────────────────────────────────────────────────
#: Suffixes that mean "this is a trained checkpoint", routed to the LeRobot loader.
CHECKPOINT_SUFFIXES = frozenset({".safetensors", ".pt", ".ckpt", ".bin"})

_SCRIPTED_FIELDS = frozenset(
    {"kind", "controller", "checkpoint", "fps", "input_features", "action_features", "actions"}
)


def load_policy(path: str | Path, *, controller: str | None = None) -> PolicyHandle:
    """Load a policy from a file. The single seam where a file becomes a handle.

    Two kinds of file are recognized, and the distinction is deliberate:

    * A trained checkpoint -- a directory, or a file with a checkpoint suffix -- is handed
      to :class:`LeRobotPolicyHandle`, which refuses. Loading one needs the dataset feature
      metadata that maps this robot's ``{joint}.pos`` scalars and camera frames into the
      policy's input tensors, and that metadata comes from the recording/dataset decision
      that has not been made. An invented mapping would mis-order joints against a real
      checkpoint without ever failing, so the refusal is the correct behaviour and this
      function is where it becomes reachable from a command line.
    * A ``.json`` scripted-policy file replays a fixed action sequence through
      :class:`ScriptedPolicy`. This is not a trained policy and never pretends to be. It
      exists so the one-action session, the FSM's gates and the whole hardware path can be
      brought up and rehearsed on a real arm before a checkpoint can be loaded.

    Resolution happens before anything is claimed or energized, so a missing or malformed
    file costs no lease.
    """

    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise PolicyLoadError(f"policy file {str(resolved)!r} does not exist")

    if resolved.is_dir() or resolved.suffix.lower() in CHECKPOINT_SUFFIXES:
        # Raises NotImplementedError, with the reason. Deliberately not caught: a caller
        # that asked for a trained checkpoint must not silently get something else.
        return LeRobotPolicyHandle(resolved, controller=controller)

    if resolved.suffix.lower() == ".json":
        return _load_scripted_policy(resolved, controller=controller)

    raise PolicyLoadError(
        f"cannot tell what kind of policy {str(resolved)!r} is: expected a checkpoint "
        f"directory, a checkpoint file ({', '.join(sorted(CHECKPOINT_SUFFIXES))}), or a "
        f".json scripted-policy file"
    )


def _load_scripted_policy(path: Path, *, controller: str | None) -> ScriptedPolicy:
    """Read a scripted-policy file strictly.

    Strict for the same reason :meth:`~everest_robot.domain.ReplayRequest.from_json` is: a
    misspelled field that silently takes a default would move a real arm along a sequence
    nobody wrote. Every action is checked against the declared action space here rather
    than at send time, so a bad file fails before the robot is claimed.
    """

    try:
        document = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise PolicyLoadError(f"{path}: {error}") from None
    if not isinstance(document, dict):
        raise PolicyLoadError(f"{path}: a scripted policy file must be a JSON object")

    unknown = sorted(set(document) - _SCRIPTED_FIELDS)
    if unknown:
        raise PolicyLoadError(f"{path}: unknown field(s): {', '.join(unknown)}")
    kind = document.get("kind", "scripted")
    if kind != "scripted":
        raise PolicyLoadError(f"{path}: unsupported kind {kind!r} (expected 'scripted')")
    missing = sorted({"controller", "action_features", "actions"} - set(document))
    if missing:
        raise PolicyLoadError(f"{path}: missing field(s): {', '.join(missing)}")

    action_features = tuple(str(name) for name in document["action_features"])
    if not action_features:
        raise PolicyLoadError(f"{path}: action_features must not be empty")
    if len(set(action_features)) != len(action_features):
        raise PolicyLoadError(f"{path}: action_features contains a duplicate")

    raw_actions = document["actions"]
    if not isinstance(raw_actions, list) or not raw_actions:
        raise PolicyLoadError(f"{path}: actions must be a non-empty list")
    expected = set(action_features)
    actions: list[Mapping[str, float]] = []
    for index, entry in enumerate(raw_actions):
        if not isinstance(entry, dict):
            raise PolicyLoadError(f"{path}: action {index} must be a JSON object")
        if set(entry) != expected:
            raise PolicyLoadError(
                f"{path}: action {index} does not match action_features: expected "
                f"{sorted(expected)}, got {sorted(entry)}"
            )
        values = {}
        for name, value in entry.items():
            number = float(value)
            if not math.isfinite(number):
                raise PolicyLoadError(f"{path}: action {index}[{name!r}] must be finite")
            values[name] = number
        actions.append(values)

    features = document.get("input_features")
    if features is None:
        input_features: Mapping[str, tuple[int, ...]] = {name: () for name in action_features}
    elif not isinstance(features, dict):
        raise PolicyLoadError(f"{path}: input_features must be a JSON object")
    else:
        input_features = {
            str(name): tuple(int(axis) for axis in shape) for name, shape in features.items()
        }

    fps = document.get("fps")
    return ScriptedPolicy(
        controller=controller or str(document["controller"]),
        checkpoint=str(document.get("checkpoint") or path),
        fps=None if fps is None else float(fps),
        input_features=input_features,
        action_features=action_features,
        actions=actions,
    )
