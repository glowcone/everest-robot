"""Policy and VLA rollout.

The runner owns everything around inference: compatibility validation before anything is
enabled, control timing and missed-deadline accounting, clipping through the hardware
safety boundary, cancellation and duration/step limits, recording, heartbeats, and a
durable result.

Inference itself sits behind :class:`PolicyHandle`, so the loop is exercised against a
deterministic fake. :class:`LeRobotPolicyHandle` is the real thing: one loader for every
LeRobot checkpoint -- ACT, SmolVLA, anything else registered -- because the architecture is
named by the checkpoint's own config rather than chosen here.

Joint-action policies only. A Cartesian-action policy would need an approved IK processor
and would still have to pass through this same boundary; IK is explicitly out of scope.
"""

from __future__ import annotations

import contextlib
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
from everest_robot.robot.lerobot_bridge import POSITION_SUFFIX, RobotBridgeCore
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


# ── loading a trained checkpoint ───────────────────────────────────────────────────
# The three feature names LeRobot's dataset convention uses. A checkpoint's config is
# expressed in these; this robot's features are ``{joint}.pos`` and bare camera names, and
# the loader below is the translation between the two.
POLICY_STATE_FEATURE = "observation.state"
POLICY_IMAGE_PREFIX = "observation.images."
POLICY_ACTION_FEATURE = "action"

# Policy types whose behaviour is conditioned on a language instruction. Running one with
# no task is not a neutral default: the checkpoint was trained with a sentence in that slot
# and an empty string is a prompt it never saw, so it is refused rather than defaulted.
LANGUAGE_CONDITIONED = ("smolvla", "pi0", "pi05", "pi0_fast", "groot")


class PolicyLoadError(RuntimeError):
    """A checkpoint cannot be used on this arm. Raised before anything is claimed."""


def robot_observation_features(
    state_shape: Sequence[int] | None,
    image_shapes: Mapping[str, Sequence[int]],
    joint_names: Sequence[str],
) -> dict[str, Any]:
    """A checkpoint's observation features, restated in this robot's feature space.

    The point of restating them is that :func:`compatibility_problems` can then compare a
    checkpoint against the robot it is about to drive using the robot's own names, and
    refuse before anything is energized. Two conversions happen here and nowhere else:

    * ``observation.state`` is one flat vector; this robot names each element
      ``{joint}.pos``. A width that is not the joint count is refused -- there is no
      correct way to spread a 7-wide state over 6 joints, and guessing produces a rollout
      that commands the wrong axis without ever raising.
    * ``observation.images.{name}`` is channel-first ``(C, H, W)`` in a policy config,
      while a camera produces ``(H, W, C)``.
    """

    features: dict[str, Any] = {}
    if state_shape is not None:
        shape = tuple(int(value) for value in state_shape)
        if len(shape) != 1 or shape[0] != len(joint_names):
            raise PolicyLoadError(
                f"the checkpoint's state is {shape}, but this arm has {len(joint_names)} "
                f"joints ({', '.join(joint_names)}). A checkpoint trained on a different "
                "arm cannot be reshaped onto this one."
            )
        features.update({f"{name}{POSITION_SUFFIX}": () for name in joint_names})

    for name, raw in image_shapes.items():
        shape = tuple(int(value) for value in raw)
        if len(shape) != 3 or shape[0] not in (1, 3):
            raise PolicyLoadError(
                f"camera feature {name!r} has shape {shape}; expected channel-first "
                "(C, H, W) with 1 or 3 channels"
            )
        features[name] = (shape[1], shape[2], shape[0])
    return features


def robot_action_features(
    action_shape: Sequence[int] | None,
    joint_names: Sequence[str],
    action_names: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """The action keys a checkpoint produces, in the order its output vector carries them.

    ``action_names`` is used when the checkpoint recorded them (the pi0 family does); it is
    returned unchanged so a checkpoint trained on a *differently ordered* arm is caught by
    the action-space check rather than silently permuted. Without them the robot's own
    joint order is assumed, which is the same assumption the dataset frame documents --
    see ``docs/lerobot-frame-reconciliation.md``.
    """

    if action_shape is None:
        raise PolicyLoadError("the checkpoint declares no action feature")
    shape = tuple(int(value) for value in action_shape)
    if len(shape) != 1 or shape[0] != len(joint_names):
        raise PolicyLoadError(
            f"the checkpoint produces a {shape} action, but this arm takes "
            f"{len(joint_names)} joint positions ({', '.join(joint_names)})"
        )
    if action_names is not None:
        if len(action_names) != shape[0]:
            raise PolicyLoadError(
                f"the checkpoint names {len(action_names)} action features for a {shape} "
                "action vector"
            )
        return tuple(str(name) for name in action_names)
    return tuple(f"{name}{POSITION_SUFFIX}" for name in joint_names)


class LeRobotPolicyHandle:
    """A trained LeRobot checkpoint, loaded with its own processors.

    Generic across architectures on purpose. The checkpoint's config names its own policy
    class (``act``, ``smolvla``, or anything else registered with LeRobot), and its own
    pre/post-processor pipelines carry the normalization statistics it was trained with, so
    a new architecture loads through this same path without a code change. What *is*
    checked here is the interface -- state width, action width, camera shapes, and the
    language slot -- because that is what actually differs between a checkpoint and this
    arm, and it is checked at load time, before the robot is claimed.

    Units are the bridge's: observations arrive as ``{joint}.pos`` degrees plus camera
    frames, and the returned action is in the same space.
    ``PolicyRunner``/``RobotBridgeCore`` own the clipping into soft limits; nothing here
    talks to a motor.

    ``select_action`` never returns ``None``: a LeRobot policy has no notion of being
    finished, so a rollout ends on the runner's step, duration or cancellation limits.
    """

    def __init__(
        self,
        checkpoint: str,
        joint_names: Sequence[str],
        *,
        task: str | None = None,
        fps: float | None = None,
        device: str | None = None,
        revision: str | None = None,
        robot_type: str | None = None,
    ) -> None:
        # Imported here, not at module scope: everything above this class runs without the
        # hardware extra, and importing LeRobot pulls in torch.
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors
        from lerobot.utils.feature_utils import hw_to_dataset_features

        self._checkpoint = str(checkpoint)
        self._joint_names = tuple(str(name) for name in joint_names)
        self.task = task
        self.robot_type = robot_type
        self._fps = fps

        try:
            config = PreTrainedConfig.from_pretrained(self._checkpoint, revision=revision)
        except Exception as error:  # noqa: BLE001 - a missing repo, a bad path, a bad config
            raise PolicyLoadError(
                f"could not read a policy config from {self._checkpoint!r}: {error}"
            ) from error
        if device is not None:
            config.device = device
        self._config = config
        self._controller = config.type

        if self._controller in LANGUAGE_CONDITIONED and not (task or "").strip():
            raise PolicyLoadError(
                f"{self._controller} is conditioned on a language instruction; pass the "
                "task it was trained with rather than running it on an empty prompt"
            )

        inputs = dict(config.input_features or {})
        state = inputs.pop(POLICY_STATE_FEATURE, None)
        images = {
            key[len(POLICY_IMAGE_PREFIX) :]: feature.shape
            for key, feature in inputs.items()
            if key.startswith(POLICY_IMAGE_PREFIX)
        }
        unsupported = sorted(
            key for key in inputs if not key.startswith(POLICY_IMAGE_PREFIX)
        )
        if unsupported:
            # Anything else -- an environment state, a reward, a second modality -- would
            # have to be produced by the robot, and this bridge produces joints and images.
            raise PolicyLoadError(
                f"the checkpoint needs observation feature(s) {', '.join(unsupported)}, "
                "which this robot does not produce"
            )

        outputs = dict(config.output_features or {})
        action = outputs.get(POLICY_ACTION_FEATURE)
        self._input_features = robot_observation_features(
            None if state is None else state.shape, images, self._joint_names
        )
        self._action_features = robot_action_features(
            None if action is None else action.shape,
            self._joint_names,
            getattr(config, "action_feature_names", None),
        )

        # The dataset feature metadata LeRobot assembles an inference frame against. It is
        # derived from what the checkpoint asked for, so the frame is by construction the
        # shape the model was trained on; whether *this* robot can supply it is
        # `compatibility_problems`' question, asked before the arm is enabled.
        hardware = {
            key: (float if key.endswith(POSITION_SUFFIX) else value)
            for key, value in self._input_features.items()
        }
        self._observation_ds = hw_to_dataset_features(hardware, "observation", use_video=False)
        self._action_ds = hw_to_dataset_features(
            dict.fromkeys(self._action_features, float), POLICY_ACTION_FEATURE
        )

        try:
            policy_class = get_policy_class(self._controller)
            self._policy = policy_class.from_pretrained(
                self._checkpoint, config=config, revision=revision
            )
            self._preprocessor, self._postprocessor = make_pre_post_processors(
                config, pretrained_path=self._checkpoint, pretrained_revision=revision
            )
        except PolicyLoadError:
            raise
        except Exception as error:  # noqa: BLE001 - a missing weight file, an absent extra
            raise PolicyLoadError(
                f"could not load the {self._controller} checkpoint at "
                f"{self._checkpoint!r}: {error}"
            ) from error

        self._torch = torch
        self._device = torch.device(config.device)

    # ── what the runner validates against ──────────────────────────────────────────
    @property
    def controller(self) -> str:
        return self._controller

    @property
    def checkpoint(self) -> str:
        return self._checkpoint

    @property
    def fps(self) -> float | None:
        return self._fps

    @property
    def input_features(self) -> Mapping[str, tuple[int, ...]]:
        return dict(self._input_features)

    @property
    def action_features(self) -> tuple[str, ...]:
        return self._action_features

    # ── inference ──────────────────────────────────────────────────────────────────
    def reset(self) -> None:
        """Clear the policy's and the processors' caches between rollouts.

        Action-chunking policies (ACT, SmolVLA) hold a queue of future actions; carrying
        one across rollouts would command the arm from an observation it no longer has.
        """

        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()

    def select_action(
        self, observation: Mapping[str, Any], task: str | None = None
    ) -> Mapping[str, float]:
        from lerobot.policies.utils import build_inference_frame, make_robot_action

        instruction = task if task is not None else self.task
        frame = build_inference_frame(
            dict(observation),
            self._device,
            self._observation_ds,
            instruction,
            self.robot_type,
        )
        torch = self._torch
        with (
            torch.inference_mode(),
            torch.autocast(device_type=self._device.type)
            if self._device.type == "cuda" and self._config.use_amp
            else contextlib.nullcontext(),
        ):
            action = self._postprocessor(self._policy.select_action(self._preprocessor(frame)))
        return make_robot_action(action, self._action_ds)
