"""In-process finite-state orchestrator for carabiner attachment.

The FSM owns decisions and budgets, while :class:`AttachmentFSMHandlers` owns perception,
policy state, and physical actions. There is intentionally no Absurd dependency here: one
runner invocation is one continuously claimed physical attempt.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Protocol


class AttachmentState(StrEnum):
    INITIAL = "initial"
    SEARCH_RL = "search_rl"
    SEARCH_CV = "search_cv"
    CLIP_RL = "clip_rl"
    SUCCESS = "success"
    FAILED = "failed"
    ABORTED = "aborted"


ACTIVE_STATES = frozenset(
    {
        AttachmentState.INITIAL,
        AttachmentState.SEARCH_RL,
        AttachmentState.SEARCH_CV,
        AttachmentState.CLIP_RL,
    }
)


@dataclass(frozen=True, slots=True)
class InitialObservation:
    """The motion-free decision made from the initial coherent observation."""

    already_attached: bool
    carabiner_detected: bool
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class SearchRLStep:
    """Result after exactly one learned search action and a fresh CV check."""

    carabiner_detected: bool
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class SearchCVStep:
    """Result after one bounded visual-following action."""

    target_visible: bool
    followed: bool
    confidence: float | None = None
    pixel_error: float | None = None


@dataclass(frozen=True, slots=True)
class ClipRLStep:
    """Result after one learned grasp/clip action and fresh recovery checks."""

    attachment_verified: bool
    returned_to_neutral: bool = False
    carabiner_visible: bool = False
    alignment_degraded: bool = False
    carabiner_grasped: bool = False
    confidence: float | None = None


class AttachmentFSMHandlers(Protocol):
    """Integration boundary; each step method may perform at most one physical action."""

    def enter_state(
        self, state: AttachmentState, previous: AttachmentState | None
    ) -> None: ...

    def observe_initial(self) -> InitialObservation: ...

    def search_rl_step(self) -> SearchRLStep: ...

    def search_cv_step(self) -> SearchCVStep: ...

    def clip_rl_step(self) -> ClipRLStep: ...

    def hold(self, reason: str) -> None: ...


class AttachmentAbort(RuntimeError):
    """A handled cancellation or safety stop that should terminate as ``ABORTED``."""


@dataclass(frozen=True, slots=True)
class AttachmentFSMConfig:
    max_total_actions: int = 1_000
    max_duration_s: float = 180.0
    max_search_rl_actions: int = 400
    max_search_cv_actions: int = 300
    max_clip_rl_actions: int = 400

    def __post_init__(self) -> None:
        for name in (
            "max_total_actions",
            "max_search_rl_actions",
            "max_search_cv_actions",
            "max_clip_rl_actions",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if not math.isfinite(self.max_duration_s) or self.max_duration_s <= 0:
            raise ValueError("max_duration_s must be finite and positive")

    def state_budget(self, state: AttachmentState) -> int:
        return {
            AttachmentState.SEARCH_RL: self.max_search_rl_actions,
            AttachmentState.SEARCH_CV: self.max_search_cv_actions,
            AttachmentState.CLIP_RL: self.max_clip_rl_actions,
        }[state]


@dataclass(frozen=True, slots=True)
class StateTransition:
    source: AttachmentState
    destination: AttachmentState
    reason: str
    action_index: int
    result: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AttachmentFSMResult:
    final_state: AttachmentState
    reason: str
    actions: int
    elapsed_s: float
    resets: int
    state_actions: Mapping[str, int]
    transitions: tuple[StateTransition, ...]

    @property
    def succeeded(self) -> bool:
        return self.final_state is AttachmentState.SUCCESS

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class AttachmentFSM:
    """Run one bounded attachment attempt against injected state handlers."""

    def __init__(
        self,
        handlers: AttachmentFSMHandlers,
        config: AttachmentFSMConfig | None = None,
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.handlers = handlers
        self.config = config or AttachmentFSMConfig()
        self.monotonic = monotonic

    def run(self) -> AttachmentFSMResult:
        started = self.monotonic()
        state = AttachmentState.INITIAL
        actions = 0
        action_states = ACTIVE_STATES - {AttachmentState.INITIAL}
        counts = {candidate.value: 0 for candidate in action_states}
        cycle_counts = dict(counts)
        resets = 0
        transitions: list[StateTransition] = []

        try:
            self.handlers.enter_state(state, None)

            while state in ACTIVE_STATES:
                if state is AttachmentState.INITIAL:
                    initial = self.handlers.observe_initial()
                    if initial.already_attached:
                        next_state = AttachmentState.SUCCESS
                        reason = "attachment already verified"
                    elif initial.carabiner_detected:
                        next_state = AttachmentState.SEARCH_CV
                        reason = "carabiner initially detected"
                    else:
                        next_state = AttachmentState.SEARCH_RL
                        reason = "carabiner not initially detected"
                    state = self._transition(
                        state, next_state, reason, actions, initial, transitions, enter=True
                    )
                    continue

                elapsed = self.monotonic() - started
                if actions >= self.config.max_total_actions:
                    state = self._transition(
                        state,
                        AttachmentState.FAILED,
                        "total action budget exhausted",
                        actions,
                        None,
                        transitions,
                    )
                    break
                if elapsed >= self.config.max_duration_s:
                    state = self._transition(
                        state,
                        AttachmentState.FAILED,
                        "wall-clock budget exhausted",
                        actions,
                        None,
                        transitions,
                    )
                    break
                if cycle_counts[state.value] >= self.config.state_budget(state):
                    state = self._transition(
                        state,
                        AttachmentState.FAILED,
                        f"{state.value} action budget exhausted",
                        actions,
                        None,
                        transitions,
                    )
                    break

                result, next_state, reason = self._step(state)
                actions += 1
                counts[state.value] += 1
                cycle_counts[state.value] += 1
                if next_state is not state:
                    if next_state is AttachmentState.INITIAL:
                        cycle_counts = {candidate.value: 0 for candidate in action_states}
                        resets += 1
                    state = self._transition(
                        state, next_state, reason, actions, result, transitions, enter=True
                    )

        except AttachmentAbort as error:
            state = self._transition(
                state, AttachmentState.ABORTED, str(error), actions, None, transitions
            )
        except BaseException:
            self.handlers.hold("unhandled exception")
            raise

        reason = transitions[-1].reason if transitions else "no transition"
        if state is not AttachmentState.SUCCESS:
            self.handlers.hold(reason)
        return AttachmentFSMResult(
            final_state=state,
            reason=reason,
            actions=actions,
            elapsed_s=self.monotonic() - started,
            resets=resets,
            state_actions=dict(sorted(counts.items())),
            transitions=tuple(transitions),
        )

    def _step(
        self, state: AttachmentState
    ) -> tuple[object, AttachmentState, str]:
        if state is AttachmentState.SEARCH_RL:
            result = self.handlers.search_rl_step()
            if result.carabiner_detected:
                return result, AttachmentState.SEARCH_CV, "carabiner detected"
            return result, state, "search policy continuing"

        if state is AttachmentState.SEARCH_CV:
            result = self.handlers.search_cv_step()
            if not result.target_visible:
                return result, AttachmentState.SEARCH_RL, "CV lost carabiner"
            if result.followed:
                return result, AttachmentState.CLIP_RL, "CV followed carabiner"
            return result, state, "CV following carabiner"

        if state is AttachmentState.CLIP_RL:
            result = self.handlers.clip_rl_step()
            if result.returned_to_neutral:
                return result, AttachmentState.INITIAL, "policy returned to neutral"
            if result.attachment_verified:
                return result, AttachmentState.SUCCESS, "attachment verified"
            if result.carabiner_visible and result.alignment_degraded:
                return result, AttachmentState.SEARCH_CV, "alignment degraded"
            if not result.carabiner_visible and not result.carabiner_grasped:
                return result, AttachmentState.SEARCH_RL, "ungrasped carabiner lost"
            return result, state, "clip policy continuing"

        raise AssertionError(f"no step handler for {state}")

    def _transition(
        self,
        source: AttachmentState,
        destination: AttachmentState,
        reason: str,
        action_index: int,
        result: object | None,
        transitions: list[StateTransition],
        *,
        enter: bool = False,
    ) -> AttachmentState:
        payload = asdict(result) if result is not None else {}
        transitions.append(
            StateTransition(source, destination, reason, action_index, payload)
        )
        if enter and destination in ACTIVE_STATES:
            self.handlers.enter_state(destination, source)
        return destination
