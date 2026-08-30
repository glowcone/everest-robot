from collections import deque
from dataclasses import dataclass, field

import pytest

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


@dataclass
class ScriptedHandlers:
    initial: InitialObservation = InitialObservation(False, False)
    search: deque[SearchRLStep] = field(
        default_factory=lambda: deque([SearchRLStep(True)])
    )
    cv: deque[SearchCVStep] = field(
        default_factory=lambda: deque([SearchCVStep(True, True)])
    )
    clip: deque[ClipRLStep] = field(
        default_factory=lambda: deque([ClipRLStep(True, carabiner_grasped=True)])
    )
    entered: list[tuple[AttachmentState, AttachmentState | None]] = field(
        default_factory=list
    )
    holds: list[str] = field(default_factory=list)
    cv_calls: int = 0

    def enter_state(
        self, state: AttachmentState, previous: AttachmentState | None
    ) -> None:
        self.entered.append((state, previous))

    def observe_initial(self) -> InitialObservation:
        return self.initial

    def search_rl_step(self) -> SearchRLStep:
        return self.search.popleft()

    def search_cv_step(self) -> SearchCVStep:
        self.cv_calls += 1
        return self.cv.popleft()

    def clip_rl_step(self) -> ClipRLStep:
        return self.clip.popleft()

    def hold(self, reason: str) -> None:
        self.holds.append(reason)


def destinations(result):
    return [transition.destination for transition in result.transitions]


def test_initial_observation_can_finish_without_motion():
    handlers = ScriptedHandlers(initial=InitialObservation(True, True))

    result = AttachmentFSM(handlers).run()

    assert result.final_state is AttachmentState.SUCCESS
    assert result.actions == 0
    assert destinations(result) == [AttachmentState.SUCCESS]
    assert handlers.entered == [(AttachmentState.INITIAL, None)]


def test_initial_detection_skips_search_policy():
    handlers = ScriptedHandlers(initial=InitialObservation(False, True))

    result = AttachmentFSM(handlers).run()

    assert result.succeeded
    assert result.actions == 2
    assert destinations(result) == [
        AttachmentState.SEARCH_CV,
        AttachmentState.CLIP_RL,
        AttachmentState.SUCCESS,
    ]


def test_nominal_path_searches_follows_and_clips():
    handlers = ScriptedHandlers()

    result = AttachmentFSM(handlers).run()

    assert result.succeeded
    assert result.actions == 3
    assert destinations(result) == [
        AttachmentState.SEARCH_RL,
        AttachmentState.SEARCH_CV,
        AttachmentState.CLIP_RL,
        AttachmentState.SUCCESS,
    ]
    assert result.state_actions == {"clip_rl": 1, "search_cv": 1, "search_rl": 1}


def test_cv_loss_returns_to_rl_search():
    handlers = ScriptedHandlers(
        search=deque([SearchRLStep(True), SearchRLStep(True)]),
        cv=deque([SearchCVStep(False, False), SearchCVStep(True, True)]),
    )

    result = AttachmentFSM(handlers).run()

    assert result.succeeded
    assert destinations(result) == [
        AttachmentState.SEARCH_RL,
        AttachmentState.SEARCH_CV,
        AttachmentState.SEARCH_RL,
        AttachmentState.SEARCH_CV,
        AttachmentState.CLIP_RL,
        AttachmentState.SUCCESS,
    ]


@pytest.mark.parametrize(
    ("clip_result", "recovery"),
    [
        (
            ClipRLStep(
                attachment_verified=False,
                carabiner_visible=True,
                alignment_degraded=True,
            ),
            AttachmentState.SEARCH_CV,
        ),
        (ClipRLStep(attachment_verified=False), AttachmentState.SEARCH_RL),
    ],
)
def test_clip_recovery_uses_observed_carabiner_state(clip_result, recovery):
    handlers = ScriptedHandlers(
        search=deque([SearchRLStep(True), SearchRLStep(True)]),
        cv=deque([SearchCVStep(True, True), SearchCVStep(True, True)]),
        clip=deque([clip_result, ClipRLStep(True, carabiner_grasped=True)]),
    )

    result = AttachmentFSM(handlers).run()

    assert result.succeeded
    assert recovery in destinations(result)


def test_policy_return_to_neutral_resets_through_initial():
    handlers = ScriptedHandlers(
        initial=InitialObservation(False, True),
        clip=deque(
            [
                ClipRLStep(
                    attachment_verified=True,
                    returned_to_neutral=True,
                    carabiner_grasped=True,
                )
            ]
        ),
    )
    observations = iter(
        [InitialObservation(False, True), InitialObservation(True, False)]
    )
    handlers.observe_initial = lambda: next(observations)  # type: ignore[method-assign]

    result = AttachmentFSM(handlers).run()

    assert result.succeeded
    assert result.resets == 1
    assert result.actions == 2
    assert destinations(result) == [
        AttachmentState.SEARCH_CV,
        AttachmentState.CLIP_RL,
        AttachmentState.INITIAL,
        AttachmentState.SUCCESS,
    ]
    assert handlers.entered.count((AttachmentState.INITIAL, AttachmentState.CLIP_RL)) == 1


# ── the loop with visual following turned off ──────────────────────────────────────
def test_skipping_cv_hands_a_detection_straight_to_the_clip_policy():
    handlers = ScriptedHandlers()
    config = AttachmentFSMConfig(use_search_cv=False)

    result = AttachmentFSM(handlers, config).run()

    assert result.succeeded
    assert result.actions == 2
    assert destinations(result) == [
        AttachmentState.SEARCH_RL,
        AttachmentState.CLIP_RL,
        AttachmentState.SUCCESS,
    ]
    assert result.state_actions == {"clip_rl": 1, "search_cv": 0, "search_rl": 1}
    assert not handlers.cv_calls


def test_skipping_cv_also_bypasses_it_out_of_initial():
    handlers = ScriptedHandlers(initial=InitialObservation(False, True))
    config = AttachmentFSMConfig(use_search_cv=False)

    result = AttachmentFSM(handlers, config).run()

    assert result.succeeded
    assert result.actions == 1
    assert destinations(result) == [AttachmentState.CLIP_RL, AttachmentState.SUCCESS]
    assert handlers.entered == [
        (AttachmentState.INITIAL, None),
        (AttachmentState.CLIP_RL, AttachmentState.INITIAL),
    ]
    assert not handlers.cv_calls


def test_a_degraded_alignment_stays_with_the_clip_policy_when_cv_is_off():
    """Nothing else re-establishes the approach, so the FSM does not bounce to search."""

    handlers = ScriptedHandlers(
        initial=InitialObservation(False, True),
        clip=deque(
            [
                ClipRLStep(
                    attachment_verified=False,
                    carabiner_visible=True,
                    alignment_degraded=True,
                    carabiner_grasped=True,
                ),
                ClipRLStep(True, carabiner_grasped=True),
            ]
        ),
    )
    config = AttachmentFSMConfig(use_search_cv=False)

    result = AttachmentFSM(handlers, config).run()

    assert result.succeeded
    assert destinations(result) == [AttachmentState.CLIP_RL, AttachmentState.SUCCESS]
    assert result.state_actions["clip_rl"] == 2
    assert not handlers.cv_calls


def test_a_lost_ungrasped_carabiner_still_returns_to_search_when_cv_is_off():
    handlers = ScriptedHandlers(
        initial=InitialObservation(False, True),
        search=deque([SearchRLStep(True)]),
        clip=deque(
            [
                ClipRLStep(attachment_verified=False, carabiner_visible=False),
                ClipRLStep(True, carabiner_grasped=True),
            ]
        ),
    )
    config = AttachmentFSMConfig(use_search_cv=False)

    result = AttachmentFSM(handlers, config).run()

    assert result.succeeded
    assert destinations(result) == [
        AttachmentState.CLIP_RL,
        AttachmentState.SEARCH_RL,
        AttachmentState.CLIP_RL,
        AttachmentState.SUCCESS,
    ]
    assert not handlers.cv_calls


def test_state_budget_stops_a_policy_that_never_finds_the_target():
    handlers = ScriptedHandlers(search=deque([SearchRLStep(False)] * 3))
    config = AttachmentFSMConfig(max_search_rl_actions=2)

    result = AttachmentFSM(handlers, config).run()

    assert result.final_state is AttachmentState.FAILED
    assert result.actions == 2
    assert result.reason == "search_rl action budget exhausted"
    assert handlers.holds == [result.reason]


def test_a_handled_safety_stop_becomes_aborted_and_holds():
    handlers = ScriptedHandlers()

    def abort():
        raise AttachmentAbort("operator cancelled")

    handlers.search_rl_step = abort  # type: ignore[method-assign]
    result = AttachmentFSM(handlers).run()

    assert result.final_state is AttachmentState.ABORTED
    assert result.reason == "operator cancelled"
    assert handlers.holds == ["operator cancelled"]


def test_an_unexpected_base_exception_holds_and_propagates():
    handlers = ScriptedHandlers()

    def interrupt():
        raise KeyboardInterrupt

    handlers.search_rl_step = interrupt  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt):
        AttachmentFSM(handlers).run()
    assert handlers.holds == ["unhandled exception"]


def test_invalid_budgets_are_refused_before_running():
    with pytest.raises(ValueError, match="max_total_actions"):
        AttachmentFSMConfig(max_total_actions=0)
    with pytest.raises(ValueError, match="max_duration_s"):
        AttachmentFSMConfig(max_duration_s=float("inf"))
