# 0003. Real-time finite-state orchestrator for carabiner attachment

- Status: Accepted
- Date: 2026-08-30

## Context

The attachment workflow currently treats localization, pickup, a move to a named position,
policy rollout, and verification as coarse Absurd checkpoints. The learned policy now works
well from grasp through attachment, while the classical-vision follower can report when it
has successfully followed the carabiner and left the end effector in a pose from which the
policy can grasp it.

The controller therefore needs to arbitrate after individual actions. It must alternate
between learned search and classical vision, then hand control to learned grasp-and-clip
behavior. Absurd checkpoints are not suitable for this control rate: replaying a stored
checkpoint does not establish the arm's current physical pose, and a database round trip
must not sit in a robot servo loop.

The existing `PolicyRunner.run()` is also not a one-action primitive. Each call resets the
policy and processor, begins a recording episode, and holds the arm on return. Repeatedly
calling it with `max_steps=1` would discard recurrent state and cached action chunks and
would not be equivalent to a continuous rollout.

## Decision

Implement a synchronous, in-process finite-state machine as a standalone entrypoint. It
holds one `RobotSession` and its exclusive lease for the complete attempt. Absurd is not in
the inner control path. A deployment may later use Absurd or another service to launch and
observe attempts, but the FSM remains independently runnable and owns its transitions.

The state machine is:

```text
                                      +---------------------+
                                      |       INITIAL       |
                                      |                     |
                                      | - Run preflight     |
                                      | - Claim robot       |
                                      | - Observe scene     |
                                      +----------+----------+
                                                 |
                           +---------------------+---------------------+
                           |                     |                     |
                  carabiner detected      not detected        already attached
                           |                     |                     |
                           v                     v                     v
                 +-----------------+   +-----------------+   +-----------------+
          +----->|    SEARCH_CV    |   |    SEARCH_RL    |   |     SUCCESS     |
          |      |                 |   |                 |   +-----------------+
          |      | Detect, follow, |   | Roll out one RL |
          |      | and approach    |   | action at a time|
          |      +-------+---------+   +-------+---------+
          |              |                     |
          |              | followed / aligned  | carabiner detected
          |              |                     +-----------------+
          |              v                                       |
          |      +-----------------+                              |
          |      |     CLIP_RL     |                              |
          |      |                 |                              |
          |      | Grasp and clip, |                              |
          |      | one RL action   |                              |
          |      | at a time       |                              |
          |      +-------+---------+                              |
          |              |                                        |
          |       +------+----------------------+                 |
          |       |                             |                 |
          |       |                    carabiner visible,         |
          |       |                    alignment degraded         |
          |       |                             |                 |
          |       |                             +-----------------+
          |       |                             |
          |       |                             v
          |       |                     +-----------------+
          |       |                     |    SEARCH_CV    |
          |       |                     +-----------------+
          |       |
          |       | carabiner lost before grasp
          |       v
          |  +-----------------+
          +--|    SEARCH_RL    |
             +-----------------+

                 CLIP_RL -- attachment verified --> SUCCESS

                 CLIP_RL -- policy returned to neutral --> INITIAL

                 Any active state -- cancelled / safety fault --> ABORTED

                 Any active state -- action or time budget exhausted --> FAILED
```

`INITIAL` performs validation and one coherent observation before any motion. With torque
off it requires multiple advancing, finite, fault-free, in-limit and stationary feedback
samples plus configured camera shapes. It always appears in the transition trace, even
when that observation immediately selects another state. There is no automatic move to a
neutral pose.

`SEARCH_RL` executes one learned action and then gives the detector an opportunity to take
over. A detection transitions to `SEARCH_CV`; otherwise it remains in `SEARCH_RL` until its
budget is exhausted.

`SEARCH_CV` executes one bounded visual-following action. Its built-in `followed` signal is
the authoritative transition to `CLIP_RL`. Losing the target returns to `SEARCH_RL`.

`CLIP_RL` executes one learned action and then checks attachment and recovery observations.
Successful verification terminates. A visible but poorly aligned carabiner returns to
`SEARCH_CV`; losing an ungrasped carabiner returns to `SEARCH_RL`; otherwise rollout
continues in `CLIP_RL`. If the policy reports that it has returned to neutral, that signal
takes precedence over its same-step verification result and transitions to `INITIAL`. The
FSM clears its per-cycle state budgets and policy context, then takes a fresh motion-free
observation; lifetime action and wall-clock budgets do not reset.

Search and clip may use the same checkpoint, but they are separate policy sessions. CV has
physically intervened between them, so cached actions and recurrent state from before CV
are stale. Entering either RL state resets or reseeds the policy from a fresh observation.

## Assumption pending verification: how a policy reports its return to neutral

**Status: UNVERIFIED. Implemented as described below; not yet confirmed against a trained
checkpoint on hardware.**

The decision above requires `CLIP_RL` to observe "the policy reports that it has returned to
neutral", but does not say what that report physically is. The implementation reads it from
the existing policy boundary: `PolicyHandle.select_action` returning `None`, which the
protocol already defines as "the policy considers the task finished". A `PolicySession` step
that terminates with `COMPLETED` is therefore a neutral candidate. The handler reports
`returned_to_neutral=True` only after fresh feedback confirms the operator-captured neutral
pose and stationarity, and then the FSM resets to `INITIAL`.

This is a mapping chosen to avoid inventing a second completion channel, not an observed
property of the policy that will run here. It is wrong if any of the following turns out to
be true of the trained checkpoint:

- the policy signals neutral some other way -- an action-space flag, a terminal action, a
  pose the caller is expected to recognize -- and returns `None` for a different reason, or
  never returns `None` at all;
- the policy returns `None` on outcomes that are *not* a return to neutral, such as giving
  up or losing the carabiner; measured confirmation prevents a false reset, but the attempt
  aborts instead of selecting a more specific recovery; or
- the policy returns to neutral without reporting anything, in which case the reset never
  fires and the attempt spends its lifetime budget in `CLIP_RL`.

None of these is detectable without hardware, because a scripted policy exhibits whatever
mapping the test asserts. Verifying it requires observing a real checkpoint's terminal
behaviour on the arm and confirming that `select_action` returning `None` coincides with the
end effector actually being at neutral.

Measured neutral-pose confirmation is mandatory even while checkpoint completion remains
the trigger. It uses an operator-captured named position in robot configuration, never
hand-written joint values. Do not infer neutral from elapsed steps or action count.

All active states have explicit action and wall-clock budgets. Cancellation and any
`BaseException` unwind through the robot session after holding the arm. A process crash is
not automatically resumable from a recorded FSM state: the hardware must be observed and
a new attempt explicitly authorized.

Every transition records the source, destination, reason, action index, and handler result.
The trace is diagnostic evidence, not a physical-effect checkpoint.

## Consequences

- Fast arbitration and camera checks stay local to the process holding the robot lease.
- The FSM can be tested deterministically with scripted handlers without hardware, models,
  cameras, a database, or Absurd.
- Policy integration requires a stateful one-action interface; the existing whole-rollout
  API remains valid for uninterrupted rollouts but must not be adapted by repeated resets.
- CV following can reuse `VisualTracker`, but detection, pixel-to-joint targeting, and the
  definition of `followed` remain behind an integration handler.
- The orchestrator never commands a known-position/neutral stage. It observes the RL
  policy's normal return to neutral and uses that event to reset for another cycle. How that
  event is observed is an unverified assumption; see the section above.
- Outer durability is optional. If added, one attempt is one non-retryable physical stage,
  not one durable checkpoint per FSM action.
