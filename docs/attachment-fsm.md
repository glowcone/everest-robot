# Attachment FSM integration guide

`robot-attach-fsm` is the standalone, synchronous orchestrator selected by
[ADR-0003](adr/0003-realtime-attachment-fsm.md). It does not require PostgreSQL or an
Absurd worker. One invocation owns one robot lease and one physical attempt.

Exercise the complete state graph without cameras or hardware:

```bash
just attach-fsm-fake
just attach-fsm-fake --initial-detection
```

`just attach-fsm` selects the hardware adapter. It currently stops at a guarded
`NotImplementedError`; this is intentional. The state machine and safety envelope exist,
but commanding a real arm would require guessing policy and perception contracts that the
team has not integrated yet.

## Handler contract

The integration surface is `EverestAttachmentFSMHandlers` in `adapters.py`. Each step
method performs **at most one** physical action and returns a typed result from
`attachment_fsm.py`. The orchestrator alone chooses the next state. Handlers must not call
one another or hide retry loops.

All observations used to decide a result must be captured after that step's action. Do not
return a detection cached before the arm moved. Keep third-party camera, model, and driver
imports lazy so hardware-free tests remain importable.

### `enter_state(state, previous)`

The remaining stub applies to entry into `SEARCH_RL` and `CLIP_RL`. Create or reset a
persistent stepwise policy session and seed it from a fresh observation. Search and clip
may select the same checkpoint, but they are separate sessions because CV physically moves
the arm between them.

Do not implement this as repeated `PolicyRunner.run(max_steps=1)`: `run()` resets processor
and policy state, starts a recording episode, and holds the arm on every return. Add a
runtime API with lifecycle equivalent to:

```text
policy_session.start(task)
policy_session.step() -> one commanded action
policy_session.finish(reason)
```

The session must retain recurrent state and action chunks between `step()` calls, discard
unused cached actions when leaving an RL state, use the existing bridge compatibility and
identity checks, heartbeat/cancellation hooks, pacing, recording, and `BaseException`
stop-and-hold behavior.

### `observe_initial()`

Capture one coherent, motion-free scene observation. Run:

- attachment verification, producing `already_attached`;
- carabiner detection, producing `carabiner_detected` and measured confidence.

Attachment wins if both are true. This handler must not enable the arm or move to a named
or neutral position. Detector thresholds and camera selection belong in validated robot
configuration, not in the FSM or CLI.

### `search_rl_step()`

Command exactly one action from the active search policy session. After the command, read a
fresh camera frame and run the lightweight carabiner detector. Return
`SearchRLStep(carabiner_detected, confidence)`. The FSM switches to CV on a detection.

The action and detection together are one orchestrator step; do not let the policy run an
uninterruptible multi-action chunk. Safety faults or operator cancellation should raise
`AttachmentAbort` after holding, while unexpected failures must unwind through the session.

### `search_cv_step()`

Run one detector/calibration/servo tick:

1. Detect the carabiner in the current frame.
2. Convert the accepted pixel target through the calibrated pixel map.
3. Pass that target, or `None`, to one `VisualTracker.tick()` call.
4. Return the CV subsystem's built-in `target_visible` and `followed` decisions along with
   confidence and pixel error.

Reuse `VisualTracker`; it owns lock-on behavior, per-tick velocity bounds, holds on missing
detections, and limit refusal. Do not reproduce those controls in the handler. `followed`
must mean the CV subsystem has completed its approach tolerance, not merely that one
tracker tick reported motion. A lost target returns to RL search; `followed=True` enters
clip RL.

### `clip_rl_step()`

Command exactly one action from the post-CV policy session. Then obtain fresh evidence for:

- `attachment_verified`;
- `returned_to_neutral` from the policy/runtime;
- `carabiner_grasped`;
- `carabiner_visible`;
- `alignment_degraded`;
- verification confidence.

Verification should use the eventual sensor/CV/VLM fusion behind the adapter. A verified
attachment succeeds. A visible, degraded alignment returns to CV. An invisible, ungrasped
carabiner returns to RL search. A grasped carabiner stays with clip RL even when it occludes
the detector. `returned_to_neutral` takes precedence: it returns to `INITIAL`, clears
per-cycle action budgets and policy context, and takes a fresh initial observation. Do not
infer neutral from elapsed steps; use the policy's completion signal or a configured,
measured neutral-pose tolerance.

### `hold(reason)`

This handler is implemented: if the arm is enabled, it holds measured position. Preserve
that behavior. It is called for bounded failure, handled abort, and before propagating any
unexpected `BaseException`. The enclosing `RobotSession` subsequently disables,
disconnects, and releases the lease.

## Results, budgets, and restart behavior

The CLI prints a JSON `AttachmentFSMResult` containing the terminal state, total and
per-state action counts, elapsed time, and every actual state transition with its guard
evidence. It also reports how many neutral resets occurred. Configure action and time
bounds through `AttachmentFSMConfig`; keep finite defaults in production. A neutral reset
clears per-cycle state budgets but never the lifetime action or wall-clock bound.

The transition trace is evidence, not a resumable checkpoint. After interruption, never
restore an FSM state and repeat its next action blindly. Observe the physical robot and
authorize a new attempt. If an outer scheduler is added, it should treat the whole run as
one non-automatically-retried physical attempt.
