# Attachment FSM integration guide

`robot-attach-fsm` is the standalone, synchronous orchestrator selected by
[ADR-0003](adr/0003-realtime-attachment-fsm.md). It does not require PostgreSQL or an
Absurd worker. One invocation owns one robot lease and one physical attempt.

Exercise the complete state graph without cameras or hardware:

```bash
just attach-fsm-fake
just attach-fsm-fake --initial-detection
```

`just attach-fsm` selects the hardware adapter. The learned states are implemented: both
load a policy from a file (`--search-policy`, `--clip-policy`) and step it one action at a
time through a persistent `PolicySession`. The run still stops at a guarded
`NotImplementedError`, now from perception alone -- `INITIAL`, `SEARCH_CV`, and the gate
signals `CLIP_RL` reads after each action have no detector or verifier behind them yet, and
refuse rather than guess. The refusal happens in `attachment_fsm_handlers()` before the
robot is claimed, so discovering it costs no lease and no energized arm.

One assumption inside the implemented half is **unverified against real hardware**: how a
clip policy reports that it has returned to neutral. See
[the ADR section](adr/0003-realtime-attachment-fsm.md#assumption-pending-verification-how-a-policy-reports-its-return-to-neutral)
and `clip_rl_step()` below before running a trained checkpoint.

## Policy files

Each learned state loads its own policy from a path, and they are separate sessions even
when the two paths are identical:

```bash
just attach-fsm --search-policy <path> --clip-policy <path>
```

`load_policy()` in `robot/policy.py` is the only place a path becomes a `PolicyHandle`, and
it recognizes two kinds of file:

- **A trained checkpoint** -- a directory, or a file ending `.safetensors`, `.pt`, `.ckpt`
  or `.bin` -- goes to `LeRobotPolicyHandle`, which **refuses**. Loading one needs the
  dataset feature metadata that maps this robot's `{joint}.pos` scalars and camera frames
  into the policy's input tensors, and that metadata comes from the recording/dataset
  decision that has not been made. An invented mapping would mis-order joints against a real
  checkpoint without ever failing, so the refusal is correct and this function is where it
  becomes reachable from a command line.
- **A `.json` scripted-policy file** replays a fixed action sequence. It is not a trained
  policy and never pretends to be; it exists so the one-action session, the FSM's gates and
  the whole hardware path can be rehearsed on a real arm before a checkpoint can be loaded.

```json
{
  "controller": "search-v0",
  "fps": 30.0,
  "action_features": ["shoulder_pan.pos", "shoulder_lift.pos", "gripper.pos"],
  "actions": [{"shoulder_pan.pos": 0.5, "shoulder_lift.pos": 0.0, "gripper.pos": 0.0}]
}
```

The reader is strict for the same reason `ReplayRequest.from_json` is: a misspelled field
that silently took a default would move a real arm along a sequence nobody wrote. Unknown
fields are rejected, every action must match `action_features` exactly, and non-finite
values are refused. Both files are resolved **before** the robot is claimed, so a missing or
malformed policy costs no lease.

## Handler contract

The integration surface is `EverestAttachmentFSMHandlers` in `adapters.py`. Each step
method performs **at most one** physical action and returns a typed result from
`attachment_fsm.py`. The orchestrator alone chooses the next state. Handlers must not call
one another or hide retry loops.

All observations used to decide a result must be captured after that step's action. Do not
return a detection cached before the arm moved. Keep third-party camera, model, and driver
imports lazy so hardware-free tests remain importable.

### `enter_state(state, previous)`

Implemented. Entry into `SEARCH_RL` and `CLIP_RL` re-seeds that state's own
`PolicySession` (`robot/policy.py`) from where the arm is standing now. Search and clip may
select the same checkpoint, but they are separate sessions because CV physically moves the
arm between them, and entering one leaves the other alone.

The session lifecycle is `seed()` / `step()` / `close()`. It retains recurrent state and
action chunks between `step()` calls, and reuses the existing bridge compatibility and
identity checks, the clipping safety boundary, heartbeat and cancellation hooks, and
stop-and-hold on `BaseException`.

It is deliberately *not* repeated `PolicyRunner.run(max_steps=1)`: `run()` resets processor
and policy state, starts a recording episode, and holds the arm on every return. The two
also pace differently on purpose. `PolicyRunner` follows an absolute schedule
(`start + n / fps`) because it owns an uninterrupted rollout where drift means running
below the trained rate. A session's caller does a detection between actions, so lateness is
expected rather than drift: each deadline is set from when the previous action went out, and
a late step is absorbed and reported, never made up. Commanding a backlog faster would drive
the arm at a rate nothing validated -- the same rule replay follows for a late frame.

Nothing is held on the way *out* of a learned state. On the paths that leave one, CV is
about to command the arm and an interposed hold would fight it; the FSM holds on every
terminal outcome and on any exception, which is where a hold belongs.

### `observe_initial()`

Capture one coherent, motion-free scene observation. Run:

- attachment verification, producing `already_attached`;
- carabiner detection, producing `carabiner_detected` and measured confidence.

Attachment wins if both are true. This handler must not enable the arm or move to a named
or neutral position. Detector thresholds and camera selection belong in validated robot
configuration, not in the FSM or CLI.

### `search_rl_step()`

The action is implemented; the detection is not. One `PolicySession.step()` commands
exactly one action, then the result of `AttachmentPerception.carabiner_detection()` is
returned as `SearchRLStep(carabiner_detected, confidence)`. The FSM switches to CV on a
detection.

Still to integrate: read a fresh camera frame and run the lightweight carabiner detector
behind `AttachmentPerception`. The default implementation refuses, and refuses in
`preflight()` -- before the claim -- so a gate that cannot be read is never discovered after
a learned action has already moved the arm.

The action and detection together are one orchestrator step; the policy must not run an
uninterruptible multi-action chunk. Safety faults and operator cancellation are raised as
`AttachmentAbort` after holding, while unexpected failures unwind through the session.

A search policy that reports itself finished without a detection raises `AttachmentAbort`
rather than continuing: calling a spent rollout until the budget runs out is a livelock, not
progress.

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

The action and `returned_to_neutral` are implemented; the remaining evidence is not. One
`PolicySession.step()` commands exactly one action from the post-CV session, then fresh
evidence is needed for:

- `attachment_verified` -- **not implemented**, behind `AttachmentPerception`;
- `returned_to_neutral` -- implemented, but on an **unverified assumption**; see below;
- `carabiner_grasped` -- **not implemented**;
- `carabiner_visible` -- **not implemented**;
- `alignment_degraded` -- **not implemented**;
- verification confidence -- **not implemented**.

Verification should use the eventual sensor/CV/VLM fusion behind the adapter. A verified
attachment succeeds. A visible, degraded alignment returns to CV. An invisible, ungrasped
carabiner returns to RL search. A grasped carabiner stays with clip RL even when it occludes
the detector. `returned_to_neutral` takes precedence: it returns to `INITIAL`, clears
per-cycle action budgets and policy context, and takes a fresh initial observation. Because
the FSM re-observes on arrival at `INITIAL`, the handler short-circuits on this signal and
does not spend a perception call answering a question that is about to be asked again.

> **Unverified assumption -- confirm before running a trained checkpoint.**
> `returned_to_neutral` is read from `PolicyHandle.select_action` returning `None`, which the
> policy protocol already defines as "the policy considers the task finished": a
> `PolicySession` step terminating with `COMPLETED` is reported as
> `returned_to_neutral=True`. This mapping was chosen to avoid inventing a second completion
> channel. It has **not** been confirmed against a real checkpoint on hardware, and a
> scripted policy cannot confirm it -- a fake exhibits whatever mapping the test asserts.
>
> It is wrong if the policy signals neutral some other way, returns `None` for unrelated
> reasons such as giving up (the FSM would reset and re-observe instead of failing), or
> returns to neutral silently (the reset would never fire and the attempt would spend its
> lifetime budget in `CLIP_RL`). Verify by observing a trained checkpoint's terminal
> behaviour on the arm and confirming `None` coincides with the end effector actually being
> at neutral. Rationale and the alternative are in
> [ADR-0003](adr/0003-realtime-attachment-fsm.md#assumption-pending-verification-how-a-policy-reports-its-return-to-neutral).

Do not infer neutral from elapsed steps. The alternative to the mapping above is a
configured, measured neutral-pose tolerance evaluated against the arm's own joint feedback,
which needs no cooperation from the checkpoint.

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
