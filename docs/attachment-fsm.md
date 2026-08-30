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
time through a persistent `PolicySession`. `SEARCH_CV` is implemented too, against the
fixed camera and the calibrated pixel map. The run still stops at a guarded
`NotImplementedError`, now from perception alone -- `INITIAL` and the gate signals
`CLIP_RL` reads after each action have no detector or verifier behind them yet, and refuse
rather than guess. The refusal happens in `attachment_fsm_handlers()` before the robot is
claimed, so discovering it costs no lease and no energized arm.

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
malformed policy costs no lease. The pixel map `SEARCH_CV` servos on is loaded and checked
against the connected arm in the same window, for the same reason.

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

Entry into `SEARCH_CV` builds a fresh `CarabinerFollower` and calls `start()` on it, and
entry into anything else stops and discards the one that was running. The follower's
lock-on count and its tracker command are seeded from joint feedback, and both are stale
the moment a policy moves the arm, so this is a rebuild rather than a resume. The fixed
camera itself is opened once per attempt and released by `close()`; re-opening it would
cost a capture-session spin-up on every hand-back with the arm claimed and energized.

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

The hardware half is implemented by `InitialReadinessChecker`. With torque off it requires
three advancing feedback samples, finite telemetry, no faults, in-limit positions, a
stationary arm, and every configured camera name and shape. The handler retains the report
for diagnosis. The perception half remains behind
`AttachmentPerception.initial_observation()`: capture one coherent, motion-free scene and run:

- attachment verification, producing `already_attached`;
- carabiner detection, producing `carabiner_detected` and measured confidence.

Attachment wins if both are true. This handler must not enable the arm or move to a named
or neutral position. Detector thresholds and camera selection belong in validated robot
configuration, not in the FSM or CLI.

The hardware entrypoint requires an operator-captured `neutral` named position (or the name
passed as `neutral_position`) before claiming the arm. Never hand-write its joint values.

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

Implemented. One step is one `CarabinerFollower.step()`
(`robot/carabiner_follower.py`), which is one detector/calibration/servo tick:

1. Read the fixed camera and detect the carabiner in that frame.
2. Convert the accepted centroid through the calibrated pixel map
   (`EVEREST_PIXEL_MAP`, default `config/pixel_map.json`), then over the measured pose with
   `full_target()` so joints the map never fitted hold where they are.
3. Pass that target, or `None`, to one `VisualTracker.tick()` call.
4. Return the follower's `target_visible` and `followed` decisions along with its measured
   pixel error.

The map is taught on *pre-grasp* poses, so tracking to its prediction tracks to
directly-above-the-carabiner by construction; nothing in this path models the table plane,
the camera's obliquity or the arm's kinematics. `VisualTracker` owns lock-on behaviour,
per-tick velocity bounds, holds on missing detections, and limit refusal, and none of that
is reproduced above it.

Three judgements belong to the follower rather than to the FSM:

- **Loss.** One dropped frame is a threshold segmentation flickering, not a lost carabiner.
  Only `lost_after_misses` consecutive frames without a detection report
  `target_visible=False`, which is what returns the FSM to RL search.
- **`followed`.** The measured pose must be within `settle_tolerance_rad` of the map's
  target for `settle_ticks` consecutive ticks. A tick that reported motion means the arm is
  on its way, and handing a half-finished approach to the clip policy is the failure this
  state exists to prevent.
- **Pacing.** The tracker's speed lock is a per-tick clamp of `max_velocity_rad_s /
  rate_hz`, so the follower waits out the remainder of the period itself rather than
  trusting the FSM's call rate.

A detection the map refuses -- outside the taught convex hull, or too far from the previous
frame to be the same blob -- holds the arm but stays *visible*. The camera is bolted down,
so no arm motion moves the carabiner back inside the hull, and a search policy would only
hand back the detection the follower already has.

`confidence` is `None`: the two-white-tape segmentation is a hard threshold with no score,
and for the same reason `attach_clip` reports no force, a number here would be fiction.
`pixel_error` is real -- the joint-space servo error inverted through the fit's Jacobian at
that pixel, so it reads as *the gripper is standing where a carabiner this many pixels away
would have put it*, in the same units as the hull margin and the jump gate. It is derived
from the fit and inherits the fit's errors; it is not independent evidence about where the
gripper physically is.

An arm fault, lost feedback or a refused command raises `TrackerStopped` out of the tracker
after it has held the arm, and the handler turns that into `AttachmentAbort`.

### `clip_rl_step()`

The action and `returned_to_neutral` are implemented; the remaining evidence is not. One
`PolicySession.step()` commands exactly one action from the post-CV session, then fresh
evidence is needed for:

- `attachment_verified` -- **not implemented**, behind `AttachmentPerception`;
- `returned_to_neutral` -- implemented from policy completion plus measured pose;
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
> A neutral candidate is read from `PolicyHandle.select_action` returning `None`, which the
> policy protocol already defines as "the policy considers the task finished": a
> `PolicySession` step terminating with `COMPLETED` is reported as
> completion signal. Fresh feedback must then show a stationary arm within the captured
> neutral pose's tolerance before `returned_to_neutral=True` is reported. The trigger has
> **not** been confirmed against a real checkpoint on hardware, and a
> scripted policy cannot confirm it -- a fake exhibits whatever mapping the test asserts.
>
> It is wrong if the policy signals neutral some other way, returns `None` for unrelated
> reasons such as giving up (measured confirmation prevents reset but aborts the attempt), or
> returns to neutral silently (the reset would never fire and the attempt would spend its
> lifetime budget in `CLIP_RL`). Verify by observing a trained checkpoint's terminal
> behaviour on the arm and confirming `None` coincides with the end effector actually being
> at neutral. Rationale and the alternative are in
> [ADR-0003](adr/0003-realtime-attachment-fsm.md#assumption-pending-verification-how-a-policy-reports-its-return-to-neutral).

Do not infer neutral from elapsed steps. Measured neutral-pose confirmation against the
arm's own joint feedback is mandatory, not an alternative to the completion trigger.

## First enable gate

`RobstrideMitPort.enable()` waits past the bus cache TTL, requires a new feedback stamp from
every motor, reconciles only an explainable full-turn wrap, and refuses non-finite or
out-of-limit feedback. It enables torque and immediately seeds every MIT goal with that
same measured pose; a partial failure disables torque before propagating. Preserve this
ordering so the first target can never be a zero, cache entry, or previous process command.

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
