# Replaying a stored session

Replay takes a pinned LeRobot dataset revision and drives this arm through one episode's
recorded actions at the recorded cadence. It is the most directly dangerous operation in
the system: it moves the whole arm through a trajectory that was recorded somewhere else,
with no perception in the loop and no ability to react to anything it was not recorded
against.

Everything below exists to make that safe to authorize.

## What runs before the arm is touched

`ReplayRunner.preflight()` completes before anything is claimed or energized. In order:

1. **Approval.** The dataset must appear in `approved_replays` in the robot parameters
   file, with a matching revision and episode. A dataset carries no record of which
   physical arm produced it, so nothing else can establish that replaying it here is
   meaningful.
2. **Request identity.** The request's `robot_id` and `calibration_id` must match this
   deployment's.
3. **Resolution.** The revision must be a full 40-character commit SHA, and it is
   downloaded to an immutable local snapshot. A branch name would make the same workflow
   parameters drive different motion on different days.
4. **Schema.** Codebase version, robot type, FPS, feature names, dtypes and shapes. Joint
   names must match this robot's exactly, in order — actions are rebuilt by name, never by
   position.
5. **The whole selected range.** Every action is checked for finiteness and against the
   driver's active soft limits, expressed in the dataset's degree frame. Frame-to-frame
   steps are measured and reported, and rejected if `max_step_deg` is configured.
6. **The recorded initial state**, under the same limit policy as the actions.

Any failure here raises rather than returning a result: the request is wrong, and there was
no physical attempt to record.

## Two things about the target dataset

Both were found by running preflight against `h8i76dfsd9/test1_20260829_130743` @ `55e5611`.

**The recorded start pose is 1.70 degrees outside this driver's `elbow_flex` limit.** The
recorded *actions* exceed the limits by at most 0.8 degrees, which is what a 1.0-degree
clamping tolerance covers — but `observation.state` is a measured position, and measured
positions overshoot the commands that produced them. Replay clamps the start pose into the
limits and aligns to the clamped value, reporting the residual as
`initial_state_clipping_deg`. Accepting this dataset therefore needs
`max_limit_deviation_deg: 2.0`, not the 1.0 the action ranges alone suggest. Two degrees
still rejects a materially different episode.

**The frame-to-frame steps are much larger than a per-step bound of 0.25 degrees.** Episode
0 contains steps up to 7.7 degrees (gripper) with a 99th percentile of 2.4 degrees, at
30 fps. `max_step_deg` therefore ships unenforced, with the observed maximum reported in
every preflight, so the bound can be set from measurement rather than from a guess.

## Timing

The base period is the dataset's FPS; `speed` scales it and is capped at 1.0, because
faster-than-recorded replay is a separate hardware acceptance question.

A late frame is **never** made up. Replay commands absolute joint positions, so losing time
is harmless — the arm simply moves more slowly through the same path. Commanding the
backlog faster is not harmless: it would drive the recorded path at a speed nobody
validated. Consecutive missed deadlines beyond `max_consecutive_missed_deadlines` abort the
replay rather than continuing to fall behind.

## During the replay

Each tick reads joint feedback before commanding, and aborts on a motor fault, missing
feedback, or a tracking error that stays above `tracking_error_limit_deg` for several
consecutive frames — one sample over the bound is lag behind a fast step, a run of them is
a joint that is stuck or fighting the command.

Cancellation, faults and timing failures return a `ReplayResult` with `completed=False`, a
machine-readable `stopped_reason`, and the frames actually sent. That count is what an
operator needs before deciding how to recover.

## Recovery, and why there are no automatic retries

The replay task is registered with one attempt. If a worker dies after 200 frames, the arm
is in an unknown intermediate pose; restarting from frame zero is a different physical
motion, not a retry. Progress is persisted for observability, not as authority to resume.

After an interrupted replay: inspect the arm, realign through an approved path, and
authorize a new attempt with a new workflow id.

## Hardware acceptance procedure

Run only after the unit and fake-hardware tests pass, and only with the workspace clear and
an operator on the e-stop.

1. Confirm the exact arm and calibration that recorded the dataset. Set them in
   `approved_replays`, and record who approved it.
2. Verify the frame reconciliation on hardware — see
   [`lerobot-frame-reconciliation.md`](lerobot-frame-reconciliation.md). This is the single
   most consequential value in the whole path.
3. Watch the episode's recorded video before driving it.
4. Run `just replay-preflight <repo> <revision> <episode>` and review the report: action
   ranges against the limits, the clipping count and maximum, the frame offsets, the
   initial-state residual, and the maximum frame-to-frame step.
5. Clear and control the workspace. Support the arm if the episode passes through a pose
   where it could fall.
6. Align only: run the replay with `--start-frame 0 --end-frame 0` to drive the alignment
   and stop.
7. Replay a short range (say 30 frames) at `--speed 0.25`. Confirm joint directions,
   timing, tracking, and that cancellation and e-stop both stop it.
8. Expand gradually — a few hundred frames, then the full episode, raising the speed only
   after each range is clean.
9. Exercise the failure paths deliberately: pull the CAN connection, stall feedback, and
   cancel the workflow mid-replay. Confirm the arm holds and the lease is released each
   time.
10. Set `max_step_deg` from the reported maximum once the episode has run cleanly.
