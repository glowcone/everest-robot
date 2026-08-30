# Everest Robot

A Python/uv scaffold for a deterministic, durable robot workflow powered by
[Absurd](https://earendil-works.github.io/absurd/) and Postgres.

The durable workflow can run end to end against deterministic scaffold adapters. A real
Maker Arm runtime is also present for exclusive hardware sessions, bounded named-position
motion, and policy rollout, while the perception and policy-loading integrations described
below remain intentionally incomplete. The workflow has this shape:

1. Localize the carabiner with CV and pick it up using GraspNet.
2. Move the robot to a named, known attachment position.
3. Attach the clip with an RL policy or VLA.
4. Verify the attachment and route back to pickup, positioning, or attachment. Each
   recovery cycle uses unique checkpoint names so feedback creates new work.

The workflow returns a durable `complete` result only after verification succeeds.

## Running the system

Prerequisites: Python 3.12+, [uv](https://docs.astral.sh/uv/), and a Postgres -- either
Docker or a PostgreSQL installed on the host. Every command below is a
[just](https://just.systems/) recipe; run `just` to see them grouped by purpose. Recipes
load `.env`, so copy the example first.

```bash
cp .env.example .env
just setup      # locked environment: no torch, no CAN
just db-up      # Postgres, in Docker or on the host
just db-init    # Absurd schema and the robot queue
just config     # what this deployment will actually do
```

`just config` reads the robot parameters file and the environment and prints the resolved
result. It touches no hardware and no database, so it is the first thing to run when
something is configured but not behaving.

### Where Postgres comes from

Docker is optional. `just db-up` uses `compose.yaml` when Docker Compose *and* a reachable
daemon are both present, and otherwise drives a PostgreSQL the host already has -- a
Homebrew install, a distribution package, or a server elsewhere. `just db-backend` prints
which one it picked and against which URL; `EVEREST_DB_BACKEND=docker|native` forces the
choice. Every other command goes through `ABSURD_DATABASE_URL` and cannot tell the two
apart.

```bash
brew install postgresql@17     # macOS, if you would rather not run Docker
just db-backend                # -> native (server at localhost:5432)
just db-up                     # starts it via `brew services`, then creates role+database
```

The two backends differ in exactly two places, both deliberate. On the native backend
`just db-down` stops nothing, because that server is shared with the rest of the machine;
stop it yourself if you mean to. And `just db-reset` drops and recreates the `robot`
database rather than deleting a Docker volume. The native backend also needs `psql` on
`PATH`, which Homebrew leaves unlinked for the versioned formulae.

### Carabiner marker vision

The fixed overhead camera can locate the carabiner from two white tape markers and its
black gate. The debug commands only read the camera and print or draw image-space points;
they never command the robot. See [`docs/carabiner-marker-vision.md`](docs/carabiner-marker-vision.md)
for the one-frame and continuous commands, ROI setup, output meanings, and optional
pixel-to-robot configuration.

### The attachment state machine

`robot-attach-fsm` is the real-time orchestrator from
[ADR-0003](docs/adr/0003-realtime-attachment-fsm.md). It is *not* an Absurd workflow: one
invocation holds one robot lease for one physical attempt, needs no worker and no
PostgreSQL, and arbitrates after every individual action rather than at coarse checkpoints.

Exercise the whole state graph with no hardware, camera, or database:

```bash
just attach-fsm-fake                     # INITIAL -> SEARCH_RL -> SEARCH_CV -> CLIP_RL -> SUCCESS
just attach-fsm-fake --initial-detection # the carabiner is already visible; skips SEARCH_RL
```

Both print a JSON result: the terminal state, per-state action counts, elapsed time, and
every transition with the evidence that caused it.

On hardware, each learned state loads its own policy from a file and steps it one action at
a time:

```bash
just attach-fsm <search-policy> <clip-policy>
```

**This refuses today, before claiming the robot.** The learned half of `SEARCH_RL` and
`CLIP_RL` is implemented, and so is `SEARCH_CV`, which servos on the fixed camera and the
calibrated pixel map; the rest of perception is not. `INITIAL` and the gate signals
`CLIP_RL` reads after each action have no detector or verifier behind them yet and refuse
rather than guess. Policy files and the pixel map are resolved and the perception gates are
checked *before* `open_session()`, so a missing checkpoint, a stale calibration or an
unavailable detector costs no lease and no energized arm to discover.

A trained checkpoint (a directory, or `.safetensors` / `.pt` / `.ckpt` / `.bin`) also still
refuses: loading one needs the dataset feature metadata that the deferred recording/dataset
decision will settle. A `.json` scripted-policy file works today and exists so the one-action
session and the FSM's gates can be rehearsed on a real arm before a checkpoint can be loaded.

One assumption in the implemented half is **unverified against hardware**: a clip policy's
return to neutral is read from `select_action()` returning `None`. Read
[`docs/attachment-fsm.md`](docs/attachment-fsm.md) before running a trained checkpoint — it
carries the handler contract, the scripted-policy file format, and what verifying that
assumption requires.

### The two workflows

| Task | Spawn it with | What it does |
| --- | --- | --- |
| `attach-carabiner` | `just start <id>` | The carabiner attachment state machine, with verification-driven recovery. Runs on the scaffold by default. |
| `replay-session` | `just replay ...` | Replays one episode of a pinned dataset revision onto the arm. |

One worker serves both:

```bash
just worker                 # leave this running
just start demo             # in another shell
just start retry-demo 1     # one simulated failed verification before success
```

Spawns are idempotent per workflow id, so a second `just start demo` returns the original
task rather than starting new physical work. Use a fresh id for a fresh run.

### Watching what happened

```bash
just tasks                  # recent tasks and their state
just task <task-id>         # parameters and the durable result
just checkpoints <task-id>  # which stages actually committed, and what they returned
just cancel <task-id>       # stop a running task
```

Checkpoints are the useful view during physical work: they record which stages completed,
so a retry resumes after them instead of repeating them. Cancelling sets the task's state;
a running stage notices at its next heartbeat, holds the arm, and releases the lease on the
way out.

On the Docker backend these recipes run `psql` inside the container, so the host needs no
PostgreSQL client; on the native backend they use the host's own. If old tasks from a
previous revision of a workflow are cluttering the picture, `just db-reset` clears
everything and reinitializes.

### Replaying a stored session

Replay drives the whole arm through a recorded trajectory with no perception in the loop,
so the recipes are numbered and meant to be run in order. Read
[`docs/session-replay.md`](docs/session-replay.md) before the first powered run — it
explains what each step proves and carries the full hardware acceptance procedure.

```bash
# 1. Validate locally. Claims nothing, energizes nothing, needs no worker.
just replay-preflight <repo-id> <40-char-revision> <episode>

# 2. The same validation through the durable workflow. Still no motion.
just replay-dry-run <repo-id> <revision> <episode>

# 3. Drive to the episode's recorded start pose and stop. First motion.
just replay-align <repo-id> <revision> <episode>

# 4. A short range at reduced speed; widen only once each run is clean.
just replay-range <repo-id> <revision> <episode> first30 30 0.25

# 5. The full episode.
just replay <repo-id> <revision> <episode> run-1
```

The preflight report is the document to review before authorizing motion. It states the
action range against the active limits, how many frames would be clamped and by how much,
the joint-frame offsets being applied, the residual between the recorded start pose and the
reachable one, and the largest frame-to-frame step in the episode.

Replay refuses until the dataset is listed in `approved_replays` in the parameters file: a
dataset carries no record of which arm produced it, so someone has to say so. An
interrupted replay is never restarted automatically — see the recovery notes in the replay
doc.

### Calibration teleoperation and joint feedback

`just monitor` opens a powered calibration session: it owns the follower lease, follows the
Star 102 leader at bounded speed, and displays every follower joint's encoder feedback.

```bash
just monitor           # POWERED Star-leader following plus the 10 Hz TUI
just monitor 2         # same control loop, slower display refresh
just monitor-read-only # torque-disabled hand positioning, no leader
just monitor-once     # one snapshot as plain text; redirect it next to a captured pose
just monitor-fake     # the display against the deterministic fake arm, no hardware
```

Keys: `q` stop and hold, `p` capture a pose, `z` mark the current pose as the reference,
`Z` clear it, `space` pause/resume leader following, and `?` for the on-screen guide — what
this mode is doing to the arm, what every column and joint state means, and how a capture
becomes a named position. A held capture is shown as `POSE HELD` in the status line.

Set `EVEREST_STAR_PORT` before using powered mode. Startup reads all seven leader servos,
compares the mapped pose with the follower, and requires confirmation for a large difference
before enabling. Use `just monitor-read-only` when the arm should remain torque-disabled.

Press `p`, then `q`, and the pose is printed in canonical joint radians with its calibration
identity — and then offered for saving as a named position:

```
Save this pose to config/maker_arm_v1.yaml as a named position?
  name (blank to skip) > clip-attachment-ready
  approved by > <operator name or team>
  notes (optional) > measured with the clip fixture at station 2
```

That closes the loop with `just goto`: measure a pose here, name it, and it becomes a
destination. The prompt runs after the session has released the arm, so nothing is claimed
while you type, and it refuses poses that describe no physical arm — `--fake` numbers, a
joint with no feedback, one outside the driver's soft limits, one measured during a fault.
Re-saving a name is a re-approval and must be confirmed. The file is edited as text so its
comments survive, then re-read through the strict loader and rolled back if anything about
the result is not what was written. `--no-save` prints the pose and offers nothing.

A saved preset is *not* yet validated: `docs/named-position-capture.md` step 3 is the speed
ladder that makes it trustworthy, and the prompt prints those commands when it is done.

All modes claim the robot lease. Powered following and monitoring therefore happen in the
same process; run it *instead of* a worker, not alongside one. A joint whose feedback counter
stops advancing is flagged `QUIET`.

### Driving to a named position

Teleoperation is how a pose gets captured; `just goto` is how the arm gets back to one.
It is the counterpart to `just raise`: that command nudges one joint relative to measured
feedback, this one drives the whole arm to a pose someone captured, approved, and committed
to `config/maker_arm_v1.yaml`. Both go through the same motion controller as the workflow,
so the limit checks, bounded interpolation, settling and fault handling are identical.

```bash
just goto-list                      # approved positions and the transitions that reach them
just goto-dry clip-attachment-ready # claims and validates; energizes nothing
just goto clip-attachment-ready     # POWERED, at speed scale 0.25
just goto clip-attachment-ready 1.0 # at the position's approved bounds
```

The recipes are ordered on purpose, and `docs/named-position-capture.md` step 3 is the
procedure they exist for: dry run, then 0.25, then 0.5, then full speed, from every pose the
move can start at.

No pose is invented here. An unknown destination is refused with the list of approved ones
*before* the robot is claimed, and a preset that falls outside the driver's live soft limits
is refused rather than clamped. Where a `named_transitions` sequence ends at the requested
position, `just goto` takes it and says so — the transition exists precisely because the
straight line was not shown to be collision-free, so there is no flag to override it. A move
that swings any joint by more than 0.35 rad asks for confirmation before energizing; pass
`--yes` to skip that, which `just goto` does not do for you.

### Configuration

Two files hold *robot behaviour*, and both are versioned:

| File | Holds |
| --- | --- |
| `config/maker_arm_v1.yaml` | Robot and calibration identity, joint order, motion bounds, approved named positions and transitions, the LeRobot joint-frame offsets, replay bounds, and approved datasets. |
| `maker-arm-sdk`'s own profile | Motor directions, offsets, gains and mechanical limits. Not restated by Everest; the driver enforces them. |

Everything *deployment-specific* comes from the environment instead, so the same versioned
configuration works on a workstation and in the cell:

| Variable | Purpose |
| --- | --- |
| `ABSURD_DATABASE_URL`, `ROBOT_QUEUE` | Postgres connection and queue name. |
| `EVEREST_DB_BACKEND` | `docker` or `native`; auto-detected when unset. Only the `database` recipes read it. |
| `EVEREST_ROBOT_BACKEND` | `scaffold` (default) or `hardware`. |
| `EVEREST_ROBOT_ID`, `EVEREST_CALIBRATION_ID` | Identity the replay recipes pass; must match the parameters file. |
| `EVEREST_CAN_PORT`, `EVEREST_CAN_BACKEND` | `can0` with `socketcan`, or a serial port with `slcan`. |
| `EVEREST_ROBOT_PARAMETERS`, `EVEREST_ARM_PROFILE` | Override either configuration file's path. |
| `EVEREST_LEASE_BACKEND` | `postgres` (default when the database is configured) or `file`. |
| `EVEREST_CAMERAS`, `EVEREST_CAMERAS_FILE` | Camera devices as JSON, inline or from a file. `just camera-scan` identifies which id is which camera; see [`docs/camera-identification.md`](docs/camera-identification.md). |
| `EVEREST_POLICY_DEVICE` | Inference device: `auto` (default; CUDA, then MPS, then CPU), or an explicit `cuda`/`mps`/`cpu`, which is never silently downgraded. |
| `EVEREST_WRIST_CAMERA`, `EVEREST_WRIST_CAMERA_COLOR` | Which configured camera the attachment detector looks through (default `wrist`), and whether it produces `rgb` (default) or `bgr`. |
| `EVEREST_GRASP_GRIPPER_BELOW_RAD` | Gripper position, in joint radians, below which it counts as holding the carabiner. Unset means grasp is never asserted. |
| `EVEREST_ALIGNMENT_TOLERANCE_PX` | Drift of the carabiner's insertion point from the CV hand-over pose that counts as a degraded alignment (default 60). |
| `HF_TOKEN` | Read by `huggingface_hub` for a private dataset or model. Never passed through workflow parameters, and never printed. |

`src/everest_robot/robot/deployment.py` is the only module that reads these.

### When something refuses

The runtime prefers refusing to guessing, so most surprises are a deliberate check:

| Message | Meaning |
| --- | --- |
| `... is not in approved_replays` | No operator-approved mapping from this dataset to this arm. Add one, or set `require_approved_dataset: false` for a bench experiment. |
| `is not a full 40-character commit SHA` | Replay pins immutable revisions; a branch name would replay different motion on different days. |
| `parameters do not match the connected arm` | The calibration identity in the parameters file differs from the deployment's. Presets and datasets from another calibration describe different physical poses. |
| `unknown named position` | The destination is not in `named_positions`. `just goto-list` shows what is; capture new ones with `docs/named-position-capture.md`. |
| `lies outside the active limits` | The episode leaves the driver's soft limits by more than the configured tolerance. The preflight report says by how much and on which joint. |
| `is claimed by another worker` / `by another process` | Something else holds the robot lease. `just monitor` takes it too; only one process may own the arm. |
| `maker_arm is not installed` | `just setup-hardware`. |
| `OpenCV cannot open a window` | The environment has the headless OpenCV build that `lerobot` pulls in, which overwrites the one with `imshow`. `uv pip install --reinstall opencv-python`, or pass `--no-window`. |
| `NotImplementedError` from a workflow stage | That stage has no hardware implementation yet — see Integration points below. |

### Development

```bash
just check          # lint and unit tests; run before handing off
just test -k replay # pass anything through to pytest
just test-network   # also run the tests that download the real dataset
```

The whole suite runs without the hardware extra: the robot SDKs are imported lazily and
every layer has a deterministic fake.

## The robot SDK layer

`src/everest_robot/robot/` holds the runtime that drives a real Maker Arm: an exclusive
lease and session lifecycle, a strictly validated robot parameters file, bounded motion
between operator-approved named positions, LeRobot's `Robot` contract over the arm, a
synchronous policy rollout runner, and the lease-local calibration teleoperator/monitor
behind `just monitor`. The motor driver decision is recorded in
[`docs/adr/0002-mit-protocol-motor-operation.md`](docs/adr/0002-mit-protocol-motor-operation.md):
this arm's motors run RobStride's MIT protocol, driven by `RobstrideMitPort`
(`EVEREST_ARM_DRIVER=mit`) with Everest-owned compensating controls; the superseded
private-protocol analysis lives in
[`docs/adr/0001-production-motor-protocol.md`](docs/adr/0001-production-motor-protocol.md).

`maker-arm` and `lerobot` are pinned to commit SHAs behind the `hardware` extra
(`just setup-hardware`) and imported lazily, so the package and its whole test suite run
without them.

Named positions ship empty on purpose. Capture them from a measured, operator-approved arm
state by following [`docs/named-position-capture.md`](docs/named-position-capture.md); the
loader refuses any preset whose calibration identity does not match the connected arm.
`just goto` drives to them once they exist.

Stored-session **replay** is implemented: it resolves a pinned Hugging Face dataset
revision, validates the whole episode against this arm before claiming it, aligns to the
recorded start pose, and replays the recorded actions at the recorded cadence. See
[`docs/session-replay.md`](docs/session-replay.md), including the hardware acceptance
procedure and two findings about the first target dataset.

Session *recording* and LeRobot checkpoint loading remain interface-only stubs pending the
dataset-writing decision, and refuse rather than guessing a format.

## Integration points

`ScaffoldRobot` in `src/everest_robot/adapters.py` is the default and supports the complete
demonstration workflow. `EverestRobot` is selected on the worker with
`EVEREST_ROBOT_BACKEND=hardware`; its deployment settings are documented in
`src/everest_robot/robot/deployment.py`.

The hardware backend is intentionally partial. It currently provides the exclusive robot
lease and session, Maker Arm connection, bounded motion to approved named positions, the
policy rollout runner, and the one-action policy session the attachment FSM's learned states
run on. It does **not** yet provide:

- integration of the standalone carabiner CV target into the hardware workflow, or
  GraspNet pickup;
- a loader for a trained RL/VLA checkpoint -- `load_policy()` resolves the path and routes
  it to the guarded `LeRobotPolicyHandle`, which still needs the dataset feature metadata
  the deferred recording/dataset decision will settle; or
- attachment verification using sensors, CV, or a VLM, which is also what the attachment
  FSM's `AttachmentPerception` gates are missing.

Those stages raise actionable `NotImplementedError` exceptions instead of returning
plausible-looking results, so merely setting `EVEREST_ROBOT_BACKEND=hardware` does not yet
produce an end-to-end hardware run. Named positions also ship empty; after following the
capture guide, either name the attachment preset `clip-attachment-ready` to match the
current workflow default or pass a different `attachment_position` task parameter.

Keep side effects idempotent: Absurd guarantees persisted checkpoints, while a worker can
briefly overlap execution around crashes -- which is what the robot lease exists to make
safe. Long-running policy/control calls heartbeat from inside their control loops, and
treat a raising heartbeat (Absurd's cancellation signal) as stop-and-hold.

Postgres data lives in the Docker volume `postgres-data` on the Docker backend, and in the
host server's own `robot` database on the native one. `just db-reset` clears whichever it
is.
