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

Prerequisites: Docker, Python 3.12+, and [uv](https://docs.astral.sh/uv/). Every command
below is a [just](https://just.systems/) recipe; run `just` to see them grouped by purpose.
Recipes load `.env`, so copy the example first.

```bash
cp .env.example .env
just setup      # locked environment: no torch, no CAN
just db-up      # Postgres in Docker
just db-init    # Absurd schema and the robot queue
just config     # what this deployment will actually do
```

`just config` reads the robot parameters file and the environment and prints the resolved
result. It touches no hardware and no database, so it is the first thing to run when
something is configured but not behaving.

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

The host needs no PostgreSQL client — these recipes run `psql` inside the container. If old
tasks from a previous revision of a workflow are cluttering the picture, `just db-reset`
clears everything and reinitializes.

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

### Reading the joints

`just monitor` opens a terminal display of every joint's encoder feedback: the angle in
radians and degrees, how far it has moved since a marked reference, velocity, torque,
temperature, and where the joint sits inside the driver's soft limits.

```bash
just monitor          # the TUI, polling at 10 Hz
just monitor 2        # slower, when you only want to watch one number settle
just monitor-once     # one snapshot as plain text; redirect it next to a captured pose
just monitor-fake     # the display against the deterministic fake arm, no hardware
```

Keys: `q` quit, `z` mark the current pose as the reference deltas are measured from, `Z`
clear it, `space` pause.

It reads and nothing else — it never enables the motors and never sends a target — so it
is the right tool for step 1 of
[`docs/named-position-capture.md`](docs/named-position-capture.md), where the arm is moved
by hand with motors disabled and the pose is read back. Press `z` at the start of a
measurement and the `d deg` column reports exactly how far each joint has been moved.

It does claim the robot lease, which is not a formality: reading feedback from a connected
arm makes the driver poll the CAN bus, so the monitor is a bus participant rather than a
passive tap. Run it *instead of* a worker, not alongside one. A joint whose feedback
counter stops advancing is flagged `QUIET`, which is a different failure from a value that
is fresh but wrong.

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
| `EVEREST_ROBOT_BACKEND` | `scaffold` (default) or `hardware`. |
| `EVEREST_ROBOT_ID`, `EVEREST_CALIBRATION_ID` | Identity the replay recipes pass; must match the parameters file. |
| `EVEREST_CAN_PORT`, `EVEREST_CAN_BACKEND` | `can0` with `socketcan`, or a serial port with `slcan`. |
| `EVEREST_ROBOT_PARAMETERS`, `EVEREST_ARM_PROFILE` | Override either configuration file's path. |
| `EVEREST_LEASE_BACKEND` | `postgres` (default when the database is configured) or `file`. |
| `EVEREST_CAMERAS`, `EVEREST_CAMERAS_FILE` | Camera devices as JSON, inline or from a file. |
| `HF_TOKEN` | Read by `huggingface_hub` for a private dataset. Never passed through workflow parameters, and never printed. |

`src/everest_robot/robot/deployment.py` is the only module that reads these.

### When something refuses

The runtime prefers refusing to guessing, so most surprises are a deliberate check:

| Message | Meaning |
| --- | --- |
| `... is not in approved_replays` | No operator-approved mapping from this dataset to this arm. Add one, or set `require_approved_dataset: false` for a bench experiment. |
| `is not a full 40-character commit SHA` | Replay pins immutable revisions; a branch name would replay different motion on different days. |
| `parameters do not match the connected arm` | The calibration identity in the parameters file differs from the deployment's. Presets and datasets from another calibration describe different physical poses. |
| `lies outside the active limits` | The episode leaves the driver's soft limits by more than the configured tolerance. The preflight report says by how much and on which joint. |
| `is claimed by another worker` / `by another process` | Something else holds the robot lease. `just monitor` takes it too; only one process may own the arm. |
| `maker_arm is not installed` | `just setup-hardware`. |
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
synchronous policy rollout runner, and the read-only joint monitor behind `just monitor`. The production driver decision is recorded in
[`docs/adr/0001-production-motor-protocol.md`](docs/adr/0001-production-motor-protocol.md):
the arm keeps running maker-arm-sdk's RobStride private protocol, which stays the hardware
safety boundary.

`maker-arm` and `lerobot` are pinned to commit SHAs behind the `hardware` extra
(`just setup-hardware`) and imported lazily, so the package and its whole test suite run
without them.

Named positions ship empty on purpose. Capture them from a measured, operator-approved arm
state by following [`docs/named-position-capture.md`](docs/named-position-capture.md); the
loader refuses any preset whose calibration identity does not match the connected arm.

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
lease and session, Maker Arm connection, bounded motion to approved named positions, and
the policy rollout runner. It does **not** yet provide:

- carabiner CV localization or GraspNet pickup;
- a policy factory that loads the requested RL/VLA checkpoint; or
- attachment verification using sensors, CV, or a VLM.

Those stages raise actionable `NotImplementedError` exceptions instead of returning
plausible-looking results, so merely setting `EVEREST_ROBOT_BACKEND=hardware` does not yet
produce an end-to-end hardware run. Named positions also ship empty; after following the
capture guide, either name the attachment preset `clip-attachment-ready` to match the
current workflow default or pass a different `attachment_position` task parameter.

Keep side effects idempotent: Absurd guarantees persisted checkpoints, while a worker can
briefly overlap execution around crashes -- which is what the robot lease exists to make
safe. Long-running policy/control calls heartbeat from inside their control loops, and
treat a raising heartbeat (Absurd's cancellation signal) as stop-and-hold.

Postgres data lives in the Docker volume `postgres-data`, which `just db-reset` deletes.
