# Repository guidance

Use `just` as the primary interface for development and operations. Run `just` to list the
recipes, which are grouped: `setup` (`setup`, `setup-hardware`, `config`), `robot`
(`monitor` -- powered calibration teleoperation; `goto` -- powered named-position motion;
`goto-list`, `goto-dry`, `monitor-read-only`, `monitor-once`, `monitor-fake` -- no motion),
`database`
(`db-backend`, `db-up`, `db-init`, `db-reset`, `psql`), `calibration` (the numbered
`pixel-*` path that teaches and then uses the fixed camera's pixel-to-joint map),
`workflow` (`worker`, `start`, `tasks`, `task`,
`checkpoints`, `cancel`), `replay` (the numbered path from `replay-preflight` to `replay`),
and `dev` (`check`, `test`, `lint`, `fmt`, `test-network`). Recipes load `.env`
automatically. Run `just check` before handing off changes.

The Justfile is documentation as much as tooling: a recipe's comment is what `just --list`
shows, so keep it to one line and put longer explanation in the section header or in
`docs/`. Anything that can move a real arm belongs in the `replay` or `workflow` group with
its ordering made explicit. `absurdctl`'s subcommands shell out to `psql`, which the host
is not assumed to have, so task inspection recipes go through `robot-db sql`, which runs
`psql` inside the Postgres container on the Docker backend and the host's client on the
native one.

Docker is optional. `src/everest_robot/database.py` owns the choice: Docker Compose when
both it and its daemon answer, otherwise a PostgreSQL the host already runs (Homebrew, a
package, a remote server), forced with `EVEREST_DB_BACKEND=docker|native`. No recipe or
module outside that file may name `docker` -- everything else goes through
`ABSURD_DATABASE_URL` and cannot tell the backends apart. Two asymmetries are deliberate
and should not be "fixed": the native backend's `down` stops nothing, because the server is
shared with the rest of the workstation, and its `reset` drops the configured database
rather than a volume.

Keep robot hardware, policy, and perception code behind the adapter boundary in
`src/everest_robot/adapters.py`. Keep durable orchestration and checkpoint naming in
`src/everest_robot/workflow.py`. Any physical side effect must be designed to tolerate
retries and brief overlapping execution.

## The robot SDK layer

`src/everest_robot/robot/` is the runtime the hardware adapter drives. It is a separate
concern from `domain.py`: `robot/contracts.py` holds high-frequency runtime structures and
the runtime's own durable results, while `domain.py` holds the workflow-visible types the
Absurd checkpoints store.

- `ports.py` is the single hardware boundary. Nothing above it imports a driver.
  `robstride_mit_port.py` (`EVEREST_ARM_DRIVER=mit`) is this arm's driver per
  `docs/adr/0002-mit-protocol-motor-operation.md`; `maker_arm_port.py` adapts
  `maker_arm.Arm` for private-protocol motors; `fake_arm.py` is the deterministic
  stand-in every layer above is tested against. The MIT driver is qualified for the
  calibration monitor only -- replay and workflow stay unqualified until ADR-0002's
  checklist is done.
- The ownership split for limits, watchdogs, velocity limiting, fault handling and
  coordinate conversion is the ADR-0002 ownership table; do not re-implement a concern a
  driver already owns, and treat the table's "gap" rows as real, accepted risk rather
  than something to quietly patch.
- `parameters.py` loads `config/maker_arm_v1.yaml` strictly: unknown fields are rejected,
  and a preset whose `calibration_id` does not match the file's robot is refused. Presets
  come from `docs/named-position-capture.md`, never from hand-edited joint values.
- `capture.py` is the only writer of that file. It splices one `named_positions` entry in
  as text, because the file's comments carry content no YAML dumper preserves, and it earns
  that by verifying: atomic replace, re-read through the strict loader, and a rollback to
  the original bytes unless the preset that comes back matches the one written. Keep every
  new refusal ahead of the write, and keep provenance (`approved_by`, `captured_at`) coming
  from a person -- it is an attestation, not metadata.
- `motion.py`, `policy.py` and `session.py` take an injectable `Clock`
  (`clock.ManualClock` in tests), a heartbeat and a cancellation check. Absurd signals
  cancellation by raising from `ctx.heartbeat()`, so any loop that commands the arm must
  leave it held on `BaseException`, not only on its own failure paths.
- `session.py` releases torque only *after* driving the arm to `park_position` -- an arm
  released at an arbitrary pose falls onto the bench, and every exit reaches `close()`:
  a clean return, an exception, a cancelled workflow, Ctrl-C. Three invariants make that
  safe and must be kept. The park route is resolved in the constructor, so a session that
  cannot bring the arm home is refused before it claims anything. Parking is skipped when
  the arm was never `ENABLED` (a dry run, `--read-only`, a destination refused before
  anything was energized) -- energizing a cold arm in a teardown would create the hazard,
  not remove it -- and when the arm is faulted or missing feedback, because a pose nobody
  can measure cannot be interpolated from; both say so on stderr rather than passing
  silently. `defer_interrupt` absorbs the first SIGINT so the Ctrl-C that *ends* a session
  is not also what abandons the park half-way; the second one releases the arm where it
  stands, and the physical e-stop is the authority over both. `park_on_success=False` is
  for the commands whose purpose is to leave the arm somewhere specific (`robot-goto`,
  `robot-raise`); they still park when interrupted or failed.
- `robot/routing.py` owns route resolution (`resolve_route`, which prefers an approved
  `named_transitions` sequence over the direct line). It lives under `robot/` because the
  session teardown needs it: a CLI may import the runtime, never the reverse. `goto.py`
  re-publishes those names, `GotoRefused` included.
- `recording.py` is a guarded stub pending the dataset-writing decision. Keep it refusing
  with an actionable message rather than guessing a format.
- `policy.LeRobotPolicyHandle` loads a trained checkpoint, and it is one loader for every
  architecture on purpose: the checkpoint's own config names its policy class and carries
  its processors, so ACT, SmolVLA and anything else LeRobot registers arrive through the
  same path. What it validates is the *interface* -- state width, action width, camera
  shapes, and a language-conditioned policy given no task -- because that is what differs
  between a checkpoint and this arm. `robot_observation_features` and
  `robot_action_features` restate a checkpoint's features in this robot's names so
  `compatibility_problems` can refuse before anything is energized; they are pure, and
  they are where a checkpoint trained on a different arm is caught. Load through
  `deployment.load_policy_handle`, before a session is opened: a bad checkpoint must cost
  no claim. Never reshape or reorder a mismatched action space at runtime.
- `datasets.py` reads a pinned LeRobot v3 snapshot directly (parquet + `meta/info.json`)
  rather than through `LeRobotDataset`, so replay needs no torch. Only the documented v3
  layout is supported; extend the reader deliberately rather than loosening its checks.
- `replay.py` must keep its ordering: everything decidable without hardware happens in
  `preflight()`, which raises, and only then is the robot claimed. Failures after that
  point return a `ReplayResult` with `stopped_reason`, because a physical attempt happened.
- Replay pacing deliberately differs from the policy runner's: a late frame is absorbed,
  never made up, because commanding a backlog faster drives the recorded path at an
  unvalidated speed.
- The `lerobot_frame` offsets in the parameters file are the most consequential values in
  the replay path -- the two drivers do not share a zero pose. See
  `docs/lerobot-frame-reconciliation.md` before touching them.
- Third-party robotics dependencies live in the `hardware` extra and are imported lazily.
  A new import of `lerobot` or `maker_arm` at module scope breaks the hardware-free tests.
- `deployment.py` owns every environment-specific value (CAN interface, cameras, lease
  backend, parameters path -- `parameters_path()`, which is also where a captured preset is
  written back). Do not read the environment anywhere else in the runtime.
- `visual_tracking.py` is the bounded servo loop for a target that changes every frame. It
  takes a full joint target or `None` per tick and clamps the commanded step, so the arm's
  speed is bounded by construction and a missing detection holds rather than coasting.
  Where the target comes from is the caller's problem: `pixel_map.py` fits the fixed
  camera's pixels to taught pre-grasp poses and refuses to extrapolate outside their convex
  hull, and `calibrate_pixel_map.py` (`robot-pixel-map`) is the operator's CLI over both.
  The map is taught on pre-grasp poses, so the tracked target *is* the pose to hand over
  from; there is no height, standoff or lift offset applied on top of it, and adding one
  back would be modelling a table plane this deliberately does not model.
  `visual_tracking.ArrivalGate` decides when servoing is done -- on measured feedback, held
  for consecutive ticks, reset by any miss -- because a loop whose target is recomputed
  every frame has no natural end. `robot-pixel-map track --policy` uses that to hand the
  arm to a checkpoint: the tracker is stopped and holding, and the fixed camera released,
  before the rollout's first observation. Keep it that way -- two controllers must never
  command the arm at once, and the policy's cameras are the deployment's, not this
  calibration's.
- `robot/monitor.py` remains the read-only feedback view model. The `robot-monitor` CLI
  defaults to a powered, lease-local calibration session: `robot/teleoperation.py` owns the
  Star leader and commands the already-claimed follower while the same process renders that
  view. `--read-only` and `--once` never enable. Never split control and monitoring across
  processes or allow either to bypass the robot lease. Pressing `p` captures a pose; the
  save prompt for it runs in `monitor.py` after the session has closed, so the arm is
  parked, disabled and released before anyone is waiting on a keyboard. Do not move that prompt
  inside the session or into the curses loop. `_KEY_BINDINGS` is the one key table: the
  footer and the `?` guide are both derived from it, and tests assert the guide covers
  every key the loop handles, every column the table can draw and every state a joint can
  report. The guide's overlay never blocks on input -- a teleoperation failure has to be
  able to end it.
- `goto.py` (`robot-goto`) and `jog.py` (`robot-raise`) are the only motion commands outside
  the durable workflow, and both are thin: they resolve a destination and hand it to
  `JointMotionController`. Keep it that way -- no second path to the motors. `goto.py`
  resolves its route from the parameters file *before* opening a session, so an unknown or
  ambiguous destination costs no claim, and it must keep preferring an approved
  `named_transitions` sequence over the direct line with no flag to override it. Both take
  `--park`/`--no-park`; the parking move itself is the session's, not a second path.

## Working on workflow components

Shared data contracts belong in `src/everest_robot/domain.py`. Add or change a typed
result there before altering an adapter or workflow stage, and keep every persisted value
JSON-serializable. Treat persisted field names and checkpoint names as durable interfaces:
renaming either can affect workflows that started on an older revision.

The placeholder implementations live on `ScaffoldRobot` in
`src/everest_robot/adapters.py`. Preserve this boundary when adding production code:

- **Carabiner localization and pickup:** implement CV detection and GraspNet grasp
  planning in `localize_and_pick_up_carabiner()`. Return the detected pose, coordinate
  frame, detector, grasp planner, and explicit `secured` result. Raise only for retryable
  execution failures; make physical commands idempotent or guard them with a state check.
- **Known-position motion:** implement deterministic navigation or arm motion in
  `go_to_known_position()`. Named positions belong in robot configuration and should be
  resolved and validated inside the adapter rather than embedded in orchestration.
- **Attachment control:** implement the RL policy or VLA in `attach_clip()`. Return the
  controller identity and measurable control results. Keep policy rollout and low-level
  control inside the adapter; use Absurd checkpoints for stage boundaries, not each action.
- **Attachment verification:** implement sensor/CV/VLM fusion in
  `verify_attachment()`. Return both the verdict and a `RecoveryTarget` so the workflow
  can deliberately route back to localization/pickup, known-position motion, or
  attachment. Add a new recovery target only alongside its orchestration branch and tests.

The durable state machine lives in `src/everest_robot/workflow.py`. Recovery iterations
must use deterministic, cycle-specific checkpoint names; reusing a completed name causes
Absurd to replay its stored value instead of executing new physical work. Keep the final
`complete` result behind successful verification, cap recovery loops, and configure
task-level retries for transient failures. Long-running model or motion calls must send
heartbeats often enough to retain their worker claim.

The CLI entry points are `src/everest_robot/client.py`,
`src/everest_robot/worker.py`, `src/everest_robot/database.py`,
`src/everest_robot/monitor.py`, `src/everest_robot/goto.py`, and
`src/everest_robot/jog.py`. Keep hardware-specific
configuration out of these modules; pass selection and tuning data through validated task
parameters or environment-backed configuration.

Add deterministic unit tests under `tests/` for every adapter result and recovery branch.
Before handing off a change, run `just check`. For orchestration changes, also run the
database-backed smoke path on whichever backend `just db-backend` reports: `just db-up`,
`just db-init`, `just worker`, then in another terminal
`just start <workflow-id> <verification-failures>`.
