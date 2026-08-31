# Repository guidance

Use `just` as the primary interface for development and operations. Run `just` to list the
recipes, which are grouped: `setup` (`setup`, `setup-hardware`, `config`, and the
read-only `camera-scan`, `camera-scan-json`, `camera-show`), `robot`
(`monitor` -- powered calibration teleoperation; `goto` -- powered named-position motion;
`goto-list`, `goto-dry`, `monitor-read-only`, `monitor-once`, `monitor-fake` -- no motion),
`database`
(`db-backend`, `db-up`, `db-init`, `db-reset`, `psql`), `calibration` (two numbered paths
for the same `SEARCH_CV` state -- `pixel-*` teaches the fixed camera's pixel-to-joint map,
`wrist-*` teaches the wrist camera's image Jacobian),
`workflow` (`attach-fsm-fake`, `attach-fsm`, `worker`, `start`, `tasks`, `task`,
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

The real-time attachment state machine is the deliberate exception to the durable workflow:
`src/everest_robot/attachment_fsm.py` owns only states, transition guards, budgets, and its
diagnostic trace. It has no Absurd or hardware dependency. Its production handlers live
behind `EverestAttachmentFSMHandlers` in `adapters.py`, and one invocation holds one
`RobotSession` lease throughout. Read `docs/adr/0003-realtime-attachment-fsm.md` and
`docs/attachment-fsm.md` before integrating a handler. Never put an Absurd checkpoint or
database call inside its action loop.

Run it with `just attach-fsm-fake` (add `--initial-detection` to enter at `SEARCH_CV`),
which exercises every state and prints the JSON result with no hardware, camera, or
database -- this is the command to reach for when changing the FSM. `--skip-cv`
(`AttachmentFSMConfig(use_search_cv=False)`, `just attach-fsm-rl`) removes `SEARCH_CV` from
the graph so a learned policy can be measured without visual following: detections hand
straight to `CLIP_RL`, a degraded alignment stays there, and neither calibration is read. It gives
up the guarantee the state exists for -- that the measured pose settled on the taught
pre-grasp target before clipping began -- so keep it an option, never the default. `just attach-fsm
<search-policy> <clip-policy>` is the hardware form, and `just attach-fsm-act <checkpoint>`
is the intended shape: one model in both learned states with classical CV between them,
still as two separate `PolicySession`s. Every state is implemented -- `INITIAL` and the
`CLIP_RL` gates come from `robot/carabiner_perception.py` over the wrist camera. `SEARCH_CV`
comes in two interchangeable forms selected by `EVEREST_SEARCH_CV` / `--search-cv`: `fixed`
(the bench camera's pixel map) and `wrist` (the wrist camera's image Jacobian). Exactly one
calibration is loaded; supplying both is refused rather than ranked. Two signals
are still refused rather than invented: attachment verification (behind
`AttachmentVerifier`; without it `SUCCESS` is unreachable, so it must be acknowledged with
`--no-attachment-verification`) and grasp detection (needs a measured
`EVEREST_GRASP_GRIPPER_BELOW_RAD`; unset means a conservative "not grasped"). Keep both
refusals in `preflight()`, before the claim: do not let a gate that cannot be read be
discovered after a learned action has already moved the arm. Perception shares the open
session's `CameraRuntime` -- never open the wrist camera a second time.

`CLIP_RL` reads "the policy returned to neutral" from `select_action()` returning `None`.
That mapping is marked UNVERIFIED in ADR-0003 and in `clip_rl_step`'s docstring, and a
scripted policy cannot confirm it. Do not delete the marker or write a test that claims to
verify it; confirming it needs a trained checkpoint on the arm. Completion is only a
candidate: reset also requires fresh stationary feedback within the operator-captured
`neutral` named-position tolerance. Never reset from the policy signal alone.

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
- ACT inference enables LeRobot temporal ensembling by default with coefficient `0.01` and
  `n_action_steps = 1`. Keep the override in `robot/policy.py`, before model construction;
  `--no-temporal-ensemble` is the operator escape hatch for ordinary chunk execution.
- `policy.PolicySession` is the one-action primitive the FSM's learned states run on:
  `seed()` / `step()` / `close()`, keeping policy context across the decisions the caller
  makes in between. Never emulate it with repeated `PolicyRunner.run(max_steps=1)`, which
  resets policy/processor state, recording, and the arm hold on every call. It is separate
  from `PolicyRunner` on purpose and the two pace differently: the runner follows an
  absolute schedule because it owns an uninterrupted rollout, while a session sets each
  deadline from when the previous action went out and absorbs a late step rather than making
  it up -- the same rule replay follows for a late frame. Do not unify them.
- `robot/readiness.py` owns INITIAL's passive admission check. It runs with torque off and
  requires advancing finite feedback, no faults, in-limit positions, stationarity, and
  configured camera shapes. `RobstrideMitPort.enable()` obtains another fresh sample and
  seeds the first MIT goal from that measured pose. Never weaken freshness to accept cache.
- `policy.load_policy()` is the only place a reference becomes a `PolicyHandle`. A checkpoint
  directory or a Hugging Face repo id routes to `LeRobotPolicyHandle`; a `.json` scripted
  policy is read strictly and is for rehearsal, never a trained policy. All are resolved --
  downloaded, cross-checked, weights loaded, device chosen -- before the robot is claimed.
  Add a new format here rather than teaching a caller what a checkpoint is.
- `checkpoints.py` owns the feature mapping, and it is the reason a checkpoint can be loaded
  at all: which `{joint}.pos` and which camera become which tensor slice is read from the
  *training dataset's* `meta/info.json`, found via the checkpoint's own `train_config.json`.
  Never derive it from the connected robot, and never fall back to positional order --
  `compatibility_problems()` compares the result against the arm, in order, and refuses a
  mismatch. A checkpoint with no recorded training dataset is refused, not guessed at.
- Device selection lives in `policy.resolve_torch_device` (`EVEREST_POLICY_DEVICE`): `auto`
  prefers CUDA, then MPS, then CPU, and an explicit choice is never silently downgraded. A
  checkpoint records the device it was trained on in both its config and its saved
  preprocessor's `device_processor` step; both are overridden at load time, which is what
  lets a CUDA-trained checkpoint run on Metal. Do not remove either override.
- `recording.py` is still a guarded stub pending the dataset-writing decision. Keep it
  refusing with an actionable message rather than guessing a format.
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
  written back; the two `SEARCH_CV` calibration paths -- `EVEREST_PIXEL_MAP` and
  `EVEREST_WRIST_SERVO`, loaded and refused up front by `load_pixel_map()` and
  `load_wrist_servo()`, with `search_cv_backend()` (`EVEREST_SEARCH_CV`) choosing between
  them). Do not read the environment anywhere else in the runtime.
- `visual_tracking.py` is the bounded servo loop for a target that changes every frame. It
  takes a full joint target or `None` per tick and clamps the commanded step, so the arm's
  speed is bounded by construction and a missing detection holds rather than coasting.
  `start()` is re-callable and enables only from `CONNECTED`, because the FSM re-enters
  `SEARCH_CV` on an arm the clip policy left energized and both drivers refuse `enable()`
  from `ENABLED`. Where the target comes from is the caller's problem, and there are two
  callers because there are two cameras. Do not merge them and do not let either become
  the other's special case.
- **Fixed camera.** `pixel_map.py` fits the bench camera's pixels to taught pre-grasp poses
  and refuses to extrapolate outside their convex hull; `calibrate_pixel_map.py`
  (`robot-pixel-map`) is the operator's CLI. `robot/carabiner_follower.py` is the caller:
  camera, two-white-tape detector and map, one tracker tick at a time. It owns the three
  judgements neither the tracker nor the FSM can make -- how many consecutive misses mean
  the target is lost, what "followed" means (the *measured* pose settled, not one tick that
  moved), and the pacing that keeps the per-tick clamp equal to `max_velocity_rad_s`. It
  never chooses a next state. `calibrate_pixel_map.py` re-exports its camera and detector,
  so the `track` subcommand and the FSM run the same loop minus the arbitration.
- **Wrist camera.** `robot/wrist_servo.py` holds an image Jacobian measured *on the arm* --
  a wrist pixel names a direction relative to the gripper, not a place on the bench, so
  there is nothing to sample and no pose to look up. `robot/wrist_follower.py` closes the
  loop in the image against a taught goal, owns the same three judgements, and returns the
  same `FollowTick` so the handler cannot tell the two apart. It reads frames through the
  session's `CameraRuntime` -- the wrist camera is a policy observation and is already
  open. `calibrate_wrist_servo.py` (`robot-wrist-servo`) teaches it by bumping one joint at
  a time, using the *measured* displacement, and refuses the whole teach if the goal image
  has drifted by the end -- that check is what catches a carabiner nudged mid-procedure,
  which nothing in the fitted numbers would reveal. A solve past `max_delta_rad` is refused,
  never clamped. `robot-wrist-servo centre` is the debug loop and the one to reach for
  first: it retargets the stored calibration **in memory** onto the frame centre, servoing
  the spine midpoint with `scale_px` and `spine_deg` set to `IGNORED_TOLERANCE`, so a
  Jacobian column with the wrong sign shows up at once as the arm going the wrong way. Keep
  it writing nothing, and keep "ignore this feature" expressed as a loose tolerance rather
  than a branch in the solve. Read `docs/wrist-servo-calibration.md` before changing the
  model, the feature set or the tolerances; the tolerances are also the solve's weights, so
  they are not free to retune in isolation.
- `robot/monitor.py` remains the read-only feedback view model. The `robot-monitor` CLI
  defaults to a powered, lease-local calibration session: `robot/teleoperation.py` owns the
  Star leader and commands the already-claimed follower while the same process renders that
  view. `--read-only` and `--once` never enable. Never split control and monitoring across
  processes or allow either to bypass the robot lease. Pressing `p` captures a pose; the
  save prompt for it runs in `monitor.py` after the session has closed, so the arm is held,
  disabled and released before anyone is waiting on a keyboard. Do not move that prompt
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
  `named_transitions` sequence over the direct line with no flag to override it.

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

The standalone attachment FSM is not durable orchestration. Its handler methods perform at
most one physical action and return the typed result that the FSM uses for its transition;
handlers must not contain hidden retry loops or choose the next state. `INITIAL` is
motion-free, `SEARCH_CV` is integrated and steps one `CarabinerFollower` tick (the
calibration is verified against the connected arm at handler construction, not at the first
servo tick), and `SEARCH_RL`/`CLIP_RL` use distinct persistent policy sessions even when
they share a checkpoint. There is no automatic neutral
or known-position command. When `CLIP_RL` reports that the policy returned to neutral, the
FSM returns to `INITIAL` and clears per-cycle state budgets; keep the lifetime total-action
and wall-clock budgets finite. Treat the trace as diagnostics rather than resumable physical
state.

The CLI entry points are `src/everest_robot/client.py`,
`src/everest_robot/worker.py`, `src/everest_robot/database.py`,
`src/everest_robot/monitor.py`, `src/everest_robot/goto.py`,
`src/everest_robot/jog.py`, the two calibration CLIs
`src/everest_robot/calibrate_pixel_map.py` and `src/everest_robot/calibrate_wrist_servo.py`,
plus the local `src/everest_robot/fsm_cli.py` and the
hardware-free `src/everest_robot/policy_check.py` and
`src/everest_robot/cameras_cli.py`. Keep hardware-specific
configuration out of these modules; pass selection and tuning data through validated task
parameters or environment-backed configuration.

`src/everest_robot/cameras_cli.py` (`robot-cameras`) is the operator's answer to "which
camera is `wrist`" -- the one pairing in `EVEREST_CAMERAS` that no check in the system can
verify, because a swapped id satisfies every shape, feature and readiness gate and only
looks through the wrong lens. `scan` probes the host's capture devices and draws each id
into its own window; `show` opens the *configured* cameras through the same `CameraRuntime`
a rollout uses, which is what catches a size the driver quietly refused. Every decision it
makes -- which ids to probe, which by-id link is the stable one, what the label says, what
the skeleton configuration would be -- is a pure function tested without a camera or a
window, and it never claims the robot lease. Keep it that way: it is an instrument, and the
moment it can move something it stops being safe to run beside a held arm. Note that
`lerobot` depends on `opencv-python-headless`, which overwrites `opencv-python`'s `cv2`, so
a hardware environment can end up with no `imshow` at all -- `_no_window_message` names that
cause, and it applies to `robot-pixel-map`'s windows too. See
`docs/camera-identification.md`.

`src/everest_robot/carabiner_detect.py` is the wrist-camera carabiner detector. It lives
inside the installed package rather than in the repository's top-level `carabiner/` (now a
re-export shim) because `robot-attach-fsm` is a console script: `sys.path` starts at the
script's directory, not the working directory, so a top-level package is not importable from
it. Runtime perception code must import `everest_robot.carabiner_detect`.

Its two arbitration rules are load-bearing and were both built from real failures. The
hysteresis floor is `LOW_THRESHOLDS`, walked **strictest first**, descending only when
nothing validated above it -- lowering it globally wins far-field frames and loses more
near-field ones, which is measurable on the capture corpus, so never turn the ladder into a
single tuned constant. And `_plausible` returns *every* size-plausible component, largest
first, for `_validate` to arbitrate: the largest teal thing in view is often a screen or a
plant, and committing to it is what made a carabiner in plain view report "no detection".
Neither of those can reject a ring-shaped object across the room -- nothing in one frame
can -- so `detect(frame, roi_xywh)` takes a search region from `EVEREST_WRIST_ROI`. Keep the
ROI masking the *score* rather than cropping the image: every caller works in absolute
pixels.

Add deterministic unit tests under `tests/` for every adapter result and recovery branch.
Before handing off a change, run `just check`. For orchestration changes, also run the
database-backed smoke path on whichever backend `just db-backend` reports: `just db-up`,
`just db-init`, `just worker`, then in another terminal
`just start <workflow-id> <verification-failures>`.
