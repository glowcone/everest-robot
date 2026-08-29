# Repository guidance

Use `just` as the primary interface for development commands. Run `just` to list the
available recipes, `just setup` to install dependencies, `just db-up` and `just db-init`
for first-time database setup, `just worker` to run the Absurd worker, and `just check`
before handing off changes. Recipes automatically load local values from `.env`.

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
  `maker_arm_port.py` adapts `maker_arm.Arm`; `fake_arm.py` is the deterministic stand-in
  every layer above is tested against.
- Soft limits, watchdogs, velocity limiting, fault handling and coordinate conversion
  belong to `maker-arm-sdk` and must not be re-implemented here. See
  `docs/adr/0001-production-motor-protocol.md` for the ownership split.
- `parameters.py` loads `config/maker_arm_v1.yaml` strictly: unknown fields are rejected,
  and a preset whose `calibration_id` does not match the file's robot is refused. Presets
  come from `docs/named-position-capture.md`, never from hand-edited joint values.
- `motion.py`, `policy.py` and `session.py` take an injectable `Clock`
  (`clock.ManualClock` in tests), a heartbeat and a cancellation check. Absurd signals
  cancellation by raising from `ctx.heartbeat()`, so any loop that commands the arm must
  leave it held on `BaseException`, not only on its own failure paths.
- `recording.py`, `replay.py` and `policy.LeRobotPolicyHandle` are guarded stubs pending
  the stored-session format decision. Keep them refusing with an actionable message rather
  than guessing a format or a feature mapping.
- Third-party robotics dependencies live in the `hardware` extra and are imported lazily.
  A new import of `lerobot` or `maker_arm` at module scope breaks the hardware-free tests.
- `deployment.py` owns every environment-specific value (CAN interface, cameras, lease
  backend, parameters path). Do not read the environment anywhere else in the runtime.

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
`src/everest_robot/worker.py`, and `src/everest_robot/database.py`. Keep hardware-specific
configuration out of these modules; pass selection and tuning data through validated task
parameters or environment-backed configuration.

Add deterministic unit tests under `tests/` for every adapter result and recovery branch.
Before handing off a change, run `just check`. For orchestration changes, also run the
Docker-backed smoke path: `just db-up`, `just db-init`, `just worker`, then in another
terminal `just start <workflow-id> <verification-failures>`.
