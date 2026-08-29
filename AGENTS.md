# Repository guidance

Use `just` as the primary interface for development commands. Run `just` to list the
available recipes, `just setup` to install dependencies, `just db-up` and `just db-init`
for first-time database setup, `just worker` to run the Absurd worker, and `just check`
before handing off changes. Recipes automatically load local values from `.env`.

Keep robot hardware, policy, and perception code behind the adapter boundary in
`src/everest_robot/adapters.py`. Keep durable orchestration and checkpoint naming in
`src/everest_robot/workflow.py`. Any physical side effect must be designed to tolerate
retries and brief overlapping execution.

## Working on workflow components

Shared data contracts belong in `src/everest_robot/domain.py`. Add or change a typed
result there before altering an adapter or workflow stage, and keep every persisted value
JSON-serializable. Treat persisted field names and checkpoint names as durable interfaces:
renaming either can affect workflows that started on an older revision.

The placeholder implementations live on `ScaffoldRobot` in
`src/everest_robot/adapters.py`. Preserve this boundary when adding production code:

- **Carabiner pickup:** implement RL-policy or VLA invocation in
  `pick_up_carabiner()`. Return an explicit `secured` result. Raise only for retryable
  execution failures; a completed but unsuccessful grasp should remain observable to the
  workflow. Make robot commands idempotent or guard them with a physical-state check.
- **Rope localization:** implement deterministic CV or VLM inference in `locate_rope()`.
  Return a pose with an explicit coordinate frame and record which detector produced it.
  Do not hide coordinate transforms or mutable camera state in the workflow.
- **Attachment control:** implement deterministic motion and force control in
  `attach_carabiner()`. Consume the typed rope pose and return measurable control results.
  Keep control loops inside the adapter; use Absurd checkpoints for stage boundaries, not
  for every servo update.
- **Attachment verification:** implement sensor/CV/VLM fusion in
  `verify_attachment()`. Return both the verdict and a `RecoveryTarget` so the workflow
  can deliberately route back to pickup, localization, or attachment. Add a new recovery
  target only alongside its orchestration branch and tests.

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
