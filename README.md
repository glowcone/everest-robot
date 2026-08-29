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

## Run locally

Prerequisites: Docker, Python 3.12+, and [uv](https://docs.astral.sh/uv/).
Developer commands use [just](https://just.systems/); run `just` to see all recipes.

```bash
cp .env.example .env
just setup
just db-up
just db-init
```

Run the worker:

```bash
just worker
```

In another shell, start a workflow:

```bash
just start
```

To demonstrate verification feedback, request one simulated failed verification before
success. Use a new workflow ID because start requests are idempotent:

```bash
just start retry-demo 1
```

Inspect recent tasks through the PostgreSQL client inside the container:

```bash
docker compose exec postgres psql -U robot -d robot \
  -c "SELECT task_id, task_name, state, attempts FROM absurd.t_robot ORDER BY task_id DESC LIMIT 20;"
```

Run local checks with `just check`. The `db-init` recipe uses the SQL bundled with the
locked `absurdctl` package through psycopg, so PostgreSQL client tools do not need to be
installed on the host.

## The robot SDK layer

`src/everest_robot/robot/` holds the runtime that drives a real Maker Arm: an exclusive
lease and session lifecycle, a strictly validated robot parameters file, bounded motion
between operator-approved named positions, LeRobot's `Robot` contract over the arm, and a
synchronous policy rollout runner. The production driver decision is recorded in
[`docs/adr/0001-production-motor-protocol.md`](docs/adr/0001-production-motor-protocol.md):
the arm keeps running maker-arm-sdk's RobStride private protocol, which stays the hardware
safety boundary.

`maker-arm` and `lerobot` are pinned to commit SHAs behind an optional extra and imported
lazily, so the package and its whole test suite run without them:

```bash
uv sync                    # core: no torch, no CAN
uv sync --extra hardware   # add the real SDKs
```

Named positions ship empty on purpose. Capture them from a measured, operator-approved arm
state by following [`docs/named-position-capture.md`](docs/named-position-capture.md); the
loader refuses any preset whose calibration identity does not match the connected arm.

Session recording, replay and LeRobot checkpoint loading are interface-only stubs pending
the stored-session format decision, and refuse rather than guessing a format.

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

The queue and database are configured with `ROBOT_QUEUE` and `ABSURD_DATABASE_URL`.
Postgres data lives in the Docker volume `postgres-data`.
