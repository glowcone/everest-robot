# Everest Robot

A Python/uv scaffold for a deterministic, durable robot workflow powered by
[Absurd](https://earendil-works.github.io/absurd/) and Postgres.

The physical robot integrations are deliberately placeholders. The scaffold makes their
boundaries explicit while exercising the orchestration shape:

1. Pick up a carabiner with an RL policy or VLA. A failed action raises so Absurd retries
   the task; a successful result is checkpointed in Postgres.
2. Locate the rope with deterministic CV or a VLM.
3. Attach the carabiner with deterministic control.
4. Verify the attachment and route back to pickup, localization, or attachment. Each
   recovery cycle uses unique checkpoint names so feedback creates new work.
5. Return a durable `complete` result only after verification succeeds.

## Run locally

Prerequisites: Docker, Python 3.11+, and [uv](https://docs.astral.sh/uv/).
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

Inspect tasks with `uv run absurdctl list-tasks --queue robot`. Run local checks with
`just check`.
The `db-init` recipe uses the SQL bundled with the locked `absurdctl` package through
psycopg, so PostgreSQL client tools do not need to be installed on the host.

## Integration points

Replace `ScaffoldRobot` methods in `src/everest_robot/adapters.py` with hardware-facing
adapters. Keep side effects idempotent: Absurd guarantees persisted checkpoints, while a
worker can briefly overlap execution around crashes. Long-running policy/control calls
should also heartbeat before the worker claim expires.

The queue and database are configured with `ROBOT_QUEUE` and `ABSURD_DATABASE_URL`.
Postgres data lives in the Docker volume `postgres-data`.
