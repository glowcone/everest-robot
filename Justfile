set dotenv-load

default:
    @just --list

# Install the locked Python environment.
setup:
    uv sync

# Start Postgres and wait until it is healthy.
db-up:
    docker compose up -d --wait postgres

# Stop the local containers without deleting database data.
db-down:
    docker compose down

# Install Absurd's schema and create the robot queue (first-time setup).
db-init:
    uv run robot-db-init

# Run the durable workflow worker.
worker:
    uv run robot-worker

# Spawn a workflow. Example: just start retry-demo 1
start workflow_id="demo" verification_failures="0":
    uv run robot-start --workflow-id {{ workflow_id }} --verification-failures {{ verification_failures }}

# Validate a replay request locally. Claims nothing, energizes nothing.
# Example: just replay-preflight h8i76dfsd9/test1_20260829_130743 55e5611... 0
replay-preflight repo revision episode="0" policy="clamp_within_tolerance" deviation="2.0":
    uv run robot-replay-preflight \
        --repo-id {{ repo }} --revision {{ revision }} --episode {{ episode }} \
        --robot-id "$EVEREST_ROBOT_ID" --calibration-id "$EVEREST_CALIBRATION_ID" \
        --limit-policy {{ policy }} --max-limit-deviation-deg {{ deviation }}

# Spawn a replay workflow. Add --dry-run through `just` by passing dry="--dry-run".
replay repo revision episode="0" workflow_id="replay" dry="":
    uv run robot-replay --workflow-id {{ workflow_id }} {{ dry }} \
        --repo-id {{ repo }} --revision {{ revision }} --episode {{ episode }} \
        --robot-id "$EVEREST_ROBOT_ID" --calibration-id "$EVEREST_CALIBRATION_ID" \
        --limit-policy clamp_within_tolerance --max-limit-deviation-deg 2.0

# Run lint and unit tests.
check:
    uv run ruff check .
    uv run pytest
