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

# Run lint and unit tests.
check:
    uv run ruff check .
    uv run pytest
