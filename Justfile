# Everest robot: development and operations commands.
#
# Recipes load .env automatically (copy .env.example to .env first). Run `just` to list
# them by group, or `just --list` for a flat list.
#
# Two workflows exist:
#   attach-carabiner  the carabiner attachment state machine  (just start)
#   replay-session    replay of a stored dataset episode      (just replay ...)
#
# Anything that can move a real arm is grouped under "replay", and every one of those
# recipes is safe to read before it is safe to run: see docs/session-replay.md. The "robot"
# group is read-only: those recipes claim the arm but never energize it.

set dotenv-load

default:
    @just --list --unsorted

# ── setup ──────────────────────────────────────────────────────────────────────────

# Install the locked environment (no torch, no CAN). Enough for everything but real hardware.
[group('setup')]
setup:
    uv sync

# Add the pinned robot SDKs (maker-arm, lerobot+torch). Only needed to drive a real arm.
[group('setup')]
setup-hardware:
    uv sync --extra hardware

# Print the resolved robot parameters and deployment environment. Touches nothing.
[group('setup')]
config:
    uv run robot-config

# ── robot ──────────────────────────────────────────────────────────────────────────
# Read-only instruments. They claim the arm, so a worker cannot be holding it, but they
# never enable the motors and never command a target.

# Watch every joint's encoder feedback in a TUI. Claims the arm; enables nothing.
[group('robot')]
monitor poll_hz="10":
    uv run robot-monitor --poll-hz {{ poll_hz }}

# Print one joint-feedback snapshot as plain text. Redirect it next to a captured pose.
[group('robot')]
monitor-once:
    uv run robot-monitor --once

# Show the TUI against the deterministic fake arm. No CAN, no claim, no real numbers.
[group('robot')]
monitor-fake:
    uv run robot-monitor --fake

# Convert one camera pixel into robot-base X/Y from a measured calibration; no hardware.
[group('robot')]
camera-to-xy u v config="pickup_config.json":
    uv run robot-camera-to-xy --config {{ config }} --point {{ u }} {{ v }} --show-matrix

# Convert one LeRobot seven-joint frame (degrees) into robot-base tool XYZ; no hardware.
[group('robot')]
dataset-fk q1 q2 q3 q4 q5 q6 gripper:
    uv run robot-kinematics dataset-fk --joints-deg \
        {{ q1 }} {{ q2 }} {{ q3 }} {{ q4 }} {{ q5 }} {{ q6 }} {{ gripper }}

# Read encoders and emit one camera/robot calibration pair; connects but never enables.
[group('robot')]
capture-calibration-point u v:
    uv run robot-kinematics capture --camera-u {{ u }} --camera-v {{ v }}

# ── database ───────────────────────────────────────────────────────────────────────

# Start Postgres and wait until it is healthy.
[group('database')]
db-up:
    docker compose up -d --wait postgres

# Stop the local containers without deleting database data.
[group('database')]
db-down:
    docker compose down

# Install Absurd's schema and create the robot queue (first-time setup).
[group('database')]
db-init:
    uv run robot-db-init

# Delete all database state and start over, clearing tasks from older workflow revisions.
[group('database')]
db-reset:
    docker compose down -v
    just db-up
    just db-init

# Open a psql shell in the Postgres container (the host needs no PostgreSQL client).
[group('database')]
psql:
    docker compose exec postgres psql -U robot -d robot

# ── workflows ──────────────────────────────────────────────────────────────────────

# Run the durable workflow worker. Leave it running; it serves both task types.
[group('workflow')]
worker:
    uv run robot-worker

# Spawn an attachment workflow; arg 2 simulates that many failed verifications first.
[group('workflow')]
start workflow_id="demo" verification_failures="0":
    uv run robot-start --workflow-id {{ workflow_id }} --verification-failures {{ verification_failures }}

# List recent tasks and their state.
[group('workflow')]
tasks limit="20":
    docker compose exec -T postgres psql -U robot -d robot -c \
        "SELECT task_id, task_name, state, attempts, enqueue_at \
         FROM absurd.t_robot ORDER BY enqueue_at DESC LIMIT {{ limit }};"

# Show one task's parameters and its durable result.
[group('workflow')]
task task_id:
    docker compose exec -T postgres psql -U robot -d robot -c \
        "SELECT task_name, state, attempts, jsonb_pretty(params) AS params, \
                jsonb_pretty(completed_payload) AS result \
         FROM absurd.t_robot WHERE task_id = '{{ task_id }}';"

# Show one task's committed checkpoints: the record of which stages actually completed.
[group('workflow')]
checkpoints task_id:
    docker compose exec -T postgres psql -U robot -d robot -c \
        "SELECT checkpoint_name, status, updated_at, jsonb_pretty(state) AS value \
         FROM absurd.c_robot WHERE task_id = '{{ task_id }}' ORDER BY updated_at;"

# Cancel a task. A running stage stops at its next heartbeat, holding the arm on the way out.
[group('workflow')]
cancel task_id:
    docker compose exec -T postgres psql -U robot -d robot -c \
        "SELECT absurd.cancel_task('${ROBOT_QUEUE:-robot}', '{{ task_id }}');"

# ── replay ─────────────────────────────────────────────────────────────────────────
# The numbered recipes are the ordered path to a powered replay. Run them in order; step 0
# is the smallest powered motion check, and docs/session-replay.md covers the rest.

# 0. Raise one joint a little from its measured pose. Pass dry="--dry-run" to plan only.
[group('replay')]
raise delta="0.10" speed="0.25" dry="":
    uv run robot-raise --delta-rad {{ delta }} --speed-scale {{ speed }} {{ dry }}

# 1. Print the preflight report locally. Claims nothing, energizes nothing, needs no worker.
[group('replay')]
replay-preflight repo revision episode="0" policy="clamp_within_tolerance" deviation="2.0":
    uv run robot-replay-preflight \
        --repo-id {{ repo }} --revision {{ revision }} --episode {{ episode }} \
        --robot-id "$EVEREST_ROBOT_ID" --calibration-id "$EVEREST_CALIBRATION_ID" \
        --limit-policy {{ policy }} --max-limit-deviation-deg {{ deviation }}

# 2. The same validation through the durable workflow. Still commands no motion.
[group('replay')]
replay-dry-run repo revision episode="0" workflow_id="replay-dry":
    just _replay {{ repo }} {{ revision }} {{ episode }} {{ workflow_id }} "--dry-run" "1.0" "0" ""

# 3. Drive to the episode's recorded start pose and stop. The first recipe that MOVES the arm.
[group('replay')]
replay-align repo revision episode="0" workflow_id="replay-align":
    just _replay {{ repo }} {{ revision }} {{ episode }} {{ workflow_id }} "" "0.25" "0" "0"

# 4. Replay a short range at reduced speed. Widen the range and speed only once each is clean.
[group('replay')]
replay-range repo revision episode="0" workflow_id="replay-range" end_frame="30" speed="0.25":
    just _replay {{ repo }} {{ revision }} {{ episode }} {{ workflow_id }} "" {{ speed }} "0" {{ end_frame }}

# 5. Replay the full episode. Each attempt needs its own workflow id.
[group('replay')]
replay repo revision episode="0" workflow_id="replay" speed="1.0":
    just _replay {{ repo }} {{ revision }} {{ episode }} {{ workflow_id }} "" {{ speed }} "0" ""

# Shared spawn used by the replay recipes above.
[private]
_replay repo revision episode workflow_id dry speed start_frame end_frame:
    uv run robot-replay --workflow-id {{ workflow_id }} {{ dry }} \
        --repo-id {{ repo }} --revision {{ revision }} --episode {{ episode }} \
        --robot-id "$EVEREST_ROBOT_ID" --calibration-id "$EVEREST_CALIBRATION_ID" \
        --speed {{ speed }} --start-frame {{ start_frame }} \
        {{ if end_frame == "" { "" } else { "--end-frame " + end_frame } }} \
        --limit-policy clamp_within_tolerance --max-limit-deviation-deg 2.0

# ── development ────────────────────────────────────────────────────────────────────

# Lint and unit tests. Run before handing off any change.
[group('dev')]
check:
    uv run ruff check .
    uv run pytest

# Run the unit tests, passing any extra arguments through to pytest.
[group('dev')]
test *args:
    uv run pytest {{ args }}

# Lint only.
[group('dev')]
lint:
    uv run ruff check .

# Apply ruff's automatic fixes.
[group('dev')]
fmt:
    uv run ruff check --fix .

# Also run the tests that download the real dataset from Hugging Face.
[group('dev')]
test-network:
    EVEREST_HF_NETWORK_TESTS=1 uv run pytest
