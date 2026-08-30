# Everest robot: development and operations commands.
#
# Recipes load .env automatically (copy .env.example to .env first). Run `just` to list
# them by group, or `just --list` for a flat list.
#
# Two workflows exist:
#   attach-carabiner  the carabiner attachment state machine  (just start)
#   replay-session    replay of a stored dataset episode      (just replay ...)
#
# Powered calibration teleoperation lives in `monitor` and powered named-position motion in
# `goto`; replay motion stays in the ordered replay group. Every robot recipe claims the arm
# exclusively.

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

# Which camera is `wrist` and which is `front` is the one fact nothing in the running system
# can tell you: a swapped id trains and runs a policy on the wrong view while every check
# still passes. `camera-scan` puts each capture device in its own window with its id drawn
# into the picture, so you identify them by waving a hand; `camera-scan-json` prints an
# EVEREST_CAMERAS skeleton for what it found; `camera-show` opens what is already configured
# through the same runtime a rollout uses, which is what catches a resolution the driver
# quietly refused. All three are read-only and never claim the arm -- but they do hold the
# cameras, so stop them before starting a rollout. See docs/camera-identification.md.

# Show every capture device in its own window, labelled with the id to configure. No motion.
[group('setup')]
camera-scan *args:
    uv run robot-cameras scan {{ args }}

# List the cameras this host can see and print an EVEREST_CAMERAS skeleton. No windows.
[group('setup')]
camera-scan-json *args:
    uv run robot-cameras scan --no-window --json {{ args }}

# Open the configured EVEREST_CAMERAS by name and flag any shape the driver refused.
[group('setup')]
camera-show *args:
    uv run robot-cameras show {{ args }}

# ── robot ──────────────────────────────────────────────────────────────────────────
# Calibration teleoperation, named-position motion, and explicit read-only instruments.
# All claim the arm, so a worker cannot be holding it. The `goto` recipes are ordered:
# `goto-list` reads the configuration, `goto-dry` claims and validates without energizing,
# and only `goto` moves.

# Follow the Star leader at bounded speed (rad/s) while watching follower encoders. POWERED.
[group('robot')]
monitor poll_hz="10" max_velocity="0.25":
    uv run robot-monitor --poll-hz {{ poll_hz }} --max-velocity {{ max_velocity }}

# Watch encoders with follower torque disabled and no Star leader.
[group('robot')]
monitor-read-only poll_hz="10":
    uv run robot-monitor --read-only --poll-hz {{ poll_hz }}

# Print one joint-feedback snapshot as plain text. Redirect it next to a captured pose.
[group('robot')]
monitor-once:
    uv run robot-monitor --once

# Show the TUI against the deterministic fake arm. No CAN, no claim, no real numbers.
[group('robot')]
monitor-fake:
    uv run robot-monitor --fake

# List the approved named positions and the transitions that reach them. Touches nothing.
[group('robot')]
goto-list:
    uv run robot-goto --list

# Validate a named-position move against the live limits and measured pose. Moves nothing.
[group('robot')]
goto-dry position:
    uv run robot-goto {{ position }} --dry-run

# Drive the arm to an approved named position, using its transition if one exists. POWERED.
[group('robot')]
goto position speed="0.25":
    uv run robot-goto {{ position }} --speed-scale {{ speed }}

# ── calibration ────────────────────────────────────────────────────────────────────
# The fixed camera's pixels to joint positions, in the order they are run. Bolt the camera
# before step 0 and do not move it afterwards: moving it voids every sample and the whole
# procedure repeats. Steps 0 and 4 hold the arm lease and MOVE the arm -- 0 follows the
# Star leader, 4 servos to the detection at a locked speed and holds still whenever it
# sees nothing. Steps 1-3 touch no hardware. The full procedure, including the wrist-roll
# model and the two hard limits, is in src/everest_robot/calibrate_pixel_map.py.

# 0. POWERED: teleoperate to ~30 pre-grasp poses, pairing each with the object's pixel.
[group('calibration')]
pixel-collect camera x y w h speed="0.375":
    uv run robot-pixel-map collect --camera {{ camera }} \
        --roi {{ x }} {{ y }} {{ w }} {{ h }} --max-velocity {{ speed }}

# 1. Refit the stored samples and print the held-out joint error. Moves nothing.
[group('calibration')]
pixel-fit model="thin_plate_spline":
    uv run robot-pixel-map fit --model {{ model }}

# 2. Print the calibration: camera, arm, sampled region, roll offset, holdout report.
[group('calibration')]
pixel-check:
    uv run robot-pixel-map check

# 3. Print the joint vector for whatever the camera sees right now. Moves nothing.
[group('calibration')]
pixel-predict frames="1":
    uv run robot-pixel-map predict --frames {{ frames }}

# 3b. The tracking loop with the arm never energized: watch what it would command.
[group('calibration')]
pixel-track-dry speed="0.15":
    uv run robot-pixel-map track --max-velocity {{ speed }} --dry-run

# 4. POWERED: servo continuously above the detected carabiner at a locked speed.
[group('calibration')]
pixel-track speed="0.15" rate="15":
    uv run robot-pixel-map track --max-velocity {{ speed }} --rate {{ rate }}

# ── database ───────────────────────────────────────────────────────────────────────
# Docker is optional. `robot-db` uses compose.yaml when Docker Compose and its daemon are
# both available, and an already-installed PostgreSQL (Homebrew, a package) otherwise.
# EVEREST_DB_BACKEND=docker|native forces the choice; both use ABSURD_DATABASE_URL.

# Print which Postgres backend these recipes will use, and against which URL.
[group('database')]
db-backend:
    uv run robot-db backend

# Start Postgres and wait until it accepts connections.
[group('database')]
db-up:
    uv run robot-db up

# Stop the containers without deleting data. A shared host server is left running.
[group('database')]
db-down:
    uv run robot-db down

# Install Absurd's schema and create the robot queue (first-time setup).
[group('database')]
db-init:
    uv run robot-db-init

# Delete all database state and start over, clearing tasks from older workflow revisions.
[group('database')]
db-reset:
    uv run robot-db reset
    just db-init

# Open a psql shell against the robot database, wherever it is running.
[group('database')]
psql:
    uv run robot-db psql

# ── workflows ──────────────────────────────────────────────────────────────────────
# `attach-fsm` is the local real-time orchestrator from ADR-0003. It holds one robot lease
# for the complete attempt and does not require Absurd or PostgreSQL. The learned states
# (SEARCH_RL, CLIP_RL) load a checkpoint or a scripted policy, SEARCH_CV drives the fixed
# camera and the pixel map from `pixel-fit`, and INITIAL plus the CLIP_RL gates come from
# the wrist-camera detector. Everything -- checkpoints, feature mapping, perception, pixel
# map -- resolves before the robot is claimed. Use `attach-fsm-fake` to exercise the state
# machine with no hardware at all, and `pixel-track` to watch the CV follower alone.
#
# `attach-fsm-act` passes one checkpoint to both learned states, which is the loop as
# designed: search until the carabiner is found, hand to classical CV to place the gripper,
# hand back to the same model to clip. The two states still keep separate policy sessions.
# It needs EVEREST_CAMERAS to name the `front` and `wrist` cameras the checkpoint was
# trained on, and it will not report SUCCESS: attachment verification is not built, so
# --no-attachment-verification is what makes that explicit.
#
# `attach-fsm-rl` is the same loop with `--skip-cv`: SEARCH_CV is routed around entirely,
# a detection hands straight to CLIP_RL, and no pixel map is read. It exists to measure a
# policy that approaches well enough without visual following -- with CV out, nothing
# checks that the gripper was placed on the carabiner before the clip policy starts.

# Exercise the attachment FSM with deterministic handlers. No camera, database, or motion.
[group('workflow')]
attach-fsm-fake flags="":
    uv run robot-attach-fsm --backend scaffold {{ flags }}

# Run one local attachment FSM attempt with a policy per learned state. POWERED.
[group('workflow')]
attach-fsm search_policy clip_policy:
    uv run robot-attach-fsm --backend hardware \
        --search-policy {{ search_policy }} --clip-policy {{ clip_policy }}

# Run the ACT loop: one checkpoint for both learned states, CV in between. POWERED.
[group('workflow')]
attach-fsm-act checkpoint device="auto":
    uv run robot-attach-fsm --backend hardware \
        --search-policy {{ checkpoint }} --clip-policy {{ checkpoint }} \
        --device {{ device }} \
        --no-attachment-verification --allow-unverified-lerobot-frame

# Run the ACT loop with SEARCH_CV skipped: one checkpoint owns the whole approach. POWERED.
[group('workflow')]
attach-fsm-rl checkpoint device="auto":
    uv run robot-attach-fsm --backend hardware \
        --search-policy {{ checkpoint }} --clip-policy {{ checkpoint }} \
        --device {{ device }} --skip-cv \
        --no-attachment-verification --allow-unverified-lerobot-frame

# Load a checkpoint and print the feature mapping it will run under. No robot, no motion.
[group('workflow')]
policy-check checkpoint:
    uv run robot-policy-check {{ checkpoint }}

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
    uv run robot-db sql \
        "SELECT task_id, task_name, state, attempts, enqueue_at \
         FROM absurd.t_robot ORDER BY enqueue_at DESC LIMIT {{ limit }};"

# Show one task's parameters and its durable result.
[group('workflow')]
task task_id:
    uv run robot-db sql \
        "SELECT task_name, state, attempts, jsonb_pretty(params) AS params, \
                jsonb_pretty(completed_payload) AS result \
         FROM absurd.t_robot WHERE task_id = '{{ task_id }}';"

# Show one task's committed checkpoints: the record of which stages actually completed.
[group('workflow')]
checkpoints task_id:
    uv run robot-db sql \
        "SELECT checkpoint_name, status, updated_at, jsonb_pretty(state) AS value \
         FROM absurd.c_robot WHERE task_id = '{{ task_id }}' ORDER BY updated_at;"

# Cancel a task. A running stage stops at its next heartbeat, holding the arm on the way out.
[group('workflow')]
cancel task_id:
    uv run robot-db sql \
        "SELECT absurd.cancel_task('${ROBOT_QUEUE:-robot}', '{{ task_id }}');"

# ── replay ─────────────────────────────────────────────────────────────────────────
# The numbered recipes are the ordered path to a powered replay. Run them in order; step 0
# is the smallest powered motion check, and docs/session-replay.md covers the rest.

# 0. Raise one joint a little from its measured pose. Pass dry="--dry-run" to plan only.
[group('replay')]
raise delta="0.10" speed="0.25" dry="":
    uv run robot-raise --delta-rad {{ delta }} --speed-scale {{ speed }} {{ dry }}

# 0b. The same command against the FakeArm: exercises the path with no CAN adapter present.
[group('replay')]
raise-fake delta="0.10" speed="0.25":
    uv run robot-raise --delta-rad {{ delta }} --speed-scale {{ speed }} --fake

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
