"""Spawn robot workflows, and preflight a replay, from the command line."""

import argparse
import json
import sys

from absurd_sdk import Absurd

from everest_robot.workflow import QUEUE_NAME


def start() -> None:
    parser = argparse.ArgumentParser(description="Start a carabiner attachment workflow")
    parser.add_argument("--workflow-id", default="demo")
    parser.add_argument("--verification-failures", type=int, default=0)
    args = parser.parse_args()

    client = Absurd(queue_name=QUEUE_NAME)
    spawned = client.spawn(
        "attach-carabiner",
        {"verification_failures": args.verification_failures},
        queue=QUEUE_NAME,
        max_attempts=10,
        retry_strategy={"kind": "exponential", "base_seconds": 1, "max_seconds": 30},
        idempotency_key=f"attach-carabiner:{args.workflow_id}",
    )
    print(json.dumps(spawned, indent=2, default=str))
    client.close()


def _replay_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-id", required=True, help="Hugging Face dataset repo id")
    parser.add_argument("--revision", required=True, help="full 40-character commit SHA")
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--robot-id", required=True)
    parser.add_argument("--calibration-id", required=True)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument(
        "--limit-policy",
        default="reject",
        choices=["reject", "clamp_within_tolerance", "clamp"],
    )
    parser.add_argument("--max-limit-deviation-deg", type=float, default=0.0)


def _replay_parameters(args: argparse.Namespace, *, dry_run: bool) -> dict:
    return {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "episode": args.episode,
        "robot_id": args.robot_id,
        "calibration_id": args.calibration_id,
        "speed": args.speed,
        "start_frame": args.start_frame,
        "end_frame": args.end_frame,
        "limit_policy": args.limit_policy,
        "max_limit_deviation_deg": args.max_limit_deviation_deg,
        "dry_run": dry_run,
    }


def start_replay() -> None:
    """Spawn a replay task.

    Spawned with a single attempt: an interrupted physical replay must not be restarted
    automatically. Re-running one deliberately needs a new workflow id.
    """

    parser = argparse.ArgumentParser(description="Replay a stored dataset episode")
    parser.add_argument("--workflow-id", default="replay")
    parser.add_argument("--dry-run", action="store_true")
    _replay_arguments(parser)
    args = parser.parse_args()

    client = Absurd(queue_name=QUEUE_NAME)
    spawned = client.spawn(
        "replay-session",
        _replay_parameters(args, dry_run=args.dry_run),
        queue=QUEUE_NAME,
        max_attempts=1,
        idempotency_key=f"replay-session:{args.workflow_id}",
    )
    print(json.dumps(spawned, indent=2, default=str))
    client.close()


def replay_preflight() -> None:
    """Validate a replay request against this deployment and print the report.

    Runs locally, claims nothing, energizes nothing. This is the report to review before
    authorizing a powered replay.
    """

    parser = argparse.ArgumentParser(
        description="Preflight a replay without claiming or energizing the arm"
    )
    _replay_arguments(parser)
    args = parser.parse_args()

    from everest_robot.domain import ReplayRequest
    from everest_robot.robot.deployment import build_replay_runner
    from everest_robot.robot.errors import ReplayError

    request = ReplayRequest.from_json(_replay_parameters(args, dry_run=True))
    try:
        plan, approval = build_replay_runner().preflight(request)
    except ReplayError as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1) from None

    report = plan.report.to_json()
    report["approved_by"] = approval.approved_by if approval else None
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    start()

