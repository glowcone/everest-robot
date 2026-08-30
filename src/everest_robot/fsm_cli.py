"""Command-line entrypoint for one local carabiner attachment FSM attempt."""

from __future__ import annotations

import argparse
import json

from everest_robot.adapters import attachment_fsm_handlers
from everest_robot.attachment_fsm import AttachmentFSM, AttachmentFSMConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one locally orchestrated carabiner attachment attempt"
    )
    parser.add_argument("--backend", choices=("scaffold", "hardware"), default=None)
    parser.add_argument(
        "--initial-detection",
        action="store_true",
        help="scaffold only: INITIAL detects the carabiner before an RL search action",
    )
    parser.add_argument("--max-actions", type=int, default=1_000)
    parser.add_argument("--max-duration", type=float, default=180.0)
    args = parser.parse_args()

    config = AttachmentFSMConfig(
        max_total_actions=args.max_actions,
        max_duration_s=args.max_duration,
    )
    params = {
        "backend": args.backend,
        "initial_detection": args.initial_detection,
    }
    with attachment_fsm_handlers(params) as handlers:
        result = AttachmentFSM(handlers, config).run()
    print(json.dumps(result.to_json(), indent=2))
    if not result.succeeded:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
