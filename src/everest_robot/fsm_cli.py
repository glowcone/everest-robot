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
    parser.add_argument(
        "--search-policy",
        help="SEARCH_RL policy: a checkpoint directory, a Hugging Face repo id, or a "
        ".json scripted policy (hardware only)",
    )
    parser.add_argument(
        "--clip-policy",
        help="CLIP_RL policy: the same forms. Pass the same reference as --search-policy "
        "to run one checkpoint in both learned states, in two separate sessions",
    )
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help="route around SEARCH_CV: a detection hands straight from INITIAL or SEARCH_RL "
        "to CLIP_RL, and no pixel map is needed. The learned clip policy then owns the "
        "whole approach, with nothing checking that the gripper was placed on the carabiner "
        "before it starts",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="torch device for inference: auto (default), cuda, mps, or cpu",
    )
    parser.add_argument(
        "--no-attachment-verification",
        action="store_true",
        help="run without an attachment verifier. The loop runs, but a successful "
        "attachment cannot be recognized, so the attempt can only end on a budget",
    )
    parser.add_argument(
        "--allow-unverified-lerobot-frame",
        action="store_true",
        help="run a checkpoint against the derived, not-yet-hardware-verified lerobot_frame "
        "offsets; see docs/lerobot-frame-reconciliation.md before using this",
    )
    parser.add_argument(
        "--policy-fps",
        type=float,
        default=None,
        help="override the configured policy rate for both learned states",
    )
    parser.add_argument("--task", default=None, help="task string passed to both policies")
    parser.add_argument("--max-actions", type=int, default=1_000)
    parser.add_argument("--max-duration", type=float, default=180.0)
    args = parser.parse_args()

    config = AttachmentFSMConfig(
        max_total_actions=args.max_actions,
        max_duration_s=args.max_duration,
        use_search_cv=not args.skip_cv,
    )
    params = {
        "backend": args.backend,
        "initial_detection": args.initial_detection,
        "skip_cv": args.skip_cv,
        "search_policy": args.search_policy,
        "clip_policy": args.clip_policy,
        "policy_fps": args.policy_fps,
        "attachment_task": args.task,
        "policy_device": args.device,
        "allow_unverified_attachment": args.no_attachment_verification,
        "allow_unverified_lerobot_frame": args.allow_unverified_lerobot_frame,
    }
    with attachment_fsm_handlers(params) as handlers:
        result = AttachmentFSM(handlers, config).run()
    print(json.dumps(result.to_json(), indent=2))
    if not result.succeeded:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
