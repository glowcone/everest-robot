"""Inspect a policy without a robot: what it expects, and whether it loads here.

Everything ``robot-attach-fsm`` resolves before it claims the arm, done on its own so an
operator can answer "will this checkpoint run on this machine, against this arm" without a
lease, a camera or a motor. It touches no hardware and commands nothing.
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve a policy and print the feature mapping it would run under"
    )
    parser.add_argument(
        "policy",
        help="a checkpoint directory, a Hugging Face repo id, or a .json scripted policy",
    )
    parser.add_argument("--revision", default=None, help="pin a checkpoint revision")
    parser.add_argument(
        "--dataset",
        default=None,
        help="training dataset repo id, for a checkpoint copied without its train_config.json",
    )
    parser.add_argument(
        "--device", default=None, help="torch device: auto (default), cuda, mps, or cpu"
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="also build the model and its processors, which is what proves the device works",
    )
    args = parser.parse_args()

    from everest_robot.robot.checkpoints import CheckpointError, resolve_checkpoint

    try:
        checkpoint = resolve_checkpoint(
            args.policy, revision=args.revision, dataset_repo_id=args.dataset
        )
    except CheckpointError as error:
        raise SystemExit(f"error: {error}") from None

    features = checkpoint.features
    print(f"checkpoint      {checkpoint.identifier}")
    print(f"policy type     {checkpoint.policy_type}")
    print(f"local path      {checkpoint.root}")
    print(f"trained on      {features.dataset_repo_id}@{features.dataset_revision[:12]}")
    print(f"robot type      {features.robot_type}")
    print(f"control rate    {features.fps} fps")
    print("\naction / state joints, in the order the tensors pack them:")
    for index, name in enumerate(features.action_names):
        print(f"  {index}  {name}")
    print("\ncameras the policy needs, keyed as the robot names them:")
    for name, shape in features.cameras.items():
        height, width, channels = shape
        print(f"  {name}: {width}x{height}x{channels}")
    if not features.cameras:
        print("  (none -- proprioception only)")

    _compare_to_this_arm(features)

    if args.load:
        from everest_robot.robot.policy import LeRobotPolicyHandle, PolicyLoadError

        try:
            handle = LeRobotPolicyHandle(checkpoint, device=args.device)
        except (PolicyLoadError, ImportError) as error:
            raise SystemExit(f"error: {error}") from None
        print(f"\nloaded on       {handle.device}")


def _compare_to_this_arm(features) -> None:
    """The comparison the policy session makes before it enables anything.

    Reported rather than enforced: this command is diagnostic, and a mismatch here is
    exactly what the operator ran it to find out.
    """

    try:
        from everest_robot.robot.cameras import load_camera_specs
        from everest_robot.robot.deployment import joint_frame, load_parameters
    except ImportError as error:  # pragma: no cover - core deps
        print(f"\ncould not read the deployment configuration: {error}", file=sys.stderr)
        return

    try:
        parameters = load_parameters()
    except (OSError, ValueError) as error:
        print(f"\nno robot parameters to compare against: {error}")
        return

    print("\nagainst this deployment:")
    expected = tuple(f"{name}.pos" for name in parameters.identity.joint_names)
    if expected == features.action_names:
        print(f"  joints          match ({parameters.identity.robot_id})")
    else:
        print(f"  joints          MISMATCH: arm has {list(expected)}")

    if parameters.policy.fps == features.fps:
        print(f"  control rate    match ({features.fps} fps)")
    else:
        print(
            f"  control rate    MISMATCH: parameters say {parameters.policy.fps} fps, "
            f"the policy was trained at {features.fps}"
        )

    configured = {spec.name: spec.frame_shape for spec in load_camera_specs()}
    for name, shape in features.cameras.items():
        if name not in configured:
            print(f"  camera {name!r}    NOT CONFIGURED in EVEREST_CAMERAS")
        elif configured[name] != shape:
            print(f"  camera {name!r}    MISMATCH: configured {configured[name]}, needs {shape}")
        else:
            print(f"  camera {name!r}    match")

    if not joint_frame(parameters).is_identity:
        print(
            "  lerobot_frame   non-zero offsets. A checkpoint trained through MakerFollower\n"
            "                  needs them, but they are derived, not measured on hardware.\n"
            "                  See docs/lerobot-frame-reconciliation.md; running anyway\n"
            "                  requires --allow-unverified-lerobot-frame."
        )


if __name__ == "__main__":
    main()
