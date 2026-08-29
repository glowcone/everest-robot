"""Spawn a robot workflow from the command line."""

import argparse
import json

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


if __name__ == "__main__":
    start()

