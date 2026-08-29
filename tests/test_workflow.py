from collections.abc import Callable
from typing import Any

from everest_robot.workflow import run_attach_carabiner


class RecordingContext:
    def __init__(self) -> None:
        self.steps: list[str] = []

    def step(self, name: str, operation: Callable[[], Any]) -> Any:
        self.steps.append(name)
        return operation()


def test_attachment_verification_retries_only_attachment_stage() -> None:
    context = RecordingContext()

    result = run_attach_carabiner({"verification_failures": 1}, context)

    assert result["status"] == "complete"
    assert result["cycles"] == 2
    assert context.steps == [
        "01-localize-and-pick-up-carabiner-cycle-00",
        "02-go-to-known-position-cycle-00",
        "03-rl-vla-attach-clip-cycle-00",
        "04-verify-attachment-cycle-00",
        "03-rl-vla-attach-clip-cycle-01",
        "04-verify-attachment-cycle-01",
    ]
