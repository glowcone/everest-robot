"""The hardware boundary the rest of the runtime is written against.

Everything above this line (motion, policy, replay, the LeRobot bridge) talks to
:class:`ArmPort` and never to a driver directly. That keeps the driver choice recorded in
docs/adr/0001-production-motor-protocol.md a single-file decision, and it lets every layer
above be tested against :class:`~everest_robot.robot.fake_arm.FakeArm`.

The port is deliberately thin: it exposes the commands and feedback the workflow needs and
nothing else. Soft limits, watchdogs, velocity limiting, fault handling and coordinate
conversion stay inside the driver, which is where the ADR assigns them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from everest_robot.robot.contracts import (
    ArmLifecycle,
    JointCommand,
    JointLimit,
    JointState,
    RobotIdentity,
)


@runtime_checkable
class ArmPort(Protocol):
    """A connected, enable-able arm in calibrated joint coordinates (radians)."""

    @property
    def identity(self) -> RobotIdentity:
        """Who this arm is, as asserted by deployment configuration and the driver."""

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Canonical joint order. Every position/target sequence follows it."""

    @property
    def lifecycle(self) -> ArmLifecycle:
        """Current driver state. Commands are only accepted in ``ENABLED``."""

    def limits(self) -> tuple[JointLimit, ...]:
        """The driver's absolute soft limits, one per joint, in joint coordinates."""

    def connect(self, timeout: float = 2.0) -> None:
        """Open the bus and prove every motor is reporting fresh feedback."""

    def disconnect(self) -> None:
        """Disable if needed and release the bus. Safe to call from any state."""

    def enable(self) -> None:
        """Arm the motors and start the driver's control loop."""

    def disable(self) -> None:
        """Release torque and stop the control loop."""

    def estop(self) -> None:
        """Immediately release torque from any state."""

    def clear_faults(self) -> None:
        """Attempt to leave ``FAULT`` after the cause has been addressed."""

    def hold_current_position(self) -> bool:
        """Freeze at the latest measured pose without releasing torque."""

    def read_state(self) -> JointState:
        """One synchronized feedback sample. Never blocks for longer than a tick."""

    def send_targets(self, targets: Sequence[float]) -> bool:
        """Command joint positions. Returns ``False`` if the driver refused them."""


def clip_to_limits(
    targets: Sequence[float],
    limits: Sequence[JointLimit],
    *,
    margin: float = 0.0,
) -> JointCommand:
    """Clamp a joint-space command into the driver's soft limits.

    Clipping is reported, never silent: a policy or replay frame that keeps hitting a limit
    is a schema or calibration problem, and the durable result has to say so. The driver
    enforces its own limits regardless; this exists so the layer above can refuse or record
    rather than discover the clamp after the fact.
    """

    names = tuple(limit.name for limit in limits)
    values = tuple(float(target) for target in targets)
    if len(values) != len(limits):
        raise ValueError(f"expected {len(limits)} targets, got {len(values)}")

    clipped: list[str] = []
    result: list[float] = []
    for value, limit in zip(values, limits, strict=True):
        if not math.isfinite(value):
            raise ValueError(f"{limit.name}: target must be finite, got {value!r}")
        bounded = limit.clamp(value, margin=margin)
        if bounded != value:
            clipped.append(limit.name)
        result.append(bounded)
    return JointCommand(names=names, targets=tuple(result), clipped_joints=tuple(clipped))


def violations(
    targets: Sequence[float],
    limits: Sequence[JointLimit],
    *,
    margin: float = 0.0,
) -> tuple[str, ...]:
    """Joints whose target lies outside the soft limits, without modifying anything.

    Used before enabling motion: a named preset that needs clipping is not an approved
    preset, and the move is refused rather than quietly clamped.
    """

    return tuple(
        limit.name
        for value, limit in zip(targets, limits, strict=True)
        if not math.isfinite(value) or not limit.contains(float(value), margin=margin)
    )
