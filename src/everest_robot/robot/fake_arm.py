"""Deterministic arm hardware for tests and hardware-free workflow runs.

Reproduces the parts of ``maker_arm.Arm``'s behaviour the layers above depend on: the
state machine and its refusals, soft-limit clamping, velocity-limited approach to a
target, per-joint feedback counters, and faults. It is driven by an injected
:class:`~everest_robot.robot.clock.Clock`, so a ten-second timeout costs microseconds and
tick counts are exactly reproducible.

It is a behavioural stand-in, not a physics model. It will not tell you whether a pose is
reachable, whether a path collides, or how the arm settles under load.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from everest_robot.robot.clock import Clock, SystemClock
from everest_robot.robot.contracts import (
    ArmLifecycle,
    JointLimit,
    JointState,
    RobotIdentity,
)


class FakeArmError(RuntimeError):
    """Raised for the same state violations the real driver raises on."""


@dataclass
class FakeArm:
    """An arm that moves toward its targets at a bounded speed, and can be made to fail.

    ``tracking_offset_rad`` adds a constant lag between the command and the measurement so
    tests can exercise tracking-error handling; ``fault_after_commands``,
    ``stale_after_commands`` and ``refuse_targets`` inject the failures the motion layer
    has to survive.
    """

    identity: RobotIdentity
    joint_limits: tuple[JointLimit, ...]
    clock: Clock = field(default_factory=SystemClock)
    positions: list[float] = field(default_factory=list)
    max_velocity_rad_s: float = 5.0
    tracking_offset_rad: float = 0.0
    fault_after_commands: int | None = None
    stale_after_commands: int | None = None
    refuse_targets: bool = False

    def __post_init__(self) -> None:
        if len(self.joint_limits) != len(self.identity.joint_names):
            raise ValueError("joint_limits must cover every joint in the identity")
        if not self.positions:
            # Park at the midpoint of each soft limit: a defined, in-range starting pose.
            self.positions = [
                (limit.lower_rad + limit.upper_rad) / 2 for limit in self.joint_limits
            ]
        if len(self.positions) != len(self.joint_limits):
            raise ValueError("positions must cover every joint")
        self._targets = list(self.positions)
        self._lifecycle = ArmLifecycle.DISCONNECTED
        self._sequence = [0] * len(self.positions)
        self._last_integration_s = self.clock.monotonic()
        self._commands = 0
        self._fault_reason: str | None = None
        self.sent_commands: list[tuple[float, ...]] = []

    # ── identity and limits ────────────────────────────────────────────────────────
    @property
    def joint_names(self) -> tuple[str, ...]:
        return self.identity.joint_names

    @property
    def lifecycle(self) -> ArmLifecycle:
        return self._lifecycle

    def limits(self) -> tuple[JointLimit, ...]:
        return self.joint_limits

    # ── lifecycle ──────────────────────────────────────────────────────────────────
    def connect(self, timeout: float = 2.0) -> None:
        if self._lifecycle is not ArmLifecycle.DISCONNECTED:
            raise FakeArmError(f"connect() requires DISCONNECTED, currently {self._lifecycle}")
        self._lifecycle = ArmLifecycle.CONNECTED
        self._bump_feedback()

    def disconnect(self) -> None:
        self._lifecycle = ArmLifecycle.DISCONNECTED

    def enable(self) -> None:
        if self._lifecycle is not ArmLifecycle.CONNECTED:
            raise FakeArmError(f"enable() requires CONNECTED, currently {self._lifecycle}")
        # The real driver refuses to arm from a pose outside the soft limits.
        outside = [
            limit.name
            for position, limit in zip(self.positions, self.joint_limits, strict=True)
            if not limit.contains(position)
        ]
        if outside:
            raise FakeArmError(f"enable() refused: {', '.join(outside)} outside soft limits")
        self._targets = list(self.positions)
        self._lifecycle = ArmLifecycle.ENABLED

    def disable(self) -> None:
        if self._lifecycle in (ArmLifecycle.ENABLED, ArmLifecycle.FAULT):
            self._lifecycle = ArmLifecycle.CONNECTED

    def estop(self) -> None:
        self._integrate()
        self._targets = list(self.positions)
        if self._lifecycle is ArmLifecycle.ENABLED:
            self._lifecycle = ArmLifecycle.CONNECTED

    def clear_faults(self) -> None:
        if self._lifecycle is ArmLifecycle.FAULT:
            self._lifecycle = ArmLifecycle.CONNECTED
            self._fault_reason = None

    def hold_current_position(self) -> bool:
        if self._lifecycle is not ArmLifecycle.ENABLED:
            return False
        self._integrate()
        self._targets = list(self.positions)
        return True

    # ── feedback and commands ──────────────────────────────────────────────────────
    def read_state(self) -> JointState:
        self._integrate()
        measured = tuple(position + self.tracking_offset_rad for position in self.positions)
        return JointState(
            names=self.joint_names,
            positions=measured,
            velocities=tuple(self._velocities),
            torques=tuple(0.0 for _ in self.positions),
            temperatures=tuple(30.0 for _ in self.positions),
            fault_bits=tuple(
                1 if self._lifecycle is ArmLifecycle.FAULT else 0 for _ in self.positions
            ),
            sequence=tuple(self._sequence),
            monotonic_s=self.clock.monotonic(),
            lifecycle=self._lifecycle,
            fault_reason=self._fault_reason,
        )

    def send_targets(self, targets: Sequence[float]) -> bool:
        if self.refuse_targets or self._lifecycle is not ArmLifecycle.ENABLED:
            return False
        if len(targets) != len(self.positions) or not all(math.isfinite(t) for t in targets):
            return False

        self._integrate()
        self._commands += 1
        self.sent_commands.append(tuple(float(t) for t in targets))
        # The driver clamps into its own soft limits; nothing above it can command past them.
        self._targets = [
            limit.clamp(float(target))
            for target, limit in zip(targets, self.joint_limits, strict=True)
        ]
        if self.fault_after_commands is not None and self._commands >= self.fault_after_commands:
            self.inject_fault("injected motor fault")
        return True

    # ── failure injection ──────────────────────────────────────────────────────────
    def inject_fault(self, reason: str) -> None:
        """Enter FAULT and hold, mirroring the driver's hold_on_fault default."""

        self._integrate()
        self._targets = list(self.positions)
        self._fault_reason = reason
        self._lifecycle = ArmLifecycle.FAULT

    # ── internals ──────────────────────────────────────────────────────────────────
    @property
    def _velocities(self) -> list[float]:
        return [
            0.0 if abs(target - position) < 1e-12 else math.copysign(
                self.max_velocity_rad_s, target - position
            )
            for position, target in zip(self.positions, self._targets, strict=True)
        ]

    def _integrate(self) -> None:
        now = self.clock.monotonic()
        elapsed = max(0.0, now - self._last_integration_s)
        self._last_integration_s = now
        if elapsed == 0.0 or self._lifecycle is ArmLifecycle.DISCONNECTED:
            return

        if self._lifecycle is ArmLifecycle.ENABLED:
            step = self.max_velocity_rad_s * elapsed
            for index, (position, target) in enumerate(
                zip(self.positions, self._targets, strict=True)
            ):
                delta = target - position
                if delta == 0.0:
                    continue
                self.positions[index] = position + math.copysign(min(abs(delta), step), delta)
        # Real motors report on every control tick whether or not they moved, and the
        # driver polls the bus while merely connected (see MakerArmPort.read_state), so the
        # counters advance in every connected state -- not only under torque. Freshness is
        # what the layer above checks, so getting this wrong would make a read-only
        # observer of a disabled arm see staleness that the real driver never reports.
        self._bump_feedback()

    def _bump_feedback(self) -> None:
        """Advance the per-joint feedback counters unless staleness is being injected."""

        if self.stale_after_commands is not None and self._commands >= self.stale_after_commands:
            return
        self._sequence = [value + 1 for value in self._sequence]
