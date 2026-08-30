"""Bounded continuous servoing toward a vision-derived joint target.

The loop is deliberately dull. Each tick it is handed a full joint target or ``None``,
and it moves the arm at most ``max_velocity_rad_s / rate_hz`` radians per joint toward
that target. Nothing here decides *where* the target is; that is the caller's camera,
detector and calibration. Keeping the decision out means this file can be tested against
:class:`~everest_robot.robot.fake_arm.FakeArm` without a lens.

Three rules make it safe to point at a live arm:

* **No target, no motion.** A missing or rejected detection holds position. It never
  coasts toward the last known target, because the object may have been picked up, moved,
  or occluded by the operator's hand, and continuing would be acting on a stale belief.
* **Re-lock after every miss.** After a hold, motion resumes only once ``lock_frames``
  consecutive ticks have produced a target. A detector that flickers between a good blob
  and nothing must not translate into the arm twitching toward it.
* **Speed locked.** The per-tick step is a hard clamp on the *commanded* value, not a
  request to a planner, so the arm's speed is bounded by construction rather than by a
  profile that a large target jump could outrun.

The clamp mirrors :class:`~everest_robot.robot.teleoperation.TeleoperationController`
rather than going through :class:`~everest_robot.robot.motion.JointMotionController`: that
controller runs a bounded move to a *fixed* target and returns when it settles, which is
the wrong shape for a target that changes every frame.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from everest_robot.robot.clock import Clock, SystemClock
from everest_robot.robot.contracts import ArmLifecycle
from everest_robot.robot.ports import ArmPort, violations

# Reasons a tick did not move the arm. They are strings rather than an enum because they
# are shown to an operator on a live overlay, not branched on.
NO_TARGET = "no detection"
LOCKING = "locking on"
FAULT = "arm fault"
NO_FEEDBACK = "no joint feedback"
OUT_OF_LIMITS = "target outside soft limits"
REFUSED = "driver refused the command"
TRACKING = "tracking"
DRY_RUN = "dry run"


class TrackerStopped(RuntimeError):
    """The tracker will not command anything further; the arm is held."""


@dataclass(frozen=True, slots=True)
class TrackTick:
    """What one control tick did, and why."""

    index: int
    moved: bool
    reason: str
    target: tuple[float, ...] | None = None
    command: tuple[float, ...] | None = None
    measured: tuple[float, ...] = ()
    remaining_rad: float = 0.0
    locked_frames: int = 0

    @property
    def settled(self) -> bool:
        return self.moved and self.remaining_rad <= 1e-6


@dataclass
class VisualTracker:
    """Follows a per-frame joint target at a bounded speed, or holds."""

    port: ArmPort
    rate_hz: float = 15.0
    max_velocity_rad_s: float = 0.15
    lock_frames: int = 3
    dry_run: bool = False
    clock: Clock = field(default_factory=SystemClock)

    def __post_init__(self) -> None:
        if not math.isfinite(self.rate_hz) or self.rate_hz <= 0.0:
            raise ValueError("rate_hz must be finite and positive")
        if not math.isfinite(self.max_velocity_rad_s) or self.max_velocity_rad_s <= 0.0:
            raise ValueError("max_velocity_rad_s must be finite and positive")
        if self.lock_frames < 1:
            raise ValueError("lock_frames must be at least 1")
        self._command: tuple[float, ...] = ()
        self._locked = 0
        self._index = 0
        self._started = False
        self.stopped_reason: str | None = None

    @property
    def period_s(self) -> float:
        return 1.0 / self.rate_hz

    @property
    def max_step_rad(self) -> float:
        """The hard per-tick displacement bound. This is the speed lock."""

        return self.max_velocity_rad_s * self.period_s

    @property
    def locked(self) -> bool:
        return self._locked >= self.lock_frames

    def start(self) -> tuple[float, ...]:
        """Seed the command from measured feedback and energize, unless this is a dry run."""

        state = self.port.read_state()
        if not state.all_finite:
            raise TrackerStopped("the arm has missing position feedback; nothing to servo from")
        if state.has_fault:
            raise TrackerStopped(f"the arm is in fault: {state.fault_reason or state.fault_bits}")
        self._command = tuple(state.positions)
        if not self.dry_run:
            self.port.enable()
        self._started = True
        return self._command

    def tick(self, target: Sequence[float] | None, reason: str = NO_TARGET) -> TrackTick:
        """Advance one control step toward ``target``, or hold when there is none.

        Never raises for an ordinary miss: a detector that loses the object is the normal
        case this loop exists to survive, and it is reported in the returned tick.
        """

        if not self._started:
            raise TrackerStopped("call start() before tick()")
        if self.stopped_reason is not None:
            raise TrackerStopped(self.stopped_reason)
        self._index += 1

        state = self.port.read_state()
        if state.has_fault:
            return self._stop(FAULT, state.positions)
        if not state.all_finite:
            return self._stop(NO_FEEDBACK, state.positions)

        if target is None:
            return self._hold(reason or NO_TARGET, state.positions)

        values = tuple(float(value) for value in target)
        if len(values) != len(self.port.joint_names):
            raise ValueError(
                f"expected {len(self.port.joint_names)} joints, got {len(values)}"
            )
        outside = violations(values, self.port.limits())
        if outside:
            return self._hold(f"{OUT_OF_LIMITS}: {', '.join(outside)}", state.positions)

        self._locked += 1
        if not self.locked:
            return TrackTick(
                index=self._index,
                moved=False,
                reason=LOCKING,
                target=values,
                measured=tuple(state.positions),
                locked_frames=self._locked,
            )

        step = self.max_step_rad
        command = tuple(
            current + max(-step, min(step, desired - current))
            for current, desired in zip(self._command, values, strict=True)
        )
        remaining = max(abs(a - b) for a, b in zip(command, values, strict=True))
        if self.dry_run:
            self._command = command
            return TrackTick(
                index=self._index,
                moved=False,
                reason=DRY_RUN,
                target=values,
                command=command,
                measured=tuple(state.positions),
                remaining_rad=remaining,
                locked_frames=self._locked,
            )
        if not self.port.send_targets(command):
            return self._stop(f"{REFUSED} while {self.port.lifecycle.value}", state.positions)
        self._command = command
        return TrackTick(
            index=self._index,
            moved=True,
            reason=TRACKING,
            target=values,
            command=command,
            measured=tuple(state.positions),
            remaining_rad=remaining,
            locked_frames=self._locked,
        )

    def stop(self, *, hold: bool = True) -> None:
        """Stop tracking. Safe to call from any state, including twice.

        ``hold=True`` freezes the arm where it is, which is what a caller that keeps the
        session open wants. ``hold=False`` is for a teardown that is about to cut torque
        anyway: commanding a hold first only snaps the arm toward its last target a moment
        before it goes limp, which the operator feels as a lurch on Ctrl-C.
        """

        self._locked = 0
        if hold and self.port.lifecycle is ArmLifecycle.ENABLED:
            self.port.hold_current_position()

    # ── internals ──────────────────────────────────────────────────────────────────
    def _hold(self, reason: str, positions: Sequence[float]) -> TrackTick:
        """Freeze, and require a fresh lock before moving again."""

        self._locked = 0
        self._command = tuple(float(value) for value in positions)
        if not self.dry_run and self.port.lifecycle is ArmLifecycle.ENABLED:
            self.port.hold_current_position()
        return TrackTick(
            index=self._index,
            moved=False,
            reason=reason,
            measured=tuple(positions),
        )

    def _stop(self, reason: str, positions: Sequence[float]) -> TrackTick:
        tick = self._hold(reason, positions)
        self.stopped_reason = reason
        return tick
