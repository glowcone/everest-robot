"""Read-only joint feedback, derived into something a human can watch.

This is the view model behind ``robot-monitor`` (:mod:`everest_robot.monitor`, which owns
the terminal rendering). It polls an :class:`~everest_robot.robot.ports.ArmPort` and turns
each :class:`~everest_robot.robot.contracts.JointState` into per-joint rows carrying the
things an operator standing at the arm actually needs: the angle in both units, where it
sits inside the driver's soft limits, how far it has moved since a marked reference, and
whether the motor is still talking.

It is strictly read-only. It calls :meth:`~everest_robot.robot.ports.ArmPort.read_state`
and :meth:`~everest_robot.robot.ports.ArmPort.limits` and nothing else -- never
``enable()``, never ``send_targets()``. That is the property that makes it safe to run
while someone hand-moves the arm with motors disabled, which is step 1 of
docs/named-position-capture.md.

Staleness is judged the way :mod:`everest_robot.robot.motion` judges it -- by each motor's
feedback counter, against the hardware's own clock -- but per joint rather than for the
arm as a whole, because the point of a monitor is to say *which* motor went quiet.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from everest_robot.robot.clock import Clock, SystemClock
from everest_robot.robot.contracts import ArmLifecycle, JointLimit, RobotIdentity
from everest_robot.robot.ports import ArmPort

# A motor whose feedback counter has not advanced for this long is reporting a cached
# value. Deliberately looser than the control loop's tolerance: a monitor polling at a few
# hertz should not cry stale over one dropped frame.
DEFAULT_STALE_AFTER_S = 1.0

# Fast enough to see a joint move under your hand, slow enough to leave the bus alone.
DEFAULT_POLL_HZ = 10.0


@dataclass(frozen=True, slots=True)
class JointReading:
    """One joint at one instant, with everything the display derives already derived.

    ``position_rad`` is ``nan`` when the motor reported no feedback. Every derived value
    guards on that rather than propagating the ``nan`` into a formatted number.
    """

    name: str
    limit: JointLimit
    position_rad: float
    velocity_rad_s: float
    torque: float
    temperature_c: float
    fault_bits: int
    sequence: int
    quiet_for_s: float
    stale: bool
    reference_rad: float | None = None

    @property
    def has_feedback(self) -> bool:
        return math.isfinite(self.position_rad)

    @property
    def position_deg(self) -> float:
        return math.degrees(self.position_rad)

    @property
    def velocity_deg_s(self) -> float:
        return math.degrees(self.velocity_rad_s)

    @property
    def delta_rad(self) -> float | None:
        """Movement since the marked reference pose, or ``None`` without one."""

        if self.reference_rad is None or not self.has_feedback:
            return None
        if not math.isfinite(self.reference_rad):
            return None
        return self.position_rad - self.reference_rad

    @property
    def delta_deg(self) -> float | None:
        delta = self.delta_rad
        return None if delta is None else math.degrees(delta)

    @property
    def within_limits(self) -> bool:
        return self.has_feedback and self.limit.contains(self.position_rad)

    @property
    def span_fraction(self) -> float | None:
        """Where the joint sits in its soft-limit span, clamped to ``[0, 1]``.

        Clamped because this drives a bar: a joint outside its limits is drawn at the end
        of the bar and reported as out of range by :attr:`within_limits`, which is the
        field to branch on.
        """

        span = self.limit.upper_rad - self.limit.lower_rad
        if not self.has_feedback or span <= 0:
            return None
        return min(1.0, max(0.0, (self.position_rad - self.limit.lower_rad) / span))

    @property
    def has_fault(self) -> bool:
        return bool(self.fault_bits)


@dataclass(frozen=True, slots=True)
class MonitorSample:
    """One poll of the whole arm."""

    index: int
    monotonic_s: float
    lifecycle: ArmLifecycle
    fault_reason: str | None
    readings: tuple[JointReading, ...]

    @property
    def has_fault(self) -> bool:
        return self.lifecycle is ArmLifecycle.FAULT or any(r.has_fault for r in self.readings)

    @property
    def stale_joints(self) -> tuple[str, ...]:
        return tuple(r.name for r in self.readings if r.stale)

    @property
    def missing_feedback(self) -> tuple[str, ...]:
        return tuple(r.name for r in self.readings if not r.has_feedback)

    @property
    def out_of_limits(self) -> tuple[str, ...]:
        return tuple(r.name for r in self.readings if r.has_feedback and not r.within_limits)


class JointMonitor:
    """Polls one arm and derives a :class:`MonitorSample` per poll.

    The caller owns the arm's lifecycle -- claiming it, connecting it and disconnecting it.
    The monitor only ever reads, so it is equally correct against an arm that is merely
    connected (the hand-teaching case) and one another part of this process has enabled.
    """

    def __init__(
        self,
        port: ArmPort,
        *,
        clock: Clock | None = None,
        stale_after_s: float = DEFAULT_STALE_AFTER_S,
    ) -> None:
        self.port = port
        self.clock = clock or SystemClock()
        self.stale_after_s = float(stale_after_s)
        self._limits = tuple(port.limits())
        if len(self._limits) != len(port.joint_names):
            raise ValueError(
                f"the driver reports {len(self._limits)} soft limits but "
                f"{len(port.joint_names)} joints; one of them is describing a different arm"
            )
        self._samples = 0
        self._last_sequence: tuple[int, ...] | None = None
        self._quiet_since_s: list[float] = []
        self._last_positions: tuple[float, ...] = ()
        self._reference: tuple[float, ...] | None = None

    # ── what the display labels itself with ────────────────────────────────────────
    @property
    def identity(self) -> RobotIdentity:
        return self.port.identity

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self.port.joint_names

    @property
    def limits(self) -> tuple[JointLimit, ...]:
        return self._limits

    @property
    def samples(self) -> int:
        return self._samples

    @property
    def reference(self) -> tuple[float, ...] | None:
        """The pose deltas are measured from, or ``None`` while none is marked."""

        return self._reference

    # ── reference pose ─────────────────────────────────────────────────────────────
    def mark_reference(self) -> None:
        """Measure movement from here on. Ignored before the first sample."""

        if self._last_positions:
            self._reference = self._last_positions

    def clear_reference(self) -> None:
        self._reference = None

    # ── polling ────────────────────────────────────────────────────────────────────
    def sample(self) -> MonitorSample:
        """Read the arm once and derive the row for every joint."""

        state = self.port.read_state()
        if len(state.positions) != len(self._limits):
            raise ValueError(
                f"the driver returned {len(state.positions)} joint positions but "
                f"{len(self._limits)} soft limits"
            )

        # The hardware's own timestamp, so "quiet for 2s" means two seconds of bus silence
        # rather than two seconds of this process being descheduled.
        now = state.monotonic_s
        if self._last_sequence is None:
            self._quiet_since_s = [now] * len(self._limits)
        else:
            for index, (previous, current) in enumerate(
                zip(self._last_sequence, state.sequence, strict=True)
            ):
                if previous != current:
                    self._quiet_since_s[index] = now
        self._last_sequence = tuple(state.sequence)
        self._last_positions = tuple(state.positions)

        readings: list[JointReading] = []
        for index, limit in enumerate(self._limits):
            quiet_for = max(0.0, now - self._quiet_since_s[index])
            readings.append(
                JointReading(
                    name=limit.name,
                    limit=limit,
                    position_rad=float(state.positions[index]),
                    velocity_rad_s=float(state.velocities[index]),
                    torque=float(state.torques[index]),
                    temperature_c=float(state.temperatures[index]),
                    fault_bits=int(state.fault_bits[index]),
                    sequence=int(state.sequence[index]),
                    quiet_for_s=quiet_for,
                    # The first sample has nothing to compare against: nothing is stale yet.
                    stale=self._samples > 0 and quiet_for > self.stale_after_s,
                    reference_rad=None if self._reference is None else self._reference[index],
                )
            )

        sample = MonitorSample(
            index=self._samples,
            monotonic_s=now,
            lifecycle=state.lifecycle,
            fault_reason=state.fault_reason,
            readings=tuple(readings),
        )
        self._samples += 1
        return sample

    def stream(
        self, *, poll_hz: float = DEFAULT_POLL_HZ, limit: int | None = None
    ) -> Iterator[MonitorSample]:
        """Yield samples at ``poll_hz`` forever, or until ``limit`` of them.

        Paced with the injected clock, so a test drives a thousand polls without waiting.
        """

        if poll_hz <= 0:
            raise ValueError(f"poll_hz must be positive, got {poll_hz}")
        interval = 1.0 / poll_hz
        count = 0
        while limit is None or count < limit:
            started = self.clock.monotonic()
            yield self.sample()
            count += 1
            if limit is not None and count >= limit:
                return
            # Absorb a late poll rather than making it up: this is a display, and a burst
            # of catch-up reads only adds bus traffic.
            self.clock.sleep(max(0.0, interval - (self.clock.monotonic() - started)))


def format_table(sample: MonitorSample, *, width: int = 100) -> Sequence[str]:
    """Render a sample as plain text lines, for a pipe or a non-terminal.

    Shares nothing with the curses renderer on purpose: this one has to stay readable when
    it is redirected into a file next to a captured pose.
    """

    header = (
        f"{'joint':<16}{'rad':>10}{'deg':>10}{'Δdeg':>9}"
        f"{'deg/s':>9}{'torque':>9}{'°C':>6}  {'state':<12}"
    )
    lines = [header, "-" * min(width, len(header))]
    for reading in sample.readings:
        if reading.has_feedback:
            position = f"{reading.position_rad:>10.4f}{reading.position_deg:>10.2f}"
        else:
            position = f"{'--':>10}{'--':>10}"
        delta = reading.delta_deg
        lines.append(
            f"{reading.name:<16}{position}"
            f"{'--' if delta is None else f'{delta:+.2f}':>9}"
            f"{reading.velocity_deg_s:>9.2f}{reading.torque:>9.3f}"
            f"{reading.temperature_c:>6.1f}  {_flags(reading):<12}"
        )
    return lines


def _flags(reading: JointReading) -> str:
    """The short reason a joint is not simply fine, or ``ok``."""

    if not reading.has_feedback:
        return "NO FEEDBACK"
    flags = []
    if reading.has_fault:
        flags.append(f"FAULT:{reading.fault_bits:#x}")
    if reading.stale:
        flags.append(f"STALE {reading.quiet_for_s:.1f}s")
    if not reading.within_limits:
        flags.append("OUT OF RANGE")
    return " ".join(flags) if flags else "ok"
