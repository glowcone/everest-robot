"""Lease-local Star 102 leader following for the calibration monitor.

The caller owns the follower's :class:`RobotSession` and therefore its lease.  This
module opens only the leader bus and drives that already-claimed follower; it must never
construct a second arm or claim.  Leader positions are mapped with maker-arm-sdk's
private-protocol Star mapping, not LeRobot's MIT/degree frame.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from everest_robot.robot.contracts import ArmLifecycle
from everest_robot.robot.ports import ArmPort


class LeaderPort(Protocol):
    """The read-only leader side of a teleoperation pair."""

    @property
    def servo_ids(self) -> tuple[int, ...]: ...

    def connect(self) -> None: ...

    def read_positions(self) -> Mapping[int, float]: ...

    def disconnect(self) -> None: ...


class Star102LeaderPort:
    """Star Arm 102 UART bus, exposing every servo that answered this cycle.

    The library's ``reliable`` flag is not used to gate readings: its glitch filter
    treats a raw angle of ~0 deg as a suspected power-cycle glitch, so a joint genuinely
    resting at zero -- which is how a leader arm sits -- is flagged unreliable forever
    (measured on this arm: servos parked at 0.0 deg read 4% "reliable", the same servos
    bent read 100%). Loss detection uses per-servo ``read_angle`` instead, which raises
    when a servo truly does not answer; a missing servo is simply absent from the
    returned mapping and the controller's leader-loss timeout does the rest.
    """

    def __init__(self, port: str, servo_ids: Sequence[int] = tuple(range(7))) -> None:
        self.port = port
        self._servo_ids = tuple(int(value) for value in servo_ids)
        self._bus: Any = None
        self._bus_error: type[Exception] = Exception

    @property
    def servo_ids(self) -> tuple[int, ...]:
        return self._servo_ids

    def connect(self) -> None:
        try:
            from motorbridge_smart_servo import FashionStarServo, ServoBusError
        except ImportError as error:  # pragma: no cover - hardware-extra failure
            raise RuntimeError(
                "motorbridge-smart-servo is unavailable; run `just setup-hardware`"
            ) from error
        self._bus_error = ServoBusError
        self._bus = FashionStarServo(self.port, baudrate=1_000_000)
        # The native layer raises after N consecutive misses on its own; disable that so
        # loss policy lives in one place, the controller's leader-loss timeout.
        self._bus.set_loss_threshold(0)

    def read_positions(self) -> Mapping[int, float]:
        if self._bus is None:
            raise RuntimeError("the Star leader is not connected")
        readings: dict[int, float] = {}
        for servo_id in self._servo_ids:
            try:
                sample = self._bus.read_angle(servo_id, multi_turn=True)
            except self._bus_error:
                continue  # no answer this cycle; the loss timeout accounts for it
            readings[servo_id] = float(sample.raw_deg)
        return readings

    def disconnect(self) -> None:
        bus, self._bus = self._bus, None
        if bus is not None:
            bus.close()


def load_star_mapper(path: str | Path | None = None) -> Any:
    """Load maker-arm-sdk's calibrated absolute Star-to-Maker mapping."""

    from maker_arm.mapping import JointMapper

    if path is None:
        from maker_arm.profiles import DEFAULT_STAR_MAPPING

        path = DEFAULT_STAR_MAPPING
    return JointMapper.from_json(str(path))


class TeleoperationController:
    """Follow a Star leader on a background control loop while the TUI reads feedback."""

    def __init__(
        self,
        follower: ArmPort,
        leader: LeaderPort,
        mapper: Any,
        *,
        rate_hz: float = 25.0,
        max_velocity_rad_s: float = 0.25,
        leader_loss_timeout_s: float = 0.5,
        clamp_joints: Sequence[str] = ("gripper",),
    ) -> None:
        if rate_hz <= 0 or not math.isfinite(rate_hz):
            raise ValueError("rate_hz must be finite and positive")
        if max_velocity_rad_s <= 0 or not math.isfinite(max_velocity_rad_s):
            raise ValueError("max_velocity_rad_s must be finite and positive")
        if leader_loss_timeout_s <= 0 or not math.isfinite(leader_loss_timeout_s):
            raise ValueError("leader_loss_timeout_s must be finite and positive")
        self.follower = follower
        self.leader = leader
        self.mapper = mapper
        self.rate_hz = float(rate_hz)
        self.max_velocity_rad_s = float(max_velocity_rad_s)
        self.leader_loss_timeout_s = float(leader_loss_timeout_s)
        self.clamp_joints = frozenset(clamp_joints)
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_raw: dict[int, float] = {}
        self._last_seen: dict[int, float] = {}
        self._command: tuple[float, ...] = ()
        self.error: str | None = None

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def connect_and_measure(self, timeout_s: float = 2.0) -> float:
        """Connect the leader and return maximum initial follower/leader difference."""

        self.leader.connect()
        try:
            deadline = time.monotonic() + timeout_s
            while len(self._last_raw) != len(self.leader.servo_ids) and time.monotonic() < deadline:
                self._last_raw.update(self.leader.read_positions())
            missing = sorted(set(self.leader.servo_ids) - set(self._last_raw))
            if missing:
                raise RuntimeError(f"Star leader has no reliable reading for servo(s) {missing}")
            now = time.monotonic()
            self._last_seen = dict.fromkeys(self.leader.servo_ids, now)
            targets = self._mapped_targets(self._last_raw)
            current = self.follower.read_state()
            if not current.all_finite:
                raise RuntimeError("follower has missing position feedback")
            self._command = tuple(current.positions)
            return max(
                abs(target - position)
                for target, position in zip(targets, current.positions, strict=True)
            )
        except BaseException:
            self.leader.disconnect()
            raise

    def start(self) -> None:
        if not self._command:
            raise RuntimeError("call connect_and_measure() before start()")
        self.follower.enable()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="calibration-teleoperation", daemon=True
        )
        self._thread.start()

    def toggle_pause(self) -> bool:
        if self.paused:
            self._paused.clear()
        else:
            self._paused.set()
            self.follower.hold_current_position()
            state = self.follower.read_state()
            if state.all_finite:
                self._command = tuple(state.positions)
        return self.paused

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(2.0, 4.0 / self.rate_hz))
        self._thread = None
        if self.follower.lifecycle is ArmLifecycle.ENABLED:
            self.follower.hold_current_position()

    def close(self) -> None:
        try:
            self.stop()
        finally:
            self.leader.disconnect()

    def _run(self) -> None:
        period = 1.0 / self.rate_hz
        last_step = time.perf_counter()
        try:
            while not self._stop.is_set() and self.follower.lifecycle is ArmLifecycle.ENABLED:
                started = time.perf_counter()
                if self.paused:
                    last_step = started
                    self._stop.wait(min(period, 0.05))
                    continue
                now = time.monotonic()
                readings = dict(self.leader.read_positions())
                for servo_id, value in readings.items():
                    self._last_raw[servo_id] = value
                    self._last_seen[servo_id] = now
                lost = [
                    servo_id
                    for servo_id in self.leader.servo_ids
                    if now - self._last_seen[servo_id] > self.leader_loss_timeout_s
                ]
                if lost:
                    raise RuntimeError(f"Star leader readings lost for servo(s) {lost}")
                desired = self._mapped_targets(readings)
                # Clamp velocity against the measured tick time, not the nominal period:
                # a tick that overran its period (bus contention with the TUI thread)
                # must not slow the arm below the configured velocity. The cap keeps a
                # stalled tick from turning into a jump.
                dt = min(started - last_step, 4.0 * period)
                last_step = started
                max_step = self.max_velocity_rad_s * dt
                command = tuple(
                    current + max(-max_step, min(max_step, target - current))
                    for current, target in zip(self._command, desired, strict=True)
                )
                if not self.follower.send_targets(command):
                    raise RuntimeError(
                        f"follower refused a target while {self.follower.lifecycle.value}"
                    )
                self._command = command
                remaining = period - (time.perf_counter() - started)
                if remaining > 0:
                    self._stop.wait(remaining)
        except BaseException as error:
            self.error = f"{type(error).__name__}: {error}"
            self._stop.set()
        finally:
            if self.follower.lifecycle is ArmLifecycle.ENABLED:
                self.follower.hold_current_position()

    def _mapped_targets(self, readings: Mapping[int, float]) -> tuple[float, ...]:
        """Map leader readings to follower joint targets, gated by the soft limits.

        An arm joint mapping outside the follower's limits is refused: the fixed Star
        mapping should never produce it, so it means the mapping or the frame is wrong.
        Joints in ``clamp_joints`` (the gripper) are clamped instead: closing a MakerMod
        gripper deliberately commands a position past the object so the motor stalls
        compliantly (grip force is kp times position error), and the gripper mapping's
        rest point sits exactly on the follower's soft limit, so squeezing the leader
        always maps past it. The follower clamps to its own limits regardless; clamping
        here just applies the same policy before the gate.
        """

        mapped = tuple(float(value) for value in self.mapper.map(dict(readings)))
        if len(mapped) != len(self.follower.joint_names):
            raise RuntimeError(
                f"Star mapping produced {len(mapped)} joints; follower expects "
                f"{len(self.follower.joint_names)}"
            )
        targets: list[float] = []
        violations: list[str] = []
        for target, limit in zip(mapped, self.follower.limits(), strict=True):
            if math.isfinite(target) and limit.name in self.clamp_joints:
                target = limit.clamp(target)
            if not math.isfinite(target) or not limit.contains(target):
                violations.append(limit.name)
            targets.append(target)
        if violations:
            raise RuntimeError(f"Star mapping is outside follower limits: {', '.join(violations)}")
        return tuple(targets)
