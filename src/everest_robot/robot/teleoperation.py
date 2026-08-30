"""Lease-local Star 102 leader following, and the park that has to follow it.

The caller owns the follower's :class:`RobotSession` and therefore its lease.  This
module opens only the leader bus and drives that already-claimed follower; it must never
construct a second arm or claim.  Leader positions are mapped with maker-arm-sdk's
private-protocol Star mapping, not LeRobot's MIT/degree frame.

:func:`park_at_start_pose` belongs here rather than in either CLI because it is the other
half of :meth:`TeleoperationController.connect_and_measure`: that method is the only thing
that records where the arm was before it was energized, and every caller that enables the
follower owes the operator a drive back to it before torque comes off.  Both
``robot-monitor`` and ``robot-pixel-map collect`` call it; a third caller that follows a
leader must too.
"""

from __future__ import annotations

import math
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from everest_robot.robot.contracts import ArmLifecycle, MotionProfile
from everest_robot.robot.ports import ArmPort, clip_to_limits

if TYPE_CHECKING:
    from everest_robot.robot.session import RobotSession

# Every powered session ends by driving the follower back to the pose it was measured at
# before it was enabled, and only then releasing torque. Reduced speed for `robot-goto`'s
# reason: the leader may have walked the arm a long way from where it started, and this
# interpolation is a direct one with no obstacle avoidance behind it.
PARK_SPEED_SCALE = 0.25
PARK_TARGET_NAME = "teleoperation-start"

# How far outside the soft limits a start pose may sit and still be parked at the nearest
# in-limit pose instead of being refused outright.
#
# It is not slack in the limits: nothing is ever commanded outside them. It is the width of
# the disagreement the drivers already contain. `RobstrideMitPort.enable()` accepts a joint
# reading up to WRAP_GRACE_DEG (20 deg, mirroring MakerFollower) outside its soft limits
# before it looks for a whole-turn wrap, so an arm resting under gravity -- shoulder_lift
# and elbow_flex drooped past their bounds -- energizes normally and then reports a pose
# `JointMotionController` must refuse as a target. That refusal used to cost the whole park.
# Beyond this width the pose is not a droop and is refused as before: 20 deg of gravity is a
# resting arm, 4 rad is a joint that no longer means what the calibration says it means.
PARK_LIMIT_TOLERANCE_RAD = math.radians(20.0)


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
        out_of_range_timeout_s: float = 2.0,
        clamp_joints: Sequence[str] = ("gripper",),
    ) -> None:
        if rate_hz <= 0 or not math.isfinite(rate_hz):
            raise ValueError("rate_hz must be finite and positive")
        if max_velocity_rad_s <= 0 or not math.isfinite(max_velocity_rad_s):
            raise ValueError("max_velocity_rad_s must be finite and positive")
        if leader_loss_timeout_s <= 0 or not math.isfinite(leader_loss_timeout_s):
            raise ValueError("leader_loss_timeout_s must be finite and positive")
        if out_of_range_timeout_s <= 0 or not math.isfinite(out_of_range_timeout_s):
            raise ValueError("out_of_range_timeout_s must be finite and positive")
        self.follower = follower
        self.leader = leader
        self.mapper = mapper
        self.rate_hz = float(rate_hz)
        self.max_velocity_rad_s = float(max_velocity_rad_s)
        self.leader_loss_timeout_s = float(leader_loss_timeout_s)
        self.out_of_range_timeout_s = float(out_of_range_timeout_s)
        self.clamp_joints = frozenset(clamp_joints)
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_raw: dict[int, float] = {}
        self._last_seen: dict[int, float] = {}
        self._command: tuple[float, ...] = ()
        self._start_pose: tuple[float, ...] = ()
        self._clamped: tuple[str, ...] = ()
        self._sustained: tuple[str, ...] = ()
        self._excursion_joints: tuple[str, ...] = ()
        self._out_of_range_since: float | None = None
        self.error: str | None = None

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    @property
    def clamped_joints(self) -> tuple[str, ...]:
        """Arm joints whose leader pose is currently being held at a soft limit.

        Written by the control loop and read by the TUI, which is why it is a single
        tuple rebind rather than a mutated list: the reader always sees one whole cycle's
        answer. Empty is the normal case, and the deliberate gripper clamp never appears
        here -- an operator needs to see the excursions that are not routine.
        """

        return self._clamped

    @property
    def sustained_excursions(self) -> tuple[str, ...]:
        """Clamped joints that have stayed clamped for ``out_of_range_timeout_s``.

        A louder grade of :attr:`clamped_joints`, not a different condition, and never a
        reason to stop: the follower is held inside its own limits either way. It exists
        because a leader joint parked past the follower's travel looks exactly like a dead
        joint -- the follower simply does not move -- and an operator who has been pushing
        for two seconds deserves to be told which joint is not coming with them.

        Live, like :attr:`clamped_joints`: it empties when the leader comes back.
        :attr:`excursion_joints` is the record that survives.
        """

        return self._sustained

    @property
    def excursion_joints(self) -> tuple[str, ...]:
        """Every joint that has had a sustained excursion this session, in first-seen order.

        Reported once when the session ends. A joint that appears here repeatedly across
        sessions is the Star mapping, not the operator: see the wrist_flex note in
        :meth:`_mapped_targets`.
        """

        return self._excursion_joints

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def start_pose(self) -> tuple[float, ...]:
        """The follower's measured pose from before it was ever enabled, or ``()``.

        Recorded by :meth:`connect_and_measure`, which reads it while the arm is still
        merely connected, and never written again -- following must not drag it along.
        The caller needs somewhere safe to put the arm before torque comes off, and this
        is the one pose that is known to have held the arm without power: it is where the
        operator parked it. Empty until the leader has been measured, which is exactly
        the case where nothing was enabled and there is nothing to undo.
        """

        return self._start_pose

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
            targets, out_of_range = self._mapped_targets(self._last_raw)
            if out_of_range:
                raise RuntimeError(
                    f"Star mapping is outside follower limits: {', '.join(out_of_range)}"
                )
            current = self.follower.read_state()
            if not current.all_finite:
                raise RuntimeError("follower has missing position feedback")
            self._command = tuple(current.positions)
            self._start_pose = self._command
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
            self._clamped = ()
            self._sustained = ()
            self._out_of_range_since = None
            self.follower.hold_current_position()
            state = self.follower.read_state()
            if state.all_finite:
                self._command = tuple(state.positions)
        return self.paused

    def stop(self, *, hold: bool = True) -> None:
        """Stop following. ``hold=True`` freezes the follower where it is.

        ``hold=False`` is for a teardown that is about to cut torque anyway. Commanding a
        hold first only snaps the arm toward its last target a moment before it goes limp,
        which the operator feels as a lurch on Ctrl-C.
        """

        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(2.0, 4.0 / self.rate_hz))
        self._thread = None
        if hold and self.follower.lifecycle is ArmLifecycle.ENABLED:
            self.follower.hold_current_position()

    def close(self, *, hold: bool = True) -> None:
        try:
            self.stop(hold=hold)
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
                desired, out_of_range = self._mapped_targets(readings)
                self._clamped = out_of_range
                if not out_of_range:
                    self._out_of_range_since = None
                    self._sustained = ()
                elif self._out_of_range_since is None:
                    self._out_of_range_since = now
                elif now - self._out_of_range_since > self.out_of_range_timeout_s:
                    self._sustained = out_of_range
                    self._excursion_joints = tuple(
                        dict.fromkeys(self._excursion_joints + out_of_range)
                    )
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
            self._clamped = ()
            self._sustained = ()
            if self.follower.lifecycle is ArmLifecycle.ENABLED:
                self.follower.hold_current_position()

    def _mapped_targets(
        self, readings: Mapping[int, float]
    ) -> tuple[tuple[float, ...], tuple[str, ...]]:
        """Map leader readings to follower targets, and say which hit a soft limit.

        Every joint is clamped into the follower's limits; the second element names the
        arm joints that needed it, and the callers decide what that means. Before the
        follower is enabled a single excursion is fatal -- the fixed Star mapping should
        never produce one from the pose the arms are actually standing in, so it means the
        mapping or the frame is wrong. Once following, an excursion is ordinary and never
        stops the session, however long it lasts: the follower is held at the soft limit
        until the leader comes back, and the loop only escalates how loudly it says so
        (:attr:`clamped_joints`, then :attr:`sustained_excursions`).

        It is ordinary because the fixed mapping does not give every joint headroom on
        both sides. wrist_flex is the worst: its ``base_rad`` (2.122) is *exactly* the
        follower's upper soft limit, so every Star reading below the mapping's zero_deg
        maps out of range from the first degree. Half that joint's leader travel is
        unreachable by construction, and treating it as evidence of a broken mapping ended
        calibration sessions over a pose the arm could hold perfectly well.

        Joints in ``clamp_joints`` (the gripper) are clamped without being reported:
        closing a MakerMod gripper deliberately commands a position past the object so
        the motor stalls compliantly (grip force is kp times position error), and the
        gripper mapping's rest point sits exactly on the follower's soft limit, so
        squeezing the leader always maps past it. The follower clamps to its own limits
        regardless; clamping here just applies the same policy one layer up.
        """

        mapped = tuple(float(value) for value in self.mapper.map(dict(readings)))
        if len(mapped) != len(self.follower.joint_names):
            raise RuntimeError(
                f"Star mapping produced {len(mapped)} joints; follower expects "
                f"{len(self.follower.joint_names)}"
            )
        targets: list[float] = []
        out_of_range: list[str] = []
        for target, limit in zip(mapped, self.follower.limits(), strict=True):
            # A non-finite target is not an excursion and cannot be clamped into one:
            # the mapper has produced a number that means nothing, so stop either way.
            if not math.isfinite(target):
                raise RuntimeError(f"Star mapping produced a non-finite target for {limit.name}")
            if not limit.contains(target):
                target = limit.clamp(target)
                if limit.name not in self.clamp_joints:
                    out_of_range.append(limit.name)
            targets.append(target)
        return tuple(targets), tuple(out_of_range)


# ── parking the arm before torque comes off ────────────────────────────────────────
def park_at_start_pose(session: RobotSession, controller: TeleoperationController) -> None:
    """Drive the follower back to its pre-teleoperation pose before torque comes off.

    :meth:`RobotSession.close` disables the arm on every exit path, so whichever pose the
    leader left the follower in is the pose it goes limp from. That is fine where the
    operator parked it and nowhere else, which is why this runs on the failure paths and
    not only on a clean quit: a leader-loss or sustained-excursion stop is precisely when
    the arm is somewhere it must not simply be dropped from.

    Skipping this does not leave the arm where it was -- it leaves it wherever gravity
    takes it, fast, and because ``start_pose`` was measured with the torque already off
    that is *almost exactly this pose*. A missing park therefore looks like a working one
    that happens to be instant, which is how ``robot-pixel-map collect`` went without one
    for as long as it did. Any caller that enables the follower needs this.

    Reported, never raised. The teleoperation error is the one the operator has to see,
    and a park that could not run must not replace it in the traceback. Whatever happens
    the arm is left held rather than moving -- ``JointMotionController`` holds on every
    failure path of its own -- so a failure here costs the parking, not the safety.
    """

    from everest_robot.robot.contracts import FailureReason

    start = controller.start_pose
    if not start:
        return  # the leader was never measured, so the arm was never enabled
    if session.port.lifecycle is not ArmLifecycle.ENABLED:
        # Disabled or faulted under the driver's own policy. There is nothing to command,
        # and saying so beats a silent skip: the arm is wherever it stopped.
        print(
            f"not returning to the starting pose: the arm is {session.port.lifecycle.value}",
            file=sys.stderr,
        )
        return

    start = _parkable(session, start)
    profile = _park_profile(session, controller, start)
    print(
        "returning the arm to the pose it started from at "
        f"{profile.max_velocity_rad_s:.3f} rad/s "
        f"(about {_planned_park_s(session, start, profile):.0f}s)...",
        file=sys.stderr,
    )
    try:
        # speed_scale is already baked into the profile; scaling twice would halve it again.
        result = session.motion.go_to_joint_target(PARK_TARGET_NAME, start, profile=profile)
    except BaseException as error:  # including a second Ctrl-C during the park
        print(
            f"return to the starting pose FAILED  {type(error).__name__}: {error}\n"
            "the arm is held where it is and torque is about to come off; support it.",
            file=sys.stderr,
        )
        return

    if result.failure_reason is not None:
        detail = result.failure_detail
        if result.failure_reason is FailureReason.LIMIT_VIOLATION:
            # `_parkable` already accepted everything within PARK_LIMIT_TOLERANCE_RAD, so
            # this pose is further out than a resting arm can droop: a re-zeroed or wrapped
            # joint. Not something to clamp our way past.
            detail = (
                f"{detail} -- further outside them than a resting arm can droop, so the "
                "arm's limits are not the ones it started under"
            )
        print(
            f"return to the starting pose FAILED  {result.failure_reason}: {detail}\n"
            "the arm is held where it is and torque is about to come off; support it.",
            file=sys.stderr,
        )
    elif result.already_at_target:
        print("already at the pose it started from; the arm did not move.", file=sys.stderr)
    else:
        print(
            f"back at the starting pose after {result.elapsed_s:.2f}s "
            f"(max tracking error {result.max_tracking_error_rad:.4f} rad).",
            file=sys.stderr,
        )


def _parkable(session: RobotSession, start: Sequence[float]) -> tuple[float, ...]:
    """The start pose, or the nearest in-limit pose to it, or the start pose unchanged.

    The pose the arm was measured at is not necessarily a pose the arm may be *commanded*
    to: `enable()` admits a joint drooped up to its wrap grace past a soft limit, and
    `JointMotionController` refuses any target outside them. A gravity-loaded arm is the
    ordinary case, not an exotic one, so refusing the whole park over it means the powered
    session ends with no park at all -- the arm going limp from wherever the leader left it,
    which is the failure this module exists to prevent.

    So park at the limit instead, and say so. The arm then settles the last few degrees when
    torque comes off, landing where it started: a short, bounded drop from a held pose, and
    strictly better than the full one. Past PARK_LIMIT_TOLERANCE_RAD nothing is clamped and
    the pose is returned untouched, for the motion controller to refuse and report.
    """

    command = clip_to_limits(start, session.port.limits())
    if not command.clipped_joints:
        return tuple(float(value) for value in start)

    excess = max(
        abs(clipped - measured)
        for clipped, measured in zip(command.targets, start, strict=True)
    )
    if excess > PARK_LIMIT_TOLERANCE_RAD:
        return tuple(float(value) for value in start)

    print(
        f"the pose it started from is {math.degrees(excess):.1f} deg outside the soft "
        f"limits on {', '.join(command.clipped_joints)}; parking at the nearest in-limit "
        "pose instead -- the arm settles the rest of the way when torque comes off.",
        file=sys.stderr,
    )
    return command.targets


def _park_profile(
    session: RobotSession,
    controller: TeleoperationController,
    start: Sequence[float],
) -> MotionProfile:
    """The bounds the park runs under: slow, and given the time to actually be slow.

    Two ceilings, whichever is lower. ``PARK_SPEED_SCALE`` of the configured motion
    defaults is ``robot-goto``'s reason -- this is a direct joint-space interpolation with
    nothing avoiding obstacles, and the leader may have walked the arm a long way from
    where it started. The controller's own ``max_velocity_rad_s`` is the second: it is the
    speed the operator has watched this arm move at for the whole session, and the park is
    not the moment to exceed it.

    The timeout is derived from the planned trajectory rather than taken from the file.
    Scaling a profile scales its velocity and acceleration and leaves ``timeout_s`` alone,
    so a slow-enough park is one the fixed timeout cannot finish: at 0.125 rad/s a 10 s
    budget buys about 1.2 rad of travel, and a longer park would fail partway and then be
    disabled wherever it had got to. That would make speed and completion trade against
    each other, and the speed is the part that is not negotiable.
    """

    defaults = session.parameters.motion_defaults
    # scaled() rejects anything above 1.0, so the controller's cap can only slow this down.
    scale = min(PARK_SPEED_SCALE, controller.max_velocity_rad_s / defaults.max_velocity_rad_s)
    profile = defaults.scaled(min(1.0, max(scale, 1e-3)))
    planned = _planned_park_s(session, start, profile)
    # Half again over the plan, plus the settle window and a second of slack, so the budget
    # still catches an arm that is not tracking rather than one that is merely taking the
    # time it was told to take.
    return replace(
        profile,
        timeout_s=max(defaults.timeout_s, 1.5 * planned + profile.settle_time_s + 1.0),
    )


def _planned_park_s(
    session: RobotSession, start: Sequence[float], profile: MotionProfile
) -> float:
    """How long the park is supposed to take from where the arm is standing now."""

    from everest_robot.robot.motion import TrapezoidPath

    state = session.port.read_state()
    if not state.all_finite:
        return 0.0
    displacement = max(
        abs(target - current) for target, current in zip(start, state.positions, strict=True)
    )
    return TrapezoidPath.plan(displacement, profile).duration_s
