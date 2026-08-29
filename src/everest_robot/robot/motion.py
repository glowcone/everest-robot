"""Bounded motion between operator-approved joint poses.

There is no inverse kinematics here and none is planned for this phase. A named position
is a measured set of joint angles, and getting there is a time-parameterized interpolation
under velocity and acceleration bounds, watched every tick.

What this layer does *not* provide is obstacle avoidance. A direct interpolation between
two approved poses is only safe if it has been driven at reduced speed on the physical
setup first (docs/named-position-capture.md). Where it has not, the parameters file
records a ``named_transitions`` waypoint sequence and :meth:`JointMotionController.
follow_transition` is the only approved way between those poses.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from everest_robot.robot.clock import Clock, SystemClock
from everest_robot.robot.contracts import (
    ArmLifecycle,
    CancelCheck,
    FailureReason,
    Heartbeat,
    JointState,
    MotionProfile,
    MotionResult,
)
from everest_robot.robot.parameters import NamedPosition, ParameterError, RobotParameters
from everest_robot.robot.ports import ArmPort, violations


@dataclass(frozen=True, slots=True)
class MotionTarget:
    """One destination for the controller: a label, joint values, and its bounds.

    Named presets become targets through :meth:`NamedPosition` lookup. Replay's initial
    alignment builds one directly, because the pose it must reach is whatever the episode
    recorded and cannot be a preset captured in advance. Everything downstream -- limit
    validation, bounded interpolation, settling, fault handling -- is identical either way,
    which is the point: there is no second, laxer path to the motors.
    """

    name: str
    joints: tuple[float, ...]
    profile: MotionProfile

    @classmethod
    def from_position(cls, position: NamedPosition) -> MotionTarget:
        return cls(name=position.name, joints=position.joints, profile=position.profile)


@dataclass(frozen=True, slots=True)
class TrapezoidPath:
    """A scalar path parameter s(t) from 0 to 1 under velocity and acceleration bounds.

    All joints share one time parameter, so the arm follows a straight line in joint space
    and every joint starts and stops together. Bounds are expressed per-joint and converted
    into path space by dividing by the largest joint displacement, which is what makes the
    slowest joint the one that sets the schedule.
    """

    duration_s: float
    _accel_s: float
    _cruise_v: float
    _accel_time_s: float

    @classmethod
    def plan(cls, displacement: float, profile: MotionProfile) -> TrapezoidPath:
        peak = abs(displacement)
        if peak == 0.0:
            return cls(0.0, 0.0, 0.0, 0.0)

        velocity = profile.max_velocity_rad_s / peak
        acceleration = profile.max_acceleration_rad_s2 / peak
        accel_time = velocity / acceleration
        if acceleration * accel_time * accel_time >= 1.0:
            # Triangular: the move is too short to ever reach the velocity bound.
            accel_time = math.sqrt(1.0 / acceleration)
            return cls(2.0 * accel_time, acceleration, acceleration * accel_time, accel_time)
        return cls(1.0 / velocity + accel_time, acceleration, velocity, accel_time)

    def at(self, elapsed_s: float) -> float:
        if self.duration_s <= 0.0 or elapsed_s >= self.duration_s:
            return 1.0
        if elapsed_s <= 0.0:
            return 0.0
        if elapsed_s <= self._accel_time_s:
            return 0.5 * self._accel_s * elapsed_s * elapsed_s
        remaining = self.duration_s - elapsed_s
        if remaining <= self._accel_time_s:
            return 1.0 - 0.5 * self._accel_s * remaining * remaining
        return self._cruise_v * (elapsed_s - self._accel_time_s / 2.0)


class JointMotionController:
    """Drives the arm between named positions and reports what physically happened.

    The controller assumes it already owns a connected arm: acquiring the arm, the
    exclusive lease and the reconnect path belong to
    :class:`~everest_robot.robot.session.RobotSession`. It will enable a merely-connected
    arm, and it holds position rather than releasing torque when a move ends or fails.
    """

    def __init__(
        self,
        port: ArmPort,
        parameters: RobotParameters,
        *,
        clock: Clock | None = None,
        heartbeat: Heartbeat | None = None,
        cancel: CancelCheck | None = None,
        limit_margin_rad: float = 0.0,
        tracking_grace_s: float = 0.25,
        max_tracking_error_rad: float | None = None,
        stale_feedback_s: float = 0.25,
        heartbeat_interval_s: float = 1.0,
        estop_on_failure: bool = False,
    ) -> None:
        self.port = port
        self.parameters = parameters
        self.clock = clock or SystemClock()
        self.heartbeat = heartbeat
        self.cancel = cancel
        self.limit_margin_rad = limit_margin_rad
        # Feedback necessarily lags the command while accelerating, so an instantaneous
        # deviation means nothing. The error must exceed the limit continuously for
        # `tracking_grace_s` before the move is called failed.
        self.tracking_grace_s = tracking_grace_s
        self.max_tracking_error_rad = max_tracking_error_rad
        self.stale_feedback_s = stale_feedback_s
        self.heartbeat_interval_s = heartbeat_interval_s
        # Holding is the default on failure: releasing torque drops whatever is in the
        # gripper. E-stop is opt-in and belongs to a deployment's safety policy.
        self.estop_on_failure = estop_on_failure
        self._last_heartbeat_s = -math.inf

    # ── public API ─────────────────────────────────────────────────────────────────
    def go_to_known_position(
        self,
        position_name: str,
        *,
        speed_scale: float = 1.0,
        dry_run: bool = False,
    ) -> MotionResult:
        """Move to one approved preset by direct joint-space interpolation.

        Returns a failed :class:`MotionResult` rather than raising for every condition the
        workflow can route on. Only a programming error (a bad ``speed_scale``) raises.
        """

        try:
            position = self.parameters.position(position_name)
        except ParameterError as error:
            return self._refused(position_name, FailureReason.UNKNOWN_POSITION, str(error))
        return self._run_legs(
            position_name,
            (MotionTarget.from_position(position),),
            speed_scale=speed_scale,
            dry_run=dry_run,
        )

    def go_to_joint_target(
        self,
        label: str,
        joints: Sequence[float],
        *,
        profile: MotionProfile | None = None,
        speed_scale: float = 1.0,
        dry_run: bool = False,
    ) -> MotionResult:
        """Move to an explicit joint target that is not a named preset.

        For poses that come from data rather than from an operator capture -- a recorded
        episode's initial state, for instance. The target still has to pass every check a
        preset does, including the active hardware limits, and the caller is responsible
        for having reached a pose from which a direct interpolation is known to be safe.
        """

        target = MotionTarget(
            name=label,
            joints=tuple(float(value) for value in joints),
            profile=profile or self.parameters.motion_defaults,
        )
        if len(target.joints) != len(self.port.joint_names):
            return self._refused(
                label,
                FailureReason.SCHEMA_MISMATCH,
                f"{label}: expected {len(self.port.joint_names)} joint values, "
                f"got {len(target.joints)}",
            )
        return self._run_legs(label, (target,), speed_scale=speed_scale, dry_run=dry_run)

    def follow_transition(
        self,
        transition_name: str,
        *,
        speed_scale: float = 1.0,
        dry_run: bool = False,
    ) -> MotionResult:
        """Move through an approved waypoint sequence, stopping at the first failure."""

        try:
            transition = self.parameters.transition(transition_name)
            legs = tuple(
                MotionTarget.from_position(self.parameters.position(name))
                for name in transition.waypoints
            )
        except ParameterError as error:
            return self._refused(transition_name, FailureReason.UNKNOWN_POSITION, str(error))
        return self._run_legs(
            transition.waypoints[-1], legs, speed_scale=speed_scale, dry_run=dry_run
        )

    # ── orchestration ──────────────────────────────────────────────────────────────
    def _run_legs(
        self,
        result_name: str,
        legs: Sequence[MotionTarget],
        *,
        speed_scale: float,
        dry_run: bool,
    ) -> MotionResult:
        started_s = self.clock.monotonic()
        waypoints = tuple(leg.name for leg in legs)
        detail = self.parameters.identity.mismatch_detail(self.port.identity)
        if detail is not None:
            return self._refused(result_name, FailureReason.IDENTITY_MISMATCH, detail, waypoints)

        state = self.port.read_state()
        try:
            return self._walk(
                result_name, legs, speed_scale, dry_run, waypoints, started_s, state
            )
        except BaseException:
            # A heartbeat raises when the workflow run is cancelled, and an interrupt can
            # arrive anywhere. Neither may leave a moving arm behind.
            self._safe_stop()
            raise

    def _walk(
        self,
        result_name: str,
        legs: Sequence[MotionTarget],
        speed_scale: float,
        dry_run: bool,
        waypoints: tuple[str, ...],
        started_s: float,
        state: JointState,
    ) -> MotionResult:
        commands = 0
        max_error = 0.0
        planned = 0.0
        clipped: tuple[str, ...] = ()
        already = True

        # A dry run never moves, so each leg after the first must be planned from where the
        # previous waypoint would have left the arm, not from where it is standing now.
        assumed: tuple[float, ...] | None = None
        for leg in legs:
            outcome = self._move_to(
                leg, speed_scale=speed_scale, dry_run=dry_run, assumed_start=assumed
            )
            assumed = leg.joints if dry_run else None
            commands += outcome.commands_sent
            planned += outcome.planned_duration_s
            max_error = max(max_error, outcome.max_tracking_error_rad)
            clipped = tuple(dict.fromkeys(clipped + outcome.clipped_joints))
            already = already and outcome.already_at_target
            state = outcome.state
            if outcome.failure_reason is not None:
                return self._result(
                    result_name,
                    reached=False,
                    state=state,
                    max_error=max_error,
                    elapsed_s=self.clock.monotonic() - started_s,
                    commands=commands,
                    waypoints=waypoints,
                    clipped=clipped,
                    dry_run=dry_run,
                    planned_duration_s=planned,
                    failure_reason=outcome.failure_reason,
                    failure_detail=outcome.failure_detail,
                )

        return self._result(
            result_name,
            reached=not dry_run,
            state=state,
            max_error=max_error,
            elapsed_s=self.clock.monotonic() - started_s,
            commands=commands,
            waypoints=waypoints,
            clipped=clipped,
            dry_run=dry_run,
            planned_duration_s=planned,
            already_at_target=already and not dry_run,
        )

    # ── one leg ────────────────────────────────────────────────────────────────────
    def _move_to(
        self,
        position: MotionTarget,
        *,
        speed_scale: float,
        dry_run: bool,
        assumed_start: tuple[float, ...] | None = None,
    ) -> _LegOutcome:
        profile = position.profile.scaled(speed_scale)
        targets = position.joints
        limits = self.port.limits()

        # A preset that needs clipping is not an approved preset: refuse, never clamp.
        outside = violations(targets, limits, margin=self.limit_margin_rad)
        if outside:
            return self._leg_failure(
                FailureReason.LIMIT_VIOLATION,
                f"{position.name}: {', '.join(outside)} outside the active hardware limits",
            )

        state = self.port.read_state()
        if state.lifecycle is ArmLifecycle.FAULT:
            return self._leg_failure(
                FailureReason.MOTOR_FAULT, f"arm is in fault: {state.fault_reason}", state
            )
        if state.lifecycle is ArmLifecycle.DISCONNECTED:
            return self._leg_failure(
                FailureReason.NOT_ENABLED, "arm is disconnected; open a session first", state
            )
        if not state.all_finite:
            return self._leg_failure(
                FailureReason.STALE_FEEDBACK, "joint feedback is missing", state
            )

        origin = assumed_start or state.positions
        displacement = max(
            abs(target - current) for target, current in zip(targets, origin, strict=True)
        )
        path = TrapezoidPath.plan(displacement, profile)

        if dry_run:
            return _LegOutcome(state=state, planned_duration_s=path.duration_s)
        if displacement <= profile.tolerance_rad:
            # Already there. Enabling and re-commanding would move the arm for no reason.
            return _LegOutcome(state=state, already_at_target=True)

        if state.lifecycle is ArmLifecycle.CONNECTED:
            self.port.enable()

        return self._execute(position, targets, state, path, profile)

    def _execute(
        self,
        position: MotionTarget,
        targets: Sequence[float],
        start_state: JointState,
        path: TrapezoidPath,
        profile: MotionProfile,
    ) -> _LegOutcome:
        start = start_state.positions
        deltas = tuple(target - origin for target, origin in zip(targets, start, strict=True))
        dt = 1.0 / profile.control_rate_hz
        tolerance = profile.tolerance_rad
        tracking_limit = self.max_tracking_error_rad or max(8.0 * tolerance, 0.1)

        began_s = self.clock.monotonic()
        command: tuple[float, ...] = tuple(start)
        deviating_since_s: float | None = None
        last_sequence = start_state.sequence
        last_change_s = began_s
        settled_since_s: float | None = None
        commands = 0
        max_error = 0.0
        state = start_state

        while True:
            now = self.clock.monotonic()
            elapsed = now - began_s
            self._beat(now)

            if self.cancel is not None and self.cancel():
                return self._leg_failure(
                    FailureReason.CANCELLED, f"{position.name}: cancelled after {elapsed:.2f}s",
                    state, commands, max_error, path.duration_s,
                )
            if elapsed > profile.timeout_s:
                return self._leg_failure(
                    FailureReason.TIMEOUT,
                    f"{position.name}: not settled within {profile.timeout_s}s "
                    f"(max tracking error {max_error:.4f} rad)",
                    state, commands, max_error, path.duration_s,
                )

            fraction = path.at(elapsed)
            command = tuple(
                origin + delta * fraction for origin, delta in zip(start, deltas, strict=True)
            )
            if not self.port.send_targets(command):
                return self._leg_failure(
                    FailureReason.NOT_ENABLED,
                    f"{position.name}: the driver refused a joint command",
                    state, commands, max_error, path.duration_s,
                )
            commands += 1

            self.clock.sleep(dt)
            state = self.port.read_state()

            if state.has_fault:
                return self._leg_failure(
                    FailureReason.MOTOR_FAULT,
                    f"{position.name}: {state.fault_reason or 'motor fault'}",
                    state, commands, max_error, path.duration_s,
                )
            if state.sequence != last_sequence:
                last_sequence = state.sequence
                last_change_s = state.monotonic_s
            elif state.monotonic_s - last_change_s > self.stale_feedback_s:
                return self._leg_failure(
                    FailureReason.STALE_FEEDBACK,
                    f"{position.name}: no fresh feedback for "
                    f"{state.monotonic_s - last_change_s:.2f}s",
                    state, commands, max_error, path.duration_s,
                )

            error = state.max_tracking_error(command)
            max_error = max(max_error, error)
            if error > tracking_limit:
                deviating_since_s = now if deviating_since_s is None else deviating_since_s
                if now - deviating_since_s >= self.tracking_grace_s:
                    return self._leg_failure(
                        FailureReason.TRACKING_ERROR,
                        f"{position.name}: tracking error {error:.4f} rad exceeded "
                        f"{tracking_limit} rad for {self.tracking_grace_s}s",
                        state, commands, max_error, path.duration_s,
                    )
            else:
                deviating_since_s = None

            within = state.max_tracking_error(targets) <= tolerance
            if within and fraction >= 1.0:
                # Every joint must stay inside tolerance for the whole settle window;
                # touching it once says nothing about where the arm comes to rest.
                settled_since_s = settled_since_s if settled_since_s is not None else now
                if now - settled_since_s >= profile.settle_time_s:
                    self.port.hold_current_position()
                    return _LegOutcome(
                        state=state,
                        commands_sent=commands,
                        max_tracking_error_rad=max_error,
                        planned_duration_s=path.duration_s,
                    )
            else:
                settled_since_s = None

    # ── helpers ────────────────────────────────────────────────────────────────────
    def _safe_stop(self) -> None:
        """Hold the arm where it is, or e-stop if the deployment asked for that.

        A driver already in FAULT is holding under its own policy and is not commanded
        further.
        """

        if self.port.lifecycle is ArmLifecycle.ENABLED:
            if self.estop_on_failure:
                self.port.estop()
            else:
                self.port.hold_current_position()

    def _beat(self, now: float) -> None:
        if self.heartbeat is None:
            return
        if now - self._last_heartbeat_s >= self.heartbeat_interval_s:
            self._last_heartbeat_s = now
            self.heartbeat()

    def _leg_failure(
        self,
        reason: FailureReason,
        detail: str,
        state: JointState | None = None,
        commands: int = 0,
        max_error: float = 0.0,
        planned_duration_s: float = 0.0,
    ) -> _LegOutcome:
        """Leave the arm safe, then report.

        Holding beats releasing: torque release drops whatever the gripper is carrying. A
        driver already in FAULT is holding by its own policy and is not commanded further.
        """

        self._safe_stop()
        return _LegOutcome(
            state=state if state is not None else self.port.read_state(),
            commands_sent=commands,
            max_tracking_error_rad=max_error,
            planned_duration_s=planned_duration_s,
            failure_reason=reason,
            failure_detail=detail,
        )

    def _refused(
        self,
        position_name: str,
        reason: FailureReason,
        detail: str,
        waypoints: tuple[str, ...] = (),
    ) -> MotionResult:
        """A refusal decided before any command was sent."""

        state = self.port.read_state()
        return self._result(
            position_name,
            reached=False,
            state=state,
            max_error=0.0,
            elapsed_s=0.0,
            commands=0,
            waypoints=waypoints,
            clipped=(),
            dry_run=False,
            planned_duration_s=0.0,
            failure_reason=reason,
            failure_detail=detail,
        )

    def _result(
        self,
        position_name: str,
        *,
        reached: bool,
        state: JointState,
        max_error: float,
        elapsed_s: float,
        commands: int,
        waypoints: tuple[str, ...],
        clipped: tuple[str, ...],
        dry_run: bool,
        planned_duration_s: float,
        already_at_target: bool = False,
        failure_reason: FailureReason | None = None,
        failure_detail: str | None = None,
    ) -> MotionResult:
        return MotionResult(
            position_name=position_name,
            reached=reached,
            joint_names=self.port.joint_names,
            final_joints=state.positions,
            max_tracking_error_rad=max_error,
            elapsed_s=elapsed_s,
            commands_sent=commands,
            robot_id=self.parameters.identity.robot_id,
            calibration_id=self.parameters.identity.calibration_id,
            config_digest=self.parameters.config_digest,
            already_at_target=already_at_target,
            dry_run=dry_run,
            planned_duration_s=planned_duration_s,
            clipped_joints=clipped,
            waypoints=waypoints if len(waypoints) > 1 else (),
            failure_reason=failure_reason,
            failure_detail=failure_detail,
        )


@dataclass(frozen=True, slots=True)
class _LegOutcome:
    """One waypoint's result, before it is folded into the durable MotionResult."""

    state: JointState
    commands_sent: int = 0
    max_tracking_error_rad: float = 0.0
    # Motion refuses out-of-limit presets rather than clamping them, so this stays empty
    # here; it exists because the same fold is used by the layers that do clip.
    clipped_joints: tuple[str, ...] = ()
    planned_duration_s: float = 0.0
    already_at_target: bool = False
    failure_reason: FailureReason | None = None
    failure_detail: str | None = None
