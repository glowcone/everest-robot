"""Translation tests for the MIT-protocol adapter.

Everything runs against a stub bus: no lerobot import, no CAN. The port's frame
conversion is checked by round-tripping through the same :class:`JointFrame` the
deployment path builds. Behaviour on a live arm belongs to the bench procedure in the
ADR addendum, not here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import pytest

from everest_robot.robot.contracts import ArmLifecycle, RobotIdentity
from everest_robot.robot.lerobot_bridge import JointFrame
from everest_robot.robot.robstride_mit_port import RobstrideMitPort

JOINTS = ("shoulder_pan", "shoulder_lift", "gripper")
IDENTITY = RobotIdentity("maker-arm-02", "maker-arm-v1", "cal-2026-08-20", JOINTS)
OFFSETS_DEG = (-120.0, 30.0, -0.25)
FRAME = JointFrame(JOINTS, offsets_deg=OFFSETS_DEG)
LIMITS_DEG = {
    "shoulder_pan": (-150.0, 150.0),
    "shoulder_lift": (-170.0, -5.0),
    "gripper": (-120.0, -2.5),
}
GAINS = {
    "shoulder_pan": (60.0, 4.0),
    "shoulder_lift": (150.0, 4.5),
    "gripper": (20.0, 0.5),
}


@dataclass
class StubBus:
    """The subset of ``RobstrideMotorsBus`` the port touches."""

    positions: dict[str, float] = field(
        default_factory=lambda: {"shoulder_pan": -110.0, "shoulder_lift": -40.0, "gripper": -5.0}
    )
    connected: bool = False
    torque_enabled: bool = False
    faulted: dict[str, str] = field(default_factory=dict)
    silent: set[str] = field(default_factory=set)
    goal_writes: list[dict[str, float]] = field(default_factory=list)
    gain_writes: dict[str, dict[str, float]] = field(default_factory=dict)
    last_feedback_time: dict[str, float | None] = field(
        default_factory=lambda: dict.fromkeys(JOINTS, 1.0)
    )
    cleared: list[str] = field(default_factory=list)

    def connect(self, handshake: bool = True) -> None:
        self.connected = True

    def disconnect(self, disable_torque: bool = True) -> None:
        self.connected = False
        if disable_torque:
            self.torque_enabled = False

    def enable_torque(self) -> None:
        self.torque_enabled = True

    def disable_torque(self) -> None:
        self.torque_enabled = False

    def sync_write(self, data_name: str, values: dict[str, float]) -> None:
        if data_name in ("Kp", "Kd"):
            self.gain_writes[data_name] = dict(values)
        elif data_name == "Goal_Position":
            self.goal_writes.append(dict(values))
        else:  # pragma: no cover - would be a port bug
            raise ValueError(data_name)

    def read(self, data_name: str, motor: str) -> float:
        self._check(motor)
        if data_name == "Present_Position":
            return self.positions[motor]
        if data_name == "Present_Velocity":
            return 90.0  # deg/s
        if data_name == "Present_Torque":
            return 0.5
        if data_name == "Temperature_MOS":
            return 31.0
        raise ValueError(data_name)  # pragma: no cover

    def update_motor_state(self, motor: str) -> bool:
        self._check(motor)
        self.cleared.append(motor)
        return True

    def _check(self, motor: str) -> None:
        if motor in self.faulted:
            raise RuntimeError(self.faulted[motor])
        if motor in self.silent:
            raise ConnectionError(f"no response from {motor}")


def make_port(**overrides: Any) -> RobstrideMitPort:
    bus = StubBus()
    for name, value in overrides.items():
        setattr(bus, name, value)
    return RobstrideMitPort(bus, IDENTITY, FRAME, limits_deg=LIMITS_DEG, gains=GAINS)


def enabled_port(**overrides: Any) -> RobstrideMitPort:
    port = make_port(**overrides)
    port.connect()
    port.enable()
    return port


# ── construction ───────────────────────────────────────────────────────────────────
def test_a_frame_for_different_joints_is_refused() -> None:
    with pytest.raises(ValueError, match="same order"):
        RobstrideMitPort(
            StubBus(), IDENTITY, JointFrame(("a", "b", "c")), limits_deg=LIMITS_DEG, gains=GAINS
        )


def test_a_joint_missing_from_the_mit_tables_is_refused() -> None:
    with pytest.raises(ValueError, match="shoulder_lift"):
        RobstrideMitPort(
            StubBus(),
            IDENTITY,
            FRAME,
            limits_deg={"shoulder_pan": (-1.0, 1.0), "gripper": (-1.0, 1.0)},
            gains=GAINS,
        )


def test_from_deployment_refuses_an_identity_frame_before_importing_lerobot() -> None:
    with pytest.raises(ValueError, match="lerobot_frame"):
        RobstrideMitPort.from_deployment(
            IDENTITY, JointFrame(JOINTS), port="/dev/ttyACM0", backend="slcan"
        )


# ── lifecycle ──────────────────────────────────────────────────────────────────────
def test_lifecycle_transitions_and_torque_side_effects() -> None:
    port = make_port()
    assert port.lifecycle is ArmLifecycle.DISCONNECTED

    port.connect()
    assert port.lifecycle is ArmLifecycle.CONNECTED
    assert port._bus.torque_enabled is False

    port.enable()
    assert port.lifecycle is ArmLifecycle.ENABLED
    assert port._bus.torque_enabled is True

    port.disable()
    assert port.lifecycle is ArmLifecycle.CONNECTED
    assert port._bus.torque_enabled is False

    port.enable()
    port.estop()
    assert port.lifecycle is ArmLifecycle.CONNECTED
    assert port._bus.torque_enabled is False

    port.disconnect()
    assert port.lifecycle is ArmLifecycle.DISCONNECTED
    assert port._bus.connected is False


def test_disconnect_releases_torque_from_enabled() -> None:
    port = enabled_port()
    port.disconnect()
    assert port._bus.torque_enabled is False
    assert port.lifecycle is ArmLifecycle.DISCONNECTED


def test_commands_are_refused_outside_enabled() -> None:
    port = make_port()
    assert port.send_targets([0.0, -1.5, -1.0]) is False
    port.connect()
    assert port.send_targets([0.0, -1.5, -1.0]) is False
    assert port.hold_current_position() is False
    assert port._bus.goal_writes == []


def test_enable_is_refused_outside_connected() -> None:
    port = make_port()
    with pytest.raises(RuntimeError, match="cannot enable"):
        port.enable()


def test_enable_writes_the_follow_gains_before_torque() -> None:
    port = enabled_port()
    assert port._bus.gain_writes["Kp"] == {name: kp for name, (kp, _) in GAINS.items()}
    assert port._bus.gain_writes["Kd"] == {name: kd for name, (_, kd) in GAINS.items()}


# ── enable limit gating and full-turn correction ──────────────────────────────────
def test_enable_refuses_a_pose_outside_limits_and_not_by_a_whole_turn() -> None:
    port = make_port(
        positions={"shoulder_pan": -110.0, "shoulder_lift": 90.0, "gripper": -5.0}
    )
    port.connect()
    with pytest.raises(RuntimeError, match="shoulder_lift"):
        port.enable()
    assert port._bus.torque_enabled is False


def test_enable_applies_a_full_turn_shift_when_a_joint_is_a_whole_turn_out() -> None:
    port = make_port(
        positions={"shoulder_pan": -110.0, "shoulder_lift": 320.0, "gripper": -5.0}
    )
    port.connect()
    port.enable()

    # Feedback is corrected by the shift: 320 - 360 = -40 deg.
    state = port.read_state()
    expected = FRAME.to_radians([-110.0, -40.0, -5.0])
    assert state.positions == pytest.approx(expected)

    # Commands are shifted back into the motor's own frame.
    assert port.send_targets(expected) is True
    write = port._bus.goal_writes[-1]
    assert write["shoulder_lift"] == pytest.approx(-40.0 + 360.0)
    assert write["shoulder_pan"] == pytest.approx(-110.0)


# ── frame conversion ───────────────────────────────────────────────────────────────
def test_read_state_converts_degrees_through_the_frame() -> None:
    port = make_port()
    port.connect()
    state = port.read_state()

    assert state.names == JOINTS
    assert state.positions == pytest.approx(FRAME.to_radians([-110.0, -40.0, -5.0]))
    assert state.velocities == pytest.approx((math.radians(90.0),) * 3)
    assert state.temperatures == (31.0, 31.0, 31.0)
    assert state.lifecycle is ArmLifecycle.CONNECTED


def test_send_targets_round_trips_through_the_frame() -> None:
    port = enabled_port()
    targets = FRAME.to_radians([-100.0, -60.0, -10.0])

    assert port.send_targets(targets) is True
    assert port._bus.goal_writes == [
        {
            name: pytest.approx(deg)
            for name, deg in zip(JOINTS, [-100.0, -60.0, -10.0], strict=True)
        }
    ]
    assert port.last_command.clipped_joints == ()


def test_limits_are_the_mit_tables_in_calibrated_radians() -> None:
    limits = make_port().limits()

    assert tuple(limit.name for limit in limits) == JOINTS
    lows = FRAME.to_radians([LIMITS_DEG[name][0] for name in JOINTS])
    highs = FRAME.to_radians([LIMITS_DEG[name][1] for name in JOINTS])
    for limit, low, high in zip(limits, lows, highs, strict=True):
        assert limit.lower_rad == pytest.approx(low)
        assert limit.upper_rad == pytest.approx(high)
        assert limit.lower_rad < limit.upper_rad


# ── command validation ─────────────────────────────────────────────────────────────
def test_malformed_targets_never_reach_the_bus() -> None:
    port = enabled_port()

    assert port.send_targets([0.1, 0.2]) is False
    assert port.send_targets([0.1, math.nan, 0.0]) is False
    assert port._bus.goal_writes == []


def test_out_of_limit_targets_are_clipped_and_reported() -> None:
    port = enabled_port()
    inside = FRAME.to_radians([-100.0, -60.0, -10.0])
    beyond = (inside[0], FRAME.to_radians([0.0, -400.0, 0.0])[1], inside[2])

    assert port.send_targets(beyond) is True
    assert port.last_command.clipped_joints == ("shoulder_lift",)
    assert port._bus.goal_writes[-1]["shoulder_lift"] == pytest.approx(
        LIMITS_DEG["shoulder_lift"][0]
    )


def test_hold_current_position_resends_the_last_read_pose() -> None:
    port = enabled_port()
    state = port.read_state()

    assert port.hold_current_position() is True
    assert port._bus.goal_writes[-1] == {
        name: pytest.approx(deg)
        for name, deg in zip(JOINTS, FRAME.to_degrees(state.positions), strict=True)
    }


def test_hold_refuses_without_a_finite_pose() -> None:
    port = enabled_port()
    port._bus.silent.add("shoulder_lift")
    port.read_state()  # shoulder_lift yields nan

    assert port.hold_current_position() is False
    assert port._bus.goal_writes == []


# ── feedback freshness and faults ──────────────────────────────────────────────────
def test_sequence_advances_only_on_fresh_feedback() -> None:
    port = make_port()
    port.connect()

    first = port.read_state()
    second = port.read_state()
    assert first.sequence == second.sequence  # last_feedback_time unchanged

    port._bus.last_feedback_time["shoulder_pan"] = 2.0
    third = port.read_state()
    assert third.sequence[0] == second.sequence[0] + 1
    assert third.sequence[1:] == second.sequence[1:]


def test_a_silent_motor_reads_nan_without_faulting() -> None:
    port = make_port(silent={"gripper"})
    port.connect()
    state = port.read_state()

    assert math.isnan(state.positions[2])
    assert not state.all_finite
    assert state.lifecycle is ArmLifecycle.CONNECTED
    assert not state.has_fault


def test_a_motor_fault_moves_the_port_to_fault_with_a_reason() -> None:
    port = make_port(faulted={"shoulder_lift": "overtemp"})
    port.connect()
    state = port.read_state()

    assert state.lifecycle is ArmLifecycle.FAULT
    assert state.has_fault
    assert state.fault_bits == (0, 1, 0)
    assert "shoulder_lift" in state.fault_reason
    assert port.send_targets([0.0, -1.5, -1.0]) is False


def test_clear_faults_requeries_every_motor_and_recovers() -> None:
    port = make_port(faulted={"shoulder_lift": "overtemp"})
    port.connect()
    port.read_state()
    assert port.lifecycle is ArmLifecycle.FAULT

    port._bus.faulted.clear()
    port.clear_faults()
    assert port.lifecycle is ArmLifecycle.CONNECTED
    assert port._bus.cleared == list(JOINTS)
    assert port.read_state().fault_reason is None


def test_clear_faults_stays_in_fault_while_a_motor_still_reports_one() -> None:
    port = make_port(faulted={"shoulder_lift": "overtemp"})
    port.connect()
    port.read_state()

    with pytest.raises(RuntimeError):
        port.clear_faults()
    assert port.lifecycle is ArmLifecycle.FAULT
