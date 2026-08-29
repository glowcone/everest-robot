"""Translation tests for the maker-arm adapter.

The stub tests always run. The tests marked with ``importorskip`` check our assumptions
against the real driver class and run only where the ``hardware`` extra is installed;
they need no CAN bus. Behaviour on a live arm belongs to the hardware acceptance suite,
not here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import pytest

from everest_robot.robot.contracts import ArmLifecycle, RobotIdentity
from everest_robot.robot.maker_arm_port import MakerArmPort

JOINTS = ("shoulder_pan", "shoulder_lift", "gripper")
IDENTITY = RobotIdentity("maker-arm-02", "maker-arm-v1", "cal-2026-08-20", JOINTS)


@dataclass
class StubJoint:
    lo: float
    hi: float


@dataclass
class StubMotor:
    feedback_sequence: int = 0


@dataclass
class StubState:
    value: str


@dataclass
class StubConfig:
    joints: list[StubJoint]


@dataclass
class StubArm:
    """The subset of ``maker_arm.Arm`` the port touches."""

    config: StubConfig
    state: StubState = field(default_factory=lambda: StubState("connected"))
    fault_reason: str | None = None
    positions: list[float] = field(default_factory=lambda: [0.1, -0.2, 0.0])
    accepted: list[list[float]] = field(default_factory=list)
    refreshed: int = 0
    motors: list[StubMotor] = field(default_factory=lambda: [StubMotor(7) for _ in range(3)])

    def refresh(self, wait: bool = False, timeout: float = 0.05) -> Any:
        self.refreshed += 1
        return None

    def get_joint_positions(self) -> list[float]:
        return list(self.positions)

    def get_joint_velocities(self) -> list[float]:
        return [0.0] * len(self.positions)

    def get_joint_torques(self) -> list[float]:
        return [0.0] * len(self.positions)

    def get_temperatures(self) -> list[float]:
        return [31.0] * len(self.positions)

    def get_faults(self) -> list[int]:
        return [0] * len(self.positions)

    def set_joint_targets(self, targets: list[float]) -> bool:
        self.accepted.append(list(targets))
        return True

    def get_commanded_positions(self) -> list[float]:
        return list(self.positions)


def make_port(**overrides: Any) -> MakerArmPort:
    arm = StubArm(
        config=StubConfig([StubJoint(-1.0, 1.0), StubJoint(-2.0, 0.5), StubJoint(-2.0, 0.0)])
    )
    for name, value in overrides.items():
        setattr(arm, name, value)
    return MakerArmPort(arm, IDENTITY, profile_digest="sha256:profile")


def test_a_joint_count_mismatch_is_refused_at_construction() -> None:
    arm = StubArm(config=StubConfig([StubJoint(-1.0, 1.0)]))

    with pytest.raises(ValueError, match="different arm"):
        MakerArmPort(arm, IDENTITY)


def test_limits_take_their_names_from_the_configured_joint_order() -> None:
    limits = make_port().limits()

    assert tuple(limit.name for limit in limits) == JOINTS
    assert (limits[1].lower_rad, limits[1].upper_rad) == (-2.0, 0.5)


def test_driver_states_map_onto_the_lifecycle_enum() -> None:
    for name, expected in [
        ("disconnected", ArmLifecycle.DISCONNECTED),
        ("connected", ArmLifecycle.CONNECTED),
        ("enabled", ArmLifecycle.ENABLED),
        ("fault", ArmLifecycle.FAULT),
    ]:
        assert make_port(state=StubState(name)).lifecycle is expected


def test_reading_while_connected_refreshes_but_reading_while_enabled_does_not() -> None:
    connected = make_port()
    connected.read_state()
    # Nothing else polls the bus until the driver's control loop is running.
    assert connected.arm.refreshed == 1

    enabled = make_port(state=StubState("enabled"))
    enabled.read_state()
    assert enabled.arm.refreshed == 0


def test_state_carries_joint_names_feedback_counters_and_fault_reason() -> None:
    port = make_port(state=StubState("fault"), fault_reason="feedback timeout")

    state = port.read_state()

    assert state.names == JOINTS
    assert state.positions == (0.1, -0.2, 0.0)
    assert state.sequence == (7, 7, 7)
    assert state.fault_reason == "feedback timeout"
    assert state.has_fault


def test_malformed_targets_never_reach_the_driver() -> None:
    port = make_port()

    assert port.send_targets([0.1, 0.2]) is False
    assert port.send_targets([0.1, math.inf, 0.0]) is False
    assert port.arm.accepted == []

    assert port.send_targets([0.1, 0.2, 0.0]) is True
    assert port.arm.accepted == [[0.1, 0.2, 0.0]]


def test_the_real_driver_exposes_everything_the_port_translates() -> None:
    maker_arm = pytest.importorskip("maker_arm")
    from maker_arm.profiles import DEFAULT_ARM_CONFIG
    from maker_arm.transport.mock import MockBackend

    arm = maker_arm.Arm(maker_arm.ArmConfig.from_yaml(str(DEFAULT_ARM_CONFIG)), MockBackend())
    identity = RobotIdentity(
        "maker-arm-02",
        "maker-arm-v1",
        "cal-2026-08-20",
        tuple(f"joint_{index}" for index in range(len(arm.config.joints))),
    )

    port = MakerArmPort(arm, identity)

    # No bus traffic is needed for any of this: it is the shape of the driver we depend on.
    assert port.lifecycle is ArmLifecycle.DISCONNECTED
    assert len(port.limits()) == len(arm.config.joints)
    assert all(limit.lower_rad < limit.upper_rad for limit in port.limits())
    assert port.read_state().names == identity.joint_names
    assert len(port.commanded_positions()) == len(arm.config.joints)
