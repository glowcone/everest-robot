""":class:`~everest_robot.robot.ports.ArmPort` over ``maker_arm.Arm``.

The RobStride **private**-protocol driver, with maker-arm-sdk as the hardware safety
boundary. ADR-0001 made this the production driver; ADR-0002 superseded that for
maker-arm-02, whose motors are provisioned in MIT mode (see
docs/adr/0002-mit-protocol-motor-operation.md and ``robstride_mit_port.py``). This port
remains the path for private-protocol arms. It is a translation layer and nothing more:
it must not re-implement limits, watchdogs, velocity limiting or fault handling -- doing
so would create a second, divergent safety story.

``maker_arm`` is imported lazily so the rest of the runtime installs and tests without the
``hardware`` extra.

Calibration identity is asserted by deployment configuration, not discovered: the driver's
hardware profile records motor geometry but carries no calibration name. What this module
can and does verify structurally is that the connected arm has exactly the joints the
parameters file declares, and it records the digest of the hardware profile actually loaded
so a run can be traced to it.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from everest_robot.robot.contracts import (
    ArmLifecycle,
    JointLimit,
    JointState,
    RobotIdentity,
)

_LIFECYCLE_BY_NAME = {
    "disconnected": ArmLifecycle.DISCONNECTED,
    "connected": ArmLifecycle.CONNECTED,
    "enabled": ArmLifecycle.ENABLED,
    "fault": ArmLifecycle.FAULT,
}


class HardwareUnavailableError(RuntimeError):
    """The ``hardware`` extra is not installed in this environment."""


def load_maker_arm() -> Any:
    """Import ``maker_arm`` on demand, with an actionable message when it is missing."""

    try:
        import maker_arm
    except ImportError as error:  # pragma: no cover - exercised only without the extra
        raise HardwareUnavailableError(
            "maker_arm is not installed; install the hardware extra "
            "(`uv sync --extra hardware`) to talk to a real arm"
        ) from error
    return maker_arm


class MakerArmPort:
    """Adapts ``maker_arm.Arm`` to the Everest port.

    Takes an already-constructed arm so tests and callers can supply a mock CAN backend;
    use :meth:`from_profile` for the ordinary deployment path.
    """

    def __init__(
        self,
        arm: Any,
        identity: RobotIdentity,
        *,
        profile_digest: str = "",
    ) -> None:
        joint_count = len(arm.config.joints)
        if joint_count != len(identity.joint_names):
            raise ValueError(
                f"hardware profile has {joint_count} joints but the parameters file declares "
                f"{len(identity.joint_names)} ({', '.join(identity.joint_names)}); one of them "
                "is describing a different arm"
            )
        self._arm = arm
        self._identity = identity
        self.profile_digest = profile_digest

    @classmethod
    def from_profile(
        cls,
        identity: RobotIdentity,
        *,
        config_path: str | Path | None = None,
        backend: str = "socketcan",
        **backend_kwargs: Any,
    ) -> MakerArmPort:
        """Build a port from a maker-arm hardware profile.

        ``config_path`` defaults to the profile shipped with the installed SDK, which is
        the version the driver's own limits and gains were captured against.
        """

        maker_arm = load_maker_arm()
        if config_path is None:
            from maker_arm.profiles import DEFAULT_ARM_CONFIG

            config_path = DEFAULT_ARM_CONFIG
        path = Path(config_path)
        digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        arm = maker_arm.Arm.from_yaml(str(path), backend=backend, **backend_kwargs)
        return cls(arm, identity, profile_digest=digest)

    # ── identity and limits ────────────────────────────────────────────────────────
    @property
    def arm(self) -> Any:
        """The underlying driver. For diagnostics; do not command through it."""

        return self._arm

    @property
    def identity(self) -> RobotIdentity:
        return self._identity

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self._identity.joint_names

    @property
    def lifecycle(self) -> ArmLifecycle:
        return _LIFECYCLE_BY_NAME[self._arm.state.value]

    def limits(self) -> tuple[JointLimit, ...]:
        """The driver's soft limits, named with the parameters file's joint order.

        Pairing is positional: index *i* of ``joint_order`` is index *i* of the hardware
        profile's joint list. The constructor's count check is what makes that safe.
        """

        return tuple(
            JointLimit(name=name, lower_rad=joint.lo, upper_rad=joint.hi)
            for name, joint in zip(self.joint_names, self._arm.config.joints, strict=True)
        )

    # ── lifecycle ──────────────────────────────────────────────────────────────────
    def connect(self, timeout: float = 2.0) -> None:
        self._arm.connect(timeout=timeout)

    def disconnect(self) -> None:
        self._arm.disconnect()

    def enable(self) -> None:
        self._arm.enable()

    def disable(self) -> None:
        self._arm.disable()

    def estop(self) -> None:
        self._arm.estop()

    def clear_faults(self) -> None:
        self._arm.clear_faults()

    def hold_current_position(self) -> bool:
        return bool(self._arm.hold_current_position())

    # ── feedback and commands ──────────────────────────────────────────────────────
    def read_state(self) -> JointState:
        """Sample feedback.

        While the arm is enabled its own control loop is polling the bus, so this reads the
        driver's cache. While it is merely connected, nothing else is polling, so a
        non-blocking refresh is issued first.
        """

        lifecycle = self.lifecycle
        if lifecycle is not ArmLifecycle.ENABLED:
            self._arm.refresh()
        arm = self._arm
        return JointState(
            names=self.joint_names,
            positions=tuple(arm.get_joint_positions()),
            velocities=tuple(arm.get_joint_velocities()),
            torques=tuple(arm.get_joint_torques()),
            temperatures=tuple(arm.get_temperatures()),
            fault_bits=tuple(arm.get_faults()),
            # Per-motor feedback counters: two reads with an unchanged counter mean the
            # value is cached, which is a different failure from a value that is wrong.
            sequence=tuple(motor.feedback_sequence for motor in arm.motors),
            monotonic_s=time.monotonic(),
            lifecycle=lifecycle,
            fault_reason=arm.fault_reason,
        )

    def send_targets(self, targets: Sequence[float]) -> bool:
        values = [float(target) for target in targets]
        if len(values) != len(self.joint_names) or not all(math.isfinite(v) for v in values):
            return False
        return bool(self._arm.set_joint_targets(values))

    def commanded_positions(self) -> tuple[float, ...]:
        """The rate-limited command the driver is currently sending to the motors."""

        return tuple(self._arm.get_commanded_positions())
