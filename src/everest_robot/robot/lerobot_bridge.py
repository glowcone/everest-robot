"""LeRobot's ``Robot`` contract over the Everest arm port.

This is the adapter the ADR commits to: the arm keeps running maker-arm-sdk's private
protocol, and LeRobot's policy, processor, dataset and replay machinery talks to it through
its normal interface.

The logic lives in :class:`RobotBridgeCore`, which has no LeRobot import and is what the
tests exercise. :func:`make_lerobot_robot` builds the actual ``Robot`` subclass on demand,
because subclassing requires importing LeRobot (and therefore torch), which the core
package does not depend on.

**Units.** ``MakerFollower`` normalizes every motor in degrees and names features
``{joint}.pos``; ``maker_arm`` works in radians in calibrated joint coordinates. The bridge
keeps the feature names and the degree convention so datasets and checkpoints line up, and
converts at this boundary. The zero pose is a separate question from the scale: with the
shipped hardware profile every joint has ``direction: 1`` and ``offset: 0``, so joint and
motor coordinates coincide, but that is a property of the profile, not a guarantee. Any
policy or dataset produced against ``MakerFollower`` must have its frame checked and
recorded in :class:`JointFrame` before it is reused here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from everest_robot.robot.cameras import CameraRuntime
from everest_robot.robot.contracts import ArmLifecycle, JointCommand
from everest_robot.robot.ports import ArmPort, clip_to_limits

# LeRobot's per-motor feature suffix. Kept identical to MakerFollower's so a checkpoint or
# dataset recorded against either driver has matching keys.
POSITION_SUFFIX = ".pos"

_RAD_TO_DEG = 180.0 / math.pi

# Built once and cached: registering the config subclass twice would collide in LeRobot's
# draccus choice registry.
_LEROBOT_CLASSES: tuple[Any, Any] | None = None


@dataclass(frozen=True, slots=True)
class JointFrame:
    """Conversion between Everest joint radians and the LeRobot feature convention.

    ``offset_deg`` is per joint and defaults to zero, which asserts that the two drivers
    agree on the zero pose. Where they do not, the measured offsets belong here and in the
    run's stored metadata -- never applied ad hoc at a call site.
    """

    joint_names: tuple[str, ...]
    offsets_deg: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.offsets_deg:
            object.__setattr__(self, "offsets_deg", tuple(0.0 for _ in self.joint_names))
        if len(self.offsets_deg) != len(self.joint_names):
            raise ValueError("offsets_deg must cover every joint")

    @property
    def is_identity(self) -> bool:
        """True when the frames differ only by the radian-to-degree scale."""

        return not any(self.offsets_deg)

    def to_degrees(self, radians: Sequence[float]) -> tuple[float, ...]:
        return tuple(
            value * _RAD_TO_DEG + offset
            for value, offset in zip(radians, self.offsets_deg, strict=True)
        )

    def to_radians(self, degrees: Sequence[float]) -> tuple[float, ...]:
        return tuple(
            (value - offset) / _RAD_TO_DEG
            for value, offset in zip(degrees, self.offsets_deg, strict=True)
        )


class RobotBridgeCore:
    """The observation/action translation, with no LeRobot dependency.

    Actions arriving from a policy are clipped into the driver's soft limits and the
    clipped action is what gets returned, matching LeRobot's contract that ``send_action``
    reports what was actually sent. Clipping is also recorded so a rollout that spent its
    time against a limit says so in its durable result.
    """

    def __init__(
        self,
        port: ArmPort,
        *,
        cameras: CameraRuntime | None = None,
        frame: JointFrame | None = None,
    ) -> None:
        self.port = port
        self.cameras = cameras or CameraRuntime({})
        self.frame = frame or JointFrame(port.joint_names)
        if self.frame.joint_names != port.joint_names:
            raise ValueError("the joint frame must describe the port's joints, in the same order")
        self.clipped_joints: set[str] = set()

    # ── features ───────────────────────────────────────────────────────────────────
    @property
    def motor_features(self) -> dict[str, type]:
        return {f"{name}{POSITION_SUFFIX}": float for name in self.port.joint_names}

    @property
    def observation_features(self) -> dict[str, Any]:
        return {**self.motor_features, **self.cameras.features()}

    @property
    def action_features(self) -> dict[str, type]:
        return dict(self.motor_features)

    @property
    def is_connected(self) -> bool:
        return (
            self.port.lifecycle is not ArmLifecycle.DISCONNECTED
            and (not self.cameras.names or self.cameras.is_connected)
        )

    # ── lifecycle ──────────────────────────────────────────────────────────────────
    def connect(self) -> None:
        self.port.connect()
        try:
            self.cameras.connect()
        except Exception:
            # Never leave the bus claimed because a camera failed.
            self.port.disconnect()
            raise

    def disconnect(self) -> None:
        try:
            self.cameras.disconnect()
        finally:
            self.port.disconnect()

    # ── observations and actions ───────────────────────────────────────────────────
    def get_observation(self) -> dict[str, Any]:
        state = self.port.read_state()
        degrees = self.frame.to_degrees(state.positions)
        observation: dict[str, Any] = {
            f"{name}{POSITION_SUFFIX}": value
            for name, value in zip(self.port.joint_names, degrees, strict=True)
        }
        observation.update(self.cameras.observation())
        return observation

    def parse_action(self, action: Mapping[str, Any]) -> JointCommand:
        """Validate a LeRobot action and turn it into a clipped joint-space command.

        A missing joint is an error rather than a hold: a policy whose action space does
        not cover the arm is the wrong policy for this robot, and the policy runner
        validates that before any motion. Extra keys are an error for the same reason.
        """

        expected = list(self.action_features)
        missing = [key for key in expected if key not in action]
        if missing:
            raise KeyError(f"action is missing {', '.join(missing)}")
        unexpected = sorted(set(action) - set(expected))
        if unexpected:
            raise KeyError(f"action carries unexpected key(s) {', '.join(unexpected)}")

        degrees = []
        for key in expected:
            value = float(action[key])
            if not math.isfinite(value):
                raise ValueError(f"action[{key!r}] must be finite, got {action[key]!r}")
            degrees.append(value)

        command = clip_to_limits(self.frame.to_radians(degrees), self.port.limits())
        self.clipped_joints.update(command.clipped_joints)
        return command

    def send_action(self, action: Mapping[str, Any]) -> dict[str, float]:
        """Send a policy action and return what was actually commanded, in degrees."""

        command = self.parse_action(action)
        if not self.port.send_targets(command.targets):
            raise RuntimeError(
                f"the driver refused a joint command (arm is {self.port.lifecycle})"
            )
        return {
            f"{name}{POSITION_SUFFIX}": value
            for name, value in zip(
                self.port.joint_names, self.frame.to_degrees(command.targets), strict=True
            )
        }


def make_lerobot_robot(
    core: RobotBridgeCore,
    *,
    robot_id: str | None = None,
    calibration_dir: Any = None,
) -> Any:
    """Wrap a core in a real LeRobot ``Robot``.

    Built at call time: subclassing ``Robot`` imports LeRobot, which the core package does
    not depend on. Calibration is a no-op here on purpose -- it belongs to maker-arm-sdk's
    hardware profile, and letting LeRobot write a second calibration file would create the
    divergent safety story the ADR exists to prevent.
    """

    config_class, robot_class = _lerobot_classes()
    identity = core.port.identity
    config = config_class(id=robot_id or identity.robot_id, calibration_dir=calibration_dir)
    return robot_class(config, core)


def _lerobot_classes() -> tuple[Any, Any]:
    global _LEROBOT_CLASSES
    if _LEROBOT_CLASSES is not None:
        return _LEROBOT_CLASSES

    from dataclasses import dataclass as _dataclass

    from lerobot.robots.config import RobotConfig
    from lerobot.robots.robot import Robot

    @RobotConfig.register_subclass("everest_maker_arm")
    @_dataclass(kw_only=True)
    class EverestMakerArmConfig(RobotConfig):
        pass

    class EverestMakerArm(Robot):
        """LeRobot-facing view of an Everest-owned arm."""

        config_class = EverestMakerArmConfig
        name = "everest_maker_arm"

        def __init__(self, config: Any, core: RobotBridgeCore) -> None:
            super().__init__(config)
            self.core = core

        @property
        def observation_features(self) -> dict:
            return self.core.observation_features

        @property
        def action_features(self) -> dict:
            return self.core.action_features

        @property
        def is_connected(self) -> bool:
            return self.core.is_connected

        def connect(self, calibrate: bool = True) -> None:
            self.core.connect()

        @property
        def is_calibrated(self) -> bool:
            return True

        def calibrate(self) -> None:
            return None

        def configure(self) -> None:
            return None

        def get_observation(self) -> dict:
            return self.core.get_observation()

        def send_action(self, action: dict) -> dict:
            return self.core.send_action(action)

        def disconnect(self) -> None:
            self.core.disconnect()

    _LEROBOT_CLASSES = (EverestMakerArmConfig, EverestMakerArm)
    return _LEROBOT_CLASSES
