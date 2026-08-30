"""Passive admission checks for the attachment FSM's INITIAL state."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from everest_robot.robot.clock import Clock, SystemClock
from everest_robot.robot.contracts import ArmLifecycle
from everest_robot.robot.ports import ArmPort, violations


@dataclass(frozen=True, slots=True)
class InitialReadinessReport:
    """Evidence gathered without enabling or commanding the arm."""

    ready: bool
    samples: int
    positions: tuple[float, ...]
    max_velocity_rad_s: float
    neutral_confirmed: bool | None
    neutral_error_rad: float | None
    camera_shapes: Mapping[str, tuple[int, ...]]
    problems: tuple[str, ...]


class InitialReadinessChecker:
    """Require fresh, finite, stationary feedback and live cameras while torque is off."""

    def __init__(
        self,
        port: ArmPort,
        *,
        camera_observation: Callable[[], Mapping[str, np.ndarray]] | None = None,
        expected_camera_shapes: Mapping[str, tuple[int, int, int]] | None = None,
        neutral_position: Sequence[float] | None = None,
        neutral_tolerance_rad: float = 0.05,
        sample_count: int = 3,
        sample_interval_s: float = 0.03,
        max_stationary_velocity_rad_s: float = 0.05,
        clock: Clock | None = None,
    ) -> None:
        if sample_count < 2:
            raise ValueError("sample_count must be at least 2 to prove feedback freshness")
        if sample_interval_s <= 0 or max_stationary_velocity_rad_s < 0:
            raise ValueError("readiness timing and velocity bounds must be non-negative")
        if neutral_tolerance_rad <= 0:
            raise ValueError("neutral_tolerance_rad must be positive")
        self.port = port
        self.camera_observation = camera_observation
        self.expected_camera_shapes = dict(expected_camera_shapes or {})
        self.neutral_position = (
            tuple(float(value) for value in neutral_position)
            if neutral_position is not None
            else None
        )
        self.neutral_tolerance_rad = neutral_tolerance_rad
        self.sample_count = sample_count
        self.sample_interval_s = sample_interval_s
        self.max_stationary_velocity_rad_s = max_stationary_velocity_rad_s
        self.clock = clock or SystemClock()

    def check(self) -> InitialReadinessReport:
        problems: list[str] = []
        if self.port.lifecycle is not ArmLifecycle.CONNECTED:
            problems.append(
                f"arm must be connected with torque off, got {self.port.lifecycle.value}"
            )

        states = []
        for index in range(self.sample_count):
            if index:
                self.clock.sleep(self.sample_interval_s)
            states.append(self.port.read_state())

        for index, state in enumerate(states):
            if state.has_fault:
                problems.append(f"feedback sample {index + 1} reports a fault")
            numeric = (*state.positions, *state.velocities, *state.torques, *state.temperatures)
            if not all(math.isfinite(value) for value in numeric):
                problems.append(f"feedback sample {index + 1} contains non-finite values")
            outside = violations(state.positions, self.port.limits())
            if outside:
                problems.append(
                    f"feedback sample {index + 1} is outside limits: {', '.join(outside)}"
                )

        for earlier, later in zip(states[:-1], states[1:], strict=True):
            stale = [
                name
                for name, before, after in zip(
                    self.port.joint_names, earlier.sequence, later.sequence, strict=True
                )
                if after <= before
            ]
            if stale:
                problems.append(f"feedback did not advance for: {', '.join(stale)}")

        last = states[-1]
        max_velocity = max((abs(value) for value in last.velocities), default=0.0)
        if math.isfinite(max_velocity) and max_velocity > self.max_stationary_velocity_rad_s:
            problems.append(
                f"arm is moving at {max_velocity:.3f} rad/s; "
                f"limit is {self.max_stationary_velocity_rad_s:.3f} rad/s"
            )

        camera_shapes: dict[str, tuple[int, ...]] = {}
        if self.camera_observation is not None:
            try:
                frames = self.camera_observation()
                camera_shapes = {name: tuple(frame.shape) for name, frame in frames.items()}
            except Exception as error:  # noqa: BLE001 - readiness reports device failures
                problems.append(f"camera observation failed: {error}")
            else:
                missing = sorted(set(self.expected_camera_shapes) - set(frames))
                if missing:
                    problems.append(f"missing camera frames: {', '.join(missing)}")
                for name, expected in self.expected_camera_shapes.items():
                    actual = camera_shapes.get(name)
                    if actual is not None and actual != expected:
                        problems.append(
                            f"camera {name!r} shape is {actual}, expected {expected}"
                        )

        neutral_confirmed: bool | None = None
        neutral_error: float | None = None
        if self.neutral_position is not None:
            if len(self.neutral_position) != len(last.positions):
                problems.append(
                    f"neutral pose has {len(self.neutral_position)} joints, "
                    f"expected {len(last.positions)}"
                )
            elif all(math.isfinite(value) for value in last.positions):
                neutral_error = max(
                    abs(actual - target)
                    for actual, target in zip(
                        last.positions, self.neutral_position, strict=True
                    )
                )
                neutral_confirmed = neutral_error <= self.neutral_tolerance_rad

        return InitialReadinessReport(
            ready=not problems,
            samples=len(states),
            positions=tuple(last.positions),
            max_velocity_rad_s=max_velocity,
            neutral_confirmed=neutral_confirmed,
            neutral_error_rad=neutral_error,
            camera_shapes=camera_shapes,
            problems=tuple(dict.fromkeys(problems)),
        )
