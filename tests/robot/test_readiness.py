import math

import numpy as np

from everest_robot.robot.clock import ManualClock
from everest_robot.robot.contracts import JointLimit, RobotIdentity
from everest_robot.robot.fake_arm import FakeArm
from everest_robot.robot.readiness import InitialReadinessChecker

JOINTS = ("shoulder", "elbow")
IDENTITY = RobotIdentity("arm-1", "maker-arm-v1", "cal-1", JOINTS)
LIMITS = (JointLimit("shoulder", -1.0, 1.0), JointLimit("elbow", -2.0, 0.0))


def connected_arm(clock=None, **overrides):
    arm = FakeArm(
        IDENTITY,
        LIMITS,
        positions=[0.1, -1.0],
        clock=clock or ManualClock(),
        **overrides,
    )
    arm.connect()
    return arm


def test_passive_readiness_never_enables_or_commands():
    clock = ManualClock()
    arm = connected_arm(clock)

    report = InitialReadinessChecker(arm, clock=clock).check()

    assert report.ready
    assert report.samples == 3
    assert arm.sent_commands == []
    assert arm.lifecycle.value == "connected"


def test_readiness_refuses_enabled_and_nonfinite_feedback():
    arm = connected_arm()
    arm.enable()
    arm.positions[1] = math.nan

    report = InitialReadinessChecker(arm, clock=arm.clock).check()

    assert not report.ready
    assert any("torque off" in problem for problem in report.problems)
    assert any("non-finite" in problem for problem in report.problems)


def test_readiness_checks_camera_names_and_shapes():
    arm = connected_arm()
    report = InitialReadinessChecker(
        arm,
        camera_observation=lambda: {"wrist": np.zeros((10, 12, 3), dtype=np.uint8)},
        expected_camera_shapes={"wrist": (10, 10, 3), "overhead": (10, 10, 3)},
        clock=arm.clock,
    ).check()

    assert not report.ready
    assert any("missing camera" in problem for problem in report.problems)
    assert any("shape" in problem for problem in report.problems)


def test_readiness_reports_measured_neutral_without_moving_there():
    arm = connected_arm()

    near = InitialReadinessChecker(
        arm,
        neutral_position=(0.11, -1.01),
        neutral_tolerance_rad=0.02,
        clock=arm.clock,
    ).check()
    far = InitialReadinessChecker(
        arm,
        neutral_position=(0.5, -1.0),
        neutral_tolerance_rad=0.02,
        clock=arm.clock,
    ).check()

    assert near.neutral_confirmed is True
    assert far.neutral_confirmed is False
    assert arm.sent_commands == []
