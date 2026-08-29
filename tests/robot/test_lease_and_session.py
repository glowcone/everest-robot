import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from everest_robot.robot.clock import ManualClock
from everest_robot.robot.contracts import ArmLifecycle, JointLimit, RobotIdentity
from everest_robot.robot.fake_arm import FakeArm
from everest_robot.robot.lease import (
    FileLease,
    InMemoryLease,
    PostgresAdvisoryLease,
    RobotBusy,
    RobotLease,
)
from everest_robot.robot.parameters import IdentityMismatch, RobotParameters
from everest_robot.robot.session import RobotSession

JOINTS = ("shoulder_pan", "shoulder_lift", "gripper")
LIMITS = (
    JointLimit("shoulder_pan", -1.0, 1.0),
    JointLimit("shoulder_lift", -2.0, 0.5),
    JointLimit("gripper", -2.0, 0.0),
)
CALIBRATION = "cal-2026-08-20"
IDENTITY = RobotIdentity("maker-arm-02", "maker-arm-v1", CALIBRATION, JOINTS)


def parameters() -> RobotParameters:
    return RobotParameters.from_mapping(
        {
            "schema_version": 1,
            "robot": {
                "id": "maker-arm-02",
                "model": "maker-arm-v1",
                "calibration_id": CALIBRATION,
                "joint_order": list(JOINTS),
                "units": "radians",
            },
            "motion_defaults": {
                "max_velocity_rad_s": 0.5,
                "max_acceleration_rad_s2": 2.0,
                "tolerance_rad": 0.02,
                "settle_time_s": 0.2,
                "timeout_s": 10.0,
                "control_rate_hz": 50,
            },
            "named_positions": {},
            "named_transitions": {},
            "policy": {"default_controller": "vla", "fps": 10, "max_duration_s": 5.0},
            "replay": {
                "require_matching_robot_id": True,
                "require_matching_calibration_id": True,
                "safe_start_position": None,
                "max_speed_scale": 1.0,
            },
        },
        config_digest="sha256:test",
        source="test.yaml",
    )


def make_arm(identity: RobotIdentity = IDENTITY) -> FakeArm:
    return FakeArm(
        identity=identity,
        joint_limits=LIMITS,
        clock=ManualClock(),
        positions=[0.0, -1.0, -0.1],
    )


# ── leases ─────────────────────────────────────────────────────────────────────────
def test_lease_implementations_satisfy_the_protocol(tmp_path) -> None:
    assert isinstance(InMemoryLease("a"), RobotLease)
    assert isinstance(FileLease("a", tmp_path), RobotLease)
    assert isinstance(PostgresAdvisoryLease("a", "postgresql://unused"), RobotLease)


def test_a_second_in_process_claim_is_refused() -> None:
    first = InMemoryLease("maker-arm-02")
    second = InMemoryLease("maker-arm-02")

    with first, pytest.raises(RobotBusy, match="already claimed"):
        second.acquire()

    second.acquire()
    assert second.held
    second.release()


def test_a_file_lease_excludes_a_second_holder_and_frees_on_release(tmp_path) -> None:
    first = FileLease("maker-arm-02", tmp_path)
    second = FileLease("maker-arm-02", tmp_path)

    first.acquire()
    with pytest.raises(RobotBusy, match="another process"):
        second.acquire()

    first.release()
    second.acquire()
    assert second.held
    second.release()


def test_different_robots_do_not_contend(tmp_path) -> None:
    with FileLease("maker-arm-02", tmp_path), FileLease("maker-arm-03", tmp_path):
        pass


def test_a_killed_holder_does_not_leave_the_robot_claimed(tmp_path) -> None:
    """The case that matters: a worker dies mid-stage without unwinding."""

    source_root = Path(__file__).resolve().parents[2] / "src"
    script = textwrap.dedent(
        f"""
        import sys, time
        sys.path.insert(0, {str(source_root)!r})
        from everest_robot.robot.lease import FileLease
        lease = FileLease("maker-arm-02", {str(tmp_path)!r})
        lease.acquire()
        print("held", flush=True)
        time.sleep(30)
        """
    )
    holder = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "held"

        contender = FileLease("maker-arm-02", tmp_path)
        with pytest.raises(RobotBusy):
            contender.acquire()

        holder.kill()
        holder.wait(timeout=10)

        # The kernel drops the flock with the process; no cleanup ran.
        contender.acquire()
        assert contender.held
        contender.release()
    finally:
        holder.kill()


def test_the_postgres_lock_key_is_stable_and_namespaced() -> None:
    lease = PostgresAdvisoryLease("maker-arm-02", "postgresql://unused")

    assert lease.lock_key == PostgresAdvisoryLease("maker-arm-02", "x").lock_key
    assert lease.lock_key != PostgresAdvisoryLease("maker-arm-03", "x").lock_key


# ── sessions ───────────────────────────────────────────────────────────────────────
def test_a_session_claims_connects_and_releases() -> None:
    arm = make_arm()
    lease = InMemoryLease("maker-arm-02")

    with RobotSession(arm, parameters(), lease=lease) as session:
        assert session.is_open
        assert lease.held
        assert arm.lifecycle is ArmLifecycle.CONNECTED
        assert session.snapshot().names == JOINTS

    assert not lease.held
    assert arm.lifecycle is ArmLifecycle.DISCONNECTED


def test_a_session_leaves_the_arm_safe_even_when_the_body_raises() -> None:
    arm = make_arm()
    lease = InMemoryLease("maker-arm-02")

    with (
        pytest.raises(RuntimeError, match="stage failed"),
        RobotSession(arm, parameters(), lease=lease) as session,
    ):
        session.port.enable()
        session.port.send_targets([0.5, -0.5, -0.1])
        raise RuntimeError("stage failed")

    assert arm.lifecycle is ArmLifecycle.DISCONNECTED
    assert not lease.held


def test_the_wrong_arm_ends_the_session_and_frees_the_claim() -> None:
    arm = make_arm(RobotIdentity("maker-arm-02", "maker-arm-v1", "recalibrated", JOINTS))
    lease = InMemoryLease("maker-arm-02")

    with pytest.raises(IdentityMismatch, match="calibration_id"):
        RobotSession(arm, parameters(), lease=lease).open()

    # A refused session must not hold the robot hostage.
    assert not lease.held
    assert arm.lifecycle is ArmLifecycle.DISCONNECTED


def test_a_busy_robot_refuses_a_second_session() -> None:
    first_arm, second_arm = make_arm(), make_arm()

    with (
        RobotSession(first_arm, parameters(), lease=InMemoryLease("maker-arm-02")),
        pytest.raises(RobotBusy),
    ):
        RobotSession(second_arm, parameters(), lease=InMemoryLease("maker-arm-02")).open()

    assert second_arm.lifecycle is ArmLifecycle.DISCONNECTED


def test_reconnect_keeps_the_claim_and_rechecks_identity() -> None:
    arm = make_arm()
    lease = InMemoryLease("maker-arm-02")

    with RobotSession(arm, parameters(), lease=lease) as session:
        session.port.enable()
        session.reconnect()

        assert lease.held
        assert arm.lifecycle is ArmLifecycle.CONNECTED


def test_the_controllers_are_only_available_while_open() -> None:
    session = RobotSession(make_arm(), parameters(), lease=InMemoryLease("maker-arm-02"))

    with pytest.raises(RuntimeError, match="not open"):
        session.motion  # noqa: B018

    with session:
        assert session.motion is session.motion
        assert session.policy is session.policy
        assert session.motion.clock is session.clock
