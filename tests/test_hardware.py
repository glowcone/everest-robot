import math
from enum import Enum

import pytest

from everest_robot.hardware import GRIPPER_INDEX, MakerArmMapping, MakerModArm
from everest_robot.kinematics import JointAngles

MAPPING = MakerArmMapping(
    ik_motor_indices=(0, 1, 2, 4),
    ik_signs=(1.0, -1.0, 1.0, 1.0),
    ik_zeros=(0.0, 0.5, 5.9, 2.1),
    neutral_joints=(0.0, 0.0, 5.9, 0.0, 2.1, 3.6, -1.0),
    gripper_open=-2.0,
    gripper_closed=-0.1,
    settle_tolerance=0.05,
    settle_timeout=1.0,
    gripper_dwell=0.0,
)


class FakeState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    ENABLED = "enabled"


class FakeSdkArm:
    """Mimics maker_arm.Arm: non-blocking targets that settle instantly."""

    def __init__(self, settles: bool = True) -> None:
        self.state = FakeState.DISCONNECTED
        self.settles = settles
        self.positions = [math.nan] * 7
        self.commanded = [math.nan] * 7
        self.target_history: list[list[float]] = []
        self.connect_calls = 0
        self.enable_calls = 0

    def connect(self) -> None:
        self.state = FakeState.CONNECTED
        self.connect_calls += 1

    def enable(self) -> None:
        self.state = FakeState.ENABLED
        self.enable_calls += 1

    def set_joint_targets(self, targets: list[float]) -> bool:
        if self.state is not FakeState.ENABLED:
            return False
        self.target_history.append(list(targets))
        self.commanded = list(targets)
        if self.settles:
            self.positions = list(targets)
        return True

    def get_joint_positions(self) -> list[float]:
        return list(self.positions)

    def get_commanded_positions(self) -> list[float]:
        return list(self.commanded)


def test_move_maps_ik_angles_onto_motors() -> None:
    sdk = FakeSdkArm()
    arm = MakerModArm(sdk, MAPPING)

    arm.move_to_joints(JointAngles(base_yaw=0.3, shoulder=0.8, elbow=1.2, wrist=-0.4))

    assert sdk.connect_calls == 1
    assert sdk.enable_calls == 1
    targets = sdk.target_history[-1]
    assert targets[0] == pytest.approx(0.0 + 0.3)
    assert targets[1] == pytest.approx(0.5 - 0.8)
    assert targets[2] == pytest.approx(5.9 + 1.2)
    assert targets[4] == pytest.approx(2.1 - 0.4)
    # Unmapped motors hold the neutral pose on the first command.
    assert targets[3] == pytest.approx(MAPPING.neutral_joints[3])
    assert targets[5] == pytest.approx(MAPPING.neutral_joints[5])
    assert targets[GRIPPER_INDEX] == pytest.approx(MAPPING.neutral_joints[GRIPPER_INDEX])


def test_gripper_commands_only_change_gripper_joint() -> None:
    sdk = FakeSdkArm()
    arm = MakerModArm(sdk, MAPPING)
    arm.move_to_joints(JointAngles(base_yaw=0.3, shoulder=0.8, elbow=1.2, wrist=-0.4))
    arm_pose = sdk.target_history[-1]

    arm.open_gripper()
    assert sdk.target_history[-1][GRIPPER_INDEX] == pytest.approx(MAPPING.gripper_open)
    assert sdk.target_history[-1][:GRIPPER_INDEX] == pytest.approx(arm_pose[:GRIPPER_INDEX])

    arm.close_gripper()
    assert sdk.target_history[-1][GRIPPER_INDEX] == pytest.approx(MAPPING.gripper_closed)
    assert sdk.target_history[-1][:GRIPPER_INDEX] == pytest.approx(arm_pose[:GRIPPER_INDEX])


def test_connect_and_enable_happen_once() -> None:
    sdk = FakeSdkArm()
    arm = MakerModArm(sdk, MAPPING)

    arm.move_to_joints(JointAngles(0.1, 0.9, 1.0, -0.2))
    arm.open_gripper()
    arm.move_to_joints(JointAngles(0.2, 0.9, 1.0, -0.2))

    assert sdk.connect_calls == 1
    assert sdk.enable_calls == 1


def test_move_raises_when_arm_never_settles() -> None:
    sdk = FakeSdkArm(settles=False)
    arm = MakerModArm(
        sdk,
        MakerArmMapping(
            ik_motor_indices=MAPPING.ik_motor_indices,
            ik_signs=MAPPING.ik_signs,
            ik_zeros=MAPPING.ik_zeros,
            neutral_joints=MAPPING.neutral_joints,
            gripper_open=MAPPING.gripper_open,
            gripper_closed=MAPPING.gripper_closed,
            settle_tolerance=0.05,
            settle_timeout=0.1,
            gripper_dwell=0.0,
        ),
    )

    with pytest.raises(RuntimeError, match="did not settle"):
        arm.move_to_joints(JointAngles(0.1, 0.9, 1.0, -0.2))


def test_mapping_from_env_validates_lengths(monkeypatch) -> None:
    monkeypatch.setenv("ROBOT_ARM_NEUTRAL_JOINTS", "0,0,0")
    with pytest.raises(ValueError, match="all 7 joint values"):
        MakerArmMapping.from_env()


def test_mapping_from_env_rejects_gripper_in_ik_map(monkeypatch) -> None:
    monkeypatch.setenv("ROBOT_IK_MOTOR_INDICES", "0,1,2,6")
    with pytest.raises(ValueError, match="motors 0-5"):
        MakerArmMapping.from_env()
