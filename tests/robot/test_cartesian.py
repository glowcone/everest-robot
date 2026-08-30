import numpy as np
import pytest

from everest_robot.robot.cartesian import dataset_joints_to_tool_transform
from everest_robot.robot.lerobot_bridge import JointFrame


class StubKinematics:
    joint_names = ("a", "b")

    def __init__(self):
        self.received = None

    def forward(self, joints_rad):
        self.received = joints_rad
        transform = np.eye(4)
        transform[:2, 3] = joints_rad[:2]
        return transform


def test_dataset_joints_are_reconciled_before_forward_kinematics() -> None:
    model = StubKinematics()
    frame = JointFrame(("a", "b", "gripper"), offsets_deg=(10.0, -20.0, 0.0))

    transform = dataset_joints_to_tool_transform(model, frame, [100.0, 70.0, -2.0])

    assert model.received == pytest.approx((np.pi / 2, np.pi / 2))
    assert transform[:2, 3] == pytest.approx([np.pi / 2, np.pi / 2])


def test_dataset_fk_requires_every_named_joint() -> None:
    with pytest.raises(ValueError, match="expected 3 dataset joints"):
        dataset_joints_to_tool_transform(
            StubKinematics(), JointFrame(("a", "b", "gripper")), [1.0, 2.0]
        )


def test_dataset_fk_rejects_wrong_model_joint_order() -> None:
    model = StubKinematics()
    model.joint_names = ("b", "a")

    with pytest.raises(ValueError, match="joint order"):
        dataset_joints_to_tool_transform(
            model, JointFrame(("a", "b", "gripper")), [1.0, 2.0, 3.0]
        )
