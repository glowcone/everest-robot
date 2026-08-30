"""The feature mapping a checkpoint is loaded under, and every way it can be wrong.

These run without torch or LeRobot: resolution is pure JSON reading and cross-checking, and
that is deliberate -- it is the half that decides which joint goes in which tensor slot, and
it is checked before the robot is claimed.
"""

from __future__ import annotations

import json

import pytest

from everest_robot.robot.checkpoints import CheckpointError, resolve_checkpoint

JOINTS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_yaw.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def dataset_info(joints=None, cameras=("front", "wrist"), fps=30):
    joints = list(JOINTS if joints is None else joints)
    features = {
        "action": {"dtype": "float32", "shape": [len(joints)], "names": joints},
        "observation.state": {"dtype": "float32", "shape": [len(joints)], "names": joints},
    }
    for name in cameras:
        features[f"observation.images.{name}"] = {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
        }
    return {
        "codebase_version": "v3.0",
        "fps": fps,
        "robot_type": "maker_follower",
        "features": features,
    }


def checkpoint_config(joints=None, cameras=("front", "wrist")):
    count = len(JOINTS if joints is None else joints)
    inputs: dict = {"observation.state": {"type": "STATE", "shape": [count]}}
    for name in cameras:
        inputs[f"observation.images.{name}"] = {"type": "VISUAL", "shape": [3, 480, 640]}
    return {
        "type": "act",
        "input_features": inputs,
        "output_features": {"action": {"type": "ACTION", "shape": [count]}},
    }


@pytest.fixture
def checkpoint(tmp_path):
    """A local checkpoint directory whose training dataset is stubbed out."""

    root = tmp_path / "act"
    root.mkdir()
    (root / "config.json").write_text(json.dumps(checkpoint_config()))
    (root / "train_config.json").write_text(json.dumps({"dataset": {"repo_id": "ns/teleop"}}))
    return root


@pytest.fixture
def dataset(monkeypatch, tmp_path):
    """Serve ``meta/info.json`` from disk instead of the hub."""

    def serve(document) -> None:
        path = tmp_path / "snapshots" / "abc123def456" / "meta"
        path.mkdir(parents=True, exist_ok=True)
        (path / "info.json").write_text(json.dumps(document))
        monkeypatch.setattr(
            "huggingface_hub.hf_hub_download", lambda *a, **k: str(path / "info.json")
        )

    return serve


def test_the_joint_order_comes_from_the_training_dataset(checkpoint, dataset) -> None:
    dataset(dataset_info())

    resolved = resolve_checkpoint(checkpoint)

    assert resolved.policy_type == "act"
    assert resolved.features.action_names == tuple(JOINTS)
    assert resolved.features.state_names == tuple(JOINTS)
    assert resolved.features.fps == 30.0
    assert resolved.features.robot_type == "maker_follower"
    # The pinned commit the metadata actually came from, not the branch that was asked for.
    assert resolved.features.dataset_revision == "abc123def456"


def test_the_policys_inputs_are_expressed_the_way_the_robot_names_them(checkpoint, dataset) -> None:
    """This is what `compatibility_problems` compares against the connected arm."""

    dataset(dataset_info())

    features = resolve_checkpoint(checkpoint).features.input_features

    assert features["shoulder_pan.pos"] == ()
    assert features["front"] == (480, 640, 3)
    assert features["wrist"] == (480, 640, 3)
    assert "observation.state" not in features


def test_the_frame_builder_gets_shapes_it_can_pack(checkpoint, dataset) -> None:
    """LeRobot's `build_dataset_frame` branches on a 1-D tuple shape, not a JSON list."""

    dataset(dataset_info())

    ds_features = resolve_checkpoint(checkpoint).features.ds_features

    assert ds_features["observation.state"]["shape"] == (7,)
    assert ds_features["action"]["names"] == JOINTS
    assert ds_features["observation.images.front"]["shape"] == (480, 640, 3)


def test_a_dataset_that_names_different_joints_for_state_and_action_is_refused(
    checkpoint, dataset, tmp_path
) -> None:
    info = dataset_info()
    info["features"]["observation.state"]["names"] = list(reversed(JOINTS))
    dataset(info)

    with pytest.raises(CheckpointError, match="name different joints"):
        resolve_checkpoint(checkpoint)


def test_a_joint_count_the_checkpoint_disagrees_with_is_refused(checkpoint, dataset) -> None:
    """Six names against a seven-wide tensor would pack every joint one slot over."""

    dataset(dataset_info(joints=JOINTS[:6]))

    with pytest.raises(CheckpointError, match="must agree"):
        resolve_checkpoint(checkpoint)


def test_a_camera_the_dataset_does_not_describe_is_refused(checkpoint, dataset) -> None:
    dataset(dataset_info(cameras=("front",)))

    with pytest.raises(CheckpointError, match="observation.images.wrist"):
        resolve_checkpoint(checkpoint)


def test_a_camera_resolution_mismatch_is_refused(checkpoint, dataset) -> None:
    info = dataset_info()
    info["features"]["observation.images.wrist"]["shape"] = [240, 320, 3]
    dataset(info)

    with pytest.raises(CheckpointError, match="resolution does not match"):
        resolve_checkpoint(checkpoint)


def test_a_repeated_joint_name_is_refused(checkpoint, dataset) -> None:
    dataset(dataset_info(joints=[*JOINTS[:5], "gripper.pos", "gripper.pos"]))

    with pytest.raises(CheckpointError, match="repeats joint"):
        resolve_checkpoint(checkpoint)


def test_a_checkpoint_without_a_training_dataset_refuses_rather_than_guessing(
    tmp_path, dataset
) -> None:
    """No recorded dataset means no authoritative joint order. Nothing is invented."""

    root = tmp_path / "orphan"
    root.mkdir()
    (root / "config.json").write_text(json.dumps(checkpoint_config()))
    dataset(dataset_info())

    with pytest.raises(CheckpointError, match="training dataset is unknown"):
        resolve_checkpoint(root)


def test_an_explicit_training_dataset_can_stand_in_for_a_missing_train_config(
    tmp_path, dataset
) -> None:
    root = tmp_path / "orphan"
    root.mkdir()
    (root / "config.json").write_text(json.dumps(checkpoint_config()))
    dataset(dataset_info())

    resolved = resolve_checkpoint(root, dataset_repo_id="ns/teleop")

    assert resolved.features.action_names == tuple(JOINTS)


def test_a_reference_that_is_neither_a_directory_nor_a_repo_id_is_refused(tmp_path) -> None:
    with pytest.raises(CheckpointError, match="neither an existing directory"):
        resolve_checkpoint(tmp_path / "nope")


def test_a_weights_file_is_not_a_checkpoint(tmp_path) -> None:
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"")

    with pytest.raises(CheckpointError, match="is a directory"):
        resolve_checkpoint(path)
