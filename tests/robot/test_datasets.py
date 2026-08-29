"""Reader tests against a genuine LeRobot v3 fixture.

The fixture under tests/fixtures/lerobot_v3_mini is a 12-frame slice of the real target
dataset (h8i76dfsd9/test1_20260829_130743 @ 55e5611), with the video features removed. Its
values, feature names, dtypes and file layout are the dataset's own, so these tests
exercise the real format offline.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from everest_robot.robot.datasets import (
    DatasetSnapshot,
    HuggingFaceDatasetResolver,
    LeRobotV3Reader,
)
from everest_robot.robot.errors import DatasetCompatibilityError, DatasetResolutionError

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "lerobot_v3_mini"
REVISION = "55e561161026be306d06354a8941c4431e8e805f"
JOINTS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_yaw.pos",
    "wrist_roll.pos",
    "gripper.pos",
)


def snapshot(root: Path = FIXTURE) -> DatasetSnapshot:
    return DatasetSnapshot("h8i76dfsd9/test1_20260829_130743", REVISION, root)


def reader(root: Path = FIXTURE) -> LeRobotV3Reader:
    return LeRobotV3Reader(snapshot(root))


def mutated(tmp_path: Path, mutate) -> Path:
    """A copy of the fixture with meta/info.json edited."""

    root = tmp_path / "dataset"
    shutil.copytree(FIXTURE, root)
    info = json.loads((root / "meta/info.json").read_text())
    mutate(info)
    (root / "meta/info.json").write_text(json.dumps(info))
    return root


# ── revision pinning ───────────────────────────────────────────────────────────────
def test_a_moving_revision_is_refused_in_production() -> None:
    resolver = HuggingFaceDatasetResolver()

    with pytest.raises(DatasetResolutionError, match="immutable revision"):
        resolver.validate("h8i76dfsd9/test1_20260829_130743", "main")
    with pytest.raises(DatasetResolutionError, match="immutable revision"):
        resolver.validate("h8i76dfsd9/test1_20260829_130743", REVISION[:8])

    resolver.validate("h8i76dfsd9/test1_20260829_130743", REVISION)


def test_a_short_revision_is_allowed_only_when_pinning_is_disabled() -> None:
    HuggingFaceDatasetResolver(require_full_revision=False).validate("a/b", "main")


def test_a_malformed_repo_id_is_refused() -> None:
    resolver = HuggingFaceDatasetResolver()

    for repo_id in ("", "no-namespace", "a/b/c", "/leading"):
        with pytest.raises(DatasetResolutionError, match="repo id"):
            resolver.validate(repo_id, REVISION)


def test_a_missing_revision_is_refused() -> None:
    with pytest.raises(DatasetResolutionError, match="revision is required"):
        HuggingFaceDatasetResolver().validate("a/b", "")


def test_resolution_failures_never_echo_the_underlying_error() -> None:
    # Hub errors can quote request headers, and those carry the token.
    resolver = HuggingFaceDatasetResolver(allow_download=False, cache_dir="/nonexistent")

    with pytest.raises(DatasetResolutionError) as error:
        resolver.resolve("h8i76dfsd9/test1_20260829_130743", REVISION)

    assert "HF_TOKEN" in str(error.value)
    assert "could not resolve" in str(error.value)


# ── manifest ───────────────────────────────────────────────────────────────────────
def test_the_fixture_reports_the_datasets_own_metadata() -> None:
    source = reader()

    assert source.codebase_version == "v3.0"
    assert source.robot_type == "maker_follower"
    assert source.fps == 30
    assert source.joint_names() == JOINTS


def test_a_missing_snapshot_file_is_a_resolution_failure(tmp_path: Path) -> None:
    with pytest.raises(DatasetResolutionError, match="missing meta/info.json"):
        reader(tmp_path).joint_names()


def test_action_and_state_features_must_name_the_same_joints(tmp_path: Path) -> None:
    def swap(info: dict) -> None:
        info["features"]["observation.state"]["names"] = list(reversed(JOINTS))

    with pytest.raises(DatasetCompatibilityError, match="name different joints"):
        reader(mutated(tmp_path, swap)).joint_names()


def test_duplicate_joint_names_are_refused(tmp_path: Path) -> None:
    def duplicate(info: dict) -> None:
        names = list(JOINTS)
        names[1] = names[0]
        info["features"]["action"]["names"] = names

    with pytest.raises(DatasetCompatibilityError, match="repeats joint"):
        reader(mutated(tmp_path, duplicate)).joint_names()


def test_a_shape_that_disagrees_with_the_names_is_refused(tmp_path: Path) -> None:
    def shrink(info: dict) -> None:
        info["features"]["action"]["shape"] = [6]

    with pytest.raises(DatasetCompatibilityError, match="shape"):
        reader(mutated(tmp_path, shrink)).joint_names()


def test_a_non_float_action_dtype_is_refused(tmp_path: Path) -> None:
    def retype(info: dict) -> None:
        info["features"]["action"]["dtype"] = "int64"

    with pytest.raises(DatasetCompatibilityError, match="dtype"):
        reader(mutated(tmp_path, retype)).joint_names()


def test_a_missing_action_feature_is_refused(tmp_path: Path) -> None:
    def drop(info: dict) -> None:
        del info["features"]["action"]

    with pytest.raises(DatasetCompatibilityError, match="'action' is missing"):
        reader(mutated(tmp_path, drop)).joint_names()


# ── episodes ───────────────────────────────────────────────────────────────────────
def test_episode_frames_carry_named_actions_and_states() -> None:
    episode = reader().read_episode(0)

    assert len(episode) == 12
    assert episode.joint_names == JOINTS
    assert episode.actions.shape == (12, 7)
    assert episode.metadata.task == "pick up the carabiner and hook into rope"
    assert episode.metadata.fps == 30

    # The real first frame of the real episode, rebuilt by name rather than by position.
    action = episode.named_action(0)
    assert set(action) == set(JOINTS)
    assert action["shoulder_pan.pos"] == pytest.approx(4.34, abs=0.01)
    assert action["gripper.pos"] == pytest.approx(-28.167, abs=0.01)
    assert episode.named_state(0)["shoulder_pan.pos"] == pytest.approx(4.231, abs=0.01)


def test_each_episode_is_sliced_by_episode_index_not_by_offset() -> None:
    first = reader().read_episode(0)
    second = reader().read_episode(1)

    # Both episodes live in one data file; frame_index restarts per episode.
    assert np.array_equal(first.frame_indices, np.arange(12))
    assert np.array_equal(second.frame_indices, np.arange(12))
    assert not np.array_equal(first.actions, second.actions)


def test_timestamps_advance_at_the_declared_rate() -> None:
    episode = reader().read_episode(0)

    steps = np.diff(episode.timestamps)

    assert np.all(steps > 0)
    assert steps.mean() == pytest.approx(1 / 30, abs=1e-4)


def test_an_episode_outside_the_dataset_is_refused() -> None:
    with pytest.raises(DatasetCompatibilityError, match="episode 7 is not in this dataset"):
        reader().read_episode(7)


def test_a_length_disagreement_between_index_and_data_is_refused(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path / "dataset"
    shutil.copytree(FIXTURE, root)
    path = root / "meta/episodes/chunk-000/file-000.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0]["length"] = 99
    pq.write_table(pa.Table.from_pylist(rows), path)

    with pytest.raises(DatasetCompatibilityError, match="index declares 99 frames"):
        reader(root).read_episode(0)


def test_an_episode_count_disagreement_is_refused(tmp_path: Path) -> None:
    def inflate(info: dict) -> None:
        info["total_episodes"] = 5

    with pytest.raises(DatasetCompatibilityError, match="episode index lists 2 episodes"):
        reader(mutated(tmp_path, inflate)).read_episode(0)
