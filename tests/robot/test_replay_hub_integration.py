"""Integration against the real Hugging Face dataset.

Skipped unless EVEREST_HF_NETWORK_TESTS=1, because it downloads. Everything it proves
about the format is also covered offline by the fixture tests; what only this can show is
that resolution, revision pinning and offline reuse work against the real Hub.
"""

from __future__ import annotations

import os

import pytest

from everest_robot.robot.clock import ManualClock
from everest_robot.robot.datasets import HuggingFaceDatasetResolver, LeRobotV3Reader
from everest_robot.robot.errors import DatasetResolutionError
from everest_robot.robot.fake_arm import FakeArm
from everest_robot.robot.lease import InMemoryLease
from everest_robot.robot.replay import ReplayRunner

from .test_replay import (
    CALIBRATION,
    IDENTITY,
    JOINTS,
    OFFSETS,
    REPO,
    REVISION,
    SDK_LIMITS,
    parameters,
    request,
)

pytestmark = pytest.mark.skipif(
    os.getenv("EVEREST_HF_NETWORK_TESTS") != "1",
    reason="set EVEREST_HF_NETWORK_TESTS=1 to run tests that download from the Hub",
)


def test_the_pinned_revision_resolves_and_then_loads_offline(tmp_path) -> None:
    online = HuggingFaceDatasetResolver(cache_dir=tmp_path)

    snapshot = online.resolve(REPO, REVISION)
    episode = LeRobotV3Reader(snapshot).read_episode(0)

    assert episode.metadata.robot_type == "maker_follower"
    assert len(episode) == 669
    assert episode.metadata.total_frames == 3393

    # The same revision now loads with the network disallowed.
    offline = HuggingFaceDatasetResolver(allow_download=False, cache_dir=tmp_path)
    again = offline.resolve(REPO, REVISION)

    assert LeRobotV3Reader(again).read_episode(0).actions.shape == (669, 7)

    # A revision that was never fetched cannot be served from the cache.
    with pytest.raises(DatasetResolutionError):
        offline.resolve(REPO, "b" * 40)


def test_a_full_real_episode_replays_into_fake_hardware(tmp_path) -> None:
    resolver = HuggingFaceDatasetResolver(cache_dir=tmp_path)
    snapshot = resolver.resolve(REPO, REVISION)
    episode = LeRobotV3Reader(snapshot).read_episode(0)

    from everest_robot.robot.lerobot_bridge import JointFrame

    frame = JointFrame(JOINTS, offsets_deg=OFFSETS)
    start = [
        limit.clamp(value)
        for value, limit in zip(frame.to_radians(episode.states[0]), SDK_LIMITS, strict=True)
    ]

    clock = ManualClock()
    arm = FakeArm(
        identity=IDENTITY,
        joint_limits=SDK_LIMITS,
        clock=clock,
        positions=start,
        max_velocity_rad_s=20.0,
    )
    approved = parameters()
    runner = ReplayRunner(
        arm,
        approved,
        resolver=resolver,
        lease=InMemoryLease("maker-arm-02"),
        clock=clock,
    )

    result = runner.run(request(episode=0))

    assert result.completed
    assert result.frames_sent == 669
    assert result.elapsed_s == pytest.approx(669 / 30, abs=1e-6)
    assert result.calibration_id == CALIBRATION
    # The known endpoint excursions, and nothing worse.
    assert 0 < result.max_clipping_deg <= 1.0
