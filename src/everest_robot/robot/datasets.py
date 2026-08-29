"""Resolving and reading a pinned LeRobot v3 dataset snapshot.

Replay needs three things out of a dataset: the recorded actions, the recorded initial
state, and the timing. All three live in parquet columns next to a small JSON manifest, so
this module reads the snapshot directly instead of going through ``LeRobotDataset``.

That is a deliberate trade. Reading the files ourselves means:

* no torch, torchcodec or ``datasets`` on the robot control host merely to read joint
  angles, and no video decoding for a joint-action replay;
* the whole replay path stays importable and testable without the ``hardware`` extra; and
* an explicit, auditable schema check instead of whatever a dataset loader accepts.

The cost is that we own the format: only the layout documented below is supported, and
anything else is refused rather than guessed at. The layout is LeRobot's v3
(``lerobot/datasets/utils.py``):

```text
meta/info.json                              codebase_version, fps, features, data_path, ...
meta/episodes/chunk-{c:03d}/file-{f:03d}.parquet   per-episode index and stats
data/chunk-{c:03d}/file-{f:03d}.parquet            frames: action, observation.state, ...
```
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from everest_robot.robot.errors import DatasetCompatibilityError, DatasetResolutionError

SUPPORTED_CODEBASE_VERSIONS = ("v3.0",)

INFO_PATH = "meta/info.json"
EPISODES_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"

ACTION_FEATURE = "action"
STATE_FEATURE = "observation.state"

# Videos are not read by a joint-action replay, and they are the bulk of a dataset.
METADATA_AND_DATA_ONLY = ("meta/*", "meta/**/*", "data/*", "data/**/*")

_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_FULL_REVISION = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True, slots=True)
class DatasetSnapshot:
    """An immutable local copy of one dataset revision."""

    repo_id: str
    revision: str
    root: Path

    def path(self, relative: str) -> Path:
        candidate = self.root / relative
        if not candidate.is_file():
            raise DatasetResolutionError(
                f"{self.repo_id}@{self.revision[:12]}: missing {relative} in the local snapshot"
            )
        return candidate


class HuggingFaceDatasetResolver:
    """Resolves a repo id and revision to an immutable local snapshot.

    Never resolves a moving head in production: a branch name would make the same workflow
    parameters replay different physical motion on different days.
    """

    def __init__(
        self,
        *,
        require_full_revision: bool = True,
        allow_download: bool = True,
        include_videos: bool = False,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.require_full_revision = require_full_revision
        self.allow_download = allow_download
        self.include_videos = include_videos
        self.cache_dir = cache_dir

    def validate(self, repo_id: str, revision: str) -> None:
        if not _REPO_ID.match(repo_id or ""):
            raise DatasetResolutionError(
                f"invalid dataset repo id {repo_id!r} (expected 'namespace/name')"
            )
        if not revision:
            raise DatasetResolutionError(f"{repo_id}: a dataset revision is required")
        if self.require_full_revision and not _FULL_REVISION.match(revision):
            raise DatasetResolutionError(
                f"{repo_id}: {revision!r} is not a full 40-character commit SHA. Replay pins an "
                "immutable revision so the same request always drives the same motion"
            )

    def resolve(self, repo_id: str, revision: str) -> DatasetSnapshot:
        """Download or reuse the snapshot. Touches no hardware.

        Authentication comes from the environment (``HF_TOKEN``), read by
        ``huggingface_hub`` itself, so no credential passes through workflow parameters.
        """

        self.validate(repo_id, revision)
        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:  # pragma: no cover - huggingface_hub is a core dep
            raise DatasetResolutionError("huggingface_hub is not installed") from error

        patterns = None if self.include_videos else list(METADATA_AND_DATA_ONLY)
        try:
            root = snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                allow_patterns=patterns,
                local_files_only=not self.allow_download,
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
            )
        except Exception as error:
            # Never surface the underlying message verbatim: hub errors can echo request
            # headers, and those carry the token.
            raise DatasetResolutionError(
                f"{repo_id}@{revision[:12]}: could not resolve the dataset snapshot "
                f"({type(error).__name__}). Check the repo id, the revision, network access, "
                "and HF_TOKEN for a private dataset"
            ) from None
        return DatasetSnapshot(repo_id=repo_id, revision=revision, root=Path(root))


@dataclass(frozen=True, slots=True)
class EpisodeMetadata:
    """What the dataset says about itself and the selected episode."""

    repo_id: str
    revision: str
    codebase_version: str
    robot_type: str | None
    fps: float
    episode: int
    length: int
    total_episodes: int
    total_frames: int
    joint_names: tuple[str, ...]
    task: str | None


@dataclass(frozen=True, slots=True)
class Episode:
    """One episode's recorded actions and states, in dataset units (degrees).

    ``actions`` and ``states`` are ``(frames, joints)`` with columns in ``joint_names``
    order. Names come from the dataset's own feature metadata, never from position.
    """

    metadata: EpisodeMetadata
    joint_names: tuple[str, ...]
    actions: np.ndarray
    states: np.ndarray
    timestamps: np.ndarray
    frame_indices: np.ndarray

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    def named_action(self, frame: int) -> dict[str, float]:
        """The action at ``frame`` as ``{joint: degrees}``, rebuilt from the names."""

        return dict(zip(self.joint_names, (float(v) for v in self.actions[frame]), strict=True))

    def named_state(self, frame: int) -> dict[str, float]:
        return dict(zip(self.joint_names, (float(v) for v in self.states[frame]), strict=True))


class LeRobotV3Reader:
    """Reads episodes out of a resolved v3 snapshot."""

    def __init__(self, snapshot: DatasetSnapshot) -> None:
        self.snapshot = snapshot
        self._info: dict[str, Any] | None = None

    # ── manifest ───────────────────────────────────────────────────────────────────
    @property
    def info(self) -> dict[str, Any]:
        if self._info is None:
            raw = self.snapshot.path(INFO_PATH).read_text()
            try:
                document = json.loads(raw)
            except json.JSONDecodeError as error:
                raise DatasetCompatibilityError(
                    f"{INFO_PATH} is not valid JSON: {error}"
                ) from error
            if not isinstance(document, dict):
                raise DatasetCompatibilityError(f"{INFO_PATH} must contain an object")
            self._info = document
        return self._info

    @property
    def codebase_version(self) -> str:
        return str(self.info.get("codebase_version", ""))

    @property
    def fps(self) -> float:
        return float(self.info.get("fps", 0.0))

    @property
    def robot_type(self) -> str | None:
        value = self.info.get("robot_type")
        return None if value is None else str(value)

    def joint_names(self) -> tuple[str, ...]:
        """The action feature's joint names, validated against the state feature.

        Positional order is not trusted anywhere: every frame is rebuilt by name, and the
        two features must agree so the recorded initial state can be compared with the
        recorded actions.
        """

        features = self.info.get("features")
        if not isinstance(features, dict):
            raise DatasetCompatibilityError(f"{INFO_PATH}: 'features' is missing or not an object")

        action_names = self._feature_names(features, ACTION_FEATURE)
        state_names = self._feature_names(features, STATE_FEATURE)
        if action_names != state_names:
            raise DatasetCompatibilityError(
                f"'{ACTION_FEATURE}' and '{STATE_FEATURE}' name different joints "
                f"({list(action_names)} vs {list(state_names)}); the recorded initial state "
                "cannot be matched to the recorded actions"
            )
        return action_names

    @staticmethod
    def _feature_names(features: dict[str, Any], key: str) -> tuple[str, ...]:
        feature = features.get(key)
        if not isinstance(feature, dict):
            raise DatasetCompatibilityError(f"{INFO_PATH}: feature {key!r} is missing")

        names = feature.get("names")
        if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
            raise DatasetCompatibilityError(f"{INFO_PATH}: feature {key!r} has no joint names")
        if len(set(names)) != len(names):
            duplicates = sorted({n for n in names if names.count(n) > 1})
            raise DatasetCompatibilityError(
                f"{INFO_PATH}: feature {key!r} repeats joint(s) {', '.join(duplicates)}"
            )

        shape = feature.get("shape")
        if not isinstance(shape, list) or list(shape) != [len(names)]:
            raise DatasetCompatibilityError(
                f"{INFO_PATH}: feature {key!r} has shape {shape} but names {len(names)} joints"
            )
        dtype = str(feature.get("dtype", ""))
        if dtype not in ("float32", "float64"):
            raise DatasetCompatibilityError(
                f"{INFO_PATH}: feature {key!r} has dtype {dtype!r}; expected float32 or float64"
            )
        return tuple(names)

    # ── episodes ───────────────────────────────────────────────────────────────────
    def episode_rows(self) -> list[dict[str, Any]]:
        """The per-episode index, which says where each episode's frames live."""

        import pyarrow.parquet as pq

        chunks = int(self.info.get("chunks_size", 1000))
        if chunks <= 0:
            raise DatasetCompatibilityError(f"{INFO_PATH}: chunks_size must be positive")

        rows: list[dict[str, Any]] = []
        total = int(self.info.get("total_episodes", 0))
        # The episode index is itself chunked; walk files until every episode is accounted
        # for rather than assuming a single file.
        chunk_index = file_index = 0
        while len(rows) < total:
            relative = EPISODES_PATH.format(chunk_index=chunk_index, file_index=file_index)
            if not (self.snapshot.root / relative).is_file():
                if file_index == 0:
                    break
                chunk_index, file_index = chunk_index + 1, 0
                continue
            table = pq.read_table(self.snapshot.path(relative))
            columns = [
                name
                for name in (
                    "episode_index",
                    "length",
                    "data/chunk_index",
                    "data/file_index",
                    "tasks",
                )
                if name in table.column_names
            ]
            rows.extend(table.select(columns).to_pylist())
            file_index += 1

        if len(rows) != total:
            raise DatasetCompatibilityError(
                f"episode index lists {len(rows)} episodes but {INFO_PATH} declares {total}"
            )
        return rows

    def read_episode(self, episode: int) -> Episode:
        """Load one episode's actions, states and timestamps."""

        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        joint_names = self.joint_names()
        rows = {int(row["episode_index"]): row for row in self.episode_rows()}
        total_episodes = int(self.info.get("total_episodes", len(rows)))
        if episode not in rows:
            raise DatasetCompatibilityError(
                f"episode {episode} is not in this dataset (it has {total_episodes}: "
                f"0-{max(total_episodes - 1, 0)})"
            )
        row = rows[episode]

        data_path = str(self.info.get("data_path", ""))
        if not data_path:
            raise DatasetCompatibilityError(f"{INFO_PATH}: 'data_path' is missing")
        relative = data_path.format(
            chunk_index=int(row.get("data/chunk_index", 0)),
            file_index=int(row.get("data/file_index", 0)),
        )
        table = pq.read_table(self.snapshot.path(relative))

        for column in (ACTION_FEATURE, STATE_FEATURE, "episode_index", "frame_index", "timestamp"):
            if column not in table.column_names:
                raise DatasetCompatibilityError(f"{relative}: missing column {column!r}")

        # Select by episode_index rather than by row offset: a data file holds several
        # episodes and the file's own base offset is not stated anywhere.
        selected = table.filter(pc.equal(table.column("episode_index"), episode))
        if selected.num_rows == 0:
            raise DatasetCompatibilityError(f"{relative}: contains no frames for episode {episode}")

        expected = int(row.get("length", selected.num_rows))
        if selected.num_rows != expected:
            raise DatasetCompatibilityError(
                f"episode {episode}: index declares {expected} frames but {relative} holds "
                f"{selected.num_rows}"
            )

        actions = _as_matrix(selected.column(ACTION_FEATURE), len(joint_names), ACTION_FEATURE)
        states = _as_matrix(selected.column(STATE_FEATURE), len(joint_names), STATE_FEATURE)
        timestamps = np.asarray(selected.column("timestamp").to_pylist(), dtype=np.float64)
        frame_indices = np.asarray(selected.column("frame_index").to_pylist(), dtype=np.int64)

        order = np.argsort(frame_indices, kind="stable")
        if not np.array_equal(frame_indices[order], np.arange(len(frame_indices))):
            raise DatasetCompatibilityError(
                f"episode {episode}: frame indices are not a contiguous "
                f"0..{len(frame_indices) - 1} range; the episode is incomplete or interleaved"
            )

        task = None
        tasks = row.get("tasks")
        if isinstance(tasks, list) and tasks:
            task = str(tasks[0])

        metadata = EpisodeMetadata(
            repo_id=self.snapshot.repo_id,
            revision=self.snapshot.revision,
            codebase_version=self.codebase_version,
            robot_type=self.robot_type,
            fps=self.fps,
            episode=episode,
            length=int(selected.num_rows),
            total_episodes=total_episodes,
            total_frames=int(self.info.get("total_frames", 0)),
            joint_names=joint_names,
            task=task,
        )
        return Episode(
            metadata=metadata,
            joint_names=joint_names,
            actions=actions[order],
            states=states[order],
            timestamps=timestamps[order],
            frame_indices=frame_indices[order],
        )


def _as_matrix(column: Any, joints: int, name: str) -> np.ndarray:
    """Turn a parquet list column into a ``(frames, joints)`` float array."""

    values = column.to_pylist()
    if any(row is None for row in values):
        raise DatasetCompatibilityError(f"{name}: contains a null row")
    widths = {len(row) for row in values}
    if widths != {joints}:
        raise DatasetCompatibilityError(
            f"{name}: expected {joints} values per frame, found {sorted(widths)}"
        )
    try:
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise DatasetCompatibilityError(f"{name}: values are not numeric") from error
    return matrix
