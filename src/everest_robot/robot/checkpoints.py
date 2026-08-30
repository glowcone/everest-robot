"""Resolving a trained checkpoint, and the feature mapping it was trained under.

Loading a checkpoint needs two things: the weights, and the dataset feature metadata that
says how this robot's ``{joint}.pos`` scalars and camera frames become the policy's input
tensors. The weights are the easy half. The metadata is the half that used to block
:class:`~everest_robot.robot.policy.LeRobotPolicyHandle`, because getting it wrong
mis-orders joints against a real checkpoint without ever failing.

It is not invented here, and it is not derived from the robot. It is read from the dataset
the checkpoint was *trained* on, which LeRobot records in the checkpoint's own
``train_config.json``:

```text
config.json        policy type, input/output feature shapes, normalization
train_config.json  dataset.repo_id  ->  meta/info.json  ->  features, names, fps
```

``meta/info.json`` names every joint of ``observation.state`` and ``action`` in the exact
order the policy consumes them. That is the authoritative mapping, and everything below is
cross-checking it: the dataset's shapes against the shapes baked into the checkpoint
config, the state names against the action names, and -- once the handle reaches
:func:`~everest_robot.robot.policy.compatibility_problems` -- the whole action space
against the connected arm's, in order. A checkpoint trained on a differently-ordered robot
fails that comparison rather than commanding the wrong axis.

Resolution touches no hardware and is done before the robot is claimed, so a wrong repo id,
an unreachable dataset or a feature mismatch costs no lease.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CHECKPOINT_CONFIG = "config.json"
TRAIN_CONFIG = "train_config.json"
DATASET_INFO = "meta/info.json"

STATE_FEATURE = "observation.state"
ACTION_FEATURE = "action"
IMAGE_PREFIX = "observation.images."

#: Everything needed to run inference. Deliberately excludes ``checkpoints/**`` (the
#: per-step training snapshots) and ``training_state/**`` (optimizer and RNG state), which
#: are the bulk of the repo and are of no use to a rollout.
INFERENCE_PATTERNS = (
    CHECKPOINT_CONFIG,
    TRAIN_CONFIG,
    "model.safetensors",
    "policy_preprocessor*",
    "policy_postprocessor*",
)

_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


class CheckpointError(RuntimeError):
    """A checkpoint could not be resolved, or does not describe what it was trained on."""


@dataclass(frozen=True, slots=True)
class CheckpointFeatures:
    """The dataset feature metadata a checkpoint was trained under.

    ``ds_features`` is the subset of the training dataset's feature dictionary that
    inference consumes, in LeRobot's own schema, so it can be handed straight to
    ``build_inference_frame`` and ``make_robot_action``. ``state_names`` and
    ``action_names`` are the ``{joint}.pos`` keys in the order the tensors pack them.
    """

    dataset_repo_id: str
    dataset_revision: str
    fps: float
    robot_type: str | None
    ds_features: Mapping[str, Mapping[str, Any]]
    state_names: tuple[str, ...]
    action_names: tuple[str, ...]
    cameras: Mapping[str, tuple[int, int, int]]

    @property
    def input_features(self) -> dict[str, tuple[int, ...]]:
        """What the policy needs, keyed the way the *robot* names its observations.

        Joint scalars carry an empty shape: the robot reports them as ``float``, and
        presence is the only thing worth checking. Cameras carry ``(H, W, C)``, which is
        what a resolution mismatch shows up as.
        """

        features: dict[str, tuple[int, ...]] = {name: () for name in self.state_names}
        features.update(self.cameras)
        return features


@dataclass(frozen=True, slots=True)
class ResolvedCheckpoint:
    """A local, inference-ready checkpoint plus the mapping it was trained under."""

    source: str
    root: Path
    revision: str | None
    policy_type: str
    features: CheckpointFeatures

    @property
    def identifier(self) -> str:
        """What to record as the checkpoint in a durable result."""

        return self.source if self.revision is None else f"{self.source}@{self.revision[:12]}"


def resolve_checkpoint(
    reference: str | Path,
    *,
    revision: str | None = None,
    dataset_repo_id: str | None = None,
    dataset_revision: str | None = None,
    allow_download: bool = True,
    cache_dir: str | Path | None = None,
) -> ResolvedCheckpoint:
    """Resolve a local directory or a Hugging Face repo id to an inference-ready checkpoint.

    ``dataset_repo_id`` overrides the training dataset recorded in ``train_config.json``.
    It exists for a checkpoint directory copied out of a training run without its
    ``train_config.json``; it is not a way to attach a different robot's feature order to a
    checkpoint, and the cross-checks below still have to pass.
    """

    source = str(reference)
    local = Path(source).expanduser()
    if local.exists():
        if not local.is_dir():
            raise CheckpointError(
                f"{source}: a trained checkpoint is a directory (LeRobot writes "
                f"{CHECKPOINT_CONFIG} and model.safetensors side by side), not a single file"
            )
        root, resolved_revision = local, None
    elif _REPO_ID.match(source):
        root, resolved_revision = _download(
            source, revision=revision, allow_download=allow_download, cache_dir=cache_dir
        )
    else:
        raise CheckpointError(
            f"{source!r} is neither an existing directory nor a Hugging Face repo id of the "
            f"form 'namespace/name'"
        )

    config = _read_json(root / CHECKPOINT_CONFIG, source)
    policy_type = str(config.get("type") or "")
    if not policy_type:
        raise CheckpointError(f"{source}: {CHECKPOINT_CONFIG} declares no policy 'type'")

    repo_id = dataset_repo_id or _training_dataset(root, source)
    features = _resolve_features(
        config,
        repo_id=repo_id,
        revision=dataset_revision,
        allow_download=allow_download,
        cache_dir=cache_dir,
        source=source,
    )
    return ResolvedCheckpoint(
        source=source,
        root=root,
        revision=resolved_revision,
        policy_type=policy_type,
        features=features,
    )


# ── the hub ────────────────────────────────────────────────────────────────────────
def _download(
    repo_id: str,
    *,
    revision: str | None,
    allow_download: bool,
    cache_dir: str | Path | None,
) -> tuple[Path, str | None]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:  # pragma: no cover - huggingface_hub is a core dep
        raise CheckpointError("huggingface_hub is not installed") from error

    try:
        root = snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            allow_patterns=list(INFERENCE_PATTERNS),
            local_files_only=not allow_download,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    except Exception as error:
        # Never surface the underlying message: hub errors can echo request headers, and
        # those carry HF_TOKEN. Same rule as the dataset resolver.
        raise CheckpointError(
            f"{repo_id}: could not resolve the checkpoint ({type(error).__name__}). Check the "
            f"repo id, the revision, network access, and HF_TOKEN for a private model"
        ) from None
    return Path(root), _snapshot_revision(Path(root), revision)


def _snapshot_revision(root: Path, requested: str | None) -> str | None:
    """The commit the cache actually materialized, which is what the run should record.

    ``huggingface_hub`` lays a snapshot down under ``snapshots/<sha>/``, so the resolved
    commit is readable from the path without a second network call -- and it stays readable
    when the download was served entirely from cache.
    """

    if root.parent.name == "snapshots":
        return root.name
    return requested


def _read_json(path: Path, source: str) -> dict[str, Any]:
    if not path.is_file():
        raise CheckpointError(f"{source}: missing {path.name} in the checkpoint")
    try:
        document = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise CheckpointError(f"{source}: {path.name} could not be read ({error})") from None
    if not isinstance(document, dict):
        raise CheckpointError(f"{source}: {path.name} must contain an object")
    return document


def _training_dataset(root: Path, source: str) -> str:
    document = root / TRAIN_CONFIG
    if not document.is_file():
        raise CheckpointError(
            f"{source}: no {TRAIN_CONFIG}, so the training dataset is unknown and the joint "
            f"order the checkpoint expects cannot be established. Pass the training dataset "
            f"explicitly if you know it"
        )
    dataset = _read_json(document, source).get("dataset")
    repo_id = dataset.get("repo_id") if isinstance(dataset, Mapping) else None
    if not repo_id:
        raise CheckpointError(
            f"{source}: {TRAIN_CONFIG} names no dataset.repo_id; the joint order the "
            f"checkpoint expects cannot be established"
        )
    return str(repo_id)


# ── the feature mapping ────────────────────────────────────────────────────────────
def _resolve_features(
    config: Mapping[str, Any],
    *,
    repo_id: str,
    revision: str | None,
    allow_download: bool,
    cache_dir: str | Path | None,
    source: str,
) -> CheckpointFeatures:
    info, resolved_revision = _dataset_info(
        repo_id, revision=revision, allow_download=allow_download, cache_dir=cache_dir
    )

    raw = info.get("features")
    if not isinstance(raw, Mapping):
        raise CheckpointError(f"{repo_id}: {DATASET_INFO} has no 'features' object")

    state_names = _joint_names(raw, STATE_FEATURE, repo_id)
    action_names = _joint_names(raw, ACTION_FEATURE, repo_id)
    if state_names != action_names:
        raise CheckpointError(
            f"{repo_id}: '{STATE_FEATURE}' and '{ACTION_FEATURE}' name different joints "
            f"({list(state_names)} vs {list(action_names)}); an observation cannot be matched "
            f"to the action it produces"
        )

    declared = config.get("input_features")
    if not isinstance(declared, Mapping):
        raise CheckpointError(f"{source}: {CHECKPOINT_CONFIG} has no 'input_features'")

    ds_features: dict[str, Mapping[str, Any]] = {
        STATE_FEATURE: _vector_feature(raw[STATE_FEATURE], state_names),
        ACTION_FEATURE: _vector_feature(raw[ACTION_FEATURE], action_names),
    }
    cameras: dict[str, tuple[int, int, int]] = {}

    for key, spec in declared.items():
        if key == STATE_FEATURE:
            _check_vector_shape(spec, len(state_names), key, source)
            continue
        if not key.startswith(IMAGE_PREFIX):
            raise CheckpointError(
                f"{source}: input feature {key!r} is neither {STATE_FEATURE!r} nor a camera. "
                f"Only proprioception and images are supported here"
            )
        if key not in raw:
            raise CheckpointError(
                f"{source}: needs camera feature {key!r}, which the training dataset "
                f"{repo_id} does not describe"
            )
        shape = _camera_shape(raw[key], key, repo_id)
        _check_camera_shape(spec, shape, key, source)
        ds_features[key] = dict(raw[key]) | {"shape": shape}
        cameras[key.removeprefix(IMAGE_PREFIX)] = shape

    outputs = config.get("output_features")
    action_spec = outputs.get(ACTION_FEATURE) if isinstance(outputs, Mapping) else None
    if action_spec is None:
        raise CheckpointError(
            f"{source}: {CHECKPOINT_CONFIG} declares no {ACTION_FEATURE!r} output feature"
        )
    _check_vector_shape(action_spec, len(action_names), ACTION_FEATURE, source)

    fps = float(info.get("fps") or 0.0)
    if fps <= 0.0:
        raise CheckpointError(f"{repo_id}: {DATASET_INFO} declares no usable fps")

    robot_type = info.get("robot_type")
    return CheckpointFeatures(
        dataset_repo_id=repo_id,
        dataset_revision=resolved_revision,
        fps=fps,
        robot_type=None if robot_type is None else str(robot_type),
        ds_features=ds_features,
        state_names=state_names,
        action_names=action_names,
        cameras=cameras,
    )


def _dataset_info(
    repo_id: str,
    *,
    revision: str | None,
    allow_download: bool,
    cache_dir: str | Path | None,
) -> tuple[Mapping[str, Any], str]:
    """Read just ``meta/info.json``. The parquet and the videos are of no interest here."""

    if not _REPO_ID.match(repo_id):
        raise CheckpointError(f"invalid dataset repo id {repo_id!r} (expected 'namespace/name')")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:  # pragma: no cover - huggingface_hub is a core dep
        raise CheckpointError("huggingface_hub is not installed") from error

    try:
        path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=DATASET_INFO,
                repo_type="dataset",
                revision=revision,
                local_files_only=not allow_download,
                cache_dir=str(cache_dir) if cache_dir else None,
            )
        )
    except Exception as error:
        raise CheckpointError(
            f"{repo_id}: could not read {DATASET_INFO}, so the joint order the checkpoint "
            f"expects cannot be established ({type(error).__name__}). Check network access "
            f"and HF_TOKEN for a private dataset"
        ) from None

    document = _read_json(path, repo_id)
    # `.../snapshots/<sha>/meta/info.json` -- walk up past `meta/`.
    resolved = _snapshot_revision(path.parent.parent, revision) or (revision or "unpinned")
    return document, resolved


def _joint_names(features: Mapping[str, Any], key: str, repo_id: str) -> tuple[str, ...]:
    feature = features.get(key)
    if not isinstance(feature, Mapping):
        raise CheckpointError(f"{repo_id}: {DATASET_INFO} has no feature {key!r}")
    names = feature.get("names")
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        raise CheckpointError(f"{repo_id}: feature {key!r} has no joint names")
    if len(set(names)) != len(names):
        duplicates = sorted({n for n in names if names.count(n) > 1})
        raise CheckpointError(
            f"{repo_id}: feature {key!r} repeats joint(s) {', '.join(duplicates)}"
        )
    shape = feature.get("shape")
    if list(shape or ()) != [len(names)]:
        raise CheckpointError(
            f"{repo_id}: feature {key!r} has shape {shape} but names {len(names)} joints"
        )
    return tuple(names)


def _vector_feature(feature: Mapping[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    """A 1-D feature in the shape ``build_dataset_frame`` expects (a tuple, not a list)."""

    return dict(feature) | {"dtype": "float32", "shape": (len(names),), "names": list(names)}


def _camera_shape(feature: Mapping[str, Any], key: str, repo_id: str) -> tuple[int, int, int]:
    shape = feature.get("shape")
    if not isinstance(shape, (list, tuple)) or len(shape) != 3:
        raise CheckpointError(
            f"{repo_id}: camera feature {key!r} has shape {shape}, expected (H, W, C)"
        )
    height, width, channels = (int(axis) for axis in shape)
    if channels != 3:
        raise CheckpointError(
            f"{repo_id}: camera feature {key!r} has {channels} channels; only RGB is supported"
        )
    return (height, width, channels)


def _check_vector_shape(spec: Any, expected: int, key: str, source: str) -> None:
    shape = spec.get("shape") if isinstance(spec, Mapping) else None
    if list(shape or ()) != [expected]:
        raise CheckpointError(
            f"{source}: {key!r} is {shape} in the checkpoint but the training dataset names "
            f"{expected} joints. These must agree or the tensor is packed wrong"
        )


def _check_camera_shape(
    spec: Any, dataset_shape: tuple[int, int, int], key: str, source: str
) -> None:
    """The checkpoint stores images channel-first; the dataset stores them channel-last."""

    shape = spec.get("shape") if isinstance(spec, Mapping) else None
    height, width, channels = dataset_shape
    if list(shape or ()) != [channels, height, width]:
        raise CheckpointError(
            f"{source}: {key!r} is {shape} (C, H, W) in the checkpoint but {list(dataset_shape)} "
            f"(H, W, C) in the training dataset; the camera resolution does not match"
        )
