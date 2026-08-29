"""Camera configuration and observation assembly.

Camera device paths and indices are deployment data, not robot policy, so they never live
in ``config/maker_arm_v1.yaml``. They come from ``EVEREST_CAMERAS`` (inline JSON) or
``EVEREST_CAMERAS_FILE`` (a path to the same JSON), which is what changes between a
workstation, a test rig and the production cell.

The port is deliberately shaped like LeRobot's ``Camera``, so a LeRobot camera satisfies it
as-is and :class:`FakeCamera` can stand in anywhere without an adapter.
"""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

CAMERAS_ENV = "EVEREST_CAMERAS"
CAMERAS_FILE_ENV = "EVEREST_CAMERAS_FILE"

# Kinds resolvable to a real device. "fake" is always available and needs no hardware.
_LEROBOT_KINDS = ("opencv", "realsense")


class CameraConfigError(ValueError):
    """Camera deployment configuration is malformed."""


@runtime_checkable
class CameraPort(Protocol):
    """A connected camera producing RGB frames of a fixed size."""

    fps: int | None
    width: int | None
    height: int | None

    @property
    def is_connected(self) -> bool: ...

    def connect(self, warmup: bool = True) -> None: ...

    def disconnect(self) -> None: ...

    def read(self) -> np.ndarray: ...

    def async_read(self, timeout_ms: float = 200.0) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class CameraSpec:
    """One camera as named by the policy or dataset that consumes it.

    ``name`` is the observation key and must match what a policy checkpoint expects; the
    policy runner validates that before any motion.
    """

    name: str
    kind: str
    index_or_path: str
    width: int
    height: int
    fps: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], where: str) -> CameraSpec:
        allowed = {"name", "kind", "index_or_path", "width", "height", "fps"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise CameraConfigError(f"{where}: unknown field(s) {', '.join(unknown)}")
        missing = sorted(allowed - set(value))
        if missing:
            raise CameraConfigError(f"{where}: missing field(s) {', '.join(missing)}")

        kind = str(value["kind"])
        if kind not in (*_LEROBOT_KINDS, "fake"):
            raise CameraConfigError(
                f"{where}.kind: {kind!r} is not supported "
                f"(expected one of {', '.join((*_LEROBOT_KINDS, 'fake'))})"
            )
        try:
            width, height, fps = int(value["width"]), int(value["height"]), int(value["fps"])
        except (TypeError, ValueError) as error:
            raise CameraConfigError(f"{where}: width, height and fps must be integers") from error
        if min(width, height, fps) <= 0:
            raise CameraConfigError(f"{where}: width, height and fps must be positive")

        return cls(
            name=str(value["name"]),
            kind=kind,
            index_or_path=str(value["index_or_path"]),
            width=width,
            height=height,
            fps=fps,
        )

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        """The observation feature shape LeRobot uses: (height, width, channels)."""

        return (self.height, self.width, 3)


def load_camera_specs(
    source: str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> tuple[CameraSpec, ...]:
    """Read camera specs from JSON, an explicit path, or the environment.

    Returns an empty tuple when nothing is configured: a proprioception-only rollout is a
    legitimate configuration, not an error.
    """

    environ = os.environ if environ is None else environ
    raw = source
    where = "camera configuration"
    if raw is None:
        if path := environ.get(CAMERAS_FILE_ENV):
            raw = Path(path).read_text()
            where = path
        elif inline := environ.get(CAMERAS_ENV):
            raw = inline
            where = CAMERAS_ENV
    if not raw:
        return ()

    try:
        document = json.loads(raw)
    except json.JSONDecodeError as error:
        raise CameraConfigError(f"{where}: invalid JSON ({error})") from error
    if not isinstance(document, list):
        raise CameraConfigError(f"{where}: expected a list of camera objects")

    specs = tuple(
        CameraSpec.from_mapping(entry, f"{where}[{index}]")
        if isinstance(entry, Mapping)
        else _not_a_mapping(f"{where}[{index}]")
        for index, entry in enumerate(document)
    )
    names = [spec.name for spec in specs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise CameraConfigError(f"{where}: duplicate camera name(s) {', '.join(duplicates)}")
    return specs


def _not_a_mapping(where: str) -> CameraSpec:
    raise CameraConfigError(f"{where}: expected a camera object")


class FakeCamera:
    """A camera that produces deterministic frames.

    Each frame is a solid value derived from a counter, so a test can tell frames apart
    and assert that observations advance without carrying image fixtures around.
    """

    def __init__(self, spec: CameraSpec) -> None:
        self.spec = spec
        self.fps: int | None = spec.fps
        self.width: int | None = spec.width
        self.height: int | None = spec.height
        self.frames_read = 0
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, warmup: bool = True) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def read(self) -> np.ndarray:
        if not self._connected:
            raise RuntimeError(f"camera {self.spec.name!r} is not connected")
        self.frames_read += 1
        return np.full(self.spec.frame_shape, self.frames_read % 256, dtype=np.uint8)

    def async_read(self, timeout_ms: float = 200.0) -> np.ndarray:
        return self.read()


def make_camera(spec: CameraSpec) -> CameraPort:
    """Build one camera. LeRobot camera classes are imported only when actually needed."""

    if spec.kind == "fake":
        return FakeCamera(spec)

    if spec.kind == "opencv":
        from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

        index_or_path: int | Path
        try:
            index_or_path = int(spec.index_or_path)
        except ValueError:
            index_or_path = Path(spec.index_or_path)
        return OpenCVCamera(
            OpenCVCameraConfig(
                index_or_path=index_or_path,
                fps=spec.fps,
                width=spec.width,
                height=spec.height,
            )
        )

    from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig

    return RealSenseCamera(
        RealSenseCameraConfig(
            serial_number_or_name=spec.index_or_path,
            fps=spec.fps,
            width=spec.width,
            height=spec.height,
        )
    )


class CameraRuntime:
    """Owns the camera set and produces one synchronized observation per tick.

    Reads are ``async_read``: a control loop wants the most recent frame, not to block on
    a slow device until it produces a new one.
    """

    def __init__(self, cameras: Mapping[str, CameraPort], specs: Sequence[CameraSpec] = ()) -> None:
        self.cameras = dict(cameras)
        self.specs = {spec.name: spec for spec in specs}

    @classmethod
    def from_specs(cls, specs: Sequence[CameraSpec]) -> CameraRuntime:
        return cls({spec.name: make_camera(spec) for spec in specs}, specs)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> CameraRuntime:
        return cls.from_specs(load_camera_specs(environ=environ))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.cameras)

    @property
    def is_connected(self) -> bool:
        return all(camera.is_connected for camera in self.cameras.values())

    def features(self) -> dict[str, tuple[int, int, int]]:
        """Observation features in LeRobot's shape convention, keyed by camera name."""

        return {
            name: self.specs[name].frame_shape
            if name in self.specs
            else (int(camera.height or 0), int(camera.width or 0), 3)
            for name, camera in self.cameras.items()
        }

    def connect(self) -> None:
        """Connect every camera, releasing any that already connected if one fails."""

        connected: list[CameraPort] = []
        try:
            for camera in self.cameras.values():
                camera.connect()
                connected.append(camera)
        except Exception:
            for camera in connected:
                _safe_disconnect(camera)
            raise

    def disconnect(self) -> None:
        for camera in self.cameras.values():
            _safe_disconnect(camera)

    def observation(self, timeout_ms: float = 200.0) -> dict[str, np.ndarray]:
        return {
            name: camera.async_read(timeout_ms=timeout_ms)
            for name, camera in self.cameras.items()
        }


def _safe_disconnect(camera: CameraPort) -> None:
    """Never let one camera's teardown strand the others or mask the original failure."""

    with contextlib.suppress(Exception):
        camera.disconnect()
