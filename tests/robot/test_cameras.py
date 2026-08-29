import json

import numpy as np
import pytest

from everest_robot.robot.cameras import (
    CAMERAS_ENV,
    CAMERAS_FILE_ENV,
    CameraConfigError,
    CameraPort,
    CameraRuntime,
    CameraSpec,
    FakeCamera,
    load_camera_specs,
)

SPECS = [
    {"name": "wrist", "kind": "fake", "index_or_path": "0", "width": 8, "height": 4, "fps": 30},
    {"name": "scene", "kind": "fake", "index_or_path": "1", "width": 8, "height": 4, "fps": 30},
]


def runtime() -> CameraRuntime:
    return CameraRuntime.from_specs(load_camera_specs(json.dumps(SPECS)))


def test_fake_camera_satisfies_the_port() -> None:
    assert isinstance(FakeCamera(CameraSpec("wrist", "fake", "0", 8, 4, 30)), CameraPort)


def test_specs_come_from_inline_json_or_a_file(tmp_path) -> None:
    from_inline = load_camera_specs(environ={CAMERAS_ENV: json.dumps(SPECS)})

    path = tmp_path / "cameras.json"
    path.write_text(json.dumps(SPECS))
    from_file = load_camera_specs(environ={CAMERAS_FILE_ENV: str(path)})

    assert from_inline == from_file
    assert [spec.name for spec in from_inline] == ["wrist", "scene"]


def test_no_configured_cameras_is_a_valid_configuration() -> None:
    assert load_camera_specs(environ={}) == ()
    assert load_camera_specs("") == ()


def test_malformed_configuration_is_rejected() -> None:
    with pytest.raises(CameraConfigError, match="invalid JSON"):
        load_camera_specs("{oops")
    with pytest.raises(CameraConfigError, match="list of camera objects"):
        load_camera_specs(json.dumps({"name": "wrist"}))
    with pytest.raises(CameraConfigError, match="missing field"):
        load_camera_specs(json.dumps([{"name": "wrist", "kind": "fake"}]))
    with pytest.raises(CameraConfigError, match="unknown field"):
        load_camera_specs(json.dumps([{**SPECS[0], "exposure": 10}]))
    with pytest.raises(CameraConfigError, match="not supported"):
        load_camera_specs(json.dumps([{**SPECS[0], "kind": "webcam"}]))
    with pytest.raises(CameraConfigError, match="positive"):
        load_camera_specs(json.dumps([{**SPECS[0], "width": 0}]))


def test_duplicate_camera_names_are_rejected() -> None:
    # Observation keys must be unique or one camera silently shadows the other.
    with pytest.raises(CameraConfigError, match="duplicate"):
        load_camera_specs(json.dumps([SPECS[0], SPECS[0]]))


def test_features_use_lerobot_height_width_channel_shapes() -> None:
    assert runtime().features() == {"wrist": (4, 8, 3), "scene": (4, 8, 3)}


def test_observations_carry_one_frame_per_camera_and_advance() -> None:
    cameras = runtime()
    cameras.connect()

    first = cameras.observation()
    second = cameras.observation()

    assert set(first) == {"wrist", "scene"}
    assert first["wrist"].shape == (4, 8, 3)
    assert first["wrist"].dtype == np.uint8
    assert not np.array_equal(first["wrist"], second["wrist"])

    cameras.disconnect()
    assert not cameras.is_connected


def test_a_failing_camera_does_not_strand_the_ones_already_connected() -> None:
    class Broken(FakeCamera):
        def connect(self, warmup: bool = True) -> None:
            raise RuntimeError("no such device")

    good = FakeCamera(CameraSpec("wrist", "fake", "0", 8, 4, 30))
    broken = Broken(CameraSpec("scene", "fake", "1", 8, 4, 30))
    cameras = CameraRuntime({"wrist": good, "scene": broken})

    with pytest.raises(RuntimeError, match="no such device"):
        cameras.connect()

    assert not good.is_connected


def test_reading_a_disconnected_camera_fails_loudly() -> None:
    cameras = runtime()

    with pytest.raises(RuntimeError, match="not connected"):
        cameras.observation()
