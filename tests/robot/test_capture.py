from datetime import date
from pathlib import Path

import pytest

from everest_robot.robot.capture import (
    CapturedPose,
    CaptureRefused,
    insert_preset,
    render_preset,
    save_preset,
    validate_name,
)
from everest_robot.robot.parameters import RobotParameters

CALIBRATION = "maker-arm-02-2026-08-20"
JOINTS = ("shoulder_pan", "shoulder_lift", "gripper")

# A parameters file with the shape that matters: comments carrying content no loader
# preserves, sitting on both sides of the block being edited.
FILE = """schema_version: 1

robot:
  id: maker-arm-02
  model: maker-arm-v1
  calibration_id: maker-arm-02-2026-08-20
  joint_order: [shoulder_pan, shoulder_lift, gripper]
  units: radians

motion_defaults:
  max_velocity_rad_s: 0.5
  max_acceleration_rad_s2: 2.0
  tolerance_rad: 0.02
  settle_time_s: 0.2
  timeout_s: 10.0
  control_rate_hz: 30

# No presets are defined yet, and none may be invented.
# Capture them by following docs/named-position-capture.md.
named_positions: {}

# Waypoint sequences for transitions that are not known to be collision-free.
named_transitions: {}

policy:
  default_controller: vla
  fps: 30
  max_duration_s: 30

replay:
  require_matching_robot_id: true
  require_matching_calibration_id: true
  safe_start_position: null
  max_speed_scale: 1.0
"""


def pose(name: str = "stage", joints: tuple[float, ...] = (0.2, -0.4, -1.0)) -> CapturedPose:
    return CapturedPose(
        name=name,
        joints=joints,
        calibration_id=CALIBRATION,
        approved_by="operator",
        captured_at=date(2026, 8, 29),
        notes=None,
    )


def written(tmp_path: Path, text: str = FILE) -> Path:
    path = tmp_path / "maker_arm_v1.yaml"
    path.write_text(text)
    return path


# ── names ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["stage", "clip-attachment-ready", "lift_clear2"])
def test_ordinary_preset_names_are_accepted(name: str):
    assert validate_name(name) == name


@pytest.mark.parametrize(
    "name",
    ["", "2fast", "has space", "quote\"key", "a: b", "-leading", "../etc", "name\nkey: x"],
)
def test_a_name_that_could_alter_the_file_s_structure_is_refused(name: str):
    with pytest.raises(CaptureRefused, match="not a usable preset name"):
        validate_name(name)


# ── rendering ──────────────────────────────────────────────────────────────────────


def test_render_writes_only_schema_fields():
    """robot_id and config_digest are identity for the operator, not preset data."""

    block = render_preset(pose())

    assert "robot_id" not in block
    assert "config_digest" not in block
    assert "calibration_id: maker-arm-02-2026-08-20" in block
    assert "captured_at: 2026-08-29" in block


def test_render_keeps_enough_precision_to_reproduce_the_measurement():
    block = render_preset(pose(joints=(0.12345678, -0.4, -1.0)))

    assert "0.1234568" in block


def test_operator_text_cannot_break_out_of_its_scalar():
    block = render_preset(
        CapturedPose(
            name="stage",
            joints=(0.0, 0.0, 0.0),
            calibration_id=CALIBRATION,
            approved_by='he said "measure twice"',
            captured_at=date(2026, 8, 29),
            notes="fixture at station 2: watch the cable",
        )
    )
    loaded = RobotParameters.from_mapping(
        _document(block), config_digest="sha256:test", source="test.yaml"
    )

    assert loaded.named_positions["stage"].approved_by == 'he said "measure twice"'
    assert loaded.named_positions["stage"].notes == "fixture at station 2: watch the cable"


def _document(block: str) -> dict:
    """Parse a rendered block the way the loader will see it."""

    import yaml

    return yaml.safe_load(
        FILE.replace("named_positions: {}\n", "named_positions:\n" + block)
    )


# ── splicing ───────────────────────────────────────────────────────────────────────


def test_the_first_preset_replaces_the_empty_mapping():
    result = insert_preset(FILE, "stage", render_preset(pose()))

    assert "named_positions: {}" not in result
    assert "named_positions:\n  stage:\n" in result


def test_splicing_leaves_every_other_byte_alone():
    """The file's comments carry content no YAML dumper would preserve."""

    result = insert_preset(FILE, "stage", render_preset(pose()))

    for line in FILE.splitlines():
        if line.strip() and line != "named_positions: {}":
            assert line in result.splitlines()


def test_a_second_preset_joins_the_first_without_disturbing_the_next_key():
    once = insert_preset(FILE, "stage", render_preset(pose()))
    twice = insert_preset(once, "lift", render_preset(pose("lift", (0.0, 0.5, -1.0))))

    lines = twice.splitlines()
    block = lines[lines.index("named_positions:") + 1 :]
    assert block[0] == "  stage:"
    assert "  lift:" in block
    # The comment introducing the next key stays attached to that key.
    assert lines[lines.index("named_transitions: {}") - 1].startswith("# Waypoint")


def test_an_existing_preset_is_not_silently_overwritten():
    once = insert_preset(FILE, "stage", render_preset(pose()))

    with pytest.raises(CaptureRefused, match="re-approval"):
        insert_preset(once, "stage", render_preset(pose()))


def test_replacing_a_preset_leaves_its_neighbours_intact():
    once = insert_preset(FILE, "stage", render_preset(pose()))
    twice = insert_preset(once, "lift", render_preset(pose("lift", (0.0, 0.5, -1.0))))

    result = insert_preset(
        twice, "stage", render_preset(pose(joints=(0.9, -0.4, -1.0))), replace=True
    )

    assert "0.9000000" in result
    assert "0.2000000" not in result
    assert "  lift:" in result
    assert result.count("  stage:") == 1


def test_replacing_something_that_is_not_there_is_refused():
    with pytest.raises(CaptureRefused, match="the block is empty"):
        insert_preset(FILE, "stage", render_preset(pose()), replace=True)

    once = insert_preset(FILE, "stage", render_preset(pose()))
    with pytest.raises(CaptureRefused, match="no preset named 'lift'"):
        insert_preset(once, "lift", render_preset(pose("lift")), replace=True)


def test_a_file_without_the_block_is_not_restructured():
    with pytest.raises(CaptureRefused, match="no top-level"):
        insert_preset("schema_version: 1\n", "stage", render_preset(pose()))


def test_an_inline_populated_block_is_refused_rather_than_guessed():
    inline = FILE.replace("named_positions: {}", "named_positions: {stage: {joints: []}}")

    with pytest.raises(CaptureRefused, match="inline value"):
        insert_preset(inline, "lift", render_preset(pose("lift")))


# ── saving ─────────────────────────────────────────────────────────────────────────


def test_saving_produces_a_file_the_strict_loader_accepts(tmp_path: Path):
    path = written(tmp_path)

    save_preset(path, pose())

    loaded = RobotParameters.from_yaml(path)
    assert loaded.named_positions["stage"].joints == pytest.approx((0.2, -0.4, -1.0))
    assert loaded.named_positions["stage"].approved_by == "operator"


def test_a_pose_from_another_calibration_is_refused(tmp_path: Path):
    path = written(tmp_path)
    before = path.read_text()
    other = CapturedPose(
        name="stage",
        joints=(0.2, -0.4, -1.0),
        calibration_id="maker-arm-02-2025-01-01",
        approved_by="operator",
        captured_at=date(2026, 8, 29),
    )

    with pytest.raises(CaptureRefused, match="different physical poses"):
        save_preset(path, other)

    assert path.read_text() == before


def test_a_pose_with_no_feedback_is_refused(tmp_path: Path):
    path = written(tmp_path)
    before = path.read_text()

    with pytest.raises(CaptureRefused, match="no feedback"):
        save_preset(path, pose(joints=(0.2, float("nan"), -1.0)))

    assert path.read_text() == before


def test_the_wrong_number_of_joints_is_refused(tmp_path: Path):
    path = written(tmp_path)

    with pytest.raises(CaptureRefused, match="4 joint values but this arm has 3"):
        save_preset(path, pose(joints=(0.0, 0.0, 0.0, 0.0)))


def test_a_file_that_does_not_load_is_never_edited(tmp_path: Path):
    path = written(tmp_path, FILE.replace("units: radians", "units: furlongs"))
    before = path.read_text()

    with pytest.raises(CaptureRefused, match="does not load as it stands"):
        save_preset(path, pose())

    assert path.read_text() == before


def test_an_edit_that_does_not_read_back_is_rolled_back(tmp_path: Path, monkeypatch):
    """The verify-and-restore step is what makes editing the file as text acceptable."""

    path = written(tmp_path)
    before = path.read_text()
    monkeypatch.setattr(
        "everest_robot.robot.capture.render_preset",
        lambda _pose: "  stage:\n    joints: [0.2, -0.4]\n    calibration_id: x\n",
    )

    with pytest.raises(CaptureRefused, match="restored unchanged"):
        save_preset(path, pose())

    assert path.read_text() == before


def test_saving_twice_needs_replace_and_then_updates_in_place(tmp_path: Path):
    path = written(tmp_path)
    save_preset(path, pose())

    with pytest.raises(CaptureRefused, match="re-approval"):
        save_preset(path, pose(joints=(0.9, -0.4, -1.0)))

    save_preset(path, pose(joints=(0.9, -0.4, -1.0)), replace=True)

    loaded = RobotParameters.from_yaml(path)
    assert loaded.named_positions["stage"].joints == pytest.approx((0.9, -0.4, -1.0))
    assert len(loaded.named_positions) == 1


def test_no_temporary_file_is_left_behind(tmp_path: Path):
    path = written(tmp_path)

    save_preset(path, pose())

    assert [child.name for child in tmp_path.iterdir()] == ["maker_arm_v1.yaml"]
