import hashlib
from datetime import date
from pathlib import Path

import pytest
import yaml

from everest_robot.robot.contracts import RobotIdentity
from everest_robot.robot.parameters import (
    IdentityMismatch,
    ParameterError,
    RobotParameters,
)

SHIPPED_CONFIG = Path(__file__).resolve().parents[2] / "config" / "maker_arm_v1.yaml"

JOINTS = ["shoulder_pan", "shoulder_lift", "gripper"]
CALIBRATION = "maker-arm-02-2026-08-20"


def document(**overrides: object) -> dict:
    base = {
        "schema_version": 1,
        "robot": {
            "id": "maker-arm-02",
            "model": "maker-arm-v1",
            "calibration_id": CALIBRATION,
            "joint_order": list(JOINTS),
            "units": "radians",
        },
        "motion_defaults": {
            "max_velocity_rad_s": 0.5,
            "max_acceleration_rad_s2": 1.0,
            "tolerance_rad": 0.03,
            "settle_time_s": 0.25,
            "timeout_s": 10.0,
            "control_rate_hz": 30,
        },
        "named_positions": {},
        "named_transitions": {},
        "policy": {"default_controller": "vla", "fps": 30, "max_duration_s": 30},
        "replay": {
            "require_matching_robot_id": True,
            "require_matching_calibration_id": True,
            "safe_start_position": None,
            "max_speed_scale": 1.0,
        },
    }
    base.update(overrides)  # type: ignore[arg-type]
    return base


def preset(**overrides: object) -> dict:
    entry = {
        "joints": [0.1, -0.2, 0.0],
        "calibration_id": CALIBRATION,
        "approved_by": "operator",
        "captured_at": "2026-08-21",
    }
    entry.update(overrides)  # type: ignore[arg-type]
    return entry


def load(doc: dict) -> RobotParameters:
    return RobotParameters.from_mapping(doc, config_digest="sha256:test", source="test.yaml")


def test_shipped_config_loads_and_defines_no_presets() -> None:
    parameters = RobotParameters.from_yaml(SHIPPED_CONFIG)

    # Poses may only come from a measured, operator-approved arm state.
    assert parameters.named_positions == {}
    assert parameters.named_transitions == {}
    assert parameters.identity.units == "radians"
    assert parameters.replay.safe_start_position is None


def test_digest_is_the_file_content_hash() -> None:
    parameters = RobotParameters.from_yaml(SHIPPED_CONFIG)

    expected = hashlib.sha256(SHIPPED_CONFIG.read_bytes()).hexdigest()

    assert parameters.config_digest == f"sha256:{expected}"


def test_unknown_fields_are_rejected_at_every_level() -> None:
    with pytest.raises(ParameterError, match="unknown field"):
        load(document(extra_section={}))
    with pytest.raises(ParameterError, match="unknown field"):
        load(document(robot={**document()["robot"], "serial": "x"}))
    with pytest.raises(ParameterError, match="unknown field"):
        # A misspelled override would otherwise silently run at the default speed.
        load(document(named_positions={"ready": preset(max_velocity_rad=0.1)}))


def test_schema_version_must_match() -> None:
    with pytest.raises(ParameterError, match="schema_version"):
        load(document(schema_version=2))


def test_preset_from_another_calibration_is_rejected() -> None:
    with pytest.raises(ParameterError, match="recapture"):
        load(document(named_positions={"ready": preset(calibration_id="maker-arm-02-2025-01-01")}))


def test_preset_provenance_is_mandatory() -> None:
    entry = preset()
    del entry["approved_by"]
    with pytest.raises(ParameterError, match="approved_by"):
        load(document(named_positions={"ready": entry}))


def test_preset_must_cover_every_joint_exactly_once() -> None:
    with pytest.raises(ParameterError, match="one value per joint"):
        load(document(named_positions={"ready": preset(joints=[0.1, 0.2])}))


def test_preset_values_must_be_finite_numbers() -> None:
    with pytest.raises(ParameterError, match="finite"):
        load(document(named_positions={"ready": preset(joints=[0.1, float("nan"), 0.0])}))
    with pytest.raises(ParameterError, match="expected a number"):
        load(document(named_positions={"ready": preset(joints=[0.1, "0.2", 0.0])}))


def test_duplicate_joint_order_entries_are_rejected() -> None:
    with pytest.raises(ParameterError, match="duplicate"):
        load(document(robot={**document()["robot"], "joint_order": ["a", "b", "a"]}))


def test_only_radians_are_accepted() -> None:
    with pytest.raises(ParameterError, match="radians"):
        load(document(robot={**document()["robot"], "units": "degrees"}))


def test_preset_overrides_inherit_the_remaining_defaults() -> None:
    parameters = load(
        document(named_positions={"ready": preset(max_velocity_rad_s=0.2, timeout_s=12.0)})
    )

    profile = parameters.position("ready").profile

    assert profile.max_velocity_rad_s == pytest.approx(0.2)
    assert profile.timeout_s == pytest.approx(12.0)
    assert profile.tolerance_rad == pytest.approx(parameters.motion_defaults.tolerance_rad)
    assert parameters.position("ready").captured_at == date(2026, 8, 21)


def test_transitions_must_reference_known_positions() -> None:
    doc = document(
        named_positions={"ready": preset()},
        named_transitions={"pickup_to_ready": {"waypoints": ["ready", "nowhere"]}},
    )
    with pytest.raises(ParameterError, match="not a named position"):
        load(doc)


def test_transitions_need_at_least_two_distinct_waypoints() -> None:
    with pytest.raises(ParameterError, match="at least two"):
        load(document(named_positions={"ready": preset()},
                      named_transitions={"t": {"waypoints": ["ready"]}}))
    with pytest.raises(ParameterError, match="repeats"):
        load(document(named_positions={"ready": preset()},
                      named_transitions={"t": {"waypoints": ["ready", "ready"]}}))


def test_replay_speed_may_not_exceed_the_recorded_session() -> None:
    replay = {**document()["replay"], "max_speed_scale": 1.5}
    with pytest.raises(ParameterError, match="max_speed_scale"):
        load(document(replay=replay))


def test_unknown_position_lookup_lists_what_is_known() -> None:
    parameters = load(document(named_positions={"ready": preset()}))

    with pytest.raises(ParameterError, match="known: ready"):
        parameters.position("missing")


def test_identity_is_verified_against_the_connected_arm() -> None:
    parameters = load(document())
    hardware = RobotIdentity("maker-arm-02", "maker-arm-v1", CALIBRATION, tuple(JOINTS))

    parameters.verify_identity(hardware)

    recalibrated = RobotIdentity("maker-arm-02", "maker-arm-v1", "other", tuple(JOINTS))
    with pytest.raises(IdentityMismatch, match="calibration_id"):
        parameters.verify_identity(recalibrated)


def test_yaml_round_trip_matches_the_mapping_loader(tmp_path: Path) -> None:
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(document(named_positions={"ready": preset()})))

    parameters = RobotParameters.from_yaml(path)

    assert parameters.position("ready").joints == (0.1, -0.2, 0.0)
    assert parameters.source == str(path)
