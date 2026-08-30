import math

import numpy as np
import pytest

from everest_robot.pixel_map import (
    MIN_SAMPLES,
    QUADRATIC,
    CameraSource,
    DetectorSpec,
    OutsideCalibratedRegion,
    PixelJointMap,
    PixelMapError,
    RobotStamp,
    Sample,
    convex_hull,
    fit_quadratic,
    fit_thin_plate_spline,
    hull_margin_px,
)

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
FIT_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")


def truth(u: float, v: float) -> tuple[float, float, float, float]:
    """A smooth, mildly non-linear stand-in for the arm's real pixel-to-pose surface."""

    x = (u - 320.0) / 320.0
    y = (v - 240.0) / 240.0
    return (
        0.8 * x + 0.1 * y * y,
        -0.6 * y + 0.05 * x * y,
        0.3 * x * x - 0.4 * y,
        0.2 * x - 0.2 * y,
    )


def grid_samples(columns: int = 6, rows: int = 5, spine: bool = True) -> list[Sample]:
    samples = []
    for column in range(columns):
        for row in range(rows):
            u = 120.0 + column * 80.0
            v = 90.0 + row * 75.0
            pan, lift, elbow, flex = truth(u, v)
            # wrist_roll follows the planted roll law; gripper is not a function of pixel.
            spine_rad = 0.4 * math.sin(u / 200.0) if spine else None
            roll = 0.0 if spine_rad is None else _wrap(spine_rad - pan + 0.25)
            samples.append(
                Sample(
                    pixel=(u, v),
                    joints=(pan, lift, elbow, flex, roll, -1.0),
                    spine_rad=spine_rad,
                )
            )
    return samples


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def calibration(samples: list[Sample] | None = None) -> PixelJointMap:
    return PixelJointMap(
        camera=CameraSource(index_or_path="0", backend="auto"),
        detector=DetectorSpec(kind="two-white-black", roi_xywh=(100, 80, 400, 380)),
        robot=RobotStamp("maker-arm-02", "maker-arm-02-2026-08-20", "sha256:abc"),
        joint_names=JOINTS,
        samples=tuple(grid_samples() if samples is None else samples),
        approach_offset_rad=tuple(0.05 for _ in JOINTS),
    )


# ── the fits ───────────────────────────────────────────────────────────────────────
def test_thin_plate_spline_reproduces_every_sample():
    samples = _reduced(grid_samples())
    model = fit_thin_plate_spline(samples, FIT_JOINTS)
    predicted = model.predict([sample.pixel for sample in samples])
    expected = np.asarray([sample.joints for sample in samples])
    assert np.allclose(predicted, expected, atol=1e-9)


def test_thin_plate_spline_interpolates_between_samples():
    model = fit_thin_plate_spline(_reduced(grid_samples()), FIT_JOINTS)
    # A point in the middle of four taught cells, which no sample sits on.
    predicted = model.predict([[240.0, 165.0]])[0]
    assert np.allclose(predicted, truth(240.0, 165.0), atol=0.02)


def test_smoothing_trades_sample_fidelity_for_flatness():
    samples = _reduced(grid_samples())
    smoothed = fit_thin_plate_spline(samples, FIT_JOINTS, smoothing=1.0)
    residual = np.abs(
        smoothed.predict([sample.pixel for sample in samples])
        - np.asarray([sample.joints for sample in samples])
    ).max()
    assert residual > 1e-6


def test_quadratic_recovers_a_quadratic_field():
    model = fit_quadratic(_reduced(grid_samples()), FIT_JOINTS, ridge=1e-9)
    assert np.allclose(model.predict([[240.0, 165.0]])[0], truth(240.0, 165.0), atol=1e-6)


def test_collinear_samples_are_refused_with_an_actionable_message():
    samples = [Sample(pixel=(float(index) * 10.0, 0.0), joints=(0.0,)) for index in range(6)]
    with pytest.raises(PixelMapError, match="collinear or duplicated"):
        fit_thin_plate_spline(samples, ("shoulder_pan",))


def _reduced(samples: list[Sample]) -> list[Sample]:
    return [
        Sample(pixel=sample.pixel, joints=sample.joints[:4], spine_rad=sample.spine_rad)
        for sample in samples
    ]


# ── the region the fit is trusted in ───────────────────────────────────────────────
def test_convex_hull_of_a_grid_is_its_four_corners():
    hull = convex_hull([sample.pixel for sample in grid_samples()])
    assert len(hull) == 4
    assert {(120.0, 90.0), (520.0, 90.0), (520.0, 390.0), (120.0, 390.0)} == {
        (float(u), float(v)) for u, v in hull
    }


def test_hull_margin_is_positive_inside_and_negative_outside():
    hull = convex_hull([sample.pixel for sample in grid_samples()])
    assert hull_margin_px((320.0, 240.0), hull) > 0.0
    assert hull_margin_px((120.0, 90.0), hull) == pytest.approx(0.0, abs=1e-9)
    assert hull_margin_px((520.0, 500.0), hull) == pytest.approx(-110.0, abs=1e-9)


def test_prediction_outside_the_taught_region_is_refused_not_extrapolated():
    fitted = calibration().fitted(fit_joints=FIT_JOINTS)
    with pytest.raises(OutsideCalibratedRegion, match="outside the calibrated region"):
        fitted.predict((600.0, 240.0))


def test_required_margin_rejects_pixels_near_the_hull_edge():
    fitted = calibration().fitted(fit_joints=FIT_JOINTS)
    edged = PixelJointMap(
        camera=fitted.camera,
        detector=fitted.detector,
        robot=fitted.robot,
        joint_names=fitted.joint_names,
        samples=fitted.samples,
        model=fitted.model,
        hull=fitted.hull,
        required_margin_px=30.0,
        roll=fitted.roll,
    )
    edged.predict((320.0, 240.0))
    with pytest.raises(OutsideCalibratedRegion):
        edged.predict((125.0, 240.0))


# ── wrist roll ─────────────────────────────────────────────────────────────────────
def test_roll_offset_and_sign_are_recovered_from_the_samples():
    fitted = calibration().fitted(fit_joints=FIT_JOINTS)
    assert fitted.roll is not None
    assert fitted.roll.sign == 1
    assert math.degrees(fitted.roll.offset_rad) == pytest.approx(math.degrees(0.25), abs=0.5)
    assert fitted.roll.residual_std_deg < 1.0


def test_roll_is_predicted_only_when_a_spine_angle_is_supplied():
    fitted = calibration().fitted(fit_joints=FIT_JOINTS)
    assert fitted.predict((320.0, 240.0)).roll_rad is None
    assert "wrist_roll" not in fitted.predict((320.0, 240.0)).joints

    spine = 0.4 * math.sin(320.0 / 200.0)
    prediction = fitted.predict((320.0, 240.0), spine)
    expected = _wrap(spine - prediction.joints["shoulder_pan"] + 0.25)
    assert prediction.roll_rad == pytest.approx(expected, abs=0.02)


def test_no_spine_angles_means_no_roll_model():
    fitted = calibration(grid_samples(spine=False)).fitted(fit_joints=FIT_JOINTS)
    assert fitted.roll is None


# ── turning a prediction into a command ────────────────────────────────────────────
def test_unfitted_joints_hold_their_measured_value():
    fitted = calibration().fitted(fit_joints=FIT_JOINTS)
    measured = (0.0, 0.0, 0.0, 0.0, 0.0, -1.7)
    target = fitted.full_target(fitted.predict((320.0, 240.0)), measured)
    assert len(target) == len(JOINTS)
    # wrist_roll and gripper were not fitted and no spine angle was given.
    assert target[4] == 0.0
    assert target[5] == -1.7
    assert target[0] != 0.0


def test_approach_pose_backs_off_every_joint_in_one_direction():
    fitted = calibration().fitted(fit_joints=FIT_JOINTS)
    target = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    assert fitted.approach_pose(target) == pytest.approx(
        tuple(value - 0.05 for value in target)
    )


def test_full_target_rejects_a_measured_pose_of_the_wrong_width():
    fitted = calibration().fitted(fit_joints=FIT_JOINTS)
    with pytest.raises(PixelMapError, match="expected 6 measured joints"):
        fitted.full_target(fitted.predict((320.0, 240.0)), (0.0, 0.0))


# ── the file ───────────────────────────────────────────────────────────────────────
def test_round_trip_through_the_file_predicts_identically(tmp_path):
    fitted = calibration().fitted(fit_joints=FIT_JOINTS)
    path = fitted.save(tmp_path / "pixel_map.json")
    reloaded = PixelJointMap.load(path)

    spine = 0.3
    before = fitted.predict((300.0, 200.0), spine)
    after = reloaded.predict((300.0, 200.0), spine)
    assert after.joints == pytest.approx(before.joints)
    assert after.roll_rad == pytest.approx(before.roll_rad)
    assert reloaded.samples == fitted.samples
    assert reloaded.camera.index_or_path == "0"
    assert reloaded.approach_offset_rad == fitted.approach_offset_rad


def test_a_quadratic_fit_also_round_trips(tmp_path):
    fitted = calibration().fitted(fit_joints=FIT_JOINTS, kind=QUADRATIC)
    reloaded = PixelJointMap.load(fitted.save(tmp_path / "pixel_map.json"))
    assert reloaded.model is not None
    assert reloaded.model.kind == QUADRATIC
    assert reloaded.predict((300.0, 200.0)).joints == pytest.approx(
        fitted.predict((300.0, 200.0)).joints
    )


def test_an_unfitted_calibration_refuses_to_predict():
    with pytest.raises(PixelMapError, match="no fit yet"):
        calibration().predict((320.0, 240.0))


def test_fitting_refuses_below_the_minimum_sample_count():
    with pytest.raises(PixelMapError, match="not enough to fit"):
        calibration(grid_samples()[: MIN_SAMPLES - 1]).fitted(fit_joints=FIT_JOINTS)


def test_a_calibration_taught_on_another_arm_is_refused():
    stamp = RobotStamp("maker-arm-02", "maker-arm-02-2026-08-20")
    with pytest.raises(PixelMapError, match="Recapture the samples"):
        stamp.verify(RobotStamp("maker-arm-02", "maker-arm-02-2027-01-01"))


def test_unknown_fit_joints_are_named_in_the_refusal():
    with pytest.raises(PixelMapError, match="unknown joint"):
        calibration().fitted(fit_joints=("elbow_twist",))


def test_a_file_from_another_schema_version_is_refused(tmp_path):
    fitted = calibration().fitted(fit_joints=FIT_JOINTS)
    document = fitted.to_json()
    document["schema_version"] = 99
    path = tmp_path / "pixel_map.json"
    path.write_text(__import__("json").dumps(document))
    with pytest.raises(PixelMapError, match="schema_version"):
        PixelJointMap.load(path)


# ── validation ─────────────────────────────────────────────────────────────────────
def test_holdout_report_measures_generalization_not_memorization():
    report = calibration().fitted(fit_joints=FIT_JOINTS).validation
    assert report is not None
    assert report["held_out"] == 6
    assert report["trained_on"] == 24
    # The planted field is smooth, so a spline trained on 24 of 30 samples should predict
    # the rest to well under a degree. A regression here means the fit stopped working.
    assert max(report["rms_error_deg"].values()) < 1.0
    assert len(report["worst_sample_pixel"]) == 2
