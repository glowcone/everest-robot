"""The wrist camera's calibration: what it measures, what it refuses, what it stores.

The Jacobian is checked against a synthetic linear camera, because that is the one case
where the right answer is known exactly -- if bumping a joint moves the image by a matrix
we chose, the fit has to return that matrix. Everything else here is a refusal: the
calibration's job is as much to say "not on this arm", "not that far" and "not from this
file" as it is to produce a step.
"""

import json

import numpy as np
import pytest

from everest_robot.pixel_map import RobotStamp
from everest_robot.robot.wrist_servo import (
    DEFAULT_GAIN,
    DEFAULT_TOLERANCE,
    FEATURE_NAMES,
    BumpTrial,
    UnsupportedSolve,
    WristServoCalibration,
    WristServoDraft,
    WristServoError,
    feature_error,
    features_of,
    fit_jacobian,
    wrap_deg,
)

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
SERVO = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
GOAL = (320.0, 240.0, 150.0, 10.0)

#: A plausible wrist view: pan sweeps the image sideways, lift and elbow move it
#: vertically and change apparent size, roll rotates the spine one-for-one.
JACOBIAN = np.array(
    [
        [-220.0, 10.0, 5.0, 0.0, 2.0],
        [8.0, 180.0, -90.0, 6.0, 1.0],
        [1.0, 40.0, 70.0, 3.0, 0.0],
        [0.5, 0.0, 0.0, 1.0, 57.3],
    ]
)


class Detection:
    """The attributes ``features_of`` reads off a ``carabiner_detect`` result."""

    def __init__(self, insert, area, spine_angle, spine=None, aperture=None):
        self.insert = insert
        self.spine = spine or (insert[0] + 20.0, insert[1])
        self.aperture = aperture or (insert[0] - 10.0, insert[1])
        self.area = area
        self.spine_angle = spine_angle


def calibration(**overrides) -> WristServoCalibration:
    settings = dict(
        robot=RobotStamp("maker-arm-02", "maker-arm-02-2026-08-20"),
        camera_name="wrist",
        joint_names=JOINTS,
        servo_joints=SERVO,
        goal=GOAL,
        jacobian=JACOBIAN,
        trials=(BumpTrial("shoulder_pan", 0.08, GOAL, GOAL),),
    )
    settings.update(overrides)
    return WristServoCalibration(**settings)


# ── features ───────────────────────────────────────────────────────────────────────
def test_scale_is_the_square_root_of_the_mask_area():
    """Square root, not area: it has to move linearly with the other three features or the
    normalized least squares is weighting a quadratic against three linears."""

    features = features_of(Detection((100.0, 50.0), 2500.0, 12.0))

    assert features == (100.0, 50.0, 50.0, 12.0)


def test_a_detection_with_no_area_is_refused_rather_than_scaled():
    with pytest.raises(WristServoError, match="no mask area"):
        features_of(Detection((100.0, 50.0), 0.0, 12.0))


def test_the_spine_difference_folds_across_the_ninety_degree_seam():
    """The detector's spine angle is undirected, so +89 and -89 are two degrees apart. An
    unwrapped subtraction would call that 178 and swing the wrist most of a half turn."""

    error = feature_error((320.0, 240.0, 150.0, 89.0), (320.0, 240.0, 150.0, -89.0))

    assert error[3] == pytest.approx(-2.0)


def test_only_the_spine_feature_is_wrapped():
    error = feature_error((320.0 + 200.0, 240.0, 150.0, 10.0), GOAL)

    assert error[0] == pytest.approx(200.0)


# ── fitting ────────────────────────────────────────────────────────────────────────
def bump(joint: str, delta: float) -> BumpTrial:
    """One trial through the synthetic camera, so the fit's right answer is known."""

    column = SERVO.index(joint)
    after = np.array(GOAL) + JACOBIAN[:, column] * delta
    after[3] = wrap_deg(after[3])
    return BumpTrial(joint, delta, GOAL, tuple(after))


def test_the_fit_recovers_the_jacobian_it_was_bumped_through():
    trials = [bump(joint, sign * 0.08) for joint in SERVO for sign in (1, -1)]

    fitted, report = fit_jacobian(trials, SERVO)

    assert np.allclose(fitted, JACOBIAN)
    assert report["wrist_roll"]["trials"] == 2


def test_a_joint_with_no_bumps_is_refused_rather_than_left_as_a_zero_column():
    """A zero column does not read as missing evidence; it reads as a joint that provably
    does not move the image, and the solve would confidently never use it."""

    trials = [bump(joint, 0.08) for joint in SERVO if joint != "elbow_flex"]

    with pytest.raises(WristServoError, match="elbow_flex.*no bump trials"):
        fit_jacobian(trials, SERVO)


def test_a_disagreeing_column_is_reported_rather_than_hidden():
    """Two bumps of one joint that disagree mean something moved that should not have.
    The fit still returns -- averaging is the right response -- but it has to say so."""

    good = bump("shoulder_pan", 0.08)
    contaminated = BumpTrial(
        "shoulder_pan", -0.08, GOAL, tuple(np.array(GOAL) + np.array([40.0, 0.0, 0.0, 0.0]))
    )
    trials = [good, contaminated] + [bump(joint, 0.08) for joint in SERVO[1:]]

    _, report = fit_jacobian(trials, SERVO)

    assert report["shoulder_pan"]["residual_rms"][0] > 10.0
    assert report["shoulder_lift"]["residual_rms"][0] == pytest.approx(0.0, abs=1e-9)


def test_a_draft_fits_from_its_own_trials_and_keeps_the_return_check():
    draft = WristServoDraft(
        robot=RobotStamp("maker-arm-02", "maker-arm-02-2026-08-20"),
        camera_name="wrist",
        joint_names=JOINTS,
        servo_joints=SERVO,
        goal=GOAL,
    )
    for joint in SERVO:
        trial = bump(joint, 0.08)
        draft.record(joint, trial.delta_rad, trial.before, trial.after)
    draft.return_error = (1.0, -2.0, 0.5, 0.25)

    fitted = draft.fitted(gain=None, approved_by="operator")

    assert np.allclose(fitted.jacobian, JACOBIAN)
    # A `None` override is an unpassed flag, not a request for no gain.
    assert fitted.gain == DEFAULT_GAIN
    assert fitted.approved_by == "operator"
    assert fitted.validation["return_error"] == [1.0, -2.0, 0.5, 0.25]


# ── solving ────────────────────────────────────────────────────────────────────────
def test_the_solve_walks_the_image_onto_the_goal():
    """The whole point, checked end to end through the same synthetic camera: iterate the
    solve and the image has to arrive, not merely move in a plausible direction."""

    fitted = calibration()
    features = np.array([360.0, 265.0, 130.0, 30.0])

    for _ in range(30):
        solve = fitted.solve(tuple(features))
        if solve.settled:
            break
        features = features + JACOBIAN @ np.array([solve.delta_rad[j] for j in SERVO])
        features[3] = wrap_deg(features[3])
    else:
        pytest.fail(f"never settled; ended at {features}")

    assert fitted.settled(tuple(features))


def test_settling_is_judged_per_feature_not_on_a_pooled_distance():
    """A pooled norm lets a large spine error hide behind three tight position ones. Every
    feature has to be inside its own tolerance, because the clip policy needs all four."""

    fitted = calibration()
    off_spine = (GOAL[0], GOAL[1], GOAL[2], GOAL[3] + 20.0)

    assert fitted.settled(GOAL)
    assert not fitted.settled(off_spine)


def test_a_step_past_the_taught_range_is_refused_rather_than_clamped():
    """A clamp would turn "this Jacobian cannot answer that" into a small confident move in
    a direction nothing measured. The follower holds on this and stays visible."""

    fitted = calibration(max_delta_rad=0.05)

    with pytest.raises(UnsupportedSolve, match="past the 0.050 rad"):
        fitted.solve((320.0 + 400.0, 240.0, 150.0, 10.0))


def test_the_solve_normalizes_by_tolerance_rather_than_mixing_pixels_with_degrees():
    """Doubling a feature's tolerance has to make the servo care less about it. Without
    normalization the units alone would decide the weighting."""

    positional = DEFAULT_TOLERANCE[:3]
    tight = calibration(tolerance=(*positional, 1.0))
    loose = calibration(tolerance=(*positional, 40.0))
    off_spine = (GOAL[0], GOAL[1], GOAL[2], GOAL[3] + 25.0)

    assert abs(tight.solve(off_spine).delta_rad["wrist_roll"]) > abs(
        loose.solve(off_spine).delta_rad["wrist_roll"]
    )


def test_an_untaught_joint_holds_where_it_is_measured():
    """The calibration measures a derivative, not a pose. A joint outside `servo_joints` is
    one it has no opinion about, so its target is exactly its feedback."""

    fitted = calibration()
    measured = [0.1, 0.2, 0.3, 0.4, 0.5, 0.9]

    target = fitted.joint_target(fitted.solve((360.0, 265.0, 130.0, 30.0)), measured)

    assert target[JOINTS.index("gripper")] == pytest.approx(0.9)
    assert target[0] != pytest.approx(0.1)


def test_a_measured_pose_of_the_wrong_length_is_refused():
    fitted = calibration()

    with pytest.raises(WristServoError, match="expected 6 measured joints"):
        fitted.joint_target(fitted.solve(GOAL), [0.0, 0.0])


# ── the file ───────────────────────────────────────────────────────────────────────
def test_a_calibration_taught_on_another_arm_is_refused():
    """A derivative is only a derivative of the machine and zeroing it was measured on."""

    with pytest.raises(WristServoError, match="Re-teach it"):
        calibration().verify(RobotStamp("maker-arm-03", "maker-arm-03-2026-08-20"))


def test_a_servo_joint_this_arm_does_not_have_is_refused():
    with pytest.raises(WristServoError, match="unknown servo joint"):
        calibration(servo_joints=("shoulder_pan", "third_elbow"))


def test_a_jacobian_that_does_not_match_the_declared_shape_is_refused():
    with pytest.raises(WristServoError, match="4x5 but"):
        calibration(servo_joints=SERVO[:3])


def test_the_file_round_trips(tmp_path):
    fitted = calibration()

    path = fitted.save(tmp_path / "wrist_servo.json")
    back = WristServoCalibration.load(path)

    assert np.allclose(back.jacobian, fitted.jacobian)
    assert back.goal == fitted.goal
    assert back.feature_names == FEATURE_NAMES
    assert back.trials == fitted.trials


def test_an_older_schema_is_refused_rather_than_read_optimistically(tmp_path):
    path = tmp_path / "wrist_servo.json"
    document = calibration().to_json()
    document["schema_version"] = 0
    path.write_text(json.dumps(document))

    with pytest.raises(WristServoError, match="schema_version 0"):
        WristServoCalibration.load(path)


def test_a_missing_file_names_the_command_that_makes_one(tmp_path):
    with pytest.raises(WristServoError, match="robot-wrist-servo teach"):
        WristServoCalibration.load(tmp_path / "absent.json")


# ── the operator CLI ───────────────────────────────────────────────────────────────
def test_a_measurement_is_the_median_so_one_leaked_mask_cannot_move_it():
    """The failure mode of a threshold segmentation is not noise around the truth; it is
    an occasional frame that is wrong by a lot. A mean would carry that into the column."""

    from everest_robot.calibrate_wrist_servo import _measure

    frames = [
        Detection((320.0, 240.0), 22500.0, 10.0),
        Detection((321.0, 241.0), 22500.0, 11.0),
        Detection((900.0, 20.0), 400.0, -80.0),  # mask leaked into a finger
        Detection((319.0, 239.0), 22500.0, 9.0),
        Detection((320.0, 240.0), 22500.0, 10.0),
    ]

    class Reader:
        def __init__(self):
            self.left = list(frames)

        def frame(self):
            return self.left.pop(0)

        @staticmethod
        def detect(frame):
            return frame

    measured = _measure(Reader(), 5)

    assert measured[0] == pytest.approx(320.0)
    assert measured[2] == pytest.approx(150.0)
    assert measured[3] == pytest.approx(10.0)


def test_the_median_unwraps_the_spine_before_combining_it():
    """Two readings straddling the +/-90 fold average to the perpendicular unless the
    unwrap happens first, which would teach a goal the carabiner was never at."""

    from everest_robot.calibrate_wrist_servo import _measure

    class Reader:
        def __init__(self):
            self.left = [
                Detection((320.0, 240.0), 22500.0, 89.0),
                Detection((320.0, 240.0), 22500.0, -89.0),
                Detection((320.0, 240.0), 22500.0, 89.0),
            ]

        def frame(self):
            return self.left.pop(0)

        @staticmethod
        def detect(frame):
            return frame

    assert wrap_deg(_measure(Reader(), 3)[3]) == pytest.approx(89.0)


def test_check_refuses_a_file_that_was_never_taught_on_an_arm(tmp_path, capsys):
    """`check` is the command an operator runs before trusting a file, so a calibration
    with no measured evidence behind it has to exit non-zero, not merely print."""

    import argparse

    from everest_robot.calibrate_wrist_servo import cmd_check

    path = calibration(trials=()).save(tmp_path / "wrist_servo.json")

    code = cmd_check(argparse.Namespace(config=str(path), samples=1, fake=False))

    assert code == 1
    assert "cannot have been taught on an arm" in capsys.readouterr().out


# ── which point u, v track ─────────────────────────────────────────────────────────
def test_the_feature_point_selects_which_detected_pixel_is_servoed():
    """All three are points on one rigid object. Which one is aimed at is a choice about
    where the gripper should end up, not an implementation detail of the detector."""

    detection = Detection((100.0, 50.0), 2500.0, 12.0)

    assert features_of(detection, "insert")[:2] == (100.0, 50.0)
    assert features_of(detection, "spine")[:2] == (120.0, 50.0)
    assert features_of(detection, "aperture")[:2] == (90.0, 50.0)


def test_an_unknown_feature_point_is_refused_rather_than_defaulted():
    with pytest.raises(WristServoError, match="unknown feature point"):
        features_of(Detection((100.0, 50.0), 2500.0, 12.0), "handle")


def test_a_calibration_measures_detections_with_its_own_taught_point():
    detection = Detection((100.0, 50.0), 2500.0, 12.0)

    assert calibration(point="spine").features(detection)[:2] == (120.0, 50.0)


# ── retargeting, for the debug loop ────────────────────────────────────────────────
def test_retargeting_changes_the_goal_and_keeps_the_measured_jacobian():
    """The Jacobian is evidence about how the arm moves the image, and aiming somewhere
    else does not make it less true. Only what counts as arrived changes."""

    fitted = calibration()

    aimed = fitted.retargeted(goal=(10.0, 20.0, 30.0, 40.0), point="spine")

    assert np.allclose(aimed.jacobian, fitted.jacobian)
    assert aimed.trials == fitted.trials
    assert aimed.goal == (10.0, 20.0, 30.0, 40.0)
    assert aimed.point == "spine"


def test_an_ignored_tolerance_takes_a_feature_out_of_the_servo_without_a_special_case():
    """The debug loop centres translation only. A very loose tolerance is how that is said,
    and it has to fall out of the existing normalization rather than needing a branch."""

    from everest_robot.robot.wrist_servo import IGNORED_TOLERANCE

    aimed = calibration(
        tolerance=(DEFAULT_TOLERANCE[0], DEFAULT_TOLERANCE[1], IGNORED_TOLERANCE, IGNORED_TOLERANCE)
    )
    way_off_in_range_and_rotation = (GOAL[0], GOAL[1], GOAL[2] + 60.0, GOAL[3] + 45.0)

    assert aimed.settled(way_off_in_range_and_rotation)
    assert all(
        abs(value) < 1e-3 for value in aimed.solve(way_off_in_range_and_rotation).delta_rad.values()
    )


def test_the_feature_point_survives_the_file():
    fitted = calibration(point="spine")

    assert WristServoCalibration.from_json(fitted.to_json()).point == "spine"
