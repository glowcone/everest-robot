"""Teach and use the wrist camera's image Jacobian -- the ``robot-wrist-servo`` CLI.

This is the calibration step the wrist-camera form of ``SEARCH_CV`` needs, and it is a
different procedure from ``robot-pixel-map`` because it measures a different thing. The
pixel map is taught by *sampling*: put the arm at thirty pre-grasp poses and pair each with
where the object appeared to a bolted-down camera. That works because a fixed camera's
pixels name places. The wrist camera's do not, so there is nothing to sample; what is
measured instead is a derivative, and a derivative is measured by moving one joint at a
time and watching what happens.

The procedure, in the order the subcommands run:

1. ``teach`` -- **MOVES THE ARM.** Put the arm where you want ``SEARCH_CV`` to hand over
   to the clip policy: gripper at pre-grasp over the carabiner, carabiner clearly in the
   wrist view. Everything below is measured from there.

   a. The image at that pose is recorded as the *goal*. This is the only place the goal
      comes from -- it is an operator attestation that "this is what handed-over looks
      like", the same standing as an approved named position.
   b. Each servo joint is bumped by ``--delta`` in each direction and returned, with the
      image measured at both ends. The *measured* joint displacement is used, never the
      commanded one, because backlash and tracking error make those different numbers and
      the derivative is only as good as its denominator.
   c. The arm returns to the start and the image is measured once more. If it has not come
      back to the goal within tolerance, something moved that should not have -- the
      carabiner was nudged, or the arm did not return -- and the whole teach is refused
      rather than saved with a quietly wrong goal in it.

2. ``check`` -- print the stored calibration, its per-column residuals and its return
   check. Touches no hardware.

3. ``look`` -- print what the wrist camera sees right now and how far it is from the goal,
   in both features and the joint step that would close it. Claims the arm to share its
   camera, but never enables it.

4. ``centre`` -- **MOVES THE ARM.** The debugging loop: slowly bring the carabiner's spine
   midpoint to the centre of the wrist frame, servoing translation only. This is the one to
   reach for first, because it is the one whose result can be checked by eye against the
   detector's overlay, and a Jacobian column with the wrong sign shows up immediately as the
   arm going the wrong way.

5. ``track`` -- **MOVES THE ARM.** The whole handover on the taught goal, outside the FSM.

Both servo commands take ``--dry-run``, which runs the entire loop with the arm never
energized.

Why bumps rather than one least-squares fit over an arbitrary set of poses: each trial here
moves exactly one joint, so each Jacobian column is identified independently and a joint
with a bad column -- backlash, or barely any effect on the image -- shows up as its own
residual instead of being smeared across its neighbours. See
:mod:`everest_robot.robot.wrist_servo` for the model and
:mod:`everest_robot.robot.wrist_follower` for what consumes it.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from collections.abc import Sequence
from typing import Any

from everest_robot.pixel_map import RobotStamp
from everest_robot.robot.wrist_servo import (
    DEFAULT_TOLERANCE,
    FEATURE_NAMES,
    FEATURE_POINTS,
    MM_PER_PX,
    UnsupportedSolve,
    WristServoCalibration,
    WristServoDraft,
    WristServoError,
    feature_error,
    features_of,
    wrap_deg,
)

#: Joints bumped by default. The gripper is excluded: opening and closing it changes the
#: image by occluding it, not by moving the camera, so its "derivative" would be an
#: artefact of the fingers entering the frame.
DEFAULT_SERVO_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")

#: How far each joint is moved to measure its column. Large enough that the image
#: displacement is well above detector noise, small enough that the linearization holds.
DEFAULT_DELTA_RAD = 0.08


# ── plumbing ───────────────────────────────────────────────────────────────────────
def _open_session(args: argparse.Namespace) -> Any:
    """Claim and connect the deployed arm, with its cameras. The caller closes it.

    Cameras, unlike ``robot-pixel-map``: the wrist camera *is* a policy observation, and
    this calibration is taught through the same ``CameraRuntime`` the FSM will read it
    through. Teaching through a second device handle would risk teaching through different
    exposure or white balance than the one that is used.
    """

    if getattr(args, "fake", False):
        from everest_robot.robot.cameras import CameraRuntime
        from everest_robot.robot.contracts import JointLimit
        from everest_robot.robot.deployment import load_parameters
        from everest_robot.robot.fake_arm import FakeArm
        from everest_robot.robot.session import RobotSession

        print("FAKE ARM: nothing below this line describes a physical robot.\n")
        parameters = load_parameters()
        limits = tuple(
            JointLimit(name, -2.0, 2.0) for name in parameters.identity.joint_names
        )
        return RobotSession(
            FakeArm(parameters.identity, limits), parameters, cameras=CameraRuntime.from_env()
        ).open()

    from everest_robot.robot.deployment import open_session

    return open_session()


def _roi(args: argparse.Namespace) -> Any:
    """The detector's search region: the flag if given, else EVEREST_WRIST_ROI."""

    from everest_robot.robot.deployment import wrist_roi

    given = getattr(args, "roi", None)
    return tuple(given) if given else wrist_roi()


def _stamp_of(session: Any) -> RobotStamp:
    identity = session.port.identity
    return RobotStamp(
        robot_id=identity.robot_id,
        calibration_id=identity.calibration_id,
        config_digest=getattr(session.parameters, "config_digest", ""),
    )


def _frames(session: Any, camera_name: str, color_mode: str) -> Any:
    from everest_robot.robot.wrist_follower import wrist_frames

    names = session.bridge.cameras.names
    if camera_name not in names:
        configured = ", ".join(names) or "none"
        raise WristServoError(
            f"camera {camera_name!r} is not configured (EVEREST_CAMERAS has: {configured})"
        )
    return wrist_frames(session.bridge.cameras, camera_name, color_mode)


def _measure(read: Any, samples: int) -> tuple[float, ...]:
    """The feature vector, as the median of ``samples`` consecutive detections.

    Median rather than mean because the failure mode of a threshold segmentation is not
    noise around the truth: it is an occasional frame where the mask leaked into a finger
    or a shadow and the answer is wrong by a lot. A mean carries that into the derivative;
    a median discards it. The spine is unwrapped against the first sample first, so a pair
    straddling the +/-90 fold does not average to the perpendicular.
    """

    from everest_robot.carabiner_detect import NotFound

    rows: list[tuple[float, ...]] = []
    misses = 0
    while len(rows) < samples:
        try:
            rows.append(features_of(read.detect(read.frame()), getattr(read, "point", "insert")))
        except (NotFound, ValueError, WristServoError) as error:
            misses += 1
            if misses > 2 * samples:
                raise WristServoError(
                    f"the carabiner was not detected in {misses} frames ({error}). Check the "
                    "wrist view, the lighting, and EVEREST_WRIST_CAMERA_COLOR"
                ) from None
    columns = list(zip(*rows, strict=True))
    reference = rows[0][FEATURE_NAMES.index("spine_deg")]
    return tuple(
        reference + statistics.median(wrap_deg(value - reference) for value in column)
        if name == "spine_deg"
        else statistics.median(column)
        for name, column in zip(FEATURE_NAMES, columns, strict=True)
    )


class _Reader:
    """One frame source plus the detector, so ``_measure`` needs a single argument."""

    def __init__(self, frames: Any, point: str = "insert", roi: Any = None) -> None:
        self.frame = frames
        self.point = point
        self.roi = roi

    def detect(self, frame: Any) -> Any:
        from everest_robot.carabiner_detect import detect

        return detect(frame, self.roi)


def _print_features(label: str, values: Sequence[float]) -> None:
    parts = " ".join(
        f"{name}={value:+.1f}" for name, value in zip(FEATURE_NAMES, values, strict=True)
    )
    print(f"{label:<12}{parts}")


def _print_error(error: Sequence[float], tolerance: Sequence[float]) -> None:
    print(f"{'feature':<12}{'error':>10}{'tolerance':>12}{'':>4}")
    for name, value, limit in zip(FEATURE_NAMES, error, tolerance, strict=True):
        unit = "deg" if name == "spine_deg" else "px"
        extra = "" if unit == "deg" else f" ({value * MM_PER_PX:+.1f} mm)"
        mark = "" if abs(value) <= limit else "  <- outside"
        print(f"{name:<12}{value:>+10.2f}{limit:>12.2f} {unit}{extra}{mark}")


def _confirm(prompt: str, word: str, skip: bool) -> None:
    if skip:
        return
    if input(f"{prompt} Type {word} to continue > ").strip() != word:
        raise WristServoError("cancelled")


# ── teach ──────────────────────────────────────────────────────────────────────────
def cmd_teach(args: argparse.Namespace) -> int:
    from everest_robot.robot.deployment import wrist_servo_path

    servo_joints = tuple(name.strip() for name in args.joints.split(",") if name.strip())
    if not servo_joints:
        raise WristServoError("--joints must name at least one joint")
    if not math.isfinite(args.delta) or args.delta <= 0.0:
        raise WristServoError("--delta must be finite and positive")
    approved_by = args.approved_by or input("who is approving this calibration? > ").strip()
    if not approved_by:
        raise WristServoError(
            "a wrist servo calibration records who stood at the bench and approved the goal "
            "pose; --approved-by is required"
        )

    session = _open_session(args)
    try:
        joint_names = tuple(session.port.joint_names)
        unknown = [name for name in servo_joints if name not in joint_names]
        if unknown:
            raise WristServoError(
                f"unknown joint(s) {', '.join(unknown)}; this arm has {', '.join(joint_names)}"
            )
        reader = _Reader(_frames(session, args.camera, args.color), args.point, _roi(args))

        print(
            "\nTEACH MOVES THE ARM. Each of "
            f"{', '.join(servo_joints)} will be moved +/-{args.delta:.3f} rad from where it "
            "stands now and returned. The pose the arm is in right now becomes the goal "
            "SEARCH_CV hands over at, so it must already be the pre-grasp pose over the "
            "carabiner.\n"
        )
        goal = _measure(reader, args.samples)
        _print_features("goal", goal)
        _confirm("Clear the area.", "TEACH", args.yes)

        draft = WristServoDraft(
            robot=_stamp_of(session),
            camera_name=args.camera,
            joint_names=joint_names,
            servo_joints=servo_joints,
            goal=goal,
            point=args.point,
            tolerance=tuple(args.tolerance) if args.tolerance else DEFAULT_TOLERANCE,
            color_mode=args.color,
        )
        home = tuple(session.snapshot().positions)
        _bump_all(session, reader, draft, home, args)

        # The arm is back at the start. If the image is not, the evidence just gathered was
        # gathered about a scene that changed underneath it.
        draft.return_error = _confirm_return(reader, draft, args)

        calibration = draft.fitted(
            gain=args.gain,
            damping=args.damping,
            max_delta_rad=args.max_delta,
            approved_by=approved_by,
        )
    finally:
        session.close()

    path = calibration.save(args.config or wrist_servo_path())
    print(f"\nwrote {path}")
    _print_calibration(calibration)
    return 0


def _bump_all(
    session: Any,
    reader: _Reader,
    draft: WristServoDraft,
    home: tuple[float, ...],
    args: argparse.Namespace,
) -> None:
    """Move each servo joint both ways from ``home``, measuring the image at both ends."""

    joint_names = draft.joint_names
    for joint in draft.servo_joints:
        index = joint_names.index(joint)
        for sign in (1.0, -1.0):
            before_joints = _return_to(session, home, args, label="home")
            before = _measure(reader, args.samples)

            target = list(home)
            target[index] = home[index] + sign * args.delta
            result = session.motion.go_to_joint_target(
                f"bump {joint} {sign:+.0f}", target, speed_scale=args.speed_scale
            )
            if not result.reached:
                raise WristServoError(
                    f"bumping {joint} by {sign * args.delta:+.3f} rad did not reach the "
                    f"target: {result.failure_reason} -- {result.failure_detail}"
                )
            after_joints = tuple(session.snapshot().positions)
            after = _measure(reader, args.samples)

            # The measured displacement, not the commanded one. A commanded 0.08 rad that
            # the arm answered with 0.071 would put a 12% error straight into the column.
            measured_delta = after_joints[index] - before_joints[index]
            if abs(measured_delta) < 0.25 * args.delta:
                raise WristServoError(
                    f"{joint} only moved {measured_delta:+.4f} rad of the "
                    f"{sign * args.delta:+.3f} asked for; check its limits and its driver"
                )
            draft.record(joint, measured_delta, before, after)
            error = feature_error(after, before)
            print(
                f"  {joint:<15}{measured_delta:+.4f} rad -> "
                f"du={error[0]:+7.1f} dv={error[1]:+7.1f} "
                f"dscale={error[2]:+6.1f} dspine={error[3]:+6.1f}"
            )
    _return_to(session, home, args, label="home")


def _return_to(
    session: Any, home: Sequence[float], args: argparse.Namespace, *, label: str
) -> tuple[float, ...]:
    result = session.motion.go_to_joint_target(label, home, speed_scale=args.speed_scale)
    if not result.reached:
        raise WristServoError(
            f"could not return to the start pose: {result.failure_reason} -- "
            f"{result.failure_detail}"
        )
    return tuple(session.snapshot().positions)


def _confirm_return(
    reader: _Reader, draft: WristServoDraft, args: argparse.Namespace
) -> tuple[float, ...]:
    """Refuse the whole teach if the goal image did not come back.

    This is the one check that can catch the failure that matters most and is otherwise
    invisible: the carabiner was nudged partway through. Every column measured after that
    point is about a different scene, and nothing in the numbers themselves would say so.

    The threshold is a multiple of the arrival tolerance rather than the tolerance itself,
    because the detector's own frame-to-frame spread is not zero and a teach that fails on
    noise would just be re-run until it passed, which is worse than a slightly loose gate.
    """

    returned = _measure(reader, args.samples)
    error = feature_error(returned, draft.goal)
    print()
    _print_error(error, draft.tolerance)
    outside = [
        name
        for name, value, limit in zip(FEATURE_NAMES, error, draft.tolerance, strict=True)
        if abs(value) > args.return_scale * limit
    ]
    if outside:
        raise WristServoError(
            f"the arm returned to the start pose but the image did not: "
            f"{', '.join(outside)} is outside {args.return_scale:g}x tolerance. The carabiner "
            "was probably moved during the teach. Nothing was saved; reset the scene and "
            "run teach again"
        )
    return error


# ── check / look ───────────────────────────────────────────────────────────────────
def _print_calibration(calibration: WristServoCalibration) -> None:
    print(f"\ntaught    {calibration.created_at}  by {calibration.approved_by or 'unrecorded'}")
    print(f"arm       {calibration.robot.robot_id} / {calibration.robot.calibration_id}")
    print(f"camera    {calibration.camera_name} ({calibration.color_mode})")
    print(f"scale     {MM_PER_PX:.2f} mm/px at the goal pose")
    _print_features("goal", calibration.goal)
    print(f"\n{'joint':<16}" + "".join(f"{name:>12}" for name in calibration.feature_names))
    columns = (calibration.validation or {}).get("columns", {})
    for index, joint in enumerate(calibration.servo_joints):
        row = "".join(
            f"{float(calibration.jacobian[feature][index]):>12.1f}"
            for feature in range(len(calibration.feature_names))
        )
        print(f"{joint:<16}{row}   per rad")
        residual = columns.get(joint, {}).get("residual_rms")
        if residual:
            print(
                f"{'':<16}"
                + "".join(f"{value:>12.1f}" for value in residual)
                + f"   residual rms over {columns[joint]['trials']} trials"
            )
    print(f"\ngain {calibration.gain}  damping {calibration.damping}  "
          f"max step {calibration.max_delta_rad} rad")
    print("tolerance " + " ".join(
        f"{name}={value:.1f}"
        for name, value in zip(calibration.feature_names, calibration.tolerance, strict=True)
    ))


def cmd_check(args: argparse.Namespace) -> int:
    from everest_robot.robot.deployment import wrist_servo_path

    calibration = WristServoCalibration.load(args.config or wrist_servo_path())
    _print_calibration(calibration)
    if not calibration.trials:
        print("\nWARNING: no bump trials recorded; this file cannot have been taught on an arm.")
        return 1
    return 0


def cmd_look(args: argparse.Namespace) -> int:
    """What the wrist camera sees now, against the goal. Claims the arm; never enables it."""

    from everest_robot.robot.deployment import wrist_servo_path

    calibration = WristServoCalibration.load(args.config or wrist_servo_path())
    session = _open_session(args)
    try:
        calibration.verify(_stamp_of(session))
        reader = _Reader(
            _frames(session, calibration.camera_name, calibration.color_mode),
            calibration.point,
            _roi(args),
        )
        features = _measure(reader, args.samples)
        _print_features("measured", features)
        _print_features("goal", calibration.goal)
        print()
        _print_error(feature_error(features, calibration.goal), calibration.tolerance)
        try:
            solve = calibration.solve(features)
        except UnsupportedSolve as error:
            print(f"\nno usable step: {error}")
            return 1
        print(f"\nnormalized error {solve.normalized_error:.2f} "
              f"({'settled' if solve.settled else 'not settled'})")
        for joint, value in solve.delta_rad.items():
            print(f"  {joint:<16}{value:+.4f} rad")
    finally:
        session.close()
    return 0


# ── centre: the SEARCH_CV loop, debugged one concern at a time ─────────────────────
def cmd_centre(args: argparse.Namespace) -> int:
    """MOVES THE ARM: slowly bring the carabiner's spine midpoint to the frame centre.

    This is ``pixel-track``'s counterpart for the wrist camera, and it exists because
    ``track`` answers a question that is too big to debug: "did the whole handover
    converge?" mixes translation, range, rotation, the taught goal and the detector
    together, and when it does not converge, none of them is ruled out.

    So this strips the loop to the one part a person can check by eye. It servos the *spine
    midpoint* -- the middle of the bar the gripper closes on, which the detector already
    marks on its overlay -- to the centre of the wrist frame, and it gives ``scale_px`` and
    ``spine_deg`` :data:`IGNORED_TOLERANCE`, so the arm does not try to change its range or
    its rotation while doing it. If the marker walks to the middle of the frame and stays
    there, the camera, the detector, the Jacobian's signs and the servo all work, and any
    remaining trouble is in the taught goal. If it walks the wrong way, a Jacobian column
    has the wrong sign, which is the failure this is fastest at showing.

    The goal is the frame's own centre, read from the first frame rather than configured:
    the image size is a property of the camera in front of us, and asking an operator to
    restate it is asking them to get it wrong.

    Nothing is written. The stored calibration is retargeted in memory only, so this can be
    run at any time without disturbing what the FSM uses.
    """

    from everest_robot.robot.deployment import wrist_servo_path
    from everest_robot.robot.wrist_servo import IGNORED_TOLERANCE

    calibration = WristServoCalibration.load(args.config or wrist_servo_path())
    session = _open_session(args)
    follower = None
    try:
        calibration.verify(_stamp_of(session))
        frames = _frames(session, calibration.camera_name, calibration.color_mode)
        height, width = frames().shape[:2]
        centre = (width / 2.0, height / 2.0)

        aimed = calibration.retargeted(
            point=args.point,
            goal=(centre[0], centre[1], calibration.goal[2], calibration.goal[3]),
            tolerance=(args.tolerance, args.tolerance, IGNORED_TOLERANCE, IGNORED_TOLERANCE),
        )
        print(
            f"\ncentring the {args.point} point on ({centre[0]:.0f}, {centre[1]:.0f}) "
            f"in a {width}x{height} frame, to within {args.tolerance:.0f} px "
            f"({args.tolerance * MM_PER_PX:.0f} mm).\n"
            "Range and rotation are not servoed in this mode."
        )
        if args.point != calibration.point:
            # Honest about the one approximation being made. The three candidate points are
            # on one rigid object, so the u/v rows of the Jacobian transfer between them
            # under small camera translations; the scale and spine rows do not, which is
            # why this mode does not servo either of them.
            print(
                f"NOTE: the Jacobian was taught on the {calibration.point!r} point. Only its "
                "u/v rows are used here, and those transfer between points on one object."
            )
        follower = _build_follower(session, aimed, args)
        return _run_follower(follower, args, done="centred")
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 1
    finally:
        if follower is not None:
            follower.stop(hold=False)
        session.close()


def _build_follower(session: Any, calibration: WristServoCalibration, args: argparse.Namespace):
    from everest_robot.robot.visual_tracking import VisualTracker
    from everest_robot.robot.wrist_follower import (
        WristCarabinerFollower,
        detect_carabiner_wrist,
    )

    return WristCarabinerFollower(
        calibration=calibration,
        tracker=VisualTracker(
            session.port,
            rate_hz=args.rate,
            max_velocity_rad_s=args.max_velocity,
            lock_frames=args.lock_frames,
            dry_run=args.dry_run,
            clock=session.clock,
        ),
        frames=_frames(session, calibration.camera_name, calibration.color_mode),
        detect=lambda frame: detect_carabiner_wrist(frame, _roi(args)),
        clock=session.clock,
        settle_ticks=args.settle_ticks,
    )


def _run_follower(follower: Any, args: argparse.Namespace, *, done: str) -> int:
    """Step the follower and print one line per tick until it settles or the budget ends."""

    from everest_robot.robot.visual_tracking import TrackerStopped

    if not args.dry_run:
        _confirm("THIS MOVES THE ARM. Clear the area.", "GO", args.yes)
    follower.start()
    print(f"\n{'tick':>5}{'visible':>9}{'done':>6}{'px err':>9}{'misses':>8}   reason")
    try:
        while True:
            tick = follower.step()
            error = "     -" if tick.pixel_error_px is None else f"{tick.pixel_error_px:6.1f}"
            print(
                f"{tick.index:>5}{str(tick.target_visible):>9}{str(tick.followed):>6}"
                f"{error:>9}{tick.misses:>8}   {tick.reason}"
            )
            if tick.followed:
                print(f"\n{done}: held within tolerance for {args.settle_ticks} ticks.")
                return 0
            if args.max_ticks and tick.index >= args.max_ticks:
                print(f"\nstopped after {args.max_ticks} ticks without settling.")
                return 1
    except TrackerStopped as error:
        print(f"\ntracker stopped: {error}", file=sys.stderr)
        return 1


# ── track ──────────────────────────────────────────────────────────────────────────
def cmd_track(args: argparse.Namespace) -> int:
    """MOVES THE ARM: run the wrist follower alone, on the taught goal, outside the FSM.

    The whole handover, end to end. Reach for ``centre`` first when it does not converge:
    this one mixes translation, range, rotation and the taught goal together, so a failure
    here rules nothing out on its own.
    """

    from everest_robot.robot.deployment import wrist_servo_path

    calibration = WristServoCalibration.load(args.config or wrist_servo_path())
    session = _open_session(args)
    follower = None
    try:
        calibration.verify(_stamp_of(session))
        follower = _build_follower(session, calibration, args)
        return _run_follower(follower, args, done="followed")
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 1
    finally:
        if follower is not None:
            follower.stop(hold=False)
        session.close()


# ── the parser ─────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--config", default=None, help="calibration path; defaults to EVEREST_WRIST_SERVO"
    )
    shared.add_argument("--fake", action="store_true", help="run against FakeArm; nothing physical")
    shared.add_argument(
        "--roi", nargs=4, type=int, default=None, metavar=("X", "Y", "W", "H"),
        help="restrict the detector to this rectangle; defaults to EVEREST_WRIST_ROI",
    )
    shared.add_argument(
        "--samples", type=int, default=5,
        help="detections median-combined into one measurement (default 5)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    teach = subparsers.add_parser(
        "teach", parents=[shared],
        help="MOVES THE ARM: record the goal image and measure the Jacobian by bumping joints",
    )
    teach.add_argument("--camera", default="wrist", help="EVEREST_CAMERAS name (default wrist)")
    teach.add_argument("--color", default="rgb", choices=("rgb", "bgr"))
    teach.add_argument(
        "--point", default="insert", choices=FEATURE_POINTS,
        help="which detected point the goal and the Jacobian are taught on (default insert, "
             "the detector's own fingertip target)",
    )
    teach.add_argument("--joints", default=",".join(DEFAULT_SERVO_JOINTS))
    teach.add_argument(
        "--delta", type=float, default=DEFAULT_DELTA_RAD,
        help=f"radians each joint is bumped either way (default {DEFAULT_DELTA_RAD})",
    )
    teach.add_argument("--speed-scale", type=float, default=0.3)
    teach.add_argument(
        "--tolerance", nargs=len(FEATURE_NAMES), type=float, default=None,
        metavar=tuple(name.upper() for name in FEATURE_NAMES),
        help="per-feature arrival tolerance: px px px deg (default "
        f"{tuple(round(value, 1) for value in DEFAULT_TOLERANCE)}, which is 8/8/10 mm at "
        f"{MM_PER_PX} mm/px, plus 6 deg of spine)",
    )
    teach.add_argument(
        "--return-scale", type=float, default=1.5,
        help="multiple of tolerance the goal image may drift over the whole teach (default 1.5)",
    )
    teach.add_argument("--gain", type=float, default=None)
    teach.add_argument("--damping", type=float, default=None)
    teach.add_argument("--max-delta", type=float, default=None)
    teach.add_argument("--approved-by", default=None, help="who approved the goal pose")
    teach.add_argument("--yes", action="store_true", help="skip the confirmation")
    teach.set_defaults(handler=cmd_teach)

    check = subparsers.add_parser(
        "check", parents=[shared], help="print the calibration; touches no hardware"
    )
    check.set_defaults(handler=cmd_check)

    look = subparsers.add_parser(
        "look", parents=[shared],
        help="print the current image, its error against the goal, and the step that closes it",
    )
    look.set_defaults(handler=cmd_look)

    def add_servo_arguments(parser: argparse.ArgumentParser, *, max_velocity: float) -> None:
        parser.add_argument("--rate", type=float, default=15.0)
        parser.add_argument(
            "--max-velocity", type=float, default=max_velocity,
            help="speed lock: radians per second per joint, as a per-tick clamp "
                 f"(default {max_velocity})",
        )
        parser.add_argument("--lock-frames", type=int, default=3)
        parser.add_argument("--settle-ticks", type=int, default=3)
        parser.add_argument("--max-ticks", type=int, default=300)
        parser.add_argument("--dry-run", action="store_true", help="never energize the arm")
        parser.add_argument("--yes", action="store_true", help="skip the confirmation")

    centre = subparsers.add_parser(
        "centre",
        aliases=["center"],
        parents=[shared],
        help="MOVES THE ARM: slowly centre the carabiner's spine in the wrist view",
    )
    # Slower than `track` by default. This is the command run while watching an overlay to
    # see which way the arm goes, so it should be easy to stop and hard to be surprised by.
    add_servo_arguments(centre, max_velocity=0.06)
    centre.add_argument(
        "--point", default="spine", choices=FEATURE_POINTS,
        help="which detected point to bring to the frame centre (default spine)",
    )
    centre.add_argument(
        "--tolerance", type=float, default=DEFAULT_TOLERANCE[0],
        help="pixels from the frame centre that count as centred "
             f"(default {DEFAULT_TOLERANCE[0]:.0f}, which is 8 mm at {MM_PER_PX} mm/px)",
    )
    centre.set_defaults(handler=cmd_centre)

    track = subparsers.add_parser(
        "track", parents=[shared],
        help="MOVES THE ARM: run the whole wrist follower on the taught goal",
    )
    add_servo_arguments(track, max_velocity=0.15)
    track.set_defaults(handler=cmd_track)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        raise SystemExit(args.handler(args))
    except WristServoError as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
