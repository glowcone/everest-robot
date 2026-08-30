"""``robot-pixel-map``: teach, fit and use a fixed-camera pixel -> joint map.

The whole procedure lives behind five subcommands, in the order they are run:

1. ``collect``  Bolt the camera. Put the carabiner somewhere, teleoperate to the pose that
   grasps it, press ``c``. About 30 positions on a rough 6x5 grid plus the corners; roughly
   25 minutes. Samples are appended to the calibration file as they are taken, so an
   interrupted session loses nothing.
2. ``fit``      Fit the map and print the held-out error. Re-runnable at any time; it only
   ever reads the samples already in the file.
3. ``check``    Print what the file contains without touching a camera or an arm.
4. ``predict``  Point the camera at the carabiner and print the joint vector, no motion.
5. ``track``    Continuously servo the gripper to the taught pre-grasp pose above whatever
   the camera sees, at a locked speed, holding still whenever it sees nothing.

Everything lands in one JSON file (``--config``, default ``config/pixel_map.json``): the
camera id, the detector and its ROI, the arm and calibration the samples were taught on,
every sample, the fitted coefficients, the convex hull the fit is valid inside, and the
wrist-roll offset. One file is the whole calibration.

Why ``track`` needs no notion of height. The map is taught on *pre-grasp* poses, so
"directly above the carabiner" is what every sample already encodes; tracking to the map's
prediction tracks to directly-above by construction. Nothing here models the table plane,
the camera's obliquity or the arm's kinematics, which is the point --
:mod:`everest_robot.pixel_map` explains why.

Both ``collect`` and ``track`` claim the robot lease and own the arm for the duration.
Never run either alongside a worker or a monitor: they would be two participants on one
CAN bus.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from everest_robot.pixel_map import (
    DEFAULT_BASE_JOINT,
    DEFAULT_FIT_JOINTS,
    DEFAULT_ROLL_JOINT,
    MIN_SAMPLES,
    QUADRATIC,
    THIN_PLATE_SPLINE,
    CameraSource,
    DetectorSpec,
    OutsideCalibratedRegion,
    PixelJointMap,
    PixelMapError,
    RobotStamp,
    Sample,
    now_stamp,
)

# The camera, the detector and the follower live in the robot SDK layer, because the
# attachment FSM's SEARCH_CV state uses exactly the same three. They are re-exported here:
# this CLI is where an operator meets them, and `robot-pixel-map track` is the same loop
# the FSM runs, minus the arbitration.
from everest_robot.robot.carabiner_follower import (
    DEFAULT_MAX_JUMP_PX,
    CarabinerPixel,
    detect_carabiner,
    load_cv2,
    open_capture,
    read_frame,
)

DEFAULT_CONFIG = "config/pixel_map.json"

# How far each joint backs off before the final one-way move onto the target.
DEFAULT_APPROACH_RAD = 0.05

__all__ = [
    "CarabinerPixel",
    "build_parser",
    "detect_carabiner",
    "main",
    "open_capture",
    "read_frame",
]


def _draw(
    frame: Any, roi: Sequence[int], detection: CarabinerPixel | None, lines: list[str]
) -> None:
    cv2 = load_cv2()
    colour = (0, 220, 0) if detection is not None else (0, 0, 255)
    cv2.rectangle(frame, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), colour, 2)
    if detection is not None:
        for point in (detection.white_a, detection.white_b):
            cv2.circle(frame, (int(point[0]), int(point[1])), 7, (255, 255, 255), -1)
            cv2.circle(frame, (int(point[0]), int(point[1])), 7, (0, 0, 0), 2)
        cv2.circle(frame, (int(detection.gate[0]), int(detection.gate[1])), 7, (0, 0, 0), -1)
        centre = (int(detection.centroid[0]), int(detection.centroid[1]))
        cv2.drawMarker(frame, centre, (0, 220, 255), cv2.MARKER_CROSS, 22, 2)
    for row, text in enumerate(lines):
        cv2.putText(
            frame, text, (12, 26 + 22 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4
        )
        cv2.putText(
            frame, text, (12, 26 + 22 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )


def _draw_hull(frame: Any, hull: Any) -> None:
    import numpy as np

    cv2 = load_cv2()
    cv2.polylines(frame, [np.asarray(hull, dtype=np.int32)], True, (255, 180, 0), 2)


# ── the robot side ─────────────────────────────────────────────────────────────────
def _open_session(args: argparse.Namespace) -> Any:
    """Claim and connect the deployed arm. The caller closes it."""

    from everest_robot.robot.deployment import build_lease, build_port, load_parameters
    from everest_robot.robot.session import RobotSession

    if args.fake:
        from everest_robot.jog import open_fake_session

        print("FAKE ARM: nothing below this line describes a physical robot.\n")
        return open_fake_session()

    parameters = load_parameters()
    # No cameras on the session: the fixed camera here is not a policy observation, and
    # routing it through CameraRuntime would make it look like one.
    session = RobotSession(
        build_port(parameters), parameters, lease=build_lease(parameters), cameras=None
    )
    return session.open()


def _stamp_of(session: Any) -> RobotStamp:
    identity = session.port.identity
    return RobotStamp(
        robot_id=identity.robot_id,
        calibration_id=identity.calibration_id,
        config_digest=getattr(session.parameters, "config_digest", ""),
    )


def _start_teleoperation(session: Any, args: argparse.Namespace) -> Any:
    """Open the Star leader against the already-claimed follower, as the monitor does."""

    import os

    from everest_robot.robot.teleoperation import (
        Star102LeaderPort,
        TeleoperationController,
        load_star_mapper,
    )

    star_port = args.star_port or os.environ.get("EVEREST_STAR_PORT")
    if not star_port:
        raise PixelMapError(
            "Star leader port is required: set EVEREST_STAR_PORT or pass --star-port; "
            "use --no-teleop to hand-teach a torque-disabled arm instead"
        )
    try:
        star_ids = tuple(int(value.strip()) for value in args.star_ids.split(","))
    except ValueError as error:
        raise PixelMapError("--star-ids must be comma-separated integers") from error

    controller = TeleoperationController(
        session.port,
        Star102LeaderPort(star_port, star_ids),
        load_star_mapper(args.star_map),
        rate_hz=args.leader_rate,
        max_velocity_rad_s=args.max_velocity,
    )
    difference = controller.connect_and_measure()
    if difference > args.sync_threshold and not args.yes:
        answer = input(
            f"leader/follower pose difference is {difference:.2f} rad; "
            "clear the area and type FOLLOW to enable > "
        )
        if answer.strip() != "FOLLOW":
            controller.close()
            raise PixelMapError("teleoperation cancelled before enabling")
    controller.start()
    return controller


# ── collect ────────────────────────────────────────────────────────────────────────
def _existing_or_new_map(args: argparse.Namespace, session: Any) -> PixelJointMap:
    """Reuse the file's camera, ROI and samples when it already exists.

    A calibration only means anything for one camera pose on one arm calibration, so
    appending to a file taught on a different arm is refused rather than merged.
    """

    stamp = _stamp_of(session)
    camera = CameraSource(
        index_or_path=str(args.camera),
        backend=args.camera_backend,
        width=args.width,
        height=args.height,
    )
    detector = DetectorSpec(kind="two-white-black", roi_xywh=tuple(args.roi) if args.roi else None)

    path = Path(args.config)
    if not path.exists():
        if detector.roi_xywh is None:
            raise PixelMapError("--roi X Y W H is required when starting a new calibration")
        return PixelJointMap(
            camera=camera,
            detector=detector,
            robot=stamp,
            joint_names=tuple(session.port.joint_names),
            samples=(),
            required_margin_px=0.0 if args.margin_px is None else args.margin_px,
            approach_offset_rad=tuple(
                DEFAULT_APPROACH_RAD if args.approach_rad is None else args.approach_rad
                for _ in session.port.joint_names
            ),
            created_at=now_stamp(),
        )

    existing = PixelJointMap.load(path)
    existing.robot.verify(stamp)
    if existing.joint_names != tuple(session.port.joint_names):
        raise PixelMapError(
            f"{path} was taught with joints {', '.join(existing.joint_names)}; "
            f"this arm reports {', '.join(session.port.joint_names)}"
        )
    if str(args.camera) != existing.camera.index_or_path:
        raise PixelMapError(
            f"{path} was taught on camera {existing.camera.index_or_path!r}, not "
            f"{str(args.camera)!r}. A different camera means a different calibration."
        )
    print(f"appending to {path} ({len(existing.samples)} samples already taught)")
    # A new --roi is honoured and *recorded*. The detector searches the ROI passed on the
    # command line, so leaving the stored one behind would file these samples under a crop
    # they were not taken through, and `track` would later search somewhere the operator
    # never taught. Widening is the normal case: the region a prediction is trusted in is
    # the sample hull, not this rectangle, so a bigger crop grants no extra trust.
    detector = existing.detector
    if args.roi and tuple(args.roi) != existing.detector.roi_xywh:
        detector = DetectorSpec(kind=existing.detector.kind, roi_xywh=tuple(args.roi))
        print(f"  recording the new ROI {tuple(args.roi)} (was {existing.detector.roi_xywh})")
    # The file's own margin and approach offsets win unless this run names new ones, so a
    # re-run to add a few samples cannot quietly reset a value the operator chose earlier.
    return PixelJointMap(
        camera=existing.camera,
        detector=detector,
        robot=existing.robot,
        joint_names=existing.joint_names,
        samples=existing.samples,
        model=existing.model,
        hull=existing.hull,
        required_margin_px=(
            existing.required_margin_px if args.margin_px is None else args.margin_px
        ),
        roll=existing.roll,
        approach_offset_rad=(
            existing.approach_offset_rad
            if args.approach_rad is None
            else tuple(args.approach_rad for _ in existing.joint_names)
        ),
        validation=existing.validation,
        created_at=existing.created_at,
    )


def cmd_collect(args: argparse.Namespace) -> int:
    """Teach correspondences: object pixel now, joint vector that grasps it now."""

    cv2 = load_cv2()
    session = _open_session(args)
    controller = None
    try:
        calibration = _existing_or_new_map(args, session)
        roi = args.roi or calibration.detector.roi_xywh
        if roi is None:
            raise PixelMapError("--roi X Y W H is required")
        camera = calibration.camera
        capture = open_capture(camera)

        if not args.no_teleop and not args.fake:
            controller = _start_teleoperation(session, args)

        samples = list(calibration.samples)
        print(
            "terminal: Enter captures, u undoes, q quits"
            if args.no_window
            else "keys: c capture · u undo last · q save and quit"
        )
        print(
            "approach every pre-grasp pose from the SAME direction, during capture and "
            "during deployment. Backlash makes the reached pose depend on it."
        )

        try:
            while True:
                frame = read_frame(capture, camera)
                detection: CarabinerPixel | None
                try:
                    detection = detect_carabiner(frame, roi)
                    miss = ""
                except (RuntimeError, ValueError) as error:
                    detection = None
                    miss = str(error)

                status = [
                    f"samples {len(samples)}   camera {camera.index_or_path}",
                    (
                        f"pixel ({detection.centroid[0]:.0f}, {detection.centroid[1]:.0f})  "
                        f"spine {math.degrees(detection.spine_rad):+.0f} deg"
                        if detection is not None
                        else f"NO DETECTION  {miss[:60]}"
                    ),
                ]
                if args.no_window:
                    key = _prompt_key(status)
                else:
                    _draw(frame, roi, detection, status)
                    cv2.imshow("pixel-map collect", frame)
                    key = cv2.waitKey(1) & 0xFF

                if key in {ord("q"), 27}:
                    break
                if key == ord("u") and samples:
                    dropped = samples.pop()
                    print(f"dropped sample at ({dropped.pixel[0]:.0f}, {dropped.pixel[1]:.0f})")
                elif key == ord("c"):
                    if detection is None:
                        print("refused: no detection to pair with the pose")
                        continue
                    state = session.port.read_state()
                    if not state.all_finite:
                        print("refused: the arm has missing position feedback")
                        continue
                    samples.append(
                        Sample(
                            pixel=detection.centroid,
                            joints=tuple(float(value) for value in state.positions),
                            spine_rad=detection.spine_rad,
                            captured_at=now_stamp(),
                        )
                    )
                    print(
                        f"[{len(samples):>3}] pixel ({detection.centroid[0]:7.1f},"
                        f" {detection.centroid[1]:7.1f})  q = "
                        + ", ".join(f"{value:+.4f}" for value in state.positions)
                    )
                if controller is not None and controller.error:
                    print(f"teleoperation stopped: {controller.error}", file=sys.stderr)
                    break
        finally:
            capture.release()
            if not args.no_window:
                with contextlib.suppress(Exception):
                    cv2.destroyAllWindows()

        calibration = calibration.with_samples(samples)
        if len(samples) >= MIN_SAMPLES and not args.no_fit:
            calibration = _fit(calibration, args)
        path = calibration.save(args.config)
        print(f"\nwrote {len(samples)} samples to {path}")
        if len(samples) < MIN_SAMPLES:
            print(
                f"not fitted: {len(samples)} samples, need at least {MIN_SAMPLES} "
                "(the procedure asks for about 30). Re-run collect to add more."
            )
        else:
            _print_fit(calibration)
    finally:
        # Torque off, without a parting hold command. Ctrl-C during a teaching session
        # should leave the arm limp and movable by hand, not snapped to wherever it was
        # last told to go.
        if controller is not None:
            controller.close(hold=False)
        session.close()
    return 0


def _prompt_key(status: Sequence[str]) -> int:
    """The no-window capture trigger: Enter captures, ``q`` quits, ``u`` undoes."""

    for line in status:
        print(line)
    answer = input("[Enter] capture · u undo · q quit > ").strip().lower()
    if answer.startswith("q"):
        return ord("q")
    if answer.startswith("u"):
        return ord("u")
    return ord("c")


# ── fit / check / predict ──────────────────────────────────────────────────────────
def _fit(calibration: PixelJointMap, args: argparse.Namespace) -> PixelJointMap:
    return calibration.fitted(
        fit_joints=tuple(args.joints),
        kind=args.model,
        smoothing=args.smoothing,
        ridge=args.ridge,
        roll_joint=None if args.no_roll else args.roll_joint,
        base_joint=args.base_joint,
        holdout=args.holdout,
    )


def _print_fit(calibration: PixelJointMap) -> None:
    model = calibration.model
    if model is None:
        print("no fit stored")
        return
    print(f"\nmodel      {model.kind} over {', '.join(model.joints)}")
    print(f"samples    {len(calibration.samples)}")
    if calibration.hull is not None:
        print(f"valid area {len(calibration.hull)}-vertex hull, "
              f"margin {calibration.required_margin_px:g} px required")
    if calibration.roll is not None:
        roll = calibration.roll
        print(
            f"wrist roll {roll.joint} = {roll.sign:+d}*spine - {roll.base_joint} "
            f"{math.degrees(roll.offset_rad):+.2f} deg "
            f"(residual {roll.residual_std_deg:.2f} deg over {roll.samples_used} samples)"
        )
    report = calibration.validation or {}
    if not report.get("held_out"):
        print(f"holdout    {report.get('note', 'not evaluated')}")
        return
    print(f"holdout    {report['held_out']} of {len(calibration.samples)} samples")
    print(f"{'joint':<16}{'rms deg':>10}{'max deg':>10}")
    for joint in model.joints:
        print(
            f"{joint:<16}{report['rms_error_deg'][joint]:>10.3f}"
            f"{report['max_abs_error_deg'][joint]:>10.3f}"
        )
    print(
        "\nJoint degrees are not millimetres. Predict, execute and measure six held-out "
        "picks with a ruler; 3-8 mm is the expected band."
    )


def cmd_fit(args: argparse.Namespace) -> int:
    calibration = _fit(PixelJointMap.load(args.config), args)
    path = calibration.save(args.config)
    _print_fit(calibration)
    print(f"\nwrote {path}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    calibration = PixelJointMap.load(args.config)
    print(f"camera     {calibration.camera.index_or_path} ({calibration.camera.backend})")
    print(f"detector   {calibration.detector.kind}  roi {calibration.detector.roi_xywh}")
    print(f"arm        {calibration.robot.robot_id} / {calibration.robot.calibration_id}")
    print(f"joints     {', '.join(calibration.joint_names)}")
    print(f"approach   {', '.join(f'{value:+.3f}' for value in calibration.approach_offset_rad)}")
    _print_fit(calibration)
    if args.pixel:
        _print_prediction(calibration, tuple(args.pixel), None)
    return 0


def _print_prediction(
    calibration: PixelJointMap, pixel: tuple[float, float], spine_rad: float | None
) -> None:
    prediction = calibration.predict(pixel, spine_rad)
    print(f"\npixel ({pixel[0]:.1f}, {pixel[1]:.1f})  {prediction.hull_margin_px:.1f} px inside")
    for joint, value in prediction.joints.items():
        print(f"  {joint:<16}{value:>+10.4f} rad{math.degrees(value):>10.2f} deg")


def cmd_predict(args: argparse.Namespace) -> int:
    """Print what the map says for a pixel, or for whatever the camera sees. No motion."""

    calibration = PixelJointMap.load(args.config)
    if args.pixel:
        _print_prediction(calibration, (args.pixel[0], args.pixel[1]), None)
        return 0

    camera = _camera_for(args, calibration)
    roi = args.roi or calibration.detector.roi_xywh
    if roi is None:
        raise PixelMapError("this calibration stored no ROI; pass --roi X Y W H")
    capture = open_capture(camera)
    try:
        for _ in range(max(1, args.frames)):
            frame = read_frame(capture, camera)
            try:
                detection = detect_carabiner(frame, roi)
            except (RuntimeError, ValueError) as error:
                print(f"no detection: {error}")
                continue
            try:
                _print_prediction(calibration, detection.centroid, detection.spine_rad)
            except OutsideCalibratedRegion as error:
                print(f"refused: {error}")
    finally:
        capture.release()
    return 0


def _camera_for(args: argparse.Namespace, calibration: PixelJointMap) -> CameraSource:
    """The stored camera unless the operator overrode it on the command line."""

    if args.camera is None:
        return calibration.camera
    return CameraSource(
        index_or_path=str(args.camera),
        backend=args.camera_backend,
        width=args.width or calibration.camera.width,
        height=args.height or calibration.camera.height,
    )


# ── track ──────────────────────────────────────────────────────────────────────────
def cmd_track(args: argparse.Namespace) -> int:
    """Servo continuously to the taught pre-grasp pose above the detected carabiner.

    Speed locked and detection gated: the arm moves at most ``--max-velocity`` rad/s per
    joint, and holds whenever the detector misses, the pixel jumps, or the pixel leaves
    the calibrated region. It never coasts toward a stale target.
    """

    cv2 = load_cv2()
    from everest_robot.robot.visual_tracking import TrackerStopped, VisualTracker

    calibration = PixelJointMap.load(args.config)
    if calibration.model is None:
        raise PixelMapError(f"{args.config} has no fit yet; run `robot-pixel-map fit` first")
    camera = _camera_for(args, calibration)
    roi = args.roi or calibration.detector.roi_xywh
    if roi is None:
        raise PixelMapError("this calibration stored no ROI; pass --roi X Y W H")

    session = _open_session(args)
    capture = None
    tracker = None
    try:
        if not args.fake:
            calibration.robot.verify(_stamp_of(session))
        if calibration.joint_names != tuple(session.port.joint_names):
            raise PixelMapError(
                f"calibration joints {', '.join(calibration.joint_names)} do not match this "
                f"arm's {', '.join(session.port.joint_names)}"
            )
        capture = open_capture(camera)
        tracker = VisualTracker(
            session.port,
            rate_hz=args.rate,
            max_velocity_rad_s=args.max_velocity,
            lock_frames=args.lock_frames,
            dry_run=args.dry_run,
        )
        if not _confirm_first_target(session, calibration, capture, camera, roi, args):
            return 1
        tracker.start()
        reasons = _track_loop(session, calibration, tracker, capture, camera, roi, args, cv2)
    except TrackerStopped as error:
        print(f"tracker stopped: {error}", file=sys.stderr)
        return 1
    finally:
        # No parting hold: the session teardown below cuts torque immediately after, so
        # a hold could only snap the arm toward its last target on the way out.
        if tracker is not None:
            tracker.stop(hold=False)
        if capture is not None:
            capture.release()
        if not args.no_window:
            with contextlib.suppress(Exception):
                cv2.destroyAllWindows()
        session.close()

    print("\nticks by outcome:")
    for reason, count in sorted(reasons.items(), key=lambda item: -item[1]):
        print(f"  {count:>6}  {reason}")
    return 0


def _confirm_first_target(
    session: Any,
    calibration: PixelJointMap,
    capture: Any,
    camera: CameraSource,
    roi: Sequence[int],
    args: argparse.Namespace,
) -> bool:
    """Say how far the first move would be before anything is energized.

    The arm may be parked anywhere. The speed lock bounds how fast it closes that gap, not
    how large the gap is, and the operator is the one who knows what is in the way.
    """

    if args.yes or args.dry_run:
        return True
    measured = session.port.read_state()
    for _ in range(30):
        frame = read_frame(capture, camera)
        try:
            detection = detect_carabiner(frame, roi)
            prediction = calibration.predict(detection.centroid, detection.spine_rad)
        except (RuntimeError, ValueError, PixelMapError):
            continue
        target = calibration.full_target(prediction, measured.positions)
        gap = max(abs(a - b) for a, b in zip(target, measured.positions, strict=True))
        answer = input(
            f"first target is {gap:.2f} rad away at {args.max_velocity} rad/s; "
            "clear the area and type TRACK to enable > "
        )
        return answer.strip() == "TRACK"
    print("no detection in 30 frames; nothing to confirm against", file=sys.stderr)
    return False


def _track_loop(
    session: Any,
    calibration: PixelJointMap,
    tracker: Any,
    capture: Any,
    camera: CameraSource,
    roi: Sequence[int],
    args: argparse.Namespace,
    cv2: Any,
) -> dict[str, int]:
    from everest_robot.robot.clock import SystemClock

    clock = SystemClock()
    reasons: dict[str, int] = {}
    previous: tuple[float, float] | None = None

    with contextlib.suppress(KeyboardInterrupt):
        while tracker.stopped_reason is None:
            started = clock.monotonic()
            frame = read_frame(capture, camera)
            target, detection, reason = _target_from_frame(
                session, calibration, frame, roi, previous, args
            )
            previous = None if detection is None else detection.centroid

            tick = tracker.tick(target, reason)
            reasons[tick.reason] = reasons.get(tick.reason, 0) + 1

            if not args.no_window:
                lines = [
                    f"{'MOVING' if tick.moved else 'HOLD'}  {tick.reason}",
                    f"speed lock {args.max_velocity:g} rad/s"
                    f"  ({tracker.max_step_rad:.4f} rad/tick)",
                ]
                if tick.remaining_rad:
                    lines.append(f"remaining {tick.remaining_rad:.3f} rad")
                _draw(frame, roi, detection, lines)
                if calibration.hull is not None:
                    _draw_hull(frame, calibration.hull)
                cv2.imshow("pixel-map track", frame)
                if (cv2.waitKey(1) & 0xFF) in {ord("q"), 27}:
                    break
            elif tick.index % max(1, int(args.rate)) == 0:
                print(f"[{tick.index:>6}] {'move' if tick.moved else 'hold'}  {tick.reason}")

            clock.sleep(tracker.period_s - (clock.monotonic() - started))

    if tracker.stopped_reason is not None:
        print(f"tracker stopped: {tracker.stopped_reason}", file=sys.stderr)
    return reasons


def _target_from_frame(
    session: Any,
    calibration: PixelJointMap,
    frame: Any,
    roi: Sequence[int],
    previous: tuple[float, float] | None,
    args: argparse.Namespace,
) -> tuple[tuple[float, ...] | None, CarabinerPixel | None, str]:
    """One frame to one joint target, or to a reason for holding still."""

    try:
        detection = detect_carabiner(frame, roi)
    except (RuntimeError, ValueError) as error:
        return None, None, f"no detection ({str(error).split('.')[0][:50]})"

    if previous is not None:
        jump = math.dist(previous, detection.centroid)
        if jump > args.max_jump_px:
            return None, detection, f"detection jumped {jump:.0f} px"

    try:
        prediction = calibration.predict(
            detection.centroid, None if args.no_roll_tracking else detection.spine_rad
        )
    except OutsideCalibratedRegion:
        return None, detection, "outside the calibrated region"

    state = session.port.read_state()
    if not state.all_finite:
        return None, detection, "no joint feedback"
    return calibration.full_target(prediction, state.positions), detection, "tracking"


# ── entry point ────────────────────────────────────────────────────────────────────
def _add_camera_arguments(parser: argparse.ArgumentParser, *, required: bool) -> None:
    parser.add_argument(
        "--camera",
        required=required,
        default=None,
        help="camera id: an index (0, 1) or a device path. Bolt it down before sample one.",
    )
    parser.add_argument(
        "--camera-backend", default="auto", choices=("auto", "avfoundation", "v4l2", "any")
    )
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument(
        "--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
        help="pickup-zone crop the detector searches; stored in the calibration",
    )


def _add_fit_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--joints", nargs="+", default=list(DEFAULT_FIT_JOINTS),
        help=f"joints to fit against the pixel (default {' '.join(DEFAULT_FIT_JOINTS)})",
    )
    parser.add_argument(
        "--model", default=THIN_PLATE_SPLINE, choices=(THIN_PLATE_SPLINE, QUADRATIC)
    )
    parser.add_argument(
        "--smoothing", type=float, default=0.0,
        help="spline smoothing in normalized-pixel units; 0 reproduces every sample exactly",
    )
    parser.add_argument("--ridge", type=float, default=1e-3, help="quadratic ridge penalty")
    parser.add_argument("--roll-joint", default=DEFAULT_ROLL_JOINT)
    parser.add_argument("--base-joint", default=DEFAULT_BASE_JOINT)
    parser.add_argument(
        "--no-roll", action="store_true", help="do not fit the wrist-roll offset"
    )
    parser.add_argument("--holdout", type=int, default=6, help="samples to hold out for validation")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="robot-pixel-map",
        description="Teach, fit and use a fixed-camera pixel to joint-position map.",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help=f"default {DEFAULT_CONFIG}")
    parser.add_argument(
        "--fake", action="store_true",
        help="run against the deterministic FakeArm: no CAN, no claim, no real numbers",
    )
    # The same two options are accepted on either side of the subcommand, because an
    # operator will type them on whichever side they thought of them. SUPPRESS keeps the
    # subcommand's defaults from silently overwriting a value given before it.
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--config", default=argparse.SUPPRESS)
    shared.add_argument("--fake", action="store_true", default=argparse.SUPPRESS)
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser(
        "collect", parents=[shared], help="teach pixel/pose pairs by teleoperation"
    )
    _add_camera_arguments(collect, required=True)
    _add_fit_arguments(collect)
    collect.add_argument(
        "--margin-px", type=float, default=None,
        help="pixels of hull margin a later detection must have to be believed (default 0)",
    )
    collect.add_argument(
        "--approach-rad", type=float, default=None,
        help="backlash approach offset per joint, subtracted to make the final move one-way "
             f"(default {DEFAULT_APPROACH_RAD}). Both are kept from the file when re-collecting.",
    )
    collect.add_argument(
        "--no-teleop", action="store_true",
        help="hand-teach a torque-disabled arm instead of following the Star leader",
    )
    collect.add_argument("--no-fit", action="store_true", help="save samples without fitting")
    collect.add_argument("--no-window", action="store_true", help="capture from the terminal")
    collect.add_argument("--star-port", help="Star 102 serial port; defaults to EVEREST_STAR_PORT")
    collect.add_argument("--star-ids", default="0,1,2,3,4,5,6")
    collect.add_argument("--star-map", help="Star mapping JSON; defaults to maker-arm's profile")
    collect.add_argument("--leader-rate", type=float, default=25.0)
    collect.add_argument(
        "--max-velocity", type=float, default=0.375,
        help="how fast the follower may chase the leader, rad/s per joint (default 0.375)",
    )
    collect.add_argument("--sync-threshold", type=float, default=0.8)
    collect.add_argument("--yes", action="store_true", help="skip the enable confirmation")
    collect.set_defaults(handler=cmd_collect)

    fit = subparsers.add_parser(
        "fit", parents=[shared], help="fit the stored samples and report holdout error"
    )
    _add_fit_arguments(fit)
    fit.set_defaults(handler=cmd_fit)

    check = subparsers.add_parser(
        "check", parents=[shared], help="print the calibration; touches no hardware"
    )
    check.add_argument("--pixel", nargs=2, type=float, metavar=("U", "V"))
    check.set_defaults(handler=cmd_check)

    predict = subparsers.add_parser(
        "predict", parents=[shared], help="print the joint vector for a pixel. No motion."
    )
    _add_camera_arguments(predict, required=False)
    predict.add_argument("--pixel", nargs=2, type=float, metavar=("U", "V"))
    predict.add_argument("--frames", type=int, default=1)
    predict.set_defaults(handler=cmd_predict)

    track = subparsers.add_parser(
        "track",
        parents=[shared],
        help="MOVES THE ARM: servo continuously above the detected carabiner",
    )
    _add_camera_arguments(track, required=False)
    track.add_argument("--rate", type=float, default=15.0, help="control ticks per second")
    track.add_argument(
        "--max-velocity", type=float, default=0.15,
        help="speed lock: radians per second per joint, enforced as a per-tick clamp",
    )
    track.add_argument(
        "--lock-frames", type=int, default=3,
        help="consecutive good detections required before motion resumes",
    )
    track.add_argument("--max-jump-px", type=float, default=DEFAULT_MAX_JUMP_PX)
    track.add_argument(
        "--no-roll-tracking", action="store_true", help="leave wrist roll where it is"
    )
    track.add_argument(
        "--dry-run", action="store_true", help="run the whole loop without energizing the arm"
    )
    track.add_argument("--no-window", action="store_true")
    track.add_argument("--yes", action="store_true", help="skip the enable confirmation")
    track.set_defaults(handler=cmd_track)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        raise SystemExit(args.handler(args))
    except PixelMapError as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
