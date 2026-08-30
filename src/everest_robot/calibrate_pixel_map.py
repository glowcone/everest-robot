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
   the camera sees, at a locked speed, holding still whenever it sees nothing. With
   ``--policy`` this is the approach half of a pick: once the arm has settled on the
   tracked pose, tracking stops and a trained checkpoint (ACT, SmolVLA, anything else
   LeRobot loads) is handed the arm to do the rest.

Everything lands in one JSON file (``--config``, default ``config/pixel_map.json``): the
camera id, the detector and its ROI, the arm and calibration the samples were taught on,
every sample, the fitted coefficients, the convex hull the fit is valid inside, and the
wrist-roll offset. One file is the whole calibration.

Why ``track`` needs no notion of height. The map is taught on *pre-grasp* poses, so
"directly above the carabiner" is what every sample already encodes; tracking to the map's
prediction tracks to directly-above by construction, and the policy takes over from that
same taught pose. Nothing here models the table plane, the camera's obliquity or the arm's
kinematics, which is the point -- :mod:`everest_robot.pixel_map` explains why.

The two controllers never overlap: the tracker is stopped and holding, and the fixed
camera released, before the rollout's first observation is read.

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
from dataclasses import dataclass
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

DEFAULT_CONFIG = "config/pixel_map.json"

# A detection that jumps further than this between frames is a different blob, not the
# same carabiner moving. Treated as a miss so the arm holds instead of lunging.
DEFAULT_MAX_JUMP_PX = 150.0

# How far each joint backs off before the final one-way move onto the target.
DEFAULT_APPROACH_RAD = 0.05

# When the servo counts as finished, for `track --policy`. Every joint's *measured*
# position must be this close to the tracked target for this many consecutive ticks --
# roughly a third of a second at the default rate, which is long enough that a detection
# flickering near the current pose does not read as an arrival.
DEFAULT_ARRIVE_RAD = 0.02
DEFAULT_ARRIVE_TICKS = 5

# Where the powered subcommands leave the arm before torque comes off. Both of them end
# with the arm held out over the pickup zone, which is the worst place to drop it.
DEFAULT_PARK_POSITION = "zero"


# ── camera ─────────────────────────────────────────────────────────────────────────
def _cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as error:
        raise PixelMapError(
            "OpenCV is required for the camera. Install with: uv add opencv-python"
        ) from error
    return cv2


def _capture_api(cv2: Any, backend: str) -> int:
    if backend == "auto":
        return cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    apis = {
        "avfoundation": cv2.CAP_AVFOUNDATION,
        "v4l2": cv2.CAP_V4L2,
        "any": cv2.CAP_ANY,
    }
    if backend not in apis:
        raise PixelMapError(f"unknown camera backend {backend!r} (expected {', '.join(apis)})")
    return apis[backend]


def open_capture(camera: CameraSource) -> Any:
    """Open the fixed camera, or say plainly which id failed."""

    cv2 = _cv2()
    try:
        index_or_path: int | str = int(camera.index_or_path)
    except ValueError:
        index_or_path = camera.index_or_path
    capture = cv2.VideoCapture(index_or_path, _capture_api(cv2, camera.backend))
    if not capture.isOpened():
        raise PixelMapError(
            f"could not open camera {camera.index_or_path!r} "
            f"(backend {camera.backend}); check the id and that nothing else holds it"
        )
    if camera.width:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera.width)
    if camera.height:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera.height)
    return capture


def read_frame(capture: Any, camera: CameraSource, *, attempts: int = 20) -> Any:
    """One frame, retrying briefly before giving up.

    A single failed ``read()`` means very little. AVFoundation returns false for the first
    few reads while the capture session spins up, and a momentary miss mid-session should
    not end a 25-minute teaching run. Only a sustained failure is real -- and on macOS its
    usual cause is another process already holding the device, because ``VideoCapture``
    still reports ``isOpened()`` in that case and simply never yields a frame.
    """

    import time

    for attempt in range(attempts):
        ok, frame = capture.read()
        if ok:
            return frame
        if attempt + 1 < attempts:
            time.sleep(0.05)
    raise PixelMapError(
        f"camera {camera.index_or_path!r} opened but produced no frame in {attempts} reads. "
        "Another process is probably holding it -- close any running "
        "robot-two-white-black-live, tools/preview.py or robot-pixel-map window. "
        "If nothing else is running, check the camera permission for your terminal."
    )


# ── detection ──────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class CarabinerPixel:
    """One detection, reduced to what the map consumes."""

    centroid: tuple[float, float]
    spine_rad: float
    white_a: tuple[float, float]
    white_b: tuple[float, float]
    gate: tuple[float, float]


def detect_carabiner(frame: Any, roi_xywh: Sequence[int]) -> CarabinerPixel:
    """The two-white-tape plus black-gate detection, reduced to a centroid and an angle.

    The centroid is the midpoint of the two white tapes -- the *object's* pixel, never the
    gripper's. Pairing that pixel with the joint vector that grasped it is what absorbs
    the camera's parallax into the fit instead of leaving it to be modelled.

    The spine angle is measured in image coordinates, where +v runs downward, so it is
    left-handed with respect to the arm. ``RollModel`` picks up the sign; nothing here
    needs to know how the camera is mounted.
    """

    from everest_robot.vision import detect_two_white_tapes_and_black_gate_bgr

    detection = detect_two_white_tapes_and_black_gate_bgr(
        frame, roi_xywh=(int(roi_xywh[0]), int(roi_xywh[1]), int(roi_xywh[2]), int(roi_xywh[3]))
    )
    a, b = detection.white_points_px
    gate = detection.black_gate_px
    midpoint = ((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)

    axis = (b.x - a.x, b.y - a.y)
    length = math.hypot(*axis)
    if length < 1e-6:
        raise PixelMapError("the two white tapes landed on the same pixel")
    axis = (axis[0] / length, axis[1] / length)
    # Two identical tapes leave the axis sign ambiguous; the gate resolves it, exactly as
    # `pickup.carabiner_pose_from_axis_points_and_side_point` does in the robot frame.
    left = (-axis[1], axis[0])
    if left[0] * (gate.x - midpoint[0]) + left[1] * (gate.y - midpoint[1]) < 0.0:
        axis = (-axis[0], -axis[1])

    return CarabinerPixel(
        centroid=midpoint,
        spine_rad=math.atan2(axis[1], axis[0]),
        white_a=(a.x, a.y),
        white_b=(b.x, b.y),
        gate=(gate.x, gate.y),
    )


def _draw(
    frame: Any, roi: Sequence[int], detection: CarabinerPixel | None, lines: list[str]
) -> None:
    cv2 = _cv2()
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

    cv2 = _cv2()
    cv2.polylines(frame, [np.asarray(hull, dtype=np.int32)], True, (255, 180, 0), 2)


# ── the robot side ─────────────────────────────────────────────────────────────────
def _open_session(args: argparse.Namespace) -> Any:
    """Claim and connect the deployed arm. The caller closes it.

    ``--park`` is handed to the session rather than run by the command, so the arm comes
    home on *every* exit -- a clean quit, a detector failure, an unhandled exception,
    Ctrl-C -- and not only on the paths a command remembered to cover.
    """

    from everest_robot.robot.deployment import build_lease, build_port, load_parameters
    from everest_robot.robot.session import RobotSession

    park = getattr(args, "park", None)
    if args.fake:
        from everest_robot.jog import open_fake_session

        print("FAKE ARM: nothing below this line describes a physical robot.\n")
        return open_fake_session(park)

    parameters = load_parameters()
    # No cameras on the session: the fixed camera here is not a policy observation, and
    # routing it through CameraRuntime would make it look like one.
    session = RobotSession(
        build_port(parameters),
        parameters,
        lease=build_lease(parameters),
        cameras=None,
        park_position=park,
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

    cv2 = _cv2()
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

        # Ctrl-C ends the teaching loop the same way `q` does. The samples taught so far
        # are written below either way: an interrupted session must not throw away the
        # poses the operator has already stood there and taught.
        try:
            with contextlib.suppress(KeyboardInterrupt):
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
        # The leader goes first, holding the follower where it is so it cannot drift while
        # the bus is handed over; the session then drives it home and releases torque. On
        # a hand-taught session (--no-teleop) the arm was never enabled, so the session
        # skips parking and it stays limp and movable, which is what that mode is for.
        if controller is not None:
            controller.close()
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

    With ``--policy`` the servo is the *approach* rather than the whole job: once the arm
    has actually settled on the tracked pose, tracking stops and the trained checkpoint
    takes over from there. The two never command the arm at the same time.
    """

    cv2 = _cv2()
    from everest_robot.robot.visual_tracking import ArrivalGate, TrackerStopped, VisualTracker

    calibration = PixelJointMap.load(args.config)
    if calibration.model is None:
        raise PixelMapError(f"{args.config} has no fit yet; run `robot-pixel-map fit` first")
    camera = _camera_for(args, calibration)
    roi = args.roi or calibration.detector.roi_xywh
    if roi is None:
        raise PixelMapError("this calibration stored no ROI; pass --roi X Y W H")

    # The checkpoint is resolved and loaded before the robot is claimed. A missing path, a
    # checkpoint trained on a different arm, or a VLA given no task is an operator's error,
    # and it must not cost a lease or leave an arm energized while a model downloads.
    handle = _load_policy(args)
    gate = ArrivalGate(tolerance_rad=args.arrive_rad, ticks=args.arrive_ticks)

    session = _open_session(args)
    capture = None
    tracker = None
    result = None
    try:
        if not args.fake:
            calibration.robot.verify(_stamp_of(session))
        if calibration.joint_names != tuple(session.port.joint_names):
            raise PixelMapError(
                f"calibration joints {', '.join(calibration.joint_names)} do not match this "
                f"arm's {', '.join(session.port.joint_names)}"
            )
        runner, policy_cameras = (None, None) if handle is None else _policy_runner(session, args)
        if handle is not None:
            # Schema, identity and camera checks now, while nothing is energized: a
            # checkpoint that cannot drive this arm should never get an approach move.
            _refuse_incompatible_policy(runner, handle, args)

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
        reasons, arrived = _track_loop(
            session, calibration, tracker, capture, camera, roi, args, cv2, gate
        )
        if arrived and handle is not None:
            # Hand over deliberately: stop servoing and hold, then let go of the fixed
            # camera before the policy's cameras are opened -- they may be the same device,
            # and a second reader gets a capture that opens and never yields a frame.
            tracker.stop()
            capture.release()
            capture = None
            if not args.no_window:
                with contextlib.suppress(Exception):
                    cv2.destroyAllWindows()
            result = _run_policy(runner, policy_cameras, handle, args)
    except TrackerStopped as error:
        print(f"tracker stopped: {error}", file=sys.stderr)
        return 1
    finally:
        # Stop servoing and hold, so the arm cannot drift on a stale target while the
        # camera and windows are torn down; the session then parks it and releases torque.
        # Parking lives in the session rather than here so it also covers the paths this
        # block never sees -- an exception before the tracker exists, or during teardown.
        if tracker is not None:
            tracker.stop()
        if capture is not None:
            capture.release()
        if not args.no_window:
            with contextlib.suppress(Exception):
                cv2.destroyAllWindows()
        session.close()

    print("\nticks by outcome:")
    for reason, count in sorted(reasons.items(), key=lambda item: -item[1]):
        print(f"  {count:>6}  {reason}")
    if handle is not None and result is None:
        if args.dry_run:
            # Nothing was commanded, so the arm cannot have arrived. The checkpoint was
            # still resolved and checked against this robot, which is what a rehearsal is.
            print(f"\ndry run: {handle.controller} was loaded and checked, never called.")
        else:
            print("\nthe arm never settled on a target, so the policy was not called.")
            return 1
    if result is not None:
        _print_policy_result(result)
        return 0 if result.failure_reason is None else 1
    return 0


# ── the policy handoff ─────────────────────────────────────────────────────────────
def _load_policy(args: argparse.Namespace) -> Any:
    """Load the checkpoint named on the command line, or return ``None``. No hardware."""

    if not args.policy:
        return None

    from everest_robot.robot.deployment import load_policy_handle
    from everest_robot.robot.policy import PolicyLoadError

    try:
        handle = load_policy_handle(
            args.policy,
            task=args.policy_task,
            device=args.policy_device,
            revision=args.policy_revision,
        )
    except PolicyLoadError as error:
        raise PixelMapError(str(error)) from None
    print(
        f"policy     {handle.controller} from {handle.checkpoint}"
        + (f'  task "{args.policy_task}"' if args.policy_task else "")
    )
    return handle


def _policy_runner(session: Any, args: argparse.Namespace) -> tuple[Any, Any]:
    """A rollout runner over the already-claimed arm, plus the cameras it will need.

    The cameras are built but not connected: the fixed calibration camera is still open at
    this point, and the rollout's cameras are only opened once it is released. They are the
    deployment's (``EVEREST_CAMERAS``), not this calibration's -- the fixed camera is what
    aims the arm, and the policy's own views are what it was trained on.
    """

    from everest_robot.robot.cameras import CameraRuntime
    from everest_robot.robot.deployment import joint_frame
    from everest_robot.robot.lerobot_bridge import RobotBridgeCore
    from everest_robot.robot.policy import PolicyRunner

    cameras = CameraRuntime.from_env()
    bridge = RobotBridgeCore(
        session.port, cameras=cameras, frame=joint_frame(session.parameters)
    )
    runner = PolicyRunner(
        bridge,
        session.parameters,
        allow_non_identity_frame=args.allow_frame_offsets,
    )
    return runner, cameras


def _refuse_incompatible_policy(runner: Any, handle: Any, args: argparse.Namespace) -> None:
    """Run every check the rollout would run, without energizing anything."""

    check = runner.run(handle, task=args.policy_task, fps=args.policy_fps, dry_run=True)
    if check.failure_reason is not None:
        raise PixelMapError(f"{check.failure_reason.value}: {check.failure_detail}")


def _run_policy(runner: Any, cameras: Any, handle: Any, args: argparse.Namespace) -> Any:
    """Hand the arm to the checkpoint. The tracker is already stopped and holding."""

    print(f"\narrived; handing over to {handle.controller}")
    cameras.connect()
    try:
        return runner.run(
            handle,
            task=args.policy_task,
            fps=args.policy_fps,
            max_steps=args.policy_steps,
            max_duration_s=args.policy_duration,
        )
    finally:
        cameras.disconnect()


def _print_policy_result(result: Any) -> None:
    print(f"\npolicy     {result.controller} from {result.checkpoint}")
    print(f"ended      {result.termination.value}")
    print(
        f"ran        {result.steps} steps in {result.elapsed_s:.1f} s at {result.fps:g} fps "
        f"({result.missed_deadlines} missed deadlines, "
        f"worst step {result.max_step_latency_s * 1000:.0f} ms)"
    )
    if result.clipped_joints:
        print(f"clipped    {', '.join(result.clipped_joints)}")
    if result.failure_reason is not None:
        print(f"failed     {result.failure_reason.value}: {result.failure_detail}", file=sys.stderr)


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
    gate: Any,
) -> tuple[dict[str, int], bool]:
    """Servo until the operator quits, the tracker stops, or the arm arrives.

    Arrival only ends the loop when there is something to hand over to; without
    ``--policy`` this tracks for as long as it is left running, which is what makes it
    usable for eyeballing a fresh calibration.
    """

    from everest_robot.robot.clock import SystemClock

    clock = SystemClock()
    reasons: dict[str, int] = {}
    previous: tuple[float, float] | None = None
    hand_over = args.policy is not None
    arrived = False

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
            arrived = gate.update(tick)

            if not args.no_window:
                lines = [
                    f"{'MOVING' if tick.moved else 'HOLD'}  {tick.reason}",
                    f"speed lock {args.max_velocity:g} rad/s"
                    f"  ({tracker.max_step_rad:.4f} rad/tick)",
                ]
                if tick.remaining_rad:
                    lines.append(f"remaining {tick.remaining_rad:.3f} rad")
                if hand_over:
                    lines.append(
                        f"settled {gate.consecutive}/{gate.ticks} ticks within "
                        f"{gate.tolerance_rad:g} rad"
                    )
                _draw(frame, roi, detection, lines)
                if calibration.hull is not None:
                    _draw_hull(frame, calibration.hull)
                cv2.imshow("pixel-map track", frame)
                if (cv2.waitKey(1) & 0xFF) in {ord("q"), 27}:
                    break
            elif tick.index % max(1, int(args.rate)) == 0:
                print(f"[{tick.index:>6}] {'move' if tick.moved else 'hold'}  {tick.reason}")

            if arrived and hand_over:
                break
            clock.sleep(tracker.period_s - (clock.monotonic() - started))

    if tracker.stopped_reason is not None:
        print(f"tracker stopped: {tracker.stopped_reason}", file=sys.stderr)
    return reasons, arrived and hand_over


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
def _add_park_arguments(parser: argparse.ArgumentParser) -> None:
    """Where the arm goes before torque comes off, for the two powered subcommands.

    The session enforces this on every exit path, so it is not a "when it ends cleanly"
    option: the whole point is that an exception or a Ctrl-C is covered too.
    """

    parser.add_argument(
        "--park",
        default=DEFAULT_PARK_POSITION,
        help=(
            "approved named position the arm is driven to before torque comes off, "
            f"however the command ends (default {DEFAULT_PARK_POSITION!r})"
        ),
    )
    parser.add_argument(
        "--no-park",
        action="store_const",
        const=None,
        dest="park",
        help="release torque where the arm stopped instead of driving it home first",
    )


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
    _add_park_arguments(collect)
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
    _add_park_arguments(track)
    track.add_argument(
        "--no-roll-tracking", action="store_true", help="leave wrist roll where it is"
    )
    track.add_argument(
        "--dry-run", action="store_true",
        help="run the whole loop without energizing the arm. The arm never arrives, so a "
             "--policy is loaded and checked but never called.",
    )
    track.add_argument("--no-window", action="store_true")
    track.add_argument("--yes", action="store_true", help="skip the enable confirmation")
    _add_policy_arguments(track)
    track.set_defaults(handler=cmd_track)
    return parser


def _add_policy_arguments(parser: argparse.ArgumentParser) -> None:
    """The handoff: what runs after the approach, and when the approach counts as done."""

    parser.add_argument(
        "--policy",
        help="MOVES THE ARM: a trained LeRobot checkpoint (a local directory or a hub id) "
             "handed the arm once it settles on the tracked pose. ACT, SmolVLA and any "
             "other architecture load through the same path -- the checkpoint names its "
             "own. Without this the loop just tracks.",
    )
    parser.add_argument(
        "--policy-task",
        help="the language instruction the checkpoint was trained with. Required for a "
             "language-conditioned policy (SmolVLA, pi0); ignored by ACT.",
    )
    parser.add_argument(
        "--policy-fps", type=float, default=None,
        help="rollout rate; must be the rate the checkpoint was trained at "
             "(default: the parameters file's policy.fps)",
    )
    parser.add_argument(
        "--policy-steps", type=int, default=None, help="stop the rollout after this many steps"
    )
    parser.add_argument(
        "--policy-duration", type=float, default=None,
        help="stop the rollout after this many seconds "
             "(default: the parameters file's policy.max_duration_s)",
    )
    parser.add_argument(
        "--policy-device", default=None,
        help="torch device for inference; defaults to EVEREST_POLICY_DEVICE, then to the "
             "checkpoint's own config",
    )
    parser.add_argument("--policy-revision", default=None, help="hub revision of the checkpoint")
    parser.add_argument(
        "--allow-frame-offsets", action="store_true",
        help="acknowledge that this arm's lerobot_frame has non-zero offsets and that the "
             "checkpoint was trained in that frame. See docs/lerobot-frame-reconciliation.md.",
    )
    parser.add_argument(
        "--arrive-rad", type=float, default=DEFAULT_ARRIVE_RAD,
        help="how close every joint's measured position must be to the tracked target "
             f"before the policy is called (default {DEFAULT_ARRIVE_RAD})",
    )
    parser.add_argument(
        "--arrive-ticks", type=int, default=DEFAULT_ARRIVE_TICKS,
        help=f"consecutive settled ticks the arrival needs (default {DEFAULT_ARRIVE_TICKS})",
    )


def main() -> None:
    from everest_robot.robot.routing import RouteRefused

    args = build_parser().parse_args()
    try:
        raise SystemExit(args.handler(args))
    # A --park name this arm does not have is refused in the session constructor, before
    # the claim. It is an operator's typo, so it gets an operator's message rather than a
    # traceback out of a teardown they have not reached yet.
    except (PixelMapError, RouteRefused) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
