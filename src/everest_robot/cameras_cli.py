"""``robot-cameras``: put every camera on screen with its id burned into the picture.

This is the tool that turns "which one of these is the wrist camera?" into a question you
answer by waving a hand in front of a lens. It exists because ``EVEREST_CAMERAS`` names
cameras the way a policy checkpoint names them (``wrist``, ``front``) while the host names
them by device id, and nothing in the running system can tell you which is which -- a
mis-paired id trains or runs a policy on the wrong view, and every downstream check still
passes because both cameras produce perfectly valid frames.

Two subcommands, in the order you need them:

* ``scan``  probes every capture device on the host, opens one window per camera that
  actually yields frames, and labels it with the id to paste into ``EVEREST_CAMERAS``.
  ``--json`` prints a configuration skeleton built from what it found.
* ``show``  opens the cameras that are *already* configured, through the same
  :class:`~everest_robot.robot.cameras.CameraRuntime` a policy rollout uses. A wrong id, a
  device another process is holding, or a resolution the driver quietly refused shows up
  here rather than at the first inference tick.

Read-only. It opens cameras, draws on the frames, and commands nothing -- it never claims
the robot lease, so it can run while the arm is held by something else. It does contend
for the *cameras*, though: stop this before starting a rollout.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from everest_robot.robot.cameras import CameraConfigError, CameraSpec, load_camera_specs

# LeRobot's own scan order, so the ids this prints are the ids it would report.
DEV_ROOT = Path("/dev")
BY_ID_DIR = Path("/dev/v4l/by-id")
MAX_OPENCV_INDEX = 8

UNOPENABLE = "did not open"
FRAMELESS = "opened but produced no frame"

DEFAULT_DISPLAY_WIDTH = 480
SNAPSHOT_DIR = Path("out/camera-ids")


class CameraViewerError(RuntimeError):
    """The viewer cannot do what was asked. Reported as a plain message, never a traceback."""


def load_cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as error:  # pragma: no cover - opencv is a core dependency
        raise CameraViewerError(
            "OpenCV is required for the camera windows. Install with: just setup"
        ) from error
    return cv2


# ── what there is to look at ───────────────────────────────────────────────────────


def _video_sort_key(path: Path) -> tuple[int, str]:
    """Sort ``video10`` after ``video9`` rather than after ``video1``."""

    suffix = path.name.removeprefix("video")
    return (int(suffix), "") if suffix.isdigit() else (1 << 30, path.name)


def candidate_targets(
    *,
    system: str | None = None,
    dev_root: Path = DEV_ROOT,
    max_index: int = MAX_OPENCV_INDEX,
) -> tuple[str, ...]:
    """Every capture device worth probing on this host.

    Linux enumerates real device nodes, so scan them: an index is just the node's number
    and skipping the filesystem would hide the gaps that appear once a camera is unplugged.
    Everywhere else there is nothing to enumerate, so probe indices and let the open fail.
    """

    system = platform.system() if system is None else system
    if system == "Linux":
        return tuple(str(path) for path in sorted(dev_root.glob("video*"), key=_video_sort_key))
    return tuple(str(index) for index in range(max_index))


def stable_ids(by_id_dir: Path = BY_ID_DIR) -> dict[str, str]:
    """Map each ``/dev/videoN`` to a by-id link that survives a replug, where one exists.

    ``/dev/videoN`` numbers are handed out in probe order, so they move when a camera is
    unplugged, when another is added, and sometimes across a plain reboot -- which silently
    swaps ``wrist`` and ``front`` in a configuration that was correct yesterday. The by-id
    link is derived from the device's own USB identity and does not move, so it is what
    this tool recommends putting in ``EVEREST_CAMERAS``.
    """

    if not by_id_dir.is_dir():
        return {}
    mapping: dict[str, str] = {}
    for link in sorted(by_id_dir.iterdir()):
        try:
            target = link.resolve(strict=True)
        except OSError:
            continue
        mapping.setdefault(str(target), str(link))
    return mapping


# ── one camera, as the operator needs to see it ────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CameraView:
    """What one window shows: which device it is, and what it actually delivers.

    ``configured`` is present only in ``show``, where there is a declared shape to hold the
    delivered one against. Shapes are LeRobot's ``(height, width, channels)`` throughout,
    so they compare directly against ``CameraSpec.frame_shape``.
    """

    target: str
    name: str | None = None
    stable_id: str | None = None
    delivered: tuple[int, ...] | None = None
    configured: tuple[int, int, int] | None = None
    driver_fps: float | None = None
    problem: str | None = None

    @property
    def key(self) -> str:
        """How this camera is titled and filed. Unique across a run."""

        return self.target if self.name is None else f"{self.name} ({self.target})"

    @property
    def recommended_id(self) -> str:
        """The id to configure: the stable link when the host offers one."""

        return self.stable_id or self.target


def shape_mismatch(view: CameraView) -> str | None:
    """The one failure a picture cannot show: a driver that ignored the requested size.

    A camera asked for 1280x720 that hands back 640x480 still looks perfectly correct in
    the window. It is the checkpoint's feature shape that breaks, at the first inference.
    """

    if view.configured is None or view.delivered is None or view.configured == view.delivered:
        return None
    return (
        f"MISMATCH: configured {view.configured[1]}x{view.configured[0]}, "
        f"delivering {view.delivered[1]}x{view.delivered[0]}"
    )


def elide_middle(text: str, max_chars: int) -> str:
    """Shorten from the middle, because both ends of a by-id name identify the camera.

    ``usb-Chicony_Electronics_Co._Ltd._Integrated_Camera-video-index0`` is wider than any
    window it would be drawn in. Cutting the tail loses ``index0`` versus ``index1``, which
    is exactly what distinguishes a camera's video node from its metadata node.
    """

    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return "." * max_chars
    keep = max_chars - 3
    head = (keep + 1) // 2
    tail = keep - head
    return f"{text[:head]}...{text[len(text) - tail:]}" if tail else f"{text[:head]}..."


def overlay_lines(view: CameraView, measured_fps: float | None = None) -> tuple[str, ...]:
    """The label drawn into the frame.

    Drawn into the picture rather than left to the window title because the title is what
    gets truncated, hidden by a tiling window manager, and lost the moment anyone takes a
    screenshot of four stacked windows to ask which one is which.
    """

    head = f"id {view.target}" if view.name is None else f"{view.name}   id {view.target}"
    lines = [head]
    if view.stable_id and view.stable_id != view.target:
        lines.append(f"stable id  {view.stable_id}")
    if view.delivered is not None:
        height, width = view.delivered[0], view.delivered[1]
        rate = "" if measured_fps is None else f"   {measured_fps:.0f} fps"
        lines.append(f"{width}x{height}{rate}")
    if (mismatch := shape_mismatch(view)) is not None:
        lines.append(mismatch)
    if view.problem:
        lines.append(view.problem)
    return tuple(lines)


def cameras_env_document(views: Sequence[CameraView]) -> list[dict[str, Any]]:
    """An ``EVEREST_CAMERAS`` skeleton for what was found, for the operator to rename.

    The names are placeholders on purpose. Which camera is ``wrist`` is exactly the fact
    this tool exists to establish by eye, and inventing it here would defeat the point.
    """

    document = []
    for index, view in enumerate(views):
        height, width = (view.delivered or (0, 0))[0], (view.delivered or (0, 0))[1]
        document.append(
            {
                "name": view.name or f"camera{index}",
                "kind": "opencv",
                "index_or_path": view.recommended_id,
                "width": int(width),
                "height": int(height),
                "fps": int(round(view.driver_fps or 30.0)),
            }
        )
    return document


def window_positions(
    count: int, *, width: int, height: int, screen_width: int = 1920, margin: int = 24
) -> tuple[tuple[int, int], ...]:
    """Lay the windows out in a grid so they do not open on top of one another.

    Stacked windows are the whole problem: an operator comparing views has to be able to
    see them side by side without dragging four of them apart first.
    """

    columns = max(1, screen_width // max(1, width + margin))
    return tuple(
        ((index % columns) * (width + margin), (index // columns) * (height + margin + 32))
        for index in range(count)
    )


# ── drawing ────────────────────────────────────────────────────────────────────────


def _draw_label(cv2: Any, frame: Any, lines: Sequence[str]) -> Any:
    """Burn the label into a copy of the frame.

    Dimmed band, then outlined text: the label has to stay legible over a bright ceiling
    and a dark bench alike, because which camera an operator is looking at is the entire
    output of this program.
    """

    canvas = frame.copy()
    band = min(canvas.shape[0], 12 + 24 * len(lines))
    shade = canvas.copy()
    cv2.rectangle(shade, (0, 0), (canvas.shape[1], band), (0, 0, 0), -1)
    cv2.addWeighted(shade, 0.45, canvas, 0.55, 0, canvas)

    for row, text in enumerate(lines):
        origin = (10, 26 + 24 * row)
        scale = 0.7 if row == 0 else 0.45
        colour = (0, 255, 255) if row == 0 else (255, 255, 255)
        if text.startswith(("MISMATCH", "no frame")):
            colour = (60, 120, 255)
        # ~10.5px per character at this font and scale; keep the text inside the frame.
        fitted = elide_middle(text, int((canvas.shape[1] - 20) / (15.0 * scale)))
        cv2.putText(canvas, fitted, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
        cv2.putText(canvas, fitted, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1)
    return canvas


def _scaled(cv2: Any, frame: Any, display_width: int) -> Any:
    if display_width <= 0 or frame.shape[1] <= display_width:
        return frame
    height = max(1, round(frame.shape[0] * display_width / frame.shape[1]))
    return cv2.resize(frame, (display_width, height), interpolation=cv2.INTER_AREA)


class _Rate:
    """A smoothed frame rate, so the number on screen does not flicker every tick."""

    def __init__(self) -> None:
        self.value: float | None = None
        self._previous: float | None = None

    def tick(self, now: float) -> None:
        if self._previous is not None and now > self._previous:
            instant = 1.0 / (now - self._previous)
            self.value = instant if self.value is None else 0.9 * self.value + 0.1 * instant
        self._previous = now


# ── the shared window loop ─────────────────────────────────────────────────────────

ReadAll = Callable[[], Mapping[str, Any]]


def _display_loop(
    cv2: Any,
    views: Sequence[CameraView],
    read_all: ReadAll,
    *,
    display_width: int,
    rate_hz: float,
) -> None:
    """One window per view, redrawn until the operator quits.

    A camera that stops delivering holds its last frame with the failure written across it
    rather than closing its window: a window that vanishes tells the operator nothing about
    which camera went away.
    """

    titles = {view.key: f"robot-cameras: {view.key}" for view in views}
    rates = {view.key: _Rate() for view in views}
    last: dict[str, Any] = {}
    placed = False
    period = 1.0 / rate_hz if rate_hz > 0 else 0.0

    try:
        for title in titles.values():
            cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    except cv2.error as error:
        raise CameraViewerError(_no_window_message(error)) from None

    print("\nkeys:  q / Esc  quit    s  save a snapshot of every window")
    try:
        while True:
            started = time.monotonic()
            frames = read_all()
            for view in views:
                frame = frames.get(view.key)
                if frame is not None:
                    last[view.key] = frame
                    rates[view.key].tick(started)
                shown = last.get(view.key)
                if shown is None:
                    continue
                stale = None if frame is not None else "no frame -- camera stopped delivering"
                labelled = _draw_label(
                    cv2,
                    _scaled(cv2, shown, display_width),
                    overlay_lines(
                        view if stale is None else replace(view, problem=stale),
                        rates[view.key].value,
                    ),
                )
                cv2.imshow(titles[view.key], labelled)

            if not placed and last:
                _place_windows(cv2, views, titles, last, display_width)
                placed = True

            key = cv2.waitKey(1) & 0xFF
            if key in {ord("q"), 27}:
                break
            if key == ord("s"):
                _save_snapshots(cv2, views, last)

            if period:
                remaining = period - (time.monotonic() - started)
                if remaining > 0:
                    time.sleep(remaining)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        # Some backends only actually tear the windows down on the next event pump.
        for _ in range(4):
            cv2.waitKey(1)


def _no_window_message(error: Exception) -> str:
    """Name the one cause that is otherwise a bewildering OpenCV traceback.

    ``lerobot`` depends on opencv-python-headless, which installs the same ``cv2`` module
    as opencv-python and overwrites it. So the hardware environment -- the robot host,
    where the camera actually is -- ends up with a cv2 that has no ``imshow`` at all, and
    every window in this repository fails the same way.
    """

    return (
        f"OpenCV cannot open a window: {error}\n"
        "The usual cause is a headless OpenCV build: lerobot depends on "
        "opencv-python-headless, which ships the same 'cv2' module as opencv-python and "
        "overwrites it, so `just setup-hardware` can leave a cv2 with no imshow. Repair it "
        "with `uv pip install --reinstall opencv-python`, or use --no-window, which reports "
        "everything except the picture."
    )


def _place_windows(
    cv2: Any,
    views: Sequence[CameraView],
    titles: Mapping[str, str],
    last: Mapping[str, Any],
    display_width: int,
) -> None:
    sample = next(iter(last.values()))
    width = min(display_width, sample.shape[1]) if display_width > 0 else sample.shape[1]
    height = round(sample.shape[0] * width / sample.shape[1])
    positions = window_positions(len(views), width=width, height=height)
    for view, (x, y) in zip(views, positions, strict=True):
        try:
            cv2.moveWindow(titles[view.key], x, y)
        except cv2.error:  # pragma: no cover - depends on the window backend
            return


def _save_snapshots(cv2: Any, views: Sequence[CameraView], last: Mapping[str, Any]) -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time())
    for view in views:
        frame = last.get(view.key)
        if frame is None:
            continue
        safe = view.key.replace("/", "_").replace(" ", "_")
        path = SNAPSHOT_DIR / f"{stamp}-{safe}.png"
        cv2.imwrite(str(path), _draw_label(cv2, frame, overlay_lines(view)))
        print(f"saved {path}")


# ── scan ───────────────────────────────────────────────────────────────────────────


def _capture_api(cv2: Any, backend: str) -> int:
    apis = {
        "auto": cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY,
        "avfoundation": cv2.CAP_AVFOUNDATION,
        "v4l2": cv2.CAP_V4L2,
        "any": cv2.CAP_ANY,
    }
    if backend not in apis:
        raise CameraViewerError(f"unknown --backend {backend!r} (expected {', '.join(apis)})")
    return apis[backend]


def _fourcc(cv2: Any, code: str) -> float:
    fourcc = getattr(cv2, "VideoWriter_fourcc", None) or cv2.VideoWriter.fourcc
    return float(fourcc(*code))


def _first_frame(capture: Any, attempts: int) -> Any:
    """A frame, retrying briefly.

    The first few reads fail routinely while a capture session spins up, and on Linux the
    metadata node that every UVC camera exposes alongside its video node opens cleanly and
    then never yields anything -- which is exactly how this tells the two apart.
    """

    for attempt in range(attempts):
        ok, frame = capture.read()
        if ok and frame is not None:
            return frame
        if attempt + 1 < attempts:
            time.sleep(0.05)
    return None


def _open_scan_targets(cv2: Any, targets: Iterable[str], args: argparse.Namespace) -> tuple[
    list[tuple[CameraView, Any]], list[CameraView]
]:
    """Open every candidate, keeping the ones that deliver a frame.

    Cameras are opened cumulatively and left open, because seeing them side by side is the
    point. That is also the one thing that can fail for a reason other than a bad id: several
    uncompressed streams can exceed what one USB controller will carry, which is what
    ``--fourcc MJPG`` and ``--only`` are for.
    """

    api = _capture_api(cv2, args.backend)
    stable = stable_ids()
    opened: list[tuple[CameraView, Any]] = []
    skipped: list[CameraView] = []

    for target in targets:
        index_or_path: int | str = int(target) if target.isdigit() else target
        capture = cv2.VideoCapture(index_or_path, api)
        if not capture.isOpened():
            capture.release()
            skipped.append(CameraView(target, problem=UNOPENABLE))
            continue
        if args.fourcc:
            capture.set(cv2.CAP_PROP_FOURCC, _fourcc(cv2, args.fourcc))
        if args.width:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        if args.height:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

        frame = _first_frame(capture, args.warmup_reads)
        if frame is None:
            capture.release()
            skipped.append(CameraView(target, problem=FRAMELESS))
            continue
        opened.append(
            (
                CameraView(
                    target=target,
                    stable_id=stable.get(target),
                    delivered=tuple(frame.shape),
                    driver_fps=capture.get(cv2.CAP_PROP_FPS) or None,
                ),
                capture,
            )
        )
    return opened, skipped


def _report(opened: Sequence[CameraView], skipped: Sequence[CameraView]) -> None:
    print(f"{len(opened)} camera(s) delivering frames:")
    for view in opened:
        height, width = (view.delivered or (0, 0))[0], (view.delivered or (0, 0))[1]
        fps = "" if view.driver_fps is None else f" @ {view.driver_fps:g} fps"
        print(f"  {view.target:<28} {width}x{height}{fps}")
        if view.stable_id and view.stable_id != view.target:
            print(f"  {'':<28} configure as {view.stable_id}")
    if skipped:
        print(f"\n{len(skipped)} probed and skipped:")
        for view in skipped:
            print(f"  {view.target:<28} {view.problem}")
        if any(view.problem == FRAMELESS for view in skipped):
            print(
                "  (a device that opens but never delivers is normally the metadata node a\n"
                "   UVC camera exposes next to its video node -- not a camera to configure)"
            )


def scan(args: argparse.Namespace) -> int:
    cv2 = load_cv2()
    targets = tuple(args.only) if args.only else candidate_targets()
    if not targets:
        raise CameraViewerError(
            "no capture devices found. On Linux check that /dev/video* exists and that you "
            "are in the 'video' group; elsewhere pass --only with an index."
        )

    opened, skipped = _open_scan_targets(cv2, targets, args)
    try:
        views = [view for view, _ in opened]
        _report(views, skipped)
        if args.json:
            print("\nEVEREST_CAMERAS skeleton (rename each camera to what the policy calls it):")
            print(json.dumps(cameras_env_document(views), indent=2))
        if not views:
            return 1
        if args.no_window:
            return 0

        captures = {view.key: capture for view, capture in opened}

        def read_all() -> dict[str, Any]:
            frames = {}
            for key, capture in captures.items():
                ok, frame = capture.read()
                frames[key] = frame if ok else None
            return frames

        _display_loop(
            cv2, views, read_all, display_width=args.display_width, rate_hz=args.rate
        )
        return 0
    finally:
        for _, capture in opened:
            capture.release()


# ── show ───────────────────────────────────────────────────────────────────────────


def _configured_specs(args: argparse.Namespace) -> tuple[CameraSpec, ...]:
    try:
        specs = load_camera_specs()
    except CameraConfigError as error:
        raise CameraViewerError(str(error)) from None
    if not specs:
        raise CameraViewerError(
            "no cameras configured. Set EVEREST_CAMERAS (or EVEREST_CAMERAS_FILE) -- run "
            "`just camera-scan-json` to get a skeleton built from what this host can see."
        )
    if args.camera:
        wanted = set(args.camera)
        unknown = sorted(wanted - {spec.name for spec in specs})
        if unknown:
            known = ", ".join(spec.name for spec in specs)
            raise CameraViewerError(
                f"no configured camera named {', '.join(unknown)} (have {known})"
            )
        specs = tuple(spec for spec in specs if spec.name in wanted)
    return specs


def show(args: argparse.Namespace) -> int:
    """Open the configured cameras the way a rollout opens them, and label each by name."""

    cv2 = load_cv2()
    specs = _configured_specs(args)

    from everest_robot.robot.cameras import CameraRuntime

    stable = stable_ids()
    runtime = CameraRuntime.from_specs(specs)
    print(f"opening {len(specs)} configured camera(s) through CameraRuntime:")
    for spec in specs:
        print(f"  {spec.name:<12} {spec.kind:<10} {spec.index_or_path}  {spec.width}x{spec.height}")
    try:
        runtime.connect()
    except Exception as error:
        raise CameraViewerError(
            f"connecting the configured cameras failed: {error}\n"
            "Check the ids with `just camera-scan`, and that no rollout or preview holds them."
        ) from None

    try:
        first = runtime.observation(timeout_ms=args.timeout_ms)
        views = [
            CameraView(
                target=spec.index_or_path,
                name=spec.name,
                stable_id=stable.get(spec.index_or_path),
                delivered=tuple(first[spec.name].shape),
                configured=spec.frame_shape,
                driver_fps=float(spec.fps),
            )
            for spec in specs
        ]
        print("\ndelivering:")
        for view in views:
            delivered = view.delivered or ()
            shape = "x".join(str(size) for size in (delivered[1], delivered[0]))
            print(f"  {view.name:<12} {shape}   {shape_mismatch(view) or 'matches configuration'}")
            if view.stable_id and view.stable_id != view.target:
                print(
                    f"  {'':<12} configured by a device number that moves; prefer "
                    f"{view.stable_id}"
                )

        mismatched = [view for view in views if shape_mismatch(view)]
        if mismatched:
            print(
                "\nA delivered shape that is not the configured one fails the checkpoint's\n"
                "feature check and INITIAL's readiness gate. Fix EVEREST_CAMERAS to the size\n"
                "the driver actually gives, or ask the camera for one it supports."
            )
        if args.no_window:
            return 1 if mismatched else 0

        # LeRobot's cameras hand back RGB; OpenCV draws and shows BGR. Converting here
        # rather than reading the device raw keeps this looking at exactly the array the
        # policy is handed.
        to_bgr = args.color == "rgb"

        def read_all() -> dict[str, Any]:
            observation = runtime.observation(timeout_ms=args.timeout_ms)
            frames = {}
            for view in views:
                frame = observation.get(view.name)
                if frame is None:
                    continue
                frames[view.key] = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if to_bgr else frame
            return frames

        _display_loop(
            cv2, views, read_all, display_width=args.display_width, rate_hz=args.rate
        )
        return 1 if mismatched else 0
    finally:
        runtime.disconnect()


# ── cli ────────────────────────────────────────────────────────────────────────────


def _add_window_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--display-width",
        type=int,
        default=DEFAULT_DISPLAY_WIDTH,
        help="downscale each window to this width for display only (0 keeps full size)",
    )
    parser.add_argument("--rate", type=float, default=30.0, help="redraw rate, Hz")
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="probe and report without opening any window, for a headless or ssh session",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="robot-cameras",
        description="Show every camera in its own window, labelled with the id to configure.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    scan_parser = subcommands.add_parser(
        "scan", help="probe every capture device on the host and label each by id"
    )
    scan_parser.add_argument(
        "--only",
        nargs="+",
        metavar="ID",
        help="probe just these ids (an index or a device path) instead of every device",
    )
    scan_parser.add_argument("--width", type=int, default=None, help="request this frame width")
    scan_parser.add_argument("--height", type=int, default=None, help="request this frame height")
    scan_parser.add_argument(
        "--fourcc",
        default=None,
        metavar="CODE",
        help="request a pixel format, e.g. MJPG -- the fix when several USB cameras "
        "cannot stream at once",
    )
    scan_parser.add_argument(
        "--backend", default="auto", choices=("auto", "avfoundation", "v4l2", "any")
    )
    scan_parser.add_argument(
        "--warmup-reads",
        type=int,
        default=12,
        help="reads to allow before calling a device frameless",
    )
    scan_parser.add_argument(
        "--json",
        action="store_true",
        help="also print an EVEREST_CAMERAS skeleton for what was found",
    )
    _add_window_arguments(scan_parser)
    scan_parser.set_defaults(handler=scan)

    show_parser = subcommands.add_parser(
        "show", help="open the cameras EVEREST_CAMERAS already configures, labelled by name"
    )
    show_parser.add_argument(
        "--camera",
        nargs="+",
        metavar="NAME",
        help="show only these configured cameras, by name",
    )
    show_parser.add_argument(
        "--color",
        default="rgb",
        choices=("rgb", "bgr"),
        help="how the camera's frames are ordered; LeRobot's cameras deliver rgb",
    )
    show_parser.add_argument("--timeout-ms", type=float, default=500.0)
    _add_window_arguments(show_parser)
    show_parser.set_defaults(handler=show)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        raise SystemExit(args.handler(args))
    except CameraViewerError as error:
        raise SystemExit(f"error: {error}") from None


if __name__ == "__main__":
    main()
