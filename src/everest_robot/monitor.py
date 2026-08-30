"""``robot-monitor``: teleoperate for calibration while watching joint feedback.

The default interactive mode owns both the follower and Star leader under one follower
lease, follows the leader at a bounded rate, and renders the same encoder TUI.  A separate
process cannot monitor an active controller safely because both would participate on the
CAN bus.  ``--read-only`` and ``--once`` retain the torque-disabled hand-teaching path.

It does claim the robot lease, and that is not incidental. Reading feedback from a merely
connected arm makes the driver poll the bus (``MakerArmPort.read_state`` refreshes when
the arm is not enabled), so a monitor is a bus participant, not a passive tap. Running one
alongside a worker would put two participants on one CAN bus, which this repo does not
treat as a degraded mode. If the lease is held, the monitor refuses and names the holder.

However a powered session ends -- ``q``, a teleoperation failure, an unhandled exception,
Ctrl-C -- the follower is driven back to ``--park`` (``zero`` by default) before torque is
released. A teleoperation session finishes with the arm wherever the operator's hands last
put the leader, and that is not a pose anyone chose to be safe to drop from. A second
Ctrl-C abandons the parking move; the physical e-stop overrides everything.

The derivation lives in :mod:`everest_robot.robot.monitor`; this module is the terminal.
"""

from __future__ import annotations

import argparse
import contextlib
import curses
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date
from typing import TYPE_CHECKING

from everest_robot.robot.contracts import ArmLifecycle
from everest_robot.robot.monitor import (
    DEFAULT_POLL_HZ,
    DEFAULT_STALE_AFTER_S,
    JointMonitor,
    JointReading,
    MonitorSample,
    format_table,
)

if TYPE_CHECKING:
    from everest_robot.robot.parameters import RobotParameters
    from everest_robot.robot.session import RobotSession
    from everest_robot.robot.teleoperation import TeleoperationController

# Column widths, and the terminal width each optional column needs before it earns its
# place. A 7-joint arm has to stay readable in an 80-column window.
_NAME_W, _RAD_W, _DEG_W, _DELTA_W = 16, 10, 9, 9
_VEL_W, _TORQUE_W, _TEMP_W, _FLAGS_W = 8, 9, 6, 13
_BASE_W = _NAME_W + _RAD_W + _DEG_W + _DELTA_W + _FLAGS_W
_NEEDS_VEL = _BASE_W + _VEL_W
_NEEDS_TEMP = _NEEDS_VEL + _TEMP_W
_NEEDS_TORQUE = _NEEDS_TEMP + _TORQUE_W
_BAR_W = 18
_NEEDS_BAR = _NEEDS_TORQUE + _BAR_W

# The keys, once. The footer is derived from the short labels and the help overlay from
# the long ones, so a binding cannot be added to the loop and forgotten in the guide.
# Where the follower is driven before torque comes off, however the session ends.
DEFAULT_PARK_POSITION = "zero"

_KEY_BINDINGS: tuple[tuple[str, str], ...] = (
    ("q", "quit+park"),
    ("p", "capture pose"),
    ("z", "mark reference"),
    ("Z", "clear"),
    ("space", "pause"),
    ("?", "help"),
)
_HELP = " · ".join(f"{key} {label}" for key, label in _KEY_BINDINGS)

_KEY_GUIDE: tuple[tuple[str, str], ...] = (
    ("q", "stop following, park the arm at its rest pose, release it, and exit"),
    ("p", "capture this pose; you are offered it as a named position on exit"),
    ("z", "mark this pose as the reference the d deg column measures from"),
    ("Z", "clear that reference"),
    ("space", "pause and resume"),
    ("?", "this guide"),
)

_COLUMN_GUIDE: tuple[tuple[str, str], ...] = (
    ("rad", "calibrated joint angle -- the unit presets are stored in"),
    ("deg", "the same angle in degrees"),
    ("d deg", "change since the pose marked with z, or -- when none is marked"),
    ("deg/s", "joint velocity"),
    ("degC", "motor temperature"),
    ("torque", "the torque the driver reports for this joint"),
    ("state", "ok, or why this joint's reading cannot be trusted"),
    ("soft limits", "the driver's limit span, o where the joint sits, ! outside it"),
)

_STATE_GUIDE: tuple[tuple[str, str], ...] = (
    ("ok", "reporting fresh feedback, inside its soft limits"),
    ("QUIET 1.2s", "the feedback counter has not advanced for that long"),
    ("OUT OF RANGE", "outside the driver's soft limits; motion refuses to go there"),
    ("NO FEEDBACK", "the motor reported nothing, so this joint is not measured"),
    ("FAULT 0x2", "the driver's fault bits for this joint"),
)


@dataclass(frozen=True, slots=True)
class MonitorContext:
    """What the header says about where these numbers are coming from."""

    robot_id: str
    model: str
    calibration_id: str
    config_digest: str
    poll_hz: float
    fake: bool
    powered: bool = False
    # The rest pose the follower is driven to before torque comes off, or None when the
    # operator asked for the arm to be released where it stands.
    park: str | None = None


# ── the terminal ───────────────────────────────────────────────────────────────────
class _Palette:
    """Curses attributes, or nothing at all on a terminal without colour."""

    def __init__(self) -> None:
        self.ok = self.warn = self.bad = self.dim = self.head = curses.A_NORMAL
        if not curses.has_colors():
            return
        curses.start_color()
        curses.use_default_colors()
        for index, colour in enumerate(
            (curses.COLOR_GREEN, curses.COLOR_YELLOW, curses.COLOR_RED, curses.COLOR_CYAN), start=1
        ):
            curses.init_pair(index, colour, -1)
        self.ok = curses.color_pair(1)
        self.warn = curses.color_pair(2)
        self.bad = curses.color_pair(3) | curses.A_BOLD
        self.head = curses.color_pair(4) | curses.A_BOLD
        self.dim = curses.A_DIM


def _put(screen: curses.window, row: int, column: int, text: str, attr: int = 0) -> None:
    """Write inside the window, or not at all.

    Curses raises when a write reaches the last cell, and a monitor that dies because
    someone narrowed their terminal would be a poor instrument.
    """

    height, width = screen.getmaxyx()
    if not 0 <= row < height or column >= width - 1:
        return
    # Defensive: the bounds above already hold, but a resize between getmaxyx() and the
    # write is exactly the race that would take the monitor down.
    with contextlib.suppress(curses.error):
        screen.addnstr(row, column, text, width - column - 1, attr)


def _bar(reading: JointReading, cells: int) -> str:
    """A soft-limit track with the joint's position marked on it."""

    fraction = reading.span_fraction
    if fraction is None:
        return "[" + "?" * cells + "]"
    marker = min(cells - 1, max(0, round(fraction * (cells - 1))))
    track = ["-"] * cells
    track[marker] = "!" if not reading.within_limits else "o"
    return "[" + "".join(track) + "]"


def _row(reading: JointReading, width: int) -> str:
    """One joint's line, dropping the least important columns on a narrow terminal."""

    if reading.has_feedback:
        cells = [f"{reading.name:<{_NAME_W}}", f"{reading.position_rad:>{_RAD_W}.4f}"]
        cells.append(f"{reading.position_deg:>{_DEG_W}.2f}")
    else:
        cells = [f"{reading.name:<{_NAME_W}}", f"{'--':>{_RAD_W}}", f"{'--':>{_DEG_W}}"]
    delta = reading.delta_deg
    cells.append(f"{'--' if delta is None else format(delta, '+.2f'):>{_DELTA_W}}")
    if width >= _NEEDS_VEL:
        cells.append(f"{reading.velocity_deg_s:>{_VEL_W}.2f}")
    if width >= _NEEDS_TEMP:
        cells.append(f"{reading.temperature_c:>{_TEMP_W}.1f}")
    if width >= _NEEDS_TORQUE:
        cells.append(f"{reading.torque:>{_TORQUE_W}.3f}")
    cells.append(f"  {_state_of(reading):<{_FLAGS_W - 2}}")
    if width >= _NEEDS_BAR:
        cells.append(_bar(reading, _BAR_W - 2))
    return "".join(cells)


def _header_row(width: int) -> str:
    cells = [
        f"{'joint':<{_NAME_W}}",
        f"{'rad':>{_RAD_W}}",
        f"{'deg':>{_DEG_W}}",
        f"{'d deg':>{_DELTA_W}}",
    ]
    if width >= _NEEDS_VEL:
        cells.append(f"{'deg/s':>{_VEL_W}}")
    if width >= _NEEDS_TEMP:
        cells.append(f"{'degC':>{_TEMP_W}}")
    if width >= _NEEDS_TORQUE:
        cells.append(f"{'torque':>{_TORQUE_W}}")
    cells.append(f"  {'state':<{_FLAGS_W - 2}}")
    if width >= _NEEDS_BAR:
        cells.append(f"{'soft limits':<{_BAR_W}}")
    return "".join(cells)


def _state_of(reading: JointReading) -> str:
    if not reading.has_feedback:
        return "NO FEEDBACK"
    if reading.has_fault:
        return f"FAULT {reading.fault_bits:#x}"
    if reading.stale:
        return f"QUIET {reading.quiet_for_s:.1f}s"
    if not reading.within_limits:
        return "OUT OF RANGE"
    return "ok"


def _attr_for(reading: JointReading, palette: _Palette) -> int:
    if not reading.has_feedback or reading.has_fault or reading.stale:
        return palette.bad
    if not reading.within_limits:
        return palette.warn
    return palette.ok


def _draw(
    screen: curses.window,
    monitor: JointMonitor,
    sample: MonitorSample,
    context: MonitorContext,
    palette: _Palette,
    *,
    paused: bool,
    captured: int | None = None,
    clamped: Sequence[str] = (),
) -> None:
    screen.erase()
    _, width = screen.getmaxyx()

    # The FAKE label must survive a narrow terminal: it is the difference between a
    # demonstration and a reading off a real machine.
    if context.fake:
        source = "FAKE ARM -- no hardware" if width >= 80 else "FAKE ARM"
    elif context.powered:
        source = "STAR TELEOP -- POWERED" if width >= 80 else "POWERED"
    else:
        source = "READ ONLY -- never enables" if width >= 80 else "READ ONLY"
    title = f"everest joint monitor  {context.robot_id} ({context.model})"
    _put(screen, 0, 0, title, palette.head)
    _put(
        screen,
        0,
        max(len(title) + 2, width - len(source) - 2),
        source,
        palette.warn if context.fake or context.powered else palette.dim,
    )
    _put(
        screen,
        1,
        0,
        f"calibration {context.calibration_id}  config {context.config_digest[:19]}",
        palette.dim,
    )
    _put(
        screen,
        2,
        0,
        _status_line(monitor, sample, context, paused=paused, captured=captured, clamped=clamped),
        palette.warn
        if paused
        or clamped
        or sample.lifecycle not in (ArmLifecycle.CONNECTED, ArmLifecycle.ENABLED)
        else palette.dim,
    )

    _put(screen, 4, 0, _header_row(width), palette.head)
    for offset, reading in enumerate(sample.readings):
        _put(screen, 5 + offset, 0, _row(reading, width), _attr_for(reading, palette))

    footer = 6 + len(sample.readings)
    _put(screen, footer, 0, _summary(sample), _summary_attr(sample, palette))
    _put(screen, footer + 1, 0, _HELP, palette.dim)
    screen.noutrefresh()
    curses.doupdate()


def _status_line(
    monitor: JointMonitor,
    sample: MonitorSample,
    context: MonitorContext,
    *,
    paused: bool,
    captured: int | None,
    clamped: Sequence[str],
) -> str:
    """The line under the header: what the session is doing, right now.

    Pressing p is otherwise invisible, which leaves the operator unsure whether the pose
    they meant to keep was taken; a clamped joint is likewise invisible, because the
    follower simply stops moving while the leader keeps going. Naming the joint is the
    difference between "the arm is stuck" and "that joint is at the end of its travel".
    """

    reference = "reference marked" if monitor.reference is not None else "no reference"
    parts = [
        f"lifecycle {sample.lifecycle.value.upper()}",
        f"{context.poll_hz:g} Hz",
        f"sample {sample.index}",
        reference,
    ]
    if paused:
        parts.append("FOLLOW PAUSED")
    if clamped:
        parts.append(f"CLAMPED: {', '.join(clamped)}")
    if captured is not None:
        parts.append(f"POSE HELD (sample {captured})")
    return "  ·  ".join(parts)


def help_lines(context: MonitorContext) -> list[str]:
    """The on-screen guide, as plain text so it can be checked without a terminal.

    An operator reading unfamiliar numbers off a powered arm should not have to leave the
    session to find out what a column means, so everything needed to trust or distrust a
    reading is here: the keys, the columns, the per-joint states, and what this particular
    mode is doing to the hardware.
    """

    lines = [f"everest joint monitor -- guide  ({context.robot_id})", ""]

    lines.append("WHAT THIS SESSION IS DOING")
    lines.extend(f"  {line}" for line in _mode_guide(context))
    lines.append("")

    lines.append("KEYS")
    lines.extend(f"  {key:<7} {description}" for key, description in _KEY_GUIDE)
    lines.append("")

    lines.append("COLUMNS")
    lines.extend(f"  {name:<12} {description}" for name, description in _COLUMN_GUIDE)
    lines.append("  narrow terminals drop columns from the right; rad and deg always stay")
    lines.append("")

    lines.append("STATE")
    lines.extend(f"  {name:<13} {description}" for name, description in _STATE_GUIDE)
    lines.append("")

    lines.append("CAPTURING A NAMED POSITION")
    lines.extend(
        f"  {line}"
        for line in (
            "Move the arm to the pose, press p, then q. The pose is printed and",
            "offered as a named position; `just goto <name>` drives back to it.",
            "Measure it three times from different directions first -- a pose that",
            "does not repeat is a fixture or calibration problem, not a preset.",
            "Saving does not make it safe: docs/named-position-capture.md step 3 is",
            "the reduced-speed ladder that does.",
        )
    )
    return lines


def _mode_guide(context: MonitorContext) -> tuple[str, ...]:
    """What this mode is doing to the arm. The one thing that must never be ambiguous."""

    if context.fake:
        return (
            "FAKE ARM. Every number below is generated by a deterministic stand-in.",
            "No CAN bus, no claim, nothing measured. A pose cannot be saved from here.",
        )
    if context.powered:
        lines = [
            "POWERED. The follower is enabled and tracking the Star leader at a",
            "bounded velocity. space pauses following and holds; q ends the session.",
            "This process holds the robot lease, so no worker can run.",
            "A leader pose the follower cannot reach is held at the soft limit and",
            "named as CLAMPED above; bring the leader back and following resumes.",
            "Staying out of range stops the session, because that is the mapping.",
        ]
        if context.park:
            lines += [
                "However the session ends -- q, a failure, Ctrl-C -- the arm is driven",
                f"back to {context.park!r} before torque comes off, so it is never dropped",
                "from a teleoperated pose. A second Ctrl-C abandons that move and releases",
                "the arm where it stands; the e-stop cuts power outright.",
            ]
        else:
            lines += [
                "--no-park was passed, so torque is released wherever the leader left the",
                "arm and it WILL fall unless you are supporting it.",
            ]
        return tuple(lines)
    return (
        "READ ONLY. The arm is never enabled, so you can position it by hand.",
        "space freezes the display. This process still holds the robot lease.",
    )


def _summary(sample: MonitorSample) -> str:
    problems = []
    if sample.fault_reason:
        problems.append(f"fault: {sample.fault_reason}")
    if sample.missing_feedback:
        problems.append(f"no feedback: {', '.join(sample.missing_feedback)}")
    if sample.stale_joints:
        problems.append(f"quiet: {', '.join(sample.stale_joints)}")
    if sample.out_of_limits:
        problems.append(f"outside soft limits: {', '.join(sample.out_of_limits)}")
    return "  ·  ".join(problems) if problems else "all joints reporting, all within soft limits"


def _summary_attr(sample: MonitorSample, palette: _Palette) -> int:
    if sample.has_fault or sample.missing_feedback or sample.stale_joints:
        return palette.bad
    if sample.out_of_limits:
        return palette.warn
    return palette.ok


def _show_help(
    screen: curses.window,
    context: MonitorContext,
    palette: _Palette,
    controller: TeleoperationController | None,
) -> None:
    """Draw the guide until a key dismisses it, scrolling when it does not fit.

    Never blocks: ``getch`` keeps the poll timeout, so a teleoperation failure ends the
    guide instead of waiting behind it. The controller holds the arm itself either way,
    but an operator reading help should not be the last to hear that following stopped.
    """

    lines = help_lines(context)
    top = 0
    while True:
        height, _ = screen.getmaxyx()
        body = max(1, height - 1)
        top = max(0, min(top, max(0, len(lines) - body)))
        scrollable = len(lines) > body

        screen.erase()
        for offset in range(min(body, len(lines) - top)):
            line = lines[top + offset]
            heading = line[:1].isupper() and line == line.upper() and not line.startswith(" ")
            _put(screen, offset, 0, line, palette.head if heading else 0)
        if scrollable:
            shown = f"{top + 1}-{min(top + body, len(lines))} of {len(lines)}"
            hint = f"up/down scroll · any other key returns  ({shown})"
        else:
            hint = "any key returns"
        _put(screen, height - 1, 0, hint, palette.dim)
        screen.noutrefresh()
        curses.doupdate()

        key = screen.getch()
        if controller is not None and controller.error:
            return
        if key in (-1, curses.KEY_RESIZE):
            continue
        if key in (curses.KEY_DOWN, ord("j")):
            top += 1
        elif key in (curses.KEY_UP, ord("k")):
            top -= 1
        elif key == curses.KEY_NPAGE:
            top += body
        elif key == curses.KEY_PPAGE:
            top -= body
        else:
            return


def run_tui(
    monitor: JointMonitor,
    context: MonitorContext,
    controller: TeleoperationController | None = None,
) -> MonitorSample | None:
    """Poll and redraw until the operator quits; return the last captured pose."""

    def loop(screen: curses.window) -> MonitorSample | None:
        curses.curs_set(0)
        screen.nodelay(False)
        # getch() doubles as the pacing wait: a keypress is answered at once, and a
        # timeout is the cue to poll the arm again.
        screen.timeout(max(1, int(1000.0 / context.poll_hz)))
        palette = _Palette()
        paused = False
        captured: MonitorSample | None = None
        sample = monitor.sample()
        while True:
            _draw(
                screen,
                monitor,
                sample,
                context,
                palette,
                paused=paused,
                captured=None if captured is None else captured.index,
                clamped=() if controller is None else controller.clamped_joints,
            )
            if controller is not None and controller.error:
                return captured
            key = screen.getch()
            if key in (ord("q"), ord("Q"), 27):
                return captured
            if key == ord("z"):
                monitor.mark_reference()
            elif key == ord("Z"):
                monitor.clear_reference()
            elif key == ord("p"):
                captured = monitor.sample()
                sample = captured
            elif key == ord(" "):
                paused = controller.toggle_pause() if controller is not None else not paused
            elif key in (ord("?"), ord("h"), curses.KEY_F1):
                _show_help(screen, context, palette, controller)
            elif key == curses.KEY_RESIZE:
                continue
            if not paused or controller is not None:
                sample = monitor.sample()

    return curses.wrapper(loop)


# ── entry point ────────────────────────────────────────────────────────────────────
def _build_fake_monitor(parameters: RobotParameters, stale_after_s: float) -> JointMonitor:
    """A monitor over :class:`FakeArm`, for looking at the display without an arm.

    The soft limits are invented here because real ones belong to the driver's hardware
    profile and there is no driver in this mode. Nothing measured in this mode means
    anything about a physical arm, which is why the header says FAKE ARM.
    """

    from everest_robot.robot.contracts import JointLimit
    from everest_robot.robot.fake_arm import FakeArm

    names = parameters.identity.joint_names
    limits = tuple(JointLimit(name, -2.0, 2.0) for name in names)
    # Fan the joints across their invented range so the display has something to show.
    positions = [
        limit.lower_rad + (limit.upper_rad - limit.lower_rad) * (index + 1) / (len(names) + 1)
        for index, limit in enumerate(limits)
    ]
    arm = FakeArm(parameters.identity, limits, positions=positions)
    arm.connect()
    return JointMonitor(arm, stale_after_s=stale_after_s)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Teleoperate from a Star 102 leader while watching follower feedback."
    )
    parser.add_argument("--poll-hz", type=float, default=DEFAULT_POLL_HZ)
    parser.add_argument(
        "--stale-after",
        type=float,
        default=DEFAULT_STALE_AFTER_S,
        help="seconds without a fresh feedback counter before a joint is flagged quiet",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="read-only: print one snapshot as plain text and exit",
    )
    parser.add_argument(
        "--fake",
        action="store_true",
        help="run against the deterministic FakeArm: no CAN, no claim, no real numbers",
    )
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="watch a torque-disabled arm without opening the Star leader",
    )
    parser.add_argument("--star-port", help="Star 102 serial port; defaults to EVEREST_STAR_PORT")
    parser.add_argument("--star-ids", default="0,1,2,3,4,5,6")
    parser.add_argument("--star-map", help="Star mapping JSON; defaults to maker-arm's profile")
    parser.add_argument("--leader-rate", type=float, default=25.0)
    parser.add_argument("--max-velocity", type=float, default=0.25)
    parser.add_argument("--leader-loss-timeout", type=float, default=0.5)
    parser.add_argument(
        "--out-of-range-timeout",
        type=float,
        default=2.0,
        help="seconds the leader may map outside the follower's soft limits before "
        "following stops; shorter excursions are clamped and shown as CLAMPED",
    )
    parser.add_argument("--sync-threshold", type=float, default=0.8)
    parser.add_argument(
        "--yes", action="store_true", help="accept a startup pose difference without prompting"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="print a captured pose but never offer to write it to the parameters file",
    )
    parser.add_argument(
        "--park",
        default=DEFAULT_PARK_POSITION,
        help=(
            "named position the follower returns to before torque comes off, however the "
            f"session ends (default {DEFAULT_PARK_POSITION!r})"
        ),
    )
    parser.add_argument(
        "--no-park",
        action="store_const",
        const=None,
        dest="park",
        help="release torque where the leader left the arm instead of returning it home",
    )
    args = parser.parse_args()
    if args.poll_hz <= 0:
        parser.error(f"--poll-hz must be positive, got {args.poll_hz}")

    from everest_robot.robot.deployment import build_lease, build_port, load_parameters
    from everest_robot.robot.session import RobotSession

    parameters = load_parameters()
    identity = parameters.identity
    context = MonitorContext(
        robot_id=identity.robot_id,
        model=identity.model,
        calibration_id=identity.calibration_id,
        config_digest=parameters.config_digest,
        poll_hz=args.poll_hz,
        fake=args.fake,
        powered=not (args.fake or args.once or args.read_only),
        park=args.park,
    )
    if not context.powered:
        # Nothing is energized in these modes, so there is nothing to park; saying so in
        # the guide beats implying an arm that is never under torque is about to be driven.
        context = replace(context, park=None)

    if args.fake:
        monitor = _build_fake_monitor(parameters, args.stale_after)
        captured = _present(monitor, context, once=args.once)
        _after_session(captured, context, parameters, save=not args.no_save)
        return

    # No cameras: calibration teleoperation and joint monitoring consume no images.
    #
    # Parking is unconditional in the powered mode, unlike `robot-goto`: a teleoperation
    # session ends with the arm wherever the operator's hands last put the leader, which is
    # exactly the pose nobody chose to be safe to release torque at. The read-only and
    # --once modes never enable, so they would skip parking anyway -- and they are asked for
    # precisely when the arm is not in a state to be driven, so they must not be refused
    # for a rest pose that is missing or unreachable.
    session = RobotSession(
        build_port(parameters),
        parameters,
        lease=build_lease(parameters),
        cameras=None,
        park_position=args.park if context.powered else None,
    )
    captured = _run_session(session, args, context)
    # Deliberately outside the session: `close()` has held the arm, disabled it and
    # released the lease, so nobody is waiting on a claimed robot while a name is typed.
    _after_session(captured, context, parameters, save=not args.no_save)


def _run_session(
    session: RobotSession,
    args: argparse.Namespace,
    context: MonitorContext,
) -> MonitorSample | None:
    """Own the arm for the length of the session and return the pose captured with ``p``."""

    import os

    with session:
        monitor = JointMonitor(session.port, stale_after_s=args.stale_after)
        if args.once or args.read_only:
            return _present(monitor, context, once=args.once)

        from everest_robot.robot.teleoperation import (
            Star102LeaderPort,
            TeleoperationController,
            load_star_mapper,
        )

        star_port = args.star_port or os.environ.get("EVEREST_STAR_PORT")
        if not star_port:
            raise SystemExit(
                "Star leader port is required: set EVEREST_STAR_PORT or pass --star-port; "
                "use --read-only to monitor without powered following"
            )
        try:
            star_ids = tuple(int(value.strip()) for value in args.star_ids.split(","))
        except ValueError as error:
            raise SystemExit("--star-ids must be comma-separated integers") from error
        controller = TeleoperationController(
            session.port,
            Star102LeaderPort(star_port, star_ids),
            load_star_mapper(args.star_map),
            rate_hz=args.leader_rate,
            max_velocity_rad_s=args.max_velocity,
            leader_loss_timeout_s=args.leader_loss_timeout,
            out_of_range_timeout_s=args.out_of_range_timeout,
        )
        try:
            difference = controller.connect_and_measure()
            if difference > args.sync_threshold and not args.yes:
                answer = input(
                    f"leader/follower pose difference is {difference:.2f} rad; "
                    "clear the area and type FOLLOW to enable > "
                )
                if answer.strip() != "FOLLOW":
                    raise SystemExit("teleoperation cancelled before enabling")
            controller.start()
            captured = _present(monitor, context, once=False, controller=controller)
        finally:
            controller.close()
        if controller.error:
            raise SystemExit(f"teleoperation stopped: {controller.error}")
        return captured


def _present(
    monitor: JointMonitor,
    context: MonitorContext,
    *,
    once: bool,
    controller: TeleoperationController | None = None,
) -> MonitorSample | None:
    if once:
        print(f"{context.robot_id} ({context.model})  calibration {context.calibration_id}")
        for line in format_table(monitor.sample()):
            print(line)
        return None
    if not sys.stdout.isatty():
        raise SystemExit(
            "robot-monitor needs a terminal; use --once for a single snapshot you can redirect"
        )
    with contextlib.suppress(KeyboardInterrupt):
        return run_tui(monitor, context, controller)
    return None


def _print_capture(sample: MonitorSample, context: MonitorContext) -> None:
    """Print copyable canonical radians after curses restores the terminal."""

    values = ", ".join(f"{reading.position_rad:.7f}" for reading in sample.readings)
    print("captured calibration pose (canonical radians):")
    print(f"  robot_id: {context.robot_id}")
    print(f"  calibration_id: {context.calibration_id}")
    print(f"  config_digest: {context.config_digest}")
    print(f"  joints: [{values}]")


def _after_session(
    captured: MonitorSample | None,
    context: MonitorContext,
    parameters: RobotParameters,
    *,
    save: bool,
) -> None:
    """Report the captured pose and, if it can be trusted, offer to record it."""

    if captured is None:
        return
    _print_capture(captured, context)
    if not save:
        return
    reason = _unsaveable(captured, context)
    if reason is not None:
        print(f"\nnot offering to save: {reason}")
        return
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return
    try:
        _prompt_save(captured, parameters)
    except (EOFError, KeyboardInterrupt):
        print("\nnot saved.")


def _unsaveable(sample: MonitorSample, context: MonitorContext) -> str | None:
    """Why this sample must not become a preset, if it must not.

    A preset is a claim about where a physical arm was. Fake numbers are not, and a joint
    that reported nothing or sits outside its own soft limits describes a pose the motion
    layer would refuse to drive to anyway.
    """

    if context.fake:
        return "--fake numbers describe no physical arm"
    missing = sample.missing_feedback
    if missing:
        return f"no feedback from {', '.join(missing)}"
    outside = sample.out_of_limits
    if outside:
        return f"{', '.join(outside)} outside the driver's soft limits"
    if sample.has_fault:
        return "the arm reported a fault while this pose was measured"
    return None


def _prompt_save(sample: MonitorSample, parameters: RobotParameters) -> None:
    """Ask what to call the pose, and write it. A blank name skips the whole thing."""

    from everest_robot.robot.capture import CapturedPose, CaptureRefused, save_preset
    from everest_robot.robot.deployment import parameters_path

    path = parameters_path()
    print(f"\nSave this pose to {path} as a named position?")
    name = input("  name (blank to skip) > ").strip()
    if not name:
        return

    replace = name in parameters.named_positions
    if replace:
        print(f"  {name!r} already exists. Recapturing it is a re-approval: every")
        print("  transition to it has to be re-validated afterwards.")
        if input("  type REPLACE to overwrite > ").strip() != "REPLACE":
            print("left unchanged.")
            return

    approved_by = input("  approved by > ").strip()
    if not approved_by:
        print("not saved: a preset records who approved the pose, and that cannot be blank.")
        return
    notes = input("  notes (optional) > ").strip()

    pose = CapturedPose(
        name=name,
        joints=tuple(reading.position_rad for reading in sample.readings),
        calibration_id=parameters.identity.calibration_id,
        approved_by=approved_by,
        captured_at=date.today(),
        notes=notes or None,
    )
    try:
        written = save_preset(path, pose, replace=replace)
    except CaptureRefused as error:
        print(f"not saved: {error}", file=sys.stderr)
        return

    print(f"\nwrote named position {name!r} to {written}")
    print("It is not validated yet. Before anything relies on it:")
    print(f"  just goto-dry {name}          # planned motion, nothing energized")
    print(f"  just goto {name} 0.25         # then 0.5, then 1.0")
    print("from every pose the move can start at -- docs/named-position-capture.md step 3.")


if __name__ == "__main__":
    main()
