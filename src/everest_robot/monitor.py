"""``robot-monitor``: watch every joint's encoder feedback in the terminal.

A read-only instrument. It claims the arm, connects, checks it is the arm the parameters
file describes, and then does nothing but call ``read_state()`` in a loop. It never
enables the motors and never sends a target, so the arm stays exactly as limp or as held
as you found it -- which is what makes it the right tool for step 1 of
docs/named-position-capture.md, where the operator moves the arm by hand and reads the
pose back.

It does claim the robot lease, and that is not incidental. Reading feedback from a merely
connected arm makes the driver poll the bus (``MakerArmPort.read_state`` refreshes when
the arm is not enabled), so a monitor is a bus participant, not a passive tap. Running one
alongside a worker would put two participants on one CAN bus, which this repo does not
treat as a degraded mode. If the lease is held, the monitor refuses and names the holder.

The derivation lives in :mod:`everest_robot.robot.monitor`; this module is the terminal.
"""

from __future__ import annotations

import argparse
import contextlib
import curses
import sys
from dataclasses import dataclass
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

_HELP = "q quit · z mark reference · Z clear · space pause"


@dataclass(frozen=True, slots=True)
class MonitorContext:
    """What the header says about where these numbers are coming from."""

    robot_id: str
    model: str
    calibration_id: str
    config_digest: str
    poll_hz: float
    fake: bool


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
) -> None:
    screen.erase()
    _, width = screen.getmaxyx()

    # The FAKE label must survive a narrow terminal: it is the difference between a
    # demonstration and a reading off a real machine.
    if context.fake:
        source = "FAKE ARM -- no hardware" if width >= 80 else "FAKE ARM"
    else:
        source = "READ ONLY -- never enables" if width >= 80 else "READ ONLY"
    title = f"everest joint monitor  {context.robot_id} ({context.model})"
    _put(screen, 0, 0, title, palette.head)
    _put(
        screen,
        0,
        max(len(title) + 2, width - len(source) - 2),
        source,
        palette.warn if context.fake else palette.dim,
    )
    _put(
        screen,
        1,
        0,
        f"calibration {context.calibration_id}  config {context.config_digest[:19]}",
        palette.dim,
    )
    reference = "reference marked" if monitor.reference is not None else "no reference"
    _put(
        screen,
        2,
        0,
        f"lifecycle {sample.lifecycle.value.upper()}  ·  {context.poll_hz:g} Hz  ·  "
        f"sample {sample.index}  ·  {reference}{'  ·  PAUSED' if paused else ''}",
        palette.warn if paused or sample.lifecycle is not ArmLifecycle.CONNECTED else palette.dim,
    )

    _put(screen, 4, 0, _header_row(width), palette.head)
    for offset, reading in enumerate(sample.readings):
        _put(screen, 5 + offset, 0, _row(reading, width), _attr_for(reading, palette))

    footer = 6 + len(sample.readings)
    _put(screen, footer, 0, _summary(sample), _summary_attr(sample, palette))
    _put(screen, footer + 1, 0, _HELP, palette.dim)
    screen.noutrefresh()
    curses.doupdate()


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


def run_tui(monitor: JointMonitor, context: MonitorContext) -> None:
    """Poll and redraw until the operator quits."""

    def loop(screen: curses.window) -> None:
        curses.curs_set(0)
        screen.nodelay(False)
        # getch() doubles as the pacing wait: a keypress is answered at once, and a
        # timeout is the cue to poll the arm again.
        screen.timeout(max(1, int(1000.0 / context.poll_hz)))
        palette = _Palette()
        paused = False
        sample = monitor.sample()
        while True:
            _draw(screen, monitor, sample, context, palette, paused=paused)
            key = screen.getch()
            if key in (ord("q"), ord("Q"), 27):
                return
            if key == ord("z"):
                monitor.mark_reference()
            elif key == ord("Z"):
                monitor.clear_reference()
            elif key == ord(" "):
                paused = not paused
            elif key == curses.KEY_RESIZE:
                continue
            if not paused:
                sample = monitor.sample()

    curses.wrapper(loop)


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
        description="Watch every joint's encoder feedback. Reads only; never enables the arm."
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
        help="print one snapshot as plain text and exit; works when redirected",
    )
    parser.add_argument(
        "--fake",
        action="store_true",
        help="run against the deterministic FakeArm: no CAN, no claim, no real numbers",
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
    )

    if args.fake:
        monitor = _build_fake_monitor(parameters, args.stale_after)
        _present(monitor, context, once=args.once)
        return

    # No cameras: a joint monitor has no use for them, and opening a camera is a side
    # effect this program has no business having.
    session = RobotSession(
        build_port(parameters), parameters, lease=build_lease(parameters), cameras=None
    )
    with session:
        _present(
            JointMonitor(session.port, stale_after_s=args.stale_after), context, once=args.once
        )


def _present(monitor: JointMonitor, context: MonitorContext, *, once: bool) -> None:
    if once:
        print(f"{context.robot_id} ({context.model})  calibration {context.calibration_id}")
        for line in format_table(monitor.sample()):
            print(line)
        return
    if not sys.stdout.isatty():
        raise SystemExit(
            "robot-monitor needs a terminal; use --once for a single snapshot you can redirect"
        )
    with contextlib.suppress(KeyboardInterrupt):
        run_tui(monitor, context)


if __name__ == "__main__":
    main()
