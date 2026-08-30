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

_HELP = "q quit+hold · p capture pose · z mark reference · Z clear · space pause follow"


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
    reference = "reference marked" if monitor.reference is not None else "no reference"
    _put(
        screen,
        2,
        0,
        f"lifecycle {sample.lifecycle.value.upper()}  ·  {context.poll_hz:g} Hz  ·  "
        f"sample {sample.index}  ·  {reference}{'  ·  FOLLOW PAUSED' if paused else ''}",
        palette.warn
        if paused or sample.lifecycle not in (ArmLifecycle.CONNECTED, ArmLifecycle.ENABLED)
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
            _draw(screen, monitor, sample, context, palette, paused=paused)
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
    parser.add_argument("--sync-threshold", type=float, default=0.8)
    parser.add_argument(
        "--yes", action="store_true", help="accept a startup pose difference without prompting"
    )
    args = parser.parse_args()
    if args.poll_hz <= 0:
        parser.error(f"--poll-hz must be positive, got {args.poll_hz}")

    import os

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
    )

    if args.fake:
        monitor = _build_fake_monitor(parameters, args.stale_after)
        _present(monitor, context, once=args.once)
        return

    # No cameras: calibration teleoperation and joint monitoring consume no images.
    session = RobotSession(
        build_port(parameters), parameters, lease=build_lease(parameters), cameras=None
    )
    with session:
        monitor = JointMonitor(session.port, stale_after_s=args.stale_after)
        if args.once or args.read_only:
            _present(monitor, context, once=args.once)
            return

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
        if captured is not None:
            _print_capture(captured, context)


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


if __name__ == "__main__":
    main()
