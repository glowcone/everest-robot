"""``robot-goto``: drive the arm to an operator-approved named position.

The counterpart to ``robot-raise``. That command moves one joint a little, relative to
measured feedback, to prove the stack works; this one moves the whole arm to a pose someone
captured, approved and committed to ``config/maker_arm_v1.yaml``. Between them they are the
only ways to command motion outside the durable workflow, and both go through
:class:`~everest_robot.robot.motion.JointMotionController` -- the same limit checks,
bounded interpolation, settling and fault handling the workflow gets.

Nothing here invents a pose. A destination that is not in ``named_positions`` is refused
with the list of ones that are; presets are captured, never authored
(docs/named-position-capture.md). Where a ``named_transitions`` waypoint sequence ends at
the requested position, this command uses it and says so, because a direct joint-space
interpolation between two safe poses is not itself known to be safe. There is deliberately
no flag to bypass an approved transition.

Like ``robot-monitor`` this claims the robot lease for the whole run, so it cannot be used
while a worker holds the arm, and a large move asks for confirmation before it energizes
anything.

A move that is interrupted or fails partway leaves the arm somewhere nobody chose, and
releasing torque there drops it. So an interrupted or failed run returns to ``--park``
(``zero`` by default) before disengaging; a run that reaches its destination stays there,
because arriving is what the command is for.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from everest_robot.robot.contracts import JointLimit, MotionResult
from everest_robot.robot.parameters import ParameterError, RobotParameters
from everest_robot.robot.routing import Route, RouteRefused, resolve_route, transitions_ending_at
from everest_robot.robot.session import RobotSession

# Reduced speed by default. The capture guide's whole procedure is to drive a new path
# slowly first, and a named position may be a long way from wherever the arm is standing.
DEFAULT_SPEED_SCALE = 0.25
# Above this displacement on any one joint, the operator is asked before anything is
# energized. It is `robot-raise`'s ceiling for the same reason: past it, this stops being a
# nudge someone can watch and becomes a real move across the workspace.
CONFIRM_ABOVE_RAD = 0.35
CONFIRM_WORD = "GO"
# Where an interrupted or failed move leaves the arm. A clean move does not park: reaching
# the requested position is the whole point of this command, and driving away from it again
# would make `robot-goto` unable to do its job.
DEFAULT_PARK_POSITION = "zero"

# Route resolution moved to everest_robot.robot.routing so the session teardown could use it
# without the runtime importing a CLI. The names stay published here, under the one this
# command has always raised.
GotoRefused = RouteRefused


def widest_displacement(
    start: Sequence[float], route: Route, joint_names: Sequence[str]
) -> tuple[float, str]:
    """The largest single-joint displacement of any leg, and which joint it is on.

    Measured leg by leg rather than start-to-finish: a transition exists to take the arm
    somewhere the straight line does not go, so its individual legs are what the operator
    is actually being asked to authorize.
    """

    origin = tuple(float(value) for value in start)
    widest, joint = 0.0, joint_names[0] if joint_names else "?"
    for leg in route.legs:
        for name, target, current in zip(joint_names, leg.joints, origin, strict=True):
            displacement = abs(target - current)
            if displacement > widest:
                widest, joint = displacement, name
        origin = leg.joints
    return widest, joint


def go_to(
    session: RobotSession,
    route: Route,
    *,
    speed_scale: float = DEFAULT_SPEED_SCALE,
    dry_run: bool = False,
) -> MotionResult:
    """Run ``route`` on an already-open session, through the approved motion path."""

    motion = session.motion
    if route.transition is not None:
        return motion.follow_transition(
            route.transition, speed_scale=speed_scale, dry_run=dry_run
        )
    return motion.go_to_known_position(
        route.destination, speed_scale=speed_scale, dry_run=dry_run
    )


# ── presentation ───────────────────────────────────────────────────────────────────


def _print_positions(parameters: RobotParameters) -> None:
    """What this arm is allowed to be driven to, and who said so."""

    names = parameters.identity.joint_names
    print(f"robot {parameters.identity.robot_id} ({parameters.identity.calibration_id})")
    if not parameters.named_positions:
        print(
            "\nno named positions. They are captured from a measured, operator-approved "
            "arm state -- see docs/named-position-capture.md -- never hand-written."
        )
        return

    print("\nnamed positions:")
    for name, position in sorted(parameters.named_positions.items()):
        routes = transitions_ending_at(parameters, name)
        via = f"  via {', '.join(routes)}" if routes else ""
        print(f"  {name}{via}")
        print(f"    approved by {position.approved_by} on {position.captured_at}")
        print(
            "    "
            + ", ".join(
                f"{joint}={value:+.4f}"
                for joint, value in zip(names, position.joints, strict=True)
            )
        )
        if position.notes:
            print(f"    {position.notes}")

    print("\nnamed transitions:")
    if not parameters.named_transitions:
        print("  none; every approved position is reached by direct interpolation")
        return
    for name, transition in sorted(parameters.named_transitions.items()):
        print(f"  {name}: {' -> '.join(transition.waypoints)}")
        if transition.notes:
            print(f"    {transition.notes}")


def _print_plan(
    joint_names: Sequence[str],
    measured: Sequence[float],
    target: Sequence[float],
    limits: Sequence[JointLimit],
) -> None:
    """Measured pose against the destination pose, with the active soft limits beside it."""

    print(f"{'joint':<16}{'measured':>11}{'target':>11}{'delta':>10}   soft limits")
    for name, now, goal, limit in zip(joint_names, measured, target, limits, strict=True):
        mark = ""
        if not limit.contains(goal):
            mark = "  OUTSIDE LIMITS"
        elif abs(goal - now) > 1e-9:
            mark = "  <-"
        print(
            f"{name:<16}{now:>+11.4f}{goal:>+11.4f}{goal - now:>+10.4f}   "
            f"[{limit.lower_rad:+.3f}, {limit.upper_rad:+.3f}]{mark}"
        )


def _report(result: MotionResult) -> int:
    if result.failure_reason is not None:
        print(
            f"FAILED  {result.failure_reason}: {result.failure_detail}",
            file=sys.stderr,
        )
        return 1
    if result.dry_run:
        print(
            f"dry run OK: would take {result.planned_duration_s:.2f}s. Nothing moved, "
            "nothing was energized."
        )
        return 0
    if result.already_at_target:
        print("already within tolerance of the target; the arm did not move.")
        return 0
    print(
        f"reached {result.position_name} in {result.elapsed_s:.2f}s "
        f"({result.commands_sent} commands, "
        f"max tracking error {result.max_tracking_error_rad:.4f} rad)"
    )
    print(f"final: {', '.join(f'{value:+.4f}' for value in result.final_joints)}")
    return 0


def _confirm(widest: float, joint: str, route: Route) -> None:
    """Ask before a large move. Refuses rather than assuming consent off a terminal."""

    prompt = (
        f"{route.describe()} moves {joint} by {widest:.3f} rad "
        f"({widest * 57.29578:.1f} deg). Clear the area and type {CONFIRM_WORD} to move > "
    )
    if not sys.stdin.isatty():
        raise GotoRefused(
            f"{route.describe()} moves {joint} by {widest:.3f} rad, which needs "
            "confirmation, but stdin is not a terminal. Rerun interactively or pass --yes."
        )
    if input(prompt).strip() != CONFIRM_WORD:
        raise GotoRefused("cancelled before the arm was energized")


# ── entry point ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Drive the arm to an operator-approved named position. POWERED.",
    )
    parser.add_argument(
        "position",
        nargs="?",
        help="the named position to drive to; omit with --list",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the approved positions and transitions and exit; touches nothing",
    )
    parser.add_argument(
        "--transition",
        help="which approved waypoint sequence to take when more than one ends there",
    )
    parser.add_argument(
        "--speed-scale",
        type=float,
        default=DEFAULT_SPEED_SCALE,
        help=(
            "fraction of the position's velocity/acceleration bounds "
            f"(default {DEFAULT_SPEED_SCALE})"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="claim and validate against the live limits and measured pose, but move nothing",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip the confirmation prompt for a large move",
    )
    parser.add_argument(
        "--park",
        default=DEFAULT_PARK_POSITION,
        help=(
            "named position the arm returns to if the move is interrupted or fails, before "
            f"torque comes off (default {DEFAULT_PARK_POSITION!r}); a move that completes "
            "stays where it was sent"
        ),
    )
    parser.add_argument(
        "--no-park",
        action="store_const",
        const=None,
        dest="park",
        help="release torque where the arm stands instead of returning it to the rest pose",
    )
    args = parser.parse_args()

    if not 0.0 < args.speed_scale <= 1.0:
        parser.error(f"--speed-scale must be in (0, 1], got {args.speed_scale}")

    from everest_robot.robot.deployment import build_lease, build_port, load_parameters

    try:
        parameters = load_parameters()
    except (OSError, ParameterError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1) from None

    if args.list:
        _print_positions(parameters)
        return
    if not args.position:
        parser.error("a named position is required (or --list to see them)")

    # Route resolution needs only the parameters file, so an unknown or ambiguous
    # destination is refused without claiming the robot or touching the CAN bus.
    try:
        route = resolve_route(parameters, args.position, transition=args.transition)
    except GotoRefused as error:
        print(f"refused: {error}", file=sys.stderr)
        raise SystemExit(1) from None

    print(f"{route.describe()}  at speed scale {args.speed_scale}")

    try:
        # No cameras: a named-position move consumes no images. Building the port is part
        # of what can fail on a misconfigured host, so it shares the session's reporting.
        # An unknown --park destination is refused here too, before anything is claimed:
        # a session that cannot bring the arm home should not start.
        session = RobotSession(
            build_port(parameters),
            parameters,
            lease=build_lease(parameters),
            cameras=None,
            park_position=args.park,
            park_on_success=False,
        )
        session.open()
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1) from None

    failed = True
    try:
        measured = tuple(session.snapshot().positions)
        joint_names = session.port.joint_names
        _print_plan(joint_names, measured, route.target.joints, session.port.limits())
        print()

        widest, joint = widest_displacement(measured, route, joint_names)
        if not args.dry_run and not args.yes and widest > CONFIRM_ABOVE_RAD:
            try:
                _confirm(widest, joint, route)
            except GotoRefused as error:
                print(f"refused: {error}", file=sys.stderr)
                raise SystemExit(1) from None

        result = go_to(
            session, route, speed_scale=args.speed_scale, dry_run=args.dry_run
        )
        status = _report(result)
        # Only a move that actually arrived is allowed to leave the arm where it is. A
        # failed one stopped somewhere nobody chose, which is the case parking is for.
        failed = status != 0
        raise SystemExit(status)
    finally:
        session.close(failed=failed)


if __name__ == "__main__":
    main()
