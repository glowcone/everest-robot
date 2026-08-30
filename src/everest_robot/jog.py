"""Raise the arm a little, as a first powered motion check.

Deliberately the smallest possible physical command: one joint, a small bounded delta,
measured from wherever the arm actually is. It exists to prove the CAN link, the identity
check, the lease and the motion controller all work end to end before anything harder is
attempted.

The move is **relative to measured feedback**, not to an assumed home pose. The caller may
know the arm is parked at zero, but this command still reads the pose and moves from there,
so a wrong assumption about what "zero" means cannot turn into a wrong absolute target.
That matters on this arm: the joint frames do not share a zero (docs/lerobot-frame-
reconciliation.md), and only some joints have 0.0 rad inside their soft limits at all.

No preset is invented here. Named positions are captured, never authored
(docs/named-position-capture.md), so this goes through ``go_to_joint_target`` -- the same
limit checks, bounded interpolation and settling as any other move.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence

from everest_robot.robot.contracts import JointLimit, MotionResult
from everest_robot.robot.session import RobotSession

# "A little". Small enough to watch, big enough to see the arm actually move.
DEFAULT_DELTA_RAD = 0.10
# A hard ceiling on this command regardless of what is passed. Anything larger is a real
# move and belongs in a captured, approved named position rather than an ad-hoc jog.
MAX_DELTA_RAD = 0.35
# Increasing shoulder_lift raises the arm. Derived from the recorded episodes: all five
# start at shoulder_lift = -3.2 deg with the wrist camera facing the room and reach
# -70..-106 deg by the time it faces the table. The LeRobot/Everest mapping is a pure
# positive offset, so the sign carries into radians unchanged.
DEFAULT_JOINT = "shoulder_lift"


class JogRefused(RuntimeError):
    """The requested jog was rejected before anything was energized."""


def plan_jog(
    positions: Sequence[float],
    joint_names: Sequence[str],
    limits: Sequence[JointLimit],
    *,
    joint: str = DEFAULT_JOINT,
    delta_rad: float = DEFAULT_DELTA_RAD,
) -> tuple[float, ...]:
    """The target pose for raising ``joint`` by ``delta_rad`` from the measured pose.

    Every other joint holds its measured value, so the arm moves in exactly one axis.
    Raises rather than clamping: a jog that does not fit is a jog the operator should see
    refused, not one silently shortened into something they did not ask for.
    """

    if joint not in joint_names:
        raise JogRefused(f"unknown joint {joint!r}; this arm has {', '.join(joint_names)}")
    if not math.isfinite(delta_rad) or delta_rad == 0.0:
        raise JogRefused(f"delta must be a non-zero finite value, got {delta_rad!r}")
    if abs(delta_rad) > MAX_DELTA_RAD:
        raise JogRefused(
            f"delta {delta_rad:+.3f} rad exceeds this command's {MAX_DELTA_RAD} rad ceiling; "
            "capture an approved named position for a move this size"
        )

    index = list(joint_names).index(joint)
    current = positions[index]
    if not math.isfinite(current):
        raise JogRefused(f"{joint}: no joint feedback, so there is no pose to move relative to")

    target = current + delta_rad
    limit = limits[index]
    if not limit.contains(target):
        headroom = limit.upper_rad - current if delta_rad > 0 else current - limit.lower_rad
        raise JogRefused(
            f"{joint}: {current:+.4f} {delta_rad:+.4f} = {target:+.4f} rad is outside the "
            f"driver's soft limits [{limit.lower_rad:+.4f}, {limit.upper_rad:+.4f}]. "
            f"Only {headroom:+.4f} rad of travel remains in that direction."
        )

    return tuple(
        target if position_index == index else float(value)
        for position_index, value in enumerate(positions)
    )


def raise_arm(
    session: RobotSession,
    *,
    joint: str = DEFAULT_JOINT,
    delta_rad: float = DEFAULT_DELTA_RAD,
    speed_scale: float = 0.25,
    dry_run: bool = False,
) -> tuple[MotionResult, tuple[float, ...], tuple[float, ...]]:
    """Raise one joint by a small delta on an already-open session.

    Returns the motion result along with the measured start pose and the commanded target,
    because what the arm was doing beforehand is half of what makes the result readable.
    """

    state = session.snapshot()
    start = tuple(state.positions)
    target = plan_jog(
        start,
        session.port.joint_names,
        session.port.limits(),
        joint=joint,
        delta_rad=delta_rad,
    )
    result = session.motion.go_to_joint_target(
        f"jog:{joint}{delta_rad:+.3f}rad",
        target,
        speed_scale=speed_scale,
        dry_run=dry_run,
    )
    return result, start, target


def open_fake_session() -> RobotSession:
    """A session over :class:`FakeArm`, for exercising this command without an arm.

    Mirrors ``robot-monitor --fake``: the soft limits are invented here because real ones
    belong to the driver's hardware profile and there is no driver in this mode. Nothing
    this reports means anything about a physical arm.
    """

    from everest_robot.robot.contracts import JointLimit
    from everest_robot.robot.deployment import load_parameters
    from everest_robot.robot.fake_arm import FakeArm

    parameters = load_parameters()
    limits = tuple(JointLimit(name, -2.0, 2.0) for name in parameters.identity.joint_names)
    return RobotSession(FakeArm(parameters.identity, limits), parameters, cameras=None).open()


def _print_plan(joint_names: Sequence[str], start, target, limits) -> None:
    print(f"{'joint':<16}{'measured':>11}{'target':>11}{'delta':>10}   soft limits")
    for name, a, b, limit in zip(joint_names, start, target, limits, strict=True):
        mark = "  <-" if abs(b - a) > 1e-9 else ""
        print(
            f"{name:<16}{a:>+11.4f}{b:>+11.4f}{b - a:>+10.4f}   "
            f"[{limit.lower_rad:+.3f}, {limit.upper_rad:+.3f}]{mark}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Raise the arm slightly from wherever it is now, as a motion check."
    )
    parser.add_argument(
        "--joint", default=DEFAULT_JOINT, help=f"joint to move (default {DEFAULT_JOINT})"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--delta-rad", type=float, default=None,
        help=f"radians to raise by (default {DEFAULT_DELTA_RAD}, max {MAX_DELTA_RAD})",
    )
    group.add_argument("--delta-deg", type=float, default=None, help="the same, in degrees")
    parser.add_argument(
        "--down", action="store_true", help="lower instead of raise (negates the delta)"
    )
    parser.add_argument(
        "--speed-scale", type=float, default=0.25,
        help="fraction of the configured velocity/acceleration bounds (default 0.25)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="validate and report the planned motion without energizing the arm",
    )
    parser.add_argument(
        "--fake", action="store_true",
        help="run against the deterministic FakeArm: no CAN, no claim, no real numbers",
    )
    args = parser.parse_args()

    if args.delta_deg is not None:
        delta = math.radians(args.delta_deg)
    elif args.delta_rad is not None:
        delta = args.delta_rad
    else:
        delta = DEFAULT_DELTA_RAD
    if args.down:
        delta = -delta

    from everest_robot.robot.deployment import open_session

    if args.fake:
        print("FAKE ARM: nothing below this line describes a physical robot.\n")
    try:
        session = open_fake_session() if args.fake else open_session()
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1) from None

    try:
        try:
            result, start, target = raise_arm(
                session,
                joint=args.joint,
                delta_rad=delta,
                speed_scale=args.speed_scale,
                dry_run=args.dry_run,
            )
        except JogRefused as error:
            print(f"refused: {error}", file=sys.stderr)
            raise SystemExit(1) from None

        _print_plan(session.port.joint_names, start, target, session.port.limits())
        print()
        if result.failure_reason is not None:
            print(f"FAILED  {result.failure_reason}: {result.failure_detail}", file=sys.stderr)
            raise SystemExit(1)
        if result.dry_run:
            print(f"dry run OK: would take {result.planned_duration_s:.2f}s. Nothing moved.")
        elif result.already_at_target:
            print("already within tolerance of the target; the arm did not move.")
        else:
            print(
                f"raised in {result.elapsed_s:.2f}s "
                f"({result.commands_sent} commands, "
                f"max tracking error {result.max_tracking_error_rad:.4f} rad)"
            )
            print(f"final: {', '.join(f'{v:+.4f}' for v in result.final_joints)}")
    finally:
        session.close()


if __name__ == "__main__":
    main()
