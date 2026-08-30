"""Deciding how the arm gets to a named position, from configuration alone.

Route resolution is pure: it reads :class:`~everest_robot.robot.parameters.RobotParameters`
and returns a plan. Nothing here claims the arm, touches the bus, or moves anything, which
is what lets ``robot-goto`` refuse an unknown destination before it costs a claim and lets
:class:`~everest_robot.robot.session.RobotSession` validate its park destination in its
constructor, before ``open()``.

It lives under ``robot/`` rather than in ``goto.py`` because two callers need it and one of
them is the session teardown. A CLI importing the runtime is fine; the runtime importing a
CLI is not.
"""

from __future__ import annotations

from dataclasses import dataclass

from everest_robot.robot.parameters import NamedPosition, ParameterError, RobotParameters


class RouteRefused(RuntimeError):
    """The destination was rejected before anything was claimed or energized."""


@dataclass(frozen=True, slots=True)
class Route:
    """How the arm is to reach one destination: directly, or along approved waypoints."""

    destination: str
    legs: tuple[NamedPosition, ...]
    transition: str | None = None

    @property
    def waypoints(self) -> tuple[str, ...]:
        return tuple(leg.name for leg in self.legs)

    @property
    def target(self) -> NamedPosition:
        return self.legs[-1]

    def describe(self) -> str:
        if self.transition is None:
            return f"direct to {self.destination}"
        return f"transition {self.transition}: {' -> '.join(self.waypoints)}"


def transitions_ending_at(parameters: RobotParameters, destination: str) -> tuple[str, ...]:
    """Approved waypoint sequences that finish at ``destination``, in a stable order."""

    return tuple(
        sorted(
            name
            for name, transition in parameters.named_transitions.items()
            if transition.waypoints[-1] == destination
        )
    )


def resolve_route(
    parameters: RobotParameters,
    destination: str,
    *,
    transition: str | None = None,
) -> Route:
    """Decide how to reach ``destination``, from configuration alone.

    An approved transition wins over a direct move whenever one ends at the destination:
    the transition exists precisely because the direct interpolation was not shown to be
    collision-free. Two of them ending at the same pose is an operator choice this command
    will not make silently, so it refuses and names them.
    """

    try:
        target = parameters.position(destination)
    except ParameterError as error:
        raise RouteRefused(str(error)) from None

    if transition is not None:
        try:
            chosen = parameters.transition(transition)
        except ParameterError as error:
            raise RouteRefused(str(error)) from None
        if chosen.waypoints[-1] != destination:
            raise RouteRefused(
                f"transition {transition!r} ends at {chosen.waypoints[-1]!r}, not at "
                f"{destination!r}; it is not a way to get there"
            )
        return Route(
            destination=destination,
            legs=tuple(parameters.position(name) for name in chosen.waypoints),
            transition=transition,
        )

    candidates = transitions_ending_at(parameters, destination)
    if len(candidates) > 1:
        raise RouteRefused(
            f"{len(candidates)} approved transitions end at {destination!r} "
            f"({', '.join(candidates)}); choose one with --transition"
        )
    if candidates:
        chosen = parameters.transition(candidates[0])
        return Route(
            destination=destination,
            legs=tuple(parameters.position(name) for name in chosen.waypoints),
            transition=candidates[0],
        )
    return Route(destination=destination, legs=(target,))
