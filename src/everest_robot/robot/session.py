"""Robot session lifecycle: claim, connect, verify, work, release.

One session owns one arm for one stretch of workflow. It exists so that every physical
call in the workflow happens under an explicit claim, against an arm that has been checked
against the configuration, with a teardown that runs even when the body raises.

Teardown drives the arm to an approved rest pose, disables it, disconnects and releases the
claim. If the process dies outright instead, both real lease backends release on their own;
what the motors do then is the driver's business, and for the MIT driver
(docs/adr/0002-mit-protocol-motor-operation.md) the answer is that the last commanded pose
stays in force, which is a hold rather than a collapse.

**Parking on the way out.** A session built with ``park_position`` will not release torque
from wherever the arm happens to be standing. Torque release at an arbitrary pose means the
arm falls, and a seven-joint arm falling onto the bench is the failure this teardown exists
to prevent -- so the arm is driven home first, under the ordinary bounded-motion path, and
only then disengaged. This runs on every exit that reaches ``close()``: a clean return, an
unhandled exception, a cancelled workflow, Ctrl-C. It is deliberately not a best-effort
nicety -- ``park_position`` is validated in the constructor so a session that cannot park is
refused before it claims anything.

Two things bound the risk that introduces. Parking is a real commanded move, so the first
Ctrl-C is deferred while it runs and a second one abandons it and cuts torque immediately;
and the physical e-stop remains the authority over both. Parking is skipped, loudly, when
the arm is in fault or has no feedback, because interpolating from a pose nobody can measure
would be worse than releasing where it stands.
"""

from __future__ import annotations

import contextlib
import signal
import sys
import threading
from collections.abc import Callable, Iterator
from types import TracebackType

from everest_robot.robot.cameras import CameraRuntime
from everest_robot.robot.clock import Clock, SystemClock
from everest_robot.robot.contracts import ArmLifecycle, CancelCheck, Heartbeat, JointState
from everest_robot.robot.lease import InMemoryLease, RobotLease
from everest_robot.robot.lerobot_bridge import JointFrame, RobotBridgeCore
from everest_robot.robot.motion import JointMotionController
from everest_robot.robot.parameters import RobotParameters
from everest_robot.robot.policy import PolicyRunner
from everest_robot.robot.ports import ArmPort
from everest_robot.robot.recording import SessionRecorder
from everest_robot.robot.routing import Route, resolve_route

# Parking is a move the operator did not explicitly ask for and may not be watching, so it
# runs at the same reduced speed the ordered recipes use for a first pass at a new path.
DEFAULT_PARK_SPEED_SCALE = 0.25


@contextlib.contextmanager
def defer_interrupt(notice: str) -> Iterator[Callable[[], bool]]:
    """Hold off one Ctrl-C so a safety-critical move can finish; let the second through.

    A parking move exists precisely because the operator wants the arm somewhere survivable
    before torque goes. Letting the Ctrl-C that *started* the teardown also abort it would
    defeat that, so the first SIGINT inside this block is absorbed and reported. The second
    raises :class:`KeyboardInterrupt` in the usual way, because an operator pressing it
    twice is telling us to stop now and the alternative to a released arm is an arm that
    ignores them -- and the physical e-stop is the authority over both.

    Yields a predicate reporting whether an interrupt was absorbed. Signal handlers can only
    be installed from the main thread, so off it this is a no-op that shields nothing rather
    than an error: a background thread is not where Ctrl-C is delivered anyway.
    """

    absorbed = False

    def pressed() -> bool:
        return absorbed

    if threading.current_thread() is not threading.main_thread():
        yield pressed
        return

    def handle(signum: int, frame: object) -> None:
        nonlocal absorbed
        if absorbed:
            # Restore first: whatever unwinds from here must not come back through us.
            signal.signal(signal.SIGINT, previous)
            raise KeyboardInterrupt
        absorbed = True
        print(notice, file=sys.stderr, flush=True)

    try:
        previous = signal.signal(signal.SIGINT, handle)
    except ValueError:  # pragma: no cover - main-thread check above already covers this
        yield pressed
        return
    try:
        yield pressed
    finally:
        with contextlib.suppress(ValueError):
            signal.signal(signal.SIGINT, previous)


class RobotSession:
    """An open, exclusively-claimed, identity-checked arm.

    Use as a context manager. The motion controller and policy runner are built lazily and
    share the session's clock, heartbeat and cancellation, so a workflow that passes those
    once gets them everywhere.

    ``park_position`` names the approved preset the arm is driven to before torque is
    released -- see this module's docstring. It is resolved here, in the constructor, so an
    unknown or ambiguous destination is refused before the robot is claimed rather than
    discovered in a teardown that has no way to recover. ``park_on_success`` exists for the
    commands whose whole purpose is to leave the arm somewhere specific: ``robot-goto`` must
    not undo its own move on a clean exit, but it must still come home when it is
    interrupted or fails partway.
    """

    def __init__(
        self,
        port: ArmPort,
        parameters: RobotParameters,
        *,
        lease: RobotLease | None = None,
        cameras: CameraRuntime | None = None,
        frame: JointFrame | None = None,
        clock: Clock | None = None,
        heartbeat: Heartbeat | None = None,
        cancel: CancelCheck | None = None,
        recorder: SessionRecorder | None = None,
        connect_timeout_s: float = 2.0,
        park_position: str | None = None,
        park_speed_scale: float = DEFAULT_PARK_SPEED_SCALE,
        park_on_success: bool = True,
    ) -> None:
        self.port = port
        self.parameters = parameters
        self.lease = lease or InMemoryLease(parameters.identity.robot_id)
        self.clock = clock or SystemClock()
        self.heartbeat = heartbeat
        self.cancel = cancel
        self.recorder = recorder
        self.connect_timeout_s = connect_timeout_s
        self.bridge = RobotBridgeCore(port, cameras=cameras, frame=frame)
        self.park_speed_scale = park_speed_scale
        self.park_on_success = park_on_success
        # Raises RouteRefused for an unknown or ambiguous destination -- deliberately here,
        # where it costs nothing, and not in close(), where there would be no way to act on
        # it. `parked` records what actually happened for the caller to report.
        self.park_route: Route | None = (
            None if park_position is None else resolve_route(parameters, park_position)
        )
        self.parked: str | None = None
        self._open = False
        self._motion: JointMotionController | None = None
        self._policy: PolicyRunner | None = None

    # ── lifecycle ──────────────────────────────────────────────────────────────────
    @property
    def is_open(self) -> bool:
        return self._open

    def open(self) -> RobotSession:
        """Claim the robot, connect, and refuse to proceed if it is the wrong arm."""

        self.lease.acquire()
        try:
            self.port.connect(timeout=self.connect_timeout_s)
        except Exception:
            self.lease.release()
            raise
        try:
            self.bridge.cameras.connect()
            # Identity before anything is enabled: presets and checkpoints are only
            # meaningful for the calibration they were captured under.
            self.parameters.verify_identity(self.port.identity)
        except Exception:
            # A refused session must leave nothing behind -- neither a claimed robot nor
            # a held CAN bus, which would block the next attempt just as effectively.
            self.bridge.disconnect()
            self.lease.release()
            raise
        self._open = True
        return self

    def close(self, *, failed: bool = False) -> None:
        """Park the arm, disengage it and release the claim, whatever happened before.

        Runs to the end on every path. Each stage is nested in the next one's ``finally``
        so that a park that fails still disables, a disable that fails still disconnects,
        and a disconnect that fails still releases the claim -- the ordering matters most
        exactly when something has already gone wrong.

        ``failed`` says whether the session body raised. It only decides whether a
        ``park_on_success=False`` session parks; a session that parks at all parks on the
        failure path regardless, because that is the path where nobody chose where the arm
        ended up.
        """

        try:
            if failed or self.park_on_success:
                try:
                    self._park()
                except BaseException as error:  # noqa: BLE001 - teardown must not raise
                    # _park handles its own failures; this is the backstop for the reads
                    # and reporting around them. Whatever went wrong, the stages below
                    # still have to run, and the operator still has to hear about it.
                    self._park_refused(f"{type(error).__name__}: {error}")
        finally:
            try:
                if self.port.lifecycle is ArmLifecycle.ENABLED:
                    self.port.disable()
            finally:
                try:
                    self.bridge.disconnect()
                finally:
                    self._open = False
                    self.lease.release()

    def _park(self) -> None:
        """Drive to the rest pose before torque comes off. Never raises.

        This is teardown: it is often already unwinding somebody else's failure, and losing
        that failure to a parking problem would be a poor trade. Every outcome is therefore
        reported on stderr and recorded in :attr:`parked` rather than raised.

        Refusals come before any command. A fault or missing feedback means the arm's pose
        is not known well enough to interpolate from, and guessing would turn a controlled
        release into a swing; the operator is told the arm is being released where it stands
        so they can support it or hit the e-stop.

        The move itself goes through :class:`~everest_robot.robot.motion.
        JointMotionController` -- the same limit checks, bounded interpolation, tracking
        watch and settling as any other move. It is not a special low-level path to the
        motors, and it inherits the honest caveat that a direct joint-space interpolation
        from an arbitrary pose is only known to be safe where an approved
        ``named_transitions`` sequence says so.
        """

        route = self.park_route
        if route is None or not self._open:
            return
        if self.port.lifecycle is ArmLifecycle.FAULT:
            self._park_refused("the arm is in fault, so its pose cannot be trusted")
            return
        if self.port.lifecycle is not ArmLifecycle.ENABLED:
            # Nothing to park. Torque was never applied by this session -- a dry run, a
            # read-only monitor, a destination refused before anything was energized -- so
            # there is no fall to prevent, and energizing a cold arm to drive it somewhere
            # nobody asked for would create the hazard instead of removing it.
            self.parked = f"skipped: the arm was never enabled ({self.port.lifecycle.value})"
            return
        if not self.port.read_state().all_finite:
            self._park_refused("some joints are not reporting a position")
            return

        notice = (
            f"\nparking at {route.destination!r} before torque comes off "
            f"({route.describe()}, speed scale {self.park_speed_scale:g}). "
            "Press Ctrl-C again to abandon the park and release the arm where it stands; "
            "use the e-stop to cut power."
        )
        print(notice, file=sys.stderr, flush=True)
        with defer_interrupt(
            "parking in progress -- press Ctrl-C again to release the arm here"
        ) as interrupted:
            try:
                if route.transition is not None:
                    result = self.motion.follow_transition(
                        route.transition, speed_scale=self.park_speed_scale
                    )
                else:
                    result = self.motion.go_to_known_position(
                        route.destination, speed_scale=self.park_speed_scale
                    )
            except BaseException as error:
                # The motion controller has already held the arm on its way out of this.
                self._park_refused(f"{type(error).__name__}: {error}")
                return
            if interrupted():
                print(
                    "the interrupt that arrived during parking has been honoured; "
                    "the arm is at its rest pose and torque is coming off now.",
                    file=sys.stderr,
                    flush=True,
                )

        if result.failure_reason is not None:
            self._park_refused(f"{result.failure_reason}: {result.failure_detail}")
            return
        self.parked = route.destination
        print(f"parked at {route.destination!r}.", file=sys.stderr, flush=True)

    def _park_refused(self, detail: str) -> None:
        """Say that the arm is about to be released where it stands, and why."""

        self.parked = f"not parked: {detail}"
        print(
            f"NOT PARKED ({detail}). Torque is being released at the arm's current pose -- "
            "support it or use the e-stop.",
            file=sys.stderr,
            flush=True,
        )

    def reconnect(self) -> None:
        """Recover the hardware connection while keeping the claim.

        For a driver-level failure inside a session that is still ours -- a bus reset, a
        transport error. Ownership is not re-negotiated, because we never lost it.
        """

        try:
            self.port.disconnect()
        finally:
            self.port.connect(timeout=self.connect_timeout_s)
            self.parameters.verify_identity(self.port.identity)

    def __enter__(self) -> RobotSession:
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close(failed=exc_type is not None)

    # ── the things a workflow stage actually uses ──────────────────────────────────
    @property
    def motion(self) -> JointMotionController:
        self._require_open()
        if self._motion is None:
            self._motion = JointMotionController(
                self.port,
                self.parameters,
                clock=self.clock,
                heartbeat=self.heartbeat,
                cancel=self.cancel,
            )
        return self._motion

    @property
    def policy(self) -> PolicyRunner:
        self._require_open()
        if self._policy is None:
            self._policy = PolicyRunner(
                self.bridge,
                self.parameters,
                clock=self.clock,
                heartbeat=self.heartbeat,
                cancel=self.cancel,
                recorder=self.recorder,
            )
        return self._policy

    def snapshot(self) -> JointState:
        """Current joint state, for deciding whether a physical effect already happened.

        A retried workflow stage should ask the hardware before repeating a move.
        """

        self._require_open()
        return self.port.read_state()

    def _require_open(self) -> None:
        if not self._open:
            raise RuntimeError("the robot session is not open; call open() first")
