"""Robot session lifecycle: claim, connect, verify, work, release.

One session owns one arm for one stretch of workflow. It exists so that every physical
call in the workflow happens under an explicit claim, against an arm that has been checked
against the configuration, with a teardown that runs even when the body raises.

Teardown holds position and disables rather than releasing torque abruptly, then
disconnects and releases the claim. If the process dies instead, both real lease backends
release on their own and the driver's motor-side CAN watchdog stops the motors.
"""

from __future__ import annotations

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


class RobotSession:
    """An open, exclusively-claimed, identity-checked arm.

    Use as a context manager. The motion controller and policy runner are built lazily and
    share the session's clock, heartbeat and cancellation, so a workflow that passes those
    once gets them everywhere.
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

    def close(self) -> None:
        """Leave the arm safe and release the claim, whatever happened before."""

        try:
            if self.port.lifecycle is ArmLifecycle.ENABLED:
                self.port.hold_current_position()
                self.port.disable()
        finally:
            try:
                self.bridge.disconnect()
            finally:
                self._open = False
                self.lease.release()

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
        self.close()

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
