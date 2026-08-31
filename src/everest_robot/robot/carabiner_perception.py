"""The ADR-0003 attachment gates, answered by the wrist-camera carabiner detector.

:mod:`everest_robot.carabiner_detect` finds a green Petzl Spirit carabiner in a wrist-camera
frame and returns where its aperture, spine and insertion point are. That answers three of the five
signals the FSM arbitrates on, and this module is the wiring: detection for ``SEARCH_RL``,
and visibility plus alignment for ``CLIP_RL``.

It does not answer the other two by guessing, because neither is visible to this detector:

* **Attachment.** Whether the carabiner ended up clipped onto the anchor is a fusion
  problem -- ADR-0003 names sensor, CV and VLM together -- and a detector that can see a
  carabiner cannot see whether it is *through* something. It is delegated to
  :class:`AttachmentVerifier`, whose only implementation today is
  :class:`UnverifiableAttachment`, which always answers "not verified". A run configured
  that way can never reach ``SUCCESS``, so it has to be asked for explicitly.
* **Grasp.** Whether the gripper is holding the carabiner is read from the gripper joint
  stalling short of closed, which needs a threshold measured on this arm with this
  carabiner. Left unset, the answer is a conservative "not grasped".

What it deliberately never reports is a confidence. The detector is a hysteresis threshold
with shape validation: it either accepts a component or raises. There is no score behind it,
and inventing one would put a number an operator could act on in a diagnostic trace with
nothing measured underneath -- the same reason ``search_cv_step`` reports ``None``.

Frames come from the session's own :class:`~everest_robot.robot.cameras.CameraRuntime`,
never from a second capture: the policy already holds the wrist camera, and on macOS and
V4L2 alike a second open on the same device either fails or starves the first.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from everest_robot.attachment_fsm import (
    AttachmentState,
    ClipRLStep,
    InitialObservation,
    SearchRLStep,
)
from everest_robot.robot.cameras import CameraRuntime
from everest_robot.robot.ports import ArmPort

#: LeRobot cameras hand back RGB by default; ``carabiner.detect`` works in Lab derived from
#: BGR. Getting this backwards swaps a* and b*, which is exactly the axis the teal score is
#: measured on, so it is configuration rather than an assumption.
COLOR_MODES = ("rgb", "bgr")


class PerceptionUnavailable(RuntimeError):
    """A gate cannot be served with the configuration given. Raised only from preflight."""


@runtime_checkable
class AttachmentVerifier(Protocol):
    """Whether the carabiner is attached to the anchor, right now.

    Separate from detection because it fails separately and will be built separately: this
    is the sensor/CV/VLM fusion ADR-0003 defers, and it is the one gate that decides
    ``SUCCESS``.
    """

    def preflight(self) -> None:
        """Raise unless this verifier can answer. Called before the robot is claimed."""

    def verify(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class UnverifiableAttachment:
    """Always answers "not attached", and says so loudly in preflight.

    This is not a stub that pretends: "not verified" is a truthful answer, and the FSM
    handles it correctly by continuing to clip until a budget stops it. What it is not is a
    configuration anyone should end up in by accident, because ``SUCCESS`` becomes
    unreachable and a perfectly successful attachment reports a budget failure. So it
    refuses unless the operator has explicitly accepted that.
    """

    acknowledged: bool = False

    def preflight(self) -> None:
        if not self.acknowledged:
            raise PerceptionUnavailable(
                "no attachment verifier is configured, so a successful attachment cannot be "
                "recognized and the run can only end on a budget. That is a legitimate way to "
                "exercise the loop on hardware, but it has to be asked for: pass "
                "--no-attachment-verification (or a real verifier) to say so deliberately"
            )

    def verify(self) -> bool:
        return False


@dataclass
class CarabinerVisionPerception:
    """The FSM's perception gates, over the wrist camera and the arm's own feedback.

    ``alignment_degraded`` is measured against a baseline rather than an assumed frame
    centre. ``SEARCH_CV`` hands over only once the arm has settled on the pixel map's
    pre-grasp target, so the first clip observation after entering ``CLIP_RL`` *is* the
    aligned pose; drift is measured from there. Nothing anywhere establishes that a settled
    pre-grasp puts the carabiner at the centre of the wrist view -- the map is taught on the
    fixed camera -- so assuming it would be inventing a fact about the rig.

    ``cameras`` and ``port`` are bound by :meth:`bind`, not passed in, because they do not
    exist yet when this is built. Everything decidable without hardware is checked in
    :meth:`preflight` before the robot is claimed; the session that owns the camera runtime
    and the arm port is opened after that, and binding it re-checks what only a live runtime
    can answer. ``configured_cameras`` is what makes the early half possible: the camera
    names come from deployment configuration, which is known long before a device is open.
    """

    verifier: AttachmentVerifier = field(default_factory=UnverifiableAttachment)
    camera_name: str = "wrist"
    color_mode: str = "rgb"
    #: Drift of the insertion point from the hand-over baseline that counts as degraded.
    alignment_tolerance_px: float = 60.0
    #: Where in the wrist frame the detector may look, in full-frame pixels, or the whole
    #: frame. Set it whenever the bench is not the only thing in view -- see
    #: ``carabiner_detect.in_roi`` for why background clutter cannot be rejected otherwise.
    roi_xywh: tuple[int, int, int, int] | None = None
    #: Gripper position, in this arm's joint radians, below which it is holding something.
    #: Measured by closing the gripper on the carabiner and reading the monitor's feedback.
    grasp_gripper_below_rad: float | None = None
    gripper_joint: str = "gripper"
    #: Camera names this deployment configures, for the pre-claim half of preflight.
    configured_cameras: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.color_mode not in COLOR_MODES:
            raise ValueError(f"color_mode must be one of {', '.join(COLOR_MODES)}")
        if not math.isfinite(self.alignment_tolerance_px) or self.alignment_tolerance_px <= 0.0:
            raise ValueError("alignment_tolerance_px must be finite and positive")
        if self.grasp_gripper_below_rad is not None and not math.isfinite(
            self.grasp_gripper_below_rad
        ):
            raise ValueError("grasp_gripper_below_rad must be finite")
        self.cameras: CameraRuntime | None = None
        self.port: ArmPort | None = None
        self._baseline: tuple[float, float] | None = None

    # ── the boundary ───────────────────────────────────────────────────────────────
    def preflight(self) -> None:
        """Raise unless every gate below can be served. Runs before the robot is claimed."""

        try:
            import everest_robot.carabiner_detect  # noqa: F401
        except ImportError as error:
            # In practice this is OpenCV or NumPy missing, not the module itself.
            raise PerceptionUnavailable(
                f"the carabiner detector could not be imported ({error})"
            ) from None

        if self.camera_name not in self.configured_cameras:
            configured = ", ".join(self.configured_cameras) or "none"
            raise PerceptionUnavailable(
                f"perception needs camera {self.camera_name!r}, which is not configured "
                f"(EVEREST_CAMERAS has: {configured})"
            )
        self.verifier.preflight()

    def bind(self, cameras: CameraRuntime, port: ArmPort) -> None:
        """Adopt the open session's camera runtime and arm port.

        The runtime is shared rather than rebuilt: the policy already holds the wrist
        camera, and a second open on the same device either fails or starves the first.
        """

        if self.camera_name not in cameras.names:
            available = ", ".join(cameras.names) or "none"
            raise PerceptionUnavailable(
                f"perception needs camera {self.camera_name!r}, which the open session does "
                f"not have (it has: {available})"
            )
        if self.grasp_gripper_below_rad is not None and self.gripper_joint not in port.joint_names:
            raise PerceptionUnavailable(
                f"grasp detection needs joint {self.gripper_joint!r}, which this arm does not "
                f"have (it has: {', '.join(port.joint_names)})"
            )
        self.cameras = cameras
        self.port = port

    def enter_state(self, state: AttachmentState, previous: AttachmentState | None) -> None:
        """Drop the alignment baseline whenever ``CLIP_RL`` is entered afresh."""

        del previous
        if state is AttachmentState.CLIP_RL:
            self._baseline = None

    def initial_observation(self) -> InitialObservation:
        """The motion-free look ``INITIAL`` decides from."""

        return InitialObservation(
            already_attached=self.verifier.verify(),
            carabiner_detected=self._detect() is not None,
            confidence=None,
        )

    def carabiner_detection(self) -> SearchRLStep:
        return SearchRLStep(carabiner_detected=self._detect() is not None, confidence=None)

    def clip_observations(self) -> ClipRLStep:
        target = self._detect()
        visible = target is not None

        degraded = False
        if target is not None:
            insert = (float(target.insert[0]), float(target.insert[1]))
            if self._baseline is None:
                # The hand-over pose. `SEARCH_CV` settled on the map's pre-grasp target to
                # get here, so this is what aligned looks like on this attempt.
                self._baseline = insert
            else:
                degraded = math.dist(self._baseline, insert) > self.alignment_tolerance_px

        return ClipRLStep(
            attachment_verified=self.verifier.verify(),
            returned_to_neutral=False,
            carabiner_visible=visible,
            alignment_degraded=degraded,
            carabiner_grasped=self._grasped(),
            confidence=None,
        )

    # ── helpers ────────────────────────────────────────────────────────────────────
    def _detect(self) -> Any | None:
        """One frame through the detector. A miss is a normal outcome, not an error."""

        import numpy as np

        from everest_robot.carabiner_detect import NotFound, detect

        if self.cameras is None:
            raise PerceptionUnavailable("perception was not bound to an open robot session")
        frame = self.cameras.observation()[self.camera_name]
        if self.color_mode == "rgb":
            # Contiguous, not a reversed view: OpenCV rejects negative strides.
            frame = np.ascontiguousarray(frame[:, :, ::-1])
        try:
            return detect(frame, self.roi_xywh)
        except (NotFound, ValueError):
            # ValueError covers the degenerate geometry the detector's own validation does
            # not reach (a hull with no interior, an unsolvable spine fit). Both mean the
            # same thing to the FSM: nothing carabiner-shaped in this frame.
            return None

    def _grasped(self) -> bool:
        """Whether the gripper is closed onto something, if that has been measured."""

        if self.grasp_gripper_below_rad is None or self.port is None:
            return False
        state = self.port.read_state()
        index = self.port.joint_names.index(self.gripper_joint)
        position = state.positions[index]
        if not math.isfinite(position):
            return False
        return position < self.grasp_gripper_below_rad
