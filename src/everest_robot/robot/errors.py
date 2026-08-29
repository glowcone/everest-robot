"""Typed replay failures.

Classified so a workflow and an operator can tell remediation apart: a bad revision is a
configuration problem, an out-of-range action is a compatibility problem, and a motor
fault is a hardware problem. They are deliberately distinct exception types rather than one
error with a string.

Error messages carry identifiers and frame indices. They must never carry credentials --
Hugging Face tokens come from the environment and are never echoed into a message, a
result, or a log line.
"""

from __future__ import annotations


class ReplayError(RuntimeError):
    """Base class for every replay failure."""


class DatasetResolutionError(ReplayError):
    """Download, authentication, revision, or missing-file failure."""


class DatasetCompatibilityError(ReplayError):
    """Unsupported schema, features, robot type, FPS, or episode."""


class ReplayLimitError(ReplayError):
    """An action or state violates the active limits under the configured policy."""


class CalibrationMismatchError(ReplayError):
    """This robot or calibration is not approved for this dataset revision."""


class InitialAlignmentError(ReplayError):
    """The arm could not safely reach the recorded initial state."""


class ReplayCancelled(ReplayError):
    """An intentional cancellation. Not a fault: the arm is held, not faulted."""


class ReplayTimingError(ReplayError):
    """The control loop repeatedly missed its deadline."""


class RobotFaultError(ReplayError):
    """Stale feedback, CAN failure, motor fault, or a rejected command."""
