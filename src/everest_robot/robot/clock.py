"""Injectable time source.

Motion, replay and policy loops are timing-critical, so they never call
:func:`time.monotonic` or :func:`time.sleep` directly. Tests drive them with
:class:`ManualClock`, which makes a ten-second timeout take microseconds and removes the
scheduler from the assertions.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """A monotonic clock that can also wait."""

    def monotonic(self) -> float:
        """Seconds from an arbitrary epoch; never decreases."""

    def sleep(self, seconds: float) -> None:
        """Advance to ``monotonic() + seconds``."""


class SystemClock:
    """The real clock. Used everywhere outside tests."""

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


class ManualClock:
    """A clock that only moves when someone moves it.

    ``sleep()`` advances instantly, so a control loop runs at its nominal rate in zero
    wall-clock time and its tick count is exactly reproducible.
    """

    def __init__(self, start: float = 0.0) -> None:
        self._now = float(start)

    def monotonic(self) -> float:
        return self._now

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            self._now += float(seconds)

    def advance(self, seconds: float) -> None:
        """Move time forward without a control loop asking for it."""
        self._now += float(seconds)
