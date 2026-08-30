"""Exclusive ownership of one physical robot.

A durable workflow retries, and a retry can start while the previous attempt's worker is
still shutting down. Two controllers on one CAN bus is not a degraded mode -- it is two
sets of targets and two watchdogs fighting over the same motors -- so ownership is taken
explicitly and is enforced outside this process.

Both real backends release automatically when the holder dies, which is the property that
matters: a killed worker must not leave the arm permanently claimed.

* :class:`PostgresAdvisoryLease` -- a session-scoped advisory lock on the Absurd database.
  Correct across hosts; released when the connection drops.
* :class:`FileLease` -- ``flock`` on a lock file. Correct for one host; released when the
  file descriptor closes, including on process death.
"""

from __future__ import annotations

import fcntl
import os
import zlib
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# Advisory lock keys are namespaced so a robot lease can never collide with another
# component's advisory lock in the same database.
LOCK_NAMESPACE = 0x45564552  # "EVER"

DEFAULT_LEASE_DIR = Path(os.getenv("EVEREST_LEASE_DIR", "/tmp"))  # noqa: S108


class RobotBusy(RuntimeError):
    """Another holder owns this robot. Never retry blindly: find out who."""


@runtime_checkable
class RobotLease(Protocol):
    """An exclusive claim on one robot, identified by ``robot_id``."""

    @property
    def robot_id(self) -> str: ...

    @property
    def held(self) -> bool: ...

    def acquire(self) -> None:
        """Claim the robot, or raise :class:`RobotBusy`."""

    def release(self) -> None:
        """Release the claim. Idempotent."""


class InMemoryLease:
    """Process-local lease. For tests and single-process runs only.

    Useless against a second process, which is exactly the case the real backends exist
    for, so it must never be the default in a deployment.
    """

    _holders: dict[str, InMemoryLease] = {}

    def __init__(self, robot_id: str) -> None:
        self._robot_id = robot_id
        self._held = False

    @property
    def robot_id(self) -> str:
        return self._robot_id

    @property
    def held(self) -> bool:
        return self._held

    def acquire(self) -> None:
        holder = InMemoryLease._holders.get(self._robot_id)
        if holder is not None and holder is not self:
            raise RobotBusy(f"robot {self._robot_id!r} is already claimed in this process")
        InMemoryLease._holders[self._robot_id] = self
        self._held = True

    def release(self) -> None:
        if InMemoryLease._holders.get(self._robot_id) is self:
            del InMemoryLease._holders[self._robot_id]
        self._held = False

    def __enter__(self) -> InMemoryLease:
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


class FileLease:
    """``flock``-based lease. Correct for a single host.

    The lock lives on the open file descriptor, so the kernel releases it when the process
    exits for any reason, including a kill. The file itself is left behind on purpose: it
    is a name, not state, and unlinking it would race another claimant.
    """

    def __init__(self, robot_id: str, directory: Path | str | None = None) -> None:
        self._robot_id = robot_id
        base = Path(directory) if directory is not None else DEFAULT_LEASE_DIR
        base.mkdir(parents=True, exist_ok=True)
        self.path = base / f"everest-robot-{robot_id}.lock"
        self._fd: int | None = None

    @property
    def robot_id(self) -> str:
        return self._robot_id

    @property
    def held(self) -> bool:
        return self._fd is not None

    def acquire(self) -> None:
        if self._fd is not None:
            return
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            os.close(fd)
            raise RobotBusy(
                f"robot {self._robot_id!r} is claimed by another process (see {self.path})"
            ) from error
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        self._fd = fd

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> FileLease:
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


class PostgresAdvisoryLease:
    """Session-scoped advisory lock in the Absurd database. Correct across hosts.

    Holds its own connection: the lock's lifetime is the connection's, so a worker that
    dies without unwinding releases it as soon as the server notices the connection is
    gone. Sharing a pooled connection would tie the lock to whatever else used it.
    """

    def __init__(self, robot_id: str, database_url: str | None = None) -> None:
        self._robot_id = robot_id
        self.database_url = database_url or os.getenv(
            "ABSURD_DATABASE_URL", "postgresql://robot:robot@localhost:5432/robot"
        )
        self._connection: Any = None

    @property
    def robot_id(self) -> str:
        return self._robot_id

    @property
    def held(self) -> bool:
        return self._connection is not None

    @property
    def lock_key(self) -> int:
        """A stable 32-bit key derived from the robot id, inside our namespace.

        Wrapped into the signed range because the two-argument advisory lock functions
        take `integer`; an unsigned crc32 above 2**31 would be sent as a bigint and match
        no overload.
        """

        unsigned = zlib.crc32(self._robot_id.encode())
        return unsigned - 2**32 if unsigned >= 2**31 else unsigned

    def acquire(self) -> None:
        if self._connection is not None:
            return
        import psycopg

        connection = psycopg.connect(self.database_url, autocommit=True)
        try:
            row = connection.execute(
                "SELECT pg_try_advisory_lock(%s::int, %s::int)",
                (LOCK_NAMESPACE, self.lock_key),
            ).fetchone()
        except Exception:
            connection.close()
            raise
        if not row or not row[0]:
            connection.close()
            raise RobotBusy(f"robot {self._robot_id!r} is claimed by another worker")
        self._connection = connection

    def release(self) -> None:
        connection, self._connection = self._connection, None
        if connection is None:
            return
        try:
            connection.execute(
                "SELECT pg_advisory_unlock(%s::int, %s::int)",
                (LOCK_NAMESPACE, self.lock_key),
            )
        finally:
            connection.close()

    def __enter__(self) -> PostgresAdvisoryLease:
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()
