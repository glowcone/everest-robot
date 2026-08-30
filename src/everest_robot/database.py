"""Local Postgres lifecycle, over Docker Compose or an already-installed server.

Absurd needs one Postgres. How it gets there is a property of the workstation, not of the
robot, so this module keeps two interchangeable backends behind one command:

* ``docker``  -- ``compose.yaml``'s ``postgres`` service. Self-contained and disposable.
* ``native``  -- a PostgreSQL the host already has (Homebrew, a distribution package, a
  server on another machine). Shared with whatever else uses it, so this backend starts
  and provisions but never stops it.

The backend is auto-detected and can be forced with ``EVEREST_DB_BACKEND``. Both backends
speak to the same ``ABSURD_DATABASE_URL``, so nothing above this module can tell them
apart. ``psql`` still runs inside the container on the Docker backend, which is why the
host is not assumed to have a PostgreSQL client there.
"""

import argparse
import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from urllib.parse import unquote, urlsplit

import psycopg
from psycopg import sql

DEFAULT_DATABASE_URL = "postgresql://robot:robot@localhost:5432/robot"

BACKENDS = ("docker", "native")
BACKEND_ENV_VAR = "EVEREST_DB_BACKEND"

# The compose service and the container-local client the Docker backend drives.
COMPOSE_SERVICE = "postgres"
# Both spellings of Compose; the plugin is preferred, the standalone binary still exists.
COMPOSE_CANDIDATES = (("docker", "compose"), ("docker-compose",))

# How long a freshly started server gets to accept connections.
STARTUP_TIMEOUT_SECONDS = 30.0


class DatabaseError(RuntimeError):
    """A backend could not be selected or could not do what was asked of it."""


@dataclass(frozen=True)
class DatabaseTarget:
    """The pieces of ``ABSURD_DATABASE_URL`` the lifecycle commands need."""

    url: str
    host: str
    port: int
    user: str
    password: str | None
    dbname: str

    @property
    def is_local(self) -> bool:
        """Whether this process can plausibly start the server itself."""

        return self.host in {"localhost", "127.0.0.1", "::1", ""}


def parse_database_url(url: str) -> DatabaseTarget:
    """Split a libpq URL into the fields the backends provision and connect with."""

    parts = urlsplit(url)
    if parts.scheme not in {"postgresql", "postgres"}:
        raise DatabaseError(f"{url!r} is not a postgresql:// URL")
    dbname = unquote(parts.path.lstrip("/"))
    if not dbname:
        raise DatabaseError(f"{url!r} names no database")
    return DatabaseTarget(
        url=url,
        host=unquote(parts.hostname or "localhost"),
        port=parts.port or 5432,
        user=unquote(parts.username or "postgres"),
        password=unquote(parts.password) if parts.password else None,
        dbname=dbname,
    )


def target_from_environment(environ: dict[str, str] | None = None) -> DatabaseTarget:
    environ = os.environ if environ is None else environ
    return parse_database_url(environ.get("ABSURD_DATABASE_URL", DEFAULT_DATABASE_URL))


# ── backend selection ──────────────────────────────────────────────────────────────


def select_backend(
    requested: str | None,
    *,
    compose_available: bool,
    daemon_available: bool,
    native_available: bool,
) -> str:
    """Choose a backend, or explain what to install. Pure, so it is unit-tested directly.

    An explicit request always wins. Otherwise Docker is preferred when it can actually
    run something -- an installed Compose with an unreachable daemon starts nothing -- and
    a host PostgreSQL client is the fallback.
    """

    if requested:
        choice = requested.strip().lower()
        if choice not in BACKENDS:
            raise DatabaseError(
                f"{BACKEND_ENV_VAR}={requested!r} is not one of {', '.join(BACKENDS)}"
            )
        return choice
    if compose_available and daemon_available:
        return "docker"
    if native_available:
        return "native"
    if compose_available:
        raise DatabaseError(
            "Docker Compose is installed but the Docker daemon is not reachable. Start "
            "Docker, or install PostgreSQL (`brew install postgresql@17`) and rerun; set "
            f"{BACKEND_ENV_VAR}=native to skip this probe."
        )
    raise DatabaseError(
        "no local Postgres backend found. Either install Docker Desktop (or the Compose "
        "plugin), or install PostgreSQL on the host -- `brew install postgresql@17` on "
        "macOS -- and rerun."
    )


def _run(argv: list[str], *, timeout: float = 30.0) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(argv, capture_output=True, timeout=timeout)


def _succeeds(argv: list[str], *, timeout: float = 30.0) -> bool:
    if not shutil.which(argv[0]):
        return False
    try:
        return _run(argv, timeout=timeout).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def compose_command() -> list[str] | None:
    """The working ``docker compose`` invocation on this host, if there is one."""

    for candidate in COMPOSE_CANDIDATES:
        if _succeeds([*candidate, "version"], timeout=15.0):
            return list(candidate)
    return None


def docker_daemon_available() -> bool:
    """Whether the Docker daemon answers. Compose without a daemon can start nothing."""

    return _succeeds(["docker", "info"], timeout=20.0)


def native_available() -> bool:
    """Whether the host has a PostgreSQL client to drive the native backend with."""

    return shutil.which("psql") is not None


def resolve_backend(
    target: DatabaseTarget, environ: dict[str, str] | None = None
) -> "DockerBackend | NativeBackend":
    """Detect and construct the backend for this host."""

    environ = os.environ if environ is None else environ
    requested = environ.get(BACKEND_ENV_VAR)
    compose = compose_command() if requested in {None, "", "docker"} else None
    choice = select_backend(
        requested,
        compose_available=compose is not None,
        daemon_available=docker_daemon_available() if compose is not None else False,
        native_available=native_available(),
    )
    if choice == "docker":
        if compose is None:
            raise DatabaseError(
                f"{BACKEND_ENV_VAR}=docker, but neither `docker compose` nor "
                "`docker-compose` runs on this host."
            )
        return DockerBackend(compose, target)
    return NativeBackend(target)


# ── shared helpers ─────────────────────────────────────────────────────────────────


def _listening(target: DatabaseTarget, timeout: float = 1.0) -> bool:
    """A pure reachability check: no authentication, so it cannot be confused by roles."""

    try:
        with socket.create_connection((target.host or "localhost", target.port), timeout):
            return True
    except OSError:
        return False


def _wait_until_listening(target: DatabaseTarget, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _listening(target):
            return
        time.sleep(0.5)
    raise DatabaseError(
        f"Postgres did not start accepting connections on {target.host}:{target.port} "
        f"within {timeout:.0f}s"
    )


def _exec(argv: list[str]) -> int:
    """Run a child in the foreground, inheriting stdio so ``psql`` stays interactive."""

    try:
        return subprocess.run(argv).returncode
    except FileNotFoundError as error:
        raise DatabaseError(f"{argv[0]} is not on PATH") from error


# ── the Docker backend ─────────────────────────────────────────────────────────────


class DockerBackend:
    """``compose.yaml``'s ``postgres`` service, with ``psql`` run inside the container."""

    name = "docker"

    def __init__(self, compose: list[str], target: DatabaseTarget) -> None:
        self._compose = compose
        self._target = target

    def describe(self) -> str:
        return f"docker ({' '.join(self._compose)}, service {COMPOSE_SERVICE})"

    def _compose_run(self, *args: str) -> None:
        code = _exec([*self._compose, *args])
        if code != 0:
            raise DatabaseError(f"`{' '.join([*self._compose, *args])}` failed ({code})")

    def up(self) -> None:
        self._compose_run("up", "-d", "--wait", COMPOSE_SERVICE)

    def down(self) -> None:
        self._compose_run("down")

    def reset(self) -> None:
        self._compose_run("down", "-v")
        self.up()

    def psql_argv(self, extra: list[str], *, interactive: bool) -> list[str]:
        flags = [] if interactive else ["-T"]
        return [
            *self._compose,
            "exec",
            *flags,
            COMPOSE_SERVICE,
            "psql",
            "-U",
            self._target.user,
            "-d",
            self._target.dbname,
            *extra,
        ]


# ── the native backend ─────────────────────────────────────────────────────────────


class NativeBackend:
    """A PostgreSQL the host already runs.

    ``up`` starts it if it is local and idle, then makes sure the role and database named
    by ``ABSURD_DATABASE_URL`` exist. ``down`` deliberately does nothing: this server is
    shared, and stopping it would take out whatever else on the workstation uses it.
    """

    name = "native"

    def __init__(self, target: DatabaseTarget) -> None:
        self._target = target

    def describe(self) -> str:
        return f"native (server at {self._target.host}:{self._target.port})"

    def up(self) -> None:
        if not _listening(self._target):
            if not self._target.is_local:
                raise DatabaseError(
                    f"nothing is listening at {self._target.host}:{self._target.port}, and "
                    "it is not this host, so these recipes cannot start it."
                )
            self._start_local_server()
            _wait_until_listening(self._target, STARTUP_TIMEOUT_SECONDS)
        self._provision()

    def down(self) -> None:
        print(
            "The native backend uses a PostgreSQL shared with the rest of this host, so "
            "nothing was stopped. Stop it yourself if you mean to -- "
            "`brew services stop postgresql@17` -- or use the Docker backend for a "
            "disposable one."
        )

    def reset(self) -> None:
        self.up()
        target = self._target
        with self._admin_connection() as connection:
            connection.execute(
                sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(
                    sql.Identifier(target.dbname)
                )
            )
        self._provision()

    def psql_argv(self, extra: list[str], *, interactive: bool) -> list[str]:
        del interactive  # psql on the host is interactive exactly when its stdin is
        if shutil.which("psql") is None:
            raise DatabaseError(
                "the native backend needs a psql client on PATH. Homebrew keeps the "
                "versioned PostgreSQL formulae unlinked, so add its bin directory -- "
                "`brew --prefix postgresql@17`/bin -- to PATH, or `brew install libpq`."
            )
        return ["psql", self._target.url, *extra]

    # -- starting and provisioning ---------------------------------------------------

    def _start_local_server(self) -> None:
        formula = _homebrew_postgres_formula()
        if formula is None:
            raise DatabaseError(
                f"no Postgres is listening on {self._target.host}:{self._target.port} and "
                "no Homebrew PostgreSQL formula was found to start. Start your server, or "
                "install one: `brew install postgresql@17`."
            )
        print(f"Starting {formula} via `brew services`...")
        result = _run(["brew", "services", "start", formula], timeout=120.0)
        if result.returncode != 0:
            detail = result.stderr.decode(errors="replace").strip()
            raise DatabaseError(f"`brew services start {formula}` failed: {detail}")

    def _admin_connection(self) -> psycopg.Connection:
        """Connect to the maintenance database with enough rights to create things.

        The host's own account is tried first -- that is the superuser Homebrew's initdb
        creates -- and the configured role second, which covers a server someone already
        provisioned by hand.
        """

        target = self._target
        attempts: list[dict[str, object]] = [
            {"host": target.host, "port": target.port, "dbname": "postgres"},
            {
                "host": target.host,
                "port": target.port,
                "dbname": "postgres",
                "user": target.user,
                "password": target.password,
            },
        ]
        errors: list[str] = []
        for attempt in attempts:
            try:
                return psycopg.connect(autocommit=True, **attempt)  # type: ignore[arg-type]
            except psycopg.OperationalError as error:
                errors.append(str(error).strip())
        raise DatabaseError(
            "could not open an administrative connection to "
            f"{target.host}:{target.port}/postgres. Tried this host's own account and "
            f"{target.user!r}:\n  " + "\n  ".join(errors)
        )

    def _provision(self) -> None:
        """Create the role and database named by the URL. Idempotent by design."""

        target = self._target
        with self._admin_connection() as connection:
            role_exists = connection.execute(
                "SELECT 1 FROM pg_roles WHERE rolname = %s", (target.user,)
            ).fetchone()
            if not role_exists:
                statement = sql.SQL("CREATE ROLE {} LOGIN").format(sql.Identifier(target.user))
                if target.password:
                    statement = sql.SQL("{} PASSWORD {}").format(
                        statement, sql.Literal(target.password)
                    )
                connection.execute(statement)
                print(f"Created role {target.user!r}")
            database_exists = connection.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s", (target.dbname,)
            ).fetchone()
            if not database_exists:
                connection.execute(
                    sql.SQL("CREATE DATABASE {} OWNER {}").format(
                        sql.Identifier(target.dbname), sql.Identifier(target.user)
                    )
                )
                print(f"Created database {target.dbname!r} owned by {target.user!r}")


def _homebrew_postgres_formula() -> str | None:
    """The newest installed ``postgresql*`` formula, if Homebrew is present."""

    if not shutil.which("brew"):
        return None
    try:
        result = _run(["brew", "list", "--formula"], timeout=60.0)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    formulae = [
        name
        for name in result.stdout.decode(errors="replace").split()
        if name.startswith("postgresql")
    ]
    return max(formulae) if formulae else None


# ── commands ───────────────────────────────────────────────────────────────────────


def init() -> None:
    """Install the locked Absurd schema and idempotently create the robot queue."""

    # absurdctl is a dev dependency; the lifecycle commands below must not need it.
    from absurdctl import BUNDLED_SCHEMA_SQL

    database_url = os.getenv("ABSURD_DATABASE_URL", DEFAULT_DATABASE_URL)
    queue_name = os.getenv("ROBOT_QUEUE", "robot")
    with psycopg.connect(database_url) as connection:
        connection.execute(BUNDLED_SCHEMA_SQL)
        connection.execute("SELECT absurd.create_queue(%s)", (queue_name,))
    print(f"Absurd schema initialized; queue {queue_name!r} is ready")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="robot-db",
        description=(
            "Manage the local Postgres behind Absurd, using Docker Compose when it is "
            "available and an already-installed PostgreSQL otherwise. Override the "
            f"choice with {BACKEND_ENV_VAR}=docker|native."
        ),
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("backend", help="print which backend these commands will use")
    subcommands.add_parser("up", help="start Postgres and wait until it accepts connections")
    subcommands.add_parser("down", help="stop the Docker containers; a shared server is left up")
    subcommands.add_parser("reset", help="delete all database state and start over")
    subcommands.add_parser("init", help="install the Absurd schema and create the queue")
    shell = subcommands.add_parser("psql", help="open an interactive psql shell")
    shell.add_argument("args", nargs=argparse.REMAINDER, help="extra arguments for psql")
    statement = subcommands.add_parser("sql", help="run one statement and print the result")
    statement.add_argument("statement", help="the SQL to run")
    return parser


def main(argv: list[str] | None = None) -> int:
    """The ``robot-db`` entry point behind the ``database`` group of just recipes."""

    arguments = _build_parser().parse_args(argv)
    try:
        target = target_from_environment()
        if arguments.command == "init":
            init()
            return 0
        backend = resolve_backend(target)
        match arguments.command:
            case "backend":
                print(f"{backend.name}: {backend.describe()} -> {target.url}")
            case "up":
                backend.up()
                print(f"Postgres is ready on the {backend.name} backend at {target.url}")
            case "down":
                backend.down()
            case "reset":
                backend.reset()
                print(f"Database {target.dbname!r} is empty again")
            case "psql":
                return _exec(backend.psql_argv(arguments.args, interactive=True))
            case "sql":
                return _exec(
                    backend.psql_argv(["-c", arguments.statement], interactive=False)
                )
    except DatabaseError as error:
        print(f"robot-db: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
