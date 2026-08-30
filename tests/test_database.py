import pytest

from everest_robot import database
from everest_robot.database import (
    DatabaseError,
    DockerBackend,
    NativeBackend,
    parse_database_url,
    select_backend,
    target_from_environment,
)

TARGET = parse_database_url("postgresql://robot:robot@localhost:5432/robot")


def test_parse_database_url_splits_the_fields_the_backends_provision() -> None:
    assert TARGET.host == "localhost"
    assert TARGET.port == 5432
    assert TARGET.user == "robot"
    assert TARGET.password == "robot"
    assert TARGET.dbname == "robot"
    assert TARGET.is_local


def test_parse_database_url_defaults_port_and_reads_a_remote_host() -> None:
    target = parse_database_url("postgresql://cell@cell-host/robot")

    assert target.port == 5432
    assert target.password is None
    assert not target.is_local


@pytest.mark.parametrize("url", ["mysql://robot@localhost/robot", "postgresql://robot@host"])
def test_parse_database_url_refuses_what_it_cannot_provision(url: str) -> None:
    with pytest.raises(DatabaseError):
        parse_database_url(url)


def test_target_from_environment_falls_back_to_the_documented_default() -> None:
    assert target_from_environment({}) == TARGET
    override = target_from_environment({"ABSURD_DATABASE_URL": "postgresql://a:b@h:6000/db"})
    assert override.port == 6000


def test_docker_is_preferred_when_it_can_actually_run_something() -> None:
    assert (
        select_backend(
            None, compose_available=True, daemon_available=True, native_available=True
        )
        == "docker"
    )


def test_a_host_postgres_is_used_when_compose_is_missing() -> None:
    assert (
        select_backend(
            None, compose_available=False, daemon_available=False, native_available=True
        )
        == "native"
    )


def test_an_installed_compose_with_a_dead_daemon_falls_back_rather_than_failing() -> None:
    assert (
        select_backend(
            None, compose_available=True, daemon_available=False, native_available=True
        )
        == "native"
    )


def test_a_dead_daemon_with_no_host_postgres_says_which_to_fix() -> None:
    with pytest.raises(DatabaseError, match="daemon is not reachable"):
        select_backend(
            None, compose_available=True, daemon_available=False, native_available=False
        )


def test_neither_backend_available_names_both_ways_out() -> None:
    with pytest.raises(DatabaseError, match="no local Postgres backend"):
        select_backend(
            None, compose_available=False, daemon_available=False, native_available=False
        )


def test_an_explicit_request_wins_over_what_is_installed() -> None:
    assert (
        select_backend(
            "native", compose_available=True, daemon_available=True, native_available=False
        )
        == "native"
    )
    assert (
        select_backend(
            "Docker", compose_available=False, daemon_available=False, native_available=True
        )
        == "docker"
    )


def test_an_unknown_backend_is_refused_rather_than_guessed() -> None:
    with pytest.raises(DatabaseError, match="podman"):
        select_backend(
            "podman", compose_available=True, daemon_available=True, native_available=True
        )


def test_docker_runs_psql_inside_the_container_so_the_host_needs_no_client() -> None:
    backend = DockerBackend(["docker", "compose"], TARGET)

    assert backend.psql_argv([], interactive=True) == [
        "docker",
        "compose",
        "exec",
        "postgres",
        "psql",
        "-U",
        "robot",
        "-d",
        "robot",
    ]
    assert "-T" in backend.psql_argv(["-c", "SELECT 1;"], interactive=False)


def test_native_drives_the_host_client_with_the_configured_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(database.shutil, "which", lambda name: f"/usr/bin/{name}")
    backend = NativeBackend(TARGET)

    assert backend.psql_argv(["-c", "SELECT 1;"], interactive=False) == [
        "psql",
        "postgresql://robot:robot@localhost:5432/robot",
        "-c",
        "SELECT 1;",
    ]


def test_native_says_how_to_get_a_client_rather_than_failing_on_exec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(database.shutil, "which", lambda name: None)

    with pytest.raises(DatabaseError, match="needs a psql client"):
        NativeBackend(TARGET).psql_argv([], interactive=True)
