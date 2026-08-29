"""Initialize Absurd without requiring a host installation of ``psql``."""

import os

import psycopg
from absurdctl import BUNDLED_SCHEMA_SQL

DEFAULT_DATABASE_URL = "postgresql://robot:robot@localhost:5432/robot"


def init() -> None:
    """Install the locked Absurd schema and idempotently create the robot queue."""

    database_url = os.getenv("ABSURD_DATABASE_URL", DEFAULT_DATABASE_URL)
    queue_name = os.getenv("ROBOT_QUEUE", "robot")
    with psycopg.connect(database_url) as connection:
        connection.execute(BUNDLED_SCHEMA_SQL)
        connection.execute("SELECT absurd.create_queue(%s)", (queue_name,))
    print(f"Absurd schema initialized; queue {queue_name!r} is ready")


if __name__ == "__main__":
    init()
