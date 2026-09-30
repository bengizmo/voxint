"""The ORM models describe the schema the migrations build (issue #692).

Migrations own the schema; the app never calls ``create_all`` on a real
database. So nothing else notices when ``src/voxint/db/models.py`` drifts from
what ``alembic upgrade head`` produces, yet the models decide which operators
the ORM compiles (JSON vs JSONB) and document constraints the database
enforces (a partial unique index, NOT NULL). This test runs ``alembic check``
itself, through ``alembic/env.py``, against this worker's database at head, so
it compares exactly what the command compares and cannot drift from it. It
also refuses a database that is not at head.

Alembic autogenerate does not see CHECK constraints or triggers, and with
env.py's settings it does not compare server-default text, so drift in those
is out of scope here.
"""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.util.exc import CommandError
from sqlalchemy import Engine

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_models_match_schema_at_head(engine: Engine) -> None:
    # ``engine`` migrated this worker's database to head and pointed
    # DATABASE_URL (which env.py reads) at it.
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "alembic"))
    try:
        command.check(cfg)
    except CommandError as exc:
        raise AssertionError(
            "`alembic check` failed: either the ORM models drift from the migrated "
            "schema (fix models.py or add a migration) or this database is not at "
            f"head:\n{exc}"
        ) from None
