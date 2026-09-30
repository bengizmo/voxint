"""The ORM models describe the schema the migrations build (issue #692).

Migrations own the schema; the app never calls ``create_all`` on a real
database. So nothing else notices when ``src/voxint/db/models.py`` drifts from
what ``alembic upgrade head`` produces, yet the models decide which operators
the ORM compiles (JSON vs JSONB) and document constraints the database
enforces (a partial unique index, NOT NULL). This test is ``alembic check`` as
a gate: autogenerate compares ``Base.metadata`` with a database at head and
must find no upgrade operations.

It compares exactly what ``alembic check`` compares. ``alembic/env.py`` sets
none of ``compare_type``, ``compare_server_default``, ``include_object`` or
``include_name``, so both use alembic's defaults: tables, columns, column
types, nullability, indexes, unique constraints and foreign keys. Alembic
autogenerate does not see CHECK constraints, triggers, or server-default
text, so drift in those is out of scope here.
"""

from __future__ import annotations

import pprint

from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import Engine

from voxint.db.models import Base


def test_models_match_schema_at_head(engine: Engine) -> None:
    with engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    assert diff == [], (
        "ORM models drift from the migrated schema; fix models.py or add a "
        "migration so `alembic check` reports nothing:\n" + pprint.pformat(diff)
    )
