"""Migration 0009 (source metadata + operator notes), up/down + model parity.

Mirrors test_migration_0006: runs the real alembic up/down against the shared
test database and restores head in the fixture teardown. Also asserts the ORM
model matches the migrated schema — no autogenerate parity test exists in-tree
and the whole suite builds its schema from the alembic chain, so a drift here
would surface far from its cause.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, inspect, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.types import TypeEngine

from voxint.db.models import MediaSourceMetadata

REPO_ROOT = Path(__file__).resolve().parents[2]


def _pg_type(type_: TypeEngine[object]) -> str:
    """Render a type as its postgres DDL string so model and reflected agree.

    Postgres has no distinct FLOAT type — ``FLOAT`` *is* ``DOUBLE PRECISION``
    (float8), so the model's ``Float`` compiles to "FLOAT" while reflection
    returns "DOUBLE PRECISION". Normalize that one server-side equivalence so
    the parity check compares real types, not spelling.
    """
    compiled = type_.compile(dialect=postgresql.dialect())
    return "DOUBLE PRECISION" if compiled == "FLOAT" else compiled


@pytest.fixture()
def alembic_cfg(engine: Engine) -> Iterator[Config]:
    # Depending on `engine` guarantees DATABASE_URL is exported and the schema
    # starts at head. The finally restores a pristine head regardless of outcome
    # so a mid-suite migration test never leaves the DB downgraded for the tests
    # that follow.
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "alembic"))
    try:
        yield cfg
    finally:
        with engine.connect() as conn:
            conn.execute(text("DROP SCHEMA public CASCADE"))
            conn.execute(text("CREATE SCHEMA public"))
            conn.commit()
        command.upgrade(cfg, "head")


def _tables(engine: Engine) -> set[str]:
    return set(inspect(engine).get_table_names())


def _pipeline_run_columns(engine: Engine) -> set[str]:
    return {c["name"] for c in inspect(engine).get_columns("pipeline_runs")}


def test_migration_0009_roundtrip(alembic_cfg: Config, engine: Engine) -> None:
    # --- at 0008: neither the table nor the notes column exists yet ---
    command.downgrade(alembic_cfg, "0008")
    assert "media_source_metadata" not in _tables(engine)
    assert "operator_notes" not in _pipeline_run_columns(engine)

    # --- upgrade to 0009: both appear ---
    command.upgrade(alembic_cfg, "0009")
    assert "media_source_metadata" in _tables(engine)
    assert "operator_notes" in _pipeline_run_columns(engine)

    with engine.connect() as conn:
        conn.execute(
            text(
                "INSERT INTO media_items (id, source_path) VALUES"
                " ('00000000-0000-0000-0000-000000000001', 'incoming/x/source')"
            )
        )
        # a minimal row relies on the tags server default
        conn.execute(
            text(
                "INSERT INTO media_source_metadata"
                " (id, media_item_id, source_kind, raw_schema_version, acquired_at)"
                " VALUES ('00000000-0000-0000-0000-000000000002',"
                " '00000000-0000-0000-0000-000000000001', 'ytdlp', 1, now())"
            )
        )
        conn.commit()
        row = conn.execute(
            text("SELECT tags, raw FROM media_source_metadata")
        ).one()
    assert row.tags == []
    assert row.raw is None

    # the unique FK pins the snapshot to zero-or-one per MediaItem
    with engine.connect() as conn, pytest.raises(IntegrityError):
        conn.execute(
            text(
                "INSERT INTO media_source_metadata"
                " (id, media_item_id, source_kind, raw_schema_version, acquired_at)"
                " VALUES ('00000000-0000-0000-0000-000000000003',"
                " '00000000-0000-0000-0000-000000000001', 'ytdlp', 1, now())"
            )
        )

    # CHECKs: unknown source_kind, negative duration, schema_version < 1
    for bad in (
        "('00000000-0000-0000-0000-000000000004',"
        " '00000000-0000-0000-0000-000000000001', 'scraped', 1, now())",
        "('00000000-0000-0000-0000-000000000005',"
        " '00000000-0000-0000-0000-000000000001', 'ytdlp', 0, now())",
    ):
        with engine.connect() as conn, pytest.raises(IntegrityError):
            conn.execute(
                text(
                    "INSERT INTO media_source_metadata"
                    " (id, media_item_id, source_kind, raw_schema_version, acquired_at)"
                    f" VALUES {bad}"
                )
            )
    with engine.connect() as conn, pytest.raises(IntegrityError):
        conn.execute(
            text(
                "UPDATE media_source_metadata SET duration_seconds = -1"
            )
        )

    # --- downgrade is clean (purely additive migration) ---
    command.downgrade(alembic_cfg, "0008")
    assert "media_source_metadata" not in _tables(engine)
    assert "operator_notes" not in _pipeline_run_columns(engine)


def test_media_source_metadata_model_matches_migration(engine: Engine) -> None:
    """The ORM model and the migrated table agree on columns + nullability.

    The suite builds its schema from the alembic chain, not create_all, and there
    is no autogenerate parity check — so assert it here for the new table.
    """
    insp = inspect(engine)
    migrated = {c["name"]: c for c in insp.get_columns("media_source_metadata")}
    model = {c.name: c for c in MediaSourceMetadata.__table__.columns}
    assert set(migrated) == set(model)
    for name, col in model.items():
        assert migrated[name]["nullable"] == col.nullable, f"{name} nullability drift"
        assert _pg_type(migrated[name]["type"]) == _pg_type(col.type), (
            f"{name} type drift: migrated={_pg_type(migrated[name]['type'])} "
            f"model={_pg_type(col.type)}"
        )
    check_names = {c["name"] for c in insp.get_check_constraints("media_source_metadata")}
    assert {
        "media_source_metadata_kind_check",
        "media_source_metadata_duration_nonneg_check",
        "media_source_metadata_raw_schema_version_check",
    } <= check_names
    fks = insp.get_foreign_keys("media_source_metadata")
    fk = next(f for f in fks if f["constrained_columns"] == ["media_item_id"])
    assert fk["referred_table"] == "media_items"
    assert fk["referred_columns"] == ["id"]
    uniques = {
        tuple(u["column_names"]) for u in insp.get_unique_constraints("media_source_metadata")
    }
    assert ("media_item_id",) in uniques


def test_operator_notes_column_matches_model(engine: Engine) -> None:
    migrated = {c["name"]: c for c in inspect(engine).get_columns("pipeline_runs")}
    assert "operator_notes" in migrated
    assert migrated["operator_notes"]["nullable"] is True
    assert _pg_type(migrated["operator_notes"]["type"]) == "TEXT"
