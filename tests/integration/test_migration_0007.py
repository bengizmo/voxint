"""Migration 0007 (speaker roster lifecycle), up/down + model↔migration parity.

Mirrors test_migration_0006: runs the real alembic up/down against the shared
test database and restores head in the fixture teardown, and asserts the ORM
``Speaker`` model matches the migrated schema (the suite builds its schema from
the alembic chain and there is no autogenerate parity check).
"""

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, inspect, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.types import TypeEngine

from voxint.db.models import Speaker

REPO_ROOT = Path(__file__).resolve().parents[2]

NOW = datetime.now(UTC)

LIFECYCLE_COLUMNS = {"merged_into_id", "merged_at", "deleted_at"}
LIFECYCLE_CHECKS = {
    "speakers_no_self_merge_check",
    "speakers_merge_fields_together_check",
    "speakers_not_merged_and_deleted_check",
}


def _pg_type(type_: TypeEngine[object]) -> str:
    """Render a type as its postgres DDL string so model and reflected agree."""
    return type_.compile(dialect=postgresql.dialect())


@pytest.fixture()
def alembic_cfg(engine: Engine) -> Iterator[Config]:
    # Same teardown contract as test_migration_0006: whatever happens, rebuild a
    # pristine head so later tests never see a downgraded schema.
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


def _speaker_columns(engine: Engine) -> set[str]:
    return {c["name"] for c in inspect(engine).get_columns("speakers")}


def _insert_speaker(conn: object, **cols: object) -> uuid.UUID:
    speaker_id = uuid.uuid4()
    names = ", ".join(["id", "display_name", *cols])
    params = ", ".join([":id", ":display_name", *(f":{c}" for c in cols)])
    conn.execute(  # type: ignore[attr-defined]
        text(f"INSERT INTO speakers ({names}) VALUES ({params})"),
        {"id": speaker_id, "display_name": f"Speaker {speaker_id}", **cols},
    )
    return speaker_id


def test_migration_0007_roundtrip(alembic_cfg: Config, engine: Engine) -> None:
    # --- at 0006: no lifecycle columns ---
    command.downgrade(alembic_cfg, "0006")
    assert not (LIFECYCLE_COLUMNS & _speaker_columns(engine))

    # --- upgrade to 0007: columns + constraints appear ---
    command.upgrade(alembic_cfg, "0007")
    assert _speaker_columns(engine) >= LIFECYCLE_COLUMNS

    insp = inspect(engine)
    checks = {c["name"] for c in insp.get_check_constraints("speakers")}
    assert checks >= LIFECYCLE_CHECKS
    # display_name uniqueness survives untouched — names are never reusable.
    uniques = {u["name"] for u in insp.get_unique_constraints("speakers")}
    assert "speakers_display_name_key" in uniques

    with engine.connect() as conn:
        target = _insert_speaker(conn)
        # a well-formed merge tombstone and a well-formed archive both insert
        _insert_speaker(conn, merged_into_id=target, merged_at=NOW)
        _insert_speaker(conn, deleted_at=NOW)
        conn.commit()

    # self-merge rejected
    with engine.connect() as conn, pytest.raises(IntegrityError):
        bad = uuid.uuid4()
        conn.execute(
            text(
                "INSERT INTO speakers (id, display_name, merged_into_id, merged_at)"
                " VALUES (:id, :name, :id, now())"
            ),
            {"id": bad, "name": f"Speaker {bad}"},
        )

    # merged_into_id without merged_at rejected (and vice versa)
    with engine.connect() as conn, pytest.raises(IntegrityError):
        _insert_speaker(conn, merged_into_id=target)
    with engine.connect() as conn, pytest.raises(IntegrityError):
        _insert_speaker(conn, merged_at=NOW)

    # merged AND deleted rejected
    with engine.connect() as conn, pytest.raises(IntegrityError):
        _insert_speaker(
            conn, merged_into_id=target, merged_at=NOW, deleted_at=NOW
        )

    # --- downgrade is clean (purely additive migration) ---
    command.downgrade(alembic_cfg, "0006")
    assert not (LIFECYCLE_COLUMNS & _speaker_columns(engine))


def test_speaker_model_matches_migration(engine: Engine) -> None:
    """The ORM model and the migrated speakers table agree at head."""
    insp = inspect(engine)
    migrated = {c["name"]: c for c in insp.get_columns("speakers")}
    model = {c.name: c for c in Speaker.__table__.columns}
    assert set(migrated) == set(model)
    for name, col in model.items():
        assert migrated[name]["nullable"] == col.nullable, f"{name} nullability drift"
        assert _pg_type(migrated[name]["type"]) == _pg_type(col.type), (
            f"{name} type drift: migrated={_pg_type(migrated[name]['type'])} "
            f"model={_pg_type(col.type)}"
        )
    checks = {c["name"] for c in insp.get_check_constraints("speakers")}
    assert checks >= LIFECYCLE_CHECKS
    # The self-FK must point merged_into_id -> speakers.id.
    fks = insp.get_foreign_keys("speakers")
    fk = next(f for f in fks if f["constrained_columns"] == ["merged_into_id"])
    assert fk["referred_table"] == "speakers"
    assert fk["referred_columns"] == ["id"]
