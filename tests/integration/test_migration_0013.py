"""Migration 0013 (audio_artifacts reclamation columns), up/down + parity.

Mirrors the 0012 migration test: real alembic up/down against the shared test
database (head restored in teardown), every named CHECK exercised with a
rejecting row, the partial index asserted present, and ORM/DDL column parity.
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

from voxint.db.models import AudioArtifact

REPO_ROOT = Path(__file__).resolve().parents[2]

MEDIA_ID = "00000000-0000-0000-0000-00000000030a"
RUN_ID = "00000000-0000-0000-0000-00000000030b"
ART_ID = "00000000-0000-0000-0000-000000000310"


def _pg_type(type_: TypeEngine[object]) -> str:
    compiled = type_.compile(dialect=postgresql.dialect())
    return "DOUBLE PRECISION" if compiled == "FLOAT" else compiled


@pytest.fixture()
def alembic_cfg(engine: Engine) -> Iterator[Config]:
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


def _seed_run(engine: Engine) -> None:
    with engine.connect() as conn:
        conn.execute(
            text(
                "INSERT INTO media_items (id, source_path) VALUES"
                f" ('{MEDIA_ID}', 'incoming/gc/source')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO pipeline_runs (id, media_item_id, status, revision)"
                f" VALUES ('{RUN_ID}', '{MEDIA_ID}', 'completed', 0)"
            )
        )
        conn.commit()


def _insert_artifact(engine: Engine, **overrides: str) -> None:
    values = {
        "id": f"'{ART_ID}'",
        "pipeline_run_id": f"'{RUN_ID}'",
        "kind": "'preprocessed_audio'",
        "path": f"'artifacts/{RUN_ID}/normalized.wav'",
        **overrides,
    }
    columns = ", ".join(values)
    row = ", ".join(values.values())
    with engine.connect() as conn:
        conn.execute(text(f"INSERT INTO audio_artifacts ({columns}) VALUES ({row})"))
        conn.commit()


def test_migration_0013_roundtrip_and_checks(alembic_cfg: Config, engine: Engine) -> None:
    command.downgrade(alembic_cfg, "0012")
    cols = {c["name"] for c in inspect(engine).get_columns("audio_artifacts")}
    assert "reclaimed_at" not in cols
    assert "reclaimed_bytes" not in cols

    command.upgrade(alembic_cfg, "0013")
    cols = {c["name"] for c in inspect(engine).get_columns("audio_artifacts")}
    assert {"reclaimed_at", "reclaimed_bytes"} <= cols
    index_names = {ix["name"] for ix in inspect(engine).get_indexes("audio_artifacts")}
    assert "ix_audio_artifacts_reclaimable" in index_names

    _seed_run(engine)
    # A fresh, unreclaimed artifact (both NULL) is fine.
    _insert_artifact(engine)

    bad = "00000000-0000-0000-0000-0000000003"
    # Paired-nullability: exactly one of the two set is rejected, both ways.
    with pytest.raises(IntegrityError, match="audio_artifacts_reclaimed_shape_check"):
        _insert_artifact(engine, id=f"'{bad}11'", reclaimed_at="now()")
    with pytest.raises(IntegrityError, match="audio_artifacts_reclaimed_shape_check"):
        _insert_artifact(engine, id=f"'{bad}12'", reclaimed_bytes="0")
    # Non-negative bytes.
    with pytest.raises(IntegrityError, match="audio_artifacts_reclaimed_bytes_nonneg_check"):
        _insert_artifact(engine, id=f"'{bad}13'", reclaimed_at="now()", reclaimed_bytes="-1")
    # A fully-stamped row (both set, bytes >= 0) is accepted.
    _insert_artifact(engine, id=f"'{bad}14'", reclaimed_at="now()", reclaimed_bytes="123")

    command.downgrade(alembic_cfg, "0012")
    cols = {c["name"] for c in inspect(engine).get_columns("audio_artifacts")}
    assert "reclaimed_at" not in cols
    assert "reclaimed_bytes" not in cols


def test_audio_artifact_model_matches_migrated_schema(engine: Engine) -> None:
    reflected = {
        col["name"]: _pg_type(col["type"])
        for col in inspect(engine).get_columns("audio_artifacts")
    }
    model = {col.name: _pg_type(col.type) for col in AudioArtifact.__table__.columns}
    assert reflected == model
