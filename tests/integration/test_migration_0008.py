"""Migration 0008 (transcript FTS indexes), up/down + model↔migration parity.

Mirrors test_migration_0007: real alembic up/down against the shared test
database (head restored in teardown), plus parity between the ORM-declared
``Index`` objects on ``TranscriptSegment`` and the migrated indexes. The
upgrade is exercised over pre-seeded rows to cover the build-scans-existing-
data path (there is no separate backfill — the index build IS the backfill).
"""

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, inspect, text

from voxint.db.models import TranscriptSegment
from voxint.db.search import ENHANCED_FTS_INDEX_NAME, RAW_FTS_INDEX_NAME, TS_CONFIG

REPO_ROOT = Path(__file__).resolve().parents[2]

FTS_INDEXES = {RAW_FTS_INDEX_NAME, ENHANCED_FTS_INDEX_NAME}


@pytest.fixture()
def alembic_cfg(engine: Engine) -> Iterator[Config]:
    # Same teardown contract as test_migration_0007: whatever happens, rebuild a
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


def _segment_indexes(engine: Engine) -> set[str]:
    return {
        name
        for i in inspect(engine).get_indexes("transcript_segments")
        if (name := i["name"]) is not None
    }


def _seed_segment(engine: Engine, raw: str, enhanced: str | None) -> None:
    media_id, run_id = uuid.uuid4(), uuid.uuid4()
    with engine.connect() as conn:
        conn.execute(
            text(
                "INSERT INTO media_items (id, source_path)"
                " VALUES (:id, :source_path)"
            ),
            {"id": media_id, "source_path": f"incoming/{media_id}.wav"},
        )
        conn.execute(
            text(
                "INSERT INTO pipeline_runs (id, media_item_id, status, revision)"
                " VALUES (:id, :media_item_id, 'completed', 0)"
            ),
            {"id": run_id, "media_item_id": media_id},
        )
        conn.execute(
            text(
                "INSERT INTO transcript_segments"
                " (id, pipeline_run_id, segment_index, start_seconds, end_seconds,"
                "  raw_text, enhanced_text, suspect)"
                " VALUES (:id, :run_id, 0, 0.0, 1.0, :raw, :enhanced, false)"
            ),
            {"id": uuid.uuid4(), "run_id": run_id, "raw": raw, "enhanced": enhanced},
        )
        conn.commit()


def test_migration_0008_roundtrip(alembic_cfg: Config, engine: Engine) -> None:
    # --- at 0007: no FTS indexes ---
    command.downgrade(alembic_cfg, "0007")
    assert not (FTS_INDEXES & _segment_indexes(engine))

    # Seed rows BEFORE the upgrade — the index build must cover existing data.
    _seed_segment(engine, "the compresser was leaking", "the compressor was leaking")
    _seed_segment(engine, "raw only segment", None)

    # --- upgrade to 0008: both GIN indexes appear and already serve queries ---
    command.upgrade(alembic_cfg, "0008")
    assert _segment_indexes(engine) >= FTS_INDEXES

    with engine.connect() as conn:
        for name in sorted(FTS_INDEXES):
            indexdef = conn.execute(
                text("SELECT indexdef FROM pg_indexes WHERE indexname = :n"),
                {"n": name},
            ).scalar_one()
            assert "USING gin" in indexdef
            assert f"'{TS_CONFIG}'" in indexdef

        # Pre-existing rows are searchable through BOTH variants: the raw
        # misrendering and its enhanced correction each match (the
        # anti-coalesce guarantee), and a NULL-enhanced row matches via raw.
        def hits(query: str) -> int:
            return conn.execute(
                text(
                    "SELECT count(*) FROM transcript_segments"
                    f" WHERE to_tsvector('{TS_CONFIG}', raw_text)"
                    f"       @@ websearch_to_tsquery('{TS_CONFIG}', :q)"
                    f"    OR to_tsvector('{TS_CONFIG}', enhanced_text)"
                    f"       @@ websearch_to_tsquery('{TS_CONFIG}', :q)"
                ),
                {"q": query},
            ).scalar_one()

        assert hits("compresser") == 1  # raw survives enhancement
        assert hits("compressor") == 1  # enhanced correction findable
        assert hits("segment") == 1  # NULL enhanced_text row via raw

    # --- downgrade is clean (purely additive migration) ---
    command.downgrade(alembic_cfg, "0007")
    assert not (FTS_INDEXES & _segment_indexes(engine))


def test_segment_model_indexes_match_migration(engine: Engine) -> None:
    """The ORM-declared FTS Index names exist in the migrated schema at head."""
    model_indexes = {i.name for i in TranscriptSegment.__table__.indexes}
    assert model_indexes >= FTS_INDEXES
    assert _segment_indexes(engine) >= FTS_INDEXES
    # The migrated expressions use the one dictionary the app compiles with.
    with engine.connect() as conn:
        for name in sorted(FTS_INDEXES):
            indexdef = conn.execute(
                text("SELECT indexdef FROM pg_indexes WHERE indexname = :n"),
                {"n": name},
            ).scalar_one()
            assert f"to_tsvector('{TS_CONFIG}'::regconfig" in indexdef
