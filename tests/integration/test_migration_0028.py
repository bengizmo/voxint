"""Migration 0028 (transcript_segments correction trace + version), up/down.

Real alembic up/down against the shared test database (head restored in
teardown): the two columns appear/disappear, a segment row that predates the
migration backfills to ``correction_trace == []`` and ``corrector_version IS
NULL`` (the server-default backfill, so the NOT-NULL column needs no data pass),
and the ``{version, input_base, entries}`` envelope round-trips. Issue #82.
"""

import json
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, inspect, text

REPO_ROOT = Path(__file__).resolve().parents[2]

_NEW_COLUMNS = ("correction_trace", "corrector_version")


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


def _segment_columns(engine: Engine) -> set[str]:
    return {c["name"] for c in inspect(engine).get_columns("transcript_segments")}


def _seed_pre_migration_segment(engine: Engine) -> str:
    """Insert a media_item → pipeline_run → transcript_segment row via RAW SQL.

    Raw SQL (not the ORM) because the ORM model already carries the two new
    columns, which do not exist at revision 0027 — an ORM insert would name them.
    Returns the segment id (hex).
    """
    media_id = uuid.uuid4().hex
    run_id = uuid.uuid4().hex
    seg_id = uuid.uuid4().hex
    with engine.connect() as conn:
        conn.execute(
            text("INSERT INTO media_items (id, source_path) VALUES (:id, :p)"),
            {"id": media_id, "p": f"incoming/{media_id}.wav"},
        )
        conn.execute(
            text(
                "INSERT INTO pipeline_runs (id, media_item_id, status, revision)"
                " VALUES (:id, :m, 'queued', 0)"
            ),
            {"id": run_id, "m": media_id},
        )
        conn.execute(
            text(
                "INSERT INTO transcript_segments"
                " (id, pipeline_run_id, segment_index, start_seconds, end_seconds,"
                "  raw_text, suspect)"
                " VALUES (:id, :r, 0, 0.0, 1.0, 'the zoom board met', false)"
            ),
            {"id": seg_id, "r": run_id},
        )
        conn.commit()
    return seg_id


def test_migration_0028_columns_backfill_and_roundtrip(
    alembic_cfg: Config, engine: Engine
) -> None:
    # Before: at 0027 neither new column exists.
    command.downgrade(alembic_cfg, "0027")
    before = _segment_columns(engine)
    for name in _NEW_COLUMNS:
        assert name not in before, name

    seg_id = _seed_pre_migration_segment(engine)

    command.upgrade(alembic_cfg, "0028")

    after = _segment_columns(engine)
    for name in _NEW_COLUMNS:
        assert name in after, name

    _select = (
        "SELECT correction_trace, corrector_version FROM transcript_segments WHERE id = :id"
    )
    with engine.connect() as conn:
        # Server-default backfill: the pre-existing row reads [] + NULL (no data
        # migration, and the NOT-NULL constraint holds).
        row = conn.execute(text(_select), {"id": seg_id}).one()
        assert row[0] == []
        assert row[1] is None

        # The persisted envelope + version round-trip.
        envelope = {
            "version": 1,
            "input_base": "llm",
            "entries": [
                {"id": "zb", "from": "zoom board", "to": "Zoning Board", "span": [4, 16]}
            ],
        }
        conn.execute(
            text(
                "UPDATE transcript_segments SET correction_trace = CAST(:t AS jsonb),"
                " corrector_version = 1 WHERE id = :id"
            ),
            {"t": json.dumps(envelope), "id": seg_id},
        )
        conn.commit()
        row = conn.execute(text(_select), {"id": seg_id}).one()
        assert row[0] == envelope
        assert row[1] == 1

    # Downgrade drops both columns again.
    command.downgrade(alembic_cfg, "0027")
    reverted = _segment_columns(engine)
    for name in _NEW_COLUMNS:
        assert name not in reverted, name
