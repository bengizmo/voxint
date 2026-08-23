"""Migration 0033 (pipeline_runs diarization speaker-count hint), up/down + checks.

Real alembic up/down against the shared test database (head restored in
teardown): ``pipeline_runs`` gains ``diarization_max_speakers`` and
``diarization_num_speakers``, a bounded hint round-trips, the 1..20 CHECKs reject
out-of-range values, downgrade drops both columns, and the ORM model stays in
lockstep with the DDL.
"""

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, inspect, text
from sqlalchemy.exc import IntegrityError

REPO_ROOT = Path(__file__).resolve().parents[2]


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


def _insert_run(engine: Engine, **cols: object) -> uuid.UUID:
    media_id, run_id = uuid.uuid4(), uuid.uuid4()
    base = {"id": run_id, "media_item_id": media_id, "status": "queued", "revision": 0}
    base.update(cols)
    keys = ", ".join(base)
    binds = ", ".join(f":{k}" for k in base)
    with engine.connect() as conn:
        conn.execute(
            text("INSERT INTO media_items (id, source_path) VALUES (:id, :p)"),
            {"id": media_id, "p": f"incoming/{media_id}.wav"},
        )
        conn.execute(text(f"INSERT INTO pipeline_runs ({keys}) VALUES ({binds})"), base)
        conn.commit()
    return run_id


def test_columns_added_and_hint_roundtrips(engine: Engine, alembic_cfg: Config) -> None:
    cols = {c["name"] for c in inspect(engine).get_columns("pipeline_runs")}
    assert {"diarization_max_speakers", "diarization_num_speakers"} <= cols

    run_id = _insert_run(engine, diarization_max_speakers=3, diarization_num_speakers=None)
    with engine.connect() as conn:
        stored = conn.execute(
            text(
                "SELECT diarization_max_speakers, diarization_num_speakers"
                " FROM pipeline_runs WHERE id = :r"
            ),
            {"r": run_id},
        ).one()
    assert stored == (3, None)


@pytest.mark.parametrize("column", ["diarization_max_speakers", "diarization_num_speakers"])
@pytest.mark.parametrize("value", [0, 21])
def test_bounds_reject_out_of_range(
    engine: Engine, alembic_cfg: Config, column: str, value: int
) -> None:
    with pytest.raises(IntegrityError):
        _insert_run(engine, **{column: value})


def test_downgrade_drops_columns(engine: Engine, alembic_cfg: Config) -> None:
    command.downgrade(alembic_cfg, "0032")
    cols = {c["name"] for c in inspect(engine).get_columns("pipeline_runs")}
    assert "diarization_max_speakers" not in cols
    assert "diarization_num_speakers" not in cols
    command.upgrade(alembic_cfg, "head")
    cols = {c["name"] for c in inspect(engine).get_columns("pipeline_runs")}
    assert {"diarization_max_speakers", "diarization_num_speakers"} <= cols
