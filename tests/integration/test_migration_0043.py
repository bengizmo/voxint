"""Migration 0043 (widen activity_events.kind), issue #162.

Real alembic up/down against the shared test database (head restored in
teardown): after 0043 the ``kind`` CHECK admits ``speaker_identified`` alongside
``run_completed`` and still rejects an unknown kind; the up/down/up roundtrip is
clean; and the downgrade deletes ONLY ``speaker_identified`` rows while
preserving ``run_completed`` rows (the narrow CHECK cannot be restored otherwise).
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, inspect, text
from sqlalchemy.exc import IntegrityError, ProgrammingError
from sqlalchemy.orm import Session, sessionmaker

from voxint.db.models import ActivityEvent

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


def _seed_run(session: Session) -> uuid.UUID:
    rid = uuid.uuid4()
    mid = uuid.uuid4()
    session.execute(
        text("INSERT INTO media_items (id, source_path) VALUES (:mid, :sp)"),
        {"mid": mid, "sp": f"incoming/{mid}/source"},
    )
    session.execute(
        text(
            "INSERT INTO pipeline_runs (id, media_item_id, status, revision,"
            " created_at, updated_at) VALUES (:rid, :mid, 'completed', 1, now(), now())"
        ),
        {"rid": rid, "mid": mid},
    )
    return rid


def test_widened_check_admits_speaker_and_rejects_unknown(
    engine: Engine, alembic_cfg: Config
) -> None:
    factory = sessionmaker(engine, expire_on_commit=False)
    with factory() as session:
        rid = _seed_run(session)
        session.flush()

        # speaker_identified is now a valid kind.
        session.add(
            ActivityEvent(
                kind="speaker_identified",
                pipeline_run_id=rid,
                title="Alice",
                href=f"/jobs/{rid}",
                occurrence_key=f"decision:{uuid.uuid4()}:identified",
            )
        )
        session.flush()

        # run_completed still valid.
        session.add(
            ActivityEvent(
                kind="run_completed",
                pipeline_run_id=rid,
                title="recording.mp3",
                href=f"/jobs/{rid}",
                occurrence_key=f"run:{rid}:completed",
            )
        )
        session.flush()

        # A genuinely-unknown kind is still refused.
        with pytest.raises((IntegrityError, ProgrammingError)), session.begin_nested():
            session.add(
                ActivityEvent(
                    kind="bogus_kind",
                    pipeline_run_id=rid,
                    title="x",
                    href=f"/jobs/{rid}",
                    occurrence_key=f"k-{uuid.uuid4()}",
                )
            )
            session.flush()


def test_downgrade_deletes_only_speaker_rows(engine: Engine, alembic_cfg: Config) -> None:
    factory = sessionmaker(engine, expire_on_commit=False)
    with factory() as session:
        rid = _seed_run(session)
        run_key = f"run:{rid}:completed"
        spk_key = f"decision:{uuid.uuid4()}:identified"
        session.add_all(
            [
                ActivityEvent(
                    kind="run_completed",
                    pipeline_run_id=rid,
                    title="recording.mp3",
                    href=f"/jobs/{rid}",
                    occurrence_key=run_key,
                ),
                ActivityEvent(
                    kind="speaker_identified",
                    pipeline_run_id=rid,
                    title="Alice",
                    href=f"/jobs/{rid}",
                    occurrence_key=spk_key,
                ),
            ]
        )
        session.commit()

    # Downgrade past 0043: speaker rows are deleted so the narrow CHECK can be
    # restored; run_completed rows survive.
    command.downgrade(alembic_cfg, "0042")

    with engine.connect() as conn:
        kinds = [
            row[0]
            for row in conn.execute(
                text("SELECT kind FROM activity_events ORDER BY id")
            )
        ]
    assert kinds == ["run_completed"]

    # And the restored narrow CHECK now rejects speaker_identified again.
    insert = text(
        "INSERT INTO activity_events"
        " (kind, pipeline_run_id, title, href, occurrence_key)"
        " VALUES ('speaker_identified', :rid, 'x', :h, :k)"
    )
    with factory() as session:
        params = {"rid": rid, "h": f"/jobs/{rid}", "k": f"k-{uuid.uuid4()}"}
        with pytest.raises((IntegrityError, ProgrammingError)), session.begin_nested():
            session.execute(insert, params)


def test_up_down_up_roundtrip(engine: Engine, alembic_cfg: Config) -> None:
    command.downgrade(alembic_cfg, "0042")
    command.upgrade(alembic_cfg, "0043")
    assert inspect(engine).has_table("activity_events")
    # After the roundtrip the widened CHECK is in force again.
    factory = sessionmaker(engine, expire_on_commit=False)
    with factory() as session:
        rid = _seed_run(session)
        session.add(
            ActivityEvent(
                kind="speaker_identified",
                pipeline_run_id=rid,
                title="Bob",
                href=f"/jobs/{rid}",
                occurrence_key=f"decision:{uuid.uuid4()}:identified",
            )
        )
        session.flush()
