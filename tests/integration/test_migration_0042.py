"""Migration 0042 (activity_events), issue #162.

Real alembic up/down against the shared test database (head restored in
teardown): the table appears with its constraints; the kind CHECK, the
occurrence_key uniqueness, and the length bounds hold; the downgrade drops the
table; the ORM model matches the migrated DDL column-for-column.
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
    """A media item + a run so the activity FK is satisfiable."""
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


def test_downgrade_drops_and_upgrade_restores(engine: Engine, alembic_cfg: Config) -> None:
    assert inspect(engine).has_table("activity_events")
    command.downgrade(alembic_cfg, "0041")
    assert not inspect(engine).has_table("activity_events")
    command.upgrade(alembic_cfg, "head")
    assert inspect(engine).has_table("activity_events")


def test_schema_constraints_hold(engine: Engine, alembic_cfg: Config) -> None:
    factory = sessionmaker(engine, expire_on_commit=False)
    with factory() as session:
        rid = _seed_run(session)
        session.flush()

        # An unknown kind is refused by the CHECK.
        with pytest.raises((IntegrityError, ProgrammingError)), session.begin_nested():
            session.add(
                ActivityEvent(
                    kind="speaker_identified",
                    pipeline_run_id=rid,
                    title="x",
                    href="/jobs/x",
                    occurrence_key=f"k-{uuid.uuid4()}",
                )
            )
            session.flush()

        # An over-long occurrence_key is refused by the length CHECK.
        with pytest.raises(IntegrityError), session.begin_nested():
            session.add(
                ActivityEvent(
                    kind="run_completed",
                    pipeline_run_id=rid,
                    title="x",
                    href="/jobs/x",
                    occurrence_key="k" * 201,
                )
            )
            session.flush()

        # The occurrence key is unique.
        key = f"run:{rid}:completed"
        session.add(
            ActivityEvent(
                kind="run_completed",
                pipeline_run_id=rid,
                title="a",
                href="/jobs/a",
                occurrence_key=key,
            )
        )
        session.flush()
        with pytest.raises(IntegrityError), session.begin_nested():
            session.add(
                ActivityEvent(
                    kind="run_completed",
                    pipeline_run_id=rid,
                    title="b",
                    href="/jobs/b",
                    occurrence_key=key,
                )
            )
            session.flush()
        session.rollback()


def test_orm_matches_migrated_schema(engine: Engine, alembic_cfg: Config) -> None:
    reflected = {c["name"] for c in inspect(engine).get_columns("activity_events")}
    model = {c.name for c in ActivityEvent.__table__.columns}
    assert reflected == model
