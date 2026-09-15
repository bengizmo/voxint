"""Migration 0062 (learned_corrections), issue #476.

Up/down against the shared test database: the new column and tables appear with
their constraints; CHECKs, unique, and cascade behaviours hold; ORM models match
the migrated DDL.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from voxint.db.models import (
    LearnedCorrection,
    LearnedCorrectionEvidence,
    Project,
)

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


def _seed_project(session: Session) -> uuid.UUID:
    """A project with its own corrections list (non-null) so learning is valid."""
    pid = uuid.uuid4()
    session.execute(
        text(
            "INSERT INTO projects (id, name, corrections, learn_corrections)"
            " VALUES (:pid, :name, :corrections, true)"
        ),
        {"pid": pid, "name": f"proj-{pid}", "corrections": "[]"},
    )
    return pid


def _seed_segment(session: Session) -> uuid.UUID:
    """A media item + run + segment so the evidence FK is satisfiable."""
    mid = uuid.uuid4()
    rid = uuid.uuid4()
    sid = uuid.uuid4()
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
    session.execute(
        text(
            "INSERT INTO transcript_segments (id, pipeline_run_id, segment_index,"
            " start_seconds, end_seconds, raw_text)"
            " VALUES (:sid, :rid, 0, 0.0, 1.0, 'hello')"
        ),
        {"sid": sid, "rid": rid},
    )
    return sid


def test_downgrade_drops_and_upgrade_restores(
    engine: Engine, alembic_cfg: Config
) -> None:
    insp = inspect(engine)
    assert insp.has_table("learned_corrections")
    assert insp.has_table("learned_correction_evidence")
    cols = {c["name"] for c in insp.get_columns("projects")}
    assert "learn_corrections" in cols

    command.downgrade(alembic_cfg, "0061")
    insp = inspect(engine)
    assert not insp.has_table("learned_corrections")
    assert not insp.has_table("learned_correction_evidence")
    cols = {c["name"] for c in insp.get_columns("projects")}
    assert "learn_corrections" not in cols

    command.upgrade(alembic_cfg, "head")
    insp = inspect(engine)
    assert insp.has_table("learned_corrections")
    assert insp.has_table("learned_correction_evidence")


def test_status_check_holds(engine: Engine, alembic_cfg: Config) -> None:
    factory = sessionmaker(engine, expire_on_commit=False)
    with factory() as session:
        pid = _seed_project(session)
        session.flush()

        with pytest.raises(IntegrityError), session.begin_nested():
            session.execute(
                text(
                    "INSERT INTO learned_corrections"
                    " (id, project_id, match, replace, status)"
                    " VALUES (:id, :pid, 'foo', 'bar', 'bogus')"
                ),
                {"id": uuid.uuid4(), "pid": pid},
            )

        session.rollback()


def test_accepted_rule_id_check_holds(engine: Engine, alembic_cfg: Config) -> None:
    factory = sessionmaker(engine, expire_on_commit=False)
    with factory() as session:
        pid = _seed_project(session)
        session.flush()

        # accepted without rule_id
        with pytest.raises(IntegrityError), session.begin_nested():
            session.execute(
                text(
                    "INSERT INTO learned_corrections"
                    " (id, project_id, match, replace, status, accepted_rule_id)"
                    " VALUES (:id, :pid, 'foo', 'bar', 'accepted', NULL)"
                ),
                {"id": uuid.uuid4(), "pid": pid},
            )

        # suggested with rule_id
        with pytest.raises(IntegrityError), session.begin_nested():
            session.execute(
                text(
                    "INSERT INTO learned_corrections"
                    " (id, project_id, match, replace, status, accepted_rule_id)"
                    " VALUES (:id, :pid, 'baz', 'qux', 'suggested', 'some-id')"
                ),
                {"id": uuid.uuid4(), "pid": pid},
            )

        session.rollback()


def test_unique_constraint_holds(engine: Engine, alembic_cfg: Config) -> None:
    factory = sessionmaker(engine, expire_on_commit=False)
    with factory() as session:
        pid = _seed_project(session)
        session.flush()

        session.execute(
            text(
                "INSERT INTO learned_corrections"
                " (id, project_id, match, replace, status)"
                " VALUES (:id, :pid, 'foo', 'bar', 'suggested')"
            ),
            {"id": uuid.uuid4(), "pid": pid},
        )
        session.flush()

        with pytest.raises(IntegrityError), session.begin_nested():
            session.execute(
                text(
                    "INSERT INTO learned_corrections"
                    " (id, project_id, match, replace, status)"
                    " VALUES (:id, :pid, 'foo', 'bar', 'suggested')"
                ),
                {"id": uuid.uuid4(), "pid": pid},
            )

        session.rollback()


def test_project_cascade_deletes_rows(engine: Engine, alembic_cfg: Config) -> None:
    factory = sessionmaker(engine, expire_on_commit=False)
    with factory() as session:
        pid = _seed_project(session)
        sid = _seed_segment(session)
        lcid = uuid.uuid4()
        session.execute(
            text(
                "INSERT INTO learned_corrections"
                " (id, project_id, match, replace, status)"
                " VALUES (:id, :pid, 'foo', 'bar', 'suggested')"
            ),
            {"id": lcid, "pid": pid},
        )
        session.execute(
            text(
                "INSERT INTO learned_correction_evidence"
                " (learned_correction_id, segment_id)"
                " VALUES (:lcid, :sid)"
            ),
            {"lcid": lcid, "sid": sid},
        )
        session.flush()

        session.execute(text("DELETE FROM projects WHERE id = :pid"), {"pid": pid})
        session.flush()

        row = session.execute(
            text("SELECT count(*) FROM learned_corrections WHERE project_id = :pid"),
            {"pid": pid},
        ).scalar()
        assert row == 0

        row = session.execute(
            text(
                "SELECT count(*) FROM learned_correction_evidence"
                " WHERE learned_correction_id = :lcid"
            ),
            {"lcid": lcid},
        ).scalar()
        assert row == 0

        session.rollback()


def test_segment_cascade_deletes_evidence(
    engine: Engine, alembic_cfg: Config
) -> None:
    factory = sessionmaker(engine, expire_on_commit=False)
    with factory() as session:
        pid = _seed_project(session)
        sid = _seed_segment(session)
        lcid = uuid.uuid4()
        session.execute(
            text(
                "INSERT INTO learned_corrections"
                " (id, project_id, match, replace, status)"
                " VALUES (:id, :pid, 'foo', 'bar', 'suggested')"
            ),
            {"id": lcid, "pid": pid},
        )
        session.execute(
            text(
                "INSERT INTO learned_correction_evidence"
                " (learned_correction_id, segment_id)"
                " VALUES (:lcid, :sid)"
            ),
            {"lcid": lcid, "sid": sid},
        )
        session.flush()

        session.execute(
            text("DELETE FROM transcript_segments WHERE id = :sid"), {"sid": sid}
        )
        session.flush()

        row = session.execute(
            text(
                "SELECT count(*) FROM learned_correction_evidence"
                " WHERE learned_correction_id = :lcid"
            ),
            {"lcid": lcid},
        ).scalar()
        assert row == 0

        session.rollback()


def test_orm_matches_migrated_schema(engine: Engine, alembic_cfg: Config) -> None:
    insp = inspect(engine)

    reflected = {c["name"] for c in insp.get_columns("learned_corrections")}
    model = {c.name for c in LearnedCorrection.__table__.columns}
    assert reflected == model

    reflected = {c["name"] for c in insp.get_columns("learned_correction_evidence")}
    model = {c.name for c in LearnedCorrectionEvidence.__table__.columns}
    assert reflected == model

    # The new column on projects
    proj_cols = {c["name"] for c in insp.get_columns("projects")}
    proj_model = {c.name for c in Project.__table__.columns}
    assert proj_model.issubset(proj_cols)
    assert "learn_corrections" in proj_cols
