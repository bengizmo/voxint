"""Migration 0064 (project_archived_at), issue #477.

Up/down against the shared test database: the nullable column appears on
projects, downgrade removes it, and the ORM model matches the migrated DDL.
Project rows and folder links survive rollback; re-upgrade resets archive state.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

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


def _col_names(engine: Engine) -> set[str]:
    inspector = inspect(engine)
    return {c["name"] for c in inspector.get_columns("projects")}


def test_upgrade_adds_column(alembic_cfg: Config, engine: Engine) -> None:
    command.downgrade(alembic_cfg, "0063")
    assert "archived_at" not in _col_names(engine)

    command.upgrade(alembic_cfg, "0064")

    assert "archived_at" in _col_names(engine)


def test_downgrade_removes_column(alembic_cfg: Config, engine: Engine) -> None:
    command.upgrade(alembic_cfg, "0064")
    assert "archived_at" in _col_names(engine)

    command.downgrade(alembic_cfg, "0063")

    assert "archived_at" not in _col_names(engine)


def test_column_is_nullable(
    alembic_cfg: Config, engine: Engine, session_factory: sessionmaker[Session]
) -> None:
    command.upgrade(alembic_cfg, "0064")
    column = next(c for c in inspect(engine).get_columns("projects") if c["name"] == "archived_at")
    assert column["nullable"]
    with session_factory() as session:
        project_id = session.execute(
            text(
                "INSERT INTO projects (id, name)"
                " VALUES (gen_random_uuid(), 'Active Project') RETURNING id"
            )
        ).scalar_one()
        archived_at = session.execute(
            text("SELECT archived_at FROM projects WHERE id = :pid"),
            {"pid": project_id},
        ).scalar_one()
        assert archived_at is None


def test_model_columns_match_migration(engine: Engine) -> None:
    from voxint.db.models import Project

    db_cols = _col_names(engine)
    model_cols = {c.name for c in Project.__table__.columns}
    assert "archived_at" in db_cols & model_cols


def test_round_trip_preserves_projects_and_folder_links(
    alembic_cfg: Config, engine: Engine
) -> None:
    command.upgrade(alembic_cfg, "0064")
    with engine.begin() as conn:
        active_id = conn.execute(
            text(
                "INSERT INTO projects (id, name)"
                " VALUES (gen_random_uuid(), 'Active Project') RETURNING id"
            )
        ).scalar_one()
        archived_id = conn.execute(
            text(
                "INSERT INTO projects (id, name, archived_at)"
                " VALUES (gen_random_uuid(), 'Archived Project', NOW()) RETURNING id"
            )
        ).scalar_one()
        folder_ids = []
        for path, project_id in (
            ("/test/f1", active_id),
            ("/test/f2", archived_id),
        ):
            folder_ids.append(
                conn.execute(
                    text(
                        "INSERT INTO media_folders (id, path, project_id)"
                        " VALUES (gen_random_uuid(), :path, :pid) RETURNING id"
                    ),
                    {"path": path, "pid": project_id},
                ).scalar_one()
            )
        assert conn.execute(
            text("SELECT archived_at FROM projects WHERE id = :pid"),
            {"pid": archived_id},
        ).scalar_one() is not None

    command.downgrade(alembic_cfg, "0063")
    assert "archived_at" not in _col_names(engine)

    with engine.connect() as conn:
        for folder_id, project_id, name in (
            (folder_ids[0], active_id, "Active Project"),
            (folder_ids[1], archived_id, "Archived Project"),
        ):
            assert conn.execute(
                text("SELECT name FROM projects WHERE id = :pid"), {"pid": project_id}
            ).scalar_one() == name
            assert conn.execute(
                text("SELECT project_id FROM media_folders WHERE id = :fid"),
                {"fid": folder_id},
            ).scalar_one() == project_id
    assert any(
        fk["constrained_columns"] == ["project_id"]
        and fk["referred_table"] == "projects"
        and fk["referred_columns"] == ["id"]
        for fk in inspect(engine).get_foreign_keys("media_folders")
    )

    command.upgrade(alembic_cfg, "0064")
    assert "archived_at" in _col_names(engine)
    with engine.connect() as conn:
        for project_id in (active_id, archived_id):
            assert conn.execute(
                text("SELECT archived_at FROM projects WHERE id = :pid"),
                {"pid": project_id},
            ).scalar_one() is None
