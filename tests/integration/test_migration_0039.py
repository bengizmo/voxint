"""Migration 0039 (projects + media_folders + media location split), issue #153.

Real alembic up/down against the shared test database (head restored in
teardown): the ``projects`` / ``media_folders`` tables and the two new
``media_items`` columns appear with their constraints; the backfill from the
legacy ``app_settings`` registrations creates one folder row per registration,
seeds ``current_path`` from ``source_path``, assigns folder membership by
deepest-ancestor, and preserves the effective domain pack. A nested registration
whose pack cannot survive a flat folder relation aborts the upgrade.
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

REPO_ROOT = Path(__file__).resolve().parents[2]

_TABLES = ("projects", "media_folders")
_MEDIA_COLUMNS = ("current_path", "media_folder_id")


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


def _seed_app_settings(
    conn: object, *, folders: list[str], packs: dict[str, str]
) -> None:
    conn.execute(  # type: ignore[attr-defined]
        text(
            "INSERT INTO app_settings (id, onboarding_complete, media_folders,"
            " folder_domain_packs) VALUES (1, true, :folders, CAST(:packs AS jsonb))"
        ),
        {"folders": folders, "packs": _json(packs)},
    )


def _json(value: dict[str, str]) -> str:
    import json

    return json.dumps(value)


def _seed_media(conn: object, source_path: str) -> uuid.UUID:
    media_id = uuid.uuid4()
    conn.execute(  # type: ignore[attr-defined]
        text("INSERT INTO media_items (id, source_path) VALUES (:id, :p)"),
        {"id": media_id, "p": source_path},
    )
    return media_id


def test_schema_present_and_constraints_hold(
    engine: Engine, alembic_cfg: Config
) -> None:
    inspector = inspect(engine)
    for table in _TABLES:
        assert inspector.has_table(table)
    media_cols = {c["name"] for c in inspector.get_columns("media_items")}
    assert media_cols >= set(_MEDIA_COLUMNS)

    with engine.connect() as conn:
        pid = uuid.uuid4()
        conn.execute(
            text("INSERT INTO projects (id, name) VALUES (:id, 'P')"), {"id": pid}
        )
        conn.commit()
    # Duplicate project name refused.
    with engine.connect() as conn, pytest.raises(IntegrityError):
        conn.execute(
            text("INSERT INTO projects (id, name) VALUES (:id, 'P')"),
            {"id": uuid.uuid4()},
        )
    # Blank name refused.
    with engine.connect() as conn, pytest.raises(IntegrityError):
        conn.execute(
            text("INSERT INTO projects (id, name) VALUES (:id, '   ')"),
            {"id": uuid.uuid4()},
        )
    # corrections must be a JSON array when present.
    with engine.connect() as conn, pytest.raises(IntegrityError):
        conn.execute(
            text(
                "INSERT INTO projects (id, name, corrections)"
                " VALUES (:id, 'Q', CAST('{}' AS jsonb))"
            ),
            {"id": uuid.uuid4()},
        )

    with engine.connect() as conn:
        fid = uuid.uuid4()
        conn.execute(
            text(
                "INSERT INTO media_folders (id, path, project_id)"
                " VALUES (:id, 'audio', :pid)"
            ),
            {"id": fid, "pid": pid},
        )
        conn.commit()
    # Duplicate folder path refused.
    with engine.connect() as conn, pytest.raises(IntegrityError):
        conn.execute(
            text("INSERT INTO media_folders (id, path) VALUES (:id, 'audio')"),
            {"id": uuid.uuid4()},
        )
    # Deleting the project nulls the folder FK (ON DELETE SET NULL), not the row.
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM projects WHERE id = :id"), {"id": pid})
        conn.commit()
        remaining = conn.execute(
            text("SELECT project_id FROM media_folders WHERE id = :id"), {"id": fid}
        ).scalar_one()
        assert remaining is None


def test_backfill_from_legacy_settings(engine: Engine, alembic_cfg: Config) -> None:
    command.downgrade(alembic_cfg, "0038")
    with engine.connect() as conn:
        _seed_app_settings(
            conn, folders=["audio", "podcasts"], packs={"audio": "p1"}
        )
        under_audio = _seed_media(conn, "audio/ep1.wav")
        under_pod = _seed_media(conn, "podcasts/show/ep2.wav")
        upload = _seed_media(conn, f"incoming/{uuid.uuid4()}/source")
        conn.commit()

    command.upgrade(alembic_cfg, "0039")

    with engine.connect() as conn:
        folders = dict(
            conn.execute(text("SELECT path, domain_pack FROM media_folders")).all()
        )
        assert folders == {"audio": "p1", "podcasts": None}
        # current_path seeded from source_path for every row.
        missing = conn.execute(
            text("SELECT count(*) FROM media_items WHERE current_path IS NULL")
        ).scalar_one()
        assert missing == 0
        # Membership assigned by deepest registered-folder ancestor.
        audio_folder = conn.execute(
            text("SELECT id FROM media_folders WHERE path = 'audio'")
        ).scalar_one()
        pod_folder = conn.execute(
            text("SELECT id FROM media_folders WHERE path = 'podcasts'")
        ).scalar_one()
        rows = dict(
            conn.execute(
                text("SELECT id, media_folder_id FROM media_items")
            ).all()
        )
        assert rows[under_audio] == audio_folder
        assert rows[under_pod] == pod_folder
        assert rows[upload] is None  # outside every registered folder


def test_backfill_aborts_on_nested_pack_shadow(
    engine: Engine, alembic_cfg: Config
) -> None:
    command.downgrade(alembic_cfg, "0038")
    with engine.connect() as conn:
        # 'audio' carries the pack; the nested 'audio/pods' does not. A file under
        # 'audio/pods' resolves to 'p1' today but to the deeper (packless) folder
        # under the flat relation, so the effective pack cannot be preserved.
        _seed_app_settings(
            conn, folders=["audio", "audio/pods"], packs={"audio": "p1"}
        )
        _seed_media(conn, "audio/pods/ep.wav")
        conn.commit()

    with pytest.raises(Exception, match="effective domain pack"):
        command.upgrade(alembic_cfg, "0039")


def test_downgrade_drops_and_upgrade_restores(
    engine: Engine, alembic_cfg: Config
) -> None:
    command.downgrade(alembic_cfg, "0038")
    inspector = inspect(engine)
    for table in _TABLES:
        assert not inspector.has_table(table)
    media_cols = {c["name"] for c in inspector.get_columns("media_items")}
    assert not media_cols & set(_MEDIA_COLUMNS)
    # The legacy columns are retained, never dropped by this migration.
    app_cols = {c["name"] for c in inspector.get_columns("app_settings")}
    assert {"media_folders", "folder_domain_packs"} <= app_cols

    command.upgrade(alembic_cfg, "0039")
    inspector = inspect(engine)
    for table in _TABLES:
        assert inspector.has_table(table)
