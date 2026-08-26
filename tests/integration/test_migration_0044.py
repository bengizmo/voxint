"""Migration 0044 (media_operations journal), issue #155.

Real alembic up/down against the shared test database: after 0044 the
``media_operations`` and ``media_operation_files`` tables exist with their
CHECK constraints and indexes, the partial unique index enforces one active
operation per media item, ``media_items`` has ``trashed_at`` and ``purged_at``
columns, and the downgrade drops everything cleanly.
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


def _seed_media(session: Session) -> uuid.UUID:
    mid = uuid.uuid4()
    session.execute(
        text("INSERT INTO media_items (id, source_path) VALUES (:mid, :sp)"),
        {"mid": mid, "sp": f"incoming/{mid}/source"},
    )
    session.commit()
    return mid


class TestUpgrade:
    def test_tables_created(self, engine: Engine, alembic_cfg: Config) -> None:
        command.upgrade(alembic_cfg, "0044")
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        assert "media_operations" in tables
        assert "media_operation_files" in tables

    def test_media_items_columns(self, engine: Engine, alembic_cfg: Config) -> None:
        command.upgrade(alembic_cfg, "0044")
        inspector = inspect(engine)
        columns = {c["name"] for c in inspector.get_columns("media_items")}
        assert "trashed_at" in columns
        assert "purged_at" in columns

    def test_partial_unique_index_exists(
        self, engine: Engine, alembic_cfg: Config
    ) -> None:
        command.upgrade(alembic_cfg, "0044")
        inspector = inspect(engine)
        indexes = inspector.get_indexes("media_operations")
        partial_idx = [i for i in indexes if i["name"] == "uq_media_operations_active_per_item"]
        assert len(partial_idx) == 1
        assert partial_idx[0]["unique"] is True

    def test_reconciler_index_exists(
        self, engine: Engine, alembic_cfg: Config
    ) -> None:
        command.upgrade(alembic_cfg, "0044")
        inspector = inspect(engine)
        indexes = inspector.get_indexes("media_operations")
        idx = [i for i in indexes if i["name"] == "ix_media_operations_reconciler"]
        assert len(idx) == 1


class TestCheckConstraints:
    def test_valid_operation_type(self, engine: Engine, alembic_cfg: Config) -> None:
        command.upgrade(alembic_cfg, "0044")
        factory = sessionmaker(engine, expire_on_commit=False)
        with factory() as session:
            mid = _seed_media(session)
            session.execute(
                text(
                    "INSERT INTO media_operations (id, media_id, operation_type) "
                    "VALUES (:oid, :mid, 'move')"
                ),
                {"oid": uuid.uuid4(), "mid": mid},
            )
            session.commit()

    def test_invalid_operation_type_rejected(
        self, engine: Engine, alembic_cfg: Config
    ) -> None:
        command.upgrade(alembic_cfg, "0044")
        factory = sessionmaker(engine, expire_on_commit=False)
        with factory() as session:
            mid = _seed_media(session)
            with pytest.raises(IntegrityError, match="media_operations_operation_type_check"):
                session.execute(
                    text(
                        "INSERT INTO media_operations (id, media_id, operation_type) "
                        "VALUES (:oid, :mid, 'delete')"
                    ),
                    {"oid": uuid.uuid4(), "mid": mid},
                )
                session.flush()

    def test_all_valid_states(self, engine: Engine, alembic_cfg: Config) -> None:
        command.upgrade(alembic_cfg, "0044")
        factory = sessionmaker(engine, expire_on_commit=False)
        with factory() as session:
            mid = _seed_media(session)
            for state in ("planned", "fs_applied", "db_applied", "awaiting_retry",
                          "completed", "failed"):
                session.execute(
                    text(
                        "INSERT INTO media_operations (id, media_id, operation_type, state) "
                        "VALUES (:oid, :mid, 'move', :state)"
                    ),
                    {"oid": uuid.uuid4(), "mid": mid, "state": state},
                )
                session.commit()
                session.execute(
                    text("DELETE FROM media_operations WHERE media_id = :mid"),
                    {"mid": mid},
                )
                session.commit()

    def test_invalid_state_rejected(self, engine: Engine, alembic_cfg: Config) -> None:
        command.upgrade(alembic_cfg, "0044")
        factory = sessionmaker(engine, expire_on_commit=False)
        with factory() as session:
            mid = _seed_media(session)
            with pytest.raises(IntegrityError, match="media_operations_state_check"):
                session.execute(
                    text(
                        "INSERT INTO media_operations (id, media_id, operation_type, state) "
                        "VALUES (:oid, :mid, 'move', 'bogus')"
                    ),
                    {"oid": uuid.uuid4(), "mid": mid},
                )
                session.flush()

    def test_invalid_file_kind_rejected(
        self, engine: Engine, alembic_cfg: Config
    ) -> None:
        command.upgrade(alembic_cfg, "0044")
        factory = sessionmaker(engine, expire_on_commit=False)
        with factory() as session:
            mid = _seed_media(session)
            oid = uuid.uuid4()
            session.execute(
                text(
                    "INSERT INTO media_operations (id, media_id, operation_type, state) "
                    "VALUES (:oid, :mid, 'purge', 'completed')"
                ),
                {"oid": oid, "mid": mid},
            )
            session.commit()
            with pytest.raises(IntegrityError, match="media_operation_files_file_kind_check"):
                session.execute(
                    text(
                        "INSERT INTO media_operation_files "
                        "(id, operation_id, file_path, file_kind) "
                        "VALUES (:fid, :oid, 'f.wav', 'bogus')"
                    ),
                    {"fid": uuid.uuid4(), "oid": oid},
                )
                session.flush()


class TestPartialUniqueIndex:
    def test_one_active_per_item(self, engine: Engine, alembic_cfg: Config) -> None:
        command.upgrade(alembic_cfg, "0044")
        factory = sessionmaker(engine, expire_on_commit=False)
        with factory() as session:
            mid = _seed_media(session)
            session.execute(
                text(
                    "INSERT INTO media_operations (id, media_id, operation_type) "
                    "VALUES (:oid, :mid, 'move')"
                ),
                {"oid": uuid.uuid4(), "mid": mid},
            )
            session.commit()
            with pytest.raises(IntegrityError):
                session.execute(
                    text(
                        "INSERT INTO media_operations (id, media_id, operation_type) "
                        "VALUES (:oid, :mid, 'trash')"
                    ),
                    {"oid": uuid.uuid4(), "mid": mid},
                )
                session.flush()

    def test_terminal_does_not_block(
        self, engine: Engine, alembic_cfg: Config
    ) -> None:
        command.upgrade(alembic_cfg, "0044")
        factory = sessionmaker(engine, expire_on_commit=False)
        with factory() as session:
            mid = _seed_media(session)
            session.execute(
                text(
                    "INSERT INTO media_operations (id, media_id, operation_type, state) "
                    "VALUES (:oid, :mid, 'move', 'completed')"
                ),
                {"oid": uuid.uuid4(), "mid": mid},
            )
            session.commit()
            session.execute(
                text(
                    "INSERT INTO media_operations (id, media_id, operation_type) "
                    "VALUES (:oid, :mid, 'trash')"
                ),
                {"oid": uuid.uuid4(), "mid": mid},
            )
            session.commit()


class TestRawInsertBehavior:
    def test_media_item_raw_insert_without_new_columns(
        self, engine: Engine, alembic_cfg: Config
    ) -> None:
        """Raw INSERT INTO media_items (id, source_path) still works.

        Verifies that the new nullable columns do not break existing raw
        inserts that omit them (the O5 NOT NULL deferral).
        """
        command.upgrade(alembic_cfg, "0044")
        factory = sessionmaker(engine, expire_on_commit=False)
        with factory() as session:
            mid = uuid.uuid4()
            session.execute(
                text("INSERT INTO media_items (id, source_path) VALUES (:mid, :sp)"),
                {"mid": mid, "sp": f"incoming/{mid}/source"},
            )
            session.commit()
            row = session.execute(
                text(
                    "SELECT trashed_at, purged_at FROM media_items WHERE id = :mid"
                ),
                {"mid": mid},
            ).one()
            assert row.trashed_at is None
            assert row.purged_at is None


class TestDowngrade:
    def test_downgrade_drops_tables(
        self, engine: Engine, alembic_cfg: Config
    ) -> None:
        command.upgrade(alembic_cfg, "0044")
        command.downgrade(alembic_cfg, "0043")
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        assert "media_operations" not in tables
        assert "media_operation_files" not in tables

    def test_downgrade_drops_columns(
        self, engine: Engine, alembic_cfg: Config
    ) -> None:
        command.upgrade(alembic_cfg, "0044")
        command.downgrade(alembic_cfg, "0043")
        inspector = inspect(engine)
        columns = {c["name"] for c in inspector.get_columns("media_items")}
        assert "trashed_at" not in columns
        assert "purged_at" not in columns

    def test_roundtrip(self, engine: Engine, alembic_cfg: Config) -> None:
        command.upgrade(alembic_cfg, "0044")
        command.downgrade(alembic_cfg, "0043")
        command.upgrade(alembic_cfg, "0044")
        inspector = inspect(engine)
        assert "media_operations" in inspector.get_table_names()
