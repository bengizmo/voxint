"""reconcile_orphaned_incoming removes files with no matching MediaItem (M4)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from voxint.ingest.service import reconcile_orphaned_incoming


def _session_all_orphaned() -> MagicMock:
    """Mock session: no DB rows match any path."""
    session = MagicMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    session.execute.return_value = result
    return session


def _session_all_known() -> MagicMock:
    """Mock session: every path has a matching DB row."""
    session = MagicMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = 1
    session.execute.return_value = result
    return session


def test_removes_orphaned_files(tmp_path: Path) -> None:
    incoming = tmp_path / "incoming" / "abc123"
    incoming.mkdir(parents=True)
    orphan = incoming / "test.wav"
    orphan.write_bytes(b"\x00" * 100)

    session = _session_all_orphaned()
    removed = reconcile_orphaned_incoming(session, tmp_path)

    assert "incoming/abc123/test.wav" in removed
    assert not orphan.exists()
    assert not incoming.exists()


def test_preserves_files_with_db_rows(tmp_path: Path) -> None:
    incoming = tmp_path / "incoming" / "def456"
    incoming.mkdir(parents=True)
    kept = incoming / "audio.wav"
    kept.write_bytes(b"\x00" * 100)

    session = _session_all_known()
    removed = reconcile_orphaned_incoming(session, tmp_path)

    assert removed == []
    assert kept.exists()


def test_no_incoming_dir_is_noop(tmp_path: Path) -> None:
    session = MagicMock()
    removed = reconcile_orphaned_incoming(session, tmp_path)
    assert removed == []


def test_empty_incoming_dir(tmp_path: Path) -> None:
    (tmp_path / "incoming").mkdir()
    session = MagicMock()
    removed = reconcile_orphaned_incoming(session, tmp_path)
    assert removed == []
