"""Recording date schema and metadata boundaries (#741)."""

import json
import subprocess
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.script import ScriptDirectory

from voxint.db.models import MediaItem
from voxint.ingest import sidecar
from voxint.media import recorded_date


def test_recorded_date_migration_head() -> None:
    """Recording dates extend the 0068 migration chain."""
    scripts = ScriptDirectory(str(Path(__file__).resolve().parents[2] / "alembic"))
    revision = scripts.get_revision("0069")
    assert revision is not None and revision.down_revision == "0068"


def test_recorded_date_column() -> None:
    """Unknown recording dates remain nullable SQL dates."""
    column = MediaItem.__table__.c.recorded_on
    assert isinstance(column.type, sa.Date)
    assert column.nullable


def test_recorded_date_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only Apple's offset-bearing tag is consulted; UTC-only creation_time never is."""
    assert recorded_date.CREATION_DATE_TAG == "com.apple.quicktime.creationdate"
    seen: list[list[str]] = []

    def run(cmd: list[str], *, timeout_seconds: float) -> subprocess.CompletedProcess[str]:
        seen.append(cmd)
        tags = {"creation_time": "2026-10-03T05:30:00-0600"}
        return subprocess.CompletedProcess(cmd, 0, json.dumps({"format": {"tags": tags}}), "")

    monkeypatch.setattr(recorded_date, "_run", run)
    assert recorded_date.probe_recorded_on(Path("source.m4a")) is None
    assert "format_tags=com.apple.quicktime.creationdate" in seen[0]


def test_recorded_date_sidecar_key() -> None:
    """Recording date is an applied sidecar key."""
    assert "recorded" in sidecar._APPLIED_KEYS
