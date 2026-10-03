"""Recording date schema and metadata boundaries (#741)."""

from pathlib import Path

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


def test_recorded_date_tag() -> None:
    """Only Apple's offset-bearing tag is consulted."""
    assert recorded_date.CREATION_DATE_TAG == "com.apple.quicktime.creationdate"
    assert recorded_date.__file__ is not None
    assert "creation_time" not in Path(recorded_date.__file__).read_text()


def test_recorded_date_sidecar_key() -> None:
    """Recording date is an applied sidecar key."""
    assert "recorded" in sidecar._APPLIED_KEYS
