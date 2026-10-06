"""The clean-up variant's storage invariants stay declared in one place (#758)."""

import re
from pathlib import Path
from typing import cast

import pytest
from sqlalchemy import CheckConstraint, Table

from voxint.db.models import CleanupJob, CleanupJobStatus, RunCleanup

ROOT = Path(__file__).resolve().parents[2]
MIGRATION = ROOT / "alembic/versions/0073_transcript_cleanup.py"
MODELS = ROOT / "src/voxint/db/models.py"


def _checks(model: type[RunCleanup] | type[CleanupJob]) -> set[str]:
    return {
        str(constraint.name)
        for constraint in cast(Table, model.__table__).constraints
        if isinstance(constraint, CheckConstraint)
    }


@pytest.mark.parametrize("model", [RunCleanup, CleanupJob])
def test_named_checks_match_migration_and_model(
    model: type[RunCleanup] | type[CleanupJob],
) -> None:
    migration = MIGRATION.read_text()
    checks = _checks(model)
    assert checks
    for name in checks:
        assert f'name="{name}"' in migration, name
    declared = {
        line.split('name="')[1].split('"')[0]
        for line in migration.splitlines()
        if 'name="' in line and model.__tablename__ in line and "_check" in line
    }
    assert declared == checks


def test_job_status_enum_matches_the_migration_check() -> None:
    values = ", ".join(f"'{status.value}'" for status in CleanupJobStatus)
    assert f"status IN ({values})" in MIGRATION.read_text()


def test_immutability_trigger_is_declared_and_documented() -> None:
    migration = MIGRATION.read_text()
    assert "CREATE TRIGGER run_cleanups_content_immutable_trigger" in migration
    assert "BEFORE INSERT OR UPDATE OR DELETE ON run_cleanups" in migration
    assert "run_cleanups_content_immutable_trigger" in MODELS.read_text()
    # Every content column is frozen by the trigger; only the stamp may change.
    columns = {column.name for column in RunCleanup.__table__.columns}
    for column in columns - {"superseded_by_cleanup_id"}:
        assert f"NEW.{column} IS DISTINCT FROM OLD.{column}" in migration, column


def test_record_cleanup_is_the_only_writer() -> None:
    writers = [
        path.relative_to(ROOT)
        for path in (ROOT / "src/voxint").rglob("*.py")
        if re.search(r"(?<!class )\bRunCleanup\(", path.read_text())
    ]
    assert writers == [Path("src/voxint/enrichment/cleanups.py")]
