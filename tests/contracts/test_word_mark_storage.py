"""Clean-up marks stay independent of the attribution ledger and its readers."""

import ast
from pathlib import Path

import pytest
from sqlalchemy import CheckConstraint

from voxint.db.models import SegmentWordMark

ROOT = Path(__file__).resolve().parents[2]


def test_attribution_readers_never_reference_word_marks() -> None:
    adjudication = ROOT / "src/voxint/adjudication"
    readers = {adjudication / f"{name}.py" for name in ("resolver", "undo", "naming", "ledger")}
    readers.update(
        path
        for path in adjudication.glob("*.py")
        if "AdjudicationDecision" in path.read_text() or "newest_in_scope" in path.read_text()
    )
    # Restart impact deliberately loads the overlay separately from attribution counts.
    readers.add(ROOT / "src/voxint/ingest/service.py")
    for path in sorted(readers):
        source = path.read_text()
        assert "segment_word_marks" not in source, path
        assert "SegmentWordMark" not in source, path


@pytest.mark.parametrize("module", ["fillers", "turn_filters"])
def test_filter_types_do_not_import_word_mark_writers(module: str) -> None:
    tree = ast.parse((ROOT / f"src/voxint/export/{module}.py").read_text())
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert "voxint.adjudication.turns" in imports
    assert all(
        f"voxint.adjudication.{name}" not in imports for name in ("word_marks", "ledger", "undo")
    )


@pytest.mark.parametrize(
    "name",
    [
        "segment_word_marks_word_range_bounds_check",
        "segment_word_marks_action_check",
        "segment_word_marks_undo_check",
        "segment_word_marks_no_self_void_check",
        "segment_word_marks_user_not_system_check",
        "segment_word_marks_system_not_user_check",
    ],
)
def test_named_checks_match_migration_and_model(name: str) -> None:
    checks = {
        constraint.name
        for constraint in SegmentWordMark.__table__.constraints
        if isinstance(constraint, CheckConstraint)
    }
    assert name in checks
    assert name in (ROOT / "alembic/versions/0072_segment_word_marks.py").read_text()
    assert name in (ROOT / "src/voxint/db/models.py").read_text()


def test_append_only_trigger_is_declared_and_documented() -> None:
    migration = (ROOT / "alembic/versions/0072_segment_word_marks.py").read_text()
    assert "CREATE TRIGGER segment_word_marks_append_only_trigger" in migration
    assert "BEFORE UPDATE OR DELETE ON segment_word_marks" in migration
    assert "EXECUTE FUNCTION segment_word_marks_append_only()" in migration
    assert (
        "segment_word_marks_append_only_trigger" in (ROOT / "src/voxint/db/models.py").read_text()
    )
    assert "pg_trigger_depth" not in migration
    assert "TRUNCATE" not in migration
