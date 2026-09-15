"""Contract: Project.archived_at column exists on the model (issue #477)."""

from voxint.db.models import Project


def test_project_has_archived_at_column() -> None:
    col_names = {c.name for c in Project.__table__.columns}
    assert "archived_at" in col_names
