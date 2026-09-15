"""Project archive and restore lifecycle (issue #477).

Service-layer operations for the project lifecycle, independent of HTTP.
Mirrors :mod:`voxint.speakers.roster` (archive/restore with FOR UPDATE,
idempotent, flush not commit).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from voxint.db.models import Project


class ProjectNotFoundError(Exception):
    pass


class ProjectArchivedError(Exception):
    pass


def archive_project(session: Session, project_id: uuid.UUID) -> Project:
    """Stamp ``archived_at``. Idempotent: returns unchanged if already archived."""
    project = session.execute(
        select(Project).where(Project.id == project_id).with_for_update()
    ).scalar_one_or_none()
    if project is None:
        raise ProjectNotFoundError(f"no project {project_id}")
    if project.archived_at is not None:
        return project
    project.archived_at = datetime.now(tz=UTC)
    session.flush()
    return project


def restore_project(session: Session, project_id: uuid.UUID) -> Project:
    """Clear ``archived_at``. Idempotent: returns unchanged if already active."""
    project = session.execute(
        select(Project).where(Project.id == project_id).with_for_update()
    ).scalar_one_or_none()
    if project is None:
        raise ProjectNotFoundError(f"no project {project_id}")
    if project.archived_at is None:
        return project
    project.archived_at = None
    session.flush()
    return project


def require_active_project(session: Session, project_id: uuid.UUID) -> Project:
    """Return the project or raise ``ProjectNotFoundError`` / ``ProjectArchivedError``.

    No FOR UPDATE: this is a guard, not a mutation. Routes that mutate the
    project row hold their own transactional safety (begin_nested or simple
    attribute write). Locking here would create a lock-order inversion with
    observe_segment_edit, which locks evidence rows before the project row.
    """
    project = session.get(Project, project_id)
    if project is None:
        raise ProjectNotFoundError(f"no project {project_id}")
    if project.archived_at is not None:
        raise ProjectArchivedError(f"project {project_id} is archived")
    return project


def describe_project_name_owner(owner: Project) -> str:
    if owner.archived_at is not None:
        return (
            f"A project named “{owner.name}” is archived. "
            "Restore it instead of creating it again."
        )
    return f"A project named “{owner.name}” already exists."
