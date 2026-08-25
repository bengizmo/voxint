"""Read-only queries behind the projects pages (Console 2.0 P2b, issue #153).

Same shape as :mod:`voxint.api.home_query` and :mod:`voxint.api.media_query`:
frozen dataclasses plus functions that take a :class:`~sqlalchemy.orm.Session`
and issue bounded ``SELECT``s, no HTTP and no side effects.

A project's speakers are DERIVED, not stored: they are the distinct speakers its
runs resolve to, preferring a human adjudication over a grounded cosine match.
That precedence already lives in :func:`voxint.adjudication.resolver.label_states`
(the one resolver the workbench and exports share), so this module walks the
project's completed runs and reuses it rather than re-deriving the rule. The walk
is bounded by the project's run count (a single-operator, modest-project tool),
one ``label_states`` call per completed run.
"""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import func
from sqlalchemy import select as sa_select
from sqlalchemy.orm import Session

from voxint.adjudication.resolver import label_states
from voxint.db.models import (
    MediaFolder,
    MediaItem,
    PipelineRun,
    Project,
    RunStatus,
)


@dataclass(frozen=True)
class ProjectSummary:
    """One row on the projects list: identity plus how many folders it owns."""

    id: uuid.UUID
    name: str
    description: str | None
    folder_count: int
    created_at: datetime


@dataclass(frozen=True)
class ProjectFolder:
    """A folder that belongs to a project, with its pack and media count."""

    id: uuid.UUID
    path: str
    domain_pack: str | None
    media_count: int


@dataclass(frozen=True)
class ProjectSpeaker:
    """A speaker derived from a project's runs, and how many runs they appear in."""

    id: uuid.UUID
    name: str | None
    run_count: int


@dataclass(frozen=True)
class AssignableFolder:
    """An unassigned folder (``project_id IS NULL``) offered by the assign form."""

    id: uuid.UUID
    path: str
    domain_pack: str | None


@dataclass(frozen=True)
class ProjectDetail:
    """Everything the project detail page renders."""

    id: uuid.UUID
    name: str
    description: str | None
    # A project with its own vocabulary or corrections overrides its folders'
    # packs (ADR 0002 per-field replacement). No editor exists yet (P2c), so this
    # is False today; the assign note reads differently once it can be True.
    has_own_config: bool
    folders: list[ProjectFolder]
    speakers: list[ProjectSpeaker]
    assignable: list[AssignableFolder]


def list_projects(session: Session) -> list[ProjectSummary]:
    """Every project in name order, each with its folder count (one query)."""
    rows = session.execute(
        sa_select(
            Project.id,
            Project.name,
            Project.description,
            Project.created_at,
            func.count(MediaFolder.id).label("folder_count"),
        )
        .outerjoin(MediaFolder, MediaFolder.project_id == Project.id)
        .group_by(Project.id)
        .order_by(func.lower(Project.name), Project.id)
    )
    return [
        ProjectSummary(
            id=row.id,
            name=row.name,
            description=row.description,
            folder_count=row.folder_count,
            created_at=row.created_at.astimezone(UTC),
        )
        for row in rows
    ]


def _member_folders(session: Session, project_id: uuid.UUID) -> list[ProjectFolder]:
    rows = session.execute(
        sa_select(
            MediaFolder.id,
            MediaFolder.path,
            MediaFolder.domain_pack,
            func.count(MediaItem.id).label("media_count"),
        )
        .outerjoin(MediaItem, MediaItem.media_folder_id == MediaFolder.id)
        .where(MediaFolder.project_id == project_id)
        .group_by(MediaFolder.id)
        .order_by(MediaFolder.path)
    )
    return [
        ProjectFolder(
            id=row.id,
            path=row.path,
            domain_pack=row.domain_pack,
            media_count=row.media_count,
        )
        for row in rows
    ]


def _derived_speakers(session: Session, project_id: uuid.UUID) -> list[ProjectSpeaker]:
    """Distinct speakers across the project's completed runs, human over grounded.

    One ``label_states`` call per completed, non-archived run (it already applies
    the human-decision-over-grounded-cosine precedence and canonicalizes merged
    speakers), counting each speaker once per run they resolve a label in.
    """
    run_ids = (
        session.execute(
            sa_select(PipelineRun.id)
            .join(MediaItem, MediaItem.id == PipelineRun.media_item_id)
            .join(MediaFolder, MediaFolder.id == MediaItem.media_folder_id)
            .where(MediaFolder.project_id == project_id)
            .where(PipelineRun.status == RunStatus.COMPLETED.value)
            .where(PipelineRun.archived_at.is_(None))
        )
        .scalars()
        .all()
    )
    # id -> (name, run_count). A speaker counts once per run regardless of how
    # many of its labels they hold.
    tally: dict[uuid.UUID, tuple[str | None, int]] = {}
    for run_id in run_ids:
        seen: set[uuid.UUID] = set()
        for state in label_states(session, run_id):
            sid = state.speaker_id
            if sid is None or sid in seen:
                continue
            seen.add(sid)
            name, count = tally.get(sid, (state.speaker_name, 0))
            tally[sid] = (state.speaker_name or name, count + 1)
    speakers = [
        ProjectSpeaker(id=sid, name=name, run_count=count)
        for sid, (name, count) in tally.items()
    ]
    # Name order, with unnamed speakers last, then by id for a stable tiebreak.
    speakers.sort(key=lambda s: (s.name is None, (s.name or "").lower(), str(s.id)))
    return speakers


def _assignable_folders(session: Session) -> list[AssignableFolder]:
    rows = session.execute(
        sa_select(MediaFolder.id, MediaFolder.path, MediaFolder.domain_pack)
        .where(MediaFolder.project_id.is_(None))
        .order_by(MediaFolder.path)
    )
    return [
        AssignableFolder(id=row.id, path=row.path, domain_pack=row.domain_pack)
        for row in rows
    ]


def project_detail(
    session: Session, project_id: uuid.UUID
) -> ProjectDetail | None:
    """The detail page's data, or ``None`` when the project does not exist."""
    project = session.get(Project, project_id)
    if project is None:
        return None
    return ProjectDetail(
        id=project.id,
        name=project.name,
        description=project.description,
        has_own_config=(
            project.vocabulary is not None or project.corrections is not None
        ),
        folders=_member_folders(session, project_id),
        speakers=_derived_speakers(session, project_id),
        assignable=_assignable_folders(session),
    )
