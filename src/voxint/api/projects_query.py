"""Read-only queries behind the projects pages (Console 2.0 P2b, issue #153).

Same shape as :mod:`voxint.api.home_query` and :mod:`voxint.api.media_query`:
frozen dataclasses plus functions that take a :class:`~sqlalchemy.orm.Session`
and issue bounded ``SELECT``s, no HTTP and no side effects.

A project's speakers are DERIVED, not stored: they are the distinct speakers its
runs resolve to, preferring a human adjudication over a grounded cosine match.
That precedence already lives in :func:`voxint.adjudication.resolver.label_states`
(the one resolver the workbench and exports share), so this module walks the
project's canonical runs and reuses it rather than re-deriving the rule. Canonical
means the newest completed, non-archived run per media item, so a re-run recording
counts once, not once per historical run. The walk is bounded by the project's
recording count (a single-operator, modest-project tool), one ``label_states``
call per canonical run — a single pass that feeds both the speaker list and the
speakers-by-recordings coverage matrix (issue #336).
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
    MediaSourceMetadata,
    PipelineRun,
    Project,
    RunStatus,
)

# The coverage matrix stays scannable: at most this many recording columns,
# newest first, with the count of older recordings disclosed instead of drawn.
_MATRIX_MAX_COLUMNS = 20


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
class ProjectRecording:
    """One canonical recording in the project: its media item and newest run."""

    media_id: uuid.UUID
    run_id: uuid.UUID
    title: str
    created_at: datetime


@dataclass(frozen=True)
class MatrixRow:
    """One speaker's presence across the matrix's recording columns."""

    speaker_id: uuid.UUID
    name: str | None
    cells: tuple[bool, ...]


@dataclass(frozen=True)
class SpeakerMatrix:
    """Speakers-by-recordings coverage: dot = the speaker appears in it."""

    columns: tuple[ProjectRecording, ...]
    rows: tuple[MatrixRow, ...]
    omitted_recordings: int


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
    # The project's OWN config overrides (ADR 0002 per-field replacement). Each is
    # nullable: None = inherit the folder pack / global baseline, [] = explicitly
    # none (wins). The editors on the detail page write these directly.
    vocabulary: list[str] | None
    corrections: list[dict[str, object]] | None
    # Per-field: the project overrides this field (not None) and so replaces the
    # folder pack / global baseline for it. Resolution is per field (ADR 0002), so
    # the supersede copy must name the fields actually overridden, not blanket both.
    has_own_vocabulary: bool
    has_own_corrections: bool
    # True when either override is set: the project has some config of its own.
    has_own_config: bool
    folders: list[ProjectFolder]
    speakers: list[ProjectSpeaker]
    matrix: SpeakerMatrix
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


def canonical_project_recordings(
    session: Session, project_id: uuid.UUID
) -> list[ProjectRecording]:
    """The newest completed, non-archived run per media item, newest media first.

    Same canonical-run rule as :func:`voxint.api.speaker_insights._canonical_runs`,
    scoped to the project's folders. Titles prefer the scraped source title over
    the raw source path, matching the Explore page.
    """
    ranked = (
        sa_select(
            PipelineRun.id.label("run_id"),
            PipelineRun.media_item_id.label("media_id"),
            func.row_number()
            .over(
                partition_by=PipelineRun.media_item_id,
                order_by=(PipelineRun.created_at.desc(), PipelineRun.id.desc()),
            )
            .label("rank"),
        )
        .join(MediaItem, MediaItem.id == PipelineRun.media_item_id)
        .join(MediaFolder, MediaFolder.id == MediaItem.media_folder_id)
        .where(MediaFolder.project_id == project_id)
        .where(PipelineRun.status == RunStatus.COMPLETED.value)
        .where(PipelineRun.archived_at.is_(None))
        .subquery()
    )
    rows = session.execute(
        sa_select(
            ranked.c.run_id,
            ranked.c.media_id,
            func.coalesce(MediaSourceMetadata.title, MediaItem.source_path).label("title"),
            MediaItem.created_at,
        )
        .join(MediaItem, MediaItem.id == ranked.c.media_id)
        .outerjoin(
            MediaSourceMetadata,
            MediaSourceMetadata.media_item_id == MediaItem.id,
        )
        .where(ranked.c.rank == 1)
        .order_by(MediaItem.created_at.desc(), MediaItem.id.desc())
    ).all()
    return [
        ProjectRecording(
            media_id=row.media_id,
            run_id=row.run_id,
            title=row.title,
            created_at=row.created_at.astimezone(UTC),
        )
        for row in rows
    ]


def _speakers_and_matrix(
    session: Session, recordings: list[ProjectRecording]
) -> tuple[list[ProjectSpeaker], SpeakerMatrix]:
    """One ``label_states`` pass feeding both the speaker list and the matrix.

    ``label_states`` already applies the human-decision-over-grounded-cosine
    precedence and canonicalizes merged speakers; a speaker counts once per
    recording regardless of how many of its labels they hold.
    """
    # speaker id -> (name, set of recording indexes they appear in).
    tally: dict[uuid.UUID, tuple[str | None, set[int]]] = {}
    for index, recording in enumerate(recordings):
        for state in label_states(session, recording.run_id):
            sid = state.speaker_id
            if sid is None:
                continue
            name, present = tally.get(sid, (state.speaker_name, set()))
            present.add(index)
            tally[sid] = (state.speaker_name or name, present)
    speakers = [
        ProjectSpeaker(id=sid, name=name, run_count=len(present))
        for sid, (name, present) in tally.items()
    ]
    # Name order, with unnamed speakers last, then by id for a stable tiebreak.
    speakers.sort(key=lambda s: (s.name is None, (s.name or "").lower(), str(s.id)))

    columns = tuple(recordings[:_MATRIX_MAX_COLUMNS])
    rows = tuple(
        MatrixRow(
            speaker_id=speaker.id,
            name=speaker.name,
            cells=tuple(
                index in tally[speaker.id][1] for index in range(len(columns))
            ),
        )
        for speaker in speakers
    )
    matrix = SpeakerMatrix(
        columns=columns,
        rows=rows,
        omitted_recordings=max(0, len(recordings) - len(columns)),
    )
    return speakers, matrix


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
    vocabulary = list(project.vocabulary) if project.vocabulary is not None else None
    corrections = (
        [dict(rule) for rule in project.corrections]
        if project.corrections is not None
        else None
    )
    recordings = canonical_project_recordings(session, project_id)
    speakers, matrix = _speakers_and_matrix(session, recordings)
    return ProjectDetail(
        id=project.id,
        name=project.name,
        description=project.description,
        vocabulary=vocabulary,
        corrections=corrections,
        has_own_vocabulary=vocabulary is not None,
        has_own_corrections=corrections is not None,
        has_own_config=(vocabulary is not None or corrections is not None),
        folders=_member_folders(session, project_id),
        speakers=speakers,
        matrix=matrix,
        assignable=_assignable_folders(session),
    )
