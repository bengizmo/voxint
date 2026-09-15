"""Observation and lifecycle for learned correction suggestions (#476).

Called from the edit route when an operator changes a segment's corrected text.
Records substitution evidence per-segment, retracting on re-edit, and manages
the accept/dismiss lifecycle. A learning bug is logged but never loses the edit.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from voxint.adjudication.transcript import effective_text
from voxint.db.models import (
    LearnedCorrection,
    LearnedCorrectionEvidence,
    MediaFolder,
    MediaItem,
    PipelineRun,
    Project,
    SegmentReviewState,
    TranscriptSegment,
)
from voxint.domain_packs.corrections import normalize_operator_corrections
from voxint.domain_packs.learning import (
    MAX_SUGGESTIONS_PER_PROJECT,
    Substitution,
    extract_substitutions,
)

logger = logging.getLogger(__name__)


def _resolve_project(session: Session, segment: TranscriptSegment) -> Project | None:
    """Walk segment -> run -> media_item -> folder -> active project (#477)."""
    row = session.execute(
        select(Project)
        .join(MediaFolder, MediaFolder.project_id == Project.id)
        .join(MediaItem, MediaItem.media_folder_id == MediaFolder.id)
        .join(PipelineRun, PipelineRun.media_item_id == MediaItem.id)
        .where(PipelineRun.id == segment.pipeline_run_id)
        .where(Project.archived_at.is_(None))
    ).scalar_one_or_none()
    return row


def _retract_evidence(session: Session, segment_id: uuid.UUID) -> None:
    """Remove all evidence from this segment and prune orphaned suggested rows."""
    session.execute(
        delete(LearnedCorrectionEvidence).where(
            LearnedCorrectionEvidence.segment_id == segment_id
        )
    )
    session.flush()
    orphans = (
        select(LearnedCorrection.id)
        .where(
            LearnedCorrection.status == "suggested",
            ~select(LearnedCorrectionEvidence.learned_correction_id)
            .where(
                LearnedCorrectionEvidence.learned_correction_id
                == LearnedCorrection.id
            )
            .correlate(LearnedCorrection)
            .exists(),
        )
    )
    session.execute(
        delete(LearnedCorrection).where(LearnedCorrection.id.in_(orphans))
    )
    session.flush()


def _suggested_count(session: Session, project_id: uuid.UUID) -> int:
    return session.execute(
        select(func.count())
        .select_from(LearnedCorrection)
        .where(
            LearnedCorrection.project_id == project_id,
            LearnedCorrection.status == "suggested",
        )
    ).scalar_one()


def _record_pair(
    session: Session,
    project_id: uuid.UUID,
    sub: Substitution,
    segment_id: uuid.UUID,
    suggested_count: int,
) -> int:
    """Upsert a learned_corrections row and record evidence. Returns new suggested count."""
    existing = session.execute(
        select(LearnedCorrection).where(
            LearnedCorrection.project_id == project_id,
            LearnedCorrection.match == sub.match,
            LearnedCorrection.replace == sub.replace,
        )
    ).scalar_one_or_none()

    if existing is not None:
        lc_id = existing.id
    elif suggested_count >= MAX_SUGGESTIONS_PER_PROJECT:
        return suggested_count
    else:
        lc_id = uuid.uuid4()
        session.execute(
            pg_insert(LearnedCorrection.__table__)  # type: ignore[arg-type]
            .values(
                id=lc_id,
                project_id=project_id,
                match=sub.match,
                replace=sub.replace,
                status="suggested",
            )
            .on_conflict_do_nothing(
                constraint="learned_corrections_project_match_replace_key"
            )
        )
        session.flush()
        row = session.execute(
            select(LearnedCorrection.id).where(
                LearnedCorrection.project_id == project_id,
                LearnedCorrection.match == sub.match,
                LearnedCorrection.replace == sub.replace,
            )
        ).scalar_one_or_none()
        if row is None:
            return suggested_count
        lc_id = row
        suggested_count += 1

    session.execute(
        pg_insert(LearnedCorrectionEvidence.__table__)  # type: ignore[arg-type]
        .values(learned_correction_id=lc_id, segment_id=segment_id)
        .on_conflict_do_nothing()
    )
    session.flush()
    return suggested_count


def observe_segment_edit(session: Session, segment: TranscriptSegment) -> None:
    """Called after set_correction changed the stored corrected text.

    Wrapped in a savepoint so a learning bug never loses the operator's edit.
    """
    try:
        with session.begin_nested():
            _retract_evidence(session, segment.id)

            project = _resolve_project(session, segment)
            if project is None:
                return
            if not project.learn_corrections:
                return
            if project.corrections is None:
                return

            # Serialize per-project to prevent concurrent-run races on
            # suggestion creation, orphan cleanup, and the 256-row cap.
            # The archived predicate closes the resolve-then-lock window (#477).
            locked = session.execute(
                select(Project.id)
                .where(Project.id == project.id, Project.archived_at.is_(None))
                .with_for_update()
            ).scalar_one_or_none()
            if locked is None:
                return

            review_state = session.get(SegmentReviewState, segment.id)
            corrected_text = review_state.corrected_text if review_state else None
            if corrected_text is None:
                return

            base = effective_text(segment, None)
            raw = segment.raw_text

            subs = extract_substitutions(base, corrected_text, raw=raw)
            if not subs:
                return

            count = _suggested_count(session, project.id)
            for sub in subs:
                count = _record_pair(session, project.id, sub, segment.id, count)
    except Exception:
        logger.exception("learned-corrections observation failed for segment %s", segment.id)


def accept_suggestion(
    session: Session, project: Project, learned: LearnedCorrection
) -> list[dict[str, Any]]:
    """Accept a suggestion: append it as a validated rule to the project's list.

    Returns the new corrections list. Raises OperatorCorrectionError on
    validation failure (caller renders the error).
    """
    new_rule = {
        "id": "",
        "match": learned.match,
        "replace": learned.replace,
        "case_sensitive": True,
        "whole_word": True,
    }
    current = list(project.corrections or [])
    current.append(new_rule)
    normalized = normalize_operator_corrections(current)
    project.corrections = normalized
    learned.status = "accepted"
    learned.accepted_rule_id = normalized[-1]["id"]
    session.flush()
    return project.corrections


def dismiss_suggestion(session: Session, learned: LearnedCorrection) -> None:
    """Dismiss a suggestion: delete it entirely."""
    session.delete(learned)
    session.flush()
