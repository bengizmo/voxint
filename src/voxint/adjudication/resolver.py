"""Read-time speaker attribution: the one resolver the UI and exports share.

Machine proposals and human rulings are never merged in storage — this module
is where they meet, at read time:

- The **effective decision** for a (run, label) is the newest ledger row
  (``created_at DESC, id DESC`` — corrections are appends, never edits).
- A human decision always beats machine evidence. Without one, only a
  *grounded* cosine proposal counts as identity; an ``llm_hint`` name is
  surfaced as a suggestion, never as attribution.
- ``exclude`` suppresses speaker attribution, never transcript text.

A label is **unresolved** (needs adjudication) when it has neither an
effective human decision nor a grounded cosine proposal. A COMPLETED run with
at least one unresolved label is in the adjudication queue.
"""

import enum
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from voxint.db.models import (
    AdjudicationDecision,
    AssignmentMethod,
    Decision,
    DiarizationTurn,
    PipelineRun,
    RunStatus,
    Speaker,
    SpeakerAssignment,
)


class Resolution(enum.StrEnum):
    """How a label's attribution was settled — or that it wasn't."""

    HUMAN_ASSIGN = "human_assign"
    HUMAN_EXCLUDE = "human_exclude"
    HUMAN_UNKNOWN = "human_unknown"
    GROUNDED_COSINE = "grounded_cosine"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class LabelState:
    """Everything the workbench and exports need to know about one label."""

    label: str
    turn_count: int
    total_seconds: float
    resolution: Resolution
    speaker_id: uuid.UUID | None
    speaker_name: str | None
    # Machine evidence, shown as context regardless of resolution.
    cosine_speaker_id: uuid.UUID | None
    cosine_speaker_name: str | None
    cosine_confidence: float | None
    cosine_grounded: bool
    llm_hint_name: str | None
    effective_decision: AdjudicationDecision | None


def effective_decisions(
    session: Session, run_id: uuid.UUID
) -> dict[str, AdjudicationDecision]:
    """Newest ledger row per label (created_at DESC, id DESC tie-break)."""
    rows = session.execute(
        select(AdjudicationDecision)
        .where(AdjudicationDecision.pipeline_run_id == run_id)
        .order_by(
            AdjudicationDecision.diarization_label,
            AdjudicationDecision.created_at.desc(),
            AdjudicationDecision.id.desc(),
        )
    ).scalars()
    effective: dict[str, AdjudicationDecision] = {}
    for row in rows:
        effective.setdefault(row.diarization_label, row)
    return effective


def label_states(session: Session, run_id: uuid.UUID) -> list[LabelState]:
    """Resolve every diarization label of a run, in label order."""
    turn_stats = session.execute(
        select(
            DiarizationTurn.label,
            func.count().label("turns"),
            func.sum(DiarizationTurn.end_seconds - DiarizationTurn.start_seconds),
        )
        .where(DiarizationTurn.pipeline_run_id == run_id)
        .group_by(DiarizationTurn.label)
        .order_by(DiarizationTurn.label)
    ).all()
    proposals = (
        session.execute(
            select(SpeakerAssignment).where(SpeakerAssignment.pipeline_run_id == run_id)
        )
        .scalars()
        .all()
    )
    cosine_by_label = {
        p.diarization_label: p for p in proposals if p.method == AssignmentMethod.COSINE.value
    }
    hint_by_label = {
        p.diarization_label: p
        for p in proposals
        if p.method == AssignmentMethod.LLM_HINT.value
    }
    decisions = effective_decisions(session, run_id)

    speaker_ids = {p.speaker_id for p in cosine_by_label.values() if p.speaker_id} | {
        d.speaker_id for d in decisions.values() if d.speaker_id
    }
    names: dict[uuid.UUID, str] = (
        {
            sid: name
            for sid, name in session.execute(
                select(Speaker.id, Speaker.display_name).where(Speaker.id.in_(speaker_ids))
            ).tuples()
        }
        if speaker_ids
        else {}
    )

    states: list[LabelState] = []
    for label, turns, seconds in turn_stats:
        cosine = cosine_by_label.get(label)
        hint = hint_by_label.get(label)
        decision = decisions.get(label)

        speaker_id: uuid.UUID | None = None
        if decision is not None:
            if decision.decision == Decision.ASSIGN.value:
                resolution = Resolution.HUMAN_ASSIGN
                speaker_id = decision.speaker_id
            elif decision.decision == Decision.EXCLUDE.value:
                resolution = Resolution.HUMAN_EXCLUDE
            else:
                resolution = Resolution.HUMAN_UNKNOWN
        elif cosine is not None and cosine.grounded:
            resolution = Resolution.GROUNDED_COSINE
            speaker_id = cosine.speaker_id
        else:
            resolution = Resolution.UNRESOLVED

        states.append(
            LabelState(
                label=label,
                turn_count=int(turns),
                total_seconds=float(seconds or 0.0),
                resolution=resolution,
                speaker_id=speaker_id,
                speaker_name=names.get(speaker_id) if speaker_id else None,
                cosine_speaker_id=cosine.speaker_id if cosine else None,
                cosine_speaker_name=(
                    names.get(cosine.speaker_id) if cosine and cosine.speaker_id else None
                ),
                cosine_confidence=cosine.confidence if cosine else None,
                cosine_grounded=bool(cosine.grounded) if cosine else False,
                llm_hint_name=hint.proposed_name if hint else None,
                effective_decision=decision,
            )
        )
    return states


@dataclass(frozen=True)
class QueueEntry:
    run_id: uuid.UUID
    source_path: str
    unresolved_labels: int
    total_labels: int
    claimed_by: str | None


def adjudication_queue(session: Session) -> list[QueueEntry]:
    """COMPLETED runs with at least one unresolved label, oldest first.

    Exact per-run resolution at single-operator scale — correctness over a
    clever SQL reduction of the precedence rules.
    """
    now = datetime.now(tz=UTC)
    runs = session.execute(
        select(PipelineRun)
        .where(PipelineRun.status == RunStatus.COMPLETED.value)
        .order_by(PipelineRun.created_at)
    ).scalars()
    entries: list[QueueEntry] = []
    for run in runs:
        states = label_states(session, run.id)
        unresolved = sum(1 for s in states if s.resolution is Resolution.UNRESOLVED)
        if unresolved == 0:
            continue
        claim_live = (
            run.review_claim_expires_at is not None and run.review_claim_expires_at > now
        )
        entries.append(
            QueueEntry(
                run_id=run.id,
                source_path=run.media_item.source_path,
                unresolved_labels=unresolved,
                total_labels=len(states),
                claimed_by=run.review_claimed_by if claim_live else None,
            )
        )
    return entries
