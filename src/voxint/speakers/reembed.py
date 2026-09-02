"""In-place migration of stored voice vectors to the embedder's current space."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from voxint.clients.base import EmbedderClient, EmbeddingResult
from voxint.db.models import (
    AssignmentMethod,
    DiarizationTurn,
    PipelineRun,
    RunStatus,
    SpeakerAssignment,
    SpeakerEmbedding,
)
from voxint.pipeline.stages.context import StageDataError, normalized_audio_path
from voxint.speakers.matching import (
    MatchingGates,
    NameHintProposal,
    eligible_label_vectors,
    evaluate_run,
    label_centroid,
    replace_run_match_candidates,
    replace_run_proposals,
)


class EmbeddingSpaceDriftError(RuntimeError):
    """The service changed vector spaces during one migration invocation."""


@dataclass(frozen=True)
class ReembedPlan:
    target_space: str
    run_ids: tuple[uuid.UUID, ...]
    turn_count: int
    enrollment_count: int
    unmigratable_count: int


@dataclass(frozen=True)
class RunReembedResult:
    turns: int
    enrollments: int
    stale_enrollment_ids: tuple[uuid.UUID, ...]


def candidate_run_ids(session: Session, run_id: uuid.UUID | None = None) -> tuple[uuid.UUID, ...]:
    """Completed runs with at least one stored voice vector, for fallback discovery."""
    query = (
        select(PipelineRun.id)
        .where(
            PipelineRun.status == RunStatus.COMPLETED.value,
            select(DiarizationTurn.id)
            .where(
                DiarizationTurn.pipeline_run_id == PipelineRun.id,
                DiarizationTurn.embedding.is_not(None),
            )
            .correlate(PipelineRun)
            .exists(),
        )
        .order_by(PipelineRun.id)
    )
    if run_id is not None:
        query = query.where(PipelineRun.id == run_id)
    return tuple(session.scalars(query))


def embed_run_turns(
    session: Session,
    run_id: uuid.UUID,
    embedder: EmbedderClient,
    media_root: Path,
) -> tuple[list[DiarizationTurn], EmbeddingResult]:
    """Embed every existing turn in stable turn order without writing it."""
    turns = list(
        session.scalars(
            select(DiarizationTurn)
            .where(DiarizationTurn.pipeline_run_id == run_id)
            .order_by(DiarizationTurn.turn_index)
        )
    )
    if not turns:
        raise StageDataError(f"run {run_id}: expected at least one diarization turn")
    audio = normalized_audio_path(session, run_id, media_root)
    result = embedder.embed(
        audio, tuple((turn.start_seconds, turn.end_seconds) for turn in turns)
    )
    if len(result.entries) != len(turns):
        raise StageDataError(
            f"run {run_id}: {len(result.entries)} embedding entries for {len(turns)} turns"
        )
    return turns, result


def discover_space_from_run(
    session: Session,
    run_id: uuid.UUID,
    embedder: EmbedderClient,
    media_root: Path,
) -> str:
    """Discover the service space from a real run when health omits it."""
    _turns, result = embed_run_turns(session, run_id, embedder, media_root)
    return result.embedding_space


def build_plan(
    session: Session, target_space: str, run_id: uuid.UUID | None = None
) -> ReembedPlan:
    """Describe stale completed-run work without changing the database."""
    stale_turns = (
        select(DiarizationTurn.id)
        .where(
            DiarizationTurn.pipeline_run_id == PipelineRun.id,
            DiarizationTurn.embedding.is_not(None),
            DiarizationTurn.embedding_space != target_space,
        )
        .correlate(PipelineRun)
        .exists()
    )
    stale_enrollments = (
        select(SpeakerEmbedding.id)
        .where(
            SpeakerEmbedding.source_pipeline_run_id == PipelineRun.id,
            SpeakerEmbedding.embedding_space != target_space,
        )
        .correlate(PipelineRun)
        .exists()
    )
    runs_query = (
        select(PipelineRun.id)
        .where(
            PipelineRun.status == RunStatus.COMPLETED.value,
            stale_turns | stale_enrollments,
        )
        .order_by(PipelineRun.id)
    )
    if run_id is not None:
        runs_query = runs_query.where(PipelineRun.id == run_id)
    run_ids = tuple(session.scalars(runs_query))
    turn_count = 0
    enrollment_count = 0
    if run_ids:
        turn_count = session.scalar(
            select(func.count(DiarizationTurn.id)).where(
                DiarizationTurn.pipeline_run_id.in_(run_ids),
            )
        ) or 0
        enrollment_count = session.scalar(
            select(func.count(SpeakerEmbedding.id)).where(
                SpeakerEmbedding.source_pipeline_run_id.in_(run_ids),
                SpeakerEmbedding.embedding_space != target_space,
            )
        ) or 0
    unmigratable_count = session.scalar(
        select(func.count(SpeakerEmbedding.id)).where(
            SpeakerEmbedding.source_pipeline_run_id.is_(None),
            SpeakerEmbedding.embedding_space != target_space,
        )
    ) or 0
    return ReembedPlan(
        target_space=target_space,
        run_ids=run_ids,
        turn_count=turn_count,
        enrollment_count=enrollment_count,
        unmigratable_count=unmigratable_count,
    )


def reembed_run(
    session: Session,
    run_id: uuid.UUID,
    target_space: str,
    embedder: EmbedderClient,
    media_root: Path,
    gates: MatchingGates,
) -> RunReembedResult:
    """Migrate one run. The caller owns the surrounding transaction."""
    turns, result = embed_run_turns(session, run_id, embedder, media_root)
    if result.embedding_space != target_space:
        raise EmbeddingSpaceDriftError(
            f"run {run_id}: embedding_space changed from {target_space!r} "
            f"to {result.embedding_space!r}"
        )
    for turn, entry in zip(turns, result.entries, strict=True):
        turn.embedding = list(entry.embedding) if entry.embedding is not None else None
        turn.embedding_space = target_space if entry.embedding is not None else None
        turn.snr_db = entry.snr_db
        turn.skip_reason = entry.skip_reason
    session.flush()

    enrollment_rows = list(
        session.scalars(
            select(SpeakerEmbedding)
            .where(
                SpeakerEmbedding.source_pipeline_run_id == run_id,
                SpeakerEmbedding.embedding_space != target_space,
            )
            .order_by(SpeakerEmbedding.id)
        )
    )
    by_label = eligible_label_vectors(session, run_id, gates)
    migrated = 0
    stale: list[uuid.UUID] = []
    for row in enrollment_rows:
        label = row.source_diarization_label
        label_data = by_label.get(label) if label is not None else None
        if label_data is None:
            stale.append(row.id)
            continue
        space, vectors = label_data
        vector = label_centroid(vectors, gates.turn_weight_cap_seconds)
        if vector is None:
            stale.append(row.id)
            continue
        if space != target_space:
            raise EmbeddingSpaceDriftError(
                f"run {run_id} label {label!r}: derived unexpected space {space!r}"
            )
        row.embedding = vector
        row.embedding_space = target_space
        migrated += 1
    session.flush()

    return RunReembedResult(len(turns), migrated, tuple(stale))


def refresh_run_matches(
    session: Session, run_id: uuid.UUID, gates: MatchingGates
) -> None:
    """Re-derive proposals and match candidates from the current roster."""
    hints = tuple(
        NameHintProposal(row.diarization_label, row.proposed_name)
        for row in session.scalars(
            select(SpeakerAssignment).where(
                SpeakerAssignment.pipeline_run_id == run_id,
                SpeakerAssignment.method == AssignmentMethod.LLM_HINT.value,
            )
        )
        if row.proposed_name is not None
    )
    decisions = evaluate_run(session, run_id, gates)
    proposals = tuple(d.proposal for d in decisions if d.proposal is not None)
    replace_run_proposals(session, run_id, proposals, hints)
    replace_run_match_candidates(session, run_id, decisions)
