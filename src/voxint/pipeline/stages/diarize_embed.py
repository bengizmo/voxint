"""Diarize + embed stage: turn ledger, per-window embeddings, segment labels.

Persists one ``diarization_turns`` row per turn — including skipped windows,
whose ``skip_reason`` stays auditable — and stamps each transcript segment
with the label of the turn it overlaps most. No Speaker rows are created here:
identity is P4's matching problem, fed by these observations.
"""

import uuid

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from voxint.clients.base import DiarizationTurn as TurnResult
from voxint.db.models import DiarizationTurn, TranscriptSegment
from voxint.pipeline.stages.context import StageContext, StageDataError, normalized_audio_path


def run(ctx: StageContext, session: Session, run_id: uuid.UUID) -> None:
    audio = normalized_audio_path(session, run_id, ctx.media_root)
    turns = ctx.diarizer.diarize(
        audio,
        max_speakers=ctx.diarization_max_speakers,
        num_speakers=ctx.diarization_num_speakers,
    ).turns

    session.execute(
        delete(DiarizationTurn).where(DiarizationTurn.pipeline_run_id == run_id)
    )
    session.execute(
        update(TranscriptSegment)
        .where(TranscriptSegment.pipeline_run_id == run_id)
        .values(diarization_label=None)
    )

    if not turns:
        return  # nothing spoke; segments keep NULL labels

    embedding = ctx.embedder.embed(
        audio, tuple((t.start_seconds, t.end_seconds) for t in turns)
    )
    if len(embedding.entries) != len(turns):
        # The HTTP client already enforces this per batch; fakes must too.
        raise StageDataError(
            f"run {run_id}: {len(embedding.entries)} embedding entries"
            f" for {len(turns)} turns"
        )

    for index, (turn, entry) in enumerate(zip(turns, embedding.entries, strict=True)):
        session.add(
            DiarizationTurn(
                pipeline_run_id=run_id,
                turn_index=index,
                start_seconds=turn.start_seconds,
                end_seconds=turn.end_seconds,
                label=turn.label,
                overlap=turn.overlap,
                overlap_seconds=turn.overlap_seconds,
                snr_db=entry.snr_db,
                skip_reason=entry.skip_reason,
                embedding=list(entry.embedding) if entry.embedding is not None else None,
                embedding_space=(
                    embedding.embedding_space if entry.embedding is not None else None
                ),
            )
        )

    segments = (
        session.execute(
            select(TranscriptSegment)
            .where(TranscriptSegment.pipeline_run_id == run_id)
            .order_by(TranscriptSegment.segment_index)
        )
        .scalars()
        .all()
    )
    for segment in segments:
        segment.diarization_label = _dominant_label(
            segment.start_seconds, segment.end_seconds, turns
        )


def _dominant_label(
    start: float, end: float, turns: tuple[TurnResult, ...]
) -> str | None:
    """Label of the turn with maximum temporal intersection; ties go to the
    earliest turn (deterministic across retries); no intersection → None.

    One label per segment is a stated v1 simplification for overlapped speech —
    the full turn ledger keeps the truth.
    """
    best_label: str | None = None
    best_overlap = 0.0
    for turn in turns:
        overlap = min(end, turn.end_seconds) - max(start, turn.start_seconds)
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = turn.label
    return best_label
