"""Orchestration and caching for speaker insights (issue #335)."""

from __future__ import annotations

import hashlib
import logging
import uuid
from collections import Counter
from datetime import datetime
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from voxint.adjudication.attribution import Emission, walk_attributions, winning_attribution
from voxint.adjudication.resolver import Resolution
from voxint.api.speaker_stats import (
    compute_ego_transitions,
    compute_log_odds,
    compute_wpm,
    tokenize_text,
)
from voxint.api.term_stats import source_hash
from voxint.db.models import (
    AdjudicationDecision,
    CorpusAnalysisArtifact,
    CorpusAnalysisArtifactKind,
    MediaItem,
    PipelineRun,
    RunStatus,
    SegmentReviewState,
    Speaker,
    TranscriptSegment,
)

logger = logging.getLogger(__name__)

_MIN_SPEAKER_TOKENS = 500
_MIN_SPEAKER_RECORDINGS = 2


def _advisory_lock_key() -> int:
    return int(hashlib.sha256(b"voxint:speaker_insights").hexdigest()[:8], 16) & 0x7FFFFFFF


def _canonical_runs(session: Session) -> list[tuple[uuid.UUID, uuid.UUID, datetime]]:
    """Return the newest completed, non-archived run for every media item."""
    ranked = (
        select(
            PipelineRun.id.label("run_id"),
            PipelineRun.media_item_id.label("media_id"),
            func.row_number()
            .over(
                partition_by=PipelineRun.media_item_id,
                order_by=(PipelineRun.created_at.desc(), PipelineRun.id.desc()),
            )
            .label("rank"),
        )
        .where(
            PipelineRun.status == RunStatus.COMPLETED.value,
            PipelineRun.archived_at.is_(None),
        )
        .subquery()
    )
    rows = session.execute(
        select(ranked.c.run_id, ranked.c.media_id, MediaItem.created_at)
        .join(MediaItem, MediaItem.id == ranked.c.media_id)
        .where(ranked.c.rank == 1)
        .order_by(MediaItem.created_at.desc(), MediaItem.id.desc())
    ).all()
    return [(row.run_id, row.media_id, row.created_at) for row in rows]


def _insights_fingerprint(
    session: Session,
    run_ids: list[uuid.UUID] | None = None,
) -> str:
    """Return a global fingerprint covering insight inputs and speaker identity."""
    if run_ids is None:
        run_ids = [run_id for run_id, _, _ in _canonical_runs(session)]

    run_rows: list[tuple[uuid.UUID, datetime]] = []
    if run_ids:
        run_rows = [
            (row.id, row.updated_at)
            for row in session.execute(
                select(PipelineRun.id, PipelineRun.updated_at)
                .where(PipelineRun.id.in_(run_ids))
                .order_by(PipelineRun.id)
            )
        ]

    corrections: dict[uuid.UUID, str] = {}
    if run_ids:
        correction_rows = session.execute(
            select(
                TranscriptSegment.pipeline_run_id,
                func.max(SegmentReviewState.corrected_at),
            )
            .join(
                SegmentReviewState,
                SegmentReviewState.transcript_segment_id == TranscriptSegment.id,
            )
            .where(TranscriptSegment.pipeline_run_id.in_(run_ids))
            .group_by(TranscriptSegment.pipeline_run_id)
        ).all()
        corrections = {rid: str(corrected_at) for rid, corrected_at in correction_rows}

    corpus_fingerprint = source_hash([
        (str(rid), f"{updated_at}:{corrections.get(rid, '')}")
        for rid, updated_at in run_rows
    ])
    latest_decision = session.execute(
        select(func.max(AdjudicationDecision.created_at))
    ).scalar_one()
    roster_rows = session.execute(
        select(
            Speaker.id, Speaker.display_name, Speaker.merged_into_id, Speaker.deleted_at,
        ).order_by(Speaker.id)
    ).all()
    roster_fingerprint = source_hash([
        (
            str(row.id),
            f"{row.display_name}:{row.merged_into_id or ''}:{row.deleted_at or ''}",
        )
        for row in roster_rows
    ])
    return source_hash([
        ("adjudication", str(latest_decision or "")),
        ("corpus", corpus_fingerprint),
        ("roster", roster_fingerprint),
    ])


def get_speaker_insights(
    session: Session, speaker_id: uuid.UUID
) -> dict[str, Any] | None:
    """Return cached speaker-insights payload, or None if not yet computed.

    This is the profile-page read path: it never computes, never fingerprints.
    Staleness is handled by the Celery task; the profile serves whatever is
    cached. A missing artifact means the task hasn't run yet.
    """
    artifact = session.execute(
        select(CorpusAnalysisArtifact)
        .where(
            CorpusAnalysisArtifact.artifact_kind
            == CorpusAnalysisArtifactKind.SPEAKER_STATS.value,
            CorpusAnalysisArtifact.scope_kind == "speaker",
            CorpusAnalysisArtifact.scope_id == speaker_id,
        )
        .order_by(CorpusAnalysisArtifact.generation.desc())
        .limit(1)
    ).scalar_one_or_none()
    if artifact is None:
        return None
    return artifact.payload


def _effective_text(emission: Emission) -> str:
    if emission.child is not None:
        return emission.child.text
    if emission.review is not None and emission.review.corrected_text is not None:
        return emission.review.corrected_text
    if emission.seg.enhanced_text is not None:
        return emission.seg.enhanced_text
    return emission.seg.raw_text


def _batch_compute_all_speakers(
    session: Session,
    canonical: list[tuple[uuid.UUID, uuid.UUID, datetime]],
) -> dict[uuid.UUID, dict[str, Any]]:
    """Fold canonical runs once and produce every eligible speaker's insights."""
    counts: dict[uuid.UUID, Counter[str]] = {}
    appearances: dict[uuid.UUID, set[uuid.UUID]] = {}
    word_counts: dict[uuid.UUID, list[int]] = {}
    durations: dict[uuid.UUID, list[float]] = {}
    speaker_names: dict[uuid.UUID, str] = {}
    sequences: list[list[str | None]] = []

    for run_id, _, _ in canonical:
        sequence: list[str | None] = []
        for emission in walk_attributions(session, run_id):
            _, resolution, speaker_id, speaker_name = winning_attribution(emission)
            if resolution not in (Resolution.HUMAN_ASSIGN, Resolution.GROUNDED_COSINE):
                speaker_id = None
            sequence.append(str(speaker_id) if speaker_id is not None else None)
            if speaker_id is None:
                continue

            if speaker_name is not None:
                speaker_names[speaker_id] = speaker_name
            appearances.setdefault(speaker_id, set()).add(run_id)
            counts.setdefault(speaker_id, Counter()).update(
                tokenize_text(_effective_text(emission))
            )

            words = emission.seg.words
            if words is None or emission.seg.suspect:
                continue
            if emission.child is not None:
                count = len(words[emission.child.word_start : emission.child.word_end])
                duration = emission.child.end_seconds - emission.child.start_seconds
            else:
                count = len(words)
                duration = emission.seg.end_seconds - emission.seg.start_seconds
            word_counts.setdefault(speaker_id, []).append(count)
            durations.setdefault(speaker_id, []).append(duration)
        sequences.append(sequence)

    corpus_counts: Counter[str] = Counter()
    for speaker_counts in counts.values():
        corpus_counts.update(speaker_counts)

    payloads: dict[uuid.UUID, dict[str, Any]] = {}
    for speaker_id, speaker_counts in counts.items():
        token_count = sum(speaker_counts.values())
        recording_count = len(appearances[speaker_id])
        if token_count < _MIN_SPEAKER_TOKENS or recording_count < _MIN_SPEAKER_RECORDINGS:
            continue

        terms = compute_log_odds(speaker_counts, corpus_counts - speaker_counts)
        transitions_in, transitions_out = compute_ego_transitions(sequences, str(speaker_id))
        wpm, timed_segments, total_segments = compute_wpm(
            word_counts.get(speaker_id, []), durations.get(speaker_id, [])
        )
        payloads[speaker_id] = {
            "distinctive_terms": [
                {
                    "term": term.term,
                    "count": term.count,
                    "log_odds": term.log_odds,
                    "z_score": term.z_score,
                }
                for term in terms
            ],
            "ego_transitions_in": [
                {
                    "speaker_id": str(edge.from_speaker_id),
                    "speaker_name": speaker_names.get(edge.from_speaker_id),
                    "count": edge.count,
                }
                for edge in transitions_in
            ],
            "ego_transitions_out": [
                {
                    "speaker_id": str(edge.to_speaker_id),
                    "speaker_name": speaker_names.get(edge.to_speaker_id),
                    "count": edge.count,
                }
                for edge in transitions_out
            ],
            "wpm": wpm,
            "wpm_timed_segments": timed_segments,
            "wpm_total_segments": total_segments,
            "speaker_token_count": token_count,
            "recording_count": recording_count,
        }

    logger.info("Computed speaker insights for %d eligible speakers", len(payloads))
    return payloads


def _write_all_speaker_artifacts(
    session: Session,
    payloads: dict[uuid.UUID, dict[str, Any]],
    fingerprint: str,
) -> None:
    """Replace the speaker-insights cache as one advisory-locked batch.

    Lock is already held by the caller (compute_all_speaker_insights).
    """
    fresh_exists = session.execute(
        select(CorpusAnalysisArtifact.id)
        .where(
            CorpusAnalysisArtifact.artifact_kind
            == CorpusAnalysisArtifactKind.SPEAKER_STATS.value,
            CorpusAnalysisArtifact.source_hash == fingerprint,
        )
        .limit(1)
    ).scalar_one_or_none()
    if fresh_exists is not None:
        return

    session.execute(
        delete(CorpusAnalysisArtifact).where(
            CorpusAnalysisArtifact.artifact_kind
            == CorpusAnalysisArtifactKind.SPEAKER_STATS.value
        )
    )
    artifacts = [
        CorpusAnalysisArtifact(
            scope_kind="speaker",
            scope_id=speaker_id,
            artifact_kind=CorpusAnalysisArtifactKind.SPEAKER_STATS.value,
            generation=1,
            source_hash=fingerprint,
            payload=payload,
        )
        for speaker_id, payload in payloads.items()
    ]
    if artifacts:
        session.add_all(artifacts)
    session.flush()


def compute_all_speaker_insights(session: Session) -> int:
    """Compute and cache speaker insights when the corpus fingerprint is stale.

    Advisory lock is held for the full duration so a slower task with an older
    fingerprint cannot overwrite fresher artifacts.
    """
    session.execute(select(func.pg_advisory_xact_lock(_advisory_lock_key())))

    canonical = _canonical_runs(session)
    run_ids = [run_id for run_id, _, _ in canonical]
    fingerprint = _insights_fingerprint(session, run_ids=run_ids)

    fresh_exists = session.execute(
        select(CorpusAnalysisArtifact.id)
        .where(
            CorpusAnalysisArtifact.artifact_kind
            == CorpusAnalysisArtifactKind.SPEAKER_STATS.value,
            CorpusAnalysisArtifact.source_hash == fingerprint,
        )
        .limit(1)
    ).scalar_one_or_none()
    if fresh_exists is not None:
        return 0

    payloads = _batch_compute_all_speakers(session, canonical)
    _write_all_speaker_artifacts(session, payloads, fingerprint)
    return len(payloads)
