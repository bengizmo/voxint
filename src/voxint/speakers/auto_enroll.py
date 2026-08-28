"""Post-completion auto-enrollment of unmatched voices (#275).

When a pipeline run finishes, labels that passed through the whole pipeline
with no human decision and no grounded cosine match are candidates for
auto-enrollment. Eligible candidates become unnamed roster speakers
("Voice 1", "Voice 2", ...) so the operator can see and name them on the
speakers page. Cross-run consolidation happens naturally: once enrolled,
the centroid participates in future runs' ``evaluate_run()`` matching.

Failure here never fails the run (caller catches and logs).
"""

import logging
import math
import uuid
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from voxint.adjudication.ledger import record_decision
from voxint.adjudication.resolver import Resolution, label_states
from voxint.db.models import (
    AdjudicationDecision,
    Decision,
    PipelineRun,
    Speaker,
    SpeakerEmbedding,
)
from voxint.speakers.matching import (
    MatchingGates,
    eligible_label_vectors,
    label_centroid,
    roster_centroids,
)
from voxint.speakers.roster import is_active

logger = logging.getLogger(__name__)

OPERATOR = "system:auto_enroll"
NAME_PREFIX = "Voice "
MAX_NAME_RETRIES = 3
LOCK_KEY = 0x_564F_5849_4E54  # "VOXINT" in hex, fits a pg bigint


@dataclass(frozen=True)
class AutoEnrollResult:
    created: int
    matched: int
    skipped: int


def _next_voice_number(session: Session) -> int:
    """The next available "Voice N" number across all speakers."""
    result = session.execute(
        text(
            "SELECT COALESCE(MAX("
            "  CAST((regexp_match(display_name, '^Voice (\\d{1,9})$'))[1] AS int)"
            "), 0) + 1"
            " FROM speakers"
        )
    ).scalar_one()
    return int(result)


def _cosine_match(
    centroid: np.ndarray,
    roster: dict[uuid.UUID, np.ndarray],
    entries: list[tuple[np.ndarray, float]],
    gates: MatchingGates,
) -> uuid.UUID | None:
    """Check if a label centroid matches an existing roster speaker at grounding tier.

    Returns the matched speaker_id or None. Uses the same cosine/margin/vote
    math as ``evaluate_run`` but with grounding-tier thresholds.
    """
    if not roster:
        return None

    ranked = sorted(
        ((float(centroid @ c), sid) for sid, c in roster.items()),
        key=lambda pair: (-pair[0], pair[1]),
    )
    top_sim, top_speaker = ranked[0]
    top_sim = max(-1.0, min(1.0, top_sim))
    margin = top_sim - ranked[1][0] if len(ranked) > 1 else math.inf

    if top_sim < gates.grounded_min_cosine:
        return None
    if margin < gates.grounded_min_margin:
        return None

    weighted = [(v, min(usable, gates.turn_weight_cap_seconds)) for v, usable in entries]

    def _nearest(vector: np.ndarray) -> uuid.UUID:
        return min(
            ((float(vector @ c), sid) for sid, c in roster.items()),
            key=lambda pair: (-pair[0], pair[1]),
        )[1]

    agree_weight = sum(w for v, w in weighted if _nearest(v) == top_speaker)
    total_weight = sum(w for _, w in weighted)
    vote_agreement = agree_weight / total_weight if total_weight > 0 else 0.0

    if vote_agreement < gates.grounded_min_vote_agreement:
        return None

    return top_speaker


def _lock_run(session: Session, run_id: uuid.UUID) -> None:
    """Acquire FOR UPDATE on the run row, serializing with operator rulings."""
    session.execute(
        select(PipelineRun.id)
        .where(PipelineRun.id == run_id)
        .with_for_update()
    )


def _lock_speaker(session: Session, speaker_id: uuid.UUID) -> Speaker | None:
    """FOR SHARE on the speaker row, preventing concurrent archive/merge.

    Returns the Speaker if still active, None if archived/merged since the
    roster was read.
    """
    speaker = session.execute(
        select(Speaker)
        .where(Speaker.id == speaker_id)
        .with_for_update(read=True)
    ).scalar_one_or_none()
    if speaker is None or not is_active(speaker):
        return None
    return speaker


def _label_has_decision(
    session: Session, run_id: uuid.UUID, label: str
) -> bool:
    """Re-check: did a decision appear for this label since we started?"""
    return (
        session.execute(
            select(AdjudicationDecision.id).where(
                AdjudicationDecision.pipeline_run_id == run_id,
                AdjudicationDecision.diarization_label == label,
                AdjudicationDecision.transcript_segment_id.is_(None),
            )
        ).scalar_one_or_none()
        is not None
    )


def auto_enroll_run(
    session: Session,
    run_id: uuid.UUID,
    gates: MatchingGates,
) -> AutoEnrollResult:
    """Auto-enroll unmatched voices from a completed run.

    Must be called inside a transaction the caller will commit. Acquires a
    pg advisory lock to serialize across concurrent workers.
    """
    states = label_states(session, run_id)
    unresolved = [s for s in states if s.resolution is Resolution.UNRESOLVED]
    if not unresolved:
        return AutoEnrollResult(created=0, matched=0, skipped=0)

    by_label = eligible_label_vectors(session, run_id, gates)
    centroids_by_label: dict[str, tuple[str, np.ndarray, list[tuple[np.ndarray, float]]]] = {}
    candidate_labels: list[str] = []
    for s in sorted(unresolved, key=lambda s: -s.total_seconds):
        if s.label not in by_label:
            continue
        space, entries = by_label[s.label]
        eligible_turns = len(entries)
        eligible_seconds = sum(usable for _, usable in entries)
        if eligible_turns < gates.grounded_min_turns:
            continue
        if eligible_seconds < gates.grounded_min_seconds:
            continue
        centroid = label_centroid(entries, gates.turn_weight_cap_seconds)
        if centroid is not None:
            centroids_by_label[s.label] = (space, centroid, entries)
            candidate_labels.append(s.label)

    if not centroids_by_label:
        return AutoEnrollResult(created=0, matched=0, skipped=len(unresolved))

    session.execute(text(f"SELECT pg_advisory_xact_lock({LOCK_KEY})"))
    _lock_run(session, run_id)

    created = 0
    matched = 0
    skipped = len(unresolved) - len(candidate_labels)

    for label in candidate_labels:
        space, centroid, entries = centroids_by_label[label]

        try:
            with session.begin_nested():
                if _label_has_decision(session, run_id, label):
                    skipped += 1
                    continue

                roster = roster_centroids(session, space)
                match_speaker_id = _cosine_match(centroid, roster, entries, gates)

                if match_speaker_id is not None:
                    locked = _lock_speaker(session, match_speaker_id)
                    if locked is None:
                        skipped += 1
                        continue
                    record_decision(
                        session,
                        pipeline_run_id=run_id,
                        diarization_label=label,
                        decision=Decision.AUTO_ENROLL,
                        operator=OPERATOR,
                        idempotency_key=f"auto_enroll:{run_id}:{label}",
                        speaker_id=match_speaker_id,
                    )
                    matched += 1
                    logger.debug(
                        "auto-enroll run %s label %s: matched existing speaker %s",
                        run_id, label, match_speaker_id,
                    )
                else:
                    voice_number = _next_voice_number(session)
                    speaker: Speaker | None = None
                    for attempt in range(MAX_NAME_RETRIES):
                        name = f"{NAME_PREFIX}{voice_number + attempt}"
                        try:
                            with session.begin_nested():
                                speaker = Speaker(display_name=name)
                                session.add(speaker)
                                session.flush()
                            break
                        except IntegrityError:
                            speaker = None
                            voice_number = _next_voice_number(session)
                    if speaker is None:
                        logger.warning(
                            "auto-enroll run %s label %s: naming collision after %d retries",
                            run_id, label, MAX_NAME_RETRIES,
                        )
                        skipped += 1
                        continue

                    decision = record_decision(
                        session,
                        pipeline_run_id=run_id,
                        diarization_label=label,
                        decision=Decision.AUTO_ENROLL,
                        operator=OPERATOR,
                        idempotency_key=f"auto_enroll:{run_id}:{label}",
                        speaker_id=speaker.id,
                    )
                    session.add(
                        SpeakerEmbedding(
                            speaker_id=speaker.id,
                            embedding_space=space,
                            embedding=centroid,
                            source_pipeline_run_id=run_id,
                            source_diarization_label=label,
                            source_adjudication_decision_id=decision.id,
                        )
                    )
                    session.flush()
                    created += 1
                    logger.debug(
                        "auto-enroll run %s label %s: created speaker %s (%s)",
                        run_id, label, speaker.id, speaker.display_name,
                    )
        except Exception:
            logger.exception(
                "auto-enroll run %s label %s: failed, continuing",
                run_id, label,
            )
            skipped += 1

    logger.info(
        "auto-enroll run %s: %d created, %d matched, %d skipped",
        run_id, created, matched, skipped,
    )
    return AutoEnrollResult(created=created, matched=matched, skipped=skipped)
