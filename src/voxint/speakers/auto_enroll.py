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
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from voxint.adjudication.ledger import record_decision
from voxint.adjudication.resolver import Resolution, label_states
from voxint.db.models import (
    AdjudicationDecision,
    AutoEnrollEvidence,
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

# Auto-enroll evidence decisions (issue #434)
AE_DECISION_LINKED = "linked"
AE_DECISION_CREATED = "created"
AE_DECISION_SKIPPED = "skipped"

# Reasons
AE_REASON_MATCHED = "matched"
AE_REASON_NO_ELIGIBLE_TURNS = "no_eligible_turns"
AE_REASON_TOO_FEW_TURNS = "too_few_turns"
AE_REASON_TOO_LITTLE_SPEECH = "too_little_speech"
AE_REASON_DEGENERATE_CENTROID = "degenerate_centroid"
AE_REASON_NO_ROSTER = "no_roster"
AE_REASON_BELOW_COSINE = "below_cosine"
AE_REASON_BELOW_MARGIN = "below_margin"
AE_REASON_BELOW_VOTE_AGREEMENT = "below_vote_agreement"
AE_REASON_EXISTING_DECISION = "existing_decision"
AE_REASON_SPEAKER_INACTIVE = "speaker_inactive"
AE_REASON_NAMING_COLLISION = "naming_collision"
AE_REASON_EXCEPTION = "exception"


@dataclass(frozen=True)
class AutoEnrollResult:
    created: int
    matched: int
    skipped: int


@dataclass(frozen=True)
class AutoEnrollMatchResult:
    """What _cosine_match decided and why, preserving all diagnostic numbers."""

    matched_speaker_id: uuid.UUID | None
    reason: str
    top_speaker_id: uuid.UUID | None
    similarity: float | None
    margin: float | None
    vote_agreement: float | None
    roster_size: int


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
) -> AutoEnrollMatchResult:
    """Check if a label centroid matches an existing roster speaker.

    Uses standard-tier thresholds for the link-or-create decision (grounding
    tier is reserved for the ``grounded`` flag in ``evaluate_run``). When the
    roster has only one speaker, the margin gate is meaningless so the cosine
    bar rises to the grounding tier as a singleton guard.
    """
    if not roster:
        return AutoEnrollMatchResult(
            matched_speaker_id=None,
            reason=AE_REASON_NO_ROSTER,
            top_speaker_id=None,
            similarity=None,
            margin=None,
            vote_agreement=None,
            roster_size=0,
        )

    ranked = sorted(
        ((float(centroid @ c), sid) for sid, c in roster.items()),
        key=lambda pair: (-pair[0], pair[1]),
    )
    top_sim, top_speaker = ranked[0]
    top_sim = max(-1.0, min(1.0, top_sim))

    # Singleton guard: with only one roster speaker the margin is infinite
    # and provides no discrimination, so require the higher grounding cosine.
    cosine_floor = gates.grounded_min_cosine if len(ranked) == 1 else gates.min_cosine
    if top_sim < cosine_floor:
        return AutoEnrollMatchResult(
            matched_speaker_id=None,
            reason=AE_REASON_BELOW_COSINE,
            top_speaker_id=top_speaker,
            similarity=top_sim,
            margin=None,
            vote_agreement=None,
            roster_size=len(ranked),
        )

    margin: float | None = None
    if len(ranked) > 1:
        margin = top_sim - ranked[1][0]
        if margin < gates.min_margin:
            return AutoEnrollMatchResult(
                matched_speaker_id=None,
                reason=AE_REASON_BELOW_MARGIN,
                top_speaker_id=top_speaker,
                similarity=top_sim,
                margin=margin,
                vote_agreement=None,
                roster_size=len(ranked),
            )

    weighted = [(v, min(usable, gates.turn_weight_cap_seconds)) for v, usable in entries]

    def _nearest(vector: np.ndarray) -> uuid.UUID:
        return min(
            ((float(vector @ c), sid) for sid, c in roster.items()),
            key=lambda pair: (-pair[0], pair[1]),
        )[1]

    agree_weight = sum(w for v, w in weighted if _nearest(v) == top_speaker)
    total_weight = sum(w for _, w in weighted)
    vote_agreement = agree_weight / total_weight if total_weight > 0 else 0.0

    if vote_agreement < gates.min_vote_agreement:
        return AutoEnrollMatchResult(
            matched_speaker_id=None,
            reason=AE_REASON_BELOW_VOTE_AGREEMENT,
            top_speaker_id=top_speaker,
            similarity=top_sim,
            margin=margin,
            vote_agreement=vote_agreement,
            roster_size=len(ranked),
        )

    return AutoEnrollMatchResult(
        matched_speaker_id=top_speaker,
        reason=AE_REASON_MATCHED,
        top_speaker_id=top_speaker,
        similarity=top_sim,
        margin=margin,
        vote_agreement=vote_agreement,
        roster_size=len(ranked),
    )


def _lock_run(session: Session, run_id: uuid.UUID) -> None:
    """Acquire FOR UPDATE on the run row, serializing with operator rulings."""
    session.execute(select(PipelineRun.id).where(PipelineRun.id == run_id).with_for_update())


def _lock_speaker(session: Session, speaker_id: uuid.UUID) -> Speaker | None:
    """FOR SHARE on the speaker row, preventing concurrent archive/merge.

    Returns the Speaker if still active, None if archived/merged since the
    roster was read.
    """
    speaker = session.execute(
        select(Speaker).where(Speaker.id == speaker_id).with_for_update(read=True)
    ).scalar_one_or_none()
    if speaker is None or not is_active(speaker):
        return None
    return speaker


def _label_has_decision(session: Session, run_id: uuid.UUID, label: str) -> bool:
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


def _persist_evidence(
    session: Session,
    run_id: uuid.UUID,
    label: str,
    decision: str,
    reason: str,
    embedding_space: str | None,
    match_result: AutoEnrollMatchResult | None,
    eligible_turns: int,
    eligible_seconds: float,
) -> None:
    """Upsert one auto_enroll_evidence row (per-label, not run-wide)."""
    values: dict[str, Any] = {
        "id": uuid.uuid4(),
        "pipeline_run_id": run_id,
        "diarization_label": label,
        "decision": decision,
        "reason": reason,
        "embedding_space": embedding_space,
        "top_speaker_id": match_result.top_speaker_id if match_result else None,
        "similarity": match_result.similarity if match_result else None,
        "margin": match_result.margin if match_result else None,
        "vote_agreement": match_result.vote_agreement if match_result else None,
        "eligible_turns": eligible_turns,
        "eligible_seconds": eligible_seconds,
        "roster_size": match_result.roster_size if match_result else None,
    }
    stmt = pg_insert(AutoEnrollEvidence).values(**values)
    stmt = stmt.on_conflict_do_update(
        constraint="auto_enroll_evidence_label_key",
        set_={
            key: stmt.excluded[key]
            for key in values
            if key not in ("id", "pipeline_run_id", "diarization_label")
        },
    )
    session.execute(stmt)


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
    eligibility_skips: dict[str, tuple[str, str | None, int, float]] = {}
    for s in sorted(unresolved, key=lambda s: -s.total_seconds):
        if s.label not in by_label:
            eligibility_skips[s.label] = (AE_REASON_NO_ELIGIBLE_TURNS, None, 0, 0.0)
            continue
        space, entries = by_label[s.label]
        eligible_turns = len(entries)
        eligible_seconds = sum(usable for _, usable in entries)
        if eligible_turns < gates.grounded_min_turns:
            eligibility_skips[s.label] = (
                AE_REASON_TOO_FEW_TURNS,
                space,
                eligible_turns,
                eligible_seconds,
            )
            continue
        if eligible_seconds < gates.grounded_min_seconds:
            eligibility_skips[s.label] = (
                AE_REASON_TOO_LITTLE_SPEECH,
                space,
                eligible_turns,
                eligible_seconds,
            )
            continue
        centroid = label_centroid(entries, gates.turn_weight_cap_seconds)
        if centroid is None:
            eligibility_skips[s.label] = (
                AE_REASON_DEGENERATE_CENTROID,
                space,
                eligible_turns,
                eligible_seconds,
            )
            continue
        centroids_by_label[s.label] = (space, centroid, entries)
        candidate_labels.append(s.label)

    for label, (
        reason,
        skipped_space,
        eligible_turns,
        eligible_seconds,
    ) in eligibility_skips.items():
        try:
            _persist_evidence(
                session,
                run_id,
                label,
                AE_DECISION_SKIPPED,
                reason,
                skipped_space,
                None,
                eligible_turns,
                eligible_seconds,
            )
        except Exception:
            logger.exception(
                "auto-enroll run %s label %s: failed to persist eligibility evidence",
                run_id,
                label,
            )

    if not centroids_by_label:
        return AutoEnrollResult(created=0, matched=0, skipped=len(unresolved))

    session.execute(text(f"SELECT pg_advisory_xact_lock({LOCK_KEY})"))
    _lock_run(session, run_id)

    # Re-derive unresolved set under the lock: a concurrent worker may have
    # enrolled these labels between our initial label_states() read and the
    # lock acquisition. Without this, the per-label _label_has_decision check
    # would overwrite good linked/created evidence with skipped/existing_decision.
    still_unresolved = {
        s.label
        for s in label_states(session, run_id)
        if s.resolution is Resolution.UNRESOLVED
    }
    candidate_labels = [lbl for lbl in candidate_labels if lbl in still_unresolved]

    created = 0
    matched = 0
    skipped = len(unresolved) - len(candidate_labels)

    for label in candidate_labels:
        space, centroid, entries = centroids_by_label[label]
        eligible_turns = len(entries)
        eligible_seconds = sum(usable for _, usable in entries)
        match_result: AutoEnrollMatchResult | None = None

        try:
            with session.begin_nested():
                if _label_has_decision(session, run_id, label):
                    _persist_evidence(
                        session,
                        run_id,
                        label,
                        AE_DECISION_SKIPPED,
                        AE_REASON_EXISTING_DECISION,
                        space,
                        None,
                        eligible_turns,
                        eligible_seconds,
                    )
                    skipped += 1
                    continue

                roster = roster_centroids(session, space)
                match_result = _cosine_match(centroid, roster, entries, gates)
                match_speaker_id = match_result.matched_speaker_id

                if match_speaker_id is not None:
                    locked = _lock_speaker(session, match_speaker_id)
                    if locked is None:
                        _persist_evidence(
                            session,
                            run_id,
                            label,
                            AE_DECISION_SKIPPED,
                            AE_REASON_SPEAKER_INACTIVE,
                            space,
                            match_result,
                            eligible_turns,
                            eligible_seconds,
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
                        speaker_id=match_speaker_id,
                    )
                    session.add(
                        SpeakerEmbedding(
                            speaker_id=match_speaker_id,
                            embedding_space=space,
                            embedding=centroid,
                            source_pipeline_run_id=run_id,
                            source_diarization_label=label,
                            source_adjudication_decision_id=decision.id,
                        )
                    )
                    session.flush()
                    _persist_evidence(
                        session,
                        run_id,
                        label,
                        AE_DECISION_LINKED,
                        AE_REASON_MATCHED,
                        space,
                        match_result,
                        eligible_turns,
                        eligible_seconds,
                    )
                    matched += 1
                    logger.debug(
                        "auto-enroll run %s label %s: matched existing speaker %s",
                        run_id,
                        label,
                        match_speaker_id,
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
                            run_id,
                            label,
                            MAX_NAME_RETRIES,
                        )
                        _persist_evidence(
                            session,
                            run_id,
                            label,
                            AE_DECISION_SKIPPED,
                            AE_REASON_NAMING_COLLISION,
                            space,
                            match_result,
                            eligible_turns,
                            eligible_seconds,
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
                    _persist_evidence(
                        session,
                        run_id,
                        label,
                        AE_DECISION_CREATED,
                        match_result.reason,
                        space,
                        match_result,
                        eligible_turns,
                        eligible_seconds,
                    )
                    created += 1
                    logger.debug(
                        "auto-enroll run %s label %s: created speaker %s (%s)",
                        run_id,
                        label,
                        speaker.id,
                        speaker.display_name,
                    )
        except Exception:
            logger.exception(
                "auto-enroll run %s label %s: failed, continuing",
                run_id,
                label,
            )
            skipped += 1
            try:
                with session.begin_nested():
                    _persist_evidence(
                        session,
                        run_id,
                        label,
                        AE_DECISION_SKIPPED,
                        AE_REASON_EXCEPTION,
                        space,
                        match_result,
                        eligible_turns,
                        eligible_seconds,
                    )
            except Exception:
                logger.exception(
                    "auto-enroll run %s label %s: failed to persist exception evidence",
                    run_id,
                    label,
                )

    logger.info(
        "auto-enroll run %s: %d created, %d matched, %d skipped",
        run_id,
        created,
        matched,
        skipped,
    )
    return AutoEnrollResult(created=created, matched=matched, skipped=skipped)
