"""Speaker enrollment: turn an unmatched voice into a roster identity.

The operator names a voice; this module creates the ``speakers`` row, one
duration-weighted enrollment centroid in ``speaker_embeddings`` (built from
exactly the same eligibility rules and centroid math as cosine matching, via
``speakers.matching``), and the ``assign`` ledger ruling — atomically. Raw
per-turn vectors stay in ``diarization_turns``; the centroid is a derived
convenience, fully re-derivable.

Replay safety is layered: the ledger's idempotency key makes the decision
append-once; checking it *first* means a duplicate POST returns the original
outcome without touching the roster; and the unique constraint on
``speaker_embeddings.source_adjudication_decision_id`` backstops the invariant
at the database.
"""

import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from voxint.adjudication.ledger import ConflictingReplayError, record_decision
from voxint.db.models import (
    AdjudicationDecision,
    Decision,
    Speaker,
    SpeakerEmbedding,
)
from voxint.speakers.matching import (
    MatchingGates,
    eligible_label_vectors,
    label_centroid,
)
from voxint.speakers.roster import (
    MAX_DISPLAY_NAME_LENGTH as MAX_DISPLAY_NAME_LENGTH,  # re-export (API imports it here)
)
from voxint.speakers.roster import (
    describe_name_owner,
    is_active,
    normalize_display_name,
)


class EnrollmentError(Exception):
    """The enrollment cannot proceed as requested — operator-visible."""


@dataclass(frozen=True)
class EnrollmentResult:
    speaker_id: uuid.UUID
    decision_id: uuid.UUID
    created_speaker: bool


def enroll_new_speaker(
    session: Session,
    *,
    run_id: uuid.UUID,
    diarization_label: str,
    display_name: str,
    operator: str,
    idempotency_key: str,
    gates: MatchingGates,
) -> EnrollmentResult:
    """Create a named speaker from a run's label and rule it assigned.

    Caller owns the transaction: everything here commits or rolls back as one.
    """
    try:
        name = normalize_display_name(display_name)
    except ValueError as exc:
        raise EnrollmentError(str(exc)) from exc

    # Replay first: a duplicate POST of a completed enrollment must return the
    # original outcome, not attempt a second speaker. Only an exact replay
    # qualifies — the same run, label, and operator; anything else reusing the
    # key is a bug or a stale form, never a silent success. The submitted name
    # is deliberately NOT compared: display_name is mutable (roster rename), so
    # a replay arriving after a rename must still resolve to the original
    # outcome — the durable identity is the ledger row, not the name.
    existing = session.execute(
        select(AdjudicationDecision).where(
            AdjudicationDecision.idempotency_key == idempotency_key
        )
    ).scalar_one_or_none()
    if existing is not None:
        if existing.speaker_id is None:
            raise EnrollmentError("idempotency key was used for a non-assign decision")
        if (
            existing.pipeline_run_id != run_id
            or existing.diarization_label != diarization_label
            or existing.operator != operator
        ):
            raise ConflictingReplayError(idempotency_key)
        return EnrollmentResult(
            speaker_id=existing.speaker_id,
            decision_id=existing.id,
            created_speaker=False,
        )

    owner = session.execute(
        select(Speaker).where(Speaker.display_name == name)
    ).scalar_one_or_none()
    if owner is not None:
        # Enrolling means a NEW voice; attaching to an existing identity is
        # the plain assign flow, which needs no centroid. Names are globally
        # unique across the whole lifecycle — a merged or archived owner gets
        # lifecycle-specific guidance instead.
        if is_active(owner):
            raise EnrollmentError(
                f"speaker {name!r} already exists — use assign-to-existing instead"
            )
        raise EnrollmentError(describe_name_owner(owner))

    by_label = eligible_label_vectors(session, run_id, gates)
    if diarization_label not in by_label:
        raise EnrollmentError(
            f"label {diarization_label!r} has no eligible embedded turns to enroll from"
        )
    space, entries = by_label[diarization_label]
    centroid = label_centroid(entries, gates.turn_weight_cap_seconds)
    if centroid is None:
        raise EnrollmentError(
            f"label {diarization_label!r} produced no usable centroid"
        )

    speaker = Speaker(display_name=name)
    try:
        # Savepoint: the pre-check above races a concurrent enrollment of the
        # same name; the unique index decides, and the loser gets the same
        # operator-visible error as the sequential case.
        with session.begin_nested():
            session.add(speaker)  # flushes on savepoint exit; id assigned
    except IntegrityError as exc:
        raise EnrollmentError(
            f"speaker {name!r} already exists — use assign-to-existing instead"
        ) from exc

    decision = record_decision(
        session,
        pipeline_run_id=run_id,
        diarization_label=diarization_label,
        decision=Decision.ASSIGN,
        operator=operator,
        idempotency_key=idempotency_key,
        speaker_id=speaker.id,
    )
    session.add(
        SpeakerEmbedding(
            speaker_id=speaker.id,
            embedding_space=space,
            embedding=centroid,
            source_pipeline_run_id=run_id,
            source_diarization_label=diarization_label,
            source_adjudication_decision_id=decision.id,
        )
    )
    session.flush()
    return EnrollmentResult(
        speaker_id=speaker.id, decision_id=decision.id, created_speaker=True
    )
