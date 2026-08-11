"""Append-only access to the adjudication decision ledger.

This module is the only sanctioned way to write ``adjudication_decisions``.
The table itself rejects UPDATE/DELETE via a trigger (see alembic 0001);
:func:`record_decision` adds idempotent-replay semantics on top: replaying the
same idempotency key with the same payload returns the existing row, replaying
it with a *different* payload is an error, never a silent overwrite.
"""

import uuid

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from voxint.db.models import AdjudicationDecision, Decision


class ConflictingReplayError(Exception):
    def __init__(self, idempotency_key: str) -> None:
        super().__init__(
            f"idempotency key {idempotency_key!r} was already used with a different payload"
        )
        self.idempotency_key = idempotency_key


def _payload_matches(
    row: AdjudicationDecision,
    pipeline_run_id: uuid.UUID,
    diarization_label: str,
    decision: Decision,
    speaker_id: uuid.UUID | None,
    operator: str,
) -> bool:
    return (
        row.pipeline_run_id == pipeline_run_id
        and row.diarization_label == diarization_label
        and row.decision == decision.value
        and row.speaker_id == speaker_id
        and row.operator == operator
    )


def record_decision(
    session: Session,
    *,
    pipeline_run_id: uuid.UUID,
    diarization_label: str,
    decision: Decision,
    operator: str,
    idempotency_key: str,
    speaker_id: uuid.UUID | None = None,
) -> AdjudicationDecision:
    """Append a ruling; replaying an identical request returns the existing row."""
    existing = session.execute(
        select(AdjudicationDecision).where(
            AdjudicationDecision.idempotency_key == idempotency_key
        )
    ).scalar_one_or_none()
    if existing is None:
        row = AdjudicationDecision(
            pipeline_run_id=pipeline_run_id,
            diarization_label=diarization_label,
            decision=decision.value,
            speaker_id=speaker_id,
            operator=operator,
            idempotency_key=idempotency_key,
        )
        try:
            # Savepoint, not a bare flush: callers compose this into larger
            # transactions (P5 enrollment creates the speaker + embedding in
            # the same one), and losing the race here must not roll their
            # work back with it.
            with session.begin_nested():
                session.add(row)
        except IntegrityError:
            # Only the savepoint rolled back. If a concurrent writer inserted
            # the same key between our SELECT and flush, adopt their row —
            # any other constraint violation (FK, CHECK) is not a replay and
            # must not be masked as one.
            existing = session.execute(
                select(AdjudicationDecision).where(
                    AdjudicationDecision.idempotency_key == idempotency_key
                )
            ).scalar_one_or_none()
            if existing is None:
                raise
        else:
            return row
    if _payload_matches(
        existing, pipeline_run_id, diarization_label, decision, speaker_id, operator
    ):
        return existing
    raise ConflictingReplayError(idempotency_key)
