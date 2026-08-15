"""Single sanctioned writer for the profile-review decision trail.

:func:`record_profile_decision` is the only way a human verdict on an
enrichment candidate is recorded. The table rejects UPDATE/DELETE via a
trigger (alembic 0010); this writer adds, on top:

- **Terminal decisions**: one accept/reject per candidate (UNIQUE in-schema).
  A rejected-in-error claim is corrected by re-running the producer, which
  yields a fresh candidate — history is never edited.
- **Stale protection**: deciding a superseded candidate raises
  :class:`StaleCandidateError` — the operator was looking at a claim a newer
  producer generation has already retired.
- **Idempotent replay** (pattern: ``adjudication/ledger.py``): the same
  ``idempotency_key`` with the same payload returns the existing row; a
  different payload raises :class:`ConflictingReplayError`.

Accepting a ``name`` claim records the act only. It never touches
``speakers.display_name``, ``speaker_assignments``, or the attribution
ledger — a read name is never grounded identity (docs/quality-gates.md).
"""

import uuid

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from voxint.db.models import (
    EnrichmentCandidate,
    ProfileDecision,
    ProfileReviewDecision,
)

MAX_OPERATOR_CHARS = 200
MAX_NOTE_CHARS = 2_000


class ConflictingReplayError(Exception):
    def __init__(self, idempotency_key: str) -> None:
        super().__init__(
            f"idempotency key {idempotency_key!r} was already used with a different payload"
        )
        self.idempotency_key = idempotency_key


class StaleCandidateError(Exception):
    """The candidate is superseded or already carries a decision."""

    def __init__(self, candidate_id: uuid.UUID, reason: str) -> None:
        super().__init__(f"candidate {candidate_id} cannot be decided: {reason}")
        self.candidate_id = candidate_id
        self.reason = reason


def _payload_matches(
    row: ProfileReviewDecision,
    candidate_id: uuid.UUID,
    decision: ProfileDecision,
    operator: str,
    note: str | None,
) -> bool:
    return (
        row.candidate_id == candidate_id
        and row.decision == decision.value
        and row.operator == operator
        and row.note == note
    )


def record_profile_decision(
    session: Session,
    *,
    candidate_id: uuid.UUID,
    decision: ProfileDecision,
    operator: str,
    idempotency_key: str,
    note: str | None = None,
) -> ProfileReviewDecision:
    """Append a profile-review verdict; identical replays return the existing row."""
    if not operator.strip() or len(operator) > MAX_OPERATOR_CHARS:
        raise ValueError(f"operator empty or over {MAX_OPERATOR_CHARS} chars")
    if note is not None and (not note.strip() or len(note) > MAX_NOTE_CHARS):
        raise ValueError(f"note empty or over {MAX_NOTE_CHARS} chars")
    if not idempotency_key.strip():
        raise ValueError("idempotency_key must be non-empty")

    def _existing_by_key() -> ProfileReviewDecision | None:
        return session.execute(
            select(ProfileReviewDecision).where(
                ProfileReviewDecision.idempotency_key == idempotency_key
            )
        ).scalar_one_or_none()

    existing = _existing_by_key()
    if existing is None:
        # Lock the candidate so a concurrent supersession (which UPDATEs this
        # row) or a concurrent decision serializes against us instead of
        # racing. The lock also proves the candidate exists.
        candidate = session.execute(
            select(EnrichmentCandidate)
            .where(EnrichmentCandidate.id == candidate_id)
            .with_for_update()
        ).scalar_one_or_none()
        if candidate is None:
            raise StaleCandidateError(candidate_id, "no such candidate")
        if candidate.superseded_by_producer_run_id is not None:
            raise StaleCandidateError(
                candidate_id,
                "superseded by a newer producer run"
                f" ({candidate.superseded_by_producer_run_id})",
            )
        prior = session.execute(
            select(ProfileReviewDecision).where(
                ProfileReviewDecision.candidate_id == candidate_id
            )
        ).scalar_one_or_none()
        if prior is not None:
            raise StaleCandidateError(
                candidate_id, f"already decided ({prior.decision})"
            )
        row = ProfileReviewDecision(
            candidate_id=candidate_id,
            decision=decision.value,
            operator=operator,
            note=note,
            idempotency_key=idempotency_key,
        )
        try:
            # Savepoint, not a bare flush: losing the race must not roll back
            # the caller's enclosing transaction (ledger.py pattern).
            with session.begin_nested():
                session.add(row)
        except IntegrityError:
            # Either UNIQUE can have lost a race: the idempotency key (adopt
            # an identical replay) or the one-decision-per-candidate key
            # (someone else decided while we looked).
            existing = _existing_by_key()
            if existing is None:
                concurrent = session.execute(
                    select(ProfileReviewDecision).where(
                        ProfileReviewDecision.candidate_id == candidate_id
                    )
                ).scalar_one_or_none()
                if concurrent is not None:
                    raise StaleCandidateError(
                        candidate_id, f"already decided ({concurrent.decision})"
                    ) from None
                raise
        else:
            return row
    if _payload_matches(existing, candidate_id, decision, operator, note):
        return existing
    raise ConflictingReplayError(idempotency_key)
