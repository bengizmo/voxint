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

Accepting a speaker-scoped ``bio``/``affiliation``/``link`` claim ALSO
materializes the value into ``speaker_profiles`` (issue #159): the decision
funnel is the single writer, so every caller — roster page, or any future one
— keeps the profile consistent with the trail. Materialization runs under a
canonical ``speakers`` row lock (taken BEFORE the candidate lock, one
consistent order with ``speakers/profile.py``); a fresh accept overwrites the
field (the newest operator act wins, even over an older manual edit — the
displaced value stays recoverable from the trail), while an idempotent REPLAY
only fills an absent row or one already referencing the same candidate, so a
retry can never reverse a LATER manual edit.
"""

import uuid

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from voxint.db.models import (
    PROFILE_FIELDS,
    EnrichmentCandidate,
    ProfileDecision,
    ProfileProvenance,
    ProfileReviewDecision,
    Speaker,
    SpeakerProfile,
)
from voxint.speakers.roster import canonicalize, merge_map

MAX_OPERATOR_CHARS = 200
MAX_NOTE_CHARS = 2_000


class ConflictingReplayError(Exception):
    def __init__(self, idempotency_key: str) -> None:
        super().__init__(
            "this action conflicts with a previous submission — refresh the page and try again"
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


def _materializable(candidate: EnrichmentCandidate) -> bool:
    """True iff accepting this candidate writes a ``speaker_profiles`` row:
    a speaker-scoped profile field. NAME never materializes (identity stays on
    ``speakers.display_name``); run-scoped claims have no profile home."""
    return (
        candidate.speaker_id is not None
        and candidate.field in {f.value for f in PROFILE_FIELDS}
    )


def lock_canonical_speaker(session: Session, speaker_id: uuid.UUID) -> uuid.UUID:
    """Canonicalize through merge tombstones and take the canonical row's lock.

    THE serialization point for every ``speaker_profiles`` write (this module's
    materialize-on-accept and ``speakers/profile.py``'s manual edit/reconcile):
    one consistent order — speaker lock first, candidate lock second — so
    concurrent accepts, manual edits, and merges cannot deadlock or interleave
    on one speaker's profile. ``populate_existing`` refreshes an already-cached
    identity so post-lock reads see committed state, not a stale map entry.
    """
    current = speaker_id
    while True:
        canonical = canonicalize(current, merge_map(session))
        session.execute(
            select(Speaker)
            .where(Speaker.id == canonical)
            .with_for_update()
            .execution_options(populate_existing=True)
        ).scalar_one_or_none()
        # Re-check AFTER holding the lock: a merge committed between the map
        # read and the lock acquisition would have tombstoned this row — chase
        # the new target so a profile write never lands on a tombstone.
        post = canonicalize(canonical, merge_map(session))
        if post == canonical:
            return canonical
        current = post


def _upsert_profile_from_accept(
    session: Session,
    canonical_speaker_id: uuid.UUID,
    candidate: EnrichmentCandidate,
    operator: str,
    *,
    replay: bool,
) -> None:
    """Materialize an accepted claim into the current profile (issue #159).

    Caller holds the canonical speaker lock. A FRESH accept upserts
    unconditionally (the newest operator act wins the field). A REPLAY only
    fills an absent row or refreshes one already referencing this same
    candidate — never a row a later act (manual edit, other accept) now owns.
    """
    row = session.execute(
        select(SpeakerProfile).where(
            SpeakerProfile.speaker_id == canonical_speaker_id,
            SpeakerProfile.field == candidate.field,
        )
    ).scalar_one_or_none()
    if row is None:
        session.add(
            SpeakerProfile(
                speaker_id=canonical_speaker_id,
                field=candidate.field,
                value=candidate.value,
                provenance=ProfileProvenance.ENRICHMENT.value,
                accepted_candidate_id=candidate.id,
                operator=operator,
            )
        )
        session.flush()
        return
    if replay and row.accepted_candidate_id != candidate.id:
        return
    row.value = candidate.value
    row.provenance = ProfileProvenance.ENRICHMENT.value
    row.accepted_candidate_id = candidate.id
    row.operator = operator
    row.updated_at = func.now()
    session.flush()


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

    def _candidate_peek() -> EnrichmentCandidate | None:
        return session.execute(
            select(EnrichmentCandidate).where(EnrichmentCandidate.id == candidate_id)
        ).scalar_one_or_none()

    existing = _existing_by_key()
    if existing is None:
        # An accept that will materialize takes the canonical speaker lock
        # FIRST (see lock_canonical_speaker) — the unlocked peek only routes;
        # every materialization fact is re-read under the locks below.
        canonical_speaker: uuid.UUID | None = None
        if decision is ProfileDecision.ACCEPT:
            peek = _candidate_peek()
            if peek is not None and _materializable(peek):
                assert peek.speaker_id is not None  # _materializable; narrows
                canonical_speaker = lock_canonical_speaker(session, peek.speaker_id)
        # Lock the candidate so a concurrent supersession (which UPDATEs this
        # row) or a concurrent decision serializes against us instead of
        # racing. The lock also proves the candidate exists.
        candidate = session.execute(
            select(EnrichmentCandidate)
            .where(EnrichmentCandidate.id == candidate_id)
            .with_for_update()
            .execution_options(populate_existing=True)
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
            if decision is ProfileDecision.ACCEPT and _materializable(candidate):
                if canonical_speaker is None:
                    # The routing peek missed the candidate; take the speaker
                    # lock now. Rare enough that the inverted order (candidate
                    # lock already held) is acceptable — Postgres aborts one
                    # transaction on a true deadlock, and the replay repair
                    # re-materializes.
                    assert candidate.speaker_id is not None  # _materializable
                    canonical_speaker = lock_canonical_speaker(session, candidate.speaker_id)
                _upsert_profile_from_accept(
                    session, canonical_speaker, candidate, operator, replay=False
                )
            return row
    if _payload_matches(existing, candidate_id, decision, operator, note):
        # Replay repair: a prior accept whose materialization was lost (crash
        # between decision and profile write, or a decision recorded by a
        # pre-0041 binary) is filled in — but never over a LATER act.
        if decision is ProfileDecision.ACCEPT:
            candidate = _candidate_peek()
            if candidate is not None and _materializable(candidate):
                assert candidate.speaker_id is not None
                canonical = lock_canonical_speaker(session, candidate.speaker_id)
                _upsert_profile_from_accept(
                    session, canonical, candidate, existing.operator, replay=True
                )
        return existing
    raise ConflictingReplayError(idempotency_key)
