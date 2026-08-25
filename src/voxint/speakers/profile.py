"""Speaker-profile reads, manual edits, and reconciliation (issue #159).

The profile page's API over ``speaker_profiles``. Writes serialize on the same
canonical ``speakers`` row lock the accept-materialization uses
(``enrichment.review.lock_canonical_speaker``), so a manual edit, a concurrent
accept, and a merge can never interleave on one speaker's profile. Reads are
alias-aware: rows normally live under the canonical id (writes canonicalize,
and ``roster.merge_speakers`` repoints on merge), but a row stranded under a
tombstone by historical drift is still surfaced — canonical row wins a
per-field conflict, else the newest ``updated_at``.
"""

import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from voxint.db.models import (
    PROFILE_FIELDS,
    EnrichmentCandidate,
    ProfileDecision,
    ProfileProvenance,
    ProfileReviewDecision,
    SpeakerProfile,
)
from voxint.enrichment.review import (
    MAX_OPERATOR_CHARS,
    lock_canonical_speaker,
)
from voxint.speakers.roster import alias_ids, canonicalize, merge_map

MAX_VALUE_CHARS = 4_000  # mirrors the speaker_profiles / enrichment_candidates CHECK

_FIELD_VALUES = tuple(f.value for f in PROFILE_FIELDS)


class ProfileFieldError(ValueError):
    """A manual profile edit that cannot be stored (bad field/value/operator)."""


def _validate_field(field: str) -> None:
    if field not in _FIELD_VALUES:
        raise ProfileFieldError(
            f"unknown profile field {field!r} (expected one of {', '.join(_FIELD_VALUES)})"
        )


def _validate_operator(operator: str) -> None:
    if not operator.strip() or len(operator) > MAX_OPERATOR_CHARS:
        raise ProfileFieldError(f"operator empty or over {MAX_OPERATOR_CHARS} chars")


def profile_for(session: Session, speaker_id: uuid.UUID) -> dict[str, SpeakerProfile]:
    """The speaker's current profile, keyed by field, alias-aware.

    Looks across every id that canonicalizes into this speaker (the input may
    itself be a tombstone). On the rare per-field conflict between an alias row
    and a canonical row, the canonical row wins; between two alias rows, the
    newest ``updated_at`` (then id, deterministic) wins.
    """
    aliases = alias_ids(session, speaker_id)
    canonical = canonicalize(speaker_id, merge_map(session))
    rows = (
        session.execute(
            select(SpeakerProfile)
            .where(SpeakerProfile.speaker_id.in_(aliases))
            .order_by(SpeakerProfile.updated_at.desc(), SpeakerProfile.id.desc())
        )
        .scalars()
        .all()
    )
    out: dict[str, SpeakerProfile] = {}
    for row in rows:
        held = out.get(row.field)
        # First (newest) row wins, unless a later canonical row displaces a
        # non-canonical holder.
        if held is None or (held.speaker_id != canonical and row.speaker_id == canonical):
            out[row.field] = row
    return out


def set_profile_field(
    session: Session,
    *,
    speaker_id: uuid.UUID,
    field: str,
    value: str,
    operator: str,
) -> SpeakerProfile:
    """Manually set one profile field (provenance ``manual``).

    Overwrites whatever holds the field — including an enrichment-accepted
    value, whose draft-claim history stays fully recoverable from the decision
    trail. The write lands under the canonical id, under the speaker lock.
    """
    _validate_field(field)
    _validate_operator(operator)
    cleaned = value.strip()
    if not cleaned or len(cleaned) > MAX_VALUE_CHARS:
        raise ProfileFieldError(f"value empty or over {MAX_VALUE_CHARS} chars")
    canonical = lock_canonical_speaker(session, speaker_id)
    row = session.execute(
        select(SpeakerProfile).where(
            SpeakerProfile.speaker_id == canonical, SpeakerProfile.field == field
        )
    ).scalar_one_or_none()
    if row is None:
        row = SpeakerProfile(
            speaker_id=canonical,
            field=field,
            value=cleaned,
            provenance=ProfileProvenance.MANUAL.value,
            accepted_candidate_id=None,
            operator=operator,
        )
        session.add(row)
    else:
        row.value = cleaned
        row.provenance = ProfileProvenance.MANUAL.value
        row.accepted_candidate_id = None
        row.operator = operator
        row.updated_at = func.now()
    session.flush()
    return row


def clear_profile_field(
    session: Session, *, speaker_id: uuid.UUID, field: str, operator: str
) -> bool:
    """Remove one profile field entirely (a deliberate manual act).

    Returns whether a row existed. The clearing act itself leaves no profile
    row to carry provenance — accepted-claim history stays in the decision
    trail, which is the durable record. Alias rows for the field are removed
    too, so a cleared field cannot resurrect through a stranded tombstone row.
    """
    _validate_field(field)
    _validate_operator(operator)
    lock_canonical_speaker(session, speaker_id)
    aliases = alias_ids(session, speaker_id)
    rows = (
        session.execute(
            select(SpeakerProfile).where(
                SpeakerProfile.speaker_id.in_(aliases), SpeakerProfile.field == field
            )
        )
        .scalars()
        .all()
    )
    for row in rows:
        session.delete(row)
    session.flush()
    return bool(rows)


def reconcile_speaker_profiles(session: Session) -> int:
    """Idempotent repair: materialize accepted claims that never reached
    ``speaker_profiles`` (a crash between decision and profile write, or a
    decision recorded by a pre-0041 binary). Fills ABSENT (canonical, field)
    rows only — an existing row is a later or equal act and is never touched.
    Returns the number of rows created.
    """
    rows = session.execute(
        select(ProfileReviewDecision, EnrichmentCandidate)
        .join(
            EnrichmentCandidate,
            EnrichmentCandidate.id == ProfileReviewDecision.candidate_id,
        )
        .where(
            ProfileReviewDecision.decision == ProfileDecision.ACCEPT.value,
            EnrichmentCandidate.speaker_id.is_not(None),
            EnrichmentCandidate.field.in_(_FIELD_VALUES),
        )
        .order_by(ProfileReviewDecision.created_at.desc(), ProfileReviewDecision.id.desc())
    ).all()
    created = 0
    seen: set[tuple[uuid.UUID, str]] = set()
    for decision, candidate in rows:
        assert candidate.speaker_id is not None  # WHERE narrows
        canonical = lock_canonical_speaker(session, candidate.speaker_id)
        key = (canonical, candidate.field)
        if key in seen:
            continue  # a newer accepted decision already owns this field
        seen.add(key)
        existing = session.execute(
            select(SpeakerProfile).where(
                SpeakerProfile.speaker_id == canonical,
                SpeakerProfile.field == candidate.field,
            )
        ).scalar_one_or_none()
        if existing is not None:
            continue
        session.add(
            SpeakerProfile(
                speaker_id=canonical,
                field=candidate.field,
                value=candidate.value,
                provenance=ProfileProvenance.ENRICHMENT.value,
                accepted_candidate_id=candidate.id,
                operator=decision.operator,
            )
        )
        created += 1
    session.flush()
    return created
