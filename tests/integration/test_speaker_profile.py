"""speaker_profiles write path (issue #159): materialize-on-accept in the
decision funnel, manual edits, replay safety, merge repointing, reconcile.

The acceptance round-trip the issue pins: accept a draft claim, see the
provenance-tagged field; a manual edit overrides it; the draft-claim history
(candidate + decision rows) is never touched; an idempotent replay of the old
accept can NOT reverse the later manual edit.
"""

import threading
import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy.orm import Session, sessionmaker

from voxint.db.models import (
    EnrichmentCandidate,
    EnrichmentProducerRun,
    ProfileDecision,
    ProfileReviewDecision,
    Speaker,
    SpeakerProfile,
)
from voxint.enrichment.review import record_profile_decision
from voxint.speakers.profile import (
    ProfileFieldError,
    clear_profile_field,
    profile_for,
    reconcile_speaker_profiles,
    set_profile_field,
)
from voxint.speakers.roster import merge_speakers


def _speaker(session: Session, name: str) -> uuid.UUID:
    speaker = Speaker(display_name=name)
    session.add(speaker)
    session.flush()
    return speaker.id


def _candidate(
    session: Session,
    speaker_id: uuid.UUID,
    field: str,
    value: str,
) -> uuid.UUID:
    now = datetime.now(UTC)
    run = EnrichmentProducerRun(
        producer="test-producer",
        producer_version="1",
        target_kind="speaker",
        speaker_id=speaker_id,
        covered_fields=[field],
        generation=1,
        outcome="found",
        idempotency_key=f"prod-{uuid.uuid4()}",
        started_at=now,
        completed_at=now,
    )
    session.add(run)
    session.flush()
    cand = EnrichmentCandidate(
        producer_run_id=run.id,
        target_kind="speaker",
        speaker_id=speaker_id,
        field=field,
        value=value,
    )
    session.add(cand)
    session.flush()
    return cand.id


def _accept(
    session: Session, candidate_id: uuid.UUID, *, key: str, operator: str = "op"
) -> ProfileReviewDecision:
    return record_profile_decision(
        session,
        candidate_id=candidate_id,
        decision=ProfileDecision.ACCEPT,
        operator=operator,
        idempotency_key=key,
    )


def test_accept_materializes_then_manual_edit_overrides_history_intact(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        speaker = _speaker(session, "Alice")
        cand = _candidate(session, speaker, "bio", "Wrote the book on birds.")
        _accept(session, cand, key="k-accept")
        session.commit()

        profile = profile_for(session, speaker)
        assert profile["bio"].value == "Wrote the book on birds."
        assert profile["bio"].provenance == "enrichment"
        assert profile["bio"].accepted_candidate_id == cand

        set_profile_field(
            session, speaker_id=speaker, field="bio", value="Ornithologist.", operator="ben"
        )
        session.commit()
        profile = profile_for(session, speaker)
        assert profile["bio"].value == "Ornithologist."
        assert profile["bio"].provenance == "manual"
        assert profile["bio"].accepted_candidate_id is None
        assert profile["bio"].operator == "ben"

        # History preserved: the candidate and its decision are untouched.
        assert session.get(EnrichmentCandidate, cand).value == "Wrote the book on birds."
        decisions = session.query(ProfileReviewDecision).filter_by(candidate_id=cand).all()
        assert [d.decision for d in decisions] == ["accept"]

        # Idempotent REPLAY of the old accept must not reverse the manual edit.
        _accept(session, cand, key="k-accept")
        session.commit()
        assert profile_for(session, speaker)["bio"].value == "Ornithologist."


def test_fresh_accept_overwrites_older_manual_value(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        speaker = _speaker(session, "Bella")
        set_profile_field(
            session, speaker_id=speaker, field="affiliation", value="Old Corp", operator="ben"
        )
        session.commit()
        cand = _candidate(session, speaker, "affiliation", "New Institute")
        _accept(session, cand, key="k-new")
        session.commit()
        row = profile_for(session, speaker)["affiliation"]
        assert row.value == "New Institute"
        assert row.provenance == "enrichment"


def test_name_and_reject_never_materialize(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        speaker = _speaker(session, "Cara")
        name_cand = _candidate(session, speaker, "name", "Dr. Cara Grey")
        _accept(session, name_cand, key="k-name")
        bio_cand = _candidate(session, speaker, "bio", "A bio.")
        record_profile_decision(
            session,
            candidate_id=bio_cand,
            decision=ProfileDecision.REJECT,
            operator="op",
            idempotency_key="k-reject",
        )
        session.commit()
        assert profile_for(session, speaker) == {}
        assert session.query(SpeakerProfile).count() == 0


def test_accept_on_tombstone_lands_under_canonical(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        a = _speaker(session, "A")
        b = _speaker(session, "B")
        cand = _candidate(session, a, "bio", "bio via A")
        merge_speakers(session, a, b)
        session.commit()
        _accept(session, cand, key="k-a")
        session.commit()
        rows = session.query(SpeakerProfile).all()
        assert len(rows) == 1
        assert rows[0].speaker_id == b
        # Alias-aware read works from either id.
        assert profile_for(session, a)["bio"].value == "bio via A"
        assert profile_for(session, b)["bio"].value == "bio via A"


def test_merge_repoints_profiles_target_wins_conflicts(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        src = _speaker(session, "Src")
        dst = _speaker(session, "Dst")
        set_profile_field(session, speaker_id=src, field="bio", value="src bio", operator="op")
        set_profile_field(session, speaker_id=src, field="link", value="src link", operator="op")
        set_profile_field(session, speaker_id=dst, field="bio", value="dst bio", operator="op")
        session.commit()
        merge_speakers(session, src, dst)
        session.commit()
        rows = {(r.speaker_id, r.field): r.value for r in session.query(SpeakerProfile).all()}
        # Conflict: target's bio wins; the source's unique link follows the identity.
        assert rows == {(dst, "bio"): "dst bio", (dst, "link"): "src link"}


def test_clear_profile_field(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        speaker = _speaker(session, "Dee")
        set_profile_field(session, speaker_id=speaker, field="bio", value="x", operator="op")
        session.commit()
        assert clear_profile_field(session, speaker_id=speaker, field="bio", operator="op")
        session.commit()
        assert profile_for(session, speaker) == {}
        assert not clear_profile_field(session, speaker_id=speaker, field="bio", operator="op")


def test_validation_rejects_bad_input(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        speaker = _speaker(session, "Eve")
        with pytest.raises(ProfileFieldError):
            set_profile_field(
                session, speaker_id=speaker, field="name", value="x", operator="op"
            )
        with pytest.raises(ProfileFieldError):
            set_profile_field(
                session, speaker_id=speaker, field="bio", value="   ", operator="op"
            )
        with pytest.raises(ProfileFieldError):
            set_profile_field(
                session, speaker_id=speaker, field="bio", value="x", operator=" "
            )


def test_reconcile_materializes_funnel_bypassing_accepts(
    session_factory: sessionmaker[Session],
) -> None:
    """A decision appended WITHOUT the funnel (pre-0041 binary) is repaired,
    and an already-materialized or manually-owned field is left alone."""
    with session_factory() as session:
        speaker = _speaker(session, "Fay")
        orphan = _candidate(session, speaker, "bio", "orphan bio")
        # Simulate the legacy writer: decision row only, no materialization.
        session.add(
            ProfileReviewDecision(
                candidate_id=orphan,
                decision="accept",
                operator="legacy",
                note=None,
                idempotency_key="legacy-key",
            )
        )
        manual_owner = _speaker(session, "Gus")
        owned = _candidate(session, manual_owner, "bio", "should not surface")
        session.add(
            ProfileReviewDecision(
                candidate_id=owned,
                decision="accept",
                operator="legacy",
                note=None,
                idempotency_key="legacy-key-2",
            )
        )
        set_profile_field(
            session, speaker_id=manual_owner, field="bio", value="manual wins", operator="ben"
        )
        session.commit()

        created = reconcile_speaker_profiles(session)
        session.commit()
        assert created == 1
        assert profile_for(session, speaker)["bio"].value == "orphan bio"
        assert profile_for(session, speaker)["bio"].operator == "legacy"
        assert profile_for(session, manual_owner)["bio"].value == "manual wins"
        # Idempotent: a second pass creates nothing.
        assert reconcile_speaker_profiles(session) == 0


def test_concurrent_accept_serializes_on_speaker_lock(
    session_factory: sessionmaker[Session],
) -> None:
    """While one session holds the speaker's profile write lock, a concurrent
    accept blocks; after the first commits, the accept lands as the newest act
    and wins the field."""
    with session_factory() as session:
        speaker = _speaker(session, "Hal")
        cand = _candidate(session, speaker, "bio", "accepted bio")
        session.commit()

    holder = session_factory()
    set_profile_field(
        holder, speaker_id=speaker, field="bio", value="manual first", operator="ben"
    )
    # Lock held (uncommitted). The accept in another session must block.
    done = threading.Event()

    def _concurrent_accept() -> None:
        with session_factory() as other:
            _accept(other, cand, key="k-race")
            other.commit()
        done.set()

    worker = threading.Thread(target=_concurrent_accept)
    worker.start()
    assert not done.wait(timeout=1.0), "accept should block on the speaker lock"
    holder.commit()
    holder.close()
    worker.join(timeout=10)
    assert done.is_set()

    with session_factory() as session:
        row = profile_for(session, speaker)["bio"]
        assert row.value == "accepted bio"
        assert row.provenance == "enrichment"


def test_clear_survives_identical_accept_replay(
    session_factory: sessionmaker[Session],
) -> None:
    """accept -> clear -> identical replay: the clear tombstone stands, the
    cleared value is never resurrected (#159 pre-landing review, codex)."""
    with session_factory() as session:
        speaker = _speaker(session, "Hana")
        cand = _candidate(session, speaker, "bio", "Old accepted bio.")
        _accept(session, cand, key="k-hana")
        session.commit()
        assert profile_for(session, speaker)["bio"].value == "Old accepted bio."

        clear_profile_field(session, speaker_id=speaker, field="bio", operator="ben")
        session.commit()
        assert "bio" not in profile_for(session, speaker)
        # The tombstone row is the durable marker of the clearing act.
        row = session.query(SpeakerProfile).filter_by(speaker_id=speaker).one()
        assert row.value is None
        assert row.provenance == "manual"
        assert row.accepted_candidate_id is None

        # Identical replay (same idempotency key, e.g. a stale tab resubmit).
        _accept(session, cand, key="k-hana")
        session.commit()
        assert "bio" not in profile_for(session, speaker)


def test_clear_survives_reconcile(session_factory: sessionmaker[Session]) -> None:
    """accept -> clear -> reconcile: the repair pass sees the tombstone as a
    later manual act and creates nothing (#159 pre-landing review, codex)."""
    with session_factory() as session:
        speaker = _speaker(session, "Iris")
        cand = _candidate(session, speaker, "affiliation", "Old Institute")
        _accept(session, cand, key="k-iris")
        clear_profile_field(
            session, speaker_id=speaker, field="affiliation", operator="ben"
        )
        session.commit()
        assert reconcile_speaker_profiles(session) == 0
        session.commit()
        assert "affiliation" not in profile_for(session, speaker)


def test_fresh_accept_overrides_a_clear(session_factory: sessionmaker[Session]) -> None:
    """A NEW accept after a clear is a later operator act: it overwrites the
    tombstone (newest act wins), exactly like it overwrites a manual value."""
    with session_factory() as session:
        speaker = _speaker(session, "Jude")
        first = _candidate(session, speaker, "bio", "First bio.")
        _accept(session, first, key="k-jude-1")
        clear_profile_field(session, speaker_id=speaker, field="bio", operator="ben")
        session.commit()
        second = _candidate(session, speaker, "bio", "Second bio.")
        _accept(session, second, key="k-jude-2")
        session.commit()
        row = profile_for(session, speaker)["bio"]
        assert row.value == "Second bio."
        assert row.provenance == "enrichment"
