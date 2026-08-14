"""Roster curation (issue #7): rename, merge, archive/restore, embedding removal.

Pins the lifecycle invariants: the append-only ledger is never rewritten, merge
repoints the mutable side and canonicalizes at read time, archive purges stale
machine grounding, and matching only ever sees active speakers.
"""

import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

# Reuse the enrollment suite's run/turn builders — same eligibility rules.
from tests.integration.test_speaker_enrollment import (
    GATES,
    SPACE,
    add_turn,
    make_completed_run,
    unit,
)
from voxint.adjudication.enrollment import EnrollmentError, enroll_new_speaker
from voxint.adjudication.resolver import Resolution, label_states
from voxint.db.models import (
    AdjudicationDecision,
    Speaker,
    SpeakerAssignment,
    SpeakerEmbedding,
)
from voxint.speakers.matching import (
    CosineProposal,
    MatchingGates,
    _roster_centroids,
    replace_run_proposals,
)
from voxint.speakers.roster import (
    RosterConflictError,
    RosterError,
    RosterNotFoundError,
    active_speakers,
    archive_speaker,
    canonicalize,
    delete_embedding,
    merge_map,
    merge_speakers,
    rename_speaker,
    restore_speaker,
    roster_overview,
)


def enroll(
    session: Session,
    run_id: uuid.UUID,
    label: str,
    name: str,
    *,
    key: str | None = None,
) -> uuid.UUID:
    result = enroll_new_speaker(
        session,
        run_id=run_id,
        diarization_label=label,
        display_name=name,
        operator="ben",
        idempotency_key=key or f"k-{uuid.uuid4()}",
        gates=GATES,
    )
    return result.speaker_id


def seed_two_speakers(session: Session) -> tuple[uuid.UUID, uuid.UUID, uuid.UUID]:
    """One run, two enrolled speakers (Alice from S0, Bob from S1)."""
    run_id = make_completed_run(session)
    add_turn(session, run_id, 0, "S0", vector=unit(0))
    add_turn(session, run_id, 1, "S0", vector=unit(0))
    add_turn(session, run_id, 2, "S1", vector=unit(1))
    add_turn(session, run_id, 3, "S1", vector=unit(1))
    session.commit()
    alice = enroll(session, run_id, "S0", "Alice")
    bob = enroll(session, run_id, "S1", "Bob")
    session.commit()
    return run_id, alice, bob


def add_cosine_assignment(
    session: Session, run_id: uuid.UUID, label: str, speaker_id: uuid.UUID
) -> None:
    replace_run_proposals(
        session,
        run_id,
        (
            CosineProposal(
                diarization_label=label,
                speaker_id=speaker_id,
                similarity=0.95,
                margin=0.5,
                vote_agreement=1.0,
                grounded=True,
            ),
        ),
        (),
    )
    session.flush()


# --- rename ---


def test_rename_updates_name_and_history_renders_new_name(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id, alice, _ = seed_two_speakers(session)
        rename_speaker(session, alice, "  Alice Verified  ")
        session.commit()
        speaker = session.get(Speaker, alice)
        assert speaker is not None and speaker.display_name == "Alice Verified"
        # The historical assign decision now renders under the new name —
        # decisions store ids, never names.
        by_label = {s.label: s for s in label_states(session, run_id)}
        assert by_label["S0"].speaker_name == "Alice Verified"


def test_rename_collision_and_validation(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        _, alice, _ = seed_two_speakers(session)
        with pytest.raises(RosterConflictError, match="already exists"):
            rename_speaker(session, alice, "Bob")
        with pytest.raises(RosterError, match="1-120"):
            rename_speaker(session, alice, "   ")
        with pytest.raises(RosterNotFoundError):
            rename_speaker(session, uuid.uuid4(), "Nobody")
        # No-op rename is fine.
        assert rename_speaker(session, alice, "Alice").display_name == "Alice"


def test_rename_rejects_inactive_and_guides_to_owner_state(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        _, alice, bob = seed_two_speakers(session)
        merge_speakers(session, bob, alice)
        session.commit()
        with pytest.raises(RosterError, match="only active"):
            rename_speaker(session, bob, "Robert")
        # A name owned by a tombstone stays reserved, with lifecycle guidance.
        with pytest.raises(RosterConflictError, match="merged"):
            rename_speaker(session, alice, "Bob")


# --- merge ---


def test_merge_repoints_embeddings_and_assignments_ledger_untouched(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id, alice, bob = seed_two_speakers(session)
        add_cosine_assignment(session, run_id, "S1", bob)
        ledger_before = session.execute(
            select(
                AdjudicationDecision.id,
                AdjudicationDecision.speaker_id,
                AdjudicationDecision.created_at,
            ).order_by(AdjudicationDecision.created_at)
        ).all()
        session.commit()

        result = merge_speakers(session, bob, alice)
        session.commit()

        assert not result.already_merged
        assert result.embeddings_moved == 1
        assert result.assignments_moved == 1
        # Mutable side repointed…
        assert {
            row
            for row in session.execute(
                select(SpeakerEmbedding.speaker_id).distinct()
            ).scalars()
        } == {alice}
        assert session.execute(
            select(SpeakerAssignment.speaker_id).distinct()
        ).scalar_one() == alice
        # …the ledger byte-identical (append-only trigger never even fires)…
        ledger_after = session.execute(
            select(
                AdjudicationDecision.id,
                AdjudicationDecision.speaker_id,
                AdjudicationDecision.created_at,
            ).order_by(AdjudicationDecision.created_at)
        ).all()
        assert ledger_after == ledger_before
        # …and the tombstone canonicalizes.
        bob_row = session.get(Speaker, bob)
        assert bob_row is not None
        assert bob_row.merged_into_id == alice and bob_row.merged_at is not None
        assert canonicalize(bob, merge_map(session)) == alice


def test_merge_resolves_historical_decision_to_target_name(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id, alice, bob = seed_two_speakers(session)
        merge_speakers(session, bob, alice)
        session.commit()
        by_label = {s.label: s for s in label_states(session, run_id)}
        # S1 was human-assigned to Bob; the merge renders it as Alice now.
        assert by_label["S1"].resolution is Resolution.HUMAN_ASSIGN
        assert by_label["S1"].speaker_id == alice
        assert by_label["S1"].speaker_name == "Alice"
        # The immutable reference is untouched.
        decision = by_label["S1"].effective_decision
        assert decision is not None and decision.speaker_id == bob


def test_merge_collapses_chains_to_depth_one(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id, alice, bob = seed_two_speakers(session)
        add_turn(session, run_id, 4, "S2", vector=unit(2))
        add_turn(session, run_id, 5, "S2", vector=unit(2))
        session.commit()
        carol = enroll(session, run_id, "S2", "Carol")
        session.commit()
        merge_speakers(session, carol, bob)
        result = merge_speakers(session, bob, alice)
        session.commit()
        assert result.aliases_collapsed == 1
        carol_row = session.get(Speaker, carol)
        bob_row = session.get(Speaker, bob)
        assert carol_row is not None and carol_row.merged_into_id == alice
        assert bob_row is not None and bob_row.merged_into_id == alice


def test_merge_replay_and_conflicts(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id, alice, bob = seed_two_speakers(session)
        add_turn(session, run_id, 4, "S2", vector=unit(2))
        add_turn(session, run_id, 5, "S2", vector=unit(2))
        session.commit()
        carol = enroll(session, run_id, "S2", "Carol")
        session.commit()

        with pytest.raises(RosterError, match="itself"):
            merge_speakers(session, alice, alice)

        merge_speakers(session, bob, alice)
        session.commit()
        # Replay of the completed merge is a success no-op.
        assert merge_speakers(session, bob, alice).already_merged
        # Merged elsewhere is a stale form.
        with pytest.raises(RosterConflictError, match="different speaker"):
            merge_speakers(session, bob, carol)
        # Merging INTO a tombstone is a stale form.
        with pytest.raises(RosterConflictError, match="no longer an active"):
            merge_speakers(session, carol, bob)
        # An archived source must be restored first.
        archive_speaker(session, carol)
        with pytest.raises(RosterError, match="restore"):
            merge_speakers(session, carol, alice)


# --- archive / restore ---


def test_archive_excludes_from_matching_and_purges_assignments(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id, alice, bob = seed_two_speakers(session)
        add_cosine_assignment(session, run_id, "S1", bob)
        session.commit()

        deleted = archive_speaker(session, bob)
        session.commit()

        assert deleted == 1
        assert session.execute(select(SpeakerAssignment.id)).first() is None
        # Out of the matching roster and the assignable list…
        assert bob not in _roster_centroids(session, SPACE)
        assert alice in _roster_centroids(session, SPACE)
        assert {s.id for s in active_speakers(session)} == {alice}
        # …but human attribution still renders the historical name.
        by_label = {s.label: s for s in label_states(session, run_id)}
        assert by_label["S1"].resolution is Resolution.HUMAN_ASSIGN
        assert by_label["S1"].speaker_name == "Bob"
        # Replayed archive is a no-op, not an error.
        assert archive_speaker(session, bob) == 0

        restore_speaker(session, bob)
        session.commit()
        assert bob in _roster_centroids(session, SPACE)
        # Purged machine assignments are NOT resurrected.
        assert session.execute(select(SpeakerAssignment.id)).first() is None


def test_archive_and_restore_reject_tombstones(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        _, alice, bob = seed_two_speakers(session)
        merge_speakers(session, bob, alice)
        with pytest.raises(RosterError, match="tombstone"):
            archive_speaker(session, bob)
        with pytest.raises(RosterError, match="cannot be restored"):
            restore_speaker(session, bob)


def test_archived_name_blocks_reenrollment_with_guidance(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id, _, bob = seed_two_speakers(session)
        add_turn(session, run_id, 4, "S2", vector=unit(2))
        add_turn(session, run_id, 5, "S2", vector=unit(2))
        session.commit()
        archive_speaker(session, bob)
        session.commit()
        # Global uniqueness holds across the lifecycle: re-creating "Bob" is
        # refused with restore guidance instead of minting a duplicate identity.
        with pytest.raises(EnrollmentError, match="archived — restore"):
            enroll(session, run_id, "S2", "Bob")


# --- embedding removal ---


def test_delete_embedding_keeps_ledger_and_blocks_rematch(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id, alice, bob = seed_two_speakers(session)
        add_cosine_assignment(session, run_id, "S1", bob)
        embedding_id = session.execute(
            select(SpeakerEmbedding.id).where(SpeakerEmbedding.speaker_id == bob)
        ).scalar_one()
        decision_count = len(
            session.execute(select(AdjudicationDecision.id)).all()
        )
        session.commit()

        removal = delete_embedding(session, bob, embedding_id)
        session.commit()

        assert removal.assignments_deleted == 1
        assert removal.remaining_in_space == 0
        # The centroid row is gone; the decision that minted it survives.
        assert session.get(SpeakerEmbedding, embedding_id) is None
        assert (
            len(session.execute(select(AdjudicationDecision.id)).all())
            == decision_count
        )
        # With no embeddings left, Bob drops out of the matching roster.
        assert bob not in _roster_centroids(session, SPACE)
        assert alice in _roster_centroids(session, SPACE)


def test_delete_embedding_replayed_enrollment_does_not_remint(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = make_completed_run(session)
        add_turn(session, run_id, 0, "S0", vector=unit(0))
        add_turn(session, run_id, 1, "S0", vector=unit(0))
        session.commit()
        original = enroll_new_speaker(
            session,
            run_id=run_id,
            diarization_label="S0",
            display_name="Alice",
            operator="ben",
            idempotency_key="k-remint",
            gates=GATES,
        )
        session.commit()
        embedding_id = session.execute(
            select(SpeakerEmbedding.id).where(
                SpeakerEmbedding.speaker_id == original.speaker_id
            )
        ).scalar_one()
        delete_embedding(session, original.speaker_id, embedding_id)
        session.commit()
        # The replay hits the idempotency check and returns the original
        # outcome — it never re-mints the deleted centroid.
        replayed = enroll_new_speaker(
            session,
            run_id=run_id,
            diarization_label="S0",
            display_name="Alice",
            operator="ben",
            idempotency_key="k-remint",
            gates=GATES,
        )
        assert replayed.created_speaker is False
        assert (
            session.execute(select(SpeakerEmbedding.id)).first() is None
        )


def test_delete_embedding_ownership_check(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        _, alice, bob = seed_two_speakers(session)
        bob_embedding = session.execute(
            select(SpeakerEmbedding.id).where(SpeakerEmbedding.speaker_id == bob)
        ).scalar_one()
        with pytest.raises(RosterNotFoundError, match="embedding"):
            delete_embedding(session, alice, bob_embedding)


# --- overview + canonicalization plumbing ---


def test_roster_overview_partitions_and_describes(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id, alice, bob = seed_two_speakers(session)
        add_turn(session, run_id, 4, "S2", vector=unit(2))
        add_turn(session, run_id, 5, "S2", vector=unit(2))
        session.commit()
        carol = enroll(session, run_id, "S2", "Carol")
        session.commit()
        merge_speakers(session, bob, alice)
        archive_speaker(session, carol)
        session.commit()

        overview = roster_overview(session)
        assert [e.speaker.id for e in overview.active] == [alice]
        assert {e.speaker.id for e in overview.inactive} == {bob, carol}
        alice_entry = overview.active[0]
        # Bob's embedding moved to Alice in the merge: 2 centroids, provenance intact.
        assert len(alice_entry.embeddings) == 2
        assert {e.embedding_space for e in alice_entry.embeddings} == {SPACE}
        assert all(len(e.vector) > 0 for e in alice_entry.embeddings)
        bob_entry = next(e for e in overview.inactive if e.speaker.id == bob)
        assert bob_entry.merged_into_name == "Alice"
        carol_entry = next(e for e in overview.inactive if e.speaker.id == carol)
        assert carol_entry.merged_into_name is None
        assert carol_entry.speaker.deleted_at is not None


def test_canonicalize_fails_loudly_on_cycle() -> None:
    a, b = uuid.uuid4(), uuid.uuid4()
    with pytest.raises(RosterError, match="cycle"):
        canonicalize(a, {a: b, b: a})


def test_matching_gates_still_default() -> None:
    # Guard: the module-level GATES fixture must stay at defaults — several
    # geometry assertions above depend on the grounded thresholds.
    assert MatchingGates() == GATES
