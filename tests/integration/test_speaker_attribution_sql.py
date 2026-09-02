"""``speaker_attributed_exists`` must agree with ``label_states`` — always.

The SQL predicate is a second definition of "which speaker does this label
belong to", so every test here computes the same answer twice: once through
the Python resolver (the ground truth the workbench renders) and once through
the SQL predicate, and asserts they agree on the tricky ledgers — newest
decision wins (including same-timestamp id tie-breaks), any decision
suppressing machine evidence, grounded-only cosine, merge-tombstone
canonicalization (via ``roster.alias_ids`` expansion), and label anchoring in
``diarization_turns``.
"""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from voxint.adjudication.resolver import label_states, speaker_attributed_exists
from voxint.db.models import (
    EMBEDDING_DIM,
    AdjudicationDecision,
    DiarizationTurn,
    MediaItem,
    PipelineRun,
    RunStatus,
    Speaker,
    SpeakerAssignment,
)
from voxint.speakers.roster import RosterError, alias_ids

SPACE = "titanet-large-v2"
BASE = datetime(2026, 8, 14, 12, 0, 0, tzinfo=UTC)


def _unit(dim: int) -> list[float]:
    vector = [0.0] * EMBEDDING_DIM
    vector[dim % EMBEDDING_DIM] = 1.0
    return vector


def make_speaker(session: Session, **cols: object) -> uuid.UUID:
    speaker = Speaker(display_name=f"spk-{uuid.uuid4().hex[:10]}", **cols)
    session.add(speaker)
    session.flush()
    return speaker.id


def make_run(session: Session, labels: tuple[str, ...] = ("SPEAKER_00",)) -> uuid.UUID:
    media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
    session.add(media)
    session.flush()
    run = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
    session.add(run)
    session.flush()
    for index, label in enumerate(labels):
        session.add(
            DiarizationTurn(
                pipeline_run_id=run.id,
                turn_index=index,
                start_seconds=float(index * 10),
                end_seconds=float(index * 10 + 8),
                label=label,
                embedding=_unit(index),
                embedding_space=SPACE,
            )
        )
    session.flush()
    return run.id


def add_decision(
    session: Session,
    run_id: uuid.UUID,
    label: str,
    decision: str,
    speaker_id: uuid.UUID | None = None,
    at: datetime = BASE,
    decision_id: uuid.UUID | None = None,
) -> None:
    session.add(
        AdjudicationDecision(
            id=decision_id or uuid.uuid4(),
            pipeline_run_id=run_id,
            diarization_label=label,
            decision=decision,
            speaker_id=speaker_id,
            operator="reviewer",
            idempotency_key=uuid.uuid4().hex,
            created_at=at,
        )
    )
    session.flush()


def add_cosine(
    session: Session,
    run_id: uuid.UUID,
    label: str,
    speaker_id: uuid.UUID,
    *,
    grounded: bool = True,
) -> None:
    session.add(
        SpeakerAssignment(
            pipeline_run_id=run_id,
            diarization_label=label,
            speaker_id=speaker_id,
            method="cosine",
            confidence=0.9,
            grounded=grounded,
        )
    )
    session.flush()


def sql_attributed(session: Session, target: uuid.UUID) -> set[uuid.UUID]:
    """Run ids the SQL predicate matches, expanding the target like the route."""
    ids = alias_ids(session, target)
    return set(
        session.execute(
            select(PipelineRun.id).where(speaker_attributed_exists(PipelineRun.id, ids))
        ).scalars()
    )


def resolver_attributed(session: Session, target: uuid.UUID) -> set[uuid.UUID]:
    """Ground truth: runs where label_states attributes (canonicalized) target."""
    run_ids = session.execute(select(PipelineRun.id)).scalars().all()
    return {
        run_id
        for run_id in run_ids
        if any(s.speaker_id == target for s in label_states(session, run_id))
    }


def assert_agreement(
    session: Session, target: uuid.UUID, expected: set[uuid.UUID]
) -> None:
    assert sql_attributed(session, target) == expected
    assert resolver_attributed(session, target) == expected


def test_effective_decision_wins_over_history(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        alice = make_speaker(session)
        bob = make_speaker(session)

        assigned_then_excluded = make_run(session)
        add_decision(session, assigned_then_excluded, "SPEAKER_00", "assign", alice)
        add_decision(
            session,
            assigned_then_excluded,
            "SPEAKER_00",
            "exclude",
            at=BASE + timedelta(minutes=1),
        )

        excluded_then_assigned = make_run(session)
        add_decision(session, excluded_then_assigned, "SPEAKER_00", "exclude")
        add_decision(
            session,
            excluded_then_assigned,
            "SPEAKER_00",
            "assign",
            alice,
            at=BASE + timedelta(minutes=1),
        )

        reassigned = make_run(session)
        add_decision(session, reassigned, "SPEAKER_00", "assign", alice)
        add_decision(
            session,
            reassigned,
            "SPEAKER_00",
            "assign",
            bob,
            at=BASE + timedelta(minutes=1),
        )

        unknown_after_assign = make_run(session)
        add_decision(session, unknown_after_assign, "SPEAKER_00", "assign", alice)
        add_decision(
            session,
            unknown_after_assign,
            "SPEAKER_00",
            "unknown",
            at=BASE + timedelta(minutes=1),
        )

        session.commit()
        assert_agreement(session, alice, {excluded_then_assigned})
        assert_agreement(session, bob, {reassigned})


def test_same_timestamp_id_tiebreak_matches_effective_decisions(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        alice = make_speaker(session)
        run = make_run(session)
        low = uuid.UUID(int=1)
        high = uuid.UUID(int=2**127)
        # Same created_at: the higher id is the effective row, per the
        # (created_at DESC, id DESC) order effective_decisions uses.
        add_decision(session, run, "SPEAKER_00", "assign", alice, decision_id=low)
        add_decision(session, run, "SPEAKER_00", "exclude", decision_id=high)
        session.commit()
        assert_agreement(session, alice, set())

    with session_factory() as session:
        alice = make_speaker(session)
        run = make_run(session)
        add_decision(
            session, run, "SPEAKER_00", "exclude", decision_id=uuid.UUID(int=3)
        )
        add_decision(
            session,
            run,
            "SPEAKER_00",
            "assign",
            alice,
            decision_id=uuid.UUID(int=2**127 + 1),
        )
        session.commit()
        assert_agreement(session, alice, {run})


def test_cosine_counts_only_grounded_and_undecided(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        alice = make_speaker(session)

        grounded_run = make_run(session)
        add_cosine(session, grounded_run, "SPEAKER_00", alice)

        ungrounded_run = make_run(session)
        add_cosine(session, ungrounded_run, "SPEAKER_00", alice, grounded=False)

        suppressed_run = make_run(session)
        add_cosine(session, suppressed_run, "SPEAKER_00", alice)
        add_decision(session, suppressed_run, "SPEAKER_00", "unknown")

        session.commit()
        assert_agreement(session, alice, {grounded_run})


def test_orphan_evidence_without_turn_is_ignored(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        alice = make_speaker(session)
        run = make_run(session, labels=("SPEAKER_00",))
        # Evidence for a label with no diarization turn is not a label of the
        # run — both definitions must ignore it.
        add_decision(session, run, "SPEAKER_99", "assign", alice)
        add_cosine(session, run, "SPEAKER_98", alice)
        session.commit()
        assert_agreement(session, alice, set())


def test_merge_tombstones_canonicalize_both_directions(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        target = make_speaker(session)
        now = datetime.now(tz=UTC)
        source = make_speaker(session, merged_into_id=target, merged_at=now)
        chained = make_speaker(session, merged_into_id=source, merged_at=now)

        historical = make_run(session)
        add_decision(session, historical, "SPEAKER_00", "assign", source)

        deep = make_run(session)
        add_decision(session, deep, "SPEAKER_00", "assign", chained)

        direct = make_run(session)
        add_decision(session, direct, "SPEAKER_00", "assign", target)

        session.commit()
        expected = {historical, deep, direct}
        # Searching the canonical target finds historical assigns to sources
        # at any chain depth; searching a stale tombstone id resolves to the
        # same set (alias_ids canonicalizes its input first).
        assert_agreement(session, target, expected)
        assert sql_attributed(session, source) == expected
        assert sql_attributed(session, chained) == expected


def test_archived_speaker_remains_attributable(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        archived = make_speaker(session, deleted_at=datetime.now(tz=UTC))
        run = make_run(session)
        add_decision(session, run, "SPEAKER_00", "assign", archived)
        session.commit()
        # Archive keeps human decisions effective — the facet offers archived
        # speakers (marked) precisely so these runs stay discoverable.
        assert_agreement(session, archived, {run})


def test_shared_labels_do_not_bleed_across_runs(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        alice = make_speaker(session)
        attributed = make_run(session)
        add_decision(session, attributed, "SPEAKER_00", "assign", alice)
        # Same label text on another run, with no evidence of its own.
        bystander = make_run(session)
        session.commit()
        assert bystander != attributed
        assert_agreement(session, alice, {attributed})


def test_multi_label_run_matches_on_any_label(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        alice = make_speaker(session)
        run = make_run(session, labels=("SPEAKER_00", "SPEAKER_01"))
        add_decision(session, run, "SPEAKER_00", "exclude")
        add_decision(session, run, "SPEAKER_01", "assign", alice)
        session.commit()
        assert_agreement(session, alice, {run})


def test_alias_ids_cycle_fails_loudly(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        now = datetime.now(tz=UTC)
        a = make_speaker(session)
        b = make_speaker(session, merged_into_id=a, merged_at=now)
        # Corrupt data: close the loop a -> b (bypassing service guards).
        session.execute(
            select(Speaker).where(Speaker.id == a)
        )  # ensure identity loaded
        session.get(Speaker, a).merged_into_id = b  # type: ignore[union-attr]
        session.get(Speaker, a).merged_at = now  # type: ignore[union-attr]
        session.flush()
        with pytest.raises(RosterError):
            alias_ids(session, b)
        session.rollback()
