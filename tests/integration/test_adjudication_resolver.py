"""Read-time attribution: decision precedence, queue membership, correction order."""

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from voxint.adjudication.ledger import record_decision
from voxint.adjudication.resolver import (
    Resolution,
    adjudication_queue,
    effective_decisions,
    label_states,
    needs_ruling,
    review_backlog_count,
    review_needed_label_count,
)
from voxint.adjudication.slots import claim_run
from voxint.db.models import (
    EMBEDDING_DIM,
    Decision,
    DiarizationTurn,
    MatchCandidate,
    MediaItem,
    MediaSourceMetadata,
    PipelineRun,
    RunStatus,
    Speaker,
    SpeakerAssignment,
)
from voxint.speakers.matching import MatchingGates
from voxint.speakers.policy import MatchBand

SPACE = "titanet-large-v2"
DEFAULT_GATES = MatchingGates()


def make_completed_run(session: Session) -> uuid.UUID:
    media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
    session.add(media)
    session.flush()
    run = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
    session.add(run)
    session.flush()
    return run.id


def add_turn(session: Session, run_id: uuid.UUID, index: int, label: str) -> None:
    vector = [0.0] * EMBEDDING_DIM
    vector[index % EMBEDDING_DIM] = 1.0
    session.add(
        DiarizationTurn(
            pipeline_run_id=run_id,
            turn_index=index,
            start_seconds=float(index * 10),
            end_seconds=float(index * 10 + 8),
            label=label,
            embedding=vector,
            embedding_space=SPACE,
        )
    )


def add_speaker(session: Session, name: str) -> uuid.UUID:
    speaker = Speaker(display_name=name)
    session.add(speaker)
    session.flush()
    return speaker.id


def test_precedence_human_over_grounded_over_unresolved(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = make_completed_run(session)
        for index, label in enumerate(["S0", "S0", "S1", "S1", "S2", "S2", "S3"]):
            add_turn(session, run_id, index, label)
        alice = add_speaker(session, "Alice")
        bob = add_speaker(session, "Bob")
        carol = add_speaker(session, "Carol")
        # S0: grounded cosine — machine identity stands.
        session.add(
            SpeakerAssignment(
                pipeline_run_id=run_id,
                diarization_label="S0",
                speaker_id=alice,
                method="cosine",
                confidence=0.9,
                grounded=True,
            )
        )
        # S1: grounded cosine says Bob, but a human ruled Carol — human wins.
        session.add(
            SpeakerAssignment(
                pipeline_run_id=run_id,
                diarization_label="S1",
                speaker_id=bob,
                method="cosine",
                confidence=0.85,
                grounded=True,
            )
        )
        record_decision(
            session,
            pipeline_run_id=run_id,
            diarization_label="S1",
            decision=Decision.ASSIGN,
            operator="ben",
            idempotency_key="k-s1",
            speaker_id=carol,
        )
        # S2: only a non-grounded cosine + an llm hint — unresolved.
        session.add(
            SpeakerAssignment(
                pipeline_run_id=run_id,
                diarization_label="S2",
                speaker_id=bob,
                method="cosine",
                confidence=0.65,
                grounded=False,
            )
        )
        session.add(
            SpeakerAssignment(
                pipeline_run_id=run_id,
                diarization_label="S2",
                method="llm_hint",
                proposed_name="Dave",
                grounded=False,
            )
        )
        session.commit()

        by_label = {s.label: s for s in label_states(session, run_id)}
        assert by_label["S0"].resolution is Resolution.GROUNDED_COSINE
        assert by_label["S0"].speaker_name == "Alice"
        assert by_label["S1"].resolution is Resolution.HUMAN_ASSIGN
        assert by_label["S1"].speaker_name == "Carol"
        assert by_label["S1"].cosine_speaker_name == "Bob"  # evidence still shown
        assert by_label["S2"].resolution is Resolution.UNRESOLVED
        assert by_label["S2"].llm_hint_name == "Dave"
        assert by_label["S2"].speaker_name is None  # hints are never identity
        assert by_label["S3"].resolution is Resolution.UNRESOLVED


def test_exclude_and_unknown_resolve_without_identity(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = make_completed_run(session)
        add_turn(session, run_id, 0, "S0")
        add_turn(session, run_id, 1, "S1")
        record_decision(
            session,
            pipeline_run_id=run_id,
            diarization_label="S0",
            decision=Decision.EXCLUDE,
            operator="ben",
            idempotency_key="k-ex",
        )
        record_decision(
            session,
            pipeline_run_id=run_id,
            diarization_label="S1",
            decision=Decision.UNKNOWN,
            operator="ben",
            idempotency_key="k-unk",
        )
        session.commit()
        by_label = {s.label: s for s in label_states(session, run_id)}
        assert by_label["S0"].resolution is Resolution.HUMAN_EXCLUDE
        assert by_label["S1"].resolution is Resolution.HUMAN_UNKNOWN
        assert by_label["S0"].speaker_id is None
        assert by_label["S1"].speaker_id is None


def test_correction_newest_decision_wins(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id = make_completed_run(session)
        add_turn(session, run_id, 0, "S0")
        alice = add_speaker(session, "Alice")
        record_decision(
            session,
            pipeline_run_id=run_id,
            diarization_label="S0",
            decision=Decision.UNKNOWN,
            operator="ben",
            idempotency_key="k-1",
        )
        # Separate transaction: created_at is transaction time, and the ledger
        # trigger forbids editing timestamps — a real correction is always a
        # later transaction.
        session.commit()
        record_decision(
            session,
            pipeline_run_id=run_id,
            diarization_label="S0",
            decision=Decision.ASSIGN,
            operator="ben",
            idempotency_key="k-2",
            speaker_id=alice,
        )
        session.commit()
        effective = effective_decisions(session, run_id)
        assert effective["S0"].decision == Decision.ASSIGN.value
        state = label_states(session, run_id)[0]
        assert state.resolution is Resolution.HUMAN_ASSIGN
        assert state.speaker_name == "Alice"


def test_queue_lists_only_completed_runs_with_unresolved_labels(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        # Unresolved + completed: queued.
        queued_run = make_completed_run(session)
        add_turn(session, queued_run, 0, "S0")
        # Completed but fully ruled: not queued.
        ruled_run = make_completed_run(session)
        add_turn(session, ruled_run, 0, "S0")
        record_decision(
            session,
            pipeline_run_id=ruled_run,
            diarization_label="S0",
            decision=Decision.EXCLUDE,
            operator="ben",
            idempotency_key="k-ruled",
        )
        # Unresolved but still running: not queued.
        media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
        session.add(media)
        session.flush()
        running = PipelineRun(
            media_item_id=media.id,
            status=RunStatus.RUNNING.value,
            current_stage="prepare",
        )
        session.add(running)
        session.flush()
        add_turn(session, running.id, 0, "S0")
        session.commit()

        entries = adjudication_queue(session, gates=DEFAULT_GATES)
        assert [e.run_id for e in entries] == [queued_run]
        assert entries[0].unresolved_labels == 1
        assert entries[0].claimed_by is None

        # A live claim surfaces in the queue row.
        claim_run(session, queued_run, reviewer="ben", ttl_seconds=600)
        session.commit()
        assert adjudication_queue(session, gates=DEFAULT_GATES)[0].claimed_by == "ben"


def test_queue_entry_carries_display_context(
    session_factory: sessionmaker[Session],
) -> None:
    """The queue row exposes title, probed duration, and created_at (issue #56)."""
    with session_factory() as session:
        media = MediaItem(
            source_path=f"incoming/{uuid.uuid4()}.wav",
            duration_seconds=125.0,
        )
        session.add(media)
        session.flush()
        session.add(
            MediaSourceMetadata(
                media_item_id=media.id,
                source_kind="ytdlp",
                title="City Council 2026-08",
                raw={"id": "abc"},
                raw_schema_version=1,
                acquired_at=datetime(2026, 8, 1, 12, 0, tzinfo=UTC),
            )
        )
        run = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
        session.add(run)
        session.flush()
        add_turn(session, run.id, 0, "S0")
        session.commit()

        entry = adjudication_queue(session, gates=DEFAULT_GATES)[0]
        assert entry.title == "City Council 2026-08"
        assert entry.duration_seconds == 125.0
        assert entry.created_at is not None

    # An upload with no metadata snapshot leaves title/duration None, no error.
    with session_factory() as session:
        bare = make_completed_run(session)
        add_turn(session, bare, 0, "S0")
        session.commit()
        entry = next(
            e for e in adjudication_queue(session, gates=DEFAULT_GATES) if e.run_id == bare
        )
        assert entry.title is None
        assert entry.duration_seconds is None
        assert entry.created_at is not None


def test_queue_sidecar_title_wins_over_scraped_title(
    session_factory: sessionmaker[Session],
) -> None:
    """The frozen sidecar title (issue #104, operator intent) beats the
    acquisition-metadata title; a tampered snapshot degrades to the scraped one."""
    with session_factory() as session:
        media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
        session.add(media)
        session.flush()
        session.add(
            MediaSourceMetadata(
                media_item_id=media.id,
                source_kind="ytdlp",
                title="Scraped title",
                raw={"id": "abc"},
                raw_schema_version=1,
                acquired_at=datetime(2026, 8, 1, 12, 0, tzinfo=UTC),
            )
        )
        run = PipelineRun(
            media_item_id=media.id,
            status=RunStatus.COMPLETED.value,
            sidecar={"title": "Operator title", "content_item_id": 1},
        )
        session.add(run)
        session.flush()
        add_turn(session, run.id, 0, "S0")
        session.commit()
        entry = next(
            e
            for e in adjudication_queue(session, gates=DEFAULT_GATES)
            if e.run_id == run.id
        )
        assert entry.title == "Operator title"

    # A sidecar without a usable title falls back to the scraped one.
    with session_factory() as session:
        media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
        session.add(media)
        session.flush()
        session.add(
            MediaSourceMetadata(
                media_item_id=media.id,
                source_kind="ytdlp",
                title="Scraped title",
                raw={"id": "abc"},
                raw_schema_version=1,
                acquired_at=datetime(2026, 8, 1, 12, 0, tzinfo=UTC),
            )
        )
        run = PipelineRun(
            media_item_id=media.id,
            status=RunStatus.COMPLETED.value,
            sidecar={"notes": "no title here"},
        )
        session.add(run)
        session.flush()
        add_turn(session, run.id, 0, "S0")
        session.commit()
        entry = next(
            e
            for e in adjudication_queue(session, gates=DEFAULT_GATES)
            if e.run_id == run.id
        )
        assert entry.title == "Scraped title"


def test_queue_sort_unresolved_orders_by_voice_count(
    session_factory: sessionmaker[Session],
) -> None:
    """sort="unresolved" surfaces the most-unresolved runs first, oldest-tie-broken."""
    with session_factory() as session:
        # Created oldest→newest: one-voice, then two-voice, then another one-voice.
        # Explicit, distinct created_at values give a well-defined oldest-first
        # order (same-transaction rows share the server default now(), so their
        # order is otherwise decided only by the id tie-breaker).
        base = datetime(2026, 8, 1, 12, 0, 0, tzinfo=UTC)
        one_a = make_completed_run(session)
        add_turn(session, one_a, 0, "S0")
        two = make_completed_run(session)
        add_turn(session, two, 0, "S0")
        add_turn(session, two, 1, "S1")
        one_b = make_completed_run(session)
        add_turn(session, one_b, 0, "S0")
        for offset, rid in enumerate((one_a, two, one_b)):
            session.get(PipelineRun, rid).created_at = base + timedelta(minutes=offset)
        session.commit()

        # Default: oldest-first (FIFO), unchanged behaviour.
        assert [e.run_id for e in adjudication_queue(session, gates=DEFAULT_GATES)] == [
            one_a,
            two,
            one_b,
        ]

        # Unresolved-first: the two-voice run leads; the two one-voice runs keep
        # their oldest-first order among the tie (stable sort).
        by_work = adjudication_queue(session, gates=DEFAULT_GATES, sort="unresolved")
        assert [e.run_id for e in by_work] == [two, one_a, one_b]
        assert [e.unresolved_labels for e in by_work] == [2, 1, 1]

        # An unknown sort degrades to the default order rather than erroring.
        assert [
            e.run_id
            for e in adjudication_queue(session, gates=DEFAULT_GATES, sort="bogus")
        ] == [
            one_a,
            two,
            one_b,
        ]


def test_review_backlog_count_matches_queue_length(
    session_factory: sessionmaker[Session],
) -> None:
    """The dashboard headline (issue #117) equals the queue it links to, by
    construction, across every eligibility outcome — completed-and-unresolved,
    fully-ruled, still-running, and archived-with-unresolved."""
    # Empty system: no runs, no backlog.
    with session_factory() as session:
        assert (
            review_backlog_count(session, gates=DEFAULT_GATES)
            == len(adjudication_queue(session, gates=DEFAULT_GATES))
            == 0
        )

    with session_factory() as session:
        # Eligible: completed with an unresolved voice.
        eligible = make_completed_run(session)
        add_turn(session, eligible, 0, "S0")
        # Completed but fully ruled: not eligible.
        ruled = make_completed_run(session)
        add_turn(session, ruled, 0, "S0")
        record_decision(
            session,
            pipeline_run_id=ruled,
            diarization_label="S0",
            decision=Decision.EXCLUDE,
            operator="ben",
            idempotency_key="k-ruled-backlog",
        )
        # Unresolved but still running: not eligible.
        media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
        session.add(media)
        session.flush()
        running = PipelineRun(
            media_item_id=media.id,
            status=RunStatus.RUNNING.value,
            current_stage="prepare",
        )
        session.add(running)
        session.flush()
        add_turn(session, running.id, 0, "S0")
        # Archived completed run with an unresolved voice: hidden from the queue.
        archived = make_completed_run(session)
        add_turn(session, archived, 0, "S0")
        session.get(PipelineRun, archived).archived_at = datetime.now(tz=UTC)
        session.commit()

        assert (
            review_backlog_count(session, gates=DEFAULT_GATES)
            == len(adjudication_queue(session, gates=DEFAULT_GATES))
            == 1
        )

        # Resolving the one eligible run drops the backlog to zero, still in step.
        record_decision(
            session,
            pipeline_run_id=eligible,
            diarization_label="S0",
            decision=Decision.EXCLUDE,
            operator="ben",
            idempotency_key="k-eligible-backlog",
        )
        session.commit()
        assert (
            review_backlog_count(session, gates=DEFAULT_GATES)
            == len(adjudication_queue(session, gates=DEFAULT_GATES))
            == 0
        )

        # A confirmable auto-enroll is the only open question on this run.
        auto_enrolled = make_completed_run(session)
        add_turn(session, auto_enrolled, 0, "S0")
        candidate = add_speaker(session, "Backlog Candidate")
        voice = add_speaker(session, "Voice Backlog")
        add_candidate(
            session,
            auto_enrolled,
            "S0",
            decision="accepted",
            reason="accepted",
            top_speaker_id=candidate,
            similarity=0.65,
            margin=0.10,
            vote_agreement=0.75,
            eligible_turns=3,
            eligible_seconds=15.0,
            roster_size=2,
            grounded=False,
        )
        record_decision(
            session,
            pipeline_run_id=auto_enrolled,
            diarization_label="S0",
            decision=Decision.AUTO_ENROLL,
            operator="auto_enroll",
            idempotency_key="k-auto-enroll-backlog",
            speaker_id=voice,
        )
        session.commit()

        queue = adjudication_queue(session, gates=DEFAULT_GATES)
        assert review_backlog_count(session, gates=DEFAULT_GATES) == len(queue) == 1
        assert queue[0].run_id == auto_enrolled
        assert queue[0].unresolved_labels == 1


def add_candidate(
    session: Session,
    run_id: uuid.UUID,
    label: str,
    *,
    decision: str,
    reason: str,
    top_speaker_id: uuid.UUID | None,
    similarity: float | None = None,
    margin: float | None = None,
    vote_agreement: float | None = None,
    grounded: bool | None = None,
    eligible_turns: int = 4,
    eligible_seconds: float = 30.0,
    roster_size: int | None = 3,
) -> None:
    session.add(
        MatchCandidate(
            pipeline_run_id=run_id,
            diarization_label=label,
            decision=decision,
            reason=reason,
            embedding_space=SPACE,
            top_speaker_id=top_speaker_id,
            similarity=similarity,
            margin=margin,
            vote_agreement=vote_agreement,
            grounded=grounded,
            eligible_turns=eligible_turns,
            eligible_seconds=eligible_seconds,
            roster_size=roster_size,
        )
    )


def test_review_needed_python_sql_parity(
    session_factory: sessionmaker[Session],
) -> None:
    """The SQL dashboard count and Python resolver agree on open questions."""
    gates = DEFAULT_GATES
    with session_factory() as session:
        runs: dict[str, uuid.UUID] = {}

        def one_label_run(scenario: str) -> uuid.UUID:
            run_id = make_completed_run(session)
            add_turn(session, run_id, 0, "S0")
            runs[scenario] = run_id
            return run_id

        def auto_enroll_run(
            scenario: str,
            *,
            similarity: float,
            margin: float | None,
            roster_size: int = 2,
            archive_candidate: bool = False,
        ) -> tuple[uuid.UUID, uuid.UUID]:
            run_id = one_label_run(scenario)
            candidate = add_speaker(session, f"{scenario} Candidate")
            if archive_candidate:
                candidate_row = session.get(Speaker, candidate)
                assert candidate_row is not None
                candidate_row.deleted_at = datetime.now(tz=UTC)
            voice = add_speaker(session, f"{scenario} Voice")
            add_candidate(
                session,
                run_id,
                "S0",
                decision="accepted",
                reason="accepted",
                top_speaker_id=candidate,
                similarity=similarity,
                margin=margin,
                vote_agreement=0.75,
                eligible_turns=3,
                eligible_seconds=15.0,
                roster_size=roster_size,
                grounded=False,
            )
            record_decision(
                session,
                pipeline_run_id=run_id,
                diarization_label="S0",
                decision=Decision.AUTO_ENROLL,
                operator="auto_enroll",
                idempotency_key=f"parity-{scenario}-auto-enroll",
                speaker_id=voice,
            )
            return run_id, candidate

        # No decision and no grounded cosine.
        one_label_run("UNRESOLVED")

        grounded = one_label_run("GROUNDED_COSINE")
        grounded_speaker = add_speaker(session, "Grounded Speaker")
        session.add(
            SpeakerAssignment(
                pipeline_run_id=grounded,
                diarization_label="S0",
                speaker_id=grounded_speaker,
                method="cosine",
                confidence=0.9,
                grounded=True,
            )
        )

        human = one_label_run("HUMAN_ASSIGN")
        human_speaker = add_speaker(session, "Human Speaker")
        record_decision(
            session,
            pipeline_run_id=human,
            diarization_label="S0",
            decision=Decision.ASSIGN,
            operator="ben",
            idempotency_key="parity-human-assign",
            speaker_id=human_speaker,
        )

        auto_enroll_run("AUTO_ENROLL_REVIEW", similarity=0.65, margin=0.10)
        auto_enroll_run("AUTO_ENROLL_ABSTAIN", similarity=0.50, margin=0.10)
        auto_enroll_run("AUTO_ENROLL_AMBIGUOUS", similarity=0.65, margin=0.06)
        auto_enroll_run(
            "AUTO_ENROLL_SINGLE_SPEAKER",
            similarity=0.65,
            margin=None,
            roster_size=1,
        )

        superseded, _ = auto_enroll_run(
            "AUTO_ENROLL_SUPERSEDED", similarity=0.65, margin=0.10
        )
        session.commit()
        replacement = add_speaker(session, "Superseding Human")
        record_decision(
            session,
            pipeline_run_id=superseded,
            diarization_label="S0",
            decision=Decision.ASSIGN,
            operator="ben",
            idempotency_key="parity-auto-enroll-superseded-human",
            speaker_id=replacement,
        )

        auto_enroll_run(
            "AUTO_ENROLL_ARCHIVED",
            similarity=0.65,
            margin=0.10,
            archive_candidate=True,
        )
        session.commit()

        for scenario, run_id in runs.items():
            sql_count = session.execute(
                select(review_needed_label_count(run_id, gates))
            ).scalar()
            states = label_states(session, run_id, gates=gates)
            python_count = sum(1 for state in states if needs_ruling(state))
            assert sql_count == python_count, (
                f"SQL={sql_count}, Python={python_count} for {scenario}"
            )


def test_label_states_carry_policy_candidate_and_evidence(
    session_factory: sessionmaker[Session],
) -> None:
    """#115: the rail confirms the POLICY candidate, so it must be on LabelState.

    The candidate comes from ``match_candidates.top_speaker_id`` and its name
    resolves through the same names map as every other speaker, so a formerly
    rejected row that a gate change reclassifies as review still names someone.
    """
    with session_factory() as session:
        run_id = make_completed_run(session)
        for index, label in enumerate(["S0", "S0", "S1", "S1", "S2", "S2", "S3", "S4"]):
            add_turn(session, run_id, index, label)
        alice = add_speaker(session, "Alice")
        bob = add_speaker(session, "Bob")
        carol = add_speaker(session, "Carol")
        # S0: grounded proposal, resolved by the machine.
        session.add(
            SpeakerAssignment(
                pipeline_run_id=run_id,
                diarization_label="S0",
                speaker_id=alice,
                method="cosine",
                confidence=0.9,
                grounded=True,
            )
        )
        add_candidate(
            session,
            run_id,
            "S0",
            decision="accepted",
            reason="accepted",
            top_speaker_id=alice,
            similarity=0.9,
            margin=0.2,
            vote_agreement=0.9,
            grounded=True,
        )
        # S1: accepted but not grounded (the review band) — proposal row + evidence.
        session.add(
            SpeakerAssignment(
                pipeline_run_id=run_id,
                diarization_label="S1",
                speaker_id=bob,
                method="cosine",
                confidence=0.65,
                grounded=False,
            )
        )
        add_candidate(
            session,
            run_id,
            "S1",
            decision="accepted",
            reason="accepted",
            top_speaker_id=bob,
            similarity=0.65,
            margin=0.1,
            vote_agreement=0.8,
            grounded=False,
        )
        # S2: rejected — no proposal row, evidence names the nearest voice.
        add_candidate(
            session,
            run_id,
            "S2",
            decision="rejected",
            reason="below_cosine",
            top_speaker_id=carol,
            similarity=0.4,
            margin=0.1,
            vote_agreement=0.8,
        )
        # S3: ineligible — no candidate at all.
        add_candidate(
            session,
            run_id,
            "S3",
            decision="ineligible",
            reason="too_few_turns",
            top_speaker_id=None,
            eligible_turns=1,
            eligible_seconds=2.0,
            roster_size=None,
        )
        # S4: never evaluated (no evidence row).
        session.commit()

        by_label = {s.label: s for s in label_states(session, run_id, gates=DEFAULT_GATES)}

        s0 = by_label["S0"]
        assert s0.band is MatchBand.AUTO_ATTRIBUTE
        assert s0.candidate_speaker_id == alice
        assert s0.candidate_speaker_name == "Alice"

        s1 = by_label["S1"]
        assert s1.resolution is Resolution.UNRESOLVED
        assert s1.band is MatchBand.REVIEW
        assert s1.candidate_prompt_allowed is True
        # Policy candidate and proposal speaker agree for an accepted row.
        assert s1.candidate_speaker_id == s1.cosine_speaker_id == bob
        assert s1.candidate_speaker_name == "Bob"
        assert s1.match_similarity == 0.65
        assert s1.match_vote_agreement == 0.8

        s2 = by_label["S2"]
        assert s2.band is MatchBand.ABSTAIN
        assert s2.cosine_speaker_id is None  # no proposal row
        assert s2.candidate_speaker_id == carol  # nearest voice still named
        assert s2.candidate_speaker_name == "Carol"
        assert s2.candidate_prompt_allowed is False  # never confirmable

        s3 = by_label["S3"]
        assert s3.band is MatchBand.ABSTAIN
        assert s3.candidate_speaker_id is None
        assert s3.match_reason == "too_few_turns"

        s4 = by_label["S4"]
        assert s4.band is MatchBand.ABSTAIN
        assert s4.match_decision is None
        assert s4.candidate_speaker_id is None
        assert s4.match_similarity is None


def test_auto_enrolled_label_is_banded_by_live_evidence(
    session_factory: sessionmaker[Session],
) -> None:
    """#115: an auto-enrolled label is a placeholder, so its band comes from the
    recorded evidence like an unresolved one — the review tail stays visible
    even after auto-enroll (#275) has saved the voice as ``Voice N``."""
    with session_factory() as session:
        run_id = make_completed_run(session)
        for index, label in enumerate(["S0", "S0", "S1", "S1"]):
            add_turn(session, run_id, index, label)
        alice = add_speaker(session, "Alice")
        bob = add_speaker(session, "Bob")
        voice_1 = add_speaker(session, "Voice 1")
        voice_2 = add_speaker(session, "Voice 2")
        # S0: auto-enrolled as Voice 1, but the matcher had an ungrounded
        # candidate (Bob) — the operator should still get to confirm Bob.
        session.add(
            SpeakerAssignment(
                pipeline_run_id=run_id,
                diarization_label="S0",
                speaker_id=bob,
                method="cosine",
                confidence=0.65,
                grounded=False,
            )
        )
        add_candidate(
            session,
            run_id,
            "S0",
            decision="accepted",
            reason="accepted",
            top_speaker_id=bob,
            similarity=0.65,
            margin=0.1,
            vote_agreement=0.8,
            grounded=False,
        )
        record_decision(
            session,
            pipeline_run_id=run_id,
            diarization_label="S0",
            decision=Decision.AUTO_ENROLL,
            operator="auto_enroll",
            idempotency_key="k-ae-s0",
            speaker_id=voice_1,
        )
        # S1: auto-enrolled as Voice 2 with a clear rejection — nothing to confirm.
        add_candidate(
            session,
            run_id,
            "S1",
            decision="rejected",
            reason="below_cosine",
            top_speaker_id=alice,
            similarity=0.3,
            margin=0.05,
            vote_agreement=0.7,
        )
        record_decision(
            session,
            pipeline_run_id=run_id,
            diarization_label="S1",
            decision=Decision.AUTO_ENROLL,
            operator="auto_enroll",
            idempotency_key="k-ae-s1",
            speaker_id=voice_2,
        )
        session.commit()

        by_label = {s.label: s for s in label_states(session, run_id, gates=DEFAULT_GATES)}

        s0 = by_label["S0"]
        assert s0.resolution is Resolution.AUTO_ENROLL  # resolution untouched
        assert s0.speaker_name == "Voice 1"
        assert s0.band is MatchBand.REVIEW
        assert s0.candidate_speaker_id == bob
        assert s0.candidate_speaker_name == "Bob"
        assert s0.candidate_prompt_allowed is True

        s1 = by_label["S1"]
        assert s1.resolution is Resolution.AUTO_ENROLL
        assert s1.speaker_name == "Voice 2"
        assert s1.band is MatchBand.ABSTAIN
        assert s1.candidate_prompt_allowed is False

        queue = adjudication_queue(session, gates=DEFAULT_GATES)
        assert len(queue) == 1
        assert queue[0].run_id == run_id
        assert queue[0].unresolved_labels == 1  # S0 is confirmable


def test_candidate_follows_merges_and_never_names_an_archived_speaker(
    session_factory: sessionmaker[Session],
) -> None:
    """#115: the Confirm target is the merge target, and an archived speaker is
    never offered (the decide route would reject it), while its name still
    resolves for display."""
    with session_factory() as session:
        run_id = make_completed_run(session)
        for index, label in enumerate(["S0", "S0", "S1", "S1"]):
            add_turn(session, run_id, index, label)
        bob = add_speaker(session, "Bob")
        bobby = add_speaker(session, "Bobby")
        bobby_row = session.get(Speaker, bobby)
        assert bobby_row is not None
        bobby_row.merged_into_id = bob
        bobby_row.merged_at = datetime.now(tz=UTC)
        gone = add_speaker(session, "Gone")
        gone_row = session.get(Speaker, gone)
        assert gone_row is not None
        gone_row.deleted_at = datetime.now(tz=UTC)
        # S0: evidence names the merged-away source; Confirm must target Bob.
        add_candidate(
            session,
            run_id,
            "S0",
            decision="accepted",
            reason="accepted",
            top_speaker_id=bobby,
            similarity=0.65,
            margin=0.1,
            vote_agreement=0.8,
            grounded=False,
        )
        # S1: evidence names a speaker archived since matching.
        add_candidate(
            session,
            run_id,
            "S1",
            decision="accepted",
            reason="accepted",
            top_speaker_id=gone,
            similarity=0.65,
            margin=0.1,
            vote_agreement=0.8,
            grounded=False,
        )
        session.commit()

        by_label = {s.label: s for s in label_states(session, run_id, gates=DEFAULT_GATES)}

        s0 = by_label["S0"]
        assert s0.band is MatchBand.REVIEW
        assert s0.candidate_speaker_id == bob
        assert s0.candidate_speaker_name == "Bob"

        s1 = by_label["S1"]
        assert s1.band is MatchBand.REVIEW
        assert s1.candidate_speaker_id is None
        assert s1.candidate_speaker_name is None
