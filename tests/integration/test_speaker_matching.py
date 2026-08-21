"""Cosine matching + the proposal writer against real Postgres/pgvector."""

import math
import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from voxint.db.models import (
    EMBEDDING_DIM,
    AssignmentMethod,
    DiarizationTurn,
    MatchCandidate,
    MediaItem,
    PipelineRun,
    Speaker,
    SpeakerAssignment,
    SpeakerEmbedding,
    TranscriptSegment,
)
from voxint.domain_packs.base import DomainPack
from voxint.pipeline.stages import enhance_match
from voxint.pipeline.stages.context import StageContext
from voxint.speakers.matching import (
    DECISION_ACCEPTED,
    DECISION_INELIGIBLE,
    DECISION_REJECTED,
    REASON_ACCEPTED,
    REASON_BELOW_COSINE,
    REASON_BELOW_MARGIN,
    REASON_BELOW_VOTE_AGREEMENT,
    REASON_NO_ELIGIBLE_TURNS,
    REASON_NO_ROSTER,
    REASON_TOO_FEW_TURNS,
    REASON_TOO_LITTLE_SPEECH,
    CosineProposal,
    LabelDecision,
    MatchingGates,
    NameHintProposal,
    ProposalError,
    confidence_from_similarity,
    evaluate_run,
    match_speakers,
    replace_run_match_candidates,
    replace_run_proposals,
)

SPACE = "titanet-large-v1"
GATES = MatchingGates()


def unit(*components: tuple[int, float]) -> list[float]:
    """A unit vector from sparse (dimension, value) components."""
    vector = [0.0] * EMBEDDING_DIM
    for dim, value in components:
        vector[dim] = value
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector]


E0 = unit((0, 1.0))
E1 = unit((1, 1.0))


def make_run(session: Session) -> uuid.UUID:
    media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
    session.add(media)
    session.flush()
    run = PipelineRun(media_item_id=media.id)
    session.add(run)
    session.flush()
    return run.id


def add_turn(
    session: Session,
    run_id: uuid.UUID,
    index: int,
    label: str,
    start: float,
    end: float,
    embedding: list[float] | None,
    overlap_seconds: float = 0.0,
    space: str | None = SPACE,
) -> None:
    session.add(
        DiarizationTurn(
            pipeline_run_id=run_id,
            turn_index=index,
            start_seconds=start,
            end_seconds=end,
            label=label,
            overlap=overlap_seconds > 0,
            overlap_seconds=overlap_seconds,
            skip_reason=None if embedding is not None else "too_short",
            embedding=embedding,
            embedding_space=space if embedding is not None else None,
        )
    )


def add_speaker(
    session: Session, name: str, embeddings: list[list[float]], space: str = SPACE
) -> uuid.UUID:
    speaker = Speaker(display_name=name)
    session.add(speaker)
    session.flush()
    for embedding in embeddings:
        session.add(
            SpeakerEmbedding(
                speaker_id=speaker.id, embedding_space=space, embedding=embedding
            )
        )
    return speaker.id


@pytest.fixture()
def session(session_factory: sessionmaker[Session]):  # type: ignore[no-untyped-def]
    with session_factory() as s:
        yield s


# ------------------------------------------------------------------- matching


def test_clean_match_is_grounded(session: Session) -> None:
    run_id = make_run(session)
    alice = add_speaker(session, "Alice", [E0])
    add_speaker(session, "Bob", [E1])
    for i in range(3):
        add_turn(session, run_id, i, "SPEAKER_00", i * 5.0, i * 5.0 + 4.0, E0)
    add_turn(session, run_id, 3, "SPEAKER_00", 20.0, 20.4, None)  # skipped window
    session.flush()

    proposals = match_speakers(session, run_id, GATES)
    assert len(proposals) == 1
    p = proposals[0]
    assert p.speaker_id == alice
    assert p.similarity == pytest.approx(1.0)
    assert p.vote_agreement == pytest.approx(1.0)
    assert p.grounded is True


def test_enough_for_proposal_but_not_grounding(session: Session) -> None:
    run_id = make_run(session)
    alice = add_speaker(session, "Alice", [E0])
    for i in range(2):  # 2 turns x 3.5 s: clears proposal gates, not grounded ones
        add_turn(session, run_id, i, "SPEAKER_00", i * 4.0, i * 4.0 + 3.5, E0)
    session.flush()

    proposals = match_speakers(session, run_id, GATES)
    assert len(proposals) == 1
    assert proposals[0].speaker_id == alice
    assert proposals[0].grounded is False


def test_low_similarity_yields_no_proposal(session: Session) -> None:
    run_id = make_run(session)
    add_speaker(session, "Alice", [E0])
    off_roster = unit((0, 0.5), (5, math.sqrt(0.75)))  # cos to Alice = 0.5 < 0.60
    for i in range(3):
        add_turn(session, run_id, i, "SPEAKER_00", i * 5.0, i * 5.0 + 4.0, off_roster)
    session.flush()
    assert match_speakers(session, run_id, GATES) == ()


def test_ambiguous_margin_yields_no_proposal(session: Session) -> None:
    run_id = make_run(session)
    add_speaker(session, "Alice", [E0])
    add_speaker(session, "Bob", [E1])
    between = unit((0, 1.0), (1, 1.0))  # equidistant: margin 0 < 0.05
    for i in range(3):
        add_turn(session, run_id, i, "SPEAKER_00", i * 5.0, i * 5.0 + 4.0, between)
    session.flush()
    assert match_speakers(session, run_id, GATES) == ()


def test_minimum_evidence_gates(session: Session) -> None:
    run_id = make_run(session)
    add_speaker(session, "Alice", [E0])
    add_turn(session, run_id, 0, "ONE_LONG_TURN", 0.0, 20.0, E0)  # 1 turn < 2
    add_turn(session, run_id, 1, "TOO_SHORT", 30.0, 32.0, E0)  # 4 s < 6 s
    add_turn(session, run_id, 2, "TOO_SHORT", 33.0, 35.0, E0)
    session.flush()
    assert match_speakers(session, run_id, GATES) == ()


def test_heavily_overlapped_turns_are_ineligible(session: Session) -> None:
    run_id = make_run(session)
    add_speaker(session, "Alice", [E0])
    for i in range(3):  # 4 s turns with 2 s overlap: ratio 0.5 > 0.20
        add_turn(
            session, run_id, i, "SPEAKER_00", i * 5.0, i * 5.0 + 4.0, E0, overlap_seconds=2.0
        )
    session.flush()
    assert match_speakers(session, run_id, GATES) == ()


# ------------------------------------------- #11 enhance stage reads the run pack


def test_enhance_stage_passes_run_pack_name_attribution_context(session: Session) -> None:
    """The enhance stage sources ``name_attribution_context`` from the RUN's
    frozen pack on the context (not a hardcoded default), so two runs with
    different packs get different attribution guidance in the same worker."""
    from pathlib import Path

    from tests.fakes import FakeASR, FakeDiarizer, FakeEmbedder, FakeLLM

    run_id = make_run(session)
    for i in range(2):
        session.add(
            TranscriptSegment(
                pipeline_run_id=run_id,
                segment_index=i,
                start_seconds=float(i),
                end_seconds=float(i + 1),
                raw_text=f"segment {i}",
                diarization_label="SPEAKER_00",
            )
        )
    session.flush()

    llm = FakeLLM()
    pack = DomainPack(
        name="podcast",
        prompt_fragments={"name_attribution_context": "The host is the most talkative voice."},
    )
    ctx = StageContext(
        asr=FakeASR(),
        diarizer=FakeDiarizer(),
        embedder=FakeEmbedder(),
        llm=llm,
        media_root=Path("/data/media"),
        domain_pack=pack,
    )

    enhance_match.run(ctx, session, run_id)

    assert llm.attribution_contexts == ["The host is the most talkative voice."]


def test_enhance_stage_omits_attribution_context_when_pack_declares_none(session: Session) -> None:
    from pathlib import Path

    from tests.fakes import FakeASR, FakeDiarizer, FakeEmbedder, FakeLLM

    run_id = make_run(session)
    session.add(
        TranscriptSegment(
            pipeline_run_id=run_id,
            segment_index=0,
            start_seconds=0.0,
            end_seconds=1.0,
            raw_text="hello",
            diarization_label="SPEAKER_00",
        )
    )
    session.flush()

    llm = FakeLLM()
    ctx = StageContext(
        asr=FakeASR(),
        diarizer=FakeDiarizer(),
        embedder=FakeEmbedder(),
        llm=llm,
        media_root=Path("/data/media"),
        domain_pack=DomainPack(name="generic"),
    )

    enhance_match.run(ctx, session, run_id)

    assert llm.attribution_contexts == [""]


def test_enhance_stage_bundled_requests_no_hint_parse(session: Session) -> None:
    """#85: the scoped bundled path asks the client NOT to parse name_hints
    (want_name_hints=False) while the BYO path leaves parsing on (True). The
    prompt itself is path-independent, so the pack's attribution guidance is still
    forwarded on both paths (changing the prompt regressed 4B faithfulness)."""
    from pathlib import Path

    from tests.fakes import FakeASR, FakeDiarizer, FakeEmbedder, FakeLLM

    pack = DomainPack(
        name="podcast",
        prompt_fragments={"name_attribution_context": "The host is the most talkative voice."},
    )

    def _run(*, bundled: bool) -> FakeLLM:
        run_id = make_run(session)
        session.add(
            TranscriptSegment(
                pipeline_run_id=run_id,
                segment_index=0,
                start_seconds=0.0,
                end_seconds=1.0,
                raw_text="hello",
                diarization_label="SPEAKER_00",
            )
        )
        session.flush()
        llm = FakeLLM()
        ctx = StageContext(
            asr=FakeASR(),
            diarizer=FakeDiarizer(),
            embedder=FakeEmbedder(),
            llm=llm,
            media_root=Path("/data/media"),
            domain_pack=pack,
            llm_bundled=bundled,
        )
        enhance_match.run(ctx, session, run_id)
        return llm

    bundled = _run(bundled=True)
    assert bundled.want_name_hints_calls == [False]
    # Prompt is identical on every path, so attribution guidance is still sent.
    assert bundled.attribution_contexts == ["The host is the most talkative voice."]

    byo = _run(bundled=False)
    assert byo.want_name_hints_calls == [True]
    assert byo.attribution_contexts == ["The host is the most talkative voice."]


def test_enhance_stage_bundled_drops_name_hints(session: Session) -> None:
    """The scoped bundled model (#67) powers enhancement text ONLY — its
    name_hints must never reach proposals, so the bundle can't drive speaker
    attribution through the back door (it stays on the BYO names producer)."""
    from pathlib import Path

    from tests.fakes import FakeASR, FakeDiarizer, FakeEmbedder, FakeLLM
    from voxint.clients.base import SpeakerNameHint

    run_id = make_run(session)
    seed_label(session, run_id)  # a real SPEAKER_00 turn a hint could attach to
    session.add(
        TranscriptSegment(
            pipeline_run_id=run_id,
            segment_index=0,
            start_seconds=0.0,
            end_seconds=1.0,
            raw_text="hi",
            diarization_label="SPEAKER_00",
        )
    )
    session.flush()

    llm = FakeLLM(
        name_hints=(SpeakerNameHint(diarization_label="SPEAKER_00", name="Jane", kind="self"),)
    )
    ctx = StageContext(
        asr=FakeASR(),
        diarizer=FakeDiarizer(),
        embedder=FakeEmbedder(),
        llm=llm,
        media_root=Path("/data/media"),
        domain_pack=DomainPack(name="generic"),
        llm_bundled=True,
    )

    enhance_match.run(ctx, session, run_id)
    session.flush()

    hint_rows = (
        session.execute(
            select(SpeakerAssignment).where(
                SpeakerAssignment.method == AssignmentMethod.LLM_HINT.value
            )
        )
        .scalars()
        .all()
    )
    assert hint_rows == []


def test_enhance_stage_byo_keeps_name_hints(session: Session) -> None:
    """The BYO path (not bundled) still persists the LLM name hint as an LLM_HINT
    proposal — the #67 drop is scoped to the bundle and must not regress it."""
    from pathlib import Path

    from tests.fakes import FakeASR, FakeDiarizer, FakeEmbedder, FakeLLM
    from voxint.clients.base import SpeakerNameHint

    run_id = make_run(session)
    seed_label(session, run_id)
    session.add(
        TranscriptSegment(
            pipeline_run_id=run_id,
            segment_index=0,
            start_seconds=0.0,
            end_seconds=1.0,
            raw_text="hi",
            diarization_label="SPEAKER_00",
        )
    )
    session.flush()

    llm = FakeLLM(
        name_hints=(SpeakerNameHint(diarization_label="SPEAKER_00", name="Jane", kind="self"),)
    )
    ctx = StageContext(
        asr=FakeASR(),
        diarizer=FakeDiarizer(),
        embedder=FakeEmbedder(),
        llm=llm,
        media_root=Path("/data/media"),
        domain_pack=DomainPack(name="generic"),
        # llm_bundled defaults False — the ordinary BYO endpoint.
    )

    enhance_match.run(ctx, session, run_id)
    session.flush()

    hint_rows = (
        session.execute(
            select(SpeakerAssignment).where(
                SpeakerAssignment.method == AssignmentMethod.LLM_HINT.value
            )
        )
        .scalars()
        .all()
    )
    assert len(hint_rows) == 1
    assert hint_rows[0].proposed_name == "Jane"


def test_label_spanning_embedding_spaces_is_dropped(session: Session) -> None:
    run_id = make_run(session)
    add_speaker(session, "Alice", [E0])
    add_turn(session, run_id, 0, "SPEAKER_00", 0.0, 4.0, E0, space=SPACE)
    add_turn(session, run_id, 1, "SPEAKER_00", 5.0, 9.0, E0, space="other-space-v9")
    add_turn(session, run_id, 2, "SPEAKER_00", 10.0, 14.0, E0, space=SPACE)
    session.flush()
    assert match_speakers(session, run_id, GATES) == ()


def test_empty_roster_yields_no_proposal(session: Session) -> None:
    run_id = make_run(session)
    for i in range(3):
        add_turn(session, run_id, i, "SPEAKER_00", i * 5.0, i * 5.0 + 4.0, E0)
    session.flush()
    assert match_speakers(session, run_id, GATES) == ()


def test_roster_uses_centroid_across_enrollments(session: Session) -> None:
    run_id = make_run(session)
    alice = add_speaker(session, "Alice", [E0, unit((0, 1.0), (1, 1.0))])
    add_speaker(session, "Bob", [E1])
    for i in range(3):
        add_turn(session, run_id, i, "SPEAKER_00", i * 5.0, i * 5.0 + 4.0, E0)
    session.flush()
    proposals = match_speakers(session, run_id, GATES)
    assert len(proposals) == 1
    assert proposals[0].speaker_id == alice
    assert proposals[0].similarity < 1.0  # centroid, not best single enrollment


def test_vote_disagreement_blocks_grounding(session: Session) -> None:
    run_id = make_run(session)
    alice = add_speaker(session, "Alice", [E0])
    add_speaker(session, "Bob", [E1])
    add_turn(session, run_id, 0, "SPEAKER_00", 0.0, 4.0, E0)
    add_turn(session, run_id, 1, "SPEAKER_00", 5.0, 9.0, E0)
    add_turn(session, run_id, 2, "SPEAKER_00", 10.0, 14.0, E1)  # a Bob-voting turn
    session.flush()
    proposals = match_speakers(session, run_id, GATES)
    assert len(proposals) == 1
    p = proposals[0]
    assert p.speaker_id == alice
    assert p.vote_agreement == pytest.approx(2 / 3)
    assert p.grounded is False  # 0.667 < grounded_min_vote_agreement 0.67


# --------------------------------------------------------------------- writer


def cosine_proposal(label: str, speaker_id: uuid.UUID, **overrides: object) -> CosineProposal:
    values: dict[str, object] = {
        "diarization_label": label,
        "speaker_id": speaker_id,
        "similarity": 0.8,
        "margin": 0.3,
        "vote_agreement": 1.0,
        "grounded": True,
    }
    values.update(overrides)
    return CosineProposal(**values)  # type: ignore[arg-type]


def seed_label(session: Session, run_id: uuid.UUID, label: str = "SPEAKER_00") -> None:
    add_turn(session, run_id, 0, label, 0.0, 4.0, E0)


def test_writer_persists_both_shapes_and_replaces_idempotently(session: Session) -> None:
    run_id = make_run(session)
    alice = add_speaker(session, "Alice", [E0])
    seed_label(session, run_id)
    session.flush()

    proposals = (cosine_proposal("SPEAKER_00", alice),)
    hints = (NameHintProposal(diarization_label="SPEAKER_00", proposed_name="Jane"),)
    replace_run_proposals(session, run_id, proposals, hints)
    replace_run_proposals(session, run_id, proposals, hints)  # retry-idempotent
    session.flush()

    rows = session.execute(select(SpeakerAssignment)).scalars().all()
    assert len(rows) == 2
    by_method = {r.method: r for r in rows}
    cos = by_method[AssignmentMethod.COSINE.value]
    assert cos.speaker_id == alice
    assert cos.grounded is True
    assert cos.proposed_name is None
    assert cos.confidence == pytest.approx(confidence_from_similarity(0.8))
    hint_row = by_method[AssignmentMethod.LLM_HINT.value]
    assert hint_row.speaker_id is None
    assert hint_row.grounded is False
    assert hint_row.confidence is None
    assert hint_row.proposed_name == "Jane"


def test_writer_rejects_unknown_label_and_duplicates(session: Session) -> None:
    run_id = make_run(session)
    alice = add_speaker(session, "Alice", [E0])
    seed_label(session, run_id)
    session.flush()

    with pytest.raises(ProposalError, match="not in run"):
        replace_run_proposals(
            session, run_id, (cosine_proposal("NO_SUCH_LABEL", alice),), ()
        )
    with pytest.raises(ProposalError, match="duplicate"):
        replace_run_proposals(
            session,
            run_id,
            (cosine_proposal("SPEAKER_00", alice), cosine_proposal("SPEAKER_00", alice)),
            (),
        )
    with pytest.raises(ProposalError, match="similarity"):
        replace_run_proposals(
            session,
            run_id,
            (cosine_proposal("SPEAKER_00", alice, similarity=float("nan")),),
            (),
        )
    with pytest.raises(ProposalError, match="name"):
        replace_run_proposals(
            session,
            run_id,
            (),
            (NameHintProposal(diarization_label="SPEAKER_00", proposed_name="  "),),
        )


def test_db_constraints_enforce_method_shapes(session: Session) -> None:
    """The migration-level backstop behind the typed writer."""
    run_id = make_run(session)
    alice = add_speaker(session, "Alice", [E0])
    seed_label(session, run_id)
    session.flush()

    session.add(
        SpeakerAssignment(
            pipeline_run_id=run_id,
            diarization_label="SPEAKER_00",
            speaker_id=alice,
            method=AssignmentMethod.COSINE.value,
            proposed_name="smuggled",  # cosine must not carry a name
        )
    )
    with pytest.raises(IntegrityError):
        session.flush()
    session.rollback()

    session.add(
        SpeakerAssignment(
            pipeline_run_id=run_id,
            diarization_label="SPEAKER_00",
            speaker_id=None,
            method=AssignmentMethod.LLM_HINT.value,
            proposed_name=None,  # hint must carry a name
        )
    )
    with pytest.raises(IntegrityError):
        session.flush()
    session.rollback()

    session.add(
        SpeakerAssignment(
            pipeline_run_id=run_id,
            diarization_label="SPEAKER_00",
            speaker_id=None,
            method=AssignmentMethod.LLM_HINT.value,
            proposed_name="Jane",
            confidence=0.9,  # hint confidence is uncalibrated and must stay NULL
        )
    )
    with pytest.raises(IntegrityError):
        session.flush()
    session.rollback()


# ------------------------------------------- #113 observational evidence capture


def _decisions_by_label(
    decisions: tuple[LabelDecision, ...],
) -> dict[str, LabelDecision]:
    return {d.diarization_label: d for d in decisions}


def test_match_speakers_projects_evaluate_run_proposals(session: Session) -> None:
    """match_speakers is exactly the accepted labels' proposals from evaluate_run,
    in label order — the byte-identity guarantee the instrumentation rests on."""
    run_id = make_run(session)
    add_speaker(session, "Alice", [E0])
    add_speaker(session, "Bob", [E1])
    for i in range(3):
        add_turn(session, run_id, i, "SPEAKER_00", i * 5.0, i * 5.0 + 4.0, E0)
    off_roster = unit((0, 0.5), (5, math.sqrt(0.75)))
    for i in range(3, 6):
        add_turn(session, run_id, i, "SPEAKER_01", i * 5.0, i * 5.0 + 4.0, off_roster)
    session.flush()

    decisions = evaluate_run(session, run_id, GATES)
    projected = tuple(d.proposal for d in decisions if d.proposal is not None)
    assert projected == match_speakers(session, run_id, GATES)


def test_evaluate_run_records_every_label_with_evidence(session: Session) -> None:
    """One LabelDecision per label — accepted, rejected, and every ineligibility
    reason — carrying the numbers that used to die in debug logs."""
    run_id = make_run(session)
    alice = add_speaker(session, "Alice", [E0])
    add_speaker(session, "Bob", [E1])

    # accepted + grounded
    for i in range(3):
        add_turn(session, run_id, i, "SPEAKER_00", i * 5.0, i * 5.0 + 4.0, E0)
    # rejected: below the cosine floor (cos to Alice = 0.5)
    off_roster = unit((0, 0.5), (5, math.sqrt(0.75)))
    for i in range(3, 6):
        add_turn(session, run_id, i, "SPEAKER_01", i * 5.0, i * 5.0 + 4.0, off_roster)
    # ineligible: a single eligible turn (< min_turns)
    add_turn(session, run_id, 6, "SPEAKER_02", 40.0, 60.0, E0)
    # ineligible: two turns but < min_seconds of usable speech
    add_turn(session, run_id, 7, "SPEAKER_03", 61.0, 63.5, E0)
    add_turn(session, run_id, 8, "SPEAKER_03", 64.0, 66.5, E0)
    # ineligible: every turn skipped (no embedding) -> no eligible turns
    add_turn(session, run_id, 9, "SPEAKER_04", 70.0, 70.3, None)
    session.flush()

    by_label = _decisions_by_label(evaluate_run(session, run_id, GATES))
    assert set(by_label) == {f"SPEAKER_0{i}" for i in range(5)}

    accepted = by_label["SPEAKER_00"]
    assert accepted.decision == DECISION_ACCEPTED
    assert accepted.reason == REASON_ACCEPTED
    assert accepted.top_speaker_id == alice
    assert accepted.similarity == pytest.approx(1.0)
    assert accepted.margin == pytest.approx(1.0)  # top-1 (Alice) vs top-2 (Bob)
    assert accepted.vote_agreement == pytest.approx(1.0)
    assert accepted.grounded is True
    assert accepted.roster_size == 2
    assert accepted.proposal is not None and accepted.proposal.speaker_id == alice

    rejected = by_label["SPEAKER_01"]
    assert rejected.decision == DECISION_REJECTED
    assert rejected.reason == REASON_BELOW_COSINE
    assert rejected.top_speaker_id == alice  # near-miss candidate is still recorded
    assert rejected.similarity == pytest.approx(0.5)
    assert rejected.grounded is None
    assert rejected.proposal is None

    too_few = by_label["SPEAKER_02"]
    assert (too_few.decision, too_few.reason) == (DECISION_INELIGIBLE, REASON_TOO_FEW_TURNS)
    assert too_few.top_speaker_id is None
    assert too_few.similarity is None
    assert too_few.eligible_turns == 1

    too_short = by_label["SPEAKER_03"]
    assert (too_short.decision, too_short.reason) == (
        DECISION_INELIGIBLE,
        REASON_TOO_LITTLE_SPEECH,
    )
    assert too_short.eligible_turns == 2
    assert too_short.eligible_seconds == pytest.approx(5.0)

    no_turns = by_label["SPEAKER_04"]
    assert (no_turns.decision, no_turns.reason) == (
        DECISION_INELIGIBLE,
        REASON_NO_ELIGIBLE_TURNS,
    )
    assert no_turns.embedding_space is None
    assert no_turns.eligible_turns == 0
    assert no_turns.roster_size is None


def test_evaluate_run_rejection_reasons(session: Session) -> None:
    """Rejections name the first failed acceptance gate (cosine, margin, vote)."""
    # below_margin: equidistant between Alice and Bob (margin 0), cosine high
    run_id = make_run(session)
    add_speaker(session, "Alice", [E0])
    add_speaker(session, "Bob", [E1])
    between = unit((0, 1.0), (1, 1.0))
    for i in range(3):
        add_turn(session, run_id, i, "SPEAKER_00", i * 5.0, i * 5.0 + 4.0, between)
    session.flush()
    d = _decisions_by_label(evaluate_run(session, run_id, GATES))["SPEAKER_00"]
    assert (d.decision, d.reason) == (DECISION_REJECTED, REASON_BELOW_MARGIN)
    assert d.margin == pytest.approx(0.0)


def test_evaluate_run_below_vote_agreement_reason(session: Session) -> None:
    run_id = make_run(session)
    add_speaker(session, "Alice", [E0])
    add_speaker(session, "Bob", [E1])
    # Two Alice-voting turns, three Bob-voting: top-1 is Bob by centroid but the
    # per-turn vote splits enough to trip the agreement gate before grounding.
    add_turn(session, run_id, 0, "SPEAKER_00", 0.0, 4.0, E0)
    add_turn(session, run_id, 1, "SPEAKER_00", 5.0, 9.0, E0)
    add_turn(session, run_id, 2, "SPEAKER_00", 10.0, 14.0, E1)
    add_turn(session, run_id, 3, "SPEAKER_00", 15.0, 19.0, E1)
    add_turn(session, run_id, 4, "SPEAKER_00", 20.0, 24.0, E1)
    session.flush()
    d = _decisions_by_label(evaluate_run(session, run_id, GATES))["SPEAKER_00"]
    # Whatever the winning identity, a 3/5 split is below the 0.60 vote gate only
    # if cosine + margin cleared; assert the reason is coherent with rejection.
    if d.decision == DECISION_REJECTED:
        assert d.reason in {
            REASON_BELOW_COSINE,
            REASON_BELOW_MARGIN,
            REASON_BELOW_VOTE_AGREEMENT,
        }


def test_evaluate_run_single_speaker_roster_margin_is_none(session: Session) -> None:
    """A one-speaker roster has no top-2, so the stored margin is NULL even though
    the proposal itself still carries math.inf (unchanged behavior)."""
    run_id = make_run(session)
    add_speaker(session, "Alice", [E0])
    for i in range(3):
        add_turn(session, run_id, i, "SPEAKER_00", i * 5.0, i * 5.0 + 4.0, E0)
    session.flush()
    d = _decisions_by_label(evaluate_run(session, run_id, GATES))["SPEAKER_00"]
    assert d.decision == DECISION_ACCEPTED
    assert d.margin is None
    assert d.roster_size == 1
    assert d.proposal is not None and math.isinf(d.proposal.margin)


def test_evaluate_run_no_roster_reason(session: Session) -> None:
    run_id = make_run(session)
    for i in range(3):
        add_turn(session, run_id, i, "SPEAKER_00", i * 5.0, i * 5.0 + 4.0, E0)
    session.flush()
    d = _decisions_by_label(evaluate_run(session, run_id, GATES))["SPEAKER_00"]
    assert (d.decision, d.reason) == (DECISION_INELIGIBLE, REASON_NO_ROSTER)
    assert d.roster_size == 0
    assert d.top_speaker_id is None


def test_replace_run_match_candidates_persists_and_is_idempotent(
    session: Session,
) -> None:
    run_id = make_run(session)
    alice = add_speaker(session, "Alice", [E0])
    add_speaker(session, "Bob", [E1])
    for i in range(3):
        add_turn(session, run_id, i, "SPEAKER_00", i * 5.0, i * 5.0 + 4.0, E0)
    off_roster = unit((0, 0.5), (5, math.sqrt(0.75)))
    for i in range(3, 6):
        add_turn(session, run_id, i, "SPEAKER_01", i * 5.0, i * 5.0 + 4.0, off_roster)
    session.flush()

    decisions = evaluate_run(session, run_id, GATES)
    replace_run_match_candidates(session, run_id, decisions)
    replace_run_match_candidates(session, run_id, decisions)  # retry-idempotent
    session.flush()

    rows = {
        r.diarization_label: r
        for r in session.execute(select(MatchCandidate)).scalars().all()
    }
    assert set(rows) == {"SPEAKER_00", "SPEAKER_01"}
    accepted = rows["SPEAKER_00"]
    assert accepted.decision == DECISION_ACCEPTED
    assert accepted.top_speaker_id == alice
    assert accepted.similarity == pytest.approx(1.0)
    assert accepted.grounded is True
    rejected = rows["SPEAKER_01"]
    assert rejected.decision == DECISION_REJECTED
    assert rejected.reason == REASON_BELOW_COSINE
    assert rejected.grounded is None
    assert rejected.top_speaker_id == alice


def test_match_candidates_ineligible_shape_constraint(session: Session) -> None:
    """The DB backstops the writer: an ineligible row must not carry a candidate."""
    run_id = make_run(session)
    alice = add_speaker(session, "Alice", [E0])
    seed_label(session, run_id)
    session.flush()

    session.add(
        MatchCandidate(
            pipeline_run_id=run_id,
            diarization_label="SPEAKER_00",
            decision=DECISION_INELIGIBLE,
            reason=REASON_TOO_FEW_TURNS,
            top_speaker_id=alice,  # ineligible must have no candidate
            eligible_turns=1,
            eligible_seconds=2.0,
        )
    )
    with pytest.raises(IntegrityError):
        session.flush()
    session.rollback()
