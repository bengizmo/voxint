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
    CosineProposal,
    MatchingGates,
    NameHintProposal,
    ProposalError,
    confidence_from_similarity,
    match_speakers,
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
