"""Integration tests for auto_enroll_run() (#275).

Exercises the DB-coupled auto-enrollment flow against a real Postgres session:
idempotency, human-decision preemption (pre-filter and mid-run recheck),
naming offset, intra-run consolidation, per-label failure containment,
archived-speaker skipping, and grounding-floor enforcement.
"""

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from voxint.adjudication.ledger import record_decision
from voxint.adjudication.resolver import Resolution, label_states
from voxint.db.models import (
    EMBEDDING_DIM,
    AdjudicationDecision,
    Decision,
    DiarizationTurn,
    MediaItem,
    PipelineRun,
    RunStatus,
    Speaker,
    SpeakerEmbedding,
)
from voxint.speakers.auto_enroll import auto_enroll_run
from voxint.speakers.matching import MatchingGates

SPACE = "titanet-large-v2"
GATES = MatchingGates()


def _unit(dim: int) -> list[float]:
    vector = [0.0] * EMBEDDING_DIM
    vector[dim % EMBEDDING_DIM] = 1.0
    return vector


def _make_run(session: Session) -> uuid.UUID:
    media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
    session.add(media)
    session.flush()
    run = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
    session.add(run)
    session.flush()
    return run.id


def _add_turns(
    session: Session,
    run_id: uuid.UUID,
    label: str,
    *,
    count: int = 4,
    vector: list[float] | None = None,
    duration: float = 8.0,
    start_index: int = 0,
) -> None:
    vec = vector or _unit(0)
    for i in range(count):
        idx = start_index + i
        session.add(
            DiarizationTurn(
                pipeline_run_id=run_id,
                turn_index=idx,
                start_seconds=float(idx * 20),
                end_seconds=float(idx * 20) + duration,
                label=label,
                overlap=False,
                overlap_seconds=0.0,
                embedding=vec,
                embedding_space=SPACE,
            )
        )


def _enroll_speaker(
    session: Session,
    name: str,
    vector: list[float] | None = None,
) -> uuid.UUID:
    """Pre-seed an active roster speaker (no provenance FK -- these represent
    hand-enrolled speakers, not auto-enrolled ones)."""
    speaker = Speaker(display_name=name)
    session.add(speaker)
    session.flush()
    session.add(
        SpeakerEmbedding(
            speaker_id=speaker.id,
            embedding_space=SPACE,
            embedding=vector or _unit(0),
        )
    )
    session.flush()
    return speaker.id


# -- Basic enrollment ---------------------------------------------------------

def test_auto_enroll_creates_speakers_for_unresolved_labels(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = _make_run(session)
        _add_turns(session, run_id, "S0", vector=_unit(0), start_index=0)
        _add_turns(session, run_id, "S1", vector=_unit(1), start_index=10)
        session.commit()

    with session_factory() as session:
        result = auto_enroll_run(session, run_id, GATES)
        session.commit()

    assert result.created == 2
    assert result.matched == 0
    assert result.skipped == 0

    with session_factory() as session:
        speakers = session.execute(
            select(Speaker).where(Speaker.display_name.like("Voice %"))
        ).scalars().all()
        assert len(speakers) == 2
        names = sorted(s.display_name for s in speakers)
        assert names == ["Voice 1", "Voice 2"]

        decisions = session.execute(
            select(AdjudicationDecision).where(
                AdjudicationDecision.pipeline_run_id == run_id,
                AdjudicationDecision.decision == Decision.AUTO_ENROLL.value,
            )
        ).scalars().all()
        assert len(decisions) == 2

        for d in decisions:
            assert d.speaker_id is not None
            assert d.operator == "system:auto_enroll"
            assert d.idempotency_key.startswith("auto_enroll:")

        embeddings = session.execute(
            select(SpeakerEmbedding).where(
                SpeakerEmbedding.source_pipeline_run_id == run_id
            )
        ).scalars().all()
        assert len(embeddings) == 2
        for e in embeddings:
            assert e.embedding_space == SPACE
            assert e.source_diarization_label in ("S0", "S1")
            assert e.source_adjudication_decision_id is not None

        states = label_states(session, run_id)
        for s in states:
            assert s.resolution is Resolution.AUTO_ENROLL
            assert s.speaker_id is not None


# -- Idempotency --------------------------------------------------------------

def test_auto_enroll_second_invocation_is_noop(
    session_factory: sessionmaker[Session],
) -> None:
    """Second call finds all labels already resolved; returns (0, 0, 0)."""
    with session_factory() as session:
        run_id = _make_run(session)
        _add_turns(session, run_id, "S0", vector=_unit(0), start_index=0)
        _add_turns(session, run_id, "S1", vector=_unit(1), start_index=10)
        session.commit()

    with session_factory() as session:
        first = auto_enroll_run(session, run_id, GATES)
        session.commit()

    assert first.created == 2

    with session_factory() as session:
        second = auto_enroll_run(session, run_id, GATES)
        session.commit()

    assert second.created == 0
    assert second.matched == 0
    assert second.skipped == 0

    with session_factory() as session:
        speaker_count = session.scalar(
            select(func.count()).select_from(Speaker).where(
                Speaker.display_name.like("Voice %")
            )
        )
        assert speaker_count == 2

        decision_count = session.scalar(
            select(func.count()).select_from(AdjudicationDecision).where(
                AdjudicationDecision.pipeline_run_id == run_id,
            )
        )
        assert decision_count == 2


# -- Human decision preemption ------------------------------------------------

def test_auto_enroll_skips_label_with_human_decision(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = _make_run(session)
        _add_turns(session, run_id, "S0", vector=_unit(0), start_index=0)
        _add_turns(session, run_id, "S1", vector=_unit(1), start_index=10)

        human_speaker = Speaker(display_name="Alice")
        session.add(human_speaker)
        session.flush()

        record_decision(
            session,
            pipeline_run_id=run_id,
            diarization_label="S0",
            decision=Decision.ASSIGN,
            operator="ben",
            idempotency_key="human-assign-s0",
            speaker_id=human_speaker.id,
        )
        session.commit()

    with session_factory() as session:
        result = auto_enroll_run(session, run_id, GATES)
        session.commit()

    assert result.created == 1
    assert result.skipped == 0
    assert result.matched == 0

    with session_factory() as session:
        voices = session.execute(
            select(Speaker).where(Speaker.display_name.like("Voice %"))
        ).scalars().all()
        assert len(voices) == 1
        assert voices[0].display_name == "Voice 1"

        states = label_states(session, run_id)
        s0 = next(s for s in states if s.label == "S0")
        s1 = next(s for s in states if s.label == "S1")
        assert s0.resolution is Resolution.HUMAN_ASSIGN
        assert s1.resolution is Resolution.AUTO_ENROLL


# -- Decision injected after candidate selection (exercises _label_has_decision)

def test_auto_enroll_decision_injected_after_candidate_selection(
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A human decision committed (from another session) after candidates are
    selected but before per-label processing: _label_has_decision() catches it
    and the label is skipped."""
    with session_factory() as session:
        run_id = _make_run(session)
        _add_turns(session, run_id, "S0", vector=_unit(0), start_index=0)
        session.commit()

    from voxint.speakers import auto_enroll as _ae
    from voxint.speakers.matching import eligible_label_vectors as _real_elv

    def _elv_with_injection(session: Session, rid: uuid.UUID, gates: MatchingGates) -> dict:  # type: ignore[type-arg]
        result = _real_elv(session, rid, gates)
        with session_factory() as s2:
            speaker = Speaker(display_name="Injected")
            s2.add(speaker)
            s2.flush()
            record_decision(
                s2,
                pipeline_run_id=rid,
                diarization_label="S0",
                decision=Decision.ASSIGN,
                operator="ben",
                idempotency_key="injected-decision",
                speaker_id=speaker.id,
            )
            s2.commit()
        return result

    monkeypatch.setattr(_ae, "eligible_label_vectors", _elv_with_injection)

    with session_factory() as session:
        result = auto_enroll_run(session, run_id, GATES)
        session.commit()

    assert result.created == 0
    assert result.matched == 0
    assert result.skipped == 1

    with session_factory() as session:
        voice_count = session.scalar(
            select(func.count()).select_from(Speaker).where(
                Speaker.display_name.like("Voice %")
            )
        )
        assert voice_count == 0

        auto_decisions = session.execute(
            select(AdjudicationDecision).where(
                AdjudicationDecision.pipeline_run_id == run_id,
                AdjudicationDecision.decision == Decision.AUTO_ENROLL.value,
            )
        ).scalars().all()
        assert len(auto_decisions) == 0

        states = label_states(session, run_id)
        s0 = next(s for s in states if s.label == "S0")
        assert s0.resolution is Resolution.HUMAN_ASSIGN


# -- Naming offset with pre-existing Voice N ----------------------------------

def test_auto_enroll_names_past_existing_voice_numbers(
    session_factory: sessionmaker[Session],
) -> None:
    """_next_voice_number returns MAX(N)+1; tests the gap-skipping SQL."""
    with session_factory() as session:
        session.add(Speaker(display_name="Voice 1"))
        session.add(Speaker(display_name="Voice 3"))
        session.flush()

        run_id = _make_run(session)
        _add_turns(session, run_id, "S0", vector=_unit(0), start_index=0)
        session.commit()

    with session_factory() as session:
        result = auto_enroll_run(session, run_id, GATES)
        session.commit()

    assert result.created == 1

    with session_factory() as session:
        new_speaker = session.execute(
            select(Speaker).where(
                Speaker.display_name.like("Voice %"),
                Speaker.display_name.notin_(["Voice 1", "Voice 3"]),
            )
        ).scalar_one()
        assert new_speaker.display_name == "Voice 4"


# -- Intra-run consolidation --------------------------------------------------

def test_auto_enroll_two_labels_same_voice_consolidate(
    session_factory: sessionmaker[Session],
) -> None:
    """Two unresolved labels with identical embeddings: the first creates a
    speaker, the second matches it (roster re-read per label)."""
    vec = _unit(5)
    with session_factory() as session:
        run_id = _make_run(session)
        _add_turns(session, run_id, "S0", vector=vec, start_index=0)
        _add_turns(session, run_id, "S1", vector=vec, start_index=10)
        session.commit()

    with session_factory() as session:
        result = auto_enroll_run(session, run_id, GATES)
        session.commit()

    assert result.created == 1
    assert result.matched == 1
    assert result.skipped == 0

    with session_factory() as session:
        voices = session.execute(
            select(Speaker).where(Speaker.display_name.like("Voice %"))
        ).scalars().all()
        assert len(voices) == 1

        decisions = session.execute(
            select(AdjudicationDecision).where(
                AdjudicationDecision.pipeline_run_id == run_id,
                AdjudicationDecision.decision == Decision.AUTO_ENROLL.value,
            )
        ).scalars().all()
        assert len(decisions) == 2
        speaker_ids = {d.speaker_id for d in decisions}
        assert len(speaker_ids) == 1

        embeddings = session.execute(
            select(SpeakerEmbedding).where(
                SpeakerEmbedding.source_pipeline_run_id == run_id
            )
        ).scalars().all()
        assert len(embeddings) == 2
        labels = {e.source_diarization_label for e in embeddings}
        assert labels == {"S0", "S1"}


# -- Per-label failure containment --------------------------------------------

def test_auto_enroll_per_label_failure_containment(
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exception inside one label's savepoint does not roll back siblings."""
    with session_factory() as session:
        run_id = _make_run(session)
        _add_turns(session, run_id, "S0", vector=_unit(0), start_index=0)
        _add_turns(session, run_id, "S1", vector=_unit(1), start_index=10)
        session.commit()

    from voxint.speakers import auto_enroll as _ae

    real_record = record_decision

    def _fail_on_s0(*args: object, **kwargs: object) -> object:
        if kwargs.get("diarization_label") == "S0":
            raise RuntimeError("injected failure for S0")
        return real_record(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(_ae, "record_decision", _fail_on_s0)

    with session_factory() as session:
        result = auto_enroll_run(session, run_id, GATES)
        session.commit()

    assert result.created == 1
    assert result.skipped == 1

    with session_factory() as session:
        voices = session.execute(
            select(Speaker).where(Speaker.display_name.like("Voice %"))
        ).scalars().all()
        assert len(voices) == 1

        decisions = session.execute(
            select(AdjudicationDecision).where(
                AdjudicationDecision.pipeline_run_id == run_id,
                AdjudicationDecision.decision == Decision.AUTO_ENROLL.value,
            )
        ).scalars().all()
        assert len(decisions) == 1
        assert decisions[0].diarization_label == "S1"

        states = label_states(session, run_id)
        s0 = next(s for s in states if s.label == "S0")
        s1 = next(s for s in states if s.label == "S1")
        assert s0.resolution is Resolution.UNRESOLVED
        assert s1.resolution is Resolution.AUTO_ENROLL


# -- Archived speaker not matched ---------------------------------------------

def test_auto_enroll_does_not_match_archived_speaker(
    session_factory: sessionmaker[Session],
) -> None:
    vec = _unit(7)
    with session_factory() as session:
        archived_id = _enroll_speaker(session, "Archived Person", vector=vec)
        archived = session.get(Speaker, archived_id)
        assert archived is not None
        archived.deleted_at = datetime.now(UTC)
        session.flush()

        run_id = _make_run(session)
        _add_turns(session, run_id, "S0", vector=vec, start_index=0)
        session.commit()

    with session_factory() as session:
        result = auto_enroll_run(session, run_id, GATES)
        session.commit()

    assert result.created == 1
    assert result.matched == 0

    with session_factory() as session:
        new_speaker = session.execute(
            select(Speaker).where(Speaker.display_name == "Voice 1")
        ).scalar_one()
        assert new_speaker.id != archived_id


# -- Matching an existing active speaker --------------------------------------

def test_auto_enroll_matches_active_roster_speaker(
    session_factory: sessionmaker[Session],
) -> None:
    vec = _unit(3)
    with session_factory() as session:
        existing_id = _enroll_speaker(session, "Known Speaker", vector=vec)

        run_id = _make_run(session)
        _add_turns(session, run_id, "S0", vector=vec, start_index=0)
        session.commit()

    with session_factory() as session:
        result = auto_enroll_run(session, run_id, GATES)
        session.commit()

    assert result.created == 0
    assert result.matched == 1
    assert result.skipped == 0

    with session_factory() as session:
        decision = session.execute(
            select(AdjudicationDecision).where(
                AdjudicationDecision.pipeline_run_id == run_id,
                AdjudicationDecision.decision == Decision.AUTO_ENROLL.value,
            )
        ).scalar_one()
        assert decision.speaker_id == existing_id

        voice_count = session.scalar(
            select(func.count()).select_from(Speaker).where(
                Speaker.display_name.like("Voice %")
            )
        )
        assert voice_count == 0


# -- Grounding floor: independent gate predicates -----------------------------

def test_auto_enroll_skips_label_below_turn_floor(
    session_factory: sessionmaker[Session],
) -> None:
    """Enough seconds but too few turns: grounded_min_turns gate fires."""
    with session_factory() as session:
        run_id = _make_run(session)
        _add_turns(session, run_id, "S0", vector=_unit(0), count=4, start_index=0)
        _add_turns(
            session, run_id, "S1", vector=_unit(1),
            count=GATES.grounded_min_turns - 1,
            duration=GATES.grounded_min_seconds + 1.0,
            start_index=10,
        )
        session.commit()

    with session_factory() as session:
        result = auto_enroll_run(session, run_id, GATES)
        session.commit()

    assert result.created == 1
    assert result.skipped == 1

    with session_factory() as session:
        states = label_states(session, run_id)
        s0 = next(s for s in states if s.label == "S0")
        s1 = next(s for s in states if s.label == "S1")
        assert s0.resolution is Resolution.AUTO_ENROLL
        assert s1.resolution is Resolution.UNRESOLVED


def test_auto_enroll_skips_label_below_seconds_floor(
    session_factory: sessionmaker[Session],
) -> None:
    """Enough turns but too few seconds: grounded_min_seconds gate fires."""
    seconds_per_turn = (GATES.grounded_min_seconds - 1.0) / (GATES.grounded_min_turns + 1)
    with session_factory() as session:
        run_id = _make_run(session)
        _add_turns(session, run_id, "S0", vector=_unit(0), count=4, start_index=0)
        _add_turns(
            session, run_id, "S1", vector=_unit(1),
            count=GATES.grounded_min_turns + 1,
            duration=seconds_per_turn,
            start_index=10,
        )
        session.commit()

    with session_factory() as session:
        result = auto_enroll_run(session, run_id, GATES)
        session.commit()

    assert result.created == 1
    assert result.skipped == 1

    with session_factory() as session:
        states = label_states(session, run_id)
        s0 = next(s for s in states if s.label == "S0")
        s1 = next(s for s in states if s.label == "S1")
        assert s0.resolution is Resolution.AUTO_ENROLL
        assert s1.resolution is Resolution.UNRESOLVED
