"""Integration behavior of the additive LLM name pass (issue #38, names.llm).

Uses the FakeLLM/FailingLLM seams from tests/fakes.py — the producer's
evidence discipline (verbatim location or dropped), gating, additivity to the
offline producer's claims, abort-on-LLM-failure, and input-signature replay.
"""

import uuid
from collections.abc import Iterator

import pytest
from sqlalchemy.orm import Session, sessionmaker

from tests.fakes import FailingLLM, FakeLLM
from voxint.clients.base import SpeakerNameHint
from voxint.config import Settings
from voxint.db.models import MediaItem, PipelineRun, TranscriptSegment
from voxint.enrichment.producers.names import (
    NameProducerError,
    run_offline_name_producer,
)
from voxint.enrichment.producers.names_llm import run_llm_name_producer
from voxint.enrichment.queries import CandidateState, candidates_for_run

SETTINGS = Settings(_env_file=None, llm_enabled=True, enrichment_names_llm_enabled=True)
DISABLED = Settings(_env_file=None)


@pytest.fixture()
def session(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    with session_factory() as sess:
        yield sess


@pytest.fixture()
def run_id(session: Session) -> uuid.UUID:
    media = MediaItem(source_path=f"incoming/llm-names/{uuid.uuid4()}.wav")
    session.add(media)
    session.flush()
    run = PipelineRun(media_item_id=media.id)
    session.add(run)
    session.flush()
    for index, (label, text) in enumerate(
        [
            ("S0", "well jane doe checking in as always"),
            ("S1", "great to be here"),
        ]
    ):
        session.add(
            TranscriptSegment(
                pipeline_run_id=run.id,
                segment_index=index,
                start_seconds=float(index) * 10.0,
                end_seconds=float(index) * 10.0 + 5.0,
                raw_text=text,
                diarization_label=label,
            )
        )
    session.commit()
    return run.id


def test_located_self_hint_becomes_run_label_candidate(session: Session, run_id: uuid.UUID) -> None:
    llm = FakeLLM(name_hints=(SpeakerNameHint("S0", "jane doe", "self"),))
    producer_run = run_llm_name_producer(session, run_id=run_id, settings=SETTINGS, client=llm)
    session.commit()
    assert producer_run.outcome == "found"
    (view,) = candidates_for_run(session, run_id)
    assert view.candidate.value == "Jane Doe"
    assert view.candidate.target_kind == "run_label"
    assert view.candidate.diarization_label == "S0"
    assert view.candidate.score == 0.5
    assert view.candidate.score_components == {"llm": 1.0}
    (evidence,) = view.evidence
    assert evidence.kind == "transcript_segment"
    assert evidence.detail is not None
    assert evidence.detail["pattern_id"] == "llm_extraction"


def test_unlocatable_hint_is_dropped(session: Session, run_id: uuid.UUID) -> None:
    llm = FakeLLM(name_hints=(SpeakerNameHint("S0", "maria lopez", "self"),))
    producer_run = run_llm_name_producer(session, run_id=run_id, settings=SETTINGS, client=llm)
    session.commit()
    assert producer_run.outcome == "none"
    assert candidates_for_run(session, run_id) == []


def test_self_hint_located_only_in_other_label_is_dropped(
    session: Session, run_id: uuid.UUID
) -> None:
    # "jane doe" appears only in S0's speech; a self-hint for S1 must not
    # produce a cluster claim from another cluster's words.
    llm = FakeLLM(name_hints=(SpeakerNameHint("S1", "jane doe", "self"),))
    producer_run = run_llm_name_producer(session, run_id=run_id, settings=SETTINGS, client=llm)
    session.commit()
    assert producer_run.outcome == "none"


def test_other_hint_becomes_run_level(session: Session, run_id: uuid.UUID) -> None:
    llm = FakeLLM(name_hints=(SpeakerNameHint("S1", "jane doe", "other"),))
    run_llm_name_producer(session, run_id=run_id, settings=SETTINGS, client=llm)
    session.commit()
    (view,) = candidates_for_run(session, run_id)
    assert view.candidate.target_kind == "run"
    assert view.candidate.diarization_label is None


def test_additive_beside_offline_producer(session: Session, run_id: uuid.UUID) -> None:
    run_offline_name_producer(session, run_id=run_id, settings=SETTINGS)
    llm = FakeLLM(name_hints=(SpeakerNameHint("S0", "jane doe", "self"),))
    run_llm_name_producer(session, run_id=run_id, settings=SETTINGS, client=llm)
    session.commit()
    views = candidates_for_run(session, run_id)
    # The offline sweep found nothing pattern-shaped in this text; the LLM
    # claim exists beside it and neither superseded the other (distinct
    # producers own distinct lineages).
    assert [view.state for view in views] == [CandidateState.PROPOSED]
    # Rerunning the offline producer never touches the LLM claim.
    session.add(
        TranscriptSegment(
            pipeline_run_id=run_id,
            segment_index=2,
            start_seconds=20.0,
            end_seconds=25.0,
            raw_text="my name is bob smith",
            diarization_label="S0",
        )
    )
    session.commit()
    run_offline_name_producer(session, run_id=run_id, settings=SETTINGS)
    session.commit()
    states = {view.candidate.value: view.state for view in candidates_for_run(session, run_id)}
    assert states["Jane Doe"] is CandidateState.PROPOSED
    assert states["Bob Smith"] is CandidateState.PROPOSED


def test_llm_failure_aborts_without_recording(session: Session, run_id: uuid.UUID) -> None:
    with pytest.raises(NameProducerError, match="LLM name pass failed"):
        run_llm_name_producer(session, run_id=run_id, settings=SETTINGS, client=FailingLLM())


def test_disabled_flags_refuse(session: Session, run_id: uuid.UUID) -> None:
    with pytest.raises(NameProducerError, match="disabled"):
        run_llm_name_producer(session, run_id=run_id, settings=DISABLED, client=FakeLLM())


def test_identical_rerun_replays_without_requerying(session: Session, run_id: uuid.UUID) -> None:
    llm = FakeLLM(name_hints=(SpeakerNameHint("S0", "jane doe", "self"),))
    first = run_llm_name_producer(session, run_id=run_id, settings=SETTINGS, client=llm)
    session.commit()
    second = run_llm_name_producer(session, run_id=run_id, settings=SETTINGS, client=llm)
    session.commit()
    assert second.id == first.id
    assert len(llm.calls) == 1  # the short-circuit never re-queried the model


def test_substring_hit_is_not_verbatim_location(session: Session, run_id: uuid.UUID) -> None:
    # "Ann" appears only inside "joanne" — a substring, not a whole-word
    # match — so the hallucinated hint must be dropped, not evidenced.
    session.add(
        TranscriptSegment(
            pipeline_run_id=run_id,
            segment_index=2,
            start_seconds=20.0,
            end_seconds=25.0,
            raw_text="joanne welcomed everyone to the studio",
            diarization_label="S0",
        )
    )
    session.commit()
    llm = FakeLLM(name_hints=(SpeakerNameHint("S0", "Ann", "self"),))
    producer_run = run_llm_name_producer(session, run_id=run_id, settings=SETTINGS, client=llm)
    session.commit()
    assert producer_run.outcome == "none"


def test_batch_config_change_mints_new_generation(session: Session, run_id: uuid.UUID) -> None:
    # Batching shapes the model's context, so it is part of the replay
    # identity: a changed batch bound re-queries instead of replaying.
    llm = FakeLLM(name_hints=(SpeakerNameHint("S0", "jane doe", "self"),))
    first = run_llm_name_producer(session, run_id=run_id, settings=SETTINGS, client=llm)
    session.commit()
    resized = Settings(
        _env_file=None,
        llm_enabled=True,
        enrichment_names_llm_enabled=True,
        llm_batch_max_segments=1,
    )
    second = run_llm_name_producer(session, run_id=run_id, settings=resized, client=llm)
    session.commit()
    assert second.id != first.id
    assert second.generation == 2
    assert len(llm.calls) > 1
