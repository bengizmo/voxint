"""Integration behavior of the offline name-candidate producer (issue #38).

Real-Postgres coverage of persistence through the #37 draft layer: run vs
run_label targets, evidence rows pointing at the real metadata snapshot and
transcript segments, outcome derivation, the input-signature idempotency
short-circuit, supersession across input changes, and decided candidates
surviving reruns.
"""

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from voxint.config import Settings
from voxint.db.models import (
    EnrichmentCandidate,
    EnrichmentCandidateEvidence,
    MediaItem,
    MediaSourceMetadata,
    PipelineRun,
    ProfileDecision,
    TranscriptSegment,
)
from voxint.enrichment.producers.names import (
    NameProducerError,
    run_offline_name_producer,
)
from voxint.enrichment.queries import CandidateState, candidates_for_run
from voxint.enrichment.review import record_profile_decision

NOW = datetime.now(tz=UTC)
SETTINGS = Settings(_env_file=None)


@pytest.fixture()
def session(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    with session_factory() as sess:
        yield sess


@pytest.fixture()
def run_id(session: Session) -> uuid.UUID:
    media = MediaItem(source_path=f"incoming/names/{uuid.uuid4()}.wav")
    session.add(media)
    session.flush()
    run = PipelineRun(media_item_id=media.id)
    session.add(run)
    session.commit()
    return run.id


def _add_metadata(
    session: Session,
    run_id: uuid.UUID,
    *,
    title: str | None = None,
    description: str | None = None,
    channel: str | None = None,
) -> MediaSourceMetadata:
    run = session.get(PipelineRun, run_id)
    assert run is not None
    snapshot = MediaSourceMetadata(
        media_item_id=run.media_item_id,
        source_kind="ytdlp",
        title=title,
        description=description,
        channel=channel,
        raw_schema_version=1,
        acquired_at=NOW,
    )
    session.add(snapshot)
    session.commit()
    return snapshot


def _add_segment(
    session: Session,
    run_id: uuid.UUID,
    *,
    index: int,
    text: str,
    label: str | None = "SPEAKER_00",
    suspect: bool = False,
) -> TranscriptSegment:
    segment = TranscriptSegment(
        pipeline_run_id=run_id,
        segment_index=index,
        start_seconds=float(index) * 10.0,
        end_seconds=float(index) * 10.0 + 5.0,
        raw_text=text,
        diarization_label=label,
        suspect=suspect,
    )
    session.add(segment)
    session.commit()
    return segment


def _produce(session: Session, run_id: uuid.UUID):  # type: ignore[no-untyped-def]
    producer_run = run_offline_name_producer(session, run_id=run_id, settings=SETTINGS)
    session.commit()
    return producer_run


def test_persists_run_and_run_label_candidates_with_real_evidence(
    session: Session, run_id: uuid.UUID
) -> None:
    snapshot = _add_metadata(session, run_id, title="Interview with Jane Doe")
    segment = _add_segment(
        session, run_id, index=0, text="hi my name is bob smith", label="SPEAKER_01"
    )

    producer_run = _produce(session, run_id)
    assert producer_run.outcome == "found"
    assert producer_run.generation == 1
    assert producer_run.covered_fields == ["name"]

    views = candidates_for_run(session, run_id)
    by_name = {view.candidate.value: view for view in views}
    assert set(by_name) == {"Jane Doe", "Bob Smith"}

    jane = by_name["Jane Doe"]
    assert jane.candidate.target_kind == "run"
    (jane_evidence,) = jane.evidence
    assert jane_evidence.kind == "metadata_field"
    assert jane_evidence.source_metadata_id == snapshot.id
    assert jane_evidence.source_field == "title"
    assert jane_evidence.detail is not None
    assert jane_evidence.detail["pattern_id"] == "title_interview_with"

    bob = by_name["Bob Smith"]
    assert bob.candidate.target_kind == "run_label"
    assert bob.candidate.diarization_label == "SPEAKER_01"
    (bob_evidence,) = bob.evidence
    assert bob_evidence.kind == "transcript_segment"
    assert bob_evidence.transcript_segment_id == segment.id
    assert bob_evidence.timestamp_seconds == segment.start_seconds
    assert bob.candidate.score is not None
    assert bob.candidate.score_components["base"] == 0.9


def test_empty_run_records_authoritative_none(session: Session, run_id: uuid.UUID) -> None:
    producer_run = _produce(session, run_id)
    assert producer_run.outcome == "none"
    assert candidates_for_run(session, run_id) == []


def test_identical_rerun_short_circuits_to_same_row(session: Session, run_id: uuid.UUID) -> None:
    _add_metadata(session, run_id, title="Interview with Jane Doe")
    first = _produce(session, run_id)
    second = _produce(session, run_id)
    assert second.id == first.id
    assert second.generation == 1
    # No duplicate candidates, nothing superseded.
    views = candidates_for_run(session, run_id)
    assert [view.state for view in views] == [CandidateState.PROPOSED]


def test_changed_input_supersedes_previous_generation(session: Session, run_id: uuid.UUID) -> None:
    _add_metadata(session, run_id, title="Interview with Jane Doe")
    first = _produce(session, run_id)
    _add_segment(session, run_id, index=0, text="my name is bob smith")
    second = _produce(session, run_id)

    assert second.id != first.id
    assert second.generation == 2
    views = candidates_for_run(session, run_id)
    by_gen: dict[int, list[CandidateState]] = {}
    for view in views:
        gen = session.get(type(second), view.candidate.producer_run_id)
        assert gen is not None
        by_gen.setdefault(gen.generation, []).append(view.state)
    assert by_gen[1] == [CandidateState.SUPERSEDED]
    assert set(by_gen[2]) == {CandidateState.PROPOSED}


def test_decided_candidate_survives_rerun(session: Session, run_id: uuid.UUID) -> None:
    _add_metadata(session, run_id, title="Interview with Jane Doe")
    _produce(session, run_id)
    (view,) = candidates_for_run(session, run_id)
    record_profile_decision(
        session,
        candidate_id=view.candidate.id,
        decision=ProfileDecision.ACCEPT,
        operator="ben",
        idempotency_key=f"accept-{view.candidate.id}",
    )
    session.commit()

    _add_segment(session, run_id, index=0, text="my name is bob smith")
    _produce(session, run_id)

    states: dict[str, set[CandidateState]] = {}
    for candidate_view in candidates_for_run(session, run_id):
        states.setdefault(candidate_view.candidate.value, set()).add(candidate_view.state)
    # The human act stands untouched; the rerun re-extracts the same name as a
    # fresh proposed duplicate beside it (decided candidates are terminal and
    # never superseded) — the console groups these by value.
    assert states["Jane Doe"] == {CandidateState.ACCEPTED, CandidateState.PROPOSED}
    assert states["Bob Smith"] == {CandidateState.PROPOSED}


def test_missing_run_raises_instead_of_recording_none(session: Session) -> None:
    with pytest.raises(NameProducerError, match="not found"):
        run_offline_name_producer(session, run_id=uuid.uuid4(), settings=SETTINGS)


def test_cross_run_isolation(session: Session, run_id: uuid.UUID) -> None:
    """Evidence and candidates bind to the invoked run, never a sibling's."""
    _add_metadata(session, run_id, title="Interview with Jane Doe")

    other_media = MediaItem(source_path=f"incoming/names/{uuid.uuid4()}.wav")
    session.add(other_media)
    session.flush()
    other_run = PipelineRun(media_item_id=other_media.id)
    session.add(other_run)
    session.commit()
    _add_segment(session, other_run.id, index=0, text="my name is mallory intruder")

    _produce(session, run_id)
    values = [view.candidate.value for view in candidates_for_run(session, run_id)]
    assert values == ["Jane Doe"]

    evidence_rows = list(
        session.execute(
            select(EnrichmentCandidateEvidence)
            .join(
                EnrichmentCandidate,
                EnrichmentCandidateEvidence.candidate_id == EnrichmentCandidate.id,
            )
            .where(EnrichmentCandidate.pipeline_run_id == run_id)
        ).scalars()
    )
    assert all(row.transcript_segment_id is None for row in evidence_rows)


def test_suspect_segment_penalty_recorded(session: Session, run_id: uuid.UUID) -> None:
    _add_segment(session, run_id, index=0, text="my name is jane doe", suspect=True)
    _produce(session, run_id)
    (view,) = candidates_for_run(session, run_id)
    assert view.candidate.score_components["base"] == 0.45
    assert view.candidate.score_components["suspect_penalty_applied"] == 1.0
    (evidence,) = view.evidence
    assert evidence.detail is not None
    assert evidence.detail["suspect"] is True


def test_config_records_versions_and_signature(session: Session, run_id: uuid.UUID) -> None:
    _add_metadata(session, run_id, title="Interview with Jane Doe")
    producer_run = _produce(session, run_id)
    assert producer_run.config is not None
    assert producer_run.config["pattern_set_version"] == 1
    assert producer_run.config["scoring_version"] == 1
    assert producer_run.config["input_signature"]
    assert producer_run.idempotency_key.endswith(producer_run.config["input_signature"][:16])
