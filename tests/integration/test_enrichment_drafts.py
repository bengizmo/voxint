"""Integration behavior of the enrichment drafts writer (issue #37).

Real-Postgres coverage of the atomic finalization: outcome derivation,
generation allocation, the supersession matrix (covered-fields-only,
same-producer-same-scope-only, decided-candidates-untouched, none-run
supersedes), and idempotent replay.
"""

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from sqlalchemy import select, text
from sqlalchemy.orm import Session, sessionmaker

from voxint.db.models import (
    ClaimField,
    EnrichmentCandidate,
    EnrichmentProducerRun,
    MediaItem,
    PipelineRun,
    ProfileDecision,
    Speaker,
)
from voxint.enrichment.drafts import (
    CandidateDraft,
    EnrichmentScope,
    MetadataEvidence,
    TranscriptEvidence,
    UrlEvidence,
    record_producer_run,
)
from voxint.enrichment.queries import (
    CandidateState,
    candidates_for_run,
    candidates_for_speaker,
    latest_producer_run,
)
from voxint.enrichment.review import ConflictingReplayError, record_profile_decision

NOW = datetime.now(tz=UTC)


@pytest.fixture()
def session(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    with session_factory() as sess:
        yield sess


@pytest.fixture()
def speaker_id(session: Session) -> uuid.UUID:
    speaker = Speaker(display_name="Draft Target Speaker")
    session.add(speaker)
    session.commit()
    return speaker.id


@pytest.fixture()
def run_id(session: Session) -> uuid.UUID:
    media = MediaItem(source_path="incoming/enrichment/source.wav")
    session.add(media)
    session.flush()
    run = PipelineRun(media_item_id=media.id)
    session.add(run)
    session.commit()
    return run.id


def _bio_draft(speaker_id: uuid.UUID, value: str = "Podcast host.") -> CandidateDraft:
    return CandidateDraft(
        target=EnrichmentScope.speaker(speaker_id),
        field=ClaimField.BIO,
        value=value,
        evidence=(UrlEvidence(url="https://example.com/about", retrieved_at=NOW),),
    )


def _record(
    session: Session,
    scope: EnrichmentScope,
    candidates: tuple[CandidateDraft, ...],
    *,
    producer: str = "name_miner",
    covered: tuple[ClaimField, ...] = (ClaimField.BIO, ClaimField.NAME),
    key: str | None = None,
) -> EnrichmentProducerRun:
    run = record_producer_run(
        session,
        producer=producer,
        producer_version="1.0",
        scope=scope,
        covered_fields=covered,
        candidates=candidates,
        idempotency_key=key or f"k-{uuid.uuid4()}",
        started_at=NOW,
        completed_at=NOW,
    )
    session.commit()
    return run


def test_found_run_with_mixed_targets_and_evidence(
    session: Session, run_id: uuid.UUID
) -> None:
    """A run-scope invocation may emit run-level AND label-level claims."""
    media_row = session.execute(select(MediaItem)).scalar_one()
    session.execute(
        text(
            "INSERT INTO media_source_metadata"
            " (id, media_item_id, source_kind, raw_schema_version, acquired_at)"
            f" VALUES ('{uuid.uuid4()}', '{media_row.id}', 'ytdlp', 1, now())"
        )
    )
    metadata_id = session.execute(
        text("SELECT id FROM media_source_metadata")
    ).scalar_one()
    session.execute(
        text(
            "INSERT INTO transcript_segments"
            " (id, pipeline_run_id, segment_index, start_seconds, end_seconds, raw_text)"
            f" VALUES ('{uuid.uuid4()}', '{run_id}', 0, 0, 5, 'I am Jane')"
        )
    )
    segment_id = session.execute(
        text("SELECT id FROM transcript_segments")
    ).scalar_one()
    session.commit()

    drafts = (
        CandidateDraft(
            target=EnrichmentScope.run(run_id),
            field=ClaimField.NAME,
            value="Jane Interviewee",
            evidence=(
                MetadataEvidence(
                    source_metadata_id=metadata_id,
                    source_field="title",
                    snippet="Interview with Jane Interviewee",
                ),
            ),
            score=0.6,
            score_components={"title_match": 0.6},
        ),
        CandidateDraft(
            target=EnrichmentScope.run_label(run_id, "SPEAKER_00"),
            field=ClaimField.NAME,
            value="Jane Interviewee",
            evidence=(
                TranscriptEvidence(
                    transcript_segment_id=segment_id,
                    timestamp_seconds=1.2,
                    snippet="I am Jane",
                ),
                UrlEvidence(url="https://example.com/ep1"),
            ),
        ),
    )
    run = _record(session, EnrichmentScope.run(run_id), drafts, covered=(ClaimField.NAME,))

    assert run.outcome == "found"
    assert run.generation == 1
    views = candidates_for_run(session, run_id)
    assert [v.state for v in views] == [CandidateState.PROPOSED, CandidateState.PROPOSED]
    by_kind = {v.candidate.target_kind: v for v in views}
    assert by_kind["run"].evidence[0].source_field == "title"
    label_view = by_kind["run_label"]
    assert [e.ordinal for e in label_view.evidence] == [0, 1]
    assert label_view.evidence[0].kind == "transcript_segment"
    assert label_view.evidence[1].kind == "url"


def test_none_outcome_is_recorded(session: Session, speaker_id: uuid.UUID) -> None:
    """'We looked and found nothing' persists as an explicit outcome row."""
    run = _record(session, EnrichmentScope.speaker(speaker_id), ())
    assert run.outcome == "none"
    assert candidates_for_speaker(session, speaker_id) == []
    latest = latest_producer_run(
        session, "name_miner", EnrichmentScope.speaker(speaker_id)
    )
    assert latest is not None and latest.id == run.id


def test_supersession_matrix(session: Session, speaker_id: uuid.UUID) -> None:
    scope = EnrichmentScope.speaker(speaker_id)
    gen1 = _record(
        session,
        scope,
        (
            _bio_draft(speaker_id, "Old bio."),
            CandidateDraft(
                target=scope,
                field=ClaimField.NAME,
                value="Jane",
                evidence=(UrlEvidence(url="https://example.com/a"),),
            ),
        ),
    )
    # a decided candidate must survive supersession
    accepted = next(
        v
        for v in candidates_for_speaker(session, speaker_id)
        if v.candidate.field == ClaimField.NAME.value
    )
    record_profile_decision(
        session,
        candidate_id=accepted.candidate.id,
        decision=ProfileDecision.ACCEPT,
        operator="ben",
        idempotency_key="d1",
    )
    session.commit()
    # a different producer over the same scope must not supersede anything
    _record(session, scope, (_bio_draft(speaker_id, "Other producer bio."),),
            producer="web_researcher")
    # a different scope of the same producer must not supersede anything
    other = Speaker(display_name="Unrelated Speaker")
    session.add(other)
    session.commit()
    _record(session, EnrichmentScope.speaker(other.id), (_bio_draft(other.id),))

    # the rerun covers only BIO — the accepted NAME row and uncovered fields survive
    gen2 = _record(
        session,
        scope,
        (_bio_draft(speaker_id, "New bio."),),
        covered=(ClaimField.BIO,),
    )
    assert gen2.generation == gen1.generation + 1

    states = {
        (v.candidate.field, v.candidate.value): v.state
        for v in candidates_for_speaker(session, speaker_id)
    }
    assert states[("bio", "Old bio.")] is CandidateState.SUPERSEDED
    assert states[("bio", "New bio.")] is CandidateState.PROPOSED
    assert states[("name", "Jane")] is CandidateState.ACCEPTED
    assert states[("bio", "Other producer bio.")] is CandidateState.PROPOSED
    superseded = session.execute(
        select(EnrichmentCandidate).where(
            EnrichmentCandidate.superseded_by_producer_run_id.is_not(None)
        )
    ).scalars().all()
    assert [c.superseded_by_producer_run_id for c in superseded] == [gen2.id]
    # the unrelated speaker's draft is untouched
    assert [
        v.state for v in candidates_for_speaker(session, other.id)
    ] == [CandidateState.PROPOSED]


def test_none_run_supersedes_covered_proposals(
    session: Session, speaker_id: uuid.UUID
) -> None:
    scope = EnrichmentScope.speaker(speaker_id)
    _record(session, scope, (_bio_draft(speaker_id),), covered=(ClaimField.BIO,))
    none_run = _record(session, scope, (), covered=(ClaimField.BIO,))
    assert none_run.outcome == "none"
    (view,) = candidates_for_speaker(session, speaker_id)
    assert view.state is CandidateState.SUPERSEDED
    assert view.candidate.superseded_by_producer_run_id == none_run.id


def test_idempotent_replay(session: Session, speaker_id: uuid.UUID) -> None:
    scope = EnrichmentScope.speaker(speaker_id)
    first = _record(session, scope, (_bio_draft(speaker_id),), key="replay-key")
    replay = _record(session, scope, (_bio_draft(speaker_id),), key="replay-key")
    assert replay.id == first.id
    assert replay.generation == first.generation
    assert len(candidates_for_speaker(session, speaker_id)) == 1

    with pytest.raises(ConflictingReplayError):
        record_producer_run(
            session,
            producer="different_producer",
            producer_version="1.0",
            scope=scope,
            covered_fields=(ClaimField.BIO,),
            candidates=(),
            idempotency_key="replay-key",
            started_at=NOW,
            completed_at=NOW,
        )


def test_replay_with_divergent_body_conflicts(
    session: Session, speaker_id: uuid.UUID
) -> None:
    """The idempotency contract covers the FULL payload: a reused key with
    different candidates, evidence, or config is a conflict, never a silent
    first-write-wins."""
    scope = EnrichmentScope.speaker(speaker_id)
    _record(session, scope, (_bio_draft(speaker_id, "The bio."),), key="body-key")

    def _attempt(
        candidates: tuple[CandidateDraft, ...],
        config: dict[str, int] | None = None,
        config_schema_version: int | None = None,
    ) -> None:
        record_producer_run(
            session,
            producer="name_miner",
            producer_version="1.0",
            scope=scope,
            covered_fields=(ClaimField.BIO, ClaimField.NAME),
            candidates=candidates,
            idempotency_key="body-key",
            started_at=NOW,
            completed_at=NOW,
            config=config,
            config_schema_version=config_schema_version,
        )

    # different claim value
    with pytest.raises(ConflictingReplayError):
        _attempt((_bio_draft(speaker_id, "A different bio."),))
    # different evidence for the same claim
    with pytest.raises(ConflictingReplayError):
        _attempt(
            (
                CandidateDraft(
                    target=scope,
                    field=ClaimField.BIO,
                    value="The bio.",
                    evidence=(UrlEvidence(url="https://example.com/elsewhere"),),
                ),
            )
        )
    # dropped candidates (would have been outcome='none')
    with pytest.raises(ConflictingReplayError):
        _attempt(())
    # different config
    with pytest.raises(ConflictingReplayError):
        _attempt(
            (_bio_draft(speaker_id, "The bio."),),
            config={"budget": 9},
            config_schema_version=1,
        )
    # ... while the identical payload still adopts
    replayed = _record(
        session, scope, (_bio_draft(speaker_id, "The bio."),), key="body-key"
    )
    assert replayed.idempotency_key == "body-key"


def test_decision_beats_concurrent_supersession(
    session_factory: sessionmaker[Session], speaker_id: uuid.UUID
) -> None:
    """READ COMMITTED race: a decision committed while the superseding run
    waits on the candidate's row lock must survive — the waiting statement
    must not stamp the just-decided candidate from its stale snapshot."""
    import threading

    with session_factory() as setup:
        run = record_producer_run(
            setup,
            producer="name_miner",
            producer_version="1.0",
            scope=EnrichmentScope.speaker(speaker_id),
            covered_fields=(ClaimField.BIO,),
            candidates=(_bio_draft(speaker_id, "Contested bio."),),
            idempotency_key="race-gen1",
            started_at=NOW,
            completed_at=NOW,
        )
        setup.commit()
        assert run.generation == 1

    decider = session_factory()
    (view,) = candidates_for_speaker(decider, speaker_id)
    record_profile_decision(
        decider,
        candidate_id=view.candidate.id,
        decision=ProfileDecision.ACCEPT,
        operator="ben",
        idempotency_key="race-decide",
    )
    decider.flush()  # decision row inserted, candidate FOR UPDATE held, no commit

    superseder_error: list[Exception] = []
    started = threading.Event()

    def _supersede() -> None:
        try:
            with session_factory() as other:
                started.set()
                record_producer_run(
                    other,
                    producer="name_miner",
                    producer_version="1.0",
                    scope=EnrichmentScope.speaker(speaker_id),
                    covered_fields=(ClaimField.BIO,),
                    candidates=(),
                    idempotency_key="race-gen2",
                    started_at=NOW,
                    completed_at=NOW,
                )
                other.commit()
        except Exception as exc:  # pragma: no cover - surfaced via assert below
            superseder_error.append(exc)

    thread = threading.Thread(target=_supersede)
    thread.start()
    started.wait(timeout=5)
    # let the superseder reach the candidate row lock, then commit the decision
    import time

    time.sleep(0.5)
    decider.commit()
    thread.join(timeout=15)
    decider.close()
    assert not thread.is_alive(), "superseding run deadlocked"
    assert superseder_error == []

    with session_factory() as check:
        (after,) = candidates_for_speaker(check, speaker_id)
        assert after.state is CandidateState.ACCEPTED
        assert after.candidate.superseded_by_producer_run_id is None


def test_config_snapshot_roundtrip(session: Session, speaker_id: uuid.UUID) -> None:
    run = record_producer_run(
        session,
        producer="web_researcher",
        producer_version="0.1",
        scope=EnrichmentScope.speaker(speaker_id),
        covered_fields=(ClaimField.BIO,),
        candidates=(),
        idempotency_key="cfg-key",
        started_at=NOW,
        completed_at=NOW,
        config={"max_searches": 3, "max_url_reads": 5},
        config_schema_version=1,
    )
    session.commit()
    stored = session.get(EnrichmentProducerRun, run.id)
    assert stored is not None
    assert stored.config == {"max_searches": 3, "max_url_reads": 5}
    assert stored.config_schema_version == 1
