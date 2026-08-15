"""Web-research jobs end-to-end over real Postgres (issue #40): job lifecycle,
draft persistence, supersession, honest failure — plus the console routes.

The loop's own protocol/grounding behavior is covered in
``tests/unit/test_research_agent.py``; here the fakes drive the REAL producer,
#37 writer, and job service against the migrated schema.
"""

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from tests.unit.test_research_agent import (
    ACTION_READ,
    ACTION_SEARCH,
    CLAIM_OK,
    PAGE_TEXT,
    PUBLIC_A,
    SEARCH_RESULT,
    FakeLLM,
    FakeProvider,
    conclude,
    page_factory,
    resolver_map,
)
from voxint.api.app import create_app
from voxint.api.csrf import (
    CSRF_PROFILE_DECISION,
    CSRF_RESEARCH_CANCEL,
    CSRF_RESEARCH_START,
    mint_csrf_token,
)
from voxint.config import Settings
from voxint.db.models import (
    EnrichmentCandidate,
    EnrichmentProducerRun,
    ResearchJob,
    ResearchJobStatus,
    Speaker,
)
from voxint.enrichment.queries import CandidateState, accepted_claims, candidates_for_speaker
from voxint.enrichment.research_jobs import (
    ResearchJobError,
    claim_job,
    create_job,
    execute_job,
    request_cancel,
)

CREDS = ("reviewer", "s3cret")
_CSRF_KEY = "web-researcher-test-csrf-key"

GATES: dict[str, object] = {
    "voxint_web_research": True,
    "web_search_base_url": "http://searx.lan:8888",
    "llm_enabled": True,
    "enrichment_web_research_enabled": True,
}


def research_settings(**overrides: object) -> Settings:
    merged = {**GATES, **overrides}
    return Settings(_env_file=None, **merged)  # type: ignore[arg-type]


def seed_speaker(session_factory: sessionmaker[Session], name: str = "Jane Doe") -> uuid.UUID:
    with session_factory() as session:
        speaker = Speaker(display_name=name)
        session.add(speaker)
        session.commit()
        return speaker.id


def run_job(
    session_factory: sessionmaker[Session],
    speaker_id: uuid.UUID,
    replies: list[dict[str, object]],
    *,
    settings: Settings | None = None,
    llm: FakeLLM | None = None,
) -> uuid.UUID:
    settings = settings or research_settings()
    with session_factory() as session:
        job = create_job(session, speaker_id=speaker_id, settings=settings)
        job_id = job.id
        session.commit()
    execute_job(
        session_factory,
        job_id,
        settings=settings,
        llm=llm or FakeLLM(replies),
        search_provider=FakeProvider([SEARCH_RESULT]),
        read_client_factory=page_factory(PAGE_TEXT, []),
        read_resolver=resolver_map({"example.com": [PUBLIC_A]}),
    )
    return job_id


def get_job(session_factory: sessionmaker[Session], job_id: uuid.UUID) -> ResearchJob:
    with session_factory() as session:
        job = session.get(ResearchJob, job_id)
        assert job is not None
        return job


def producer_runs(session_factory: sessionmaker[Session]) -> list[EnrichmentProducerRun]:
    with session_factory() as session:
        return list(session.execute(select(EnrichmentProducerRun)).scalars())


def test_job_records_grounded_drafts_and_succeeds(
    session_factory: sessionmaker[Session],
) -> None:
    speaker_id = seed_speaker(session_factory)
    job_id = run_job(session_factory, speaker_id, [ACTION_SEARCH, ACTION_READ, conclude(CLAIM_OK)])
    job = get_job(session_factory, job_id)
    assert job.status == ResearchJobStatus.SUCCEEDED.value
    assert job.producer_run_id is not None
    assert (job.searches_used, job.reads_used, job.rounds_used) == (1, 1, 2)
    with session_factory() as session:
        [view] = candidates_for_speaker(session, speaker_id)
        assert view.state is CandidateState.PROPOSED
        assert view.candidate.field == "affiliation"
        assert view.candidate.speaker_id == speaker_id
        [evidence] = view.evidence
        assert evidence.kind == "url"
        assert evidence.url == "https://example.com/jane"
        assert evidence.snippet and evidence.snippet in PAGE_TEXT
        run = session.get(EnrichmentProducerRun, job.producer_run_id)
        assert run is not None and run.outcome == "found"
        assert run.idempotency_key == f"web_researcher:speaker:{speaker_id}:{job_id}"
        assert run.config is not None and run.config["job_id"] == str(job_id)


def test_rerun_supersedes_and_found_false_records_none(
    session_factory: sessionmaker[Session],
) -> None:
    speaker_id = seed_speaker(session_factory)
    run_job(session_factory, speaker_id, [ACTION_SEARCH, ACTION_READ, conclude(CLAIM_OK)])
    second = run_job(session_factory, speaker_id, [conclude(found=False, reason="nothing solid")])
    job = get_job(session_factory, second)
    assert job.status == ResearchJobStatus.SUCCEEDED.value
    with session_factory() as session:
        [view] = candidates_for_speaker(session, speaker_id)
        assert view.state is CandidateState.SUPERSEDED
        runs = sorted(producer_runs(session_factory), key=lambda r: r.generation)
        assert [r.outcome for r in runs] == ["found", "none"]
        assert runs[1].config is not None and runs[1].config["found"] is False


def test_llm_failure_fails_job_and_records_no_producer_run(
    session_factory: sessionmaker[Session],
) -> None:
    speaker_id = seed_speaker(session_factory)
    # Two malformed replies: one repair attempt, then the loop must fail.
    job_id = run_job(session_factory, speaker_id, [{"garbage": 1}, {"garbage": 2}])
    job = get_job(session_factory, job_id)
    assert job.status == ResearchJobStatus.FAILED.value
    assert job.error and "protocol" in job.error
    assert producer_runs(session_factory) == []


def test_cancel_requested_stops_before_first_round(
    session_factory: sessionmaker[Session],
) -> None:
    speaker_id = seed_speaker(session_factory)
    settings = research_settings()
    with session_factory() as session:
        job = create_job(session, speaker_id=speaker_id, settings=settings)
        job.cancel_requested = True
        job_id = job.id
        session.commit()
    execute_job(session_factory, job_id, settings=settings, llm=FakeLLM([]))
    job_row = get_job(session_factory, job_id)
    assert job_row.status == ResearchJobStatus.CANCELLED.value
    assert producer_runs(session_factory) == []


def test_duplicate_delivery_noops_and_cancel_of_queued_is_terminal(
    session_factory: sessionmaker[Session],
) -> None:
    speaker_id = seed_speaker(session_factory)
    settings = research_settings()
    with session_factory() as session:
        job = create_job(session, speaker_id=speaker_id, settings=settings)
        job_id = job.id
        session.commit()
    with session_factory() as session:
        assert claim_job(session, job_id) is not None
    with session_factory() as session:
        assert claim_job(session, job_id) is None  # duplicate delivery
    with session_factory() as session:
        other = create_job(session, speaker_id=speaker_id, settings=settings)
        other_id = other.id
        session.commit()
    with session_factory() as session:
        assert request_cancel(session, other_id) is True
        session.commit()
    assert get_job(session_factory, other_id).status == ResearchJobStatus.CANCELLED.value
    # execute_job on the cancelled-when-queued job must no-op via the claim.
    execute_job(session_factory, other_id, settings=settings, llm=FakeLLM([]))
    assert get_job(session_factory, other_id).status == ResearchJobStatus.CANCELLED.value


def test_worker_rechecks_gates(session_factory: sessionmaker[Session]) -> None:
    speaker_id = seed_speaker(session_factory)
    with session_factory() as session:
        job = create_job(session, speaker_id=speaker_id, settings=research_settings())
        job_id = job.id
        session.commit()
    # The capability was shut off between enqueue and execution.
    execute_job(
        session_factory,
        job_id,
        settings=Settings(_env_file=None),
        llm=FakeLLM([]),
    )
    job_row = get_job(session_factory, job_id)
    assert job_row.status == ResearchJobStatus.FAILED.value
    assert job_row.error and "disabled" in job_row.error


def test_create_job_refuses_gates_off_and_bad_targets(
    session_factory: sessionmaker[Session],
) -> None:
    speaker_id = seed_speaker(session_factory)
    with session_factory() as session:
        with pytest.raises(ResearchJobError, match="disabled"):
            create_job(session, speaker_id=speaker_id, settings=Settings(_env_file=None))
        with pytest.raises(ResearchJobError, match="not found"):
            create_job(session, speaker_id=uuid.uuid4(), settings=research_settings())


# ------------------------------------------------------------- console routes


def _build_client(session_factory: sessionmaker[Session], **overrides: object) -> TestClient:
    settings = research_settings(
        voxint_user=CREDS[0], voxint_password=CREDS[1], csrf_secret=_CSRF_KEY, **overrides
    )
    client = TestClient(create_app(settings=settings, session_factory=session_factory))
    client.auth = CREDS
    seed_onboarded(session_factory)
    return client


def test_start_route_creates_job_and_publishes(
    session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    published: list[uuid.UUID] = []
    monkeypatch.setattr("voxint.api.app._publish_research_job", published.append)
    client = _build_client(session_factory)
    speaker_id = seed_speaker(session_factory)
    response = client.post(
        f"/speakers/{speaker_id}/research/start",
        data={
            "csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_RESEARCH_START),
            "operator_note": "likely the podcast host",
        },
    )
    assert response.status_code == 200
    with session_factory() as session:
        [job] = session.execute(select(ResearchJob)).scalars()
        assert job.speaker_id == speaker_id
        assert job.status == ResearchJobStatus.QUEUED.value
        assert job.operator_note == "likely the podcast host"
        assert job.budget["max_searches"] == 3
        assert published == [job.id]
    # A second start while one is active is refused with an inline error.
    again = client.post(
        f"/speakers/{speaker_id}/research/start",
        data={"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_RESEARCH_START)},
    )
    assert again.status_code == 200
    assert "already active" in again.text
    with session_factory() as session:
        assert len(list(session.execute(select(ResearchJob)).scalars())) == 1


def test_cancel_route_sets_flag(
    session_factory: sessionmaker[Session],
) -> None:
    client = _build_client(session_factory)
    speaker_id = seed_speaker(session_factory)
    with session_factory() as session:
        job = create_job(session, speaker_id=speaker_id, settings=research_settings())
        job_id = job.id
        session.commit()
    response = client.post(
        f"/speakers/{speaker_id}/research/{job_id}/cancel",
        data={"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_RESEARCH_CANCEL)},
    )
    assert response.status_code == 200
    job_row = get_job(session_factory, job_id)
    assert job_row.cancel_requested is True
    assert job_row.status == ResearchJobStatus.CANCELLED.value  # was still queued


def test_profile_decision_route_accepts_field_by_field(
    session_factory: sessionmaker[Session],
) -> None:
    client = _build_client(session_factory)
    speaker_id = seed_speaker(session_factory)
    run_job(session_factory, speaker_id, [ACTION_SEARCH, ACTION_READ, conclude(CLAIM_OK)])
    with session_factory() as session:
        [view] = candidates_for_speaker(session, speaker_id)
        candidate_id = view.candidate.id
    response = client.post(
        f"/speakers/{speaker_id}/research/candidates/{candidate_id}/decision",
        data={
            "csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_PROFILE_DECISION),
            "nonce": uuid.uuid4().hex,
            "verdict": "accept",
        },
    )
    assert response.status_code == 200
    with session_factory() as session:
        [accepted] = accepted_claims(session, speaker_id)
        assert accepted.candidate.id == candidate_id


def test_profile_decision_route_refuses_foreign_and_name_candidates(
    session_factory: sessionmaker[Session],
) -> None:
    client = _build_client(session_factory)
    speaker_id = seed_speaker(session_factory)
    other_id = seed_speaker(session_factory, name="Someone Else")
    run_job(session_factory, speaker_id, [ACTION_SEARCH, ACTION_READ, conclude(CLAIM_OK)])
    with session_factory() as session:
        [view] = candidates_for_speaker(session, speaker_id)
        candidate_id = view.candidate.id
    response = client.post(
        f"/speakers/{other_id}/research/candidates/{candidate_id}/decision",
        data={
            "csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_PROFILE_DECISION),
            "nonce": uuid.uuid4().hex,
            "verdict": "accept",
        },
    )
    assert response.status_code == 404
    with session_factory() as session:
        assert session.execute(select(EnrichmentCandidate)).scalars().one().id == candidate_id


def test_research_fragment_disabled_copy_when_gates_off(
    session_factory: sessionmaker[Session],
) -> None:
    client = _build_client(
        session_factory,
        enrichment_web_research_enabled=False,
        voxint_web_research=False,
        llm_enabled=False,
        web_search_base_url="",
    )
    speaker_id = seed_speaker(session_factory)
    response = client.get(f"/speakers/{speaker_id}/research")
    assert response.status_code == 200
    assert "Web research is off" in response.text
