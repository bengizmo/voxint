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


def test_malformed_base_url_fails_not_stuck_running(
    session_factory: sessionmaker[Session],
) -> None:
    # A malformed env-only LLM_BASE_URL raises httpx.InvalidURL while the worker
    # builds its own client (llm=None). Construction must be inside the failure
    # boundary — but narrowly, around construction ONLY: the research loop also
    # drives httpx (search/read), so a raw HTTP error there must not be relabeled
    # "endpoint misconfigured". The job lands FAILED with the closed-vocabulary
    # init message, never stranded RUNNING.
    speaker_id = seed_speaker(session_factory)
    settings = research_settings(llm_base_url="http://[::1")
    with session_factory() as session:
        job = create_job(session, speaker_id=speaker_id, settings=settings)
        job_id = job.id
        session.commit()
    execute_job(session_factory, job_id, settings=settings, llm=None)
    job = get_job(session_factory, job_id)
    assert job.status == ResearchJobStatus.FAILED.value
    assert job.error == (
        "LLM endpoint could not be initialized"
        " (check the LLM endpoint setting or LLM_BASE_URL)"
    )
    assert job.producer_run_id is None


def test_rerun_supersedes_and_found_false_records_none(
    session_factory: sessionmaker[Session],
) -> None:
    speaker_id = seed_speaker(session_factory)
    run_job(session_factory, speaker_id, [ACTION_SEARCH, ACTION_READ, conclude(CLAIM_OK)])
    # An authoritative "looked, found nothing" needs a real (bounded)
    # investigation — a zero-work conclude would fail the job instead.
    second = run_job(
        session_factory,
        speaker_id,
        [ACTION_SEARCH, conclude(found=False, reason="nothing solid")],
    )
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


def test_cancel_before_delivery_wins_and_nothing_runs(
    session_factory: sessionmaker[Session],
) -> None:
    """Cancel lands between enqueue and delivery: the claim refuses the job
    (terminal CANCELLED), the loop never starts, nothing is persisted."""
    speaker_id = seed_speaker(session_factory)
    settings = research_settings()
    with session_factory() as session:
        job = create_job(session, speaker_id=speaker_id, settings=settings)
        job_id = job.id
        session.commit()
    with session_factory() as session:
        assert request_cancel(session, job_id) is True
        session.commit()
    # The delivery arrives after the cancel: FakeLLM([]) would blow up on any
    # call — the claim must refuse first.
    execute_job(session_factory, job_id, settings=settings, llm=FakeLLM([]))
    job_row = get_job(session_factory, job_id)
    assert job_row.status == ResearchJobStatus.CANCELLED.value
    assert job_row.cancel_requested is True
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
    # A second speaker for the queued-cancel path — the first speaker's
    # RUNNING job holds its one-active-job slot.
    other_speaker = seed_speaker(session_factory, name="Bob Smith")
    with session_factory() as session:
        other = create_job(session, speaker_id=other_speaker, settings=settings)
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


def test_concurrent_start_blocked_by_partial_unique_index(
    session_factory: sessionmaker[Session],
) -> None:
    """The friendly pre-check is check-then-insert; the DB index is the real
    one-active-job-per-speaker invariant."""
    from sqlalchemy.exc import IntegrityError

    speaker_id = seed_speaker(session_factory)
    settings = research_settings()
    with session_factory() as session:
        create_job(session, speaker_id=speaker_id, settings=settings)
        session.commit()
    with (
        session_factory() as session,
        pytest.raises(IntegrityError, match="research_jobs_one_active_per_speaker"),
    ):
        create_job(session, speaker_id=speaker_id, settings=settings)
        session.commit()
    # A terminal job frees the slot.
    with session_factory() as session:
        [job] = session.execute(select(ResearchJob)).scalars()
        job.status = ResearchJobStatus.FAILED.value
        session.commit()
    with session_factory() as session:
        create_job(session, speaker_id=speaker_id, settings=settings)
        session.commit()


def test_stale_running_job_can_be_force_cancelled(
    session_factory: sessionmaker[Session],
) -> None:
    """A worker crash leaves RUNNING; once the row provably outlived its own
    wall-clock budget, cancel terminates it so the speaker is not blocked
    forever."""
    from datetime import UTC, datetime, timedelta

    speaker_id = seed_speaker(session_factory)
    settings = research_settings()
    with session_factory() as session:
        job = create_job(session, speaker_id=speaker_id, settings=settings)
        job_id = job.id
        session.commit()
    with session_factory() as session:
        assert claim_job(session, job_id) is not None
    # Fresh RUNNING: cancel only sets the cooperative flag.
    with session_factory() as session:
        assert request_cancel(session, job_id) is True
        session.commit()
    fresh = get_job(session_factory, job_id)
    assert fresh.status == ResearchJobStatus.RUNNING.value
    assert fresh.cancel_requested is True
    # Backdate far past deadline + llm timeout + grace: now cancel terminates.
    # (created_at moves too — the schema requires started_at >= created_at.)
    with session_factory() as session:
        row = session.get(ResearchJob, job_id)
        assert row is not None
        row.created_at = datetime.now(tz=UTC) - timedelta(hours=3)
        row.started_at = datetime.now(tz=UTC) - timedelta(hours=2)
        session.commit()
    with session_factory() as session:
        assert request_cancel(session, job_id) is True
        session.commit()
    stale = get_job(session_factory, job_id)
    assert stale.status == ResearchJobStatus.CANCELLED.value
    assert stale.finished_at is not None


def test_execution_honors_the_budget_snapshot_not_live_settings(
    session_factory: sessionmaker[Session],
) -> None:
    """The preview the operator approved is the budget that runs — a settings
    change between enqueue and execution never silently applies."""
    speaker_id = seed_speaker(session_factory)
    approved = research_settings(research_max_searches=1, research_max_rounds=2)
    with session_factory() as session:
        job = create_job(session, speaker_id=speaker_id, settings=approved)
        job_id = job.id
        session.commit()
    inflated = research_settings(research_max_searches=9, research_max_rounds=9)
    execute_job(
        session_factory,
        job_id,
        settings=inflated,
        llm=FakeLLM([ACTION_SEARCH, ACTION_READ, conclude(CLAIM_OK)]),
        search_provider=FakeProvider([SEARCH_RESULT]),
        read_client_factory=page_factory(PAGE_TEXT, []),
        read_resolver=resolver_map({"example.com": [PUBLIC_A]}),
    )
    job_row = get_job(session_factory, job_id)
    assert job_row.status == ResearchJobStatus.SUCCEEDED.value
    with session_factory() as session:
        run = session.get(EnrichmentProducerRun, job_row.producer_run_id)
        assert run is not None and run.config is not None
        assert run.config["max_searches"] == 1  # the approved value, not 9
        assert run.config["max_rounds"] == 2


class _CancelOnLastReply:
    """Sets the job's cancel flag while serving the final scripted reply —
    modeling a cancel that lands during the conclude LLM call, after the
    loop's last between-round check."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        job_id: uuid.UUID,
        replies: list[dict[str, object]],
    ) -> None:
        self._inner = FakeLLM(replies)
        self._session_factory = session_factory
        self._job_id = job_id
        self._calls_left = len(replies)

    def chat_json(self, messages: object) -> dict[str, object]:
        self._calls_left -= 1
        if self._calls_left == 0:
            with self._session_factory() as session:
                assert request_cancel(session, self._job_id) is True
                session.commit()
        return self._inner.chat_json(messages)  # type: ignore[arg-type]


def test_cancel_racing_the_conclude_call_wins_and_keeps_no_drafts(
    session_factory: sessionmaker[Session],
) -> None:
    """A cancel set during the final LLM call is invisible to the loop's
    between-round checks; the pre-persist check must catch it — the job ends
    CANCELLED and neither the producer run nor the drafts survive."""
    speaker_id = seed_speaker(session_factory)
    settings = research_settings()
    with session_factory() as session:
        job = create_job(session, speaker_id=speaker_id, settings=settings)
        job_id = job.id
        session.commit()
    execute_job(
        session_factory,
        job_id,
        settings=settings,
        llm=_CancelOnLastReply(
            session_factory, job_id, [ACTION_SEARCH, ACTION_READ, conclude(CLAIM_OK)]
        ),
        search_provider=FakeProvider([SEARCH_RESULT]),
        read_client_factory=page_factory(PAGE_TEXT, []),
        read_resolver=resolver_map({"example.com": [PUBLIC_A]}),
    )
    job_row = get_job(session_factory, job_id)
    assert job_row.status == ResearchJobStatus.CANCELLED.value
    assert producer_runs(session_factory) == []
    with session_factory() as session:
        assert candidates_for_speaker(session, speaker_id) == []


def test_finalization_db_error_lands_failed_not_forever_running(
    session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A DB error while recording the outcome must land as an honest FAILED
    row (closed-vocabulary error), never a forever-RUNNING job."""
    speaker_id = seed_speaker(session_factory)
    settings = research_settings()
    with session_factory() as session:
        job = create_job(session, speaker_id=speaker_id, settings=settings)
        job_id = job.id
        session.commit()

    def boom(*args: object, **kwargs: object) -> object:
        raise RuntimeError("disk full")

    monkeypatch.setattr("voxint.enrichment.research_jobs.record_research_outcome", boom)
    execute_job(
        session_factory,
        job_id,
        settings=settings,
        llm=FakeLLM([ACTION_SEARCH, ACTION_READ, conclude(CLAIM_OK)]),
        search_provider=FakeProvider([SEARCH_RESULT]),
        read_client_factory=page_factory(PAGE_TEXT, []),
        read_resolver=resolver_map({"example.com": [PUBLIC_A]}),
    )
    job_row = get_job(session_factory, job_id)
    assert job_row.status == ResearchJobStatus.FAILED.value
    assert job_row.error == "unexpected error (RuntimeError) — see worker logs"
    assert producer_runs(session_factory) == []


def test_cancel_racing_the_outcome_write_loses_the_stamp_and_rolls_back(
    session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cancel committed AFTER the pre-persist check but before the success
    stamp: the guarded CAS must lose the race, roll the drafts back, and
    resolve the job CANCELLED."""
    import voxint.enrichment.research_jobs as research_jobs_module

    speaker_id = seed_speaker(session_factory)
    settings = research_settings()
    with session_factory() as session:
        job = create_job(session, speaker_id=speaker_id, settings=settings)
        job_id = job.id
        session.commit()
    real_record = research_jobs_module.record_research_outcome

    def record_then_cancel(session: Session, **kwargs: object) -> object:
        run = real_record(session, **kwargs)  # type: ignore[arg-type]
        with session_factory() as other:
            assert request_cancel(other, job_id) is True
            other.commit()
        return run

    monkeypatch.setattr(research_jobs_module, "record_research_outcome", record_then_cancel)
    execute_job(
        session_factory,
        job_id,
        settings=settings,
        llm=FakeLLM([ACTION_SEARCH, ACTION_READ, conclude(CLAIM_OK)]),
        search_provider=FakeProvider([SEARCH_RESULT]),
        read_client_factory=page_factory(PAGE_TEXT, []),
        read_resolver=resolver_map({"example.com": [PUBLIC_A]}),
    )
    job_row = get_job(session_factory, job_id)
    assert job_row.status == ResearchJobStatus.CANCELLED.value
    assert job_row.producer_run_id is None
    assert producer_runs(session_factory) == []
    with session_factory() as session:
        assert candidates_for_speaker(session, speaker_id) == []


def test_stale_cutoff_uses_db_clock_not_app_clock(
    session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: a skewed app clock must not force-cancel a fresh RUNNING
    job — the cutoff compares DB clock to DB clock."""
    from datetime import UTC, datetime, timedelta, tzinfo

    import voxint.enrichment.research_jobs as research_jobs_module

    class _FutureDatetime:
        @staticmethod
        def now(tz: tzinfo | None = None) -> datetime:
            return datetime.now(tz=tz) + timedelta(hours=10)

    speaker_id = seed_speaker(session_factory)
    settings = research_settings()
    with session_factory() as session:
        job = create_job(session, speaker_id=speaker_id, settings=settings)
        job_id = job.id
        session.commit()
    with session_factory() as session:
        assert claim_job(session, job_id) is not None
    monkeypatch.setattr(research_jobs_module, "datetime", _FutureDatetime)
    monkeypatch.setattr(research_jobs_module, "UTC", UTC, raising=False)
    with session_factory() as session:
        assert request_cancel(session, job_id) is True
        session.commit()
    job_row = get_job(session_factory, job_id)
    assert job_row.status == ResearchJobStatus.RUNNING.value  # flag only
    assert job_row.cancel_requested is True


def test_stale_bound_allows_the_forced_conclude_repair_call(
    session_factory: sessionmaker[Session],
) -> None:
    """Regression for the widened stale bound: past deadline + ONE llm timeout
    the loop may still legitimately be alive (the forced conclude gets its own
    single repair attempt), so cancel must only set the cooperative flag."""
    from datetime import UTC, datetime, timedelta

    speaker_id = seed_speaker(session_factory)
    settings = research_settings()  # deadline 300 s, llm timeout 300 s
    with session_factory() as session:
        job = create_job(session, speaker_id=speaker_id, settings=settings)
        job_id = job.id
        session.commit()
    with session_factory() as session:
        assert claim_job(session, job_id) is not None
    # Past the old one-timeout bound (300 + 300 + 60 = 660) but inside the
    # honest two-timeout bound (300 + 600 + 60 = 960).
    with session_factory() as session:
        row = session.get(ResearchJob, job_id)
        assert row is not None
        row.created_at = datetime.now(tz=UTC) - timedelta(seconds=800)
        row.started_at = datetime.now(tz=UTC) - timedelta(seconds=700)
        session.commit()
    with session_factory() as session:
        assert request_cancel(session, job_id) is True
        session.commit()
    job_row = get_job(session_factory, job_id)
    assert job_row.status == ResearchJobStatus.RUNNING.value  # flag only
    assert job_row.cancel_requested is True


def test_worker_client_uses_the_snapshotted_llm_timeout(
    session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    """request_cancel's stale bound reasons from the enqueue-time snapshot, so
    the worker's own client must run under the same timeout — never live
    settings changed since enqueue."""
    import voxint.enrichment.research_jobs as research_jobs_module
    from voxint.clients.llm import LLMError

    recorded: list[float] = []

    class _RecordingClient:
        def __init__(self, base_url: str, model: str, api_key: str, timeout: float) -> None:
            recorded.append(timeout)

        def chat_json(self, messages: object) -> dict[str, object]:
            raise LLMError("connect: injected transport failure")

        def close(self) -> None:
            return None

    monkeypatch.setattr(research_jobs_module, "HttpLLMClient", _RecordingClient)
    speaker_id = seed_speaker(session_factory)
    approved = research_settings(llm_timeout_seconds=123.0)
    with session_factory() as session:
        job = create_job(session, speaker_id=speaker_id, settings=approved)
        job_id = job.id
        session.commit()
    execute_job(
        session_factory,
        job_id,
        settings=research_settings(llm_timeout_seconds=456.0),
        search_provider=FakeProvider([SEARCH_RESULT]),
        read_client_factory=page_factory(PAGE_TEXT, []),
        read_resolver=resolver_map({"example.com": [PUBLIC_A]}),
    )
    assert recorded == [123.0]
    assert get_job(session_factory, job_id).status == ResearchJobStatus.FAILED.value


def test_snapshot_llm_timeout_falls_back_to_the_shared_default() -> None:
    """A pre-0.11 snapshot without the key gets the SAME default the runtime
    uses — the old hard-coded 90.0 could drift from the settings default."""
    from voxint.config import DEFAULT_LLM_TIMEOUT_SECONDS
    from voxint.enrichment.research_jobs import _snapshot_llm_timeout

    assert _snapshot_llm_timeout({}) == DEFAULT_LLM_TIMEOUT_SECONDS
    assert _snapshot_llm_timeout({"llm_timeout_seconds": 42}) == 42.0
    assert _snapshot_llm_timeout({"llm_timeout_seconds": "bad"}) == DEFAULT_LLM_TIMEOUT_SECONDS


class _ForceCancelThenGarbage:
    """Force-cancels the job (backdate → stale-RUNNING cancel) during the LLM
    call, then feeds garbage so the worker reaches its FAILED verdict against
    an already-terminal row."""

    def __init__(self, session_factory: sessionmaker[Session], job_id: uuid.UUID) -> None:
        self._session_factory = session_factory
        self._job_id = job_id
        self._cancelled = False

    def chat_json(self, messages: object) -> dict[str, object]:
        from datetime import UTC, datetime, timedelta

        if not self._cancelled:
            self._cancelled = True
            with self._session_factory() as session:
                row = session.get(ResearchJob, self._job_id)
                assert row is not None
                row.created_at = datetime.now(tz=UTC) - timedelta(hours=3)
                row.started_at = datetime.now(tz=UTC) - timedelta(hours=2)
                session.commit()
            with self._session_factory() as session:
                assert request_cancel(session, self._job_id) is True
                session.commit()
        return {"garbage": 1}


def test_late_worker_failure_does_not_clobber_terminal_cancelled(
    session_factory: sessionmaker[Session],
) -> None:
    """A force-cancel resolved the row mid-run; the worker's late FAILED
    verdict must not overwrite the terminal state."""
    speaker_id = seed_speaker(session_factory)
    settings = research_settings()
    with session_factory() as session:
        job = create_job(session, speaker_id=speaker_id, settings=settings)
        job_id = job.id
        session.commit()
    execute_job(
        session_factory,
        job_id,
        settings=settings,
        llm=_ForceCancelThenGarbage(session_factory, job_id),
        search_provider=FakeProvider([SEARCH_RESULT]),
        read_client_factory=page_factory(PAGE_TEXT, []),
        read_resolver=resolver_map({"example.com": [PUBLIC_A]}),
    )
    job_row = get_job(session_factory, job_id)
    assert job_row.status == ResearchJobStatus.CANCELLED.value
    assert job_row.error is None  # the late FAILED text never landed


class _FlagThenGarbage:
    """Sets the cooperative cancel flag on the first LLM call, then feeds
    garbage — the protocol failure and the pending cancel race inside one
    round, with no between-round check to arbitrate."""

    def __init__(self, session_factory: sessionmaker[Session], job_id: uuid.UUID) -> None:
        self._session_factory = session_factory
        self._job_id = job_id
        self._flagged = False

    def chat_json(self, messages: object) -> dict[str, object]:
        if not self._flagged:
            self._flagged = True
            with self._session_factory() as session:
                assert request_cancel(session, self._job_id) is True
                session.commit()
        return {"garbage": 1}


def test_failed_verdict_racing_operator_cancel_resolves_cancelled(
    session_factory: sessionmaker[Session],
) -> None:
    """The operator asked for CANCELLED; a worker failure arriving with the
    flag already set must resolve to that, not FAILED."""
    speaker_id = seed_speaker(session_factory)
    settings = research_settings()
    with session_factory() as session:
        job = create_job(session, speaker_id=speaker_id, settings=settings)
        job_id = job.id
        session.commit()
    execute_job(
        session_factory,
        job_id,
        settings=settings,
        llm=_FlagThenGarbage(session_factory, job_id),
        search_provider=FakeProvider([SEARCH_RESULT]),
        read_client_factory=page_factory(PAGE_TEXT, []),
        read_resolver=resolver_map({"example.com": [PUBLIC_A]}),
    )
    job_row = get_job(session_factory, job_id)
    assert job_row.status == ResearchJobStatus.CANCELLED.value


# ------------------------------------------------------------- console routes


def _build_client(session_factory: sessionmaker[Session], **overrides: object) -> TestClient:
    settings = research_settings(
        voxint_user=CREDS[0], voxint_password=CREDS[1], csrf_secret=_CSRF_KEY, **overrides
    )
    client = TestClient(create_app(settings=settings, session_factory=session_factory))
    client.auth = CREDS
    # Gates resolve enablement row-over-env (issue #10): seed the onboarded row to
    # match this client's effective env enablement so the gate reflects test intent.
    seed_onboarded(session_factory, llm_enabled=settings.llm_enabled)
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


def test_profile_decision_route_refuses_name_field_candidates(
    session_factory: sessionmaker[Session],
) -> None:
    """The profile surface serves bio/affiliation/link only — a NAME claim,
    even one correctly scoped to this speaker, stays on the workbench flow."""
    from datetime import UTC, datetime

    from voxint.db.models import ClaimField
    from voxint.enrichment.drafts import (
        CandidateDraft,
        EnrichmentScope,
        UrlEvidence,
        record_producer_run,
    )

    client = _build_client(session_factory)
    speaker_id = seed_speaker(session_factory)
    with session_factory() as session:
        scope = EnrichmentScope.speaker(speaker_id)
        run = record_producer_run(
            session,
            producer="test_producer",
            producer_version="1",
            scope=scope,
            covered_fields=(ClaimField.NAME,),
            candidates=(
                CandidateDraft(
                    target=scope,
                    field=ClaimField.NAME,
                    value="Jane Doe",
                    evidence=(UrlEvidence(url="https://example.com/jane"),),
                ),
            ),
            idempotency_key=f"test-name-{speaker_id}",
            started_at=datetime.now(tz=UTC),
            completed_at=datetime.now(tz=UTC),
        )
        session.commit()
        candidate_id = run.candidates[0].id
    response = client.post(
        f"/speakers/{speaker_id}/research/candidates/{candidate_id}/decision",
        data={
            "csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_PROFILE_DECISION),
            "nonce": uuid.uuid4().hex,
            "verdict": "accept",
        },
    )
    assert response.status_code == 404
    with session_factory() as session:
        assert accepted_claims(session, speaker_id) == []


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
