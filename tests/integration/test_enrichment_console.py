"""The name-suggestion console surface (issue #38): trigger → render → decide.

End-to-end over the real app + Postgres: the workbench's "Name hints" block,
the claim-gated synchronous producer trigger, per-label self-intro hints, the
accept/reject profile-review wiring (a review record, never identity), the
Enroll prefill from an accepted suggestion, and the stale/foreign-candidate
refusals.
"""

import uuid
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.api.csrf import CSRF_CLAIM, mint_csrf_token
from voxint.config import Settings
from voxint.db.models import (
    DiarizationTurn,
    EnrichmentCandidate,
    MediaItem,
    MediaSourceMetadata,
    PipelineRun,
    ProfileReviewDecision,
    RunStatus,
    Speaker,
    SpeakerAssignment,
    TranscriptSegment,
)

CREDS = ("reviewer", "s3cret")
_CSRF_KEY = "enrichment-console-test-csrf-key"
NOW = datetime.now(tz=UTC)


def _build_client(
    session_factory: sessionmaker[Session], **settings_overrides: object
) -> TestClient:
    settings = Settings(
        _env_file=None,
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        csrf_secret=_CSRF_KEY,
        **settings_overrides,  # type: ignore[arg-type]
    )
    client = TestClient(create_app(settings=settings, session_factory=session_factory))
    client.auth = CREDS
    seed_onboarded(session_factory)
    return client


@pytest.fixture()
def client(session_factory: sessionmaker[Session]) -> TestClient:
    return _build_client(session_factory)


def seed_run(
    session: Session,
    *,
    title: str | None = "Interview with Jane Doe",
    intro: str | None = "hi my name is bob smith",
) -> uuid.UUID:
    media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
    session.add(media)
    session.flush()
    run = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
    session.add(run)
    session.flush()
    if title is not None:
        session.add(
            MediaSourceMetadata(
                media_item_id=media.id,
                source_kind="ytdlp",
                title=title,
                raw_schema_version=1,
                acquired_at=NOW,
            )
        )
    if intro is not None:
        session.add(
            TranscriptSegment(
                pipeline_run_id=run.id,
                segment_index=0,
                start_seconds=0.0,
                end_seconds=5.0,
                raw_text=intro,
                diarization_label="S0",
            )
        )
        # The workbench's label cards come from diarization turns.
        session.add(
            DiarizationTurn(
                pipeline_run_id=run.id,
                turn_index=0,
                start_seconds=0.0,
                end_seconds=5.0,
                label="S0",
                skip_reason="too_short",
            )
        )
    session.commit()
    return run.id


def claim_token(client: TestClient, run_id: uuid.UUID) -> str:
    resp = client.post(
        f"/review/{run_id}/claim",
        data={"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_CLAIM)},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    return resp.headers["location"].split("token=")[1]


def _enrich(client: TestClient, run_id: uuid.UUID, token: str) -> str:
    resp = client.post(
        f"/review/{run_id}/enrich/names",
        data={"token": token},
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200
    return resp.text


def _candidate_id(
    session_factory: sessionmaker[Session], run_id: uuid.UUID, value: str
) -> uuid.UUID:
    with session_factory() as session:
        return session.execute(
            select(EnrichmentCandidate.id).where(
                EnrichmentCandidate.pipeline_run_id == run_id,
                EnrichmentCandidate.value == value,
                EnrichmentCandidate.superseded_by_producer_run_id.is_(None),
            )
        ).scalar_one()


def test_workbench_offers_generate_before_first_sweep(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        run_id = seed_run(session)
    token = claim_token(client, run_id)
    page = client.get(f"/review/{run_id}?token={token}").text
    assert "Name hints" in page
    assert "No name sweep has run yet" in page
    assert "Generate name suggestions" in page


def test_trigger_renders_run_and_label_suggestions(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        run_id = seed_run(session)
    token = claim_token(client, run_id)
    fragment = _enrich(client, run_id, token)
    assert "Jane Doe" in fragment  # run-level, from the title
    assert "Self-introduced (unverified): “Bob Smith”" in fragment
    assert "Re-run name suggestions" in fragment
    # Suggestions carry accept/reject forms while proposed.
    assert 'name="verdict" value="accept"' in fragment


def test_trigger_on_empty_run_reports_none(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, title=None, intro=None)
    token = claim_token(client, run_id)
    fragment = _enrich(client, run_id, token)
    assert "No name suggestions found" in fragment


def test_trigger_requires_claim(client: TestClient, session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id = seed_run(session)
    resp = client.post(f"/review/{run_id}/enrich/names", data={"token": str(uuid.uuid4())})
    assert resp.status_code == 409


def test_accept_records_review_only_and_prefills_enroll(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        run_id = seed_run(session)
    token = claim_token(client, run_id)
    _enrich(client, run_id, token)
    candidate_id = _candidate_id(session_factory, run_id, "Bob Smith")

    resp = client.post(
        f"/review/{run_id}/candidates/{candidate_id}/decision",
        data={"token": token, "nonce": uuid.uuid4().hex, "verdict": "accept"},
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200
    assert "accepted" in resp.text
    # The accepted per-label suggestion prefills the Enroll input.
    assert 'value="Bob Smith"' in resp.text

    with session_factory() as session:
        decision = session.execute(select(ProfileReviewDecision)).scalar_one()
        assert decision.candidate_id == candidate_id
        assert decision.decision == "accept"
        # A review record, never identity: nothing enters the roster or ledger.
        assert session.execute(select(Speaker)).scalar_one_or_none() is None
        assert session.execute(select(SpeakerAssignment)).scalar_one_or_none() is None


def test_reject_renders_rejected_pill(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, intro=None)
    token = claim_token(client, run_id)
    _enrich(client, run_id, token)
    candidate_id = _candidate_id(session_factory, run_id, "Jane Doe")
    resp = client.post(
        f"/review/{run_id}/candidates/{candidate_id}/decision",
        data={"token": token, "nonce": uuid.uuid4().hex, "verdict": "reject"},
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200
    assert "rejected" in resp.text


def test_decide_superseded_candidate_conflicts(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, intro=None)
    token = claim_token(client, run_id)
    _enrich(client, run_id, token)
    stale_id = _candidate_id(session_factory, run_id, "Jane Doe")
    # New input → new generation supersedes the first sweep's proposal.
    with session_factory() as session:
        session.add(
            TranscriptSegment(
                pipeline_run_id=run_id,
                segment_index=0,
                start_seconds=0.0,
                end_seconds=5.0,
                raw_text="my name is bob smith",
                diarization_label="S0",
            )
        )
        session.commit()
    _enrich(client, run_id, token)
    resp = client.post(
        f"/review/{run_id}/candidates/{stale_id}/decision",
        data={"token": token, "nonce": uuid.uuid4().hex, "verdict": "accept"},
    )
    assert resp.status_code == 409
    assert "superseded" in resp.json()["detail"]


def test_foreign_candidate_is_404(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        run_a = seed_run(session)
        run_b = seed_run(session, title="Interview with Maria Lopez", intro=None)
    token_a = claim_token(client, run_a)
    token_b = claim_token(client, run_b)
    _enrich(client, run_b, token_b)
    foreign = _candidate_id(session_factory, run_b, "Maria Lopez")
    resp = client.post(
        f"/review/{run_a}/candidates/{foreign}/decision",
        data={"token": token_a, "nonce": uuid.uuid4().hex, "verdict": "accept"},
    )
    assert resp.status_code == 404


def test_disabled_flag_hides_surface_and_blocks_trigger(
    session_factory: sessionmaker[Session],
) -> None:
    client = _build_client(session_factory, enrichment_names_enabled=False)
    with session_factory() as session:
        run_id = seed_run(session)
    token = claim_token(client, run_id)
    page = client.get(f"/review/{run_id}?token={token}").text
    assert "Name hints" not in page
    resp = client.post(f"/review/{run_id}/enrich/names", data={"token": token})
    assert resp.status_code == 404


def test_non_name_candidate_is_not_decidable_here(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    # The route serves the NAME surface only: a BIO claim on the same run —
    # never rendered here — must 404 even with a valid claim token.
    from datetime import timedelta

    from voxint.db.models import ClaimField
    from voxint.enrichment.drafts import (
        CandidateDraft,
        EnrichmentScope,
        UrlEvidence,
        record_producer_run,
    )

    with session_factory() as session:
        run_id = seed_run(session)
        record_producer_run(
            session,
            producer="test.bio",
            producer_version="1",
            scope=EnrichmentScope.run(run_id),
            covered_fields=(ClaimField.BIO,),
            candidates=(
                CandidateDraft(
                    target=EnrichmentScope.run(run_id),
                    field=ClaimField.BIO,
                    value="A bio claim.",
                    evidence=(UrlEvidence(url="https://example.com/about"),),
                ),
            ),
            idempotency_key=f"bio-{run_id}",
            started_at=NOW,
            completed_at=NOW + timedelta(seconds=1),
        )
        session.commit()
    bio_id = _candidate_id(session_factory, run_id, "A bio claim.")
    token = claim_token(client, run_id)
    resp = client.post(
        f"/review/{run_id}/candidates/{bio_id}/decision",
        data={"token": token, "nonce": uuid.uuid4().hex, "verdict": "accept"},
    )
    assert resp.status_code == 404


def test_disabled_flag_blocks_decisions_too(
    session_factory: sessionmaker[Session],
) -> None:
    # Seed candidates while enabled, then flip the flag off: the decision
    # route disappears along with the surface.
    enabled = _build_client(session_factory)
    with session_factory() as session:
        run_id = seed_run(session)
    token = claim_token(enabled, run_id)
    _enrich(enabled, run_id, token)
    candidate_id = _candidate_id(session_factory, run_id, "Bob Smith")

    disabled = _build_client(session_factory, enrichment_names_enabled=False)
    token2 = claim_token(disabled, run_id)
    resp = disabled.post(
        f"/review/{run_id}/candidates/{candidate_id}/decision",
        data={"token": token2, "nonce": uuid.uuid4().hex, "verdict": "accept"},
    )
    assert resp.status_code == 404
