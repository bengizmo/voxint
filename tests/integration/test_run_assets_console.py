"""The run-asset console surface (#41): fragment → generate → cancel → export.

End-to-end over the real app + Postgres: the run detail page's asset block,
the generate-all / per-kind POST (publish monkeypatched — no broker), cancel,
the gates-off message, and the export envelope's additive
``enrichment_assets`` key with staleness.
"""

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from tests.integration.test_run_assets import (
    MENTIONS_BODY,
    SUMMARY_BODY,
    FakeLLM,
    make_settings,
    seed_run,
)
from voxint.api.app import create_app
from voxint.config import Settings
from voxint.db.models import RunAssetJob, RunAssetKind, TranscriptSegment
from voxint.enrichment.asset_jobs import execute_job

CREDS = ("reviewer", "s3cret")
_CSRF_KEY = "run-assets-console-test-csrf-key"


def _build_client(session_factory: sessionmaker[Session], *, gates_open: bool = True) -> TestClient:
    overrides: dict[str, object] = (
        {"llm_enabled": True, "enrichment_run_assets_enabled": True} if gates_open else {}
    )
    settings = Settings(
        _env_file=None,
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        csrf_secret=_CSRF_KEY,
        **overrides,  # type: ignore[arg-type]
    )
    client = TestClient(create_app(settings=settings, session_factory=session_factory))
    client.auth = CREDS
    seed_onboarded(session_factory, llm_enabled=gates_open)
    return client


def _generate(client: TestClient, run_id: uuid.UUID, *, kind: str | None = None):  # type: ignore[no-untyped-def]
    from voxint.api.csrf import CSRF_ASSETS_GENERATE, mint_csrf_token

    data: dict[str, str] = {"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_ASSETS_GENERATE)}
    if kind is not None:
        data["kind"] = kind
    return client.post(f"/runs/{run_id}/assets/generate", data=data)


class TestConsole:
    @pytest.fixture()
    def published(self, monkeypatch: pytest.MonkeyPatch) -> list[uuid.UUID]:
        sink: list[uuid.UUID] = []
        monkeypatch.setattr("voxint.api.app._publish_run_asset_job", sink.append)
        return sink

    def test_run_detail_includes_asset_block(self, session_factory: sessionmaker[Session]) -> None:
        client = _build_client(session_factory)
        with session_factory() as session:
            run_id = seed_run(session)
        page = client.get(f"/runs/{run_id}")
        assert page.status_code == 200
        assert "Run assets" in page.text
        assert "Machine-generated" in page.text

    def test_gates_off_message(self, session_factory: sessionmaker[Session]) -> None:
        client = _build_client(session_factory, gates_open=False)
        with session_factory() as session:
            run_id = seed_run(session)
        fragment = client.get(f"/runs/{run_id}/assets")
        assert fragment.status_code == 200
        # Plain-language remediation pointing at the in-UI Settings toggles, not a
        # raw env var (issue #62).
        assert "ENRICHMENT_RUN_ASSETS_ENABLED" not in fragment.text
        assert "Run assets are off" in fragment.text
        assert "/settings#features" in fragment.text

    def test_generate_all_creates_three_jobs_and_publishes(
        self, session_factory: sessionmaker[Session], published: list[uuid.UUID]
    ) -> None:
        client = _build_client(session_factory)
        with session_factory() as session:
            run_id = seed_run(session)
        resp = _generate(client, run_id)
        assert resp.status_code == 200
        with session_factory() as session:
            jobs = session.execute(select(RunAssetJob)).scalars().all()
            assert sorted(j.asset_kind for j in jobs) == [
                "entity_mentions",
                "summary",
                "topics",
            ]
        assert len(published) == 3
        # A repeat "generate all" while everything is active skips all three.
        resp = _generate(client, run_id)
        assert resp.status_code == 200
        with session_factory() as session:
            assert len(session.execute(select(RunAssetJob)).scalars().all()) == 3
        assert len(published) == 3

    def test_generate_single_kind_and_cancel(
        self, session_factory: sessionmaker[Session], published: list[uuid.UUID]
    ) -> None:
        from voxint.api.csrf import CSRF_ASSETS_CANCEL, mint_csrf_token

        client = _build_client(session_factory)
        with session_factory() as session:
            run_id = seed_run(session)
        resp = _generate(client, run_id, kind="summary")
        assert resp.status_code == 200
        with session_factory() as session:
            job = session.execute(select(RunAssetJob)).scalar_one()
            assert job.asset_kind == "summary"
            job_id = job.id
        resp = client.post(
            f"/runs/{run_id}/assets/{job_id}/cancel",
            data={"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_ASSETS_CANCEL)},
        )
        assert resp.status_code == 200
        with session_factory() as session:
            assert session.get(RunAssetJob, job_id).status == "cancelled"

    def test_generate_unknown_kind_is_422(
        self, session_factory: sessionmaker[Session], published: list[uuid.UUID]
    ) -> None:
        client = _build_client(session_factory)
        with session_factory() as session:
            run_id = seed_run(session)
        resp = _generate(client, run_id, kind="sentiment")
        assert resp.status_code == 422

    def test_generate_without_transcript_reports_inline(
        self, session_factory: sessionmaker[Session], published: list[uuid.UUID]
    ) -> None:
        client = _build_client(session_factory)
        with session_factory() as session:
            run_id = seed_run(session)
            session.query(TranscriptSegment).filter_by(pipeline_run_id=run_id).delete()
            session.commit()
        resp = _generate(client, run_id)
        assert resp.status_code == 200
        assert "no transcript segments" in resp.text
        assert not published

    def test_fragment_renders_finished_asset_with_staleness(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        client = _build_client(session_factory)
        settings = make_settings()
        with session_factory() as session:
            run_id = seed_run(session)
        # Generate mentions inline (the worker path) with a fake LLM.
        with session_factory() as session:
            from voxint.enrichment.asset_jobs import create_jobs

            created, _ = create_jobs(
                session,
                pipeline_run_id=run_id,
                kinds=(RunAssetKind.ENTITY_MENTIONS,),
                settings=settings,
            )
            session.commit()
            job_id = created[0].id
        execute_job(session_factory, job_id, settings=settings, llm=FakeLLM([MENTIONS_BODY]))
        fragment = client.get(f"/runs/{run_id}/assets")
        assert "Acme Corp" in fragment.text
        assert "stale" not in fragment.text
        # Change the source → the badge appears.
        with session_factory() as session:
            segment = session.query(TranscriptSegment).filter_by(pipeline_run_id=run_id).one()
            segment.enhanced_text = "Hello, I am Joanne from Acme Corporation."
            session.commit()
        fragment = client.get(f"/runs/{run_id}/assets")
        assert "stale" in fragment.text


class TestExport:
    def test_envelope_carries_assets_additively(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        client = _build_client(session_factory)
        settings = make_settings()
        with session_factory() as session:
            run_id = seed_run(session)
        body = client.get(f"/runs/{run_id}/export.json").json()
        assert body["schema_version"] == 1  # additive key, no bump
        assert body["enrichment_assets"] is None
        with session_factory() as session:
            from voxint.enrichment.asset_jobs import create_jobs

            created, _ = create_jobs(
                session,
                pipeline_run_id=run_id,
                kinds=(RunAssetKind.SUMMARY,),
                settings=settings,
            )
            session.commit()
            job_id = created[0].id
        execute_job(session_factory, job_id, settings=settings, llm=FakeLLM([SUMMARY_BODY]))
        body = client.get(f"/runs/{run_id}/export.json").json()
        assert body["schema_version"] == 1
        exported = body["enrichment_assets"]["summary"]
        assert exported["payload"] == SUMMARY_BODY
        assert exported["machine_generated"] is True
        assert exported["stale"] is False
        assert exported["generation"] == 1
        assert exported["producer"] == "run_assets.llm"
        assert len(exported["source_content_hash"]) == 64
        # Source change flips staleness in the export too.
        with session_factory() as session:
            segment = session.query(TranscriptSegment).filter_by(pipeline_run_id=run_id).one()
            segment.enhanced_text = "Edited transcript text."
            session.commit()
        body = client.get(f"/runs/{run_id}/export.json").json()
        assert body["enrichment_assets"]["summary"]["stale"] is True


class TestConsoleHardening:
    def test_polling_only_while_active(
        self, session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from voxint.api.csrf import CSRF_ASSETS_CANCEL, mint_csrf_token

        monkeypatch.setattr("voxint.api.app._publish_run_asset_job", lambda _job_id: True)
        client = _build_client(session_factory)
        with session_factory() as session:
            run_id = seed_run(session)
        fragment = client.get(f"/runs/{run_id}/assets")
        assert 'hx-trigger="every 3s"' not in fragment.text  # nothing active
        resp = _generate(client, run_id, kind="summary")
        assert 'hx-trigger="every 3s"' in resp.text  # queued job → poll
        with session_factory() as session:
            job_id = session.execute(select(RunAssetJob)).scalar_one().id
        resp = client.post(
            f"/runs/{run_id}/assets/{job_id}/cancel",
            data={"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_ASSETS_CANCEL)},
        )
        assert 'hx-trigger="every 3s"' not in resp.text  # terminal → poll stops

    def test_broker_outage_is_reported_inline(
        self, session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("voxint.api.app._publish_run_asset_job", lambda _job_id: False)
        client = _build_client(session_factory)
        with session_factory() as session:
            run_id = seed_run(session)
        resp = _generate(client, run_id)
        assert resp.status_code == 200
        assert "broker unavailable" in resp.text
