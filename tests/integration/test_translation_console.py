"""The translation console surface (#133): fragment → generate → cancel.

End-to-end over the real app + Postgres: the run detail page's translation
card, the generate POST (publish monkeypatched — no broker) with the
preferred-language default and a per-run override, same-language and
no-target refusals as inline card errors, cancel, the gates-off message,
polling only while a job is active, and the stale banner after an edit.
"""

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from tests.integration.test_translation_jobs import FakeLLM, make_settings, seed_run
from voxint.adjudication.review_state import set_correction
from voxint.api.app import create_app
from voxint.api.csrf import (
    CSRF_TRANSLATION_CANCEL,
    CSRF_TRANSLATION_GENERATE,
    mint_csrf_token,
)
from voxint.config import Settings
from voxint.db.models import TranscriptSegment, TranslationJob, TranslationJobStatus
from voxint.enrichment.translation_jobs import execute_job

CREDS = ("reviewer", "s3cret")
_CSRF_KEY = "translation-console-test-csrf-key"


def _build_client(
    session_factory: sessionmaker[Session],
    *,
    gates_open: bool = True,
    **extra: object,
) -> TestClient:
    overrides: dict[str, object] = {"llm_enabled": True} if gates_open else {}
    overrides.update(extra)
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


def _generate(client: TestClient, run_id: uuid.UUID, *, target: str | None = None):  # type: ignore[no-untyped-def]
    data: dict[str, str] = {
        "csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_TRANSLATION_GENERATE)
    }
    if target is not None:
        data["target_language"] = target
    return client.post(f"/runs/{run_id}/translation/generate", data=data)


class TestConsole:
    @pytest.fixture()
    def published(self, monkeypatch: pytest.MonkeyPatch) -> list[uuid.UUID]:
        sink: list[uuid.UUID] = []

        def _sink(job_id: uuid.UUID) -> bool:
            sink.append(job_id)
            return True

        monkeypatch.setattr("voxint.api.app._publish_translation_job", _sink)
        return sink

    def test_run_detail_includes_translation_card(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
        client = _build_client(session_factory)
        body = client.get(f"/runs/{run_id}").text
        assert f"run-translation-{run_id}" in body
        assert "Translate to" in body

    def test_gates_off_message(self, session_factory: sessionmaker[Session]) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
        client = _build_client(session_factory, gates_open=False)
        body = client.get(f"/runs/{run_id}/translation").text
        assert "Translation is off" in body

    def test_generate_with_explicit_target_creates_job(
        self, session_factory: sessionmaker[Session], published: list[uuid.UUID]
    ) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
        client = _build_client(session_factory)
        response = _generate(client, run_id, target="es")
        assert response.status_code == 200
        with session_factory() as session:
            job = session.execute(select(TranslationJob)).scalar_one()
            assert job.target_language == "es"
            assert job.status == TranslationJobStatus.QUEUED.value
        assert len(published) == 1
        # A duplicate while active is an inline card error, not a second job.
        again = _generate(client, run_id, target="es")
        assert again.status_code == 200
        assert "already in progress" in again.text
        assert len(published) == 1
        # A DIFFERENT language while one is active is refused the same way —
        # the card is a single-job surface with one cancel control.
        other = _generate(client, run_id, target="fr")
        assert other.status_code == 200
        assert "already in progress" in other.text
        assert len(published) == 1

    def test_generate_defaults_to_preferred_language(
        self, session_factory: sessionmaker[Session], published: list[uuid.UUID]
    ) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
        client = _build_client(session_factory, translation_target_language="fr")
        response = _generate(client, run_id)
        assert response.status_code == 200
        with session_factory() as session:
            job = session.execute(select(TranslationJob)).scalar_one()
            assert job.target_language == "fr"
        assert len(published) == 1

    def test_preferred_matching_detected_falls_back_to_explicit_choice(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        # The detected language is dropped from the select's options, so a
        # preferred language equal to it cannot be the default: the browser
        # would silently pick the first remaining option. The card must show
        # the explicit-choice placeholder instead.
        with session_factory() as session:
            run_id = seed_run(session, detected_language="es")
        client = _build_client(session_factory, translation_target_language="es")
        body = client.get(f"/runs/{run_id}/translation").text
        assert "Choose a language" in body
        assert body.count(" selected") == 1  # only the placeholder

    def test_generate_without_any_target_reports_inline(
        self, session_factory: sessionmaker[Session], published: list[uuid.UUID]
    ) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
        client = _build_client(session_factory)
        response = _generate(client, run_id)
        assert response.status_code == 200
        assert "Pick a language" in response.text
        assert not published

    def test_generate_same_language_reports_inline(
        self, session_factory: sessionmaker[Session], published: list[uuid.UUID]
    ) -> None:
        with session_factory() as session:
            run_id = seed_run(session, detected_language="es")
        client = _build_client(session_factory)
        response = _generate(client, run_id, target="es")
        assert response.status_code == 200
        assert "already in" in response.text
        assert not published

    def test_generate_requires_csrf(
        self, session_factory: sessionmaker[Session], published: list[uuid.UUID]
    ) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
        client = _build_client(session_factory)
        response = client.post(
            f"/runs/{run_id}/translation/generate", data={"target_language": "es"}
        )
        assert response.status_code == 403
        assert not published

    def test_cancel_resolves_queued_job(
        self, session_factory: sessionmaker[Session], published: list[uuid.UUID]
    ) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
        client = _build_client(session_factory)
        _generate(client, run_id, target="es")
        with session_factory() as session:
            job_id = session.execute(select(TranslationJob.id)).scalar_one()
        response = client.post(
            f"/runs/{run_id}/translation/{job_id}/cancel",
            data={"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_TRANSLATION_CANCEL)},
        )
        assert response.status_code == 200
        with session_factory() as session:
            job = session.get(TranslationJob, job_id)
            assert job.status == TranslationJobStatus.CANCELLED.value

    def test_polling_only_while_active_and_stale_banner_after_edit(
        self, session_factory: sessionmaker[Session], published: list[uuid.UUID]
    ) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
        client = _build_client(session_factory)
        _generate(client, run_id, target="es")
        active = client.get(f"/runs/{run_id}/translation").text
        assert "hx-trigger" in active  # polls while the job is queued/running
        with session_factory() as session:
            job_id = session.execute(select(TranslationJob.id)).scalar_one()
        execute_job(session_factory, job_id, settings=make_settings(), llm=FakeLLM())
        done = client.get(f"/runs/{run_id}/translation").text
        assert "hx-trigger" not in done  # terminal render stops polling
        assert "Spanish (es)" in done
        assert "out of date" not in done
        # An operator edit stales the generation: banner, no partial alignment.
        with session_factory() as session:
            segment = session.execute(
                select(TranscriptSegment).where(
                    TranscriptSegment.pipeline_run_id == run_id,
                    TranscriptSegment.segment_index == 0,
                )
            ).scalar_one()
            set_correction(session, segment=segment, text="Edited afterwards.")
            session.commit()
        stale = client.get(f"/runs/{run_id}/translation").text
        assert "out of date" in stale
        assert "Re-translate" in stale
