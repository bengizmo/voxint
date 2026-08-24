"""The translation read surfaces (#133 Slice B): interleaved transcript view,
fail-closed ``?lang=`` exports, the export menu's fresh-only links, and the
review stepper's terminal Translate props + JSON generate contract.

End-to-end over the real app + Postgres. Generations are produced by the real
executor with the fake LLM (prefixing ``ES:``), so freshness/staleness comes
from the genuine source-hash pipeline, never fixtures pretending at it.
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
from voxint.api.csrf import CSRF_TRANSLATION_GENERATE, mint_csrf_token
from voxint.config import Settings
from voxint.db.models import TranscriptSegment, TranslationJob
from voxint.enrichment.translation_jobs import create_job, execute_job

CREDS = ("reviewer", "s3cret")
_CSRF_KEY = "translation-view-test-csrf-key"

EXPORT_FORMATS = ("txt", "md", "srt", "vtt", "json")


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


def _translate(
    session_factory: sessionmaker[Session], run_id: uuid.UUID, target: str = "es"
) -> None:
    """Produce one real fresh generation via the executor + fake LLM."""
    with session_factory() as session:
        job, already = create_job(
            session,
            pipeline_run_id=run_id,
            target_language=target,
            settings=make_settings(),
        )
        assert job is not None and not already
        job_id = job.id
        session.commit()
    execute_job(session_factory, job_id, settings=make_settings(), llm=FakeLLM())


def _stale_edit(session_factory: sessionmaker[Session], run_id: uuid.UUID) -> None:
    with session_factory() as session:
        segment = session.execute(
            select(TranscriptSegment).where(
                TranscriptSegment.pipeline_run_id == run_id,
                TranscriptSegment.segment_index == 0,
            )
        ).scalar_one()
        set_correction(session, segment=segment, text="Edited afterwards.")
        session.commit()


class TestTranscriptView:
    def test_fresh_translation_interleaves(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
        _translate(session_factory, run_id)
        client = _build_client(session_factory)
        # The toggle offers the fresh generation on the plain page.
        plain = client.get(f"/runs/{run_id}/transcript").text
        assert "translation=es" in plain
        # Selecting it interleaves the translated lines in the fallback AND
        # hands them to the island props, with honest provenance copy.
        body = client.get(f"/runs/{run_id}/transcript?translation=es").text
        assert 'class="tp-translation"' in body
        assert "ES:Hello there." in body
        assert "ES:Goodbye now." in body
        assert "Machine-translated to Spanish" in body
        assert '"translation"' in body or "&#34;translation&#34;" in body

    def test_raw_variant_shows_note_not_lines(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
        _translate(session_factory, run_id)
        client = _build_client(session_factory)
        body = client.get(f"/runs/{run_id}/transcript?text=raw&translation=es").text
        assert "pair with the reviewed text" in body
        assert "ES:Hello there." not in body

    def test_toggle_absent_on_raw_and_enhanced_variants(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        # Translations pair with the reviewed text: raw/enhanced views must not
        # advertise a switcher whose links silently change the rendition.
        with session_factory() as session:
            run_id = seed_run(session)
        _translate(session_factory, run_id)
        client = _build_client(session_factory)
        assert (
            'aria-label="Translation"'
            in client.get(f"/runs/{run_id}/transcript").text
        )
        for text in ("raw", "enhanced"):
            body = client.get(f"/runs/{run_id}/transcript?text={text}").text
            assert 'aria-label="Translation"' not in body, text

    def test_stale_translation_never_interleaves(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
        _translate(session_factory, run_id)
        _stale_edit(session_factory, run_id)
        client = _build_client(session_factory)
        plain = client.get(f"/runs/{run_id}/transcript").text
        assert "out of date" in plain
        body = client.get(f"/runs/{run_id}/transcript?translation=es").text
        assert "out of date" in body
        assert "ES:Hello there." not in body
        assert 'class="tp-translation"' not in body

    def test_no_translations_renders_no_toggle(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
        client = _build_client(session_factory)
        body = client.get(f"/runs/{run_id}/transcript").text
        assert 'aria-label="Translation"' not in body
        assert 'class="tp-translation"' not in body

    def test_unknown_translation_shows_note(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
        client = _build_client(session_factory)
        body = client.get(f"/runs/{run_id}/transcript?translation=fr").text
        assert "No such translation" in body

    def test_export_menu_lists_fresh_only(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
        _translate(session_factory, run_id)
        client = _build_client(session_factory)
        fresh = client.get(f"/runs/{run_id}/transcript").text
        assert "export.srt?lang=es" in fresh
        _stale_edit(session_factory, run_id)
        stale = client.get(f"/runs/{run_id}/transcript").text
        assert "export.srt?lang=es" not in stale


class TestExports:
    def test_fresh_translation_substitutes_in_every_format(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
        _translate(session_factory, run_id)
        client = _build_client(session_factory)
        for fmt in EXPORT_FORMATS:
            response = client.get(f"/review/{run_id}/export.{fmt}?lang=es")
            assert response.status_code == 200, fmt
            assert "ES:Hello there." in response.text, fmt
            # Never mixed-language: the original wording is fully replaced.
            assert "] Hello there." not in response.text, fmt

    def test_missing_translation_is_409(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
        client = _build_client(session_factory)
        for fmt in EXPORT_FORMATS:
            response = client.get(f"/review/{run_id}/export.{fmt}?lang=es")
            assert response.status_code == 409, fmt
            assert "no Spanish (es) translation" in response.json()["detail"]

    def test_stale_translation_is_409(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
        _translate(session_factory, run_id)
        _stale_edit(session_factory, run_id)
        client = _build_client(session_factory)
        for fmt in EXPORT_FORMATS:
            response = client.get(f"/review/{run_id}/export.{fmt}?lang=es")
            assert response.status_code == 409, fmt
            assert "out of date" in response.json()["detail"]

    def test_running_translation_is_409_with_honest_detail(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
            job, _ = create_job(
                session,
                pipeline_run_id=run_id,
                target_language="es",
                settings=make_settings(),
            )
            assert job is not None
            session.commit()
        client = _build_client(session_factory)
        response = client.get(f"/review/{run_id}/export.txt?lang=es")
        assert response.status_code == 409
        assert "still being generated" in response.json()["detail"]

    def test_unreadable_source_is_409(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        # A generation whose source transcript is gone (segments deleted after
        # the fact) must fail closed, never serve the orphaned rendition.
        with session_factory() as session:
            run_id = seed_run(session)
        _translate(session_factory, run_id)
        with session_factory() as session:
            for segment in session.execute(
                select(TranscriptSegment).where(
                    TranscriptSegment.pipeline_run_id == run_id
                )
            ).scalars():
                session.delete(segment)
            session.commit()
        client = _build_client(session_factory)
        response = client.get(f"/review/{run_id}/export.txt?lang=es")
        assert response.status_code == 409

    def test_unreadable_source_marks_run_card_out_of_date(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        # The run card must fail closed like the view and exports: with the
        # source transcript gone, the generation renders "out of date", never a
        # "view it" promise the transcript page would refuse.
        with session_factory() as session:
            run_id = seed_run(session)
        _translate(session_factory, run_id)
        with session_factory() as session:
            for segment in session.execute(
                select(TranscriptSegment).where(
                    TranscriptSegment.pipeline_run_id == run_id
                )
            ).scalars():
                session.delete(segment)
            session.commit()
        client = _build_client(session_factory)
        body = client.get(f"/runs/{run_id}/translation").text
        assert "out of date" in body
        assert "View it beneath each line" not in body

    def test_lang_with_raw_or_enhanced_is_422(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
        _translate(session_factory, run_id)
        client = _build_client(session_factory)
        for text in ("raw", "enhanced"):
            response = client.get(f"/review/{run_id}/export.txt?lang=es&text={text}")
            assert response.status_code == 422, text
        # Explicit corrected is the default and stays allowed.
        ok = client.get(f"/review/{run_id}/export.txt?lang=es&text=corrected")
        assert ok.status_code == 200

    def test_unknown_lang_code_is_422(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
        client = _build_client(session_factory)
        response = client.get(f"/review/{run_id}/export.txt?lang=zz")
        assert response.status_code == 422

    def test_srt_cue_timing_unchanged(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        # Substitution, no reflow: translated cues keep the original timing.
        with session_factory() as session:
            run_id = seed_run(session)
        _translate(session_factory, run_id)
        client = _build_client(session_factory)
        original = client.get(f"/review/{run_id}/export.srt").text
        translated = client.get(f"/review/{run_id}/export.srt?lang=es").text
        orig_times = [ln for ln in original.splitlines() if "-->" in ln]
        trans_times = [ln for ln in translated.splitlines() if "-->" in ln]
        assert orig_times == trans_times

    def test_rttm_ignores_lang(self, session_factory: sessionmaker[Session]) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
        client = _build_client(session_factory)
        response = client.get(f"/review/{run_id}/export.rttm?lang=es")
        assert response.status_code == 200  # unknown query param, speaker turns only


class TestStepperTranslate:
    def test_props_present_with_preferred_language(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
        client = _build_client(
            session_factory, translation_target_language="fr"
        )
        body = client.get(f"/review/{run_id}/transcript").text
        assert "defaultTarget" in body
        assert "French" in body

    def test_props_null_when_gates_closed(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
        client = _build_client(session_factory, gates_open=False)
        body = client.get(f"/review/{run_id}/transcript").text
        assert '"translate": null' in body

    def test_preferred_matching_detected_gives_no_default(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = seed_run(session, detected_language="es")
        client = _build_client(
            session_factory, translation_target_language="es"
        )
        body = client.get(f"/review/{run_id}/transcript").text
        assert '"defaultTarget": null' in body

    def test_generate_json_contract(
        self,
        session_factory: sessionmaker[Session],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        published: list[uuid.UUID] = []

        def _sink(job_id: uuid.UUID) -> bool:
            published.append(job_id)
            return True

        monkeypatch.setattr("voxint.api.routers.legacy_runs._publish_translation_job", _sink)
        with session_factory() as session:
            run_id = seed_run(session)
        client = _build_client(session_factory)
        response = client.post(
            f"/runs/{run_id}/translation/generate",
            data={
                "csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_TRANSLATION_GENERATE),
                "target_language": "es",
            },
            headers={"Accept": "application/json"},
        )
        assert response.status_code == 200
        assert response.json() == {"started": True, "error": None}
        assert len(published) == 1
        with session_factory() as session:
            job = session.execute(select(TranslationJob)).scalar_one()
            assert job.target_language == "es"
        # A card-level refusal arrives as {started: False, error}, never HTML.
        again = client.post(
            f"/runs/{run_id}/translation/generate",
            data={
                "csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_TRANSLATION_GENERATE),
                "target_language": "es",
            },
            headers={"Accept": "application/json"},
        )
        assert again.status_code == 200
        payload = again.json()
        assert payload["started"] is False
        assert "already in progress" in payload["error"]

    def test_generate_json_reports_broker_outage(
        self,
        session_factory: sessionmaker[Session],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "voxint.api.routers.legacy_runs._publish_translation_job", lambda job_id: False
        )
        with session_factory() as session:
            run_id = seed_run(session)
        client = _build_client(session_factory)
        response = client.post(
            f"/runs/{run_id}/translation/generate",
            data={
                "csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_TRANSLATION_GENERATE),
                "target_language": "es",
            },
            headers={"Accept": "application/json"},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["started"] is False
        assert "broker unavailable" in payload["error"]

    def test_generate_without_target_reports_json_error(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = seed_run(session)
        client = _build_client(session_factory)
        response = client.post(
            f"/runs/{run_id}/translation/generate",
            data={"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_TRANSLATION_GENERATE)},
            headers={"Accept": "application/json"},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["started"] is False
        assert "Pick a language" in payload["error"]
