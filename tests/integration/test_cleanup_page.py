"""The Cleaned page and its routes (#758 slice 5) over the real app + Postgres.

Generations come from the real job executor with a scripted fake model, so
the diff, the counts and the staleness follow the stored rows and the genuine
source hash. Covers the gates, the D6 language states, generate and cancel,
the polled status block, the historical diff when stale, and the export-menu
row with its byte goldens.
"""

import re
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from httpx import Response
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from tests.integration.test_cleanup_export import _edit, _generate, _queue
from tests.integration.test_cleanup_writer import seed
from tests.integration.test_runs_api import _export_menu_html
from tests.integration.test_translation_jobs import seed_run
from tests.integration.test_translation_view_export import _CSRF_KEY, _build_client
from voxint.api.app import create_app
from voxint.api.auth import SESSION_COOKIE, create_session, new_session_token
from voxint.api.csrf import CSRF_CLEANUP_CANCEL, CSRF_CLEANUP_GENERATE, mint_csrf_token
from voxint.config import Settings
from voxint.db.models import CleanupJob, CleanupJobStatus, PipelineRun, UserRole
from voxint.enrichment.translations import TranslationError
from voxint.users import create_user

FIXTURES = Path(__file__).parent / "fixtures"
PUBLISH = "voxint.api.routers.legacy_runs._publish_cleanup_job"
DOWNLOAD_FORMATS = ("txt", "md", "srt", "vtt", "json")
ROW = (
    '\n        <p class="text-sm">\n'
    '          <a href="/runs/RUN_ID/cleanup">LLM clean-up</a>:\n'
    '          <span class="muted">a separate copy with filler phrases removed by the'
    " language model. Review and download it on its own page.</span>\n"
    "        </p>"
)


@pytest.fixture
def run_id(session_factory: sessionmaker[Session]) -> uuid.UUID:
    with session_factory() as session:
        return seed(session)


@pytest.fixture
def published(monkeypatch: pytest.MonkeyPatch) -> list[uuid.UUID]:
    sink: list[uuid.UUID] = []

    def _publish(job_id: uuid.UUID) -> bool:
        sink.append(job_id)
        return True

    monkeypatch.setattr(PUBLISH, _publish)
    return sink


def _page(client: TestClient, run_id: uuid.UUID, status: int = 200) -> str:
    response = client.get(f"/runs/{run_id}/cleanup")
    assert response.status_code == status, response.text
    return str(response.text)


def _post_generate(client: TestClient, run_id: uuid.UUID) -> Response:
    response: Response = client.post(
        f"/runs/{run_id}/cleanup/generate",
        data={"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_CLEANUP_GENERATE)},
        follow_redirects=False,
    )
    return response


def _post_cancel(client: TestClient, run_id: uuid.UUID, job_id: uuid.UUID) -> Response:
    response: Response = client.post(
        f"/runs/{run_id}/cleanup/{job_id}/cancel",
        data={"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_CLEANUP_CANCEL)},
        follow_redirects=False,
    )
    return response


def _jobs(session_factory: sessionmaker[Session], run_id: uuid.UUID) -> list[CleanupJob]:
    with session_factory() as session:
        return list(
            session.scalars(select(CleanupJob).where(CleanupJob.pipeline_run_id == run_id))
        )


def _has_downloads(page: str) -> bool:
    return "text=cleaned" in page


# ------------------------------------------------------------------ gates


def test_empty_page_offers_generate(
    session_factory: sessionmaker[Session], run_id: uuid.UUID
) -> None:
    page = _page(_build_client(session_factory), run_id)
    assert "Generate clean-up" in page
    assert f'action="/runs/{run_id}/cleanup/generate"' in page
    assert "Summary" not in page and not _has_downloads(page)
    assert "not verified as" not in page
    assert "hx-get" not in page


def test_gated_off_hides_generate_and_refuses(
    session_factory: sessionmaker[Session], run_id: uuid.UUID, published: list[uuid.UUID]
) -> None:
    client = _build_client(session_factory, gates_open=False)
    page = _page(client, run_id)
    assert "Clean-up is off" in page and "/settings#llm" in page
    assert "/cleanup/generate" not in page
    response = _post_generate(client, run_id)
    assert response.status_code == 409
    assert "Clean-up is off: it needs the language model enabled" in response.text
    assert _jobs(session_factory, run_id) == [] and published == []


def test_cancel_works_with_the_llm_disabled(
    session_factory: sessionmaker[Session], run_id: uuid.UUID
) -> None:
    _queue(session_factory, run_id)
    client = _build_client(session_factory, gates_open=False)
    (job,) = _jobs(session_factory, run_id)
    page = _page(client, run_id)
    assert f'action="/runs/{run_id}/cleanup/{job.id}/cancel"' in page
    response = _post_cancel(client, run_id, job.id)
    assert response.status_code == 303
    assert response.headers["location"] == f"/runs/{run_id}/cleanup"
    (job,) = _jobs(session_factory, run_id)
    assert job.status == CleanupJobStatus.CANCELLED.value
    page = _page(client, run_id)
    assert "The last clean-up cancelled" in page
    assert "Clean-up is off" in page


# ------------------------------------------------------------------ language


def test_other_language_is_refused(
    session_factory: sessionmaker[Session], published: list[uuid.UUID]
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, detected_language="es")
    client = _build_client(session_factory)
    page = _page(client, run_id)
    assert "Clean-up works on English transcripts only" in page and "Spanish" in page
    assert "/cleanup/generate" not in page
    response = _post_generate(client, run_id)
    assert response.status_code == 409
    assert "English transcripts only" in response.text
    assert published == []


def test_undetected_language_is_allowed_and_flagged(
    session_factory: sessionmaker[Session], published: list[uuid.UUID]
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, detected_language=None)
    client = _build_client(session_factory)
    assert "not verified as\n  English" in _page(client, run_id)
    response = _post_generate(client, run_id)
    assert response.status_code == 303
    assert len(published) == 1


# ------------------------------------------------------------------ generate / cancel


def test_generate_queues_publishes_and_polls(
    session_factory: sessionmaker[Session], run_id: uuid.UUID, published: list[uuid.UUID]
) -> None:
    client = _build_client(session_factory)
    response = _post_generate(client, run_id)
    assert response.status_code == 303
    assert response.headers["location"] == f"/runs/{run_id}/cleanup"
    (job,) = _jobs(session_factory, run_id)
    assert published == [job.id] and job.status == CleanupJobStatus.QUEUED.value
    page = _page(client, run_id)
    assert "Cleaning up: queued" in page
    assert f'hx-get="/runs/{run_id}/cleanup/status"' in page
    assert "/cleanup/generate" not in page
    again = _post_generate(client, run_id)
    assert again.status_code == 409
    assert "A clean-up is already in progress." in again.text
    assert len(_jobs(session_factory, run_id)) == 1 and len(published) == 1


def test_broker_outage_is_reported(
    session_factory: sessionmaker[Session], run_id: uuid.UUID, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(PUBLISH, lambda job_id: False)
    response = _post_generate(_build_client(session_factory), run_id)
    assert response.status_code == 503
    assert "the worker queue could not be reached" in response.text
    (job,) = _jobs(session_factory, run_id)
    assert job.status == CleanupJobStatus.QUEUED.value


def test_csrf_and_foreign_job_are_refused(
    session_factory: sessionmaker[Session], run_id: uuid.UUID
) -> None:
    _queue(session_factory, run_id)
    (job,) = _jobs(session_factory, run_id)
    with session_factory() as session:
        other = seed(session)
    client = _build_client(session_factory)
    assert client.post(f"/runs/{run_id}/cleanup/generate").status_code == 403
    assert client.post(f"/runs/{run_id}/cleanup/{job.id}/cancel").status_code == 403
    assert _post_cancel(client, other, job.id).status_code == 404
    assert _post_cancel(client, run_id, uuid.uuid4()).status_code == 404
    assert client.get(f"/runs/{uuid.uuid4()}/cleanup").status_code == 404
    (job,) = _jobs(session_factory, run_id)
    assert job.status == CleanupJobStatus.QUEUED.value


def test_status_fragment_refreshes_once_the_job_ends(
    session_factory: sessionmaker[Session], run_id: uuid.UUID
) -> None:
    _queue(session_factory, run_id)
    (job,) = _jobs(session_factory, run_id)
    client = _build_client(session_factory)
    url = f"/runs/{run_id}/cleanup/status"
    active = client.get(url, headers={"HX-Request": "true"})
    assert active.status_code == 200 and "HX-Redirect" not in active.headers
    assert f'id="cleanup-status-{run_id}"' in active.text and "hx-get" in active.text
    assert _post_cancel(client, run_id, job.id).status_code == 303
    ended = client.get(url, headers={"HX-Request": "true"})
    assert ended.headers["HX-Redirect"] == f"/runs/{run_id}/cleanup"
    assert "hx-get" not in ended.text
    assert "HX-Redirect" not in client.get(url).headers


# ------------------------------------------------------------------ the diff


def test_current_generation_renders_diff_counts_and_downloads(
    session_factory: sessionmaker[Session], run_id: uuid.UUID
) -> None:
    _generate(
        session_factory,
        run_id,
        {0: "I think it's fine.", 1: "so we went", 2: "to Paris, yesterday.", 3: "Okay."},
    )
    page = _page(_build_client(session_factory), run_id)
    flat = " ".join(page.split())
    assert '<p><del class="cleanup-removed">I mean,</del> I think it&#39;s fine.</p>' in flat
    assert '<p><del class="cleanup-removed">Um,</del> so we went</p>' in flat
    assert '<p>to Paris, <del class="cleanup-removed">uh,</del> yesterday.</p>' in flat
    assert "<p>So -- we go.</p>" in flat
    assert "3 of 5 lines changed, 4 words removed." in flat
    assert "<li>no word timings to anchor the change: 1 line</li>" in flat
    assert (
        '<p>Okay then. <span class="muted text-sm">(kept: no word timings to anchor'
        " the change)</span></p>"
    ) in flat
    assert "Out of date" not in page
    for fmt in DOWNLOAD_FORMATS:
        assert f"/review/{run_id}/export.{fmt}?text=cleaned" in page
    assert "Generate again" in page
    assert "prompt version 1" in page and "generation 1" in page


def test_stale_generation_shows_the_historical_diff_without_downloads(
    session_factory: sessionmaker[Session], run_id: uuid.UUID
) -> None:
    _generate(session_factory, run_id)
    _edit(session_factory, run_id)
    page = _page(_build_client(session_factory), run_id)
    assert "Out of date." in page
    assert not _has_downloads(page)
    assert '<del class="cleanup-removed">I mean,</del>' in page
    assert "Edited afterwards." not in page


def test_transcript_text_is_escaped(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = seed(session, texts=("Um, <b>bold</b> & so on.",))
    _generate(session_factory, run_id, {0: "<b>bold</b> & so on."})
    page = _page(_build_client(session_factory), run_id)
    assert '<del class="cleanup-removed">Um,</del> &lt;b&gt;bold&lt;/b&gt; &amp; so on.' in page
    assert "<b>bold</b>" not in page


# ------------------------------------------------------------------ entry points


def _menu(
    session_factory: sessionmaker[Session],
    run_id: uuid.UUID,
    *,
    editor: bool,
    gates_open: bool,
) -> str:
    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        media_id = run.media_item_id
    client = _build_client(session_factory, gates_open=gates_open)
    response = client.get(
        f"/media/{media_id}/editor"
        if editor
        else f"/runs/{run_id}/transcript?read=1&text=corrected&timestamps=false"
    )
    assert response.status_code == 200
    menu = _export_menu_html(response.text).replace(str(run_id), "RUN_ID")
    return menu.replace(run_id.hex[:8], "RUN_STEM")


@pytest.mark.parametrize("editor", [False, True])
def test_export_menu_row(
    session_factory: sessionmaker[Session], run_id: uuid.UUID, editor: bool
) -> None:
    """The row is the only difference from the unchanged golden: present when
    the LLM is on, or when a generation exists with it off."""
    expected = (
        FIXTURES / ("export_menu_editor.html" if editor else "export_menu_read.html")
    ).read_text()
    anchor = "\n      </details>\n    </div>\n  </details>\n</div>"
    assert expected.endswith("</ul>" + anchor)
    with_row = expected[: -len(anchor)] + ROW + anchor
    assert _menu(session_factory, run_id, editor=editor, gates_open=False) == expected
    assert _menu(session_factory, run_id, editor=editor, gates_open=True) == with_row
    _generate(session_factory, run_id)
    assert _menu(session_factory, run_id, editor=editor, gates_open=False) == with_row


def test_read_page_links_to_the_cleaned_page(
    session_factory: sessionmaker[Session], run_id: uuid.UUID
) -> None:
    url = f"/runs/{run_id}/transcript?read=1&text=corrected&timestamps=false"
    on = _build_client(session_factory).get(url).text
    nav = re.search(r'aria-label="Reading view options">.*?</nav>', on, re.S)
    assert nav is not None and f'href="/runs/{run_id}/cleanup"' in nav[0]
    off = _build_client(session_factory, gates_open=False).get(url).text
    assert f"/runs/{run_id}/cleanup" not in off


# ------------------------------------------------------------------ review round 1


def test_regeneration_keeps_the_current_copy_downloadable_like_the_export(
    session_factory: sessionmaker[Session], run_id: uuid.UUID, published: list[uuid.UUID]
) -> None:
    """While a regeneration runs, the page offers the current copy exactly
    when the export still serves it (slice 4's rule)."""
    _generate(session_factory, run_id)
    client = _build_client(session_factory)
    assert _post_generate(client, run_id).status_code == 303
    page = _page(client, run_id)
    assert "Cleaning up: queued" in page and _has_downloads(page)
    export = client.get(f"/review/{run_id}/export.txt", params={"text": "cleaned"})
    assert export.status_code == 200


def test_unreadable_source_says_so_instead_of_claiming_a_change(
    session_factory: sessionmaker[Session], run_id: uuid.UUID, monkeypatch: pytest.MonkeyPatch
) -> None:
    _generate(session_factory, run_id)

    def _unreadable(session: Session, run_id: uuid.UUID) -> None:
        raise TranslationError("the transcript changed while it was being read; try again")

    monkeypatch.setattr("voxint.api.routers.legacy_runs.load_translation_source", _unreadable)
    page = _page(_build_client(session_factory), run_id)
    assert "The current transcript could not be read" in page
    assert "The transcript changed after this clean-up" not in page
    assert "the transcript changed while it was being read" in page
    assert not _has_downloads(page)


def test_viewer_cannot_generate_or_cancel(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id = seed(session)
    _queue(session_factory, run_id)
    (job,) = _jobs(session_factory, run_id)
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        voxint_multi_user=True,
        database_url="postgresql+psycopg://x/x",
        csrf_secret=_CSRF_KEY,
        llm_enabled=True,
    )
    app = create_app(settings=settings, session_factory=session_factory)
    seed_onboarded(session_factory, llm_enabled=True)
    with session_factory() as session:
        create_user(session, username="admin", password="adminpass", role=UserRole.ADMIN)
        viewer = create_user(
            session, username="watcher", password="watcherpass", role=UserRole.VIEWER
        )
        token = new_session_token()
        create_session(session, user_id=viewer.id, token=token, ttl_seconds=3600)
        session.commit()
    client = TestClient(app, cookies={SESSION_COOKIE: token})
    page = _page(client, run_id)
    assert "/cancel" not in page and "/cleanup/generate" not in page
    for response in (_post_generate(client, run_id), _post_cancel(client, run_id, job.id)):
        assert response.status_code == 403
        assert response.json()["detail"] == "write access required"
    (job,) = _jobs(session_factory, run_id)
    assert job.status == CleanupJobStatus.QUEUED.value and not job.cancel_requested
