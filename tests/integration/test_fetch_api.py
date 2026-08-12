"""POST /fetch — browser URL ingestion, end to end against real Postgres.

Wiring over the already-tested submit_url service (its DB semantics —
replay/conflict/SSRF validation — are covered in test_ingest_service.py and
test_ingest_url.py). These exercise the ROUTE: it creates a source_url MediaItem
+ QUEUED run and publishes commit-before-publish, maps the service's typed
errors to status codes, refuses cleanly when ytdlp_enabled is off, gives the
upload and fetch forms independent submission ids, and shows a run's provenance
as a bare host — never the raw URL (whose query can carry a signed token).

Synthetic data is neutral: example.com and the IETF TEST-NET documentation
ranges only, never a private/internal host.
"""

import re
import uuid

import pytest
from celery.exceptions import OperationalError
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from voxint.api.app import create_app
from voxint.api.csrf import CSRF_FETCH, CSRF_SUBMIT, mint_csrf_token
from voxint.config import Settings
from voxint.db.models import MediaItem, PipelineRun, RunStatus

CREDS = ("reviewer", "s3cret")
_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
_CSRF_KEY = "fetch-api-test-csrf-key"  # low-entropy; a known secret lets tests mint


def _fd(**kwargs: str) -> dict[str, str]:
    """Form data with a valid /fetch CSRF token merged in (the real forms carry
    one; posting without it is 403 — see test_fetch_rejected_without_csrf_token)."""
    return {"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_FETCH), **kwargs}


def make_client(
    session_factory: sessionmaker[Session], *, ytdlp_enabled: bool = True
) -> TestClient:
    settings = Settings(
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        ytdlp_enabled=ytdlp_enabled,
        csrf_secret=_CSRF_KEY,
    )
    client = TestClient(create_app(settings=settings, session_factory=session_factory))
    client.auth = CREDS
    return client


@pytest.fixture()
def client(session_factory: sessionmaker[Session]) -> TestClient:
    return make_client(session_factory)


@pytest.fixture()
def published(monkeypatch: pytest.MonkeyPatch) -> list[uuid.UUID]:
    """Capture commit-before-publish enqueues without a live broker."""
    calls: list[uuid.UUID] = []
    monkeypatch.setattr("voxint.api.app._publish_run", calls.append)
    return calls


def _run_id_from_redirect(location: str) -> uuid.UUID:
    assert location.startswith("/runs/")
    return uuid.UUID(location.removeprefix("/runs/").split("?", 1)[0])


# --- happy path ---------------------------------------------------------------


def test_fetch_creates_source_url_run_and_publishes(
    client: TestClient,
    session_factory: sessionmaker[Session],
    published: list[uuid.UUID],
) -> None:
    sub = uuid.uuid4().hex
    resp = client.post(
        "/fetch", data=_fd(url=_URL, submission_id=sub), follow_redirects=False
    )
    assert resp.status_code == 303
    run_id = _run_id_from_redirect(resp.headers["location"])

    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        assert run.status == RunStatus.QUEUED.value
        media = session.get(MediaItem, run.media_item_id)
        assert media is not None
        assert media.source_url == _URL
        # No file is written — source_path is the pre-assigned, uuid-namespaced
        # location the worker's ACQUIRE stage will download into.
        assert media.source_path == f"incoming/{sub}/source"

    assert published == [run_id]  # commit-before-publish fired exactly once


def test_runs_page_offers_the_fetch_form(client: TestClient) -> None:
    body = client.get("/runs").text
    assert 'action="/fetch"' in body
    assert 'name="url"' in body


def test_upload_and_fetch_forms_get_independent_submission_ids(
    client: TestClient,
) -> None:
    # The two forms must NOT share a submission_id (they would collide on the
    # source_path namespace). Both hidden fields render, with distinct values.
    body = client.get("/runs").text
    ids = re.findall(r'name="submission_id" value="([0-9a-f]+)"', body)
    assert len(ids) == 2
    assert ids[0] != ids[1]


# --- validation / error mapping (URL never echoed) ----------------------------


def test_fetch_bad_url_is_422_and_creates_nothing(
    client: TestClient,
    session_factory: sessionmaker[Session],
    published: list[uuid.UUID],
) -> None:
    resp = client.post(
        "/fetch",
        data=_fd(url="ftp://example.com/f.mp3", submission_id=uuid.uuid4().hex),
        follow_redirects=False,
    )
    assert resp.status_code == 422
    assert published == []
    with session_factory() as session:
        assert session.execute(select(PipelineRun)).first() is None
        assert session.execute(select(MediaItem)).first() is None


def test_fetch_error_body_never_echoes_the_url(client: TestClient) -> None:
    # A rejected URL's signed query must not leak into the 422 body. The host is
    # a TEST-NET literal (non-global) so validation fails on the host, and the
    # error message is generic by construction.
    secret = "SUPERSECRETSIGNATURE"
    resp = client.post(
        "/fetch",
        data=_fd(
            url=f"http://192.0.2.9/media?token={secret}",
            submission_id=uuid.uuid4().hex,
        ),
        follow_redirects=False,
    )
    assert resp.status_code == 422
    assert secret not in resp.text


def test_fetch_non_uuid_submission_id_is_422(
    client: TestClient, published: list[uuid.UUID]
) -> None:
    resp = client.post(
        "/fetch",
        data=_fd(url=_URL, submission_id="not-a-uuid"),
        follow_redirects=False,
    )
    assert resp.status_code == 422
    assert published == []


# --- CSRF ---------------------------------------------------------------------


def test_fetch_rejected_without_csrf_token(
    client: TestClient,
    session_factory: sessionmaker[Session],
    published: list[uuid.UUID],
) -> None:
    # No csrf_token field ⇒ 403 before any DB write (a forged cross-site POST).
    resp = client.post(
        "/fetch",
        data={"url": _URL, "submission_id": uuid.uuid4().hex},  # NB: no _fd() token
        follow_redirects=False,
    )
    assert resp.status_code == 403
    assert published == []
    with session_factory() as session:
        assert session.execute(select(PipelineRun)).first() is None
        assert session.execute(select(MediaItem)).first() is None


def test_fetch_rejected_with_wrong_action_token(
    client: TestClient, published: list[uuid.UUID]
) -> None:
    # A token minted for /submit is not valid on /fetch (action binding).
    resp = client.post(
        "/fetch",
        data={
            "url": _URL,
            "submission_id": uuid.uuid4().hex,
            "csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_SUBMIT),
        },
        follow_redirects=False,
    )
    assert resp.status_code == 403
    assert published == []


def test_runs_page_renders_fetch_csrf_token(client: TestClient) -> None:
    # The fetch form carries a hidden csrf_token that verifies for /fetch.
    body = client.get("/runs").text
    match = re.search(
        r'action="/fetch".*?name="csrf_token" value="([^"]+)"', body, re.DOTALL
    )
    assert match is not None
    from voxint.api.csrf import verify_csrf_token

    assert verify_csrf_token(_CSRF_KEY, CSRF_FETCH, match.group(1))


# --- replay idempotency -------------------------------------------------------


def test_fetch_replay_same_url_returns_same_run(
    client: TestClient,
    session_factory: sessionmaker[Session],
    published: list[uuid.UUID],
) -> None:
    sub = uuid.uuid4().hex
    first = client.post(
        "/fetch", data=_fd(url=_URL, submission_id=sub), follow_redirects=False
    )
    second = client.post(
        "/fetch", data=_fd(url=_URL, submission_id=sub), follow_redirects=False
    )
    assert first.status_code == second.status_code == 303
    run_id = _run_id_from_redirect(first.headers["location"])
    assert _run_id_from_redirect(second.headers["location"]) == run_id

    with session_factory() as session:
        assert len(session.execute(select(PipelineRun)).scalars().all()) == 1
        assert len(session.execute(select(MediaItem)).scalars().all()) == 1
    assert published == [run_id, run_id]  # at-least-once; the worker dedups


def test_fetch_replay_different_url_conflicts(
    client: TestClient,
    session_factory: sessionmaker[Session],
    published: list[uuid.UUID],
) -> None:
    sub = uuid.uuid4().hex
    first = client.post(
        "/fetch", data=_fd(url=_URL, submission_id=sub), follow_redirects=False
    )
    assert first.status_code == 303
    run_id = _run_id_from_redirect(first.headers["location"])

    clash = client.post(
        "/fetch",
        data=_fd(url="https://example.com/other.mp3", submission_id=sub),
        follow_redirects=False,
    )
    assert clash.status_code == 409

    with session_factory() as session:
        assert len(session.execute(select(PipelineRun)).scalars().all()) == 1
        media = session.execute(select(MediaItem)).scalar_one()
        assert media.source_url == _URL  # the first url wins; unchanged
    assert published == [run_id]  # the conflicting POST never publishes


# --- ytdlp_enabled refusal ----------------------------------------------------


def test_fetch_refused_when_ytdlp_disabled(
    session_factory: sessionmaker[Session], published: list[uuid.UUID]
) -> None:
    client = make_client(session_factory, ytdlp_enabled=False)
    resp = client.post(
        "/fetch",
        data=_fd(url=_URL, submission_id=uuid.uuid4().hex),
        follow_redirects=False,
    )
    assert resp.status_code == 403
    assert published == []
    with session_factory() as session:
        assert session.execute(select(PipelineRun)).first() is None
        assert session.execute(select(MediaItem)).first() is None


def test_runs_page_hides_fetch_form_when_disabled(
    session_factory: sessionmaker[Session],
) -> None:
    client = make_client(session_factory, ytdlp_enabled=False)
    body = client.get("/runs").text
    assert 'action="/fetch"' not in body
    assert "URL ingestion is disabled" in body


# --- provenance display (host, never the raw URL) -----------------------------


def test_run_detail_shows_host_not_raw_url(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    secret_url = "https://cdn.example.com/media.mp3?token=SUPERSECRETSIGNATURE"
    sub = uuid.uuid4().hex
    resp = client.post(
        "/fetch", data=_fd(url=secret_url, submission_id=sub), follow_redirects=False
    )
    run_id = _run_id_from_redirect(resp.headers["location"])

    detail = client.get(f"/runs/{run_id}").text
    assert "cdn.example.com" in detail  # provenance host is shown
    assert "SUPERSECRETSIGNATURE" not in detail  # the signed token is not
    assert "token=" not in detail
    assert secret_url not in detail  # the raw URL never reaches the view


def test_run_detail_local_run_omits_the_source_line(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    # A local/uploaded run has no source_url; provenance_host returns None and the
    # detail view must OMIT the "Source" line entirely (never render "Source: None").
    from voxint.pipeline.engine import submit

    with session_factory() as session:
        media = MediaItem(source_path="incoming/local/clip.wav")  # source_url is None
        session.add(media)
        session.flush()
        run_id = submit(session, media.id).id
        session.commit()

    detail = client.get(f"/runs/{run_id}").text
    assert "Source:" not in detail  # the provenance line is absent, not "Source: None"


# --- broker-down degradation --------------------------------------------------


def test_broker_down_fetch_leaves_run_queued(
    client: TestClient,
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Commit-before-publish: a broker outage at enqueue is non-fatal — the run
    # is durably QUEUED (never FAILED, no error) and the redirect flags the
    # deferred-enqueue banner for the recovery sweep.
    def _broker_down(_run_id: uuid.UUID) -> None:
        raise OperationalError("Error 111 connecting to redis. Connection refused.")

    monkeypatch.setattr("voxint.api.app._publish_run", _broker_down)
    sub = uuid.uuid4().hex
    resp = client.post(
        "/fetch", data=_fd(url=_URL, submission_id=sub), follow_redirects=False
    )
    assert resp.status_code == 303
    assert resp.headers["location"].endswith("?enqueue=deferred")
    run_id = _run_id_from_redirect(resp.headers["location"])

    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        assert run.status == RunStatus.QUEUED.value  # never FAILED
        assert run.error is None  # a broker outage is not a stage failure
