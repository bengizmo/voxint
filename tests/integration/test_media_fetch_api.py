"""POST /media/fetch — browser URL ingestion, end to end against real Postgres.

Wiring over the already-tested submit_url service (its DB semantics —
replay/conflict/SSRF validation — are covered in test_ingest_service.py and
test_ingest_url.py). These exercise the ROUTE: it creates a source_url MediaItem
+ QUEUED run and publishes commit-before-publish, maps the service's typed
errors to status codes, refuses cleanly when ytdlp_enabled is off, gives the
upload and fetch forms independent submission ids, and shows a run's provenance
as a bare host — never the raw URL (whose query can carry a signed token). They
moved here from the legacy ``POST /fetch`` when #682 removed that route.

Synthetic data is neutral: example.com and the IETF TEST-NET documentation
ranges only, never a private/internal host.
"""

import re
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.api.csrf import CSRF_MEDIA_FETCH, CSRF_MEDIA_SUBMIT, mint_csrf_token
from voxint.config import Settings
from voxint.db.models import MediaItem, PipelineRun, RunStatus

CREDS = ("reviewer", "s3cret")
_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
_CSRF_KEY = "fetch-api-test-csrf-key"  # low-entropy; a known secret lets tests mint
_QUEUED = "/media?submitted=1"
_DEFERRED = "/media?submitted=deferred"


def _fd(**kwargs: str) -> dict[str, str]:
    """Form data with a valid /media/fetch CSRF token merged in (the real forms
    carry one; posting without it is 403 — see
    test_fetch_rejected_without_csrf_token)."""
    return {"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_MEDIA_FETCH), **kwargs}


def make_client(
    session_factory: sessionmaker[Session], tmp_path: Path, *, ytdlp_enabled: bool = True
) -> TestClient:
    settings = Settings(
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        media_root=tmp_path,
        ytdlp_enabled=ytdlp_enabled,
        csrf_secret=_CSRF_KEY,
    )
    client = TestClient(create_app(settings=settings, session_factory=session_factory))
    client.auth = CREDS
    seed_onboarded(session_factory)
    return client


@pytest.fixture()
def client(session_factory: sessionmaker[Session], tmp_path: Path) -> TestClient:
    return make_client(session_factory, tmp_path)


@pytest.fixture()
def published(monkeypatch: pytest.MonkeyPatch) -> list[uuid.UUID]:
    """Capture commit-before-publish enqueues without a live broker."""
    calls: list[uuid.UUID] = []
    monkeypatch.setattr(
        "voxint.api.routers.deps._publish_run", lambda run_id, **_kwargs: calls.append(run_id)
    )

    from voxint.ingest.service import SubmissionResult

    def _record_publish(self: SubmissionResult) -> bool:
        calls.append(self.run_id)
        return True

    monkeypatch.setattr(SubmissionResult, "publish", _record_publish)
    return calls


def _run_id_for(session_factory: sessionmaker[Session], submission_id: str) -> uuid.UUID:
    """The one run the fetch under ``submission_id`` created (the library
    redirect carries no run id)."""
    with session_factory() as session:
        media = session.execute(
            select(MediaItem).where(MediaItem.source_path == f"incoming/{submission_id}/source")
        ).scalar_one()
        return session.execute(
            select(PipelineRun.id).where(PipelineRun.media_item_id == media.id)
        ).scalar_one()


# --- happy path ---------------------------------------------------------------


def test_fetch_creates_source_url_run_and_publishes(
    client: TestClient,
    session_factory: sessionmaker[Session],
    published: list[uuid.UUID],
) -> None:
    sub = uuid.uuid4().hex
    resp = client.post(
        "/media/fetch", data=_fd(url=_URL, submission_id=sub), follow_redirects=False
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == _QUEUED
    run_id = _run_id_for(session_factory, sub)

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


def test_media_page_offers_the_fetch_form(client: TestClient) -> None:
    body = client.get("/media").text
    assert 'action="/media/fetch"' in body
    assert 'name="url"' in body


def test_upload_and_fetch_forms_get_independent_submission_ids(
    client: TestClient,
) -> None:
    # The two forms must NOT share a submission_id (they would collide on the
    # source_path namespace). Both hidden fields render, with distinct values.
    body = client.get("/media").text
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
        "/media/fetch",
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
        "/media/fetch",
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
        "/media/fetch",
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
        "/media/fetch",
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
    # A token minted for /media/submit is not valid on /media/fetch (action binding).
    resp = client.post(
        "/media/fetch",
        data={
            "url": _URL,
            "submission_id": uuid.uuid4().hex,
            "csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_MEDIA_SUBMIT),
        },
        follow_redirects=False,
    )
    assert resp.status_code == 403
    assert published == []


def test_media_page_renders_fetch_csrf_token(client: TestClient) -> None:
    # The fetch form carries a hidden csrf_token that verifies for /media/fetch.
    body = client.get("/media").text
    match = re.search(
        r'action="/media/fetch".*?name="csrf_token" value="([^"]+)"', body, re.DOTALL
    )
    assert match is not None
    from voxint.api.csrf import verify_csrf_token

    assert verify_csrf_token(_CSRF_KEY, CSRF_MEDIA_FETCH, match.group(1))


# --- replay idempotency -------------------------------------------------------


def test_fetch_replay_same_url_returns_same_run(
    client: TestClient,
    session_factory: sessionmaker[Session],
    published: list[uuid.UUID],
) -> None:
    sub = uuid.uuid4().hex
    first = client.post(
        "/media/fetch", data=_fd(url=_URL, submission_id=sub), follow_redirects=False
    )
    second = client.post(
        "/media/fetch", data=_fd(url=_URL, submission_id=sub), follow_redirects=False
    )
    assert first.status_code == second.status_code == 303
    assert first.headers["location"] == second.headers["location"] == _QUEUED
    run_id = _run_id_for(session_factory, sub)

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
        "/media/fetch", data=_fd(url=_URL, submission_id=sub), follow_redirects=False
    )
    assert first.status_code == 303
    run_id = _run_id_for(session_factory, sub)

    clash = client.post(
        "/media/fetch",
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
    session_factory: sessionmaker[Session], tmp_path: Path, published: list[uuid.UUID]
) -> None:
    client = make_client(session_factory, tmp_path, ytdlp_enabled=False)
    resp = client.post(
        "/media/fetch",
        data=_fd(url=_URL, submission_id=uuid.uuid4().hex),
        follow_redirects=False,
    )
    assert resp.status_code == 403
    assert published == []
    with session_factory() as session:
        assert session.execute(select(PipelineRun)).first() is None
        assert session.execute(select(MediaItem)).first() is None


def test_media_page_disables_fetch_form_when_disabled(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    client = make_client(session_factory, tmp_path, ytdlp_enabled=False)
    body = client.get("/media").text
    # The form stays (the add menu keeps its shape) but cannot be submitted, and
    # the page says why and where to turn it on.
    assert re.search(r'<input type="url" name="url"[^>]*\brequired disabled>', body)
    assert re.search(r'<button type="submit" disabled>Fetch and transcribe</button>', body)
    assert "Fetching from a URL is turned off." in body
    assert "compose.ytdlp-egress.yaml" not in body


def test_media_page_points_at_the_egress_overlay_when_enabled(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    """The fetch panel carries the egress warning the legacy /runs form had
    (#682): fetching untrusted links needs the restricted overlay."""
    body = make_client(session_factory, tmp_path, ytdlp_enabled=True).get("/media").text
    assert "compose.ytdlp-egress.yaml" in body
    assert "URL ingestion &amp; egress security" in body
    assert "Fetching from a URL is turned off." not in body


# --- provenance display (host, never the raw URL) -----------------------------


def test_run_detail_shows_host_not_raw_url(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    secret_url = "https://cdn.example.com/media.mp3?token=SUPERSECRETSIGNATURE"
    sub = uuid.uuid4().hex
    resp = client.post(
        "/media/fetch", data=_fd(url=secret_url, submission_id=sub), follow_redirects=False
    )
    assert resp.status_code == 303
    run_id = _run_id_for(session_factory, sub)

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
    from voxint.domain_packs.base import load_default
    from voxint.pipeline.engine import submit

    with session_factory() as session:
        media = MediaItem(source_path="incoming/local/clip.wav")  # source_url is None
        session.add(media)
        session.flush()
        run_id = submit(session, media.id, domain_pack=load_default().to_mapping()).id
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
    # deferred-enqueue notice for the recovery sweep.
    from voxint.ingest.service import SubmissionResult

    monkeypatch.setattr(SubmissionResult, "publish", lambda self: False)
    sub = uuid.uuid4().hex
    resp = client.post(
        "/media/fetch", data=_fd(url=_URL, submission_id=sub), follow_redirects=False
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == _DEFERRED
    run_id = _run_id_for(session_factory, sub)

    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        assert run.status == RunStatus.QUEUED.value  # never FAILED
        assert run.error is None  # a broker outage is not a stage failure


def test_fetch_maps_domain_pack_error_to_422(
    client: TestClient,
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A freeze-time snapshot collision (issue #84) / unresolvable pack (issue #11)
    # raises DomainPackError from submit_url; the route must map it to a plain-language
    # 422 (never a raw 500) and queue no run. The raise itself (operator↔pack union
    # collision) is proven at the service level in test_ingest_service.py; here we lock
    # the ROUTE's error mapping, which the review found missing.
    from voxint.domain_packs.base import DomainPackError

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise DomainPackError(
            "domain pack corrections are not idempotent: rule 'b' (match 'Board') "
            "would re-fire on the replacement of rule 'zb'"
        )

    monkeypatch.setattr("voxint.api.routers.media.submit_url", _raise)
    resp = client.post("/media/fetch", data=_fd(url=_URL, submission_id=uuid.uuid4().hex))
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert "couldn't be applied" in detail
    assert "not idempotent" in detail  # the substance is preserved
    assert "domain pack corrections" not in detail  # raw pack jargon is softened out
    with session_factory() as session:
        assert session.execute(select(PipelineRun)).scalars().all() == []
