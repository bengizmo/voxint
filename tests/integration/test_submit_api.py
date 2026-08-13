"""POST /submit — the browser upload path, end to end against real Postgres.

Covers the happy path (bytes land uuid-namespaced, MediaItem gets the first-ever
sha256/size, run queued + published), the size cap on both the early
Content-Length gate and the authoritative stream copy, filename rejection,
submission-id replay idempotency (same bytes → same run) and its conflict
(different bytes → 409), and that a failed upload never leaves a temp behind.
"""

import hashlib
import io
import re
import threading
import time
import uuid
import wave
from pathlib import Path

import pytest
from celery.exceptions import OperationalError
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import _UPLOAD_ENVELOPE_ALLOWANCE, create_app
from voxint.api.csrf import CSRF_FETCH, CSRF_SUBMIT, mint_csrf_token
from voxint.config import Settings
from voxint.db.models import MediaItem, PipelineRun, RunStatus
from voxint.ingest.service import UploadConflictError, UploadTooLargeError, submit_upload

CREDS = ("reviewer", "s3cret")
_CSRF_KEY = "submit-api-test-csrf-key"  # low-entropy; a known secret lets tests mint


def _sd(**kwargs: str) -> dict[str, str]:
    """Form fields (submission_id, …) with a valid /submit CSRF token merged in —
    the real upload form carries one; posting without it is 403."""
    return {"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_SUBMIT), **kwargs}


def wav_bytes(seconds: float = 0.02) -> bytes:
    frames = int(16000 * seconds)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * frames)
    return buf.getvalue()


@pytest.fixture()
def media_root(tmp_path: Path) -> Path:
    return tmp_path


def make_client(
    session_factory: sessionmaker[Session], media_root: Path, *, max_bytes: int
) -> TestClient:
    settings = Settings(
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        media_root=media_root,
        upload_max_bytes=max_bytes,
        csrf_secret=_CSRF_KEY,
    )
    client = TestClient(create_app(settings=settings, session_factory=session_factory))
    client.auth = CREDS
    seed_onboarded(session_factory)
    return client


@pytest.fixture()
def client(
    session_factory: sessionmaker[Session], media_root: Path
) -> TestClient:
    return make_client(session_factory, media_root, max_bytes=10 * 1024 * 1024)


@pytest.fixture()
def published(monkeypatch: pytest.MonkeyPatch) -> list[uuid.UUID]:
    """Capture commit-before-publish enqueues without a live broker."""
    calls: list[uuid.UUID] = []
    monkeypatch.setattr("voxint.api.app._publish_run", calls.append)
    return calls


def _run_id_from_redirect(location: str) -> uuid.UUID:
    assert location.startswith("/runs/")
    return uuid.UUID(location.removeprefix("/runs/"))


# --- happy path ---------------------------------------------------------------


def test_upload_creates_namespaced_media_and_publishes(
    client: TestClient,
    session_factory: sessionmaker[Session],
    media_root: Path,
    published: list[uuid.UUID],
) -> None:
    body = wav_bytes()
    sub = uuid.uuid4().hex
    resp = client.post(
        "/submit",
        files={"file": ("episode 5.wav", body, "audio/wav")},
        data=_sd(submission_id=sub),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    run_id = _run_id_from_redirect(resp.headers["location"])

    landed = media_root / "incoming" / sub / "episode 5.wav"
    assert landed.read_bytes() == body  # exact bytes, atomically placed
    # No temp part-file left in the submission dir.
    assert [p.name for p in landed.parent.iterdir()] == ["episode 5.wav"]

    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        assert run.status == RunStatus.QUEUED.value
        media = session.get(MediaItem, run.media_item_id)
        assert media is not None
        assert media.source_path == f"incoming/{sub}/episode 5.wav"
        assert media.size_bytes == len(body)
        assert media.sha256 == hashlib.sha256(body).hexdigest()

    assert published == [run_id]  # commit-before-publish fired exactly once


def test_runs_page_offers_the_upload_form(client: TestClient) -> None:
    body = client.get("/runs").text
    assert 'action="/submit"' in body
    assert 'name="submission_id"' in body
    assert 'enctype="multipart/form-data"' in body


# --- CSRF ---------------------------------------------------------------------


def test_upload_form_renders_valid_csrf_token(client: TestClient) -> None:
    from voxint.api.csrf import verify_csrf_token

    body = client.get("/runs").text
    match = re.search(
        r'action="/submit".*?name="csrf_token" value="([^"]+)"', body, re.DOTALL
    )
    assert match is not None
    assert verify_csrf_token(_CSRF_KEY, CSRF_SUBMIT, match.group(1))


def test_submit_rejected_without_csrf_token(
    client: TestClient,
    session_factory: sessionmaker[Session],
    published: list[uuid.UUID],
) -> None:
    # No csrf_token ⇒ 403 before the file is finalized / any row is written.
    resp = client.post(
        "/submit",
        files={"file": ("ep.wav", wav_bytes(), "audio/wav")},
        data={"submission_id": uuid.uuid4().hex},  # NB: no csrf_token
        follow_redirects=False,
    )
    assert resp.status_code == 403
    assert published == []
    with session_factory() as session:
        assert session.execute(select(MediaItem)).first() is None


def test_submit_rejected_with_wrong_action_token(
    client: TestClient, published: list[uuid.UUID]
) -> None:
    # A token minted for /fetch is not valid on /submit (action binding).
    resp = client.post(
        "/submit",
        files={"file": ("ep.wav", wav_bytes(), "audio/wav")},
        data={
            "submission_id": uuid.uuid4().hex,
            "csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_FETCH),
        },
        follow_redirects=False,
    )
    assert resp.status_code == 403
    assert published == []


# --- broker-down degradation --------------------------------------------------


def test_broker_down_submit_leaves_run_queued_and_flags_banner(
    client: TestClient,
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Commit-before-publish means a broker outage at enqueue time is non-fatal:
    # the upload succeeds, the durable run stays QUEUED (never FAILED, no error),
    # and the redirect flags the deferred-enqueue banner. The recovery sweep
    # republishes it later. Simulated by making the publish raise the exact broker
    # exception _publish_or_defer catches.
    def _broker_down(_run_id: uuid.UUID) -> None:
        raise OperationalError("Error 111 connecting to redis. Connection refused.")

    monkeypatch.setattr("voxint.api.app._publish_run", _broker_down)

    sub = uuid.uuid4().hex
    resp = client.post(
        "/submit",
        files={"file": ("ep.wav", wav_bytes(), "audio/wav")},
        data=_sd(submission_id=sub),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"].endswith("?enqueue=deferred")
    run_id = _run_id_from_redirect(resp.headers["location"].split("?", 1)[0])

    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        assert run.status == RunStatus.QUEUED.value  # never FAILED
        assert run.error is None  # a broker outage is not a stage failure

    # The detail page tells the operator the run is queued and self-healing.
    detail = client.get(f"/runs/{run_id}?enqueue=deferred").text
    assert "enqueue was deferred" in detail


# --- size cap -----------------------------------------------------------------


def test_oversized_upload_rejected_and_no_run(
    session_factory: sessionmaker[Session],
    media_root: Path,
    published: list[uuid.UUID],
) -> None:
    # Body is under the coarse middleware gate but over the per-file cap, so the
    # authoritative streaming check refuses it — 413, no run, nothing published.
    client = make_client(session_factory, media_root, max_bytes=64)
    resp = client.post(
        "/submit",
        files={"file": ("big.wav", wav_bytes(seconds=0.1), "audio/wav")},
        data=_sd(submission_id=uuid.uuid4().hex),
        follow_redirects=False,
    )
    assert resp.status_code == 413
    assert published == []
    with session_factory() as session:
        assert session.execute(select(PipelineRun)).first() is None


def test_oversized_content_length_rejected_before_auth_and_body(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # An honestly-declared over-cap body is refused by the ASGI size middleware
    # BEFORE Starlette parses (spools) it and before the route's Basic-auth check
    # — proven by sending it unauthenticated and still getting 413 (not 401). This
    # is what makes "reject oversized Content-Length early" real. (A chunked/
    # absent length is not covered here — that residual is the security slice's.)
    client = make_client(session_factory, media_root, max_bytes=64)
    client.auth = None
    threshold = 64 + _UPLOAD_ENVELOPE_ALLOWANCE
    resp = client.post(
        "/submit",
        files={"file": ("big.wav", b"x" * (threshold + 1024), "audio/wav")},
        data=_sd(submission_id=uuid.uuid4().hex),
        follow_redirects=False,
    )
    assert resp.status_code == 413
    with session_factory() as session:
        assert session.execute(select(PipelineRun)).first() is None


def test_concurrent_conflicting_uploads_never_corrupt_disk(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # Regression for the dual-review's live-verified CRITICAL: two concurrent
    # uploads sharing a submission_id but carrying DIFFERENT bytes must never
    # leave the committed MediaItem's sha256/size describing a file the other
    # request overwrote. The file publish is gated on winning the source_path
    # UNIQUE insert, so the loser 409s WITHOUT ever calling os.replace; the
    # winner's row and on-disk bytes agree.
    sub = uuid.uuid4().hex
    body_a = b"A" * 5000
    body_b = b"B" * 9000
    sha_a = hashlib.sha256(body_a).hexdigest()

    a_ready = threading.Event()
    a_may_commit = threading.Event()
    errors: dict[str, BaseException] = {}
    outcomes: dict[str, object] = {}

    def worker_a() -> None:
        session = session_factory()
        try:
            run = submit_upload(
                session,
                stream=io.BytesIO(body_a),
                filename="clip.wav",
                submission_id=sub,
                media_root=media_root,
                max_bytes=10**9,
            )
            outcomes["run_a"] = run.id
            a_ready.set()  # A now holds the source_path row lock (uncommitted)
            a_may_commit.wait(timeout=10)
            session.commit()
        except BaseException as exc:  # surfaced to the test body, never swallowed
            errors["a"] = exc
            a_ready.set()
        finally:
            session.close()

    def worker_b() -> None:
        if not a_ready.wait(timeout=10):
            return
        session = session_factory()
        try:
            submit_upload(
                session,
                stream=io.BytesIO(body_b),
                filename="clip.wav",
                submission_id=sub,
                media_root=media_root,
                max_bytes=10**9,
            )
            outcomes["b"] = "created"  # a loser must never reach here
            session.commit()
        except UploadConflictError:
            outcomes["b"] = "conflict"
            session.rollback()
        except BaseException as exc:  # surfaced to the test body, never swallowed
            errors["b"] = exc
            session.rollback()
        finally:
            session.close()

    ta = threading.Thread(target=worker_a, name="A")
    tb = threading.Thread(target=worker_b, name="B")
    ta.start()
    tb.start()
    assert a_ready.wait(timeout=10)
    # Let B reach its (blocked) insert before A commits; timing only decides which
    # losing interleaving occurs — the assertions below hold for either.
    time.sleep(0.5)
    a_may_commit.set()
    ta.join(timeout=20)
    tb.join(timeout=20)

    assert not errors, errors
    assert outcomes.get("b") == "conflict"

    landed = media_root / "incoming" / sub / "clip.wav"
    assert landed.read_bytes() == body_a  # winner's bytes intact on disk
    assert sorted(p.name for p in landed.parent.iterdir()) == ["clip.wav"]  # no temp
    with session_factory() as session:
        media = session.execute(select(MediaItem)).scalars().all()
        runs = session.execute(select(PipelineRun)).scalars().all()
    assert len(media) == 1 and len(runs) == 1
    assert media[0].sha256 == sha_a and media[0].size_bytes == len(body_a)


def test_stream_copy_enforces_cap_and_cleans_temp(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # Drive submit_upload directly so the cap is hit mid-stream (not by the
    # route's early Content-Length gate): the temp must be unlinked, no dest
    # written, and no MediaItem created.
    sub = uuid.uuid4().hex
    oversized = io.BytesIO(b"\x00" * 4096)
    with session_factory() as session:
        with pytest.raises(UploadTooLargeError):
            submit_upload(
                session,
                stream=oversized,
                filename="huge.wav",
                submission_id=sub,
                media_root=media_root,
                max_bytes=1024,
            )
        session.rollback()

    dest_dir = media_root / "incoming" / sub
    assert not (dest_dir / "huge.wav").exists()
    # No leftover ".upload-*.part" temp in the submission dir.
    assert list(dest_dir.iterdir()) == []
    with session_factory() as session:
        assert session.execute(select(MediaItem)).first() is None


# --- filename policy ----------------------------------------------------------


@pytest.mark.parametrize("bad", ["../evil.wav", "sub/dir.wav", "..", ".", "..\\evil.wav"])
def test_traversal_filenames_rejected(
    client: TestClient,
    session_factory: sessionmaker[Session],
    published: list[uuid.UUID],
    bad: str,
) -> None:
    resp = client.post(
        "/submit",
        files={"file": (bad, wav_bytes(), "audio/wav")},
        data=_sd(submission_id=uuid.uuid4().hex),
        follow_redirects=False,
    )
    assert resp.status_code == 422
    assert published == []
    with session_factory() as session:
        assert session.execute(select(MediaItem)).first() is None


def test_non_uuid_submission_id_rejected(
    client: TestClient, published: list[uuid.UUID]
) -> None:
    resp = client.post(
        "/submit",
        files={"file": ("ok.wav", wav_bytes(), "audio/wav")},
        data=_sd(submission_id="not-a-uuid"),
        follow_redirects=False,
    )
    assert resp.status_code == 422
    assert published == []


# --- replay idempotency -------------------------------------------------------


def test_replay_same_bytes_returns_same_run(
    client: TestClient,
    session_factory: sessionmaker[Session],
    published: list[uuid.UUID],
) -> None:
    body = wav_bytes()
    sub = uuid.uuid4().hex
    files = {"file": ("ep.wav", body, "audio/wav")}
    first = client.post(
        "/submit", files=files, data=_sd(submission_id=sub), follow_redirects=False
    )
    second = client.post(
        "/submit",
        files={"file": ("ep.wav", body, "audio/wav")},
        data=_sd(submission_id=sub),
        follow_redirects=False,
    )
    assert first.status_code == second.status_code == 303
    run_id = _run_id_from_redirect(first.headers["location"])
    assert _run_id_from_redirect(second.headers["location"]) == run_id

    with session_factory() as session:
        runs = session.execute(select(PipelineRun)).scalars().all()
        media = session.execute(select(MediaItem)).scalars().all()
    assert len(runs) == 1  # no duplicate run for the replayed submission
    assert len(media) == 1
    assert published == [run_id, run_id]  # at-least-once: idempotent worker dedups


def test_replay_different_bytes_conflicts(
    client: TestClient,
    session_factory: sessionmaker[Session],
    media_root: Path,
    published: list[uuid.UUID],
) -> None:
    sub = uuid.uuid4().hex
    first = client.post(
        "/submit",
        files={"file": ("ep.wav", wav_bytes(seconds=0.02), "audio/wav")},
        data=_sd(submission_id=sub),
        follow_redirects=False,
    )
    assert first.status_code == 303
    run_id = _run_id_from_redirect(first.headers["location"])

    clash = client.post(
        "/submit",
        files={"file": ("ep.wav", wav_bytes(seconds=0.05), "audio/wav")},
        data=_sd(submission_id=sub),
        follow_redirects=False,
    )
    assert clash.status_code == 409

    landed = media_root / "incoming" / sub / "ep.wav"
    assert landed.read_bytes() == wav_bytes(seconds=0.02)  # original untouched
    with session_factory() as session:
        assert len(session.execute(select(PipelineRun)).scalars().all()) == 1
    assert published == [run_id]  # the conflicting POST never publishes
