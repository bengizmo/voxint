"""The first-run setup wizard (issue #3, slice 4), end to end against real Postgres.

Covers the six wizard steps' rendering and POST persistence, CSRF on every mutation,
the two deferred-review guards at the LLM step (env-seeded ``llm_enabled`` and the
fail-closed budget/key checks), the bounded media scan (net-new only, reserved-tree
and symlink exclusion, caps, missing root), and that Finish flips onboarding so the
gate then lets the protected console through. The wizard runs BEFORE onboarding, so
these clients are deliberately NOT seeded onboarded.
"""

import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.api.csrf import CSRF_CLAIM, CSRF_SETUP, mint_csrf_token
from voxint.api.setup_wizard import scan_media_folders
from voxint.app_settings import get_app_settings
from voxint.config import Settings
from voxint.db.models import AppSettings, MediaItem, PipelineRun, RunStatus
from voxint.media.registration import registered_folder_paths

CREDS = ("reviewer", "s3cret")
_CSRF_KEY = "setup-wizard-test-csrf-key"  # known secret so tests can mint tokens
_HTMX = {"HX-Request": "true"}


@pytest.fixture()
def media_root(tmp_path: Path) -> Path:
    return tmp_path


def make_client(
    session_factory: sessionmaker[Session], media_root: Path, **overrides: object
) -> TestClient:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        csrf_secret=_CSRF_KEY,
        media_root=media_root,
        **overrides,
    )
    client = TestClient(create_app(settings=settings, session_factory=session_factory))
    client.auth = CREDS
    return client


@pytest.fixture()
def client(
    session_factory: sessionmaker[Session], media_root: Path
) -> TestClient:
    return make_client(session_factory, media_root)


@pytest.fixture()
def published(monkeypatch: pytest.MonkeyPatch) -> list[uuid.UUID]:
    """Capture commit-before-publish enqueues without a live broker."""
    calls: list[uuid.UUID] = []
    monkeypatch.setattr(
        "voxint.api.routers.deps._publish_run", lambda run_id, **_kwargs: calls.append(run_id)
    )
    return calls


def _form(**fields: str) -> dict[str, str]:
    return {"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_SETUP), **fields}


def _register_folder(client: TestClient, folder: str) -> None:
    """Register a media folder through the issue #63 browser route (replaces the
    old bulk /setup/media textarea)."""
    resp = client.post(
        "/setup/folders",
        data=_form(action="add", folder=folder, path="."),
        headers=_HTMX,
    )
    assert resp.status_code == 200


def _row(session_factory: sessionmaker[Session]) -> AppSettings | None:
    with session_factory() as session:
        return get_app_settings(session)


# --------------------------------------------------------------- GET rendering


def test_get_setup_defaults_to_welcome(client: TestClient) -> None:
    body = client.get("/setup").text
    assert "Welcome to Voxint" in body
    # The protected top nav is replaced by the wizard step indicator.
    assert 'class="wizard-steps"' in body
    assert 'href="/review"' not in body
    # #93: the step's forward CTA is the one teal accent (reserve teal for wizard
    # forward motion; in-step Save/Scan and Re-check stay neutral).
    assert 'class="btn-primary">Get started →</a>' in body


def test_get_setup_unknown_step_falls_back_to_welcome(client: TestClient) -> None:
    resp = client.get("/setup?step=bogus")
    assert resp.status_code == 200
    assert "Welcome to Voxint" in resp.text


def test_get_setup_media_step_renders_folder_panel(client: TestClient) -> None:
    body = client.get("/setup?step=media").text
    # The bulk textarea is gone (issue #63); the folder browser panel renders instead.
    assert 'name="media_folders"' not in body
    assert 'id="folder-panel"' in body
    assert 'action="/setup/folders"' in body


def test_setup_folders_browse_is_readonly_and_creates_no_row(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    # On the pre-onboarding wizard the app_settings row does not exist yet; a
    # browse GET must render the panel without creating it (issue #63).
    resp = client.get("/setup/folders/browse")
    assert resp.status_code == 200
    assert 'id="folder-panel"' in resp.text
    assert _row(session_factory) is None


def test_setup_requires_auth(client: TestClient) -> None:
    resp = client.get("/setup", auth=None)
    assert resp.status_code == 401


def test_setup_is_reachable_before_onboarding(client: TestClient) -> None:
    # Sanity: the gate does not bounce /setup (it is the redirect destination).
    assert client.get("/setup", follow_redirects=False).status_code == 200


# ------------------------------------------------------------------ media step


def test_add_folder_persists(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    (media_root / "podcasts").mkdir()
    resp = client.post(
        "/setup/folders",
        data=_form(action="add", folder="podcasts", path="."),
        follow_redirects=False,
    )
    # Plain (non-HX) POST degrades to a full-page redirect that keeps the location.
    assert resp.status_code == 303
    assert resp.headers["location"].startswith("/setup?")
    assert "step=media" in resp.headers["location"]
    with session_factory() as session:
        assert registered_folder_paths(session) == ["podcasts"]


def test_add_folder_without_csrf_is_403(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    (media_root / "podcasts").mkdir()
    resp = client.post(
        "/setup/folders",
        data={"action": "add", "folder": "podcasts", "path": "."},
        follow_redirects=False,
    )
    assert resp.status_code == 403
    assert _row(session_factory) is None  # nothing written


def test_add_folder_bad_path_rerenders_without_writing(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    resp = client.post(
        "/setup/folders",
        data=_form(action="add", folder="/etc", path="."),
        headers=_HTMX,
    )
    assert resp.status_code == 200
    assert "media folder" in resp.text  # visible validation error
    # The failed add rolled back — the get_or_create insert never committed.
    assert _row(session_factory) is None


def test_add_folder_does_not_disable_env_enabled_llm(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # Since #153 registering a folder writes only the media_folders relation and no
    # longer creates the app_settings row, so the wizard touching folders cannot
    # switch an env-enabled LLM off: the row is seeded from env (settings.llm_enabled)
    # whenever a genuine app_settings write first creates it.
    (media_root / "m").mkdir()
    client = make_client(
        session_factory, media_root, llm_enabled=True, llm_api_key="sk-test"
    )
    _register_folder(client, "m")
    with session_factory() as session:
        assert registered_folder_paths(session) == ["m"]  # the folder registered
    assert _row(session_factory) is None  # app_settings untouched by a folder add


# -------------------------------------------------------------- vocabulary step


def test_post_vocabulary_persists_and_advances(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    resp = client.post(
        "/setup/vocabulary",
        data=_form(vocabulary="NUCA\nDuctless mini-split\nNUCA\n"),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/setup?step=llm"
    row = _row(session_factory)
    assert row is not None and row.vocabulary == ["NUCA", "Ductless mini-split"]


def test_post_vocabulary_overlong_term_rerenders(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    resp = client.post("/setup/vocabulary", data=_form(vocabulary="x" * 500))
    assert resp.status_code == 200
    assert "vocabulary term" in resp.text
    assert _row(session_factory) is None


# ------------------------------------------------------------------- llm step


def test_post_llm_enable_with_key_and_budget(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client = make_client(session_factory, media_root, llm_api_key="sk-test")
    resp = client.post(
        "/setup/llm",
        data=_form(enabled="true", llm_base_url="http://localhost:9000/v1", llm_model="m"),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/setup?step=services"
    row = _row(session_factory)
    assert row is not None
    assert row.llm_enabled is True
    assert row.llm_base_url == "http://localhost:9000/v1"
    assert row.llm_model == "m"


def test_post_llm_enable_without_key_fails_closed(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # No env key → enabling is refused AND the row is persisted disabled (fail
    # closed), never left in a can't-actually-run enabled state.
    client = make_client(session_factory, media_root, llm_api_key="")
    resp = client.post("/setup/llm", data=_form(enabled="true"))
    assert resp.status_code == 200
    assert "LLM_API_KEY" in resp.text
    row = _row(session_factory)
    assert row is not None and row.llm_enabled is False


def test_post_llm_enable_over_budget_fails_closed(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # Deferred finding 2: the wizard can enable the LLM with env llm_enabled=False,
    # so the env-time budget validator never ran — the wizard must re-check and
    # refuse (fail closed) when the run budget doesn't fit the stage lease.
    client = make_client(
        session_factory,
        media_root,
        llm_api_key="sk-test",
        llm_enabled=False,
        llm_run_budget_seconds=999999.0,
        stage_lease_seconds=21600,
    )
    resp = client.post("/setup/llm", data=_form(enabled="true"))
    assert resp.status_code == 200
    assert "lease" in resp.text
    row = _row(session_factory)
    assert row is not None and row.llm_enabled is False


def test_post_llm_disable_persists(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client = make_client(session_factory, media_root, llm_api_key="sk-test")
    resp = client.post("/setup/llm", data=_form(), follow_redirects=False)  # unchecked
    assert resp.status_code == 303
    row = _row(session_factory)
    assert row is not None and row.llm_enabled is False


def test_post_llm_bad_base_url_rerenders_without_persisting(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client = make_client(session_factory, media_root, llm_api_key="sk-test")
    resp = client.post("/setup/llm", data=_form(enabled="true", llm_base_url="ftp://x/v1"))
    assert resp.status_code == 200
    assert "base URL" in resp.text
    assert _row(session_factory) is None  # a format error changes nothing


# --------------------------------------------------------------- services step


def _services_down_client(
    session_factory: sessionmaker[Session], media_root: Path, **overrides: object
) -> TestClient:
    # Point the model services + redis at a definitely-closed port so each check
    # returns quickly (connection refused) and renders as failed — the wizard step
    # (issue #61) must render every dependency regardless. Postgres stays the real
    # test DB, so it reports ready.
    return make_client(
        session_factory,
        media_root,
        asr_url="http://127.0.0.1:1",
        diarizer_url="http://127.0.0.1:1",
        embedder_url="http://127.0.0.1:1",
        redis_url="redis://127.0.0.1:1/0",
        **overrides,
    )


def test_get_services_renders_doctor_checks(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client = _services_down_client(session_factory, media_root)
    body = client.get("/setup?step=services").text
    # Every dependency the doctor covers is surfaced by name.
    assert "transcription" in body
    assert "diarization" in body
    assert "speaker embedding" in body
    assert "postgres" in body
    assert "redis" in body
    # Real test DB is up → postgres ready; the closed-port deps → failed. Both pill
    # states must appear (never a false all-good with a required dep down).
    assert '<span class="pill ready">ready</span>' in body
    assert '<span class="pill failed">failed</span>' in body
    # Plain-language remediation for a down dependency, no stack trace.
    assert "Start the model services" in body


def test_services_step_has_no_hf_token_row(
    session_factory: sessionmaker[Session],
    media_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The diarization weights are vendored into the pyannote image, so the
    # wizard must not steer users toward a Hugging Face account — with or
    # without a token in the environment.
    monkeypatch.setenv("HF_TOKEN", "hf_SECRETVALUE")
    client = _services_down_client(session_factory, media_root)
    body = client.get("/setup?step=services").text
    assert "Hugging Face" not in body
    assert "hf_SECRETVALUE" not in body
    assert "hugging face" not in body.lower()
    # Truthful failure semantics replaced the old "simply waits" claim.
    assert "simply waits" not in body
    assert "requeue" in body


def test_services_step_llm_enabled_unreachable_is_unverified(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # LLM enhancement on but the endpoint unreachable → the row shows as UNVERIFIED
    # (advisory, best-effort), never ready and never a hard failure. The endpoint DSN
    # and the API key must never appear in the rendered page.
    client = _services_down_client(
        session_factory,
        media_root,
        llm_enabled=True,
        llm_base_url="http://127.0.0.1:1/v1",
        llm_api_key="sk-SECRETKEY",
    )
    body = client.get("/setup?step=services").text
    assert "llm endpoint" in body
    assert '<span class="pill unverified">unverified</span>' in body
    assert "sk-SECRETKEY" not in body
    assert "127.0.0.1:1" not in body  # no DSN / endpoint leaked


def test_services_step_llm_disabled_shows_no_llm_row(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # Default llm_enabled=False → check_llm returns None → no LLM row at all.
    client = _services_down_client(session_factory, media_root)
    body = client.get("/setup?step=services").text
    assert "llm endpoint" not in body


def test_services_step_returns_200_with_dependencies_down(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # The readiness checks must never raise into the request: with redis + models down
    # (and LLM enabled at a dead endpoint), the GET is still a clean 200, not a 500.
    client = _services_down_client(
        session_factory,
        media_root,
        llm_enabled=True,
        llm_base_url="http://127.0.0.1:1/v1",
        llm_api_key="sk-x",
    )
    resp = client.get("/setup?step=services")
    assert resp.status_code == 200


def test_services_step_renders_when_database_is_down(media_root: Path) -> None:
    # The wizard's whole point is to SHOW a down dependency. If Postgres itself is
    # unreachable, the SERVICES step must still render (200) with the failed postgres
    # row, not 500 because the page's own app_settings read raised. Bind the app to a
    # dead DB port (connection refused, fast).
    dead_factory = sessionmaker(
        create_engine("postgresql+psycopg://voxint:voxint@127.0.0.1:1/voxint_test")
    )
    client = _services_down_client(dead_factory, media_root)
    resp = client.get("/setup?step=services")
    assert resp.status_code == 200
    assert "postgres" in resp.text
    assert '<span class="pill failed">failed</span>' in resp.text


def test_services_step_llm_row_follows_row_over_env_on(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # Row wins over env (#74): env LLM off, but the app_settings row enables it →
    # the LLM readiness row appears. Guards against the doctor gate drifting from a run.
    seed_onboarded(session_factory, llm_enabled=True)
    client = _services_down_client(session_factory, media_root)  # env llm_enabled=False
    body = client.get("/setup?step=services").text
    assert "llm endpoint" in body


def test_services_step_llm_row_follows_row_over_env_off(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # Reverse: env LLM on, but the row disables it → no LLM row (the row wins).
    seed_onboarded(session_factory, llm_enabled=False)
    client = _services_down_client(session_factory, media_root, llm_enabled=True)
    body = client.get("/setup?step=services").text
    assert "llm endpoint" not in body


# ------------------------------------------------------------------ finish step


def test_get_setup_finish_step_primary_and_secondary_actions(client: TestClient) -> None:
    """#93: the finish step's forward CTA (start-tutorial) is the teal primary;
    the plain "Finish setup" alternative stays the neutral secondary."""
    body = client.get("/setup?step=finish").text
    assert "You're all set" in body
    assert 'name="start_tutorial" value="1" class="primary"' in body
    assert 'class="secondary">Finish setup →</button>' in body


def test_post_finish_flips_onboarding_and_opens_console(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    resp = client.post("/setup/finish", data=_form(), follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/review"
    row = _row(session_factory)
    assert row is not None and row.onboarding_complete is True
    # The gate now lets a protected route through.
    assert client.get("/review", follow_redirects=False).status_code == 200


def test_post_finish_without_csrf_is_403_and_not_onboarded(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    resp = client.post("/setup/finish", data={}, follow_redirects=False)
    assert resp.status_code == 403
    assert client.get("/review", follow_redirects=False).status_code == 303  # still gated


# --------------------------------------------------- scan (function-level walk)


def _write_media(root: Path, rel: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFFxxxx")


def test_scan_finds_net_new_media_only(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    _write_media(media_root, "pods/a.wav")
    _write_media(media_root, "pods/b.mp3")
    _write_media(media_root, "pods/notes.txt")  # non-media suffix → skipped
    settings = Settings(_env_file=None, media_root=media_root)  # type: ignore[call-arg]
    with session_factory() as session:
        # b.mp3 is already ingested → excluded as not net-new.
        session.add(MediaItem(source_path="pods/b.mp3"))
        session.commit()
        result = scan_media_folders(session, media_root, ["pods"], settings)
    assert result.candidates == ["pods/a.wav"]
    assert result.root_missing is False


def test_scan_excludes_reserved_trees(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    _write_media(media_root, "top.wav")
    _write_media(media_root, "incoming/u.wav")  # Voxint-owned upload tree
    _write_media(media_root, "artifacts/n.wav")  # Voxint-owned artifact tree
    settings = Settings(_env_file=None, media_root=media_root)  # type: ignore[call-arg]
    with session_factory() as session:
        result = scan_media_folders(session, media_root, ["."], settings)
    assert result.candidates == ["top.wav"]


def test_scan_skips_symlinks(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    _write_media(media_root, "real/a.wav")
    (media_root / "link").symlink_to(media_root / "real", target_is_directory=True)
    settings = Settings(_env_file=None, media_root=media_root)  # type: ignore[call-arg]
    with session_factory() as session:
        result = scan_media_folders(session, media_root, ["."], settings)
    # Only the real file, never the symlinked duplicate path.
    assert result.candidates == ["real/a.wav"]


def test_scan_respects_file_cap(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    for i in range(5):
        _write_media(media_root, f"m/f{i}.wav")
    settings = Settings(_env_file=None, media_root=media_root, setup_scan_max_files=3)  # type: ignore[call-arg]
    with session_factory() as session:
        result = scan_media_folders(session, media_root, ["m"], settings)
    assert len(result.candidates) == 3
    assert result.hit_file_cap is True


def test_scan_missing_root_is_flagged_not_raised(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    missing = tmp_path / "nope"
    settings = Settings(_env_file=None, media_root=missing)  # type: ignore[call-arg]
    with session_factory() as session:
        result = scan_media_folders(session, missing, ["."], settings)
    assert result.root_missing is True
    assert result.candidates == []


def test_scan_skips_a_folder_that_vanished(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # A stored folder can be removed between save and scan; the walk re-validates
    # each and simply skips a now-missing one rather than raising.
    _write_media(media_root, "real/a.wav")
    settings = Settings(_env_file=None, media_root=media_root)  # type: ignore[call-arg]
    with session_factory() as session:
        result = scan_media_folders(session, media_root, ["ghost", "real"], settings)
    assert result.candidates == ["real/a.wav"]


def test_scan_stops_at_entry_cap(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    for i in range(4):
        _write_media(media_root, f"m/f{i}.wav")
    settings = Settings(_env_file=None, media_root=media_root, setup_scan_max_entries=1)  # type: ignore[call-arg]
    with session_factory() as session:
        result = scan_media_folders(session, media_root, ["m"], settings)
    assert result.hit_entry_cap is True
    assert result.inspected <= 2  # stopped almost immediately


def test_scan_skips_symlinked_files(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    _write_media(media_root, "real/a.wav")
    (media_root / "real" / "dupe.wav").symlink_to(media_root / "real" / "a.wav")
    settings = Settings(_env_file=None, media_root=media_root)  # type: ignore[call-arg]
    with session_factory() as session:
        result = scan_media_folders(session, media_root, ["real"], settings)
    assert result.candidates == ["real/a.wav"]  # the symlinked file is skipped


def test_scan_skips_a_reserved_base(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # Defence-in-depth: even if a reserved tree ends up registered as a folder, the
    # scan base itself (not just its children) is excluded, so pipeline uploads are
    # never re-ingested.
    _write_media(media_root, "incoming/u.wav")
    settings = Settings(_env_file=None, media_root=media_root)  # type: ignore[call-arg]
    with session_factory() as session:
        result = scan_media_folders(session, media_root, ["incoming"], settings)
    assert result.candidates == []


def test_scan_file_cap_applies_to_net_new_not_known(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # Regression: the file cap must apply AFTER the existence filter, so an
    # already-ingested file can't fill the cap and hide genuinely new media.
    _write_media(media_root, "m/known.wav")
    _write_media(media_root, "m/fresh.wav")
    settings = Settings(_env_file=None, media_root=media_root, setup_scan_max_files=1)  # type: ignore[call-arg]
    with session_factory() as session:
        session.add(MediaItem(source_path="m/known.wav"))
        session.commit()
        result = scan_media_folders(session, media_root, ["m"], settings)
    assert result.candidates == ["m/fresh.wav"]  # net-new surfaces regardless of order
    assert result.hit_file_cap is False  # one net-new, under the cap


# --------------------------------------------------------- scan (route + confirm)


def test_scan_preview_then_confirm_queues_and_publishes(
    session_factory: sessionmaker[Session],
    media_root: Path,
    published: list[uuid.UUID],
) -> None:
    _write_media(media_root, "pods/a.wav")
    _write_media(media_root, "pods/b.wav")
    client = make_client(session_factory, media_root)
    # Register the folder first (the scan walks only registered folders).
    (media_root / "pods").mkdir(exist_ok=True)
    _register_folder(client, "pods")

    preview = client.post("/setup/scan", data=_form(), headers=_HTMX)
    assert preview.status_code == 200
    # The count is rendered inside <strong>; the confirm button label is unambiguous.
    assert "Queue these 2 for transcription" in preview.text
    assert "pods/a.wav" in preview.text and "pods/b.wav" in preview.text

    confirm = client.post("/setup/scan/confirm", data=_form(), headers=_HTMX)
    assert confirm.status_code == 200
    assert "Queued 2 runs" in confirm.text

    with session_factory() as session:
        runs = session.execute(select(PipelineRun)).scalars().all()
        media = session.execute(select(MediaItem)).scalars().all()
    assert len(runs) == 2 and len(media) == 2
    assert all(r.status == RunStatus.QUEUED.value for r in runs)
    assert sorted(m.source_path for m in media) == ["pods/a.wav", "pods/b.wav"]
    assert len(published) == 2  # commit-before-publish fired once per new run


def test_scan_confirm_is_idempotent(
    session_factory: sessionmaker[Session],
    media_root: Path,
    published: list[uuid.UUID],
) -> None:
    _write_media(media_root, "pods/a.wav")
    client = make_client(session_factory, media_root)
    (media_root / "pods").mkdir(exist_ok=True)
    _register_folder(client, "pods")

    first = client.post("/setup/scan/confirm", data=_form(), headers=_HTMX)
    assert "Queued 1 run" in first.text
    second = client.post("/setup/scan/confirm", data=_form(), headers=_HTMX)
    # The file is already ingested, so the second confirm queues nothing new.
    assert "Queued 0 runs" in second.text

    with session_factory() as session:
        assert len(session.execute(select(PipelineRun)).scalars().all()) == 1
        assert len(session.execute(select(MediaItem)).scalars().all()) == 1
    assert len(published) == 1  # no duplicate publish on the replayed confirm


def test_scan_without_csrf_is_403(client: TestClient) -> None:
    assert client.post("/setup/scan", data={}, headers=_HTMX).status_code == 403
    assert client.post("/setup/scan/confirm", data={}, headers=_HTMX).status_code == 403


def test_scan_wrong_action_token_is_403(client: TestClient) -> None:
    # A token minted for another action (claim) is not valid on the wizard routes.
    bad = {"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_CLAIM)}
    assert client.post("/setup/scan", data=bad, headers=_HTMX).status_code == 403
