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
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from voxint.api.app import create_app
from voxint.api.csrf import CSRF_CLAIM, CSRF_SETUP, mint_csrf_token
from voxint.api.setup_wizard import scan_media_folders
from voxint.app_settings import get_app_settings
from voxint.config import Settings
from voxint.db.models import AppSettings, MediaItem, PipelineRun, RunStatus

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
    monkeypatch.setattr("voxint.api.app._publish_run", calls.append)
    return calls


def _form(**fields: str) -> dict[str, str]:
    return {"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_SETUP), **fields}


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


def test_get_setup_unknown_step_falls_back_to_welcome(client: TestClient) -> None:
    resp = client.get("/setup?step=bogus")
    assert resp.status_code == 200
    assert "Welcome to Voxint" in resp.text


def test_get_setup_media_step_renders_form(client: TestClient) -> None:
    body = client.get("/setup?step=media").text
    assert 'action="/setup/media"' in body
    assert 'name="media_folders"' in body


def test_setup_requires_auth(client: TestClient) -> None:
    resp = client.get("/setup", auth=None)
    assert resp.status_code == 401


def test_setup_is_reachable_before_onboarding(client: TestClient) -> None:
    # Sanity: the gate does not bounce /setup (it is the redirect destination).
    assert client.get("/setup", follow_redirects=False).status_code == 200


# ------------------------------------------------------------------ media step


def test_post_media_persists_folders(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    (media_root / "podcasts").mkdir()
    resp = client.post(
        "/setup/media", data=_form(media_folders="podcasts\n"), follow_redirects=False
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/setup?step=media"
    row = _row(session_factory)
    assert row is not None and row.media_folders == ["podcasts"]


def test_post_media_without_csrf_is_403(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    resp = client.post("/setup/media", data={"media_folders": "x"}, follow_redirects=False)
    assert resp.status_code == 403
    assert _row(session_factory) is None  # nothing written


def test_post_media_bad_folder_rerenders_without_writing(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    resp = client.post("/setup/media", data=_form(media_folders="/etc"))
    assert resp.status_code == 200
    assert "media folder" in resp.text  # visible validation error
    assert _row(session_factory) is None  # validation ran before any get_or_create


def test_post_media_does_not_disable_env_enabled_llm(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # Deferred finding 1: the FIRST row write (here, saving media folders) must seed
    # llm_enabled from env, not the model default False — otherwise an env-enabled
    # LLM would be silently switched off the moment the wizard touches the DB.
    (media_root / "m").mkdir()
    client = make_client(
        session_factory, media_root, llm_enabled=True, llm_api_key="sk-test"
    )
    client.post("/setup/media", data=_form(media_folders="m"))
    row = _row(session_factory)
    assert row is not None and row.llm_enabled is True


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


def test_get_services_renders_probe_results(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # Point the services at a definitely-closed port so each probe returns quickly
    # (connection refused) and renders as down — the step must render regardless.
    client = make_client(
        session_factory,
        media_root,
        asr_url="http://127.0.0.1:1",
        diarizer_url="http://127.0.0.1:1",
        embedder_url="http://127.0.0.1:1",
    )
    body = client.get("/setup?step=services").text
    assert "transcription" in body
    assert "diarization" in body
    assert "speaker embedding" in body


def test_services_step_has_no_hf_token_row(
    session_factory: sessionmaker[Session],
    media_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The diarization weights are vendored into the pyannote image, so the
    # wizard must not steer users toward a Hugging Face account — with or
    # without a token in the environment.
    monkeypatch.setenv("HF_TOKEN", "hf_SECRETVALUE")
    client = make_client(
        session_factory,
        media_root,
        asr_url="http://127.0.0.1:1",
        diarizer_url="http://127.0.0.1:1",
        embedder_url="http://127.0.0.1:1",
    )
    body = client.get("/setup?step=services").text
    assert "Hugging Face" not in body
    assert "hf_SECRETVALUE" not in body
    # Truthful failure semantics replaced the old "simply waits" claim.
    assert "simply waits" not in body
    assert "requeue" in body


# ------------------------------------------------------------------ finish step


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
    client.post("/setup/media", data=_form(media_folders="pods"))

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
    client.post("/setup/media", data=_form(media_folders="pods"))

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
