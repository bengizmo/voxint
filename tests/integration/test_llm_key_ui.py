"""In-UI LLM API key (issue #10) — the setup + settings routes, end to end.

Covers the candidate-state → validate → ONE mutation contract of ``POST
/setup/llm`` and ``POST /settings/llm``: store a key, replace it, leave it
untouched on a blank submission, remove it (revert to env), fail closed on an
un-enablable key/budget while preserving the valid candidate key, reject the
contradictory remove+replacement combination, and never echo the key back into
any rendered HTML. CSRF and the 303 redirect on the settings POST are asserted
here too. Runs against real Postgres (the mutation-commit semantics the atomic
route logic exists for only reproduce with a real session).
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.api.csrf import CSRF_SETTINGS, CSRF_SETUP, mint_csrf_token
from voxint.app_settings import get_app_settings, get_or_create
from voxint.config import Settings
from voxint.db.models import AppSettings

CREDS = ("reviewer", "s3cret")
_CSRF_KEY = "llm-key-ui-test-csrf-key"

STORED_SENTINEL = "sk-STORED-do-not-render-123"
ENV_SENTINEL = "sk-ENV-fallback-456"


@pytest.fixture()
def media_root(tmp_path: Path) -> Path:
    return tmp_path


def make_client(
    session_factory: sessionmaker[Session],
    media_root: Path,
    *,
    onboarded: bool = False,
    **overrides: object,
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
    if onboarded:
        seed_onboarded(session_factory)
    return client


def _setup_form(**fields: str) -> dict[str, str]:
    return {"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_SETUP), **fields}


def _settings_form(**fields: str) -> dict[str, str]:
    return {"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_SETTINGS), **fields}


def _row(session_factory: sessionmaker[Session]) -> AppSettings | None:
    with session_factory() as session:
        return get_app_settings(session)


def _seed_stored_key(session_factory: sessionmaker[Session], key: str) -> None:
    with session_factory() as session:
        row = get_or_create(session, llm_enabled_default=False)
        row.llm_api_key = key
        session.commit()


# ------------------------------------------------------------------- /setup/llm


def test_setup_store_key_persists_and_advances(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client = make_client(session_factory, media_root, llm_api_key="")  # no env key
    resp = client.post(
        "/setup/llm",
        data=_setup_form(enabled="true", llm_api_key=STORED_SENTINEL),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/setup?step=services"
    row = _row(session_factory)
    assert row is not None
    assert row.llm_api_key == STORED_SENTINEL
    assert row.llm_enabled is True


def test_setup_blank_key_leaves_stored_key_untouched(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    _seed_stored_key(session_factory, STORED_SENTINEL)
    client = make_client(session_factory, media_root, llm_api_key="")
    # Blank key field + enable: the stored key stays, the toggle applies.
    resp = client.post(
        "/setup/llm", data=_setup_form(enabled="true"), follow_redirects=False
    )
    assert resp.status_code == 303
    row = _row(session_factory)
    assert row is not None
    assert row.llm_api_key == STORED_SENTINEL  # untouched
    assert row.llm_enabled is True


def test_setup_remove_key_reverts_to_env(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    _seed_stored_key(session_factory, STORED_SENTINEL)
    client = make_client(session_factory, media_root, llm_api_key=ENV_SENTINEL)
    resp = client.post(
        "/setup/llm",
        data=_setup_form(enabled="true", remove_llm_api_key="true"),
        follow_redirects=False,
    )
    assert resp.status_code == 303  # env key still lets enable succeed
    row = _row(session_factory)
    assert row is not None
    assert row.llm_api_key is None  # reverted to env fallback
    assert row.llm_enabled is True


def test_setup_remove_plus_replacement_is_rejected(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    _seed_stored_key(session_factory, STORED_SENTINEL)
    client = make_client(session_factory, media_root, llm_api_key="")
    resp = client.post(
        "/setup/llm",
        data=_setup_form(
            enabled="true", llm_api_key="sk-new-value", remove_llm_api_key="true"
        ),
    )
    assert resp.status_code == 200  # contradictory → re-render, nothing changed
    row = _row(session_factory)
    assert row is not None
    assert row.llm_api_key == STORED_SENTINEL  # prior key preserved


def test_setup_invalid_key_replacement_preserves_prior_and_fails_closed(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    _seed_stored_key(session_factory, STORED_SENTINEL)
    client = make_client(session_factory, media_root, llm_api_key="")
    # A key with an inner space is a format error: nothing mutates, prior key stays.
    resp = client.post(
        "/setup/llm", data=_setup_form(enabled="true", llm_api_key="sk bad key")
    )
    assert resp.status_code == 200
    row = _row(session_factory)
    assert row is not None
    assert row.llm_api_key == STORED_SENTINEL
    assert row.llm_enabled is False  # enable did not take


def test_setup_enable_without_any_key_persists_valid_candidate_disabled(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # No env key, no stored key, enable requested → fail closed (disabled) but the
    # (valid, empty) candidate persists; message shown.
    client = make_client(session_factory, media_root, llm_api_key="")
    resp = client.post("/setup/llm", data=_setup_form(enabled="true"))
    assert resp.status_code == 200
    assert "No LLM API key" in resp.text or "No API key" in resp.text
    row = _row(session_factory)
    assert row is not None and row.llm_enabled is False


def test_setup_html_never_echoes_stored_key(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    _seed_stored_key(session_factory, STORED_SENTINEL)
    client = make_client(session_factory, media_root, llm_api_key=ENV_SENTINEL)
    body = client.get("/setup?step=llm").text
    assert STORED_SENTINEL not in body
    assert ENV_SENTINEL not in body
    # But the honest status is shown.
    assert "stored" in body


# ---------------------------------------------------------------- /settings/llm


def test_settings_store_key_redirects_303(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client = make_client(session_factory, media_root, onboarded=True, llm_api_key="")
    resp = client.post(
        "/settings/llm",
        data=_settings_form(enabled="true", llm_api_key=STORED_SENTINEL),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/settings"
    row = _row(session_factory)
    assert row is not None and row.llm_api_key == STORED_SENTINEL and row.llm_enabled


def test_settings_remove_key_reverts_to_env(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client = make_client(session_factory, media_root, onboarded=True, llm_api_key=ENV_SENTINEL)
    _seed_stored_key(session_factory, STORED_SENTINEL)
    resp = client.post(
        "/settings/llm",
        data=_settings_form(enabled="true", remove_llm_api_key="true"),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    row = _row(session_factory)
    assert row is not None and row.llm_api_key is None


def test_settings_invalid_replacement_preserves_prior_and_disables(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client = make_client(session_factory, media_root, onboarded=True, llm_api_key="")
    _seed_stored_key(session_factory, STORED_SENTINEL)
    resp = client.post(
        "/settings/llm", data=_settings_form(enabled="true", llm_api_key="sk bad key")
    )
    assert resp.status_code == 200  # re-render, nothing changed
    row = _row(session_factory)
    assert row is not None and row.llm_api_key == STORED_SENTINEL


def test_settings_llm_requires_csrf(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client = make_client(session_factory, media_root, onboarded=True, llm_api_key="")
    _seed_stored_key(session_factory, STORED_SENTINEL)
    resp = client.post(
        "/settings/llm",
        data={"enabled": "true", "llm_api_key": "sk-new"},  # no csrf_token
    )
    assert resp.status_code == 403
    row = _row(session_factory)
    assert row is not None and row.llm_api_key == STORED_SENTINEL  # unchanged


def test_settings_html_never_echoes_stored_key(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client = make_client(session_factory, media_root, onboarded=True, llm_api_key="")
    _seed_stored_key(session_factory, STORED_SENTINEL)
    body = client.get("/settings").text
    assert STORED_SENTINEL not in body
    assert "stored" in body
