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


# ---------------------------- endpoint env-fallback (issue #46) ----------------
# The base_url/model inputs must render the ROW OVERRIDE (blank when inheriting
# env, env default as placeholder), and a save must store NULL — not pin the env
# value — when the submitted value is blank or merely equals the env default.

ENV_BASE = "https://env.example/v1"
ENV_MODEL = "env-model"


def test_settings_untouched_endpoint_stays_null(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # Saving the LLM form without touching base_url/model (blank fields, as the
    # forms now render for an inheriting row) leaves both columns NULL.
    client = make_client(
        session_factory, media_root, onboarded=True,
        llm_api_key=ENV_SENTINEL, llm_base_url=ENV_BASE, llm_model=ENV_MODEL,
    )
    resp = client.post(
        "/settings/llm", data=_settings_form(enabled="true"), follow_redirects=False
    )
    assert resp.status_code == 303
    row = _row(session_factory)
    assert row is not None
    assert row.llm_base_url is None
    assert row.llm_model is None


def test_settings_endpoint_equal_to_env_default_stores_null(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # An operator who re-submits the env default (e.g. copied from the placeholder)
    # keeps inheriting: the column stays NULL rather than pinning the env value.
    client = make_client(
        session_factory, media_root, onboarded=True,
        llm_api_key=ENV_SENTINEL, llm_base_url=ENV_BASE, llm_model=ENV_MODEL,
    )
    resp = client.post(
        "/settings/llm",
        data=_settings_form(enabled="true", llm_base_url=ENV_BASE, llm_model=ENV_MODEL),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    row = _row(session_factory)
    assert row is not None
    assert row.llm_base_url is None
    assert row.llm_model is None


def test_settings_endpoint_override_stores_value(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # A genuinely different value is a deliberate override and is stored verbatim.
    client = make_client(
        session_factory, media_root, onboarded=True,
        llm_api_key=ENV_SENTINEL, llm_base_url=ENV_BASE, llm_model=ENV_MODEL,
    )
    resp = client.post(
        "/settings/llm",
        data=_settings_form(
            enabled="true",
            llm_base_url="https://row.example/v1",
            llm_model="row-model",
        ),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    row = _row(session_factory)
    assert row is not None
    assert row.llm_base_url == "https://row.example/v1"
    assert row.llm_model == "row-model"


def test_settings_endpoint_inputs_blank_when_inheriting_env(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # NULL row endpoint + env set → the input value is NOT the env value (that was
    # the bug: prefilling the effective value pins it on the next save); the env
    # default is offered as the placeholder instead.
    client = make_client(
        session_factory, media_root, onboarded=True,
        llm_api_key=ENV_SENTINEL, llm_base_url=ENV_BASE, llm_model=ENV_MODEL,
    )
    _seed_stored_key(session_factory, STORED_SENTINEL)  # row exists, endpoint NULL
    body = client.get("/settings").text
    assert f'value="{ENV_BASE}"' not in body  # env value never pinned into the input
    assert f'value="{ENV_MODEL}"' not in body
    assert f'placeholder="{ENV_BASE}"' in body  # shown as the default hint
    assert f'placeholder="{ENV_MODEL}"' in body


def test_settings_endpoint_inputs_show_override(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client = make_client(
        session_factory, media_root, onboarded=True,
        llm_api_key=ENV_SENTINEL, llm_base_url=ENV_BASE, llm_model=ENV_MODEL,
    )
    with session_factory() as session:
        row = get_or_create(session, llm_enabled_default=False)
        row.llm_base_url = "https://row.example/v1"
        row.llm_model = "row-model"
        session.commit()
    body = client.get("/settings").text
    assert 'value="https://row.example/v1"' in body
    assert 'value="row-model"' in body


def test_setup_endpoint_equal_to_env_default_stores_null(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # Same tri-state on the wizard's LLM step.
    client = make_client(
        session_factory, media_root,
        llm_api_key=ENV_SENTINEL, llm_base_url=ENV_BASE, llm_model=ENV_MODEL,
    )
    resp = client.post(
        "/setup/llm",
        data=_setup_form(enabled="true", llm_base_url=ENV_BASE, llm_model=ENV_MODEL),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    row = _row(session_factory)
    assert row is not None
    assert row.llm_base_url is None
    assert row.llm_model is None


def test_setup_endpoint_inputs_blank_when_inheriting_env(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client = make_client(
        session_factory, media_root,
        llm_api_key=ENV_SENTINEL, llm_base_url=ENV_BASE, llm_model=ENV_MODEL,
    )
    body = client.get("/setup?step=llm").text
    assert f'value="{ENV_BASE}"' not in body
    assert f'placeholder="{ENV_BASE}"' in body


def test_setup_validation_rerender_preserves_blank_endpoints(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # A blank-endpoint submission that fails the enable guard (no key anywhere)
    # re-renders with the inputs still BLANK — the env default is the placeholder,
    # never echoed into `value` (which would falsely show inherited state as
    # pinned). The DB row endpoint columns stay NULL.
    client = make_client(
        session_factory, media_root,
        llm_api_key="", llm_base_url=ENV_BASE, llm_model=ENV_MODEL,
    )
    resp = client.post("/setup/llm", data=_setup_form(enabled="true"))
    assert resp.status_code == 200  # enable failed → re-render
    body = resp.text
    assert f'value="{ENV_BASE}"' not in body
    assert f'value="{ENV_MODEL}"' not in body
    assert f'placeholder="{ENV_BASE}"' in body
    assert f'placeholder="{ENV_MODEL}"' in body
    row = _row(session_factory)
    assert row is not None
    assert row.llm_base_url is None and row.llm_model is None
