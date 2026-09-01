"""Settings → Translation section (issue #133), end to end.

The preferred-target-language select (a string override: "inherit" writes
NULL, a code writes the override) and the auto-translate tri-state save
together through ``POST /settings/translation`` with the candidate → validate
→ ONE mutation contract: a rejected save writes NOTHING and re-renders the
operator's choices with a plain message; "use installation setting" reverts an
override; CSRF is required; and both rendered controls round-trip through the
route (an undeclared Form param would be dropped by Starlette and read as
"inherit", silently clearing an override).
"""

from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.api.csrf import CSRF_SETTINGS, mint_csrf_token
from voxint.app_settings import get_app_settings, get_or_create
from voxint.config import Settings
from voxint.db.models import AppSettings

CREDS = ("reviewer", "s3cret")
_CSRF_KEY = "settings-translation-test-csrf-key"


def make_client(
    session_factory: sessionmaker[Session],
    tmp_path: Path,
    **overrides: object,
) -> tuple[TestClient, Settings]:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        csrf_secret=_CSRF_KEY,
        media_root=tmp_path,
        **overrides,
    )
    client = TestClient(create_app(settings=settings, session_factory=session_factory))
    client.auth = CREDS
    seed_onboarded(session_factory)
    return client, settings


def _form(**fields: str) -> dict[str, str]:
    return {"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_SETTINGS), **fields}


def _row(session_factory: sessionmaker[Session]) -> AppSettings | None:
    with session_factory() as session:
        return get_app_settings(session)


def test_translation_section_renders(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    client, _ = make_client(session_factory, tmp_path)
    body = client.get("/settings/ai").text
    assert 'id="translation"' in body
    assert 'name="translation_target_language"' in body
    # Unset columns render as "inherit" and name the (unset) env default.
    assert 'value="inherit" selected' in body
    assert "currently not set" in body
    # Autogenerate switch renders (inherited, no badge).
    assert 'id="sw-translation_autogenerate"' in body
    assert 'role="switch"' in body


def test_save_override_and_revert_to_inherit(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    client, _ = make_client(session_factory, tmp_path)
    response = client.post(
        "/settings/translation",
        data=_form(translation_target_language="es", translation_autogenerate="on"),
        follow_redirects=False,
    )
    assert response.status_code == 303
    row = _row(session_factory)
    assert row is not None
    assert row.translation_target_language == "es"
    assert row.translation_autogenerate is True

    # The saved state renders back selected/checked.
    body = client.get("/settings/ai").text
    assert 'value="es" selected' in body
    # The saved override renders the switch checked with a "Changed" badge.
    assert 'id="sw-translation_autogenerate"' in body
    assert ">Changed</span>" in body

    response = client.post(
        "/settings/translation",
        data=_form(
            translation_target_language="inherit", translation_autogenerate="inherit"
        ),
        follow_redirects=False,
    )
    assert response.status_code == 303
    row = _row(session_factory)
    assert row is not None
    assert row.translation_target_language is None
    assert row.translation_autogenerate is None


def test_autogenerate_without_target_refused_and_writes_nothing(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    client, _ = make_client(session_factory, tmp_path)
    response = client.post(
        "/settings/translation",
        data=_form(
            translation_target_language="inherit", translation_autogenerate="on"
        ),
    )
    assert response.status_code == 200
    assert "Pick a preferred language" in response.text
    # The operator's rejected choice is preserved: switch checked + badge.
    assert 'id="sw-translation_autogenerate"' in response.text
    assert ">Changed</span>" in response.text
    row = _row(session_factory)
    assert row is None or row.translation_autogenerate is None


def test_autogenerate_rides_on_env_target(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    # An env-configured target satisfies the invariant with the row inheriting.
    client, _ = make_client(session_factory, tmp_path, translation_target_language="fr")
    response = client.post(
        "/settings/translation",
        data=_form(
            translation_target_language="inherit", translation_autogenerate="on"
        ),
        follow_redirects=False,
    )
    assert response.status_code == 303
    row = _row(session_factory)
    assert row is not None
    assert row.translation_target_language is None
    assert row.translation_autogenerate is True


def test_unknown_code_and_bad_choice_refused(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    client, _ = make_client(session_factory, tmp_path)
    response = client.post(
        "/settings/translation",
        data=_form(
            translation_target_language="klingon", translation_autogenerate="off"
        ),
    )
    assert response.status_code == 200
    assert "Unrecognized language choice" in response.text
    response = client.post(
        "/settings/translation",
        data=_form(
            translation_target_language="es", translation_autogenerate="sideways"
        ),
    )
    assert response.status_code == 200
    assert "Unrecognized auto-translate setting" in response.text
    assert _row(session_factory) is None or (
        _row(session_factory).translation_target_language is None  # type: ignore[union-attr]
    )


def test_case_normalized_code_saves_lowercase(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    client, _ = make_client(session_factory, tmp_path)
    response = client.post(
        "/settings/translation",
        data=_form(translation_target_language="ES", translation_autogenerate="off"),
        follow_redirects=False,
    )
    assert response.status_code == 303
    row = _row(session_factory)
    assert row is not None and row.translation_target_language == "es"


def test_csrf_required(session_factory: sessionmaker[Session], tmp_path: Path) -> None:
    client, _ = make_client(session_factory, tmp_path)
    response = client.post(
        "/settings/translation", data={"translation_target_language": "es"}
    )
    assert response.status_code == 403
    row = _row(session_factory)
    assert row is None or row.translation_target_language is None


def test_existing_override_survives_unrelated_sections(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    # Saving the semantic section must not clear a translation override (the
    # per-section scoping contract).
    with session_factory() as session:
        row = get_or_create(session, llm_enabled_default=False)
        row.translation_target_language = "es"
        session.commit()
    client, _ = make_client(session_factory, tmp_path)
    response = client.post(
        "/settings/semantic",
        data=_form(
            semantic_index_enabled="inherit", semantic_index_autogenerate="inherit"
        ),
        follow_redirects=False,
    )
    assert response.status_code == 303
    row = _row(session_factory)
    assert row is not None and row.translation_target_language == "es"
