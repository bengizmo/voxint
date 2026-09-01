"""Settings → Glossary section (issue #123): console glossary editing, end to end.

Covers ``POST /settings/glossary`` against real Postgres. The glossary edits the
same ``app_settings.vocabulary`` the setup wizard writes, through the SAME
``normalize_vocabulary`` gate (one term per line, deduped, 500-term / 120-char
bounds), replace-all: a valid list persists and normalizes; an over-cap term is
refused server-side with a plain-language message, writing NOTHING and keeping the
operator's submitted text; CSRF is required; a fresh deployment starts empty.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.api.csrf import CSRF_SETTINGS, mint_csrf_token
from voxint.api.setup_wizard import MAX_VOCABULARY_TERM_CHARS, MAX_VOCABULARY_TERMS
from voxint.app_settings import get_app_settings, get_or_create
from voxint.config import Settings
from voxint.db.models import AppSettings

CREDS = ("reviewer", "s3cret")
_CSRF_KEY = "settings-glossary-test-csrf-key"


@pytest.fixture()
def media_root(tmp_path: Path) -> Path:
    return tmp_path


def make_client(
    session_factory: sessionmaker[Session],
    media_root: Path,
    *,
    onboarded: bool = True,
) -> TestClient:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        csrf_secret=_CSRF_KEY,
        media_root=media_root,
    )
    client = TestClient(create_app(settings=settings, session_factory=session_factory))
    client.auth = CREDS
    if onboarded:
        seed_onboarded(session_factory)
    return client


def _form(vocabulary: str) -> dict[str, str]:
    return {
        "csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_SETTINGS),
        "vocabulary": vocabulary,
    }


def _row(session_factory: sessionmaker[Session]) -> AppSettings | None:
    with session_factory() as session:
        return get_app_settings(session)


def test_section_renders_with_form(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client = make_client(session_factory, media_root)
    resp = client.get("/settings/ai")
    assert resp.status_code == 200
    body = resp.text
    assert 'id="glossary"' in body
    assert "/settings/ai" in body
    # Fresh deployment: no terms yet, and the count reads honestly.
    assert "0 terms." in body


def test_section_prefills_stored_terms(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client = make_client(session_factory, media_root)
    with session_factory() as session:
        get_or_create(session, llm_enabled_default=False).vocabulary = ["Zoning Board", "NUCA"]
        session.commit()
    body = client.get("/settings/ai").text
    # Both terms render in the textarea, one per line, and the count is exact.
    assert "Zoning Board" in body
    assert "NUCA" in body
    assert "2 terms." in body


def test_post_valid_terms_persists_normalized(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client = make_client(session_factory, media_root)
    # Blank lines and a duplicate collapse through normalize_vocabulary.
    resp = client.post(
        "/settings/glossary",
        data=_form("Zoning Board\n\nNUCA\nZoning Board\n"),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/settings/ai#glossary"
    row = _row(session_factory)
    assert row is not None and row.vocabulary == ["Zoning Board", "NUCA"]


def test_post_replaces_the_whole_list(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client = make_client(session_factory, media_root)
    with session_factory() as session:
        get_or_create(session, llm_enabled_default=False).vocabulary = ["Old", "Stale"]
        session.commit()
    resp = client.post("/settings/glossary", data=_form("Fresh"), follow_redirects=False)
    assert resp.status_code == 303
    row = _row(session_factory)
    assert row is not None and row.vocabulary == ["Fresh"]


def test_post_empty_clears_the_glossary(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client = make_client(session_factory, media_root)
    with session_factory() as session:
        get_or_create(session, llm_enabled_default=False).vocabulary = ["Remove", "Me"]
        session.commit()
    resp = client.post("/settings/glossary", data=_form("   \n  \n"), follow_redirects=False)
    assert resp.status_code == 303
    row = _row(session_factory)
    assert row is not None and row.vocabulary == []


def test_post_over_long_term_rejected_and_writes_nothing(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client = make_client(session_factory, media_root)
    with session_factory() as session:
        get_or_create(session, llm_enabled_default=False).vocabulary = ["Keep"]
        session.commit()
    too_long = "x" * (MAX_VOCABULARY_TERM_CHARS + 1)
    resp = client.post("/settings/glossary", data=_form(too_long))
    assert resp.status_code == 422
    # The page re-renders with a plain-language error AND the submitted text, so the
    # operator does not lose their edit.
    assert f"{MAX_VOCABULARY_TERM_CHARS} characters" in resp.text
    assert too_long in resp.text
    # Nothing persisted: the prior term survives untouched.
    row = _row(session_factory)
    assert row is not None and row.vocabulary == ["Keep"]


def test_post_over_count_rejected(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client = make_client(session_factory, media_root)
    payload = "\n".join(f"term{i}" for i in range(MAX_VOCABULARY_TERMS + 1))
    resp = client.post("/settings/glossary", data=_form(payload))
    assert resp.status_code == 422
    assert f"at most {MAX_VOCABULARY_TERMS}" in resp.text
    row = _row(session_factory)
    assert row is None or not row.vocabulary


def test_rejected_save_count_tracks_submitted_text_not_stored(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    """On a rejected save the term count describes the operator's submitted lines, not
    the last stored list, so the "(N terms.)" caption never contradicts the textarea it
    sits under (issue #123). Stored one term, submit 501, see the honest 501."""
    client = make_client(session_factory, media_root)
    with session_factory() as session:
        get_or_create(session, llm_enabled_default=False).vocabulary = ["Keep"]
        session.commit()
    payload = "\n".join(f"term{i}" for i in range(MAX_VOCABULARY_TERMS + 1))
    resp = client.post("/settings/glossary", data=_form(payload))
    assert resp.status_code == 422
    # The count reflects the submitted lines (501), not the single stored term.
    assert f"{MAX_VOCABULARY_TERMS + 1} terms." in resp.text
    # And the stored list is untouched.
    row = _row(session_factory)
    assert row is not None and row.vocabulary == ["Keep"]


def test_post_requires_csrf(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client = make_client(session_factory, media_root)
    resp = client.post("/settings/glossary", data={"vocabulary": "Zoning Board"})
    assert resp.status_code == 403
    row = _row(session_factory)
    assert row is None or not row.vocabulary
