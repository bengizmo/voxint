"""Settings → Corrections section (issue #84): console rule authoring, end to end.

Covers the ``POST /settings/corrections`` candidate → validate → ONE mutation
contract against real Postgres: a valid list persists (replace-all, like
vocabulary) and returns the canonical rules with generated ids; a bad rule is
refused server-side through the SAME #80 gate with a plain-language message and
the offending row, writing NOTHING; a collision with the default pack's own rules
is caught at author time; CSRF is required; and a fresh deployment starts empty.
"""

import json
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.api.csrf import CSRF_SETTINGS, mint_csrf_token
from voxint.app_settings import get_app_settings, get_or_create
from voxint.config import Settings
from voxint.db.models import AppSettings

CREDS = ("reviewer", "s3cret")
_CSRF_KEY = "settings-corrections-test-csrf-key"

_JSON = {"accept": "application/json"}


@pytest.fixture()
def media_root(tmp_path: Path) -> Path:
    return tmp_path


def make_client(
    session_factory: sessionmaker[Session],
    media_root: Path,
    *,
    onboarded: bool = True,
    **overrides: object,
) -> tuple[TestClient, Settings]:
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
    return client, settings


def _write_pack(root: Path, name: str, **fields: object) -> Path:
    pack_dir = root / name
    pack_dir.mkdir(parents=True)
    (pack_dir / "manifest.yaml").write_text(yaml.safe_dump({"name": name, **fields}))
    return pack_dir


def _form(rules: object) -> dict[str, str]:
    return {
        "csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_SETTINGS),
        "rules": json.dumps(rules),
    }


def _row(session_factory: sessionmaker[Session]) -> AppSettings | None:
    with session_factory() as session:
        return get_app_settings(session)


def test_section_renders_with_island_mount(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client, _ = make_client(session_factory, media_root)
    resp = client.get("/settings")
    assert resp.status_code == 200
    body = resp.text
    assert 'id="corrections"' in body
    assert 'data-island="corrections-editor"' in body
    assert "/settings/corrections" in body
    # Fresh deployment: no rules yet, honest empty state.
    assert "No correction rules yet." in body


def test_post_valid_rules_persists_and_returns_canonical(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client, _ = make_client(session_factory, media_root)
    resp = client.post(
        "/settings/corrections",
        data=_form([{"match": "zoom board", "replace": "Zoning Board"}]),
        headers=_JSON,
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["ok"] is True
    rule = payload["corrections"][0]
    # id auto-generated from the match; flags defaulted true.
    assert rule == {
        "id": "zoom-board",
        "match": "zoom board",
        "replace": "Zoning Board",
        "case_sensitive": True,
        "whole_word": True,
    }
    row = _row(session_factory)
    assert row is not None and row.corrections == payload["corrections"]


def test_post_without_json_accept_redirects(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client, _ = make_client(session_factory, media_root)
    resp = client.post(
        "/settings/corrections",
        data=_form([{"id": "teh", "match": "teh", "replace": "the"}]),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/settings"
    row = _row(session_factory)
    assert row is not None and row.corrections[0]["id"] == "teh"


def test_post_replaces_the_whole_list(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client, _ = make_client(session_factory, media_root)
    with session_factory() as session:
        row = get_or_create(session, llm_enabled_default=False)
        row.corrections = [
            {"id": "a", "match": "a", "replace": "A", "case_sensitive": True, "whole_word": True},
            {"id": "b", "match": "b", "replace": "B", "case_sensitive": True, "whole_word": True},
        ]
        session.commit()
    resp = client.post(
        "/settings/corrections",
        data=_form([{"id": "c", "match": "c", "replace": "C"}]),
        headers=_JSON,
    )
    assert resp.status_code == 200
    row = _row(session_factory)
    assert row is not None
    assert [rule["id"] for rule in row.corrections] == ["c"]


def test_post_bad_rule_rejected_with_row_and_writes_nothing(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client, _ = make_client(session_factory, media_root)
    resp = client.post(
        "/settings/corrections",
        data=_form(
            [
                {"match": "ok", "replace": "OK"},
                {"match": "  ", "replace": "x"},
            ]
        ),
        headers=_JSON,
    )
    assert resp.status_code == 422
    payload = resp.json()
    assert payload["ok"] is False
    assert payload["row"] == 1
    assert "domain pack" not in payload["error"]
    assert payload["error"].startswith("Rule 2 ")
    # Nothing persisted on a rejected save.
    row = _row(session_factory)
    assert row is None or not row.corrections


def test_post_non_idempotent_rejected(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client, _ = make_client(session_factory, media_root)
    resp = client.post(
        "/settings/corrections",
        data=_form([{"id": "d", "match": "dog", "replace": "dog food"}]),
        headers=_JSON,
    )
    assert resp.status_code == 422
    assert "idempotent" in resp.json()["error"]


def test_post_malformed_json_payload_rejected(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client, _ = make_client(session_factory, media_root)
    resp = client.post(
        "/settings/corrections",
        data={"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_SETTINGS), "rules": "{not json"},
        headers=_JSON,
    )
    assert resp.status_code == 422
    assert resp.json()["ok"] is False


def test_post_collision_with_default_pack_rejected(
    session_factory: sessionmaker[Session], media_root: Path, tmp_path: Path
) -> None:
    # A configured default pack that already declares a "Zoning Board" replacement;
    # an operator "Board" rule would re-fire on it, so authoring must refuse it now.
    pack_dir = _write_pack(
        tmp_path / "packs",
        "civic",
        corrections=[{"id": "zb", "match": "zoom board", "replace": "Zoning Board"}],
    )
    client, _ = make_client(session_factory, media_root, domain_pack_path=pack_dir)
    resp = client.post(
        "/settings/corrections",
        data=_form([{"id": "b", "match": "Board", "replace": "Committee"}]),
        headers=_JSON,
    )
    assert resp.status_code == 422
    assert "idempotent" in resp.json()["error"]


def test_post_requires_csrf(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client, _ = make_client(session_factory, media_root)
    resp = client.post(
        "/settings/corrections",
        data={"rules": json.dumps([{"match": "a", "replace": "b"}])},
        headers=_JSON,
    )
    assert resp.status_code == 403
    row = _row(session_factory)
    assert row is None or not row.corrections
