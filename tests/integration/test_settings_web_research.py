"""Settings → Sources & research section (issue #76): web-research config, end to end.

Covers the ``POST /settings/web-research`` candidate → validate → ONE mutation
contract against real Postgres: the master/producer/LLM/URL combinations as a
table-driven contract test (write-nothing on any violation, exactly one base-URL
message), the trusted-domain editor's strict accept/reject with the permissive
runtime parser still reaching triage live, the secret's blank-keep/remove/typed
semantics with a non-disclosure sentinel across GET + every error re-render, the
tri-state inherit/env-echo → NULL collapse, CSRF, and that a UI toggle reaches the
research capability gate with no restart.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.api.csrf import CSRF_SETTINGS, mint_csrf_token
from voxint.app_settings import (
    get_app_settings,
    get_or_create,
    resolve_effective_source_authority_domains,
)
from voxint.config import Settings
from voxint.db.models import AppSettings
from voxint.enrichment.research_jobs import research_gates_open
from voxint.enrichment.triage import (
    EvidenceRef,
    TriageInputs,
    parse_authority_domains,
    score_profile,
)

CREDS = ("reviewer", "s3cret")
_CSRF_KEY = "settings-web-research-test-csrf-key"
_VALID_URL = "https://searxng.example.org"


@pytest.fixture()
def media_root(tmp_path: Path) -> Path:
    return tmp_path


def make_client(
    session_factory: sessionmaker[Session],
    media_root: Path,
    *,
    seed_llm_enabled: bool = False,
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
    seed_onboarded(session_factory, llm_enabled=seed_llm_enabled)
    return client, settings


def _form(**fields: str) -> dict[str, str]:
    return {"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_SETTINGS), **fields}


def _row(session_factory: sessionmaker[Session]) -> AppSettings | None:
    with session_factory() as session:
        return get_app_settings(session)


def _seed_cols(session_factory: sessionmaker[Session], **columns: object) -> None:
    with session_factory() as session:
        row = get_or_create(session, llm_enabled_default=False)
        for name, value in columns.items():
            setattr(row, name, value)
        session.commit()


def _alert_count(body: str) -> int:
    return body.count('role="alert"')


# --- Table-driven master/producer/LLM/URL contract test (acceptance criteria) ----

# (id, seed_llm, form fields, expect_commit, message substring or None)
_CASES = [
    (
        "producer_needs_master",
        True,
        {"enrichment_web_research_enabled": "on", "voxint_web_research": "off"},
        False,
        "needs Web research turned on above",
    ),
    (
        "producer_needs_llm",
        False,
        {
            "enrichment_web_research_enabled": "on",
            "voxint_web_research": "on",
            "web_search_base_url": _VALID_URL,
        },
        False,
        "needs LLM transcript enhancement turned on",
    ),
    (
        "master_needs_endpoint",
        True,
        {"voxint_web_research": "on", "web_search_base_url": ""},
        False,
        "needs a search provider endpoint",
    ),
    (
        "master_malformed_endpoint",
        True,
        {"voxint_web_research": "on", "web_search_base_url": "example.org/search"},
        False,
        "must be a full web address",
    ),
    (
        "master_on_valid",
        True,
        {"voxint_web_research": "on", "web_search_base_url": _VALID_URL},
        True,
        None,
    ),
    (
        "all_inherit",
        True,
        {},
        True,
        None,
    ),
]


@pytest.mark.parametrize(
    ("seed_llm", "fields", "expect_commit", "message"),
    [c[1:] for c in _CASES],
    ids=[c[0] for c in _CASES],
)
def test_web_research_contract_matrix(
    session_factory: sessionmaker[Session],
    media_root: Path,
    seed_llm: bool,
    fields: dict[str, str],
    expect_commit: bool,
    message: str | None,
) -> None:
    client, _ = make_client(session_factory, media_root, seed_llm_enabled=seed_llm)
    resp = client.post(
        "/settings/web-research", data=_form(**fields), follow_redirects=False
    )
    row = _row(session_factory)
    if expect_commit:
        assert resp.status_code == 303
    else:
        assert resp.status_code == 200
        assert message is not None and message in resp.text
        # Exactly one operator message — the master-on/malformed case must not
        # double-report the base URL (validate_effective_flags owns it when master
        # is on; the standalone check runs only in the master-off branch).
        assert _alert_count(resp.text) == 1
        # Nothing written: every web-research column stays NULL on a rejected save.
        assert row is not None
        for name in (
            "voxint_web_research",
            "enrichment_web_research_enabled",
            "web_search_base_url",
            "web_search_api_key",
            "source_authority_domains",
        ):
            assert getattr(row, name) is None, name


def test_master_on_valid_persists_override(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client, _ = make_client(session_factory, media_root, seed_llm_enabled=True)
    resp = client.post(
        "/settings/web-research",
        data=_form(voxint_web_research="on", web_search_base_url=_VALID_URL),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    row = _row(session_factory)
    assert row is not None
    assert row.voxint_web_research is True
    assert row.web_search_base_url == _VALID_URL


# --- Trusted-domain editor: strict accept/reject + live reach into triage --------


def test_domains_accept_stored_verbatim(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client, _ = make_client(session_factory, media_root)
    domains = "example.com, news.bbc.co.uk\nxn--80akhbyknj4f.example"
    resp = client.post(
        "/settings/web-research",
        data=_form(source_authority_domains=domains),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert _row(session_factory).source_authority_domains == domains  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "bad",
    [
        "https://example.com",
        "example.com/path",
        "example.com:8080",
        "a@example.com",
        "*.example.com",
        "localhost",
    ],
)
def test_domains_reject_non_bare(
    session_factory: sessionmaker[Session], media_root: Path, bad: str
) -> None:
    client, _ = make_client(session_factory, media_root)
    resp = client.post(
        "/settings/web-research",
        data=_form(source_authority_domains=bad),
        follow_redirects=False,
    )
    assert resp.status_code == 200
    assert "is not a plain domain" in resp.text
    # Nothing written on a malformed token.
    assert _row(session_factory).source_authority_domains is None  # type: ignore[union-attr]


def test_domain_edit_reaches_triage_live(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # Save an allowlist via the UI, then confirm the effective set the scorer reads
    # reflects it on the next triage computation (per-request read, no restart), and
    # that a subdomain of an allowlisted registrable domain gets authority credit.
    client, settings = make_client(session_factory, media_root)
    resp = client.post(
        "/settings/web-research",
        data=_form(source_authority_domains="example.com"),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    row = _row(session_factory)
    authority = parse_authority_domains(
        resolve_effective_source_authority_domains(row, settings)
    )
    assert authority == frozenset({"example.com"})
    scored = score_profile(
        TriageInputs(
            field="bio",
            producer="web_researcher",
            producer_score=None,
            producer_components={},
            evidence=(EvidenceRef(kind="url", url="https://sub.example.com/p"),),
            voice=None,
            peer_producer_count=1,
            authority_domains=authority,
        )
    )
    assert scored.components["source_authority"] == 1.0


def test_empty_row_and_env_yields_zero_authority(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # Empty override + empty env → source_authority reads 0.0 (unchanged default).
    _, settings = make_client(session_factory, media_root)
    row = _row(session_factory)
    authority = parse_authority_domains(
        resolve_effective_source_authority_domains(row, settings)
    )
    assert authority == frozenset()
    scored = score_profile(
        TriageInputs(
            field="bio",
            producer="web_researcher",
            producer_score=None,
            producer_components={},
            evidence=(EvidenceRef(kind="url", url="https://sub.example.com/p"),),
            voice=None,
            peer_producer_count=1,
            authority_domains=authority,
        )
    )
    assert scored.components["source_authority"] == 0.0


# --- Tri-state inherit / env-echo → NULL -----------------------------------------


def test_inherit_reverts_stored_overrides_to_null(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client, _ = make_client(session_factory, media_root, seed_llm_enabled=True)
    client.post(
        "/settings/web-research",
        data=_form(
            voxint_web_research="on",
            web_search_base_url=_VALID_URL,
            source_authority_domains="example.com",
        ),
    )
    assert _row(session_factory).web_search_base_url == _VALID_URL  # type: ignore[union-attr]
    resp = client.post(
        "/settings/web-research",
        data=_form(
            voxint_web_research="inherit",
            web_search_base_url="",
            source_authority_domains="",
        ),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    row = _row(session_factory)
    assert row is not None
    assert row.voxint_web_research is None
    assert row.web_search_base_url is None
    assert row.source_authority_domains is None


def test_env_echo_base_url_collapses_to_null(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # Submitting a base URL that merely echoes the env default stores NULL, so a
    # later env change still applies (no silent pin).
    client, _ = make_client(
        session_factory,
        media_root,
        seed_llm_enabled=True,
        voxint_web_research=True,
        web_search_base_url=_VALID_URL,
    )
    resp = client.post(
        "/settings/web-research",
        data=_form(voxint_web_research="inherit", web_search_base_url=_VALID_URL),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert _row(session_factory).web_search_base_url is None  # type: ignore[union-attr]


# --- Secret handling + non-disclosure --------------------------------------------


def test_key_typed_then_kept_then_removed(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    sentinel = "provider-key-SENTINEL-9f3a"
    client, _ = make_client(session_factory, media_root)
    # Typed → stored.
    client.post("/settings/web-research", data=_form(web_search_api_key=sentinel))
    assert _row(session_factory).web_search_api_key == sentinel  # type: ignore[union-attr]
    # Blank on a later save → kept (never wiped).
    client.post("/settings/web-research", data=_form(source_authority_domains="example.com"))
    assert _row(session_factory).web_search_api_key == sentinel  # type: ignore[union-attr]
    # Remove → NULL (revert to env).
    resp = client.post(
        "/settings/web-research",
        data=_form(remove_web_search_api_key="true"),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert _row(session_factory).web_search_api_key is None  # type: ignore[union-attr]


@pytest.mark.parametrize("replacement", ["new-key", "bad key"])
def test_key_remove_and_replace_is_contradiction(
    session_factory: sessionmaker[Session], media_root: Path, replacement: str
) -> None:
    # The contradiction fires from the raw submission, so remove + a replacement is
    # reported even when the replacement is itself malformed (would fail to normalize).
    _seed_cols(session_factory, web_search_api_key="old-key")
    client, _ = make_client(session_factory, media_root)
    resp = client.post(
        "/settings/web-research",
        data=_form(web_search_api_key=replacement, remove_web_search_api_key="true"),
        follow_redirects=False,
    )
    assert resp.status_code == 200
    assert "not both" in resp.text
    assert _row(session_factory).web_search_api_key == "old-key"  # type: ignore[union-attr]


def test_key_never_rendered_on_get_or_error(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    sentinel = "provider-key-SENTINEL-never-shown"
    _seed_cols(session_factory, web_search_api_key=sentinel)
    client, _ = make_client(session_factory, media_root)
    # GET never renders the stored key.
    assert sentinel not in client.get("/settings").text
    # An error re-render that echoes back the submitted non-secret fields must not
    # carry a submitted key either.
    resp = client.post(
        "/settings/web-research",
        data=_form(
            web_search_api_key=sentinel,
            voxint_web_research="on",
            web_search_base_url="",  # invalid → forces the error re-render
        ),
        follow_redirects=False,
    )
    assert resp.status_code == 200
    assert sentinel not in resp.text


# --- Live gate + CSRF ------------------------------------------------------------


def test_toggle_reaches_research_gate_without_restart(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # Open all three gates via the UI, confirm the effective gate resolves open on
    # the same app instance, then disable retrieval and confirm it closes live —
    # queued research work re-checks this gate, so no network call would follow.
    client, settings = make_client(session_factory, media_root, seed_llm_enabled=True)
    client.post(
        "/settings/web-research",
        data=_form(
            voxint_web_research="on",
            enrichment_web_research_enabled="on",
            web_search_base_url=_VALID_URL,
        ),
    )
    assert research_gates_open(settings, _row(session_factory)) is True
    # Disable retrieval (master + producer off — master-off-with-producer-on is
    # itself invalid, so an operator disabling retrieval turns both off).
    resp = client.post(
        "/settings/web-research",
        data=_form(voxint_web_research="off", enrichment_web_research_enabled="off"),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert research_gates_open(settings, _row(session_factory)) is False


def test_web_research_requires_csrf(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client, _ = make_client(session_factory, media_root, seed_llm_enabled=True)
    resp = client.post(
        "/settings/web-research",
        data={"voxint_web_research": "on"},  # no csrf_token
        follow_redirects=False,
    )
    assert resp.status_code == 403
    assert _row(session_factory).voxint_web_research is None  # type: ignore[union-attr]
