"""The LLM form refuses a disable that strands an LLM-dependent feature (issue #77).

`POST /settings/llm` and `POST /setup/llm` share ``_persist_llm_settings``, which
now cross-validates the effective feature-flag combo before turning LLM off: a
disable that would strand a feature the boot validator requires ``llm_enabled=true``
for (run assets / the LLM name pass / web-research enrichment) is refused with a
plain-language message and writes NOTHING — LLM stays on, and no dependent is
auto-disabled (#62). A disable that strands nothing still succeeds, and an
*unrelated* pre-existing violation never blocks the save (only the delta caused by
flipping LLM matters). Runs against real Postgres so the mutation-commit semantics
the atomic route logic exists for actually reproduce.
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
_CSRF_KEY = "llm-disable-guard-test-csrf-key"


@pytest.fixture()
def media_root(tmp_path: Path) -> Path:
    return tmp_path


def make_client(
    session_factory: sessionmaker[Session],
    media_root: Path,
    *,
    onboarded: bool = True,
    seed_llm_enabled: bool = True,
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
        seed_onboarded(session_factory, llm_enabled=seed_llm_enabled)
    return client, settings


def _disable_form(csrf_name: str = CSRF_SETTINGS) -> dict[str, str]:
    # No ``enabled`` field → the bool Form defaults to False (a deliberate disable).
    return {"csrf_token": mint_csrf_token(_CSRF_KEY, csrf_name)}


def _seed_flags(session_factory: sessionmaker[Session], **columns: object) -> None:
    """Force flag columns on the singleton (already-onboarded) row directly, bypassing
    the form/validator, to stage a pre-existing override the guard must reason about.
    ``llm_enabled`` is left as ``seed_onboarded`` set it."""
    with session_factory() as session:
        row = get_or_create(session, llm_enabled_default=True)
        for name, value in columns.items():
            setattr(row, name, value)
        session.commit()


def _row(session_factory: sessionmaker[Session]) -> AppSettings | None:
    with session_factory() as session:
        return get_app_settings(session)


def test_disable_blocked_when_run_assets_on_via_row(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client, _ = make_client(session_factory, media_root)
    _seed_flags(session_factory, enrichment_run_assets_enabled=True)

    resp = client.post("/settings/llm", data=_disable_form())

    assert resp.status_code == 200  # re-render, not the 303 success redirect
    assert "run assets" in resp.text
    row = _row(session_factory)
    assert row is not None
    assert row.llm_enabled is True  # write-nothing: LLM stays on
    assert row.enrichment_run_assets_enabled is True  # dependent NOT auto-disabled


def test_refused_disable_writes_no_llm_field(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # Prove "writes nothing" across EVERY LLM column, not just llm_enabled: seed
    # distinct stored values, submit endpoint/model edits alongside the disable, and
    # assert the whole row is untouched (the submitted edits must not leak either).
    client, _ = make_client(session_factory, media_root)
    _seed_flags(
        session_factory,
        enrichment_run_assets_enabled=True,
        llm_base_url="http://seeded.example:9999/v1",
        llm_model="seeded-model",
        llm_api_key="sk-SEEDED-value",
    )

    form = _disable_form()
    form["llm_base_url"] = "http://attacker.example/edited"
    form["llm_model"] = "edited-model"
    resp = client.post("/settings/llm", data=form)

    assert resp.status_code == 200
    row = _row(session_factory)
    assert row is not None
    assert row.llm_enabled is True
    assert row.llm_base_url == "http://seeded.example:9999/v1"
    assert row.llm_model == "seeded-model"
    assert row.llm_api_key == "sk-SEEDED-value"


def test_setup_disable_with_no_row_and_env_dependent_creates_no_row(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # /setup/llm before the wizard has ever saved: no app_settings row exists and the
    # dependent is env-configured. A refused disable must not call get_or_create — the
    # singleton row stays absent (the strongest form of "writes nothing").
    client, _ = make_client(
        session_factory,
        media_root,
        onboarded=False,
        llm_enabled=True,
        enrichment_run_assets_enabled=True,
    )
    assert _row(session_factory) is None  # precondition: no row yet

    resp = client.post("/setup/llm", data=_disable_form(CSRF_SETUP))

    assert resp.status_code == 200
    assert "run assets" in resp.text
    assert _row(session_factory) is None  # nothing created


def test_disable_blocked_when_run_assets_on_via_env(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # The dependent is env-configured (row column NULL → inherits env True); the guard
    # validates the *effective* combo, so it still fires.
    client, _ = make_client(
        session_factory,
        media_root,
        llm_enabled=True,
        enrichment_run_assets_enabled=True,
    )
    resp = client.post("/settings/llm", data=_disable_form())

    assert resp.status_code == 200
    assert "run assets" in resp.text
    row = _row(session_factory)
    assert row is not None
    assert row.llm_enabled is True


def test_disable_succeeds_when_nothing_depends_on_llm(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client, _ = make_client(session_factory, media_root)

    resp = client.post("/settings/llm", data=_disable_form(), follow_redirects=False)

    assert resp.status_code == 303
    row = _row(session_factory)
    assert row is not None
    assert row.llm_enabled is False


def test_setup_route_is_guarded_too(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client, _ = make_client(session_factory, media_root)
    _seed_flags(session_factory, enrichment_names_enabled=True, enrichment_names_llm_enabled=True)

    resp = client.post("/setup/llm", data=_disable_form(CSRF_SETUP))

    assert resp.status_code == 200
    assert "the LLM name pass" in resp.text
    row = _row(session_factory)
    assert row is not None
    assert row.llm_enabled is True


def test_unrelated_pre_existing_violation_does_not_block_disable(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # A stored master-web-research toggle with no endpoint is already invalid (the
    # "voxint_web_research ⇒ web_search_base_url" invariant) — but that violation is
    # independent of LLM, so it must NOT block turning LLM off. Only the delta caused
    # by flipping llm_enabled counts. No LLM-dependent feature is on here.
    client, _ = make_client(session_factory, media_root)
    _seed_flags(session_factory, voxint_web_research=True)  # base_url left blank/NULL

    resp = client.post("/settings/llm", data=_disable_form(), follow_redirects=False)

    assert resp.status_code == 303  # disable allowed despite the pre-existing violation
    row = _row(session_factory)
    assert row is not None
    assert row.llm_enabled is False


def test_already_off_dependent_is_not_re_blocked(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # LLM already effectively OFF while a dependent is stranded (a pre-existing state
    # the guard is not responsible for). Re-submitting a disable is a no-op on LLM and
    # introduces no NEW violation, so it must succeed rather than trap the operator.
    client, _ = make_client(session_factory, media_root, seed_llm_enabled=False)
    _seed_flags(session_factory, enrichment_run_assets_enabled=True)

    resp = client.post("/settings/llm", data=_disable_form(), follow_redirects=False)

    assert resp.status_code == 303
    row = _row(session_factory)
    assert row is not None
    assert row.llm_enabled is False
