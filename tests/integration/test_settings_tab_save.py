"""Settings tab-level composite POST endpoints (#407): dispatch, rollback, reset.

Covers the three tab-level composite routes added in #379 — ``POST /settings``
(General), ``POST /settings/ai`` (AI), ``POST /settings/media`` (Media) — plus
``_reconcile_switches`` idempotency and the ``_handle_reset_flag`` reset paths.
Per-section validation matrices are NOT repeated here (siblings cover those);
these tests verify the composite dispatch wiring, multi-section transaction
boundaries, and reset contracts.
"""

from pathlib import Path
from typing import Any

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
_CSRF_KEY = "settings-tab-save-test-csrf-key"


@pytest.fixture()
def media_root(tmp_path: Path) -> Path:
    return tmp_path


def _feature_flag_names() -> tuple[str, ...]:
    from voxint.api.routers.settings import _FEATURE_FLAG_NAMES

    return _FEATURE_FLAG_NAMES


def make_client(
    session_factory: sessionmaker[Session],
    media_root: Path,
    *,
    onboarded: bool = True,
    seed_llm_enabled: bool = False,
    **overrides: object,
) -> tuple[TestClient, Settings]:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        csrf_secret=_CSRF_KEY,
        media_root=media_root,
        console_settings_enabled=True,
        **overrides,
    )
    client = TestClient(create_app(settings=settings, session_factory=session_factory))
    client.auth = CREDS
    if onboarded:
        seed_onboarded(session_factory, llm_enabled=seed_llm_enabled)
    return client, settings


def _form(
    rendered: list[str] | None = None, **fields: str
) -> dict[str, str | list[str]]:
    """Build form data supporting repeated ``_rendered`` keys."""
    result: dict[str, str | list[str]] = {
        "csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_SETTINGS),
    }
    if rendered:
        result["_rendered"] = rendered
    result.update(fields)
    return result


def _row(session_factory: sessionmaker[Session]) -> AppSettings | None:
    with session_factory() as session:
        return get_app_settings(session)


def _seed_cols(session_factory: sessionmaker[Session], **columns: object) -> None:
    with session_factory() as session:
        row = get_or_create(session, llm_enabled_default=False)
        for name, value in columns.items():
            setattr(row, name, value)
        session.commit()


def _snapshot(session_factory: sessionmaker[Session]) -> dict[str, Any]:
    """Full-row snapshot through a fresh session for rollback comparison."""
    with session_factory() as session:
        row = get_app_settings(session)
        if row is None:
            return {}
        return {
            c.key: getattr(row, c.key)
            for c in row.__table__.columns  # type: ignore[union-attr]
        }


# ---------------------------------------------------------------------------
# General tab: POST /settings
# ---------------------------------------------------------------------------


class TestGeneralTabSave:
    def test_success_dispatch_all_flags_off(
        self, session_factory: sessionmaker[Session], media_root: Path
    ) -> None:
        """All feature flags flipped from on to off via the composite route."""
        flag_names = _feature_flag_names()
        client, _ = make_client(session_factory, media_root)
        _seed_cols(
            session_factory,
            **{name: True for name in flag_names},
        )
        data = _form(rendered=list(flag_names))
        resp = client.post("/settings", data=data, follow_redirects=False)
        assert resp.status_code == 303
        assert "/settings#features" in resp.headers["location"]
        row = _row(session_factory)
        assert row is not None
        for name in flag_names:
            assert getattr(row, name) is False, f"{name} should be False"

    def test_idempotency_inherit_preserved(
        self, session_factory: sessionmaker[Session], media_root: Path
    ) -> None:
        """Save without changes preserves NULL (inherit) — no silent override."""
        flag_names = _feature_flag_names()
        client, _ = make_client(
            session_factory,
            media_root,
            enrichment_names_enabled=True,
            ytdlp_enabled=False,
        )
        data = _form(
            rendered=list(flag_names),
            enrichment_names_enabled="on",
        )
        resp = client.post("/settings", data=data, follow_redirects=False)
        assert resp.status_code == 303
        row = _row(session_factory)
        assert row is not None
        for name in flag_names:
            assert getattr(row, name) is None, f"{name} should stay inherit (NULL)"


# ---------------------------------------------------------------------------
# AI tab: POST /settings/ai
# ---------------------------------------------------------------------------


class TestAiTabSave:
    def test_happy_path_all_sections(
        self, session_factory: sessionmaker[Session], media_root: Path
    ) -> None:
        """Valid LLM + semantic + translation + glossary all persist together."""
        client, _ = make_client(
            session_factory,
            media_root,
            seed_llm_enabled=True,
            semantic_index_enabled=False,
            semantic_index_autogenerate=False,
            translation_autogenerate=False,
        )
        _seed_cols(session_factory, semantic_index_enabled=False)
        data = _form(
            rendered=["semantic_index_enabled", "semantic_index_autogenerate",
                      "translation_autogenerate"],
            enabled="true",
            llm_base_url="https://llm.example.org/v1",
            llm_model="gpt-test",
            llm_api_key="sk-test-key-1234",
            semantic_index_enabled="on",
            translation_target_language="es",
            translation_autogenerate="on",
            vocabulary="Alpha\nBeta",
        )
        resp = client.post("/settings/ai", data=data, follow_redirects=False)
        assert resp.status_code == 303
        assert "/settings/ai#llm" in resp.headers["location"]
        row = _row(session_factory)
        assert row is not None
        assert row.llm_enabled is True
        assert row.llm_base_url == "https://llm.example.org/v1"
        assert row.llm_model == "gpt-test"
        assert row.llm_api_key == "sk-test-key-1234"
        assert row.semantic_index_enabled is True
        assert row.translation_target_language == "es"
        assert row.translation_autogenerate is True
        assert row.vocabulary == ["Alpha", "Beta"]

    def test_semantic_failure_preserves_valid_llm(
        self, session_factory: sessionmaker[Session], media_root: Path
    ) -> None:
        """Invalid semantic (autogenerate on, enabled off) rolls back only
        semantic; valid LLM edits persist (#403/#405)."""
        client, _ = make_client(
            session_factory, media_root, seed_llm_enabled=True
        )
        _seed_cols(
            session_factory,
            llm_base_url="https://original.example.org/v1",
            llm_model="original-model",
            llm_api_key="sk-original-key",
            semantic_index_enabled=True,
        )
        before = _snapshot(session_factory)
        data = _form(
            rendered=["semantic_index_enabled", "semantic_index_autogenerate",
                      "translation_autogenerate"],
            enabled="true",
            llm_base_url="https://replacement.example.org/v1",
            llm_model="replacement-model",
            llm_api_key="sk-replacement-key",
            semantic_index_autogenerate="on",
            translation_target_language="inherit",
            translation_autogenerate="inherit",
            vocabulary="",
        )
        resp = client.post("/settings/ai", data=data, follow_redirects=False)
        assert resp.status_code == 422
        assert "auto-index" in resp.text.lower() or "semantic search on" in resp.text.lower()
        after = _snapshot(session_factory)
        # LLM edits persisted (valid section).
        assert after["llm_base_url"] == "https://replacement.example.org/v1"
        assert after["llm_model"] == "replacement-model"
        assert after["llm_api_key"] == "sk-replacement-key"
        # Semantic rolled back (failed section).
        assert after["semantic_index_enabled"] == before["semantic_index_enabled"]
        assert after["semantic_index_autogenerate"] == before["semantic_index_autogenerate"]
        # Summary shown.
        assert "not saved" in resp.text.lower()

    def test_translation_failure_preserves_valid_llm_and_semantic(
        self, session_factory: sessionmaker[Session], media_root: Path
    ) -> None:
        """Auto-translate on without target rolls back only translation;
        LLM + semantic edits persist (#403/#405)."""
        client, _ = make_client(
            session_factory, media_root, seed_llm_enabled=True
        )
        _seed_cols(
            session_factory,
            llm_base_url="https://original.example.org/v1",
            llm_api_key="sk-original-key",
            semantic_index_enabled=True,
        )
        before = _snapshot(session_factory)
        data = _form(
            rendered=["semantic_index_enabled", "semantic_index_autogenerate",
                      "translation_autogenerate"],
            enabled="true",
            llm_base_url="https://replacement.example.org/v1",
            llm_model="",
            llm_api_key="sk-replacement-key",
            semantic_index_enabled="on",
            translation_target_language="inherit",
            translation_autogenerate="on",
            vocabulary="",
        )
        resp = client.post("/settings/ai", data=data, follow_redirects=False)
        assert resp.status_code == 422
        assert "preferred language" in resp.text.lower()
        after = _snapshot(session_factory)
        # LLM persisted.
        assert after["llm_base_url"] == "https://replacement.example.org/v1"
        assert after["llm_api_key"] == "sk-replacement-key"
        # Semantic persisted (submitted "on").
        assert after["semantic_index_enabled"] is True
        # Translation rolled back.
        assert after["translation_autogenerate"] == before["translation_autogenerate"]
        # Summary shown.
        assert "not saved" in resp.text.lower()

    def test_glossary_failure_preserves_all_prior(
        self, session_factory: sessionmaker[Session], media_root: Path
    ) -> None:
        """Overlong glossary term (422) rolls back only glossary; LLM +
        semantic + translation edits persist (#403/#405)."""
        client, _ = make_client(
            session_factory, media_root, seed_llm_enabled=True
        )
        _seed_cols(
            session_factory,
            llm_base_url="https://original.example.org/v1",
            llm_api_key="sk-original-key",
            semantic_index_enabled=True,
        )
        before = _snapshot(session_factory)
        overlong_term = "x" * 121
        data = _form(
            rendered=["semantic_index_enabled", "semantic_index_autogenerate",
                      "translation_autogenerate"],
            enabled="true",
            llm_base_url="https://replacement.example.org/v1",
            llm_model="",
            llm_api_key="sk-replacement-key",
            semantic_index_enabled="on",
            translation_target_language="es",
            translation_autogenerate="on",
            vocabulary=overlong_term,
        )
        resp = client.post("/settings/ai", data=data, follow_redirects=False)
        assert resp.status_code == 422
        assert "120 characters" in resp.text
        after = _snapshot(session_factory)
        # LLM persisted.
        assert after["llm_base_url"] == "https://replacement.example.org/v1"
        assert after["llm_api_key"] == "sk-replacement-key"
        # Semantic persisted.
        assert after["semantic_index_enabled"] is True
        # Translation persisted.
        assert after["translation_target_language"] == "es"
        # Glossary unchanged (the only failed section).
        assert after["vocabulary"] == before["vocabulary"]
        # Summary shown.
        assert "not saved" in resp.text.lower()

    def test_llm_enablement_failure_partial_save_with_siblings(
        self, session_factory: sessionmaker[Session], media_root: Path
    ) -> None:
        """LLM enable with no effective key saves key/model but forces
        llm_enabled off (partial save); siblings persist (#403/#405)."""
        client, _ = make_client(
            session_factory, media_root, seed_llm_enabled=False
        )
        _seed_cols(
            session_factory,
            llm_base_url="https://original.example.org/v1",
            llm_api_key=None,
            vocabulary=["old-term"],
        )
        data = _form(
            rendered=["semantic_index_enabled", "semantic_index_autogenerate",
                      "translation_autogenerate"],
            enabled="true",
            llm_base_url="https://replacement.example.org/v1",
            llm_model="replacement-model",
            llm_api_key="",
            translation_target_language="inherit",
            translation_autogenerate="inherit",
            vocabulary="NewTerm",
        )
        resp = client.post("/settings/ai", data=data, follow_redirects=False)
        assert resp.status_code == 422
        after = _snapshot(session_factory)
        # LLM partially saved: key/model written, enable forced off.
        assert after["llm_base_url"] == "https://replacement.example.org/v1"
        assert after["llm_model"] == "replacement-model"
        assert after["llm_enabled"] is False
        # Glossary sibling persisted.
        assert after["vocabulary"] == ["NewTerm"]
        # Summary reflects partial save.
        assert "partially saved" in resp.text.lower()

    def test_llm_strand_disable_not_saved_with_siblings(
        self, session_factory: sessionmaker[Session], media_root: Path
    ) -> None:
        """Strand-disable (LLM off while a dependent is on) saves nothing
        in the LLM section; siblings persist (#403/#405)."""
        client, _ = make_client(
            session_factory, media_root, seed_llm_enabled=True,
            llm_enabled=True, enrichment_run_assets_enabled=True,
        )
        _seed_cols(
            session_factory,
            llm_base_url="https://original.example.org/v1",
            llm_api_key="sk-original-key",
            llm_enabled=True,
            vocabulary=["old-term"],
        )
        before = _snapshot(session_factory)
        data = _form(
            rendered=["semantic_index_enabled", "semantic_index_autogenerate",
                      "translation_autogenerate"],
            enabled="",
            llm_base_url="https://replacement.example.org/v1",
            llm_model="",
            llm_api_key="",
            translation_target_language="inherit",
            translation_autogenerate="inherit",
            vocabulary="NewTerm",
        )
        resp = client.post("/settings/ai", data=data, follow_redirects=False)
        assert resp.status_code == 422
        after = _snapshot(session_factory)
        # LLM not saved (strand refused).
        assert after["llm_base_url"] == before["llm_base_url"]
        assert after["llm_api_key"] == before["llm_api_key"]
        assert after["llm_enabled"] == before["llm_enabled"]
        # Glossary sibling persisted.
        assert after["vocabulary"] == ["NewTerm"]
        # Summary names LLM as not saved.
        assert "llm not saved" in resp.text.lower()


# ---------------------------------------------------------------------------
# Media tab: POST /settings/media
# ---------------------------------------------------------------------------


class TestMediaTabSave:
    def test_success_dispatch(
        self, session_factory: sessionmaker[Session], media_root: Path
    ) -> None:
        """Watch-folder + web-research all persist in one POST."""
        client, _ = make_client(session_factory, media_root, seed_llm_enabled=True)
        data = _form(
            rendered=["watch_folder_enabled", "voxint_web_research",
                      "enrichment_web_research_enabled"],
            watch_folder_enabled="on",
            voxint_web_research="on",
            enrichment_web_research_enabled="on",
            web_search_base_url="https://searxng.example.org",
            web_search_api_key="",
            source_authority_domains="example.com",
        )
        resp = client.post("/settings/media", data=data, follow_redirects=False)
        assert resp.status_code == 303
        assert "/settings/media#folders" in resp.headers["location"]
        row = _row(session_factory)
        assert row is not None
        assert row.watch_folder_enabled is True
        assert row.voxint_web_research is True
        assert row.enrichment_web_research_enabled is True
        assert row.web_search_base_url == "https://searxng.example.org"
        assert row.source_authority_domains == "example.com"

    def test_research_failure_rolls_back_watch_folder(
        self, session_factory: sessionmaker[Session], media_root: Path
    ) -> None:
        """Master-on with no base URL rolls back watch-folder change."""
        client, _ = make_client(session_factory, media_root)
        _seed_cols(session_factory, watch_folder_enabled=False)
        before = _snapshot(session_factory)
        data = _form(
            rendered=["watch_folder_enabled", "voxint_web_research",
                      "enrichment_web_research_enabled"],
            watch_folder_enabled="on",
            voxint_web_research="on",
            enrichment_web_research_enabled="off",
            web_search_base_url="",
            web_search_api_key="",
            source_authority_domains="",
        )
        resp = client.post("/settings/media", data=data, follow_redirects=False)
        assert resp.status_code == 200
        assert "search provider" in resp.text.lower() or "base_url" in resp.text.lower()
        after = _snapshot(session_factory)
        assert after["watch_folder_enabled"] == before["watch_folder_enabled"]


# ---------------------------------------------------------------------------
# Reset paths
# ---------------------------------------------------------------------------


class TestResetPaths:
    @pytest.mark.parametrize(
        ("flag", "tab", "endpoint"),
        [
            ("ytdlp_enabled", "", "/settings"),
            ("semantic_index_enabled", "ai", "/settings/ai"),
            ("watch_folder_enabled", "media", "/settings/media"),
        ],
    )
    def test_happy_reset_clears_to_null(
        self,
        session_factory: sessionmaker[Session],
        media_root: Path,
        flag: str,
        tab: str,
        endpoint: str,
    ) -> None:
        """Resetting a flag sets its column to NULL and redirects correctly."""
        client, _ = make_client(session_factory, media_root)
        _seed_cols(session_factory, **{flag: True})
        row_before = _row(session_factory)
        assert row_before is not None
        assert getattr(row_before, flag) is True

        data = _form(reset_flag=flag)
        resp = client.post(endpoint, data=data, follow_redirects=False)
        assert resp.status_code == 303
        assert f"#sw-{flag}" in resp.headers["location"]
        if tab:
            assert f"/settings/{tab}" in resp.headers["location"]

        row_after = _row(session_factory)
        assert row_after is not None
        assert getattr(row_after, flag) is None

    def test_reset_ignores_dirty_fields(
        self, session_factory: sessionmaker[Session], media_root: Path
    ) -> None:
        """Reset is an early return — other form fields are not processed."""
        flag_names = _feature_flag_names()
        client, _ = make_client(session_factory, media_root)
        _seed_cols(session_factory, ytdlp_enabled=True, enrichment_names_enabled=False)
        data = _form(
            rendered=list(flag_names),
            reset_flag="ytdlp_enabled",
            enrichment_names_enabled="on",
        )
        resp = client.post("/settings", data=data, follow_redirects=False)
        assert resp.status_code == 303
        row = _row(session_factory)
        assert row is not None
        assert row.ytdlp_enabled is None
        assert row.enrichment_names_enabled is False, "dirty field must not be processed"

    def test_reset_invariant_violation_rolls_back(
        self, session_factory: sessionmaker[Session], media_root: Path
    ) -> None:
        """Resetting a prerequisite flag while dependent is on re-renders with error (#404)."""
        client, _ = make_client(
            session_factory,
            media_root,
            seed_llm_enabled=True,
        )
        _seed_cols(
            session_factory,
            enrichment_run_assets_enabled=True,
            enrichment_run_assets_autogenerate=True,
        )
        before = _snapshot(session_factory)
        data = _form(reset_flag="enrichment_run_assets_enabled")
        resp = client.post("/settings", data=data, follow_redirects=False)
        assert resp.status_code == 200
        assert "Reset not applied" in resp.text
        assert "run assets" in resp.text.lower()
        after = _snapshot(session_factory)
        assert after["enrichment_run_assets_enabled"] is True, "should not have been reset"
        assert (
            after["enrichment_run_assets_autogenerate"]
            == before["enrichment_run_assets_autogenerate"]
        )

    def test_reset_semantic_invariant_violation_rolls_back(
        self, session_factory: sessionmaker[Session], media_root: Path
    ) -> None:
        """Resetting semantic_index_enabled while autogenerate is on re-renders (#404)."""
        client, _ = make_client(
            session_factory,
            media_root,
            semantic_index_enabled=False,
            semantic_index_autogenerate=False,
        )
        _seed_cols(
            session_factory,
            semantic_index_enabled=True,
            semantic_index_autogenerate=True,
        )
        data = _form(reset_flag="semantic_index_enabled")
        resp = client.post("/settings/ai", data=data, follow_redirects=False)
        assert resp.status_code == 200
        assert "Reset not applied" in resp.text
        assert "semantic search" in resp.text.lower()
        row = _row(session_factory)
        assert row is not None
        assert row.semantic_index_enabled is True, "should not have been reset"

    def test_non_resettable_flag_ignored(
        self, session_factory: sessionmaker[Session], media_root: Path
    ) -> None:
        """Submitting a non-resettable flag name redirects safely, no change."""
        client, _ = make_client(
            session_factory, media_root, seed_llm_enabled=True
        )
        before = _snapshot(session_factory)
        data = _form(reset_flag="llm_enabled")
        resp = client.post("/settings/ai", data=data, follow_redirects=False)
        assert resp.status_code == 303
        after = _snapshot(session_factory)
        assert after == before


# ---------------------------------------------------------------------------
# CSRF
# ---------------------------------------------------------------------------


class TestCsrf:
    @pytest.mark.parametrize(
        "endpoint",
        ["/settings", "/settings/ai", "/settings/media"],
    )
    def test_missing_csrf_rejected(
        self,
        session_factory: sessionmaker[Session],
        media_root: Path,
        endpoint: str,
    ) -> None:
        """POST without CSRF token returns 403 and writes nothing."""
        client, _ = make_client(session_factory, media_root)
        before = _snapshot(session_factory)
        resp = client.post(endpoint, data={}, follow_redirects=False)
        assert resp.status_code == 403
        after = _snapshot(session_factory)
        assert after == before

    @pytest.mark.parametrize(
        "endpoint",
        ["/settings", "/settings/ai", "/settings/media"],
    )
    def test_wrong_scope_csrf_rejected(
        self,
        session_factory: sessionmaker[Session],
        media_root: Path,
        endpoint: str,
    ) -> None:
        """CSRF token from a different scope returns 403 and writes nothing."""
        client, _ = make_client(session_factory, media_root)
        before = _snapshot(session_factory)
        wrong_token = mint_csrf_token(_CSRF_KEY, CSRF_SETUP)
        resp = client.post(
            endpoint, data={"csrf_token": wrong_token}, follow_redirects=False
        )
        assert resp.status_code == 403
        after = _snapshot(session_factory)
        assert after == before
