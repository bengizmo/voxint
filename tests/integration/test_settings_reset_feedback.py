"""Reset-flag operator feedback (#404): blocked resets re-render with section errors.

When ``_handle_reset_flag`` rolls back because the reset would NEWLY introduce
an invariant violation, the response is a 200 re-render with a plain-language
notice in the owning section's error slot (matching the tab POST save error
pattern).  Successful resets remain a silent 303 redirect.  The before/after
delta discipline (cf. ``_llm_disable_strand_error``) ensures a pre-existing
unrelated violation never blocks or mislabels the cause.
"""

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.api.csrf import CSRF_SETTINGS, mint_csrf_token
from voxint.app_settings import get_app_settings, get_or_create
from voxint.config import Settings
from voxint.db.models import AppSettings

CREDS = ("reviewer", "s3cret")
_CSRF_KEY = "reset-feedback-test-csrf-key"


@pytest.fixture()
def media_root(tmp_path: Path) -> Path:
    return tmp_path


def _client(
    session_factory: sessionmaker[Session],
    media_root: Path,
    *,
    seed_llm_enabled: bool = False,
    **overrides: object,
) -> TestClient:
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
    seed_onboarded(session_factory, llm_enabled=seed_llm_enabled)
    return client


def _form(**fields: str) -> dict[str, str]:
    return {"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_SETTINGS), **fields}


def _seed(session_factory: sessionmaker[Session], **columns: object) -> None:
    with session_factory() as session:
        row = get_or_create(session, llm_enabled_default=False)
        for name, value in columns.items():
            setattr(row, name, value)
        session.commit()


def _row(session_factory: sessionmaker[Session]) -> AppSettings | None:
    with session_factory() as session:
        return get_app_settings(session)


def _snapshot(session_factory: sessionmaker[Session]) -> dict[str, Any]:
    with session_factory() as session:
        row = get_app_settings(session)
        if row is None:
            return {}
        return {
            c.key: getattr(row, c.key)
            for c in row.__table__.columns  # type: ignore[union-attr]
        }


# ---------------------------------------------------------------------------
# Blocked resets: 200 re-render with section error
# ---------------------------------------------------------------------------


class TestBlockedResetFeedback:
    def test_feature_flag_blocked_reset_shows_error(
        self, session_factory: sessionmaker[Session], media_root: Path
    ) -> None:
        """Resetting a feature flag prerequisite re-renders with error in features section."""
        client = _client(session_factory, media_root, seed_llm_enabled=True)
        _seed(
            session_factory,
            enrichment_run_assets_enabled=True,
            enrichment_run_assets_autogenerate=True,
        )
        resp = client.post(
            "/settings",
            data=_form(reset_flag="enrichment_run_assets_enabled"),
            follow_redirects=False,
        )
        assert resp.status_code == 200
        assert "Reset not applied" in resp.text
        assert "run assets" in resp.text.lower()
        row = _row(session_factory)
        assert row is not None
        assert row.enrichment_run_assets_enabled is True

    def test_semantic_blocked_reset_shows_error(
        self, session_factory: sessionmaker[Session], media_root: Path
    ) -> None:
        """Resetting semantic_index_enabled while autogenerate is on shows semantic error."""
        client = _client(
            session_factory,
            media_root,
            semantic_index_enabled=False,
            semantic_index_autogenerate=False,
        )
        _seed(
            session_factory,
            semantic_index_enabled=True,
            semantic_index_autogenerate=True,
        )
        resp = client.post(
            "/settings/ai",
            data=_form(reset_flag="semantic_index_enabled"),
            follow_redirects=False,
        )
        assert resp.status_code == 200
        assert "Reset not applied" in resp.text
        assert "semantic search" in resp.text.lower()
        row = _row(session_factory)
        assert row is not None
        assert row.semantic_index_enabled is True

    def test_synthdetect_blocked_reset_rolls_back(
        self, session_factory: sessionmaker[Session], media_root: Path
    ) -> None:
        """Resetting synthdetect_enabled while autogenerate is on blocks the reset.

        The synthdetect section has no template partial yet, so the error message
        is present in the context but not rendered.  This test verifies the
        invariant gate fires (200 re-render, column unchanged).
        """
        client = _client(
            session_factory,
            media_root,
            synthdetect_enabled=False,
            synthdetect_autogenerate=False,
        )
        _seed(
            session_factory,
            synthdetect_enabled=True,
            synthdetect_autogenerate=True,
        )
        resp = client.post(
            "/settings/ai",
            data=_form(reset_flag="synthdetect_enabled"),
            follow_redirects=False,
        )
        assert resp.status_code == 200
        row = _row(session_factory)
        assert row is not None
        assert row.synthdetect_enabled is True


# ---------------------------------------------------------------------------
# Successful resets: 303 redirect, column set to NULL
# ---------------------------------------------------------------------------


class TestSuccessfulReset:
    def test_successful_reset_returns_303(
        self, session_factory: sessionmaker[Session], media_root: Path
    ) -> None:
        """Resetting a flag with no invariant violation gives a 303 redirect."""
        client = _client(session_factory, media_root, seed_llm_enabled=True)
        _seed(session_factory, enrichment_run_assets_enabled=True)
        resp = client.post(
            "/settings",
            data=_form(reset_flag="enrichment_run_assets_enabled"),
            follow_redirects=False,
        )
        assert resp.status_code == 303
        row = _row(session_factory)
        assert row is not None
        assert row.enrichment_run_assets_enabled is None


# ---------------------------------------------------------------------------
# Delta discipline: pre-existing violations don't block unrelated resets
# ---------------------------------------------------------------------------


class TestDeltaDiscipline:
    def test_preexisting_violation_does_not_block_unrelated_reset(
        self, session_factory: sessionmaker[Session], media_root: Path
    ) -> None:
        """A pre-existing invariant violation must not block an unrelated reset.

        Seed a broken state (autogenerate on + enabled off for run assets)
        then reset an unrelated flag (watch_folder_enabled). The unrelated
        reset should succeed despite the pre-existing violation.
        """
        client = _client(session_factory, media_root, seed_llm_enabled=True)
        _seed(
            session_factory,
            enrichment_run_assets_autogenerate=True,
            enrichment_run_assets_enabled=False,
            watch_folder_enabled=True,
        )
        resp = client.post(
            "/settings/media",
            data=_form(reset_flag="watch_folder_enabled"),
            follow_redirects=False,
        )
        assert resp.status_code == 303
        row = _row(session_factory)
        assert row is not None
        assert row.watch_folder_enabled is None

    def test_preexisting_feature_violation_does_not_block_semantic_reset(
        self, session_factory: sessionmaker[Session], media_root: Path
    ) -> None:
        """A pre-existing feature-flag violation doesn't block a clean semantic reset."""
        client = _client(
            session_factory,
            media_root,
            seed_llm_enabled=True,
        )
        _seed(
            session_factory,
            enrichment_run_assets_autogenerate=True,
            enrichment_run_assets_enabled=False,
            semantic_index_enabled=True,
        )
        resp = client.post(
            "/settings/ai",
            data=_form(reset_flag="semantic_index_enabled"),
            follow_redirects=False,
        )
        assert resp.status_code == 303
        row = _row(session_factory)
        assert row is not None
        assert row.semantic_index_enabled is None


# ---------------------------------------------------------------------------
# Message-mapping drift test
# ---------------------------------------------------------------------------


class TestInvariantCopyDrift:
    def test_feature_invariant_copy_covers_all_reachable_messages(self) -> None:
        """Every message from validate_effective_flags has a plain-language entry."""
        from voxint.api.routers.settings import _FEATURE_INVARIANT_COPY
        from voxint.app_settings import EffectiveFlags, validate_effective_flags

        all_messages: list[str] = []
        for llm in (True, False):
            for names in (True, False):
                for llm_names in (True, False):
                    for assets in (True, False):
                        for auto_assets in (True, False):
                            for web in (True, False):
                                for web_enrich in (True, False):
                                    msgs = validate_effective_flags(
                                        EffectiveFlags(
                                            llm_enabled=llm,
                                            enrichment_names_enabled=names,
                                            enrichment_names_llm_enabled=llm_names,
                                            enrichment_run_assets_enabled=assets,
                                            enrichment_run_assets_autogenerate=auto_assets,
                                            voxint_web_research=web,
                                            enrichment_web_research_enabled=web_enrich,
                                            web_search_base_url="http://example.com",
                                        )
                                    )
                                    all_messages.extend(msgs)
        unique = set(all_messages)
        unmapped = [m for m in unique if m not in _FEATURE_INVARIANT_COPY]
        assert unmapped == [], f"Unmapped messages: {unmapped}"
