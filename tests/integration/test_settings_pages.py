"""Console 2.0 P6 settings hub + sub-pages (issue #161).

The flag ``console_settings_enabled`` branches the ``/settings`` CONTENT: off is
the current flat page byte-for-byte (its sections and anchors intact), on is the
regrouped hub that keeps those same mutable sections inline and links out to the
read-only sub-pages. The four sub-pages (status, hardware, database, plugins) are
always registered and reachable regardless of the flag (dark-ship), fail soft
when a dependency is down, and never write config or restart anything.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.api.csrf import CSRF_SETTINGS, mint_csrf_token
from voxint.api.service_identity import ServiceIdentityView
from voxint.api.settings_view import build_hardware_view
from voxint.config import Settings
from voxint.plugins import reset_plugins_cache
from voxint.plugins.base import PluginManifest, SettingsSection, VoxintPlugin
from voxint.plugins.deps import PluginRouteDeps

CREDS = ("reviewer", "s3cret")


@pytest.fixture(autouse=True)
def _reset_registry() -> Iterator[None]:
    reset_plugins_cache()
    yield
    reset_plugins_cache()


def _client(
    session_factory: sessionmaker[Session], **overrides: object
) -> TestClient:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        csrf_secret="settings-page-test-csrf-key",
        **overrides,
    )
    client = TestClient(create_app(settings=settings, session_factory=session_factory))
    client.auth = CREDS
    seed_onboarded(session_factory)
    return client


# --------------------------------------------------------------------------- #
# Flag branches /settings content; anchors and form posts survive.
# --------------------------------------------------------------------------- #
# Anchors external files deep-link into, plus the mutation endpoints; both must
# survive the hub regroup, since the deep-linking files are outside Track D.
_PRESERVED_ANCHORS = ('id="llm"', 'id="features"', 'id="glossary"')
_PRESERVED_POSTS = ("/settings/llm", "/settings/features", "/settings/glossary")


def test_flag_off_renders_flat_page(session_factory: sessionmaker[Session]) -> None:
    # The flag defaults ON since activation (P6b, #161); pass it off explicitly to
    # pin the legacy flat page that flag-off must still render byte-compatibly.
    client = _client(session_factory, console_settings_enabled=False)
    body = client.get("/settings").text
    for marker in (*_PRESERVED_ANCHORS, *_PRESERVED_POSTS):
        assert marker in body
    # The hub-only sub-page nav and groupings are absent on the flat page. (The
    # sidebar "Hardware" shortcut points at /settings/status on every page now, so
    # a hub-card-only link like /settings/database is the distinguishing marker.)
    assert "/settings/database" not in body
    assert "Everyday settings" not in body


def test_flag_on_hub_keeps_anchors_and_links_subpages(
    session_factory: sessionmaker[Session],
) -> None:
    client = _client(session_factory, console_settings_enabled=True)
    body = client.get("/settings").text
    # Anchors and form posts still resolve on the hub (deep-links keep working).
    for marker in (*_PRESERVED_ANCHORS, *_PRESERVED_POSTS):
        assert marker in body
    # The hub adds the sub-page nav and the plain-language groupings.
    for link in (
        "/settings/status",
        "/settings/hardware",
        "/settings/database",
        "/settings/plugins",
    ):
        assert link in body
    assert "Everyday settings" in body


def test_default_activates_the_hub_and_nav_cutover(
    session_factory: sessionmaker[Session],
) -> None:
    # P6b (#161) flipped the default on; pin it so an accidental flip-back is
    # caught, and pin the /resources -> /settings/status nav cutover at the exact
    # edge this slice moved (otherwise a regression to the old link would still
    # "work" through the redirect, just with a wasted 303).
    default_settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        csrf_secret="settings-page-test-csrf-key",
    )
    assert default_settings.console_settings_enabled is True
    body = _client(session_factory).get("/settings").text  # no flag override
    assert "Everyday settings" in body  # the hub, not the flat page
    assert 'href="/settings/status"' in body  # status sub-page link in hub nav
    assert 'href="/settings/hardware"' in body  # hardware sub-page link in hub nav
    assert 'href="/resources"' not in body  # the old link is gone


def test_flag_on_post_error_rerenders_hub(
    session_factory: sessionmaker[Session],
) -> None:
    # An invariant violation re-renders in place (200, not a redirect). With the
    # flag on that re-render must be the hub, so the operator does not jump to the
    # flat page mid-edit. names_llm on + names off (with env LLM off) violates the
    # cross-flag invariants, so nothing is written and the page re-renders.
    client = _client(session_factory, console_settings_enabled=True)
    resp = client.post(
        "/settings/features",
        data={
            "csrf_token": mint_csrf_token("settings-page-test-csrf-key", CSRF_SETTINGS),
            "enrichment_names_enabled": "off",
            "enrichment_names_llm_enabled": "on",
        },
    )
    assert resp.status_code == 200
    assert "Everyday settings" in resp.text  # hub chrome, not the flat page


# --------------------------------------------------------------------------- #
# Sub-pages are always registered (both flag states) and fail soft.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("flag", [False, True])
@pytest.mark.parametrize(
    "path", ["/settings/status", "/settings/hardware", "/settings/database", "/settings/plugins"]
)
def test_subpages_reachable_regardless_of_flag(
    session_factory: sessionmaker[Session], flag: bool, path: str
) -> None:
    client = _client(session_factory, console_settings_enabled=flag)
    assert client.get(path).status_code == 200


def test_status_page_reports_health_and_unknown_install(
    session_factory: sessionmaker[Session], tmp_path: Path,
) -> None:
    # No model services in the test env, so this proves the doctor checks render
    # and the page is a 200 even with dependencies down (fail-soft).
    client = _client(session_factory, media_root=tmp_path)
    body = client.get("/settings/status").text
    assert "PARTS OF VOXINT" in body
    assert "Install type unknown" in body
    assert "Console &amp; API" in body
    assert "Database" in body
    assert "Processor" in body
    assert "Memory" in body
    assert "Disk (media)" in body


def test_hardware_page_shows_env_keys(
    session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Pin the environment re-read to a clean default Settings so the test does not
    # pick up a repo-root .env and see a spurious "restart pending".
    monkeypatch.setattr(
        "voxint.api.settings_view.Settings",
        lambda: Settings(_env_file=None),  # type: ignore[call-arg]
    )
    client = _client(session_factory)
    body = client.get("/settings/hardware").text
    assert "ASR_URL" in body and "COMPUTE_TIER" in body
    # No environment change after boot, so nothing is pending.
    assert "restart pending" not in body
    assert "Could not check for pending changes" not in body


def test_restart_check_failed_is_honest_when_env_unreadable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # If the environment cannot be re-read, the view says so rather than claiming
    # nothing is pending (a false negative would hide a real change).
    def _boom() -> Settings:
        raise ValueError("bad .env")

    monkeypatch.setattr("voxint.api.settings_view.Settings", _boom)
    boot = Settings(_env_file=None)  # type: ignore[call-arg]
    view = build_hardware_view(boot, (), environ_settings=None)
    assert view.restart_check_failed is True
    assert view.restart_pending is False


def test_database_page_shows_size_or_honest_failure(
    session_factory: sessionmaker[Session],
) -> None:
    client = _client(session_factory)
    body = client.get("/settings/database").text
    # The real migrated test DB is up, so size renders (not the failure state).
    assert "Total database size" in body
    assert "Retention and cleanup" in body


# --------------------------------------------------------------------------- #
# Restart-pending is the honest configured-vs-environment comparison. Proven at
# the builder with two Settings so it is deterministic (a fresh Settings() would
# otherwise pick up any repo-root .env); this IS "flip an env-resolved value".
# --------------------------------------------------------------------------- #
def test_restart_pending_flips_when_env_value_changes() -> None:
    boot = Settings(_env_file=None, compute_tier="gpu")  # type: ignore[call-arg]
    unchanged = build_hardware_view(boot, (), environ_settings=boot)
    assert unchanged.restart_pending is False

    changed_env = Settings(_env_file=None, compute_tier="cpu")  # type: ignore[call-arg]
    pending = build_hardware_view(boot, (), environ_settings=changed_env)
    assert pending.restart_pending is True
    tier = next(f for f in pending.fields if f.env_key == "COMPUTE_TIER")
    assert tier.running == "gpu" and tier.pending == "cpu"


def test_hardware_view_redacts_url_credentials() -> None:
    boot = Settings(  # type: ignore[call-arg]
        _env_file=None, asr_url="http://user:secret@asr.local:8022"
    )
    view = build_hardware_view(boot, (), environ_settings=boot)
    asr = next(f for f in view.fields if f.env_key == "ASR_URL")
    assert "secret" not in asr.running and "asr.local" in asr.running


# --------------------------------------------------------------------------- #
# Plugins: honest empty state, kill-switch cases, and a real per-plugin page.
# --------------------------------------------------------------------------- #
def test_plugins_page_shows_synthdetect(
    session_factory: sessionmaker[Session],
) -> None:
    client = _client(session_factory)
    body = client.get("/settings/plugins").text
    assert "Synthetic Speech Detection" in body


def test_plugins_page_reports_unknown_kill_switch_id(
    session_factory: sessionmaker[Session],
) -> None:
    client = _client(session_factory, voxint_plugins_disabled="ghostplugin")
    body = client.get("/settings/plugins").text
    assert "Unknown ids in the kill switch" in body
    assert "ghostplugin" in body


def test_plugin_detail_unknown_id_404(session_factory: sessionmaker[Session]) -> None:
    client = _client(session_factory)
    assert client.get("/settings/plugins/nope").status_code == 404


class RenderPlugin(VoxintPlugin):
    manifest = PluginManifest(id="fakeplug", name="Fake Plugin", description="d")

    def enabled(self, row: object, settings: object) -> bool:
        return True

    def settings_section(self) -> SettingsSection:
        return SettingsSection(
            section_id="fakeplug", title="Fake", template="fakeplug/section.html"
        )

    def build_router(self, deps: PluginRouteDeps) -> APIRouter:
        router = APIRouter()

        @router.get("/plugins/fakeplug/ping")
        def ping() -> JSONResponse:
            return JSONResponse({"ok": True})

        return router


def test_plugin_detail_renders_active_plugin_section(
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tdir = tmp_path / "fakeplug_templates"
    tdir.mkdir()
    # The sentinel renders only when the section receives the full settings
    # context (csrf_settings), proving the detail page gives a plugin section the
    # same context it gets inline on the hub (so a real CSRF-guarded form works).
    (tdir / "section.html").write_text(
        '<section id="fakeplug"><h2>FAKE-SECTION-MARKER</h2>'
        "{% if csrf_settings %}<p>CSRF-CONTEXT-OK</p>{% endif %}</section>"
    )
    monkeypatch.setattr("voxint.plugins.BUILTIN", (RenderPlugin,))
    monkeypatch.setattr(
        "voxint.api.app._plugin_template_dirs",
        lambda registry: {"fakeplug": str(tdir)},
    )
    reset_plugins_cache()
    client = _client(session_factory)
    # Listed on the registry page, linking to its detail page.
    listing = client.get("/settings/plugins").text
    assert "Fake Plugin" in listing
    assert "/settings/plugins/fakeplug" in listing
    # Detail page renders the plugin's own contributed section under full context.
    detail = client.get("/settings/plugins/fakeplug").text
    assert "FAKE-SECTION-MARKER" in detail
    assert "CSRF-CONTEXT-OK" in detail


def test_service_identity_view_renders_on_hardware(
    session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Inject a validated identity so the reused _models.html panel renders a
    # concrete model, proving the hardware page's model panel is wired.
    view = ServiceIdentityView(
        role="asr",
        label="Transcription",
        url="http://asr.local:8022",
        reachable=True,
        model="large-v2",
        revision=None,
        engine="faster-whisper",
        configurable=True,
        verdict="validated",  # type: ignore[arg-type]
        identity_axis=None,
        detail=None,
        env_keys=("WHISPER_MODEL",),
    )
    monkeypatch.setattr(
        "voxint.api.routers.settings.collect_service_identity", lambda settings: [view]
    )
    client = _client(session_factory)
    body = client.get("/settings/hardware").text
    assert "large-v2" in body


def test_status_page_splits_the_two_ai_lanes(
    session_factory: sessionmaker[Session], tmp_path: Path,
) -> None:
    """#316: LLM work enabled but the BYO endpoint untouched (the default URL,
    no key). The page must NOT probe-and-warn the endpoint the operator never
    chose: the BYO row reads "not configured" (off dot), the bundled row is
    present (off here - no bundle in the test env), and the old conflated
    "Local AI model" label is gone. Fully offline: neither lane is probed."""
    client = _client(session_factory, media_root=tmp_path, llm_enabled=True)
    # The onboarded row wins over env for enablement; flip it on there too.
    seed_onboarded(session_factory, llm_enabled=True)
    body = client.get("/settings/status").text
    assert "Bundled AI model" in body
    assert "Your own AI endpoint" in body
    assert "Local AI model" not in body
    assert "not configured" in body
    assert "rejected (HTTP 401)" not in body
    assert "Set up" in body


def test_status_page_default_install_shows_both_ai_rows_off(
    session_factory: sessionmaker[Session], tmp_path: Path,
) -> None:
    client = _client(session_factory, media_root=tmp_path)
    body = client.get("/settings/status").text
    assert "Bundled AI model" in body
    assert "Your own AI endpoint" in body
    assert "off -- used for polish &amp; profiles" in body
    assert "Turn on" in body
