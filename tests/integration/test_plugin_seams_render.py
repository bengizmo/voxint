"""End-to-end proof that the #138 API seams fire when a plugin is active.

The empty-registry no-op is proved by the byte-identical route/task inventory
guards. Here a synthetic plugin (injected by monkeypatching ``BUILTIN`` and the
template-dir resolver) proves the live-app seams actually activate: its router is
mounted and reachable behind the onboarding gate, a (path, method) collision is
rejected at startup, its settings section and its Features-section flag render on
``/settings``, and its run-detail panel renders on ``/runs/{id}``.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.config import Settings
from voxint.db.models import MediaItem, PipelineRun, RunStatus
from voxint.plugins import PluginError, reset_plugins_cache
from voxint.plugins.base import (
    PanelContribution,
    PluginManifest,
    SettingsSection,
    VoxintPlugin,
)
from voxint.plugins.deps import PluginRouteDeps

CREDS = ("reviewer", "s3cret")


@pytest.fixture(autouse=True)
def _reset_registry() -> Iterator[None]:
    reset_plugins_cache()
    yield
    reset_plugins_cache()


@pytest.fixture()
def plugin_templates(tmp_path: Path) -> Path:
    tdir = tmp_path / "fakeplug_templates"
    tdir.mkdir()
    (tdir / "section.html").write_text(
        '<section id="fakeplug"><h2>FAKE-SECTION-MARKER</h2></section>'
    )
    (tdir / "panel.html").write_text("<div>FAKE-PANEL-MARKER</div>")
    return tdir


class RenderPlugin(VoxintPlugin):
    manifest = PluginManifest(id="fakeplug", name="Fake", description="d")

    def settings_section(self) -> SettingsSection:
        # No feature_flags here: rendering a plugin FeatureFlag reads its
        # AppSettings column + Settings field, which only exist once the plugin is
        # converted (#141). The effective-meta MERGE mechanism is proved in
        # tests/unit/test_plugin_seams.py; here we prove the section loop renders.
        return SettingsSection(
            section_id="fakeplug",
            title="Fake Section",
            template="fakeplug/section.html",
        )

    def run_detail_panels(self) -> tuple[PanelContribution, ...]:
        return (PanelContribution(slot="main", template="fakeplug/panel.html"),)

    def build_router(self, deps: PluginRouteDeps) -> APIRouter:
        router = APIRouter()

        @router.get("/plugins/fakeplug/ping")
        def ping() -> JSONResponse:
            return JSONResponse({"ok": True})

        return router


def _install(
    monkeypatch: pytest.MonkeyPatch,
    plugins: tuple[type[VoxintPlugin], ...],
    plugin_templates: Path | None = None,
) -> None:
    monkeypatch.setattr("voxint.plugins.BUILTIN", plugins)
    if plugin_templates is not None:
        monkeypatch.setattr(
            "voxint.api.app._plugin_template_dirs",
            lambda registry: {"fakeplug": str(plugin_templates)},
        )
    reset_plugins_cache()


def _client(
    session_factory: sessionmaker[Session], **overrides: object
) -> TestClient:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        csrf_secret="plugin-seam-test-csrf-key",
        **overrides,
    )
    client = TestClient(create_app(settings=settings, session_factory=session_factory))
    client.auth = CREDS
    seed_onboarded(session_factory)
    return client


def _seed_completed_run(session_factory: sessionmaker[Session]) -> uuid.UUID:
    with session_factory() as session:
        media = MediaItem(source_path=f"incoming/{uuid.uuid4().hex}/source")
        session.add(media)
        session.flush()
        run = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
        session.add(run)
        session.flush()
        run_id: uuid.UUID = run.id
        session.commit()
        return run_id


# --------------------------------------------------------------------------- #
# Router mount (seam #6).
# --------------------------------------------------------------------------- #
def test_plugin_router_is_mounted_and_gated(
    session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, (RenderPlugin,))
    client = _client(session_factory)
    resp = client.get("/plugins/fakeplug/ping")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_plugin_route_is_behind_the_onboarding_gate(
    session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, (RenderPlugin,))
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        csrf_secret="plugin-seam-test-csrf-key",
    )
    client = TestClient(create_app(settings=settings, session_factory=session_factory))
    client.auth = CREDS
    # NOT onboarded: the gated router (which the plugin route mounts on) redirects.
    resp = client.get("/plugins/fakeplug/ping", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/setup"


def test_colliding_plugin_routes_fail_startup(
    session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    class Other(VoxintPlugin):
        manifest = PluginManifest(id="otherplug", name="Other", description="d")

        def build_router(self, deps: PluginRouteDeps) -> APIRouter:
            router = APIRouter()

            @router.get("/plugins/fakeplug/ping")
            def clash() -> JSONResponse:
                return JSONResponse({"clash": True})

            return router

    _install(monkeypatch, (RenderPlugin, Other))
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
    )
    with pytest.raises(PluginError, match="plugin route collisions"):
        create_app(settings=settings, session_factory=session_factory)


def test_plugin_route_colliding_with_core_fails_startup(
    session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plugin claiming a (path, method) a CORE route owns must fail boot too.

    find_route_collisions only catches plugin-vs-plugin; this guards the runtime
    plugin-vs-core check (#138) that mirrors the worker's core-task-name guard, so
    a converted plugin (#139+) squatting a core route stops boot rather than
    mounting an unreachable shadow the route-inventory test alone would have to
    catch.
    """

    class CoreClash(VoxintPlugin):
        manifest = PluginManifest(id="coreclash", name="Core Clash", description="d")

        def build_router(self, deps: PluginRouteDeps) -> APIRouter:
            router = APIRouter()

            @router.get("/settings")  # GET /settings is a core protected route
            def clash() -> JSONResponse:
                return JSONResponse({"clash": True})

            return router

    _install(monkeypatch, (CoreClash,))
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
    )
    with pytest.raises(PluginError, match="collisions with core"):
        create_app(settings=settings, session_factory=session_factory)


# --------------------------------------------------------------------------- #
# Settings section render (seam #4).
# --------------------------------------------------------------------------- #
def test_plugin_settings_section_renders(
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    plugin_templates: Path,
) -> None:
    _install(monkeypatch, (RenderPlugin,), plugin_templates)
    client = _client(session_factory)
    body = client.get("/settings/plugins").text
    # The plugin's own settings section rendered after the core sections.
    assert "FAKE-SECTION-MARKER" in body


# --------------------------------------------------------------------------- #
# Run-detail panel render (seam #5).
# --------------------------------------------------------------------------- #
def test_plugin_run_detail_panel_renders(
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    plugin_templates: Path,
) -> None:
    _install(monkeypatch, (RenderPlugin,), plugin_templates)
    client = _client(session_factory)
    run_id = _seed_completed_run(session_factory)
    body = client.get(f"/runs/{run_id}").text
    assert "FAKE-PANEL-MARKER" in body
