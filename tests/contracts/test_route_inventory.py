"""Contract: the console's HTTP route inventory is frozen (issue #137).

The plugin epic (#136) moves whole feature areas — translation, semantic search,
LLM enrichment — out of ``api/app.py`` and into plugin packages. Route paths and
methods are frozen contract across that move: after a conversion the *all-enabled*
inventory (core + every plugin) must be byte-identical to this golden, and the
*core-only* inventory (every plugin killed) must be exactly this set minus the
converted routes. This test pins the baseline so a conversion that accidentally
renames, drops, or duplicates a route fails loudly.

The framework is dormant in #137 (empty registry), so all-enabled and core-only
are the same app; :func:`app_route_inventory` is the shared enumerator the
seam-wiring and conversion tests (#138+) reuse to build both.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute

from tests.contracts.conftest import REPO_ROOT
from voxint.api.app import create_app
from voxint.plugins import load_registry

_GOLDEN = REPO_ROOT / "tests" / "contracts" / "fixtures" / "route_inventory.json"


def _iter_api_routes(routes: Iterable[object]) -> list[APIRoute]:
    """Every APIRoute reachable from ``routes``, descending mounted sub-routers.

    FastAPI mounts an included ``APIRouter`` as a sub-route exposing
    ``original_router`` rather than flattening it, so the gated ``protected``
    router — and each plugin router included *under* it (#138) — nests one level
    deeper each. A single-level walk misses the plugin routes, which would let the
    all-enabled inventory silently drop a converted plugin's routes (#139+); recurse
    so a conversion that moves a route into a plugin still shows it here.
    """
    found: list[APIRoute] = []
    for route in routes:  # type: ignore[attr-defined]
        if isinstance(route, APIRoute):
            found.append(route)
        else:
            sub = getattr(route, "original_router", None)
            if sub is not None:
                found.extend(_iter_api_routes(sub.routes))
    return found


def app_route_inventory(app: FastAPI) -> list[list[object]]:
    """``[path, [methods…]]`` for every APIRoute, sorted and HEAD-stripped.

    Recurses through every mounted sub-router (:func:`_iter_api_routes`), so the
    gated ``protected`` routes and any plugin routes nested under it are all
    captured — the all-enabled vs core-only comparison the conversions rely on.
    """
    routes = _iter_api_routes(app.routes)
    inventory = [
        [r.path, sorted(m for m in r.methods if m != "HEAD")] for r in routes
    ]
    return sorted(inventory)


def test_route_inventory_matches_golden() -> None:
    golden = json.loads(_GOLDEN.read_text())
    actual = app_route_inventory(create_app())
    # Compare as sorted lists so the failure diff names the exact drift.
    assert actual == golden, (
        "console route inventory changed. If this is intentional, regenerate "
        f"{_GOLDEN.relative_to(REPO_ROOT)}; otherwise a route was renamed, dropped, "
        "or duplicated."
    )


def test_framework_is_dormant() -> None:
    # #137 lands the framework with an empty builtin set, so core-only and
    # all-enabled route inventories are the same app. #138 wires the registry into
    # create_app and this invariant becomes the with/without-plugins comparison.
    assert load_registry().plugins == ()


def test_inventory_captures_nested_plugin_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plugin route nested under ``protected`` must appear in the inventory.

    Guards the recursive walk (#138): plugin routers mount as sub-routers two
    levels deep, so a single-level walk drops them silently — which would make the
    all-enabled vs core-only comparison the conversions rely on meaningless (a
    converted route would vanish from the inventory rather than move). If this
    regresses, the golden inventories stop guarding conversions.
    """
    from fastapi import APIRouter
    from fastapi.responses import JSONResponse

    from voxint.plugins import reset_plugins_cache
    from voxint.plugins.base import PluginManifest, VoxintPlugin
    from voxint.plugins.deps import PluginRouteDeps

    class InventoryPlugin(VoxintPlugin):
        manifest = PluginManifest(id="invplug", name="Inv", description="d")

        def build_router(self, deps: PluginRouteDeps) -> APIRouter:
            router = APIRouter()

            @router.get("/plugins/invplug/ping")
            def ping() -> JSONResponse:
                return JSONResponse({"ok": True})

            return router

    monkeypatch.setattr("voxint.plugins.BUILTIN", (InventoryPlugin,))
    reset_plugins_cache()
    try:
        inventory = app_route_inventory(create_app())
    finally:
        reset_plugins_cache()
    assert ["/plugins/invplug/ping", ["GET"]] in inventory
