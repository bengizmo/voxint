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

from fastapi import FastAPI
from fastapi.routing import APIRoute

from tests.contracts.conftest import REPO_ROOT
from voxint.api.app import create_app
from voxint.plugins import load_registry

_GOLDEN = REPO_ROOT / "tests" / "contracts" / "fixtures" / "route_inventory.json"


def app_route_inventory(app: FastAPI) -> list[list[object]]:
    """``[path, [methods…]]`` for every APIRoute, sorted and HEAD-stripped.

    Walks included routers through ``original_router`` (FastAPI mounts an
    included router as a sub-route, so its APIRoutes are not on ``app.routes``),
    recursively: P0b nests per-area routers inside the ``protected`` router, so
    a single-level walk would silently drop the nested family routes.
    """

    def collect(candidates: Iterable[object], into: list[APIRoute]) -> None:
        for route in candidates:
            if isinstance(route, APIRoute):
                into.append(route)
            elif hasattr(route, "original_router"):
                collect(route.original_router.routes, into)

    routes: list[APIRoute] = []
    collect(app.routes, routes)
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
