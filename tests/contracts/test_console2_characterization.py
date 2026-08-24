"""Characterization contracts for the Console 2.0 refactor (issue #150).

P0b decomposes the ~7.5k-line ``api/app.py`` into per-area routers. That move
must be behavior-preserving, and later phases retire legacy routes behind
redirects. These tests pin the CURRENT observable behavior so a decomposition
that renames, drops, or re-gates a route, or a new router that forgets the CSRF
wiring, fails loudly rather than silently.

Three characterizations, all read off the FastAPI route table (no HTTP, no DB),
so they run in the plain unit lane and survive routers moving between modules:

1. Route characterization golden: path, methods, onboarding gate, operator auth
   for every route. Pins the surface without pinning which module defines it.
2. CSRF coverage: every mutating route carries one of the two mutation defenses
   (the ``csrf_token`` form token, or the ``token`` review-claim token that is
   itself the CSRF defense for claim-gated review mutations).
3. Redirect map: the declarative ``REDIRECT_MAP`` seam future phases extend, with
   a structural guard that each declared redirect has a real source route. A live
   HTTP check of the same table lives in ``tests/integration/test_console2_redirects``.
"""

from __future__ import annotations

import inspect
import json
from typing import TYPE_CHECKING, Literal, NamedTuple

from fastapi import FastAPI
from fastapi.routing import APIRoute

from tests.contracts.conftest import REPO_ROOT
from voxint.api.app import create_app
from voxint.api.auth import require_operator

if TYPE_CHECKING:
    from collections.abc import Iterator

    from fastapi.dependencies.models import Dependant

_GOLDEN = (
    REPO_ROOT / "tests" / "contracts" / "fixtures" / "console2_route_characterization.json"
)

# HTTP methods that change state and therefore need a mutation defense.
_MUTATING = {"POST", "PUT", "PATCH", "DELETE"}

# The two form fields that carry a mutation defense (app.py: the URL-borne claim
# ``token`` is "both the review lock and the CSRF defense for claim-gated
# mutations", so a review route protected by it needs no separate csrf_token).
_CSRF_FIELDS = {"csrf_token", "token"}

# Read-only POST routes that intentionally carry no mutation defense because they
# write nothing (a POST body is only how the client ships a large selection). Each
# entry needs the "persists nothing" reason recorded at the handler. Keep this
# minimal: a genuinely new mutating route must NOT be added here, so that the
# coverage test catches it.
_CSRF_EXEMPT_READONLY = frozenset(
    {
        # Renders an UNSAVED annotation selection to Markdown; persists nothing
        # (app.py export_live_pull_quote: "no claim, no nonce, no CSRF").
        ("POST", "/review/{run_id}/annotations/export/live.md"),
    }
)


class RouteRecord(NamedTuple):
    path: str
    methods: list[str]
    gate: Literal["onboarding", "exempt"]
    auth: bool


def _walk_calls(dependant: Dependant) -> Iterator[object]:
    yield dependant.call
    for sub in dependant.dependencies:
        yield from _walk_calls(sub)


def console_route_records(app: FastAPI) -> list[RouteRecord]:
    """Every APIRoute as ``(path, methods, gate, auth)``, sorted.

    ``gate`` is ``exempt`` for routes registered directly on ``app`` (liveness,
    the htmx asset, the setup wizard) and ``onboarding`` for routes on the
    ``protected`` router, which FastAPI mounts as a sub-route reachable through
    ``original_router``. ``auth`` is whether ``require_operator`` appears anywhere
    in the route's dependency tree.
    """
    records: list[RouteRecord] = []
    for route in app.routes:
        if isinstance(route, APIRoute):
            pairs: list[tuple[APIRoute, bool]] = [(route, False)]
        elif hasattr(route, "original_router"):
            pairs = [
                (r, True)
                for r in route.original_router.routes
                if isinstance(r, APIRoute)
            ]
        else:
            pairs = []
        for r, onboarding in pairs:
            methods = sorted(m for m in r.methods if m != "HEAD")
            auth = any(call is require_operator for call in _walk_calls(r.dependant))
            records.append(
                RouteRecord(
                    r.path, methods, "onboarding" if onboarding else "exempt", auth
                )
            )
    return sorted(records)


def _mutating_routes(app: FastAPI) -> list[APIRoute]:
    routes: list[APIRoute] = []
    for route in app.routes:
        candidates = (
            [route]
            if isinstance(route, APIRoute)
            else (
                [r for r in route.original_router.routes if isinstance(r, APIRoute)]
                if hasattr(route, "original_router")
                else []
            )
        )
        for r in candidates:
            if _MUTATING & set(r.methods):
                routes.append(r)
    return routes


def test_route_characterization_matches_golden() -> None:
    golden = json.loads(_GOLDEN.read_text())
    actual = [list(rec) for rec in console_route_records(create_app())]
    assert actual == golden, (
        "console route characterization changed (path/methods/gate/auth). If this "
        f"is intentional, regenerate {_GOLDEN.relative_to(REPO_ROOT)}; otherwise a "
        "route was renamed, dropped, re-gated, or lost its auth dependency."
    )


def mutation_defense_gaps(app: FastAPI) -> tuple[list[str], set[tuple[str, str]]]:
    """Return ``(uncovered, matched_exemptions)`` for the app's mutating routes.

    ``uncovered`` is every mutating route with no csrf_token/token field that is
    not on the read-only allowlist; ``matched_exemptions`` is the allowlist
    entries that were actually hit (so a stale entry is detectable).
    """
    uncovered: list[str] = []
    seen_exempt: set[tuple[str, str]] = set()
    for route in _mutating_routes(app):
        params = set(inspect.signature(route.endpoint).parameters)
        if _CSRF_FIELDS & params:
            continue
        methods = sorted(m for m in route.methods if m != "HEAD")
        matched = {(m, route.path) for m in methods} & _CSRF_EXEMPT_READONLY
        if matched:
            seen_exempt |= matched
            continue
        uncovered.append(f"{','.join(methods)} {route.path}")
    return uncovered, seen_exempt


def test_every_mutating_route_has_a_mutation_defense() -> None:
    """Every state-changing route accepts a csrf_token or a review-claim token.

    Signature presence is the wiring signal that survives P0b (a handler keeps
    its form fields whichever module it moves to). A new mutating router that
    forgets both defenses shows up here as an uncovered route.
    """
    uncovered, seen_exempt = mutation_defense_gaps(create_app())
    assert not uncovered, (
        "mutating route(s) with no CSRF defense (neither a csrf_token form field "
        "nor a review-claim token): " + "; ".join(sorted(uncovered))
    )
    # A stale allowlist entry (its route was removed or renamed) is itself drift.
    stale = _CSRF_EXEMPT_READONLY - seen_exempt
    assert not stale, f"stale _CSRF_EXEMPT_READONLY entr(ies): {sorted(stale)}"


# --- Redirect map: the seam future phases extend --------------------------


class RedirectRule(NamedTuple):
    """A legacy path the refactor redirects to a new location.

    ``status`` is the redirect code the source issues; ``auth`` marks whether the
    source sits behind the operator gate (drives the live HTTP check's setup).
    Future phases append rows here as they retire routes (P1 ``/dashboard`` -> ``/``,
    P3c ``/review`` -> ``/media/{id}``), keeping the redirect contract in one place.
    """

    source: str
    target: str
    status: int
    auth: bool


# Initially only the index redirect exists (issue #150: "empty of redirects
# beyond /"). ``/`` is unconditional once past the onboarding gate.
REDIRECT_MAP: tuple[RedirectRule, ...] = (
    RedirectRule(source="/", target="/review", status=303, auth=True),
)


def test_redirect_map_is_well_formed() -> None:
    sources = [rule.source for rule in REDIRECT_MAP]
    assert len(sources) == len(set(sources)), "duplicate redirect source(s)"
    for rule in REDIRECT_MAP:
        assert rule.status in {303, 307, 308}, f"{rule.source}: odd redirect status"
        assert rule.target and rule.target != rule.source, (
            f"{rule.source}: redirect target must be a different non-empty path"
        )


def test_redirect_sources_have_a_real_route() -> None:
    """A declared redirect must have a handler that issues it.

    Guards a future phase from declaring a legacy redirect whose source route it
    forgot to keep registered. The GET method is what a browser follows.
    """
    get_paths = {
        rec.path for rec in console_route_records(create_app()) if "GET" in rec.methods
    }
    missing = [rule.source for rule in REDIRECT_MAP if rule.source not in get_paths]
    assert not missing, f"redirect source(s) with no GET route: {missing}"


# --- Seeded-drift proofs: the checkers actually catch a regression ---------


def _synthetic_app(*, with_defense: bool) -> FastAPI:
    from typing import Annotated

    from fastapi import Depends, Form

    from voxint.api.auth import require_operator

    app = FastAPI()

    if with_defense:

        @app.post("/thing", dependencies=[Depends(require_operator)])
        def _mutate(csrf_token: Annotated[str, Form()]) -> dict[str, str]:
            return {"ok": csrf_token}
    else:

        @app.post("/thing", dependencies=[Depends(require_operator)])
        def _mutate_unprotected(payload: Annotated[str, Form()]) -> dict[str, str]:
            return {"ok": payload}

    return app


def test_csrf_check_catches_an_unprotected_mutation() -> None:
    covered, _ = mutation_defense_gaps(_synthetic_app(with_defense=True))
    assert covered == []
    uncovered, _ = mutation_defense_gaps(_synthetic_app(with_defense=False))
    assert uncovered == ["POST /thing"], (
        "the CSRF-coverage check must flag a mutating route that carries neither "
        "a csrf_token nor a review-claim token"
    )


def test_route_records_capture_gate_and_auth() -> None:
    """The enumerator distinguishes exempt/onboarding and auth/no-auth, so the
    golden would catch a route silently re-gated or stripped of auth."""
    records = console_route_records(create_app())
    by_path = {rec.path: rec for rec in records}
    # /healthz is the one exempt, unauthenticated liveness route.
    assert by_path["/healthz"].gate == "exempt"
    assert by_path["/healthz"].auth is False
    # A protected, operator-gated route (the review submit).
    assert by_path["/submit"].gate == "onboarding"
    assert by_path["/submit"].auth is True
