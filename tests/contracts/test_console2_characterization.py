"""Characterization contracts for the Console 2.0 refactor (issue #150).

P0b decomposes the ~7.5k-line ``api/app.py`` into per-area routers. That move
must be behavior-preserving, and later phases retire legacy routes behind
redirects. These tests pin the CURRENT observable behavior so a decomposition
that renames, drops, or re-gates a route, or a new router that forgets the CSRF
wiring, fails loudly rather than silently.

Three characterizations, all read off the FastAPI route table (no HTTP, no DB),
so they run in the plain unit lane and survive routers moving between modules:

1. Route characterization golden: path, methods, onboarding gate, operator auth
   for every route, read from each route's dependency tree so it survives a route
   moving between routers. Pins the surface without pinning which module defines it.
2. CSRF coverage, in two layers: a WIRING check that every mutating route accepts
   a ``csrf_token`` or review-claim ``token`` field, and an ENFORCEMENT check that
   the handler body actually calls a defense verifier (so a route that accepts a
   token and never checks it is caught, which field presence alone cannot do).
3. Redirect map: the declarative ``REDIRECT_MAP`` seam future phases extend, with
   a structural guard that each declared redirect has a real source route. A live
   HTTP check of the same table lives in ``tests/integration/test_console2_redirects``.
4. Registration order: the route table in traversal (= matching) order. The
   sorted golden above cannot see a reorder, and Starlette matches in
   registration order, so the P0b router moves pin the order separately.
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
from voxint.api.routers.deps import require_onboarded

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from fastapi.dependencies.models import Dependant

_GOLDEN = (
    REPO_ROOT / "tests" / "contracts" / "fixtures" / "console2_route_characterization.json"
)

# HTTP methods that change state and therefore need a mutation defense.
_MUTATING = {"POST", "PUT", "PATCH", "DELETE"}

# The form fields that carry a mutation defense (app.py: the URL-borne claim
# ``token`` is "both the review lock and the CSRF defense for claim-gated
# mutations", so a review route protected by it needs no separate csrf_token).
# Presence of the field is a WIRING signal only, not proof of enforcement; the
# enforcement contract is _VERIFICATION_SINKS below.
_CSRF_FIELDS = {"csrf_token", "token"}

# The functions that actually VERIFY a mutation defense: the CSRF-token check and
# the claim-token verifiers (each raises on a bad or missing token). A mutating
# handler that does not call one of these in its body is unprotected even if it
# accepts a token field. A new verifier is added here deliberately, as a
# reviewable change; a route that switches to one not listed here fails the
# enforcement test until it is added (the safe direction).
_VERIFICATION_SINKS = (
    "_require_csrf",
    "verify_claim",
    "_verify_annotation_claim",
    "release_run",
)

# Mutating routes that intentionally carry no mutation defense because they write
# nothing (a POST body is only how the client ships a large selection). Each entry
# needs the "persists nothing" reason recorded at the handler. Keep this minimal:
# a genuinely new mutating route must NOT be added here, so the coverage tests
# catch it.
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


def _iter_api_routes(routes: Iterable[object]) -> Iterator[APIRoute]:
    """Every APIRoute reachable from ``routes``, descending included sub-routers.

    FastAPI mounts an included router as a sub-route reachable through
    ``original_router``, and P0b nests those (``console.include_router(area)``),
    so the walk recurses rather than descending a single level. Gate and auth are
    then read from each route's dependency tree, not from where it sits in the
    mount topology, so a route keeps its classification whichever router holds it.
    """
    for route in routes:
        if isinstance(route, APIRoute):
            yield route
        elif hasattr(route, "original_router"):
            yield from _iter_api_routes(route.original_router.routes)


def console_route_records(app: FastAPI) -> list[RouteRecord]:
    """Every APIRoute as ``(path, methods, gate, auth)``, sorted.

    ``gate`` is ``onboarding`` when ``require_onboarded`` is in the route's
    dependency tree (the onboarding gate is a router-level dependency FastAPI
    propagates onto each route) and ``exempt`` otherwise. ``auth`` is whether
    ``require_operator`` is in the tree. Both are read from the dependency graph,
    so a behavior-preserving router move does not change the record.
    """
    records: list[RouteRecord] = []
    for r in _iter_api_routes(app.routes):
        methods = sorted(m for m in r.methods if m != "HEAD")
        calls = list(_walk_calls(r.dependant))
        gate = "onboarding" if any(c is require_onboarded for c in calls) else "exempt"
        auth = any(c is require_operator for c in calls)
        records.append(RouteRecord(r.path, methods, gate, auth))
    return sorted(records)


_ORDER_GOLDEN = (
    REPO_ROOT / "tests" / "contracts" / "fixtures" / "console2_route_order.json"
)


def test_route_registration_order_matches_golden() -> None:
    """The route table in traversal order — the order Starlette matches in.

    The characterization golden is sorted, so it cannot detect two routes
    swapping registration positions. Order is contract during the P0b
    decomposition (the issue requires registration order preserved), and
    ``_iter_api_routes`` yields depth-first in registration order, which is the
    order request matching walks.
    """
    golden = json.loads(_ORDER_GOLDEN.read_text())
    actual = [
        [r.path, sorted(m for m in r.methods if m != "HEAD")]
        for r in _iter_api_routes(create_app().routes)
    ]
    assert actual == golden, (
        "console route REGISTRATION ORDER changed. A behavior-preserving router "
        f"move must keep it; if a reorder is deliberate, regenerate "
        f"{_ORDER_GOLDEN.relative_to(REPO_ROOT)} and justify it in the commit."
    )


def _mutating_routes(app: FastAPI) -> list[APIRoute]:
    return [r for r in _iter_api_routes(app.routes) if _MUTATING & set(r.methods)]


def test_route_characterization_matches_golden() -> None:
    golden = json.loads(_GOLDEN.read_text())
    actual = [list(rec) for rec in console_route_records(create_app())]
    assert actual == golden, (
        "console route characterization changed (path/methods/gate/auth). If this "
        f"is intentional, regenerate {_GOLDEN.relative_to(REPO_ROOT)}; otherwise a "
        "route was renamed, dropped, re-gated, or lost its auth dependency."
    )


def _has_defense_field(route: APIRoute) -> bool:
    """The handler accepts a csrf_token or review-claim token form field."""
    return bool(_CSRF_FIELDS & set(inspect.signature(route.endpoint).parameters))


def _verifies_a_defense(route: APIRoute) -> bool:
    """The handler body calls one of the mutation-defense verifiers.

    This reads the handler source, so it proves a verification call exists rather
    than merely that a token field is accepted. It sees only the handler's own
    body: every current mutating handler calls its verifier directly there (a
    delegated verifier would need adding to _VERIFICATION_SINKS or would fail this
    check, which is the safe direction).
    """
    try:
        source = inspect.getsource(route.endpoint)
    except OSError:
        return False
    return any(sink in source for sink in _VERIFICATION_SINKS)


def mutation_gaps(
    app: FastAPI, is_covered: Callable[[APIRoute], bool]
) -> tuple[list[str], set[tuple[str, str]]]:
    """Return ``(uncovered, matched_exemptions)`` for the app's mutating routes.

    ``uncovered`` is every mutating route for which ``is_covered`` is false and
    which is not on the read-only allowlist; ``matched_exemptions`` is the
    allowlist entries actually hit, so a stale entry is detectable.
    """
    uncovered: list[str] = []
    seen_exempt: set[tuple[str, str]] = set()
    for route in _mutating_routes(app):
        if is_covered(route):
            continue
        methods = sorted(m for m in route.methods if m != "HEAD")
        matched = {(m, route.path) for m in methods} & _CSRF_EXEMPT_READONLY
        if matched:
            seen_exempt |= matched
            continue
        uncovered.append(f"{','.join(methods)} {route.path}")
    return uncovered, seen_exempt


def _assert_no_gaps(
    is_covered: Callable[[APIRoute], bool], what: str
) -> None:
    uncovered, seen_exempt = mutation_gaps(create_app(), is_covered)
    assert not uncovered, f"mutating route(s) with no {what}: " + "; ".join(
        sorted(uncovered)
    )
    stale = _CSRF_EXEMPT_READONLY - seen_exempt
    assert not stale, f"stale _CSRF_EXEMPT_READONLY entr(ies): {sorted(stale)}"


def test_every_mutating_route_exposes_a_defense_field() -> None:
    """WIRING check: every state-changing route accepts a csrf_token or claim token.

    Field presence is only a smoke signal, not proof of enforcement (a handler
    could accept the field and ignore it). It survives P0b because a handler keeps
    its form fields whichever module it moves to, and it fails fast on a new route
    that forgets the field entirely. The enforcement contract is the next test.
    """
    _assert_no_gaps(_has_defense_field, "csrf_token / review-claim token field")


def test_every_mutating_route_verifies_its_defense() -> None:
    """ENFORCEMENT check: every state-changing route calls a defense verifier.

    Reads each handler's source and asserts it calls one of _VERIFICATION_SINKS
    (the CSRF check or a claim verifier). This catches the case a field-presence
    check cannot: a handler that accepts a correctly named token and never
    verifies it. Only the read-only allowlist is exempt.
    """
    _assert_no_gaps(_verifies_a_defense, "mutation-defense verifier call in its body")


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


def _synthetic_app(kind: Literal["verified", "field_only", "none"]) -> FastAPI:
    from typing import Annotated

    from fastapi import Depends, Form

    from voxint.api.auth import require_operator

    def _require_csrf(*_a: object) -> None:  # stand-in for the real verifier name
        return None

    app = FastAPI()

    if kind == "verified":

        @app.post("/thing", dependencies=[Depends(require_operator)])
        def _mutate(csrf_token: Annotated[str, Form()]) -> dict[str, str]:
            _require_csrf(csrf_token)
            return {"ok": csrf_token}
    elif kind == "field_only":
        # Accepts the field but never verifies it: the exact false positive a
        # field-presence check misses.
        @app.post("/thing", dependencies=[Depends(require_operator)])
        def _mutate_unverified(csrf_token: Annotated[str, Form()]) -> dict[str, str]:
            return {"ok": csrf_token}
    else:

        @app.post("/thing", dependencies=[Depends(require_operator)])
        def _mutate_none(payload: Annotated[str, Form()]) -> dict[str, str]:
            return {"ok": payload}

    return app


def test_wiring_check_catches_a_missing_field() -> None:
    assert mutation_gaps(_synthetic_app("verified"), _has_defense_field)[0] == []
    assert mutation_gaps(_synthetic_app("field_only"), _has_defense_field)[0] == []
    assert mutation_gaps(_synthetic_app("none"), _has_defense_field)[0] == ["POST /thing"]


def test_enforcement_check_catches_a_declared_but_unverified_defense() -> None:
    # The point the wiring check cannot make: a handler that accepts a correctly
    # named token and never verifies it must still be flagged.
    assert mutation_gaps(_synthetic_app("verified"), _verifies_a_defense)[0] == []
    assert mutation_gaps(_synthetic_app("field_only"), _verifies_a_defense)[0] == [
        "POST /thing"
    ]


def _gated_route_app() -> FastAPI:
    from fastapi import Depends

    app = FastAPI()

    @app.get("/gated", dependencies=[Depends(require_onboarded)])
    def _g() -> dict[str, str]:
        return {}

    return app


def test_gate_is_read_from_dependencies_not_topology() -> None:
    """A require_onboarded route registered directly on the app (no original_router
    wrapper) must still classify as onboarding. The old mount-shape heuristic would
    have mislabeled it exempt, breaking the golden on a behavior-preserving move."""
    records = {rec.path: rec for rec in console_route_records(_gated_route_app())}
    assert records["/gated"].gate == "onboarding"


def test_route_records_capture_gate_and_auth() -> None:
    """The enumerator distinguishes exempt/onboarding and auth/no-auth, so the
    golden would catch a route silently re-gated or stripped of auth."""
    by_path = {rec.path: rec for rec in console_route_records(create_app())}
    # /healthz is the one exempt, unauthenticated liveness route.
    assert by_path["/healthz"].gate == "exempt"
    assert by_path["/healthz"].auth is False
    # A protected, operator-gated route (the review submit).
    assert by_path["/submit"].gate == "onboarding"
    assert by_path["/submit"].auth is True
