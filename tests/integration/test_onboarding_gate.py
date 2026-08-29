"""The first-run onboarding gate, end to end against real Postgres.

A not-onboarded operator is redirected to /setup on every non-exempt route (303
for a normal navigation, 204 + HX-Redirect for htmx), while authentication still
runs first (unauthenticated → 401, never a redirect that leaks onboarding state)
and /healthz, the htmx asset, and /setup stay reachable. Once onboarding is
completed — possibly by another process — the very next request passes, proving
the read is per-request and never cached across requests. A route-inventory guard
fails if a new route is added without the gate.
"""

import uuid

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.api.routers.deps import require_onboarded
from voxint.config import Settings

CREDS = ("reviewer", "s3cret")
_CSRF_KEY = "onboarding-gate-test-csrf-key"  # low-entropy; a known secret is fine here
_HTMX = {"HX-Request": "true"}
_ZERO = uuid.UUID(int=0)

# Routes deliberately reachable before onboarding: liveness, the htmx runtime the
# setup page itself needs, and the setup wizard the gate redirects TO. The wizard
# family is enumerated EXACTLY (not a blanket "/setup" prefix) so an accidentally
# ungated route under /setup would still fail this guard.
EXEMPT_PATHS = {
    "/healthz",
    "/static/htmx.min.js",
    # Island bundles (issue #48): on `app`, not the protected router, so they
    # load before onboarding completes (the setup wizard extends base.html).
    # Still auth-gated by the route's own OperatorDep, not open like /healthz.
    "/static/app/{asset_path:path}",
    "/setup",
    "/setup/folders/browse",
    "/setup/folders",
    "/setup/scan",
    "/setup/scan/confirm",
    "/setup/vocabulary",
    "/setup/llm",
    "/setup/finish",
    "/login",
    "/logout",
}


@pytest.fixture()
def client(session_factory: sessionmaker[Session]) -> TestClient:
    """A NOT-onboarded console (no app_settings row) — the gate is active."""
    settings = Settings(
        voxint_user=CREDS[0], voxint_password=CREDS[1], csrf_secret=_CSRF_KEY
    )
    test_client = TestClient(create_app(settings=settings, session_factory=session_factory))
    test_client.auth = CREDS
    return test_client


# ------------------------------------------------------------------ redirects


def test_normal_get_redirects_to_setup(client: TestClient) -> None:
    resp = client.get("/review", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/setup"


def test_htmx_get_redirects_via_header(client: TestClient) -> None:
    resp = client.get("/review", headers=_HTMX, follow_redirects=False)
    # htmx performs the client-side redirect off the header; a 303 body would be
    # swapped into the page instead, so the gate returns an empty 204.
    assert resp.status_code == 204
    assert resp.headers["hx-redirect"] == "/setup"
    assert resp.headers.get("location") is None
    assert resp.text == ""


def test_post_route_is_gated(client: TestClient) -> None:
    # A mutation route redirects too — the gate fires before the handler's own
    # CSRF/claim checks, so a not-onboarded POST never reaches them.
    resp = client.post(f"/review/{_ZERO}/claim", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/setup"


def test_head_media_route_is_gated(client: TestClient) -> None:
    resp = client.head(f"/media/{_ZERO}", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/setup"


def test_index_redirects_to_setup_when_not_onboarded(client: TestClient) -> None:
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/setup"


# ------------------------------------------------------------------ auth-first


def test_unauthenticated_gets_401_not_a_redirect(client: TestClient) -> None:
    # Auth is a prerequisite of the gate, so an anonymous request is challenged
    # rather than redirected — onboarding state never leaks to the unauthenticated.
    resp = client.get("/review", auth=None, follow_redirects=False)
    assert resp.status_code == 401
    assert "WWW-Authenticate" in resp.headers
    assert resp.headers.get("location") is None


# ------------------------------------------------------------------ exemptions


def test_healthz_open_without_onboarding(client: TestClient) -> None:
    resp = client.get("/healthz", auth=None)
    assert resp.status_code == 200


@pytest.mark.parametrize("path", ["/setup", "/static/htmx.min.js"])
def test_exempt_authed_routes_serve_without_onboarding(client: TestClient, path: str) -> None:
    resp = client.get(path, follow_redirects=False)
    assert resp.status_code == 200


# ------------------------------------------------- onboarded + cross-request


def test_onboarded_request_reaches_the_handler(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    seed_onboarded(session_factory)
    resp = client.get("/review", follow_redirects=False)
    assert resp.status_code == 200


def test_index_renders_home_when_onboarded(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    # P1 (#152): / is the real Home page now, not a redirect to /review.
    seed_onboarded(session_factory)
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code == 200
    assert "cb-breadcrumb" in resp.text
    assert "NEEDS YOUR ATTENTION" in resp.text


def test_gate_reads_onboarding_fresh_each_request(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    # No app.state cache: completing onboarding out of band (as the Celery worker
    # would, in its own process) is visible to the very next request.
    assert client.get("/review", follow_redirects=False).status_code == 303
    seed_onboarded(session_factory)
    assert client.get("/review", follow_redirects=False).status_code == 200


# ------------------------------------------------------------ inventory guard


def test_every_route_is_gated_or_explicitly_exempt() -> None:
    """A new console route must ride the protected router (or be listed exempt).

    Exemption is structural — a route is exempt iff it is registered on `app`
    rather than the protected router — so this walks every APIRoute and asserts it
    either carries the gate dependency or is one of the known-exempt paths.
    """
    app = create_app(Settings(voxint_user=CREDS[0], voxint_password=CREDS[1]))

    def dependency_calls(dependant: object) -> list[object]:
        found = [dependant.call]  # type: ignore[attr-defined]
        for sub in dependant.dependencies:  # type: ignore[attr-defined]
            found.extend(dependency_calls(sub))
        return found

    def collect(routes: list[object], into: list[APIRoute]) -> None:
        # This FastAPI mounts an included router as a sub-route; reach its
        # APIRoutes through original_router rather than app.routes. Recursive:
        # P0b nests per-area routers inside the protected router.
        for route in routes:
            if isinstance(route, APIRoute):
                into.append(route)
            elif hasattr(route, "original_router"):
                collect(route.original_router.routes, into)

    def api_routes() -> list[APIRoute]:
        routes: list[APIRoute] = []
        collect(app.routes, routes)
        return routes

    routes = api_routes()
    # Sanity: the walk actually found the console (not an empty set that would make
    # every assertion below vacuously pass).
    assert len(routes) >= 15
    for route in routes:
        gated = require_onboarded in dependency_calls(route.dependant)
        exempt = route.path in EXEMPT_PATHS
        assert gated != exempt, (
            f"{route.path} must be gated XOR exempt (gated={gated}, exempt={exempt})"
        )
