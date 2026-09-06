"""The tutorial page-to-route map stays live (Console 2.0 P1, issue #152).

The guided-tutorial banner derives its continue-links from
``voxint.tutorial.steps.PAGE_ROUTE_NAME`` via ``app.url_path_for``, so a later
phase remaps a step by editing that map instead of hunting hardcoded URLs. That
only holds if every mapped route name actually exists on the built app — a
rename or removal would otherwise strand the tutorial with a runtime
``NoMatchFound`` mid-walkthrough. Read off the route table (no HTTP, no DB),
like the console2 characterization contracts.
"""

from __future__ import annotations

from fastapi.routing import APIRoute

from tests.contracts.test_console2_characterization import _iter_api_routes
from voxint.api.app import create_app
from voxint.tutorial.steps import PAGE_ROUTE_NAME, STEP_PAGE, TutorialPage

# The path parameters each tutorial page's route needs to build a URL. The
# editor page takes media_id; the run_detail page takes run_id.
_EXPECTED_PATH_PARAMS: dict[TutorialPage, set[str]] = {
    TutorialPage.RUN_DETAIL: {"run_id"},
    TutorialPage.EDITOR: {"media_id"},
}


def _routes_by_name() -> dict[str, list[APIRoute]]:
    routes: dict[str, list[APIRoute]] = {}
    for route in _iter_api_routes(create_app().routes):
        routes.setdefault(route.name, []).append(route)
    return routes


def test_every_tutorial_page_has_a_route() -> None:
    """Every page in STEP_PAGE has a PAGE_ROUTE_NAME entry, and vice versa."""
    assert set(PAGE_ROUTE_NAME) == set(STEP_PAGE.values()), (
        "PAGE_ROUTE_NAME and STEP_PAGE disagree on the tutorial's page set"
    )


def test_every_mapped_route_name_resolves_uniquely() -> None:
    """Each mapped name is exactly one GET route with the expected path params."""
    routes = _routes_by_name()
    for page, name in PAGE_ROUTE_NAME.items():
        matches = routes.get(name, [])
        assert len(matches) == 1, (
            f"tutorial page {page.value!r} maps to route name {name!r}, which "
            f"resolves to {len(matches)} routes (need exactly 1); update "
            f"PAGE_ROUTE_NAME or re-pin name= on the moved route"
        )
        route = matches[0]
        assert "GET" in (route.methods or set()), (
            f"route {name!r} for tutorial page {page.value!r} has no GET method"
        )
        params = set(route.param_convertors)
        expected = _EXPECTED_PATH_PARAMS.get(page, set())
        assert params == expected, (
            f"route {name!r} for tutorial page {page.value!r} takes path params "
            f"{sorted(params)}, but the banner builds URLs with {sorted(expected)}"
        )
