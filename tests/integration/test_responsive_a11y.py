"""Responsive + accessibility baseline of the server-rendered console (issue #64).

The shell markup and stylesheet live in ``base.html``, so every 200 HTML page
carries them; ``/dashboard`` is the cheapest onboarded page to assert them on.
These are literal substring/attribute checks on ``resp.text`` — the same seam
every other server-page test uses (there is no template-render harness). Whether
the layout is *actually* free of horizontal overflow at phone widths is a browser
concern the maintainer verifies manually (documented in the issue #64 plan); here
we pin that the enabling markup/CSS is present and does not regress.

Per-page table-scroll regions are asserted where their data is already seeded:
runs list + stage ledger in ``test_runs_api``, review queue in ``test_review_api``,
metrics tables in ``test_dashboard_api``.
"""

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.config import Settings

CREDS = ("reviewer", "s3cret")


@pytest.fixture()
def client(session_factory: sessionmaker[Session], tmp_path: Path) -> TestClient:
    settings = Settings(
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        media_root=tmp_path,
    )
    test_client = TestClient(create_app(settings=settings, session_factory=session_factory))
    test_client.auth = CREDS
    seed_onboarded(session_factory)
    return test_client


def test_skip_link_and_main_landmark(client: TestClient) -> None:
    """A keyboard/AT user gets a skip-link that targets a real ``<main>`` landmark;
    the landmark is programmatically focusable (tabindex=-1) so activating the
    skip-link moves focus, not merely the scroll position."""
    body = client.get("/dashboard").text
    assert '<a class="skip-link" href="#main">Skip to main content</a>' in body
    assert '<main id="main" tabindex="-1">' in body
    # The skip-link target and the landmark id agree.
    assert body.index('href="#main"') < body.index('id="main"')


def test_focus_ring_and_responsive_css_shipped(client: TestClient) -> None:
    """The inline stylesheet carries the new focus-visible ring, the table scroll
    container, the page-overflow guard, the wrap-enabled nav, and the small-screen
    breakpoint. Property checks are whitespace-tolerant (a harmless reformat must
    not false-fail) but still assert the actual declaration, not just a selector
    name — a gutted ``:focus-visible {}`` must NOT satisfy the ring check
    (multi-model review: token-only assertions were vacuous)."""
    body = client.get("/dashboard").text
    # The focus ring actually draws an outline from a defined var, not nothing.
    assert re.search(r":focus-visible\s*\{[^}]*outline:\s*2px solid var\(--focus-ring\)", body)
    # The ring follows the teal brand accent (issue #91), which is itself a real
    # colour — so the outline resolves to a hue, not to an undefined var.
    assert re.search(r"--focus-ring:\s*var\(--accent\)", body)
    assert re.search(r"--accent:\s*#", body)
    assert re.search(r"\.table-wrap\s*\{[^}]*overflow-x:\s*auto", body)
    # AC1 belt-and-braces: long unbroken paths/URLs outside a table-wrap can't
    # force horizontal PAGE overflow (they break instead).
    assert re.search(r"body\s*\{[^}]*overflow-wrap:\s*break-word", body)
    assert re.search(r"nav\.top\s*\{[^}]*flex-wrap:\s*wrap", body)
    assert "max-width: 40rem" in body
    # Utility for visually-hidden-but-announced labels (e.g. the queue action col).
    # Assert the actual clip rule, not just the selector name — a gutted
    # ``.visually-hidden {}`` must not pass (same anti-vacuous standard as above).
    assert re.search(r"\.visually-hidden\s*\{[^}]*clip:\s*rect\(0 0 0 0\)", body)
    # New chrome transitions are gated behind prefers-reduced-motion (issue #91
    # added button hover transitions; #64 forbids unconditional motion).
    assert re.search(r"prefers-reduced-motion:\s*reduce[^}]*transition:\s*none", body)


def test_light_dark_and_forced_colors_preserved(client: TestClient) -> None:
    """Light/dark theming stays intact and the focus ring degrades sanely under a
    forced-colors (high-contrast) system setting."""
    body = client.get("/dashboard").text
    assert "color-scheme: light dark;" in body
    # The consolidated dark block (issue #91) redefines the themed tokens the ring
    # and error colour follow: the teal accent gets a dark value, and the ring is
    # bound to it, so keyboard focus stays visible on the dark canvas. A
    # forced-colors fallback still overrides both.
    assert re.search(r"prefers-color-scheme:\s*dark[^}]*--accent:\s*#5eb8ae", body)
    assert re.search(r"--focus-ring:\s*var\(--accent\)", body)
    # The forced-colors fallback actually retargets the ring to the system
    # Highlight colour — assert the declaration, not just the media-query name.
    assert re.search(r"forced-colors:\s*active[^}]*outline-color:\s*Highlight", body)
    # Dark-scheme error colour keeps error text legible on a dark canvas (AC4):
    # .error reads --danger, which the dark block retargets to a lighter red.
    assert re.search(r"\.error\s*\{[^}]*color:\s*var\(--danger\)", body)
    assert re.search(r"prefers-color-scheme:\s*dark[^}]*--danger:\s*#f2685e", body)
