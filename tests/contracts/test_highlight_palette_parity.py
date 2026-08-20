"""Highlight-palette parity contract (issue #86 operator annotation layer).

The annotation highlight palette size lives in three places that must agree, or a
highlight silently loses its color:

1. backend ``HIGHLIGHT_PALETTE_SIZE`` (``db.models``) — the CHECK-constrained
   ``0..N-1`` colour-index range for both tags and annotations, and the
   ``annotationLimits.highlightPalette`` the toolbar renders swatches from;
2. the ``--hl-0-rgb .. --hl-{N-1}-rgb`` design tokens in ``base.html`` — defined
   ONCE in the bare light ``:root`` (theme-agnostic, like ``--seg-rgb``; a future
   per-theme tune would add them to both dark blocks, guarded by the theme-toggle
   contract); and
3. the ``mark.hl-0 .. mark.hl-{N-1}`` paint rules in ``base.html`` — the actual
   highlight background the island and the JS-off fallback both render.

If the backend hands out colour index ``k`` but no ``--hl-k-rgb`` token or
``mark.hl-k`` rule exists, the mark resolves an empty background and the highlight
renders invisibly with no error. This pins all three to the same contiguous
``0..N-1`` set so any partial change fails here, mirroring
``test_speaker_palette_parity``.
"""

import re

from tests.contracts.conftest import REPO_ROOT, strip_css_comments
from voxint.db.models import HIGHLIGHT_PALETTE_SIZE

_BASE_HTML = REPO_ROOT / "src" / "voxint" / "api" / "templates" / "base.html"


def _token_indices(css: str) -> set[int]:
    """Indices with a ``--hl-N-rgb:`` token definition."""
    return {int(m) for m in re.findall(r"--hl-(\d+)-rgb\s*:", css)}


def _mark_indices(css: str) -> set[int]:
    """Indices with a ``mark.hl-N`` paint rule (a background declaration)."""
    return {int(m) for m in re.findall(r"mark\.hl-(\d+)\s*\{[^}]*background", css)}


def test_highlight_tokens_cover_the_palette() -> None:
    css = strip_css_comments(_BASE_HTML.read_text())
    expected = set(range(HIGHLIGHT_PALETTE_SIZE))
    assert _token_indices(css) == expected, (
        "base.html --hl-N-rgb tokens must be exactly the contiguous "
        f"0..{HIGHLIGHT_PALETTE_SIZE - 1} set that matches HIGHLIGHT_PALETTE_SIZE"
    )


def test_highlight_mark_rules_cover_the_palette() -> None:
    css = strip_css_comments(_BASE_HTML.read_text())
    expected = set(range(HIGHLIGHT_PALETTE_SIZE))
    assert _mark_indices(css) == expected, (
        "base.html mark.hl-N paint rules must be exactly the contiguous "
        f"0..{HIGHLIGHT_PALETTE_SIZE - 1} set; a backend colour index without a "
        "mark rule renders an invisible highlight"
    )
