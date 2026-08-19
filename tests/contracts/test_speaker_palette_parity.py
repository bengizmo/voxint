"""Speaker-palette parity contract (issue #57 waveform polish).

The per-speaker palette size lives in FOUR places that must agree, or a run's
diarization labels silently lose their color:

1. backend ``speaker_colors.PALETTE_SIZE`` (maps labels to indices ``0..N-1``),
2. the frontend ``WaveformStrip`` probe count (``PALETTE_SIZE`` in the .tsx),
3. the ``--spk-0 .. --spk-{N-1}`` design tokens in ``base.html`` (both the light
   ``:root`` and the ``:root`` nested in the ``prefers-color-scheme: dark`` block),
   and
4. the ``.spk-0 .. .spk-{N-1}`` -> ``--spk-accent: var(--spk-N)`` class mappings in
   ``base.html`` (the public hook the islands and this waveform read through probe
   spans).

If the backend hands out index ``k`` but no ``.spk-k`` / ``--spk-k`` exists, the
canvas resolves an empty accent and falls back to the neutral bar color, so a
real speaker turn renders as "no speaker" with no error. A synchronized change to
only the two integer constants would still leave the CSS short. This contract pins
all four to the same contiguous ``0..N-1`` set so any partial change fails here.

The CSS checks are deliberately structural: the ``:root`` bodies are brace-matched
and comments stripped so the token set is read from the ACTUAL light/dark
declarations (not merely "somewhere after the dark marker"), and each ``.spk-N``
mapping is matched WITH its ``var(--spk-M)`` target so a self-referential
``k -> k`` mapping is required (an empty or cross-wired rule fails).
"""

import re

from tests.contracts.conftest import REPO_ROOT
from voxint.api.speaker_colors import PALETTE_SIZE

_BASE_HTML = REPO_ROOT / "src" / "voxint" / "api" / "templates" / "base.html"
_WAVEFORM_TSX = REPO_ROOT / "frontend" / "src" / "components" / "WaveformStrip.tsx"


def _strip_css_comments(css: str) -> str:
    return re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)


def _root_bodies(css: str) -> list[tuple[int, str]]:
    """Return ``(selector_index, body)`` for every brace-matched ``:root { ... }``.

    ``base.html`` has several ``:root`` blocks (a bare ``color-scheme`` one, the
    light token block, and the dark token block), so callers select the one that
    actually bears ``--spk-*`` rather than assuming position. Depth matching keeps
    this correct even if a nested rule is ever added.
    """
    bodies: list[tuple[int, str]] = []
    for match in re.finditer(r":root\s*\{", css):
        open_brace = css.index("{", match.start())
        depth = 0
        for i in range(open_brace, len(css)):
            if css[i] == "{":
                depth += 1
            elif css[i] == "}":
                depth -= 1
                if depth == 0:
                    bodies.append((match.start(), css[open_brace + 1 : i]))
                    break
        else:
            raise AssertionError("unbalanced braces after :root in base.html")
    return bodies


def _spk_root_body(css: str, *, before: bool, split_at: int) -> str:
    """The single ``:root`` body bearing ``--spk-*`` on one side of ``split_at``.

    ``before=True`` selects the light block (before the dark media marker);
    ``before=False`` selects the dark block. Asserting exactly one match guards
    against a stray duplicate or a token block landing on the wrong side.
    """
    side = [
        body
        for idx, body in _root_bodies(css)
        if "--spk-" in body and (idx < split_at if before else idx > split_at)
    ]
    scope = "light" if before else "dark"
    assert len(side) == 1, (
        f"expected exactly one {scope} :root block defining --spk-* tokens in "
        f"base.html, found {len(side)}"
    )
    return side[0]


def _frontend_palette_size() -> int:
    text = _WAVEFORM_TSX.read_text()
    match = re.search(r"^const PALETTE_SIZE = (\d+);", text, re.MULTILINE)
    assert match, "WaveformStrip.tsx lost its `const PALETTE_SIZE = N;` declaration"
    return int(match.group(1))


def _token_indices(css_body: str) -> set[int]:
    """Indices with a ``--spk-N:`` token definition in the given CSS body."""
    return {int(m) for m in re.findall(r"--spk-(\d+)\s*:", css_body)}


def _mapping_pairs(css: str) -> set[tuple[int, int]]:
    """(selector, target) pairs from ``.spk-N { --spk-accent: var(--spk-M) }``."""
    pattern = r"\.spk-(\d+)\s*\{\s*--spk-accent\s*:\s*var\(\s*--spk-(\d+)\s*\)\s*\}"
    return {(int(sel), int(tok)) for sel, tok in re.findall(pattern, css)}


def test_frontend_palette_size_matches_backend() -> None:
    assert _frontend_palette_size() == PALETTE_SIZE, (
        "WaveformStrip PALETTE_SIZE drifted from speaker_colors.PALETTE_SIZE; "
        "a backend palette index would resolve no waveform accent"
    )


def test_base_html_light_and_dark_tokens_cover_the_palette() -> None:
    css = _strip_css_comments(_BASE_HTML.read_text())
    dark_at = css.index("prefers-color-scheme: dark")
    light_body = _spk_root_body(css, before=True, split_at=dark_at)
    dark_body = _spk_root_body(css, before=False, split_at=dark_at)
    expected = set(range(PALETTE_SIZE))

    assert _token_indices(light_body) == expected, (
        "light :root --spk-* tokens must be exactly the contiguous "
        f"0..{PALETTE_SIZE - 1} set that matches speaker_colors.PALETTE_SIZE"
    )
    assert _token_indices(dark_body) == expected, (
        "dark :root --spk-* tokens must be exactly the contiguous "
        f"0..{PALETTE_SIZE - 1} set that matches speaker_colors.PALETTE_SIZE"
    )


def test_base_html_class_mappings_cover_the_palette() -> None:
    css = _strip_css_comments(_BASE_HTML.read_text())
    expected = {(i, i) for i in range(PALETTE_SIZE)}
    assert _mapping_pairs(css) == expected, (
        f".spk-N -> var(--spk-N) mappings must be exactly the self-referential "
        f"0..{PALETTE_SIZE - 1} set; a missing, empty, or cross-wired rule leaves "
        "a backend index resolving to the neutral bar or another speaker's colour"
    )
