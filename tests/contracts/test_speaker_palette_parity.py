"""Speaker-palette parity contract (issue #57 waveform polish).

The per-speaker palette size lives in FOUR places that must agree, or a run's
diarization labels silently lose their color:

1. backend ``speaker_colors.PALETTE_SIZE`` (maps labels to indices ``0..N-1``),
2. the frontend ``WaveformStrip`` probe count (``PALETTE_SIZE`` in the .tsx),
3. the ``--spk-0 .. --spk-{N-1}`` design tokens in ``base.html`` — the light
   ``:root`` block AND both dark blocks (#94 theme toggle): the guarded
   ``:root:not([data-theme="light"])`` nested in the ``prefers-color-scheme:
   dark`` media query, and the explicit ``:root[data-theme="dark"]`` sibling —
   and
4. the ``.spk-0 .. .spk-{N-1}`` -> ``--spk-accent: var(--spk-N)`` class mappings in
   ``base.html`` (the public hook the islands and this waveform read through probe
   spans).

If the backend hands out index ``k`` but no ``.spk-k`` / ``--spk-k`` exists, the
canvas resolves an empty accent and falls back to the neutral bar color, so a
real speaker turn renders as "no speaker" with no error. A synchronized change to
only the two integer constants would still leave the CSS short. This contract pins
all four to the same contiguous ``0..N-1`` set so any partial change fails here.

The CSS checks are deliberately structural: the root bodies are brace-matched
and comments stripped so the token set is read from the ACTUAL light/dark
declarations (not merely "somewhere after the dark marker"), the two dark
blocks must define identical ``--spk-*`` values (they are deliberate duplicates
— CSS cannot OR a media query with a selector — so a palette tweak to only one
would silently fork system-dark from explicit-dark), and each ``.spk-N``
mapping is matched WITH its ``var(--spk-M)`` target so a self-referential
``k -> k`` mapping is required (an empty or cross-wired rule fails).
"""

import re

from tests.contracts.conftest import (
    REPO_ROOT,
)
from tests.contracts.conftest import (
    selector_bodies as _selector_bodies,
)
from tests.contracts.conftest import (
    strip_css_comments as _strip_css_comments,
)
from voxint.api.speaker_colors import PALETTE_SIZE

_BASE_HTML = REPO_ROOT / "src" / "voxint" / "api" / "templates" / "base.html"
_WAVEFORM_TSX = REPO_ROOT / "frontend" / "src" / "components" / "WaveformStrip.tsx"


def _light_root_body(css: str) -> str:
    """The single bare ``:root`` body bearing ``--spk-*`` BEFORE the dark media
    marker (base.html also carries a bare ``color-scheme: light`` root, which
    defines no palette). Asserting exactly one guards against a stray duplicate
    or a token block landing on the wrong side of the marker."""
    dark_at = css.index("prefers-color-scheme: dark")
    side = [
        body
        for idx, body in _selector_bodies(css, r":root")
        if "--spk-" in body and idx < dark_at
    ]
    assert len(side) == 1, (
        "expected exactly one light :root block defining --spk-* tokens in "
        f"base.html, found {len(side)}"
    )
    return side[0]


def _system_dark_root_body(css: str) -> str:
    """The single guarded ``:root:not([data-theme="light"])`` body INSIDE the
    single ``prefers-color-scheme: dark`` media query (#94 system-dark case)."""
    media = _selector_bodies(
        css, r"@media\s+screen\s+and\s+\(\s*prefers-color-scheme:\s*dark\s*\)"
    )
    assert len(media) == 1, (
        "expected exactly one screen and (prefers-color-scheme: dark) media "
        f"query in base.html, found {len(media)}"
    )
    guarded = _selector_bodies(media[0][1], re.escape(':root:not([data-theme="light"])'))
    assert len(guarded) == 1, (
        'expected exactly one :root:not([data-theme="light"]) block inside the '
        f"dark media query, found {len(guarded)}"
    )
    return guarded[0][1]


def _explicit_dark_root_body(css: str) -> str:
    """The single explicit ``:root[data-theme="dark"]`` body (#94 forced-dark
    case, the deliberate duplicate of the guarded system-dark block)."""
    bodies = _selector_bodies(css, re.escape(':root[data-theme="dark"]'))
    assert len(bodies) == 1, (
        'expected exactly one :root[data-theme="dark"] block in base.html, '
        f"found {len(bodies)}"
    )
    return bodies[0][1]


def _frontend_palette_size() -> int:
    text = _WAVEFORM_TSX.read_text()
    match = re.search(r"^const PALETTE_SIZE = (\d+);", text, re.MULTILINE)
    assert match, "WaveformStrip.tsx lost its `const PALETTE_SIZE = N;` declaration"
    return int(match.group(1))


def _token_indices(css_body: str) -> set[int]:
    """Indices with a ``--spk-N:`` token definition in the given CSS body."""
    return {int(m) for m in re.findall(r"--spk-(\d+)\s*:", css_body)}


def _spk_values(css_body: str) -> dict[int, str]:
    """``index -> whitespace-normalized value`` for every ``--spk-N:`` token."""
    return {
        int(idx): " ".join(value.split())
        for idx, value in re.findall(r"--spk-(\d+)\s*:\s*([^;}]+)", css_body)
    }


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
    light_body = _light_root_body(css)
    system_dark_body = _system_dark_root_body(css)
    explicit_dark_body = _explicit_dark_root_body(css)
    expected = set(range(PALETTE_SIZE))

    assert _token_indices(light_body) == expected, (
        "light :root --spk-* tokens must be exactly the contiguous "
        f"0..{PALETTE_SIZE - 1} set that matches speaker_colors.PALETTE_SIZE"
    )
    assert _token_indices(system_dark_body) == expected, (
        "system-dark (guarded media-query) --spk-* tokens must be exactly the "
        f"contiguous 0..{PALETTE_SIZE - 1} set that matches "
        "speaker_colors.PALETTE_SIZE"
    )
    assert _token_indices(explicit_dark_body) == expected, (
        ':root[data-theme="dark"] --spk-* tokens must be exactly the contiguous '
        f"0..{PALETTE_SIZE - 1} set that matches speaker_colors.PALETTE_SIZE"
    )
    assert _spk_values(system_dark_body) == _spk_values(explicit_dark_body), (
        "the two dark token blocks are deliberate duplicates (#94: media-query "
        "system-dark vs explicit data-theme dark) — their --spk-* values must "
        "stay identical, or the palette forks between the two dark modes"
    )


def test_no_client_side_palette_hash_in_frontend() -> None:
    """The speaker palette index MUST come from the server (issue #420).

    A client-side hash (the old ``speakerPaletteIndex``) disagrees with the
    server's positional palette once a label resolves, painting the same voice
    in two colors across pages. This contract ensures no such function exists."""
    frontend_src = REPO_ROOT / "frontend" / "src"
    hits = []
    for path in frontend_src.rglob("*.tsx"):
        text = path.read_text()
        if "speakerPaletteIndex" in text:
            hits.append(str(path.relative_to(REPO_ROOT)))
    assert not hits, (
        "frontend/src still contains a client-side speakerPaletteIndex hash; "
        f"palette indices must come from the server: {hits}"
    )


def test_base_html_class_mappings_cover_the_palette() -> None:
    css = _strip_css_comments(_BASE_HTML.read_text())
    expected = {(i, i) for i in range(PALETTE_SIZE)}
    assert _mapping_pairs(css) == expected, (
        f".spk-N -> var(--spk-N) mappings must be exactly the self-referential "
        f"0..{PALETTE_SIZE - 1} set; a missing, empty, or cross-wired rule leaves "
        "a backend index resolving to the neutral bar or another speaker's colour"
    )
