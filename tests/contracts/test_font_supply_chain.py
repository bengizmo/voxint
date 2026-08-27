"""Font supply-chain contract (epic #205, V1 #206).

IBM Plex Sans and Mono are self-hosted as woff2 via the Vite build pipeline.
These tests pin the invariants that would otherwise rot silently:

1. The source woff2 files exist in the frontend source tree.
2. Aggregate payload stays under the 200 KB budget.
3. The OFL-1.1 license ships alongside the fonts.
4. The Tailwind CSS source declares @font-face for every shipped weight.
5. base.html references IBM Plex in --font-ui and --font-mono with
   system-ui / ui-monospace fallbacks.
"""

from tests.contracts.conftest import REPO_ROOT

_FONTS_DIR = REPO_ROOT / "frontend" / "src" / "fonts"
_TAILWIND_CSS = REPO_ROOT / "frontend" / "src" / "styles" / "tailwind.css"
_BASE_HTML = REPO_ROOT / "src" / "voxint" / "api" / "templates" / "base.html"

_EXPECTED_FACES = [
    ("ibm-plex-sans-latin-400-normal.woff2", 400),
    ("ibm-plex-sans-latin-500-normal.woff2", 500),
    ("ibm-plex-sans-latin-600-normal.woff2", 600),
    ("ibm-plex-mono-latin-400-normal.woff2", 400),
    ("ibm-plex-mono-latin-500-normal.woff2", 500),
]

_PAYLOAD_BUDGET_BYTES = 200 * 1024


def test_all_font_files_present() -> None:
    for name, _ in _EXPECTED_FACES:
        path = _FONTS_DIR / name
        assert path.exists(), f"missing font file: {path.relative_to(REPO_ROOT)}"


def test_aggregate_payload_under_budget() -> None:
    total = sum((_FONTS_DIR / name).stat().st_size for name, _ in _EXPECTED_FACES)
    assert total <= _PAYLOAD_BUDGET_BYTES, (
        f"font payload {total / 1024:.0f} KB exceeds the {_PAYLOAD_BUDGET_BYTES // 1024} KB budget"
    )


def test_ofl_license_present() -> None:
    license_file = _FONTS_DIR / "OFL-LICENSE.txt"
    assert license_file.exists(), "OFL-1.1 license file missing from fonts directory"
    text = license_file.read_text()
    assert "SIL OPEN FONT LICENSE" in text


def test_tailwind_css_declares_all_faces() -> None:
    css = _TAILWIND_CSS.read_text()
    for name, weight in _EXPECTED_FACES:
        assert name in css, f"tailwind.css missing @font-face src for {name}"
        assert f"font-weight: {weight}" in css


def test_base_html_font_families() -> None:
    html = _BASE_HTML.read_text()
    assert "'IBM Plex Sans'" in html, "base.html --font-ui must reference IBM Plex Sans"
    assert "'IBM Plex Mono'" in html, "base.html --font-mono must reference IBM Plex Mono"
    assert "system-ui" in html, "base.html --font-ui must keep system-ui fallback"
    assert "ui-monospace" in html, "base.html --font-mono must keep ui-monospace fallback"
