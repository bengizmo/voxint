"""Chip semantics contract (epic #205, V3 #208).

Pins the canonical label-to-semantic mapping and verifies that the CSS
classes referenced by the mapping actually exist in base.html.
"""

from tests.contracts.conftest import REPO_ROOT
from voxint.api.chip_semantics import CHIP_SEMANTICS

_BASE_HTML = REPO_ROOT / "src" / "voxint" / "api" / "templates" / "base.html"

_EXPECTED_SUFFIXES = {"ok", "warn", "danger", "info", "accent", "neutral"}


def test_all_semantics_have_css_classes() -> None:
    css = _BASE_HTML.read_text()
    for label, suffix in CHIP_SEMANTICS.items():
        cls = f".chip-{suffix}"
        assert cls in css, (
            f"chip label {label!r} maps to {cls} but that class is missing from base.html"
        )


def test_semantic_suffixes_are_known() -> None:
    for label, suffix in CHIP_SEMANTICS.items():
        assert suffix in _EXPECTED_SUFFIXES, (
            f"chip label {label!r} uses unknown suffix {suffix!r}; "
            f"expected one of {sorted(_EXPECTED_SUFFIXES)}"
        )


def test_mapping_is_nonempty() -> None:
    assert len(CHIP_SEMANTICS) >= 8, (
        f"chip mapping has only {len(CHIP_SEMANTICS)} entries; expected at least 8"
    )
