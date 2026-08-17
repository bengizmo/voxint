"""The shared corrected>enhanced>raw selector (issues #53/#58).

`effective_text` is the ONE place the correction precedence lives — display,
exports, search, and enrichment all defer to it, so it is pinned here directly.
"""

from types import SimpleNamespace

from voxint.adjudication.transcript import effective_text


def _seg(raw: str, enhanced: str | None) -> SimpleNamespace:
    # effective_text only reads .raw_text / .enhanced_text (duck-typed).
    return SimpleNamespace(raw_text=raw, enhanced_text=enhanced)


def test_precedence_corrected_over_enhanced_over_raw() -> None:
    seg = _seg("raw", "enhanced")
    assert effective_text(seg, "corrected") == "corrected"  # corrected wins
    assert effective_text(seg, None) == "enhanced"  # then enhanced
    assert effective_text(_seg("raw", None), None) == "raw"  # then raw


def test_uses_is_not_null_not_truthiness() -> None:
    # An empty-string correction is NOT NULL, so it takes precedence — proving the
    # selector keys on IS NOT NULL, never Python truthiness. (The writer normalizes
    # empty input to NULL before this ever persists, but the selector must be
    # correct on its own.)
    assert effective_text(_seg("raw", "enhanced"), "") == ""
    # Likewise an empty enhanced string is honored over raw.
    assert effective_text(_seg("raw", ""), None) == ""
