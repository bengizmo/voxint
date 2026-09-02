"""The count-inflected "needs review" chip macro (#318, #386).

The generic ``chip()`` macro is an EXACT-key lookup, so a count-prefixed label
like "2 voices need review" falls through to chip-neutral — which is exactly how
the jobs page shipped a styling bug. ``needs_review_chip`` owns both the
inflection and the warn class; these tests pin BOTH the text and the class
(the class assertion is the regression trap).
"""

from voxint.api.routers.deps import templates

_TMPL = (
    '{% from "fragments/_chips.html" import needs_review_chip %}{{ needs_review_chip(n) }}'
)


def _render(count: int) -> str:
    return templates.env.from_string(_TMPL).render(n=count)


def test_singular_inflects_noun_and_verb() -> None:
    html = _render(1)
    assert "1 voice needs review" in html
    assert "chip-warn" in html


def test_plural_inflects_noun_and_verb() -> None:
    html = _render(2)
    assert "2 voices need review" in html
    assert "chip-warn" in html


def test_never_falls_through_to_neutral() -> None:
    for count in (1, 2, 9):
        assert "chip-neutral" not in _render(count)


def test_call_sites_use_the_macro() -> None:
    from tests.contracts.conftest import REPO_ROOT

    tpl = REPO_ROOT / "src" / "voxint" / "api" / "templates"
    for rel in ("home/home.html", "media/media.html", "legacy_runs/runs.html"):
        text = (tpl / rel).read_text()
        assert "needs_review_chip(" in text, f"{rel} no longer uses needs_review_chip"
        assert "need you</span>" not in text, f"{rel} regrew an inline needs-you span"
