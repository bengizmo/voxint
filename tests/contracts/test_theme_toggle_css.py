"""Theme-toggle CSS + head-order contract (issue #94).

The #94 theme toggle rests on two structural facts about ``base.html`` that
nothing at runtime would flag if they rotted:

1. **Dark-block parity.** Dark tokens live in TWO deliberately identical
   blocks — the guarded ``:root:not([data-theme="light"])`` inside the
   ``screen and (prefers-color-scheme: dark)`` media query (system dark) and
   the explicit ``:root[data-theme="dark"]`` nested in ``@media screen``
   (forced dark) — because CSS cannot OR a media query with a selector. Both
   are scoped to ``screen`` so print always renders the light tokens
   (browsers suppress page backgrounds when printing; dark-theme light ink on
   white paper is illegible — measured in the #94 browser lane). An edit to
   only one silently forks the two dark modes, so this contract parses both
   blocks (brace-balanced, comments stripped, duplicate properties rejected)
   and requires the full declaration sets to be exactly equal.

2. **Pre-paint head order.** The inline theme resolver must run before ANY
   render-blocking resource in ``<head>`` — the htmx script, the Tailwind
   island stylesheet, and the big inline ``<style>`` — or the first painted
   frame flashes the wrong theme. The resolver must also guard storage access
   (private-mode/blocked localStorage throws) and accept only the two known
   values, so a corrupt stored string can never become a bogus ``data-theme``
   attribute.

These are static source checks on the template file, same seam as the other
``base.html`` contracts (``test_speaker_palette_parity``).
"""

import re

from tests.contracts.conftest import REPO_ROOT

_BASE_HTML = REPO_ROOT / "src" / "voxint" / "api" / "templates" / "base.html"

_MEDIA_DARK = r"@media\s+screen\s+and\s+\(\s*prefers-color-scheme:\s*dark\s*\)"
_MEDIA_SCREEN_ONLY = r"@media\s+screen(?!\s+and)"
_GUARDED_ROOT = re.escape(':root:not([data-theme="light"])')
_EXPLICIT_ROOT = re.escape(':root[data-theme="dark"]')


def _strip_css_comments(text: str) -> str:
    return re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)


def _selector_bodies(css: str, selector_pattern: str) -> list[str]:
    """Brace-balanced body of every block whose selector matches the pattern.

    NOT a single broad regex: the dark media query nests a ``:root`` block, so
    bodies are extracted by walking brace depth from each selector's opening
    brace. Unbalanced braces are a hard failure, never a silent empty result.
    """
    bodies: list[str] = []
    for match in re.finditer(selector_pattern + r"\s*\{", css):
        open_brace = css.index("{", match.end() - 1)
        depth = 0
        for i in range(open_brace, len(css)):
            if css[i] == "{":
                depth += 1
            elif css[i] == "}":
                depth -= 1
                if depth == 0:
                    bodies.append(css[open_brace + 1 : i])
                    break
        else:
            raise AssertionError("unbalanced braces in base.html CSS")
    return bodies


def _declarations(body: str, context: str) -> dict[str, str]:
    """``property -> whitespace-normalized value`` for one block body.

    Split on ``;`` then on the first ``:`` per declaration. A duplicate
    property inside one block is rejected outright: the later declaration
    would silently win in CSS, letting one dark block "agree" with the other
    on a value the browser never applies.
    """
    props: dict[str, str] = {}
    for raw_decl in body.split(";"):
        decl = raw_decl.strip()
        if not decl:
            continue
        name, sep, value = decl.partition(":")
        assert sep, f"malformed declaration {decl!r} in the {context} block"
        prop = name.strip()
        assert prop not in props, f"duplicate property {prop!r} in the {context} block"
        props[prop] = " ".join(value.split())
    return props


def test_dark_token_blocks_are_declaration_identical() -> None:
    css = _strip_css_comments(_BASE_HTML.read_text())

    media_bodies = _selector_bodies(css, _MEDIA_DARK)
    assert len(media_bodies) == 1, (
        "expected exactly one screen and (prefers-color-scheme: dark) media "
        f"query in base.html, found {len(media_bodies)}"
    )
    guarded_bodies = _selector_bodies(media_bodies[0], _GUARDED_ROOT)
    assert len(guarded_bodies) == 1, (
        'expected exactly one :root:not([data-theme="light"]) block inside the '
        f"dark media query, found {len(guarded_bodies)}"
    )
    # The explicit-dark block must live inside a plain `@media screen` scope:
    # print must always render the light tokens, even with data-theme="dark".
    screen_bodies = _selector_bodies(css, _MEDIA_SCREEN_ONLY)
    assert len(screen_bodies) == 1, (
        "expected exactly one plain @media screen block in base.html, "
        f"found {len(screen_bodies)}"
    )
    explicit_bodies = _selector_bodies(screen_bodies[0], _EXPLICIT_ROOT)
    assert len(explicit_bodies) == 1, (
        'expected exactly one :root[data-theme="dark"] block inside @media '
        f"screen, found {len(explicit_bodies)}"
    )
    assert len(_selector_bodies(css, _EXPLICIT_ROOT)) == 1, (
        'a second :root[data-theme="dark"] block exists outside @media screen'
    )

    system_dark = _declarations(guarded_bodies[0], "system-dark")
    explicit_dark = _declarations(explicit_bodies[0], "explicit-dark")

    # Anti-vacuous: two empty (or gutted) blocks must not "agree". Each block
    # pins the resolved scheme and redefines at least the core surface token.
    assert system_dark.get("color-scheme") == "dark"
    assert "--paper" in system_dark, "system-dark block lost its token set"

    assert system_dark == explicit_dark, (
        "the two dark token blocks are deliberate duplicates (#94: CSS cannot "
        "OR a media query with a selector) — their declaration sets must stay "
        "identical, or system-dark and explicit-dark silently fork; edit both"
    )


def _resolver_script(raw: str) -> tuple[int, str]:
    """(start_index, content) of the pre-paint theme resolver script.

    Identified by content — it reads the ``voxint-theme`` storage key and sets
    ``data-theme`` — and taken as the EARLIEST such inline script: the
    end-of-body theme wiring touches the same names, but the resolver must be
    the first one in the document for pre-paint resolution to hold.
    """
    candidates = [
        (m.start(), m.group(1))
        for m in re.finditer(r"<script>(.*?)</script>", raw, flags=re.DOTALL)
        if "voxint-theme" in m.group(1) and "data-theme" in m.group(1)
    ]
    assert candidates, (
        "base.html lost its inline theme resolver script (voxint-theme + "
        "data-theme)"
    )
    return candidates[0]


def test_head_source_order_resolver_before_blocking_resources() -> None:
    raw = _BASE_HTML.read_text()
    resolver_idx, _ = _resolver_script(raw)
    htmx_idx = raw.index('src="/static/htmx.min.js"')
    tailwind_idx = raw.index("asset_url('tailwind')")
    # The stylesheet's opening tag sits alone on its line; a template comment
    # also says "<style>" in prose, so a bare index() would anchor too early.
    style_open = re.search(r"(?m)^\s*<style>\s*$", raw)
    assert style_open, "base.html lost its inline <style> block"
    style_idx = style_open.start()

    assert resolver_idx < htmx_idx, (
        "the pre-paint theme resolver must precede the blocking htmx script: "
        "a slow script fetch must not delay theme resolution"
    )
    assert htmx_idx < tailwind_idx, (
        "the htmx script moved after the Tailwind island stylesheet link"
    )
    assert tailwind_idx < style_idx, (
        "the Tailwind island stylesheet link must precede the inline <style> "
        "so the hand-authored block stays the later (winning) source"
    )


def test_resolver_guards_storage_and_accepts_only_known_values() -> None:
    _, resolver = _resolver_script(_BASE_HTML.read_text())
    # Storage access can throw (private mode, blocked storage); the resolver
    # must swallow that and fall back to System rather than break first paint.
    assert "try" in resolver, "theme resolver lost its try guard around storage"
    assert "catch" in resolver, "theme resolver lost its catch guard around storage"
    assert 'localStorage.getItem("voxint-theme")' in resolver
    # Only the two known values may ever become a data-theme attribute; any
    # other stored string (corrupt, stale, hand-edited) means System (no
    # attribute set).
    assert re.search(r'if\s*\(\s*\w+\s*===\s*"light"\s*\|\|\s*\w+\s*===\s*"dark"\s*\)', resolver), (
        'theme resolver must gate on exactly t === "light" || t === "dark" '
        "before setting data-theme"
    )
