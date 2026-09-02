"""Island mount points must carry ``data-props`` in a single-quoted attribute.

The ``tojson`` filter returns Markup that escapes ``<``, ``>``, ``&`` and ``'``
but deliberately leaves ``"`` alone, so the JSON must sit inside a
single-quoted attribute. The tempting alternative, a double-quoted attribute
with ``|replace('"', '&quot;')``, double-escapes under autoescape (Markup's
``replace`` escapes its replacement argument), the browser hands the island
``&quot;`` literals, ``JSON.parse`` throws, and the island silently never
hydrates behind its server-rendered fallback. This pinned three islands
(speaker timeline, temporal trends, quote board) before #385 found it.
"""

from __future__ import annotations

import re
from pathlib import Path

TEMPLATES = Path(__file__).resolve().parents[2] / "src" / "voxint" / "api" / "templates"

_DATA_PROPS = re.compile(r"""data-props\s*=\s*(?P<quote>["'])""")


def _template_files() -> list[Path]:
    files = sorted(TEMPLATES.rglob("*.html"))
    assert files, "template tree not found"
    return files


def test_every_island_mount_uses_a_single_quoted_data_props_attribute() -> None:
    offenders: list[str] = []
    mounts = 0
    for path in _template_files():
        for match in _DATA_PROPS.finditer(path.read_text()):
            mounts += 1
            if match.group("quote") != "'":
                line = path.read_text().count("\n", 0, match.start()) + 1
                offenders.append(f"{path.relative_to(TEMPLATES)}:{line}")
    assert mounts > 0, "expected at least one island mount point"
    assert offenders == [], f"double-quoted data-props (island will not hydrate): {offenders}"


def test_no_template_re_escapes_tojson_output() -> None:
    offenders = [
        str(path.relative_to(TEMPLATES))
        for path in _template_files()
        if "tojson|replace" in path.read_text()
    ]
    assert offenders == [], f"tojson output re-escaped (double-encodes): {offenders}"
