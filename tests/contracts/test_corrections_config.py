"""Contract: domain-pack corrections invariants stay pinned and documented (#80).

A pack's ``corrections:`` field declares deterministic literal-substitution rules
frozen per run. Its bounds and default flags are a versioned operator-facing
contract (bumping a bound changes which packs are accepted), and the documented
manifest example must stay parseable by the real loader — a doc that drifts from
the schema is a bug.
"""

import re

import yaml

from tests.contracts.conftest import REPO_ROOT
from voxint.domain_packs.base import DomainPack
from voxint.domain_packs.corrections import (
    MAX_CORRECTIONS_MANIFEST_BYTES,
    MAX_MATCH_CHARS,
    MAX_REPLACEMENT_CHARS,
    MAX_RULES_PER_PACK,
    parse_corrections,
)


def test_correction_bounds_pinned() -> None:
    # These bounds gate which packs load; changing one is a deliberate contract
    # change, mirrored in docs/domain-packs.md.
    assert MAX_RULES_PER_PACK == 256
    assert MAX_MATCH_CHARS == 256
    assert MAX_REPLACEMENT_CHARS == 512
    assert MAX_CORRECTIONS_MANIFEST_BYTES == 131072


def test_correction_flag_defaults_are_true() -> None:
    # case_sensitive + whole_word default true (the documented conservative
    # posture); a change would silently alter matching for every existing pack.
    rule = parse_corrections([{"id": "a", "match": "x", "replace": "y"}])[0]
    assert rule.case_sensitive is True
    assert rule.whole_word is True


def _first_yaml_block_with(marker: str) -> dict[str, object]:
    """The first fenced ```yaml block in domain-packs.md that contains ``marker``."""
    text = (REPO_ROOT / "docs" / "domain-packs.md").read_text()
    for block in re.findall(r"```yaml\n(.*?)```", text, re.DOTALL):
        if marker in block:
            loaded = yaml.safe_load(block)
            assert isinstance(loaded, dict)
            return loaded
    raise AssertionError(f"no ```yaml block in docs/domain-packs.md contains {marker!r}")


def test_docs_manifest_example_parses() -> None:
    # The documented newsroom manifest (with its corrections:) must load through
    # the real strict loader — guards doc↔code drift exactly as triage guards
    # .env.example. Keyed on "name: newsroom" (unique to the full manifest block),
    # not on "id: zoning-board" which also appears in the corrections-only snippet.
    manifest = _first_yaml_block_with("name: newsroom")
    pack = DomainPack.from_mapping(manifest)
    rule = next(r for r in pack.corrections if r.id == "zoning-board")
    assert rule.match == "zoom board"
    assert rule.replace == "Zoning Board"
    # The example omits the flags, so they must resolve to the documented defaults.
    assert rule.case_sensitive is True
    assert rule.whole_word is True
