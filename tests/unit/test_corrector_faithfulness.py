"""Gate A: the corrector-in-isolation faithfulness regression (issue #81).

Reuses the six frozen enhancement fixtures (``tests/fixtures/llm_qual/enhancement``)
as regression inputs — but NOT the LLM's authorized-edit *pass policy*. With an
**empty rule set** the corrector is the identity function, so every fixture segment
must come back byte-identical with an empty trace and no growth rejection. This
stages the wiring and proves the engine introduces no injection / translation /
filler-drop / reorder behavior of its own.

Two supporting contracts are asserted explicitly rather than left accidentally
true (design report §7-A / §12-F7):

- **NFC:** all six fixtures are already NFC, so the "the corrector does not NFC-
  normalize" guarantee is a real, checked property, not luck.
- **Scorer reuse:** the identity output still passes each fixture's own gold via the
  frozen pure scorers in ``tools/qualify_local_llm.py`` (protected tokens survive,
  no merge/split/reorder) — the spec's named reuse substrate.

The separate no-LLM/no-rules ``enhance_match`` *stage* identity test (JSON /
``ensure_ascii`` re-serialization) is #82's: it exercises the stage's persistence
path, which #81 does not wire.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any

import pytest
from tools.qualify_local_llm import score_segment

from voxint.domain_packs.corrector import apply_corrections

ENHANCEMENT_DIR = (
    Path(__file__).resolve().parents[1] / "fixtures" / "llm_qual" / "enhancement"
)

FIXTURES = sorted(ENHANCEMENT_DIR.glob("*.json"))
FIXTURE_IDS = [p.stem for p in FIXTURES]

# All six are mandatory regression cases; guard against a moved/renamed corpus.
EXPECTED_FIXTURES = frozenset(
    {
        "asr_errors",
        "disfluencies",
        "multi_speaker_swap",
        "noop_clean",
        "prompt_injection",
        "unicode",
    }
)


def test_all_expected_fixtures_present() -> None:
    assert frozenset(FIXTURE_IDS) == EXPECTED_FIXTURES


def _segments(fixture: dict[str, Any]) -> list[dict[str, Any]]:
    return fixture["segments"]


@pytest.mark.parametrize("path", FIXTURES, ids=FIXTURE_IDS)
def test_empty_rules_is_byte_identical_and_nfc(path: Path) -> None:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    gold_segments = fixture["gold"]["segments"]
    for segment in _segments(fixture):
        source = segment["text"]
        # Contract: the corpus is NFC, so byte-identity is a meaningful assertion
        # and "no NFC normalization" is explicit.
        assert unicodedata.is_normalized("NFC", source), (
            f"{path.stem} segment {segment['index']} is not NFC"
        )
        result = apply_corrections(source, [])
        assert result.text == source, f"{path.stem} segment {segment['index']} changed"
        assert result.trace == ()
        assert result.growth_rejected is False
        # Reuse the frozen scorers: the identity output still passes the fixture's
        # own gold (protected tokens preserved, no reorder). Redundant with byte-
        # identity here by design, but it exercises the named reuse substrate and
        # pins the harness boundary the design report calls for.
        gold = gold_segments[str(segment["index"])]
        assert score_segment(result.text, source, gold) == []
