"""Canonical-PCM primitive parity between the corpus and infer modules (#144, S5).

The S5 PR-2a prepare executor measures clip identity with a numpy-free reader in
``synthdetect_corpus``; the scoring path measures the same clips with the numpy
reader in ``synthdetect_infer``. Both hash the raw WAV ``data``-chunk payload, so
their canonical constants MUST stay equal, or a corpus materialized by one and
scored by the other could disagree on identity without any test noticing. This pins
that invariant as data drift, in the same commit that introduces the corpus reader.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import synthdetect_corpus as corpus  # noqa: E402
import synthdetect_infer as infer  # noqa: E402


def test_canonical_constants_match_infer() -> None:
    assert corpus.CANONICAL_SAMPLE_RATE == infer.CANONICAL_SAMPLE_RATE
    assert corpus.CANONICAL_CHANNELS == infer.CANONICAL_CHANNELS
    assert corpus.CANONICAL_SAMPLE_WIDTH == infer.CANONICAL_SAMPLE_WIDTH
    assert corpus.CANONICALIZATION_ID == infer.CANONICALIZATION_ID


def test_block_align_is_derived() -> None:
    assert corpus._BLOCK_ALIGN == corpus.CANONICAL_CHANNELS * corpus.CANONICAL_SAMPLE_WIDTH
