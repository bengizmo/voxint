"""Lightweight WER scorer for the Voxint benchmark.

Ships with the release (no extra dependencies).  Uses a minimal normalizer
and standard Levenshtein edit distance on word tokens.
"""

from __future__ import annotations

import hashlib
import re
import string
from dataclasses import dataclass

NORMALIZER_VERSION = "voxint-norm-v1"
AGGREGATION_VERSION = "voxint-agg-v1"


@dataclass(frozen=True, slots=True)
class WERCounts:
    """Per-file word error counts."""

    substitutions: int
    insertions: int
    deletions: int
    reference_words: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.insertions + self.deletions

    @property
    def wer(self) -> float:
        if self.reference_words == 0:
            return 0.0
        return self.errors / self.reference_words

    def to_dict(self) -> dict[str, int]:
        return {
            "substitutions": self.substitutions,
            "insertions": self.insertions,
            "deletions": self.deletions,
            "reference_words": self.reference_words,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    """Pooled micro-WER and hallucination metrics across a benchmark run."""

    pooled_wer: float
    total_substitutions: int
    total_insertions: int
    total_deletions: int
    total_reference_words: int
    hallucination_total_words: int
    hallucination_nonempty_count: int
    hallucination_file_count: int
    speech_file_count: int
    total_time_s: float

    def to_dict(self) -> dict[str, object]:
        return {
            "pooled_wer": self.pooled_wer,
            "total_substitutions": self.total_substitutions,
            "total_insertions": self.total_insertions,
            "total_deletions": self.total_deletions,
            "total_reference_words": self.total_reference_words,
            "hallucination_total_words": self.hallucination_total_words,
            "hallucination_nonempty_count": self.hallucination_nonempty_count,
            "hallucination_file_count": self.hallucination_file_count,
            "speech_file_count": self.speech_file_count,
            "total_time_s": self.total_time_s,
        }


_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    """Minimal benchmark normalizer: lowercase, strip ASCII punctuation, collapse whitespace."""
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_wer(reference: str, hypothesis: str) -> WERCounts:
    """Compute WER via Levenshtein edit distance on normalized word tokens.

    Both strings are normalized before tokenization.  An empty reference with
    a non-empty hypothesis counts all hypothesis words as insertions.
    """
    ref_words = normalize(reference).split()
    hyp_words = normalize(hypothesis).split()

    n = len(ref_words)
    m = len(hyp_words)

    full = [[0] * (m + 1) for _ in range(n + 1)]
    for j in range(m + 1):
        full[0][j] = j
    for i in range(1, n + 1):
        full[i][0] = i
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                full[i][j] = full[i - 1][j - 1]
            else:
                full[i][j] = 1 + min(
                    full[i - 1][j - 1],  # substitution
                    full[i - 1][j],  # deletion
                    full[i][j - 1],  # insertion
                )

    subs = 0
    ins = 0
    dels = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and full[i][j] == full[i - 1][j - 1] + 1:
            subs += 1
            i -= 1
            j -= 1
        elif i > 0 and full[i][j] == full[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1

    return WERCounts(
        substitutions=subs,
        insertions=ins,
        deletions=dels,
        reference_words=n,
    )


def protocol_hash() -> str:
    """Fingerprint of the current normalizer + aggregation protocol.

    Changes when the normalizer rules or the aggregation method change,
    making cross-version comparisons detectable.
    """
    identity = f"{NORMALIZER_VERSION}:{AGGREGATION_VERSION}"
    return hashlib.sha256(identity.encode()).hexdigest()[:16]


def pool_wer(counts: list[WERCounts]) -> float:
    """Pooled micro-WER: sum(errors) / sum(reference_words)."""
    total_errors = sum(c.errors for c in counts)
    total_ref = sum(c.reference_words for c in counts)
    if total_ref == 0:
        return 0.0
    return total_errors / total_ref
