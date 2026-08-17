"""Pooled WER scoring for the whisper bakeoff / self-parity gate (#33).

Wraps ``jiwer==4.0.0`` (edit-distance only) + the sha-pinned vendored Whisper
``EnglishTextNormalizer`` (``tests/parity/bakeoff/normalize.py``). Both are
pinned in the ``parity`` extra so the frozen denominator cannot move under a
resolver bump.

The gate metric is a **micro average**: integer S/D/I/N edit counts are pooled
across files, then ``WER = (S + D + I) / N`` on the totals — NOT a mean of
per-file WERs (which would over-weight short files). Files whose *normalized
reference is empty* (true non-speech) are excluded from the WER denominator; the
caller enforces the separate zero-insertion invariant on those via
``FileScore.insertions``.

Both reference and hypothesis are normalized identically before scoring — the
whole point of a frozen normalizer is that it is applied to raw text on both
sides at scoring time.
"""

from __future__ import annotations

from dataclasses import dataclass

import jiwer

from tests.parity.bakeoff.normalize import normalize_text


@dataclass(frozen=True)
class FileScore:
    """Per-file integer edit counts on normalized text (micro-average inputs)."""

    name: str
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    reference_empty: bool

    @property
    def ref_words(self) -> int:
        """Reference word count N = hits + substitutions + deletions."""
        return self.hits + self.substitutions + self.deletions

    @property
    def wer(self) -> float:
        """Per-file WER (diagnostic only; the gate pools counts, not WERs)."""
        n = self.ref_words
        if n == 0:
            # No reference words: WER is undefined; any hypothesis token is a
            # pure insertion. Report 0.0 for a truly-empty pair, else 1.0.
            return 0.0 if self.insertions == 0 else 1.0
        return (self.substitutions + self.deletions + self.insertions) / n


@dataclass(frozen=True)
class PooledScore:
    """Micro-averaged WER over a corpus, plus the per-file breakdown."""

    files: tuple[FileScore, ...]
    substitutions: int
    deletions: int
    insertions: int
    ref_words: int

    @property
    def wer(self) -> float:
        """Pooled WER on the totals (0.0 when no reference words remain)."""
        if self.ref_words == 0:
            return 0.0
        return (self.substitutions + self.deletions + self.insertions) / self.ref_words

    @property
    def wer_pp(self) -> float:
        """Pooled WER in percentage points (the ≤0.5pp gate unit)."""
        return self.wer * 100.0


def score_file(name: str, reference: str, hypothesis: str) -> FileScore:
    """Score one (reference, hypothesis) pair after frozen normalization.

    ``jiwer.process_words`` rejects an empty reference, so an empty normalized
    reference is scored directly: every hypothesis word is an insertion.
    """
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)

    if not ref:
        insertions = len(hyp.split())
        return FileScore(
            name=name,
            substitutions=0,
            deletions=0,
            insertions=insertions,
            hits=0,
            reference_empty=True,
        )

    out = jiwer.process_words(ref, hyp)
    return FileScore(
        name=name,
        substitutions=out.substitutions,
        deletions=out.deletions,
        insertions=out.insertions,
        hits=out.hits,
        reference_empty=False,
    )


def score_pooled(items: list[tuple[str, str, str]]) -> PooledScore:
    """Score ``(name, reference, hypothesis)`` triples and pool the counts.

    Empty-reference files still appear in ``files`` (so the caller can assert
    their zero-insertion invariant) but are excluded from the pooled WER
    denominator and numerator — a non-speech clip must not dilute or inflate the
    speech WER.
    """
    scores = tuple(score_file(name, ref, hyp) for name, ref, hyp in items)
    speech = [s for s in scores if not s.reference_empty]
    return PooledScore(
        files=scores,
        substitutions=sum(s.substitutions for s in speech),
        deletions=sum(s.deletions for s in speech),
        insertions=sum(s.insertions for s in speech),
        ref_words=sum(s.ref_words for s in speech),
    )
