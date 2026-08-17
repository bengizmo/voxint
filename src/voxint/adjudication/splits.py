"""Word-boundary segment splits (issue #59, slice 2): derivation + the one writer.

A split is stored as an append-only CUT — "split before word i" on the immutable
parent segment (:class:`~voxint.db.models.SegmentSplitBoundary`) — never as a new
row and never as a mutable overlay the append-only ledger would point at.
Children are DERIVED here at read time from the parent's immutable ``words``
tokens and the cut set ``{0, cuts…, word_count}``. The parent row is never
mutated; ``raw_text`` stays the ASR evidence of record.

Splittability is deliberately conservative (correctness over reach — Voxint's
doctrine). A segment is splittable only when the derived children reproduce its
text FAITHFULLY IN EVERY RENDERING VARIANT, which requires:

* ``words`` is a stored array of >= 2 structurally-valid tokens (non-empty string,
  finite ``start <= end``), whose timings are non-decreasing and lie within the
  parent interval;
* the tokens RECONCATENATE to ``raw_text`` exactly (only an outer-edge whitespace
  delta on the whole string is tolerated — never a per-token strip, which would
  drop the inter-word spaces that make the partition faithful); and
* ``enhanced_text`` is NULL or matches ``raw_text`` (no material enhancement) —
  so a split parent's effective text equals its word-derived text under
  ``?text=raw|enhanced|corrected`` alike, and expanding children never
  contradicts the variant/export contract. A materially-enhanced segment is
  surfaced as unsplittable rather than silently rendered raw.

Correction is orthogonal and handled at the routes: a split cannot be created on
a corrected segment, and a split parent cannot later be corrected (both deferred
to a later slice). This module does not read review state.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from voxint.db.models import SegmentSplitBoundary, TranscriptSegment


@dataclass(frozen=True)
class _Word:
    """A validated word token: verbatim string (incl. any leading space) + timing."""

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class DerivedChild:
    """One derived child of a split parent, computed from the immutable word tokens.

    ``word_start``/``word_end`` are half-open coordinates into the parent's
    ``words`` list; ``text`` is the word-derived rendering (outer whitespace
    trimmed ONCE for display, never per inner token). The interval is the child's
    first-word start to last-word end — monotonic and within the parent interval
    by construction (the guard rejects anything else).
    """

    word_start: int
    word_end: int
    start_seconds: float
    end_seconds: float
    text: str


class UnsplittableError(ValueError):
    """A segment cannot be word-split. Surfaced to the operator, never guessed
    around: the read path renders such a parent whole, and the split route rejects
    the request with this reason."""


def _validated_words(seg: TranscriptSegment) -> list[_Word] | None:
    """The parent's tokens as validated :class:`_Word`\\ s, or ``None`` if the stored
    ``words`` are absent or structurally unusable.

    Structural only: reconcatenation and enhanced-text checks live in
    :func:`splittable_words`. Requires >= 2 tokens (one word has no interior cut),
    each a dict with a non-empty string ``word`` and finite ``start <= end``, with
    non-decreasing timings bounded by the parent interval.
    """
    raw: Any = seg.words
    if not isinstance(raw, list) or len(raw) < 2:
        return None
    lo = seg.start_seconds
    hi = seg.end_seconds
    out: list[_Word] = []
    prev_start = -math.inf
    for item in raw:
        if not isinstance(item, dict):
            return None
        text = item.get("word")
        if not isinstance(text, str) or text == "":
            return None
        raw_start = item.get("start")
        raw_end = item.get("end")
        if isinstance(raw_start, bool) or isinstance(raw_end, bool):
            return None
        if not isinstance(raw_start, (int, float)) or not isinstance(raw_end, (int, float)):
            return None
        start = float(raw_start)
        end = float(raw_end)
        if not (math.isfinite(start) and math.isfinite(end)) or end < start:
            return None
        # Monotonic non-decreasing starts, and within the parent interval — so
        # every derived child gets a finite, ordered, contained span. A small
        # numeric epsilon guards float round-trips at the interval edges.
        if start < prev_start or start < lo - _EPS or end > hi + _EPS:
            return None
        prev_start = start
        out.append(_Word(start=start, end=end, text=text))
    return out


_EPS = 1e-6


def splittable_words(seg: TranscriptSegment) -> list[_Word] | None:
    """The parent's validated tokens iff the segment is word-splittable, else ``None``.

    Adds the faithfulness checks on top of :func:`_validated_words`: the tokens
    must reconcatenate to ``raw_text`` (exact, or differing only by outer-edge
    whitespace on the whole joined string), and ``enhanced_text`` must be NULL or
    match ``raw_text`` (ignoring outer whitespace). Correction state is NOT checked
    here — the routes own that guard.
    """
    words = _validated_words(seg)
    if words is None:
        return None
    joined = "".join(w.text for w in words)
    if joined != seg.raw_text and joined.strip() != seg.raw_text.strip():
        return None
    if seg.enhanced_text is not None and seg.enhanced_text.strip() != seg.raw_text.strip():
        return None
    return words


def word_count(seg: TranscriptSegment) -> int | None:
    """The number of splittable word tokens, or ``None`` when unsplittable. The
    legal interior cuts are ``0 < word_index < word_count``."""
    words = splittable_words(seg)
    return None if words is None else len(words)


def derive_children(
    seg: TranscriptSegment, cuts: list[int]
) -> list[DerivedChild] | None:
    """Partition a splittable parent's words at ``cuts`` into derived children.

    ``cuts`` are interior word indices ("split before word i"); duplicates and
    order do not matter (they are normalized). Returns ``None`` if the segment is
    unsplittable or any cut is out of ``(0, word_count)`` — the caller renders the
    parent whole rather than inventing offsets. With no valid cuts a single child
    spanning the whole segment is returned, so a parent with only invalid
    boundaries still renders exactly its own text.
    """
    words = splittable_words(seg)
    if words is None:
        return None
    n = len(words)
    interior = sorted({c for c in cuts if 0 < c < n})
    bounds = [0, *interior, n]
    children: list[DerivedChild] = []
    for start_i, end_i in pairwise(bounds):
        group = words[start_i:end_i]
        children.append(
            DerivedChild(
                word_start=start_i,
                word_end=end_i,
                start_seconds=group[0].start,
                end_seconds=group[-1].end,
                text="".join(w.text for w in group).strip(),
            )
        )
    return children


def boundaries_for_run(
    session: Session, run_id: uuid.UUID
) -> dict[uuid.UUID, list[int]]:
    """Every split boundary of a run, grouped by parent segment id (one query).

    The batch-load the read path uses to expand children without an N+1: returns
    ``{parent_segment_id: [sorted word indices]}``, empty for a run with no splits.
    """
    rows = session.execute(
        select(
            SegmentSplitBoundary.parent_segment_id,
            SegmentSplitBoundary.word_index,
        )
        .where(SegmentSplitBoundary.pipeline_run_id == run_id)
        .order_by(SegmentSplitBoundary.parent_segment_id, SegmentSplitBoundary.word_index)
    ).all()
    grouped: dict[uuid.UUID, list[int]] = {}
    for parent_id, word_index in rows:
        grouped.setdefault(parent_id, []).append(word_index)
    return grouped


def record_split(
    session: Session,
    *,
    parent: TranscriptSegment,
    word_index: int,
    operator: str,
) -> None:
    """Insert one split boundary, idempotently (issue #59, the sole split writer).

    Validates ``0 < word_index < word_count`` against the parent's splittable word
    count and raises :class:`UnsplittableError` when the segment cannot be split or
    the index is out of range. The insert is ``ON CONFLICT DO NOTHING`` on
    ``(parent_segment_id, word_index)`` — a replayed / double-clicked "split before
    word i" is a structural no-op, so no client nonce is needed. The caller owns
    the claim lock and the corrected-segment guard.
    """
    count = word_count(parent)
    if count is None:
        raise UnsplittableError(
            "segment cannot be split at a word boundary "
            "(no aligned word timings, or its text was enhanced)"
        )
    if not 0 < word_index < count:
        raise UnsplittableError(
            f"split word_index must be in (0, {count}); got {word_index}"
        )
    session.execute(
        pg_insert(SegmentSplitBoundary)
        .values(
            id=uuid.uuid4(),
            pipeline_run_id=parent.pipeline_run_id,
            parent_segment_id=parent.id,
            word_index=word_index,
            operator=operator,
        )
        .on_conflict_do_nothing(
            constraint="segment_split_boundaries_parent_word_key"
        )
    )
    session.flush()
