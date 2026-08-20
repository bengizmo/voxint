"""Operator annotation layer (issue #86): pure anchor mapping, classification,
hashing, and read-time span resolution.

This module owns the coordinate math behind the annotation contract frozen in
``docs/annotations.md``. Everything here is PURE — no session, no I/O — so the
whole anchor truth table is unit-testable without a database. The DB writer
(``capture_annotation`` and friends) and the routes wrap these functions; they
load the covered segments, their review corrections, and their split boundaries,
then call in here.

Coordinate system (see ``docs/annotations.md`` and
``voxint.adjudication.transcript``): the console renders the CORRECTED variant.
An unsplit line's rendered text is the segment's effective text
(``corrected -> enhanced -> raw``); a split child's rendered text is its word
tokens joined and outer-stripped (``splits.derive_children``). Annotations
always address the IMMUTABLE parent segment; split children are a read-time
projection. Character offsets are Unicode code points everywhere — Python ``str``
indices are code points, so this module never sees UTF-16.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from typing import Any

from voxint.adjudication.splits import derive_children, splittable_words
from voxint.adjudication.transcript import effective_text
from voxint.db.models import (
    MAX_ANNOTATION_QUOTE_CHARS,
    MAX_ANNOTATION_SPAN_SEGMENTS,
    TranscriptSegment,
)

# Anchor kinds (mirror the DB CHECK). segment_range ALWAYS means whole immutable
# parents; a whole split-child selection is word_range, not segment_range.
WORD_RANGE = "word_range"
TEXT_RANGE = "text_range"
SEGMENT_RANGE = "segment_range"

ANCHOR_SCHEMA_VERSION = 1

# The versioned, length-framed hash serialization prefix (docs/annotations.md).
# Changing this or the framing mass-stales every annotation, so it is pinned by
# a golden-hex unit test and would require an anchor_schema_version bump.
_HASH_PREFIX = "annv1"

# Timing precision labels carried on every read/API/export shape so #88 never
# treats a coarse segment interval as a clip-accurate edge.
TIMING_WORD = "word"
TIMING_SEGMENT = "segment"

__all__ = [
    "ANCHOR_SCHEMA_VERSION",
    "SEGMENT_RANGE",
    "TEXT_RANGE",
    "TIMING_SEGMENT",
    "TIMING_WORD",
    "WORD_RANGE",
    "AnnotationStaleError",
    "AnnotationValidationError",
    "CaptureEndpoint",
    "CapturePayload",
    "CoveredSegment",
    "DerivedAnchor",
    "annotation_source_hash",
    "derive_anchor",
    "word_eligible",
]


class AnnotationError(Exception):
    """Base for annotation-domain failures the routes translate to HTTP."""


class AnnotationValidationError(AnnotationError):
    """Malformed / out-of-bounds / cap-violating capture -> 422."""


class AnnotationStaleError(AnnotationError):
    """The client's quote assertion disagrees with the server-derived quote, or a
    refresh's anchor no longer identifies its span deterministically -> 409
    ``X-Voxint-Conflict: stale``."""


# --------------------------------------------------------------------------- #
# Capture payload (wire shape, direction NOT yet normalized)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CaptureEndpoint:
    """One selection endpoint as it arrives on the wire.

    ``segment_id`` is the immutable PARENT. ``child_word_start``/
    ``child_word_end`` are a nullable pair: present iff the endpoint sits in a
    rendered split child, and then they must EXACTLY name a currently rendered
    child of that parent. ``offset`` is a code-point index into that rendered
    line's text (the child's stripped text, or the whole effective text when
    unsplit).
    """

    segment_id: uuid.UUID
    offset: int
    child_word_start: int | None = None
    child_word_end: int | None = None


@dataclass(frozen=True)
class CapturePayload:
    """A full selection: two endpoints plus the client's consistency assertion.

    ``client_quote`` is the client's own slice of its props text — never
    ``Range.toString()``. The server derives the quote independently and a
    mismatch is a stale conflict; it is never stored and never picks the kind.
    """

    start: CaptureEndpoint
    end: CaptureEndpoint
    client_quote: str


# --------------------------------------------------------------------------- #
# Covered-segment view (what the DB layer loads and hands to the pure core)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CoveredSegment:
    """One parent segment in a selection's span, with everything the pure core
    needs: the ORM segment, its operator review correction (``None`` when
    absent), its captured ``segment_index``, and the current split-boundary cuts
    (interior word indices) so child ranges can be validated and derived.

    The DB layer builds these in ``segment_index`` order for the contiguous span
    from the start endpoint's segment to the end endpoint's segment.
    """

    segment: TranscriptSegment
    segment_index: int
    corrected_text: str | None
    cuts: tuple[int, ...] = ()

    @property
    def effective(self) -> str:
        """The rendered/effective text offsets index into and the hash covers."""
        return effective_text(self.segment, self.corrected_text)


# --------------------------------------------------------------------------- #
# Derived anchor (the server-owned result of validating a capture)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DerivedAnchor:
    """The validated, classified anchor plus its captured snapshot — exactly the
    fields the writer persists. Direction is already normalized to transcript
    order (start endpoint precedes end endpoint)."""

    anchor_kind: str
    start_segment_id: uuid.UUID
    end_segment_id: uuid.UUID
    start_segment_index: int
    end_segment_index: int
    start_word_index: int | None
    end_word_index: int | None
    start_char_offset: int | None
    end_char_offset: int | None
    source_text_hash: str
    start_seconds: float | None
    end_seconds: float | None
    quote_text: str
    timing_precision: str


# --------------------------------------------------------------------------- #
# Hash
# --------------------------------------------------------------------------- #


def annotation_source_hash(segment_texts: list[tuple[uuid.UUID, str]]) -> str:
    """Full sha256 hex over a versioned, length-framed serialization of the
    covered segments' effective texts, in the order given (``segment_index``
    order at capture).

    Serialization (docs/annotations.md):

        "annv1" + for each segment: "{segment_id}:{code_point_length}:{text}"

    Length framing plus the segment id make the serialization injective, so a
    segment whose text contains a delimiter cannot collide with a different
    partition. No revision-kind salt: byte-identical visible text stays
    non-stale even if its source tier changed. Pinned by a golden-hex test.
    """
    parts = [_HASH_PREFIX]
    for segment_id, text in segment_texts:
        parts.append(f"{segment_id}:{len(text)}:{text}")
    return hashlib.sha256("".join(parts).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Token-boundary math (child-local <-> parent code-point offsets)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _WordGrid:
    """Precomputed token geometry for one word-eligible parent, in the parent's
    effective-text code-point coordinates.

    ``content_start[i]`` / ``content_end[i]`` bracket token ``i``'s visible
    (non-whitespace) content; a word_range selection endpoint must land exactly
    on a ``content_start`` (as a start) or a ``content_end`` (as an end).
    ``join_starts[i]`` is token ``i``'s start in the raw joined-token string, used
    to project child-local offsets through the outer/child trim deltas.
    """

    parent_text: str
    join_starts: tuple[int, ...]
    lead_joined: int
    lead_parent: int
    content_start: tuple[int, ...]
    content_end: tuple[int, ...]
    token_texts: tuple[str, ...]


def _word_grid(words: list[Any], parent_text: str) -> _WordGrid:
    token_texts = tuple(w.text for w in words)
    joined = "".join(token_texts)
    lead_joined = len(joined) - len(joined.lstrip())
    lead_parent = len(parent_text) - len(parent_text.lstrip())
    join_starts: list[int] = []
    acc = 0
    for tok in token_texts:
        join_starts.append(acc)
        acc += len(tok)

    # Map a joined-coordinate position (within the shared inner content) to a
    # parent-effective-text position by swapping the outer-whitespace deltas.
    def to_parent(join_pos: int) -> int:
        return join_pos - lead_joined + lead_parent

    content_start: list[int] = []
    content_end: list[int] = []
    for i, tok in enumerate(token_texts):
        tok_lead = len(tok) - len(tok.lstrip())
        tok_trail = len(tok) - len(tok.rstrip())
        content_start.append(to_parent(join_starts[i] + tok_lead))
        content_end.append(to_parent(join_starts[i] + len(tok) - tok_trail))
    return _WordGrid(
        parent_text=parent_text,
        join_starts=tuple(join_starts),
        lead_joined=lead_joined,
        lead_parent=lead_parent,
        content_start=tuple(content_start),
        content_end=tuple(content_end),
        token_texts=token_texts,
    )


def _child_local_to_parent_offset(
    grid: _WordGrid, child_word_start: int, child_word_end: int, local_offset: int
) -> int:
    """Map a code-point offset local to a split child's rendered (stripped) text
    to a code-point offset in the parent's effective text.

    Accounts for the parent outer-trim delta (joined tokens vs effective text)
    and the child left-trim delta (the child strips its first token's leading
    whitespace). See ``docs/annotations.md``.
    """
    joined_child = "".join(grid.token_texts[child_word_start:child_word_end])
    child_lead = len(joined_child) - len(joined_child.lstrip())
    stripped_len = len(joined_child.strip())
    if local_offset < 0 or local_offset > stripped_len:
        raise AnnotationValidationError(
            f"child offset {local_offset} out of range 0..{stripped_len}"
        )
    join_pos = grid.join_starts[child_word_start] + child_lead + local_offset
    return join_pos - grid.lead_joined + grid.lead_parent


# --------------------------------------------------------------------------- #
# Word eligibility + endpoint resolution
# --------------------------------------------------------------------------- #


def word_eligible(seg: TranscriptSegment, corrected_text: str | None) -> list[Any] | None:
    """The parent's validated word tokens iff the segment is word-eligible for an
    annotation, else ``None``.

    Word-eligibility = ``splittable_words`` (#82 correction-trace empty, tokens
    reconcatenate to raw, ``enhanced_text`` null-or-raw) AND no operator review
    correction (#58). ``splittable_words`` deliberately does NOT read review
    state (route-owned for splits), so the ``corrected_text`` guard is applied
    here explicitly (docs/annotations.md, Classification).
    """
    if corrected_text is not None:
        return None
    words = splittable_words(seg)
    if words is None:
        return None
    # A whitespace-only token has a zero-width content span and would duplicate
    # its neighbour's boundary offset, making the word-index and precise-timing
    # derivation ambiguous (``_index_where`` would silently pick the empty
    # token). Such tokens are pathological — whisper never emits them — so refuse
    # word-eligibility and let the selection fall to text_range rather than
    # anchor to an empty token. This is deliberately stricter than split
    # eligibility (#59), which owns its own rendering.
    if any(not w.text.strip() for w in words):
        return None
    return words


def _index_where(values: tuple[int, ...], target: int) -> int | None:
    """The index whose value equals ``target``, or ``None``. Token-boundary offsets
    are distinct and sorted, so at most one matches."""
    for i, value in enumerate(values):
        if value == target:
            return i
    return None


@dataclass(frozen=True)
class _ResolvedEndpoint:
    """A validated endpoint: its covered segment, its parent-effective-text
    code-point offset, and (when word-eligible) the token grid for boundary
    classification."""

    covered: CoveredSegment
    parent_offset: int
    words: list[Any] | None
    grid: _WordGrid | None


def _resolve_endpoint(covered: CoveredSegment, ep: CaptureEndpoint) -> _ResolvedEndpoint:
    words = word_eligible(covered.segment, covered.corrected_text)
    parent_text = covered.effective
    plen = len(parent_text)
    grid = _word_grid(words, parent_text) if words is not None else None

    if ep.child_word_start is None and ep.child_word_end is None:
        # Unsplit endpoint: the offset indexes the whole effective text directly.
        if not 0 <= ep.offset <= plen:
            raise AnnotationValidationError(f"offset {ep.offset} out of range 0..{plen}")
        parent_offset = ep.offset
    else:
        # Split-child endpoint: the pair must be both-present and name a currently
        # rendered child; the offset is local to that child's stripped text.
        if ep.child_word_start is None or ep.child_word_end is None:
            raise AnnotationValidationError(
                "child_word_start and child_word_end must be both present or both absent"
            )
        if words is None:
            raise AnnotationValidationError(
                "endpoint segment is not word-eligible; a child range cannot name it"
            )
        # Child coordinates are valid ONLY when the parent is actually rendered
        # as a split — i.e. real cuts yield >= 2 children (the same gate
        # ``attributed_transcript`` uses). ``derive_children`` returns one
        # synthetic whole child for an unsplit parent, so without this length
        # check a forged ``(0, word_count)`` child range on an unsplit line would
        # be accepted (docs/annotations.md: child coords must name a CURRENTLY
        # RENDERED split child). Also closes the un-split race.
        children = derive_children(covered.segment, list(covered.cuts))
        if (
            children is None
            or len(children) <= 1
            or not any(
                c.word_start == ep.child_word_start and c.word_end == ep.child_word_end
                for c in children
            )
        ):
            raise AnnotationValidationError(
                "child_word range does not name a currently rendered split child"
            )
        assert grid is not None  # words is not None -> grid built above
        parent_offset = _child_local_to_parent_offset(
            grid, ep.child_word_start, ep.child_word_end, ep.offset
        )
    return _ResolvedEndpoint(covered=covered, parent_offset=parent_offset, words=words, grid=grid)


# --------------------------------------------------------------------------- #
# Classification + derivation
# --------------------------------------------------------------------------- #


def _derive_quote(covered_span: list[CoveredSegment], start_off: int, end_off: int) -> str:
    """The selected text: start-tail + whole middles + end-head, newline-joined.
    A single-segment selection is a plain slice."""
    if len(covered_span) == 1:
        return covered_span[0].effective[start_off:end_off]
    parts = [covered_span[0].effective[start_off:]]
    parts.extend(cs.effective for cs in covered_span[1:-1])
    parts.append(covered_span[-1].effective[:end_off])
    return "\n".join(parts)


def _classify_and_derive(
    covered_span: list[CoveredSegment],
    start_re: _ResolvedEndpoint,
    end_re: _ResolvedEndpoint,
) -> DerivedAnchor:
    start_off = start_re.parent_offset
    end_off = end_re.parent_offset
    end_len = len(end_re.covered.effective)

    whole = start_off == 0 and end_off == end_len
    all_eligible = all(
        word_eligible(cs.segment, cs.corrected_text) is not None for cs in covered_span
    )
    any_words_null = any(cs.segment.words is None for cs in covered_span)

    start_word_index: int | None = None
    end_word_index: int | None = None
    if all_eligible and start_re.grid is not None and end_re.grid is not None:
        swi = _index_where(start_re.grid.content_start, start_off)
        ewi = _index_where(end_re.grid.content_end, end_off)
        if swi is not None and ewi is not None:
            start_word_index, end_word_index = swi, ewi + 1

    on_boundaries = start_word_index is not None and end_word_index is not None
    if whole:
        kind = SEGMENT_RANGE
    elif all_eligible and on_boundaries:
        kind = WORD_RANGE
    elif any_words_null:
        # docs/annotations.md: a partial selection touching a segment with no word
        # timings degrades to a whole-segment anchor (verbatim quote preserved).
        kind = SEGMENT_RANGE
    else:
        kind = TEXT_RANGE

    quote = _derive_quote(covered_span, start_off, end_off)
    if len(quote) > MAX_ANNOTATION_QUOTE_CHARS:
        raise AnnotationValidationError(
            f"quote is {len(quote)} code points, exceeds {MAX_ANNOTATION_QUOTE_CHARS}"
        )
    source_hash = annotation_source_hash([(cs.segment.id, cs.effective) for cs in covered_span])

    if kind == WORD_RANGE:
        assert start_re.words is not None and end_re.words is not None
        assert start_word_index is not None and end_word_index is not None
        start_seconds: float | None = float(start_re.words[start_word_index].start)
        end_seconds: float | None = float(end_re.words[end_word_index - 1].end)
        timing = TIMING_WORD
        char_start: int | None = None
        char_end: int | None = None
    else:
        start_seconds = end_seconds = None
        timing = TIMING_SEGMENT
        start_word_index = end_word_index = None
        if kind == TEXT_RANGE:
            char_start, char_end = start_off, end_off
        else:  # SEGMENT_RANGE stores no offsets
            char_start = char_end = None

    return DerivedAnchor(
        anchor_kind=kind,
        start_segment_id=start_re.covered.segment.id,
        end_segment_id=end_re.covered.segment.id,
        start_segment_index=start_re.covered.segment_index,
        end_segment_index=end_re.covered.segment_index,
        start_word_index=start_word_index,
        end_word_index=end_word_index,
        start_char_offset=char_start,
        end_char_offset=char_end,
        source_text_hash=source_hash,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        quote_text=quote,
        timing_precision=timing,
    )


def derive_anchor(run_segments: list[CoveredSegment], payload: CapturePayload) -> DerivedAnchor:
    """Validate a capture payload against the current render and derive the
    server-owned anchor: normalize direction, extract the contiguous covered
    span, classify (segment/word/text range), and derive the quote, hash, and
    timing. The ``client_quote`` is compared against the server-derived quote and
    a mismatch raises :class:`AnnotationStaleError` (never stored).

    ``run_segments`` is the run's segments as :class:`CoveredSegment` in
    ``segment_index`` order. The caller (DB layer) is responsible for the 404
    fail-closed check that both endpoint segment ids belong to the run before
    calling; a missing id here is a defensive validation error.
    """
    lookup = {cs.segment.id: cs for cs in run_segments}
    position = {cs.segment.id: i for i, cs in enumerate(run_segments)}
    for ep in (payload.start, payload.end):
        if ep.segment_id not in lookup:
            raise AnnotationValidationError(f"segment {ep.segment_id} is not in this run")

    start_re = _resolve_endpoint(lookup[payload.start.segment_id], payload.start)
    end_re = _resolve_endpoint(lookup[payload.end.segment_id], payload.end)

    # Normalize direction by transcript position: (segment order, offset).
    start_key = (position[payload.start.segment_id], start_re.parent_offset)
    end_key = (position[payload.end.segment_id], end_re.parent_offset)
    if end_key < start_key:
        start_re, end_re = end_re, start_re
        start_key, end_key = end_key, start_key

    if start_key == end_key:
        raise AnnotationValidationError("empty selection (start and end coincide)")

    start_pos = position[start_re.covered.segment.id]
    end_pos = position[end_re.covered.segment.id]
    covered_span = run_segments[start_pos : end_pos + 1]
    if len(covered_span) > MAX_ANNOTATION_SPAN_SEGMENTS:
        raise AnnotationValidationError(
            f"selection spans {len(covered_span)} segments, exceeds {MAX_ANNOTATION_SPAN_SEGMENTS}"
        )

    derived = _classify_and_derive(covered_span, start_re, end_re)
    if payload.client_quote != derived.quote_text:
        raise AnnotationStaleError("client quote does not match the server-derived quote")
    return derived
