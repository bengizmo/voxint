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
import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from voxint.adjudication.resolver import review_states
from voxint.adjudication.splits import boundaries_for_run, derive_children, splittable_words
from voxint.adjudication.transcript import TranscriptLine, effective_text
from voxint.db.models import (
    HIGHLIGHT_PALETTE_SIZE,
    MAX_ANNOTATION_NOTE_CHARS,
    MAX_ANNOTATION_QUOTE_CHARS,
    MAX_ANNOTATION_SPAN_SEGMENTS,
    MAX_TAGS_PER_ANNOTATION,
    AnnotationTag,
    AnnotationTagLink,
    TranscriptAnnotation,
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
    "AnnotationError",
    "AnnotationIdempotencyError",
    "AnnotationNotFoundError",
    "AnnotationStaleError",
    "AnnotationValidationError",
    "CaptureEndpoint",
    "CapturePayload",
    "CoveredSegment",
    "DerivedAnchor",
    "ResolvedAnnotation",
    "ResolvedSpan",
    "StoredAnchor",
    "annotation_source_hash",
    "annotations_for_run",
    "capture_annotation",
    "derive_anchor",
    "load_covered_segments",
    "reanchor_annotation",
    "refresh_annotation",
    "resolve_annotation_spans",
    "soft_delete_annotation",
    "stored_anchor_from_row",
    "update_annotation",
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


class AnnotationNotFoundError(AnnotationError):
    """The run, an endpoint segment, or the annotation itself is unknown, foreign
    to this run, or forged -> 404 (fail closed; forged is not distinguished from
    missing so nothing leaks)."""


class AnnotationIdempotencyError(AnnotationError):
    """A create nonce was replayed with a DIFFERENT payload than the row it already
    keys -> 409 ``X-Voxint-Conflict: idempotency``. A same-payload replay is not an
    error (it returns the original row); only a fingerprint mismatch is."""


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


def _require_quote_within_cap(quote: str) -> None:
    """A derived quote (capture or refresh) over the stored cap is a 422 — never a
    DB CHECK violation surfaced as an uncaught IntegrityError."""
    if len(quote) > MAX_ANNOTATION_QUOTE_CHARS:
        raise AnnotationValidationError(
            f"quote is {len(quote)} code points, exceeds {MAX_ANNOTATION_QUOTE_CHARS}"
        )


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
    _require_quote_within_cap(quote)
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


# --------------------------------------------------------------------------- #
# DB writer: load the covered segments the pure core needs
# --------------------------------------------------------------------------- #


def load_covered_segments(session: Session, run_id: uuid.UUID) -> list[CoveredSegment]:
    """A run's segments as :class:`CoveredSegment` in ``segment_index`` order, each
    carrying its operator review correction (#58) and split-boundary cuts (#59).

    This is the render the annotation writer and read resolver both validate
    against: the corrected effective text every offset indexes into, and the cuts
    that derive the split children a child range must name. One batch load each for
    corrections and boundaries (no N+1), mirroring ``attributed_transcript``.
    """
    segments = (
        session.execute(
            select(TranscriptSegment)
            .where(TranscriptSegment.pipeline_run_id == run_id)
            .order_by(TranscriptSegment.segment_index)
        )
        .scalars()
        .all()
    )
    review = review_states(session, run_id)
    boundaries = boundaries_for_run(session, run_id)
    covered: list[CoveredSegment] = []
    for seg in segments:
        rs = review.get(seg.id)
        covered.append(
            CoveredSegment(
                segment=seg,
                segment_index=seg.segment_index,
                corrected_text=rs.corrected_text if rs is not None else None,
                cuts=tuple(boundaries.get(seg.id, ())),
            )
        )
    return covered


def _require_endpoints_in_run(covered: list[CoveredSegment], payload: CapturePayload) -> None:
    """Fail closed (404) when either endpoint segment id is not one of the run's
    segments — a cross-run or forged id. Runs BEFORE ``derive_anchor`` so a forged
    id is a 404, never leaked as a 422 validation detail."""
    ids = {cs.segment.id for cs in covered}
    for ep in (payload.start, payload.end):
        if ep.segment_id not in ids:
            raise AnnotationNotFoundError(f"segment {ep.segment_id} is not in this run")


# --------------------------------------------------------------------------- #
# DB writer: metadata + tag validation
# --------------------------------------------------------------------------- #


def _validate_color(color_index: int) -> None:
    if not 0 <= color_index < HIGHLIGHT_PALETTE_SIZE:
        raise AnnotationValidationError(
            f"color_index {color_index} out of range 0..{HIGHLIGHT_PALETTE_SIZE - 1}"
        )


def _normalize_note(note: str | None) -> str | None:
    """Empty is stored as NULL (no empty rendering); length is capped at 422."""
    if not note:
        return None
    if len(note) > MAX_ANNOTATION_NOTE_CHARS:
        raise AnnotationValidationError(
            f"note is {len(note)} code points, exceeds {MAX_ANNOTATION_NOTE_CHARS}"
        )
    return note


def _resolve_tag_ids(session: Session, tag_ids: list[uuid.UUID]) -> list[uuid.UUID]:
    """Deduplicate (preserving order), enforce the per-annotation cap (422), and
    verify every id names a real tag (404 unknown). Archived tags are still
    attachable — the picker hides them, but a caller may re-attach an existing one.
    """
    seen: set[uuid.UUID] = set()
    deduped: list[uuid.UUID] = []
    for tid in tag_ids:
        if tid not in seen:
            seen.add(tid)
            deduped.append(tid)
    if len(deduped) > MAX_TAGS_PER_ANNOTATION:
        raise AnnotationValidationError(
            f"{len(deduped)} tags exceeds the {MAX_TAGS_PER_ANNOTATION}-per-annotation cap"
        )
    if deduped:
        found = set(
            session.execute(select(AnnotationTag.id).where(AnnotationTag.id.in_(deduped))).scalars()
        )
        missing = [tid for tid in deduped if tid not in found]
        if missing:
            raise AnnotationNotFoundError(f"unknown tag id {missing[0]}")
    return deduped


def _replace_tag_links(
    session: Session, annotation_id: uuid.UUID, tag_ids: list[uuid.UUID]
) -> None:
    """Set an annotation's tag links to exactly ``tag_ids`` (already validated and
    deduped). Each insert is ON CONFLICT DO NOTHING so a concurrent identical link
    is a structural no-op (the composite PK is the guard)."""
    session.execute(
        delete(AnnotationTagLink).where(AnnotationTagLink.annotation_id == annotation_id)
    )
    for tid in tag_ids:
        session.execute(
            pg_insert(AnnotationTagLink)
            .values(annotation_id=annotation_id, tag_id=tid)
            .on_conflict_do_nothing()
        )


def _create_fingerprint(
    payload: CapturePayload, color_index: int, note: str | None, tag_ids: list[uuid.UUID]
) -> str:
    """A sha256 of the canonical create payload (P1-7). A replayed nonce whose
    fingerprint matches returns the original row; a mismatch is a 409 idempotency
    conflict. Deterministic over the SAME shape the writer persists, so a
    semantically-identical retry (empty vs absent note, a duplicated or reordered
    tag id) is not a false conflict: the note is normalized to its stored form
    (empty -> null) and the tags are deduplicated and sorted."""
    canonical = json.dumps(
        {
            "start": [
                str(payload.start.segment_id),
                payload.start.offset,
                payload.start.child_word_start,
                payload.start.child_word_end,
            ],
            "end": [
                str(payload.end.segment_id),
                payload.end.offset,
                payload.end.child_word_start,
                payload.end.child_word_end,
            ],
            "client_quote": payload.client_quote,
            "color_index": color_index,
            "note": note or None,
            "tags": sorted({str(t) for t in tag_ids}),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _idempotency_key(run_id: uuid.UUID, nonce: str) -> str:
    """Namespace the client nonce by run so the globally-UNIQUE key cannot collide
    across runs that happen to mint the same nonce."""
    return f"{run_id}:{nonce}"


def _assign_anchor(row: TranscriptAnnotation, derived: DerivedAnchor) -> None:
    """Copy a derived anchor + its captured snapshot onto a row (create + re-anchor
    share this so the two can never persist different shapes for one derivation)."""
    row.anchor_schema_version = ANCHOR_SCHEMA_VERSION
    row.anchor_kind = derived.anchor_kind
    row.start_segment_id = derived.start_segment_id
    row.end_segment_id = derived.end_segment_id
    row.start_segment_index = derived.start_segment_index
    row.end_segment_index = derived.end_segment_index
    row.start_word_index = derived.start_word_index
    row.end_word_index = derived.end_word_index
    row.start_char_offset = derived.start_char_offset
    row.end_char_offset = derived.end_char_offset
    row.source_text_hash = derived.source_text_hash
    row.start_seconds = derived.start_seconds
    row.end_seconds = derived.end_seconds
    row.quote_text = derived.quote_text


# --------------------------------------------------------------------------- #
# DB writer: capture (sole create path, idempotent)
# --------------------------------------------------------------------------- #


def capture_annotation(
    session: Session,
    *,
    run_id: uuid.UUID,
    payload: CapturePayload,
    operator: str,
    nonce: str,
    color_index: int,
    note: str | None = None,
    tag_ids: list[uuid.UUID] | None = None,
) -> TranscriptAnnotation:
    """The sole annotation create path (issue #86).

    Idempotent by the run-namespaced ``nonce``: replaying it with the SAME payload
    returns the original row (including a soft-deleted one — never resurrecting or
    duplicating); replaying it with a different payload is a 409 idempotency
    conflict. The replay short-circuit runs BEFORE anchor derivation, so a replay
    of a since-invalidated selection still returns its row.

    A fresh create validates metadata (colour/note/tag caps -> 422, unknown tag ->
    404), loads the run render, fails closed (404) on a cross-run/forged endpoint,
    derives the server-owned anchor (``derive_anchor`` -> 422 / 409 stale on a
    client-quote mismatch), and inserts with savepoint-adopt so a concurrent
    same-nonce writer is adopted rather than raising.
    """
    tags = list(tag_ids or [])
    key = _idempotency_key(run_id, nonce)
    fingerprint = _create_fingerprint(payload, color_index, note, tags)

    existing = session.execute(
        select(TranscriptAnnotation).where(TranscriptAnnotation.idempotency_key == key)
    ).scalar_one_or_none()
    if existing is not None:
        return _adopt_or_conflict(existing, fingerprint)

    # Fresh create: validate fully now (a replay above never reaches here).
    _validate_color(color_index)
    normalized_note = _normalize_note(note)
    deduped_tags = _resolve_tag_ids(session, tags)
    covered = load_covered_segments(session, run_id)
    if not covered:
        raise AnnotationNotFoundError(f"run {run_id} has no transcript")
    _require_endpoints_in_run(covered, payload)
    derived = derive_anchor(covered, payload)

    row = TranscriptAnnotation(
        id=uuid.uuid4(),
        pipeline_run_id=run_id,
        color_index=color_index,
        note=normalized_note,
        operator=operator,
        idempotency_key=key,
        request_fingerprint=fingerprint,
    )
    _assign_anchor(row, derived)
    try:
        # Savepoint, not a bare flush: the route composes this into the claimed
        # write transaction, and losing the same-nonce race must not roll that back.
        with session.begin_nested():
            session.add(row)
            session.flush()
    except IntegrityError:
        adopted = session.execute(
            select(TranscriptAnnotation).where(TranscriptAnnotation.idempotency_key == key)
        ).scalar_one_or_none()
        if adopted is None:
            raise  # not a replay race — a real FK/CHECK violation
        return _adopt_or_conflict(adopted, fingerprint)

    _replace_tag_links(session, row.id, deduped_tags)
    session.flush()
    return row


def _adopt_or_conflict(existing: TranscriptAnnotation, fingerprint: str) -> TranscriptAnnotation:
    """A row already keys this nonce: adopt it on a matching fingerprint, else the
    nonce was reused for a different payload (409 idempotency)."""
    if existing.request_fingerprint == fingerprint:
        return existing
    raise AnnotationIdempotencyError(
        f"nonce replayed with a different payload than annotation {existing.id}"
    )


# --------------------------------------------------------------------------- #
# DB writer: load + mutate an existing annotation
# --------------------------------------------------------------------------- #


def _load_live_annotation(
    session: Session, run_id: uuid.UUID, annotation_id: uuid.UUID
) -> TranscriptAnnotation:
    """The run's live (not soft-deleted) annotation, or 404. Run-scoped, so a
    foreign or forged id is indistinguishable from missing (fail closed)."""
    row = session.execute(
        select(TranscriptAnnotation).where(
            TranscriptAnnotation.id == annotation_id,
            TranscriptAnnotation.pipeline_run_id == run_id,
            TranscriptAnnotation.deleted_at.is_(None),
        )
    ).scalar_one_or_none()
    if row is None:
        raise AnnotationNotFoundError(f"annotation {annotation_id} not found in this run")
    return row


def update_annotation(
    session: Session,
    *,
    run_id: uuid.UUID,
    annotation_id: uuid.UUID,
    color_index: int,
    note: str | None,
    tag_ids: list[uuid.UUID] | None,
) -> TranscriptAnnotation:
    """Replace an annotation's metadata (colour, note, tag set) — last write wins.
    The anchor and its captured snapshot are untouched (that is ``reanchor``)."""
    row = _load_live_annotation(session, run_id, annotation_id)
    _validate_color(color_index)
    normalized_note = _normalize_note(note)
    deduped_tags = _resolve_tag_ids(session, list(tag_ids or []))
    row.color_index = color_index
    row.note = normalized_note
    _replace_tag_links(session, row.id, deduped_tags)
    session.flush()
    return row


def reanchor_annotation(
    session: Session,
    *,
    run_id: uuid.UUID,
    annotation_id: uuid.UUID,
    payload: CapturePayload,
) -> TranscriptAnnotation:
    """Atomically replace an annotation's anchor + snapshot from a complete fresh
    capture, re-validated exactly like a create. The recovery path for a stale
    anchor; metadata (colour/note/tags) is preserved."""
    row = _load_live_annotation(session, run_id, annotation_id)
    covered = load_covered_segments(session, run_id)
    _require_endpoints_in_run(covered, payload)
    derived = derive_anchor(covered, payload)
    _assign_anchor(row, derived)
    session.flush()
    return row


def refresh_annotation(
    session: Session, *, run_id: uuid.UUID, annotation_id: uuid.UUID
) -> TranscriptAnnotation:
    """Re-derive quote/hash/seconds when the anchor still identifies its span
    deterministically (Phase 0 #6). A matching hash is a no-op. Otherwise:

    - ``segment_range``: whole segments are stable identity, so re-derive the
      whole-span quote and hash.
    - ``word_range``: refreshable only while every covered segment still grants
      word-eligibility (immutable word timings + boundaries); a lost eligibility
      (a late correction) refuses with 409 stale-anchor.
    - ``text_range``: a changed hash means the code-point offsets are dead ->
      refuse with 409 stale-anchor (recover via re-anchor).

    A re-derived quote that would exceed the stored-quote cap (later corrections
    grew the covered text past ``MAX_ANNOTATION_QUOTE_CHARS``) is a 422 before the
    row is touched — never an uncaught DB CHECK violation.
    """
    row = _load_live_annotation(session, run_id, annotation_id)
    covered = load_covered_segments(session, run_id)
    span = _covered_span_for_row(covered, row)
    current_hash = annotation_source_hash([(cs.segment.id, cs.effective) for cs in span])
    if current_hash == row.source_text_hash:
        return row  # text unchanged -> nothing to refresh

    if row.anchor_kind == SEGMENT_RANGE:
        quote = _derive_quote(span, 0, len(span[-1].effective))
        _require_quote_within_cap(quote)
        row.quote_text = quote
        row.source_text_hash = current_hash
        session.flush()
        return row

    if row.anchor_kind == WORD_RANGE:
        eligibility = [word_eligible(cs.segment, cs.corrected_text) for cs in span]
        if any(words is None for words in eligibility):
            raise AnnotationStaleError("word-range anchor lost word-eligibility")
        assert row.start_word_index is not None and row.end_word_index is not None
        first_words = eligibility[0]
        last_words = eligibility[-1]
        assert first_words is not None and last_words is not None
        grid_first = _word_grid(first_words, span[0].effective)
        grid_last = _word_grid(last_words, span[-1].effective)
        start_off = grid_first.content_start[row.start_word_index]
        end_off = grid_last.content_end[row.end_word_index - 1]
        quote = _derive_quote(span, start_off, end_off)
        _require_quote_within_cap(quote)
        row.quote_text = quote
        row.start_seconds = float(first_words[row.start_word_index].start)
        row.end_seconds = float(last_words[row.end_word_index - 1].end)
        row.source_text_hash = current_hash
        session.flush()
        return row

    # text_range with a changed hash: the offsets no longer name the same text.
    raise AnnotationStaleError("text-range anchor is stale; re-anchor to recover")


def soft_delete_annotation(
    session: Session, *, run_id: uuid.UUID, annotation_id: uuid.UUID
) -> TranscriptAnnotation:
    """Soft-delete an annotation (idempotent). An unknown/foreign id is 404; a
    DELETE of an already-deleted row is a no-op returning the deleted row (204),
    never a 404 — the row still exists, and a create replay still finds it."""
    row = session.execute(
        select(TranscriptAnnotation).where(
            TranscriptAnnotation.id == annotation_id,
            TranscriptAnnotation.pipeline_run_id == run_id,
        )
    ).scalar_one_or_none()
    if row is None:
        raise AnnotationNotFoundError(f"annotation {annotation_id} not found in this run")
    if row.deleted_at is None:
        row.deleted_at = datetime.now(UTC)
        session.flush()
    return row


def annotations_for_run(
    session: Session, run_id: uuid.UUID, *, tag_ids: list[uuid.UUID] | None = None
) -> list[TranscriptAnnotation]:
    """A run's live annotations in transcript order (start segment index, then
    creation). Soft-deleted rows are excluded. A non-empty ``tag_ids`` filters to
    annotations carrying ANY of those tags (repeated ``?tag=`` is an OR-union,
    identically in the panel and exports)."""
    stmt = select(TranscriptAnnotation).where(
        TranscriptAnnotation.pipeline_run_id == run_id,
        TranscriptAnnotation.deleted_at.is_(None),
    )
    if tag_ids:
        stmt = stmt.where(
            TranscriptAnnotation.id.in_(
                select(AnnotationTagLink.annotation_id).where(
                    AnnotationTagLink.tag_id.in_(list(tag_ids))
                )
            )
        )
    stmt = stmt.order_by(TranscriptAnnotation.start_segment_index, TranscriptAnnotation.created_at)
    return list(session.execute(stmt).scalars())


def _covered_span_for_row(
    covered: list[CoveredSegment], row: TranscriptAnnotation
) -> list[CoveredSegment]:
    """The contiguous covered-segment span an existing anchor addresses, sliced by
    the endpoint segment ids (immutable, so ``segment_index`` order is stable)."""
    pos = {cs.segment.id: i for i, cs in enumerate(covered)}
    si = pos.get(row.start_segment_id)
    ei = pos.get(row.end_segment_id)
    if si is None or ei is None:
        # An endpoint segment vanished — only possible via a CASCADE that also
        # deletes this row, so this is defensive; treat as unresolvable.
        raise AnnotationNotFoundError(f"annotation {row.id} endpoints are no longer in the run")
    return covered[si : ei + 1]


# --------------------------------------------------------------------------- #
# Read resolver (PURE): stored anchors -> render-order spans + staleness
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StoredAnchor:
    """A persisted anchor as the pure resolver sees it (decoupled from the ORM so
    the read path is unit-testable without a database). The route builds these
    from ``TranscriptAnnotation`` rows via :func:`stored_anchor_from_row`."""

    annotation_id: uuid.UUID
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


@dataclass(frozen=True)
class ResolvedSpan:
    """One highlight piece to render: a code-point ``[start, end)`` slice of the
    line at ``line_index`` in the current render order."""

    line_index: int
    start: int
    end: int


@dataclass(frozen=True)
class ResolvedAnnotation:
    """The read-time resolution of one stored anchor against the current render:
    where to paint it (``spans``), whether its captured text drifted (``stale``,
    which suppresses inline spans), the honest timing precision + display seconds,
    the CURRENT speakers over its covered lines (attribution is always live, never
    the captured copy), and the start-line locator used when stale."""

    annotation_id: uuid.UUID
    anchor_kind: str
    stale: bool
    timing_precision: str
    start_seconds: float | None
    end_seconds: float | None
    speakers: tuple[str, ...]
    spans: tuple[ResolvedSpan, ...]
    locator_line_index: int | None


def stored_anchor_from_row(row: TranscriptAnnotation) -> StoredAnchor:
    """Project a persisted annotation row onto the pure :class:`StoredAnchor`."""
    return StoredAnchor(
        annotation_id=row.id,
        anchor_kind=row.anchor_kind,
        start_segment_id=row.start_segment_id,
        end_segment_id=row.end_segment_id,
        start_segment_index=row.start_segment_index,
        end_segment_index=row.end_segment_index,
        start_word_index=row.start_word_index,
        end_word_index=row.end_word_index,
        start_char_offset=row.start_char_offset,
        end_char_offset=row.end_char_offset,
        source_text_hash=row.source_text_hash,
        start_seconds=row.start_seconds,
        end_seconds=row.end_seconds,
    )


def _project_interval_to_lines(
    p0: int,
    p1: int,
    parent_lines: list[tuple[int, TranscriptLine]],
    grid: _WordGrid | None,
) -> list[ResolvedSpan]:
    """Map a parent-effective code-point interval ``[p0, p1)`` onto the render
    lines of that parent. An unsplit line's text IS the effective text (1:1); a
    split child covers a contiguous parent-offset window ``[A, B)`` (linear map,
    slope 1), so the overlap projects back to child-local by subtracting ``A``."""
    spans: list[ResolvedSpan] = []
    for line_index, ln in parent_lines:
        if ln.word_start is None or ln.word_end is None:
            start = max(p0, 0)
            end = min(p1, len(ln.text))
            if start < end:
                spans.append(ResolvedSpan(line_index, start, end))
            continue
        # Split child: its stripped text spans parent offsets [A, B).
        assert grid is not None  # a split child exists only for a word-eligible parent
        a = _child_local_to_parent_offset(grid, ln.word_start, ln.word_end, 0)
        b = a + len(ln.text)
        start = max(p0, a)
        end = min(p1, b)
        if start < end:
            spans.append(ResolvedSpan(line_index, start - a, end - a))
    return spans


def _word_interval(grid: _WordGrid, word_start: int, word_end: int) -> tuple[int, int]:
    """The parent-effective code-point interval covering tokens ``[word_start,
    word_end)`` — from the first token's content start to the last token's content
    end, so leading/trailing token whitespace is not highlighted."""
    return grid.content_start[word_start], grid.content_end[word_end - 1]


def resolve_annotation_spans(
    lines: list[TranscriptLine],
    covered_segments: list[CoveredSegment],
    stored_anchors: list[StoredAnchor],
) -> list[ResolvedAnnotation]:
    """Resolve stored anchors against the CURRENT render — the inverse of the
    capture mapping (issue #86). Pure: ``lines`` is the render-order line list from
    ``attributed_transcript`` and ``covered_segments`` the matching parent segments
    (both built from one run at one time), so no session is touched here.

    For each anchor: recompute the source hash over current effective texts (a
    mismatch, or a word_range whose segments lost word-eligibility, is ``stale``);
    a stale anchor yields NO inline spans (only the start-line locator), never
    painting dead offsets onto changed text. A live anchor maps to render-order
    per-line code-point spans — word_range and text_range through the shared
    interval projection (intersecting split-child windows), segment_range as whole
    covered lines. Speakers are read live from the lines; display seconds fall back
    to the coarse covered-segment interval when precise seconds were not stored.
    """
    pos_by_id = {cs.segment.id: i for i, cs in enumerate(covered_segments)}
    lines_by_parent: dict[uuid.UUID, list[tuple[int, TranscriptLine]]] = {}
    for idx, ln in enumerate(lines):
        if ln.segment_id is not None:
            lines_by_parent.setdefault(ln.segment_id, []).append((idx, ln))

    grid_cache: dict[uuid.UUID, _WordGrid | None] = {}

    def grid_for(cs: CoveredSegment) -> _WordGrid | None:
        if cs.segment.id not in grid_cache:
            words = word_eligible(cs.segment, cs.corrected_text)
            grid_cache[cs.segment.id] = (
                _word_grid(words, cs.effective) if words is not None else None
            )
        return grid_cache[cs.segment.id]

    results: list[ResolvedAnnotation] = []
    for anc in stored_anchors:
        si = pos_by_id.get(anc.start_segment_id)
        ei = pos_by_id.get(anc.end_segment_id)
        start_lines = lines_by_parent.get(anc.start_segment_id)
        locator = start_lines[0][0] if start_lines else None

        if si is None or ei is None:
            # Endpoints no longer in the render (defensive): unresolvable + stale.
            results.append(
                ResolvedAnnotation(
                    annotation_id=anc.annotation_id,
                    anchor_kind=anc.anchor_kind,
                    stale=True,
                    timing_precision=(
                        TIMING_WORD if anc.anchor_kind == WORD_RANGE else TIMING_SEGMENT
                    ),
                    start_seconds=anc.start_seconds,
                    end_seconds=anc.end_seconds,
                    speakers=(),
                    spans=(),
                    locator_line_index=locator,
                )
            )
            continue

        span = covered_segments[si : ei + 1]
        current_hash = annotation_source_hash([(cs.segment.id, cs.effective) for cs in span])
        stale = current_hash != anc.source_text_hash
        eligibility = {cs.segment.id: word_eligible(cs.segment, cs.corrected_text) for cs in span}
        if anc.anchor_kind == WORD_RANGE and any(w is None for w in eligibility.values()):
            # Lost word-eligibility (a late correction): never token-slice the
            # corrected text — degrade to stale (Classification, docs/annotations.md).
            stale = True

        # Speakers are always resolved live from the current lines (attribution
        # honesty), even when the text is stale.
        speakers: list[str] = []
        for cs in span:
            for _, ln in lines_by_parent.get(cs.segment.id, []):
                if ln.speaker not in speakers:
                    speakers.append(ln.speaker)

        timing_precision = TIMING_WORD if anc.anchor_kind == WORD_RANGE else TIMING_SEGMENT
        start_seconds = (
            anc.start_seconds if anc.start_seconds is not None else span[0].segment.start_seconds
        )
        end_seconds = (
            anc.end_seconds if anc.end_seconds is not None else span[-1].segment.end_seconds
        )

        spans: list[ResolvedSpan] = []
        if not stale:
            spans = _resolve_live_spans(anc, span, lines_by_parent, grid_for)

        results.append(
            ResolvedAnnotation(
                annotation_id=anc.annotation_id,
                anchor_kind=anc.anchor_kind,
                stale=stale,
                timing_precision=timing_precision,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                speakers=tuple(speakers),
                spans=tuple(spans),
                locator_line_index=locator,
            )
        )
    return results


def _resolve_live_spans(
    anc: StoredAnchor,
    span: list[CoveredSegment],
    lines_by_parent: dict[uuid.UUID, list[tuple[int, TranscriptLine]]],
    grid_for: Any,
) -> list[ResolvedSpan]:
    """Per-parent interval projection for a non-stale anchor. word_range and
    text_range compute a code-point interval per covered parent (whole middles,
    the endpoint's offset at each end) and project it onto that parent's lines;
    segment_range highlights each covered line whole."""
    n = len(span)
    spans: list[ResolvedSpan] = []
    for j, cs in enumerate(span):
        parent_lines = lines_by_parent.get(cs.segment.id, [])
        if not parent_lines:
            continue
        if anc.anchor_kind == SEGMENT_RANGE:
            for line_index, ln in parent_lines:
                if len(ln.text) > 0:
                    spans.append(ResolvedSpan(line_index, 0, len(ln.text)))
            continue

        grid = grid_for(cs)
        effective_len = len(cs.effective)
        if anc.anchor_kind == WORD_RANGE:
            assert anc.start_word_index is not None and anc.end_word_index is not None
            assert grid is not None
            token_count = len(grid.content_start)
            if n == 1:
                w0, w1 = anc.start_word_index, anc.end_word_index
            elif j == 0:
                w0, w1 = anc.start_word_index, token_count
            elif j == n - 1:
                w0, w1 = 0, anc.end_word_index
            else:
                w0, w1 = 0, token_count
            p0, p1 = _word_interval(grid, w0, w1)
        else:  # TEXT_RANGE
            assert anc.start_char_offset is not None and anc.end_char_offset is not None
            if n == 1:
                p0, p1 = anc.start_char_offset, anc.end_char_offset
            elif j == 0:
                p0, p1 = anc.start_char_offset, effective_len
            elif j == n - 1:
                p0, p1 = 0, anc.end_char_offset
            else:
                p0, p1 = 0, effective_len

        spans.extend(_project_interval_to_lines(p0, p1, parent_lines, grid))
    return spans
