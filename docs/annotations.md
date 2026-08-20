# Operator annotations: the anchor contract

Audience: contributors and integrators. Operator-facing usage lives in the
console how-to once the feature ships. This document freezes the contracts
behind the annotation layer (issue #86): how a selected transcript span is
addressed, validated, stored, rendered, and detected as stale. Code must match
this document; a deliberate change here is a schema-version event, not a drift.

## What an annotation is

An annotation is an operator-owned marker over a span of the reviewed
transcript: a highlight color, zero or more flat tags, and an optional margin
note. Annotations are a mutable workspace, not an adjudication record: they can
be edited and soft-deleted, and they never modify pipeline evidence. The
transcript columns `raw_text`, `enhanced_text`, and the review-state
`corrected_text` are never written by any annotation code path. A contract test
enforces this.

## Coordinate system

The console renders the transcript as the ordered line list produced by
`attributed_transcript` (`src/voxint/adjudication/transcript.py`). Split
parents expand into derived child lines at read time; children carry the
immutable parent `segment_id` plus a half-open `[word_start, word_end)` window
into the parent's word tokens. Annotations always address the immutable parent
segment, in one of three anchor kinds. Rendered-line indices are never
persisted; they are a per-render projection.

Character offsets are Unicode code points, everywhere. The browser's DOM
exposes UTF-16 code units; the client converts to code points exactly once, in
`frontend/src/lib/selection.ts`, before anything touches the wire. The server
never sees or stores UTF-16 units.

## Anchor kinds (`anchor_schema_version` = 1)

- `word_range`: parent segment endpoints plus half-open parent-word indices.
  Granted only when every covered segment is word-faithful (see
  Classification) and both selection endpoints fall exactly on token
  boundaries. The only kind with precise, word-timing-derived seconds.
- `text_range`: parent segment endpoints plus half-open code-point offsets
  into each endpoint segment's effective text as it existed at capture.
  Required whenever the displayed text is not word-faithful raw text
  (enhanced or corrected).
- `segment_range`: whole immutable parent segments, no offsets. The fallback
  for whole-segment selections and for partial selections on segments with no
  word timings (`words IS NULL`): the exact selected text is preserved as the
  quote, but display and seek degrade honestly to whole-segment granularity.
  Timing is never fabricated.

All kinds capture, server-side, at write time: the derived `quote_text`, a
`source_text_hash` over the covered effective texts, and (word_range only)
precise `start_seconds` and `end_seconds`.

## Wire schema for capture

Create, re-anchor, and the live pull-quote request share one capture payload.
Per selection endpoint (start and end):

- `segment_id`: the immutable parent segment UUID.
- `child_word_start`, `child_word_end`: a nullable pair. When present it must
  exactly name a currently rendered split child of that parent (derived from
  the current split boundaries); anything else is a 422. Null for unsplit
  lines.
- `offset`: a code-point index into that rendered line's text. Half-open
  across the pair: the start offset is inclusive, the end offset exclusive.

Plus one `client_quote` field: the client's own slice of its props text (never
`Range.toString()`, whose whitespace behavior at block boundaries is not a
stable protocol). The server derives the quote independently from stored text
and compares; a mismatch is a 409 stale conflict. `client_quote` is a
consistency assertion only. It never becomes stored text, and classification
is exclusively server-side; the client never picks the anchor kind.

Reverse selections are legal; the server normalizes direction by transcript
position before validating.

## Coordinate mapping (child-local to parent)

Rendered child text is the child's tokens joined and outer-stripped
(`splits.py`, `derive_children`). Parent coordinates are code-point offsets
into the parent's effective text. The mapping from a child-local offset to a
parent offset must account for two deltas:

- the parent outer-trim delta: joined tokens may differ from `raw_text` by
  outer whitespace only (`splittable_words` tolerates
  `joined.strip() == raw_text.strip()`), so token positions are located
  inside `raw_text` after skipping its leading whitespace;
- the child left-trim delta: each child's rendered text strips the leading
  whitespace of its own first token span.

One pure function pair in `src/voxint/adjudication/annotations.py` owns this
mapping and its inverse. An executable fixture table pins it: leading and
trailing spaces, space-prefixed tokens, embedded newlines, emoji, combining
marks, re-split, and un-split cases. Unsplit lines map one-to-one because the
rendered text equals the effective text.

## Classification (server-side truth table)

Given validated, direction-normalized endpoints:

1. If both offsets cover whole parent segments, the anchor is
   `segment_range`. `segment_range` always means whole immutable parents; a
   whole split-child selection is not a segment_range.
2. Otherwise, if every covered segment is word-eligible and both offsets land
   exactly on token boundaries, the anchor is `word_range`. Word-eligibility
   is `splittable_words(segment) is not None` (tokens validate and
   reconcatenate to raw, no fired correction-trace entry, `enhanced_text`
   null or raw-equal) and additionally no operator review correction
   (`SegmentReviewState.corrected_text` is null); the review-correction check
   is route-owned for splits and must be applied here explicitly. A whole
   split-child selection classifies as word_range.
3. Otherwise the anchor is `text_range`, with offsets stored against the
   parent effective text at capture.

Segments with `words IS NULL` can never produce `word_range`. A partial
selection on such a segment stores `segment_range` with the verbatim quote;
the interface presents its display and seek as whole-segment, and never
implies sub-segment precision.

`text_range` offsets live in immutable parent coordinates. Split children are
render projections; re-splitting or un-splitting a parent changes projections
only and never rewrites stored anchors or hashes.

## Source text hash

`source_text_hash` is the full sha256 hex digest of a versioned,
length-framed serialization of the covered segments in `segment_index` order:

    "annv1" + for each segment: "{segment_id}:{code_point_length}:{text}"

encoded UTF-8, where `text` is the segment's effective text (corrected, else
enhanced, else raw) at capture and `code_point_length` is `len(text)` in code
points. Length framing plus segment ids make the serialization injective; a
segment containing a delimiter byte cannot collide with a different segment
partition. There is no revision-kind salt: byte-identical visible text stays
non-stale even if its source tier changed. A golden-hex unit test pins this
definition; changing it would mark every existing annotation stale at once,
so any change requires bumping `anchor_schema_version`.

The hash is stored for every anchor kind. At read time the server recomputes
the hash over current effective texts; a mismatch marks the annotation stale.

## Timing honesty

`start_seconds` and `end_seconds` are nullable. Only `word_range` stores them,
derived from word timings. `text_range` and `segment_range` store null; the
read path derives coarse segment-interval bounds for display and seek and
labels every API and export shape with `timing_precision`: `"word"` or
`"segment"`. Downstream consumers (the future clip extraction in #88) must
treat `"segment"` bounds as jump targets, never as clip-accurate edges.

## Staleness, refresh, re-anchor

At read time each annotation reports `stale` (hash mismatch) and its speaker
attribution is resolved from the current adjudication state, not from
anything captured.

Stale annotations render without inline highlight marks. Painting old offsets
onto changed text would highlight words the operator never selected, so the
transcript shows only a distinctly styled locator at the annotation's start
line, labeled as approximate, and the panel shows the original captured quote
with a "text changed" badge.

Refresh (an explicit operator action) re-derives quote, hash, and seconds
only when the anchor still identifies the same span deterministically:

- `segment_range`: always refreshable (whole segments are stable identity).
- `word_range`: refreshable only while every covered segment still grants
  word-eligibility; otherwise refused.
- `text_range`: a matching hash makes refresh a no-op; a stale hash means the
  offsets are dead and refresh is refused.

A refused refresh returns a 409 stale-anchor conflict. The recovery path is
re-anchor: a PATCH carrying a complete fresh capture payload, validated
exactly like a create, which atomically replaces the anchor and its snapshot.
Refresh, re-anchor, and metadata edits are mutually exclusive PATCH
operations.

## Data model (migration 0031)

Three tables; annotations are mutable with soft delete.

- `annotation_tags` (global, flat): verbatim `name` plus a writer-computed
  `name_normalized` (trimmed, casefolded) carrying the UNIQUE constraint, a
  palette `color`, `created_at`, and nullable `archived_at`. Archived tags
  disappear from pickers but remain visible on existing annotations. There is
  no tag delete in v1.
- `transcript_annotations`: endpoint parent-segment FKs (`ON DELETE
  CASCADE`), captured `start_segment_index`/`end_segment_index` copies for
  ordering, the anchor fields gated by per-kind CHECK constraints (word pair
  present only for word_range, char pair only for text_range, neither for
  segment_range; hash always present, lowercase hex, 64 chars), nullable
  paired seconds, `quote_text`, `color_index`, `note`, `operator`,
  `idempotency_key` (UNIQUE) with a `request_fingerprint` of the canonical
  create payload, timestamps, and `deleted_at`. `pipeline_run_id` is
  denormalized by the sole writer from the endpoint segments and cascade
  deletes with the run.
- `annotation_tag_links`: composite-key many-to-many, capped per annotation.

Caps (server-enforced): 6 highlight colors, 100 segments per span, 50,000
quote code points, 4,000 note code points, 8 tags per annotation.

## API surface and error taxonomy

Reads (list, exports, stored quotes) require operator auth and onboarding
only, with no claim, so Copy works in a read-only review tab. The read routes
are `GET /review/{run_id}/annotations` (the panel list), `GET
/review/{run_id}/annotations/{annotation_id}/export.md` (one pull-quote), and
`GET /review/{run_id}/annotations/export.md` (all filtered pull-quotes, honoring
the same `?tag=` OR-union). Run-scoped writes require the active review claim
token; a lost claim is a 409 marked `X-Voxint-Conflict: claim`. Creates carry a
client nonce;
replaying the same nonce with the same fingerprint returns the original row
(including a soft-deleted one; replay never resurrects or duplicates), and
the same nonce with a different payload is a 409 idempotency conflict.
Metadata edits and deletes are naturally idempotent and carry no nonce. Tag
writes have no run context and are CSRF-gated like run notes. The live
pull-quote request (`POST /review/{run_id}/annotations/export/live.md`) persists
nothing and therefore carries neither claim, nonce, nor CSRF; it classifies and
validates an unsaved selection with the same caps and code path as create (422 on
a bad anchor or cap, 409 stale on a drifted client quote) and returns the
markdown. Its optional `note`/`tags` decorate the returned trailer only.

- 422: malformed anchors, out-of-bounds offsets, unknown child ranges,
  violated caps, bad colors.
- 404: unknown run, segment, annotation, or tag, including any cross-run or
  forged id. Fail closed; do not distinguish forged from missing.
- 409: claim loss (`X-Voxint-Conflict: claim`), archived run, stale quote or
  stale anchor (`X-Voxint-Conflict: stale`), duplicate tag name, idempotency
  payload mismatch (`X-Voxint-Conflict: idempotency`).

Repeated `?tag=` filters are a union (OR), identically in the panel and in
exports.

## Console surface (Landing 1)

The review console (the `review-stepper` island) is the sole create path.
Selecting transcript text opens a toolbar with six color swatches, a tag picker
plus an inline new-tag field, and a note field; the `h` shortcut opens it for
the current selection, and an empty selection reports that rather than doing
nothing. Saving POSTs the capture to `/review/{run_id}/annotations`. The server
owns the anchor, so a client quote that no longer matches comes back as a 409 the
operator retries; the code-unit to code-point conversion happens once, in
`frontend/src/lib/selection.ts`, before anything reaches the wire.

Highlights paint as `<mark class="hl-N">` pieces byte-identical to the line text,
so the marks never alter a character and the DOM selection offsets stay aligned.
The Highlights panel lists every annotation in transcript order with an OR-union
tag filter and per-row Jump, Edit, Delete, and, when a highlight is stale,
Refresh and Re-anchor. Re-anchor reads the current selection as the new anchor.
Speaker attribution and timing come from the read-time resolution, never the
captured copy, and timing is labeled approximate (a leading `≈`) when
`timing_precision` is `segment` rather than `word`.

A stale annotation drops its inline marks and shows an approximate locator chip
at its old start line; the panel row still shows the original quote verbatim.
Creating and editing highlights are island-only. With JavaScript off,
`review_transcript.html` renders a read-only Highlights list from the same props
the island hydrates from, so the fallback and the live panel cannot disagree;
hydration replaces the fallback wholesale.

Landing 2 adds Copy to the panel: a per-row Copy button and a Copy-all action that
honors the active tag filter. Both fetch the server-rendered markdown (the client
never reassembles a quote) and write it to the clipboard, falling back to a
selectable read-only field when the browser clipboard is unavailable (a plain-http
LAN context). Copy is a read, so it stays available without the claim; a stale
highlight's Copy is disabled until the operator refreshes or re-anchors it.

The highlight palette size (`HIGHLIGHT_PALETTE_SIZE`) is pinned across the
backend caps, the `--hl-N` design tokens, and the `mark.hl-N` rules by
`tests/contracts/test_highlight_palette_parity.py`.

## Pull-quote formatting

Stored and live pull-quotes (issue #86 Landing 2) build clipped `TranscriptLine`
values (line text sliced to the annotation span, speaker preserved) and pass them
through the existing `to_markdown` in `src/voxint/export/__init__.py`, so the quote
bytes match the file export by construction. The projection lives beside the
resolver (`clip_lines_for_export`); the markdown wrapper (`annotation_pull_quote`)
lives in the pure export module and never imports the resolver.

The body renders as the reading copy (`to_markdown(..., timestamps=False)`): timing
lives once, in the citation, so the body never shows a whole-segment bracket that
would misstate a sub-segment highlight's span. A thin trailer follows the body:

- `**Source:**` the escaped source title (sidecar title, else acquisition-metadata
  title, else a cleaned filename) then a `format_timespan` range. The range carries
  a leading `≈` marker when `timing_precision` is `segment` rather than `word`,
  mirroring the on-screen panel.
- `**Tags:**` the annotation's tag names in the panel's alphabetical order, when it
  has any.
- `**Note:**` the note, line breaks folded to spaces, when it has one.

Title, tag names, and the note are untrusted, so each is inline-escaped with the
same `_md_escape` the transcript body uses; folding the note to a single line means
no physical line can sit at a block-leading position, so a leading `#` stays literal
prose. The body and every trailer field are separated by blank lines, and the whole
quote ends in exactly one newline. A single highlight (`content_start` at `world` in
a `word_range`) renders:

```markdown
## Alice

> world

**Source:** Interview [00:00:01.000–00:00:02.000]

**Tags:** Key Point

**Note:** worth a look
```

A bulk export concatenates each highlight's block in the canonical transcript order
(rendered line index, then code-point offset, then annotation id, so a split-child
highlight orders by the line the operator sees, not by segment index) joined by the
thematic-break separator `\n---\n\n`. An empty filtered result is an empty body.

A stale highlight has no live spans, and the stored `quote_text` alone cannot
reconstruct the live speaker attribution and per-line geometry a faithful quote
needs. Exporting one is refused (409 `X-Voxint-Conflict: stale`) rather than
fabricated; a bulk export fails atomically if any matched highlight is stale. The
operator refreshes or re-anchors it first.

Annotation exports use their own one-entry media-type table (`ANNOTATION_MEDIA_TYPES`,
`text/markdown`); the transcript `TranscriptFormat`/`MEDIA_TYPES` tables are not
extended.
