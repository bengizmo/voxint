// Selection capture for the operator annotation layer (issue #86).
//
// The browser's DOM exposes offsets in UTF-16 code UNITS; the annotation wire
// protocol is Unicode code POINTS everywhere (docs/annotations.md). This module
// is the ONE place that conversion happens, before anything touches the wire, so
// the server never sees a UTF-16 unit. The pure conversions are unit-tested
// (selection.test.ts); the DOM traversal in `selectionToCapture` is exercised in
// the browser E2E lane, since it needs a real Selection/Range.

// The wire capture shape (mirrors the POST /review/{id}/annotations form). The
// client builds `clientQuote` from its OWN props text, never `Range.toString()`
// (whose whitespace behaviour at block boundaries is not a stable protocol); the
// server derives the quote independently and a mismatch is a 409 stale.
export interface CaptureEndpoint {
  segmentId: string; // the immutable PARENT segment uuid
  offset: number; // code-point index into the rendered line's text
  childWordStart: number | null; // set iff the endpoint sits in a split child
  childWordEnd: number | null;
}

export interface CapturePayload {
  start: CaptureEndpoint;
  end: CaptureEndpoint;
  clientQuote: string;
}

// --------------------------------------------------------------------------- //
// Pure code-unit <-> code-point conversion (unit-tested; no DOM)
// --------------------------------------------------------------------------- //

/** The number of Unicode code points in `text[0 .. u16)`, where `u16` is a
 *  UTF-16 code-unit index (a DOM Range offset). A surrogate pair counts as one. */
export function utf16ToCodePoint(text: string, u16: number): number {
  const limit = Math.max(0, Math.min(u16, text.length));
  let cp = 0;
  let i = 0;
  while (i < limit) {
    const code = text.charCodeAt(i);
    if (code >= 0xd800 && code <= 0xdbff && i + 1 < text.length) {
      const next = text.charCodeAt(i + 1);
      if (next >= 0xdc00 && next <= 0xdfff) {
        i += 2;
        cp += 1;
        continue;
      }
    }
    i += 1;
    cp += 1;
  }
  return cp;
}

/** The UTF-16 code-unit index at code-point offset `cp` (the inverse of
 *  `utf16ToCodePoint`). Clamped to the string length. */
export function codePointToUtf16(text: string, cp: number): number {
  let i = 0;
  let count = 0;
  while (count < cp && i < text.length) {
    const code = text.charCodeAt(i);
    if (code >= 0xd800 && code <= 0xdbff && i + 1 < text.length) {
      const next = text.charCodeAt(i + 1);
      if (next >= 0xdc00 && next <= 0xdfff) {
        i += 2;
        count += 1;
        continue;
      }
    }
    i += 1;
    count += 1;
  }
  return i;
}

/** Slice `text` by code-point offsets `[start, end)` (astral-safe: `Array.from`
 *  splits on code points, so a surrogate pair is one element). Offsets are
 *  clamped and ordered so a reversed pair yields "". */
export function sliceByCodePoints(
  text: string,
  start: number,
  end: number,
): string {
  const points = Array.from(text);
  const lo = Math.max(0, Math.min(start, points.length));
  const hi = Math.max(lo, Math.min(end, points.length));
  return points.slice(lo, hi).join("");
}

// --------------------------------------------------------------------------- //
// DOM traversal (browser-lane tested; needs a real Selection/Range)
// --------------------------------------------------------------------------- //

interface LineHit {
  segmentId: string;
  segmentIndex: number;
  childWordStart: number | null;
  childWordEnd: number | null;
  offset: number; // code points into the line's rendered text
  lineText: string;
  wrapper: HTMLElement; // the [data-seg-text] element this boundary fell in
}

/** Resolve one Range boundary (a container node + a UTF-16 offset) to the line it
 *  falls in and its code-point offset within that line's text. Returns null if the
 *  boundary is not inside a `[data-seg-text]` wrapper (chrome contact). */
function resolveBoundary(
  root: HTMLElement,
  node: Node,
  u16Offset: number,
): LineHit | null {
  // Walk up to the [data-seg-text] wrapper that holds this line's text nodes.
  let el: Node | null = node;
  while (el && el !== root) {
    if (el instanceof HTMLElement && el.dataset.segText !== undefined) {
      break;
    }
    el = el.parentNode;
  }
  if (!(el instanceof HTMLElement) || el.dataset.segText === undefined) {
    return null;
  }
  const wrapper = el;
  const line = wrapper.closest<HTMLElement>("[data-seg-index]");
  const segLine = wrapper.closest<HTMLElement>("[data-seg-id]") ?? line;
  if (!line || !segLine) {
    return null;
  }
  const segmentId = segLine.dataset.segId;
  const segmentIndex = Number(line.dataset.segIndex);
  if (segmentId === undefined || Number.isNaN(segmentIndex)) {
    return null;
  }

  // The boundary's code-point offset within the wrapper's rendered text.
  let offset: number;
  if (node.nodeType === Node.TEXT_NODE) {
    // Text container: DOM `u16Offset` is a UTF-16 index into this text node. Sum
    // the code points of every preceding text node in the wrapper, plus the code
    // points before the boundary within its own node.
    const walker = document.createTreeWalker(wrapper, NodeFilter.SHOW_TEXT);
    let acc = 0;
    let landed = false;
    let current: Node | null = walker.nextNode();
    while (current) {
      const text = current.textContent ?? "";
      if (current === node) {
        acc += utf16ToCodePoint(text, u16Offset);
        landed = true;
        break;
      }
      acc += utf16ToCodePoint(text, text.length);
      current = walker.nextNode();
    }
    if (!landed) {
      return null;
    }
    offset = acc;
  } else {
    // Element container (the wrapper itself, or an inner <mark> that
    // renderAnnotatedText emits): `u16Offset` is a CHILD index, NOT a text offset.
    // Measure the rendered text from the wrapper start to the boundary with a
    // Range — correct for every child-index (start-of-line, mid-line at a mark
    // edge, or end-of-line). Astral-safe via Array.from; within one inline wrapper
    // Range.toString() equals the textContent slice, so no block-boundary
    // whitespace is involved (the reason the cross-line quote is built from props).
    const measure = document.createRange();
    measure.selectNodeContents(wrapper);
    try {
      measure.setEnd(node, u16Offset);
    } catch {
      return null;
    }
    offset = Array.from(measure.toString()).length;
  }

  const ws = line.dataset.wordStart;
  const we = line.dataset.wordEnd;
  return {
    segmentId,
    segmentIndex,
    childWordStart: ws !== undefined && ws !== "" ? Number(ws) : null,
    childWordEnd: we !== undefined && we !== "" ? Number(we) : null,
    offset,
    lineText: wrapper.textContent ?? "",
    wrapper,
  };
}

/** Read the current window selection and build a normalized capture payload, or
 *  null when the selection is empty, collapsed, or touches non-transcript chrome.
 *  A DOM Range always has its start boundary before its end boundary in document
 *  (= render) order regardless of drag direction, so `start`/`end` already read
 *  forward and `clientQuote` reads forward with them. */
export function selectionToCapture(root: HTMLElement): CapturePayload | null {
  const selection = window.getSelection();
  if (!selection || selection.rangeCount === 0 || selection.isCollapsed) {
    return null;
  }
  const range = selection.getRangeAt(0);
  const first = resolveBoundary(root, range.startContainer, range.startOffset);
  const second = resolveBoundary(root, range.endContainer, range.endOffset);
  if (!first || !second) {
    return null;
  }
  if (first.wrapper === second.wrapper && first.offset === second.offset) {
    return null; // zero-width
  }

  const clientQuote = buildClientQuote(root, first, second);
  return {
    start: {
      segmentId: first.segmentId,
      offset: first.offset,
      childWordStart: first.childWordStart,
      childWordEnd: first.childWordEnd,
    },
    end: {
      segmentId: second.segmentId,
      offset: second.offset,
      childWordStart: second.childWordStart,
      childWordEnd: second.childWordEnd,
    },
    clientQuote,
  };
}

/** The client's own quote assertion: the selected text sliced from the RENDERED
 *  line texts (never `Range.toString()`, whose whitespace behaviour across block
 *  boundaries is not a stable protocol). A single rendered line is one slice; a
 *  multi-line selection is the first line's tail, every whole line strictly
 *  between the endpoints, and the last line's head, newline-joined — the same
 *  shape the server derives (`_derive_quote`), so a byte match is the non-stale
 *  signal. A selection that crosses a split parent's derived children still
 *  degrades to a 409 the operator retries: the server joins the parent's
 *  contiguous text with a space where the DOM renders separate child lines. */
function buildClientQuote(
  root: HTMLElement,
  first: LineHit,
  second: LineHit,
): string {
  if (first.wrapper === second.wrapper) {
    return sliceByCodePoints(first.lineText, first.offset, second.offset);
  }
  const wrappers = Array.from(
    root.querySelectorAll<HTMLElement>("[data-seg-text]"),
  );
  const firstIndex = wrappers.indexOf(first.wrapper);
  const secondIndex = wrappers.indexOf(second.wrapper);
  const middles =
    firstIndex !== -1 && secondIndex > firstIndex + 1
      ? wrappers
          .slice(firstIndex + 1, secondIndex)
          .map((w) => w.textContent ?? "")
      : [];
  return buildQuoteParts(
    first.lineText,
    first.offset,
    middles,
    second.lineText,
    second.offset,
  );
}

/** Pure assembly of a multi-line quote from the endpoint line texts + code-point
 *  offsets and the whole texts of the lines strictly between them (unit-tested;
 *  no DOM). The first line contributes its tail (from `firstOffset`), each middle
 *  line its whole text, and the last line its head (to `secondOffset`); parts are
 *  newline-joined to mirror the server's `_derive_quote`. */
export function buildQuoteParts(
  firstText: string,
  firstOffset: number,
  middleTexts: string[],
  secondText: string,
  secondOffset: number,
): string {
  const head = sliceByCodePoints(
    firstText,
    firstOffset,
    Array.from(firstText).length,
  );
  const tail = sliceByCodePoints(secondText, 0, secondOffset);
  return [head, ...middleTexts, tail].join("\n");
}

/** Flatten a capture payload to the POST/PATCH form field names (the server's
 *  wire contract). Null child indices are omitted so FastAPI sees them as absent
 *  (an unsplit endpoint), not the string "null". */
export function captureFormFields(
  payload: CapturePayload,
): Record<string, string> {
  const fields: Record<string, string> = {
    start_segment_id: payload.start.segmentId,
    start_offset: String(payload.start.offset),
    end_segment_id: payload.end.segmentId,
    end_offset: String(payload.end.offset),
    client_quote: payload.clientQuote,
  };
  if (payload.start.childWordStart !== null) {
    fields.start_child_word_start = String(payload.start.childWordStart);
  }
  if (payload.start.childWordEnd !== null) {
    fields.start_child_word_end = String(payload.start.childWordEnd);
  }
  if (payload.end.childWordStart !== null) {
    fields.end_child_word_start = String(payload.end.childWordStart);
  }
  if (payload.end.childWordEnd !== null) {
    fields.end_child_word_end = String(payload.end.childWordEnd);
  }
  return fields;
}
