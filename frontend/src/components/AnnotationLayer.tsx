import {
  useCallback,
  useMemo,
  useRef,
  useState,
  type ReactNode,
  type RefObject,
} from "react";

import { ApiError, apiFetch } from "../lib/api-client";
import { writeClipboard } from "../lib/clipboard";
import {
  annotationsExportUrl,
  filterByTags,
  sortAnnotations,
  spansByLine,
  staleLocatorLines,
  type AnnotationLimits,
  type AnnotationLineSpan,
  type AnnotationShape,
  type AnnotationTagShape,
} from "../lib/annotations";
import { makeNonce } from "../lib/nonce";
import {
  captureFormFields,
  selectionToCapture,
  type CapturePayload,
} from "../lib/selection";

// Every annotation write is form-encoded and asks for JSON explicitly (the routes
// would otherwise content-negotiate an HTML redirect for the JS-off fallback).
const FORM_HEADERS = {
  "content-type": "application/x-www-form-urlencoded",
  accept: "application/json",
};

export interface UseAnnotationsArgs {
  runId: string;
  // The live claim token (null on a read-only tab); writes are refused without it.
  reviewToken: string | null;
  // Whether this tab holds the claim AND has not lost it — gates the create toolbar
  // and every mutating panel action. Reads (the panel) render regardless.
  writable: boolean;
  // An ancestor of the transcript lines; selection.ts walks up to it. ReviewStepper
  // attaches this to its outer div.
  rootRef: RefObject<HTMLElement | null>;
  initialAnnotations: AnnotationShape[];
  initialTags: AnnotationTagShape[];
  limits: AnnotationLimits;
  tagCsrf: string | null;
  // Per-action CSRF token for extracting a highlight as an audio clip (issue #88);
  // null on a payload without it (then the clip action simply posts unauthenticated
  // and the server refuses with a 403, which surfaces as an inline error).
  clipCsrf: string | null;
  // Move the review cursor to (and play) a line — reused for panel "Jump".
  onJump: (lineIndex: number) => void;
  // A claim-loss 409 during an annotation write bubbles here so ReviewStepper stops
  // the whole loop, exactly as a verify/correct claim loss would.
  onClaimLost: () => void;
}

export interface UseAnnotationsResult {
  spansByLine: Map<number, AnnotationLineSpan[]>;
  staleLines: Set<number>;
  // Player onTextSelect (mouseup): adopt the current selection as a create draft.
  // Silent when nothing is selected. Returns the capture so the `h` path can react.
  captureSelection: () => CapturePayload | null;
  // The `h` shortcut: same capture, but announces when there is nothing selected.
  annotateFromKeyboard: () => void;
  // Re-fetch resolved annotations + tags after a transcript mutation (text edit /
  // split / relabel) so marks re-resolve against the new render instead of going
  // stale until a page reload.
  reload: () => Promise<void>;
  toolbar: ReactNode;
  panel: ReactNode;
}

// A stable identity for a selection, so a mouseup on an unchanged selection (e.g.
// clicking a swatch while the transcript text stays selected) does not reset the
// operator's in-progress draft.
function selectionKey(cap: CapturePayload): string {
  return [
    cap.start.segmentId,
    cap.start.offset,
    cap.start.childWordStart,
    cap.start.childWordEnd,
    cap.end.segmentId,
    cap.end.offset,
    cap.end.childWordStart,
    cap.end.childWordEnd,
  ].join(":");
}

export function useAnnotations({
  runId,
  reviewToken,
  writable,
  rootRef,
  initialAnnotations,
  initialTags,
  limits,
  tagCsrf,
  clipCsrf,
  onJump,
  onClaimLost,
}: UseAnnotationsArgs): UseAnnotationsResult {
  const [annotations, setAnnotations] =
    useState<AnnotationShape[]>(initialAnnotations);
  const [tags, setTags] = useState<AnnotationTagShape[]>(initialTags);
  // The selection being turned into a highlight (create mode); null when none.
  const [pending, setPending] = useState<CapturePayload | null>(null);
  // The annotation whose metadata is being edited (edit mode); null when none.
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draftColor, setDraftColor] = useState<number>(0);
  const [draftTags, setDraftTags] = useState<Set<string>>(() => new Set());
  const [draftNote, setDraftNote] = useState<string>("");
  const [newTagName, setNewTagName] = useState<string>("");
  const [newTagColor, setNewTagColor] = useState<number>(0);
  const [filterTags, setFilterTags] = useState<Set<string>>(() => new Set());
  const [busy, setBusy] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  // Copy/export (issue #86 Landing 2): a transient aria-live status, and the exact
  // fetched markdown to reveal in a selectable field when the clipboard is
  // unavailable (a plain-http LAN context) so the operator can copy it by hand.
  const [copyStatus, setCopyStatus] = useState<string | null>(null);
  const [copyFallback, setCopyFallback] = useState<string | null>(null);
  // The annotation whose clip is currently being extracted (issue #88), so only
  // that row's button shows a busy label; null when none is in flight.
  const [clippingId, setClippingId] = useState<string | null>(null);
  const clipBusyRef = useRef<boolean>(false);
  // Synchronous re-entry guard (a double word/button click races two writes before
  // `busy` state re-renders), mirroring ReviewStepper's busyRef.
  const busyRef = useRef<boolean>(false);

  const spans = useMemo(() => spansByLine(annotations), [annotations]);
  const staleLines = useMemo(
    () => staleLocatorLines(annotations),
    [annotations],
  );
  const ordered = useMemo(() => sortAnnotations(annotations), [annotations]);
  const visibleTags = useMemo(() => tags.filter((t) => !t.archived), [tags]);

  const upsert = useCallback((shape: AnnotationShape) => {
    setAnnotations((cur) =>
      cur.some((a) => a.id === shape.id)
        ? cur.map((a) => (a.id === shape.id ? shape : a))
        : [...cur, shape],
    );
  }, []);

  const resetDraft = useCallback(() => {
    setDraftColor(0);
    setDraftTags(new Set());
    setDraftNote("");
  }, []);

  const closeToolbar = useCallback(() => {
    setPending(null);
    setEditingId(null);
    setError(null);
    resetDraft();
    // Drop the live selection so a later stray click does not re-open the toolbar
    // over the same (now annotated) span.
    window.getSelection()?.removeAllRanges();
  }, [resetDraft]);

  // Adopt the current window selection as a fresh create draft — but only when it
  // is genuinely NEW. An unchanged selection (the mouseup from clicking a swatch or
  // Save, which leaves the transcript text selected) keeps the operator's draft; a
  // collapsed/off-transcript selection (a null capture) is ignored entirely.
  const captureSelection = useCallback((): CapturePayload | null => {
    const root = rootRef.current;
    if (!root) return null;
    const cap = selectionToCapture(root);
    if (!cap) return null;
    // An open edit owns the toolbar; a stray transcript selection must not silently
    // discard the operator's in-progress note/tags/colour — they Cancel to leave.
    if (editingId !== null) return null;
    // An unchanged selection (the mouseup from clicking a swatch or Save, which
    // leaves the transcript text selected) keeps the operator's current draft.
    if (pending && selectionKey(pending) === selectionKey(cap)) return pending;
    // A genuinely new selection starts a fresh create draft. State setters run in
    // sequence — never inside a setState updater, which React may replay.
    resetDraft();
    setError(null);
    setPending(cap);
    return cap;
  }, [rootRef, editingId, pending, resetDraft]);

  const annotateFromKeyboard = useCallback(() => {
    const cap = captureSelection();
    if (!cap) {
      setError(
        "Select some transcript text first, then press h to highlight it.",
      );
    }
  }, [captureSelection]);

  // One form-encoded write. Returns whether it succeeded plus the parsed body (when
  // asked). A claim-loss 409 bubbles to onClaimLost (stop the loop); every other
  // failure — a stale/idempotency/tag 409, a validation 422, a network error —
  // surfaces inline via `error` and keeps the surface live.
  const request = useCallback(
    async (
      path: string,
      init: RequestInit,
      parse: boolean,
    ): Promise<{ ok: boolean; data?: unknown }> => {
      if (busyRef.current) return { ok: false };
      busyRef.current = true;
      setBusy(true);
      setError(null);
      try {
        const res = await apiFetch(path, init);
        const data = parse ? await res.json() : undefined;
        return { ok: true, data };
      } catch (err) {
        if (
          err instanceof ApiError &&
          err.status === 409 &&
          err.conflictKind === "claim"
        ) {
          onClaimLost();
        } else {
          setError(err instanceof ApiError ? err.detail : "Request failed.");
        }
        return { ok: false };
      } finally {
        busyRef.current = false;
        setBusy(false);
      }
    },
    [onClaimLost],
  );

  const saveDraft = useCallback(async () => {
    if (!writable || reviewToken === null) return;
    const body = new URLSearchParams();
    body.set("token", reviewToken);
    if (editingId === null) {
      if (!pending) return;
      body.set("nonce", makeNonce());
      for (const [k, v] of Object.entries(captureFormFields(pending))) {
        body.set(k, v);
      }
      body.set("color_index", String(draftColor));
      if (draftNote.trim() !== "") body.set("note", draftNote);
      for (const id of draftTags) body.append("tags", id);
      const { ok, data } = await request(
        `/review/${runId}/annotations`,
        { method: "POST", headers: FORM_HEADERS, body: body.toString() },
        true,
      );
      if (ok && data) {
        upsert(data as AnnotationShape);
        closeToolbar();
      }
    } else {
      // op=edit replaces the whole metadata set (last write wins), so we always
      // send colour + the current note + the current tag set.
      body.set("op", "edit");
      body.set("color_index", String(draftColor));
      if (draftNote.trim() !== "") body.set("note", draftNote);
      for (const id of draftTags) body.append("tags", id);
      const { ok, data } = await request(
        `/review/${runId}/annotations/${editingId}`,
        { method: "PATCH", headers: FORM_HEADERS, body: body.toString() },
        true,
      );
      if (ok && data) {
        upsert(data as AnnotationShape);
        closeToolbar();
      }
    }
  }, [
    writable,
    reviewToken,
    editingId,
    pending,
    draftColor,
    draftNote,
    draftTags,
    request,
    runId,
    upsert,
    closeToolbar,
  ]);

  const deleteAnnotation = useCallback(
    async (id: string) => {
      if (!writable || reviewToken === null) return;
      const body = new URLSearchParams({ token: reviewToken });
      const { ok } = await request(
        `/review/${runId}/annotations/${id}`,
        { method: "DELETE", headers: FORM_HEADERS, body: body.toString() },
        false,
      );
      if (ok) {
        setAnnotations((cur) => cur.filter((a) => a.id !== id));
        if (editingId === id) closeToolbar();
      }
    },
    [writable, reviewToken, request, runId, editingId, closeToolbar],
  );

  const refresh = useCallback(
    async (id: string) => {
      if (!writable || reviewToken === null) return;
      const body = new URLSearchParams({ token: reviewToken, op: "refresh" });
      const { ok, data } = await request(
        `/review/${runId}/annotations/${id}`,
        { method: "PATCH", headers: FORM_HEADERS, body: body.toString() },
        true,
      );
      if (ok && data) upsert(data as AnnotationShape);
    },
    [writable, reviewToken, request, runId, upsert],
  );

  const reanchor = useCallback(
    async (id: string) => {
      if (!writable || reviewToken === null) return;
      // Selecting the new location fired a mouseup, so the same selection is
      // already the live create draft (`pending`); reuse it and fall back to a
      // fresh read only if it is somehow gone.
      const root = rootRef.current;
      const cap = pending ?? (root ? selectionToCapture(root) : null);
      if (!cap) {
        setError(
          "Select the highlight's new location in the transcript, then click Re-anchor.",
        );
        return;
      }
      const body = new URLSearchParams({ token: reviewToken, op: "reanchor" });
      for (const [k, v] of Object.entries(captureFormFields(cap))) {
        body.set(k, v);
      }
      const { ok, data } = await request(
        `/review/${runId}/annotations/${id}`,
        { method: "PATCH", headers: FORM_HEADERS, body: body.toString() },
        true,
      );
      if (ok && data) {
        upsert(data as AnnotationShape);
        // Clear the create draft the re-anchor selection opened, so its toolbar
        // cannot then Save a duplicate annotation over the same span.
        closeToolbar();
      }
    },
    [writable, reviewToken, rootRef, pending, request, runId, upsert, closeToolbar],
  );

  // Re-fetch the run's resolved annotations + tags. Call after a transcript
  // mutation (text edit / split / relabel) that changes the rendered lines: the
  // server re-derives quote/hash/seconds/line-index and staleness against the NEW
  // text, so marks repaint correctly (or drop to a locator) instead of clinging to
  // pre-edit offsets until a full page reload. No claim token needed (the GET is
  // read-only). Best-effort — a failed refresh leaves the prior state, which the
  // next reload or a page load reconciles.
  const reload = useCallback(async () => {
    try {
      const res = await apiFetch(`/review/${runId}/annotations`, {
        method: "GET",
      });
      const data = (await res.json()) as {
        annotations: AnnotationShape[];
        tags: AnnotationTagShape[];
      };
      setAnnotations(data.annotations);
      setTags(data.tags);
    } catch {
      // Leave the existing state; the next reload or page load reconciles.
    }
  }, [runId]);

  const createTag = useCallback(async () => {
    if (!writable) return;
    const name = newTagName.trim();
    if (name === "") return;
    const body = new URLSearchParams({ name, color: String(newTagColor) });
    if (tagCsrf) body.set("csrf_token", tagCsrf);
    const { ok, data } = await request(
      `/annotations/tags`,
      { method: "POST", headers: FORM_HEADERS, body: body.toString() },
      true,
    );
    if (ok && data) {
      const tag = data as AnnotationTagShape;
      setTags((cur) =>
        cur.some((t) => t.id === tag.id) ? cur : [...cur, tag],
      );
      // Auto-select the just-created tag on the current draft (subject to the cap).
      setDraftTags((cur) => {
        if (cur.size >= limits.maxTagsPerAnnotation) return cur;
        const next = new Set(cur);
        next.add(tag.id);
        return next;
      });
      setNewTagName("");
    }
  }, [
    writable,
    newTagName,
    newTagColor,
    tagCsrf,
    request,
    limits.maxTagsPerAnnotation,
  ]);

  const beginEdit = useCallback((a: AnnotationShape) => {
    setPending(null);
    setEditingId(a.id);
    setDraftColor(a.colorIndex);
    setDraftTags(new Set(a.tags.map((t) => t.id)));
    setDraftNote(a.note ?? "");
    setError(null);
  }, []);

  const toggleDraftTag = useCallback(
    (id: string) => {
      setDraftTags((cur) => {
        const next = new Set(cur);
        if (next.has(id)) next.delete(id);
        else if (next.size < limits.maxTagsPerAnnotation) next.add(id);
        return next;
      });
    },
    [limits.maxTagsPerAnnotation],
  );

  const toggleFilterTag = useCallback((id: string) => {
    setFilterTags((cur) => {
      const next = new Set(cur);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const jumpTo = useCallback(
    (a: AnnotationShape) => {
      // A rendered-LINE index only: a span's line, else the stale locator's line.
      // Never `startSegmentIndex` — that is a SEGMENT index, which diverges from
      // the line index after any split above it (it would jump to the wrong line).
      const line = a.spans[0]?.lineIndex ?? a.locatorLineIndex;
      if (line == null) return;
      onJump(line);
    },
    [onJump],
  );

  // Copy one highlight as a Markdown pull-quote. The SERVER renders the quote (bytes
  // match a file export by construction); the client only fetches and writes it, never
  // reassembling markdown. A stale highlight is refused server-side (409 stale) and its
  // Copy is disabled client-side, so this path normally sees only live rows. No claim.
  const copyOne = useCallback(
    async (id: string) => {
      setCopyStatus(null);
      setCopyFallback(null);
      try {
        const res = await apiFetch(`/review/${runId}/annotations/${id}/export.md`, {
          method: "GET",
        });
        const md = await res.text();
        if (await writeClipboard(md)) {
          setCopyStatus("Highlight copied to the clipboard.");
        } else {
          setCopyFallback(md);
          setCopyStatus(
            "Couldn’t copy automatically. The highlight is shown below; select it and copy it manually.",
          );
        }
      } catch (e) {
        if (e instanceof ApiError && e.conflictKind === "stale") {
          setCopyStatus("This highlight is stale. Refresh or re-anchor it, then copy.");
        } else {
          setCopyStatus("Couldn’t copy the highlight. Please try again.");
        }
      }
    },
    [runId],
  );

  // Copy every highlight matching the current tag filter, in transcript order. Honors
  // the same OR-union `?tag=` filter as the panel. The bulk export fails atomically if
  // any matched highlight is stale (409 stale); Copy all is also disabled client-side
  // while the visible set contains one.
  const copyAll = useCallback(async () => {
    setCopyStatus(null);
    setCopyFallback(null);
    try {
      const res = await apiFetch(annotationsExportUrl(runId, filterTags), {
        method: "GET",
      });
      const md = await res.text();
      if (md.trim() === "") {
        setCopyStatus("No highlights to copy.");
        return;
      }
      if (await writeClipboard(md)) {
        setCopyStatus("All highlights copied to the clipboard.");
      } else {
        setCopyFallback(md);
        setCopyStatus(
          "Couldn’t copy automatically. The highlights are shown below; select them and copy them manually.",
        );
      }
    } catch (e) {
      if (e instanceof ApiError && e.conflictKind === "stale") {
        setCopyStatus(
          "Some highlights are stale. Refresh or re-anchor them, then copy all.",
        );
      } else {
        setCopyStatus("Couldn’t copy the highlights. Please try again.");
      }
    }
  }, [runId, filterTags]);

  // Extract one highlight as an attributed audio clip (issue #88). POSTs to the
  // clips route (CSRF-gated, no claim), then navigates a temporary same-origin
  // anchor to the returned download URL — the server's Content-Disposition drives
  // the save, so the bytes never touch JS (no Blob buffering). Generation is
  // idempotent and content-addressed server-side, so a repeat click re-downloads
  // the one canonical clip. The clientside visibility gate mirrors the server's
  // preconditions; the server rejection is authoritative and shown inline.
  const extractClip = useCallback(
    async (id: string) => {
      if (clipBusyRef.current) return; // sync re-entry guard against a double click
      clipBusyRef.current = true;
      setClippingId(id);
      setCopyStatus(null);
      setCopyFallback(null);
      try {
        const body = new URLSearchParams();
        if (clipCsrf) body.set("csrf_token", clipCsrf);
        const res = await apiFetch(`/review/${runId}/annotations/${id}/clips`, {
          method: "POST",
          headers: FORM_HEADERS,
          body: body.toString(),
        });
        const data = (await res.json()) as { downloadUrl: string };
        const link = document.createElement("a");
        link.href = data.downloadUrl;
        document.body.appendChild(link);
        link.click();
        link.remove();
        setCopyStatus("Clip ready. Your download should start shortly.");
      } catch (e) {
        if (e instanceof ApiError && e.conflictKind === "stale") {
          setCopyStatus(
            "This highlight is stale. Refresh or re-anchor it, then extract the clip.",
          );
        } else if (e instanceof ApiError && e.status === 422) {
          setCopyStatus("This highlight’s timed span can’t be clipped.");
        } else if (e instanceof ApiError && e.status === 409) {
          setCopyStatus(
            "The processed audio for this run isn’t available, so a clip can’t be extracted.",
          );
        } else {
          setCopyStatus("Couldn’t extract the clip. Please try again.");
        }
      } finally {
        clipBusyRef.current = false;
        setClippingId(null);
      }
    },
    [runId, clipCsrf],
  );

  const toolbarOpen = writable && (pending !== null || editingId !== null);
  const toolbar: ReactNode = toolbarOpen ? (
    <AnnotationToolbar
      mode={editingId !== null ? "edit" : "create"}
      quote={pending?.clientQuote ?? null}
      palette={limits.paletteSize}
      tags={visibleTags}
      draftColor={draftColor}
      draftTags={draftTags}
      draftNote={draftNote}
      maxNoteChars={limits.maxNoteChars}
      maxTagNameChars={limits.maxTagNameChars}
      newTagName={newTagName}
      newTagColor={newTagColor}
      busy={busy}
      error={error}
      onColor={setDraftColor}
      onToggleTag={toggleDraftTag}
      onNote={setDraftNote}
      onNewTagName={setNewTagName}
      onNewTagColor={setNewTagColor}
      onCreateTag={() => void createTag()}
      onSave={() => void saveDraft()}
      onCancel={closeToolbar}
    />
  ) : null;

  const panel: ReactNode = (
    <HighlightsPanel
      annotations={filterByTags(ordered, filterTags)}
      total={annotations.length}
      tags={visibleTags}
      filterTags={filterTags}
      writable={writable}
      busy={busy}
      // A create/edit error already shows in the toolbar; only surface panel-action
      // errors (delete/refresh/reanchor) here so the same message never doubles.
      error={toolbarOpen ? null : error}
      onToggleFilter={toggleFilterTag}
      onClearFilter={() => setFilterTags(new Set())}
      onJump={jumpTo}
      onEdit={beginEdit}
      onDelete={(id) => void deleteAnnotation(id)}
      onRefresh={(id) => void refresh(id)}
      onReanchor={(id) => void reanchor(id)}
      onCopy={(id) => void copyOne(id)}
      onCopyAll={() => void copyAll()}
      onExtractClip={(id) => void extractClip(id)}
      clippingId={clippingId}
      copyStatus={copyStatus}
      copyFallback={copyFallback}
    />
  );

  return {
    spansByLine: spans,
    staleLines,
    captureSelection,
    annotateFromKeyboard,
    reload,
    toolbar,
    panel,
  };
}

// A row of palette swatches shared by the colour picker and the new-tag colour.
function ColorSwatches({
  palette,
  value,
  onChange,
  label,
}: {
  palette: number;
  value: number;
  onChange: (i: number) => void;
  label: string;
}): ReactNode {
  return (
    <span className="hl-swatches" role="group" aria-label={label}>
      {Array.from({ length: palette }, (_, i) => (
        <button
          key={i}
          type="button"
          className={`hl-swatch hl-${i}${value === i ? " is-selected" : ""}`}
          aria-pressed={value === i}
          aria-label={`Color ${i + 1}`}
          onClick={() => onChange(i)}
        />
      ))}
    </span>
  );
}

interface AnnotationToolbarProps {
  mode: "create" | "edit";
  quote: string | null;
  palette: number;
  tags: AnnotationTagShape[];
  draftColor: number;
  draftTags: Set<string>;
  draftNote: string;
  maxNoteChars: number;
  maxTagNameChars: number;
  newTagName: string;
  newTagColor: number;
  busy: boolean;
  error: string | null;
  onColor: (i: number) => void;
  onToggleTag: (id: string) => void;
  onNote: (v: string) => void;
  onNewTagName: (v: string) => void;
  onNewTagColor: (i: number) => void;
  onCreateTag: () => void;
  onSave: () => void;
  onCancel: () => void;
}

// The selection toolbar (issue #86): colour + tags + optional note, for a new
// highlight or an existing one's metadata. No Copy — export is Landing 2.
function AnnotationToolbar({
  mode,
  quote,
  palette,
  tags,
  draftColor,
  draftTags,
  draftNote,
  maxNoteChars,
  maxTagNameChars,
  newTagName,
  newTagColor,
  busy,
  error,
  onColor,
  onToggleTag,
  onNote,
  onNewTagName,
  onNewTagColor,
  onCreateTag,
  onSave,
  onCancel,
}: AnnotationToolbarProps): ReactNode {
  const title = mode === "edit" ? "Edit highlight" : "New highlight";
  return (
    <div className="annotation-toolbar my-2" role="group" aria-label={title}>
      <p className="text-sm">
        <strong>{title}</strong>
      </p>
      {mode === "create" && quote != null && (
        <p className="annotation-quote muted text-sm">“{quote}”</p>
      )}
      <div className="my-1">
        <ColorSwatches
          palette={palette}
          value={draftColor}
          onChange={onColor}
          label="Highlight color"
        />
      </div>
      {tags.length > 0 && (
        <div className="annotation-tag-pick my-1">
          {tags.map((t) => (
            <button
              key={t.id}
              type="button"
              className={`tag-pill hl-${t.color}${draftTags.has(t.id) ? " is-selected" : ""}`}
              aria-pressed={draftTags.has(t.id)}
              onClick={() => onToggleTag(t.id)}
            >
              {t.name}
            </button>
          ))}
        </div>
      )}
      <div className="annotation-new-tag my-1 text-sm">
        <input
          type="text"
          value={newTagName}
          maxLength={maxTagNameChars}
          placeholder="New tag name"
          aria-label="New tag name"
          onChange={(e) => onNewTagName(e.target.value)}
        />
        <ColorSwatches
          palette={palette}
          value={newTagColor}
          onChange={onNewTagColor}
          label="New tag color"
        />
        <button
          type="button"
          disabled={busy || newTagName.trim() === ""}
          onClick={onCreateTag}
        >
          Add tag
        </button>
      </div>
      <textarea
        className="w-full text-sm my-1"
        rows={2}
        value={draftNote}
        maxLength={maxNoteChars}
        placeholder="Optional note"
        aria-label="Highlight note"
        onChange={(e) => onNote(e.target.value)}
      />
      <div className="flex items-center my-1">
        <button
          type="button"
          className="primary mr-2"
          disabled={busy}
          onClick={onSave}
        >
          {mode === "edit" ? "Save changes" : "Save highlight"}
        </button>
        <button type="button" onClick={onCancel} disabled={busy}>
          Cancel
        </button>
      </div>
      {error && (
        <p role="alert" className="text-sm">
          {error}
        </p>
      )}
    </div>
  );
}

interface HighlightsPanelProps {
  annotations: AnnotationShape[];
  total: number;
  tags: AnnotationTagShape[];
  filterTags: Set<string>;
  writable: boolean;
  busy: boolean;
  error: string | null;
  onToggleFilter: (id: string) => void;
  onClearFilter: () => void;
  onJump: (a: AnnotationShape) => void;
  onEdit: (a: AnnotationShape) => void;
  onDelete: (id: string) => void;
  onRefresh: (id: string) => void;
  onReanchor: (id: string) => void;
  // Copy/export (issue #86 Landing 2): a read, so it renders for any viewer.
  onCopy: (id: string) => void;
  onCopyAll: () => void;
  // Extract clip (issue #88): a derived-artifact action, so like Copy it renders
  // for any viewer (CSRF-gated, not claim-gated). Only word-timed highlights show
  // it. `clippingId` is the row whose extraction is in flight (busy label).
  onExtractClip: (id: string) => void;
  clippingId: string | null;
  copyStatus: string | null;
  copyFallback: string | null;
}

// The Highlights panel (issue #86): every annotation in transcript order, with an
// OR-union tag filter, honest staleness, and click-to-jump. Read for any viewer;
// the mutating actions render only when the tab holds the claim, but Jump and Copy
// (both reads) always render.
function HighlightsPanel({
  annotations,
  total,
  tags,
  filterTags,
  writable,
  busy,
  error,
  onToggleFilter,
  onClearFilter,
  onJump,
  onEdit,
  onDelete,
  onRefresh,
  onReanchor,
  onCopy,
  onCopyAll,
  onExtractClip,
  clippingId,
  copyStatus,
  copyFallback,
}: HighlightsPanelProps): ReactNode {
  // Copy all is refused (409) if any matched highlight is stale; disable it up front
  // while the visible set contains one, and when there is nothing to copy.
  const anyStale = annotations.some((a) => a.stale);
  return (
    <section className="highlights-panel my-2" aria-label="Highlights">
      <h2 className="text-sm">
        <strong>Highlights</strong> ({total})
      </h2>
      {annotations.length > 0 && (
        <div className="highlights-bulk my-1 text-sm">
          <button
            type="button"
            onClick={onCopyAll}
            disabled={anyStale}
            title={
              anyStale
                ? "Some highlights are stale. Refresh or re-anchor them first."
                : undefined
            }
          >
            Copy all{filterTags.size > 0 ? " (filtered)" : ""}
          </button>
        </div>
      )}
      {tags.length > 0 && (
        <div
          className="highlights-filter my-1 text-sm"
          role="group"
          aria-label="Filter highlights by tag"
        >
          <button
            type="button"
            className={`tag-pill${filterTags.size === 0 ? " is-selected" : ""}`}
            aria-pressed={filterTags.size === 0}
            onClick={onClearFilter}
          >
            All
          </button>
          {tags.map((t) => (
            <button
              key={t.id}
              type="button"
              className={`tag-pill hl-${t.color}${filterTags.has(t.id) ? " is-selected" : ""}`}
              aria-pressed={filterTags.has(t.id)}
              onClick={() => onToggleFilter(t.id)}
            >
              {t.name}
            </button>
          ))}
        </div>
      )}
      {error && (
        <p role="alert" className="text-sm">
          {error}
        </p>
      )}
      {annotations.length === 0 ? (
        <p className="muted text-sm">
          {total === 0
            ? "No highlights yet. Select transcript text to add one."
            : "No highlights match the current tag filter."}
        </p>
      ) : (
        <ul className="highlights-list">
          {annotations.map((a) => (
            <li
              key={a.id}
              className={`highlight-row${a.stale ? " is-stale" : ""}`}
            >
              <span
                className={`hl-swatch hl-${a.colorIndex}`}
                aria-hidden="true"
              />
              <span className="highlight-time muted tabular-nums">
                {a.startSeconds != null && a.endSeconds != null
                  ? `${a.timingPrecision !== "word" ? "≈ " : ""}${a.startSeconds.toFixed(2)}–${a.endSeconds.toFixed(2)}s`
                  : "timing unavailable"}
              </span>
              {a.speakers.length > 0 && (
                <span className="spk-badge ml-1">{a.speakers.join(", ")}</span>
              )}
              <blockquote className="highlight-quote">
                “{a.quote}”
              </blockquote>
              {a.tags.length > 0 && (
                <span className="highlight-tags">
                  {a.tags.map((t) => (
                    <span key={t.id} className={`tag-pill hl-${t.color}`}>
                      {t.name}
                    </span>
                  ))}
                </span>
              )}
              {a.note != null && a.note !== "" && (
                <p className="highlight-note text-sm">{a.note}</p>
              )}
              {a.stale && (
                <p className="highlight-stale text-sm" role="note">
                  This highlight is stale. The transcript text it covered has
                  changed.
                  {writable
                    ? " Refresh it if the wording is back, or select its new location and Re-anchor."
                    : ""}
                </p>
              )}
              <div className="highlight-actions text-sm">
                <button type="button" onClick={() => onJump(a)}>
                  Jump
                </button>
                <button
                  type="button"
                  onClick={() => onCopy(a.id)}
                  disabled={a.stale}
                  title={
                    a.stale
                      ? "This highlight is stale. Refresh or re-anchor it first."
                      : undefined
                  }
                >
                  Copy
                </button>
                {a.timingPrecision === "word" &&
                  a.startSeconds != null &&
                  a.endSeconds != null && (
                    <button
                      type="button"
                      onClick={() => onExtractClip(a.id)}
                      disabled={a.stale || clippingId != null}
                      title={
                        a.stale
                          ? "This highlight is stale. Refresh or re-anchor it first."
                          : "Extract this highlight as an audio clip"
                      }
                    >
                      {clippingId === a.id ? "Extracting…" : "Extract clip"}
                    </button>
                  )}
                {writable && (
                  <button
                    type="button"
                    disabled={busy}
                    onClick={() => onEdit(a)}
                  >
                    Edit
                  </button>
                )}
                {writable && a.stale && (
                  <button
                    type="button"
                    disabled={busy}
                    onClick={() => onRefresh(a.id)}
                  >
                    Refresh
                  </button>
                )}
                {writable && a.stale && (
                  <button
                    type="button"
                    disabled={busy}
                    onClick={() => onReanchor(a.id)}
                  >
                    Re-anchor
                  </button>
                )}
                {writable && (
                  <button
                    type="button"
                    disabled={busy}
                    onClick={() => onDelete(a.id)}
                  >
                    Delete
                  </button>
                )}
              </div>
            </li>
          ))}
        </ul>
      )}
      {copyStatus != null && (
        <p role="status" aria-live="polite" className="highlight-copy-status text-sm">
          {copyStatus}
        </p>
      )}
      {copyFallback != null && (
        <label className="highlight-copy-fallback text-sm">
          Copy this text manually:
          <textarea
            readOnly
            rows={6}
            value={copyFallback}
            onFocus={(e) => e.currentTarget.select()}
          />
        </label>
      )}
    </section>
  );
}
