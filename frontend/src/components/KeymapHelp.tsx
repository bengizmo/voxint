import { useEffect, useRef } from "react";

export interface KeymapHelpProps {
  open: boolean;
  onClose: () => void;
  // Whether the run has a speaker roster yet. The digit-assign shortcut only
  // fires when there are speakers to assign, so the cheat-sheet says so honestly
  // rather than promising a key that would no-op on a roster-less run.
  hasRoster: boolean;
}

// One shortcut row: a <kbd> and its plain-language description. Rendered as a
// fragment so <dt>/<dd> land as direct children of the enclosing <dl>.
function Row({ keys, desc }: { keys: string; desc: string }) {
  return (
    <>
      <dt style={{ margin: 0 }}>
        <kbd>{keys}</kbd>
      </dt>
      <dd style={{ margin: 0 }}>{desc}</dd>
    </>
  );
}

// The `?` cheat-sheet overlay (issue #51): a modal dialog listing every review
// shortcut, reachable by mouse (a "⌨ Shortcuts" button) as well as the `?` key —
// the keyboard is never the sole path to discovering the keyboard. Escape, the
// close button, and a backdrop click all dismiss it; focus moves into the dialog
// on open and is restored to the opener on close, and Tab is trapped inside, so
// a keyboard operator never loses their place. Theme-adaptive via the same CSS
// system colors base.html uses (Canvas/CanvasText), so it follows OS light/dark
// without a media query.
export function KeymapHelp({ open, onClose, hasRoster }: KeymapHelpProps) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const restoreRef = useRef<HTMLElement | null>(null);
  // True only when the press that may become a dismissing click STARTED on the
  // backdrop — so a text selection begun inside the panel and released outside
  // never closes the dialog.
  const backdropDownRef = useRef<boolean>(false);

  useEffect(() => {
    if (!open) return;
    restoreRef.current = document.activeElement as HTMLElement | null;
    closeRef.current?.focus();
    // Lock background scroll for the dialog's lifetime; restore the prior value
    // (not a hard-coded "") so we never clobber a style set elsewhere.
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prevOverflow;
      restoreRef.current?.focus();
    };
  }, [open]);

  if (!open) return null;

  const onKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === "Escape") {
      event.preventDefault();
      onClose();
      return;
    }
    if (event.key !== "Tab") return;
    // Keep Tab inside the dialog so focus can't wander to the page behind it.
    const focusable = dialogRef.current?.querySelectorAll<HTMLElement>(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
    );
    if (!focusable || focusable.length === 0) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    const active = document.activeElement;
    // If focus has slipped outside the dialog — e.g. onto the scrollable panel
    // itself, which is click-focusable in Firefox — pull it back to the first
    // control rather than letting Tab escape to the page behind.
    if (active == null || !dialogRef.current?.contains(active)) {
      event.preventDefault();
      first.focus();
      return;
    }
    if (event.shiftKey && active === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && active === last) {
      event.preventDefault();
      first.focus();
    }
  };

  return (
    <div
      // Backdrop: a click that both starts and ends on it dismisses. It is purely
      // presentational — the dialog role and labelling live on the panel below.
      onMouseDown={(e) => {
        backdropDownRef.current = e.target === e.currentTarget;
      }}
      onClick={(e) => {
        if (e.target === e.currentTarget && backdropDownRef.current) onClose();
        backdropDownRef.current = false;
      }}
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,.5)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 1000,
        padding: "1rem",
      }}
    >
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="keymap-help-title"
        tabIndex={-1}
        onKeyDown={onKeyDown}
        onClick={(e) => e.stopPropagation()}
        style={{
          background: "Canvas",
          color: "CanvasText",
          border: "1px solid GrayText",
          borderRadius: ".6rem",
          padding: "1.25rem 1.5rem",
          maxWidth: "28rem",
          width: "100%",
          maxHeight: "85vh",
          overflowY: "auto",
        }}
      >
        <div className="flex items-center justify-between">
          <h2
            id="keymap-help-title"
            className="text-base"
            style={{ margin: 0 }}
          >
            Keyboard shortcuts
          </h2>
          <button
            ref={closeRef}
            type="button"
            onClick={onClose}
            aria-label="Close keyboard shortcuts"
          >
            ✕
          </button>
        </div>
        <p className="muted text-sm">
          Shortcuts work while reviewing and never fire while you are typing in a
          text box or menu. Every shortcut also has a button on the page.
        </p>
        <dl
          style={{
            display: "grid",
            gridTemplateColumns: "auto 1fr",
            gap: ".35rem .75rem",
            margin: ".5rem 0 .75rem",
          }}
        >
          <Row keys="v" desc="Verify this segment and go to the next" />
          <Row keys="n" desc="Skip to the next unreviewed segment" />
          <Row keys="p" desc="Replay the current segment" />
          <Row keys="e" desc="Edit the current segment’s text" />
          <Row keys="j / k" desc="Go to and play the next / previous segment" />
          <Row
            keys="1 – 9"
            desc={
              hasRoster
                ? "Assign this segment to the 1st–9th speaker"
                : "Assign this segment to a speaker (no speakers on this run yet)"
            }
          />
          <Row keys="0" desc="Reset this segment to its detected speaker" />
          <Row keys="?" desc="Show this list of shortcuts" />
        </dl>
        <p className="muted text-sm" style={{ marginBottom: 0 }}>
          Space plays or pauses and the arrow keys scroll — these stay with the
          audio player.
        </p>
      </div>
    </div>
  );
}
