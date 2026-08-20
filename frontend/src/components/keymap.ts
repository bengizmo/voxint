// Single source of truth for the review-console keyboard shortcuts (issues #51/#53).
//
// The drift this prevents is between the *handler* and everything that describes it:
// the `switch (event.key)` in ReviewStepper that actually dispatches each shortcut, the
// `?` cheat-sheet modal (KeymapHelp), and the inline `<kbd>` hints on the page's buttons
// all used to carry their own copies of the key literals. They are now built from the
// primitive constants below, so a rebinding flows to all three in lock step. Dispatch
// itself stays a plain, visible switch — only the key *literals* are centralized, not the
// behaviour (there is deliberately no data-driven action table to hide it behind).

// The unmodified keys the global keymap binds. Values are lower-case so the handler's
// `event.key.toLowerCase()` matches them directly (Caps Lock never disables a shortcut).
export const REVIEW_KEY = {
  verify: "v",
  skip: "n",
  replay: "p",
  edit: "e",
  next: "j",
  previous: "k",
  resetSpeaker: "0",
  help: "?",
} as const;

// Digit-assign: 1–9 assign the focused segment to the Nth roster speaker. The bounds live
// here so the handler's range check, the modal row, and the inline cue agree.
export const ASSIGN_DIGIT_MIN = 1;
export const ASSIGN_DIGIT_MAX = 9;

// Save-edit is the one shortcut that is a modifier *chord* (Ctrl/⌘+Enter), not an
// unmodified global key, and it fires only while the correction textarea is focused —
// exactly where the global keymap above is deliberately suppressed. It therefore lives
// here as its own primitive rather than in REVIEW_KEY: the textarea handler, the two
// on-screen hints ("Save edit", the unsaved-edit warning), and the cheat-sheet row all
// read this one label + matcher, so they can never drift the way they used to.
export const SAVE_EDIT_LABEL = "Ctrl/⌘+↵";
export const SAVE_EDIT_DESC = "Save the edit you’re typing (while the edit box is focused)";

// True for the key event that saves an in-progress edit from within the box. Accepts any
// event exposing the modifier flags and `key`, so React synthetic and native events match.
export function isSaveEditChord(event: {
  ctrlKey: boolean;
  metaKey: boolean;
  key: string;
}): boolean {
  return (event.ctrlKey || event.metaKey) && event.key === "Enter";
}

export interface ShortcutCtx {
  // Whether the run has a speaker roster yet. The digit-assign shortcut only fires when
  // there are speakers to assign, so the cheat-sheet says so honestly rather than
  // promising a key that would no-op on a roster-less run.
  hasRoster: boolean;
}

export interface ReviewShortcut {
  // Display label for the key(s), e.g. "v", "j / k", "1 – 9".
  keys: string;
  // Plain-language description; a function when it varies by context.
  desc: string | ((ctx: ShortcutCtx) => string);
}

// Ordered for the cheat-sheet modal. Key labels are built from the primitives above so the
// modal can never disagree with the handler that reads the same constants.
export const REVIEW_SHORTCUTS: readonly ReviewShortcut[] = [
  { keys: REVIEW_KEY.verify, desc: "Verify this segment and go to the next" },
  { keys: REVIEW_KEY.skip, desc: "Skip to the next unreviewed segment" },
  { keys: REVIEW_KEY.replay, desc: "Replay the current segment" },
  { keys: REVIEW_KEY.edit, desc: "Edit the current segment’s text" },
  {
    keys: `${REVIEW_KEY.next} / ${REVIEW_KEY.previous}`,
    desc: "Go to and play the next / previous segment",
  },
  {
    keys: `${ASSIGN_DIGIT_MIN} – ${ASSIGN_DIGIT_MAX}`,
    desc: ({ hasRoster }) =>
      hasRoster
        ? "Assign this segment to the 1st–9th speaker"
        : "Assign this segment to a speaker (no speakers on this run yet)",
  },
  { keys: REVIEW_KEY.resetSpeaker, desc: "Reset this segment to its detected speaker" },
  { keys: REVIEW_KEY.help, desc: "Show this list of shortcuts" },
];

// Resolve a shortcut's description, applying the context for the ones that vary by it.
export function shortcutDesc(shortcut: ReviewShortcut, ctx: ShortcutCtx): string {
  return typeof shortcut.desc === "function" ? shortcut.desc(ctx) : shortcut.desc;
}
