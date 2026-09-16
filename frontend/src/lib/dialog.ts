/**
 * Shared modal dialog hook (#162, extracted from KeymapHelp).
 *
 * Manages focus trap, scroll lock, Escape dismiss, focus restore, and
 * optional `inert` on a background element.
 */

import { useEffect, useRef } from "react";

export interface UseModalDialogOptions {
  /** Whether the dialog is currently open. */
  open: boolean;
  /** Called when the dialog should close (Escape key). */
  onClose: () => void;
  /** Ref to the dialog panel element (for focus trap). */
  panelRef: React.RefObject<HTMLElement | null>;
  /** Ref to the element that should receive initial focus on open. */
  initialFocusRef: React.RefObject<HTMLElement | null>;
  /**
   * Optional CSS selector for the element to mark `inert` while the dialog
   * is open (e.g. ".app-shell"). Keeps Tab and click from reaching the page.
   */
  inertSelector?: string;
}

/**
 * Hook that manages modal dialog lifecycle: focus restore, scroll lock,
 * Tab trap, Escape dismiss, and optional inert background.
 */
export function useModalDialog({
  open,
  onClose,
  panelRef,
  initialFocusRef,
  inertSelector,
}: UseModalDialogOptions): void {
  const restoreRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!open) return;

    // Capture the element to restore focus to on close.
    restoreRef.current = document.activeElement as HTMLElement | null;
    initialFocusRef.current?.focus();

    // Lock background scroll.
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    // Mark the background inert if requested.
    const inertEl = inertSelector
      ? document.querySelector<HTMLElement>(inertSelector)
      : null;
    if (inertEl) inertEl.inert = true;

    return () => {
      document.body.style.overflow = prevOverflow;
      if (inertEl) inertEl.inert = false;
      restoreRef.current?.focus();
    };
  }, [open, onClose, initialFocusRef, inertSelector]);

  // Tab trap: keep Tab cycling inside the dialog panel.
  useEffect(() => {
    if (!open) return;

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
        return;
      }
      if (event.key !== "Tab") return;

      const panel = panelRef.current;
      if (!panel) return;
      const focusable = panel.querySelectorAll<HTMLElement>(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
      );
      if (focusable.length === 0) return;

      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      const active = document.activeElement;

      // If focus has escaped the panel, pull it back.
      if (!active || !panel.contains(active)) {
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

    window.addEventListener("keydown", onKeyDown, true);
    return () => window.removeEventListener("keydown", onKeyDown, true);
  }, [open, onClose, panelRef]);
}
