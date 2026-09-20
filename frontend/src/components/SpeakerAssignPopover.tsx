import { useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { SpeakerCombobox, type Speaker } from "./SpeakerCombobox";

export interface SpeakerAssignPopoverProps {
  anchorRect: DOMRect;
  currentSpeaker: string;
  currentSpeakerId: string | null;
  currentLabel: string | null;
  resolved: boolean;
  labelSegmentCount: number;
  speakers: readonly Speaker[];
  onAssign: (speakerId: string, scope: "segment" | "label") => void;
  onCreate: (name: string, scope: "segment" | "label") => Promise<boolean>;
  onReset: (scope: "segment" | "label") => void;
  onRename: (newName: string) => void;
  onClose: () => void;
  disabled?: boolean;
}

const styles = `
.sp-popover {
  position: fixed; z-index: 1000; box-sizing: border-box;
  width: 20rem; max-width: calc(100vw - 16px); max-height: calc(100dvh - 16px);
  overflow: auto; padding: .5rem; border: 1px solid var(--line);
  border-radius: var(--r-sm, 6px); background: var(--surface);
  color: var(--ink); box-shadow: var(--shadow-1, 0 4px 16px #0003);
  font: inherit; font-size: var(--t-sm, .875rem);
}
.sp-popover .speaker-combobox { display: block; }
.sp-popover .speaker-combobox-input,
.sp-popover .speaker-combobox-trigger { box-sizing: border-box; width: 100%; }
.sp-popover .speaker-combobox-list {
  position: static; max-height: 10rem; box-shadow: none; border: 0;
}
.sp-actions, .sp-scope {
  margin: .5rem 0 0; padding: .5rem 0 0; border: 0;
  border-top: 1px solid var(--line); min-width: 0;
}
.sp-action, .sp-save, .sp-cancel {
  border: 0; border-radius: var(--r-sm, 4px); background: transparent;
  color: inherit; font: inherit; padding: .375rem .5rem; cursor: pointer;
}
.sp-action { display: block; width: 100%; text-align: left; overflow-wrap: anywhere; }
.sp-action:hover, .sp-save:hover, .sp-cancel:hover { background: var(--surface-2); }
.sp-popover :focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.sp-popover :disabled { opacity: .5; cursor: not-allowed; }
.sp-scope label { display: flex; align-items: baseline; gap: .375rem; padding: .25rem; }
.sp-scope input { accent-color: var(--accent); }
.sp-rename { display: grid; gap: .375rem; }
.sp-name {
  box-sizing: border-box; width: 100%; min-width: 0; padding: .375rem .5rem;
  background: var(--surface); color: var(--ink); font: inherit;
  border: 1px solid var(--line); border-radius: var(--r-sm, 4px);
}
.sp-rename-buttons { display: flex; justify-content: flex-end; gap: .25rem; }
`;

export function SpeakerAssignPopover({
  anchorRect,
  currentSpeaker,
  currentSpeakerId,
  currentLabel,
  resolved,
  labelSegmentCount,
  speakers,
  onAssign,
  onCreate,
  onReset,
  onRename,
  onClose,
  disabled = false,
}: SpeakerAssignPopoverProps) {
  const uid = useId();
  const panelRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLElement | null>(null);
  const renameRef = useRef<HTMLInputElement>(null);
  const renameButtonRef = useRef<HTMLButtonElement>(null);
  const [scope, setScope] = useState<"segment" | "label">(
    resolved ? "segment" : "label",
  );
  const [renaming, setRenaming] = useState(false);
  const [name, setName] = useState(currentSpeaker);
  const [creating, setCreating] = useState(false);
  const busy = disabled || creating;
  const mountedRef = useRef(false);

  useLayoutEffect(() => {
    mountedRef.current = true;
    if (!triggerRef.current && document.activeElement instanceof HTMLElement) {
      triggerRef.current = document.activeElement;
    }
    const panel = panelRef.current;
    panel?.focus({ preventScroll: true });
    return () => {
      mountedRef.current = false;
      const active = document.activeElement;
      if (panel?.contains(active) || active === document.body) {
        triggerRef.current?.focus({ preventScroll: true });
      }
    };
    // Capture the opener once, before the combobox focuses its input.
  }, []);

  useLayoutEffect(() => {
    const panel = panelRef.current;
    if (!panel) return;
    const position = () => {
      const { width, height } = panel.getBoundingClientRect();
      const margin = 8;
      const below = anchorRect.bottom + 4;
      const above = anchorRect.top - height - 4;
      const top = below + height > window.innerHeight - margin ? above : below;
      panel.style.left = `${Math.max(margin, Math.min(anchorRect.left, window.innerWidth - width - margin))}px`;
      panel.style.top = `${Math.max(margin, Math.min(top, window.innerHeight - height - margin))}px`;
    };
    position();
    const observer = new ResizeObserver(position);
    observer.observe(panel);
    window.addEventListener("resize", position);
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", position);
    };
  }, [anchorRect]);

  useEffect(() => {
    const panel = panelRef.current;
    if (!panel) return;
    const onFocusOut = () => {
      requestAnimationFrame(() => {
        if (!mountedRef.current) return;
        if (!panelRef.current?.contains(document.activeElement)) onClose();
      });
    };
    panel.addEventListener("focusout", onFocusOut);
    return () => panel.removeEventListener("focusout", onFocusOut);
  }, [onClose]);

  useEffect(() => {
    const onClick = (event: MouseEvent) => {
      if (
        event.target instanceof Node &&
        !panelRef.current?.contains(event.target)
      ) {
        onClose();
      }
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !event.isComposing) {
        event.preventDefault();
        onClose();
      }
    };
    const onScroll = () => onClose();
    window.addEventListener("scroll", onScroll, { capture: true, passive: true });
    document.addEventListener("click", onClick);
    document.addEventListener("keydown", onKeyDown, true);
    return () => {
      window.removeEventListener("scroll", onScroll, { capture: true });
      document.removeEventListener("click", onClick);
      document.removeEventListener("keydown", onKeyDown, true);
    };
  }, [onClose]);

  useEffect(() => {
    if (renaming) {
      renameRef.current?.focus();
      renameRef.current?.select();
    }
  }, [renaming]);

  return createPortal(
    <div
      ref={panelRef}
      className="sp-popover"
      role="dialog"
      aria-label={`Assign speaker for ${currentSpeaker}`}
      aria-busy={creating}
      tabIndex={-1}
      data-speaker-label={currentLabel ?? undefined}
    >
      <style>{styles}</style>
      <SpeakerCombobox
        speakers={speakers}
        mode="controlled"
        value={currentSpeakerId ?? undefined}
        label="Choose speaker"
        placeholder="Search or create speaker…"
        autoFocus
        disabled={busy}
        onSelect={(speakerId) => {
          if (busy) return;
          onAssign(speakerId, scope);
          onClose();
        }}
        onCreate={scope === "label" ? async (speakerName) => {
          if (busy) return false;
          setCreating(true);
          try {
            const ok = await onCreate(speakerName, scope);
            if (ok && mountedRef.current) onClose();
            return ok;
          } finally {
            if (mountedRef.current) setCreating(false);
          }
        } : undefined}
      />
      <div className="sp-actions">
        {currentSpeakerId !== null &&
          (renaming ? (
            <form
              className="sp-rename"
              onSubmit={(event) => {
                event.preventDefault();
                if (busy || !name.trim()) return;
                onRename(name.trim());
                onClose();
              }}
            >
              <label htmlFor={`${uid}-name`}>Rename “{currentSpeaker}”</label>
              <input
                ref={renameRef}
                id={`${uid}-name`}
                className="sp-name"
                value={name}
                onChange={(event) => setName(event.target.value)}
                disabled={busy}
                autoComplete="off"
                maxLength={120}
              />
              <div className="sp-rename-buttons">
                <button
                  type="button"
                  className="sp-cancel"
                  disabled={busy}
                  onClick={() => {
                    setRenaming(false);
                    requestAnimationFrame(() =>
                      renameButtonRef.current?.focus(),
                    );
                  }}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="sp-save"
                  disabled={busy || !name.trim()}
                >
                  Save
                </button>
              </div>
            </form>
          ) : (
            <button
              ref={renameButtonRef}
              type="button"
              className="sp-action"
              disabled={busy}
              onClick={() => {
                setName(currentSpeaker);
                setRenaming(true);
              }}
            >
              ✎ Rename “{currentSpeaker}”…
            </button>
          ))}
        <button
          type="button"
          className="sp-action"
          disabled={busy}
          onClick={() => {
            if (busy) return;
            onReset(scope);
            onClose();
          }}
        >
          ↺ Reset to detected speaker
        </button>
      </div>
      <fieldset
        className="sp-scope"
        aria-label="Assignment scope"
        disabled={busy}
      >
        <label>
          <input
            type="radio"
            name={`${uid}-scope`}
            value="label"
            checked={scope === "label"}
            onChange={() => setScope("label")}
          />
          <span>All segments with this voice ({labelSegmentCount})</span>
        </label>
        <label>
          <input
            type="radio"
            name={`${uid}-scope`}
            value="segment"
            checked={scope === "segment"}
            onChange={() => setScope("segment")}
          />
          <span>Just this segment</span>
        </label>
      </fieldset>
    </div>,
    document.body,
  );
}
