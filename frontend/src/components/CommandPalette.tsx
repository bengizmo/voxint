/**
 * Command palette island (#162): unified search + command palette.
 *
 * Three result groups: commands (navigation + per-page actions), entity
 * search (media, speakers, projects), and semantic transcript passages.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { apiFetch } from "../lib/api-client";
import { useModalDialog } from "../lib/dialog";
import {
  type EntityItem,
  type PaletteCommand,
  type PassageItem,
  type PassageState,
  type RowGroup,
  PASSAGE_STATE_COPY,
  buildRows,
  filterCommands,
  isPaletteChord,
  moveActive,
  rowAt,
  totalRows,
} from "../lib/palette";
import { getRecentPages } from "../lib/recent-pages";

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface CommandPaletteProps {
  destinations: PaletteCommand[];
  actions: PaletteCommand[];
  initialOpen?: boolean;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function CommandPalette({
  destinations,
  actions,
  initialOpen = false,
}: CommandPaletteProps) {
  const [open, setOpen] = useState(initialOpen);
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const [entities, setEntities] = useState<EntityItem[]>([]);
  const [passages, setPassages] = useState<PassageItem[]>([]);
  const [passagesState, setPassagesState] = useState<PassageState>("idle");

  const panelRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // All server-provided commands, destinations first, then per-page actions.
  const allCommands = [...destinations, ...actions];

  // Filter commands by query.
  const filtered = filterCommands(allCommands, query);

  // Show recent pages only when the query is empty.
  const recentPages = query.trim() === ""
    ? getRecentPages().filter((p) => p.href !== window.location.pathname)
    : undefined;

  // Build row groups.
  const groups: RowGroup[] = buildRows(filtered, entities, passages, recentPages);
  const count = totalRows(groups);

  // ---------------------------------------------------------------------------
  // Modal dialog lifecycle
  // ---------------------------------------------------------------------------

  // Shared close handler: reset all state.
  const closePalette = useCallback(() => {
    setOpen(false);
    setQuery("");
    setActiveIndex(0);
    setEntities([]);
    setPassages([]);
    setPassagesState("idle");
  }, []);

  useModalDialog({
    open,
    onClose: closePalette,
    panelRef,
    initialFocusRef: inputRef,
    inertSelector: ".app-shell",
  });

  // ---------------------------------------------------------------------------
  // Entity search (debounced fetch)
  // ---------------------------------------------------------------------------

  useEffect(() => {
    if (!open) return;
    const trimmed = query.trim();
    if (trimmed.length < 2) {
      setEntities([]);
      return;
    }

    setEntities([]);
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      apiFetch(`/palette/entities?q=${encodeURIComponent(trimmed)}`, {
        signal: controller.signal,
      })
        .then((res) => res.json())
        .then((data: { state: string; items: EntityItem[] }) => {
          if (data.state === "ok") {
            setEntities(data.items);
          } else {
            setEntities([]);
          }
        })
        .catch(() => {
          // AbortError or network failure -- silently degrade.
        });
    }, 150);

    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [open, query]);

  // ---------------------------------------------------------------------------
  // Passage search (debounced fetch, independent of entity fetch)
  // ---------------------------------------------------------------------------

  useEffect(() => {
    if (!open) return;
    const trimmed = query.trim();
    if (trimmed.length < 2) {
      setPassages([]);
      setPassagesState("idle");
      return;
    }

    setPassagesState("loading");
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      apiFetch(`/palette/passages?q=${encodeURIComponent(trimmed)}`, {
        signal: controller.signal,
      })
        .then((res) => res.json())
        .then(
          (data: { state: string; items: PassageItem[] }) => {
            setPassagesState(data.state as PassageState);
            if (data.state === "ok") {
              setPassages(data.items);
            } else {
              setPassages([]);
            }
          },
        )
        .catch((err: unknown) => {
          if (err instanceof DOMException && err.name === "AbortError") return;
          setPassagesState("idle");
        });
    }, 150);

    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [open, query]);

  // ---------------------------------------------------------------------------
  // Global Ctrl/Cmd+K toggle
  // ---------------------------------------------------------------------------

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (isPaletteChord(event)) {
        event.preventDefault();
        if (open) {
          closePalette();
        } else {
          setOpen(true);
        }
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open, closePalette]);

  // ---------------------------------------------------------------------------
  // Keyboard navigation inside the dialog
  // ---------------------------------------------------------------------------

  const onInputKeyDown = (event: React.KeyboardEvent) => {
    // Let the chord toggle the palette closed from inside the input.
    if (isPaletteChord(event.nativeEvent)) {
      event.preventDefault();
      closePalette();
      return;
    }

    // During IME composition, don't intercept navigation keys.
    if (event.nativeEvent.isComposing) return;

    // Stop propagation so editor shortcuts do not fire while the palette
    // input is focused (React 19 delegates at the portal container).
    event.stopPropagation();

    switch (event.key) {
      case "ArrowDown":
      case "ArrowUp":
      case "Home":
      case "End": {
        event.preventDefault();
        setActiveIndex((prev) => moveActive(prev, event.key, count));
        break;
      }
      case "Enter": {
        event.preventDefault();
        const row = rowAt(groups, activeIndex);
        if (row) {
          closePalette();
          window.location.assign(row.href);
        }
        break;
      }
      // Escape is handled by useModalDialog.
    }
  };

  // Reset active index when query or results change.
  useEffect(() => {
    setActiveIndex(0);
  }, [query, count]);

  // Scroll active option into view.
  useEffect(() => {
    if (!open) return;
    const activeRow = rowAt(groups, activeIndex);
    if (!activeRow) return;
    const el = document.getElementById(activeRow.id);
    el?.scrollIntoView({ block: "nearest" });
  }, [activeIndex, open, groups]);

  // ---------------------------------------------------------------------------
  // Backdrop click
  // ---------------------------------------------------------------------------

  const backdropDownRef = useRef(false);

  const onBackdropMouseDown = (e: React.MouseEvent) => {
    backdropDownRef.current = e.target === e.currentTarget;
  };

  const onBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget && backdropDownRef.current) {
      closePalette();
    }
    backdropDownRef.current = false;
  };

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  // The trigger pill (server-rendered fallback is replaced by React).
  const trigger = (
    <button
      type="button"
      className="cb-search"
      aria-haspopup="dialog"
      aria-expanded={open}
      aria-label="Search and commands"
      onClick={() => setOpen(true)}
    >
      <svg
        viewBox="0 0 16 16"
        width="14"
        height="14"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        aria-hidden="true"
        focusable="false"
      >
        <circle cx="6.5" cy="6.5" r="4" />
        <line x1="10" y1="10" x2="14" y2="14" strokeLinecap="round" />
      </svg>
      <span className="cb-search-label">Search</span>
      <kbd>Ctrl K</kbd>
    </button>
  );

  const dialog = open
    ? createPortal(
        <div
          className="palette-backdrop"
          onMouseDown={onBackdropMouseDown}
          onClick={onBackdropClick}
        >
          <div
            ref={panelRef}
            className="palette-panel"
            role="dialog"
            aria-modal="true"
            aria-label="Search and commands"
            onClick={(e) => e.stopPropagation()}
            onKeyDown={(e) => {
              // Shield all dialog descendants from page-level shortcuts.
              // Escape and Tab are handled by useModalDialog's capture listener.
              if (e.key !== "Escape" && e.key !== "Tab") {
                e.stopPropagation();
              }
            }}
          >
            {/* Search input */}
            <div className="palette-input-wrap">
              <svg
                viewBox="0 0 16 16"
                width="14"
                height="14"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                aria-hidden="true"
                focusable="false"
              >
                <circle cx="6.5" cy="6.5" r="4" />
                <line
                  x1="10"
                  y1="10"
                  x2="14"
                  y2="14"
                  strokeLinecap="round"
                />
              </svg>
              <input
                ref={inputRef}
                className="palette-input"
                type="text"
                placeholder="Search or jump to..."
                role="combobox"
                aria-autocomplete="list"
                aria-expanded={count > 0}
                aria-controls="palette-listbox"
                aria-activedescendant={
                  count > 0
                    ? rowAt(groups, activeIndex)?.id
                    : undefined
                }
                autoComplete="off"
                spellCheck={false}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={onInputKeyDown}
              />
            </div>

            {/* Results */}
            <div className="palette-results" role="listbox" id="palette-listbox">
              {groups.map((group) => (
                <div key={group.label} role="group" aria-label={group.label}>
                  <div className="palette-group-label">{group.label}</div>
                  {group.rows.map((row) => (
                    <div
                      key={row.id}
                      id={row.id}
                      className="palette-option"
                      role="option"
                      aria-selected={
                        row.id === rowAt(groups, activeIndex)?.id
                      }
                      tabIndex={-1}
                      onMouseEnter={() => {
                        // Find the flat index for this row.
                        let idx = 0;
                        for (const g of groups) {
                          for (const r of g.rows) {
                            if (r.id === row.id) {
                              setActiveIndex(idx);
                              return;
                            }
                            idx++;
                          }
                        }
                      }}
                      onClick={() => {
                        closePalette();
                        window.location.assign(row.href);
                      }}
                    >
                      <span className="palette-option-label">{row.label}</span>
                      {row.sublabel && (
                        <span className="palette-option-hint">
                          {row.sublabel}
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              ))}
              {count === 0 && query.trim() && passagesState !== "loading" && (
                <div className="palette-empty">No results</div>
              )}
              {/* Passage state notices */}
              {PASSAGE_STATE_COPY[passagesState] && (
                <div className="palette-status">
                  {PASSAGE_STATE_COPY[passagesState]}
                </div>
              )}
              {/* "See all in Explore" link when passages are showing */}
              {passages.length > 0 && query.trim() && (
                <div className="palette-status">
                  <a
                    href={`/explore?q=${encodeURIComponent(query.trim())}`}
                    style={{ color: "var(--accent)" }}
                  >
                    See all results in Explore
                  </a>
                </div>
              )}
            </div>
          </div>
        </div>,
        document.body,
      )
    : null;

  return (
    <>
      {trigger}
      {dialog}
    </>
  );
}
