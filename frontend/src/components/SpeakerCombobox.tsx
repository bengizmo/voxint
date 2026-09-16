import {
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
} from "react";

import { moveActive, rankItems } from "../lib/combobox";

export type Speaker = { id: string; displayName: string };

type CreateAction = "create";
type InheritAction = "inherit";
type SpecialRow = CreateAction | InheritAction;

interface SpeakerComboboxProps {
  speakers: readonly Speaker[];
  mode: "command" | "controlled";
  value?: string;
  onSelect: (speakerId: string) => void;
  onCreate?: (name: string) => Promise<boolean>;
  onInherit?: () => void;
  placeholder?: string;
  label: string;
  disabled?: boolean;
  digitPrefixes?: boolean;
}

export function SpeakerCombobox({
  speakers,
  mode,
  value,
  onSelect,
  onCreate,
  onInherit,
  placeholder = "Choose speaker…",
  label,
  disabled = false,
  digitPrefixes = false,
}: SpeakerComboboxProps) {
  const uid = useId();
  const listboxId = `${uid}-listbox`;
  const optionId = (idx: number) => `${uid}-opt-${idx}`;

  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const [creating, setCreating] = useState(false);

  const wrapRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const returnFocusRef = useRef(false);

  const filtered = rankItems(speakers, query, (s) => s.displayName);

  const trimmed = query.trim();
  const exactMatch = trimmed
    ? speakers.some(
        (s) => s.displayName.toLowerCase() === trimmed.toLowerCase(),
      )
    : true;
  const showCreate = onCreate != null && trimmed.length > 0 && !exactMatch;

  const rows: Array<Speaker | SpecialRow> = [
    ...filtered,
    ...(onInherit ? (["inherit"] as const) : []),
    ...(showCreate ? (["create"] as const) : []),
  ];
  const count = rows.length;

  const selectedLabel =
    mode === "controlled" && value
      ? speakers.find((s) => s.id === value)?.displayName
      : undefined;

  // [H1] Restore focus to trigger on keyboard-initiated close.
  const close = useCallback(
    (restoreFocus = false) => {
      setOpen(false);
      setQuery("");
      setActiveIndex(0);
      if (restoreFocus) returnFocusRef.current = true;
    },
    [],
  );

  // [H1] Effect: when open becomes false and returnFocus is flagged, focus trigger.
  useEffect(() => {
    if (!open && returnFocusRef.current) {
      returnFocusRef.current = false;
      triggerRef.current?.focus();
    }
  }, [open]);

  // [M1] Map speaker id -> unfiltered roster index for stable digit prefixes.
  const speakerRosterIndex = useMemo(() => {
    const m = new Map<string, number>();
    speakers.forEach((s, i) => m.set(s.id, i));
    return m;
  }, [speakers]);

  // [H2] Close after commit in both modes. [H4/L5] Guard on disabled/creating.
  const commitRow = useCallback(
    async (row: Speaker | SpecialRow) => {
      if (disabled || creating) return;
      if (typeof row === "object") {
        onSelect(row.id);
        close(true);
        return;
      }
      if (row === "inherit" && onInherit) {
        onInherit();
        close(true);
        return;
      }
      if (row === "create" && onCreate) {
        setCreating(true);
        try {
          const ok = await onCreate(trimmed);
          if (ok) close(true);
        } finally {
          setCreating(false);
        }
      }
    },
    [onSelect, onInherit, onCreate, close, trimmed, disabled, creating],
  );

  useEffect(() => {
    setActiveIndex(0);
  }, [query, count]);

  useEffect(() => {
    if (!open) return;
    const id = `${uid}-opt-${activeIndex}`;
    const el = listRef.current?.querySelector(`#${CSS.escape(id)}`);
    el?.scrollIntoView({ block: "nearest" });
  }, [activeIndex, open, uid]);

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) {
        close();
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open, close]);

  const onKeyDown = (e: KeyboardEvent) => {
    if (e.nativeEvent.isComposing) return;
    e.stopPropagation();

    switch (e.key) {
      case "ArrowDown":
      case "ArrowUp":
        e.preventDefault();
        setActiveIndex((prev) => moveActive(prev, e.key, count));
        return;
      case "Home":
      case "End":
        e.preventDefault();
        setActiveIndex((prev) => moveActive(prev, e.key, count));
        return;
      case "Enter":
        e.preventDefault();
        if (activeIndex >= 0 && activeIndex < count) {
          void commitRow(rows[activeIndex]);
        }
        return;
      case "Escape":
        e.preventDefault();
        close(true);
        return;
      case "Tab":
        close();
        return;
    }
  };

  const handleTriggerClick = () => {
    if (disabled) return;
    setOpen(true);
    requestAnimationFrame(() => inputRef.current?.focus());
  };

  // [H3] Only stopPropagation for keys the trigger handles.
  const handleTriggerKeyDown = (e: KeyboardEvent) => {
    if (disabled) return;
    if (e.key === "ArrowDown" || e.key === "Enter" || e.key === " ") {
      e.stopPropagation();
      e.preventDefault();
      setOpen(true);
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  };

  // [M1] Digit prefixes use unfiltered roster index; hidden when filtering.
  const rowLabel = (row: Speaker | SpecialRow): string => {
    if (row === "inherit") return "↺ Reset to detected speaker";
    if (row === "create") return `Create "${trimmed}"`;
    if (digitPrefixes && !trimmed) {
      const rosterIdx = speakerRosterIndex.get(row.id);
      if (rosterIdx != null && rosterIdx < 9) {
        return `${rosterIdx + 1}. ${row.displayName}`;
      }
    }
    return row.displayName;
  };

  return (
    <div
      ref={wrapRef}
      className="speaker-combobox"
    >
      {!open && (
        <button
          ref={triggerRef}
          type="button"
          className="speaker-combobox-trigger text-sm"
          disabled={disabled}
          aria-haspopup="listbox"
          aria-expanded={false}
          aria-label={label}
          onClick={handleTriggerClick}
          onKeyDown={handleTriggerKeyDown}
        >
          {selectedLabel ?? placeholder}
        </button>
      )}
      {open && (
        <>
          <input
            ref={inputRef}
            className="speaker-combobox-input text-sm"
            type="text"
            role="combobox"
            aria-autocomplete="list"
            aria-haspopup="listbox"
            aria-expanded={true}
            aria-controls={listboxId}
            aria-activedescendant={
              activeIndex >= 0 && activeIndex < count
                ? optionId(activeIndex)
                : undefined
            }
            aria-label={label}
            autoComplete="off"
            spellCheck={false}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder={placeholder}
            disabled={creating || disabled}
          />
          <div
            ref={listRef}
            id={listboxId}
            role="listbox"
            aria-label={label}
            className="speaker-combobox-list"
          >
            {rows.map((row, i) => {
              const isSpeaker = typeof row === "object";
              const isActive = i === activeIndex;
              const isSelected =
                isSpeaker && mode === "controlled" && row.id === value;
              return (
                <div
                  key={isSpeaker ? row.id : row}
                  id={optionId(i)}
                  role="option"
                  aria-selected={isActive}
                  className={`speaker-combobox-option${isSelected ? " selected" : ""}${row === "create" ? " create-option" : ""}`}
                  tabIndex={-1}
                  onMouseEnter={() => setActiveIndex(i)}
                  onMouseDown={(e) => e.preventDefault()}
                  onClick={() => void commitRow(row)}
                >
                  {rowLabel(row)}
                </div>
              );
            })}
            {count === 0 && (
              <div className="speaker-combobox-empty" role="status">
                No matches
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
