"""Turn post-processing — pure python, torch-free by design.

Fixed operation order (contract): drop short turns → merge same-speaker gaps →
mark overlap → compute speaker summaries. Kept importable without GPU deps so
unit/contract tests can exercise it directly.
"""

from typing import Any


def merge_short_gaps(
    turns: list[dict[str, Any]], min_duration_off: float
) -> list[dict[str, Any]]:
    """Merge adjacent same-speaker turns separated by less than ``min_duration_off``.

    Natural speech pauses otherwise fragment a speaker's contiguous speech into
    many short turns.
    """
    if len(turns) < 2:
        return [t.copy() for t in turns]

    ordered = sorted(turns, key=lambda t: t["start_seconds"])
    merged = [ordered[0].copy()]
    for current in ordered[1:]:
        prev = merged[-1]
        gap = current["start_seconds"] - prev["end_seconds"]
        if prev["label"] == current["label"] and 0 <= gap <= min_duration_off:
            prev["end_seconds"] = max(prev["end_seconds"], current["end_seconds"])
        else:
            merged.append(current.copy())
    return merged


def mark_overlaps(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Set ``overlap`` and ``overlap_seconds`` on each turn.

    ``overlap_seconds`` sums the intersection with every different-speaker turn,
    so callers can distinguish a grazing overlap from a fully-overlapped turn.
    Overlapped speech embeds poorly — embedding extraction should skip or trim
    heavily-overlapped turns.
    """
    ordered = sorted(turns, key=lambda t: t["start_seconds"])
    for turn in ordered:
        overlap_total = 0.0
        for other in ordered:
            if other is turn or other["label"] == turn["label"]:
                continue
            lo = max(turn["start_seconds"], other["start_seconds"])
            hi = min(turn["end_seconds"], other["end_seconds"])
            if hi > lo:
                overlap_total += hi - lo
        turn["overlap"] = overlap_total > 0.0
        turn["overlap_seconds"] = round(overlap_total, 6)
    return ordered


def summarize_speakers(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-speaker totals over the returned turns, most talk-time first.

    Overlapping time counts for every speaker involved (no de-duplication).
    """
    totals: dict[str, dict[str, Any]] = {}
    for turn in turns:
        entry = totals.setdefault(
            turn["label"], {"label": turn["label"], "total_seconds": 0.0, "num_turns": 0}
        )
        entry["total_seconds"] += turn["end_seconds"] - turn["start_seconds"]
        entry["num_turns"] += 1
    return sorted(totals.values(), key=lambda s: s["total_seconds"], reverse=True)


def process_turns(
    raw_turns: list[dict[str, Any]],
    *,
    min_turn_seconds: float,
    min_duration_off: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Full post-processing chain. Returns (turns, speaker summaries)."""
    kept = [
        t for t in raw_turns if (t["end_seconds"] - t["start_seconds"]) >= min_turn_seconds
    ]
    merged = merge_short_gaps(kept, min_duration_off)
    marked = mark_overlaps(merged)
    return marked, summarize_speakers(marked)
