"""Per-speaker identity colors (issue #50).

A deterministic, order-independent map from a run's diarization labels to small
palette indices. Assignment is derived from ONE canonical per-run label universe
(the run's `label_states`) so the transcript page and the workbench card for the
same label always agree, and JS-off fallback markup matches hydrated islands.
Color is a SUPPLEMENTAL cue only — the raw label text is the primary, non-color
identifier shared across both surfaces (accessibility: never color alone).
"""

from collections.abc import Iterable

# Curated palette size. Contrast-safe accent colors live in CSS (base.html) as
# `.spk-0 .. .spk-{N-1}`. Beyond this the palette repeats by design (the diarizer
# supports more labels than any set of mutually-distinguishable, contrast-safe
# colors); the raw-label badge disambiguates repeats.
PALETTE_SIZE = 8


def speaker_palette(labels: Iterable[str]) -> dict[str, int]:
    """Map each distinct label to a palette index in [0, PALETTE_SIZE).

    Positional over the sorted distinct labels: deterministic and independent of
    input order. Callers MUST pass the run's canonical label universe (from
    `label_states`) so independently-rendered surfaces cannot drift.
    """
    return {label: i % PALETTE_SIZE for i, label in enumerate(sorted(set(labels)))}
