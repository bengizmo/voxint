"""Per-speaker identity colors (issue #50).

A deterministic, order-independent map from a run's diarization labels to small
palette indices. Callers build it from ONE canonical per-run label universe (the
union of the run's diarization-turn and transcript-segment labels — see
`_run_label_universe`) so the transcript page and the workbench card for the
same label always agree, and JS-off fallback markup matches hydrated islands.
Color is a SUPPLEMENTAL cue only — the raw label text is the primary, non-color
identifier shared across both surfaces (accessibility: never color alone).
"""

import uuid
from collections.abc import Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from voxint.db.models import DiarizationTurn, TranscriptSegment

# Curated palette size. Contrast-safe accent colors live in CSS (base.html) as
# `.spk-0 .. .spk-{N-1}`. Beyond this the palette repeats by design (the diarizer
# supports more labels than any set of mutually-distinguishable, contrast-safe
# colors); the raw-label badge disambiguates repeats.
PALETTE_SIZE = 8


def run_label_universe(session: Session, run_id: uuid.UUID) -> set[str]:
    """Every diarization label present in a run, from BOTH its diarization turns
    and its transcript segments.

    A transcript segment may carry a label with no turn (the supported degenerate
    case the resolver's turn-derived ``label_states`` does not enumerate), and a
    turn's label may have no segment; the union covers both. This is the ONE
    canonical universe the per-speaker palette (#50) is built from, so the
    transcript page, its JS-off fallback, and the workbench cards color a given
    label identically. Two cheap indexed ``DISTINCT`` queries — deliberately not
    ``label_states`` (which resolves turn stats, proposals, decisions, and merges)."""
    turn_labels = session.execute(
        select(DiarizationTurn.label)
        .where(DiarizationTurn.pipeline_run_id == run_id)
        .distinct()
    ).scalars()
    segment_labels = session.execute(
        select(TranscriptSegment.diarization_label)
        .where(TranscriptSegment.pipeline_run_id == run_id)
        .distinct()
    ).scalars()
    return {*turn_labels, *(label for label in segment_labels if label is not None)}


def speaker_palette(labels: Iterable[str]) -> dict[str, int]:
    """Map each distinct label to a palette index in [0, PALETTE_SIZE).

    Positional over the sorted distinct labels: deterministic and independent of
    input order. Callers MUST pass the run's canonical label universe (from
    `_run_label_universe`) so independently-rendered surfaces cannot drift.
    """
    return {label: i % PALETTE_SIZE for i, label in enumerate(sorted(set(labels)))}
