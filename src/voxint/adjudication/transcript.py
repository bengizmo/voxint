"""Read-time transcript presentation: the one attributed view HTML + export share.

Both the ``/runs/{id}/transcript`` HTML page and the ``/review/{id}/export.txt``
download resolve speaker names the same way — through :func:`label_states` — so
the two can never disagree. Attribution lives here, once; each caller only owns
its own formatting (HTML vs plain text).
"""

import enum
import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from voxint.adjudication.resolver import LabelState, Resolution, label_states
from voxint.db.models import TranscriptSegment


class TranscriptText(enum.StrEnum):
    """Which stored text a transcript view renders."""

    ENHANCED = "enhanced"  # prefer enhanced_text, fall back to raw when NULL
    RAW = "raw"  # the immutable ASR output, always


@dataclass(frozen=True)
class TranscriptLine:
    """One attributed segment: interval, resolved speaker, and its text."""

    start_seconds: float
    end_seconds: float
    speaker: str
    text: str
    diarization_label: str | None = None  # raw label (identity key for #50 colors)


def parse_transcript_text(raw: str | None) -> TranscriptText:
    """A blank/absent value means 'enhanced'; anything else must be a variant."""
    if raw in (None, ""):
        return TranscriptText.ENHANCED
    try:
        return TranscriptText(raw)
    except ValueError as exc:
        raise ValueError(f"unknown transcript text {raw!r}") from exc


def display_name(state: LabelState | None, seg: TranscriptSegment) -> str:
    """The speaker string for a segment, given its label's resolved state.

    A grounded/assigned label shows its speaker name; exclude/unknown rulings
    annotate the local label; everything unresolved falls back to the raw label.
    """
    label = seg.diarization_label or "(no speaker)"
    if state is None:
        return label
    if state.resolution in (Resolution.HUMAN_ASSIGN, Resolution.GROUNDED_COSINE):
        return state.speaker_name or label
    if state.resolution is Resolution.HUMAN_EXCLUDE:
        return f"(excluded) {label}"
    if state.resolution is Resolution.HUMAN_UNKNOWN:
        return f"Unknown ({label})"
    return label


def attributed_transcript(
    session: Session, run_id: uuid.UUID, *, text: TranscriptText
) -> list[TranscriptLine]:
    """Every segment of a run in order, each attributed through the resolver."""
    states = {s.label: s for s in label_states(session, run_id)}
    segments = session.execute(
        select(TranscriptSegment)
        .where(TranscriptSegment.pipeline_run_id == run_id)
        .order_by(TranscriptSegment.segment_index)
    ).scalars()
    lines: list[TranscriptLine] = []
    for seg in segments:
        body = seg.raw_text if text is TranscriptText.RAW else (seg.enhanced_text or seg.raw_text)
        lines.append(
            TranscriptLine(
                start_seconds=seg.start_seconds,
                end_seconds=seg.end_seconds,
                speaker=display_name(states.get(seg.diarization_label or ""), seg),
                text=body,
                diarization_label=seg.diarization_label,
            )
        )
    return lines
