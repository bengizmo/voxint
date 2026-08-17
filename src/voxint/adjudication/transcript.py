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

from voxint.adjudication.resolver import (
    LabelState,
    Resolution,
    SegmentOverride,
    label_states,
    review_states,
    segment_states,
    word_range_states,
)
from voxint.adjudication.splits import boundaries_for_run, derive_children
from voxint.db.models import TranscriptSegment


class TranscriptText(enum.StrEnum):
    """Which stored text a transcript view renders (precedence ladder).

    CORRECTED is the default: it applies the operator's per-segment corrections
    (issue #58) on top of the pipeline text. ENHANCED is the pipeline rendering
    with no corrections; RAW is the immutable ASR evidence. Order here is the
    order the variant switcher shows.
    """

    CORRECTED = "corrected"  # corrected → enhanced → raw (operator-effective; default)
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
    # ASR confidence (exp(avg_logprob), a transformed likelihood — NOT a
    # calibrated probability). None when unknown; the #53 review console flags
    # low-confidence segments. Carried on the resolved line so the JS-off
    # fallback and the island flag identically; exports ignore it.
    confidence: float | None = None
    # The transcript segment this line resolves — the write target for the
    # verify / correct routes (issues #53/#58). Always set by the resolver;
    # optional only so export-formatter tests can build a line without one.
    segment_id: uuid.UUID | None = None
    # Operator review state (issues #53/#58): whether this segment is verified,
    # and whether the rendered text is an operator correction (drives the
    # "edited" badge). Both default to the unreviewed state.
    verified: bool = False
    corrected: bool = False
    # Word-boundary split provenance (issue #59). A split parent expands into
    # several derived child lines: every child carries the immutable PARENT id as
    # ``source_segment_id`` (the verify/correct/split write target — a split's
    # verification stays PARENT-scoped, never per child), while exactly one child
    # per parent has ``review_target=True`` so the N-of-M queue counts one target
    # per parent and no child is double-counted. An unsplit line is its own source
    # and its own review target. ``None``/``False`` on a synthetic export line.
    source_segment_id: uuid.UUID | None = None
    review_target: bool = False
    # Word-range coordinates of a split child (issue #59 slice 3): the exact
    # ``[word_start, word_end)`` this line covers within its parent. Set only on
    # a split parent's derived children — the coordinate the per-child reassign
    # picker posts to ``/relabel`` to scope a ruling to this child alone. ``None``
    # on unsplit lines and synthetic/export lines (no partitionable range).
    word_start: int | None = None
    word_end: int | None = None


def parse_transcript_text(raw: str | None) -> TranscriptText:
    """A blank/absent value means the default 'corrected' (operator-effective)
    view; anything else must be a named variant."""
    if raw in (None, ""):
        return TranscriptText.CORRECTED
    try:
        return TranscriptText(raw)
    except ValueError as exc:
        raise ValueError(f"unknown transcript text {raw!r}") from exc


def effective_text(seg: TranscriptSegment, corrected_text: str | None) -> str:
    """The operator-effective rendering: corrected → enhanced → raw.

    ``IS NOT NULL``, never truthiness — the ONE selector shared by the CORRECTED
    display default, the exports, the search index, and the run-asset/enrichment
    generators, so those four can never drift. Callers with no review row pass
    ``corrected_text=None``.
    """
    if corrected_text is not None:
        return corrected_text
    if seg.enhanced_text is not None:
        return seg.enhanced_text
    return seg.raw_text


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


def segment_speaker(override: SegmentOverride, seg: TranscriptSegment) -> str:
    """The speaker string for a segment carrying an active per-segment override.

    Segment scope only assigns (inherit is filtered out upstream), so an override
    always names a speaker; fall back to the raw label only if the assigned
    speaker somehow has no name.
    """
    return override.speaker_name or seg.diarization_label or "(no speaker)"


def attributed_transcript(
    session: Session, run_id: uuid.UUID, *, text: TranscriptText
) -> list[TranscriptLine]:
    """Every segment of a run in order, each attributed through the resolver.

    A per-segment override (issue #54 Phase B) wins for its own segment; every
    other segment falls through to its label's resolution — so a segment that
    inherits (or was never overridden) tracks later label rulings live, never a
    frozen copy. Both the HTML transcript page and the text export share this
    one function, so they can never disagree.
    """
    states = {s.label: s for s in label_states(session, run_id)}
    overrides = segment_states(session, run_id)
    # Per-word-range overrides (issue #59 slice 3): a reassigned split child wins
    # over its parent's whole-segment/label speaker, for its exact range only.
    range_overrides = word_range_states(session, run_id)
    review = review_states(session, run_id)
    # Every split boundary of the run, grouped by parent id, in one query — so the
    # child expansion below adds no per-segment round-trip (issue #59).
    boundaries = boundaries_for_run(session, run_id)
    segments = session.execute(
        select(TranscriptSegment)
        .where(TranscriptSegment.pipeline_run_id == run_id)
        .order_by(TranscriptSegment.segment_index)
    ).scalars()
    lines: list[TranscriptLine] = []
    for seg in segments:
        rs = review.get(seg.id)
        corrected_text = rs.corrected_text if rs is not None else None
        body = _resolve_body(seg, corrected_text, text)
        override = overrides.get(seg.id)
        if override is not None:
            speaker = segment_speaker(override, seg)
        else:
            speaker = display_name(states.get(seg.diarization_label or ""), seg)
        verified = rs is not None and rs.verified_at is not None
        # Reflects whether a correction EXISTS, independent of which variant is
        # rendered — so a ?text=raw view still badges "this segment was corrected"
        # while showing the raw evidence.
        corrected = corrected_text is not None
        cuts = boundaries.get(seg.id)
        # A split parent (>= 2 derived children) expands into per-child lines that
        # inherit the parent's resolved speaker and review state; every other
        # segment stays a single line. ``derive_children`` returns None for an
        # unsplittable parent — fail closed by rendering it whole (never invent
        # child offsets), which cannot happen for a boundary the writer accepted
        # (segments are immutable, so splittability is stable once checked).
        children = derive_children(seg, cuts) if cuts else None
        if children is not None and len(children) > 1:
            for child_i, child in enumerate(children):
                # Most-specific scope wins: a word-range override for this exact
                # child range beats the parent's whole-segment/label speaker; with
                # no such override the child inherits the parent's ``speaker``.
                child_override = range_overrides.get(
                    (seg.id, child.word_start, child.word_end)
                )
                child_speaker = (
                    segment_speaker(child_override, seg)
                    if child_override is not None
                    else speaker
                )
                lines.append(
                    TranscriptLine(
                        start_seconds=child.start_seconds,
                        end_seconds=child.end_seconds,
                        speaker=child_speaker,
                        text=child.text,
                        diarization_label=seg.diarization_label,
                        confidence=seg.confidence,
                        # Verify/correct/split all target the immutable parent;
                        # segment_id stays the parent id so a child's write lands
                        # on the parent, and review state is parent-scoped.
                        segment_id=seg.id,
                        verified=verified,
                        corrected=corrected,
                        source_segment_id=seg.id,
                        # One queue entry per parent: only the first child counts.
                        review_target=child_i == 0,
                        # The child's exact word range — what the per-child
                        # reassign picker posts to scope a ruling to this child.
                        word_start=child.word_start,
                        word_end=child.word_end,
                    )
                )
        else:
            lines.append(
                TranscriptLine(
                    start_seconds=seg.start_seconds,
                    end_seconds=seg.end_seconds,
                    speaker=speaker,
                    text=body,
                    diarization_label=seg.diarization_label,
                    confidence=seg.confidence,
                    segment_id=seg.id,
                    verified=verified,
                    corrected=corrected,
                    source_segment_id=seg.id,
                    review_target=True,
                )
            )
    return lines


def _resolve_body(
    seg: TranscriptSegment, corrected_text: str | None, variant: TranscriptText
) -> str:
    """The text a variant renders for one segment. RAW is the immutable ASR
    evidence; ENHANCED is the pipeline text (no corrections); CORRECTED (default)
    applies the operator's correction via :func:`effective_text`."""
    if variant is TranscriptText.RAW:
        return seg.raw_text
    if variant is TranscriptText.ENHANCED:
        # IS NOT NULL (not truthiness), so an intentionally-emptied enhanced
        # segment does not resurrect raw text here while the default view shows
        # it empty — the two renderings must agree on the empty-enhanced edge.
        return seg.enhanced_text if seg.enhanced_text is not None else seg.raw_text
    return effective_text(seg, corrected_text)
