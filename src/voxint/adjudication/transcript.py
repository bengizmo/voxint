"""Read-time transcript presentation: the one attributed view HTML + export share.

Both the ``/runs/{id}/transcript`` HTML page and the ``/review/{id}/export.txt``
download resolve speaker names the same way — through :func:`label_states` — so
the two can never disagree. Attribution lives here, once; each caller only owns
its own formatting (HTML vs plain text).
"""

import enum
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy.orm import Session

from voxint.adjudication.attribution import walk_attributions
from voxint.adjudication.resolver import LabelState, Resolution, SegmentOverride
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
    # The canonical speaker id of the child's OWN active word-range override, if
    # any (issue #59 slice 3). ``None`` when the child has no range-scoped ruling
    # and merely inherits the segment's whole-segment/label speaker. The reassign
    # picker binds its ``<select>`` to this — so the control shows a child-scoped
    # assignment ONLY when one truly exists, never an inherited speaker dressed up
    # as a child ruling, and "inherit" is selected exactly when this is ``None``.
    word_range_speaker_id: uuid.UUID | None = None
    # Deterministic domain-pack correction provenance (#83). Carried straight from
    # the segment so the console can show which pack/rule produced each edit and
    # offer the immutable raw text one action away. Set on a WHOLE segment only —
    # a split parent's derived children leave these ``None`` (provenance is
    # parent-scoped; a corrected segment is never split, and a child slice must
    # never claim the parent's enhanced-text-coordinate spans). ``correction_trace``
    # is the persisted envelope (or ``[]``); ``corrector_version`` gates read-time
    # reconstruction; ``raw_text`` is the immutable ASR evidence for the compare/
    # reset affordance.
    correction_trace: dict[str, object] | list[object] | None = None
    corrector_version: int | None = None
    raw_text: str | None = None


@dataclass(frozen=True)
class TranscriptParagraph:
    """Consecutive same-speaker lines merged into one reading paragraph.

    The single grouping unit shared by the on-screen read mode and the Markdown
    export, so the two can never drift on where a paragraph begins or ends.
    ``text`` is the constituent lines joined with the boundary rule below;
    ``start_seconds``/``end_seconds`` span the first line's start to the last
    line's end.
    """

    speaker: str
    start_seconds: float
    end_seconds: float
    text: str


def _join_segment_texts(texts: Sequence[str]) -> str:
    """Join consecutive segment texts into one paragraph body.

    Embedded newlines inside a segment are preserved. A single ASCII space is
    inserted at a segment boundary only when neither side already carries
    whitespace there, so ``"Hello." + "Still here."`` becomes
    ``"Hello. Still here."`` while a segment that already ends in a newline is
    not given an extra space. Empty segments contribute nothing (no double
    spaces).
    """
    out = ""
    for piece in texts:
        if not piece:
            continue
        if out and not out[-1].isspace() and not piece[0].isspace():
            out += " "
        out += piece
    return out


def paragraphize_transcript(
    lines: Sequence[TranscriptLine],
) -> list[TranscriptParagraph]:
    """Merge adjacent same-speaker lines into reading paragraphs.

    Grouping is on the already-resolved ``speaker`` (exact equality, never the
    diarization label), and only for lines that are adjacent in transcript order
    — a speaker returning after someone else starts a new paragraph, so
    chronology is preserved. Pure: no DB, no I/O.
    """
    paragraphs: list[TranscriptParagraph] = []
    group: list[TranscriptLine] = []

    def flush() -> None:
        if not group:
            return
        paragraphs.append(
            TranscriptParagraph(
                speaker=group[0].speaker,
                start_seconds=group[0].start_seconds,
                end_seconds=group[-1].end_seconds,
                text=_join_segment_texts([ln.text for ln in group]),
            )
        )

    for line in lines:
        if group and line.speaker != group[0].speaker:
            flush()
            group = []
        group.append(line)
    flush()
    return paragraphs


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
    if state.resolution in (
        Resolution.HUMAN_ASSIGN,
        Resolution.GROUNDED_COSINE,
        Resolution.AUTO_ENROLL,
    ):
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

    The precedence walk itself (batch loads, split expansion, word-range >
    whole-segment > label) lives in :func:`walk_attributions`, shared with the
    identity-grade :func:`~voxint.adjudication.attribution.attributed_intervals`
    (issue #159) — this function only owns the display formatting.
    """
    lines: list[TranscriptLine] = []
    for emission in walk_attributions(session, run_id):
        seg = emission.seg
        rs = emission.review
        corrected_text = rs.corrected_text if rs is not None else None
        # Most-specific scope wins for the rendered name: an active whole-segment
        # override beats the label's resolution (issue #54 Phase B); a word-range
        # override (below) beats both for its exact child range.
        if emission.seg_override is not None:
            speaker = segment_speaker(emission.seg_override, seg)
        else:
            speaker = display_name(emission.label_state, seg)
        verified = rs is not None and rs.verified_at is not None
        # Reflects whether a correction EXISTS, independent of which variant is
        # rendered — so a ?text=raw view still badges "this segment was corrected"
        # while showing the raw evidence.
        corrected = corrected_text is not None
        if emission.child is not None:
            child = emission.child
            child_override = emission.range_override
            child_speaker = (
                segment_speaker(child_override, seg) if child_override is not None else speaker
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
                    review_target=emission.child_index == 0,
                    # The child's exact word range — what the per-child
                    # reassign picker posts to scope a ruling to this child.
                    word_start=child.word_start,
                    word_end=child.word_end,
                    # The child's OWN range override id (None ⇒ inheriting) —
                    # what the picker's <select> binds to, so it reflects true
                    # child-scope, not the resolved (possibly inherited) speaker.
                    word_range_speaker_id=(
                        child_override.speaker_id if child_override is not None else None
                    ),
                )
            )
        else:
            lines.append(
                TranscriptLine(
                    start_seconds=seg.start_seconds,
                    end_seconds=seg.end_seconds,
                    speaker=speaker,
                    text=_resolve_body(seg, corrected_text, text),
                    diarization_label=seg.diarization_label,
                    confidence=seg.confidence,
                    segment_id=seg.id,
                    verified=verified,
                    corrected=corrected,
                    source_segment_id=seg.id,
                    review_target=True,
                    # Whole-segment correction provenance (#83). Split children
                    # above deliberately omit these (parent-scoped).
                    correction_trace=seg.correction_trace,
                    corrector_version=seg.corrector_version,
                    raw_text=seg.raw_text,
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
