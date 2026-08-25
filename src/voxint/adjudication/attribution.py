"""Structured effective attribution: the machine-readable twin of the transcript.

:func:`attributed_transcript` renders a run for people (display names, text
variants, review badges). Aggregation (issue #159) needs the same fold with
identity intact: WHICH canonical speaker won each interval, at WHAT scope, by
WHAT kind of ruling. ``TranscriptLine`` deliberately carries none of that (its
``speaker`` is a display string and its ``verified`` is segment-review state,
not attribution), so this module owns the shared walk both views consume —
precedence (word-range > whole-segment > label), split expansion, and recency
all live here exactly once, and the two projections can never disagree.
"""

import enum
import uuid
from collections.abc import Iterator
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from voxint.adjudication.resolver import (
    LabelState,
    Resolution,
    SegmentOverride,
    WordRangeKey,
    label_states,
    review_states,
    segment_states,
    word_range_states,
)
from voxint.adjudication.splits import DerivedChild, boundaries_for_run, derive_children
from voxint.db.models import SegmentReviewState, TranscriptSegment


class AttributionScope(enum.StrEnum):
    """Which grain of ruling won an interval (most-specific-wins order)."""

    WORD_RANGE = "word_range"  # an active override for the child's exact range
    SEGMENT = "segment"  # an active whole-segment override
    LABEL = "label"  # the label's resolution (human or grounded machine)


@dataclass(frozen=True)
class Emission:
    """One walk step: a segment (or derived split child) plus every resolver
    overlay that could attribute it. The shared intermediate both projections
    map over — :func:`attributed_intervals` keeps the identity fields,
    ``attributed_transcript`` formats the display fields. ``child`` is ``None``
    for an unsplit segment; ``range_override`` is the override for exactly this
    child's range (never a sibling's)."""

    seg: TranscriptSegment
    child: DerivedChild | None
    child_index: int | None
    label_state: LabelState | None
    seg_override: SegmentOverride | None
    range_override: SegmentOverride | None
    review: SegmentReviewState | None


@dataclass(frozen=True)
class AttributedInterval:
    """One interval's winning attribution, identity-grade.

    ``speaker_id`` is the CANONICAL id (merge tombstones already followed by the
    resolver) and is set iff the winning ruling attributes a speaker — a human
    assign at any scope, or a grounded cosine match. Excluded/unknown/unresolved
    intervals carry ``None``. ``resolution`` is the winning ruling's kind:
    override scopes are human assigns by construction; label scope carries the
    label's resolution (``UNRESOLVED`` when the label has no turns and so no
    state). ``segment_id`` is always the immutable parent segment.
    """

    start_seconds: float
    end_seconds: float
    segment_id: uuid.UUID
    diarization_label: str | None
    scope: AttributionScope
    resolution: Resolution
    speaker_id: uuid.UUID | None
    speaker_name: str | None
    # Set only on a split parent's derived children (half-open word coords).
    word_start: int | None
    word_end: int | None

    @property
    def is_human_assign(self) -> bool:
        """True iff a human assign won this interval (any scope)."""
        return self.resolution is Resolution.HUMAN_ASSIGN and self.speaker_id is not None


def walk_attributions(session: Session, run_id: uuid.UUID) -> Iterator[Emission]:
    """Walk a run's segments in order, expanding split children, yielding each
    with its resolver overlays. THE shared fold: every batch load and the
    split-expansion rule live here so no projection re-derives them. A split
    parent whose boundaries cannot derive >= 2 children is emitted whole
    (fail-closed, matching the transcript rendering)."""
    states = {s.label: s for s in label_states(session, run_id)}
    overrides = segment_states(session, run_id)
    range_overrides = word_range_states(session, run_id)
    review = review_states(session, run_id)
    boundaries = boundaries_for_run(session, run_id)
    segments = session.execute(
        select(TranscriptSegment)
        .where(TranscriptSegment.pipeline_run_id == run_id)
        .order_by(TranscriptSegment.segment_index)
    ).scalars()
    for seg in segments:
        label_state = states.get(seg.diarization_label or "")
        seg_override = overrides.get(seg.id)
        rs = review.get(seg.id)
        cuts = boundaries.get(seg.id)
        children = derive_children(seg, cuts) if cuts else None
        if children is not None and len(children) > 1:
            for child_i, child in enumerate(children):
                key: WordRangeKey = (seg.id, child.word_start, child.word_end)
                yield Emission(
                    seg=seg,
                    child=child,
                    child_index=child_i,
                    label_state=label_state,
                    seg_override=seg_override,
                    range_override=range_overrides.get(key),
                    review=rs,
                )
        else:
            yield Emission(
                seg=seg,
                child=None,
                child_index=None,
                label_state=label_state,
                seg_override=seg_override,
                range_override=None,
                review=rs,
            )


def winning_attribution(
    emission: Emission,
) -> tuple[AttributionScope, Resolution, uuid.UUID | None, str | None]:
    """The most-specific active ruling for one emission.

    Word-range override > whole-segment override > label resolution. Override
    scopes are human assigns by construction (``_active_overrides`` keeps only
    assigns); label scope reports the label's own resolution, ``UNRESOLVED``
    when the label never had a state (no diarization turns).
    """
    if emission.range_override is not None:
        return (
            AttributionScope.WORD_RANGE,
            Resolution.HUMAN_ASSIGN,
            emission.range_override.speaker_id,
            emission.range_override.speaker_name,
        )
    if emission.seg_override is not None:
        return (
            AttributionScope.SEGMENT,
            Resolution.HUMAN_ASSIGN,
            emission.seg_override.speaker_id,
            emission.seg_override.speaker_name,
        )
    state = emission.label_state
    if state is None:
        return (AttributionScope.LABEL, Resolution.UNRESOLVED, None, None)
    if state.resolution in (Resolution.HUMAN_ASSIGN, Resolution.GROUNDED_COSINE):
        return (AttributionScope.LABEL, state.resolution, state.speaker_id, state.speaker_name)
    return (AttributionScope.LABEL, state.resolution, None, None)


def attributed_intervals(session: Session, run_id: uuid.UUID) -> list[AttributedInterval]:
    """Every interval of a run with its winning attribution, in transcript order.

    The aggregation-facing projection of :func:`walk_attributions`: same
    expansion and precedence as the rendered transcript, but carrying canonical
    speaker ids and ruling kinds instead of display strings. An interval's
    bounds are the child's word-derived span for split children, else the
    segment's own interval.
    """
    out: list[AttributedInterval] = []
    for emission in walk_attributions(session, run_id):
        scope, resolution, speaker_id, speaker_name = winning_attribution(emission)
        seg = emission.seg
        child = emission.child
        out.append(
            AttributedInterval(
                start_seconds=child.start_seconds if child is not None else seg.start_seconds,
                end_seconds=child.end_seconds if child is not None else seg.end_seconds,
                segment_id=seg.id,
                diarization_label=seg.diarization_label,
                scope=scope,
                resolution=resolution,
                speaker_id=speaker_id,
                speaker_name=speaker_name,
                word_start=child.word_start if child is not None else None,
                word_end=child.word_end if child is not None else None,
            )
        )
    return out
