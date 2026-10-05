"""Shared render-time turn filters with word-token removal counts.

Apply fillers before repeats, preserving filler seams as repeat boundaries.
Count whitespace-separated tokens containing a letter or digit at each step.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from voxint.adjudication.turns import EffectiveMarks, SpeakerTurn, join_pieces
from voxint.export.filler_lists import FillerList
from voxint.export.fillers import Protected, Removal, drop_fillers_with_trace
from voxint.export.repeats import drop_repeats as remove_repeats


@dataclass(frozen=True)
class FilteredTurns:
    turns: list[SpeakerTurn]
    fillers_removed: int
    repeats_removed: int
    trace: tuple[Removal | Protected, ...] = ()
    omitted: int = 0
    kept: int = 0


def _word_count(turns: Sequence[SpeakerTurn]) -> int:
    return sum(
        any(char.isalnum() for char in token)
        for turn in turns
        for token in join_pieces(turn.pieces).split()
    )


def apply_turn_filters(
    turns: Sequence[SpeakerTurn], *, fillers: FillerList | None, drop_repeats: bool,
    marks: EffectiveMarks | None = None,
) -> FilteredTurns:
    """Apply marks with fillers only, then repeats; count final cleans, never probes."""
    filtered = list(turns)
    fillers_removed = repeats_removed = omitted = kept = 0
    trace: tuple[Removal | Protected, ...] = ()
    seams: list[frozenset[int]] | None = None
    if fillers is not None:
        before = _word_count(filtered)
        cleaned, trace = drop_fillers_with_trace(filtered, fillers=fillers, marks=marks)
        filtered = [turn for turn, _ in cleaned]
        seams = [turn_seams for _, turn_seams in cleaned]
        omitted_words = 0
        if marks:
            omitted = len(
                {source for entry in trace if entry.cause == "omit" for source in entry.sources}
            )
            kept = len(
                {
                    source
                    for entry in trace
                    if entry.cause == "kept"
                    for source in entry.sources
                    if marks and marks.get(source) == "keep"
                }
            )
            omitted_words = sum(
                any(char.isalnum() for char in token)
                for entry in trace
                if entry.cause == "omit"
                for token in entry.text.split()
            )
        fillers_removed = before - _word_count(filtered) - omitted_words
    if drop_repeats:
        before = _word_count(filtered)
        filtered = remove_repeats(filtered, seams)
        repeats_removed = before - _word_count(filtered)
    return FilteredTurns(filtered, fillers_removed, repeats_removed, trace, omitted, kept)
