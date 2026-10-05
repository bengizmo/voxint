"""Shared render-time turn filters with word-token removal counts.

Apply fillers before repeats, preserving filler seams as repeat boundaries.
Count whitespace-separated tokens containing a letter or digit at each step.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from voxint.adjudication.turns import SpeakerTurn, join_pieces
from voxint.export.filler_lists import FillerList
from voxint.export.fillers import drop_fillers_with_seams
from voxint.export.repeats import drop_repeats as remove_repeats


@dataclass(frozen=True)
class FilteredTurns:
    turns: list[SpeakerTurn]
    fillers_removed: int
    repeats_removed: int


def _word_count(turns: Sequence[SpeakerTurn]) -> int:
    return sum(
        any(char.isalnum() for char in token)
        for turn in turns
        for token in join_pieces(turn.pieces).split()
    )


def apply_turn_filters(
    turns: Sequence[SpeakerTurn], *, fillers: FillerList | None, drop_repeats: bool
) -> FilteredTurns:
    """Compose turn filters and measure each step on its own input text."""
    filtered = list(turns)
    fillers_removed = repeats_removed = 0
    seams: list[frozenset[int]] | None = None
    if fillers is not None:
        before = _word_count(filtered)
        cleaned = drop_fillers_with_seams(filtered, fillers=fillers)
        filtered = [turn for turn, _ in cleaned]
        seams = [turn_seams for _, turn_seams in cleaned]
        fillers_removed = before - _word_count(filtered)
    if drop_repeats:
        before = _word_count(filtered)
        filtered = remove_repeats(filtered, seams)
        repeats_removed = before - _word_count(filtered)
    return FilteredTurns(filtered, fillers_removed, repeats_removed)
