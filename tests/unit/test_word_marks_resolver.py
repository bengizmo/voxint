"""Word marks resolve by sequence, with clear and undo restoring prior state."""

import uuid

from voxint.adjudication.word_marks import WordMarkRow, resolve_marks

SEGMENT = uuid.UUID(int=1)
OTHER_SEGMENT = uuid.UUID(int=2)


def row(
    seq: int,
    action: str,
    *,
    segment: uuid.UUID = SEGMENT,
    start: int = 0,
    end: int = 1,
    voids: int | None = None,
) -> WordMarkRow:
    return WordMarkRow(
        id=uuid.UUID(int=seq + 10),
        seq=seq,
        segment_id=segment,
        start_word_index=start,
        end_word_index=end,
        action=action,
        voids_mark_id=None if voids is None else uuid.UUID(int=voids + 10),
    )


def test_highest_seq_wins_even_when_input_is_unordered() -> None:
    assert resolve_marks(iter([row(3, "omit"), row(1, "omit"), row(2, "keep")])) == {
        (SEGMENT, 0, 1): "omit"
    }


def test_clear_removes_entry() -> None:
    assert resolve_marks([row(1, "keep"), row(2, "clear")]) == {}


def test_undo_restores_previous_mark() -> None:
    assert resolve_marks([row(3, "undo", voids=2), row(2, "omit"), row(1, "keep")]) == {
        (SEGMENT, 0, 1): "keep"
    }


def test_undo_clear_restores_mark_under_it() -> None:
    assert resolve_marks([row(1, "omit"), row(2, "clear"), row(3, "undo", voids=2)]) == {
        (SEGMENT, 0, 1): "omit"
    }


def test_different_units_are_independent() -> None:
    assert resolve_marks(
        [
            row(1, "keep"),
            row(2, "omit", start=1, end=3),
            row(3, "omit", segment=OTHER_SEGMENT),
            row(4, "clear"),
        ]
    ) == {(SEGMENT, 1, 3): "omit", (OTHER_SEGMENT, 0, 1): "omit"}


def test_undo_only_mark_leaves_no_entry() -> None:
    assert resolve_marks([row(1, "omit"), row(2, "undo", voids=1)]) == {}
