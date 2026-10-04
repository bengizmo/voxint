"""Exact reading-filter note copy, independent of HTTP and stored data."""

import pytest

from voxint.api.routers.legacy_runs import read_filter_note


@pytest.mark.parametrize("fillers,repeats,f_count,r_count,phrase", [
    (False, False, 0, 0, None),
    (False, False, 4, 2, None),
    (True, True, 4, 1, "4 filler words and 1 repeated word"),
    (True, False, 0, 9, "no filler words"),
    (True, False, 1, 9, "1 filler word"),
    (True, True, 0, 0, "no filler words and no repeated words"),
    (False, True, 9, 2, "2 repeated words"),
    (False, True, 9, 1, "1 repeated word"),
    (False, True, 9, 0, "no repeated words"),
])
def test_read_filter_note(
    fillers: bool, repeats: bool, f_count: int, r_count: int, phrase: str | None,
) -> None:
    expected = f"Left out {phrase}. The saved transcript is unchanged." if phrase else None
    assert read_filter_note(
        drop_fillers=fillers, drop_repeats=repeats,
        fillers_removed=f_count, repeats_removed=r_count,
    ) == expected
