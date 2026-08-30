"""Pure-logic tests for the "More like this passage" shaping (#357).

The DB-backed query path is covered by the integration suite; these pin what
must be right without a database: source-span overlap exclusion (including the
identical-timestamp split chunks a long paragraph produces), exact-span
dedupe, the per-run cap, deterministic order, and the plain-text preview.
"""

import uuid
from datetime import UTC, datetime

from voxint.api.meaning_query import Candidate
from voxint.api.similar_query import _preview, shape_similar

_SOURCE_RUN = uuid.UUID("00000000-0000-0000-0000-0000000000a0")
_RUN_B = uuid.UUID("00000000-0000-0000-0000-0000000000b0")
_RUN_C = uuid.UUID("00000000-0000-0000-0000-0000000000c0")
_WHEN = datetime(2026, 8, 30, 12, 0, tzinfo=UTC)


def _cand(
    *,
    run_id: uuid.UUID = _RUN_B,
    chunk_index: int = 0,
    start: float = 0.0,
    end: float = 5.0,
    distance: float | None = 0.5,
) -> Candidate:
    return Candidate(
        id=uuid.uuid4(),
        run_id=run_id,
        chunk_index=chunk_index,
        run_created_at=_WHEN,
        title="A recording",
        source_path="incoming/x.wav",
        speaker_label="S0",
        start_seconds=start,
        end_seconds=end,
        chunk_text="a passage",
        distance=distance,
    )


def _shape(candidates: list[Candidate], **kwargs: float) -> list[Candidate]:
    return shape_similar(
        candidates,
        source_run_id=_SOURCE_RUN,
        source_start=10.0,
        source_end=20.0,
        **kwargs,  # type: ignore[arg-type]
    )


class TestSourceSpanExclusion:
    def test_source_run_overlap_excluded(self) -> None:
        overlapping = _cand(run_id=_SOURCE_RUN, start=15.0, end=25.0)
        assert _shape([overlapping]) == []

    def test_source_run_containing_paragraph_excluded(self) -> None:
        # The originating paragraph spans past the segment on both sides.
        paragraph = _cand(run_id=_SOURCE_RUN, start=5.0, end=40.0)
        assert _shape([paragraph]) == []

    def test_identical_timestamp_split_chunks_all_excluded(self) -> None:
        # Split pieces of one long paragraph share the paragraph's timestamps;
        # every piece must go, not just the "covering" one.
        pieces = [
            _cand(run_id=_SOURCE_RUN, chunk_index=i, start=5.0, end=40.0, distance=0.1 * i)
            for i in range(3)
        ]
        assert _shape(pieces) == []

    def test_source_run_non_overlapping_kept(self) -> None:
        elsewhere = _cand(run_id=_SOURCE_RUN, start=100.0, end=110.0)
        assert _shape([elsewhere]) == [elsewhere]

    def test_touching_boundary_is_not_overlap(self) -> None:
        adjacent = _cand(run_id=_SOURCE_RUN, start=20.0, end=30.0)
        assert _shape([adjacent]) == [adjacent]

    def test_other_run_overlapping_interval_kept(self) -> None:
        # Overlap exclusion is scoped to the source run; the same clock range
        # in a different recording is a legitimate result.
        other = _cand(run_id=_RUN_B, start=15.0, end=25.0)
        assert _shape([other]) == [other]


class TestSpanDedupe:
    def test_same_span_keeps_closest(self) -> None:
        far = _cand(run_id=_RUN_B, chunk_index=0, start=0.0, end=30.0, distance=0.8)
        near = _cand(run_id=_RUN_B, chunk_index=1, start=0.0, end=30.0, distance=0.2)
        assert _shape([far, near]) == [near]

    def test_dedupe_order_independent(self) -> None:
        far = _cand(run_id=_RUN_B, chunk_index=0, start=0.0, end=30.0, distance=0.8)
        near = _cand(run_id=_RUN_B, chunk_index=1, start=0.0, end=30.0, distance=0.2)
        assert _shape([near, far]) == [near]


class TestCapAndOrder:
    def test_ordered_by_distance(self) -> None:
        a = _cand(run_id=_RUN_B, start=0.0, end=5.0, distance=0.9)
        b = _cand(run_id=_RUN_C, start=0.0, end=5.0, distance=0.1)
        assert _shape([a, b]) == [b, a]

    def test_per_run_cap(self) -> None:
        b_hits = [
            _cand(
                run_id=_RUN_B,
                chunk_index=i,
                start=i * 10.0,
                end=i * 10.0 + 5.0,
                distance=0.1 * (i + 1),
            )
            for i in range(4)
        ]
        c_hit = _cand(run_id=_RUN_C, start=0.0, end=5.0, distance=0.95)
        kept = _shape([*b_hits, c_hit], per_run_cap=2)
        assert [c.run_id for c in kept] == [_RUN_B, _RUN_B, _RUN_C]

    def test_top_k_truncates(self) -> None:
        hits = [
            _cand(run_id=uuid.uuid4(), start=0.0, end=5.0, distance=0.1 * i)
            for i in range(6)
        ]
        assert len(_shape(hits, top_k=3)) == 3

    def test_missing_distance_sorts_last(self) -> None:
        unknown = _cand(run_id=_RUN_B, start=0.0, end=5.0, distance=None)
        known = _cand(run_id=_RUN_C, start=0.0, end=5.0, distance=0.9)
        assert _shape([unknown, known]) == [known, unknown]

    def test_distance_tie_breaks_deterministically(self) -> None:
        a = _cand(run_id=_RUN_B, chunk_index=2, start=0.0, end=5.0, distance=0.5)
        b = _cand(run_id=_RUN_B, chunk_index=1, start=10.0, end=15.0, distance=0.5)
        # Same run + distance: lower chunk_index first, both orders of input.
        assert _shape([a, b]) == [b, a]
        assert _shape([b, a]) == [b, a]


class TestPreview:
    def test_short_text_verbatim(self) -> None:
        assert _preview("a short passage") == "a short passage"

    def test_whitespace_collapsed(self) -> None:
        assert _preview("spread\n  over\t lines") == "spread over lines"

    def test_long_text_elided(self) -> None:
        long = "word " * 100
        out = _preview(long)
        assert out.endswith(" …")
        assert len(out) <= 205
