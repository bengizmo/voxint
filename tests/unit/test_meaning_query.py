"""Pure-logic tests for the Meaning-search ranking (#121, PR2).

The DB-backed query path is covered by the integration suite; these tests pin the
parts that must be right without a database: quoted-span parsing, RRF fusion +
exact-quote priority + per-run cap + deterministic order, and the escape-safe
passage preview.
"""

import uuid
from datetime import UTC, datetime

from markupsafe import Markup

from voxint.api import meaning_query
from voxint.api.meaning_query import (
    Candidate,
    parse_positive_quotes,
    rank_candidates,
)

_RUN_A = uuid.UUID("00000000-0000-0000-0000-0000000000a0")
_RUN_B = uuid.UUID("00000000-0000-0000-0000-0000000000b0")
_WHEN = datetime(2026, 8, 22, 12, 0, tzinfo=UTC)


def _cand(
    *,
    run_id: uuid.UUID = _RUN_A,
    chunk_index: int = 0,
    vector_rank: int | None = None,
    lexical_rank: int | None = None,
    distance: float | None = None,
    exact_quote: bool = False,
    run_created_at: datetime = _WHEN,
    chunk_text: str = "a passage",
) -> Candidate:
    return Candidate(
        id=uuid.uuid4(),
        run_id=run_id,
        chunk_index=chunk_index,
        run_created_at=run_created_at,
        title="A recording",
        source_path="incoming/x.wav",
        speaker_label="S0",
        start_seconds=float(chunk_index) * 10.0,
        end_seconds=float(chunk_index) * 10.0 + 5.0,
        chunk_text=chunk_text,
        vector_rank=vector_rank,
        lexical_rank=lexical_rank,
        distance=distance,
        exact_quote=exact_quote,
    )


class TestParsePositiveQuotes:
    def test_no_quotes(self) -> None:
        assert parse_positive_quotes("who paid for the repairs") == []

    def test_single_positive(self) -> None:
        assert parse_positive_quotes('the "reversing valve" broke') == ["reversing valve"]

    def test_negative_excluded_not_promoted(self) -> None:
        assert parse_positive_quotes('valve -"heat pump"') == []

    def test_mix_positive_and_negative(self) -> None:
        assert parse_positive_quotes('"reversing valve" -"heat pump"') == [
            "reversing valve"
        ]

    def test_multiple_positive_dedup_case_insensitive(self) -> None:
        assert parse_positive_quotes('"Merger" and "merger" and "Deal"') == [
            "Merger",
            "Deal",
        ]

    def test_blank_quote_ignored(self) -> None:
        assert parse_positive_quotes('"   " "real"') == ["real"]

    def test_unbalanced_quote_no_match(self) -> None:
        assert parse_positive_quotes('a "dangling quote') == []


class TestRankCandidates:
    def test_empty(self) -> None:
        assert rank_candidates([]) == []

    def test_both_arms_beats_single_arm(self) -> None:
        both = _cand(vector_rank=5, lexical_rank=5)
        vector_only = _cand(vector_rank=1, run_id=_RUN_B)
        ranked = rank_candidates([vector_only, both])
        # Agreement across arms outranks a rank-1 hit in only one arm.
        assert ranked[0] is both

    def test_semantic_only_still_ranks(self) -> None:
        a = _cand(vector_rank=1, run_id=_RUN_A)
        b = _cand(vector_rank=2, run_id=_RUN_B)
        assert [c.id for c in rank_candidates([b, a])] == [a.id, b.id]

    def test_lexical_only_still_ranks(self) -> None:
        a = _cand(lexical_rank=1, run_id=_RUN_A)
        b = _cand(lexical_rank=2, run_id=_RUN_B)
        assert [c.id for c in rank_candidates([b, a])] == [a.id, b.id]

    def test_exact_quote_hard_priority(self) -> None:
        # A weak exact-quote hit (poor RRF) still beats a strong ordinary hit.
        quote = _cand(vector_rank=200, exact_quote=True, run_id=_RUN_A)
        strong = _cand(vector_rank=1, lexical_rank=1, run_id=_RUN_B)
        ranked = rank_candidates([strong, quote])
        assert ranked[0] is quote
        assert ranked[1] is strong

    def test_per_run_cap(self) -> None:
        same_run = [_cand(vector_rank=i, chunk_index=i, run_id=_RUN_A) for i in range(1, 6)]
        other = _cand(vector_rank=10, run_id=_RUN_B)
        ranked = rank_candidates([*same_run, other], per_run_cap=2)
        from_a = [c for c in ranked if c.run_id == _RUN_A]
        assert len(from_a) == 2
        assert other in ranked

    def test_top_k_truncates(self) -> None:
        cands = [
            _cand(vector_rank=i, chunk_index=i, run_id=uuid.uuid4()) for i in range(1, 11)
        ]
        assert len(rank_candidates(cands, top_k=3)) == 3

    def test_tie_break_prefers_closer_distance(self) -> None:
        # Equal RRF (both vector_rank=1), differ only by cosine distance.
        near = _cand(vector_rank=1, distance=0.1, run_id=_RUN_A)
        far = _cand(vector_rank=1, distance=0.9, run_id=_RUN_B)
        assert rank_candidates([far, near])[0] is near

    def test_deterministic_order_is_stable(self) -> None:
        cands = [_cand(vector_rank=1, chunk_index=i, run_id=uuid.uuid4()) for i in range(4)]
        first = [c.id for c in rank_candidates(list(reversed(cands)))]
        second = [c.id for c in rank_candidates(cands)]
        assert first == second


class TestPreview:
    def test_escapes_hostile_markup(self) -> None:
        out = meaning_query._preview("<script>alert(1)</script> hello", [])
        assert isinstance(out, Markup)
        assert "<script>" not in out
        assert "&lt;script&gt;" in out

    def test_marks_quoted_phrase(self) -> None:
        out = meaning_query._preview("the reversing valve failed", ["reversing valve"])
        assert "<mark>reversing valve</mark>" in out

    def test_mark_is_case_insensitive(self) -> None:
        out = meaning_query._preview("The Reversing Valve failed", ["reversing valve"])
        assert "<mark>Reversing Valve</mark>" in out

    def test_no_phrases_returns_head_unmarked(self) -> None:
        out = meaning_query._preview("a plain passage of text", [])
        assert "<mark>" not in out
        assert "plain passage" in out

    def test_truncates_long_text_with_ellipsis(self) -> None:
        out = meaning_query._preview("word " * 400, [])
        assert "…" in out
        assert len(out) < len("word " * 400)

    def test_window_anchors_on_phrase(self) -> None:
        text = "intro " * 200 + "the special marker here " + "outro " * 200
        out = meaning_query._preview(text, ["special marker"])
        assert "<mark>special marker</mark>" in out
