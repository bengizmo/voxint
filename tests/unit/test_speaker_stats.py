"""Unit tests for speaker-level statistics (issue #335)."""

from __future__ import annotations

import uuid
from collections import Counter

from voxint.api.speaker_stats import (
    SpeakerTermStat,
    _collapse_consecutive,
    compute_ego_transitions,
    compute_log_odds,
    compute_wpm,
    tokenize_text,
)

# ---------------------------------------------------------------------------
# compute_log_odds
# ---------------------------------------------------------------------------


class TestComputeLogOdds:
    def test_distinctive_term_ranks_high(self) -> None:
        target = Counter({"hvac": 20, "system": 5, "the": 3, "cold": 10})
        background = Counter({"hvac": 2, "system": 50, "the": 200, "cold": 8})
        result = compute_log_odds(target, background, min_count=3)
        terms = [r.term for r in result]
        assert "hvac" in terms
        assert result[0].term == "hvac"
        assert result[0].z_score > 0

    def test_positive_z_scores_only(self) -> None:
        target = Counter({"rare": 5, "common": 3})
        background = Counter({"rare": 1, "common": 500})
        result = compute_log_odds(target, background, min_count=3)
        for stat in result:
            assert stat.z_score > 0

    def test_min_count_filtering(self) -> None:
        target = Counter({"hvac": 2, "system": 10})
        background = Counter({"hvac": 1, "system": 5})
        result = compute_log_odds(target, background, min_count=3)
        terms = [r.term for r in result]
        assert "hvac" not in terms

    def test_empty_target(self) -> None:
        assert compute_log_odds(Counter(), Counter({"a": 10})) == []

    def test_empty_background(self) -> None:
        assert compute_log_odds(Counter({"a": 10}), Counter()) == []

    def test_both_empty(self) -> None:
        assert compute_log_odds(Counter(), Counter()) == []

    def test_informative_prior_uses_background_frequency(self) -> None:
        """The prior alpha_w should be proportional to background frequency,
        not flat. A rare background term should get a smaller prior than a
        common one, making it easier for the rare term to show as distinctive
        when the speaker uses it more."""
        target = Counter({"rare_word": 5, "common_word": 5})
        background = Counter({"rare_word": 1, "common_word": 1000})
        result = compute_log_odds(target, background, min_count=3)
        by_term = {r.term: r for r in result}
        assert "rare_word" in by_term
        fallback = SpeakerTermStat("", 0, 0, 0)
        assert by_term["rare_word"].z_score > by_term.get("common_word", fallback).z_score

    def test_top_n_limits_output(self) -> None:
        target = Counter({f"term{i}": 10 for i in range(50)})
        background = Counter({f"term{i}": 1 for i in range(50)})
        result = compute_log_odds(target, background, min_count=3, top_n=5)
        assert len(result) <= 5

    def test_single_term_vocabulary_no_crash(self) -> None:
        """Regression: one-term vocabulary must not ZeroDivisionError."""
        result = compute_log_odds(Counter({"word": 3}), Counter({"word": 4}))
        assert result == []

    def test_deterministic_ordering(self) -> None:
        target = Counter({"alpha": 10, "beta": 10, "gamma": 10})
        background = Counter({"alpha": 5, "beta": 5, "gamma": 5})
        r1 = compute_log_odds(target, background, min_count=3)
        r2 = compute_log_odds(target, background, min_count=3)
        assert [s.term for s in r1] == [s.term for s in r2]


# ---------------------------------------------------------------------------
# compute_ego_transitions
# ---------------------------------------------------------------------------


class TestComputeEgoTransitions:
    def setup_method(self) -> None:
        self.alice = str(uuid.uuid4())
        self.bob = str(uuid.uuid4())
        self.carol = str(uuid.uuid4())

    def test_alternating_speakers(self) -> None:
        seq = [self.alice, self.bob, self.alice, self.bob, self.alice]
        t_in, t_out = compute_ego_transitions([seq], self.alice)
        assert len(t_in) == 1
        assert t_in[0].count == 2
        assert t_in[0].from_speaker_id == uuid.UUID(self.bob)
        assert len(t_out) == 1
        assert t_out[0].count == 2
        assert t_out[0].to_speaker_id == uuid.UUID(self.bob)

    def test_none_breaks_chain(self) -> None:
        seq = [self.alice, None, self.bob]
        t_in, t_out = compute_ego_transitions([seq], self.alice)
        assert len(t_out) == 0
        assert len(t_in) == 0

    def test_consecutive_same_speaker_collapsed(self) -> None:
        seq = [self.alice, self.alice, self.alice, self.bob]
        _t_in, t_out = compute_ego_transitions([seq], self.alice)
        assert len(t_out) == 1
        assert t_out[0].count == 1

    def test_multiple_runs_aggregated(self) -> None:
        seq1 = [self.alice, self.bob]
        seq2 = [self.alice, self.bob]
        _t_in, t_out = compute_ego_transitions([seq1, seq2], self.alice)
        assert t_out[0].count == 2

    def test_single_speaker_no_transitions(self) -> None:
        seq = [self.alice, self.alice, self.alice]
        t_in, t_out = compute_ego_transitions([seq], self.alice)
        assert len(t_in) == 0
        assert len(t_out) == 0

    def test_empty_sequences(self) -> None:
        t_in, t_out = compute_ego_transitions([], self.alice)
        assert t_in == []
        assert t_out == []

    def test_three_speakers(self) -> None:
        seq = [self.bob, self.alice, self.carol, self.alice, self.bob]
        t_in, t_out = compute_ego_transitions([seq], self.alice)
        in_ids = {str(e.from_speaker_id) for e in t_in}
        out_ids = {str(e.to_speaker_id) for e in t_out}
        assert self.bob in in_ids
        assert self.carol in out_ids
        assert self.bob in out_ids


# ---------------------------------------------------------------------------
# _collapse_consecutive
# ---------------------------------------------------------------------------


class TestCollapseConsecutive:
    def test_basic(self) -> None:
        assert _collapse_consecutive(["a", "a", "b", "b", "a"]) == ["a", "b", "a"]

    def test_with_nones(self) -> None:
        assert _collapse_consecutive(["a", None, None, "b"]) == ["a", None, "b"]

    def test_empty(self) -> None:
        assert _collapse_consecutive([]) == []

    def test_single(self) -> None:
        assert _collapse_consecutive(["a"]) == ["a"]


# ---------------------------------------------------------------------------
# compute_wpm
# ---------------------------------------------------------------------------


class TestComputeWpm:
    def test_basic_wpm(self) -> None:
        wpm, timed, total = compute_wpm([120, 120], [60.0, 60.0])
        assert wpm == 120.0
        assert timed == 2
        assert total == 2

    def test_below_minimum_threshold(self) -> None:
        wpm, timed, total = compute_wpm([30], [10.0])
        assert wpm is None
        assert timed == 1
        assert total == 1

    def test_skips_zero_duration(self) -> None:
        wpm, timed, total = compute_wpm([100, 50, 100], [60.0, 0.0, 60.0])
        assert wpm is not None
        assert timed == 2
        assert total == 3

    def test_skips_zero_word_count(self) -> None:
        wpm, timed, _total = compute_wpm([100, 0, 100], [60.0, 30.0, 60.0])
        assert wpm is not None
        assert timed == 2

    def test_empty_input(self) -> None:
        wpm, timed, total = compute_wpm([], [])
        assert wpm is None
        assert timed == 0
        assert total == 0

    def test_aggregate_not_mean(self) -> None:
        """WPM = total_words / total_seconds * 60, not mean of per-segment WPMs."""
        wpm, _, _ = compute_wpm([60, 180], [60.0, 60.0], min_timed_seconds=0)
        assert wpm == 120.0


# ---------------------------------------------------------------------------
# tokenize_text
# ---------------------------------------------------------------------------


class TestTokenizeText:
    def test_basic(self) -> None:
        counts = tokenize_text("the quick brown fox jumps over the lazy dog")
        assert counts["quick"] == 1
        assert counts["fox"] == 1
        assert "the" not in counts

    def test_empty(self) -> None:
        assert tokenize_text("") == Counter()
