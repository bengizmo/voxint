"""Behavioral tests for the dedup merge-planning logic.

These are pure-function tests: no database, no IO. They exercise
``plan_dedup_merges`` directly with synthetic ``DuplicatePair`` objects.
"""

from __future__ import annotations

import uuid

from voxint.speakers.dedup import plan_dedup_merges
from voxint.speakers.matching import DuplicatePair


def _pair(
    a: uuid.UUID, b: uuid.UUID, similarity: float
) -> DuplicatePair:
    return DuplicatePair(speaker_a_id=a, speaker_b_id=b, similarity=similarity)


# Fixed UUIDs for deterministic tests.
VOICE_1 = uuid.UUID("00000000-0000-0000-0000-000000000001")
VOICE_2 = uuid.UUID("00000000-0000-0000-0000-000000000002")
ALICE = uuid.UUID("00000000-0000-0000-0000-00000000000a")
BOB = uuid.UUID("00000000-0000-0000-0000-00000000000b")
CAROL = uuid.UUID("00000000-0000-0000-0000-00000000000c")


class TestPlaceholderDirection:
    def test_placeholder_to_real_name(self) -> None:
        pairs = [_pair(VOICE_1, ALICE, 0.92)]
        names = {VOICE_1: "Voice 1", ALICE: "Alice"}
        result = plan_dedup_merges(pairs, names, merge_threshold=0.85)
        assert len(result.merges) == 1
        assert result.merges[0].source_id == VOICE_1
        assert result.merges[0].target_id == ALICE

    def test_real_name_to_placeholder(self) -> None:
        pairs = [_pair(ALICE, VOICE_1, 0.92)]
        names = {ALICE: "Alice", VOICE_1: "Voice 1"}
        result = plan_dedup_merges(pairs, names, merge_threshold=0.85)
        assert len(result.merges) == 1
        assert result.merges[0].source_id == VOICE_1
        assert result.merges[0].target_id == ALICE


class TestAmbiguousDirection:
    def test_both_placeholders_skipped(self) -> None:
        pairs = [_pair(VOICE_1, VOICE_2, 0.92)]
        names = {VOICE_1: "Voice 1", VOICE_2: "Voice 2"}
        result = plan_dedup_merges(pairs, names, merge_threshold=0.85)
        assert len(result.merges) == 0
        assert len(result.skipped) == 1
        assert result.skipped[0].reason == "ambiguous_direction"

    def test_both_real_names_skipped(self) -> None:
        pairs = [_pair(ALICE, BOB, 0.92)]
        names = {ALICE: "Alice", BOB: "Bob"}
        result = plan_dedup_merges(pairs, names, merge_threshold=0.85)
        assert len(result.merges) == 0
        assert len(result.skipped) == 1
        assert result.skipped[0].reason == "ambiguous_direction"


class TestDisjointPairSafety:
    def test_second_pair_with_shared_speaker_skipped(self) -> None:
        pairs = [
            _pair(VOICE_1, ALICE, 0.95),
            _pair(VOICE_2, ALICE, 0.90),
        ]
        names = {VOICE_1: "Voice 1", VOICE_2: "Voice 2", ALICE: "Alice"}
        result = plan_dedup_merges(pairs, names, merge_threshold=0.85)
        assert len(result.merges) == 1
        assert result.merges[0].source_id == VOICE_1
        assert result.merges[0].target_id == ALICE
        assert len(result.skipped) == 1
        assert result.skipped[0].reason == "overlaps_prior"

    def test_source_overlap_also_caught(self) -> None:
        pairs = [
            _pair(VOICE_1, ALICE, 0.95),
            _pair(VOICE_1, BOB, 0.90),
        ]
        names = {VOICE_1: "Voice 1", ALICE: "Alice", BOB: "Bob"}
        result = plan_dedup_merges(pairs, names, merge_threshold=0.85)
        assert len(result.merges) == 1
        assert result.merges[0].target_id == ALICE
        assert len(result.skipped) == 1
        assert result.skipped[0].reason == "overlaps_prior"


class TestSimilarityOrdering:
    def test_higher_similarity_wins_contested_speaker(self) -> None:
        pairs = [
            _pair(VOICE_2, ALICE, 0.90),
            _pair(VOICE_1, ALICE, 0.95),
        ]
        names = {VOICE_1: "Voice 1", VOICE_2: "Voice 2", ALICE: "Alice"}
        result = plan_dedup_merges(pairs, names, merge_threshold=0.85)
        assert len(result.merges) == 1
        assert result.merges[0].source_id == VOICE_1
        assert result.merges[0].target_id == ALICE

    def test_equal_similarity_deterministic_via_uuid(self) -> None:
        pairs = [
            _pair(VOICE_2, BOB, 0.92),
            _pair(VOICE_1, ALICE, 0.92),
        ]
        names = {VOICE_1: "Voice 1", VOICE_2: "Voice 2", ALICE: "Alice", BOB: "Bob"}
        result = plan_dedup_merges(pairs, names, merge_threshold=0.85)
        assert len(result.merges) == 2
        first_source = result.merges[0].source_id
        second_source = result.merges[1].source_id
        assert {first_source, second_source} == {VOICE_1, VOICE_2}
        # Deterministic: VOICE_1 has a lower UUID so it sorts first
        assert result.merges[0].source_id == VOICE_1


class TestMergeThreshold:
    def test_below_threshold_excluded(self) -> None:
        pairs = [_pair(VOICE_1, ALICE, 0.80)]
        names = {VOICE_1: "Voice 1", ALICE: "Alice"}
        result = plan_dedup_merges(pairs, names, merge_threshold=0.85)
        assert len(result.merges) == 0
        assert len(result.skipped) == 0

    def test_at_threshold_included(self) -> None:
        pairs = [_pair(VOICE_1, ALICE, 0.85)]
        names = {VOICE_1: "Voice 1", ALICE: "Alice"}
        result = plan_dedup_merges(pairs, names, merge_threshold=0.85)
        assert len(result.merges) == 1

    def test_empty_pairs(self) -> None:
        result = plan_dedup_merges([], {}, merge_threshold=0.85)
        assert len(result.merges) == 0
        assert len(result.skipped) == 0


class TestPlaceholderRegex:
    """Boundary cases for the Voice N pattern."""

    def test_voice_with_multi_digit(self) -> None:
        sid = uuid.uuid4()
        pairs = [_pair(sid, ALICE, 0.92)]
        names = {sid: "Voice 10", ALICE: "Alice"}
        result = plan_dedup_merges(pairs, names, merge_threshold=0.85)
        assert result.merges[0].source_id == sid

    def test_lowercase_voice_not_placeholder(self) -> None:
        sid = uuid.uuid4()
        pairs = [_pair(sid, ALICE, 0.92)]
        names = {sid: "voice 1", ALICE: "Alice"}
        result = plan_dedup_merges(pairs, names, merge_threshold=0.85)
        assert len(result.merges) == 0
        assert result.skipped[0].reason == "ambiguous_direction"

    def test_voice_without_number_not_placeholder(self) -> None:
        sid = uuid.uuid4()
        pairs = [_pair(sid, ALICE, 0.92)]
        names = {sid: "Voice", ALICE: "Alice"}
        result = plan_dedup_merges(pairs, names, merge_threshold=0.85)
        assert len(result.merges) == 0
        assert result.skipped[0].reason == "ambiguous_direction"

    def test_voice_with_trailing_space_not_placeholder(self) -> None:
        sid = uuid.uuid4()
        pairs = [_pair(sid, ALICE, 0.92)]
        names = {sid: "Voice 1 ", ALICE: "Alice"}
        result = plan_dedup_merges(pairs, names, merge_threshold=0.85)
        assert len(result.merges) == 0
        assert result.skipped[0].reason == "ambiguous_direction"
