"""Unit tests for project entity/topic rollups (issue #336)."""

from __future__ import annotations

import uuid
from typing import Any

from voxint.api.project_insights import aggregate_entities, aggregate_topics


def _run() -> uuid.UUID:
    return uuid.uuid4()


def _mention(
    surface: str,
    kind: str | None = "person",
    occurrences: object = None,
) -> dict[str, Any]:
    if occurrences is None:
        occurrences = [{}]
    return {"surface": surface, "kind": kind, "occurrences": occurrences}


class TestAggregateEntities:
    def test_strip_casefold_merge_but_internal_whitespace_is_preserved(self) -> None:
        run = _run()
        result = aggregate_entities(
            [(run, [_mention(" Acme ", "organization"), _mention("ACME", "organization"),
                    _mention("Ac  me", "organization")])]
        )
        assert [item.label for item in result["organization"]] == ["ACME", "Ac  me"]
        assert result["organization"][0].run_count == 1
        assert result["organization"][0].occurrence_count == 2

    def test_display_casing_uses_plurality_then_lexical_ties(self) -> None:
        result = aggregate_entities(
            [
                (_run(), [_mention("acme"), _mention("Acme")]),
                (_run(), [_mention("Acme")]),
            ]
        )
        assert result["person"][0].label == "Acme"

        tied = aggregate_entities([(_run(), [_mention("BETA"), _mention("beta")])])
        assert tied["person"][0].label == "BETA"

    def test_rank_is_runs_then_occurrences_then_label(self) -> None:
        run_a, run_b = _run(), _run()
        result = aggregate_entities(
            [
                (run_a, [_mention("Zulu", occurrences=[{}, {}, {}]), _mention("Alpha")]),
                (run_b, [_mention("Alpha")]),
            ]
        )["person"]
        assert [item.label for item in result] == ["Alpha", "Zulu"]

        lexical = aggregate_entities([(_run(), [_mention("beta"), _mention("Alpha")])])
        assert [item.label for item in lexical["person"]] == ["Alpha", "beta"]

    def test_kind_majority_votes_by_run_with_priority_tie_break(self) -> None:
        run_a, run_b, run_c = _run(), _run(), _run()
        majority = aggregate_entities(
            [
                (run_a, [_mention("Acme", "organization")]),
                (run_b, [_mention("acme", "organization")]),
                (run_c, [_mention("ACME", "person")]),
            ]
        )
        assert majority["organization"][0].label == "ACME"

        tied = aggregate_entities(
            [(run_a, [_mention("X", "product")]), (run_b, [_mention("x", "person")])]
        )
        assert tied["person"][0].kind == "person"

    def test_null_and_unknown_kinds_go_to_other(self) -> None:
        result = aggregate_entities(
            [(_run(), [_mention("Unknown", None), _mention("Mystery", "location")])]
        )
        assert [item.label for item in result["other"]] == ["Mystery", "Unknown"]

    def test_caps_each_kind_at_twelve(self) -> None:
        result = aggregate_entities(
            [(_run(), [_mention(f"Person {index:02}") for index in range(15)])]
        )
        assert len(result["person"]) == 12
        assert [item.label for item in result["person"]] == [
            f"Person {index:02}" for index in range(12)
        ]

    def test_occurrence_count_and_malformed_entries_are_defensive(self) -> None:
        result = aggregate_entities(
            [
                (
                    _run(),
                    [
                        "not a dict",
                        {"kind": "person", "occurrences": [{}]},
                        {"surface": 123, "occurrences": [{}]},
                        _mention("Acme", occurrences=[{}, {}, {}]),
                        {"surface": "acme", "kind": "person"},
                        _mention("ACME", occurrences="bad"),
                    ],
                )
            ]
        )
        item = result["person"][0]
        assert item.occurrence_count == 5
        assert item.run_count == 1

    def test_empty_input_has_all_ordered_groups(self) -> None:
        assert aggregate_entities([]) == {
            "person": [],
            "organization": [],
            "product": [],
            "other": [],
        }


class TestAggregateTopics:
    def test_casefold_dedupes_and_counts_distinct_runs(self) -> None:
        run_a, run_b = _run(), _run()
        result = aggregate_topics(
            [
                (run_a, [{"label": " Climate ", "confidence": 0.2}, {"label": "CLIMATE"}]),
                (run_b, [{"label": "climate", "confidence": 0.5}]),
            ]
        )
        assert result[0].label == "CLIMATE"
        assert result[0].run_count == 2

    def test_description_comes_from_highest_confidence_and_ties_keep_first(self) -> None:
        result = aggregate_topics(
            [
                (_run(), [{"label": "Heat", "description": "low", "confidence": 0.2}]),
                (_run(), [{"label": "heat", "description": "high", "confidence": 0.9}]),
                (_run(), [{"label": "HEAT", "description": "tied", "confidence": 0.9}]),
            ]
        )
        assert result[0].description == "high"

    def test_confidence_tie_prefers_a_real_description_over_none(self) -> None:
        result = aggregate_topics(
            [
                (_run(), [{"label": "Heat", "confidence": 0.9}]),
                (_run(), [{"label": "heat", "description": "present", "confidence": 0.9}]),
            ]
        )
        assert result[0].description == "present"

    def test_none_confidence_sorts_lowest_and_first_none_wins_tie(self) -> None:
        result = aggregate_topics(
            [
                (_run(), [{"label": "Heat", "description": "first", "confidence": None}]),
                (_run(), [{"label": "heat", "description": "second"}]),
            ]
        )
        assert result[0].description == "first"

    def test_rank_and_top_ten_cap(self) -> None:
        run_a, run_b = _run(), _run()
        topics = [{"label": f"Topic {index:02}"} for index in range(12)]
        result = aggregate_topics(
            [(run_a, topics), (run_b, [{"label": "Topic 11"}])]
        )
        assert len(result) == 10
        assert result[0].label == "Topic 11"
        assert [item.label for item in result[1:]] == [
            f"Topic {index:02}" for index in range(9)
        ]

    def test_malformed_and_empty_inputs(self) -> None:
        assert aggregate_topics([]) == []
        assert aggregate_topics(
            [(_run(), ["bad", {"description": "missing"}, {"label": "   "}])]
        ) == []
