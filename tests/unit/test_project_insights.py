"""Unit tests for pure project-insight aggregation functions."""

from __future__ import annotations

import pytest

from voxint.api.project_insights import (
    AggregatedEntity,
    AggregatedTopic,
    _normalize,
    aggregate_entities,
    aggregate_topics,
)

# ---------------------------------------------------------------------------
# aggregate_entities
# ---------------------------------------------------------------------------


class TestAggregateEntities:
    def test_basic_aggregation_across_payloads(self) -> None:
        payloads = [
            {
                "mentions": [
                    {
                        "surface": "Acme Corp",
                        "kind": "ORG",
                        "occurrences": [
                            {"quote": "Acme launched the product."},
                            {"quote": "Acme hired more staff."},
                        ],
                    },
                    {
                        "surface": "Alice",
                        "kind": "PERSON",
                        "occurrences": [{"quote": "Alice spoke."}],
                    },
                ]
            },
            {
                "mentions": [
                    {
                        "surface": "Acme Corp",
                        "kind": "org",
                        "occurrences": [{"quote": "Acme expanded."}],
                    }
                ]
            },
        ]

        assert aggregate_entities(payloads) == [
            AggregatedEntity(
                surface="acme corp",
                display_surface="Acme Corp",
                kind="org",
                total_occurrences=3,
                runs_count=2,
                sample_quotes=[
                    "Acme launched the product.",
                    "Acme hired more staff.",
                    "Acme expanded.",
                ],
            ),
            AggregatedEntity(
                surface="alice",
                display_surface="Alice",
                kind="person",
                total_occurrences=1,
                runs_count=1,
                sample_quotes=["Alice spoke."],
            ),
        ]

    def test_unicode_normalization_merges_surface_forms(self) -> None:
        result = aggregate_entities(
            [
                {"mentions": [{"surface": "Jos\u00e9", "kind": "person", "occurrences": []}]},
                {
                    "mentions": [
                        {
                            "surface": "JOS\u00c9",
                            "kind": "PERSON",
                            "occurrences": [{"quote": "JOS\u00c9 arrived."}],
                        }
                    ]
                },
            ]
        )

        assert result == [
            AggregatedEntity(
                surface="jos\u00e9",
                display_surface="Jos\u00e9",
                kind="person",
                total_occurrences=2,
                runs_count=2,
                sample_quotes=["JOS\u00c9 arrived."],
            )
        ]

    def test_kind_conflict_resolved_by_majority_vote(self) -> None:
        result = aggregate_entities(
            [
                {
                    "mentions": [
                        {"surface": "Mercury", "kind": "place", "occurrences": []},
                        {"surface": "MERCURY", "kind": "ORG", "occurrences": []},
                    ]
                },
                {"mentions": [{"surface": "Mercury", "kind": " org ", "occurrences": []}]},
            ]
        )

        assert len(result) == 1
        assert result[0].kind == "org"

    def test_empty_and_whitespace_only_surfaces_are_skipped(self) -> None:
        result = aggregate_entities(
            [
                {
                    "mentions": [
                        {"surface": "", "kind": "person", "occurrences": []},
                        {"surface": "  \t\n ", "kind": "org", "occurrences": []},
                        {"kind": "place", "occurrences": []},
                        {"surface": "Valid", "kind": "other", "occurrences": []},
                    ]
                }
            ]
        )

        assert [entity.surface for entity in result] == ["valid"]

    def test_null_and_missing_kind_are_handled(self) -> None:
        result = aggregate_entities(
            [
                {
                    "mentions": [
                        {"surface": "Unknown One", "kind": None, "occurrences": []},
                        {"surface": "Unknown Two", "occurrences": []},
                    ]
                }
            ]
        )

        assert [entity.kind for entity in result] == [None, None]

    def test_quotes_are_deduplicated_trimmed_and_capped_at_five(self) -> None:
        result = aggregate_entities(
            [
                {
                    "mentions": [
                        {
                            "surface": "Acme",
                            "kind": "org",
                            "occurrences": [
                                {"quote": "  quote one  "},
                                {"quote": "quote one"},
                                {"quote": "quote two"},
                                {"quote": "quote three"},
                            ],
                        }
                    ]
                },
                {
                    "mentions": [
                        {
                            "surface": "ACME",
                            "kind": "org",
                            "occurrences": [
                                {"quote": "quote four"},
                                {"quote": "quote five"},
                                {"quote": "quote six"},
                                {"quote": "   "},
                            ],
                        }
                    ]
                },
            ]
        )

        assert result[0].sample_quotes == [
            "quote one",
            "quote two",
            "quote three",
            "quote four",
            "quote five",
        ]

    def test_sort_order(self) -> None:
        result = aggregate_entities(
            [
                {
                    "mentions": [
                        {"surface": "Alpha", "occurrences": [{}, {}, {}]},
                        {"surface": "Zeta", "occurrences": [{}]},
                        {"surface": "Gamma", "occurrences": [{}, {}]},
                        {"surface": "Beta", "occurrences": [{}, {}]},
                    ]
                },
                {"mentions": [{"surface": "ZETA", "occurrences": [{}]}]},
            ]
        )

        assert [entity.surface for entity in result] == [
            "alpha",
            "zeta",
            "beta",
            "gamma",
        ]

    def test_result_is_capped_at_fifty(self) -> None:
        result = aggregate_entities(
            [
                {
                    "mentions": [
                        {"surface": f"Entity {index:02d}", "occurrences": []} for index in range(55)
                    ]
                }
            ]
        )

        assert len(result) == 50
        assert [entity.surface for entity in result] == [
            f"entity {index:02d}" for index in range(50)
        ]

    def test_empty_payloads_returns_empty(self) -> None:
        assert aggregate_entities([]) == []

    def test_empty_mentions_list_returns_empty(self) -> None:
        assert aggregate_entities([{"mentions": []}]) == []


# ---------------------------------------------------------------------------
# aggregate_topics
# ---------------------------------------------------------------------------


class TestAggregateTopics:
    def test_basic_aggregation(self) -> None:
        result = aggregate_topics(
            [
                {
                    "topics": [
                        {
                            "label": "Climate",
                            "description": "Climate policy",
                            "confidence": 0.8,
                        },
                        {"label": "Energy", "description": None, "confidence": 0.5},
                    ]
                },
                {
                    "topics": [
                        {
                            "label": "Climate",
                            "description": "Later description",
                            "confidence": 1.0,
                        }
                    ]
                },
            ]
        )

        assert result == [
            AggregatedTopic(
                label="climate",
                display_label="Climate",
                description="Climate policy",
                avg_confidence=pytest.approx(0.9),
                confidence_count=2,
                runs_count=2,
            ),
            AggregatedTopic(
                label="energy",
                display_label="Energy",
                description=None,
                avg_confidence=0.5,
                confidence_count=1,
                runs_count=1,
            ),
        ]

    def test_label_normalization_merges_across_payloads(self) -> None:
        result = aggregate_topics(
            [
                {"topics": [{"label": "  Fullwidth \uff21\uff29  ", "confidence": 0.4}]},
                {"topics": [{"label": "fullwidth ai", "confidence": 0.6}]},
            ]
        )

        assert len(result) == 1
        assert result[0].label == "fullwidth ai"
        assert result[0].display_label == "Fullwidth \uff21\uff29"
        assert result[0].runs_count == 2

    def test_null_confidences_are_excluded_from_average(self) -> None:
        result = aggregate_topics(
            [
                {"topics": [{"label": "Science", "confidence": None}]},
                {"topics": [{"label": "SCIENCE", "confidence": 0.75}]},
                {"topics": [{"label": "science"}]},
            ]
        )

        assert result[0].avg_confidence == 0.75
        assert result[0].confidence_count == 1
        assert result[0].runs_count == 3

    def test_all_null_confidences_produce_none_average(self) -> None:
        result = aggregate_topics(
            [
                {"topics": [{"label": "Science", "confidence": None}]},
                {"topics": [{"label": "SCIENCE"}]},
            ]
        )

        assert result[0].avg_confidence is None
        assert result[0].confidence_count == 0

    def test_first_non_null_description_is_kept(self) -> None:
        result = aggregate_topics(
            [
                {"topics": [{"label": "Robotics", "description": None}]},
                {"topics": [{"label": "ROBOTICS", "description": " First description "}]},
                {"topics": [{"label": "robotics", "description": "Later description"}]},
            ]
        )

        assert result[0].description == "First description"

    def test_sort_order(self) -> None:
        result = aggregate_topics(
            [
                {
                    "topics": [
                        {"label": "Solo", "confidence": 1.0},
                        {"label": "Low", "confidence": 0.2},
                        {"label": "Zulu", "confidence": 0.8},
                        {"label": "Alpha", "confidence": 0.8},
                    ]
                },
                {
                    "topics": [
                        {"label": "LOW", "confidence": 0.2},
                        {"label": "ZULU", "confidence": 0.8},
                        {"label": "ALPHA", "confidence": 0.8},
                    ]
                },
            ]
        )

        assert [topic.label for topic in result] == [
            "alpha",
            "zulu",
            "low",
            "solo",
        ]

    def test_result_is_capped_at_thirty(self) -> None:
        result = aggregate_topics(
            [
                {
                    "topics": [
                        {"label": f"Topic {index:02d}", "confidence": 0.5} for index in range(35)
                    ]
                }
            ]
        )

        assert len(result) == 30
        assert [topic.label for topic in result] == [f"topic {index:02d}" for index in range(30)]

    def test_empty_payloads_returns_empty(self) -> None:
        assert aggregate_topics([]) == []


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_casefolds_unicode_text(self) -> None:
        assert _normalize("Jos\u00e9 JOS\u00c9") == "jos\u00e9 jos\u00e9"

    def test_nfkc_normalizes_compatibility_characters(self) -> None:
        text = "\uff26\uff55\uff4c\uff4c\uff57\uff49\uff44\uff54\uff48 \u212a"
        assert _normalize(text) == "fullwidth k"

    def test_collapses_and_trims_whitespace(self) -> None:
        assert _normalize("  Alpha\t beta\n\u2003gamma  ") == "alpha beta gamma"

    def test_empty_and_whitespace_only_text(self) -> None:
        assert _normalize("") == ""
        assert _normalize(" \t\n ") == ""
