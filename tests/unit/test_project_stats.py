from __future__ import annotations

import uuid

from voxint.api.project_stats import (
    CoverageCell,
    CoverageItem,
    CoverageMatrix,
    EntityStat,
    SpeakerPresence,
    TopicStat,
    build_project_insights_payload,
    compute_coverage_matrix,
    compute_entity_stats,
    compute_topic_stats,
)


class TestComputeEntityStats:
    def setup_method(self) -> None:
        self.run_a = str(uuid.uuid4())
        self.run_b = str(uuid.uuid4())
        self.run_c = str(uuid.uuid4())

    def test_basic_aggregation(self) -> None:
        enrichments = [
            (self.run_a, {"mentions": [
                {"surface": "Alice", "kind": "person", "occurrences": [{"start": 0}]},
                {"surface": "Bob", "kind": "person", "occurrences": [{"start": 10}]},
            ]}),
            (self.run_b, {"mentions": [
                {"surface": "Alice", "kind": "person",
                 "occurrences": [{"start": 5}, {"start": 20}]},
            ]}),
        ]
        result = compute_entity_stats(enrichments)
        by_key = {e.key: e for e in result}
        assert by_key["alice"].run_count == 2
        assert by_key["alice"].occurrence_count == 3
        assert by_key["bob"].run_count == 1
        assert by_key["bob"].occurrence_count == 1

    def test_casefold_normalization(self) -> None:
        enrichments = [
            (self.run_a, {"mentions": [
                {"surface": "John Smith", "kind": "person", "occurrences": [{}]},
            ]}),
            (self.run_b, {"mentions": [
                {"surface": "john smith", "kind": "person", "occurrences": [{}]},
            ]}),
        ]
        result = compute_entity_stats(enrichments)
        assert len(result) == 1
        assert result[0].run_count == 2

    def test_whitespace_normalization(self) -> None:
        enrichments = [
            (self.run_a, {"mentions": [
                {"surface": "John  Smith", "kind": "person", "occurrences": [{}]},
            ]}),
            (self.run_b, {"mentions": [
                {"surface": "John Smith", "kind": "person", "occurrences": [{}]},
            ]}),
        ]
        result = compute_entity_stats(enrichments)
        assert len(result) == 1
        assert result[0].key == "john smith"

    def test_kind_separation(self) -> None:
        enrichments = [
            (self.run_a, {"mentions": [
                {"surface": "Jordan", "kind": "person", "occurrences": [{}]},
                {"surface": "Jordan", "kind": "organization", "occurrences": [{}]},
            ]}),
        ]
        result = compute_entity_stats(enrichments)
        assert len(result) == 2

    def test_display_surface_most_frequent(self) -> None:
        enrichments = [
            (self.run_a, {"mentions": [
                {"surface": "John Smith", "kind": "person", "occurrences": [{}]},
                {"surface": "John Smith", "kind": "person", "occurrences": [{}]},
                {"surface": "John Smith", "kind": "person", "occurrences": [{}]},
            ]}),
            (self.run_b, {"mentions": [
                {"surface": "john smith", "kind": "person", "occurrences": [{}]},
            ]}),
        ]
        result = compute_entity_stats(enrichments)
        assert result[0].display_surface == "John Smith"

    def test_display_surface_longest_on_tie(self) -> None:
        enrichments = [
            (self.run_a, {"mentions": [
                {"surface": "Smith", "kind": "person", "occurrences": [{}]},
            ]}),
            (self.run_b, {"mentions": [
                {"surface": "smith", "kind": "person", "occurrences": [{}]},
            ]}),
        ]
        result = compute_entity_stats(enrichments)
        assert result[0].display_surface == "Smith"

    def test_sort_order(self) -> None:
        enrichments = [
            (self.run_a, {"mentions": [
                {"surface": "Alpha", "kind": "person", "occurrences": [{}]},
                {"surface": "Beta", "kind": "person", "occurrences": [{}, {}]},
                {"surface": "Gamma", "kind": "person", "occurrences": [{}]},
            ]}),
            (self.run_b, {"mentions": [
                {"surface": "Alpha", "kind": "person", "occurrences": [{}]},
                {"surface": "Gamma", "kind": "person", "occurrences": [{}]},
            ]}),
        ]
        result = compute_entity_stats(enrichments)
        keys = [e.key for e in result]
        assert keys == ["alpha", "gamma", "beta"]

    def test_cap_at_50(self) -> None:
        enrichments = [
            (self.run_a, {"mentions": [
                {"surface": f"Entity{i}", "kind": "person", "occurrences": [{}]}
                for i in range(60)
            ]}),
        ]
        result = compute_entity_stats(enrichments)
        assert len(result) == 50

    def test_empty_input(self) -> None:
        assert compute_entity_stats([]) == []

    def test_empty_mentions(self) -> None:
        enrichments = [(self.run_a, {"mentions": []})]
        assert compute_entity_stats(enrichments) == []

    def test_null_kind(self) -> None:
        enrichments = [
            (self.run_a, {"mentions": [
                {"surface": "Something", "kind": None, "occurrences": [{}]},
            ]}),
        ]
        result = compute_entity_stats(enrichments)
        assert len(result) == 1
        assert result[0].kind is None

    def test_occurrence_count_from_occurrences_list(self) -> None:
        enrichments = [
            (self.run_a, {"mentions": [
                {"surface": "Alice", "kind": "person", "occurrences": [
                    {"s": 0}, {"s": 5}, {"s": 10},
                ]},
            ]}),
        ]
        result = compute_entity_stats(enrichments)
        assert result[0].occurrence_count == 3


class TestComputeTopicStats:
    def setup_method(self) -> None:
        self.run_a = str(uuid.uuid4())
        self.run_b = str(uuid.uuid4())
        self.run_c = str(uuid.uuid4())

    def test_basic_aggregation(self) -> None:
        enrichments = [
            (self.run_a, {"topics": [
                {"label": "Budget", "description": "About budgets", "confidence": 0.9},
            ]}),
            (self.run_b, {"topics": [
                {"label": "Budget", "description": "Budget discussion", "confidence": 0.8},
                {"label": "Hiring", "description": "Staff hiring", "confidence": 0.7},
            ]}),
        ]
        result = compute_topic_stats(enrichments)
        by_key = {t.key: t for t in result}
        assert by_key["budget"].run_count == 2
        assert by_key["hiring"].run_count == 1

    def test_within_run_dedup(self) -> None:
        enrichments = [
            (self.run_a, {"topics": [
                {"label": "Budget", "confidence": 0.9},
                {"label": "Budget", "confidence": 0.8},
            ]}),
        ]
        result = compute_topic_stats(enrichments)
        assert result[0].run_count == 1

    def test_cross_run_counting(self) -> None:
        enrichments = [
            (self.run_a, {"topics": [{"label": "Budget", "confidence": 0.9}]}),
            (self.run_b, {"topics": [{"label": "Budget", "confidence": 0.8}]}),
            (self.run_c, {"topics": [{"label": "Budget", "confidence": 0.7}]}),
        ]
        result = compute_topic_stats(enrichments)
        assert result[0].run_count == 3

    def test_confidence_mean(self) -> None:
        enrichments = [
            (self.run_a, {"topics": [{"label": "Budget", "confidence": 0.9}]}),
            (self.run_b, {"topics": [{"label": "Budget", "confidence": 0.7}]}),
        ]
        result = compute_topic_stats(enrichments)
        assert result[0].confidence_mean == 0.8

    def test_confidence_max(self) -> None:
        enrichments = [
            (self.run_a, {"topics": [{"label": "Budget", "confidence": 0.6}]}),
            (self.run_b, {"topics": [{"label": "Budget", "confidence": 0.9}]}),
        ]
        result = compute_topic_stats(enrichments)
        assert result[0].confidence_max == 0.9

    def test_description_from_highest_confidence(self) -> None:
        enrichments = [
            (self.run_a, {"topics": [
                {"label": "Budget", "description": "low conf", "confidence": 0.3},
            ]}),
            (self.run_b, {"topics": [
                {"label": "Budget", "description": "high conf", "confidence": 0.9},
            ]}),
        ]
        result = compute_topic_stats(enrichments)
        assert result[0].description == "high conf"

    def test_description_fallback_no_confidence(self) -> None:
        enrichments = [
            (self.run_a, {"topics": [
                {"label": "Budget", "description": "fallback desc"},
            ]}),
        ]
        result = compute_topic_stats(enrichments)
        assert result[0].description == "fallback desc"

    def test_casefold_label(self) -> None:
        enrichments = [
            (self.run_a, {"topics": [
                {"label": "Budget Planning", "confidence": 0.9},
            ]}),
            (self.run_b, {"topics": [
                {"label": "budget planning", "confidence": 0.8},
            ]}),
        ]
        result = compute_topic_stats(enrichments)
        assert len(result) == 1
        assert result[0].run_count == 2

    def test_sort_order(self) -> None:
        enrichments = [
            (self.run_a, {"topics": [
                {"label": "Zebra", "confidence": 0.5},
                {"label": "Alpha", "confidence": 0.5},
            ]}),
            (self.run_b, {"topics": [
                {"label": "Alpha", "confidence": 0.5},
            ]}),
        ]
        result = compute_topic_stats(enrichments)
        assert [t.label for t in result] == ["Alpha", "Zebra"]

    def test_cap_at_30(self) -> None:
        enrichments = [
            (self.run_a, {"topics": [
                {"label": f"Topic{i}", "confidence": 0.5}
                for i in range(40)
            ]}),
        ]
        result = compute_topic_stats(enrichments)
        assert len(result) == 30

    def test_empty_input(self) -> None:
        assert compute_topic_stats([]) == []


class TestComputeCoverageMatrix:
    def setup_method(self) -> None:
        self.spk_a = str(uuid.uuid4())
        self.spk_b = str(uuid.uuid4())
        self.run_a = str(uuid.uuid4())
        self.run_b = str(uuid.uuid4())
        self.media_a = str(uuid.uuid4())
        self.media_b = str(uuid.uuid4())

    def test_basic_matrix(self) -> None:
        items = [
            CoverageItem(
                media_item_id=self.media_a, run_id=self.run_a, title="Rec A",
                duration_s=120, speakers=[
                    SpeakerPresence(self.spk_a, "Alice", segment_count=5),
                    SpeakerPresence(self.spk_b, "Bob", segment_count=3),
                ],
            ),
            CoverageItem(
                media_item_id=self.media_b, run_id=self.run_b, title="Rec B",
                duration_s=60, speakers=[
                    SpeakerPresence(self.spk_a, "Alice", segment_count=2),
                ],
            ),
        ]
        result = compute_coverage_matrix(items)
        assert len(result.speakers) == 2
        assert len(result.recordings) == 2
        assert len(result.cells) == 3

    def test_speaker_ordering(self) -> None:
        items = [
            CoverageItem(
                media_item_id=self.media_a, run_id=self.run_a, title="Rec",
                duration_s=120, speakers=[
                    SpeakerPresence(self.spk_a, "Zoe", segment_count=3),
                    SpeakerPresence(self.spk_b, "Alice", segment_count=10),
                ],
            ),
        ]
        result = compute_coverage_matrix(items)
        assert result.speakers[0]["label"] == "Alice"
        assert result.speakers[1]["label"] == "Zoe"

    def test_recording_ordering(self) -> None:
        items = [
            CoverageItem(
                media_item_id=self.media_a, run_id=self.run_a, title="First",
                duration_s=120, speakers=[],
            ),
            CoverageItem(
                media_item_id=self.media_b, run_id=self.run_b, title="Second",
                duration_s=60, speakers=[],
            ),
        ]
        result = compute_coverage_matrix(items)
        assert result.recordings[0]["title"] == "First"
        assert result.recordings[1]["title"] == "Second"

    def test_zero_segment_excluded(self) -> None:
        items = [
            CoverageItem(
                media_item_id=self.media_a, run_id=self.run_a, title="Rec",
                duration_s=120, speakers=[
                    SpeakerPresence(self.spk_a, "Alice", segment_count=0),
                    SpeakerPresence(self.spk_b, "Bob", segment_count=5),
                ],
            ),
        ]
        result = compute_coverage_matrix(items)
        assert len(result.cells) == 1
        assert result.cells[0].segment_count == 5

    def test_empty_input(self) -> None:
        result = compute_coverage_matrix([])
        assert result.speakers == []
        assert result.recordings == []
        assert result.cells == []
        assert result.stats["speaker_count"] == 0
        assert result.stats["recording_count"] == 0

    def test_single_recording_single_speaker(self) -> None:
        items = [
            CoverageItem(
                media_item_id=self.media_a, run_id=self.run_a, title="Only",
                duration_s=60, speakers=[
                    SpeakerPresence(self.spk_a, "Solo", segment_count=7),
                ],
            ),
        ]
        result = compute_coverage_matrix(items)
        assert len(result.speakers) == 1
        assert len(result.recordings) == 1
        assert len(result.cells) == 1
        assert result.cells[0].segment_count == 7

    def test_stats(self) -> None:
        items = [
            CoverageItem(
                media_item_id=self.media_a, run_id=self.run_a, title="Rec A",
                duration_s=120, speakers=[
                    SpeakerPresence(self.spk_a, "Alice", segment_count=5),
                ],
            ),
            CoverageItem(
                media_item_id=self.media_b, run_id=self.run_b, title="Rec B",
                duration_s=60, speakers=[],
            ),
        ]
        result = compute_coverage_matrix(items)
        assert result.stats["speaker_count"] == 1
        assert result.stats["recording_count"] == 2
        assert result.stats["recordings_with_speakers"] == 1
        assert result.stats["covered_cells"] == 1


class TestBuildPayload:
    def setup_method(self) -> None:
        self.run_a = str(uuid.uuid4())

    def test_payload_structure(self) -> None:
        entities = [
            EntityStat(
                key="alice", kind="person", display_surface="Alice",
                run_count=2, occurrence_count=5,
            ),
        ]
        topics = [
            TopicStat(
                key="budget", label="Budget", description="About budgets",
                run_count=3, confidence_mean=0.85, confidence_max=0.95,
            ),
        ]
        coverage = CoverageMatrix(
            speakers=[{"id": "s1", "label": "Speaker 1", "run_count": 1}],
            recordings=[{
                "media_item_id": "m1", "run_id": self.run_a,
                "title": "Rec", "duration_s": 60,
            }],
            cells=[CoverageCell(speaker_idx=0, recording_idx=0, segment_count=10)],
            stats={
                "speaker_count": 1, "recording_count": 1,
                "recordings_with_speakers": 1, "covered_cells": 1,
            },
        )
        payload = build_project_insights_payload(
            entities, topics, coverage,
            run_count=5, runs_with_entities=3, runs_with_topics=4,
        )
        assert payload["version"] == 1
        assert payload["stats"]["run_count"] == 5
        assert payload["stats"]["runs_with_entities"] == 3
        assert payload["stats"]["runs_with_topics"] == 4
        assert payload["stats"]["runs_with_speakers"] == 1
        assert payload["stats"]["speaker_count"] == 1
        assert payload["stats"]["media_item_count"] == 1
        assert len(payload["entities"]) == 1
        assert payload["entities"][0]["key"] == "alice"
        assert payload["entities"][0]["occurrence_count"] == 5
        assert len(payload["topics"]) == 1
        assert payload["topics"][0]["confidence_mean"] == 0.85
        assert len(payload["coverage"]["cells"]) == 1
        assert payload["coverage"]["cells"][0]["segment_count"] == 10
