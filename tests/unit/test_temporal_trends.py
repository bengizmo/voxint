"""Unit tests for pure temporal trend aggregation (issue #337)."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, date, datetime, timedelta, timezone
from typing import Any

import pytest

from voxint.api.temporal_trends import (
    ALGORITHM_VERSION,
    MAX_BUCKETS,
    MAX_ENTITIES,
    MAX_TERMS,
    SCHEMA_VERSION,
    RecordingInput,
    build_temporal_trends,
    count_entity_frequencies,
    count_term_frequencies,
    display_mode_for,
    generate_buckets,
    resolve_date,
    select_bucket_unit,
)


def recording(
    *,
    when: date,
    text: str = "",
    mentions: Any = None,
    upload_date: date | None = None,
    run_id: uuid.UUID | None = None,
) -> RecordingInput:
    return {
        "run_id": run_id or uuid.uuid4(),
        "media_id": uuid.uuid4(),
        "upload_date": upload_date,
        "created_at": datetime.combine(when, datetime.min.time(), tzinfo=UTC),
        "effective_text": text,
        "entity_mentions": mentions,
    }


class TestDateResolution:
    def test_source_upload_date_wins(self) -> None:
        assert resolve_date(date(2025, 1, 2), datetime(2026, 3, 4, tzinfo=UTC)) == (
            date(2025, 1, 2),
            "source_upload_date",
        )

    def test_aware_ingestion_timestamp_is_converted_to_utc(self) -> None:
        local = datetime(2026, 1, 2, 1, 30, tzinfo=timezone(timedelta(hours=3)))
        assert resolve_date(None, local) == (
            date(2026, 1, 1),
            "ingestion_created_at",
        )

    def test_naive_ingestion_timestamp_is_treated_as_utc(self) -> None:
        assert resolve_date(None, datetime(2026, 1, 2, 1, 30)) == (
            date(2026, 1, 2),
            "ingestion_created_at",
        )


class TestBucketSelection:
    @pytest.mark.parametrize(
        ("start", "end", "expected"),
        [
            (date(2026, 1, 1), date(2026, 3, 1), "day"),  # exactly 60 days
            (date(2026, 1, 1), date(2026, 3, 2), "week"),
            (date(2026, 1, 1), date(2027, 1, 1), "week"),  # 53 aligned weeks
            (date(2026, 1, 1), date(2027, 3, 1), "month"),
            (date(2000, 1, 1), date(2026, 1, 1), "month"),
        ],
    )
    def test_selects_finest_unit_within_target(self, start: date, end: date, expected: str) -> None:
        assert select_bucket_unit(start, end) == expected

    def test_rejects_reverse_range(self) -> None:
        with pytest.raises(ValueError, match="min_date"):
            select_bucket_unit(date(2026, 2, 1), date(2026, 1, 1))


class TestBucketGeneration:
    def test_day_buckets_include_leap_day(self) -> None:
        buckets = generate_buckets(date(2024, 2, 28), date(2024, 3, 1), "day")
        assert [bucket["start"] for bucket in buckets] == [
            "2024-02-28",
            "2024-02-29",
            "2024-03-01",
        ]
        assert buckets[-1]["end_exclusive"] == "2024-03-02"

    def test_week_buckets_start_on_monday_and_cross_year(self) -> None:
        buckets = generate_buckets(date(2025, 12, 31), date(2026, 1, 7), "week")
        assert [(bucket["start"], bucket["end_exclusive"]) for bucket in buckets] == [
            ("2025-12-29", "2026-01-05"),
            ("2026-01-05", "2026-01-12"),
        ]

    def test_month_buckets_use_calendar_boundaries(self) -> None:
        buckets = generate_buckets(date(2024, 1, 31), date(2024, 3, 2), "month")
        assert [(bucket["start"], bucket["end_exclusive"]) for bucket in buckets] == [
            ("2024-01-01", "2024-02-01"),
            ("2024-02-01", "2024-03-01"),
            ("2024-03-01", "2024-04-01"),
        ]

    def test_generated_metadata_is_zero_filled(self) -> None:
        bucket = generate_buckets(date(2026, 1, 1), date(2026, 1, 1), "day")[0]
        assert bucket["recording_count"] == 0
        assert bucket["date_sources"] == {
            "source_upload_date": 0,
            "ingestion_created_at": 0,
        }

    def test_rejects_unknown_unit(self) -> None:
        with pytest.raises(ValueError, match="unsupported bucket unit"):
            generate_buckets(date(2026, 1, 1), date(2026, 1, 2), "year")


class TestTermFrequencies:
    def test_uses_shared_tokenizer_and_dense_bucket_alignment(self) -> None:
        recordings = [
            recording(when=date(2026, 1, 1), text="Climate climate and policy"),
            recording(when=date(2026, 1, 3), text="policy energy"),
        ]
        buckets = generate_buckets(date(2026, 1, 1), date(2026, 1, 3), "day")

        terms = count_term_frequencies(recordings, buckets)

        assert terms == [
            {
                "key": "climate",
                "label": "climate",
                "total_count": 2,
                "recording_count": 1,
                "values": [2, 0, 0],
            },
            {
                "key": "policy",
                "label": "policy",
                "total_count": 2,
                "recording_count": 2,
                "values": [1, 0, 1],
            },
            {
                "key": "energy",
                "label": "energy",
                "total_count": 1,
                "recording_count": 1,
                "values": [0, 0, 1],
            },
        ]

    def test_ties_sort_alphabetically_and_results_are_capped(self) -> None:
        text = " ".join(f"term{chr(97 + index)}" for index in range(26))
        terms = count_term_frequencies(
            [recording(when=date(2026, 1, 1), text=text)],
            generate_buckets(date(2026, 1, 1), date(2026, 1, 1), "day"),
        )

        assert len(terms) == MAX_TERMS
        assert [term["key"] for term in terms] == [
            f"term{chr(97 + index)}" for index in range(MAX_TERMS)
        ]


class TestEntityFrequencies:
    def test_normalizes_nfkc_case_and_whitespace(self) -> None:
        recordings = [
            recording(
                when=date(2026, 1, 1),
                mentions={
                    "mentions": [
                        {
                            "surface": "  ACME\u00a0Corp ",
                            "kind": "ORG",
                            "occurrences": [{}, {}],
                        }
                    ]
                },
            ),
            recording(
                when=date(2026, 1, 2),
                mentions={"mentions": [{"surface": "acme corp", "kind": "org", "occurrences": []}]},
            ),
        ]
        buckets = generate_buckets(date(2026, 1, 1), date(2026, 1, 2), "day")

        assert count_entity_frequencies(recordings, buckets) == [
            {
                "key": "acme corp",
                "label": "ACME\u00a0Corp",
                "kind": "org",
                "total_count": 3,
                "recording_count": 2,
                "values": [2, 1],
            }
        ]

    @pytest.mark.parametrize(
        "mentions",
        [None, {}, {"mentions": None}, {"mentions": "bad"}, {"mentions": [None, 1]}],
    )
    def test_malformed_or_empty_payloads_are_ignored(self, mentions: Any) -> None:
        rec = recording(when=date(2026, 1, 1), mentions=mentions)
        buckets = generate_buckets(date(2026, 1, 1), date(2026, 1, 1), "day")
        assert count_entity_frequencies([rec], buckets) == []

    def test_malformed_mentions_are_skipped_and_missing_occurrences_count_once(self) -> None:
        rec = recording(
            when=date(2026, 1, 1),
            mentions={
                "mentions": [
                    {},
                    {"surface": None},
                    {"surface": "   "},
                    {"surface": "Valid", "occurrences": "bad"},
                ]
            },
        )
        buckets = generate_buckets(date(2026, 1, 1), date(2026, 1, 1), "day")
        assert count_entity_frequencies([rec], buckets)[0]["total_count"] == 1

    def test_ties_sort_alphabetically_and_results_are_capped(self) -> None:
        mentions = {
            "mentions": [
                {"surface": f"Entity {index:02d}", "occurrences": []} for index in range(25)
            ]
        }
        entities = count_entity_frequencies(
            [recording(when=date(2026, 1, 1), mentions=mentions)],
            generate_buckets(date(2026, 1, 1), date(2026, 1, 1), "day"),
        )

        assert len(entities) == MAX_ENTITIES
        assert [entity["key"] for entity in entities] == [
            f"entity {index:02d}" for index in range(MAX_ENTITIES)
        ]


class TestBuildTemporalTrends:
    def test_empty_corpus(self) -> None:
        payload = build_temporal_trends([])
        assert payload["schema_version"] == SCHEMA_VERSION
        assert payload["algorithm_version"] == ALGORITHM_VERSION
        assert payload["range"]["start"] is None
        assert payload["range"]["bucket_unit"] is None
        assert payload["buckets"] == []
        assert payload["terms"] == []
        assert payload["entities"] == []
        assert payload["display_mode"] == "empty"

    def test_single_date_corpus_and_provenance(self) -> None:
        records = [
            recording(
                when=date(2026, 5, 10),
                upload_date=date(2020, 5, 10),
                text="archive",
            ),
            recording(when=date(2020, 5, 10), text="archive"),
        ]

        payload = build_temporal_trends(records)

        assert payload["range"] == {
            "start": "2020-05-10",
            "end": "2020-05-10",
            "bucket_unit": "day",
            "week_starts_on": "monday",
            "timezone": "UTC",
        }
        assert len(payload["buckets"]) == 1
        assert payload["buckets"][0]["recording_count"] == 2
        assert payload["buckets"][0]["date_sources"] == {
            "source_upload_date": 1,
            "ingestion_created_at": 1,
        }
        assert payload["date_provenance"]["source_upload_date_recordings"] == 1
        assert payload["date_provenance"]["ingestion_created_at_recordings"] == 1
        # One distinct day: a one-point chart says nothing, so the console
        # renders a dated summary instead (#385).
        assert payload["display_mode"] == "single_date"

    def test_two_distinct_dates_select_chart_mode(self) -> None:
        records = [
            recording(when=date(2026, 1, 1), text="alpha"),
            recording(when=date(2026, 1, 2), text="alpha"),
        ]
        assert build_temporal_trends(records)["display_mode"] == "chart"

    def test_display_mode_counts_distinct_dates_not_buckets(self) -> None:
        assert display_mode_for([]) == "empty"
        assert display_mode_for([date(2026, 1, 1), date(2026, 1, 1)]) == "single_date"
        assert display_mode_for([date(2026, 1, 1), date(2026, 1, 2)]) == "chart"
        # Two dates far enough apart to bucket by month still count as two.
        far = [date(2020, 1, 1), date(2026, 1, 1)]
        assert display_mode_for(far) == "chart"
        payload = build_temporal_trends(
            [recording(when=far[0], text="a"), recording(when=far[1], text="a")]
        )
        assert payload["range"]["bucket_unit"] == "month"
        assert payload["display_mode"] == "chart"

    def test_zero_fills_gaps_and_all_series_align_to_buckets(self) -> None:
        records = [
            recording(
                when=date(2026, 1, 1),
                text="alpha alpha",
                mentions={"mentions": [{"surface": "Acme", "occurrences": []}]},
            ),
            recording(
                when=date(2026, 1, 3),
                text="alpha beta",
                mentions={"mentions": [{"surface": "Beta", "occurrences": [{}]}]},
            ),
        ]

        payload = build_temporal_trends(records)

        assert [bucket["recording_count"] for bucket in payload["buckets"]] == [1, 0, 1]
        assert all(
            len(series["values"]) == len(payload["buckets"])
            for series in [*payload["terms"], *payload["entities"]]
        )
        assert payload["terms"][0]["values"] == [2, 0, 1]
        assert payload["entities"][0]["values"] == [1, 0, 0]

    def test_topics_are_not_counted_as_terms(self) -> None:
        rec = recording(
            when=date(2026, 1, 1),
            text="spoken",
            mentions={
                "mentions": [],
                "topics": [{"label": "forbidden-topic", "confidence": 1.0}],
            },
        )
        assert [term["key"] for term in build_temporal_trends([rec])["terms"]] == ["spoken"]

    def test_empty_entity_asset_counts_as_enrichment_coverage(self) -> None:
        rec = recording(
            when=date(2026, 1, 1),
            mentions={"mentions": []},
        )
        payload = build_temporal_trends([rec])
        assert payload["entities"] == []
        assert payload["coverage"]["entity_enriched_recordings"] == 1

    def test_caps_and_truncation_flags(self) -> None:
        terms = " ".join(f"word{chr(97 + index)}" for index in range(25))
        mentions = {
            "mentions": [
                {"surface": f"Entity {index:02d}", "occurrences": []} for index in range(25)
            ]
        }
        payload = build_temporal_trends(
            [recording(when=date(2026, 1, 1), text=terms, mentions=mentions)]
        )
        assert len(payload["terms"]) == MAX_TERMS
        assert len(payload["entities"]) == MAX_ENTITIES
        assert payload["truncated"] == {"terms": True, "entities": True}

    def test_payload_is_json_serializable(self) -> None:
        payload = build_temporal_trends([recording(when=date(2026, 1, 1), text="signal")])
        assert json.loads(json.dumps(payload)) == payload

    def test_selected_bucket_count_obeys_target_when_a_supported_unit_can(self) -> None:
        payload = build_temporal_trends(
            [
                recording(when=date(2026, 1, 1)),
                recording(when=date(2027, 1, 1)),
            ]
        )
        assert len(payload["buckets"]) <= MAX_BUCKETS
