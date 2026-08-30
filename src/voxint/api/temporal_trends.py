"""Pure temporal aggregation for project-level corpus trends (issue #337).

This module deliberately has no database or framework dependencies.  Callers
provide one :class:`RecordingInput` per canonical recording; the functions
resolve the recording date, choose calendar-aligned buckets, and return a
JSON-friendly payload suitable for a ``TEMPORAL_TRENDS`` analysis artifact.
"""

from __future__ import annotations

import unicodedata
import uuid
from collections import Counter, defaultdict
from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal, TypedDict, cast

from voxint.api.term_stats import tokenize

MAX_TERMS = 20
MAX_ENTITIES = 20
MAX_BUCKETS = 60
ALGORITHM_VERSION = "1"
SCHEMA_VERSION = 1

BucketUnit = Literal["day", "week", "month"]
DateProvenance = Literal["source_upload_date", "ingestion_created_at"]


class RecordingInput(TypedDict):
    """All source data needed to aggregate one canonical recording."""

    run_id: uuid.UUID
    media_id: uuid.UUID
    upload_date: date | None
    created_at: datetime
    effective_text: str
    entity_mentions: dict[str, Any] | None


class DateSourceCounts(TypedDict):
    source_upload_date: int
    ingestion_created_at: int


class BucketMeta(TypedDict):
    """One calendar bucket; dates are ISO strings at the JSON boundary."""

    start: str
    end_exclusive: str
    recording_count: int
    date_sources: DateSourceCounts


class SeriesPoint(TypedDict):
    """Named representation of one dense series value, useful to consumers."""

    bucket_start: str
    value: int


class TermSeries(TypedDict):
    key: str
    label: str
    total_count: int
    recording_count: int
    values: list[int]


class EntitySeries(TypedDict):
    key: str
    label: str
    kind: str | None
    total_count: int
    recording_count: int
    values: list[int]


class TemporalRange(TypedDict):
    start: str | None
    end: str | None
    bucket_unit: BucketUnit | None
    week_starts_on: Literal["monday"]
    timezone: Literal["UTC"]


class DateProvenanceSummary(TypedDict):
    preference: list[str]
    source_upload_date_recordings: int
    ingestion_created_at_recordings: int
    undated_recordings: int
    label: str


class TemporalCoverage(TypedDict):
    dated_recordings: int
    term_recordings: int
    entity_enriched_recordings: int


class TruncationSummary(TypedDict):
    terms: bool
    entities: bool


class TemporalTrendsPayload(TypedDict):
    schema_version: int
    algorithm_version: str
    range: TemporalRange
    date_provenance: DateProvenanceSummary
    buckets: list[BucketMeta]
    terms: list[TermSeries]
    entities: list[EntitySeries]
    coverage: TemporalCoverage
    truncated: TruncationSummary


def resolve_date(upload_date: date | None, created_at: datetime) -> tuple[date, str]:
    """Resolve a recording's effective date and its provenance.

    A source-claimed upload date is preferred.  Ingestion timestamps are
    normalized to UTC before extracting their date so the result is independent
    of the application server's local timezone.  Naive timestamps are treated
    as already-UTC, matching PostgreSQL's UTC application convention.
    """
    if upload_date is not None:
        return upload_date, "source_upload_date"
    if created_at.tzinfo is not None:
        created_at = created_at.astimezone(UTC)
    return created_at.date(), "ingestion_created_at"


def _month_start(value: date) -> date:
    return value.replace(day=1)


def _aligned_start(value: date, unit: BucketUnit) -> date:
    if unit == "day":
        return value
    if unit == "week":
        return value - timedelta(days=value.weekday())
    return _month_start(value)


def _next_bucket(value: date, unit: BucketUnit) -> date:
    if unit == "day":
        return value + timedelta(days=1)
    if unit == "week":
        return value + timedelta(days=7)
    if value.month == 12:
        return date(value.year + 1, 1, 1)
    return date(value.year, value.month + 1, 1)


def _bucket_count(min_date: date, max_date: date, unit: BucketUnit) -> int:
    start = _aligned_start(min_date, unit)
    final = _aligned_start(max_date, unit)
    if unit == "day":
        return (final - start).days + 1
    if unit == "week":
        return (final - start).days // 7 + 1
    return (final.year - start.year) * 12 + final.month - start.month + 1


def select_bucket_unit(min_date: date, max_date: date) -> str:
    """Choose the finest calendar unit producing at most ``MAX_BUCKETS``.

    Month is the coarsest unit in the issue contract.  It is retained for very
    long corpora even when the range exceeds 60 months; consumers can thin axis
    ticks without silently dropping time periods.
    """
    if min_date > max_date:
        raise ValueError("min_date must not be after max_date")
    for unit in ("day", "week", "month"):
        if _bucket_count(min_date, max_date, unit) <= MAX_BUCKETS:
            return unit
    return "month"


def generate_buckets(min_date: date, max_date: date, unit: str) -> list[BucketMeta]:
    """Generate dense, calendar-aligned buckets spanning the date range."""
    if min_date > max_date:
        raise ValueError("min_date must not be after max_date")
    if unit not in ("day", "week", "month"):
        raise ValueError(f"unsupported bucket unit: {unit}")
    typed_unit = cast(BucketUnit, unit)
    current = _aligned_start(min_date, typed_unit)
    final = _aligned_start(max_date, typed_unit)
    buckets: list[BucketMeta] = []
    while current <= final:
        following = _next_bucket(current, typed_unit)
        buckets.append(
            {
                "start": current.isoformat(),
                "end_exclusive": following.isoformat(),
                "recording_count": 0,
                "date_sources": {
                    "source_upload_date": 0,
                    "ingestion_created_at": 0,
                },
            }
        )
        current = following
    return buckets


def _bucket_index(buckets: list[BucketMeta], value: date) -> int | None:
    """Return the bucket containing ``value`` without assuming a fixed width."""
    for index, bucket in enumerate(buckets):
        start = date.fromisoformat(bucket["start"])
        end = date.fromisoformat(bucket["end_exclusive"])
        if start <= value < end:
            return index
    return None


def count_term_frequencies(
    recordings: list[RecordingInput], buckets: list[BucketMeta]
) -> list[TermSeries]:
    """Count token occurrences per bucket and return the top term series."""
    bucket_counts: defaultdict[str, list[int]] = defaultdict(lambda: [0] * len(buckets))
    totals: Counter[str] = Counter()
    recording_counts: Counter[str] = Counter()

    for recording in recordings:
        recording_date, _ = resolve_date(recording["upload_date"], recording["created_at"])
        index = _bucket_index(buckets, recording_date)
        if index is None:
            continue
        frequencies = Counter(tokenize(recording["effective_text"]))
        for term, count in frequencies.items():
            bucket_counts[term][index] += count
            totals[term] += count
            recording_counts[term] += 1

    ranked = sorted(totals, key=lambda term: (-totals[term], term))[:MAX_TERMS]
    return [
        {
            "key": term,
            "label": term,
            "total_count": totals[term],
            "recording_count": recording_counts[term],
            "values": bucket_counts[term],
        }
        for term in ranked
    ]


def _normalize_entity(surface: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", surface).casefold().split())


def _valid_mentions(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    raw_mentions = payload.get("mentions")
    if not isinstance(raw_mentions, list):
        return []
    return [mention for mention in raw_mentions if isinstance(mention, dict)]


def count_entity_frequencies(
    recordings: list[RecordingInput], buckets: list[BucketMeta]
) -> list[EntitySeries]:
    """Count normalized entity mentions per bucket and return top series."""
    bucket_counts: defaultdict[str, list[int]] = defaultdict(lambda: [0] * len(buckets))
    totals: Counter[str] = Counter()
    recording_ids: defaultdict[str, set[uuid.UUID]] = defaultdict(set)
    labels: dict[str, str] = {}
    kinds: defaultdict[str, Counter[str]] = defaultdict(Counter)

    for recording in recordings:
        recording_date, _ = resolve_date(recording["upload_date"], recording["created_at"])
        index = _bucket_index(buckets, recording_date)
        if index is None:
            continue
        for mention in _valid_mentions(recording["entity_mentions"]):
            raw_surface = mention.get("surface")
            if not isinstance(raw_surface, str) or not raw_surface.strip():
                continue
            key = _normalize_entity(raw_surface)
            if not key:
                continue
            labels.setdefault(key, raw_surface.strip())
            raw_kind = mention.get("kind")
            if isinstance(raw_kind, str) and raw_kind.strip():
                kinds[key][raw_kind.strip().casefold()] += 1
            occurrences = mention.get("occurrences")
            count = max(1, len(occurrences)) if isinstance(occurrences, list) else 1
            bucket_counts[key][index] += count
            totals[key] += count
            recording_ids[key].add(recording["run_id"])

    ranked = sorted(totals, key=lambda key: (-totals[key], key))[:MAX_ENTITIES]
    result: list[EntitySeries] = []
    for key in ranked:
        kind_counts = kinds[key]
        # Resolve equal-frequency kinds alphabetically for order-independent output.
        resolved_kind = (
            min(kind_counts, key=lambda kind: (-kind_counts[kind], kind)) if kind_counts else None
        )
        result.append(
            {
                "key": key,
                "label": labels[key],
                "kind": resolved_kind,
                "total_count": totals[key],
                "recording_count": len(recording_ids[key]),
                "values": bucket_counts[key],
            }
        )
    return result


def _empty_payload() -> TemporalTrendsPayload:
    return {
        "schema_version": SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "range": {
            "start": None,
            "end": None,
            "bucket_unit": None,
            "week_starts_on": "monday",
            "timezone": "UTC",
        },
        "date_provenance": {
            "preference": [
                "media_source_metadata.upload_date",
                "media_items.created_at",
            ],
            "source_upload_date_recordings": 0,
            "ingestion_created_at_recordings": 0,
            "undated_recordings": 0,
            "label": ("Source-reported upload date where available; ingestion date otherwise."),
        },
        "buckets": [],
        "terms": [],
        "entities": [],
        "coverage": {
            "dated_recordings": 0,
            "term_recordings": 0,
            "entity_enriched_recordings": 0,
        },
        "truncated": {"terms": False, "entities": False},
    }


def build_temporal_trends(recordings: list[RecordingInput]) -> TemporalTrendsPayload:
    """Build the complete JSON-friendly temporal trends payload."""
    payload = _empty_payload()
    if not recordings:
        return payload

    resolved = [
        resolve_date(recording["upload_date"], recording["created_at"]) for recording in recordings
    ]
    dates = [recording_date for recording_date, _ in resolved]
    min_date = min(dates)
    max_date = max(dates)
    unit = select_bucket_unit(min_date, max_date)
    buckets = generate_buckets(min_date, max_date, unit)

    provenance_counts: Counter[str] = Counter(provenance for _, provenance in resolved)
    for recording_date, provenance in resolved:
        index = _bucket_index(buckets, recording_date)
        if index is not None:
            buckets[index]["recording_count"] += 1
            if provenance == "source_upload_date":
                buckets[index]["date_sources"]["source_upload_date"] += 1
            else:
                buckets[index]["date_sources"]["ingestion_created_at"] += 1

    terms = count_term_frequencies(recordings, buckets)
    entities = count_entity_frequencies(recordings, buckets)
    all_term_keys = {
        term for recording in recordings for term in tokenize(recording["effective_text"])
    }
    all_entity_keys = {
        _normalize_entity(surface)
        for recording in recordings
        for mention in _valid_mentions(recording["entity_mentions"])
        if isinstance((surface := mention.get("surface")), str) and surface.strip()
    }

    payload["range"] = {
        "start": min_date.isoformat(),
        "end": max_date.isoformat(),
        "bucket_unit": cast(BucketUnit, unit),
        "week_starts_on": "monday",
        "timezone": "UTC",
    }
    payload["date_provenance"].update(
        {
            "source_upload_date_recordings": provenance_counts["source_upload_date"],
            "ingestion_created_at_recordings": provenance_counts["ingestion_created_at"],
        }
    )
    payload["buckets"] = buckets
    payload["terms"] = terms
    payload["entities"] = entities
    payload["coverage"] = {
        "dated_recordings": len(recordings),
        "term_recordings": sum(
            bool(tokenize(recording["effective_text"])) for recording in recordings
        ),
        "entity_enriched_recordings": sum(
            isinstance(recording["entity_mentions"], dict) for recording in recordings
        ),
    }
    payload["truncated"] = {
        "terms": len(all_term_keys) > MAX_TERMS,
        "entities": len(all_entity_keys) > MAX_ENTITIES,
    }
    return payload
