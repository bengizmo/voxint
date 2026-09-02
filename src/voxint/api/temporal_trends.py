"""Temporal aggregation and project-scoped caching (issue #337).

The aggregation functions are pure: callers provide one :class:`RecordingInput`
per canonical recording and receive a JSON-friendly payload.  The orchestration
functions at the end of the module load those inputs and cache the result in a
``TEMPORAL_TRENDS`` analysis artifact.
"""

from __future__ import annotations

import hashlib
import unicodedata
import uuid
from collections import Counter, defaultdict
from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal, TypedDict, cast

from sqlalchemy import delete, func, literal, select
from sqlalchemy.dialects.postgresql import aggregate_order_by
from sqlalchemy.orm import Session

from voxint.api.project_insights import _canonical_project_runs
from voxint.api.term_stats import source_hash, tokenize
from voxint.db.models import (
    CorpusAnalysisArtifact,
    CorpusAnalysisArtifactKind,
    MediaItem,
    MediaSourceMetadata,
    PipelineRun,
    RunAssetKind,
    RunEnrichmentAsset,
    SegmentReviewState,
    TranscriptSegment,
)

MAX_TERMS = 20
MAX_ENTITIES = 20
MAX_BUCKETS = 60
# Version 2 (#385): payload gains ``display_mode``. ALGORITHM_VERSION feeds the
# cache fingerprint, so bumping it recomputes artifacts stored under the old
# shape instead of serving a payload without the key.
ALGORITHM_VERSION = "2"
SCHEMA_VERSION = 2

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


# How the console should render the payload (#385). Decided server-side so the
# Jinja mount and the React island never disagree on the threshold:
# ``chart`` for two or more distinct dates, ``single_date`` when every dated
# recording falls on one day (a one-point chart conveys nothing, so the mount
# renders a dated summary instead of hiding the data), ``empty`` for none.
DisplayMode = Literal["chart", "single_date", "empty"]


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
    display_mode: DisplayMode


def display_mode_for(dates: list[date]) -> DisplayMode:
    """Pick the render mode from the resolved recording dates themselves.

    Counts distinct calendar dates, not occupied buckets, so the answer does
    not depend on the bucket unit a long range happens to select.
    """
    distinct_dates = len(set(dates))
    if distinct_dates == 0:
        return "empty"
    return "single_date" if distinct_dates == 1 else "chart"


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
    recordings: list[RecordingInput],
    buckets: list[BucketMeta],
    *,
    pre_tokenized: list[list[str]] | None = None,
) -> list[TermSeries]:
    """Count token occurrences per bucket and return the top term series."""
    bucket_counts: defaultdict[str, list[int]] = defaultdict(lambda: [0] * len(buckets))
    totals: Counter[str] = Counter()
    recording_counts: Counter[str] = Counter()

    for idx, recording in enumerate(recordings):
        recording_date, _ = resolve_date(recording["upload_date"], recording["created_at"])
        index = _bucket_index(buckets, recording_date)
        if index is None:
            continue
        tokens = (
            pre_tokenized[idx]
            if pre_tokenized is not None
            else tokenize(recording["effective_text"])
        )
        frequencies = Counter(tokens)
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
        "display_mode": "empty",
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

    per_recording_tokens = [tokenize(recording["effective_text"]) for recording in recordings]
    terms = count_term_frequencies(recordings, buckets, pre_tokenized=per_recording_tokens)
    entities = count_entity_frequencies(recordings, buckets)
    all_term_keys = {term for tokens in per_recording_tokens for term in tokens}
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
        "term_recordings": sum(bool(tokens) for tokens in per_recording_tokens),
        "entity_enriched_recordings": sum(
            isinstance(recording["entity_mentions"], dict) for recording in recordings
        ),
    }
    payload["truncated"] = {
        "terms": len(all_term_keys) > MAX_TERMS,
        "entities": len(all_entity_keys) > MAX_ENTITIES,
    }
    payload["display_mode"] = display_mode_for(dates)
    return payload


# ---------------------------------------------------------------------------
# Project-scoped compute-on-read cache
# ---------------------------------------------------------------------------

CanonicalRun = tuple[uuid.UUID, uuid.UUID, str]


def _temporal_lock_key(project_id: uuid.UUID) -> int:
    """Return a deterministic signed-safe advisory-lock key for a project."""
    digest = hashlib.sha256(f"voxint:temporal_trends:{project_id}".encode())
    return int(digest.hexdigest()[:8], 16) & 0x7FFFFFFF


def _temporal_fingerprint(
    session: Session,
    project_id: uuid.UUID,
    canonical_runs: list[CanonicalRun],
) -> str:
    """Hash every mutable input consumed by temporal trend generation."""
    run_ids = [run_id for run_id, _, _ in canonical_runs]
    if not run_ids:
        return source_hash([("version", ALGORITHM_VERSION), ("project", str(project_id))])

    run_rows = session.execute(
        select(PipelineRun.id, PipelineRun.updated_at)
        .where(PipelineRun.id.in_(run_ids))
        .order_by(PipelineRun.id)
    ).all()
    run_updated = {row.id: row.updated_at for row in run_rows}

    correction_rows = session.execute(
        select(
            TranscriptSegment.pipeline_run_id,
            func.max(SegmentReviewState.corrected_at).label("latest_corrected_at"),
        )
        .join(
            SegmentReviewState,
            SegmentReviewState.transcript_segment_id == TranscriptSegment.id,
        )
        .where(
            TranscriptSegment.pipeline_run_id.in_(run_ids),
            SegmentReviewState.corrected_text.is_not(None),
        )
        .group_by(TranscriptSegment.pipeline_run_id)
    ).all()
    corrected_at = {row.pipeline_run_id: row.latest_corrected_at for row in correction_rows}

    date_rows = session.execute(
        select(
            PipelineRun.id.label("run_id"),
            MediaItem.created_at,
            MediaSourceMetadata.upload_date,
        )
        .join(MediaItem, MediaItem.id == PipelineRun.media_item_id)
        .outerjoin(MediaSourceMetadata, MediaSourceMetadata.media_item_id == MediaItem.id)
        .where(PipelineRun.id.in_(run_ids))
        .order_by(PipelineRun.id)
    ).all()
    dates = {row.run_id: f"{row.upload_date or ''}:{row.created_at}" for row in date_rows}

    asset_rows = session.execute(
        select(
            RunEnrichmentAsset.id,
            RunEnrichmentAsset.pipeline_run_id,
            RunEnrichmentAsset.completed_at,
        )
        .where(
            RunEnrichmentAsset.pipeline_run_id.in_(run_ids),
            RunEnrichmentAsset.asset_kind == RunAssetKind.ENTITY_MENTIONS.value,
            RunEnrichmentAsset.superseded_by_asset_id.is_(None),
        )
        .order_by(RunEnrichmentAsset.pipeline_run_id, RunEnrichmentAsset.id)
    ).all()
    assets = source_hash(
        [(str(row.id), f"{row.pipeline_run_id}:{row.completed_at}") for row in asset_rows]
    )

    canonical = source_hash(
        [
            (
                str(run_id),
                ":".join(
                    (
                        str(media_id),
                        str(run_updated.get(run_id, "")),
                        str(corrected_at.get(run_id, "")),
                        dates.get(run_id, ""),
                    )
                ),
            )
            for run_id, media_id, _ in canonical_runs
        ]
    )
    return source_hash(
        [
            ("version", ALGORITHM_VERSION),
            ("project", str(project_id)),
            ("canonical", canonical),
            ("entity_assets", assets),
        ]
    )


def _load_recording_inputs(
    session: Session, canonical_runs: list[CanonicalRun]
) -> list[RecordingInput]:
    """Load one aggregation input per canonical run, preserving canonical order."""
    run_ids = [run_id for run_id, _, _ in canonical_runs]
    if not run_ids:
        return []

    metadata_rows = session.execute(
        select(
            PipelineRun.id.label("run_id"),
            MediaItem.id.label("media_id"),
            MediaItem.created_at,
            MediaSourceMetadata.upload_date,
        )
        .join(MediaItem, MediaItem.id == PipelineRun.media_item_id)
        .outerjoin(MediaSourceMetadata, MediaSourceMetadata.media_item_id == MediaItem.id)
        .where(PipelineRun.id.in_(run_ids))
    ).all()
    metadata = {row.run_id: row for row in metadata_rows}

    effective_text = func.coalesce(
        SegmentReviewState.corrected_text,
        TranscriptSegment.enhanced_text,
        TranscriptSegment.raw_text,
    )
    text_rows = session.execute(
        select(
            TranscriptSegment.pipeline_run_id.label("run_id"),
            func.string_agg(
                effective_text,
                aggregate_order_by(literal(" "), TranscriptSegment.segment_index),
            ).label("text"),
        )
        .outerjoin(
            SegmentReviewState,
            SegmentReviewState.transcript_segment_id == TranscriptSegment.id,
        )
        .where(TranscriptSegment.pipeline_run_id.in_(run_ids))
        .group_by(TranscriptSegment.pipeline_run_id)
    ).all()
    texts = {row.run_id: row.text or "" for row in text_rows}

    asset_rows = session.execute(
        select(RunEnrichmentAsset.pipeline_run_id, RunEnrichmentAsset.payload)
        .where(
            RunEnrichmentAsset.pipeline_run_id.in_(run_ids),
            RunEnrichmentAsset.asset_kind == RunAssetKind.ENTITY_MENTIONS.value,
            RunEnrichmentAsset.superseded_by_asset_id.is_(None),
        )
        .order_by(RunEnrichmentAsset.pipeline_run_id, RunEnrichmentAsset.generation.desc())
    ).all()
    entities: dict[uuid.UUID, dict[str, Any]] = {}
    for asset_row in asset_rows:
        entities.setdefault(asset_row.pipeline_run_id, asset_row.payload)

    recordings: list[RecordingInput] = []
    for run_id, media_id, _ in canonical_runs:
        metadata_row = metadata.get(run_id)
        if metadata_row is None:
            continue
        recordings.append(
            {
                "run_id": run_id,
                "media_id": media_id,
                "upload_date": metadata_row.upload_date,
                "created_at": metadata_row.created_at,
                "effective_text": texts.get(run_id, ""),
                "entity_mentions": entities.get(run_id),
            }
        )
    return recordings


def compute_temporal_trends(session: Session, project_id: uuid.UUID) -> TemporalTrendsPayload:
    """Return fresh project trends, recomputing the cached artifact if stale."""
    session.execute(select(func.pg_advisory_xact_lock(_temporal_lock_key(project_id))))
    canonical_runs = _canonical_project_runs(session, project_id)
    fingerprint = _temporal_fingerprint(session, project_id, canonical_runs)

    cached = session.execute(
        select(CorpusAnalysisArtifact)
        .where(
            CorpusAnalysisArtifact.scope_kind == "project",
            CorpusAnalysisArtifact.scope_id == project_id,
            CorpusAnalysisArtifact.artifact_kind
            == CorpusAnalysisArtifactKind.TEMPORAL_TRENDS.value,
        )
        .order_by(CorpusAnalysisArtifact.generation.desc())
        .limit(1)
    ).scalar_one_or_none()
    if cached is not None and cached.source_hash == fingerprint:
        return cast(TemporalTrendsPayload, cached.payload)

    payload = build_temporal_trends(_load_recording_inputs(session, canonical_runs))
    session.execute(
        delete(CorpusAnalysisArtifact).where(
            CorpusAnalysisArtifact.scope_kind == "project",
            CorpusAnalysisArtifact.scope_id == project_id,
            CorpusAnalysisArtifact.artifact_kind
            == CorpusAnalysisArtifactKind.TEMPORAL_TRENDS.value,
        )
    )
    session.add(
        CorpusAnalysisArtifact(
            scope_kind="project",
            scope_id=project_id,
            artifact_kind=CorpusAnalysisArtifactKind.TEMPORAL_TRENDS.value,
            generation=1,
            source_hash=fingerprint,
            payload=payload,
        )
    )
    session.flush()
    return payload


def get_temporal_trends(session: Session, project_id: uuid.UUID) -> TemporalTrendsPayload:
    """Read path for project temporal trends."""
    return compute_temporal_trends(session, project_id)
