"""Single sanctioned writer + read side for run-level enrichment assets (#41).

:func:`record_asset` is the only way a successful generation persists: one
immutable ``run_enrichment_assets`` row per (run, asset kind, generation),
validated up front and fail-closed. The writer mirrors ``drafts.py``:

- serializes per (run, kind) with a transaction-scoped advisory lock so
  generation allocation, insertion, and supersession are one atomic step;
- allocates a monotonic ``generation`` under that lock — "newer" is a
  generation comparison, never wall-clock;
- supersedes the prior still-current asset of the *same kind only* in the
  same transaction — the three kinds never touch each other (the issue's
  independent-versioning requirement);
- replays idempotently: the same ``idempotency_key`` with the same payload
  returns the existing row, a different payload is an error.

The module also owns the **source snapshot**: :func:`load_source` reads
exactly what the generators are allowed to see (the ordered transcript with
its raw diarization labels — resolved speaker names are a deliberate v1 cut —
plus the #36 metadata snapshot and operator notes), and
:func:`source_content_hash` canonicalizes it into the sha256 staleness
detector stored on every asset. Content only — model/prompt versions are
recorded separately, so a prompt upgrade never masquerades as a source
change.
"""

import hashlib
import json
import math
import unicodedata
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import func, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from voxint.db.models import (
    MediaSourceMetadata,
    PipelineRun,
    RunAssetKind,
    RunEnrichmentAsset,
    TranscriptSegment,
)
from voxint.enrichment.review import ConflictingReplayError

SOURCE_SCHEMA_VERSION = 1

MAX_PRODUCER_CHARS = 200
MAX_MODEL_CHARS = 200
MAX_SUMMARY_CHARS = 4_000
MAX_TOPICS = 10
MAX_TOPIC_LABEL_CHARS = 120
MAX_TOPIC_DESCRIPTION_CHARS = 500
MAX_MENTIONS = 100
MAX_OCCURRENCES_PER_MENTION = 20
MAX_SURFACE_CHARS = 200
MAX_CONFIG_BYTES = 16_384

ENTITY_KINDS = ("person", "organization", "product")


class RunAssetError(Exception):
    """A generator submitted something the asset layer refuses to persist."""


def normalize_span_text(text: str) -> str:
    """NFKC + casefold + whitespace-collapse — the matching normalization
    shared by the producer's grounding and the writer's re-validation."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def quote_matches_surface(surface: str, quote: str) -> bool:
    """A grounded quote must be an occurrence OF the surface, not merely any
    real substring of the transcript — otherwise a model could hang an
    invented entity name on a genuinely-located "the". Containment either way
    (normalized): the quote may be a partial form of the surface ("Acme" for
    "Acme Corp") or carry surrounding words, but the two must overlap."""
    normalized_surface = normalize_span_text(surface)
    normalized_quote = normalize_span_text(quote)
    return (
        bool(normalized_surface)
        and bool(normalized_quote)
        and (normalized_surface in normalized_quote or normalized_quote in normalized_surface)
    )


def span_boundaries_ok(segment: str, start: int, end: int) -> bool:
    """Alphanumeric edges of a span must sit on non-alphanumeric boundaries —
    "Ann" must not anchor inside "Joanne", "1" not inside "2019". Shared rule:
    the producer's locate enforces it and the writer re-checks it, so the two
    can never diverge."""
    quote = segment[start:end]
    if not quote:
        return False
    if quote[0].isalnum() and start > 0 and segment[start - 1].isalnum():
        return False
    return not (quote[-1].isalnum() and end < len(segment) and segment[end].isalnum())


@dataclass(frozen=True)
class SegmentSource:
    """One transcript segment as the generators see it."""

    segment_index: int
    diarization_label: str | None
    text: str


@dataclass(frozen=True)
class RunAssetSource:
    """Everything a run-asset generation reads — the hashed staleness domain."""

    pipeline_run_id: uuid.UUID
    segments: tuple[SegmentSource, ...]
    metadata: Mapping[str, Any] | None
    operator_notes: str | None


def load_source(session: Session, pipeline_run_id: uuid.UUID) -> RunAssetSource:
    """The exact inputs a generation may read, in deterministic order.

    Segment text is the operator-facing best text (``enhanced_text`` falling
    back to ``raw_text``) — the same pinning as ``names.llm``: enhancement
    changing SHOULD make assets stale, because it changes what a regeneration
    would read. Raises :class:`RunAssetError` for an unknown run or one with
    no transcript yet (there is nothing to summarize; an asset over an empty
    source would be an authoritative-sounding lie).
    """
    run = session.get(PipelineRun, pipeline_run_id)
    if run is None:
        raise RunAssetError(f"unknown pipeline run: {pipeline_run_id}")
    rows = (
        session.execute(
            select(TranscriptSegment)
            .where(TranscriptSegment.pipeline_run_id == pipeline_run_id)
            .order_by(TranscriptSegment.segment_index)
        )
        .scalars()
        .all()
    )
    if not rows:
        raise RunAssetError(
            "run has no transcript segments yet — assets are generated from the"
            " transcript, so the run must finish transcription first"
        )
    metadata_row = session.execute(
        select(MediaSourceMetadata).where(MediaSourceMetadata.media_item_id == run.media_item_id)
    ).scalar_one_or_none()
    metadata: dict[str, Any] | None = None
    if metadata_row is not None:
        # Only the prompt-relevant context fields, with explicit nulls — never
        # the raw extractor document.
        metadata = {
            "title": metadata_row.title,
            "uploader": metadata_row.uploader,
            "channel": metadata_row.channel,
            "description": metadata_row.description,
            "upload_date": metadata_row.upload_date.isoformat()
            if metadata_row.upload_date is not None
            else None,
            "tags": list(metadata_row.tags),
        }
    return RunAssetSource(
        pipeline_run_id=pipeline_run_id,
        segments=tuple(
            SegmentSource(
                segment_index=row.segment_index,
                diarization_label=row.diarization_label,
                text=row.enhanced_text or row.raw_text,
            )
            for row in rows
        ),
        metadata=metadata,
        operator_notes=run.operator_notes,
    )


def source_content_hash(source: RunAssetSource) -> str:
    """sha256 over the canonical serialization of the generation inputs.

    Deterministic and content-only: stable key order, compact separators,
    explicit nulls. Excludes model/producer/prompt versions on purpose — those
    are provenance columns, and folding them in would make every code upgrade
    look like a source change.
    """
    payload: dict[str, Any] = {
        "source_schema_version": SOURCE_SCHEMA_VERSION,
        "pipeline_run_id": str(source.pipeline_run_id),
        "segments": [
            [segment.segment_index, segment.diarization_label, segment.text]
            for segment in source.segments
        ],
        "metadata": dict(source.metadata) if source.metadata is not None else None,
        "operator_notes": source.operator_notes,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _require_keys(
    obj: Mapping[str, Any], allowed: tuple[str, ...], required: tuple[str, ...], label: str
) -> None:
    extra = set(obj) - set(allowed)
    if extra:
        raise RunAssetError(f"{label} has unknown keys: {sorted(extra)}")
    missing = set(required) - set(obj)
    if missing:
        raise RunAssetError(f"{label} is missing keys: {sorted(missing)}")


def _validate_summary_payload(payload: Mapping[str, Any]) -> None:
    _require_keys(payload, ("summary",), ("summary",), "summary payload")
    summary = payload["summary"]
    if not isinstance(summary, str) or not summary.strip():
        raise RunAssetError("summary must be a non-empty string")
    if len(summary) > MAX_SUMMARY_CHARS:
        raise RunAssetError(f"summary over {MAX_SUMMARY_CHARS} chars")


def _validate_topics_payload(payload: Mapping[str, Any]) -> None:
    _require_keys(payload, ("topics",), ("topics",), "topics payload")
    topics = payload["topics"]
    if not isinstance(topics, list) or not (1 <= len(topics) <= MAX_TOPICS):
        raise RunAssetError(f"topics must be a list of 1..{MAX_TOPICS} entries")
    seen: set[str] = set()
    for topic in topics:
        if not isinstance(topic, Mapping):
            raise RunAssetError("each topic must be an object")
        _require_keys(
            topic,
            ("label", "description", "confidence", "vocabulary", "term_id"),
            ("label",),
            "topic",
        )
        label = topic["label"]
        if not isinstance(label, str) or not label.strip():
            raise RunAssetError("topic label must be a non-empty string")
        if len(label) > MAX_TOPIC_LABEL_CHARS:
            raise RunAssetError(f"topic label over {MAX_TOPIC_LABEL_CHARS} chars")
        folded = label.casefold()
        if folded in seen:
            raise RunAssetError(f"duplicate topic label: {label!r}")
        seen.add(folded)
        description = topic.get("description")
        if description is not None and (
            not isinstance(description, str)
            or not description.strip()
            or len(description) > MAX_TOPIC_DESCRIPTION_CHARS
        ):
            raise RunAssetError("topic description must be null or a bounded string")
        confidence = topic.get("confidence")
        if confidence is not None:
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
                raise RunAssetError("topic confidence must be null or a number")
            if not (math.isfinite(float(confidence)) and 0.0 <= float(confidence) <= 1.0):
                raise RunAssetError("topic confidence must be in [0, 1]")
        # Reserved for the #11 domain-pack vocabularies — payload v1 keeps
        # them null so their arrival is a payload_schema_version bump, not a
        # table migration.
        if topic.get("vocabulary") is not None or topic.get("term_id") is not None:
            raise RunAssetError("topic vocabulary/term_id must be null in payload v1")


def _validate_mentions_payload(payload: Mapping[str, Any], segment_text: Mapping[int, str]) -> None:
    _require_keys(
        payload, ("mentions", "diagnostics"), ("mentions", "diagnostics"), "mentions payload"
    )
    mentions = payload["mentions"]
    if not isinstance(mentions, list) or len(mentions) > MAX_MENTIONS:
        raise RunAssetError(f"mentions must be a list of 0..{MAX_MENTIONS} entries")
    diagnostics = payload["diagnostics"]
    if not isinstance(diagnostics, Mapping):
        raise RunAssetError("diagnostics must be an object")
    _require_keys(
        diagnostics,
        ("dropped_unlocatable", "dropped_out_of_run"),
        ("dropped_unlocatable", "dropped_out_of_run"),
        "mention diagnostics",
    )
    for key in ("dropped_unlocatable", "dropped_out_of_run"):
        count = diagnostics[key]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise RunAssetError(f"diagnostics.{key} must be a non-negative integer")
    for mention in mentions:
        if not isinstance(mention, Mapping):
            raise RunAssetError("each mention must be an object")
        _require_keys(
            mention, ("surface", "kind", "occurrences"), ("surface", "occurrences"), "mention"
        )
        surface = mention["surface"]
        if not isinstance(surface, str) or not surface.strip():
            raise RunAssetError("mention surface must be a non-empty string")
        if len(surface) > MAX_SURFACE_CHARS:
            raise RunAssetError(f"mention surface over {MAX_SURFACE_CHARS} chars")
        kind = mention.get("kind")
        if kind is not None and kind not in ENTITY_KINDS:
            raise RunAssetError(f"mention kind must be null or one of {ENTITY_KINDS}")
        occurrences = mention["occurrences"]
        if not isinstance(occurrences, list) or not (
            1 <= len(occurrences) <= MAX_OCCURRENCES_PER_MENTION
        ):
            raise RunAssetError(f"a mention needs 1..{MAX_OCCURRENCES_PER_MENTION} occurrences")
        for occurrence in occurrences:
            if not isinstance(occurrence, Mapping):
                raise RunAssetError("each occurrence must be an object")
            _require_keys(
                occurrence,
                ("segment_index", "quote", "start_char", "end_char"),
                ("segment_index", "quote", "start_char", "end_char"),
                "occurrence",
            )
            index = occurrence["segment_index"]
            if isinstance(index, bool) or not isinstance(index, int):
                raise RunAssetError("occurrence segment_index must be an integer")
            if index not in segment_text:
                raise RunAssetError(f"occurrence references segment_index {index} outside the run")
            quote = occurrence["quote"]
            start = occurrence["start_char"]
            end = occurrence["end_char"]
            if not isinstance(quote, str) or not quote.strip():
                raise RunAssetError("occurrence quote must be a non-empty string")
            for name, value in (("start_char", start), ("end_char", end)):
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise RunAssetError(f"occurrence {name} must be a non-negative integer")
            segment = segment_text[index]
            if not (start < end <= len(segment)):
                raise RunAssetError(
                    f"occurrence offsets [{start}, {end}) fall outside segment {index}"
                )
            if segment[start:end] != quote:
                # A span that does not reproduce the transcript verbatim is not
                # evidence — the producer's locate step should have dropped it.
                raise RunAssetError(
                    f"occurrence quote does not match segment {index} text at [{start}, {end})"
                )
            if not span_boundaries_ok(segment, start, end):
                raise RunAssetError(
                    f"occurrence span [{start}, {end}) in segment {index} anchors"
                    " inside a longer word"
                )
            if not quote_matches_surface(surface, quote):
                raise RunAssetError(
                    "occurrence quote is unrelated to the mention surface —"
                    " a grounded span must be an occurrence of the entity"
                )


def validate_payload(
    kind: RunAssetKind, payload: Mapping[str, Any], *, source: RunAssetSource
) -> None:
    """Fail-closed per-kind shape validation against the run's actual source."""
    if kind is RunAssetKind.SUMMARY:
        _validate_summary_payload(payload)
    elif kind is RunAssetKind.TOPICS:
        _validate_topics_payload(payload)
    elif kind is RunAssetKind.ENTITY_MENTIONS:
        _validate_mentions_payload(
            payload,
            {segment.segment_index: segment.text for segment in source.segments},
        )
    else:  # pragma: no cover - exhaustive over the enum
        raise RunAssetError(f"unknown asset kind: {kind!r}")


def _validate_record(
    *,
    producer: str,
    producer_version: str,
    model: str,
    payload_schema_version: int,
    hash_value: str,
    idempotency_key: str,
    started_at: datetime,
    completed_at: datetime,
    config: Mapping[str, Any] | None,
    config_schema_version: int | None,
) -> None:
    for label, value, cap in (
        ("producer", producer, MAX_PRODUCER_CHARS),
        ("producer_version", producer_version, MAX_PRODUCER_CHARS),
        ("model", model, MAX_MODEL_CHARS),
    ):
        if not value.strip() or len(value) > cap:
            raise RunAssetError(f"{label} empty or over {cap} chars")
    if payload_schema_version < 1:
        raise RunAssetError("payload_schema_version must be >= 1")
    if len(hash_value) != 64 or any(ch not in "0123456789abcdef" for ch in hash_value):
        raise RunAssetError("source_content_hash must be 64 lowercase hex chars")
    if not idempotency_key.strip():
        raise RunAssetError("idempotency_key must be non-empty")
    for label, stamp in (("started_at", started_at), ("completed_at", completed_at)):
        if stamp.tzinfo is None:
            raise RunAssetError(f"{label} must be timezone-aware")
    if completed_at < started_at:
        raise RunAssetError("completed_at precedes started_at")
    if (config is None) != (config_schema_version is None):
        raise RunAssetError("config and config_schema_version must be set together")
    if config is not None:
        if config_schema_version is not None and config_schema_version < 1:
            raise RunAssetError("config_schema_version must be >= 1")
        try:
            encoded = json.dumps(dict(config), sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise RunAssetError(f"config is not JSON-serializable: {exc}") from exc
        if len(encoded.encode()) > MAX_CONFIG_BYTES:
            raise RunAssetError(f"config over {MAX_CONFIG_BYTES} bytes")


def _canonical_json(value: Mapping[str, Any] | None) -> str:
    return json.dumps(value if value is None else dict(value), sort_keys=True)


def _replay_matches(
    row: RunEnrichmentAsset,
    *,
    pipeline_run_id: uuid.UUID,
    kind: RunAssetKind,
    payload: Mapping[str, Any],
    payload_schema_version: int,
    producer: str,
    producer_version: str,
    model: str,
    hash_value: str,
    started_at: datetime,
    completed_at: datetime,
    config: Mapping[str, Any] | None,
    config_schema_version: int | None,
) -> bool:
    """Full-payload replay equality — ANY divergence is a conflicting key reuse."""
    return (
        row.pipeline_run_id == pipeline_run_id
        and row.asset_kind == kind.value
        and _canonical_json(row.payload) == _canonical_json(payload)
        and row.payload_schema_version == payload_schema_version
        and row.producer == producer
        and row.producer_version == producer_version
        and row.model == model
        and row.source_content_hash == hash_value
        and row.started_at == started_at
        and row.completed_at == completed_at
        and _canonical_json(row.config) == _canonical_json(config)
        and row.config_schema_version == config_schema_version
    )


def record_asset(
    session: Session,
    *,
    source: RunAssetSource,
    kind: RunAssetKind,
    payload: Mapping[str, Any],
    payload_schema_version: int,
    producer: str,
    producer_version: str,
    model: str,
    idempotency_key: str,
    started_at: datetime,
    completed_at: datetime,
    config: Mapping[str, Any] | None = None,
    config_schema_version: int | None = None,
) -> RunEnrichmentAsset:
    """Atomically persist one successful generation and supersede its
    predecessor of the same kind.

    The staleness hash is computed here from ``source`` — the writer, not the
    caller, owns the hash so it always describes what was actually loaded.
    Returns the asset row (the existing one on an identical replay). Raises
    :class:`RunAssetError` for anything malformed and
    :class:`ConflictingReplayError` when ``idempotency_key`` was already used
    with a different payload.
    """
    hash_value = source_content_hash(source)
    _validate_record(
        producer=producer,
        producer_version=producer_version,
        model=model,
        payload_schema_version=payload_schema_version,
        hash_value=hash_value,
        idempotency_key=idempotency_key,
        started_at=started_at,
        completed_at=completed_at,
        config=config,
        config_schema_version=config_schema_version,
    )
    validate_payload(kind, payload, source=source)

    def _existing() -> RunEnrichmentAsset | None:
        return session.execute(
            select(RunEnrichmentAsset).where(RunEnrichmentAsset.idempotency_key == idempotency_key)
        ).scalar_one_or_none()

    def _adopt_or_conflict(row: RunEnrichmentAsset) -> RunEnrichmentAsset:
        if _replay_matches(
            row,
            pipeline_run_id=source.pipeline_run_id,
            kind=kind,
            payload=payload,
            payload_schema_version=payload_schema_version,
            producer=producer,
            producer_version=producer_version,
            model=model,
            hash_value=hash_value,
            started_at=started_at,
            completed_at=completed_at,
            config=config,
            config_schema_version=config_schema_version,
        ):
            return row
        raise ConflictingReplayError(idempotency_key)

    existing = _existing()
    if existing is not None:
        return _adopt_or_conflict(existing)

    # One finalization at a time per (run, kind): generation allocation,
    # insertion, and supersession must be atomic even when the pair has no
    # prior rows to lock. Transaction-scoped, releases on commit/rollback.
    session.execute(
        text("SELECT pg_advisory_xact_lock(hashtext(:producer), hashtext(:scope))"),
        {
            "producer": f"run_assets:{kind.value}",
            "scope": str(source.pipeline_run_id),
        },
    )
    existing = _existing()
    if existing is not None:
        return _adopt_or_conflict(existing)
    generation = (
        session.execute(
            select(func.coalesce(func.max(RunEnrichmentAsset.generation), 0)).where(
                RunEnrichmentAsset.pipeline_run_id == source.pipeline_run_id,
                RunEnrichmentAsset.asset_kind == kind.value,
            )
        ).scalar_one()
        + 1
    )
    asset = RunEnrichmentAsset(
        pipeline_run_id=source.pipeline_run_id,
        asset_kind=kind.value,
        generation=generation,
        payload=dict(payload),
        payload_schema_version=payload_schema_version,
        producer=producer,
        producer_version=producer_version,
        model=model,
        source_content_hash=hash_value,
        config_schema_version=config_schema_version,
        idempotency_key=idempotency_key,
        started_at=started_at,
        completed_at=completed_at,
    )
    # Assign only when present: an explicit ``config=None`` would serialize as
    # a JSON ``null`` (not SQL NULL) and trip the jsonb_typeof CHECK.
    if config is not None:
        asset.config = dict(config)
    try:
        # Savepoint, not a bare flush: losing an idempotency race must not
        # roll back the caller's enclosing transaction (drafts.py pattern).
        with session.begin_nested():
            session.add(asset)
            session.flush()
            session.execute(
                update(RunEnrichmentAsset)
                .where(
                    RunEnrichmentAsset.pipeline_run_id == source.pipeline_run_id,
                    RunEnrichmentAsset.asset_kind == kind.value,
                    RunEnrichmentAsset.generation < generation,
                    RunEnrichmentAsset.superseded_by_asset_id.is_(None),
                )
                .values(superseded_by_asset_id=asset.id)
            )
    except IntegrityError:
        existing = _existing()
        if existing is None:
            raise
        return _adopt_or_conflict(existing)
    return asset


def latest_assets(session: Session, pipeline_run_id: uuid.UUID) -> dict[str, RunEnrichmentAsset]:
    """The current (unsuperseded) asset per kind, keyed by kind value."""
    rows = (
        session.execute(
            select(RunEnrichmentAsset).where(
                RunEnrichmentAsset.pipeline_run_id == pipeline_run_id,
                RunEnrichmentAsset.superseded_by_asset_id.is_(None),
            )
        )
        .scalars()
        .all()
    )
    return {row.asset_kind: row for row in rows}
