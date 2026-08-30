"""Project-level entity and topic rollups for corpus visualization (issue #336).

The page uses one canonical run per recording: the newest completed,
non-archived run. Current enrichment assets are folded synchronously because
their payloads are bounded and projects are modest in this single-operator
application. The pure aggregators accept parsed payload lists and return only
template-facing scalars; occurrence evidence never reaches the view layer.
"""

from __future__ import annotations

import logging
import math
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from voxint.db.models import (
    MediaFolder,
    MediaItem,
    PipelineRun,
    RunAssetKind,
    RunEnrichmentAsset,
    RunStatus,
)

logger = logging.getLogger(__name__)

ENTITY_KIND_ORDER = ("person", "organization", "product", "other")
_ENTITY_KIND_PRIORITY = {kind: index for index, kind in enumerate(ENTITY_KIND_ORDER)}


@dataclass(frozen=True)
class EntityInsight:
    """One normalized entity and its project-level frequency."""

    label: str
    kind: str
    run_count: int
    occurrence_count: int


@dataclass(frozen=True)
class TopicInsight:
    """One normalized topic and the recordings in which it appears."""

    label: str
    run_count: int
    description: str | None


@dataclass(frozen=True)
class EnrichmentCoverage:
    """Per-kind asset coverage over one shared canonical-recording count."""

    entity_runs: int
    topic_runs: int
    total_runs: int


@dataclass(frozen=True)
class ProjectInsights:
    """The complete template-facing project insight bundle."""

    entities: dict[str, list[EntityInsight]]
    topics: list[TopicInsight]
    coverage: EnrichmentCoverage
    has_any_asset: bool
    has_assets_but_no_items: bool


def _display_label(casings: Counter[str]) -> str:
    """Choose the plurality casing with stable lexical tie-breaks."""
    return min(casings, key=lambda label: (-casings[label], label.casefold(), label))


def _entity_kind(value: object) -> str:
    if isinstance(value, str) and value in _ENTITY_KIND_PRIORITY:
        return value
    return "other"


def _majority_kind(kind_counts: Counter[str]) -> str:
    return min(
        kind_counts,
        key=lambda kind: (-kind_counts[kind], _ENTITY_KIND_PRIORITY[kind]),
    )


def aggregate_entities(
    per_run_mentions: list[tuple[uuid.UUID, list[object]]],
) -> dict[str, list[EntityInsight]]:
    """Aggregate entity mentions into the top 12 entries for each kind.

    Surfaces merge only by ``strip().casefold()``. Each recording contributes
    once to run count and once to the final kind vote for a normalized entity;
    duplicate entries within a recording resolve their local kind by majority
    before that vote.
    """
    casings: dict[str, Counter[str]] = defaultdict(Counter)
    occurrences: Counter[str] = Counter()
    runs: dict[str, set[uuid.UUID]] = defaultdict(set)
    run_kinds: dict[str, dict[uuid.UUID, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )

    for run_id, mentions in per_run_mentions:
        for mention in mentions:
            if not isinstance(mention, dict):
                logger.debug("Skipping malformed entity mention: not an object")
                continue
            surface = mention.get("surface")
            if not isinstance(surface, str) or not surface.strip():
                logger.debug("Skipping malformed entity mention: invalid surface")
                continue
            label = surface.strip()
            key = label.casefold()
            raw_occurrences = mention.get("occurrences")
            occurrence_count = len(raw_occurrences) if isinstance(raw_occurrences, list) else 1

            casings[key][label] += 1
            occurrences[key] += occurrence_count
            runs[key].add(run_id)
            run_kinds[key][run_id][_entity_kind(mention.get("kind"))] += 1

    grouped: dict[str, list[EntityInsight]] = {kind: [] for kind in ENTITY_KIND_ORDER}
    for key, casing_counts in casings.items():
        kind_votes: Counter[str] = Counter()
        for per_run_counts in run_kinds[key].values():
            kind_votes[_majority_kind(per_run_counts)] += 1
        kind = _majority_kind(kind_votes)
        grouped[kind].append(
            EntityInsight(
                label=_display_label(casing_counts),
                kind=kind,
                run_count=len(runs[key]),
                occurrence_count=occurrences[key],
            )
        )

    for kind in ENTITY_KIND_ORDER:
        grouped[kind].sort(
            key=lambda item: (
                -item.run_count,
                -item.occurrence_count,
                item.label.casefold(),
                item.label,
            )
        )
        grouped[kind] = grouped[kind][:12]
    return grouped


def _topic_confidence(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return -math.inf
    confidence = float(value)
    return confidence if math.isfinite(confidence) else -math.inf


def aggregate_topics(
    per_run_topics: list[tuple[uuid.UUID, list[object]]],
) -> list[TopicInsight]:
    """Aggregate casefolded topics, returning the top ten by recording count."""
    casings: dict[str, Counter[str]] = defaultdict(Counter)
    runs: dict[str, set[uuid.UUID]] = defaultdict(set)
    descriptions: dict[str, tuple[float, int, str | None]] = {}
    seen_index = 0

    for run_id, topics in per_run_topics:
        for topic in topics:
            if not isinstance(topic, dict):
                logger.debug("Skipping malformed topic: not an object")
                continue
            raw_label = topic.get("label")
            if not isinstance(raw_label, str) or not raw_label.strip():
                logger.debug("Skipping malformed topic: invalid label")
                continue
            label = raw_label.strip()
            key = label.casefold()
            description_value = topic.get("description")
            description = description_value if isinstance(description_value, str) else None
            confidence = _topic_confidence(topic.get("confidence"))

            casings[key][label] += 1
            runs[key].add(run_id)
            candidate = (confidence, -seen_index, description)
            current = descriptions.get(key)
            if current is None or candidate[:2] > current[:2]:
                descriptions[key] = candidate
            seen_index += 1

    insights = [
        TopicInsight(
            label=_display_label(casing_counts),
            run_count=len(runs[key]),
            description=descriptions[key][2],
        )
        for key, casing_counts in casings.items()
    ]
    insights.sort(key=lambda item: (-item.run_count, item.label.casefold(), item.label))
    return insights[:10]


def _canonical_run_ids(session: Session, project_id: uuid.UUID) -> list[uuid.UUID]:
    """Newest completed, non-archived run per project recording."""
    ranked = (
        select(
            PipelineRun.id.label("run_id"),
            func.row_number()
            .over(
                partition_by=PipelineRun.media_item_id,
                order_by=(PipelineRun.created_at.desc(), PipelineRun.id.desc()),
            )
            .label("rank"),
        )
        .join(MediaItem, MediaItem.id == PipelineRun.media_item_id)
        .join(MediaFolder, MediaFolder.id == MediaItem.media_folder_id)
        .where(
            MediaFolder.project_id == project_id,
            PipelineRun.status == RunStatus.COMPLETED.value,
            PipelineRun.archived_at.is_(None),
        )
        .subquery()
    )
    return list(
        session.execute(
            select(ranked.c.run_id).where(ranked.c.rank == 1).order_by(ranked.c.run_id)
        ).scalars()
    )


def _payload_entries(payload: object, key: str, *, run_id: uuid.UUID) -> list[object]:
    if not isinstance(payload, dict):
        logger.debug(
            "Skipping malformed %s asset for run %s: payload is not an object",
            key,
            run_id,
        )
        return []
    entries = payload.get(key)
    if not isinstance(entries, list):
        logger.debug("Skipping malformed %s asset for run %s: entries are not a list", key, run_id)
        return []
    return entries


def compute_project_insights(session: Session, project_id: uuid.UUID) -> ProjectInsights:
    """Compute current entity/topic rollups for a project's canonical runs."""
    run_ids = _canonical_run_ids(session, project_id)
    if not run_ids:
        return ProjectInsights(
            entities={kind: [] for kind in ENTITY_KIND_ORDER},
            topics=[],
            coverage=EnrichmentCoverage(entity_runs=0, topic_runs=0, total_runs=0),
            has_any_asset=False,
            has_assets_but_no_items=False,
        )

    rows = session.execute(
        select(RunEnrichmentAsset)
        .where(
            RunEnrichmentAsset.pipeline_run_id.in_(run_ids),
            RunEnrichmentAsset.asset_kind.in_(
                (RunAssetKind.ENTITY_MENTIONS.value, RunAssetKind.TOPICS.value)
            ),
            RunEnrichmentAsset.superseded_by_asset_id.is_(None),
        )
        .order_by(
            RunEnrichmentAsset.pipeline_run_id,
            RunEnrichmentAsset.asset_kind,
            RunEnrichmentAsset.generation.desc(),
            RunEnrichmentAsset.id.desc(),
        )
    ).scalars()

    current: dict[tuple[uuid.UUID, str], RunEnrichmentAsset] = {}
    for row in rows:
        current.setdefault((row.pipeline_run_id, row.asset_kind), row)

    per_run_mentions: list[tuple[uuid.UUID, list[object]]] = []
    per_run_topics: list[tuple[uuid.UUID, list[object]]] = []
    entity_runs: set[uuid.UUID] = set()
    topic_runs: set[uuid.UUID] = set()
    for (run_id, kind), asset in current.items():
        if kind == RunAssetKind.ENTITY_MENTIONS.value:
            entity_runs.add(run_id)
            per_run_mentions.append(
                (run_id, _payload_entries(asset.payload, "mentions", run_id=run_id))
            )
        elif kind == RunAssetKind.TOPICS.value:
            topic_runs.add(run_id)
            per_run_topics.append(
                (run_id, _payload_entries(asset.payload, "topics", run_id=run_id))
            )

    entities = aggregate_entities(per_run_mentions)
    topics = aggregate_topics(per_run_topics)
    has_any_asset = bool(current)
    has_items = bool(topics) or any(entities.values())
    return ProjectInsights(
        entities=entities,
        topics=topics,
        coverage=EnrichmentCoverage(
            entity_runs=len(entity_runs),
            topic_runs=len(topic_runs),
            total_runs=len(run_ids),
        ),
        has_any_asset=has_any_asset,
        has_assets_but_no_items=has_any_asset and not has_items,
    )
