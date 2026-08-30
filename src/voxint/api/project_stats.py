"""Project-level statistics for corpus visualization (issue #336).

Pure computation: entity rollup, topic rollup, coverage matrix.
No DB or framework imports.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass
class EntityStat:
    key: str
    kind: str | None
    display_surface: str
    run_count: int
    occurrence_count: int


@dataclass
class TopicStat:
    key: str
    label: str
    description: str | None
    run_count: int
    confidence_mean: float
    confidence_max: float


@dataclass
class SpeakerPresence:
    speaker_id: str
    label: str
    segment_count: int


@dataclass
class CoverageItem:
    media_item_id: str
    run_id: str
    title: str
    duration_s: int | None
    speakers: list[SpeakerPresence]


@dataclass
class CoverageCell:
    speaker_idx: int
    recording_idx: int
    segment_count: int


@dataclass
class CoverageMatrix:
    speakers: list[dict[str, Any]]
    recordings: list[dict[str, Any]]
    cells: list[CoverageCell]
    stats: dict[str, int]


def _normalize_surface(surface: str) -> str:
    return " ".join(surface.casefold().split())


def _pick_display(surfaces: list[str]) -> str:
    counts: Counter[str] = Counter(surfaces)
    max_count = max(counts.values())
    candidates = [s for s, c in counts.items() if c == max_count]
    candidates.sort(key=lambda s: (-len(s), s))
    return candidates[0]


def compute_entity_stats(
    enrichments: list[tuple[str, dict[str, Any]]],
) -> list[EntityStat]:
    groups: dict[tuple[str | None, str], dict[str, Any]] = {}

    for run_id, payload in enrichments:
        for mention in payload.get("mentions", []):
            surface = mention.get("surface", "")
            kind = mention.get("kind")
            key = _normalize_surface(surface)
            if not key:
                continue
            group_key = (kind, key)
            if group_key not in groups:
                groups[group_key] = {
                    "run_ids": set(),
                    "surfaces": [],
                    "occurrence_count": 0,
                }
            group = groups[group_key]
            group["run_ids"].add(run_id)
            group["surfaces"].append(surface)
            occurrences = mention.get("occurrences", [])
            group["occurrence_count"] += max(len(occurrences), 1)

    results: list[EntityStat] = []
    for (kind, key), group in groups.items():
        results.append(EntityStat(
            key=key,
            kind=kind,
            display_surface=_pick_display(group["surfaces"]),
            run_count=len(group["run_ids"]),
            occurrence_count=group["occurrence_count"],
        ))

    results.sort(key=lambda e: (-e.run_count, -e.occurrence_count, e.key))
    return results[:50]


def compute_topic_stats(
    enrichments: list[tuple[str, dict[str, Any]]],
) -> list[TopicStat]:
    groups: dict[str, dict[str, Any]] = {}

    for run_id, payload in enrichments:
        seen_in_run: set[str] = set()
        for topic in payload.get("topics", []):
            label = topic.get("label", "")
            key = _normalize_surface(label)
            if not key:
                continue
            description = topic.get("description")
            confidence = topic.get("confidence")

            if key not in groups:
                groups[key] = {
                    "run_ids": set(),
                    "labels": [],
                    "confidences": [],
                    "best_description": None,
                    "best_confidence": None,
                    "first_description": None,
                }
            group = groups[key]

            if key not in seen_in_run:
                group["run_ids"].add(run_id)
                seen_in_run.add(key)

            group["labels"].append(label)
            if group["first_description"] is None and description is not None:
                group["first_description"] = description
            if confidence is not None:
                group["confidences"].append(confidence)
                if (
                    group["best_confidence"] is None
                    or confidence > group["best_confidence"]
                ):
                    group["best_confidence"] = confidence
                    group["best_description"] = description

    results: list[TopicStat] = []
    for key, group in groups.items():
        confidences = group["confidences"]
        confidence_mean = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )
        confidence_max = max(confidences) if confidences else 0.0
        description = group["best_description"] or group["first_description"]

        results.append(TopicStat(
            key=key,
            label=_pick_display(group["labels"]),
            description=description,
            run_count=len(group["run_ids"]),
            confidence_mean=round(confidence_mean, 4),
            confidence_max=round(confidence_max, 4),
        ))

    results.sort(key=lambda t: (-t.run_count, t.label))
    return results[:30]


def compute_coverage_matrix(
    items: list[CoverageItem],
) -> CoverageMatrix:
    speaker_totals: dict[str, int] = {}
    speaker_labels: dict[str, str] = {}
    speaker_runs: dict[str, set[str]] = {}

    for item in items:
        for sp in item.speakers:
            speaker_totals[sp.speaker_id] = (
                speaker_totals.get(sp.speaker_id, 0) + sp.segment_count
            )
            speaker_labels[sp.speaker_id] = sp.label
            if sp.speaker_id not in speaker_runs:
                speaker_runs[sp.speaker_id] = set()
            speaker_runs[sp.speaker_id].add(item.run_id)

    sorted_speaker_ids = sorted(
        speaker_totals.keys(),
        key=lambda sid: (-speaker_totals[sid], speaker_labels[sid]),
    )

    speaker_id_to_idx = {sid: idx for idx, sid in enumerate(sorted_speaker_ids)}

    speakers_list = [
        {
            "id": sid,
            "label": speaker_labels[sid],
            "run_count": len(speaker_runs[sid]),
        }
        for sid in sorted_speaker_ids
    ]

    recordings_list = [
        {
            "media_item_id": item.media_item_id,
            "run_id": item.run_id,
            "title": item.title,
            "duration_s": item.duration_s,
        }
        for item in items
    ]

    cells: list[CoverageCell] = []
    for rec_idx, item in enumerate(items):
        for sp in item.speakers:
            if sp.segment_count > 0:
                cells.append(CoverageCell(
                    speaker_idx=speaker_id_to_idx[sp.speaker_id],
                    recording_idx=rec_idx,
                    segment_count=sp.segment_count,
                ))

    recordings_with_speakers = len({c.recording_idx for c in cells})

    return CoverageMatrix(
        speakers=speakers_list,
        recordings=recordings_list,
        cells=cells,
        stats={
            "speaker_count": len(sorted_speaker_ids),
            "recording_count": len(items),
            "recordings_with_speakers": recordings_with_speakers,
            "covered_cells": len(cells),
        },
    )


def build_project_insights_payload(
    entities: list[EntityStat],
    topics: list[TopicStat],
    coverage: CoverageMatrix,
    *,
    run_count: int,
    runs_with_entities: int,
    runs_with_topics: int,
) -> dict[str, Any]:
    return {
        "version": 1,
        "stats": {
            "run_count": run_count,
            "runs_with_entities": runs_with_entities,
            "runs_with_topics": runs_with_topics,
            "runs_with_speakers": coverage.stats["recordings_with_speakers"],
            "speaker_count": coverage.stats["speaker_count"],
            "media_item_count": len(coverage.recordings),
        },
        "entities": [
            {
                "key": e.key,
                "kind": e.kind,
                "display_surface": e.display_surface,
                "run_count": e.run_count,
                "occurrence_count": e.occurrence_count,
            }
            for e in entities
        ],
        "topics": [
            {
                "key": t.key,
                "label": t.label,
                "description": t.description,
                "run_count": t.run_count,
                "confidence_mean": t.confidence_mean,
                "confidence_max": t.confidence_max,
            }
            for t in topics
        ],
        "coverage": {
            "speakers": coverage.speakers,
            "recordings": coverage.recordings,
            "cells": [
                {
                    "speaker_idx": c.speaker_idx,
                    "recording_idx": c.recording_idx,
                    "segment_count": c.segment_count,
                }
                for c in coverage.cells
            ],
        },
    }
