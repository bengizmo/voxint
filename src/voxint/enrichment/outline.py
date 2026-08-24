"""Navigable outline (issue #87): grounded entity-mention jump targets.

The review console computes run-level enrichment (summary, topics, grounded
entity mentions) but it sits disconnected from the player. This module turns the
one *grounded* part, entity mentions anchored to a ``segment_index``, into
validated jump targets: for each occurrence it resolves the segment's immutable
``start_seconds`` so the client can seek the player there. Summaries and topics
have no source spans, so they travel as inert context, never as click-to-seek
navigation.

The honesty contract (see the issue and ``docs/`` operator copy):

* ``start_seconds`` is grounded truth. It comes from ASR word timings and is
  never touched by corrections, enhancement, or splits (splits are a read-time
  projection over immutable parent segments). So a jump target stays honest even
  when the stored ``quote`` text has drifted since generation.
* An occurrence whose ``segment_index`` no longer resolves to a live segment is
  dropped and counted, never rendered as fake navigation.
* Asset-level staleness (the source text changed since generation) is disclosed
  once, at the panel level, not stamped on every entry.

Everything above ``build_outline`` is PURE, no session and no I/O, so the whole
resolve/group/dedup/order/drop table is unit-testable without a database.
``build_outline`` is the thin session read that loads the assets and the segment
timing map, then calls the pure core and shapes the camelCase props the island
hydrates from.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from voxint.app_settings import get_app_settings
from voxint.config import Settings
from voxint.db.models import RunAssetKind, TranscriptSegment
from voxint.enrichment.asset_jobs import run_asset_gates_open
from voxint.enrichment.run_assets import latest_assets, load_source, source_content_hash

OUTLINE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class OutlineOccurrence:
    """One grounded jump target: a segment start plus the quote as generated."""

    segment_index: int
    start_seconds: float
    quote: str


@dataclass(frozen=True)
class OutlineMention:
    """A distinct entity ``(surface, kind)`` with its surviving occurrences."""

    surface: str
    kind: str | None
    occurrences: tuple[OutlineOccurrence, ...]


@dataclass(frozen=True)
class OutlineContext:
    """Ungrounded descriptive assets, rendered as inert context only."""

    summary: str | None
    topics: tuple[str, ...]


@dataclass(frozen=True)
class OutlineDiagnostics:
    """Counts of mentions the operator does not see, so completeness is honest.

    ``dropped_unlocatable`` and ``dropped_out_of_run`` are carried from the
    generator (offsets it could not locate, or references outside the run).
    ``dropped_unresolved`` is this module's read-time drop: an occurrence whose
    ``segment_index`` no longer maps to a live segment.
    """

    dropped_unlocatable: int
    dropped_out_of_run: int
    dropped_unresolved: int


@dataclass(frozen=True)
class Outline:
    """The resolved outline. ``available`` is whether an entity_mentions asset
    exists at all; ``gated`` is whether the feature flags are off (the two drive
    distinct honest empty-state copy); ``asset_stale`` drives the single panel
    banner.
    """

    available: bool
    gated: bool
    asset_stale: bool
    mentions: tuple[OutlineMention, ...]
    context: OutlineContext
    diagnostics: OutlineDiagnostics

    def to_props(self) -> dict[str, Any]:
        """The camelCase JSON hydrated into ``island_props["outline"]``."""

        return {
            "available": self.available,
            "gated": self.gated,
            "assetStale": self.asset_stale,
            "mentions": [
                {
                    "surface": mention.surface,
                    "kind": mention.kind,
                    "occurrences": [
                        {
                            "segmentIndex": occ.segment_index,
                            "startSeconds": occ.start_seconds,
                            "quote": occ.quote,
                        }
                        for occ in mention.occurrences
                    ],
                }
                for mention in self.mentions
            ],
            "context": {
                "summary": self.context.summary,
                "topics": list(self.context.topics),
            },
            "diagnostics": {
                "droppedUnlocatable": self.diagnostics.dropped_unlocatable,
                "droppedOutOfRun": self.diagnostics.dropped_out_of_run,
                "droppedUnresolved": self.diagnostics.dropped_unresolved,
            },
        }


def _context_from_payloads(
    summary_payload: Mapping[str, Any] | None,
    topics_payload: Mapping[str, Any] | None,
) -> OutlineContext:
    summary: str | None = None
    if summary_payload is not None:
        value = summary_payload.get("summary")
        if isinstance(value, str) and value.strip():
            summary = value

    topics: list[str] = []
    if topics_payload is not None:
        raw = topics_payload.get("topics")
        if isinstance(raw, Sequence) and not isinstance(raw, str | bytes):
            for topic in raw:
                if isinstance(topic, Mapping):
                    label = topic.get("label")
                    if isinstance(label, str) and label.strip():
                        topics.append(label)
    return OutlineContext(summary=summary, topics=tuple(topics))


def _int(value: Any) -> int | None:
    # bool is an int subclass; a stray True must not read as segment 1.
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def resolve_outline(
    mentions_payload: Mapping[str, Any] | None,
    summary_payload: Mapping[str, Any] | None,
    topics_payload: Mapping[str, Any] | None,
    segment_starts: Mapping[int, float],
    *,
    asset_stale: bool,
    gated: bool,
) -> Outline:
    """Pure core. Group mentions by ``(surface, kind)``, resolve every occurrence
    against ``segment_starts``, drop-and-count the unresolvable ones, order for
    chronological reading, and fold in the inert context.

    ``mentions_payload is None`` means no entity_mentions asset exists
    (``available=False``); an empty ``mentions`` list means the asset ran and
    found nothing. Both render an honest empty state; the caller distinguishes
    them via ``available`` together with the mention count.
    """

    context = _context_from_payloads(summary_payload, topics_payload)

    if mentions_payload is None:
        return Outline(
            available=False,
            gated=gated,
            asset_stale=False,
            mentions=(),
            context=context,
            diagnostics=OutlineDiagnostics(0, 0, 0),
        )

    raw_diag = mentions_payload.get("diagnostics")
    dropped_unlocatable = 0
    dropped_out_of_run = 0
    if isinstance(raw_diag, Mapping):
        dropped_unlocatable = _int(raw_diag.get("dropped_unlocatable")) or 0
        dropped_out_of_run = _int(raw_diag.get("dropped_out_of_run")) or 0

    dropped_unresolved = 0

    # Preserve first-appearance order of groups while merging duplicate
    # (surface, kind) keys; dedup occurrences on (segment_index, start_char).
    groups: dict[tuple[str, str | None], list[OutlineOccurrence]] = {}
    order: list[tuple[str, str | None]] = []
    seen_occ: dict[tuple[str, str | None], set[tuple[int, int]]] = {}

    raw_mentions = mentions_payload.get("mentions")
    if not isinstance(raw_mentions, Sequence) or isinstance(raw_mentions, str | bytes):
        raw_mentions = []

    for mention in raw_mentions:
        if not isinstance(mention, Mapping):
            continue
        surface = mention.get("surface")
        if not isinstance(surface, str) or not surface:
            continue
        kind_raw = mention.get("kind")
        kind = kind_raw if isinstance(kind_raw, str) else None
        key = (surface, kind)

        raw_occ = mention.get("occurrences")
        if not isinstance(raw_occ, Sequence) or isinstance(raw_occ, str | bytes):
            continue

        for occ in raw_occ:
            if not isinstance(occ, Mapping):
                continue
            seg_index = _int(occ.get("segment_index"))
            if seg_index is None:
                continue
            start = segment_starts.get(seg_index)
            if start is None:
                # The segment no longer resolves (re-transcription produced a
                # different set). Drop it rather than seek to nothing.
                dropped_unresolved += 1
                continue
            start_char = _int(occ.get("start_char")) or 0
            dedup_key = (seg_index, start_char)
            occ_seen = seen_occ.setdefault(key, set())
            if dedup_key in occ_seen:
                continue
            occ_seen.add(dedup_key)
            quote = occ.get("quote")
            occurrence = OutlineOccurrence(
                segment_index=seg_index,
                start_seconds=float(start),
                quote=quote if isinstance(quote, str) else "",
            )
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(occurrence)

    mentions: list[OutlineMention] = []
    for key in order:
        occurrences = groups[key]
        if not occurrences:
            continue
        occurrences.sort(key=lambda o: (o.start_seconds, o.segment_index))
        surface, kind = key
        mentions.append(OutlineMention(surface=surface, kind=kind, occurrences=tuple(occurrences)))

    # Chronological entry into the transcript; stable tiebreak on surface.
    mentions.sort(key=lambda m: (m.occurrences[0].start_seconds, m.surface))

    return Outline(
        available=True,
        gated=gated,
        asset_stale=asset_stale,
        mentions=tuple(mentions),
        context=context,
        diagnostics=OutlineDiagnostics(
            dropped_unlocatable=dropped_unlocatable,
            dropped_out_of_run=dropped_out_of_run,
            dropped_unresolved=dropped_unresolved,
        ),
    )


def _segment_starts(session: Session, pipeline_run_id: uuid.UUID) -> dict[int, float]:
    rows = session.execute(
        select(TranscriptSegment.segment_index, TranscriptSegment.start_seconds).where(
            TranscriptSegment.pipeline_run_id == pipeline_run_id
        )
    ).all()
    return {index: start for index, start in rows}


def build_outline(
    session: Session, pipeline_run_id: uuid.UUID, settings: Settings
) -> dict[str, Any]:
    """Thin session read behind the pure core. Loads the current per-kind assets,
    computes asset-level staleness and the gate state, resolves against live
    segment timings, and returns the island props.
    """

    assets = latest_assets(session, pipeline_run_id)
    mentions_asset = assets.get(RunAssetKind.ENTITY_MENTIONS.value)
    summary_asset = assets.get(RunAssetKind.SUMMARY.value)
    topics_asset = assets.get(RunAssetKind.TOPICS.value)

    gated = not run_asset_gates_open(settings, get_app_settings(session))

    mentions_payload: Mapping[str, Any] | None = None
    asset_stale = False
    if mentions_asset is not None:
        mentions_payload = mentions_asset.payload
        current_hash = source_content_hash(load_source(session, pipeline_run_id))
        asset_stale = mentions_asset.source_content_hash != current_hash

    outline = resolve_outline(
        mentions_payload,
        summary_asset.payload if summary_asset is not None else None,
        topics_asset.payload if topics_asset is not None else None,
        _segment_starts(session, pipeline_run_id),
        asset_stale=asset_stale,
        gated=gated,
    )
    return outline.to_props()
