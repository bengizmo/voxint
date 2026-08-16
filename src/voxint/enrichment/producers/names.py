"""Offline name-candidate producer: orchestration for issue #38.

Loads one run's #36 metadata snapshot and transcript segments, runs the pure
extraction (:mod:`name_patterns`) and scoring (:mod:`name_scoring`) layers,
and persists the result through the single sanctioned writer
(:func:`voxint.enrichment.drafts.record_producer_run`) at **run scope** —
always. Run-scope invocations may emit both run-level and run_label-level
candidates, and supersession keys on the invocation scope, so every rerun
cleanly retires the previous generation's still-proposed claims.

Correctness notes:

- **Evidence-to-run ownership**: the schema does not verify that evidence
  rows belong to the candidate's run, so the queries here are the guarantee —
  metadata is joined through the run's ``media_item_id`` and segments are
  filtered by ``pipeline_run_id`` exactly.
- **Idempotency key = input signature.** The key hashes the producer/pattern/
  scoring versions, the domain-pack seeds, and the exact metadata + segment
  content. Identical inputs produce the identical key, and the producer
  short-circuits to the existing row *before* minting fresh timestamps — a
  reused key must never reach the writer with a divergent payload
  (``ConflictingReplayError``). Changed inputs produce a new key and a new
  generation that supersedes the old one.
- **outcome='none' is authoritative.** A run with no extractable names is
  recorded (it retires prior proposals: "we looked again, found nothing").
  Read failures raise instead — they must never masquerade as 'none'.
- Extraction happens before the persistence call; the caller commits.
"""

import hashlib
import json
import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from voxint.config import Settings
from voxint.db.models import (
    ClaimField,
    EnrichmentProducerRun,
    MediaSourceMetadata,
    PipelineRun,
    TranscriptSegment,
)
from voxint.domain_packs.registry import domain_pack_from_snapshot
from voxint.enrichment.drafts import (
    MAX_EVIDENCE_ROWS,
    CandidateDraft,
    EnrichmentScope,
    Evidence,
    MetadataEvidence,
    TranscriptEvidence,
    record_producer_run,
)
from voxint.enrichment.producers.name_patterns import (
    PATTERN_SET_VERSION,
    MetadataRef,
    RawMention,
    extract_from_metadata,
    extract_from_segment,
)
from voxint.enrichment.producers.name_scoring import (
    SCORING_VERSION,
    CandidateLevel,
    NameCandidate,
    aggregate,
)
from voxint.enrichment.review import ConflictingReplayError

PRODUCER_NAME = "names.offline"
PRODUCER_VERSION = "1"
CONFIG_SCHEMA_VERSION = 1
DETAIL_SCHEMA_VERSION = 1


class NameProducerError(Exception):
    """The producer could not complete an authoritative scan."""


def _input_signature(
    *,
    run_id: uuid.UUID,
    name_seeds: tuple[str, ...],
    metadata: MediaSourceMetadata | None,
    segments: list[TranscriptSegment],
) -> str:
    """Content hash of everything that determines this producer's output."""
    payload: dict[str, object] = {
        "producer": PRODUCER_NAME,
        "producer_version": PRODUCER_VERSION,
        "pattern_set_version": PATTERN_SET_VERSION,
        "scoring_version": SCORING_VERSION,
        "run_id": str(run_id),
        "name_seeds": sorted(seed.casefold() for seed in name_seeds),
        "metadata": None
        if metadata is None
        else {
            "title": metadata.title,
            "description": metadata.description,
            "channel": metadata.channel,
            "uploader": metadata.uploader,
            "tags": list(metadata.tags or []),
        },
        "segments": [
            {
                "index": segment.segment_index,
                "label": segment.diarization_label,
                "text": segment.enhanced_text or segment.raw_text,
                "suspect": segment.suspect,
                "start": segment.start_seconds,
            }
            for segment in segments
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _evidence_for(
    mention: RawMention,
    *,
    metadata: MediaSourceMetadata | None,
    segments_by_index: dict[int, TranscriptSegment],
) -> Evidence:
    detail: dict[str, object] = {
        "pattern_id": mention.pattern_id,
        "reliability": mention.reliability,
        "attribution": mention.attribution.value,
        "matched": mention.raw_span,
    }
    if isinstance(mention.source, MetadataRef):
        if metadata is None:  # pragma: no cover — extraction implies a snapshot
            raise NameProducerError("metadata mention without a metadata snapshot")
        if mention.source.item_index is not None:
            detail["item_index"] = mention.source.item_index
        return MetadataEvidence(
            source_metadata_id=metadata.id,
            source_field=mention.source.field,
            snippet=mention.snippet or None,
            detail=detail,
            detail_schema_version=DETAIL_SCHEMA_VERSION,
        )
    segment = segments_by_index.get(mention.source.segment_index)
    if segment is None:  # pragma: no cover — mentions come from these segments
        raise NameProducerError(
            f"transcript mention references unknown segment {mention.source.segment_index}"
        )
    if mention.source.suspect:
        detail["suspect"] = True
    return TranscriptEvidence(
        transcript_segment_id=segment.id,
        timestamp_seconds=mention.source.start_seconds,
        snippet=mention.snippet or None,
        detail=detail,
        detail_schema_version=DETAIL_SCHEMA_VERSION,
    )


def _draft_for(
    candidate: NameCandidate,
    *,
    run_id: uuid.UUID,
    metadata: MediaSourceMetadata | None,
    segments_by_index: dict[int, TranscriptSegment],
) -> CandidateDraft:
    if candidate.level is CandidateLevel.RUN_LABEL:
        if candidate.diarization_label is None:  # pragma: no cover — scoring invariant
            raise NameProducerError("run_label candidate without a label")
        target = EnrichmentScope.run_label(run_id, candidate.diarization_label)
    else:
        target = EnrichmentScope.run(run_id)
    # Mentions are already in deterministic strongest-first order; the cap
    # keeps the strongest rows and the full count stays in score_components.
    evidence = tuple(
        _evidence_for(mention, metadata=metadata, segments_by_index=segments_by_index)
        for mention in candidate.mentions[:MAX_EVIDENCE_ROWS]
    )
    return CandidateDraft(
        target=target,
        field=ClaimField.NAME,
        value=candidate.name,
        evidence=evidence,
        score=candidate.score,
        score_components=candidate.score_components,
    )


def run_offline_name_producer(
    session: Session,
    *,
    run_id: uuid.UUID,
    settings: Settings,
) -> EnrichmentProducerRun:
    """Execute one offline sweep over a run and persist the draft claims.

    Returns the (possibly replayed) producer run; the caller commits.
    Raises :class:`NameProducerError` when the run does not exist — and lets
    read errors propagate rather than recording a false ``outcome='none'``.
    """
    run = session.get(PipelineRun, run_id)
    if run is None:
        raise NameProducerError(f"pipeline run {run_id} not found")

    # Ownership joins: metadata through the run's media item, segments by run.
    metadata = session.execute(
        select(MediaSourceMetadata).where(MediaSourceMetadata.media_item_id == run.media_item_id)
    ).scalar_one_or_none()
    segments = list(
        session.execute(
            select(TranscriptSegment)
            .where(TranscriptSegment.pipeline_run_id == run_id)
            .order_by(TranscriptSegment.segment_index)
        ).scalars()
    )

    # Read the pack the run was TRANSCRIBED with (its frozen #11 snapshot), not the
    # mutable global env — so late enrichment can never diverge from transcription,
    # and the idempotency signature (which hashes name_seeds) stays stable even if
    # the pack on disk or DOMAIN_PACK_PATH later changes.
    pack = domain_pack_from_snapshot(run.domain_pack, settings)
    signature = _input_signature(
        run_id=run_id, name_seeds=pack.name_seeds, metadata=metadata, segments=segments
    )
    idempotency_key = f"{PRODUCER_NAME}:{run_id}:{signature}"
    existing = session.execute(
        select(EnrichmentProducerRun).where(
            EnrichmentProducerRun.idempotency_key == idempotency_key
        )
    ).scalar_one_or_none()
    if existing is not None:
        return existing
    # Minted only after the short-circuit miss: a reused key must never reach
    # the writer's full-payload replay fingerprint with fresh timestamps.
    started_at = datetime.now(tz=UTC)

    mentions: list[RawMention] = []
    if metadata is not None:
        mentions.extend(
            extract_from_metadata(
                title=metadata.title,
                description=metadata.description,
                channel=metadata.channel,
                uploader=metadata.uploader,
                tags=tuple(metadata.tags or ()),
            )
        )
    for segment in segments:
        text = segment.enhanced_text or segment.raw_text
        mentions.extend(
            extract_from_segment(
                text,
                segment_index=segment.segment_index,
                diarization_label=segment.diarization_label,
                start_seconds=segment.start_seconds,
                suspect=segment.suspect,
            )
        )

    candidates = aggregate(mentions, name_seeds=pack.name_seeds)
    segments_by_index = {segment.segment_index: segment for segment in segments}
    drafts = tuple(
        _draft_for(
            candidate,
            run_id=run_id,
            metadata=metadata,
            segments_by_index=segments_by_index,
        )
        for candidate in candidates
    )

    config = {
        "producer_version": PRODUCER_VERSION,
        "pattern_set_version": PATTERN_SET_VERSION,
        "scoring_version": SCORING_VERSION,
        "domain_pack": pack.name,
        "name_seed_count": len(pack.name_seeds),
        "input_signature": signature,
    }
    try:
        return record_producer_run(
            session,
            producer=PRODUCER_NAME,
            producer_version=PRODUCER_VERSION,
            scope=EnrichmentScope.run(run_id),
            covered_fields=(ClaimField.NAME,),
            candidates=drafts,
            idempotency_key=idempotency_key,
            started_at=started_at,
            completed_at=datetime.now(tz=UTC),
            config=config,
            config_schema_version=CONFIG_SCHEMA_VERSION,
        )
    except ConflictingReplayError:
        # Same inputs raced from two entry points (CLI + console): the stored
        # row differs only in timestamps. The signature guarantees identical
        # inputs, so the existing row is the authoritative result.
        raced = session.execute(
            select(EnrichmentProducerRun).where(
                EnrichmentProducerRun.idempotency_key == idempotency_key
            )
        ).scalar_one_or_none()
        if raced is not None:
            return raced
        raise
