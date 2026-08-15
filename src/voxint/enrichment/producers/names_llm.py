"""Optional LLM name pass: strictly additive to the offline producer (#38).

A separate producer (``names.llm``) so it supersedes independently — the
offline path never depends on it and stays fully useful with no LLM
configured. Reuses the enhancement client's hardened contract: the batch
call already extracts ``name_hints`` (label-constrained, strictly parsed by
:mod:`voxint.clients.llm`), so no new prompt or parser surface is added.
Transcript-only in v1 (metadata is already covered deterministically).

Evidence discipline holds for model output: a hint survives only when its
(normalized) name is located verbatim in a real segment's text — in the
hinted label's own segments for ``kind='self'`` (those become run_label
candidates), anywhere in the transcript for ``kind='other'`` (run-level).
Unlocatable names are dropped, not persisted. Every kept candidate carries a
fixed score of 0.5 with ``{"llm": 1.0}`` components — an uncalibrated marker,
deliberately below every strong deterministic pattern.

Same idempotency scheme as the offline producer: an input-signature key over
the model + exact segment content, with a short-circuit before timestamps.
Identical inputs therefore replay the stored result rather than re-querying
the (nondeterministic) endpoint; a changed transcript or model mints a new
superseding generation. An LLM transport/contract failure raises — it must
never be recorded as an authoritative ``outcome='none'``.
"""

import hashlib
import json
import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from voxint.clients.base import EnhancementRequestSegment, LLMClient, SpeakerNameHint
from voxint.clients.llm import HttpLLMClient, LLMError
from voxint.config import Settings
from voxint.db.models import (
    ClaimField,
    EnrichmentProducerRun,
    PipelineRun,
    TranscriptSegment,
)
from voxint.enrichment.drafts import (
    MAX_EVIDENCE_ROWS,
    CandidateDraft,
    EnrichmentScope,
    TranscriptEvidence,
    record_producer_run,
)
from voxint.enrichment.producers.name_patterns import normalize_name
from voxint.enrichment.producers.names import NameProducerError
from voxint.enrichment.review import ConflictingReplayError

LLM_PRODUCER_NAME = "names.llm"
LLM_PRODUCER_VERSION = "1"
LLM_CONFIG_SCHEMA_VERSION = 1
LLM_DETAIL_SCHEMA_VERSION = 1
LLM_CANDIDATE_SCORE = 0.5


def _input_signature(*, run_id: uuid.UUID, model: str, segments: list[TranscriptSegment]) -> str:
    payload: dict[str, object] = {
        "producer": LLM_PRODUCER_NAME,
        "producer_version": LLM_PRODUCER_VERSION,
        "model": model,
        "run_id": str(run_id),
        "segments": [
            {
                "index": segment.segment_index,
                "label": segment.diarization_label,
                "text": segment.enhanced_text or segment.raw_text,
            }
            for segment in segments
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _batches(
    segments: list[TranscriptSegment], *, max_segments: int, max_chars: int
) -> list[tuple[EnhancementRequestSegment, ...]]:
    batches: list[tuple[EnhancementRequestSegment, ...]] = []
    current: list[EnhancementRequestSegment] = []
    chars = 0
    for segment in segments:
        text = segment.enhanced_text or segment.raw_text
        if current and (len(current) >= max_segments or chars + len(text) > max_chars):
            batches.append(tuple(current))
            current, chars = [], 0
        current.append(
            EnhancementRequestSegment(
                segment_index=segment.segment_index,
                text=text,
                diarization_label=segment.diarization_label,
            )
        )
        chars += len(text)
    if current:
        batches.append(tuple(current))
    return batches


def _locate(
    name: str, segments: list[TranscriptSegment], *, label: str | None
) -> TranscriptSegment | None:
    """First segment whose text contains the name verbatim (casefolded).

    ``label`` restricts the search to that diarization label's own segments —
    the requirement for a self-hint to become a cluster-level claim.
    """
    needle = name.casefold()
    for segment in segments:
        if label is not None and segment.diarization_label != label:
            continue
        if needle in (segment.enhanced_text or segment.raw_text).casefold():
            return segment
    return None


def run_llm_name_producer(
    session: Session,
    *,
    run_id: uuid.UUID,
    settings: Settings,
    client: LLMClient | None = None,
) -> EnrichmentProducerRun:
    """Execute the additive LLM name pass over one run and persist the drafts.

    Requires both ``enrichment_names_llm_enabled`` and ``llm_enabled``. Pass
    ``client`` to inject a preconfigured client (tests); otherwise one is
    built from settings and closed here. The caller commits.
    """
    if not (settings.enrichment_names_llm_enabled and settings.llm_enabled):
        raise NameProducerError(
            "the LLM name pass is disabled — set ENRICHMENT_NAMES_LLM_ENABLED=true"
            " and LLM_ENABLED=true"
        )
    started_at = datetime.now(tz=UTC)
    run = session.get(PipelineRun, run_id)
    if run is None:
        raise NameProducerError(f"pipeline run {run_id} not found")
    segments = list(
        session.execute(
            select(TranscriptSegment)
            .where(TranscriptSegment.pipeline_run_id == run_id)
            .order_by(TranscriptSegment.segment_index)
        ).scalars()
    )

    signature = _input_signature(run_id=run_id, model=settings.llm_model, segments=segments)
    idempotency_key = f"{LLM_PRODUCER_NAME}:{run_id}:{signature[:16]}"
    existing = session.execute(
        select(EnrichmentProducerRun).where(
            EnrichmentProducerRun.idempotency_key == idempotency_key
        )
    ).scalar_one_or_none()
    if existing is not None:
        return existing

    hints: list[SpeakerNameHint] = []
    if segments:
        owned = client is None
        llm: LLMClient = client or HttpLLMClient(
            settings.llm_base_url,
            settings.llm_model,
            settings.llm_api_key,
            settings.llm_timeout_seconds,
        )
        try:
            for batch in _batches(
                segments,
                max_segments=settings.llm_batch_max_segments,
                max_chars=settings.llm_batch_max_chars,
            ):
                try:
                    hints.extend(llm.enhance_segments(batch, "").name_hints)
                except LLMError as exc:
                    # Abort, never a false authoritative 'none' — a 'none'
                    # generation would retire the prior LLM proposals.
                    raise NameProducerError(f"LLM name pass failed: {exc}") from exc
        finally:
            if owned and isinstance(llm, HttpLLMClient):
                llm.close()

    # One candidate per (target, name); evidence rows accumulate per hint.
    grouped: dict[tuple[str | None, str], tuple[str, list[TranscriptEvidence]]] = {}
    for hint in hints:
        name = normalize_name(hint.name)
        if name is None:
            continue
        label = hint.diarization_label if hint.kind == "self" else None
        located = _locate(name, segments, label=label)
        if located is None:
            continue  # unlocatable model output is dropped, not persisted
        evidence = TranscriptEvidence(
            transcript_segment_id=located.id,
            timestamp_seconds=located.start_seconds,
            snippet=(located.enhanced_text or located.raw_text)[:200],
            detail={
                "pattern_id": "llm_extraction",
                "kind": hint.kind,
                "model": settings.llm_model,
            },
            detail_schema_version=LLM_DETAIL_SCHEMA_VERSION,
        )
        key = (label, name.casefold())
        if key in grouped:
            grouped[key][1].append(evidence)
        else:
            grouped[key] = (name, [evidence])

    drafts = tuple(
        CandidateDraft(
            target=(
                EnrichmentScope.run_label(run_id, label)
                if label is not None
                else EnrichmentScope.run(run_id)
            ),
            field=ClaimField.NAME,
            value=name,
            evidence=tuple(evidence_rows[:MAX_EVIDENCE_ROWS]),
            score=LLM_CANDIDATE_SCORE,
            score_components={"llm": 1.0},
        )
        for (label, _), (name, evidence_rows) in sorted(
            grouped.items(), key=lambda item: (item[0][0] or "", item[0][1])
        )
    )

    config = {
        "producer_version": LLM_PRODUCER_VERSION,
        "model": settings.llm_model,
        "input_signature": signature,
    }
    try:
        return record_producer_run(
            session,
            producer=LLM_PRODUCER_NAME,
            producer_version=LLM_PRODUCER_VERSION,
            scope=EnrichmentScope.run(run_id),
            covered_fields=(ClaimField.NAME,),
            candidates=drafts,
            idempotency_key=idempotency_key,
            started_at=started_at,
            completed_at=datetime.now(tz=UTC),
            config=config,
            config_schema_version=LLM_CONFIG_SCHEMA_VERSION,
        )
    except ConflictingReplayError:
        raced = session.execute(
            select(EnrichmentProducerRun).where(
                EnrichmentProducerRun.idempotency_key == idempotency_key
            )
        ).scalar_one_or_none()
        if raced is not None:
            return raced
        raise
