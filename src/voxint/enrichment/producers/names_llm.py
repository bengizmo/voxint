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
import re
import unicodedata
import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from voxint.adjudication.resolver import review_states
from voxint.adjudication.transcript import effective_text
from voxint.app_settings import (
    get_app_settings,
    resolve_effective_enrichment_names_llm_enabled,
    resolve_effective_llm_api_key,
    resolve_effective_llm_enabled,
    resolve_effective_llm_endpoint,
)
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


def _input_signature(
    *,
    run_id: uuid.UUID,
    settings: Settings,
    segments: list[TranscriptSegment],
    corrected: dict[uuid.UUID, str | None],
) -> str:
    # Batching shapes the LLM's per-request context (and therefore its hints),
    # and the same model name can front different endpoints — both belong in
    # the replay identity alongside the exact transcript content. ``corrected``
    # folds operator corrections (issue #58, D2) into that content via the shared
    # effective_text selector, so a correction honestly re-mines names.
    payload: dict[str, object] = {
        "producer": LLM_PRODUCER_NAME,
        "producer_version": LLM_PRODUCER_VERSION,
        "model": settings.llm_model,
        "base_url": settings.llm_base_url,
        "batch_max_segments": settings.llm_batch_max_segments,
        "batch_max_chars": settings.llm_batch_max_chars,
        "run_id": str(run_id),
        "segments": [
            {
                "index": segment.segment_index,
                "label": segment.diarization_label,
                "text": effective_text(segment, corrected.get(segment.id)),
            }
            for segment in segments
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _batches(
    segments: list[TranscriptSegment],
    *,
    max_segments: int,
    max_chars: int,
    corrected: dict[uuid.UUID, str | None],
) -> list[tuple[EnhancementRequestSegment, ...]]:
    """Contiguous batches bounded by count and characters.

    A single segment longer than ``max_chars`` travels alone rather than
    being split or dropped — deliberate parity with ``enhance_match``'s
    ``_build_batches`` (splitting mid-segment would break the label-scoped
    evidence mapping the location check depends on). The LLM sees the operator's
    effective text (issue #58, D2), the same rendering hashed and located below.
    """
    batches: list[tuple[EnhancementRequestSegment, ...]] = []
    current: list[EnhancementRequestSegment] = []
    chars = 0
    for segment in segments:
        text = effective_text(segment, corrected.get(segment.id))
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


# A letter for boundary purposes — the same class the extraction inventory
# treats as a name character, so "Ann" can never anchor inside "Joanne".
_LETTER = r"[^\W\d_]"


def _locate(
    name: str,
    segments: list[TranscriptSegment],
    *,
    label: str | None,
    corrected: dict[uuid.UUID, str | None],
) -> TranscriptSegment | None:
    """First segment containing the name as whole words (NFKC + casefolded).

    ``label`` restricts the search to that diarization label's own segments —
    the requirement for a self-hint to become a cluster-level claim. The
    match requires non-letter boundaries on both sides: a substring hit
    inside a longer word is not verbatim evidence. Searches the same effective
    text (issue #58, D2) the LLM was fed, so a name the operator introduced in a
    correction is locatable as evidence.
    """
    needle = unicodedata.normalize("NFKC", name).casefold()
    pattern = re.compile(rf"(?<!{_LETTER}){re.escape(needle)}(?!{_LETTER})", re.UNICODE)
    for segment in segments:
        if label is not None and segment.diarization_label != label:
            continue
        haystack = unicodedata.normalize(
            "NFKC", effective_text(segment, corrected.get(segment.id))
        ).casefold()
        if pattern.search(haystack):
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
    # Resolve the effective endpoint + key from the app_settings row (issue #10): a
    # UI-stored base_url/model/key win over env, and so does enablement — a UI toggle
    # applies with no restart, matching enhancement and the other producers. The
    # endpoint goes into exec_settings so the replay signature, the client, and the
    # evidence provenance all reflect the model that actually ran; the key is resolved
    # separately and never persisted.
    row = get_app_settings(session)
    if not (
        resolve_effective_enrichment_names_llm_enabled(row, settings)
        and resolve_effective_llm_enabled(row, settings)
    ):
        raise NameProducerError(
            "the LLM name pass is disabled — set ENRICHMENT_NAMES_LLM_ENABLED=true"
            " and enable LLM (env LLM_ENABLED or the in-UI toggle)"
        )
    effective_base_url, effective_model = resolve_effective_llm_endpoint(row, settings)
    exec_settings = settings.model_copy(
        update={"llm_base_url": effective_base_url, "llm_model": effective_model}
    )
    effective_key = resolve_effective_llm_api_key(row, settings)
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
    # Operator corrections overlay (issue #58, D2): the LLM reads, the signature
    # hashes, and _locate searches the same effective text the console/export
    # render — one shared selector, so the four can never disagree.
    corrected = {
        sid: state.corrected_text for sid, state in review_states(session, run_id).items()
    }

    signature = _input_signature(
        run_id=run_id, settings=exec_settings, segments=segments, corrected=corrected
    )
    idempotency_key = f"{LLM_PRODUCER_NAME}:{run_id}:{signature}"
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

    hints: list[SpeakerNameHint] = []
    if segments:
        owned = client is None
        llm: LLMClient
        if client is None:
            try:
                llm = HttpLLMClient(
                    exec_settings.llm_base_url,
                    exec_settings.llm_model,
                    effective_key,
                    exec_settings.llm_timeout_seconds,
                    disable_thinking=exec_settings.llm_disable_thinking,
                )
            except Exception as exc:
                # Building the client can fail before any request: a malformed
                # base_url raises httpx.InvalidURL, and httpx.Client(trust_env=True)
                # builds the SSL context eagerly, so a broken environment raises a
                # non-httpx error here too. The try wraps construction ONLY (the
                # batch loop below is a separate try), so this broad catch maps
                # every init failure — and nothing else — to the producer's own
                # error. That lets the CLI batch (`enrich names --llm`, which
                # catches NameProducerError per run) isolate the bad run instead
                # of aborting. Message carries no URL — it may hold unwanted detail.
                raise NameProducerError(
                    "LLM endpoint could not be initialized"
                    " (check the LLM endpoint setting or LLM_BASE_URL)"
                ) from exc
        else:
            llm = client
        try:
            for batch in _batches(
                segments,
                max_segments=settings.llm_batch_max_segments,
                max_chars=settings.llm_batch_max_chars,
                corrected=corrected,
            ):
                try:
                    # BYO-only producer: its whole purpose is the name_hints
                    # channel, so request it explicitly (never runs the bundled
                    # model, which suppresses hints under #85).
                    hints.extend(
                        llm.enhance_segments(batch, "", want_name_hints=True).name_hints
                    )
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
        located = _locate(name, segments, label=label, corrected=corrected)
        if located is None:
            continue  # unlocatable model output is dropped, not persisted
        evidence = TranscriptEvidence(
            transcript_segment_id=located.id,
            timestamp_seconds=located.start_seconds,
            snippet=effective_text(located, corrected.get(located.id))[:200],
            detail={
                "pattern_id": "llm_extraction",
                "kind": hint.kind,
                "model": exec_settings.llm_model,
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
        "model": exec_settings.llm_model,
        "base_url": exec_settings.llm_base_url,
        "batch_max_segments": exec_settings.llm_batch_max_segments,
        "batch_max_chars": exec_settings.llm_batch_max_chars,
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
