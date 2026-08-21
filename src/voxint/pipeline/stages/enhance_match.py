"""Enhance + match stage.

Two independent halves, deliberately unequal in failure semantics:

- **LLM enhancement is best-effort.** Segments go out in bounded, contiguous,
  ID-keyed batches; a failed batch is retried once, repeated failure opens a
  circuit for the rest of the run, and a wall-clock budget bounds total LLM
  time inside the stage lease. Affected segments simply keep ``enhanced_text``
  NULL — an unreachable optional endpoint never fails the run.
- **Speaker matching always runs** and its invariant violations DO fail the
  stage: proposals are science, not garnish.

Deterministic domain-pack corrections (#82) compose with the LLM output via a
raw-gated dual pass (:func:`_compose_correction`): rules run on ``raw_text`` to
fix the authoritative rule-fire set, then — when the LLM ran — only those rules
are enforced on its output, so a hallucinated surface can never be blessed as
operator-authored. enhanced_text + the correction trace/version are persisted
only when the final text materially differs from raw.

Idempotent under retry: enhanced_text, correction_trace, and corrector_version
are reset together and proposals are replaced (delete-then-insert via the single
writer) on every invocation.
"""

import logging
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from voxint.clients.base import (
    EnhancementRequestSegment,
    LLMClient,
    SpeakerNameHint,
)
from voxint.clients.llm import LLMError, enhanced_size_ceiling
from voxint.db.models import TranscriptSegment
from voxint.domain_packs.corrections import CorrectionRule
from voxint.domain_packs.corrector import (
    CORRECTOR_VERSION,
    CorrectionResult,
    apply_corrections,
)
from voxint.pipeline.stages.context import LLMPolicy, StageContext
from voxint.speakers.matching import (
    MAX_PROPOSED_NAME_LENGTH,
    NameHintProposal,
    evaluate_run,
    replace_run_match_candidates,
    replace_run_proposals,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Composition:
    """The dual-pass outcome for one segment (see :func:`_compose_correction`)."""

    final_text: str
    input_base: str  # "raw" | "llm"
    entries: tuple[dict[str, Any], ...]  # each {id, from, to, span:[s,e]}
    changed: bool  # final_text != raw_text


def _compose_correction(
    raw_text: str,
    raw_result: CorrectionResult,
    enhanced_text: str | None,
    rules: Sequence[CorrectionRule],
) -> _Composition:
    """Compose domain-pack corrections with the LLM output — the raw-gated dual
    pass (#82, design report §8).

    ``raw_result`` is the corrector applied to ``raw_text``, computed once by the
    caller BEFORE the LLM ran; its trace fixes the authoritative rule-fire set in
    the evidence. When the LLM is off or its batch failed (``enhanced_text is
    None``) the raw-pass result stands (``input_base="raw"``). Otherwise the
    enforcement pass re-applies ONLY the rules whose id matched raw to the LLM
    output (``input_base="llm"``): the operator's canonical form is final on terms
    genuinely present, but a rule with no raw basis can never fire — so the LLM
    cannot get a hallucinated surface blessed as operator-authored (§12-F6).

    Enforcement is frozen ID-set (a matched-raw rule applies to all its
    occurrences in the LLM output); ``input_base="llm"`` marks those spans as
    LLM-coordinate, never evidence.

    Both passes derive their growth ceiling from ``raw_text``
    (:func:`enhanced_size_ceiling`), NOT from the already-expanded LLM output:
    the enforcement bound must stay tied to the authoritative raw length so a
    corrected segment can never outgrow the raw→enhanced envelope. Deriving it
    from ``enhanced_text`` would compound the two bounds (~``16*len(raw)``).
    """
    raw_fire_ids = {entry.id for entry in raw_result.trace}
    if enhanced_text is None:
        final, base, trace = raw_result.text, "raw", raw_result.trace
    else:
        enforcement_rules = [rule for rule in rules if rule.id in raw_fire_ids]
        enforced = apply_corrections(
            enhanced_text,
            enforcement_rules,
            max_output_chars=enhanced_size_ceiling(raw_text),
        )
        final, base, trace = enforced.text, "llm", enforced.trace
    return _Composition(
        final_text=final,
        input_base=base,
        entries=tuple(entry.to_mapping() for entry in trace),
        changed=final != raw_text,
    )


def run(ctx: StageContext, session: Session, run_id: uuid.UUID) -> None:
    segments = (
        session.execute(
            select(TranscriptSegment)
            .where(TranscriptSegment.pipeline_run_id == run_id)
            .order_by(TranscriptSegment.segment_index)
        )
        .scalars()
        .all()
    )
    # Atomic re-enhance reset: enhanced_text, correction trace, and version clear
    # together on every invocation (retry-idempotent — a stale trace must never
    # outlive the enhanced_text it described).
    for segment in segments:
        segment.enhanced_text = None
        segment.correction_trace = []
        segment.corrector_version = None

    # Raw pass FIRST, for every segment — even when the LLM will succeed — so the
    # rule-fire set is fixed in the evidence before enhancement can rewrite it.
    rules = ctx.domain_pack.corrections
    raw_results = {
        segment.segment_index: apply_corrections(
            segment.raw_text,
            rules,
            max_output_chars=enhanced_size_ceiling(segment.raw_text),
        )
        for segment in segments
    }

    hints: list[SpeakerNameHint] = []
    if ctx.llm is not None and segments:
        # The scoped bundled local model (#67) powers enhancement text only, never
        # speaker attribution — so its reply is NOT parsed for name_hints (#85). The
        # prompt is left identical on every path (changing it regressed the 4B
        # model's segment faithfulness); only the parse differs, so a hallucinated
        # hint can't fail the batch for a channel that is discarded below anyway.
        want_name_hints = not ctx.llm_bundled
        # The run's frozen #11 pack may add a name-attribution fragment. Read it
        # straight from the pack (unlike enhancement_context it needs no derived
        # form — no vocabulary is folded in), keeping one source of truth.
        name_attribution_context = ctx.domain_pack.prompt_fragments.get(
            "name_attribution_context", ""
        )
        hints = _enhance(
            ctx.llm,
            ctx.llm_policy,
            ctx.enhancement_context,
            segments,
            run_id,
            name_attribution_context=name_attribution_context,
            want_name_hints=want_name_hints,
        )

    # Compose corrections with the (optional) LLM output and persist per segment,
    # BEFORE speaker matching. enhanced_text/trace/version are written only when
    # the final text materially differs from raw_text (design report §8).
    for segment in segments:
        composition = _compose_correction(
            segment.raw_text,
            raw_results[segment.segment_index],
            segment.enhanced_text,
            rules,
        )
        if composition.changed:
            # Pure-LLM enhancement (engine ran, no pack rule fired) persists here
            # too: entries=[], input_base="llm", corrector_version=1 (decision B).
            # Split-eligibility then keys off trace_has_entries (falsy for []) plus
            # the enhanced-text strip check, NOT the mere presence of a trace.
            segment.enhanced_text = composition.final_text
            segment.correction_trace = {
                "version": CORRECTOR_VERSION,
                "input_base": composition.input_base,
                "entries": list(composition.entries),
            }
            segment.corrector_version = CORRECTOR_VERSION
        else:
            segment.enhanced_text = None
            segment.correction_trace = []
            segment.corrector_version = None

    if ctx.llm_bundled:
        # Scoped bundled local model (issue #67): it powers enhancement text ONLY,
        # never speaker attribution (#66: it misattributes names). #85 already
        # suppresses the hints channel at the source (want_name_hints=False above),
        # so `hints` is empty here; this stays as an independent policy boundary —
        # attribution reaches proposals exclusively via the BYO names producer,
        # even if the upstream seam ever regresses.
        hints = []

    # One matcher pass yields both the accepted proposals AND the full per-label
    # decision evidence (issue #113): proposals drive attribution exactly as
    # before, while the near-miss/ineligible rows are captured observationally.
    decisions = evaluate_run(session, run_id, ctx.matching_gates)
    proposals = tuple(d.proposal for d in decisions if d.proposal is not None)
    replace_run_proposals(session, run_id, proposals, _select_hints(hints))
    replace_run_match_candidates(session, run_id, decisions)


def _enhance(
    llm: LLMClient,
    policy: LLMPolicy,
    context: str,
    segments: Sequence[TranscriptSegment],
    run_id: uuid.UUID,
    *,
    name_attribution_context: str = "",
    want_name_hints: bool = True,
) -> list[SpeakerNameHint]:
    """Write enhanced_text onto ``segments`` batch by batch; return the name
    hints heard along the way (in encounter order). ``name_attribution_context``
    is the run's #11 pack fragment guiding the name_hints pass ("" when the pack
    declares none). ``want_name_hints=False`` (the scoped bundled path, #85)
    suppresses the hints channel for every batch, so the returned list is empty."""
    batches = _build_batches(segments, policy)
    deadline = time.monotonic() + policy.run_budget_seconds
    consecutive_failures = 0
    succeeded = failed = 0
    hints: list[SpeakerNameHint] = []
    for batch_index, batch in enumerate(batches):
        if consecutive_failures >= policy.consecutive_failure_limit:
            logger.warning(
                "run %s: LLM circuit open after %d consecutive failures;"
                " skipping %d remaining batches",
                run_id,
                consecutive_failures,
                len(batches) - batch_index,
            )
            break
        if time.monotonic() >= deadline:
            logger.warning(
                "run %s: LLM budget (%.0fs) exhausted; skipping %d remaining batches",
                run_id,
                policy.run_budget_seconds,
                len(batches) - batch_index,
            )
            break
        request = tuple(
            EnhancementRequestSegment(
                segment_index=s.segment_index,
                text=s.raw_text,
                diarization_label=s.diarization_label,
            )
            for s in batch
        )
        result = None
        for attempt in range(1, policy.attempts_per_batch + 1):
            # Deadline holds between retries too; one in-flight attempt may
            # overrun by at most its own timeout (bounded by the Settings
            # validator against the stage lease).
            if attempt > 1 and time.monotonic() >= deadline:
                break
            try:
                result = llm.enhance_segments(
                    request,
                    context,
                    name_attribution_context=name_attribution_context,
                    want_name_hints=want_name_hints,
                )
                break
            except LLMError as exc:
                logger.warning(
                    "run %s: LLM batch %d attempt %d/%d failed: %s",
                    run_id,
                    batch_index,
                    attempt,
                    policy.attempts_per_batch,
                    exc,
                )
        if result is None:
            failed += 1
            consecutive_failures += 1
            continue
        succeeded += 1
        consecutive_failures = 0
        for s in batch:
            s.enhanced_text = result.enhanced[s.segment_index]
        hints.extend(result.name_hints)
    logger.info(
        "run %s: LLM enhancement %d/%d batches succeeded (%d failed)",
        run_id,
        succeeded,
        len(batches),
        failed,
    )
    return hints


def _build_batches(
    segments: Sequence[TranscriptSegment], policy: LLMPolicy
) -> list[list[TranscriptSegment]]:
    """Contiguous batches bounded by segment count and total characters; an
    oversized single segment travels alone rather than being dropped."""
    batches: list[list[TranscriptSegment]] = []
    current: list[TranscriptSegment] = []
    current_chars = 0
    for segment in segments:
        chars = len(segment.raw_text)
        if current and (
            len(current) >= policy.batch_max_segments
            or current_chars + chars > policy.batch_max_chars
        ):
            batches.append(current)
            current, current_chars = [], 0
        current.append(segment)
        current_chars += chars
    if current:
        batches.append(current)
    return batches


def _select_hints(hints: list[SpeakerNameHint]) -> tuple[NameHintProposal, ...]:
    """One hint per label: an explicit self-introduction beats being named by
    someone else; within a kind, the earliest heard wins. Unusable names
    (blank or absurdly long) are dropped here — they are LLM output, not a
    pipeline invariant violation."""
    chosen: dict[str, SpeakerNameHint] = {}
    for hint in hints:
        name = hint.name.strip()
        if not name or len(name) > MAX_PROPOSED_NAME_LENGTH:
            continue
        existing = chosen.get(hint.diarization_label)
        if existing is None or (existing.kind != "self" and hint.kind == "self"):
            chosen[hint.diarization_label] = hint
    return tuple(
        NameHintProposal(diarization_label=label, proposed_name=hint.name.strip())
        for label, hint in sorted(chosen.items())
    )
