"""Enhance + match stage.

Two independent halves, deliberately unequal in failure semantics:

- **LLM enhancement is best-effort.** Segments go out in bounded, contiguous,
  ID-keyed batches; a failed batch is retried once, repeated failure opens a
  circuit for the rest of the run, and a wall-clock budget bounds total LLM
  time inside the stage lease. Affected segments simply keep ``enhanced_text``
  NULL — an unreachable optional endpoint never fails the run.
- **Speaker matching always runs** and its invariant violations DO fail the
  stage: proposals are science, not garnish.

Idempotent under retry: enhanced_text is reset and proposals are replaced
(delete-then-insert via the single writer) on every invocation.
"""

import logging
import time
import uuid
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from voxint.clients.base import (
    EnhancementRequestSegment,
    LLMClient,
    SpeakerNameHint,
)
from voxint.clients.llm import LLMError
from voxint.db.models import TranscriptSegment
from voxint.pipeline.stages.context import LLMPolicy, StageContext
from voxint.speakers.matching import (
    MAX_PROPOSED_NAME_LENGTH,
    NameHintProposal,
    match_speakers,
    replace_run_proposals,
)

logger = logging.getLogger(__name__)


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
    for segment in segments:
        segment.enhanced_text = None

    hints: list[SpeakerNameHint] = []
    if ctx.llm is not None and segments:
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
        )

    if ctx.llm_bundled:
        # Scoped bundled local model (issue #67): it powers enhancement text ONLY,
        # never speaker attribution (#66: it misattributes names). Drop the pass's
        # name_hints so they can't reach proposals through the back door —
        # attribution stays exclusively on the BYO names producer.
        hints = []

    proposals = match_speakers(session, run_id, ctx.matching_gates)
    replace_run_proposals(session, run_id, proposals, _select_hints(hints))


def _enhance(
    llm: LLMClient,
    policy: LLMPolicy,
    context: str,
    segments: Sequence[TranscriptSegment],
    run_id: uuid.UUID,
    *,
    name_attribution_context: str = "",
) -> list[SpeakerNameHint]:
    """Write enhanced_text onto ``segments`` batch by batch; return the name
    hints heard along the way (in encounter order). ``name_attribution_context``
    is the run's #11 pack fragment guiding the name_hints pass ("" when the pack
    declares none)."""
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
                    request, context, name_attribution_context=name_attribution_context
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
