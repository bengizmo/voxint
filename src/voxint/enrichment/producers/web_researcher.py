"""The web-research producer (issue #40): one job, one speaker, one recorded run.

Wraps the research loop (:mod:`voxint.research.agent`) in the #37 draft
contract. Surviving claims become speaker-scoped bio/affiliation/link
candidates with :class:`UrlEvidence`; a confident ``found=false`` (or a
conclusion whose every claim failed grounding) records an authoritative
``outcome='none'`` generation. LLM/transport/contract failures and
cancellation RAISE — they must never be recorded as a 'none' that would
retire prior drafts.

Idempotency is per job, not per input: web research is non-deterministic, so
the key ``web_researcher:speaker:{speaker_id}:{job_id}`` means "this durable
execution", and an intentional rerun is a new job minting a new superseding
generation. The full non-secret configuration still snapshots into ``config``
for audit.
"""

import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from voxint.config import Settings
from voxint.db.models import (
    AdjudicationDecision,
    ClaimField,
    Decision,
    EnrichmentCandidate,
    EnrichmentProducerRun,
    EnrichmentTargetKind,
    MediaSourceMetadata,
    PipelineRun,
    Speaker,
)
from voxint.enrichment.drafts import (
    MAX_EVIDENCE_ROWS,
    CandidateDraft,
    EnrichmentScope,
    UrlEvidence,
    record_producer_run,
)
from voxint.enrichment.review import ConflictingReplayError
from voxint.research.agent import (
    ProgressCounters,
    ResearchConclusion,
    ResearchSeed,
    RosterMatch,
    run_research_loop,
)
from voxint.speakers import roster

PRODUCER_NAME = "web_researcher"
PRODUCER_VERSION = "1"
CONFIG_SCHEMA_VERSION = 1
DETAIL_SCHEMA_VERSION = 1
# Uncalibrated marker below every strong deterministic signal (names.llm
# precedent) — web claims are suggestions for review, never ranked truth.
CANDIDATE_SCORE = 0.5
MAX_SEED_CANDIDATE_NAMES = 5
MAX_SEED_RUNS = 3
MAX_SEED_DESCRIPTION_CHARS = 300
MAX_OPERATOR_NOTE_CHARS = 1_000


class WebResearcherError(Exception):
    """The job cannot proceed or complete — misconfiguration, missing/merged
    speaker, or a research-loop failure. Nothing is persisted to drafts."""


def _assigned_run_ids(session: Session, speaker_id: uuid.UUID) -> list[uuid.UUID]:
    """Runs with a human 'assign' ruling for this speaker (alias-chain aware),
    newest ruling first."""
    aliases = roster.alias_ids(session, speaker_id)
    rows = session.execute(
        select(AdjudicationDecision.pipeline_run_id, AdjudicationDecision.created_at)
        .where(
            AdjudicationDecision.decision == Decision.ASSIGN.value,
            AdjudicationDecision.speaker_id.in_(aliases),
            # LABEL scope only (issue #54 Phase B): a per-segment override is not
            # a whole-label attribution and must not seed run-level research.
            AdjudicationDecision.transcript_segment_id.is_(None),
        )
        .order_by(AdjudicationDecision.created_at.desc())
    ).all()
    seen: list[uuid.UUID] = []
    for run_id, _ in rows:
        if run_id not in seen:
            seen.append(run_id)
    return seen


def _candidate_names(
    session: Session, speaker_id: uuid.UUID, run_ids: list[uuid.UUID]
) -> tuple[str, ...]:
    """Distinct NAME-claim values from the labels this speaker was ruled onto.

    The #38 producers write run_label name candidates; the adjudication ledger
    ties (run, label) to the speaker. That join — not any web content — is
    what seeds alternative spellings and heard names.
    """
    aliases = roster.alias_ids(session, speaker_id)
    pairs = set(
        session.execute(
            select(
                AdjudicationDecision.pipeline_run_id,
                AdjudicationDecision.diarization_label,
            ).where(
                AdjudicationDecision.decision == Decision.ASSIGN.value,
                AdjudicationDecision.speaker_id.in_(aliases),
                AdjudicationDecision.transcript_segment_id.is_(None),
            )
        ).all()
    )
    if not pairs:
        return ()
    values: list[str] = []
    rows = session.execute(
        select(
            EnrichmentCandidate.pipeline_run_id,
            EnrichmentCandidate.diarization_label,
            EnrichmentCandidate.value,
        )
        .where(
            EnrichmentCandidate.target_kind == EnrichmentTargetKind.RUN_LABEL.value,
            EnrichmentCandidate.field == ClaimField.NAME.value,
            EnrichmentCandidate.pipeline_run_id.in_(run_ids),
        )
        .order_by(EnrichmentCandidate.created_at.desc())
    ).all()
    for run_id, label, value in rows:
        if (run_id, label) in pairs and value.casefold() not in {v.casefold() for v in values}:
            values.append(value)
        if len(values) >= MAX_SEED_CANDIDATE_NAMES:
            break
    return tuple(values)


def _seed_context(
    session: Session, run_ids: list[uuid.UUID]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(context lines, seed URLs) from the stored #36 metadata snapshots."""
    lines: list[str] = []
    urls: list[str] = []
    for run_id in run_ids[:MAX_SEED_RUNS]:
        run = session.get(PipelineRun, run_id)
        if run is None:
            continue
        metadata = session.execute(
            select(MediaSourceMetadata).where(
                MediaSourceMetadata.media_item_id == run.media_item_id
            )
        ).scalar_one_or_none()
        if metadata is None:
            continue
        parts: list[str] = []
        if metadata.title:
            parts.append(f"title: {metadata.title}")
        if metadata.channel:
            parts.append(f"channel: {metadata.channel}")
        if metadata.uploader and metadata.uploader != metadata.channel:
            parts.append(f"uploader: {metadata.uploader}")
        if metadata.upload_date:
            parts.append(f"published: {metadata.upload_date.isoformat()}")
        if metadata.description:
            parts.append(
                "description: "
                + " ".join(metadata.description.split())[:MAX_SEED_DESCRIPTION_CHARS]
            )
        if parts:
            lines.append("; ".join(parts))
        for url in (metadata.canonical_url, metadata.channel_url, metadata.uploader_url):
            if url and url.lower().startswith(("http://", "https://")) and url not in urls:
                urls.append(url)
    return tuple(lines), tuple(urls)


def build_seed(
    session: Session,
    *,
    speaker: Speaker,
    operator_note: str | None,
) -> ResearchSeed:
    """The bounded, operator-controlled context for one speaker's job."""
    run_ids = _assigned_run_ids(session, speaker.id)
    context_lines, seed_urls = _seed_context(session, run_ids)
    note = None
    if operator_note and operator_note.strip():
        note = " ".join(operator_note.split())[:MAX_OPERATOR_NOTE_CHARS]
    return ResearchSeed(
        display_name=speaker.display_name,
        candidate_names=_candidate_names(session, speaker.id, run_ids),
        context_lines=context_lines,
        seed_urls=seed_urls,
        operator_note=note,
    )


def make_roster_lookup(
    session: Session, *, target_speaker_id: uuid.UUID
) -> "Callable[[str], list[RosterMatch]]":
    """The read-only ``query_existing_speakers`` tool: canonical identities
    only, names and ids only — nothing else can leak into the model context."""

    def lookup(query: str) -> list[RosterMatch]:
        needle = roster.normalize_display_name(query).casefold()
        if not needle:
            return []
        return [
            RosterMatch(
                speaker_id=speaker.id,
                display_name=speaker.display_name,
                is_target=speaker.id == target_speaker_id,
            )
            for speaker in roster.searchable_speakers(session)
            if needle in speaker.display_name.casefold()
        ]

    return lookup


def load_research_speaker(session: Session, speaker_id: uuid.UUID) -> Speaker:
    """The speaker a job may target: existing, canonical (no merge tombstone),
    and not archived."""
    speaker = session.get(Speaker, speaker_id)
    if speaker is None:
        raise WebResearcherError(f"speaker {speaker_id} not found")
    if speaker.merged_into_id is not None:
        raise WebResearcherError(
            f"speaker {speaker.display_name!r} was merged — research the merge target instead"
        )
    if speaker.deleted_at is not None:
        raise WebResearcherError(
            f"speaker {speaker.display_name!r} is archived — restore it before researching it"
        )
    return speaker


def record_research_outcome(
    session: Session,
    *,
    job_id: uuid.UUID,
    speaker_id: uuid.UUID,
    settings: Settings,
    conclusion: ResearchConclusion,
    started_at: datetime,
) -> EnrichmentProducerRun:
    """Persist one completed loop as a #37 producer run (the caller commits)."""
    scope = EnrichmentScope.speaker(speaker_id)
    drafts = tuple(
        CandidateDraft(
            target=scope,
            field=claim.field,
            value=claim.value,
            # One evidence row per independently grounded source — multiple
            # distinct sources for one value is the corroboration signal triage
            # (#42) reads. Ordered as grounded; bounded by MAX_EVIDENCE_ROWS.
            evidence=tuple(
                UrlEvidence(
                    url=source.url,
                    retrieved_at=source.retrieved_at,
                    snippet=source.snippet,
                    detail={
                        "model": settings.llm_model,
                        "title": source.title,
                        "source_id": source.source_id,
                        # Redirects can move the read: keep the URL the loop
                        # authorized beside the final one actually fetched.
                        **(
                            {"requested_url": source.requested_url}
                            if source.requested_url != source.url
                            else {}
                        ),
                    },
                    detail_schema_version=DETAIL_SCHEMA_VERSION,
                )
                for source in claim.sources
            )[:MAX_EVIDENCE_ROWS],
            score=CANDIDATE_SCORE,
            score_components={"web": 1.0},
        )
        for claim in conclusion.claims
    )
    config: dict[str, object] = {
        "producer_version": PRODUCER_VERSION,
        "protocol_version": "1",
        "model": settings.llm_model,
        "base_url": settings.llm_base_url,
        "max_searches": settings.research_max_searches,
        "max_reads": settings.research_max_reads,
        "max_rounds": settings.research_max_rounds,
        "deadline_seconds": settings.research_deadline_seconds,
        "job_id": str(job_id),
        "found": conclusion.found,
        "searches_used": conclusion.searches_used,
        "reads_used": conclusion.reads_used,
        "rounds_used": conclusion.rounds_used,
        "dropped_claims": conclusion.dropped_claims,
    }
    # JSONB None becomes JSON null (not SQL NULL) — only assign when present.
    if conclusion.reason:
        config["reason"] = conclusion.reason

    idempotency_key = f"{PRODUCER_NAME}:speaker:{speaker_id}:{job_id}"
    try:
        return record_producer_run(
            session,
            producer=PRODUCER_NAME,
            producer_version=PRODUCER_VERSION,
            scope=scope,
            covered_fields=(ClaimField.BIO, ClaimField.AFFILIATION, ClaimField.LINK),
            candidates=drafts,
            idempotency_key=idempotency_key,
            started_at=started_at,
            completed_at=datetime.now(tz=UTC),
            config=config,
            config_schema_version=CONFIG_SCHEMA_VERSION,
        )
    except ConflictingReplayError:
        # A raced duplicate delivery finalized first with (only) different
        # timestamps/results; adopt whatever that execution recorded.
        raced = session.execute(
            select(EnrichmentProducerRun).where(
                EnrichmentProducerRun.idempotency_key == idempotency_key
            )
        ).scalar_one_or_none()
        if raced is not None:
            return raced
        raise


__all__ = [
    "PRODUCER_NAME",
    "PRODUCER_VERSION",
    "ProgressCounters",
    "ResearchConclusion",
    "WebResearcherError",
    "build_seed",
    "load_research_speaker",
    "make_roster_lookup",
    "record_research_outcome",
    "run_research_loop",
]
