"""Stage wiring: dependencies in, StageFn map out.

The P1 engine executes ``Callable[[Session, UUID], None]`` bodies; this module
builds them from a :class:`StageContext` holding the provider clients. Tests
inject fakes; the worker builds HTTP clients from settings. Stage bodies are
at-least-once (engine contract), so every one is idempotent: it deletes or
resets exactly the rows it owns for the run before writing them again.
"""

import uuid
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from voxint.clients.asr import HttpASRClient
from voxint.clients.base import ASRClient, DiarizerClient, EmbedderClient, LLMClient
from voxint.clients.diarize import HttpDiarizerClient
from voxint.clients.embed import HttpEmbedderClient
from voxint.clients.llm import HttpLLMClient
from voxint.config import Settings
from voxint.db.models import ArtifactKind, AudioArtifact, Stage
from voxint.domain_packs.base import DomainPack, load_default
from voxint.pipeline.engine import StageFn
from voxint.speakers.matching import MatchingGates


class StageDataError(Exception):
    """A stage's persisted inputs are missing or inconsistent — a pipeline bug
    or operator error, never transient."""


@dataclass(frozen=True)
class LLMPolicy:
    """Batching, retry, and budget bounds for best-effort LLM enhancement."""

    attempts_per_batch: int = 2
    batch_max_segments: int = 32
    batch_max_chars: int = 12000
    run_budget_seconds: float = 14400.0
    consecutive_failure_limit: int = 3


@dataclass(frozen=True)
class StageContext:
    asr: ASRClient
    diarizer: DiarizerClient
    embedder: EmbedderClient
    # None when llm_enabled=False: enhance_match then leaves enhanced_text NULL.
    llm: LLMClient | None
    media_root: Path
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    llm_policy: LLMPolicy = LLMPolicy()
    # Domain-pack prompt fragment appended to the enhancement system prompt.
    enhancement_context: str = ""
    matching_gates: MatchingGates = field(default_factory=MatchingGates)


def build_stage_context(settings: Settings) -> StageContext:
    """Production wiring: long-lived HTTP clients sized for media-length calls."""
    timeout = settings.gpu_http_timeout_seconds
    pack = (
        DomainPack.load(settings.domain_pack_path)
        if settings.domain_pack_path is not None
        else load_default()
    )
    return StageContext(
        asr=HttpASRClient(settings.asr_url, settings.media_root, timeout),
        diarizer=HttpDiarizerClient(settings.diarizer_url, settings.media_root, timeout),
        embedder=HttpEmbedderClient(settings.embedder_url, settings.media_root, timeout),
        llm=(
            HttpLLMClient(
                settings.llm_base_url,
                settings.llm_model,
                settings.llm_api_key,
                settings.llm_timeout_seconds,
            )
            if settings.llm_enabled
            else None
        ),
        media_root=settings.media_root,
        ffmpeg_bin=settings.ffmpeg_bin,
        ffprobe_bin=settings.ffprobe_bin,
        llm_policy=LLMPolicy(
            attempts_per_batch=settings.llm_attempts_per_batch,
            batch_max_segments=settings.llm_batch_max_segments,
            batch_max_chars=settings.llm_batch_max_chars,
            run_budget_seconds=settings.llm_run_budget_seconds,
            consecutive_failure_limit=settings.llm_consecutive_failure_limit,
        ),
        enhancement_context=pack.prompt_fragments.get("enhancement_context", ""),
        matching_gates=MatchingGates(
            max_overlap_ratio=settings.match_max_overlap_ratio,
            turn_weight_cap_seconds=settings.match_turn_weight_cap_seconds,
            min_turns=settings.match_min_turns,
            min_seconds=settings.match_min_seconds,
            min_cosine=settings.match_min_cosine,
            min_margin=settings.match_min_margin,
            min_vote_agreement=settings.match_min_vote_agreement,
            grounded_min_turns=settings.grounded_min_turns,
            grounded_min_seconds=settings.grounded_min_seconds,
            grounded_min_cosine=settings.grounded_min_cosine,
            grounded_min_margin=settings.grounded_min_margin,
            grounded_min_vote_agreement=settings.grounded_min_vote_agreement,
        ),
    )


def build_stage_fns(ctx: StageContext) -> dict[Stage, StageFn]:
    from voxint.pipeline.stages import (
        diarize_embed,
        enhance_match,
        finalize,
        prepare,
        transcribe,
    )

    return {
        Stage.PREPARE: partial(prepare.run, ctx),
        Stage.TRANSCRIBE: partial(transcribe.run, ctx),
        Stage.DIARIZE_EMBED: partial(diarize_embed.run, ctx),
        Stage.ENHANCE_MATCH: partial(enhance_match.run, ctx),
        Stage.FINALIZE: partial(finalize.run, ctx),
    }


def normalized_audio_path(session: Session, run_id: uuid.UUID, media_root: Path) -> Path:
    """Locate the prepare stage's output for this run — exactly one artifact."""
    artifacts = (
        session.execute(
            select(AudioArtifact).where(
                AudioArtifact.pipeline_run_id == run_id,
                AudioArtifact.kind == ArtifactKind.PREPROCESSED_AUDIO.value,
            )
        )
        .scalars()
        .all()
    )
    if len(artifacts) != 1:
        raise StageDataError(
            f"run {run_id}: expected exactly one preprocessed_audio artifact,"
            f" found {len(artifacts)}"
        )
    return media_root / artifacts[0].path
