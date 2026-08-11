"""Stage wiring: dependencies in, StageFn map out.

The P1 engine executes ``Callable[[Session, UUID], None]`` bodies; this module
builds them from a :class:`StageContext` holding the provider clients. Tests
inject fakes; the worker builds HTTP clients from settings. Stage bodies are
at-least-once (engine contract), so every one is idempotent: it deletes or
resets exactly the rows it owns for the run before writing them again.
"""

import uuid
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from voxint.clients.asr import HttpASRClient
from voxint.clients.base import ASRClient, DiarizerClient, EmbedderClient, LLMClient
from voxint.clients.diarize import HttpDiarizerClient
from voxint.clients.embed import HttpEmbedderClient
from voxint.config import Settings
from voxint.db.models import ArtifactKind, AudioArtifact, Stage
from voxint.pipeline.engine import StageFn


class StageDataError(Exception):
    """A stage's persisted inputs are missing or inconsistent — a pipeline bug
    or operator error, never transient."""


@dataclass(frozen=True)
class StageContext:
    asr: ASRClient
    diarizer: DiarizerClient
    embedder: EmbedderClient
    # None until P4 wires the real enhancement adapter (llm_enabled=False):
    # enhance_match then leaves enhanced_text NULL.
    llm: LLMClient | None
    media_root: Path
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"


def build_stage_context(settings: Settings) -> StageContext:
    """Production wiring: long-lived HTTP clients sized for media-length calls."""
    timeout = settings.gpu_http_timeout_seconds
    return StageContext(
        asr=HttpASRClient(settings.asr_url, settings.media_root, timeout),
        diarizer=HttpDiarizerClient(settings.diarizer_url, settings.media_root, timeout),
        embedder=HttpEmbedderClient(settings.embedder_url, settings.media_root, timeout),
        llm=None,  # P4
        media_root=settings.media_root,
        ffmpeg_bin=settings.ffmpeg_bin,
        ffprobe_bin=settings.ffprobe_bin,
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
