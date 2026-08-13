"""Stage wiring: dependencies in, StageFn map out.

The P1 engine executes ``Callable[[Session, UUID], None]`` bodies; this module
builds them from a :class:`StageContext` holding the provider clients. Tests
inject fakes; the worker builds HTTP clients from settings. Stage bodies are
at-least-once (engine contract), so every one is idempotent: it deletes or
resets exactly the rows it owns for the run before writing them again.
"""

import dataclasses
import logging
import socket
import uuid
from collections.abc import Iterable
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
from voxint.db.models import AppSettings, ArtifactKind, AudioArtifact, Stage
from voxint.domain_packs.base import DomainPack, load_default
from voxint.media.netcheck import Resolver
from voxint.media.ytdlp import Downloader, build_ytdlp_downloader
from voxint.pipeline.engine import StageFn
from voxint.speakers.matching import MatchingGates, gates_from_settings

logger = logging.getLogger(__name__)


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
    # URL acquisition (ACQUIRE stage). None ⇒ no downloader wired: a URL run then
    # fails loudly rather than silently no-oping past an un-acquired source. The
    # no-op path (source_url IS NULL) never touches these, so isolated no-op tests
    # can leave them at their defaults.
    downloader: Downloader | None = None
    # DNS resolver for the ACQUIRE stage's worker-side SSRF gate: it re-resolves a
    # URL run's host and rejects non-public addresses before the download. Injected
    # (defaults to socket.getaddrinfo) so tests never touch real DNS — the no-op
    # path (source_url IS NULL) never calls it.
    resolver: Resolver = socket.getaddrinfo
    # Authoritative post-download size cap the stage re-checks (also handed to the
    # downloader as an early --max-filesize hint).
    ytdlp_max_bytes: int = 5 * 1024**3
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    llm_policy: LLMPolicy = LLMPolicy()
    # Domain-pack prompt fragment appended to the enhancement system prompt.
    enhancement_context: str = ""
    # Effective ASR/enhancement vocabulary for the run (domain pack + user words).
    # Surfaced to the whisper initial_prompt and rendered into enhancement_context.
    vocabulary: tuple[str, ...] = ()
    matching_gates: MatchingGates = field(default_factory=MatchingGates)


def build_stage_context(settings: Settings) -> StageContext:
    """Production wiring: long-lived HTTP clients sized for media-length calls.

    This is the process-cached BASE context. The LLM client is deliberately left
    ``None`` here and wired per-run by :func:`apply_run_preferences`, which owns
    (and closes) it — the run's effective base_url/model/enabled come from the
    app_settings row, and a per-run owned ``httpx.Client`` must not outlive the run.
    The transport clients (asr/diarizer/embedder/downloader) ARE long-lived and are
    preserved across runs.
    """
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
        llm=None,  # wired per-run by apply_run_preferences (which closes it)
        media_root=settings.media_root,
        downloader=build_ytdlp_downloader(
            timeout_seconds=settings.acquire_timeout_seconds,
            socket_timeout_seconds=settings.ytdlp_socket_timeout_seconds,
            proxy=settings.ytdlp_proxy,
            cookies_file=settings.ytdlp_cookies_file,
        ),
        ytdlp_max_bytes=settings.ytdlp_max_bytes,
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
        vocabulary=pack.vocabulary,
        matching_gates=gates_from_settings(settings),
    )


@dataclass(frozen=True)
class RunPreferences:
    """Effective, non-secret pipeline preferences for a single run.

    Snapshotted once at ``run_pipeline`` start from the ``app_settings`` singleton
    layered over env (:class:`Settings`), then applied to the process-cached base
    context — so a wizard edit takes effect on the next run with no worker restart.
    ``vocabulary`` here is the *user* word list only; the domain pack's words are
    unioned in :func:`apply_run_preferences`, which holds the base context.
    """

    vocabulary: tuple[str, ...]
    llm_enabled: bool
    llm_base_url: str
    llm_model: str


def _dedup_order_preserving(items: Iterable[str]) -> tuple[str, ...]:
    """First occurrence wins; blank/whitespace-only entries dropped."""
    seen: dict[str, None] = {}
    for item in items:
        stripped = item.strip()
        if stripped and stripped not in seen:
            seen[stripped] = None
    return tuple(seen)


def resolve_run_preferences(
    row: AppSettings | None, settings: Settings
) -> RunPreferences:
    """Layer the ``app_settings`` row over env defaults (pure — no I/O, no pack).

    A NULL/absent row field falls back to the env default, so with no row at all
    this reproduces today's env-only behavior exactly. The LLM API key is never
    stored in the row; it stays env-only (see :func:`apply_run_preferences`).
    """
    llm_enabled = row.llm_enabled if row is not None else settings.llm_enabled
    llm_base_url = (
        row.llm_base_url if row is not None and row.llm_base_url else settings.llm_base_url
    )
    llm_model = row.llm_model if row is not None and row.llm_model else settings.llm_model
    # row.vocabulary is NOT NULL in the DB (server default []), but an in-memory
    # row may leave it None — guard so a bare row can't crash a pure resolve.
    vocabulary = _dedup_order_preserving(
        row.vocabulary if row is not None and row.vocabulary else ()
    )
    return RunPreferences(
        vocabulary=vocabulary,
        llm_enabled=llm_enabled,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
    )


def _augment_enhancement_context(pack_fragment: str, vocabulary: tuple[str, ...]) -> str:
    """Append a rendered vocabulary line to the domain pack's prompt fragment."""
    if not vocabulary:
        return pack_fragment
    vocab_line = "Domain vocabulary (names, jargon, acronyms): " + ", ".join(vocabulary)
    return f"{pack_fragment}\n{vocab_line}" if pack_fragment else vocab_line


def apply_run_preferences(
    base: StageContext, settings: Settings, prefs: RunPreferences
) -> StageContext:
    """Layer ``prefs`` onto the process-cached ``base`` context for one run.

    Transport clients (asr/diarizer/embedder/downloader) are kept as-is — only the
    per-run, preference-derived fields are swapped via ``dataclasses.replace``:
    the pack+user vocabulary, the enhancement context that renders it, and the LLM
    client. Enabling the LLM without an env key is an honest no-op: the run logs a
    warning and proceeds with ``llm=None`` (enhancement is best-effort, never a
    blocker), rather than failing.

    When enabled, the returned ``llm`` is a freshly built ``HttpLLMClient`` that
    OWNS its ``httpx.Client``; the caller (``run_pipeline``) must ``close()`` it
    after the run so a long-lived worker does not leak a connection pool per run.
    """
    vocabulary = _dedup_order_preserving((*base.vocabulary, *prefs.vocabulary))
    enhancement_context = _augment_enhancement_context(base.enhancement_context, vocabulary)
    llm: LLMClient | None
    if prefs.llm_enabled and settings.llm_api_key:
        llm = HttpLLMClient(
            prefs.llm_base_url,
            prefs.llm_model,
            settings.llm_api_key,
            settings.llm_timeout_seconds,
        )
    else:
        if prefs.llm_enabled and not settings.llm_api_key:
            logger.warning(
                "LLM enhancement is enabled but LLM_API_KEY is unset; "
                "proceeding with enhancement disabled for this run."
            )
        llm = None
    return dataclasses.replace(
        base, llm=llm, enhancement_context=enhancement_context, vocabulary=vocabulary
    )


def build_stage_fns(ctx: StageContext) -> dict[Stage, StageFn]:
    from voxint.pipeline.stages import (
        acquire,
        diarize_embed,
        enhance_match,
        finalize,
        prepare,
        transcribe,
    )

    return {
        Stage.ACQUIRE: partial(acquire.run, ctx),
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
