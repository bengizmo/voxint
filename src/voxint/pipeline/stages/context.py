"""Stage wiring: dependencies in, StageFn map out.

The P1 engine executes ``Callable[[Session, UUID], None]`` bodies; this module
builds them from a :class:`StageContext` holding the provider clients. Tests
inject fakes; the worker builds HTTP clients from settings. Stage bodies are
at-least-once (engine contract), so every one is idempotent: it deletes or
resets exactly the rows it owns for the run before writing them again.
"""

from __future__ import annotations

import dataclasses
import logging
import socket
import uuid
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from voxint.app_settings import build_bundled_llm_client, resolve_effective_llm_endpoint
from voxint.clients.asr import HttpASRClient
from voxint.clients.base import ASRClient, DiarizerClient, EmbedderClient, LLMClient
from voxint.clients.diarize import HttpDiarizerClient
from voxint.clients.embed import HttpEmbedderClient
from voxint.clients.llm import HttpLLMClient
from voxint.config import Settings, llm_budget_fits_stage_lease
from voxint.db.models import AppSettings, ArtifactKind, AudioArtifact, Stage
from voxint.domain_packs.base import DomainPack, dedup_order_preserving, load_default
from voxint.domain_packs.registry import default_domain_pack
from voxint.media.netcheck import Resolver
from voxint.media.ytdlp import Downloader, build_ytdlp_downloader
from voxint.pipeline.engine import StageFn
from voxint.speakers.matching import MatchingGates, gates_from_settings

logger = logging.getLogger(__name__)


def parse_config_resolution_version(pack_snapshot: dict[str, Any] | None) -> int:
    """Extract the config-resolution version from a domain-pack snapshot.

    NULL snapshots, pre-#153 rows, and malformed metadata use the version-1
    live-union path.
    """
    if not isinstance(pack_snapshot, dict):
        return 1
    raw = pack_snapshot.get("config_resolution_version", 1)
    try:
        version: int = int(raw)
        return version
    except (TypeError, ValueError, OverflowError):
        logger.warning(
            "malformed config_resolution_version %r; falling back to "
            "live-union config resolution",
            raw,
        )
        return 1


class StageDataError(Exception):
    """A stage's persisted inputs are missing or inconsistent — a pipeline bug
    or operator error, never transient."""


class StageDeferError(Exception):
    """The stage should be retried later (not a failure)."""


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
    # Caller-known secret literals (configured proxy string, cookies path)
    # scrubbed verbatim from RETAINED metadata text (media/source_metadata.py) —
    # the same contract the downloader threads to redact() for error text. The
    # values live inside the downloader closure, so the stage needs its own copy.
    metadata_secrets: tuple[str, ...] = ()
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    llm_policy: LLMPolicy = LLMPolicy()
    # True when ``llm`` is the scoped bundled local model (issue #67), not a BYO
    # endpoint. enhance_match reads this to DROP the enhancement pass's own
    # name_hints so the bundled model can never power speaker attribution through
    # the back door — attribution stays exclusively on the BYO names producer.
    llm_bundled: bool = False
    # The run's resolved domain pack (issue #11). On the process-cached base this is
    # the DEFAULT pack; apply_run_preferences swaps in the run's frozen snapshot so a
    # stage can read any fragment via ctx.domain_pack.prompt_fragments.get(key, "").
    domain_pack: DomainPack = field(default_factory=load_default)
    # Domain-pack prompt fragment appended to the enhancement system prompt.
    enhancement_context: str = ""
    # Effective ASR/enhancement vocabulary for the run (domain pack + user words).
    # Surfaced to the whisper initial_prompt and rendered into enhancement_context.
    vocabulary: tuple[str, ...] = ()
    matching_gates: MatchingGates = field(default_factory=MatchingGates)
    # Speaker-count hint handed to the diarizer for this run (issue #128). Both
    # None ⇒ no bound is sent and the service applies its own default. The max is
    # the install-wide ceiling (settings.diarization_max_speakers) unless the run
    # carries a per-recording override; the exact count, when set, pins pyannote
    # to that many speakers and wins over the max.
    diarization_max_speakers: int | None = None
    diarization_num_speakers: int | None = None


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
    # The base carries the DEFAULT pack; apply_run_preferences overrides it with the
    # run's frozen snapshot per run (issue #11), so a legacy (NULL-snapshot) run and
    # tests that skip apply_run_preferences still see a valid pack.
    pack = default_domain_pack(settings)
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
        metadata_secrets=tuple(
            secret
            for secret in (
                settings.ytdlp_proxy,
                str(settings.ytdlp_cookies_file) if settings.ytdlp_cookies_file else "",
            )
            if secret
        ),
        ffmpeg_bin=settings.ffmpeg_bin,
        ffprobe_bin=settings.ffprobe_bin,
        llm_policy=LLMPolicy(
            attempts_per_batch=settings.llm_attempts_per_batch,
            batch_max_segments=settings.llm_batch_max_segments,
            batch_max_chars=settings.llm_batch_max_chars,
            run_budget_seconds=settings.llm_run_budget_seconds,
            consecutive_failure_limit=settings.llm_consecutive_failure_limit,
        ),
        domain_pack=pack,
        enhancement_context=pack.prompt_fragments.get("enhancement_context", ""),
        vocabulary=pack.vocabulary,
        matching_gates=gates_from_settings(settings),
        diarization_max_speakers=settings.diarization_max_speakers,
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


# The canonicalization moved to voxint.domain_packs.base so the submit-time freeze
# (issue #153) shares one definition with this live path; a byte-identical v2
# snapshot depends on both sides running the same function.
_dedup_order_preserving = dedup_order_preserving


def resolve_run_preferences(
    row: AppSettings | None, settings: Settings
) -> RunPreferences:
    """Layer the ``app_settings`` row over env defaults (pure — no I/O, no pack).

    A NULL/absent row field falls back to the env default, so with no row at all
    this reproduces today's env-only behavior exactly. This carries only the
    non-secret preferences; the effective API key is resolved separately (kept out
    of this dataclass, which has a repr) via
    :func:`voxint.app_settings.resolve_effective_llm_api_key` and passed into
    :func:`apply_run_preferences`.
    """
    llm_enabled = row.llm_enabled if row is not None else settings.llm_enabled
    llm_base_url, llm_model = resolve_effective_llm_endpoint(row, settings)
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
    base: StageContext,
    settings: Settings,
    prefs: RunPreferences,
    pack: DomainPack,
    *,
    llm_api_key: str,
    bundled: bool = False,
    config_resolution_version: int = 1,
) -> StageContext:
    """Layer ``prefs`` and the run's ``pack`` onto the cached ``base`` for one run.

    Transport clients (asr/diarizer/embedder/downloader) are kept as-is — only the
    per-run fields are swapped via ``dataclasses.replace``: the run's domain pack
    (issue #11), the pack+user vocabulary, the enhancement context that renders it,
    and the LLM client. ``pack`` is the run's frozen snapshot (or the default for a
    legacy run), so vocabulary and every prompt fragment come from THAT pack, not
    the process-cached base — two queued runs with different packs get different
    contexts in the same worker with no restart. Enabling the LLM without an
    effective key is an honest no-op: the run logs a warning and proceeds with
    ``llm=None`` (enhancement is best-effort, never a blocker), rather than failing.

    ``llm_api_key`` is the effective key (a UI-stored row value wins over env),
    resolved by the caller via
    :func:`voxint.app_settings.resolve_effective_llm_api_key` and passed as a
    keyword-only ``str`` — deliberately NOT carried on the (repr-bearing, non-secret)
    :class:`RunPreferences`.

    When enabled, the returned ``llm`` is a freshly built ``HttpLLMClient`` that
    OWNS its ``httpx.Client``; the caller (``run_pipeline``) must ``close()`` it
    after the run so a long-lived worker does not leak a connection pool per run.

    ``bundled`` (issue #67): when True, the run routes enhancement to the scoped
    bundled local model — a fixed, product-owned, KEYLESS endpoint — instead of the
    BYO ``llm_*`` one. The key precondition is then satisfied without a key (the
    bundle needs none), the client is built from ``settings.llm_bundled_*`` via the
    shared factory, and the returned context carries ``llm_bundled=True`` so
    enhance_match drops the pass's name_hints. Resolved by the caller through
    :func:`voxint.app_settings.llm_bundled_active`.
    """
    # Effective vocabulary. A version-2 snapshot (issue #153) already FROZE the
    # per-field-resolved effective vocabulary at submit — project/folder/global
    # replacement is baked into pack.vocabulary — so the worker must NOT re-union
    # the live operator glossary or it would leak a later settings edit into a
    # deterministically-frozen run. A missing/absent version (every pre-#153 row,
    # config_resolution_version == 1) keeps the exact live-union path so those
    # runs stay byte-identical. Only version 2 is a freeze; an unrecognized future
    # version deliberately falls through to the live-union path (never silently
    # reinterpreted as a freeze it may not be). Fail-closing on an unknown version
    # was considered and rejected: this runs outside the execute_run failure lane,
    # so a raise here would strand the run for the recovery sweep to re-publish
    # forever, and voxint is single-operator (no mixed-version worker fleet to
    # protect against a rolling upgrade). Live-union is the safe, non-poisoning
    # fallback.
    if config_resolution_version == 2:
        vocabulary = _dedup_order_preserving(pack.vocabulary)
    else:
        vocabulary = _dedup_order_preserving((*pack.vocabulary, *prefs.vocabulary))
    enhancement_context = _augment_enhancement_context(
        pack.prompt_fragments.get("enhancement_context", ""), vocabulary
    )
    # Fail closed on three independent preconditions before building the client:
    #  - the effective API key must be present (a UI-stored row value or env);
    #  - the run budget must still fit the enhance_match lease (the wizard refuses to
    #    persist llm_enabled=True when it doesn't, but the env budget can change AFTER
    #    a True row is written while env llm_enabled is off, so the startup validator
    #    never re-checked it — this per-run guard closes that window); and
    #  - the client must actually build (a malformed base_url raises in httpx).
    # All three are honest no-ops (enhancement is best-effort): warn and proceed with
    # llm=None rather than failing — or, for the URL case, poison-looping — the run.
    budget_ok = llm_budget_fits_stage_lease(settings)
    # The caller already resolved the effective key (row wins over env) and stripped
    # it; a whitespace-only key resolves to "" and reads as absent — matching the
    # wizard's own check (setup_wizard.validate_llm_enable), so the two never
    # disagree on whether a key is set.
    key_present = bool(llm_api_key)
    # The bundled endpoint needs no key (issue #67), so it satisfies the key
    # precondition on its own; the BYO path still requires a key.
    key_ok = key_present or bundled
    llm: LLMClient | None = None
    if prefs.llm_enabled and key_ok and budget_ok:
        try:
            llm = (
                build_bundled_llm_client(settings)
                if bundled
                else HttpLLMClient(
                    prefs.llm_base_url,
                    prefs.llm_model,
                    llm_api_key,
                    settings.llm_timeout_seconds,
                    disable_thinking=settings.llm_disable_thinking,
                )
            )
        except (httpx.InvalidURL, httpx.HTTPError) as exc:
            # A malformed base_url raises while httpx builds the client. This runs in
            # run_pipeline BEFORE execute_run's failure handling, so letting it
            # propagate would leave the run QUEUED for the recovery sweep to
            # re-publish forever (a poison loop). Enhancement is best-effort, so
            # degrade to llm=None instead. The wizard rejects a bad URL up front; this
            # backstops an env-set or otherwise-persisted one.
            logger.warning(
                "LLM enhancement is enabled but the client could not be built "
                "(likely a malformed LLM base URL); proceeding with enhancement "
                "disabled for this run: %s",
                exc,
            )
            llm = None
    elif prefs.llm_enabled and not key_ok:
        # Only reachable on the BYO path (bundled makes key_ok True); the bundled
        # endpoint never emits this no-key warning.
        logger.warning(
            "LLM enhancement is enabled but no API key is configured (neither a "
            "UI-stored key nor env LLM_API_KEY); proceeding with enhancement "
            "disabled for this run."
        )
    elif prefs.llm_enabled and not budget_ok:
        logger.warning(
            "LLM enhancement is enabled but the run budget plus worst-case "
            "overrun no longer fits the enhance_match stage lease; proceeding "
            "with enhancement disabled for this run to avoid a mid-flight "
            "lease reclaim. Lower llm_run_budget_seconds or raise "
            "stage_lease_seconds."
        )
    return dataclasses.replace(
        base,
        llm=llm,
        llm_bundled=bundled and llm is not None,
        domain_pack=pack,
        enhancement_context=enhancement_context,
        vocabulary=vocabulary,
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
