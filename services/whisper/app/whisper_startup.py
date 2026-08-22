"""Fail-closed startup policy for the whisper model selection.

large-v2 is the only *validated* transcription model (v3/turbo hallucinate —
see the module docstring in ``transcription.py``). Operators may still opt into
an alternate model, but only through an explicit, warned gate: the default path
stays byte-identical and network-free, and any deviation from it fails closed
with a message naming exactly what to set.

This module is a **pure resolver** (``resolve_whisper_startup``) over an env
mapping plus a thin impure applier (``apply_whisper_startup``) that mutates
``os.environ`` and logs before the model libraries load. Keeping the decision
pure mirrors the diarizer's resolution style (``services/pyannote/app/diarizer.py``)
and lets the whole truth table be unit-tested without a model or a container.

The ``HF_HUB_OFFLINE`` subtlety: the image always bakes ``HF_HUB_OFFLINE=1`` so
the default is air-gap-safe, which means an inherited env cannot tell an
operator-set ``HF_HUB_OFFLINE=1`` apart from the image default. So this resolver
does not read ``HF_HUB_OFFLINE`` to infer intent. Instead ``WHISPER_ALLOW_DOWNLOAD=1``
is the single, documented authority that permits a network fetch: when it is set
(with a valid alternate revision) the resolver turns offline off for the alternate
cache. That is the direct, logged consequence of the opt-in, not a silent flip.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# The validated default, in both the alias and the fully-qualified repo-id forms
# faster-whisper accepts for it. Either resolves to the baked, offline snapshot.
DEFAULT_MODELS = frozenset({"large-v2", "Systran/faster-whisper-large-v2"})

# Alternate weights download into a SEPARATE writable cache, never over the baked
# large-v2 download root (/app/.cache/whisper) — a volume there would shadow the
# baked model and retain stale contents across image upgrades.
ALT_CACHE_ROOT = "/app/model-cache/whisper"

# A downloadable alternate must pin a full 40-char lowercase commit SHA. Symbolic
# refs (main, v3), short shas, and uppercase are rejected so the fetch is
# reproducible and the stamped provenance means something.
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class WhisperStartupError(RuntimeError):
    """A misconfigured whisper model selection; the service must not start."""


@dataclass(frozen=True)
class WhisperStartup:
    """The resolved startup decision.

    ``env_overrides`` is what the applier writes into ``os.environ`` before the
    model libraries load. For the default it is empty (nothing changes); for an
    alternate it points the download root at the separate cache and enables the
    fetch. ``warning`` is a one-line honest notice logged for alternates.
    """

    model_name: str
    is_override: bool
    env_overrides: Mapping[str, str]
    warning: str | None


def _clean(env: Mapping[str, str], key: str) -> str:
    """Read a stripped env value (missing/whitespace both collapse to '')."""
    return (env.get(key) or "").strip()


def resolve_whisper_startup(env: Mapping[str, str]) -> WhisperStartup:
    """Decide the whisper startup policy from ``env``. Pure; raises on misconfig.

    Raises ``WhisperStartupError`` for any alternate selection that is not fully,
    explicitly gated. The default (``large-v2`` / the baked repo id, or an unset
    ``WHISPER_MODEL``) returns a no-op decision so the shipped path is untouched.
    """
    raw = env.get("WHISPER_MODEL")
    # Unset means the baked default; an explicitly-blank value is a mistake we
    # refuse rather than silently treat as the default.
    if raw is None:
        model = "large-v2"
    else:
        model = raw.strip()
        if not model:
            raise WhisperStartupError(
                "WHISPER_MODEL is set but empty. Unset it to use the validated "
                "large-v2 default, or set it to an exact model id."
            )

    if model in DEFAULT_MODELS:
        # The validated large-v2 name must load the baked large-v2 weights, not a
        # different snapshot under the same name (#125). An operator revision that
        # is present AND differs from the baked snapshot would otherwise load
        # different weights while still reporting the validated name — no download
        # gate, no warning. Fail closed: only the baked revision is valid on the
        # default name; an alternate build must go through the explicit
        # WHISPER_ALLOW_DOWNLOAD path with its own model id. The guard requires the
        # baked reference to be present, so a bare dev venv without
        # WHISPER_BAKED_REVISION is not broken.
        operator_revision = _clean(env, "WHISPER_REVISION")
        baked = _clean(env, "WHISPER_BAKED_REVISION")
        if operator_revision and baked and operator_revision != baked:
            raise WhisperStartupError(
                f"WHISPER_MODEL={model!r} is the validated default, but "
                f"WHISPER_REVISION={operator_revision} is not the baked large-v2 "
                f"snapshot ({baked}). Unset WHISPER_REVISION to load the validated "
                "weights, or select an alternate model with WHISPER_MODEL plus "
                "WHISPER_ALLOW_DOWNLOAD=1 and that model's own commit SHA."
            )
        # The whisper compose overlays forward ``WHISPER_REVISION: ${WHISPER_REVISION:-}``,
        # which sets the container's WHISPER_REVISION to an EMPTY string whenever the
        # operator does not override it. An empty (revision-less) load resolves the
        # "main" ref, which HF_HUB_OFFLINE cannot satisfy for the baked snapshot (the
        # bake writes the commit snapshot, not refs/main), so the default path would
        # fail to load offline. Restore the baked revision from WHISPER_BAKED_REVISION
        # (baked from the same ARG and never compose-forwarded) so the shipped default
        # stays byte-identical and network-free even under the empty pass-through.
        overrides: dict[str, str] = {}
        if not operator_revision and baked:
            overrides["WHISPER_REVISION"] = baked
        return WhisperStartup(
            model_name=model, is_override=False, env_overrides=overrides, warning=None
        )

    # A local path or a deep repo id would bypass the download-and-pin policy
    # (faster-whisper will load a directory verbatim). Only Hub ids are supported
    # for alternates: a bare alias or a single ``org/name``. A value with that
    # shape can still name a real relative directory; the resolver stays pure
    # (no filesystem access), so ``apply_whisper_startup`` closes that residual
    # with an existence check at boot.
    if model.startswith(("/", ".", "~")) or model.count("/") > 1:
        raise WhisperStartupError(
            f"WHISPER_MODEL={model!r} looks like a filesystem path. Alternate "
            "models must be a Hugging Face repo id (for example "
            "'org/faster-whisper-name'); local paths are not supported."
        )

    if _clean(env, "WHISPER_ALLOW_DOWNLOAD") != "1":
        raise WhisperStartupError(
            f"WHISPER_MODEL={model!r} is not the validated default (large-v2). "
            "To load an alternate model you must opt in explicitly: set "
            "WHISPER_ALLOW_DOWNLOAD=1 and WHISPER_REVISION to that model's full "
            "40-character commit SHA. Alternate models are an unvalidated "
            "mechanism (only large-v2 is validated; v3 and turbo hallucinate)."
        )

    revision = _clean(env, "WHISPER_REVISION")
    if not _SHA_RE.fullmatch(revision):
        raise WhisperStartupError(
            f"WHISPER_MODEL={model!r} requires WHISPER_REVISION set to a full "
            "40-character lowercase commit SHA "
            f"(got {revision or 'unset'!r}); symbolic refs like 'main' or 'v3' "
            "and short SHAs are rejected so the download is reproducible."
        )

    # The image bakes WHISPER_REVISION to the large-v2 snapshot, which is itself a
    # valid 40-char SHA. An operator who sets an alternate model but forgets to
    # change the revision would otherwise pass the check above with the wrong
    # model's SHA. WHISPER_BAKED_REVISION carries the baked reference so we can
    # reject that exact mistake with an actionable message.
    baked = _clean(env, "WHISPER_BAKED_REVISION")
    if baked and revision == baked:
        raise WhisperStartupError(
            f"WHISPER_REVISION still points at the baked large-v2 snapshot "
            f"({baked}); set it to {model!r}'s own 40-character commit SHA."
        )

    warning = (
        f"Loading ALTERNATE ASR model {model!r} at revision {revision}. This is an "
        "unvalidated opt-in: only large-v2 is validated and v3/turbo hallucinate. "
        f"Network fetch is enabled and weights cache at {ALT_CACHE_ROOT}."
    )
    return WhisperStartup(
        model_name=model,
        is_override=True,
        env_overrides={
            "WHISPER_DOWNLOAD_ROOT": ALT_CACHE_ROOT,
            # Explicit, logged consequence of WHISPER_ALLOW_DOWNLOAD=1 — never a
            # silent unset of an operator's offline intent (see module docstring).
            "HF_HUB_OFFLINE": "0",
        },
        warning=warning,
    )


def apply_whisper_startup(
    environ: MutableMapping[str, str] | None = None,
) -> WhisperStartup:
    """Resolve and apply the startup policy to ``environ`` (default ``os.environ``).

    Writes the decision's env overrides and logs the alternate-model warning
    before the model libraries import, so ``HF_HUB_OFFLINE`` takes effect. Returns
    the decision so the caller can pass ``model_name`` to ``create_transcriber``.
    Propagates ``WhisperStartupError`` so a misconfigured service fails to start.

    Also the impure half of the alternate-model path guard: the pure resolver
    rejects path-shaped values by syntax alone, but a single-slash ``org/name``
    can also name a real relative directory, which faster-whisper would load
    verbatim, bypassing the pinned download. The existence check lives here so
    the resolver stays filesystem-free; it raises before any env write, so a
    refused boot leaves ``environ`` untouched.
    """
    target = os.environ if environ is None else environ
    decision = resolve_whisper_startup(target)
    if decision.is_override and os.path.isdir(decision.model_name):
        raise WhisperStartupError(
            f"WHISPER_MODEL={decision.model_name!r} names an existing local "
            "directory; faster-whisper would load it verbatim, bypassing the "
            "pinned download. Local paths are not supported: remove or rename "
            "the directory, or use a Hugging Face repo id."
        )
    for key, value in decision.env_overrides.items():
        target[key] = value
    if decision.warning:
        logger.warning(decision.warning)
    return decision
