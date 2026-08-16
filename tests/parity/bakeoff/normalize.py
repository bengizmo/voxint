"""The one frozen text normalizer for every bakeoff WER/CER number.

The pre-registered gate (``docs/gpu-contracts.md``) requires a single, versioned
normalizer applied *identically* to hypothesis, gold, and the frozen CT2
baseline. This module is that normalizer: a thin wrapper over the verbatim,
sha-pinned OpenAI Whisper ``EnglishTextNormalizer`` vendored under
``_vendor/openai_whisper_normalizers/``.

Why the Whisper normalizer and not a bespoke lowercase/punctuation pipeline:
this is a Whisper-vs-Whisper, English-only bakeoff, so the community-standard
Whisper normalizer is the correct apples-to-apples denominator. It deliberately
equates numbers, contractions, titles, diacritics, and British/American
spellings — which is exactly why the gate *also* reports raw (un-normalized)
WER, and why scoring always normalizes raw gold, raw baseline, and raw candidate
text *together* rather than trusting any stored normalized text as authority.

What actually freezes the output (not just the API):
  * the vendored ``.py`` + ``english.json`` byte digests (``provenance.json``),
  * the exact-pinned ``more-itertools`` / ``regex`` in the ``parity`` extra,
  * the recorded runtime fingerprint (:func:`runtime_fingerprint`) — the Python
    minor and ``unicodedata`` Unicode version, the likeliest source of
    exotic-character drift,
  * the golden input→output vectors asserted by the contract test.

``normalize_text`` strips edge whitespace (upstream can leave it); that stripping
is part of the versioned wrapper behavior, bumped via ``WRAPPER_REVISION``.

Scoring conventions (pre-registered here so they are fixed before the harness in
Slice 4 measures anything — see the gate in ``docs/gpu-contracts.md``):
  * WER and CER are computed with jiwer (exact-pinned ``4.0.0``) on text passed
    through :func:`normalize_text`.
  * CER uses jiwer's default character treatment, which **counts internal
    spaces as characters** — we keep that default rather than collapsing
    whitespace, so word-boundary errors remain visible in CER.
  * Empty-reference cases (e.g. normalized silence / filler-only fixtures) follow
    jiwer 4.x's documented empty-reference semantics; the zero-insertion gate,
    not WER, is the authority for true non-speech.
"""

from __future__ import annotations

import functools
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any

from ._vendor.openai_whisper_normalizers import EnglishTextNormalizer

# Bump when the wrapper's own behavior changes (e.g. the strip policy below),
# independently of an upstream re-vendor. The upstream commit is read from
# provenance.json so there is a single source of truth for it.
WRAPPER_REVISION = "voxint-wrapper-v1"

_PROVENANCE_PATH = Path(__file__).parent / "_vendor" / "provenance.json"


@functools.lru_cache(maxsize=1)
def _provenance() -> dict[str, Any]:
    return json.loads(_PROVENANCE_PATH.read_text())


@functools.lru_cache(maxsize=1)
def _normalizer() -> EnglishTextNormalizer:
    # EnglishTextNormalizer loads english.json on construction; build once.
    return EnglishTextNormalizer()


def upstream_commit() -> str:
    """The pinned openai/whisper commit the vendored normalizer came from."""
    return str(_provenance()["upstream_commit"])


# Binds the exact upstream source AND the wrapper behavior, e.g.
# "openai-whisper@5f86d1d.../voxint-wrapper-v1". Recorded in every baseline and
# bakeoff result so a number can never be silently paired with a different
# normalizer.
NORMALIZER_VERSION = f"openai-whisper@{upstream_commit()}/{WRAPPER_REVISION}"


def normalize_text(text: str) -> str:
    """Frozen normalization for WER/CER: Whisper ``EnglishTextNormalizer`` + strip.

    Apply to raw hypothesis, raw gold, and raw baseline alike, at scoring time.
    """
    return _normalizer()(text).strip()


def runtime_fingerprint() -> dict[str, str]:
    """Runtime facts that can move normalized output, for provenance records."""
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "unicodedata_unidata_version": unicodedata.unidata_version,
        "normalizer_version": NORMALIZER_VERSION,
    }
