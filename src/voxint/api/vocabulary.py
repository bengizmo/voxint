"""Ops Console plain-language vocabulary contract (COPY #209).

Maps internal/technical terms to their user-facing equivalents. This is
the single source of truth; the Jinja2 templates and the contract test
both reference it. R1-R6 screen refreshes apply these mappings; legacy
templates carry an allowlist until they are retired.

Scope: user-facing template text only. Internal identifiers (DB columns,
API fields, Python names), logs, CLI output, and Settings ``<details>``
advanced disclosures are explicitly excluded.
"""

from __future__ import annotations

VOCABULARY: dict[str, str] = {
    "diarization": "separate voices",
    "diarize": "separate voices",
    "embedding": "voice sample",
    "enrollment": "voice sample",
    "enroll": "add a voice sample",
    "adjudication": "review",
    "adjudicate": "review",
    "ruling": "decision",
    "needs_ruling": "needs you",
    "needs ruling": "needs you",
    "run finalized": "done",
    "language model": "local AI model",
}
"""Internal term -> plain-language user-facing equivalent."""

PROHIBITED_PATTERNS: tuple[str, ...] = (
    "diarization",
    "diarize",
    "adjudication",
    "adjudicate",
    "needs ruling",
    "needs_ruling",
)
"""Terms that must not appear as user-facing text in Ops Console templates.

``embedding`` and ``enrollment`` are excluded from the prohibited list
because they appear legitimately in template variable access
(``item.embedding``, ``speaker.enrollments``) and in Settings detail
prose where the technical term is appropriate.
"""
