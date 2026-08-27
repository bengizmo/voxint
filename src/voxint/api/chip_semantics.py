"""Ops Console chip semantics (V3 #208).

Canonical label-to-semantic mapping for the chip system. The Jinja2
macro in ``fragments/_chips.html`` is the rendering layer; this module
is the referenceable Python constant for contract tests and any backend
code that needs to classify a label.
"""

from __future__ import annotations

CHIP_SEMANTICS: dict[str, str] = {
    "verified": "ok",
    "reviewed": "ok",
    "all reviewed": "ok",
    "done": "ok",
    "transcribing": "info",
    "needs you": "warn",
    "failed": "danger",
    "watched": "accent",
    "learned": "info",
    "unknown": "neutral",
}
"""Map canonical chip label → semantic class suffix (ok/warn/danger/info/accent/neutral)."""
