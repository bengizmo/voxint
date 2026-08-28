"""Quote provenance manifest (issue #122).

Pure builder: no DB, no HTTP, no clock. The route handler gathers data and
passes primitives; this module serializes them into a JSON manifest for
journalists who need to defend a quote's provenance chain.

Temporal layers are explicit: ``pipeline_provenance`` is what the pipeline
observed at run time, ``quote.annotation_updated_at`` is the operator's
last edit, and ``exported_at`` is when this manifest was generated.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class QuoteLine:
    """One line of quoted text with its speaker and timing."""

    text: str
    speaker: str
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class ClipRef:
    """A reference to an extracted audio clip with integrity metadata."""

    id: uuid.UUID
    download_url: str
    filename: str
    sha256: str
    sample_rate: int
    channels: int
    start_sample: int
    end_sample: int


@dataclass(frozen=True)
class StageRole:
    """One model service role's observed identity."""

    reachable: bool
    model: str | None = None
    revision: str | None = None
    engine: str | None = None


@dataclass(frozen=True)
class StageProvenance:
    """Model identity for one pipeline stage's latest completed attempt."""

    attempt: int
    finished_at: datetime | None
    roles: dict[str, StageRole]


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt is not None else None


def _clip_dict(clip: ClipRef) -> dict[str, Any]:
    return {
        "id": clip.id.hex,
        "download_url": clip.download_url,
        "filename": clip.filename,
        "sha256": clip.sha256,
        "sample_rate": clip.sample_rate,
        "channels": clip.channels,
        "start_sample": clip.start_sample,
        "end_sample": clip.end_sample,
    }


def _stage_dict(sp: StageProvenance) -> dict[str, Any]:
    return {
        "attempt": sp.attempt,
        "finished_at": _iso(sp.finished_at),
        "roles": {
            name: {
                "reachable": role.reachable,
                "model": role.model,
                "revision": role.revision,
                "engine": role.engine,
            }
            for name, role in sp.roles.items()
        },
    }


def build_quote_manifest(
    *,
    exported_at: datetime,
    annotation_id: uuid.UUID,
    source_text_hash: str,
    annotation_updated_at: datetime,
    lines: list[QuoteLine],
    timing_precision: str,
    tags: list[str],
    note: str | None,
    clip: ClipRef | None,
    media_id: uuid.UUID,
    run_id: uuid.UUID,
    source_title: str,
    media_sha256: str | None,
    app_version: str,
    stages: dict[str, StageProvenance],
) -> dict[str, Any]:
    """Build a single-quote provenance manifest."""
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "quote_provenance",
        "exported_at": exported_at.isoformat(),
        "quote": {
            "lines": [
                {
                    "text": ln.text,
                    "speaker": ln.speaker,
                    "start_seconds": ln.start_seconds,
                    "end_seconds": ln.end_seconds,
                }
                for ln in lines
            ],
            "timing_precision": timing_precision,
            "tags": tags,
            "note": note,
            "annotation_id": annotation_id.hex,
            "source_text_hash": source_text_hash,
            "annotation_updated_at": annotation_updated_at.isoformat(),
        },
        "clip": _clip_dict(clip) if clip is not None else None,
        "source": {
            "media_id": media_id.hex,
            "run_id": run_id.hex,
            "title": source_title,
            "media_sha256": media_sha256,
        },
        "pipeline_provenance": {
            "app_version": app_version,
            "observed_before_attempt": True,
            "stages": {
                name: _stage_dict(sp) for name, sp in stages.items()
            },
        },
    }


def build_quote_bundle(
    *,
    exported_at: datetime,
    media_id: uuid.UUID,
    run_id: uuid.UUID,
    source_title: str,
    media_sha256: str | None,
    app_version: str,
    stages: dict[str, StageProvenance],
    quotes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a bulk provenance bundle with run-level facts at the envelope."""
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "quote_provenance_bundle",
        "exported_at": exported_at.isoformat(),
        "source": {
            "media_id": media_id.hex,
            "run_id": run_id.hex,
            "title": source_title,
            "media_sha256": media_sha256,
        },
        "pipeline_provenance": {
            "app_version": app_version,
            "observed_before_attempt": True,
            "stages": {
                name: _stage_dict(sp) for name, sp in stages.items()
            },
        },
        "quotes": quotes,
    }
