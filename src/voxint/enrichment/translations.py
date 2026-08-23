"""Single sanctioned writer + read side for transcript translations (#133).

:func:`record_translation` is the only way a successful generation persists:
one immutable ``run_translations`` row per (run, target language, generation),
validated up front and fail-closed, mirroring ``run_assets.record_asset``:

- serializes per (run, target language) with a transaction-scoped advisory
  lock so generation allocation, insertion, and supersession are one atomic
  step;
- allocates a monotonic ``generation`` under that lock;
- supersedes the prior still-current translation of the *same target
  language only* in the same transaction.

The module also owns the **source snapshot**: :func:`load_translation_source`
freezes the ordered output of :func:`voxint.adjudication.transcript.
attributed_transcript` (the CORRECTED rendition — the same lines the console
and every export show, split children included), and
:func:`translation_source_hash` canonicalizes it into the sha256 freshness
authority stored on every generation. Speaker names are deliberately EXCLUDED
from both the snapshot and the hash — speakers are not sent to the model, so
re-adjudicating or renaming one must never stale a translation. A text edit,
split, or unsplit changes the line list and honestly stales the whole
generation; stale lines are never partially aligned to a changed transcript.
"""

import hashlib
import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import func, select, text, update
from sqlalchemy.orm import Session

from voxint.adjudication.transcript import TranscriptText, attributed_transcript
from voxint.api.languages import LANGUAGE_NAMES
from voxint.db.models import PipelineRun, RunTranslation

SOURCE_SCHEMA_VERSION = 1
PAYLOAD_SCHEMA_VERSION = 1

MAX_PRODUCER_CHARS = 200
MAX_MODEL_CHARS = 200
# JSONB rows must stay comfortably readable in one request; a transcript that
# would blow this is beyond what a one-shot LLM translation can serve anyway.
MAX_LINES = 10_000
MAX_LINES_PAYLOAD_BYTES = 8_000_000


class TranslationError(Exception):
    """A generation submitted something the translation layer refuses to persist."""


def translated_size_ceiling(source_text: str) -> int:
    """Growth bound for one translated line.

    Translations legitimately run 10-30% longer than their source; a reply
    several times the source length is runaway generation or refusal prose,
    not a translation. The additive floor keeps short lines (greetings,
    single words) from being rejected for ordinary expansion.
    """
    return max(3 * len(source_text), len(source_text) + 500)


@dataclass(frozen=True)
class TranslationLineSource:
    """One frozen transcript line as the translator sees it.

    ``line_index`` is the identity within this snapshot (and within the
    generation it produces). The segment/word-range coordinates are frozen
    provenance carried into the stored lines — diagnostics, never read-time
    join keys.
    """

    line_index: int
    segment_id: uuid.UUID | None
    word_start: int | None
    word_end: int | None
    text: str


@dataclass(frozen=True)
class TranslationSource:
    """Everything a translation generation reads — the hashed freshness domain."""

    pipeline_run_id: uuid.UUID
    source_language: str | None
    lines: tuple[TranslationLineSource, ...]


def load_translation_source(session: Session, pipeline_run_id: uuid.UUID) -> TranslationSource:
    """Freeze the exact lines a translation may read, in display order.

    The source is the CORRECTED rendition of :func:`attributed_transcript` —
    the operator-effective text with split children expanded — because a
    translation is a rendition of the *finished* transcript. Raises
    :class:`TranslationError` for an unknown run or one with no transcript yet.
    ``source_language`` is the run's #124 detected language (a prompt hint and
    provenance; ``None`` when detection reported nothing).
    """
    run = session.get(PipelineRun, pipeline_run_id)
    if run is None:
        raise TranslationError(f"unknown pipeline run: {pipeline_run_id}")
    lines = attributed_transcript(session, pipeline_run_id, text=TranscriptText.CORRECTED)
    if not lines:
        raise TranslationError(
            "run has no transcript yet — translation reads the finished"
            " transcript, so the run must finish transcription first"
        )
    if len(lines) > MAX_LINES:
        raise TranslationError(
            f"run has {len(lines)} transcript lines against the {MAX_LINES}-line"
            " translation bound"
        )
    return TranslationSource(
        pipeline_run_id=pipeline_run_id,
        source_language=run.detected_language,
        lines=tuple(
            TranslationLineSource(
                line_index=index,
                segment_id=line.segment_id,
                word_start=line.word_start,
                word_end=line.word_end,
                text=line.text,
            )
            for index, line in enumerate(lines)
        ),
    )


def translation_source_hash(source: TranslationSource) -> str:
    """sha256 over the canonical serialization of the translation inputs.

    The ONLY freshness authority for a generation: deterministic, content-only
    (stable key order, compact separators, explicit nulls), covering the
    ordered structural + text fields plus the source-language hint the prompt
    names. Speaker names are deliberately absent (see module docstring). Model
    and prompt versions are provenance columns, never folded in — a prompt
    upgrade must not masquerade as a source change.
    """
    payload = {
        "source_schema_version": SOURCE_SCHEMA_VERSION,
        "pipeline_run_id": str(source.pipeline_run_id),
        "source_language": source.source_language,
        "lines": [
            [
                line.line_index,
                str(line.segment_id) if line.segment_id is not None else None,
                line.word_start,
                line.word_end,
                line.text,
            ]
            for line in source.lines
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_translated(source: TranslationSource, translated: Mapping[int, str]) -> None:
    expected = {line.line_index for line in source.lines}
    got = set(translated)
    if got != expected:
        missing = sorted(expected - got)[:5]
        extra = sorted(got - expected)[:5]
        raise TranslationError(
            f"translated lines do not cover the source exactly —"
            f" missing {missing}, unknown {extra}"
        )
    for line in source.lines:
        value = translated[line.line_index]
        if not isinstance(value, str):
            raise TranslationError(f"translation for line {line.line_index} is not a string")
        if "\x00" in value:
            raise TranslationError(f"translation for line {line.line_index} contains NUL")
        if line.text.strip() and not value.strip():
            raise TranslationError(
                f"translation for non-empty line {line.line_index} is empty"
            )
        if not line.text.strip() and value.strip():
            raise TranslationError(
                f"translation invents text for empty line {line.line_index}"
            )
        if len(value) > translated_size_ceiling(line.text):
            raise TranslationError(
                f"translation for line {line.line_index} is {len(value)} chars against"
                f" a {translated_size_ceiling(line.text)}-char growth bound"
            )


def record_translation(
    session: Session,
    *,
    source: TranslationSource,
    target_language: str,
    translated: Mapping[int, str],
    model: str,
    producer: str,
    producer_version: str,
    started_at: datetime,
    completed_at: datetime,
) -> RunTranslation:
    """Atomically persist one successful generation and supersede its
    predecessor for the same target language.

    The freshness hash is computed here from ``source`` — the writer, not the
    caller, owns the hash so it always describes what was actually frozen.
    Only complete generations persist: ``translated`` must cover every source
    line exactly (the executor's batch validation should already guarantee
    this; the writer re-checks so no caller can smuggle a partial generation).
    """
    if target_language not in LANGUAGE_NAMES:
        raise TranslationError(f"unknown target language code: {target_language!r}")
    for label, value, cap in (
        ("producer", producer, MAX_PRODUCER_CHARS),
        ("producer_version", producer_version, MAX_PRODUCER_CHARS),
        ("model", model, MAX_MODEL_CHARS),
    ):
        if not value.strip() or len(value) > cap:
            raise TranslationError(f"{label} empty or over {cap} chars")
    for label, stamp in (("started_at", started_at), ("completed_at", completed_at)):
        if stamp.tzinfo is None:
            raise TranslationError(f"{label} must be timezone-aware")
    if completed_at < started_at:
        raise TranslationError("completed_at precedes started_at")
    _validate_translated(source, translated)
    lines = [
        {
            "i": line.line_index,
            "segment_id": str(line.segment_id) if line.segment_id is not None else None,
            "word_start": line.word_start,
            "word_end": line.word_end,
            "source_text": line.text,
            "text": translated[line.line_index],
        }
        for line in source.lines
    ]
    encoded = json.dumps(lines, ensure_ascii=False)
    if len(encoded.encode("utf-8")) > MAX_LINES_PAYLOAD_BYTES:
        raise TranslationError(
            f"translation payload over {MAX_LINES_PAYLOAD_BYTES} bytes"
        )

    # One finalization at a time per (run, target language): generation
    # allocation, insertion, and supersession must be atomic even when the pair
    # has no prior rows to lock. Transaction-scoped, releases on commit/rollback.
    session.execute(
        text("SELECT pg_advisory_xact_lock(hashtext(:producer), hashtext(:scope))"),
        {
            "producer": f"translations:{target_language}",
            "scope": str(source.pipeline_run_id),
        },
    )
    generation = (
        session.execute(
            select(func.coalesce(func.max(RunTranslation.generation), 0)).where(
                RunTranslation.pipeline_run_id == source.pipeline_run_id,
                RunTranslation.target_language == target_language,
            )
        ).scalar_one()
        + 1
    )
    row = RunTranslation(
        pipeline_run_id=source.pipeline_run_id,
        target_language=target_language,
        generation=generation,
        source_language=source.source_language,
        lines=lines,
        payload_schema_version=PAYLOAD_SCHEMA_VERSION,
        producer=producer,
        producer_version=producer_version,
        model=model,
        source_content_hash=translation_source_hash(source),
        started_at=started_at,
        completed_at=completed_at,
    )
    session.add(row)
    session.flush()
    session.execute(
        update(RunTranslation)
        .where(
            RunTranslation.pipeline_run_id == source.pipeline_run_id,
            RunTranslation.target_language == target_language,
            RunTranslation.generation < generation,
            RunTranslation.superseded_by_translation_id.is_(None),
        )
        .values(superseded_by_translation_id=row.id)
    )
    return row


def current_translations(session: Session, pipeline_run_id: uuid.UUID) -> list[RunTranslation]:
    """Every current (unsuperseded) translation head for the run, newest first."""
    return list(
        session.execute(
            select(RunTranslation)
            .where(
                RunTranslation.pipeline_run_id == pipeline_run_id,
                RunTranslation.superseded_by_translation_id.is_(None),
            )
            .order_by(RunTranslation.created_at.desc())
        )
        .scalars()
        .all()
    )


def current_translation(
    session: Session, pipeline_run_id: uuid.UUID, target_language: str
) -> RunTranslation | None:
    """The current head for one (run, target language), if any."""
    return session.execute(
        select(RunTranslation).where(
            RunTranslation.pipeline_run_id == pipeline_run_id,
            RunTranslation.target_language == target_language,
            RunTranslation.superseded_by_translation_id.is_(None),
        )
    ).scalar_one_or_none()
