"""Single sanctioned writer + read side for the LLM clean-up variant (#758).

:func:`record_cleanup` is the only way a successful generation persists: one
immutable ``run_cleanups`` row per (run, generation), mirroring
``translations.record_translation`` (advisory lock per run, monotonic
generation, supersede in the same transaction, idempotent replay by digest).

The writer trusts nothing the producer computed. It takes each line's raw
LLM proposal (``None`` when the reply for that line could not be parsed) and
derives everything stored from it: the validator outcome
(:func:`~voxint.enrichment.cleanup.validate_proposal`), the deleted anchors,
the rendered text (:func:`~voxint.enrichment.cleanup.render_cleaned`) and the
``counts``. A deletion the validator would not choose, or a rejection reason
that did not happen, therefore cannot be stored. The proposals themselves are
not stored: the variant never carries the model's characters.

The source is the translation source (the CORRECTED ``attributed_transcript``
lines) plus each line's #757 anchors from the matching
:func:`~voxint.adjudication.attribution.walk_attributions` emission. Freshness
is :func:`~voxint.enrichment.translations.translation_source_hash`, unchanged,
so speaker renames never stale a clean-up and any text edit, split or unsplit
does.
"""

import hashlib
import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from sqlalchemy import func, select, text, update
from sqlalchemy.orm import Session, defer

from voxint.adjudication.attribution import walk_attributions
from voxint.adjudication.transcript import TranscriptText
from voxint.adjudication.turns import WordMarkKey, emission_anchors, selected_text
from voxint.db.models import PipelineRun, RunCleanup
from voxint.enrichment.cleanup import (
    REJECT_REASONS,
    Accepted,
    Outcome,
    Rejected,
    RejectReason,
    SourceLine,
    render_cleaned,
    source_line,
    validate_proposal,
)
from voxint.enrichment.translations import (
    MAX_LINES,
    TranslationError,
    TranslationLineSource,
    TranslationSource,
    load_translation_source,
    translation_source_hash,
)
from voxint.export.filler_lists import FillerList
from voxint.idempotency import savepoint_adopt_or_conflict

PAYLOAD_SCHEMA_VERSION = 1

# The console export routes' ``text=`` value for this variant. It is never a
# TranscriptText member, so the API, the CLI and every default stay unaware.
CLEANED_TEXT = "cleaned"

MAX_PRODUCER_CHARS = 200
MAX_MODEL_CHARS = 200
# Each line stores its source and its cleaned text, so the cap is the
# translation cap doubled; MAX_LINES is enforced by load_translation_source.
MAX_LINES_PAYLOAD_BYTES = 16_000_000
MAX_CONFIG_BYTES = 64_000

LineOutcome = Literal["changed", "unchanged", "rejected"]

_CHANGED = "the transcript changed while it was being read; try again"
# A deletion-only reply is never much longer than its source line; anything
# past this is runaway generation, refused before the O(n*m) alignment.
PROPOSAL_SLACK_CHARS = 200


class CleanupError(Exception):
    """The clean-up layer refuses to load a source or persist a generation."""


class ConflictingReplayError(CleanupError):
    """An idempotency key was reused with a different payload."""

    def __init__(self, idempotency_key: str) -> None:
        super().__init__(
            f"idempotency key {idempotency_key!r} already used with a different payload"
        )
        self.idempotency_key = idempotency_key


@dataclass(frozen=True)
class CleanupLineSource:
    """One frozen transcript line and its anchored words."""

    line: TranslationLineSource
    words: SourceLine


@dataclass(frozen=True)
class CleanupSource:
    """Everything a clean-up generation reads; ``translation`` is hashed."""

    translation: TranslationSource
    lines: tuple[CleanupLineSource, ...]

    @property
    def pipeline_run_id(self) -> uuid.UUID:
        return self.translation.pipeline_run_id


def load_cleanup_source(session: Session, pipeline_run_id: uuid.UUID) -> CleanupSource:
    """Freeze the CORRECTED lines with the #757 anchors of their emissions.

    ``attributed_transcript`` builds exactly one line per
    ``walk_attributions`` emission, in order; the two are separate reads, so
    any disagreement (a commit landing between them under READ COMMITTED)
    raises :class:`CleanupError` rather than pairing a line with the wrong
    anchors. Callers wanting one snapshot run under REPEATABLE READ.
    """
    if session.get(PipelineRun, pipeline_run_id) is None:
        raise CleanupError(f"unknown pipeline run: {pipeline_run_id}")
    emissions = list(walk_attributions(session, pipeline_run_id))
    if not emissions:
        raise CleanupError(
            "run has no transcript, so there is nothing to clean up"
            " (transcription has not finished, or it found no speech)"
        )
    if len(emissions) > MAX_LINES:
        raise CleanupError(
            f"run has {len(emissions)} transcript lines against the {MAX_LINES}-line bound"
        )
    try:
        translation = load_translation_source(session, pipeline_run_id)
    except TranslationError as exc:
        raise CleanupError(_CHANGED) from exc
    if len(emissions) != len(translation.lines):
        raise CleanupError(_CHANGED)
    lines: list[CleanupLineSource] = []
    for line, emission in zip(translation.lines, emissions, strict=True):
        child = emission.child
        if (
            emission.seg.id != line.segment_id
            or (child.word_start if child else None) != line.word_start
            or (child.word_end if child else None) != line.word_end
            or selected_text(emission, TranscriptText.CORRECTED) != line.text
        ):
            raise CleanupError(_CHANGED)
        anchors = emission_anchors(emission, text=TranscriptText.CORRECTED).anchors
        lines.append(CleanupLineSource(line, source_line(line.text, anchors)))
    return CleanupSource(translation, tuple(lines))


def proposal_ceiling(source_text: str) -> int:
    """The longest proposal the writer accepts for one source line."""
    return len(source_text) + PROPOSAL_SLACK_CHARS


def filler_list_snapshot(fillers: FillerList) -> dict[str, Any]:
    """The filler list as generation provenance (the prompt's hints)."""
    return {
        "preset_version": fillers.preset_version,
        "words": list(fillers.words),
        "phrases": list(fillers.phrases),
        "kept": list(fillers.kept),
    }


@dataclass(frozen=True)
class _CheckedLine:
    entry: dict[str, Any]
    outcome: LineOutcome
    words_removed: int


def _derive_line(source: CleanupLineSource, proposal: str | None) -> _CheckedLine:
    """Judge one line's raw proposal; ``None`` means the reply was malformed."""
    index = source.line.line_index
    words = source.words
    if proposal is not None and not isinstance(proposal, str):
        raise CleanupError(f"line {index}: proposal is not a string")
    if proposal is not None and "\x00" in proposal:
        raise CleanupError(f"line {index}: proposal contains NUL")
    if proposal is not None and len(proposal) > proposal_ceiling(words.text):
        raise CleanupError(
            f"line {index}: proposal is {len(proposal)} chars against a"
            f" {proposal_ceiling(words.text)}-char bound"
        )
    outcome: Outcome = (
        Rejected("malformed") if proposal is None else validate_proposal(words, proposal)
    )
    deleted_ranges: list[list[int]] = []
    removed = 0
    reason: RejectReason | None = None
    if isinstance(outcome, Accepted):
        line_outcome: LineOutcome = "changed"
        text_value = render_cleaned(words, outcome.deleted)
        for i in sorted(outcome.deleted, key=lambda i: words.words[i].anchor.lex_start):
            anchor = words.words[i].anchor
            deleted_ranges.append(
                [anchor.token_start, anchor.token_end, anchor.lex_start, anchor.lex_end]
            )
            removed += len(words.words[i].keys)
    elif isinstance(outcome, Rejected):
        line_outcome, text_value, reason = "rejected", words.text, outcome.reason
    else:
        line_outcome, text_value = "unchanged", words.text
    line = source.line
    return _CheckedLine(
        {
            "i": index,
            "segment_id": str(line.segment_id) if line.segment_id is not None else None,
            "word_start": line.word_start,
            "word_end": line.word_end,
            "source": line.text,
            "text": text_value,
            "deleted": deleted_ranges,
            "outcome": line_outcome,
            "reason": reason,
        },
        line_outcome,
        removed,
    )


def _json_exact(value: object) -> bool:
    """True when ``value`` survives a JSONB round trip unchanged.

    Floats are refused: JSONB rewrites numbers (``1e20`` reads back as an
    integer), which would make an identical replay look like a conflict, and
    NaN, infinities and NUL are not storable at all.
    """
    if value is None or isinstance(value, bool | int):
        return True
    if isinstance(value, str):
        return "\x00" not in value
    if isinstance(value, list):
        return all(_json_exact(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and "\x00" not in key and _json_exact(item)
            for key, item in value.items()
        )
    return False


def _check_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Require ``prompt_version`` and the filler-list snapshot; any other keys
    (the producer's batch knobs) are free-form JSON provenance."""
    if not isinstance(config, Mapping):
        raise CleanupError("config must be an object")
    plain = dict(config)
    try:
        # Serializing first refuses cycles, huge integers and unknown types.
        encoded = _canonical(plain)
    except (TypeError, ValueError) as exc:
        raise CleanupError("config must be JSON-serializable") from exc
    if not _json_exact(plain):
        raise CleanupError(
            "config must hold only strings without NUL, integers, booleans, null,"
            " lists and objects"
        )
    version = config.get("prompt_version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise CleanupError("config.prompt_version must be a positive integer")
    fillers = config.get("filler_list")
    if not isinstance(fillers, Mapping):
        raise CleanupError("config.filler_list must be an object")
    if not isinstance(fillers.get("preset_version"), str):
        raise CleanupError("config.filler_list.preset_version must be a string")
    for key in ("words", "phrases", "kept"):
        entries = fillers.get(key)
        if not isinstance(entries, list) or not all(isinstance(e, str) for e in entries):
            raise CleanupError(f"config.filler_list.{key} must be a list of strings")
    if len(encoded.encode("utf-8")) > MAX_CONFIG_BYTES:
        raise CleanupError(f"config over {MAX_CONFIG_BYTES} bytes")
    loaded: dict[str, Any] = json.loads(encoded)
    return loaded


def _canonical(value: object) -> str:
    """Type-sensitive canonical JSON (``True`` and ``1`` differ, unlike ``==``)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(lines: list[dict[str, Any]]) -> str:
    return hashlib.sha256(_canonical(lines).encode("utf-8")).hexdigest()


def record_cleanup(
    session: Session,
    *,
    source: CleanupSource,
    proposals: Mapping[int, str | None],
    config: Mapping[str, Any],
    model: str,
    producer: str,
    producer_version: str,
    started_at: datetime,
    completed_at: datetime,
    idempotency_key: str,
) -> RunCleanup:
    """Atomically persist one generation and supersede the run's previous head.

    ``proposals`` must cover every source line exactly: the model's text for
    the line, or ``None`` when its reply could not be parsed (``malformed``).
    A generation in which every line is ``malformed`` is refused: no line got
    a parseable reply, so storing it would report a broken model as a success.

    Returns the existing row on an identical replay; raises
    :class:`ConflictingReplayError` when ``idempotency_key`` was already used
    with a different payload.
    """
    if not idempotency_key.strip():
        raise CleanupError("idempotency_key must be non-empty")
    for label, value, cap in (
        ("producer", producer, MAX_PRODUCER_CHARS),
        ("producer_version", producer_version, MAX_PRODUCER_CHARS),
        ("model", model, MAX_MODEL_CHARS),
    ):
        if not value.strip() or len(value) > cap:
            raise CleanupError(f"{label} empty or over {cap} chars")
    for label, stamp in (("started_at", started_at), ("completed_at", completed_at)):
        if stamp.tzinfo is None:
            raise CleanupError(f"{label} must be timezone-aware")
    # UTC so replay equality survives a DST fold and a database round trip.
    started_at, completed_at = started_at.astimezone(UTC), completed_at.astimezone(UTC)
    if completed_at < started_at:
        raise CleanupError("completed_at precedes started_at")
    expected = {line.line.line_index for line in source.lines}
    if set(proposals) != expected:
        missing = sorted(expected - set(proposals))[:5]
        extra = sorted(set(proposals) - expected, key=repr)[:5]
        raise CleanupError(
            f"proposals do not cover the source exactly: missing {missing}, unknown {extra}"
        )
    checked = [_derive_line(line, proposals[line.line.line_index]) for line in source.lines]
    if all(line.entry["reason"] == "malformed" for line in checked):
        raise CleanupError("no line received a usable reply from the model")
    rejected = dict.fromkeys(REJECT_REASONS, 0)
    for line in checked:
        if line.entry["reason"] is not None:
            rejected[line.entry["reason"]] += 1
    counts: dict[str, Any] = {
        "lines": len(checked),
        "lines_changed": sum(line.outcome == "changed" for line in checked),
        "words_removed": sum(line.words_removed for line in checked),
        "rejected": rejected,
    }
    config_value = _check_config(config)
    lines = [line.entry for line in checked]
    if len(json.dumps(lines, ensure_ascii=False).encode("utf-8")) > MAX_LINES_PAYLOAD_BYTES:
        raise CleanupError(f"clean-up payload over {MAX_LINES_PAYLOAD_BYTES} bytes")
    hash_value = translation_source_hash(source.translation)
    digest = _digest(lines)
    run_id = source.pipeline_run_id

    def _existing() -> RunCleanup | None:
        return session.execute(
            select(RunCleanup)
            .options(defer(RunCleanup.lines))
            .where(RunCleanup.idempotency_key == idempotency_key)
        ).scalar_one_or_none()

    def _adopt_or_conflict(row: RunCleanup) -> RunCleanup:
        # completed_at is excluded: the executor stamps it from the clock at
        # finalization, so a redelivery after an unacked commit differs there.
        if (
            row.pipeline_run_id == run_id
            and row.replay_digest == digest
            and _canonical(row.counts) == _canonical(counts)
            and _canonical(row.config) == _canonical(config_value)
            and row.payload_schema_version == PAYLOAD_SCHEMA_VERSION
            and row.producer == producer
            and row.producer_version == producer_version
            and row.model == model
            and row.source_content_hash == hash_value
            and row.started_at == started_at
        ):
            return row
        raise ConflictingReplayError(idempotency_key)

    existing = _existing()
    if existing is not None:
        return _adopt_or_conflict(existing)

    session.execute(
        text("SELECT pg_advisory_xact_lock(hashtext(:producer), hashtext(:scope))"),
        {"producer": "cleanups", "scope": str(run_id)},
    )

    def _persist() -> RunCleanup:
        generation = (
            session.execute(
                select(func.coalesce(func.max(RunCleanup.generation), 0)).where(
                    RunCleanup.pipeline_run_id == run_id
                )
            ).scalar_one()
            + 1
        )
        row = RunCleanup(
            pipeline_run_id=run_id,
            generation=generation,
            lines=lines,
            counts=counts,
            config=config_value,
            payload_schema_version=PAYLOAD_SCHEMA_VERSION,
            producer=producer,
            producer_version=producer_version,
            model=model,
            source_content_hash=hash_value,
            idempotency_key=idempotency_key,
            replay_digest=digest,
            started_at=started_at,
            completed_at=completed_at,
        )
        session.add(row)
        session.flush()
        session.execute(
            update(RunCleanup)
            .where(
                RunCleanup.pipeline_run_id == run_id,
                RunCleanup.generation < generation,
                RunCleanup.superseded_by_cleanup_id.is_(None),
            )
            .values(superseded_by_cleanup_id=row.id)
        )
        return row

    return savepoint_adopt_or_conflict(
        session, lookup=_existing, adopt_or_conflict=_adopt_or_conflict, persist=_persist
    )


def current_cleanup(session: Session, pipeline_run_id: uuid.UUID) -> RunCleanup | None:
    """The run's current (unsuperseded) generation, if any."""
    return session.execute(
        select(RunCleanup).where(
            RunCleanup.pipeline_run_id == pipeline_run_id,
            RunCleanup.superseded_by_cleanup_id.is_(None),
        )
    ).scalar_one_or_none()


def cleanup_texts(row: RunCleanup) -> list[str]:
    """Stored cleaned strings in line order. Callers check freshness first."""
    return [str(line["text"]) for line in row.lines]


def cleanup_omits(row: RunCleanup) -> dict[WordMarkKey, Literal["omit"]]:
    """The stored deletions as #757 omit marks, for the turns projection."""
    omits: dict[WordMarkKey, Literal["omit"]] = {}
    for line in row.lines:
        for token_start, token_end, _lex_start, _lex_end in line["deleted"]:
            omits[(uuid.UUID(line["segment_id"]), token_start, token_end)] = "omit"
    return omits
