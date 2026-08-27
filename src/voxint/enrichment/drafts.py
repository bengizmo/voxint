"""Single sanctioned writer for enrichment producer runs, candidates, evidence.

:func:`record_producer_run` is the only way producers persist their output:
one **completed** invocation plus its candidate claims and their evidence, all
in one atomic finalization. The writer:

- validates everything up front (caps, shapes, scope containment) and fails
  closed — a draft layer must never accept a malformed claim;
- serializes per (producer, scope) with a transaction-scoped advisory lock so
  generation allocation, insertion, and supersession are one atomic step even
  for a scope with no prior rows (row locks cannot guard what does not exist);
- allocates a monotonic ``generation`` under that lock — "newer" is a
  generation comparison, never wall-clock, so out-of-order completion cannot
  make an older invocation supersede a newer one's claims;
- derives ``outcome`` (``'none'`` iff zero candidates), never trusts a
  caller's assertion of it;
- supersedes only still-proposed candidates of the *same producer + same
  scope + lower generation* whose field is in the new run's
  ``covered_fields`` — decided candidates are history and are never touched.
  ⚠ Scope here means the **invocation** scope, not the candidate target: a
  ``run``-scope rerun supersedes the run_label candidates its earlier
  ``run``-scope sweeps emitted, but a ``run_label``-scope invocation is a
  *different* scope and will never supersede claims from a ``run``-scope
  sweep (they hold different locks and separate generation counters).
  Producers must re-run at the scope kind they originally used;
- replays idempotently: the same ``idempotency_key`` with the same payload
  returns the existing run, a different payload is an error (pattern:
  ``adjudication/ledger.py``).

Nothing here writes ``speakers``, ``speaker_assignments``, or
``adjudication_decisions`` — drafts are suggestions about identity, not
identity (docs/quality-gates.md).
"""

import json
import math
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import exists, func, select, text, update
from sqlalchemy.orm import Session, selectinload

from voxint.db.models import (
    ClaimField,
    EnrichmentCandidate,
    EnrichmentCandidateEvidence,
    EnrichmentOutcome,
    EnrichmentProducerRun,
    EnrichmentTargetKind,
    EvidenceKind,
    ProfileReviewDecision,
)
from voxint.enrichment.review import ConflictingReplayError
from voxint.idempotency import savepoint_adopt_or_conflict

MAX_PRODUCER_CHARS = 200
MAX_VALUE_CHARS = 4_000
MAX_NAME_VALUE_CHARS = 120
MAX_EVIDENCE_ROWS = 16
MAX_SNIPPET_CHARS = 1_000
MAX_SOURCE_FIELD_CHARS = 200
MAX_URL_CHARS = 2_048
MAX_SCORE_COMPONENTS = 32
MAX_SCORE_COMPONENT_KEY_CHARS = 64
MAX_CONFIG_BYTES = 16_384


class EnrichmentDraftError(Exception):
    """A producer submitted something the draft layer refuses to persist."""


@dataclass(frozen=True)
class EnrichmentScope:
    """The XOR target of an invocation or candidate: speaker | run | run+label."""

    kind: EnrichmentTargetKind
    speaker_id: uuid.UUID | None = None
    pipeline_run_id: uuid.UUID | None = None
    diarization_label: str | None = None

    @classmethod
    def speaker(cls, speaker_id: uuid.UUID) -> "EnrichmentScope":
        return cls(EnrichmentTargetKind.SPEAKER, speaker_id=speaker_id)

    @classmethod
    def run(cls, pipeline_run_id: uuid.UUID) -> "EnrichmentScope":
        return cls(EnrichmentTargetKind.RUN, pipeline_run_id=pipeline_run_id)

    @classmethod
    def run_label(
        cls, pipeline_run_id: uuid.UUID, diarization_label: str
    ) -> "EnrichmentScope":
        return cls(
            EnrichmentTargetKind.RUN_LABEL,
            pipeline_run_id=pipeline_run_id,
            diarization_label=diarization_label,
        )

    def validate(self) -> None:
        by_kind: dict[EnrichmentTargetKind, tuple[bool, bool, bool]] = {
            EnrichmentTargetKind.SPEAKER: (True, False, False),
            EnrichmentTargetKind.RUN: (False, True, False),
            EnrichmentTargetKind.RUN_LABEL: (False, True, True),
        }
        want = by_kind.get(self.kind)
        if want is None:
            raise EnrichmentDraftError(f"unknown target kind: {self.kind!r}")
        got = (
            self.speaker_id is not None,
            self.pipeline_run_id is not None,
            self.diarization_label is not None,
        )
        if got != want:
            raise EnrichmentDraftError(
                f"scope shape mismatch for kind {self.kind}: {self!r}"
            )
        if self.diarization_label is not None and not self.diarization_label.strip():
            raise EnrichmentDraftError("diarization_label must be non-empty")

    def contains(self, candidate: "EnrichmentScope") -> bool:
        """A candidate's target must lie inside the invocation's scope.

        A run-scope invocation may emit run-level *and* run_label-level
        candidates for that run ("Interview with Jane" is a run-level claim;
        a self-introduction inside a cluster is a run_label claim). Speaker
        and run_label scopes admit only their exact target.
        """
        if self == candidate:
            return True
        return (
            self.kind is EnrichmentTargetKind.RUN
            and candidate.kind is EnrichmentTargetKind.RUN_LABEL
            and candidate.pipeline_run_id == self.pipeline_run_id
        )

    def lock_key(self) -> str:
        parts = [self.kind.value]
        for value in (self.speaker_id, self.pipeline_run_id, self.diarization_label):
            if value is not None:
                parts.append(str(value))
        return ":".join(parts)


@dataclass(frozen=True)
class MetadataEvidence:
    """A claim traced to a ``media_source_metadata`` column or ``raw.`` key."""

    source_metadata_id: uuid.UUID
    source_field: str
    snippet: str | None = None
    detail: Mapping[str, Any] | None = None
    detail_schema_version: int | None = None


@dataclass(frozen=True)
class TranscriptEvidence:
    """A claim traced to a transcript segment (+ optional in-media timestamp)."""

    transcript_segment_id: uuid.UUID
    timestamp_seconds: float | None = None
    snippet: str | None = None
    detail: Mapping[str, Any] | None = None
    detail_schema_version: int | None = None


@dataclass(frozen=True)
class UrlEvidence:
    """A claim traced to a fetched web page."""

    url: str
    retrieved_at: datetime | None = None
    snippet: str | None = None
    detail: Mapping[str, Any] | None = None
    detail_schema_version: int | None = None


Evidence = MetadataEvidence | TranscriptEvidence | UrlEvidence


@dataclass(frozen=True)
class CandidateDraft:
    """One claim a producer wants reviewed, with its evidence."""

    target: EnrichmentScope
    field: ClaimField
    value: str
    evidence: tuple[Evidence, ...]
    score: float | None = None
    score_components: Mapping[str, float] = dataclass_field(default_factory=dict)


def _validate_structural_url(url: str) -> None:
    """Refuse anything but an absolute, credential-free http(s) token.

    Mirrors the structural URL policy of ``media/source_metadata.py``: an
    evidence URL is retained for display/navigation, so it must be one
    unbroken http(s) token with no userinfo and no control characters —
    refused outright rather than kept as a mangled remnant.
    """
    if not url or len(url) > MAX_URL_CHARS:
        raise EnrichmentDraftError(f"evidence url empty or over {MAX_URL_CHARS} chars")
    if any(ch.isspace() or ord(ch) < 0x20 for ch in url):
        raise EnrichmentDraftError("evidence url contains whitespace/control chars")
    lowered = url.lower()
    if not (lowered.startswith("https://") or lowered.startswith("http://")):
        raise EnrichmentDraftError(f"evidence url must be http(s): {url[:80]!r}")
    authority = url.split("://", 1)[1].split("/", 1)[0]
    if "@" in authority:
        raise EnrichmentDraftError("evidence url must not embed credentials")


def _validate_common_evidence(item: Evidence) -> None:
    if item.snippet is not None and (
        not item.snippet.strip() or len(item.snippet) > MAX_SNIPPET_CHARS
    ):
        raise EnrichmentDraftError(
            f"evidence snippet empty or over {MAX_SNIPPET_CHARS} chars"
        )
    if (item.detail is None) != (item.detail_schema_version is None):
        raise EnrichmentDraftError(
            "evidence detail and detail_schema_version must be set together"
        )
    if item.detail_schema_version is not None and item.detail_schema_version < 1:
        raise EnrichmentDraftError("detail_schema_version must be >= 1")


def _validate_evidence(item: Evidence) -> None:
    _validate_common_evidence(item)
    if isinstance(item, MetadataEvidence):
        if not item.source_field.strip() or len(item.source_field) > MAX_SOURCE_FIELD_CHARS:
            raise EnrichmentDraftError(
                f"source_field empty or over {MAX_SOURCE_FIELD_CHARS} chars"
            )
    elif isinstance(item, TranscriptEvidence):
        if item.timestamp_seconds is not None and not (
            math.isfinite(item.timestamp_seconds) and item.timestamp_seconds >= 0
        ):
            raise EnrichmentDraftError(
                f"timestamp_seconds must be finite and >= 0: {item.timestamp_seconds}"
            )
    elif isinstance(item, UrlEvidence):
        _validate_structural_url(item.url)


def _validate_candidate(
    draft: CandidateDraft, scope: EnrichmentScope, covered: tuple[ClaimField, ...]
) -> None:
    draft.target.validate()
    if not scope.contains(draft.target):
        raise EnrichmentDraftError(
            f"candidate target {draft.target!r} is outside invocation scope {scope!r}"
        )
    if draft.field not in covered:
        raise EnrichmentDraftError(
            f"candidate field {draft.field} not in covered_fields {covered}"
        )
    value = draft.value
    if not value.strip() or len(value) > MAX_VALUE_CHARS:
        raise EnrichmentDraftError(f"value empty or over {MAX_VALUE_CHARS} chars")
    if draft.field is ClaimField.NAME and len(value) > MAX_NAME_VALUE_CHARS:
        raise EnrichmentDraftError(f"name value over {MAX_NAME_VALUE_CHARS} chars")
    if not (1 <= len(draft.evidence) <= MAX_EVIDENCE_ROWS):
        raise EnrichmentDraftError(
            f"a candidate needs 1..{MAX_EVIDENCE_ROWS} evidence rows,"
            f" got {len(draft.evidence)}"
        )
    for item in draft.evidence:
        _validate_evidence(item)
    if draft.score is not None and not (
        math.isfinite(draft.score) and 0.0 <= draft.score <= 1.0
    ):
        raise EnrichmentDraftError(f"score must be finite in [0, 1]: {draft.score}")
    if len(draft.score_components) > MAX_SCORE_COMPONENTS:
        raise EnrichmentDraftError(
            f"more than {MAX_SCORE_COMPONENTS} score components"
        )
    for key, component in draft.score_components.items():
        if not key.strip() or len(key) > MAX_SCORE_COMPONENT_KEY_CHARS:
            raise EnrichmentDraftError(f"bad score component key: {key!r:.100}")
        if not isinstance(component, (int, float)) or isinstance(component, bool):
            raise EnrichmentDraftError(
                f"score component {key!r} must be a number, got {type(component).__name__}"
            )
        if not math.isfinite(float(component)):
            raise EnrichmentDraftError(f"score component {key!r} must be finite")


def _validate_invocation(
    producer: str,
    producer_version: str,
    scope: EnrichmentScope,
    covered: tuple[ClaimField, ...],
    idempotency_key: str,
    started_at: datetime,
    completed_at: datetime,
    config: Mapping[str, Any] | None,
    config_schema_version: int | None,
) -> None:
    for label, value in (("producer", producer), ("producer_version", producer_version)):
        if not value.strip() or len(value) > MAX_PRODUCER_CHARS:
            raise EnrichmentDraftError(
                f"{label} empty or over {MAX_PRODUCER_CHARS} chars"
            )
    scope.validate()
    if not covered:
        raise EnrichmentDraftError("covered_fields must not be empty")
    if len(set(covered)) != len(covered):
        raise EnrichmentDraftError(f"duplicate covered_fields: {covered}")
    if not idempotency_key.strip():
        raise EnrichmentDraftError("idempotency_key must be non-empty")
    for label, stamp in (("started_at", started_at), ("completed_at", completed_at)):
        if stamp.tzinfo is None:
            raise EnrichmentDraftError(f"{label} must be timezone-aware")
    if completed_at < started_at:
        raise EnrichmentDraftError("completed_at precedes started_at")
    if (config is None) != (config_schema_version is None):
        raise EnrichmentDraftError(
            "config and config_schema_version must be set together"
        )
    if config is not None:
        if config_schema_version is not None and config_schema_version < 1:
            raise EnrichmentDraftError("config_schema_version must be >= 1")
        try:
            encoded = json.dumps(dict(config), sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise EnrichmentDraftError(f"config is not JSON-serializable: {exc}") from exc
        if len(encoded.encode()) > MAX_CONFIG_BYTES:
            raise EnrichmentDraftError(f"config over {MAX_CONFIG_BYTES} bytes")


def _scope_filter(scope: EnrichmentScope) -> tuple[Any, ...]:
    return (
        EnrichmentProducerRun.target_kind == scope.kind.value,
        EnrichmentProducerRun.speaker_id.is_not_distinct_from(scope.speaker_id),
        EnrichmentProducerRun.pipeline_run_id.is_not_distinct_from(
            scope.pipeline_run_id
        ),
        EnrichmentProducerRun.diarization_label.is_not_distinct_from(
            scope.diarization_label
        ),
    )


def _canonical_timestamp(stamp: datetime | None) -> str | None:
    return stamp.astimezone(UTC).isoformat() if stamp is not None else None


def _draft_evidence_key(item: Evidence) -> list[object]:
    if isinstance(item, MetadataEvidence):
        kind_cols: list[object] = [
            EvidenceKind.METADATA_FIELD.value,
            str(item.source_metadata_id),
            item.source_field,
            None,
            None,
        ]
    elif isinstance(item, TranscriptEvidence):
        kind_cols = [
            EvidenceKind.TRANSCRIPT_SEGMENT.value,
            None,
            None,
            str(item.transcript_segment_id),
            item.timestamp_seconds,
        ]
    else:
        kind_cols = [EvidenceKind.URL.value, None, None, None, None]
    url = item.url if isinstance(item, UrlEvidence) else None
    retrieved = item.retrieved_at if isinstance(item, UrlEvidence) else None
    return [
        *kind_cols,
        url,
        _canonical_timestamp(retrieved),
        item.snippet,
        dict(item.detail) if item.detail is not None else None,
        item.detail_schema_version,
    ]


def _draft_payload_fingerprint(drafts: Sequence[CandidateDraft]) -> str:
    entries = [
        [
            draft.target.kind.value,
            str(draft.target.speaker_id),
            str(draft.target.pipeline_run_id),
            draft.target.diarization_label,
            draft.field.value,
            draft.value,
            draft.score,
            sorted(
                (key, float(component))
                for key, component in draft.score_components.items()
            ),
            [_draft_evidence_key(item) for item in draft.evidence],
        ]
        for draft in drafts
    ]
    return json.dumps(sorted(entries, key=json.dumps), sort_keys=True)


def _row_evidence_key(row: EnrichmentCandidateEvidence) -> list[object]:
    return [
        row.kind,
        str(row.source_metadata_id) if row.source_metadata_id else None,
        row.source_field,
        str(row.transcript_segment_id) if row.transcript_segment_id else None,
        row.timestamp_seconds,
        row.url,
        _canonical_timestamp(row.retrieved_at),
        row.snippet,
        row.detail,
        row.detail_schema_version,
    ]


def _row_payload_fingerprint(row: EnrichmentProducerRun) -> str:
    entries = [
        [
            candidate.target_kind,
            str(candidate.speaker_id),
            str(candidate.pipeline_run_id),
            candidate.diarization_label,
            candidate.field,
            candidate.value,
            candidate.score,
            sorted(
                (key, float(component))
                for key, component in candidate.score_components.items()
            ),
            [
                _row_evidence_key(item)
                for item in sorted(candidate.evidence, key=lambda e: e.ordinal)
            ],
        ]
        for candidate in row.candidates
    ]
    return json.dumps(sorted(entries, key=json.dumps), sort_keys=True)


def _replay_matches(
    row: EnrichmentProducerRun,
    producer: str,
    producer_version: str,
    scope: EnrichmentScope,
    covered: tuple[ClaimField, ...],
    candidates: Sequence[CandidateDraft],
    started_at: datetime,
    completed_at: datetime,
    config: Mapping[str, Any] | None,
    config_schema_version: int | None,
) -> bool:
    """Full-payload replay equality — an identical replay adopts the stored
    row; ANY divergence (candidates, evidence, config, timestamps included)
    is a conflicting reuse of the key, never a silent first-write-wins."""
    return (
        row.producer == producer
        and row.producer_version == producer_version
        and row.target_kind == scope.kind.value
        and row.speaker_id == scope.speaker_id
        and row.pipeline_run_id == scope.pipeline_run_id
        and row.diarization_label == scope.diarization_label
        and tuple(row.covered_fields) == tuple(f.value for f in covered)
        and _canonical_timestamp(row.started_at) == _canonical_timestamp(started_at)
        and _canonical_timestamp(row.completed_at)
        == _canonical_timestamp(completed_at)
        and row.config == (dict(config) if config is not None else None)
        and row.config_schema_version == config_schema_version
        and _row_payload_fingerprint(row) == _draft_payload_fingerprint(candidates)
    )


def record_producer_run(
    session: Session,
    *,
    producer: str,
    producer_version: str,
    scope: EnrichmentScope,
    covered_fields: Sequence[ClaimField],
    candidates: Sequence[CandidateDraft],
    idempotency_key: str,
    started_at: datetime,
    completed_at: datetime,
    config: Mapping[str, Any] | None = None,
    config_schema_version: int | None = None,
) -> EnrichmentProducerRun:
    """Atomically persist a completed invocation, its claims, and supersession.

    Returns the invocation row (the existing one on an identical replay).
    Raises :class:`EnrichmentDraftError` for anything malformed and
    :class:`ConflictingReplayError` when ``idempotency_key`` was already used
    with a different payload.
    """
    covered = tuple(covered_fields)
    _validate_invocation(
        producer,
        producer_version,
        scope,
        covered,
        idempotency_key,
        started_at,
        completed_at,
        config,
        config_schema_version,
    )
    for draft in candidates:
        _validate_candidate(draft, scope, covered)

    def _existing() -> EnrichmentProducerRun | None:
        return session.execute(
            select(EnrichmentProducerRun)
            .where(EnrichmentProducerRun.idempotency_key == idempotency_key)
            .options(
                selectinload(EnrichmentProducerRun.candidates).selectinload(
                    EnrichmentCandidate.evidence
                )
            )
        ).scalar_one_or_none()

    def _adopt_or_conflict(row: EnrichmentProducerRun) -> EnrichmentProducerRun:
        if _replay_matches(
            row,
            producer,
            producer_version,
            scope,
            covered,
            candidates,
            started_at,
            completed_at,
            config,
            config_schema_version,
        ):
            return row
        raise ConflictingReplayError(idempotency_key)

    existing = _existing()
    if existing is not None:
        return _adopt_or_conflict(existing)

    # One finalization at a time per (producer, scope): generation allocation,
    # run+candidate insertion, and supersession must be atomic even when the
    # scope has no prior rows to lock. Transaction-scoped, so it releases on
    # commit/rollback and can never leak.
    session.execute(
        text("SELECT pg_advisory_xact_lock(hashtext(:producer), hashtext(:scope))"),
        {"producer": producer, "scope": scope.lock_key()},
    )

    def _persist() -> EnrichmentProducerRun:
        generation = (
            session.execute(
                select(func.coalesce(func.max(EnrichmentProducerRun.generation), 0)).where(
                    EnrichmentProducerRun.producer == producer, *_scope_filter(scope)
                )
            ).scalar_one()
            + 1
        )
        run = EnrichmentProducerRun(
            producer=producer,
            producer_version=producer_version,
            target_kind=scope.kind.value,
            speaker_id=scope.speaker_id,
            pipeline_run_id=scope.pipeline_run_id,
            diarization_label=scope.diarization_label,
            covered_fields=[f.value for f in covered],
            generation=generation,
            outcome=(
                EnrichmentOutcome.NONE.value
                if not candidates
                else EnrichmentOutcome.FOUND.value
            ),
            config_schema_version=config_schema_version,
            idempotency_key=idempotency_key,
            started_at=started_at,
            completed_at=completed_at,
        )
        if config is not None:
            run.config = dict(config)
        session.add(run)
        session.flush()
        _insert_candidates(session, run, candidates)
        _supersede_prior(session, run, scope, covered)
        return run

    # The helper's lookup() acts as the post-lock re-check.
    return savepoint_adopt_or_conflict(
        session,
        lookup=_existing,
        adopt_or_conflict=_adopt_or_conflict,
        persist=_persist,
    )


def _insert_candidates(
    session: Session, run: EnrichmentProducerRun, candidates: Sequence[CandidateDraft]
) -> None:
    for draft in candidates:
        candidate = EnrichmentCandidate(
            producer_run_id=run.id,
            target_kind=draft.target.kind.value,
            speaker_id=draft.target.speaker_id,
            pipeline_run_id=draft.target.pipeline_run_id,
            diarization_label=draft.target.diarization_label,
            field=draft.field.value,
            value=draft.value,
            score=draft.score,
            score_components={
                key: float(component)
                for key, component in draft.score_components.items()
            },
        )
        session.add(candidate)
        session.flush()
        for ordinal, item in enumerate(draft.evidence):
            row = EnrichmentCandidateEvidence(
                candidate_id=candidate.id,
                ordinal=ordinal,
                snippet=item.snippet,
                detail_schema_version=item.detail_schema_version,
            )
            # Assign only when present — an explicit None would serialize as
            # a JSON null, not SQL NULL (same trap as ``config`` above).
            if item.detail is not None:
                row.detail = dict(item.detail)
            if isinstance(item, MetadataEvidence):
                row.kind = EvidenceKind.METADATA_FIELD.value
                row.source_metadata_id = item.source_metadata_id
                row.source_field = item.source_field
            elif isinstance(item, TranscriptEvidence):
                row.kind = EvidenceKind.TRANSCRIPT_SEGMENT.value
                row.transcript_segment_id = item.transcript_segment_id
                row.timestamp_seconds = item.timestamp_seconds
            else:
                row.kind = EvidenceKind.URL.value
                row.url = item.url
                row.retrieved_at = item.retrieved_at
            session.add(row)
    session.flush()


def _supersede_prior(
    session: Session,
    run: EnrichmentProducerRun,
    scope: EnrichmentScope,
    covered: tuple[ClaimField, ...],
) -> None:
    """Stamp still-proposed claims of older generations of this producer+scope.

    Only fields the new run declared it covered; decided candidates (a human
    act stands) and already-superseded ones are never touched. An
    ``outcome='none'`` run supersedes too — "we looked again and found
    nothing" retires the earlier proposals.
    """
    prior_runs = select(EnrichmentProducerRun.id).where(
        EnrichmentProducerRun.producer == run.producer,
        *_scope_filter(scope),
        EnrichmentProducerRun.generation < run.generation,
    )
    eligibility = (
        EnrichmentCandidate.producer_run_id.in_(prior_runs),
        EnrichmentCandidate.field.in_([f.value for f in covered]),
        EnrichmentCandidate.superseded_by_producer_run_id.is_(None),
        ~exists(
            select(ProfileReviewDecision.id).where(
                ProfileReviewDecision.candidate_id == EnrichmentCandidate.id
            )
        ),
    )
    # Two statements on purpose (assumes READ COMMITTED, the session default).
    # A concurrent record_profile_decision holds FOR UPDATE on its candidate
    # while inserting the decision; a single bulk UPDATE that blocks on that
    # row lock would resume on its ORIGINAL statement snapshot (the row itself
    # is unchanged, so no re-check happens) and stamp a just-decided
    # candidate. Locking first, then updating, gives the UPDATE a fresh
    # snapshot in which the committed decision is visible and the row is
    # skipped.
    locked_ids = (
        session.execute(
            select(EnrichmentCandidate.id).where(*eligibility).with_for_update()
        )
        .scalars()
        .all()
    )
    if locked_ids:
        session.execute(
            update(EnrichmentCandidate)
            .where(EnrichmentCandidate.id.in_(locked_ids), *eligibility)
            .values(superseded_by_producer_run_id=run.id)
        )
