"""SQLAlchemy models — the v1 schema, owned by the alembic chain from revision 0001.

Design notes (see docs/architecture.md when it lands):

- Media identity (``media_items``) is split from execution state (``pipeline_runs`` +
  ``stage_runs``). Runs carry an explicit ``revision`` for compare-and-swap updates.
- Statuses are text + CHECK constraints in the DB, ``StrEnum`` in Python.
- ``transcript_segments`` preserves raw ASR text forever; enhancement writes
  ``enhanced_text`` beside it, never over it.
- ``speaker_assignments`` are machine proposals; ``adjudication_decisions`` is the
  immutable human ledger. The two are never merged.
- ``speaker_embeddings`` carry an ``embedding_space`` tag; vectors from different
  spaces must never be compared (enforced in ``speakers/matching.py`` queries).
- The ``enrichment_*`` tables (issue #37) hold machine-derived draft claims
  with evidence; ``profile_review_decisions`` is their separate append-only
  human trail. Drafts are suggestions *about* identity, never identity — they
  feed neither attribution nor the roster.
"""

import enum
import uuid
from datetime import date, datetime
from typing import Any, ClassVar

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from voxint.db import search

EMBEDDING_DIM = 192


class Base(DeclarativeBase):
    type_annotation_map: ClassVar = {dict[str, Any]: JSON().with_variant(JSONB(), "postgresql")}


class RunStatus(enum.StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    AWAITING_ADJUDICATION = "awaiting_adjudication"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Stage(enum.StrEnum):
    # ACQUIRE is the universal first stage: a no-op success for local/uploaded
    # media (source_url IS NULL), a yt-dlp download for URL runs. Enum order
    # mirrors STAGE_ORDER so ``_enum_values`` and ``list(Stage)`` agree with it.
    ACQUIRE = "acquire"
    PREPARE = "prepare"
    TRANSCRIBE = "transcribe"
    DIARIZE_EMBED = "diarize_embed"
    ENHANCE_MATCH = "enhance_match"
    FINALIZE = "finalize"


STAGE_ORDER: tuple[Stage, ...] = (
    Stage.ACQUIRE,
    Stage.PREPARE,
    Stage.TRANSCRIBE,
    Stage.DIARIZE_EMBED,
    Stage.ENHANCE_MATCH,
    Stage.FINALIZE,
)


class StageStatus(enum.StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ArtifactKind(enum.StrEnum):
    PREPROCESSED_AUDIO = "preprocessed_audio"
    CHUNK = "chunk"
    TRANSCRIPT_EXPORT = "transcript_export"


class SourceKind(enum.StrEnum):
    # 'rss' is reserved for future feed-item acquisition (issue #36 schema
    # accommodation); only 'ytdlp' rows are written today.
    YTDLP = "ytdlp"
    RSS = "rss"


class AssignmentMethod(enum.StrEnum):
    COSINE = "cosine"
    LLM_HINT = "llm_hint"


class Decision(enum.StrEnum):
    ASSIGN = "assign"
    EXCLUDE = "exclude"
    UNKNOWN = "unknown"


class EnrichmentTargetKind(enum.StrEnum):
    SPEAKER = "speaker"
    RUN = "run"
    RUN_LABEL = "run_label"


class EnrichmentOutcome(enum.StrEnum):
    FOUND = "found"
    NONE = "none"


class ClaimField(enum.StrEnum):
    NAME = "name"
    BIO = "bio"
    AFFILIATION = "affiliation"
    LINK = "link"


class EvidenceKind(enum.StrEnum):
    METADATA_FIELD = "metadata_field"
    TRANSCRIPT_SEGMENT = "transcript_segment"
    URL = "url"


class ProfileDecision(enum.StrEnum):
    ACCEPT = "accept"
    REJECT = "reject"


def _enum_values(e: type[enum.StrEnum]) -> str:
    return ", ".join(f"'{m.value}'" for m in e)


class MediaItem(Base):
    __tablename__ = "media_items"
    __table_args__ = (
        CheckConstraint(
            "duration_seconds IS NULL OR duration_seconds >= 0",
            name="media_items_duration_nonneg_check",
        ),
        CheckConstraint(
            "size_bytes IS NULL OR size_bytes >= 0", name="media_items_size_nonneg_check"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    source_path: Mapped[str] = mapped_column(Text, unique=True)
    # Provenance for URL-ingested media (nullable, non-unique): the origin URL a
    # run was fetched from. NULL means local/uploaded media that already sits at
    # source_path, so the ACQUIRE stage no-ops. Set only by the URL submission
    # service; source_path stays the unique file identity in every case.
    source_url: Mapped[str | None] = mapped_column(Text)
    media_type: Mapped[str | None] = mapped_column(Text)
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    sha256: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    runs: Mapped[list["PipelineRun"]] = relationship(back_populates="media_item")
    source_metadata: Mapped["MediaSourceMetadata | None"] = relationship(
        back_populates="media_item"
    )


class MediaSourceMetadata(Base):
    """Write-once acquisition context for a MediaItem (issue #36).

    Scraped source metadata — what the extractor knew about the recording at
    acquisition time. It is **context, not identity**: nothing here feeds
    speaker attribution. One row per MediaItem, inserted by the ACQUIRE stage
    and never updated: a MediaItem is already per-acquisition (URL submission
    mints a fresh uuid ``source_path``; the published bytes are immutable), so
    re-acquiring a URL creates a new MediaItem + snapshot and can never rewrite
    the context a past adjudication was made against. Human input (operator
    notes) lives on ``pipeline_runs`` — the two are never conflated.

    Normalized columns are the stable query/display surface;``raw`` holds the
    bounded, allowlisted, ``raw_schema_version``-stamped subset of the
    extractor's info-JSON built by ``media/source_metadata.py`` — never the
    full document (its ``formats``/``http_headers`` carry signed URLs and
    cookies, which must not persist at rest). ``duration_seconds`` is the
    *source-claimed* duration (context), distinct from the probed
    ``media_items.duration_seconds`` (measurement); it feeds nothing downstream.
    """

    __tablename__ = "media_source_metadata"
    __table_args__ = (
        CheckConstraint(
            f"source_kind IN ({_enum_values(SourceKind)})",
            name="media_source_metadata_kind_check",
        ),
        CheckConstraint(
            "duration_seconds IS NULL OR duration_seconds >= 0",
            name="media_source_metadata_duration_nonneg_check",
        ),
        CheckConstraint(
            "raw_schema_version >= 1",
            name="media_source_metadata_raw_schema_version_check",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    media_item_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("media_items.id"), unique=True)
    source_kind: Mapped[str] = mapped_column(Text)
    title: Mapped[str | None] = mapped_column(Text)
    uploader: Mapped[str | None] = mapped_column(Text)
    uploader_url: Mapped[str | None] = mapped_column(Text)
    channel: Mapped[str | None] = mapped_column(Text)
    channel_url: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    # Parsed from yt-dlp's YYYYMMDD string; NULL when absent or unparseable.
    upload_date: Mapped[date | None] = mapped_column(Date)
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    tags: Mapped[list[str]] = mapped_column(
        ARRAY(Text), default=list, server_default=text("'{}'::text[]")
    )
    # The extractor's canonical webpage URL (post-redirect) — sanitized by the
    # extraction allowlist, never a signed transport URL.
    canonical_url: Mapped[str | None] = mapped_column(Text)
    extractor: Mapped[str | None] = mapped_column(Text)
    extractor_version: Mapped[str | None] = mapped_column(Text)
    raw: Mapped[dict[str, Any] | None] = mapped_column()
    raw_schema_version: Mapped[int] = mapped_column(Integer)
    # When the extractor observed the source (carried in the sidecar so a
    # crash-replay repair reuses the original capture time deterministically).
    acquired_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    media_item: Mapped[MediaItem] = relationship(back_populates="source_metadata")


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"
    __table_args__ = (
        CheckConstraint(
            f"status IN ({_enum_values(RunStatus)})", name="pipeline_runs_status_check"
        ),
        CheckConstraint(
            f"current_stage IS NULL OR current_stage IN ({_enum_values(Stage)})",
            name="pipeline_runs_current_stage_check",
        ),
        CheckConstraint("revision >= 0", name="pipeline_runs_revision_nonneg_check"),
        # The reviewer claim is one unit: either a run is unclaimed (all NULL)
        # or fully claimed (all set) — no half-claimed states to reason about.
        CheckConstraint(
            "(review_claim_token IS NULL) = (review_claimed_by IS NULL)"
            " AND (review_claim_token IS NULL) = (review_claimed_at IS NULL)"
            " AND (review_claim_token IS NULL) = (review_claim_expires_at IS NULL)",
            name="pipeline_runs_review_claim_shape_check",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    media_item_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("media_items.id"), index=True)
    status: Mapped[str] = mapped_column(Text, default=RunStatus.QUEUED.value, index=True)
    current_stage: Mapped[str | None] = mapped_column(Text)
    # Optimistic-concurrency token: every state change goes through CAS on this column.
    revision: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    # Reviewer slot (P5): one adjudication claim per run, guarded by the same
    # CAS revision as pipeline transitions. The token is an opaque per-claim
    # secret so a stale browser tab can never act on a newer claim.
    review_claim_token: Mapped[uuid.UUID | None] = mapped_column()
    review_claimed_by: Mapped[str | None] = mapped_column(Text)
    review_claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    review_claim_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Operator's free-text notes for this run (issue #36) — human input, kept
    # structurally apart from scraped source metadata (media_source_metadata).
    # Editable, last-write-wins, deliberately outside the CAS revision: notes
    # are operator prose, not pipeline state.
    operator_notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    media_item: Mapped[MediaItem] = relationship(back_populates="runs")
    stage_runs: Mapped[list["StageRun"]] = relationship(back_populates="pipeline_run")


class StageRun(Base):
    """One attempt at one stage — doubles as the stage execution claim.

    A worker inserts a RUNNING row (with its ``worker_id`` and a lease) *before*
    executing the stage body; the ``(pipeline_run_id, stage, attempt)`` unique
    constraint arbitrates concurrent claims. Recovery may only touch claims whose
    lease has expired.
    """

    __tablename__ = "stage_runs"
    __table_args__ = (
        UniqueConstraint("pipeline_run_id", "stage", "attempt", name="stage_runs_attempt_key"),
        CheckConstraint(f"stage IN ({_enum_values(Stage)})", name="stage_runs_stage_check"),
        CheckConstraint(f"status IN ({_enum_values(StageStatus)})", name="stage_runs_status_check"),
        CheckConstraint("attempt >= 1", name="stage_runs_attempt_positive_check"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    pipeline_run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("pipeline_runs.id"), index=True)
    stage: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, default=StageStatus.RUNNING.value)
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    worker_id: Mapped[str | None] = mapped_column(Text)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(Text)
    metrics: Mapped[dict[str, Any] | None] = mapped_column()

    pipeline_run: Mapped[PipelineRun] = relationship(back_populates="stage_runs")


class AudioArtifact(Base):
    __tablename__ = "audio_artifacts"
    __table_args__ = (
        CheckConstraint(
            f"kind IN ({_enum_values(ArtifactKind)})", name="audio_artifacts_kind_check"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    pipeline_run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("pipeline_runs.id"), index=True)
    kind: Mapped[str] = mapped_column(Text)
    path: Mapped[str] = mapped_column(Text)
    meta: Mapped[dict[str, Any] | None] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AudioChunk(Base):
    __tablename__ = "audio_chunks"
    __table_args__ = (
        UniqueConstraint("pipeline_run_id", "chunk_index", name="audio_chunks_index_key"),
        CheckConstraint("chunk_index >= 0", name="audio_chunks_index_nonneg_check"),
        CheckConstraint(
            "start_seconds >= 0 AND end_seconds > start_seconds",
            name="audio_chunks_interval_check",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    pipeline_run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("pipeline_runs.id"), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    start_seconds: Mapped[float] = mapped_column(Float)
    end_seconds: Mapped[float] = mapped_column(Float)
    path: Mapped[str] = mapped_column(Text)


class TranscriptSegment(Base):
    __tablename__ = "transcript_segments"
    __table_args__ = (
        UniqueConstraint("pipeline_run_id", "segment_index", name="transcript_segments_index_key"),
        CheckConstraint("segment_index >= 0", name="transcript_segments_index_nonneg_check"),
        CheckConstraint(
            "start_seconds >= 0 AND end_seconds >= start_seconds",
            name="transcript_segments_interval_check",
        ),
        # FTS expression indexes (migration 0008). Declared here for the
        # model↔migration parity tests; the DDL literals must match
        # ``voxint.db.search`` (contract-tested). Safe to declare as ORM
        # metadata: nothing calls ``create_all`` — schema comes from alembic.
        Index(
            search.RAW_FTS_INDEX_NAME,
            text(f"to_tsvector('{search.TS_CONFIG}', raw_text)"),
            postgresql_using="gin",
        ),
        Index(
            search.ENHANCED_FTS_INDEX_NAME,
            text(f"to_tsvector('{search.TS_CONFIG}', enhanced_text)"),
            postgresql_using="gin",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    pipeline_run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("pipeline_runs.id"), index=True)
    segment_index: Mapped[int] = mapped_column(Integer)
    start_seconds: Mapped[float] = mapped_column(Float)
    end_seconds: Mapped[float] = mapped_column(Float)
    # Raw ASR output is immutable; enhancement writes enhanced_text, never over raw_text.
    raw_text: Mapped[str] = mapped_column(Text)
    enhanced_text: Mapped[str | None] = mapped_column(Text)
    # Local diarization label within this run (e.g. "SPEAKER_00"), not a speaker identity.
    diarization_label: Mapped[str | None] = mapped_column(Text)
    # ASR hallucination soft-tag, preserved verbatim from the service; gates weight it.
    suspect: Mapped[bool] = mapped_column(Boolean, default=False)


class DiarizationTurn(Base):
    """Run-scoped observation ledger: one row per diarization turn.

    Carries the turn interval, local label, overlap info, and that window's
    embedding outcome — either a vector (with its ``embedding_space``) or a
    ``skip_reason`` (``too_short`` / ``low_snr``), never both. Skips stay
    auditable instead of vanishing; P4 speaker matching consumes these rows.
    Labels repeat across turns — identity lives in ``speakers``, not here.
    """

    __tablename__ = "diarization_turns"
    __table_args__ = (
        UniqueConstraint("pipeline_run_id", "turn_index", name="diarization_turns_index_key"),
        CheckConstraint("turn_index >= 0", name="diarization_turns_index_nonneg_check"),
        CheckConstraint(
            "start_seconds >= 0 AND end_seconds > start_seconds",
            name="diarization_turns_interval_check",
        ),
        CheckConstraint("overlap_seconds >= 0", name="diarization_turns_overlap_nonneg_check"),
        # Exactly one of embedding / skip_reason, mirroring the titanet contract.
        CheckConstraint(
            "(embedding IS NULL) != (skip_reason IS NULL)",
            name="diarization_turns_embedding_xor_skip_check",
        ),
        CheckConstraint(
            "embedding IS NULL OR embedding_space IS NOT NULL",
            name="diarization_turns_embedding_space_check",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    pipeline_run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("pipeline_runs.id"), index=True)
    turn_index: Mapped[int] = mapped_column(Integer)
    start_seconds: Mapped[float] = mapped_column(Float)
    end_seconds: Mapped[float] = mapped_column(Float)
    label: Mapped[str] = mapped_column(Text)
    overlap: Mapped[bool] = mapped_column(Boolean, default=False)
    overlap_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    snr_db: Mapped[float | None] = mapped_column(Float)
    skip_reason: Mapped[str | None] = mapped_column(Text)
    embedding: Mapped[Any] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    embedding_space: Mapped[str | None] = mapped_column(Text)


class Speaker(Base):
    """Roster identity with a curation lifecycle (issue #7).

    A speaker is *active* while ``merged_into_id`` and ``deleted_at`` are both
    NULL — only active speakers participate in matching, dropdowns, and new
    decisions. Merging retains the source row as a tombstone (``merged_into_id``
    self-FK) so ledger FKs stay valid; readers canonicalize through it. Archive
    is reversible (``deleted_at``). ``display_name`` stays globally unique
    across every lifecycle state — restore or merge, never re-create a name.
    """

    __tablename__ = "speakers"
    __table_args__ = (
        CheckConstraint(
            "merged_into_id IS NULL OR merged_into_id != id",
            name="speakers_no_self_merge_check",
        ),
        CheckConstraint(
            "(merged_into_id IS NULL) = (merged_at IS NULL)",
            name="speakers_merge_fields_together_check",
        ),
        CheckConstraint(
            "merged_into_id IS NULL OR deleted_at IS NULL",
            name="speakers_not_merged_and_deleted_check",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    display_name: Mapped[str] = mapped_column(Text, unique=True)
    notes: Mapped[str | None] = mapped_column(Text)
    merged_into_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("speakers.id"), index=True)
    merged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class SpeakerEmbedding(Base):
    __tablename__ = "speaker_embeddings"
    __table_args__ = (
        CheckConstraint(
            "length(embedding_space) > 0", name="speaker_embeddings_space_nonempty_check"
        ),
        # One enrollment centroid per human decision — a replayed enrollment
        # POST can never mint a second embedding row.
        UniqueConstraint(
            "source_adjudication_decision_id",
            name="speaker_embeddings_source_decision_key",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    speaker_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("speakers.id"), index=True)
    # Cosine comparisons are only valid within one embedding_space (e.g. "titanet-large-v1").
    embedding_space: Mapped[str] = mapped_column(Text, index=True)
    embedding: Mapped[Any] = mapped_column(Vector(EMBEDDING_DIM))
    source_pipeline_run_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("pipeline_runs.id"))
    # Enrollment provenance (P5): which local label and which human ruling
    # produced this centroid. Raw per-turn vectors stay in diarization_turns.
    source_diarization_label: Mapped[str | None] = mapped_column(Text)
    source_adjudication_decision_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("adjudication_decisions.id")
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class SpeakerAssignment(Base):
    """Machine proposals only — human rulings live in ``adjudication_decisions``."""

    __tablename__ = "speaker_assignments"
    __table_args__ = (
        UniqueConstraint(
            "pipeline_run_id",
            "diarization_label",
            "method",
            name="speaker_assignments_proposal_key",
        ),
        CheckConstraint(
            f"method IN ({_enum_values(AssignmentMethod)})",
            name="speaker_assignments_method_check",
        ),
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="speaker_assignments_confidence_check",
        ),
        # named != grounded, enforced: only a cosine proposal with a concrete
        # speaker can claim grounding; an LLM name hint never can.
        CheckConstraint(
            "NOT grounded OR (speaker_id IS NOT NULL AND method = 'cosine')",
            name="speaker_assignments_grounded_check",
        ),
        # Method shapes: a cosine proposal names a roster speaker and carries no
        # free-text name; an llm_hint carries only a name — never a speaker_id,
        # never grounded.
        CheckConstraint(
            "method != 'cosine' OR (speaker_id IS NOT NULL AND proposed_name IS NULL)",
            name="speaker_assignments_cosine_shape_check",
        ),
        CheckConstraint(
            "method != 'llm_hint' OR (speaker_id IS NULL AND NOT grounded"
            " AND confidence IS NULL"  # model-reported confidence is not calibrated
            " AND proposed_name IS NOT NULL AND length(trim(proposed_name)) > 0)",
            name="speaker_assignments_llm_hint_shape_check",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    pipeline_run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("pipeline_runs.id"), index=True)
    diarization_label: Mapped[str] = mapped_column(Text)
    speaker_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("speakers.id"))
    method: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float | None] = mapped_column(Float)
    # llm_hint only: the explicitly spoken name the LLM heard for this label.
    proposed_name: Mapped[str | None] = mapped_column(Text)
    # named != grounded: a name proposed by an LLM is NOT grounded until it has
    # embedding-level evidence or a human ruling.
    grounded: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AdjudicationDecision(Base):
    """Immutable human ledger — rows are only ever inserted, never updated or deleted."""

    __tablename__ = "adjudication_decisions"
    __table_args__ = (
        UniqueConstraint("idempotency_key", name="adjudication_decisions_idempotency_key"),
        CheckConstraint(
            f"decision IN ({_enum_values(Decision)})", name="adjudication_decisions_check"
        ),
        CheckConstraint(
            "(decision = 'assign') = (speaker_id IS NOT NULL)",
            name="adjudication_decisions_assign_speaker_check",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    pipeline_run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("pipeline_runs.id"), index=True)
    diarization_label: Mapped[str] = mapped_column(Text)
    decision: Mapped[str] = mapped_column(Text)
    speaker_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("speakers.id"))
    operator: Mapped[str] = mapped_column(Text)
    idempotency_key: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EnrichmentProducerRun(Base):
    """One **completed** enrichment-producer invocation (issue #37).

    Enrichment producers (offline name mining, web research, …) generate
    machine-derived claims about speakers and runs. This table records each
    finished invocation: which producer (stable logical key + version), what
    scope it examined, which claim fields it covered, and whether it found
    anything. Rows are inserted only at successful completion — atomically with
    their candidate rows — by the single sanctioned writer
    (``enrichment/drafts.py``); failed or partial attempts are not audited at
    this layer. A recorded invocation is history: a trigger rejects
    UPDATE/DELETE, because scope, coverage, generation, and outcome anchor
    candidate lineage and supersession.

    - ``outcome`` is derived by the writer: ``'none'`` iff the invocation
      produced zero candidates. "We looked and found nothing" is reviewable
      information, deliberately unlike ``speaker_assignments`` where absence
      is modeled as no row (docs/quality-gates.md).
    - ``generation`` is monotonic per (producer, scope), allocated by the
      writer under a per-scope advisory lock; supersession compares
      generations, never wall-clock, so out-of-order completion cannot make an
      older run supersede a newer one. Uniqueness of (producer, scope,
      generation) cannot be a UNIQUE constraint (the scope trio contains
      NULLs) — it is writer-enforced under the lock.
    - ``covered_fields`` declares which claim fields the producer actually
      examined; a rerun supersedes only candidates in its covered fields.
    - ``config`` is a bounded, schema-versioned snapshot of the settings that
      shaped the invocation (budgets, model, thresholds) for reproducibility.
    """

    __tablename__ = "enrichment_producer_runs"
    __table_args__ = (
        UniqueConstraint("idempotency_key", name="enrichment_producer_runs_idempotency_key"),
        CheckConstraint(
            "length(trim(producer)) > 0",
            name="enrichment_producer_runs_producer_nonempty_check",
        ),
        CheckConstraint(
            "length(trim(producer_version)) > 0",
            name="enrichment_producer_runs_producer_version_nonempty_check",
        ),
        CheckConstraint(
            f"target_kind IN ({_enum_values(EnrichmentTargetKind)})",
            name="enrichment_producer_runs_target_kind_check",
        ),
        # Target shapes: exactly the columns of the declared kind, no strays.
        CheckConstraint(
            "target_kind != 'speaker' OR (speaker_id IS NOT NULL"
            " AND pipeline_run_id IS NULL AND diarization_label IS NULL)",
            name="enrichment_producer_runs_speaker_shape_check",
        ),
        CheckConstraint(
            "target_kind != 'run' OR (pipeline_run_id IS NOT NULL"
            " AND speaker_id IS NULL AND diarization_label IS NULL)",
            name="enrichment_producer_runs_run_shape_check",
        ),
        CheckConstraint(
            "target_kind != 'run_label' OR (pipeline_run_id IS NOT NULL"
            " AND diarization_label IS NOT NULL AND speaker_id IS NULL)",
            name="enrichment_producer_runs_run_label_shape_check",
        ),
        CheckConstraint(
            "diarization_label IS NULL OR length(trim(diarization_label)) > 0",
            name="enrichment_producer_runs_label_nonempty_check",
        ),
        CheckConstraint(
            "cardinality(covered_fields) >= 1 AND covered_fields <@ "
            f"ARRAY[{_enum_values(ClaimField)}]::text[]",
            name="enrichment_producer_runs_covered_fields_check",
        ),
        CheckConstraint("generation >= 1", name="enrichment_producer_runs_generation_check"),
        CheckConstraint(
            f"outcome IN ({_enum_values(EnrichmentOutcome)})",
            name="enrichment_producer_runs_outcome_check",
        ),
        CheckConstraint(
            "(config IS NULL) = (config_schema_version IS NULL)",
            name="enrichment_producer_runs_config_pair_check",
        ),
        CheckConstraint(
            "config_schema_version IS NULL OR config_schema_version >= 1",
            name="enrichment_producer_runs_config_schema_version_check",
        ),
        CheckConstraint(
            "config IS NULL OR jsonb_typeof(config) = 'object'",
            name="enrichment_producer_runs_config_object_check",
        ),
        CheckConstraint(
            "completed_at >= started_at",
            name="enrichment_producer_runs_completed_after_started_check",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    producer: Mapped[str] = mapped_column(Text)
    producer_version: Mapped[str] = mapped_column(Text)
    target_kind: Mapped[str] = mapped_column(Text)
    speaker_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("speakers.id"), index=True)
    pipeline_run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("pipeline_runs.id"), index=True
    )
    diarization_label: Mapped[str | None] = mapped_column(Text)
    covered_fields: Mapped[list[str]] = mapped_column(ARRAY(Text))
    generation: Mapped[int] = mapped_column(Integer)
    outcome: Mapped[str] = mapped_column(Text)
    config: Mapped[dict[str, Any] | None] = mapped_column()
    config_schema_version: Mapped[int | None] = mapped_column(Integer)
    idempotency_key: Mapped[str] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    candidates: Mapped[list["EnrichmentCandidate"]] = relationship(
        back_populates="producer_run",
        foreign_keys="EnrichmentCandidate.producer_run_id",
    )


class EnrichmentCandidate(Base):
    """A reviewable machine-derived claim — a *suggestion about* identity, never
    identity (issue #37).

    Producers write claims here as evidence-backed drafts; nothing in this
    table (accepted or not) ever feeds attribution, mutates
    ``speakers.display_name``/``notes``, or touches the adjudication ledger.
    A heard or read name is never grounded identity (docs/quality-gates.md) —
    only acoustic evidence or a human ruling grounds attribution.

    Claim content is immutable (a DB trigger rejects DELETE and any UPDATE
    except stamping ``superseded_by_producer_run_id``, write-once; rows are
    born unsuperseded — an initial non-NULL stamp is rejected at INSERT). There is no
    stored review state: the effective state is derived at read time
    (``enrichment/queries.py``) — a ``profile_review_decisions`` row wins
    (accepted/rejected, terminal), else a set ``superseded_by_producer_run_id``
    means superseded, else the claim is proposed. ``score`` /
    ``score_components`` are producer-local signals, never comparable across
    producers. At least one evidence row per candidate is writer-enforced.
    """

    __tablename__ = "enrichment_candidates"
    __table_args__ = (
        CheckConstraint(
            f"target_kind IN ({_enum_values(EnrichmentTargetKind)})",
            name="enrichment_candidates_target_kind_check",
        ),
        CheckConstraint(
            "target_kind != 'speaker' OR (speaker_id IS NOT NULL"
            " AND pipeline_run_id IS NULL AND diarization_label IS NULL)",
            name="enrichment_candidates_speaker_shape_check",
        ),
        CheckConstraint(
            "target_kind != 'run' OR (pipeline_run_id IS NOT NULL"
            " AND speaker_id IS NULL AND diarization_label IS NULL)",
            name="enrichment_candidates_run_shape_check",
        ),
        CheckConstraint(
            "target_kind != 'run_label' OR (pipeline_run_id IS NOT NULL"
            " AND diarization_label IS NOT NULL AND speaker_id IS NULL)",
            name="enrichment_candidates_run_label_shape_check",
        ),
        CheckConstraint(
            "diarization_label IS NULL OR length(trim(diarization_label)) > 0",
            name="enrichment_candidates_label_nonempty_check",
        ),
        CheckConstraint(
            f"field IN ({_enum_values(ClaimField)})",
            name="enrichment_candidates_field_check",
        ),
        CheckConstraint(
            "length(trim(value)) > 0 AND char_length(value) <= 4000",
            name="enrichment_candidates_value_check",
        ),
        CheckConstraint(
            "score IS NULL OR (score >= 0 AND score <= 1)",
            name="enrichment_candidates_score_check",
        ),
        CheckConstraint(
            "jsonb_typeof(score_components) = 'object'",
            name="enrichment_candidates_score_components_object_check",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    producer_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("enrichment_producer_runs.id"), index=True
    )
    target_kind: Mapped[str] = mapped_column(Text)
    speaker_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("speakers.id"), index=True)
    pipeline_run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("pipeline_runs.id"), index=True
    )
    diarization_label: Mapped[str | None] = mapped_column(Text)
    field: Mapped[str] = mapped_column(Text)
    value: Mapped[str] = mapped_column(Text)
    score: Mapped[float | None] = mapped_column(Float)
    score_components: Mapped[dict[str, Any]] = mapped_column(
        default=dict, server_default=text("'{}'::jsonb")
    )
    # Machine lifecycle fact — the only mutable column (write-once, trigger
    # enforced): set when a newer generation of the same producer + scope
    # covering this claim's field replaces it.
    superseded_by_producer_run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("enrichment_producer_runs.id")
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    producer_run: Mapped[EnrichmentProducerRun] = relationship(
        back_populates="candidates", foreign_keys=[producer_run_id]
    )
    evidence: Mapped[list["EnrichmentCandidateEvidence"]] = relationship(
        back_populates="candidate",
        order_by="EnrichmentCandidateEvidence.ordinal",
    )


class EnrichmentCandidateEvidence(Base):
    """Field-level provenance for one claim — 1:many so a single claim can cite
    a metadata field, transcript segments, and several URLs together (#37).

    Exactly one evidence shape per row (kind-shape CHECKs): a
    ``media_source_metadata`` column/``raw.``-path, a transcript segment
    (+ optional timestamp), or a fetched URL (+ optional retrieval time).
    Rows are append-only (trigger): evidence never changes after the claim is
    recorded. ``snippet`` is the bounded human-readable excerpt; ``detail`` is
    a schema-versioned seam for kind-specific extension.
    """

    __tablename__ = "enrichment_candidate_evidence"
    __table_args__ = (
        UniqueConstraint(
            "candidate_id", "ordinal", name="enrichment_candidate_evidence_ordinal_key"
        ),
        CheckConstraint("ordinal >= 0", name="enrichment_candidate_evidence_ordinal_nonneg_check"),
        CheckConstraint(
            f"kind IN ({_enum_values(EvidenceKind)})",
            name="enrichment_candidate_evidence_kind_check",
        ),
        # Evidence shapes: exactly the columns of the declared kind.
        CheckConstraint(
            "kind != 'metadata_field' OR (source_metadata_id IS NOT NULL"
            " AND source_field IS NOT NULL AND transcript_segment_id IS NULL"
            " AND timestamp_seconds IS NULL AND url IS NULL AND retrieved_at IS NULL)",
            name="enrichment_candidate_evidence_metadata_shape_check",
        ),
        CheckConstraint(
            "kind != 'transcript_segment' OR (transcript_segment_id IS NOT NULL"
            " AND source_metadata_id IS NULL AND source_field IS NULL"
            " AND url IS NULL AND retrieved_at IS NULL)",
            name="enrichment_candidate_evidence_transcript_shape_check",
        ),
        CheckConstraint(
            "kind != 'url' OR (url IS NOT NULL"
            " AND source_metadata_id IS NULL AND source_field IS NULL"
            " AND transcript_segment_id IS NULL AND timestamp_seconds IS NULL)",
            name="enrichment_candidate_evidence_url_shape_check",
        ),
        CheckConstraint(
            "source_field IS NULL OR length(trim(source_field)) > 0",
            name="enrichment_candidate_evidence_source_field_nonempty_check",
        ),
        CheckConstraint(
            "timestamp_seconds IS NULL OR timestamp_seconds >= 0",
            name="enrichment_candidate_evidence_timestamp_nonneg_check",
        ),
        CheckConstraint(
            "url IS NULL OR char_length(url) <= 2048",
            name="enrichment_candidate_evidence_url_length_check",
        ),
        CheckConstraint(
            "snippet IS NULL OR char_length(snippet) <= 1000",
            name="enrichment_candidate_evidence_snippet_length_check",
        ),
        CheckConstraint(
            "(detail IS NULL) = (detail_schema_version IS NULL)",
            name="enrichment_candidate_evidence_detail_pair_check",
        ),
        CheckConstraint(
            "detail_schema_version IS NULL OR detail_schema_version >= 1",
            name="enrichment_candidate_evidence_detail_schema_version_check",
        ),
        CheckConstraint(
            "detail IS NULL OR jsonb_typeof(detail) = 'object'",
            name="enrichment_candidate_evidence_detail_object_check",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("enrichment_candidates.id"), index=True
    )
    ordinal: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(Text)
    source_metadata_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("media_source_metadata.id")
    )
    # Normalized column name or "raw.<key>" path within the metadata snapshot.
    source_field: Mapped[str | None] = mapped_column(Text)
    transcript_segment_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("transcript_segments.id")
    )
    timestamp_seconds: Mapped[float | None] = mapped_column(Float)
    url: Mapped[str | None] = mapped_column(Text)
    retrieved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    snippet: Mapped[str | None] = mapped_column(Text)
    detail: Mapped[dict[str, Any] | None] = mapped_column()
    detail_schema_version: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    candidate: Mapped[EnrichmentCandidate] = relationship(back_populates="evidence")


class ProfileReviewDecision(Base):
    """Append-only human trail for enrichment claims (issue #37) — deliberately
    SEPARATE from ``adjudication_decisions``.

    Accepting a bio is a different act from ruling on who spoke: this trail
    records profile-review verdicts and never touches attribution. One decision
    per candidate (UNIQUE) — accept/reject is terminal; a re-proposed claim
    arrives as a fresh candidate row from a newer producer run, so corrections
    happen by re-running the producer, not by editing history. Rows reject
    UPDATE/DELETE via a trigger; writes go through the single idempotent
    writer (``enrichment/review.py``). Accepting a ``name`` claim records the
    act only — ``speakers.display_name`` and the attribution ledger are never
    written from here.
    """

    __tablename__ = "profile_review_decisions"
    __table_args__ = (
        UniqueConstraint("candidate_id", name="profile_review_decisions_candidate_key"),
        UniqueConstraint("idempotency_key", name="profile_review_decisions_idempotency_key"),
        CheckConstraint(
            f"decision IN ({_enum_values(ProfileDecision)})",
            name="profile_review_decisions_decision_check",
        ),
        CheckConstraint(
            "length(trim(operator)) > 0",
            name="profile_review_decisions_operator_nonempty_check",
        ),
        CheckConstraint(
            "note IS NULL OR char_length(note) <= 2000",
            name="profile_review_decisions_note_length_check",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    candidate_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("enrichment_candidates.id"))
    decision: Mapped[str] = mapped_column(Text)
    operator: Mapped[str] = mapped_column(Text)
    note: Mapped[str | None] = mapped_column(Text)
    idempotency_key: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AppSettings(Base):
    """Singleton (``id = 1``) store for the preferences the first-run wizard writes.

    Deliberately split from ``config.Settings`` (env-only, frozen on ``app.state``
    at process start): infra config and secrets — ``DATABASE_URL``, ``REDIS_URL``,
    ports, ``LLM_API_KEY`` — stay in the environment and the wizard never rewrites
    ``.env``. Only non-secret, user-facing preferences live here. Exactly one row
    exists, pinned by the ``id = 1`` CHECK; the API reads it per request and the
    worker snapshots it per run (see ``pipeline.stages.context``). ``llm_base_url``
    / ``llm_model`` are nullable — NULL means "fall back to the env default"; the
    LLM API key is never stored here.
    """

    __tablename__ = "app_settings"
    __table_args__ = (CheckConstraint("id = 1", name="app_settings_single_row_check"),)

    id: Mapped[int] = mapped_column(SmallInteger, primary_key=True, default=1)
    onboarding_complete: Mapped[bool] = mapped_column(Boolean, default=False)
    # Folders the wizard registered under MEDIA_ROOT (paths relative to it).
    media_folders: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    # User vocabulary (names/jargon/acronyms): augments the selected domain pack,
    # surfaced to the LLM enhancement context and the bounded whisper initial_prompt.
    vocabulary: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    llm_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    # NULL -> use the env default (config.Settings); the API key stays env-only.
    llm_base_url: Mapped[str | None] = mapped_column(Text)
    llm_model: Mapped[str | None] = mapped_column(Text)
    # The bundled guided-tutorial run, seeded idempotently by `voxint tutorial seed`.
    tutorial_run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("pipeline_runs.id", ondelete="SET NULL")
    )
    tutorial_completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class ResearchJobStatus(enum.StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResearchJob(Base):
    """One operator-initiated web-research job for a speaker (issue #40).

    Mutable orchestration state only — progress counters, cooperative-cancel
    flag, terminal status. The research *results* never live here: surviving
    claims land as immutable #37 drafts via the single sanctioned writer, and
    ``producer_run_id`` links the job to that record when one was written.
    The job id doubles as the producer-run idempotency identity
    (``web_researcher:speaker:{speaker_id}:{job_id}``): one job is one durable
    execution — an intentional rerun is a NEW job, because web research is
    non-deterministic and an input-derived key would wrongly suppress it.

    ``status`` moves queued → running (guarded claim UPDATE, so a duplicate
    Celery delivery no-ops) → succeeded | failed | cancelled. ``cancel_requested``
    is the operator's cooperative signal; the loop re-reads it between rounds.
    A ``failed``/``cancelled`` job records NO producer run — never an
    authoritative 'none' that would retire prior drafts.
    """

    __tablename__ = "research_jobs"
    __table_args__ = (
        # DB-enforced "one active job per speaker": the console's friendly
        # pre-check is check-then-insert and can race; this index cannot.
        Index(
            "research_jobs_one_active_per_speaker",
            "speaker_id",
            unique=True,
            postgresql_where=text("status IN ('queued', 'running')"),
        ),
        CheckConstraint(
            f"status IN ({_enum_values(ResearchJobStatus)})",
            name="research_jobs_status_check",
        ),
        CheckConstraint(
            "searches_used >= 0 AND reads_used >= 0 AND rounds_used >= 0",
            name="research_jobs_counters_check",
        ),
        CheckConstraint(
            "jsonb_typeof(budget) = 'object'",
            name="research_jobs_budget_object_check",
        ),
        CheckConstraint(
            "started_at IS NULL OR started_at >= created_at",
            name="research_jobs_started_after_created_check",
        ),
        CheckConstraint(
            "finished_at IS NULL OR started_at IS NOT NULL",
            name="research_jobs_finished_requires_started_check",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    speaker_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("speakers.id"), index=True)
    # Set when the job came from a run page's "research unresolved speakers"
    # fan-out; purely provenance, never a scope.
    pipeline_run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("pipeline_runs.id"), index=True
    )
    status: Mapped[str] = mapped_column(Text, default=ResearchJobStatus.QUEUED.value)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    # Snapshot of the budgets this job was started under (settings can change
    # between enqueue and execution; the preview the operator approved wins).
    budget: Mapped[dict[str, Any]] = mapped_column()
    # Bounded operator-supplied note handed to the model as seed context.
    operator_note: Mapped[str | None] = mapped_column(Text)
    searches_used: Mapped[int] = mapped_column(Integer, default=0)
    reads_used: Mapped[int] = mapped_column(Integer, default=0)
    rounds_used: Mapped[int] = mapped_column(Integer, default=0)
    # Bounded, redacted failure summary (closed vocabulary + safe detail only).
    error: Mapped[str | None] = mapped_column(Text)
    producer_run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("enrichment_producer_runs.id")
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
