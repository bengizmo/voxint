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

# Text-embedding dimension for the transcript semantic-search spine (issue
# #121). Deliberately separate from ``EMBEDDING_DIM`` (192, the TitaNet
# speaker space): the MiniLM text space is 384-dim, and the two vector spaces
# must never be conflated. Cosine is only meaningful within one
# ``embedding_space`` discriminator.
TEXT_EMBEDDING_DIM = 384


class Base(DeclarativeBase):
    type_annotation_map: ClassVar = {dict[str, Any]: JSON().with_variant(JSONB(), "postgresql")}


class RunStatus(enum.StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    AWAITING_ADJUDICATION = "awaiting_adjudication"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NotifiableEvent(enum.StrEnum):
    """Run transitions that emit a webhook (issue #12).

    A subset of ``RunStatus``: the outcomes an operator wants pushed. CANCELLED
    is deliberately excluded — the operator initiated it, so it is not news.
    Only ``COMPLETED`` is truly terminal; ``AWAITING_ADJUDICATION`` and
    ``FAILED`` are revisitable (requeue / recovery), so each arrival is keyed by
    the run's ``transition_revision`` rather than by event alone.
    """

    AWAITING_ADJUDICATION = "awaiting_adjudication"
    COMPLETED = "completed"
    FAILED = "failed"


class NotificationStatus(enum.StrEnum):
    """Lifecycle of one ``notification_deliveries`` outbox row (issue #12)."""

    PENDING = "pending"  # awaiting first delivery attempt (or a retry)
    IN_FLIGHT = "in_flight"  # claimed by a sweep with a lease; POST in progress
    DELIVERED = "delivered"  # receiver returned 2xx
    DEAD = "dead"  # gave up after notify_max_attempts
    SUPPRESSED = "suppressed"  # a FAILED arrival the run moved past before delivery


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

# The two execution lanes are an explicit partition of the canonical pipeline.
# POST_SEGMENT must remain a contiguous suffix of STAGE_ORDER: the engine hands a
# run from one lane to the other by parking it at the next stage, so an interleaved
# lane assignment would require repeated handoffs and defeat the deliberately small
# two-queue topology. Any future stage must consciously join exactly one segment
# (and preserve that suffix contract), rather than silently inheriting a queue.
GPU_SEGMENT: frozenset[Stage] = frozenset(
    {Stage.ACQUIRE, Stage.PREPARE, Stage.TRANSCRIBE, Stage.DIARIZE_EMBED}
)
POST_SEGMENT: frozenset[Stage] = frozenset({Stage.ENHANCE_MATCH, Stage.FINALIZE})


class StageStatus(enum.StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ArtifactKind(enum.StrEnum):
    PREPROCESSED_AUDIO = "preprocessed_audio"
    CHUNK = "chunk"
    TRANSCRIPT_EXPORT = "transcript_export"
    # Lazily-computed waveform amplitude envelope (issue #57); one per run,
    # cached under artifacts/{run_id}/peaks.json. Survives reclamation (the
    # sweep targets preprocessed_audio only) so a static waveform can still
    # render after the WAV is reclaimed.
    WAVEFORM_PEAKS = "waveform_peaks"


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
    # Segment-scope only (issue #54 Phase B): "this one segment inherits its
    # label's resolution" — the append-only reset that clears a per-segment
    # override without freezing a copied identity. Never used at label scope.
    INHERIT = "inherit"


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
        CheckConstraint(
            "sidecar IS NULL OR jsonb_typeof(sidecar) = 'object'",
            name="pipeline_runs_sidecar_object_check",
        ),
        # Mirror the pyannote service bounds (1..20) and the config field so a bad
        # hint can never reach the diarizer through this column.
        CheckConstraint(
            "diarization_max_speakers IS NULL"
            " OR (diarization_max_speakers >= 1 AND diarization_max_speakers <= 20)",
            name="pipeline_runs_diarization_max_speakers_check",
        ),
        CheckConstraint(
            "diarization_num_speakers IS NULL"
            " OR (diarization_num_speakers >= 1 AND diarization_num_speakers <= 20)",
            name="pipeline_runs_diarization_num_speakers_check",
        ),
        # The detection score is a probability; anything outside [0, 1] is a
        # malformed write, refused at the schema (#124).
        CheckConstraint(
            "detected_language_probability IS NULL"
            " OR (detected_language_probability >= 0"
            " AND detected_language_probability <= 1)",
            name="pipeline_runs_detected_language_probability_check",
        ),
        # A score describes a detected language: a probability with no language
        # is contradictory provenance (the reverse — a language with no score —
        # is the legitimate forced/fallback shape).
        CheckConstraint(
            "detected_language_probability IS NULL"
            " OR detected_language IS NOT NULL",
            name="pipeline_runs_detected_language_pairing_check",
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
    # Soft-archive stamp (issue #5): non-NULL hides the run from /runs and the
    # /review queue while keeping every row (incl. the append-only ledger)
    # intact. Reversible (un-archive → NULL). Operator-visibility metadata,
    # deliberately outside the CAS revision and orthogonal to status — like
    # operator_notes, last-write-wins, not pipeline state.
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Frozen domain-pack snapshot for this run (issue #11): the resolved manifest
    # (name/vocabulary/name_seeds/prompt_fragments) as JSON, stamped write-once at
    # submit from the per-folder mapping (or the default pack). The pipeline worker
    # and the enrichment producers both read THIS, not the mutable global env, so a
    # run — and its late enrichment — always sees the exact pack it was transcribed
    # with even if the manifest on disk later changes. NULL = a legacy run created
    # before #11: resolve the current default pack at execution time (see
    # DomainPack.from_mapping / resolve_run_domain_pack).
    domain_pack: Mapped[dict[str, Any] | None] = mapped_column()
    # The run's YAML sidecar, frozen write-once at submit (issue #104): the WHOLE
    # parsed mapping (JSON-normalized), so reference-only keys other tooling wrote
    # beside the media file survive for provenance. The machine-read fields
    # (title/speakers/domain_pack/notes) were already APPLIED at submit — this
    # column is the record, not a live input: editing the file after ingest
    # changes nothing, and NULL = no sidecar existed when the media was picked up
    # (one arriving later is deliberately too late). Title display reads this
    # tolerantly (api.presentation.title_from_snapshot); nothing downstream
    # re-parses it.
    sidecar: Mapped[dict[str, Any] | None] = mapped_column()
    # Per-recording diarization speaker-count hint (issue #128), frozen at submit
    # from a CLI flag or the YAML sidecar. NULL max ⇒ the worker falls back to the
    # install-wide default (settings.diarization_max_speakers) at execution;
    # a non-NULL value is an explicit per-run override. num is an EXACT count that
    # pins pyannote to that many speakers and takes precedence over max. Both are
    # bounded 1..20 (a CHECK mirrors the pyannote service and the config field).
    # Read in the worker alongside domain_pack; deliberately typed scalar columns,
    # not folded into the domain_pack manifest.
    diarization_max_speakers: Mapped[int | None] = mapped_column(Integer)
    diarization_num_speakers: Mapped[int | None] = mapped_column(Integer)
    # The bounded whisper ``initial_prompt`` this run actually decoded with (issue
    # #123): the rendered join of the effective vocabulary (pack + operator glossary,
    # deduped/capped), stamped by the transcribe stage. The frozen domain_pack
    # snapshot above records only the PACK's words; the operator's glossary is
    # unioned LIVE at run start (app_settings.vocabulary), so without this column the
    # names this run was actually told about are unrecoverable. NULL = a run not yet
    # transcribed, a run with no vocabulary (empty prompt), or a legacy run
    # transcribed before this column existed.
    initial_prompt: Mapped[str | None] = mapped_column(Text)
    # The language whisper actually transcribed this run in (issue #124), as the
    # service reported it (ISO-639-1 style code, e.g. "es"), stamped by the
    # transcribe stage after a successful decode. Voxint's client requests
    # auto-detection, so this records the model's decision, not operator input.
    # NULL = a run not yet transcribed or a legacy run transcribed before this
    # column existed (never backfilled — reconstructing it would fabricate
    # provenance). Low-cardinality; deliberately unindexed (single-operator).
    detected_language: Mapped[str | None] = mapped_column(Text)
    # Whisper's language-detection score for detected_language: a probability in
    # [0, 1] (CHECK-enforced), present only when detection actually ran. NOT a
    # calibrated confidence and NOT a code-switch signal — surfaced on the run
    # detail page with exactly that framing. NULL whenever detected_language is
    # NULL, and also when the service forced a language or substituted a
    # fallback (no honest score exists for a detection that did not happen).
    detected_language_probability: Mapped[float | None] = mapped_column(Float)
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
        # Reclamation (issue #15) stamps both columns together or neither — no
        # half-reclaimed rows to reason about (mirrors the review-claim shape).
        CheckConstraint(
            "(reclaimed_at IS NULL) = (reclaimed_bytes IS NULL)",
            name="audio_artifacts_reclaimed_shape_check",
        ),
        CheckConstraint(
            "reclaimed_bytes IS NULL OR reclaimed_bytes >= 0",
            name="audio_artifacts_reclaimed_bytes_nonneg_check",
        ),
        # Sweep predicate: unreclaimed preprocessed-audio rows for a given run.
        Index(
            "ix_audio_artifacts_reclaimable",
            "pipeline_run_id",
            postgresql_where=text("kind = 'preprocessed_audio' AND reclaimed_at IS NULL"),
        ),
        # One waveform-peaks cache row per run (issue #57): concurrent first
        # requests both compute, but INSERT … ON CONFLICT DO NOTHING against
        # this index keeps exactly one canonical row. Keep in lockstep with
        # migration 0021.
        Index(
            "uq_audio_artifacts_waveform_peaks",
            "pipeline_run_id",
            unique=True,
            postgresql_where=text("kind = 'waveform_peaks'"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    pipeline_run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("pipeline_runs.id"), index=True)
    kind: Mapped[str] = mapped_column(Text)
    path: Mapped[str] = mapped_column(Text)
    meta: Mapped[dict[str, Any] | None] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    # GC reclamation audit (issue #15): NULL until the sweep unlinks the file on
    # disk, then the UTC stamp + bytes measured at reclaim time (0 if the file
    # was already absent). The row itself is never deleted.
    reclaimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    reclaimed_bytes: Mapped[int | None] = mapped_column(BigInteger)


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
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="transcript_segments_confidence_range_check",
        ),
        # Shape backstop for the words JSONB (migration 0022), matching the
        # jsonb_typeof convention every other JSONB column uses. NULL = the run
        # had no word timing; an array (possibly empty) = it did. Paired with
        # none_as_null=True on the column so a wordless run stores SQL NULL (not
        # JSONB 'null', which would fail this CHECK).
        CheckConstraint(
            "words IS NULL OR jsonb_typeof(words) = 'array'",
            name="transcript_segments_words_array_check",
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
    # Raw ASR output is immutable; enhancement writes enhanced_text, never over
    # raw_text, and operator corrections (issue #58) live in segment_review_states
    # — also beside raw_text, never over it. raw_text stays the ASR evidence of record.
    raw_text: Mapped[str] = mapped_column(Text)
    enhanced_text: Mapped[str | None] = mapped_column(Text)
    # Local diarization label within this run (e.g. "SPEAKER_00"), not a speaker identity.
    diarization_label: Mapped[str | None] = mapped_column(Text)
    # ASR hallucination soft-tag, preserved verbatim from the service; gates weight it.
    suspect: Mapped[bool] = mapped_column(Boolean, default=False)
    # ASR confidence = exp(avg_logprob) clamped to [0, 1] (a transformed
    # likelihood, NOT a calibrated probability — docs/quality-gates.md). NULL when
    # the provider reported none (older runs, non-confidence backends); the #53
    # review console flags low-confidence segments and never fabricates a NULL.
    confidence: Mapped[float | None] = mapped_column(Float)
    # Per-word timings (issue #59), bucketed from the whisper service's flat
    # word_timestamps output: a list of {start, end, word, confidence}. NULL for
    # runs transcribed before #59 and providers without word timing — never
    # backfilled onto existing evidence rows. The word-boundary split UI reads
    # these; the ASR text/interval remain the numerics contract, words are derived
    # detail. No GIN index (never queried by content). none_as_null=True so a
    # wordless run persists SQL NULL, not JSONB 'null' — keeping `words IS NULL`
    # (and the array-shape CHECK above) honest.
    words: Mapped[list[dict[str, Any]] | None] = mapped_column(
        JSON(none_as_null=True).with_variant(JSONB(none_as_null=True), "postgresql"),
        nullable=True,
    )
    # Deterministic domain-pack correction provenance (#82). Either [] (no material
    # correction / re-enhance reset) or the envelope
    # {"version": int, "input_base": "raw"|"llm", "entries": [{id, from, to, span:[s,e]}]}.
    # Written by the enhance_match dual pass beside enhanced_text; they reset
    # atomically on re-enhance. A NON-EMPTY entries list is the authoritative
    # "a rule fired" signal the split machinery reads (never a re-diff of text).
    correction_trace: Mapped[dict[str, Any] | list[Any]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"),
        nullable=False,
        server_default=text("'[]'"),
    )
    # Corrector engine version stamped when a material correction/enhancement is
    # persisted (#82). NULL = legacy pre-#82 enhanced_text (rendered "enhanced
    # (unversioned)", never recomputed) OR a row with no persisted enhanced output.
    corrector_version: Mapped[int | None] = mapped_column(Integer, nullable=True)


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


class MatchCandidate(Base):
    """Observational per-label match evidence (issue #113).

    One row per diarization label per run recording what
    ``speakers.matching.match_speakers`` decided — accepted, rejected, or
    ineligible — and the numbers behind it (top candidate, cosine, margin,
    vote-agreement, eligibility). Unlike ``speaker_assignments`` (accepted
    proposals only), this keeps the near-misses that otherwise died in debug
    logs, so false *rejects* become visible and a baseline attribution-accuracy
    number can be measured. Purely diagnostic: nothing here feeds the resolver,
    centroids, or thresholds; writing it does not change matching.

    Written delete-then-insert by ``replace_run_match_candidates`` beside the
    proposals, idempotent under stage retry. Cascades on run deletion /
    re-transcription (disposable evidence, never a source of truth).
    """

    __tablename__ = "match_candidates"
    __table_args__ = (
        UniqueConstraint(
            "pipeline_run_id", "diarization_label", name="match_candidates_label_key"
        ),
        CheckConstraint(
            "decision IN ('accepted', 'rejected', 'ineligible')",
            name="match_candidates_decision_check",
        ),
        # An ineligible label never reached a roster comparison, so it has no
        # candidate and no numbers; accepted/rejected always do.
        CheckConstraint(
            "(decision = 'ineligible') = (top_speaker_id IS NULL)",
            name="match_candidates_ineligible_speaker_check",
        ),
        CheckConstraint(
            "(decision = 'ineligible') = (similarity IS NULL)",
            name="match_candidates_ineligible_similarity_check",
        ),
        CheckConstraint(
            "(decision = 'ineligible') = (vote_agreement IS NULL)",
            name="match_candidates_ineligible_vote_check",
        ),
        # Grounding is decided only for an accepted proposal.
        CheckConstraint(
            "(grounded IS NOT NULL) = (decision = 'accepted')",
            name="match_candidates_grounded_check",
        ),
        # Ranges mirror the matcher's own guarantees (clamped cosine, ratio
        # vote-agreement); margin can be NULL (single-speaker roster) and is left
        # unbounded because a clamped top-1 can sit a float-epsilon below top-2.
        CheckConstraint(
            "similarity IS NULL OR (similarity >= -1 AND similarity <= 1)",
            name="match_candidates_similarity_range_check",
        ),
        CheckConstraint(
            "vote_agreement IS NULL OR (vote_agreement >= 0 AND vote_agreement <= 1)",
            name="match_candidates_vote_range_check",
        ),
        CheckConstraint(
            "eligible_turns >= 0 AND eligible_seconds >= 0",
            name="match_candidates_eligibility_nonneg_check",
        ),
        CheckConstraint(
            "roster_size IS NULL OR roster_size >= 0",
            name="match_candidates_roster_size_check",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    pipeline_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("pipeline_runs.id", ondelete="CASCADE"), index=True
    )
    diarization_label: Mapped[str] = mapped_column(Text)
    decision: Mapped[str] = mapped_column(Text)
    reason: Mapped[str] = mapped_column(Text)
    embedding_space: Mapped[str | None] = mapped_column(Text)
    # The top roster candidate at match time. NULL only for an ineligible label
    # (no candidate). Speakers are soft-archived/merged (the row persists), never
    # hard-deleted, so this plain FK never blocks in practice; it refuses a stray
    # hard delete rather than silently nulling recorded evidence (which would also
    # break the ineligible-speaker coherence CHECK).
    top_speaker_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("speakers.id"))
    similarity: Mapped[float | None] = mapped_column(Float)  # raw cosine, [-1, 1]
    margin: Mapped[float | None] = mapped_column(Float)  # top-1 vs top-2; NULL if 1 speaker
    vote_agreement: Mapped[float | None] = mapped_column(Float)
    grounded: Mapped[bool | None] = mapped_column(Boolean)
    eligible_turns: Mapped[int] = mapped_column(Integer, default=0)
    eligible_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    roster_size: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


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
        # INHERIT is a segment-scope reset only; it is meaningless at label scope
        # (issue #54 Phase B). A NULL transcript_segment_id is label scope.
        CheckConstraint(
            "decision != 'inherit' OR transcript_segment_id IS NOT NULL",
            name="adjudication_decisions_inherit_segment_check",
        ),
        # Segment scope carries only assign or inherit; exclude/unknown are
        # label-scope concepts (the relabel route already enforces this — this is
        # the persistence-boundary backstop).
        CheckConstraint(
            "transcript_segment_id IS NULL OR decision IN ('assign', 'inherit')",
            name="adjudication_decisions_segment_decision_check",
        ),
        # Batch-load the per-run segment overlay in one indexed pass.
        Index(
            "ix_adjudication_decisions_run_segment",
            "pipeline_run_id",
            "transcript_segment_id",
        ),
        # Word-range scope (issue #59 slice 3): start/end are set together or
        # both NULL — a half-open [start, end) range over the parent's words, or
        # no range at all.
        CheckConstraint(
            "(start_word_index IS NULL) = (end_word_index IS NULL)",
            name="adjudication_decisions_word_range_pair_check",
        ),
        # A present range must scope a segment (never a bare label) and be a
        # well-formed non-empty half-open interval. The contrapositive also gives
        # the plan's "label-scope rows keep the range NULL": no segment ⇒ no range.
        CheckConstraint(
            "start_word_index IS NULL OR ("
            "transcript_segment_id IS NOT NULL"
            " AND start_word_index >= 0"
            " AND end_word_index > start_word_index)",
            name="adjudication_decisions_word_range_bounds_check",
        ),
        # Batch-load the per-run word-range overlay in one indexed pass, newest
        # first per (segment, range). Mirrors ix_..._run_segment for the finer
        # sub-segment grain (issue #59 slice 3).
        Index(
            "ix_adjudication_decisions_word_range",
            "pipeline_run_id",
            "transcript_segment_id",
            "start_word_index",
            "end_word_index",
            "created_at",
            "id",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    pipeline_run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("pipeline_runs.id"), index=True)
    diarization_label: Mapped[str] = mapped_column(Text)
    decision: Mapped[str] = mapped_column(Text)
    speaker_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("speakers.id"))
    # NULL = label scope (the historical grain: rules the whole (run, label)).
    # Non-NULL = segment scope: this ruling overrides just that one transcript
    # segment. The writer derives diarization_label from the segment row, so the
    # two always agree (issue #54 Phase B).
    transcript_segment_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("transcript_segments.id"), nullable=True
    )
    # Word-range scope (issue #59 slice 3): a half-open ``[start, end)`` interval
    # into the parent segment's immutable ``words`` list, addressing one derived
    # split child (or any word-range) for reassignment. Both NULL = whole-segment
    # scope (the 0018 grain); both set = this sub-segment range. The range keys on
    # the immutable parent id + word offsets, never on a disposable boundary row,
    # so it survives re-split/un-split. A CHECK pairs them and bounds the interval.
    start_word_index: Mapped[int | None] = mapped_column(nullable=True)
    end_word_index: Mapped[int | None] = mapped_column(nullable=True)
    operator: Mapped[str] = mapped_column(Text)
    idempotency_key: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# Length bound on operator-corrected text (issue #58): a segment is a short
# utterance; this is a pathological-input sanity cap, not a UX limit. The route
# validates against it and the DB CHECK below is the backstop.
MAX_CORRECTED_TEXT_CHARS = 20_000


class SegmentReviewState(Base):
    """Mutable per-segment operator workflow state (issues #53/#58): the verified
    mark (#53) and the operator's corrected text (#58). One row per reviewed
    segment, UPSERTed latest-wins.

    Deliberately NOT the append-only ``adjudication_decisions`` ledger — verified
    /corrected is orthogonal to speaker attribution and would violate its CHECK
    grammar and pollute resolution counts — and NOT columns on the immutable
    ``transcript_segments`` observation row. ``raw_text`` stays the ASR evidence
    of record; a correction is written *beside* it, never over it. See
    docs/plans/2026-08-16-2227_transcript-text-correction-provenance.md.

    ``pipeline_run_id`` is denormalized (derived from the segment by the one
    writer) so the per-run overlay, the "N of M verified" counter, and the search
    join load without a join — mirroring how ``adjudication_decisions`` carries it.
    """

    __tablename__ = "segment_review_states"
    __table_args__ = (
        # corrected_at is set exactly when corrected_text is (paired-shape).
        CheckConstraint(
            "(corrected_text IS NULL) = (corrected_at IS NULL)",
            name="segment_review_states_corrected_pair_check",
        ),
        CheckConstraint(
            f"corrected_text IS NULL OR char_length(corrected_text) <= {MAX_CORRECTED_TEXT_CHARS}",
            name="segment_review_states_corrected_len_check",
        ),
        Index("ix_segment_review_states_run", "pipeline_run_id"),
        # Corrected text is FTS-searchable (issue #58, D3): a PARTIAL GIN index,
        # since corrected_text is NULL for most rows. Declared here for the
        # model↔migration parity test; the DDL literal must match db.search and
        # migration 0020 (contract-tested). Never coalesced with raw/enhanced.
        Index(
            search.CORRECTED_FTS_INDEX_NAME,
            text(f"to_tsvector('{search.TS_CONFIG}', corrected_text)"),
            postgresql_using="gin",
            postgresql_where=text("corrected_text IS NOT NULL"),
        ),
    )

    transcript_segment_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("transcript_segments.id", ondelete="CASCADE"), primary_key=True
    )
    pipeline_run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("pipeline_runs.id"))
    # NULL = unverified; a timestamp = the operator has confirmed this segment.
    verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # NULL = no correction (segment renders enhanced-or-raw); non-NULL = operator
    # text, which takes display/export/search precedence. Empty/whitespace-only is
    # normalized to NULL at the writer, so this is never an empty rendering.
    corrected_text: Mapped[str | None] = mapped_column(Text)
    corrected_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class SegmentSplitBoundary(Base):
    """One operator word-boundary split of a transcript segment (issue #59).

    A split is stored as an append-only CUT — ``word_index`` = "split before word
    i" on the immutable parent segment — NOT as a new ``transcript_segments`` row
    and NOT as a mutable overlay the append-only ledger would point at. Children
    are DERIVED at read time from the cut set ``{0, cuts…, word_count}``; the
    parent row is never mutated (``raw_text``/interval/``words`` stay the ASR
    evidence of record). See docs/plans + the #59 memory for why real child rows
    and an overlay-id FK were both rejected.

    ``pipeline_run_id`` is denormalized (the one writer derives it from the
    parent) so ``attributed_transcript`` batch-loads a run's boundaries in one
    indexed query — mirroring how ``adjudication_decisions`` carries it. The FK to
    the parent is ON DELETE CASCADE so re-transcription (new segment ids) does not
    leak orphan boundaries.
    """

    __tablename__ = "segment_split_boundaries"
    __table_args__ = (
        # "split before word i" is INTERIOR: i == 0 (segment start) is not a cut.
        # The upper bound (i < word_count) is cross-table (parent word count), so
        # it is validated in the writer, not a CHECK.
        CheckConstraint(
            "word_index >= 1",
            name="segment_split_boundaries_word_index_interior_check",
        ),
        # A split is STRUCTURALLY idempotent: "before word i" exists at most once,
        # so a replayed / double-clicked split is a no-op (writer: ON CONFLICT DO
        # NOTHING) — no client nonce needed, unlike the relabel ledger.
        UniqueConstraint(
            "parent_segment_id",
            "word_index",
            name="segment_split_boundaries_parent_word_key",
        ),
        Index("ix_segment_split_boundaries_run", "pipeline_run_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    pipeline_run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("pipeline_runs.id"))
    parent_segment_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("transcript_segments.id", ondelete="CASCADE")
    )
    # "Split before word i" (1 <= i < word_count): the cut falls between word i-1
    # and word i of the parent's immutable ``words`` list.
    word_index: Mapped[int] = mapped_column(Integer)
    operator: Mapped[str] = mapped_column(Text)
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
    at process start): infra config and infra secrets — ``DATABASE_URL``,
    ``REDIS_URL``, ports — stay in the environment and the wizard never rewrites
    ``.env``. Exactly one row exists, pinned by the ``id = 1`` CHECK; the API reads
    it per request and the worker snapshots it per run (see
    ``pipeline.stages.context``). ``llm_base_url`` / ``llm_model`` are nullable —
    NULL means "fall back to the env default".

    ``llm_api_key`` (issue #10) is the one credential stored here so a
    non-technical operator can configure LLM enhancement entirely from the UI. A
    non-blank value WINS over env ``LLM_API_KEY``; NULL/blank falls back to it
    (mirrors ``llm_enabled``, which is taken hard from the row). It is stored
    **plaintext at rest** — an accepted trade-off for this single-operator,
    local-first deployment (not a shared multi-tenant store; a SQL dump necessarily
    contains it). It must never be rendered back to the UI, logged, put in an error
    message, or exported: resolve it only through
    ``app_settings.resolve_effective_llm_api_key`` and keep it out of any
    repr/serialization (this model defines no custom ``__repr__``, so the default
    shows only the class + primary key).

    ``web_search_api_key`` (issue #74) is the second credential stored here, for
    the web-search provider, and carries the identical contract: non-blank wins
    over env ``WEB_SEARCH_API_KEY``, plaintext at rest, never rendered/logged/
    exported, resolved only via ``app_settings.resolve_effective_web_search_api_key``.
    The tri-state feature-flag columns (issue #74) are NON-secret: NULL means
    "inherit the env default", a non-NULL value overrides it.
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
    # Per-folder domain-pack assignment (issue #11): {media_folder -> pack_name},
    # mapping a watched folder (as stored in media_folders, relative to MEDIA_ROOT)
    # to a pack resolvable by name. Consulted at submit to freeze each run's pack
    # snapshot; an unmapped folder falls back to the default pack. Default {} means
    # every folder uses the default — the pre-#11 behavior.
    folder_domain_packs: Mapped[dict[str, str]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"),
        nullable=False,
        default=dict,
        server_default=text("'{}'"),
    )
    # Operator-authored correction rules (issue #84): a list of rule mappings
    # {id, match, replace, case_sensitive, whole_word}, each already validated
    # through the #80 gate at author time. Unioned onto the selected pack's
    # corrections at submit-time freeze (see
    # ingest.service._run_domain_pack_snapshot) and stored in
    # pipeline_runs.domain_pack — NOT applied live like vocabulary — so #82
    # compose and #83 provenance read them off the frozen snapshot unchanged.
    # Named "corrections" to mirror the pack field; distinct from the manual
    # per-segment review edits in SegmentReviewState.corrected_text.
    corrections: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"),
        nullable=False,
        default=list,
        server_default=text("'[]'"),
    )
    llm_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    # NULL -> use the env default (config.Settings).
    llm_base_url: Mapped[str | None] = mapped_column(Text)
    llm_model: Mapped[str | None] = mapped_column(Text)
    # In-UI LLM API key (issue #10). NULL/blank -> fall back to env LLM_API_KEY; a
    # non-blank value wins. Credential, plaintext at rest — never render/log/export;
    # resolve only via app_settings.resolve_effective_llm_api_key. See class docstring.
    llm_api_key: Mapped[str | None] = mapped_column(Text)
    # Live-read feature flags the settings console can toggle at runtime (issue #74,
    # under the #47 arc). Tri-state: NULL means "inherit the env default"
    # (config.Settings), a non-NULL value overrides it — the llm_base_url nullable
    # pattern, NOT the llm_enabled hard-row pattern, so an operator can always revert
    # an override to the installation setting by clearing it (writes NULL). Every
    # runtime gate resolves these through app_settings.resolve_effective_<flag>, so a
    # UI toggle applies with no restart and no read-site can drift onto a bare env
    # read. Names mirror the config fields exactly.
    enrichment_names_enabled: Mapped[bool | None] = mapped_column(Boolean)
    enrichment_names_llm_enabled: Mapped[bool | None] = mapped_column(Boolean)
    enrichment_run_assets_enabled: Mapped[bool | None] = mapped_column(Boolean)
    enrichment_run_assets_autogenerate: Mapped[bool | None] = mapped_column(Boolean)
    voxint_web_research: Mapped[bool | None] = mapped_column(Boolean)
    enrichment_web_research_enabled: Mapped[bool | None] = mapped_column(Boolean)
    ytdlp_enabled: Mapped[bool | None] = mapped_column(Boolean)
    # Transcript semantic-search embedding spine (#121). Same tri-state: NULL
    # inherits the env default, non-NULL overrides. These two depend only on each
    # other (autogenerate ⇒ enabled), never on llm_enabled — the embedder is a
    # local ONNX graph with no LLM and no egress. Resolve only via
    # resolve_effective_semantic_index_{enabled,autogenerate}.
    semantic_index_enabled: Mapped[bool | None] = mapped_column(Boolean)
    semantic_index_autogenerate: Mapped[bool | None] = mapped_column(Boolean)
    # Optional bundled local LLM (issue #67). Same tri-state as the flags above:
    # NULL inherits env LLM_BUNDLED_ENABLED, non-NULL overrides. When effective
    # AND a bundled base URL is configured, enhancement + run-asset
    # summary/entities route to the keyless bundled endpoint; names + research
    # stay BYO. Resolve only via resolve_effective_llm_bundled_enabled.
    llm_bundled_enabled: Mapped[bool | None] = mapped_column(Boolean)
    # Watch-folder ingest runtime override (issue #60). Tri-state like the flags
    # above: NULL inherits the env default (config.watch_folder_enabled, off), a
    # non-NULL value overrides it — so the operator can enable/disable from the
    # Settings folders panel with no restart, and revert to the installation
    # setting by clearing it. Resolved via resolve_effective_watch_folder_enabled.
    watch_folder_enabled: Mapped[bool | None] = mapped_column(Boolean)
    # Latest watch-sweep summary (issue #60) for the plain-language Settings status
    # line — the ONLY sweep state persisted (no history/per-file ledger). Keys:
    # picked_up, already_known, settling, deferred, stat_errors, hit_entry_cap,
    # hit_file_cap, root_missing, completed_at (ISO-8601). NULL means the sweep has
    # never run.
    watch_folder_last_sweep: Mapped[dict[str, Any] | None] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql")
    )
    # External-sources config (issue #74). NULL/blank -> inherit the env default; a
    # non-blank value overrides it (the llm_base_url/llm_api_key precedent).
    source_authority_domains: Mapped[str | None] = mapped_column(Text)
    web_search_base_url: Mapped[str | None] = mapped_column(Text)
    # In-UI web-search provider credential (issue #74). NULL/blank -> fall back to env
    # WEB_SEARCH_API_KEY; a non-blank value wins. Credential, plaintext at rest — never
    # render/log/export; resolve only via app_settings.resolve_effective_web_search_api_key.
    # Same handling as llm_api_key (see class docstring).
    web_search_api_key: Mapped[str | None] = mapped_column(Text)
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


class RunAssetKind(enum.StrEnum):
    SUMMARY = "summary"
    TOPICS = "topics"
    ENTITY_MENTIONS = "entity_mentions"


class RunAssetJobStatus(enum.StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunEnrichmentAsset(Base):
    """One **successful** run-level asset generation (issue #41).

    A whole machine-generated document about a run — a summary, a topic list,
    or entity mentions — not a reviewable per-field claim (#37 handles those).
    Rows are inserted only on success by the single sanctioned writer
    (``enrichment/run_assets.py``); failed attempts live on ``run_asset_jobs``
    and never consume a generation. Content is immutable: the only permitted
    mutation is stamping ``superseded_by_asset_id`` once when a newer
    generation of the *same kind* lands (trigger-enforced, like
    ``enrichment_candidates``). ``generation`` is monotonic per
    (pipeline_run_id, asset_kind), allocated under an advisory lock —
    independence between the three kinds is exactly this per-kind keying.

    ``source_content_hash`` is the staleness detector: a sha256 over the
    canonical serialization of everything the generator read (the transcript
    with its raw diarization labels, source metadata, operator notes —
    resolved speaker names are a deliberate v1 cut) — content only, never
    model/prompt versions, so a prompt upgrade cannot masquerade as a source
    change. Stale = recomputed current hash differs. It is deliberately NOT
    the idempotency key: an operator regenerate on unchanged input is an
    intentional new generation, never silently suppressed (the #40 lesson);
    dedup applies only to duplicate delivery of the same job.
    """

    __tablename__ = "run_enrichment_assets"
    __table_args__ = (
        UniqueConstraint(
            "pipeline_run_id",
            "asset_kind",
            "generation",
            name="run_enrichment_assets_generation_key",
        ),
        UniqueConstraint("idempotency_key", name="run_enrichment_assets_idempotency_key"),
        CheckConstraint(
            f"asset_kind IN ({_enum_values(RunAssetKind)})",
            name="run_enrichment_assets_kind_check",
        ),
        CheckConstraint("generation >= 1", name="run_enrichment_assets_generation_check"),
        CheckConstraint(
            "jsonb_typeof(payload) = 'object'",
            name="run_enrichment_assets_payload_object_check",
        ),
        CheckConstraint(
            "payload_schema_version >= 1",
            name="run_enrichment_assets_payload_version_check",
        ),
        CheckConstraint(
            "length(trim(producer)) > 0",
            name="run_enrichment_assets_producer_nonempty_check",
        ),
        CheckConstraint(
            "length(trim(producer_version)) > 0",
            name="run_enrichment_assets_producer_version_nonempty_check",
        ),
        CheckConstraint(
            "length(trim(model)) > 0",
            name="run_enrichment_assets_model_nonempty_check",
        ),
        CheckConstraint(
            "source_content_hash ~ '^[0-9a-f]{64}$'",
            name="run_enrichment_assets_source_hash_check",
        ),
        CheckConstraint(
            "(config IS NULL) = (config_schema_version IS NULL)",
            name="run_enrichment_assets_config_pair_check",
        ),
        CheckConstraint(
            "config IS NULL OR jsonb_typeof(config) = 'object'",
            name="run_enrichment_assets_config_object_check",
        ),
        CheckConstraint(
            "config_schema_version IS NULL OR config_schema_version >= 1",
            name="run_enrichment_assets_config_version_check",
        ),
        CheckConstraint(
            "completed_at >= started_at",
            name="run_enrichment_assets_completed_after_started_check",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    pipeline_run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("pipeline_runs.id"), index=True)
    asset_kind: Mapped[str] = mapped_column(Text)
    generation: Mapped[int] = mapped_column(Integer)
    payload: Mapped[dict[str, Any]] = mapped_column()
    payload_schema_version: Mapped[int] = mapped_column(Integer)
    producer: Mapped[str] = mapped_column(Text)
    producer_version: Mapped[str] = mapped_column(Text)
    # The exact model identifier the generation ran with (provenance, shown
    # alongside the payload; a regeneration under a new model is visible here
    # even when the source hash is unchanged).
    model: Mapped[str] = mapped_column(Text)
    source_content_hash: Mapped[str] = mapped_column(Text)
    config: Mapped[dict[str, Any] | None] = mapped_column()
    config_schema_version: Mapped[int | None] = mapped_column(Integer)
    idempotency_key: Mapped[str] = mapped_column(Text)
    superseded_by_asset_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("run_enrichment_assets.id")
    )
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class RunAssetJob(Base):
    """One generation attempt for one (run, asset kind) — issue #41.

    Mutable orchestration state only, exactly like ``research_jobs``: the
    asset *result* is an immutable ``run_enrichment_assets`` row linked via
    ``asset_id`` when the attempt succeeded. One job = one durable execution;
    a rerun is a NEW job (the job id anchors the asset's idempotency key).
    ``status`` moves queued → running (guarded claim UPDATE, duplicate Celery
    delivery no-ops) → succeeded | failed | cancelled. A failed or cancelled
    job records NO asset and consumes NO generation, so one kind failing can
    never block or retire the other kinds' assets — the issue's failure
    isolation is structural. The partial unique index allows one active job
    per (run, kind); cancel is deadline-aware (one LLM call + grace) so a
    crashed RUNNING row cannot hold that slot forever.
    """

    __tablename__ = "run_asset_jobs"
    __table_args__ = (
        # DB-enforced "one active job per (run, kind)": the console's friendly
        # pre-check is check-then-insert and can race; this index cannot.
        Index(
            "run_asset_jobs_one_active_per_run_kind",
            "pipeline_run_id",
            "asset_kind",
            unique=True,
            postgresql_where=text("status IN ('queued', 'running')"),
        ),
        CheckConstraint(
            f"asset_kind IN ({_enum_values(RunAssetKind)})",
            name="run_asset_jobs_kind_check",
        ),
        CheckConstraint(
            f"status IN ({_enum_values(RunAssetJobStatus)})",
            name="run_asset_jobs_status_check",
        ),
        CheckConstraint(
            "jsonb_typeof(config) = 'object'",
            name="run_asset_jobs_config_object_check",
        ),
        CheckConstraint(
            "started_at IS NULL OR started_at >= created_at",
            name="run_asset_jobs_started_after_created_check",
        ),
        CheckConstraint(
            "finished_at IS NULL OR started_at IS NOT NULL",
            name="run_asset_jobs_finished_requires_started_check",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    pipeline_run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("pipeline_runs.id"), index=True)
    asset_kind: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, default=RunAssetJobStatus.QUEUED.value)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    # Snapshot of the execution settings this job was started under (model,
    # endpoint, input bound) — the worker reconstructs from this, so a
    # settings change between enqueue and execution never silently applies.
    config: Mapped[dict[str, Any]] = mapped_column()
    # Bounded, redacted failure summary (closed vocabulary + safe detail only).
    error: Mapped[str | None] = mapped_column(Text)
    asset_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("run_enrichment_assets.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class EmbeddingJobStatus(enum.StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SegmentEmbedding(Base):
    """One embedded transcript chunk — the semantic-search spine (issue #121).

    An additive artifact: the embedding producer reads finished transcript text
    (resolved via ``attributed_transcript`` → ``paragraphize_transcript``) and
    writes these rows. It never touches ASR / diarization / TitaNet, so it does
    not trip the numerics parity gate. Cosine is only valid within one
    ``embedding_space`` (the MiniLM text space, e.g.
    ``"minilm-multi-l12-onnx-fp32-mean-v1"``) — never compared against the
    192-dim TitaNet speaker space.

    Chunks are paragraph-derived and ephemeral (paragraph boundaries shift when
    a correction, split, or speaker ruling changes), so the span and text live
    ON the row rather than behind a segment FK. ``generation`` is monotonic per
    (pipeline_run_id, embedding_space): a re-embed publishes a whole new
    generation of the run's chunks atomically and the old generation is
    replaced, never half-seen. ``content_hash`` is the sha256 of the exact
    embedded string, for cheap per-chunk change detection within a rebuild.
    """

    __tablename__ = "segment_embeddings"
    __table_args__ = (
        UniqueConstraint(
            "pipeline_run_id",
            "embedding_space",
            "generation",
            "chunk_index",
            name="segment_embeddings_chunk_key",
        ),
        CheckConstraint(
            "length(trim(embedding_space)) > 0",
            name="segment_embeddings_space_nonempty_check",
        ),
        CheckConstraint("generation >= 1", name="segment_embeddings_generation_check"),
        CheckConstraint("chunk_index >= 0", name="segment_embeddings_chunk_index_check"),
        CheckConstraint(
            "start_seconds >= 0 AND end_seconds >= start_seconds",
            name="segment_embeddings_interval_check",
        ),
        CheckConstraint(
            "content_hash ~ '^[0-9a-f]{64}$'",
            name="segment_embeddings_content_hash_check",
        ),
        # The current-generation lookup path: filter by run+space, order/scan by
        # generation. No ANN index in v1 — exact cosine scan is sub-second at
        # single-operator scale; add HNSW only on measured latency evidence.
        Index(
            "ix_segment_embeddings_run_space_generation",
            "pipeline_run_id",
            "embedding_space",
            "generation",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    pipeline_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("pipeline_runs.id", ondelete="CASCADE"), index=True
    )
    embedding_space: Mapped[str] = mapped_column(Text)
    generation: Mapped[int] = mapped_column(Integer)
    chunk_index: Mapped[int] = mapped_column(Integer)
    start_seconds: Mapped[float] = mapped_column(Float)
    end_seconds: Mapped[float] = mapped_column(Float)
    # The local diarization/display speaker for the chunk's dominant span, for a
    # future speaker facet; nullable because a chunk may span an unlabeled gap.
    speaker_label: Mapped[str | None] = mapped_column(Text)
    # Which rendering the chunk text came from (corrected/enhanced/raw); a
    # paragraph assembled from mixed renderings records its dominant one.
    text_rendering: Mapped[str] = mapped_column(Text)
    # The exact text that was embedded — kept on the row so a semantic hit can
    # show a passage snippet and a jump target without re-resolving the run.
    chunk_text: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(Text)
    embedding: Mapped[Any] = mapped_column(Vector(TEXT_EMBEDDING_DIM))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EmbeddingJob(Base):
    """One embedding-index build attempt for one (run, embedding_space) — #121.

    A dedicated lane, deliberately NOT the LLM-coupled run-asset job family:
    embedding needs no LLM client and no ``llm_enabled`` gate. It reuses the
    proven lifecycle *patterns* only — ``status`` moves queued → running
    (guarded claim UPDATE, duplicate Celery delivery no-ops) → succeeded |
    failed | cancelled; a partial unique index allows one active job per
    (run, space); ``source_content_hash`` is the staleness detector (a run is
    stale when its recomputed resolved-transcript hash differs from the current
    generation's). A succeeded job is the generation manifest: it records the
    ``generation`` it published and links nothing else — the vectors live in
    ``segment_embeddings``. A failed/cancelled job publishes no generation.
    """

    __tablename__ = "embedding_jobs"
    __table_args__ = (
        Index(
            "embedding_jobs_one_active_per_run_space",
            "pipeline_run_id",
            "embedding_space",
            unique=True,
            postgresql_where=text("status IN ('queued', 'running')"),
        ),
        CheckConstraint(
            f"status IN ({_enum_values(EmbeddingJobStatus)})",
            name="embedding_jobs_status_check",
        ),
        CheckConstraint(
            "length(trim(embedding_space)) > 0",
            name="embedding_jobs_space_nonempty_check",
        ),
        CheckConstraint(
            "generation IS NULL OR generation >= 1",
            name="embedding_jobs_generation_check",
        ),
        CheckConstraint(
            "source_content_hash ~ '^[0-9a-f]{64}$'",
            name="embedding_jobs_source_hash_check",
        ),
        CheckConstraint(
            "started_at IS NULL OR started_at >= created_at",
            name="embedding_jobs_started_after_created_check",
        ),
        CheckConstraint(
            "finished_at IS NULL OR started_at IS NOT NULL",
            name="embedding_jobs_finished_requires_started_check",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    pipeline_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("pipeline_runs.id", ondelete="CASCADE"), index=True
    )
    embedding_space: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, default=EmbeddingJobStatus.QUEUED.value)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    source_content_hash: Mapped[str] = mapped_column(Text)
    # The generation this job published, stamped on success; NULL until then.
    generation: Mapped[int | None] = mapped_column(Integer)
    # Bounded, redacted failure summary (closed vocabulary + safe detail only).
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class NotificationDelivery(Base):
    """Transactional outbox for run webhooks (issue #12).

    One row per notifiable transition arrival, inserted in the SAME transaction
    as the run's ``cas_update_run`` so delivery intent is atomic with the state
    change (at-least-once, no commit-to-broker loss window). A separate beat
    sweep claims due rows under a lease, POSTs a signed payload outside any DB
    transaction, and records the outcome. The row is never the source of truth
    for run state — only for whether the operator was told.

    Keyed by ``(pipeline_run_id, transition_revision)``: only ``COMPLETED`` is
    truly terminal, so a requeued run that fails again is a *distinct* arrival
    with its own ``id`` (the receiver's dedup key). Delivery retries reuse the
    same row/id.
    """

    __tablename__ = "notification_deliveries"
    __table_args__ = (
        CheckConstraint(
            f"event IN ({_enum_values(NotifiableEvent)})",
            name="notification_deliveries_event_check",
        ),
        CheckConstraint(
            f"status IN ({_enum_values(NotificationStatus)})",
            name="notification_deliveries_status_check",
        ),
        CheckConstraint("attempts >= 0", name="notification_deliveries_attempts_nonneg_check"),
        # delivered_at is set iff the row reached DELIVERED — no half-delivered rows.
        CheckConstraint(
            "(status = 'delivered') = (delivered_at IS NOT NULL)",
            name="notification_deliveries_delivered_shape_check",
        ),
        # One outbox row per distinct transition arrival (the occurrence key).
        UniqueConstraint(
            "pipeline_run_id",
            "transition_revision",
            name="uq_notification_deliveries_run_revision",
        ),
        # Sweep predicate: due rows (pending or a lapsed in-flight lease) oldest first.
        Index(
            "ix_notification_deliveries_due",
            "next_attempt_at",
            postgresql_where=text("status IN ('pending', 'in_flight')"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    pipeline_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("pipeline_runs.id", ondelete="CASCADE"), index=True
    )
    # RunStatus revision the run held at this transition — the arrival identity.
    transition_revision: Mapped[int] = mapped_column(Integer)
    event: Mapped[str] = mapped_column(Text)
    # Frozen at emission: the exact object serialized (deterministically) and signed
    # per attempt. Versioned via a schema_version field inside it.
    payload: Mapped[dict[str, Any]] = mapped_column()
    status: Mapped[str] = mapped_column(Text, default=NotificationStatus.PENDING.value)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    # When this row next becomes eligible for a delivery attempt (FAILED rows get
    # a short initial delay so a synchronous requeue settles before we notify).
    next_attempt_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    # Set when claimed IN_FLIGHT; a sweep may reclaim the row once this passes.
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Bounded + redacted transport error (never the URL, secret, or payload).
    last_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# Operator annotation layer (issue #86). Caps are server-enforced (the route
# validates, the DB CHECK is the backstop) and mirrored in docs/annotations.md,
# the contract-of-record this schema must match. HIGHLIGHT_PALETTE_SIZE is the
# fixed 6-color highlight palette; a color index is 0..HIGHLIGHT_PALETTE_SIZE-1.
HIGHLIGHT_PALETTE_SIZE = 6
MAX_ANNOTATION_SPAN_SEGMENTS = 100
MAX_ANNOTATION_NOTE_CHARS = 4000
MAX_TAGS_PER_ANNOTATION = 8
MAX_ANNOTATION_QUOTE_CHARS = 50_000
MAX_TAG_NAME_CHARS = 64


class AnnotationTag(Base):
    """A global, flat operator tag (issue #86): a name plus a palette color.

    Tags are shared across runs (not run-scoped) and flat (no hierarchy —
    scope guard). ``name_normalized`` (trimmed, casefolded by the sole writer)
    carries the UNIQUE constraint, so "Key Point" and "key point" collide;
    ``name`` preserves the operator's verbatim casing for display. There is no
    tag delete in v1 — ``archived_at`` hides a tag from pickers while leaving it
    visible on annotations that already carry it (archive/restore via PATCH).
    """

    __tablename__ = "annotation_tags"
    __table_args__ = (
        UniqueConstraint("name_normalized", name="annotation_tags_name_normalized_key"),
        CheckConstraint("char_length(trim(name)) > 0", name="annotation_tags_name_nonempty_check"),
        CheckConstraint(
            f"char_length(name) <= {MAX_TAG_NAME_CHARS}",
            name="annotation_tags_name_len_check",
        ),
        CheckConstraint(
            f"color >= 0 AND color <= {HIGHLIGHT_PALETTE_SIZE - 1}",
            name="annotation_tags_color_range_check",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    # Verbatim operator casing, trimmed by the writer. Display source.
    name: Mapped[str] = mapped_column(Text)
    # Writer-computed trim+casefold; carries the uniqueness constraint so
    # case/whitespace variants of one tag cannot both exist.
    name_normalized: Mapped[str] = mapped_column(Text)
    color: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    # NULL = active (shown in pickers); a timestamp = archived (hidden from
    # pickers, still rendered on existing annotations). No hard delete in v1.
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class TranscriptAnnotation(Base):
    """One operator annotation over a transcript span (issue #86): a highlight
    color, an optional margin note, and zero or more tags (via
    ``annotation_tag_links``).

    A mutable operator workspace, NOT an adjudication record: rows are edited,
    re-anchored, and soft-deleted, and no annotation code path ever writes
    ``raw_text``/``enhanced_text``/``corrected_text`` (a contract test enforces
    this). Anchors always address the IMMUTABLE parent ``transcript_segments``
    id in one of three kinds (``docs/annotations.md``); split children are a
    read-time projection, so re-split/un-split never rewrites a stored anchor
    or hash.

    ``pipeline_run_id`` is denormalized by the sole writer from the endpoint
    segments (mirroring ``segment_review_states``) so the run listing loads
    without a join; ``start_segment_index``/``end_segment_index`` are captured
    copies of the endpoint segment positions (stable per run) that give the
    transcript-order listing and span iteration without a segment join.
    """

    __tablename__ = "transcript_annotations"
    __table_args__ = (
        UniqueConstraint("idempotency_key", name="transcript_annotations_idempotency_key"),
        CheckConstraint(
            "anchor_schema_version = 1",
            name="transcript_annotations_schema_version_check",
        ),
        CheckConstraint(
            "anchor_kind IN ('word_range', 'text_range', 'segment_range')",
            name="transcript_annotations_anchor_kind_check",
        ),
        # Hash is stored for every kind (staleness applies to all): full sha256,
        # lowercase hex, 64 chars. Column is NOT NULL; this pins the shape.
        CheckConstraint(
            "source_text_hash ~ '^[0-9a-f]{64}$'",
            name="transcript_annotations_source_hash_hex_check",
        ),
        # Paired nullability: each offset pair is set together or both NULL.
        CheckConstraint(
            "(start_word_index IS NULL) = (end_word_index IS NULL)",
            name="transcript_annotations_word_pair_check",
        ),
        CheckConstraint(
            "(start_char_offset IS NULL) = (end_char_offset IS NULL)",
            name="transcript_annotations_char_pair_check",
        ),
        # Per-kind shape (the classification truth table, persistence backstop):
        # word_range ⇒ word pair present, char pair absent; text_range ⇒ char
        # pair present, word pair absent; segment_range ⇒ all four absent.
        CheckConstraint(
            "(anchor_kind = 'word_range'"
            " AND start_word_index IS NOT NULL AND end_word_index IS NOT NULL"
            " AND start_char_offset IS NULL AND end_char_offset IS NULL)"
            " OR (anchor_kind = 'text_range'"
            " AND start_char_offset IS NOT NULL AND end_char_offset IS NOT NULL"
            " AND start_word_index IS NULL AND end_word_index IS NULL)"
            " OR (anchor_kind = 'segment_range'"
            " AND start_word_index IS NULL AND end_word_index IS NULL"
            " AND start_char_offset IS NULL AND end_char_offset IS NULL)",
            name="transcript_annotations_kind_shape_check",
        ),
        # Bounds: a present half-open range has start >= 0 and end >= 1.
        CheckConstraint(
            "start_word_index IS NULL OR (start_word_index >= 0 AND end_word_index >= 1)",
            name="transcript_annotations_word_bounds_check",
        ),
        CheckConstraint(
            "start_char_offset IS NULL OR (start_char_offset >= 0 AND end_char_offset >= 1)",
            name="transcript_annotations_char_bounds_check",
        ),
        # Same-segment ordering: when both endpoints land in ONE parent segment,
        # the range is a non-empty half-open interval (end > start). Across
        # different segments the two offsets index different segments' texts and
        # carry no ordering relation, so the check is scoped to equal endpoints.
        CheckConstraint(
            "start_word_index IS NULL"
            " OR start_segment_id <> end_segment_id"
            " OR end_word_index > start_word_index",
            name="transcript_annotations_word_same_segment_order_check",
        ),
        CheckConstraint(
            "start_char_offset IS NULL"
            " OR start_segment_id <> end_segment_id"
            " OR end_char_offset > start_char_offset",
            name="transcript_annotations_char_same_segment_order_check",
        ),
        # Span iterates start_segment_index .. end_segment_index inclusive.
        CheckConstraint(
            "end_segment_index >= start_segment_index",
            name="transcript_annotations_segment_index_order_check",
        ),
        # Timing (word_range only stores seconds): paired nullability + order.
        CheckConstraint(
            "(start_seconds IS NULL) = (end_seconds IS NULL)",
            name="transcript_annotations_seconds_pair_check",
        ),
        CheckConstraint(
            "start_seconds IS NULL OR end_seconds >= start_seconds",
            name="transcript_annotations_seconds_order_check",
        ),
        # Timing honesty (docs/annotations.md): precise seconds exist for EXACTLY
        # word_range (word-timing-derived, always derivable there) and are NULL
        # for text_range / segment_range, whose read path labels coarse
        # segment-interval bounds timing_precision="segment". Paired with the
        # seconds_pair_check, this gates both endpoints on the kind.
        CheckConstraint(
            "(anchor_kind = 'word_range') = (start_seconds IS NOT NULL)",
            name="transcript_annotations_seconds_kind_check",
        ),
        CheckConstraint(
            f"char_length(quote_text) <= {MAX_ANNOTATION_QUOTE_CHARS}",
            name="transcript_annotations_quote_len_check",
        ),
        CheckConstraint(
            f"note IS NULL OR char_length(note) <= {MAX_ANNOTATION_NOTE_CHARS}",
            name="transcript_annotations_note_len_check",
        ),
        CheckConstraint(
            f"color_index >= 0 AND color_index <= {HIGHLIGHT_PALETTE_SIZE - 1}",
            name="transcript_annotations_color_index_check",
        ),
        # Run listing in transcript order; skips soft-deleted rows.
        Index(
            "ix_transcript_annotations_run_order",
            "pipeline_run_id",
            "start_segment_index",
            postgresql_where=text("deleted_at IS NULL"),
        ),
        Index("ix_transcript_annotations_start_segment", "start_segment_id"),
        Index("ix_transcript_annotations_end_segment", "end_segment_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    pipeline_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("pipeline_runs.id", ondelete="CASCADE"), index=True
    )
    anchor_schema_version: Mapped[int] = mapped_column(Integer)
    anchor_kind: Mapped[str] = mapped_column(Text)
    # Endpoint parent-segment ids (immutable). CASCADE so re-transcription (new
    # segment ids) or run deletion cannot leak orphaned annotations.
    start_segment_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("transcript_segments.id", ondelete="CASCADE")
    )
    end_segment_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("transcript_segments.id", ondelete="CASCADE")
    )
    # Captured segment positions (stable per run): transcript-order key + span
    # iteration without a join back to transcript_segments.
    start_segment_index: Mapped[int] = mapped_column(Integer)
    end_segment_index: Mapped[int] = mapped_column(Integer)
    # word_range only: half-open parent-word indices at each endpoint.
    start_word_index: Mapped[int | None] = mapped_column(Integer)
    end_word_index: Mapped[int | None] = mapped_column(Integer)
    # text_range only: half-open code-point offsets into each endpoint segment's
    # effective text as it existed at capture (parent coordinates).
    start_char_offset: Mapped[int | None] = mapped_column(Integer)
    end_char_offset: Mapped[int | None] = mapped_column(Integer)
    # sha256 hex over the covered effective texts at capture (see
    # docs/annotations.md). Recomputed at read time; a mismatch = stale.
    source_text_hash: Mapped[str] = mapped_column(Text)
    # Precise only for word_range (word-timing-derived); NULL otherwise, and the
    # read path labels coarse segment-interval bounds timing_precision="segment".
    start_seconds: Mapped[float | None] = mapped_column(Float)
    end_seconds: Mapped[float | None] = mapped_column(Float)
    # Server-derived captured quote (never client text). Preserved verbatim even
    # when the anchor later goes stale, so the panel can show the original.
    quote_text: Mapped[str] = mapped_column(Text)
    color_index: Mapped[int] = mapped_column(Integer)
    note: Mapped[str | None] = mapped_column(Text)
    operator: Mapped[str] = mapped_column(Text)
    # Create idempotency: a replayed nonce with the SAME request_fingerprint
    # returns the original row (including a soft-deleted one); a different
    # fingerprint is a 409 idempotency conflict. NULL for non-create writes.
    idempotency_key: Mapped[str | None] = mapped_column(Text)
    request_fingerprint: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    # NULL = live; a timestamp = soft-deleted (excluded from lists/exports;
    # a nonce replay still returns the deleted row rather than resurrecting it).
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AnnotationTagLink(Base):
    """Many-to-many between annotations and tags (issue #86), capped per
    annotation by the writer (cross-row cap, the splits precedent). Composite
    PK makes a duplicate (annotation, tag) link a structural no-op (writer uses
    ON CONFLICT DO NOTHING). Links cascade with their annotation."""

    __tablename__ = "annotation_tag_links"
    __table_args__ = (Index("ix_annotation_tag_links_tag", "tag_id"),)

    annotation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("transcript_annotations.id", ondelete="CASCADE"),
        primary_key=True,
    )
    tag_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("annotation_tags.id"), primary_key=True)
