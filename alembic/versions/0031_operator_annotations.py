"""operator annotation layer — highlights + tags + notes (issue #86)

The review console gains an operator OUTPUT layer: a selected transcript span
becomes an annotation (a highlight color, flat tags, an optional margin note).
Three new tables, all mutable-workspace (edited, re-anchored, soft-deleted),
none of which ever mutate pipeline evidence. See ``docs/annotations.md`` — the
frozen anchor contract this schema must match.

- ``annotation_tags`` (global, flat): verbatim ``name`` for display plus a
  writer-computed ``name_normalized`` (trim+casefold) carrying the UNIQUE
  constraint, a palette ``color`` (0..5), and ``archived_at`` (hide from
  pickers without deleting; no hard delete in v1).
- ``transcript_annotations``: anchors address the IMMUTABLE parent
  ``transcript_segments`` id in one of three kinds (``word_range`` /
  ``text_range`` / ``segment_range``), gated by a per-kind shape CHECK truth
  table. ``source_text_hash`` (sha256 hex, stored for every kind) drives
  read-time staleness; ``start_seconds``/``end_seconds`` are precise ONLY for
  ``word_range`` (NULL otherwise — timing honesty). ``pipeline_run_id`` and the
  ``*_segment_index`` copies are denormalized by the sole writer so the run
  listing loads in transcript order without a join. ``idempotency_key`` +
  ``request_fingerprint`` give create-replay semantics. FKs to segments and the
  run are ON DELETE CASCADE so re-transcription / run deletion leaves no orphan.
- ``annotation_tag_links``: composite-key many-to-many, cascading with the
  annotation, capped per annotation by the writer.

Downgrade drops all three tables, discarding operator annotations (the
underlying transcript evidence is untouched). Unlike a scope-narrowing column,
this is a clean full-feature removal, not silent corruption, so no data-loss
guard is warranted.

Revision ID: 0031
Revises: 0030
Create Date: 2026-08-20 08:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0031"
down_revision: str | None = "0030"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Mirrors voxint.db.models; kept as literals so the migration is self-contained.
_HIGHLIGHT_PALETTE_MAX = 5  # HIGHLIGHT_PALETTE_SIZE - 1
_MAX_TAG_NAME_CHARS = 64
_MAX_QUOTE_CHARS = 50_000
_MAX_NOTE_CHARS = 4000


def upgrade() -> None:
    op.create_table(
        "annotation_tags",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("name_normalized", sa.Text(), nullable=False),
        sa.Column("color", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("name_normalized", name="annotation_tags_name_normalized_key"),
        sa.CheckConstraint(
            "char_length(trim(name)) > 0", name="annotation_tags_name_nonempty_check"
        ),
        sa.CheckConstraint(
            f"char_length(name) <= {_MAX_TAG_NAME_CHARS}",
            name="annotation_tags_name_len_check",
        ),
        sa.CheckConstraint(
            f"color >= 0 AND color <= {_HIGHLIGHT_PALETTE_MAX}",
            name="annotation_tags_color_range_check",
        ),
    )

    op.create_table(
        "transcript_annotations",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "pipeline_run_id",
            sa.Uuid(),
            sa.ForeignKey("pipeline_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("anchor_schema_version", sa.Integer(), nullable=False),
        sa.Column("anchor_kind", sa.Text(), nullable=False),
        sa.Column(
            "start_segment_id",
            sa.Uuid(),
            sa.ForeignKey("transcript_segments.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "end_segment_id",
            sa.Uuid(),
            sa.ForeignKey("transcript_segments.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("start_segment_index", sa.Integer(), nullable=False),
        sa.Column("end_segment_index", sa.Integer(), nullable=False),
        sa.Column("start_word_index", sa.Integer(), nullable=True),
        sa.Column("end_word_index", sa.Integer(), nullable=True),
        sa.Column("start_char_offset", sa.Integer(), nullable=True),
        sa.Column("end_char_offset", sa.Integer(), nullable=True),
        sa.Column("source_text_hash", sa.Text(), nullable=False),
        sa.Column("start_seconds", sa.Float(), nullable=True),
        sa.Column("end_seconds", sa.Float(), nullable=True),
        sa.Column("quote_text", sa.Text(), nullable=False),
        sa.Column("color_index", sa.Integer(), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("operator", sa.Text(), nullable=False),
        sa.Column("idempotency_key", sa.Text(), nullable=True),
        sa.Column("request_fingerprint", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("idempotency_key", name="transcript_annotations_idempotency_key"),
        sa.CheckConstraint(
            "anchor_schema_version = 1",
            name="transcript_annotations_schema_version_check",
        ),
        sa.CheckConstraint(
            "anchor_kind IN ('word_range', 'text_range', 'segment_range')",
            name="transcript_annotations_anchor_kind_check",
        ),
        sa.CheckConstraint(
            "source_text_hash ~ '^[0-9a-f]{64}$'",
            name="transcript_annotations_source_hash_hex_check",
        ),
        sa.CheckConstraint(
            "(start_word_index IS NULL) = (end_word_index IS NULL)",
            name="transcript_annotations_word_pair_check",
        ),
        sa.CheckConstraint(
            "(start_char_offset IS NULL) = (end_char_offset IS NULL)",
            name="transcript_annotations_char_pair_check",
        ),
        sa.CheckConstraint(
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
        sa.CheckConstraint(
            "start_word_index IS NULL OR (start_word_index >= 0 AND end_word_index >= 1)",
            name="transcript_annotations_word_bounds_check",
        ),
        sa.CheckConstraint(
            "start_char_offset IS NULL OR (start_char_offset >= 0 AND end_char_offset >= 1)",
            name="transcript_annotations_char_bounds_check",
        ),
        sa.CheckConstraint(
            "start_word_index IS NULL"
            " OR start_segment_id <> end_segment_id"
            " OR end_word_index > start_word_index",
            name="transcript_annotations_word_same_segment_order_check",
        ),
        sa.CheckConstraint(
            "start_char_offset IS NULL"
            " OR start_segment_id <> end_segment_id"
            " OR end_char_offset > start_char_offset",
            name="transcript_annotations_char_same_segment_order_check",
        ),
        sa.CheckConstraint(
            "end_segment_index >= start_segment_index",
            name="transcript_annotations_segment_index_order_check",
        ),
        sa.CheckConstraint(
            "(start_seconds IS NULL) = (end_seconds IS NULL)",
            name="transcript_annotations_seconds_pair_check",
        ),
        sa.CheckConstraint(
            "start_seconds IS NULL OR end_seconds >= start_seconds",
            name="transcript_annotations_seconds_order_check",
        ),
        # Precise seconds exist for EXACTLY word_range; NULL for the other kinds
        # (timing honesty, docs/annotations.md).
        sa.CheckConstraint(
            "(anchor_kind = 'word_range') = (start_seconds IS NOT NULL)",
            name="transcript_annotations_seconds_kind_check",
        ),
        sa.CheckConstraint(
            f"char_length(quote_text) <= {_MAX_QUOTE_CHARS}",
            name="transcript_annotations_quote_len_check",
        ),
        sa.CheckConstraint(
            f"note IS NULL OR char_length(note) <= {_MAX_NOTE_CHARS}",
            name="transcript_annotations_note_len_check",
        ),
        sa.CheckConstraint(
            f"color_index >= 0 AND color_index <= {_HIGHLIGHT_PALETTE_MAX}",
            name="transcript_annotations_color_index_check",
        ),
    )
    # Denormalized run FK also gets the SQLAlchemy index=True index.
    op.create_index(
        "ix_transcript_annotations_pipeline_run_id",
        "transcript_annotations",
        ["pipeline_run_id"],
    )
    # Run listing in transcript order, skipping soft-deleted rows.
    op.create_index(
        "ix_transcript_annotations_run_order",
        "transcript_annotations",
        ["pipeline_run_id", "start_segment_index"],
        postgresql_where=sa.text("deleted_at IS NULL"),
    )
    op.create_index(
        "ix_transcript_annotations_start_segment",
        "transcript_annotations",
        ["start_segment_id"],
    )
    op.create_index(
        "ix_transcript_annotations_end_segment",
        "transcript_annotations",
        ["end_segment_id"],
    )

    op.create_table(
        "annotation_tag_links",
        sa.Column(
            "annotation_id",
            sa.Uuid(),
            sa.ForeignKey("transcript_annotations.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "tag_id",
            sa.Uuid(),
            sa.ForeignKey("annotation_tags.id"),
            primary_key=True,
        ),
    )
    op.create_index("ix_annotation_tag_links_tag", "annotation_tag_links", ["tag_id"])


def downgrade() -> None:
    op.drop_index("ix_annotation_tag_links_tag", table_name="annotation_tag_links")
    op.drop_table("annotation_tag_links")
    op.drop_index("ix_transcript_annotations_end_segment", table_name="transcript_annotations")
    op.drop_index("ix_transcript_annotations_start_segment", table_name="transcript_annotations")
    op.drop_index("ix_transcript_annotations_run_order", table_name="transcript_annotations")
    op.drop_index(
        "ix_transcript_annotations_pipeline_run_id",
        table_name="transcript_annotations",
    )
    op.drop_table("transcript_annotations")
    op.drop_table("annotation_tags")
