"""baseline schema

Revision ID: 0001
Revises: 
Create Date: 2026-08-11 15:59:17.054213
"""
from collections.abc import Sequence

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = '0001'
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table('media_items',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('source_path', sa.Text(), nullable=False),
    sa.Column('media_type', sa.Text(), nullable=True),
    sa.Column('duration_seconds', sa.Float(), nullable=True),
    sa.Column('size_bytes', sa.BigInteger(), nullable=True),
    sa.Column('sha256', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint('duration_seconds IS NULL OR duration_seconds >= 0', name='media_items_duration_nonneg_check'),
    sa.CheckConstraint('size_bytes IS NULL OR size_bytes >= 0', name='media_items_size_nonneg_check'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('source_path')
    )
    op.create_table('speakers',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('display_name', sa.Text(), nullable=False),
    sa.Column('notes', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('display_name')
    )
    op.create_table('pipeline_runs',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('media_item_id', sa.Uuid(), nullable=False),
    sa.Column('status', sa.Text(), nullable=False),
    sa.Column('current_stage', sa.Text(), nullable=True),
    sa.Column('revision', sa.Integer(), nullable=False),
    sa.Column('error', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("current_stage IS NULL OR current_stage IN ('prepare', 'transcribe', 'diarize_embed', 'enhance_match', 'finalize')", name='pipeline_runs_current_stage_check'),
    sa.CheckConstraint("status IN ('queued', 'running', 'awaiting_adjudication', 'completed', 'failed', 'cancelled')", name='pipeline_runs_status_check'),
    sa.CheckConstraint('revision >= 0', name='pipeline_runs_revision_nonneg_check'),
    sa.ForeignKeyConstraint(['media_item_id'], ['media_items.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_pipeline_runs_media_item_id'), 'pipeline_runs', ['media_item_id'], unique=False)
    op.create_index(op.f('ix_pipeline_runs_status'), 'pipeline_runs', ['status'], unique=False)
    op.create_table('adjudication_decisions',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('pipeline_run_id', sa.Uuid(), nullable=False),
    sa.Column('diarization_label', sa.Text(), nullable=False),
    sa.Column('decision', sa.Text(), nullable=False),
    sa.Column('speaker_id', sa.Uuid(), nullable=True),
    sa.Column('operator', sa.Text(), nullable=False),
    sa.Column('idempotency_key', sa.Text(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("(decision = 'assign') = (speaker_id IS NOT NULL)", name='adjudication_decisions_assign_speaker_check'),
    sa.CheckConstraint("decision IN ('assign', 'exclude', 'unknown')", name='adjudication_decisions_check'),
    sa.ForeignKeyConstraint(['pipeline_run_id'], ['pipeline_runs.id'], ),
    sa.ForeignKeyConstraint(['speaker_id'], ['speakers.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('idempotency_key', name='adjudication_decisions_idempotency_key')
    )
    op.create_index(op.f('ix_adjudication_decisions_pipeline_run_id'), 'adjudication_decisions', ['pipeline_run_id'], unique=False)
    op.create_table('audio_artifacts',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('pipeline_run_id', sa.Uuid(), nullable=False),
    sa.Column('kind', sa.Text(), nullable=False),
    sa.Column('path', sa.Text(), nullable=False),
    sa.Column('meta', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("kind IN ('preprocessed_audio', 'chunk', 'transcript_export')", name='audio_artifacts_kind_check'),
    sa.ForeignKeyConstraint(['pipeline_run_id'], ['pipeline_runs.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_audio_artifacts_pipeline_run_id'), 'audio_artifacts', ['pipeline_run_id'], unique=False)
    op.create_table('audio_chunks',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('pipeline_run_id', sa.Uuid(), nullable=False),
    sa.Column('chunk_index', sa.Integer(), nullable=False),
    sa.Column('start_seconds', sa.Float(), nullable=False),
    sa.Column('end_seconds', sa.Float(), nullable=False),
    sa.Column('path', sa.Text(), nullable=False),
    sa.CheckConstraint('chunk_index >= 0', name='audio_chunks_index_nonneg_check'),
    sa.CheckConstraint('start_seconds >= 0 AND end_seconds > start_seconds', name='audio_chunks_interval_check'),
    sa.ForeignKeyConstraint(['pipeline_run_id'], ['pipeline_runs.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('pipeline_run_id', 'chunk_index', name='audio_chunks_index_key')
    )
    op.create_index(op.f('ix_audio_chunks_pipeline_run_id'), 'audio_chunks', ['pipeline_run_id'], unique=False)
    op.create_table('speaker_assignments',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('pipeline_run_id', sa.Uuid(), nullable=False),
    sa.Column('diarization_label', sa.Text(), nullable=False),
    sa.Column('speaker_id', sa.Uuid(), nullable=True),
    sa.Column('method', sa.Text(), nullable=False),
    sa.Column('confidence', sa.Float(), nullable=True),
    sa.Column('grounded', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("NOT grounded OR (speaker_id IS NOT NULL AND method = 'cosine')", name='speaker_assignments_grounded_check'),
    sa.CheckConstraint("method IN ('cosine', 'llm_hint')", name='speaker_assignments_method_check'),
    sa.CheckConstraint('confidence IS NULL OR (confidence >= 0 AND confidence <= 1)', name='speaker_assignments_confidence_check'),
    sa.ForeignKeyConstraint(['pipeline_run_id'], ['pipeline_runs.id'], ),
    sa.ForeignKeyConstraint(['speaker_id'], ['speakers.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('pipeline_run_id', 'diarization_label', 'method', name='speaker_assignments_proposal_key')
    )
    op.create_index(op.f('ix_speaker_assignments_pipeline_run_id'), 'speaker_assignments', ['pipeline_run_id'], unique=False)
    op.create_table('speaker_embeddings',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('speaker_id', sa.Uuid(), nullable=False),
    sa.Column('embedding_space', sa.Text(), nullable=False),
    sa.Column('embedding', pgvector.sqlalchemy.vector.VECTOR(dim=192), nullable=False),
    sa.Column('source_pipeline_run_id', sa.Uuid(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint('length(embedding_space) > 0', name='speaker_embeddings_space_nonempty_check'),
    sa.ForeignKeyConstraint(['source_pipeline_run_id'], ['pipeline_runs.id'], ),
    sa.ForeignKeyConstraint(['speaker_id'], ['speakers.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_speaker_embeddings_embedding_space'), 'speaker_embeddings', ['embedding_space'], unique=False)
    op.create_index(op.f('ix_speaker_embeddings_speaker_id'), 'speaker_embeddings', ['speaker_id'], unique=False)
    op.create_table('stage_runs',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('pipeline_run_id', sa.Uuid(), nullable=False),
    sa.Column('stage', sa.Text(), nullable=False),
    sa.Column('status', sa.Text(), nullable=False),
    sa.Column('attempt', sa.Integer(), nullable=False),
    sa.Column('worker_id', sa.Text(), nullable=True),
    sa.Column('lease_expires_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('error', sa.Text(), nullable=True),
    sa.Column('metrics', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=True),
    sa.CheckConstraint("stage IN ('prepare', 'transcribe', 'diarize_embed', 'enhance_match', 'finalize')", name='stage_runs_stage_check'),
    sa.CheckConstraint("status IN ('running', 'completed', 'failed', 'skipped')", name='stage_runs_status_check'),
    sa.CheckConstraint('attempt >= 1', name='stage_runs_attempt_positive_check'),
    sa.ForeignKeyConstraint(['pipeline_run_id'], ['pipeline_runs.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('pipeline_run_id', 'stage', 'attempt', name='stage_runs_attempt_key')
    )
    op.create_index(op.f('ix_stage_runs_pipeline_run_id'), 'stage_runs', ['pipeline_run_id'], unique=False)
    op.create_table('transcript_segments',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('pipeline_run_id', sa.Uuid(), nullable=False),
    sa.Column('segment_index', sa.Integer(), nullable=False),
    sa.Column('start_seconds', sa.Float(), nullable=False),
    sa.Column('end_seconds', sa.Float(), nullable=False),
    sa.Column('raw_text', sa.Text(), nullable=False),
    sa.Column('enhanced_text', sa.Text(), nullable=True),
    sa.Column('diarization_label', sa.Text(), nullable=True),
    sa.CheckConstraint('segment_index >= 0', name='transcript_segments_index_nonneg_check'),
    sa.CheckConstraint('start_seconds >= 0 AND end_seconds >= start_seconds', name='transcript_segments_interval_check'),
    sa.ForeignKeyConstraint(['pipeline_run_id'], ['pipeline_runs.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('pipeline_run_id', 'segment_index', name='transcript_segments_index_key')
    )
    op.create_index(op.f('ix_transcript_segments_pipeline_run_id'), 'transcript_segments', ['pipeline_run_id'], unique=False)
    # adjudication_decisions is append-only at the persistence boundary, not just by convention
    op.execute("""
        CREATE FUNCTION adjudication_decisions_append_only() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION 'adjudication_decisions is append-only (% blocked)', TG_OP;
        END $$
    """)
    op.execute("""
        CREATE TRIGGER adjudication_decisions_append_only_trigger
        BEFORE UPDATE OR DELETE ON adjudication_decisions
        FOR EACH ROW EXECUTE FUNCTION adjudication_decisions_append_only()
    """)


def downgrade() -> None:
    op.execute(
        "DROP TRIGGER IF EXISTS adjudication_decisions_append_only_trigger"
        " ON adjudication_decisions"
    )
    op.execute("DROP FUNCTION IF EXISTS adjudication_decisions_append_only()")
    op.drop_index(op.f('ix_transcript_segments_pipeline_run_id'), table_name='transcript_segments')
    op.drop_table('transcript_segments')
    op.drop_index(op.f('ix_stage_runs_pipeline_run_id'), table_name='stage_runs')
    op.drop_table('stage_runs')
    op.drop_index(op.f('ix_speaker_embeddings_speaker_id'), table_name='speaker_embeddings')
    op.drop_index(op.f('ix_speaker_embeddings_embedding_space'), table_name='speaker_embeddings')
    op.drop_table('speaker_embeddings')
    op.drop_index(op.f('ix_speaker_assignments_pipeline_run_id'), table_name='speaker_assignments')
    op.drop_table('speaker_assignments')
    op.drop_index(op.f('ix_audio_chunks_pipeline_run_id'), table_name='audio_chunks')
    op.drop_table('audio_chunks')
    op.drop_index(op.f('ix_audio_artifacts_pipeline_run_id'), table_name='audio_artifacts')
    op.drop_table('audio_artifacts')
    op.drop_index(op.f('ix_adjudication_decisions_pipeline_run_id'), table_name='adjudication_decisions')
    op.drop_table('adjudication_decisions')
    op.drop_index(op.f('ix_pipeline_runs_status'), table_name='pipeline_runs')
    op.drop_index(op.f('ix_pipeline_runs_media_item_id'), table_name='pipeline_runs')
    op.drop_table('pipeline_runs')
    op.drop_table('speakers')
    op.drop_table('media_items')
    # ### end Alembic commands ###
