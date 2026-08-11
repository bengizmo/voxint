"""diarization_turns observation ledger + transcript_segments.suspect

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-11
"""
from collections.abc import Sequence

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op

revision: str = '0002'
down_revision: str | None = '0001'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        'transcript_segments',
        sa.Column('suspect', sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.create_table(
        'diarization_turns',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('pipeline_run_id', sa.Uuid(), nullable=False),
        sa.Column('turn_index', sa.Integer(), nullable=False),
        sa.Column('start_seconds', sa.Float(), nullable=False),
        sa.Column('end_seconds', sa.Float(), nullable=False),
        sa.Column('label', sa.Text(), nullable=False),
        sa.Column('overlap', sa.Boolean(), nullable=False),
        sa.Column('overlap_seconds', sa.Float(), nullable=False),
        sa.Column('snr_db', sa.Float(), nullable=True),
        sa.Column('skip_reason', sa.Text(), nullable=True),
        sa.Column('embedding', pgvector.sqlalchemy.Vector(dim=192), nullable=True),
        sa.Column('embedding_space', sa.Text(), nullable=True),
        sa.CheckConstraint('turn_index >= 0', name='diarization_turns_index_nonneg_check'),
        sa.CheckConstraint(
            'start_seconds >= 0 AND end_seconds > start_seconds',
            name='diarization_turns_interval_check',
        ),
        sa.CheckConstraint('overlap_seconds >= 0', name='diarization_turns_overlap_nonneg_check'),
        sa.CheckConstraint(
            '(embedding IS NULL) != (skip_reason IS NULL)',
            name='diarization_turns_embedding_xor_skip_check',
        ),
        sa.CheckConstraint(
            'embedding IS NULL OR embedding_space IS NOT NULL',
            name='diarization_turns_embedding_space_check',
        ),
        sa.ForeignKeyConstraint(['pipeline_run_id'], ['pipeline_runs.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('pipeline_run_id', 'turn_index', name='diarization_turns_index_key'),
    )
    op.create_index(
        op.f('ix_diarization_turns_pipeline_run_id'),
        'diarization_turns',
        ['pipeline_run_id'],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f('ix_diarization_turns_pipeline_run_id'), table_name='diarization_turns')
    op.drop_table('diarization_turns')
    op.drop_column('transcript_segments', 'suspect')
