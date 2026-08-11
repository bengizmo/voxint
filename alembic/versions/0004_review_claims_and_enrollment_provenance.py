"""pipeline_runs reviewer claim columns + speaker_embeddings enrollment provenance

Revision ID: 0004
Revises: 0003
Create Date: 2026-08-11
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = '0004'
down_revision: str | None = '0003'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Reviewer slot: one adjudication claim per run, guarded by the run's CAS
    # revision. All four columns move together (all NULL = unclaimed).
    op.add_column('pipeline_runs', sa.Column('review_claim_token', sa.Uuid(), nullable=True))
    op.add_column('pipeline_runs', sa.Column('review_claimed_by', sa.Text(), nullable=True))
    op.add_column(
        'pipeline_runs',
        sa.Column('review_claimed_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        'pipeline_runs',
        sa.Column('review_claim_expires_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.create_check_constraint(
        'pipeline_runs_review_claim_shape_check',
        'pipeline_runs',
        '(review_claim_token IS NULL) = (review_claimed_by IS NULL)'
        ' AND (review_claim_token IS NULL) = (review_claimed_at IS NULL)'
        ' AND (review_claim_token IS NULL) = (review_claim_expires_at IS NULL)',
    )

    # Enrollment provenance: which local label and which human ruling produced
    # this centroid. Unique on the decision id so a replayed enrollment POST
    # can never mint a second embedding row (NULLs — pre-P5 rows — don't collide).
    op.add_column(
        'speaker_embeddings', sa.Column('source_diarization_label', sa.Text(), nullable=True)
    )
    op.add_column(
        'speaker_embeddings',
        sa.Column('source_adjudication_decision_id', sa.Uuid(), nullable=True),
    )
    op.create_foreign_key(
        'speaker_embeddings_source_adjudication_decision_id_fkey',
        'speaker_embeddings',
        'adjudication_decisions',
        ['source_adjudication_decision_id'],
        ['id'],
    )
    op.create_unique_constraint(
        'speaker_embeddings_source_decision_key',
        'speaker_embeddings',
        ['source_adjudication_decision_id'],
    )


def downgrade() -> None:
    op.drop_constraint(
        'speaker_embeddings_source_decision_key', 'speaker_embeddings', type_='unique'
    )
    op.drop_constraint(
        'speaker_embeddings_source_adjudication_decision_id_fkey',
        'speaker_embeddings',
        type_='foreignkey',
    )
    op.drop_column('speaker_embeddings', 'source_adjudication_decision_id')
    op.drop_column('speaker_embeddings', 'source_diarization_label')
    op.drop_constraint(
        'pipeline_runs_review_claim_shape_check', 'pipeline_runs', type_='check'
    )
    op.drop_column('pipeline_runs', 'review_claim_expires_at')
    op.drop_column('pipeline_runs', 'review_claimed_at')
    op.drop_column('pipeline_runs', 'review_claimed_by')
    op.drop_column('pipeline_runs', 'review_claim_token')
