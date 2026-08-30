"""Add corpus analysis artifacts table.

Revision ID: 0049
Revises: 0048
Create Date: 2026-08-30
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0049"
down_revision: str = "0048"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "corpus_analysis_artifacts",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("scope_kind", sa.Text(), nullable=False),
        sa.Column("scope_id", sa.Uuid(), nullable=True),
        sa.Column("artifact_kind", sa.Text(), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("source_hash", sa.Text(), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "scope_kind IN ('corpus', 'project', 'speaker')",
            name="corpus_analysis_artifacts_scope_kind_check",
        ),
        sa.CheckConstraint(
            "generation >= 1",
            name="corpus_analysis_artifacts_generation_check",
        ),
        sa.CheckConstraint(
            "jsonb_typeof(payload) = 'object'",
            name="corpus_analysis_artifacts_payload_object_check",
        ),
        sa.CheckConstraint(
            "source_hash ~ '^[0-9a-f]{64}$'",
            name="corpus_analysis_artifacts_source_hash_check",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "scope_kind",
            "scope_id",
            "artifact_kind",
            "generation",
            name="corpus_analysis_artifacts_generation_key",
        ),
    )


def downgrade() -> None:
    op.drop_table("corpus_analysis_artifacts")
