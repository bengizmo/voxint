"""Add auto-enroll decision evidence for threshold calibration (#434).

Parallel to ``match_candidates``, ``auto_enroll_evidence`` records the final
auto-enroll outcome and the comparison diagnostics for every label processed by
auto-enrollment. It is purely diagnostic and written with a per-label upsert.
The run foreign key cascades on deletion; the speaker foreign key preserves the
historical candidate reference.

Revision ID: 0058
Revises: 0057
Create Date: 2026-09-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0058"
down_revision: str | None = "0057"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "auto_enroll_evidence",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "pipeline_run_id",
            sa.Uuid(),
            sa.ForeignKey("pipeline_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("diarization_label", sa.Text(), nullable=False),
        sa.Column("decision", sa.Text(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("embedding_space", sa.Text(), nullable=True),
        sa.Column(
            "top_speaker_id",
            sa.Uuid(),
            sa.ForeignKey("speakers.id"),
            nullable=True,
        ),
        sa.Column("similarity", sa.Float(), nullable=True),
        sa.Column("margin", sa.Float(), nullable=True),
        sa.Column("vote_agreement", sa.Float(), nullable=True),
        sa.Column("eligible_turns", sa.Integer(), nullable=False),
        sa.Column("eligible_seconds", sa.Float(), nullable=False),
        sa.Column("roster_size", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "pipeline_run_id",
            "diarization_label",
            name="auto_enroll_evidence_label_key",
        ),
        sa.CheckConstraint(
            "decision IN ('linked', 'created', 'skipped')",
            name="auto_enroll_evidence_decision_check",
        ),
        sa.CheckConstraint(
            "(top_speaker_id IS NULL) = (similarity IS NULL)",
            name="auto_enroll_evidence_comparison_shape_check",
        ),
        sa.CheckConstraint(
            "decision != 'linked' OR top_speaker_id IS NOT NULL",
            name="auto_enroll_evidence_linked_speaker_check",
        ),
        sa.CheckConstraint(
            "similarity IS NULL OR (similarity >= -1 AND similarity <= 1)",
            name="auto_enroll_evidence_similarity_range_check",
        ),
        sa.CheckConstraint(
            "vote_agreement IS NULL OR (vote_agreement >= 0 AND vote_agreement <= 1)",
            name="auto_enroll_evidence_vote_range_check",
        ),
        sa.CheckConstraint(
            "eligible_turns >= 0 AND eligible_seconds >= 0",
            name="auto_enroll_evidence_eligibility_nonneg_check",
        ),
        sa.CheckConstraint(
            "roster_size IS NULL OR roster_size >= 0",
            name="auto_enroll_evidence_roster_size_check",
        ),
    )
    op.create_index(
        "ix_auto_enroll_evidence_pipeline_run_id",
        "auto_enroll_evidence",
        ["pipeline_run_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_auto_enroll_evidence_pipeline_run_id",
        table_name="auto_enroll_evidence",
    )
    op.drop_table("auto_enroll_evidence")
