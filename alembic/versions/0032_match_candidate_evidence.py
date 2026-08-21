"""match-candidate decision evidence — the speaker-matching measurement loop (#113)

Phase 1 of epic #112. ``speaker_assignments`` persists ACCEPTED proposals only,
so rejected and ineligible near-miss numbers (the cosine/margin/vote-agreement
the matcher already computes) died in debug logs — false *rejects*, the failure
that makes matching feel dead, were invisible. This adds one observational
table recording the matcher's decision for EVERY diarization label:

- ``match_candidates``: one row per (run, label). ``decision`` is
  ``accepted`` / ``rejected`` / ``ineligible``; ``reason`` refines it (which
  eligibility or acceptance gate stopped it). Accepted/rejected labels carry the
  top candidate ``top_speaker_id`` and the numbers (``similarity``, ``margin``,
  ``vote_agreement``); ineligible labels carry none (they never reached a roster
  comparison). ``grounded`` is set only for accepted proposals. ``margin`` is
  NULL with a single-speaker roster (top-1 vs top-2 is undefined).

Purely diagnostic: nothing here feeds the resolver, centroids, or thresholds —
matching behavior is byte-identical with or without the capture. Written
delete-then-insert beside the proposals, idempotent under stage retry. The run
FK cascades on run deletion / re-transcription (disposable evidence, never a
source of truth); the speaker FK is plain (speakers are soft-archived/merged,
never hard-deleted, so it never blocks in practice and refuses a stray hard
delete rather than nulling recorded evidence).

Downgrade drops the table, discarding the captured evidence; the underlying
proposals and diarization turns are untouched.

Revision ID: 0032
Revises: 0031
Create Date: 2026-08-20 22:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0032"
down_revision: str | None = "0031"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "match_candidates",
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
        sa.Column("grounded", sa.Boolean(), nullable=True),
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
            "pipeline_run_id", "diarization_label", name="match_candidates_label_key"
        ),
        sa.CheckConstraint(
            "decision IN ('accepted', 'rejected', 'ineligible')",
            name="match_candidates_decision_check",
        ),
        sa.CheckConstraint(
            "(decision = 'ineligible') = (top_speaker_id IS NULL)",
            name="match_candidates_ineligible_speaker_check",
        ),
        sa.CheckConstraint(
            "(decision = 'ineligible') = (similarity IS NULL)",
            name="match_candidates_ineligible_similarity_check",
        ),
        sa.CheckConstraint(
            "(decision = 'ineligible') = (vote_agreement IS NULL)",
            name="match_candidates_ineligible_vote_check",
        ),
        sa.CheckConstraint(
            "(grounded IS NOT NULL) = (decision = 'accepted')",
            name="match_candidates_grounded_check",
        ),
        sa.CheckConstraint(
            "similarity IS NULL OR (similarity >= -1 AND similarity <= 1)",
            name="match_candidates_similarity_range_check",
        ),
        sa.CheckConstraint(
            "vote_agreement IS NULL OR (vote_agreement >= 0 AND vote_agreement <= 1)",
            name="match_candidates_vote_range_check",
        ),
        sa.CheckConstraint(
            "eligible_turns >= 0 AND eligible_seconds >= 0",
            name="match_candidates_eligibility_nonneg_check",
        ),
        sa.CheckConstraint(
            "roster_size IS NULL OR roster_size >= 0",
            name="match_candidates_roster_size_check",
        ),
    )
    # Denormalized run FK also gets the SQLAlchemy index=True index.
    op.create_index(
        "ix_match_candidates_pipeline_run_id",
        "match_candidates",
        ["pipeline_run_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_match_candidates_pipeline_run_id", table_name="match_candidates")
    op.drop_table("match_candidates")
