"""Add user_id attribution to profile_review_decisions (#308).

Mirrors the adjudication_decisions user_id column (migration 0051) for the
profile-review audit trail. ON DELETE RESTRICT because this is an append-only
ledger and users are soft-disabled, never hard-deleted.

Revision ID: 0052
Revises: 0051
"""

import sqlalchemy as sa
from alembic import op

revision = "0052"
down_revision = "0051"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "profile_review_decisions",
        sa.Column(
            "user_id",
            sa.Uuid(),
            sa.ForeignKey("users.id", ondelete="RESTRICT"),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_profile_review_decisions_user_id",
        "profile_review_decisions",
        ["user_id"],
    )
    op.create_check_constraint(
        "profile_review_decisions_user_not_system_check",
        "profile_review_decisions",
        "user_id IS NULL OR operator NOT LIKE 'system:%'",
    )


def downgrade() -> None:
    op.drop_constraint(
        "profile_review_decisions_user_not_system_check",
        "profile_review_decisions",
        type_="check",
    )
    op.drop_index(
        "ix_profile_review_decisions_user_id",
        table_name="profile_review_decisions",
    )
    op.drop_column("profile_review_decisions", "user_id")
