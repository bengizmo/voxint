"""Widen users_role_check to allow the 'viewer' role (#363).

Revision ID: 0057
Revises: 0056
Create Date: 2026-08-31
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0057"
down_revision: str = "0056"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint("users_role_check", "users", type_="check")
    op.create_check_constraint(
        "users_role_check",
        "users",
        "role IN ('admin', 'reviewer', 'viewer')",
    )


def downgrade() -> None:
    # Precondition: no user rows with role='viewer' may exist; reassign them
    # first (e.g. ``voxint user set-role <name> reviewer``) or the narrowed
    # CHECK will reject existing data and the transaction will roll back.
    op.drop_constraint("users_role_check", "users", type_="check")
    op.create_check_constraint(
        "users_role_check",
        "users",
        "role IN ('admin', 'reviewer')",
    )
