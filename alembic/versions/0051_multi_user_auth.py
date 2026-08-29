"""Multi-user authentication.

Revision ID: 0051
Revises: 0050
"""

import sqlalchemy as sa
from alembic import op

revision: str = "0051"
down_revision: str = "0050"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("username", sa.Text(), nullable=False, unique=True),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False, server_default="reviewer"),
        sa.Column("disabled_at", sa.DateTime(timezone=True)),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "role IN ('admin', 'reviewer')",
            name="users_role_check",
        ),
        sa.CheckConstraint(
            "length(btrim(username)) > 0",
            name="users_username_nonempty_check",
        ),
        sa.CheckConstraint(
            "username ~ '^[a-z0-9][a-z0-9_.-]{0,63}$'",
            name="users_username_format_check",
        ),
        sa.CheckConstraint(
            "position(':' in username) = 0",
            name="users_username_no_colon_check",
        ),
    )

    op.create_table(
        "auth_sessions",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column(
            "user_id",
            sa.UUID(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("token_hash", sa.LargeBinary(), nullable=False, unique=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "expires_at > created_at",
            name="auth_sessions_expiry_check",
        ),
    )
    op.create_index(
        "ix_auth_sessions_expires_at",
        "auth_sessions",
        ["expires_at"],
    )

    op.add_column(
        "adjudication_decisions",
        sa.Column(
            "user_id",
            sa.UUID(),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
        ),
    )
    op.create_index(
        "ix_adjudication_decisions_user_id",
        "adjudication_decisions",
        ["user_id"],
    )
    op.create_check_constraint(
        "adjudication_decisions_user_not_system_check",
        "adjudication_decisions",
        "user_id IS NULL OR operator NOT LIKE 'system:%'",
    )
    op.create_check_constraint(
        "adjudication_decisions_system_not_user_check",
        "adjudication_decisions",
        "operator NOT LIKE 'system:%' OR user_id IS NULL",
    )


def downgrade() -> None:
    op.drop_constraint(
        "adjudication_decisions_system_not_user_check",
        "adjudication_decisions",
        type_="check",
    )
    op.drop_constraint(
        "adjudication_decisions_user_not_system_check",
        "adjudication_decisions",
        type_="check",
    )
    op.drop_index(
        "ix_adjudication_decisions_user_id",
        table_name="adjudication_decisions",
    )
    op.drop_column("adjudication_decisions", "user_id")
    op.drop_index("ix_auth_sessions_expires_at", table_name="auth_sessions")
    op.drop_table("auth_sessions")
    op.drop_table("users")
