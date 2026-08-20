"""pipeline_runs: frozen YAML sidecar snapshot

Issue #104 (watch-folder YAML sidecar metadata). Adds one column to
``pipeline_runs``:

- ``sidecar`` (JSONB, nullable) — the media file's companion YAML sidecar as a
  JSON-normalized mapping, stamped write-once at submit. The machine-read keys
  (``title``/``speakers``/``domain_pack``/``notes``) are APPLIED at submit
  (speakers union into the frozen ``domain_pack`` snapshot's ``name_seeds``,
  notes into ``operator_notes``); the column preserves the whole mapping —
  including keys Voxint does not recognize — for provenance and for the
  console's title display. NULL = no sidecar existed when the media was picked
  up; a sidecar arriving later is deliberately too late (frozen-at-submit, the
  #84 posture). The check constraint pins the stored value to a JSON object so
  a tampered scalar/array can never reach the tolerant readers.

Revision ID: 0030
Revises: 0029
Create Date: 2026-08-19 22:30:00.000000
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0030"
down_revision: str | None = "0029"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "pipeline_runs",
        sa.Column("sidecar", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.create_check_constraint(
        "pipeline_runs_sidecar_object_check",
        "pipeline_runs",
        "sidecar IS NULL OR jsonb_typeof(sidecar) = 'object'",
    )


def downgrade() -> None:
    op.drop_constraint(
        "pipeline_runs_sidecar_object_check", "pipeline_runs", type_="check"
    )
    op.drop_column("pipeline_runs", "sidecar")
