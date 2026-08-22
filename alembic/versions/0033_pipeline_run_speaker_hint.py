"""pipeline_runs: per-recording diarization speaker-count hint

Issue #128 (operator-supplied speaker count). Adds two nullable integer columns
to ``pipeline_runs``:

- ``diarization_max_speakers`` — an upper bound on the number of distinct
  speakers the diarizer may return, frozen at submit from a ``voxint submit``
  flag or the YAML sidecar. NULL means the worker falls back to the install-wide
  default (``settings.diarization_max_speakers``) at execution; a non-NULL value
  is an explicit per-run override.
- ``diarization_num_speakers`` — an EXACT count. When set, the client pins
  pyannote to that many speakers (sent as ``min_speakers == max_speakers``) and
  it takes precedence over the bound.

Both are bounded 1..20 by a check constraint, mirroring the pyannote service
request model and the ``diarization_max_speakers`` config field, so a bad hint
can never reach the diarizer through these columns. Kept as typed scalar columns
rather than folded into the frozen ``domain_pack`` manifest, which is pack
provenance, not a run-options bag.

Revision ID: 0033
Revises: 0032
Create Date: 2026-08-21 23:30:00.000000
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0033"
down_revision: str | None = "0032"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "pipeline_runs",
        sa.Column("diarization_max_speakers", sa.Integer(), nullable=True),
    )
    op.add_column(
        "pipeline_runs",
        sa.Column("diarization_num_speakers", sa.Integer(), nullable=True),
    )
    op.create_check_constraint(
        "pipeline_runs_diarization_max_speakers_check",
        "pipeline_runs",
        "diarization_max_speakers IS NULL"
        " OR (diarization_max_speakers >= 1 AND diarization_max_speakers <= 20)",
    )
    op.create_check_constraint(
        "pipeline_runs_diarization_num_speakers_check",
        "pipeline_runs",
        "diarization_num_speakers IS NULL"
        " OR (diarization_num_speakers >= 1 AND diarization_num_speakers <= 20)",
    )


def downgrade() -> None:
    op.drop_constraint(
        "pipeline_runs_diarization_num_speakers_check", "pipeline_runs", type_="check"
    )
    op.drop_constraint(
        "pipeline_runs_diarization_max_speakers_check", "pipeline_runs", type_="check"
    )
    op.drop_column("pipeline_runs", "diarization_num_speakers")
    op.drop_column("pipeline_runs", "diarization_max_speakers")
