"""ACQUIRE pipeline stage + media_items.source_url provenance

Revision ID: 0005
Revises: 0004
Create Date: 2026-08-12

Prepends ``acquire`` to the pipeline stage vocabulary and records the origin URL
for media fetched by the ACQUIRE stage. Both stage CHECK constraints
(``pipeline_runs_current_stage_check``, ``stage_runs_stage_check``) must admit the
new value — Postgres has no ALTER on a CHECK expression, so each is dropped and
recreated. The downgrade REFUSES once any run has entered ``acquire`` rather than
delete operational history.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = '0005'
down_revision: str | None = '0004'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Stage vocabularies as literal SQL fragments — the migration owns its own copy
# rather than importing the model enum, so a later enum edit can never silently
# rewrite what this revision applied.
_STAGES_WITH_ACQUIRE = (
    "'acquire', 'prepare', 'transcribe', 'diarize_embed', 'enhance_match', 'finalize'"
)
_STAGES_ORIGINAL = "'prepare', 'transcribe', 'diarize_embed', 'enhance_match', 'finalize'"


def _swap_stage_constraints(stage_values: str) -> None:
    op.drop_constraint(
        'pipeline_runs_current_stage_check', 'pipeline_runs', type_='check'
    )
    op.create_check_constraint(
        'pipeline_runs_current_stage_check',
        'pipeline_runs',
        f'current_stage IS NULL OR current_stage IN ({stage_values})',
    )
    op.drop_constraint('stage_runs_stage_check', 'stage_runs', type_='check')
    op.create_check_constraint(
        'stage_runs_stage_check',
        'stage_runs',
        f'stage IN ({stage_values})',
    )


def upgrade() -> None:
    op.add_column('media_items', sa.Column('source_url', sa.Text(), nullable=True))
    _swap_stage_constraints(_STAGES_WITH_ACQUIRE)


def downgrade() -> None:
    # Refuse to erase history: if any run ever entered ACQUIRE, narrowing the
    # constraints would either fail against the surviving rows or orphan real
    # ledger entries. Operators must resolve those runs before downgrading.
    acquire_present = op.get_bind().execute(
        sa.text(
            "SELECT EXISTS ("
            " SELECT 1 FROM stage_runs WHERE stage = 'acquire'"
            " UNION ALL"
            " SELECT 1 FROM pipeline_runs WHERE current_stage = 'acquire'"
            ")"
        )
    ).scalar()
    if acquire_present:
        raise RuntimeError(
            "refusing to downgrade 0005: acquire stage rows exist; downgrading "
            "would delete operational history — resolve those runs first"
        )
    _swap_stage_constraints(_STAGES_ORIGINAL)
    # The guard above protects operational *history* (runs that entered ACQUIRE).
    # source_url on a row whose run never acquired — e.g. a still-QUEUED URL run —
    # is dropped here without warning; that provenance loss is in scope for a
    # downgrade of this revision (the column itself is what's being removed).
    op.drop_column('media_items', 'source_url')
