"""pipeline_runs: record the language whisper detected for each run

Issue #124 (detected language). Adds two nullable columns to ``pipeline_runs``:

- ``detected_language`` — the language whisper actually transcribed the run in,
  as the service reported it (ISO-639-1 style code), stamped by the transcribe
  stage after a successful decode. Voxint's ASR client now requests
  auto-detection, so this records the model's decision, not operator input.
- ``detected_language_probability`` — whisper's language-detection score for
  that language, a probability in [0, 1] (CHECK-enforced). Present only when
  detection actually ran; not a calibrated confidence.

Nullable, no default, no backfill: NULL means a run not yet transcribed or a
legacy run transcribed before this column existed. Reconstructing a language
for old runs from filenames or transcript heuristics would fabricate
provenance, so legacy rows stay NULL. No index on ``detected_language`` — a
deliberate choice (low-cardinality, single-operator deployments), not an
omission.

Revision ID: 0037
Revises: 0036
Create Date: 2026-08-22 20:30:00.000000
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0037"
down_revision: str | None = "0036"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_PROBABILITY_CHECK = "pipeline_runs_detected_language_probability_check"
_PAIRING_CHECK = "pipeline_runs_detected_language_pairing_check"


def upgrade() -> None:
    op.add_column(
        "pipeline_runs",
        sa.Column("detected_language", sa.Text(), nullable=True),
    )
    op.add_column(
        "pipeline_runs",
        sa.Column("detected_language_probability", sa.Float(), nullable=True),
    )
    op.create_check_constraint(
        _PROBABILITY_CHECK,
        "pipeline_runs",
        "detected_language_probability IS NULL"
        " OR (detected_language_probability >= 0"
        " AND detected_language_probability <= 1)",
    )
    # A score describes a detected language: a probability with no language is
    # contradictory provenance (a language with no score is the legitimate
    # forced/fallback shape).
    op.create_check_constraint(
        _PAIRING_CHECK,
        "pipeline_runs",
        "detected_language_probability IS NULL OR detected_language IS NOT NULL",
    )


def downgrade() -> None:
    op.drop_constraint(_PAIRING_CHECK, "pipeline_runs", type_="check")
    op.drop_constraint(_PROBABILITY_CHECK, "pipeline_runs", type_="check")
    op.drop_column("pipeline_runs", "detected_language_probability")
    op.drop_column("pipeline_runs", "detected_language")
