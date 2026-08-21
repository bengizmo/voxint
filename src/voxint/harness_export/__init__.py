"""DB -> scoring-harness input exporter (issue #113, step 4).

The scoring harness (``voxint.harness``) is DB-free by contract: it scores
plain JSON/JSONL files. This package is the one piece allowed to read the
database and render live pipeline state into the shapes those scorers consume
(``docs/harness.md``). It lives OUTSIDE ``voxint.harness`` precisely so that
package keeps its no-database, no-settings guarantee.

Two families are produced, both from STORED evidence (never re-running TitaNet
or the matcher decision):

- ``name_accuracy_items`` -> the ``score name-accuracy`` items contract, one
  item per run, one slot per diarization label. ``assigned_name`` is what the
  matcher *auto-attributes* (a grounded cosine proposal), read independently of
  any human ruling; ``truth`` is the human adjudication. The two are separate
  axes on purpose: keying the machine prediction off the read-time resolution
  would make every adjudicated label look like a machine abstention.
- ``agreement_enrollment`` + ``agreement_slots`` -> the ``score agreement``
  enroll-and-re-identify contract. Voiceprints and per-label centroids are the
  EXACT vectors production compares, rebuilt from stored per-turn vectors via
  the matcher's own centroid helpers.

``evidence_snapshot`` records the export-time code / gates / roster identity
beside a baseline. It is an export-time snapshot, not a historical replay: the
database does not retain the gates or roster centroids as they were at match
time, so the snapshot is labelled ``gates_at_export`` and documented as such.
"""

from voxint.harness_export.export import (
    ABSTAIN,
    NEITHER_DETERMINABLE,
    ExportError,
    TruthAnchoring,
    agreement_enrollment,
    agreement_slots,
    evidence_snapshot,
    name_accuracy_items,
)

__all__ = [
    "ABSTAIN",
    "NEITHER_DETERMINABLE",
    "ExportError",
    "TruthAnchoring",
    "agreement_enrollment",
    "agreement_slots",
    "evidence_snapshot",
    "name_accuracy_items",
]
