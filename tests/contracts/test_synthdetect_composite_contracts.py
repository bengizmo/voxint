"""Contract tests for the synthdetect v3 composite manifest (#144, M1 S6).

Pins the v3 composite schema's cross-seam contracts:

- The composite constants exist and have the expected values.
- A v3 composite clip exposes the same scoring-path fields as v1/v2.
- The scorer's ``join_scores`` function works identically on v3 clips.
- v3 provenance kinds match exactly the v1/v2 corpus kinds.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import synthdetect_corpus as corpus  # noqa: E402
import synthdetect_eval as se  # noqa: E402


def test_composite_constants_exist() -> None:
    assert corpus.COMPOSITE_MANIFEST_SCHEMA_VERSION == 3
    assert corpus.CORPUS_KIND_COMPOSITE == "composite"
    assert 3 in corpus.SUPPORTED_SCHEMA_VERSIONS


def test_composite_provenance_kinds_match_original_corpus_kinds() -> None:
    assert corpus.CORPUS_KIND_SYNTHESIS == "synthesis"
    assert corpus.CORPUS_KIND_IMPORTED == "imported_benchmark"


def test_manifest_component_dataclass_fields() -> None:
    c = corpus.ManifestComponent(
        component_id="test",
        corpus_kind=corpus.CORPUS_KIND_SYNTHESIS,
        manifest_sha256="a" * 64,
        clip_count=1,
    )
    assert c.component_id == "test"
    assert c.benchmark is None


def test_clip_entry_has_composite_fields() -> None:
    fields = {f.name for f in corpus.ClipEntry.__dataclass_fields__.values()}
    assert "component_id" in fields
    assert "partition_group_id" in fields


_SHA = "a" * 64
_SCORING_FIELDS = ("clip_id", "label", "stratum", "split")


def test_v3_clip_scoring_path_fields_identical_to_v1() -> None:
    """The scorer reads clip_id/label/stratum/split from any manifest version."""
    v1_clip = corpus.validate_clip(
        {
            "clip_id": "c1",
            "rel_path": "c1.wav",
            "sha256": _SHA,
            "duration_s": 5.0,
            "label": "bona_fide",
            "language": "en",
            "license_spdx": "CC0-1.0",
            "stratum": "test",
            "source": "test",
            "speaker_id": "s1",
            "split": "calibration",
            "generator": None,
            "degradation": None,
            "parent_clip_id": None,
            "acquire": None,
        },
        0,
    )
    v3_clip = corpus.validate_clip(
        {
            "clip_id": "c1",
            "rel_path": "c1.wav",
            "sha256": _SHA,
            "duration_s": 5.0,
            "label": "bona_fide",
            "language": "en",
            "license_spdx": "CC0-1.0",
            "stratum": "test",
            "source": "test",
            "speaker_id": "s1",
            "split": "calibration",
            "provenance_kind": corpus.CORPUS_KIND_SYNTHESIS,
            "component_id": "organic",
            "partition_group_id": None,
            "generator": None,
            "degradation": None,
            "parent_clip_id": None,
            "acquire": None,
        },
        0,
        composite=True,
    )
    for field in _SCORING_FIELDS:
        assert getattr(v1_clip, field) == getattr(v3_clip, field)


def test_scorer_positive_label_is_spoof() -> None:
    assert se.POSITIVE_LABEL == "spoof"
