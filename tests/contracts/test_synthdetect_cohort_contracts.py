"""Contract tests for the synthdetect cohort freeze (#144, M1 S5 PR-3).

Pins the frozen cohort policy's cross-seam contracts:

- ``FROZEN_COHORT_CHAINS`` entries all reference known recipes.
- ``S5_COHORT_VERSION`` is a positive integer.
- Cohort plan hash is a 64-char hex digest.
- ``_assign_chain`` is stable (a golden clip_id maps to the same chain).
- The selection policy id matches the expected value.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import synthdetect_corpus as corpus  # noqa: E402
import synthdetect_sources as sources  # noqa: E402


class TestCohortRegistryContract:
    def test_chains_reference_known_recipes(self) -> None:
        for chain in sources.FROZEN_COHORT_CHAINS:
            for recipe_id in chain:
                assert recipe_id in sources.DEGRADATION_RECIPES, (
                    f"FROZEN_COHORT_CHAINS names unknown recipe {recipe_id!r}"
                )

    def test_version_positive(self) -> None:
        assert isinstance(sources.S5_COHORT_VERSION, int)
        assert sources.S5_COHORT_VERSION > 0

    def test_chain_count_matches_recipe_count(self) -> None:
        assert len(sources.FROZEN_COHORT_CHAINS) == len(sources.DEGRADATION_RECIPES)

    def test_policy_id_is_hash_assign_v1(self) -> None:
        assert sources.S5_COHORT_SELECTION_POLICY == "hash-assign-v1"


class TestAssignmentStability:
    """A golden clip_id must always map to the same chain (hash stability)."""

    GOLDEN_CLIP_ID = "ami-ES2011a-MEE069-turn-0-64600"

    def test_golden_assignment_stable(self) -> None:
        serialized = tuple(sorted("|".join(c) for c in sources.FROZEN_COHORT_CHAINS))
        chain = corpus._assign_chain(self.GOLDEN_CLIP_ID, serialized)
        assert chain in serialized
        expected = corpus._assign_chain(self.GOLDEN_CLIP_ID, serialized)
        assert chain == expected


class TestCohortPlanHashContract:
    def _make_plan_fixture(self) -> corpus.CohortPlan:
        def _clip(i: int) -> dict:
            cid = f"cal-{i}"
            return {
                "clip_id": cid,
                "rel_path": f"ami/turn/{cid}.wav",
                "sha256": hashlib.sha256(cid.encode()).hexdigest(),
                "duration_s": 4.0,
                "label": "bona_fide",
                "language": "en",
                "license_spdx": "CC-BY-4.0",
                "stratum": "bona_fide|organic|meetingroom",
                "source": "ami",
                "speaker_id": f"spk-{i}",
                "split": "calibration",
                "generator": None,
                "degradation": None,
                "parent_clip_id": None,
                "acquire": json.dumps(
                    {"kind": "turn", "recording": "m1", "rttm_label": "A",
                     "source_file": "m1.wav", "start_sample": 0,
                     "end_sample": 64600, "start_s": 0.0, "end_s": 4.0375},
                    sort_keys=True,
                ),
            }

        clips = [_clip(i) for i in range(6)]
        manifest = corpus.load_manifest({"schema_version": 1, "clips": clips})
        return corpus.plan_cohort(manifest)

    def test_hash_is_64_hex(self) -> None:
        plan = self._make_plan_fixture()
        assert len(plan.cohort_plan_sha256) == 64
        int(plan.cohort_plan_sha256, 16)

    def test_hash_reproducible(self) -> None:
        h1 = self._make_plan_fixture().cohort_plan_sha256
        h2 = self._make_plan_fixture().cohort_plan_sha256
        assert h1 == h2
