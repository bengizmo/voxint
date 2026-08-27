"""Cohort-freeze tests for synthdetect S5 PR-3 (issue #144).

Tests the pure, audio-free cohort plan layer: deterministic hash-based chain
assignment, one child per calibration parent, turn-only filtering, cohort
plan hash stability, and the ``freeze`` CLI verb dry-run.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import synthdetect_corpus as corpus  # noqa: E402
import synthdetect_sources as sources  # noqa: E402

_SHA = "b" * 64


def _turn_acquire(recording: str = "m1", label: str = "A") -> str:
    return json.dumps(
        {
            "source_file": f"{recording}.wav",
            "recording": recording,
            "rttm_label": label,
            "start_sample": 0,
            "end_sample": 64600,
            "start_s": 0.0,
            "end_s": 4.0375,
            "kind": "turn",
        },
        sort_keys=True,
    )


def _segment_acquire(recording: str = "m1", label: str = "A") -> str:
    return json.dumps(
        {
            "source_file": f"{recording}.wav",
            "recording": recording,
            "rttm_label": label,
            "start_sample": 0,
            "end_sample": 128000,
            "start_s": 0.0,
            "end_s": 8.0,
            "kind": "segment",
        },
        sort_keys=True,
    )


def _cal_clip(clip_id: str, source: str = "ami", **over: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "clip_id": clip_id,
        "rel_path": f"{source}/turn/{clip_id}.wav",
        "sha256": hashlib.sha256(clip_id.encode()).hexdigest(),
        "duration_s": 4.0,
        "label": "bona_fide",
        "language": "en",
        "license_spdx": "CC-BY-4.0",
        "stratum": f"bona_fide|organic|{'meetingroom' if source == 'ami' else 'webvideo'}",
        "source": source,
        "speaker_id": f"{source}-m1-A",
        "split": "calibration",
        "generator": None,
        "degradation": None,
        "parent_clip_id": None,
        "acquire": _turn_acquire(),
    }
    raw.update(over)
    return raw


def _make_manifest(clips: list[dict[str, Any]]) -> corpus.Manifest:
    return corpus.load_manifest({"schema_version": 1, "clips": clips})


# --------------------------------------------------------------------------- #
# Frozen cohort constants
# --------------------------------------------------------------------------- #
class TestFrozenCohortConstants:
    def test_six_chains(self) -> None:
        assert len(sources.FROZEN_COHORT_CHAINS) == 6

    def test_all_singletons(self) -> None:
        for chain in sources.FROZEN_COHORT_CHAINS:
            assert len(chain) == 1

    def test_all_known_recipes(self) -> None:
        for chain in sources.FROZEN_COHORT_CHAINS:
            for recipe_id in chain:
                assert recipe_id in sources.DEGRADATION_RECIPES

    def test_no_duplicate_chains(self) -> None:
        serialized = ["|".join(c) for c in sources.FROZEN_COHORT_CHAINS]
        assert len(serialized) == len(set(serialized))

    def test_version_and_policy(self) -> None:
        assert sources.S5_COHORT_VERSION == 1
        assert sources.S5_COHORT_SELECTION_POLICY == "hash-assign-v1"


# --------------------------------------------------------------------------- #
# Chain assignment
# --------------------------------------------------------------------------- #
class TestChainAssignment:
    def test_deterministic(self) -> None:
        chains = ("a-v1", "b-v1", "c-v1")
        clip_id = "test-clip-123"
        a1 = corpus._assign_chain(clip_id, chains)
        a2 = corpus._assign_chain(clip_id, chains)
        assert a1 == a2

    def test_different_clips_can_get_different_chains(self) -> None:
        chains = ("a-v1", "b-v1", "c-v1", "d-v1", "e-v1", "f-v1")
        assigned = {corpus._assign_chain(f"clip-{i}", chains) for i in range(100)}
        assert len(assigned) > 1

    def test_all_chains_reachable(self) -> None:
        serialized = tuple(sorted("|".join(c) for c in sources.FROZEN_COHORT_CHAINS))
        assigned = {
            corpus._assign_chain(f"clip-{i}", serialized) for i in range(1000)
        }
        assert assigned == set(serialized)


# --------------------------------------------------------------------------- #
# plan_cohort
# --------------------------------------------------------------------------- #
class TestPlanCohort:
    def test_basic_plan(self) -> None:
        clips = [_cal_clip(f"cal-{i}") for i in range(12)]
        manifest = _make_manifest(clips)
        plan = corpus.plan_cohort(manifest)

        assert plan.cohort_version == sources.S5_COHORT_VERSION
        assert plan.selection_policy == sources.S5_COHORT_SELECTION_POLICY
        assert len(plan.chains) == 6
        assert len(plan.assignments) == 12
        assert len(plan.children) == 12
        assert plan.cohort_plan_sha256

    def test_one_child_per_parent(self) -> None:
        clips = [_cal_clip(f"cal-{i}") for i in range(20)]
        manifest = _make_manifest(clips)
        plan = corpus.plan_cohort(manifest)

        parent_ids = [a.parent_clip_id for a in plan.assignments]
        assert len(parent_ids) == len(set(parent_ids))

    def test_only_calibration_split(self) -> None:
        clips = [
            _cal_clip("cal-0", split="calibration", speaker_id="spk-cal"),
            _cal_clip("eval-0", split="eval", speaker_id="spk-eval"),
            _cal_clip("hold-0", split="holdout", speaker_id="spk-hold"),
        ]
        manifest = _make_manifest(clips)
        plan = corpus.plan_cohort(manifest)

        assert len(plan.assignments) == 1
        assert plan.assignments[0].parent_clip_id == "cal-0"

    def test_only_turn_clips(self) -> None:
        clips = [
            _cal_clip("turn-0", acquire=_turn_acquire()),
            _cal_clip("seg-0", acquire=_segment_acquire()),
        ]
        manifest = _make_manifest(clips)
        plan = corpus.plan_cohort(manifest)

        assert len(plan.assignments) == 1
        assert plan.assignments[0].parent_clip_id == "turn-0"

    def test_rejects_manifest_with_degraded_entries(self) -> None:
        parent = _cal_clip("parent-0")
        child = {
            **_cal_clip("child-0"),
            "parent_clip_id": "parent-0",
            "degradation": "mp3-cbr48-v1",
            "stratum": "bona_fide|organic|meetingroom|mp3-cbr48-v1",
        }
        with pytest.raises(corpus.CorpusError, match="already contains degraded"):
            corpus.plan_cohort(_make_manifest([parent, child]))

    def test_no_eligible_parents_raises(self) -> None:
        clips = [_cal_clip("eval-only", split="eval")]
        manifest = _make_manifest(clips)
        with pytest.raises(corpus.CorpusError, match="no eligible"):
            corpus.plan_cohort(manifest)

    def test_deterministic_hash(self) -> None:
        clips = [_cal_clip(f"cal-{i}") for i in range(6)]
        m = _make_manifest(clips)
        h1 = corpus.plan_cohort(m).cohort_plan_sha256
        h2 = corpus.plan_cohort(m).cohort_plan_sha256
        assert h1 == h2

    def test_hash_changes_with_different_parents(self) -> None:
        m1 = _make_manifest([_cal_clip(f"a-{i}") for i in range(6)])
        m2 = _make_manifest([_cal_clip(f"b-{i}") for i in range(6)])
        h1 = corpus.plan_cohort(m1).cohort_plan_sha256
        h2 = corpus.plan_cohort(m2).cohort_plan_sha256
        assert h1 != h2

    def test_hash_independent_of_manifest_order(self) -> None:
        clips = [_cal_clip(f"cal-{i}") for i in range(6)]
        m1 = _make_manifest(clips)
        m2 = _make_manifest(list(reversed(clips)))
        h1 = corpus.plan_cohort(m1).cohort_plan_sha256
        h2 = corpus.plan_cohort(m2).cohort_plan_sha256
        assert h1 == h2

    def test_children_inherit_parent_fields(self) -> None:
        clips = [_cal_clip("cal-0")]
        manifest = _make_manifest(clips)
        plan = corpus.plan_cohort(manifest)
        child = plan.children[0]

        assert child.parent_clip_id == "cal-0"
        assert child.label == "bona_fide"
        assert child.split == "calibration"
        assert child.source == "ami"

    def test_acquire_without_kind_excluded(self) -> None:
        clips = [_cal_clip("no-kind", acquire=json.dumps({"recording": "m1"}, sort_keys=True))]
        manifest = _make_manifest(clips)
        with pytest.raises(corpus.CorpusError, match="no eligible"):
            corpus.plan_cohort(manifest)

    def test_null_acquire_excluded(self) -> None:
        clips = [_cal_clip("null-acq", acquire=None)]
        manifest = _make_manifest(clips)
        with pytest.raises(corpus.CorpusError, match="no eligible"):
            corpus.plan_cohort(manifest)

    @pytest.mark.parametrize("bad_acquire", [
        json.dumps(42),
        json.dumps([1, 2]),
        json.dumps("a string"),
        json.dumps(True),
        json.dumps(None),
    ])
    def test_non_object_acquire_excluded(self, bad_acquire: str) -> None:
        clips = [_cal_clip("bad-acq", acquire=bad_acquire)]
        manifest = _make_manifest(clips)
        with pytest.raises(corpus.CorpusError, match="no eligible"):
            corpus.plan_cohort(manifest)


# --------------------------------------------------------------------------- #
# Cohort plan hash projection
# --------------------------------------------------------------------------- #
class TestCohortPlanHash:
    def test_hash_is_64_hex(self) -> None:
        clips = [_cal_clip(f"cal-{i}") for i in range(3)]
        plan = corpus.plan_cohort(_make_manifest(clips))
        assert len(plan.cohort_plan_sha256) == 64
        int(plan.cohort_plan_sha256, 16)

    def test_hash_includes_version(self) -> None:
        clips = [_cal_clip(f"cal-{i}") for i in range(3)]
        m = _make_manifest(clips)
        h1 = corpus.plan_cohort(m, cohort_version=1).cohort_plan_sha256
        h2 = corpus.plan_cohort(m, cohort_version=2).cohort_plan_sha256
        assert h1 != h2

    def test_hash_includes_policy(self) -> None:
        clips = [_cal_clip(f"cal-{i}") for i in range(3)]
        m = _make_manifest(clips)
        h1 = corpus.plan_cohort(m, selection_policy="hash-assign-v1").cohort_plan_sha256
        h2 = corpus.plan_cohort(m, selection_policy="hash-assign-v2").cohort_plan_sha256
        assert h1 != h2


# --------------------------------------------------------------------------- #
# validate_cohort_chains
# --------------------------------------------------------------------------- #
class TestValidateCohortChains:
    def test_empty_chains_rejected(self) -> None:
        with pytest.raises(sources.SourcesError, match="non-empty"):
            sources._validate_cohort_chains(())

    def test_empty_inner_chain_rejected(self) -> None:
        with pytest.raises(sources.SourcesError, match="empty chain"):
            sources._validate_cohort_chains(((),))

    def test_unknown_recipe_rejected(self) -> None:
        with pytest.raises(sources.SourcesError, match="unknown recipe"):
            sources._validate_cohort_chains(
                (("nonexistent-v1",),), sources.DEGRADATION_RECIPES
            )

    def test_duplicate_chain_rejected(self) -> None:
        with pytest.raises(sources.SourcesError, match="duplicate chain"):
            sources._validate_cohort_chains(
                (("mp3-cbr48-v1",), ("mp3-cbr48-v1",)),
                sources.DEGRADATION_RECIPES,
            )


# --------------------------------------------------------------------------- #
# Plan-manifest binding check in materialize_cohort
# --------------------------------------------------------------------------- #
class TestPlanManifestBinding:
    def test_sha_mismatch_rejected(self, tmp_path: Path) -> None:
        clips = [_cal_clip("cal-0")]
        manifest = _make_manifest(clips)
        plan = corpus.plan_cohort(manifest)

        parent_root = tmp_path / "parent"
        parent_root.mkdir()
        manifest_bytes = json.dumps(
            {"schema_version": 1, "clips": clips}, indent=2, sort_keys=True,
        ).encode("utf-8")
        (parent_root / "manifest.json").write_bytes(manifest_bytes)
        clip_dir = parent_root / "ami" / "turn"
        clip_dir.mkdir(parents=True)
        (clip_dir / "cal-0.wav").write_bytes(b"\x00")

        tampered = corpus.CohortAssignment(
            parent_clip_id=plan.assignments[0].parent_clip_id,
            parent_pcm_sha256="a" * 64,
            source=plan.assignments[0].source,
            speaker_id=plan.assignments[0].speaker_id,
            split=plan.assignments[0].split,
            assigned_chain=plan.assignments[0].assigned_chain,
            child_clip_id=plan.assignments[0].child_clip_id,
        )
        tampered_plan = corpus.CohortPlan(
            cohort_version=plan.cohort_version,
            selection_policy=plan.selection_policy,
            chains=plan.chains,
            assignments=(tampered,),
            children=plan.children,
            cohort_plan_sha256=plan.cohort_plan_sha256,
        )
        with pytest.raises(corpus.CorpusError, match="plan/manifest mismatch"):
            corpus.materialize_cohort(
                plan=tampered_plan,
                parent_root=parent_root,
                corpus_root=tmp_path / "cohort",
                container_image="repo@sha256:" + "a" * 64,
            )


# --------------------------------------------------------------------------- #
# freeze CLI dry-run
# --------------------------------------------------------------------------- #
class TestCmdFreezeDryRun:
    def _write_manifest(self, tmp_path: Path, clips: list[dict[str, Any]]) -> Path:
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(
            json.dumps({"schema_version": 1, "clips": clips}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return manifest_path

    def test_dry_run_outputs_plan(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        clips = [_cal_clip(f"cal-{i}") for i in range(6)]
        mpath = self._write_manifest(tmp_path, clips)

        ns = corpus.argparse.Namespace(
            manifest=str(mpath),
            corpus_root=None,
            parent_root=None,
            container_image=None,
        )
        rc = corpus.cmd_freeze(ns)
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert out["cohort_plan_sha256"]
        assert out["cohort_version"] == 1
        assert out["selection_policy"] == "hash-assign-v1"
        assert out["assignments"] == 6
        assert len(out["children"]) == 6

    def test_dry_run_no_eligible_fails(self, tmp_path: Path) -> None:
        clips = [_cal_clip("eval-only", split="eval")]
        mpath = self._write_manifest(tmp_path, clips)

        ns = corpus.argparse.Namespace(
            manifest=str(mpath),
            corpus_root=None,
            parent_root=None,
            container_image=None,
        )
        rc = corpus.cmd_freeze(ns)
        assert rc == 2

    def test_execution_requires_parent_root(self, tmp_path: Path) -> None:
        clips = [_cal_clip("cal-0")]
        mpath = self._write_manifest(tmp_path, clips)

        ns = corpus.argparse.Namespace(
            manifest=str(mpath),
            corpus_root=str(tmp_path / "out"),
            parent_root=None,
            container_image=None,
        )
        rc = corpus.cmd_freeze(ns)
        assert rc == 2
