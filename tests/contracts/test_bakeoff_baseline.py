"""Contract: the committed CT2-CPU whisper baseline is bound to the frozen
corpus and honors the licensing doctrine.

The baseline (``tests/parity/fixtures/references/ct2-cpu-metal/transcribe.json``,
produced by ``tools/generate_bakeoff_baseline.py``) is the load-bearing numerics
oracle of issue #33: every Metal candidate is measured against it. This contract
pins the invariants that would otherwise rot silently:

  * it covers exactly the committable strata (AMI CC-BY + synthetic CC0) — no
    TED-LIUM3 (CC-BY-NC-ND) transcript/text ever lands here;
  * every entry is sha-bound to its manifest entry (audio + AMI gold), so a
    baseline can never be silently paired with a different corpus;
  * both decode variants the frozen engine exposes are recorded (``vad_true`` /
    ``vad_false``), matching the pre-registered self-parity gate;
  * ``meta`` pins the oracle identity — device cpu, engine faster-whisper, the
    CPU-image batch size 4, the pinned model revision, and an unmodified
    ``transcription.py``.

Skips (not fails) until the baseline is generated, mirroring
``test_bakeoff_manifest.py`` — capture is a maintainer GPU/native step.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BAKEOFF = REPO / "tests" / "parity" / "fixtures" / "bakeoff"
MANIFEST = BAKEOFF / "manifest.json"
BASELINE = (
    REPO / "tests" / "parity" / "fixtures" / "references" / "ct2-cpu-metal" / "transcribe.json"
)

# Must match WHISPER_HF_REVISION in scripts/metal/voxint-metal.sh and
# tests/parity/test_whisper_metal.py — a baseline captured against a different
# revision is not the pinned oracle.
WHISPER_HF_REVISION = "f0fe81560cb8b68660e564f55dd99207059c092e"

# The pinned oracle batch size: mirror the CPU image (Dockerfile.cpu), not the
# GPU/ROCm app default of 16.
ORACLE_BATCH_SIZE = 4

COMMITTED_DATASETS = {"ami_ihm", "synthetic"}

pytestmark = pytest.mark.skipif(
    not BASELINE.exists(),
    reason="CT2-CPU baseline not generated yet (run tools/generate_bakeoff_baseline.py)",
)


@pytest.fixture(scope="module")
def baseline() -> dict:
    return json.loads(BASELINE.read_text())


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST.read_text())


@pytest.fixture(scope="module")
def by_key(manifest: dict) -> dict:
    return {f"{e['dataset']}/{e['upstream_id']}": e for e in manifest["files"]}


def test_covers_exactly_committable_strata(baseline: dict, manifest: dict) -> None:
    got = {e["dataset"] for e in baseline["entries"].values()}
    assert got == COMMITTED_DATASETS, f"baseline datasets {got} != {COMMITTED_DATASETS}"
    # Exactly the committable manifest entries, no more, no fewer.
    expected_keys = {
        f"{e['dataset']}/{e['upstream_id']}"
        for e in manifest["files"]
        if e["dataset"] in COMMITTED_DATASETS
    }
    assert set(baseline["entries"]) == expected_keys


def test_no_ted_leakage(baseline: dict) -> None:
    for key, entry in baseline["entries"].items():
        assert entry["dataset"] != "tedlium3", key
    # Belt-and-braces: the raw bytes never mention the TED dataset id.
    assert "tedlium3" not in BASELINE.read_text()


def test_every_entry_is_sha_bound_to_the_manifest(baseline: dict, by_key: dict) -> None:
    for key, entry in baseline["entries"].items():
        assert key in by_key, f"baseline entry {key} not in manifest"
        m = by_key[key]
        assert entry["audio_sha256"] == m["sha256"], key
        # The baseline binds to committed gold text wherever the manifest has it:
        # AMI (CC-BY word-gold) and synthetic `short_clean` (CC0 known text) carry
        # a transcript_sha256; synthetic silence/bait (no speech) carry None. The
        # baseline must mirror the manifest exactly — gold-bound when gold exists,
        # None otherwise. (TED is never committed here; see test_no_ted_leakage.)
        assert entry["gold_transcript_sha256"] == m.get("transcript_sha256"), key


def test_both_vad_variants_recorded_with_required_fields(baseline: dict) -> None:
    for key, entry in baseline["entries"].items():
        variants = entry["variants"]
        assert set(variants) == {"vad_true", "vad_false"}, key
        for vname, v in variants.items():
            for field in (
                "transcript", "segments", "words", "confidence",
                "language", "duration_seconds", "suspect_segment_count", "request",
            ):
                assert field in v, f"{key}/{vname} missing {field}"
        # vad_true replays the literal production payload (no vad_filter key);
        # vad_false is the explicit diagnostic denominator.
        assert "vad_filter" not in variants["vad_true"]["request"]
        assert variants["vad_false"]["request"]["vad_filter"] is False


def test_meta_pins_the_oracle_identity(baseline: dict, manifest: dict) -> None:
    meta = baseline["meta"]
    assert meta["tier"] == "ct2-cpu-metal"
    assert meta["service_healthz"]["device"] == "cpu"
    assert meta["service_healthz"]["engine"] == "faster-whisper"
    assert meta["decode_config"]["batch_size"] == ORACLE_BATCH_SIZE
    assert meta["decode_config"]["language"] == "en"
    assert meta["model"]["revision"] == WHISPER_HF_REVISION
    assert meta["code"]["working_tree_clean_for_transcription_py"] is True
    # The baseline is bound to the exact manifest bytes it was captured against.
    import hashlib

    assert meta["corpus"]["manifest_sha256"] == hashlib.sha256(MANIFEST.read_bytes()).hexdigest()


def test_determinism_drift_within_tolerance(baseline: dict) -> None:
    det = baseline["meta"]["determinism"]
    assert det["warm_passes"] == 2
    # Recorded max drift across two warm passes must be within the gate the tool
    # enforced (it fails closed above these, so this guards a regenerated file).
    assert det["max_timestamp_drift_s"] <= det["ts_tol_s"]
