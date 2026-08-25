"""Manifest schema + split-assignment tests for synthdetect (issue #144).

Freezes the fail-closed manifest validation and the seeded, speaker-disjoint
split assignment before any audio exists, so a later edit cannot silently accept
a malformed record or leak a speaker across the calibration/eval boundary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import synthdetect_corpus as corpus  # noqa: E402

_SHA = "a" * 64


def _bona_fide(clip_id: str, speaker: str, **over: Any) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "clip_id": clip_id,
        "rel_path": f"organic/{clip_id}.wav",
        "sha256": _SHA,
        "duration_s": 5.0,
        "label": "bona_fide",
        "language": "en",
        "license_spdx": "CC0-1.0",
        "stratum": "domain_bonafide",
        "source": "maintainer",
        "speaker_id": speaker,
        "split": None,
        "generator": None,
        "degradation": None,
        "parent_clip_id": None,
        "acquire": None,
    }
    rec.update(over)
    return rec


def _spoof(clip_id: str, speaker: str, gen: str = "piper", **over: Any) -> dict[str, Any]:
    rec = _bona_fide(clip_id, speaker, label="spoof")
    rec["stratum"] = f"tts_{gen}"
    rec["generator"] = {
        "name": gen,
        "version": "1.0",
        "checkpoint_sha": None,
        "voice": "v1",
        "seed": "42",
        "text_source": "harvard",
    }
    rec.update(over)
    return rec


def _manifest(clips: list[dict[str, Any]]) -> dict[str, Any]:
    return {"schema_version": corpus.MANIFEST_SCHEMA_VERSION, "clips": clips}


# --------------------------------------------------------------------------- #
# validate_clip / load_manifest fail-closed
# --------------------------------------------------------------------------- #
def test_valid_manifest_loads() -> None:
    m = corpus.load_manifest(_manifest([_bona_fide("c1", "spk1"), _spoof("c2", "spk2")]))
    assert len(m.clips) == 2
    assert m.clips[1].generator is not None
    assert m.clips[1].generator.name == "piper"


def test_bad_schema_version_rejected() -> None:
    with pytest.raises(corpus.CorpusError, match="schema_version"):
        corpus.load_manifest({"schema_version": 99, "clips": [_bona_fide("c1", "s1")]})


def test_empty_clips_rejected() -> None:
    with pytest.raises(corpus.CorpusError, match="non-empty array"):
        corpus.load_manifest(_manifest([]))


def test_spoof_requires_generator() -> None:
    bad = _spoof("c1", "s1")
    bad["generator"] = None
    with pytest.raises(corpus.CorpusError, match="must carry generator provenance"):
        corpus.load_manifest(_manifest([bad]))


def test_bona_fide_must_not_carry_generator() -> None:
    bad = _bona_fide("c1", "s1")
    bad["generator"] = {"name": "x", "version": "1", "checkpoint_sha": None,
                        "voice": "v", "seed": "1", "text_source": "t"}
    with pytest.raises(corpus.CorpusError, match="must not carry a generator"):
        corpus.load_manifest(_manifest([bad]))


def test_bad_sha_rejected() -> None:
    with pytest.raises(corpus.CorpusError, match="sha256"):
        corpus.load_manifest(_manifest([_bona_fide("c1", "s1", sha256="xyz")]))


def test_non_positive_duration_rejected() -> None:
    with pytest.raises(corpus.CorpusError, match="duration_s"):
        corpus.load_manifest(_manifest([_bona_fide("c1", "s1", duration_s=0)]))


def test_non_finite_duration_rejected() -> None:
    # json.loads parses Infinity/NaN by default and `inf <= 0` is False, so a
    # fail-closed module must reject non-finite durations explicitly.
    with pytest.raises(corpus.CorpusError, match="finite"):
        corpus.load_manifest(_manifest([_bona_fide("c1", "s1", duration_s=float("inf"))]))
    with pytest.raises(corpus.CorpusError, match="finite"):
        corpus.load_manifest(_manifest([_bona_fide("c1", "s1", duration_s=float("nan"))]))


def test_self_parent_rejected() -> None:
    bad = _bona_fide("c1", "s1", degradation="opus16", parent_clip_id="c1")
    with pytest.raises(corpus.CorpusError, match="its own parent"):
        corpus.load_manifest(_manifest([bad]))


def test_parent_without_degradation_rejected() -> None:
    parent = _bona_fide("c1", "s1")
    child = _bona_fide("c2", "s1", parent_clip_id="c1")  # no degradation label
    with pytest.raises(corpus.CorpusError, match="without a degradation"):
        corpus.load_manifest(_manifest([parent, child]))


def test_preassigned_split_speaker_straddle_rejected() -> None:
    # Two clips from one speaker stamped into different splits must be refused;
    # speaker-disjointness is a load-time invariant, not just an assign_splits one.
    a = _bona_fide("c1", "spk1", split="calibration")
    b = _bona_fide("c2", "spk1", split="eval")
    with pytest.raises(corpus.CorpusError, match="straddles splits"):
        corpus.load_manifest(_manifest([a, b]))


def test_non_finite_fraction_rejected() -> None:
    clips = _clips_for_split()
    with pytest.raises(corpus.CorpusError, match="finite"):
        corpus.assign_splits(
            clips, eval_only_generators={"chatterbox"}, calibration_fraction=float("nan")
        )


def test_bad_label_rejected() -> None:
    with pytest.raises(corpus.CorpusError, match="label"):
        corpus.load_manifest(_manifest([_bona_fide("c1", "s1", label="maybe")]))


def test_bad_split_rejected() -> None:
    with pytest.raises(corpus.CorpusError, match="split"):
        corpus.load_manifest(_manifest([_bona_fide("c1", "s1", split="train")]))


def test_traversing_rel_path_rejected() -> None:
    with pytest.raises(corpus.CorpusError, match="relative and not traverse"):
        corpus.load_manifest(_manifest([_bona_fide("c1", "s1", rel_path="../escape.wav")]))


def test_absolute_rel_path_rejected() -> None:
    with pytest.raises(corpus.CorpusError, match="relative and not traverse"):
        corpus.load_manifest(_manifest([_bona_fide("c1", "s1", rel_path="/etc/passwd")]))


def test_unsafe_clip_id_rejected() -> None:
    with pytest.raises(corpus.CorpusError, match="clip_id"):
        corpus.load_manifest(_manifest([_bona_fide("../evil", "s1")]))


def test_duplicate_clip_id_rejected() -> None:
    with pytest.raises(corpus.CorpusError, match="duplicate clip_id"):
        corpus.load_manifest(_manifest([_bona_fide("c1", "s1"), _bona_fide("c1", "s2")]))


def test_dangling_parent_rejected() -> None:
    child = _bona_fide("c2", "s1", degradation="opus16", parent_clip_id="missing")
    with pytest.raises(corpus.CorpusError, match="not in the manifest"):
        corpus.load_manifest(_manifest([child]))


def test_degraded_clip_needs_parent() -> None:
    bad = _bona_fide("c1", "s1", degradation="opus16")
    with pytest.raises(corpus.CorpusError, match="must name its parent_clip_id"):
        corpus.load_manifest(_manifest([bad]))


def test_valid_degradation_chain_links() -> None:
    parent = _bona_fide("c1", "s1")
    child = _bona_fide("c2", "s1", degradation="opus16", parent_clip_id="c1")
    m = corpus.load_manifest(_manifest([parent, child]))
    assert m.clips[1].degradation == "opus16"
    assert m.clips[1].parent_clip_id == "c1"


def test_unexpected_key_rejected() -> None:
    with pytest.raises(corpus.CorpusError, match="unexpected keys"):
        corpus.load_manifest(_manifest([_bona_fide("c1", "s1", surprise="x")]))


def test_bad_generator_checkpoint_sha_rejected() -> None:
    bad = _spoof("c1", "s1")
    bad["generator"]["checkpoint_sha"] = "not-a-sha"
    with pytest.raises(corpus.CorpusError, match="checkpoint_sha"):
        corpus.load_manifest(_manifest([bad]))


# --------------------------------------------------------------------------- #
# assign_splits: speaker-disjoint, seeded, unseen-generator eval
# --------------------------------------------------------------------------- #
def _clips_for_split() -> tuple:
    clips = []
    # 10 bona fide speakers, 2 clips each.
    for i in range(10):
        clips.append(_bona_fide(f"b{i}a", f"spk{i}"))
        clips.append(_bona_fide(f"b{i}b", f"spk{i}"))
    # An eval-only generator "chatterbox" with its own synthetic voices.
    clips.append(_spoof("g1", "voiceA", gen="chatterbox"))
    clips.append(_spoof("g2", "voiceB", gen="chatterbox"))
    # A regular generator "piper".
    clips.append(_spoof("p1", "voiceC", gen="piper"))
    m = corpus.load_manifest(_manifest(clips))
    return m.clips


def test_split_is_speaker_disjoint() -> None:
    clips = _clips_for_split()
    assignment = corpus.assign_splits(clips, eval_only_generators={"chatterbox"})
    by_speaker: dict[str, set[str]] = {}
    for c in clips:
        by_speaker.setdefault(c.speaker_id, set()).add(assignment[c.clip_id])
    for speaker, splits in by_speaker.items():
        assert len(splits) == 1, f"speaker {speaker} straddles splits {splits}"


def test_eval_only_generator_lands_in_eval() -> None:
    clips = _clips_for_split()
    assignment = corpus.assign_splits(clips, eval_only_generators={"chatterbox"})
    assert assignment["g1"] == "eval"
    assert assignment["g2"] == "eval"


def test_split_assignment_is_deterministic() -> None:
    clips = _clips_for_split()
    a1 = corpus.assign_splits(clips, eval_only_generators={"chatterbox"})
    a2 = corpus.assign_splits(clips, eval_only_generators={"chatterbox"})
    assert a1 == a2


def test_split_changes_with_seed() -> None:
    clips = _clips_for_split()
    a1 = corpus.assign_splits(clips, eval_only_generators={"chatterbox"}, seed="seed-a")
    a2 = corpus.assign_splits(clips, eval_only_generators={"chatterbox"}, seed="seed-b")
    assert a1 != a2


def test_unknown_eval_only_generator_rejected() -> None:
    clips = _clips_for_split()
    with pytest.raises(corpus.CorpusError, match="no clips in the corpus"):
        corpus.assign_splits(clips, eval_only_generators={"nonexistent"})


def test_fractions_over_one_rejected() -> None:
    clips = _clips_for_split()
    with pytest.raises(corpus.CorpusError, match="exceed"):
        corpus.assign_splits(
            clips, eval_only_generators={"chatterbox"},
            calibration_fraction=0.7, holdout_fraction=0.5,
        )


def test_empty_clips_rejected_in_assign() -> None:
    with pytest.raises(corpus.CorpusError, match="no clips to assign"):
        corpus.assign_splits([], eval_only_generators=set())


def test_split_summary_counts() -> None:
    clips = _clips_for_split()
    assignment = corpus.assign_splits(clips, eval_only_generators={"chatterbox"})
    summary = corpus.split_summary(assignment)
    assert sum(summary.values()) == len(clips)
    assert set(summary) == set(corpus.SPLITS)


def test_all_three_splits_populated() -> None:
    clips = _clips_for_split()
    assignment = corpus.assign_splits(
        clips, eval_only_generators={"chatterbox"},
        calibration_fraction=0.5, holdout_fraction=0.2,
    )
    summary = corpus.split_summary(assignment)
    assert summary["calibration"] > 0
    assert summary["eval"] > 0
    assert summary["holdout"] > 0


# --------------------------------------------------------------------------- #
# validate CLI
# --------------------------------------------------------------------------- #
def test_cli_validate_ok(tmp_path: Path, capsys: Any) -> None:
    path = tmp_path / "m.json"
    path.write_text(json.dumps(_manifest([
        _bona_fide("c1", "s1", split="eval"), _spoof("c2", "s2", split="eval")
    ])), encoding="utf-8")
    rc = corpus.main(["validate", "--manifest", str(path)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True
    assert out["clips"] == 2
    assert out["labels"] == {"bona_fide": 1, "spoof": 1}
    assert out["split_summary"]["eval"] == 2


def test_cli_validate_bad_manifest(tmp_path: Path, capsys: Any) -> None:
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"schema_version": 99, "clips": []}), encoding="utf-8")
    rc = corpus.main(["validate", "--manifest", str(path)])
    assert rc == 2
    assert "manifest invalid" in capsys.readouterr().err


def test_cli_validate_missing_file(tmp_path: Path) -> None:
    rc = corpus.main(["validate", "--manifest", str(tmp_path / "nope.json")])
    assert rc == 2


# --------------------------------------------------------------------------- #
# Schema v2: imported-benchmark manifest (issue #144, S3)
# --------------------------------------------------------------------------- #
def _imported_bona_fide(clip_id: str, speaker: str, **over: Any) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "clip_id": clip_id,
        "rel_path": f"canonical/{clip_id}.wav",
        "sha256": _SHA,
        "duration_s": 4.0,
        "label": "bona_fide",
        "language": "und",
        "license_spdx": "LicenseRef-ASVspoof2021-DF",
        "stratum": "bona_fide|nocodec",
        "source": "asvspoof2021-df",
        "speaker_id": speaker,
        "split": "eval",
        "imported_provenance": {
            "official_trial_id": clip_id,
            "source_dataset": "asvspoof",
            "codec_condition": "nocodec",
            "official_split": "eval",
            "attack_system": None,
            "vocoder_family": "bonafide",
        },
    }
    rec.update(over)
    return rec


def _imported_spoof(clip_id: str, speaker: str, **over: Any) -> dict[str, Any]:
    rec = _imported_bona_fide(clip_id, speaker, label="spoof")
    rec["stratum"] = "spoof|nocodec"
    rec["imported_provenance"] = {
        "official_trial_id": clip_id,
        "source_dataset": "vcc2020",
        "codec_condition": "nocodec",
        "official_split": "eval",
        "attack_system": "A09",
        "vocoder_family": "traditional_vocoder",
    }
    rec.update(over)
    return rec


def _imported_manifest(
    clips: list[dict[str, Any]], benchmark: str = "asvspoof2021-df", **over: Any
) -> dict[str, Any]:
    m: dict[str, Any] = {
        "schema_version": corpus.IMPORTED_MANIFEST_SCHEMA_VERSION,
        "corpus_kind": corpus.CORPUS_KIND_IMPORTED,
        "benchmark": benchmark,
        "clips": clips,
    }
    m.update(over)
    return m


def test_imported_manifest_loads() -> None:
    clips = [_imported_bona_fide("DF_E_1", "LA_1"), _imported_spoof("DF_E_2", "TEF2")]
    m = corpus.load_manifest(_imported_manifest(clips))
    assert m.corpus_kind == corpus.CORPUS_KIND_IMPORTED
    assert m.benchmark == "asvspoof2021-df"
    spoof = m.clips[1]
    assert spoof.generator is None
    assert spoof.imported_provenance is not None
    assert spoof.imported_provenance.attack_system == "A09"
    assert spoof.imported_provenance.vocoder_family == "traditional_vocoder"
    # A bona fide imported clip carries the provenance block but no attack system;
    # its vocoder family is the official 'bonafide' marker, not null.
    bona = m.clips[0]
    assert bona.imported_provenance is not None
    assert bona.imported_provenance.attack_system is None
    assert bona.imported_provenance.vocoder_family == "bonafide"


def test_imported_spoof_missing_provenance_rejected() -> None:
    bad = _imported_spoof("DF_E_1", "s1")
    del bad["imported_provenance"]
    with pytest.raises(corpus.CorpusError, match="must carry imported_provenance"):
        corpus.load_manifest(_imported_manifest([bad]))


def test_imported_clip_with_generator_key_rejected() -> None:
    bad = _imported_spoof("DF_E_1", "s1")
    bad["generator"] = None  # a synthesis key is not allowed on an imported clip
    with pytest.raises(corpus.CorpusError, match="unexpected keys"):
        corpus.load_manifest(_imported_manifest([bad]))


def test_imported_bona_fide_with_attack_rejected() -> None:
    bad = _imported_bona_fide("DF_E_1", "s1")
    bad["imported_provenance"]["attack_system"] = "A09"
    with pytest.raises(corpus.CorpusError, match="must not carry an attack_system"):
        corpus.load_manifest(_imported_manifest([bad]))


def test_imported_spoof_missing_attack_rejected() -> None:
    bad = _imported_spoof("DF_E_1", "s1")
    bad["imported_provenance"]["attack_system"] = None
    with pytest.raises(corpus.CorpusError, match="must name its attack_system"):
        corpus.load_manifest(_imported_manifest([bad]))


@pytest.mark.parametrize("sentinel", ["unknown", "n/a", "-", "", "None", "TBD"])
def test_imported_provenance_sentinel_rejected(sentinel: str) -> None:
    bad = _imported_spoof("DF_E_1", "s1")
    bad["imported_provenance"]["source_dataset"] = sentinel
    with pytest.raises(corpus.CorpusError, match=r"placeholder|non-empty string"):
        corpus.load_manifest(_imported_manifest([bad]))


def test_imported_spoof_unknown_vocoder_accepted() -> None:
    # 'unknown' is a real official vocoder family; it must NOT be sentinel-rejected.
    ok = _imported_spoof("DF_E_1", "s1")
    ok["imported_provenance"]["vocoder_family"] = "unknown"
    m = corpus.load_manifest(_imported_manifest([ok]))
    assert m.clips[0].imported_provenance is not None
    assert m.clips[0].imported_provenance.vocoder_family == "unknown"


def test_imported_provenance_null_vocoder_rejected() -> None:
    bad = _imported_spoof("DF_E_1", "s1")
    bad["imported_provenance"]["vocoder_family"] = None
    with pytest.raises(corpus.CorpusError, match="vocoder_family must be a non-empty string"):
        corpus.load_manifest(_imported_manifest([bad]))


def test_imported_manifest_missing_corpus_kind_rejected() -> None:
    m = _imported_manifest([_imported_spoof("DF_E_1", "s1")])
    del m["corpus_kind"]
    with pytest.raises(corpus.CorpusError, match="corpus_kind"):
        corpus.load_manifest(m)


def test_imported_manifest_wrong_corpus_kind_rejected() -> None:
    m = _imported_manifest([_imported_spoof("DF_E_1", "s1")], corpus_kind="synthesis")
    with pytest.raises(corpus.CorpusError, match="corpus_kind"):
        corpus.load_manifest(m)


def test_imported_manifest_missing_benchmark_rejected() -> None:
    m = _imported_manifest([_imported_spoof("DF_E_1", "s1")], benchmark="   ")
    with pytest.raises(corpus.CorpusError, match="benchmark"):
        corpus.load_manifest(m)


def test_imported_manifest_unexpected_top_level_key_rejected() -> None:
    m = _imported_manifest([_imported_spoof("DF_E_1", "s1")], surprise="x")
    with pytest.raises(corpus.CorpusError, match="unexpected top-level keys"):
        corpus.load_manifest(m)


def test_v1_manifest_with_imported_provenance_rejected() -> None:
    # A v1 clip may not smuggle an imported_provenance block (unknown key).
    bad = _bona_fide("c1", "s1", imported_provenance={"official_trial_id": "x"})
    with pytest.raises(corpus.CorpusError, match="unexpected keys"):
        corpus.load_manifest(_manifest([bad]))


def test_v1_manifest_with_top_level_corpus_kind_rejected() -> None:
    # The v1 hardening: v1 no longer silently ignores extra top-level keys.
    m = _manifest([_bona_fide("c1", "s1")])
    m["corpus_kind"] = "imported_benchmark"
    with pytest.raises(corpus.CorpusError, match="unexpected top-level keys"):
        corpus.load_manifest(m)


# --- v2 cross-field consistency + honesty (multi-model review hardening) ----- #
def test_imported_provenance_missing_key_rejected() -> None:
    # An officially-absent field must be present-and-null, never omitted.
    bad = _imported_bona_fide("DF_E_1", "s1")
    del bad["imported_provenance"]["attack_system"]
    with pytest.raises(corpus.CorpusError, match="missing keys"):
        corpus.load_manifest(_imported_manifest([bad]))


def test_imported_trial_id_must_match_clip_id() -> None:
    bad = _imported_spoof("DF_E_1", "s1")
    bad["imported_provenance"]["official_trial_id"] = "DF_E_999"
    with pytest.raises(corpus.CorpusError, match="must equal clip_id"):
        corpus.load_manifest(_imported_manifest([bad]))


def test_imported_clip_must_be_eval_split() -> None:
    bad = _imported_spoof("DF_E_1", "s1", split="holdout")
    with pytest.raises(corpus.CorpusError, match="must have split 'eval'"):
        corpus.load_manifest(_imported_manifest([bad]))


def test_imported_official_split_must_be_eval() -> None:
    bad = _imported_spoof("DF_E_1", "s1")
    bad["imported_provenance"]["official_split"] = "progress"
    with pytest.raises(corpus.CorpusError, match="official_split must be 'eval'"):
        corpus.load_manifest(_imported_manifest([bad]))


def test_imported_stratum_must_match_label_and_codec() -> None:
    bad = _imported_spoof("DF_E_1", "s1", stratum="bona_fide|low_mp3")
    with pytest.raises(corpus.CorpusError, match="stratum"):
        corpus.load_manifest(_imported_manifest([bad]))


@pytest.mark.parametrize("sentinel", ["unknown", "n/a", "-", "None"])
def test_imported_manifest_benchmark_sentinel_rejected(sentinel: str) -> None:
    m = _imported_manifest([_imported_spoof("DF_E_1", "s1")], benchmark=sentinel)
    with pytest.raises(corpus.CorpusError, match=r"placeholder|non-empty benchmark"):
        corpus.load_manifest(m)
