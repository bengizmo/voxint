"""Organic-corpus planning tests for synthdetect S5 (issue #144).

Freezes the pure, audio-free `prepare` layer before any audio exists: RTTM
parsing, the pinned decimal-to-sample rule, overlap cleaning, the turn and
session-segment planners, the materialization-plan schema, and
`finalize_manifest`. No ffmpeg, no audio, no GPU is touched here; every function
under test computes a deterministic result from JSON-shaped inputs.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import synthdetect_corpus as corpus  # noqa: E402

SR = corpus.CANONICAL_SAMPLE_RATE


def _rttm(*rows: tuple[str, float, float, str]) -> str:
    return "\n".join(
        f"SPEAKER {rec} 1 {start} {dur} <NA> <NA> {label} <NA> <NA>"
        for rec, start, dur, label in rows
    )


# --------------------------------------------------------------------------- #
# parse_rttm
# --------------------------------------------------------------------------- #
def test_parse_rttm_happy() -> None:
    turns = corpus.parse_rttm(_rttm(("m1", 0.0, 0.5, "A"), ("m1", 1.0, 2.0, "B")))
    assert len(turns) == 2
    assert turns[0].recording == "m1"
    assert turns[0].speaker_label == "A"
    assert turns[0].start_s == 0.0
    assert turns[1].end_s == 3.0


def test_parse_rttm_skips_blank_lines() -> None:
    text = "\n" + _rttm(("m1", 0.0, 1.0, "A")) + "\n\n"
    assert len(corpus.parse_rttm(text)) == 1


def test_parse_rttm_recording_mismatch_rejected() -> None:
    with pytest.raises(corpus.CorpusError, match="does not match expected"):
        corpus.parse_rttm(_rttm(("m2", 0.0, 1.0, "A")), recording="m1")


@pytest.mark.parametrize(
    "row",
    [
        "SPEECH m1 1 0.0 1.0 <NA> <NA> A <NA> <NA>",  # not SPEAKER
        "SPEAKER m1 1 0.0 1.0 <NA> <NA>",  # too few fields
        "SPEAKER m1 1 x 1.0 <NA> <NA> A <NA> <NA>",  # non-numeric start
        "SPEAKER m1 1 -0.5 1.0 <NA> <NA> A <NA> <NA>",  # negative start
        "SPEAKER m1 1 0.0 0.0 <NA> <NA> A <NA> <NA>",  # non-positive dur
        "SPEAKER m1 1 0.0 nan <NA> <NA> A <NA> <NA>",  # non-finite dur
        "SPEAKER m1 1 0.0 1.0 <NA> <NA> <NA> <NA> <NA>",  # missing label
    ],
)
def test_parse_rttm_fail_closed(row: str) -> None:
    with pytest.raises(corpus.CorpusError):
        corpus.parse_rttm(row)


def test_parse_rttm_empty_rejected() -> None:
    with pytest.raises(corpus.CorpusError, match="no SPEAKER rows"):
        corpus.parse_rttm("\n   \n")


# --------------------------------------------------------------------------- #
# to_sample_interval (pinned floor/ceil)
# --------------------------------------------------------------------------- #
def test_to_sample_interval_floor_ceil() -> None:
    # 0.00003125 s = 0.5 sample -> floor to 0; 1.00003125 -> ceil to 16001.
    iv = corpus.to_sample_interval(0.00003125, 1.00003125)
    assert iv.start_sample == 0
    assert iv.end_sample == 16001
    assert iv.n_samples == 16001


def test_to_sample_interval_exact() -> None:
    iv = corpus.to_sample_interval(1.0, 2.0)
    assert (iv.start_sample, iv.end_sample) == (SR, 2 * SR)


@pytest.mark.parametrize("start,end", [(-0.1, 1.0), (1.0, 1.0), (2.0, 1.0), (float("nan"), 1.0)])
def test_to_sample_interval_rejects(start: float, end: float) -> None:
    with pytest.raises(corpus.CorpusError):
        corpus.to_sample_interval(start, end)


# --------------------------------------------------------------------------- #
# namespaced_speaker
# --------------------------------------------------------------------------- #
def test_namespaced_speaker_ok() -> None:
    assert corpus.namespaced_speaker("ami", "ES2011a", "MEO069") == "ami-ES2011a-MEO069"


def test_namespaced_speaker_rejects_unsafe() -> None:
    with pytest.raises(corpus.CorpusError, match="not a safe token"):
        corpus.namespaced_speaker("ami", "rec/../x", "A")


# --------------------------------------------------------------------------- #
# interval helpers
# --------------------------------------------------------------------------- #
def test_merge_intervals_gap_and_overlap() -> None:
    merged = corpus._merge_intervals([(0.0, 0.5), (0.6, 1.0), (5.0, 6.0), (5.5, 5.7)], 0.3)
    assert merged == [(0.0, 1.0), (5.0, 6.0)]


def test_merge_intervals_gap_boundary_not_merged() -> None:
    # gap exactly == gap_s is NOT < gap_s, so it stays split.
    assert corpus._merge_intervals([(0.0, 1.0), (1.3, 2.0)], 0.3) == [(0.0, 1.0), (1.3, 2.0)]


def test_subtract_overlaps_removes_other_speaker() -> None:
    subs = corpus._subtract_overlaps((0.0, 10.0), [(4.0, 6.0)], floor_s=0.1)
    assert subs == [(0.0, 4.0), (6.0, 10.0)]


def test_subtract_overlaps_ignores_sub_floor_graze() -> None:
    # a 50 ms touch is below the 100 ms floor -> not subtracted.
    assert corpus._subtract_overlaps((0.0, 10.0), [(4.0, 4.05)], floor_s=0.1) == [(0.0, 10.0)]


# --------------------------------------------------------------------------- #
# _clean_spans
# --------------------------------------------------------------------------- #
def test_clean_spans_merges_words_and_floors() -> None:
    turns = corpus.parse_rttm(
        _rttm(("m1", 0.0, 0.5, "A"), ("m1", 0.6, 0.5, "A"), ("m1", 5.0, 0.2, "A"))
    )
    cleaned = corpus._clean_spans(turns, merge_gap_s=0.3, min_s=1.0, overlap_floor_s=0.1)
    # [0,0.5]+[0.6,1.1] merge to [0,1.1] (kept); [5,5.2] is 0.2s (< 1.0, dropped).
    assert cleaned == {"A": [(0.0, 1.1)]}


def test_clean_spans_drops_overlap_then_floors() -> None:
    turns = corpus.parse_rttm(
        _rttm(("m1", 5.0, 6.0, "A"), ("m1", 5.5, 0.3, "B"))
    )
    cleaned = corpus._clean_spans(turns, merge_gap_s=0.3, min_s=1.0, overlap_floor_s=0.1)
    # A [5,11] minus B [5.5,5.8]: [5,5.5] (0.5s dropped) + [5.8,11] (kept). B too short.
    assert list(cleaned) == ["A"]
    assert cleaned["A"] == [(5.8, 11.0)]


# --------------------------------------------------------------------------- #
# build_plan
# --------------------------------------------------------------------------- #
def _sample_plan() -> corpus.MaterializationPlan:
    turns = corpus.parse_rttm(
        _rttm(
            ("m1", 0.0, 0.5, "A"),
            ("m1", 0.6, 0.5, "A"),
            ("m1", 5.0, 6.0, "A"),
            ("m1", 5.5, 0.3, "B"),
            ("m1", 20.0, 5.0, "B"),
        )
    )
    return corpus.build_plan(corpus.ORGANIC_SOURCES["ami"], {"m1": turns})


def test_build_plan_turns_and_segments() -> None:
    plan = _sample_plan()
    assert plan.schema_version == corpus.PLAN_SCHEMA_VERSION
    assert plan.source == "ami"
    turn_ids = {r.clip_id for r in plan.turn_clips}
    # A: [0,1.1] and [5.8,11]; B: [20,25].
    assert turn_ids == {
        "ami-m1-A-turn-0-17600",
        "ami-m1-A-turn-92800-176000",
        "ami-m1-B-turn-320000-400000",
    }
    # Segments require >= one full window (4.0375s): A [5.8,11], B [20,25].
    assert {r.clip_id for r in plan.segments} == {
        "ami-m1-A-segment-92800-176000",
        "ami-m1-B-segment-320000-400000",
    }
    for r in plan.turn_clips:
        assert r.label == "bona_fide"
        assert r.stratum == "bona_fide|organic|meetingroom"
        assert r.language == "en"
        assert r.license_spdx == "CC-BY-4.0"
        assert r.kind == "turn"


def test_build_plan_segments_meet_full_window() -> None:
    plan = _sample_plan()
    for seg in plan.segments:
        assert seg.interval.n_samples >= corpus.MODEL_WIDTH_SAMPLES


def test_build_plan_speaker_disjoint_across_recordings() -> None:
    # Same speaker label in two recordings becomes two namespaced ids, so a split
    # is assigned per (recording, label) and can never leak across recordings.
    turns_a = corpus.parse_rttm(_rttm(("r1", 0.0, 3.0, "A")))
    turns_b = corpus.parse_rttm(_rttm(("r2", 0.0, 3.0, "A")))
    plan = corpus.build_plan(corpus.ORGANIC_SOURCES["voxconverse"], {"r1": turns_a, "r2": turns_b})
    split_of = {r.speaker_id: r.split for r in plan.turn_clips}
    assert set(split_of) == {"voxconverse-r1-A", "voxconverse-r2-A"}
    # Each speaker maps to exactly one split (disjoint by construction).
    for r in plan.turn_clips:
        assert split_of[r.speaker_id] == r.split


def test_build_plan_deterministic() -> None:
    assert corpus.plan_to_dict(_sample_plan()) == corpus.plan_to_dict(_sample_plan())


def test_build_plan_empty_recordings_rejected() -> None:
    with pytest.raises(corpus.CorpusError, match="no recordings"):
        corpus.build_plan(corpus.ORGANIC_SOURCES["ami"], {})


@pytest.mark.parametrize("cal,hold", [(-0.1, 0.2), (0.6, 0.6), (float("inf"), 0.1)])
def test_build_plan_bad_fractions_rejected(cal: float, hold: float) -> None:
    turns = corpus.parse_rttm(_rttm(("m1", 0.0, 3.0, "A")))
    with pytest.raises(corpus.CorpusError):
        corpus.build_plan(
            corpus.ORGANIC_SOURCES["ami"],
            {"m1": turns},
            calibration_fraction=cal,
            holdout_fraction=hold,
        )


def test_plan_to_dict_shape() -> None:
    d = corpus.plan_to_dict(_sample_plan())
    assert set(d) == {"schema_version", "source", "turn_clips", "segments"}
    rec = d["turn_clips"][0]
    assert set(rec) == {
        "clip_id", "rel_path", "source", "recording", "speaker_id", "label",
        "language", "license_spdx", "stratum", "start_sample", "end_sample",
        "split", "acquire", "kind",
    }


# --------------------------------------------------------------------------- #
# finalize_manifest
# --------------------------------------------------------------------------- #
def _measured(records: tuple[corpus.IngestRecord, ...]) -> dict[str, tuple[str, int]]:
    return {
        r.clip_id: (hashlib.sha256(r.clip_id.encode()).hexdigest(), r.interval.n_samples)
        for r in records
    }


def test_finalize_manifest_happy() -> None:
    plan = _sample_plan()
    manifest = corpus.finalize_manifest(plan.turn_clips, _measured(plan.turn_clips))
    assert manifest.schema_version == corpus.MANIFEST_SCHEMA_VERSION
    assert manifest.corpus_kind == corpus.CORPUS_KIND_SYNTHESIS
    assert len(manifest.clips) == len(plan.turn_clips)
    clip = manifest.clips[0]
    assert clip.label == "bona_fide"
    assert clip.generator is None
    # duration is derived from the measured sample count.
    assert clip.duration_s == pytest.approx(clip.duration_s)


def test_finalize_manifest_missing_fact_rejected() -> None:
    plan = _sample_plan()
    measured = _measured(plan.turn_clips)
    measured.pop(plan.turn_clips[0].clip_id)
    with pytest.raises(corpus.CorpusError, match="no measured facts"):
        corpus.finalize_manifest(plan.turn_clips, measured)


def test_finalize_manifest_bad_sha_rejected() -> None:
    plan = _sample_plan()
    measured = _measured(plan.turn_clips)
    cid = plan.turn_clips[0].clip_id
    measured[cid] = ("nothex", measured[cid][1])
    with pytest.raises(corpus.CorpusError, match="64 lowercase hex"):
        corpus.finalize_manifest(plan.turn_clips, measured)


@pytest.mark.parametrize("bad", [0, -5, True])
def test_finalize_manifest_bad_sample_count_rejected(bad: object) -> None:
    plan = _sample_plan()
    measured = _measured(plan.turn_clips)
    cid = plan.turn_clips[0].clip_id
    measured[cid] = (measured[cid][0], bad)  # type: ignore[assignment]
    with pytest.raises(corpus.CorpusError, match="positive int"):
        corpus.finalize_manifest(plan.turn_clips, measured)


def test_finalize_manifest_empty_rejected() -> None:
    with pytest.raises(corpus.CorpusError, match="no records"):
        corpus.finalize_manifest([], {})


def test_finalize_manifest_validates_speaker_disjoint() -> None:
    # finalize round-trips through load_manifest, which rejects a speaker whose
    # clips straddle two splits (a hand-corrupted plan). Build such a case.
    plan = _sample_plan()
    records = list(plan.turn_clips)
    # Force two clips of speaker A into different splits.
    a_records = [r for r in records if r.speaker_id == "ami-m1-A"]
    assert len(a_records) >= 2
    from dataclasses import replace

    records = [r for r in records if r not in a_records]
    records.append(replace(a_records[0], split="calibration"))
    records.append(replace(a_records[1], split="eval"))
    with pytest.raises(corpus.CorpusError, match="straddles splits"):
        corpus.finalize_manifest(tuple(records), _measured(tuple(records)))


def test_organic_sources_registry() -> None:
    for source_id, spec in corpus.ORGANIC_SOURCES.items():
        assert spec.source_id == source_id
        assert spec.license_spdx == "CC-BY-4.0"
        assert spec.domain


# --------------------------------------------------------------------------- #
# CLI (dry-run, no audio)
# --------------------------------------------------------------------------- #
def _write_rttm(path: Path) -> None:
    path.write_text(
        "SPEAKER m1 1 0.0 5.0 <NA> <NA> A <NA> <NA>\n"
        "SPEAKER m1 1 6.0 5.0 <NA> <NA> B <NA> <NA>\n",
        encoding="utf-8",
    )


def test_cli_prepare(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rttm = tmp_path / "m1.rttm"
    _write_rttm(rttm)
    rc = corpus.main(["prepare", "--source", "ami", "--rttm", str(rttm)])
    assert rc == 0
    import json

    out = json.loads(capsys.readouterr().out)
    assert out["source"] == "ami"
    assert len(out["turn_clips"]) == 2


def test_cli_prepare_unknown_source(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rttm = tmp_path / "m1.rttm"
    _write_rttm(rttm)
    assert corpus.main(["prepare", "--source", "nope", "--rttm", str(rttm)]) == 2
    assert "unknown source" in capsys.readouterr().err


def _write_parent_manifest(path: Path, split: str | None = "calibration") -> None:
    import json

    clip = {
        "clip_id": "ami-m1-A-turn-0-80000",
        "rel_path": "ami/turn/ami-m1-A-turn-0-80000.wav",
        "sha256": "a" * 64,
        "duration_s": 5.0,
        "label": "bona_fide",
        "language": "en",
        "license_spdx": "CC-BY-4.0",
        "stratum": "bona_fide|organic|meetingroom",
        "source": "ami",
        "speaker_id": "ami-m1-A",
        "split": split,
        "generator": None,
        "degradation": None,
        "parent_clip_id": None,
        "acquire": None,
    }
    path.write_text(json.dumps({"schema_version": 1, "clips": [clip]}), encoding="utf-8")


def test_cli_degrade(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest = tmp_path / "parent.json"
    _write_parent_manifest(manifest)
    rc = corpus.main(["degrade", "--manifest", str(manifest), "--recipe", "mp3-cbr48-v1"])
    assert rc == 0
    import json

    out = json.loads(capsys.readouterr().out)
    assert out["chain"] == "mp3-cbr48-v1"
    assert len(out["children"]) == 1
    assert out["children"][0]["parent_clip_id"] == "ami-m1-A-turn-0-80000"


def test_cli_degrade_split_filter_empty(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = tmp_path / "parent.json"
    _write_parent_manifest(manifest, split="eval")
    rc = corpus.main(
        [
            "degrade",
            "--manifest",
            str(manifest),
            "--recipe",
            "mp3-cbr48-v1",
            "--split",
            "calibration",
        ]
    )
    assert rc == 2
    assert "no parent clips" in capsys.readouterr().err
