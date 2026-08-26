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
from decimal import Decimal
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


@pytest.mark.parametrize(
    "end_s,expected_end",
    [
        # These decimals x 16000 are exact integers, so ceil must NOT round up. In
        # binary float, 0.1*16000 == 1600.0000000000002 -> ceil 1601 (the bug).
        ("0.1", 1600),
        ("0.2", 3200),
        ("0.3", 4800),
        ("0.6", 9600),
        ("0.7", 11200),
    ],
)
def test_to_sample_interval_decimal_boundary_exact(end_s: str, expected_end: int) -> None:
    # Exact decimal arithmetic: an ordinary RTTM decimal maps to the exact sample.
    iv = corpus.to_sample_interval(Decimal("0.0"), Decimal(end_s))
    assert iv.end_sample == expected_end


def test_to_sample_interval_touching_turns_share_no_sample() -> None:
    # Two different-speaker turns that touch at 0.1 s must partition the samples,
    # never share sample 1600 (the float ceil-overcount would put 1600 in both).
    first = corpus.to_sample_interval(Decimal("0.0"), Decimal("0.1"))
    second = corpus.to_sample_interval(Decimal("0.1"), Decimal("0.2"))
    assert first.end_sample == second.start_sample == 1600
    assert first.n_samples == second.n_samples == 1600


def test_to_sample_interval_float_caller_recovers_intent() -> None:
    # A float caller (0.1) is routed through str, so it still lands on sample 1600.
    assert corpus.to_sample_interval(0.0, 0.1).end_sample == 1600


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
def _iv(*pairs: tuple[str, str]) -> list[tuple[Decimal, Decimal]]:
    return [(Decimal(a), Decimal(b)) for a, b in pairs]


def test_merge_intervals_gap_and_overlap() -> None:
    merged = corpus._merge_intervals(
        _iv(("0.0", "0.5"), ("0.6", "1.0"), ("5.0", "6.0"), ("5.5", "5.7")), Decimal("0.3")
    )
    assert merged == _iv(("0.0", "1.0"), ("5.0", "6.0"))


def test_merge_intervals_gap_boundary_not_merged() -> None:
    # gap exactly == gap_s is NOT < gap_s, so it stays split.
    assert corpus._merge_intervals(
        _iv(("0.0", "1.0"), ("1.3", "2.0")), Decimal("0.3")
    ) == _iv(("0.0", "1.0"), ("1.3", "2.0"))


def test_coalesce_intervals_unions_touching_and_overlapping() -> None:
    # Touching (1.0==1.0) and overlapping (2.5<3.0) both union into one region.
    merged = corpus._coalesce_intervals(_iv(("0.0", "1.0"), ("1.0", "2.0"), ("2.5", "3.0")))
    assert merged == _iv(("0.0", "2.0"), ("2.5", "3.0"))


def test_subtract_overlaps_removes_other_speaker() -> None:
    subs = corpus._subtract_overlaps(
        (Decimal("0.0"), Decimal("10.0")), _iv(("4.0", "6.0")), floor_s=Decimal("0.1")
    )
    assert subs == _iv(("0.0", "4.0"), ("6.0", "10.0"))


def test_subtract_overlaps_ignores_sub_floor_graze() -> None:
    # a 50 ms touch is below the 100 ms floor -> not subtracted.
    assert corpus._subtract_overlaps(
        (Decimal("0.0"), Decimal("10.0")), _iv(("4.0", "4.05")), floor_s=Decimal("0.1")
    ) == _iv(("0.0", "10.0"))


# --------------------------------------------------------------------------- #
# _clean_spans
# --------------------------------------------------------------------------- #
def test_clean_spans_merges_same_speaker_gap() -> None:
    turns = corpus.parse_rttm(
        _rttm(("m1", 0.0, 0.5, "A"), ("m1", 0.6, 0.5, "A"), ("m1", 5.0, 0.2, "A"))
    )
    # _clean_spans merges but no longer floors (the length floor is sample-domain,
    # applied downstream): [0,0.5]+[0.6,1.1] merge to [0,1.1]; [5,5.2] stays.
    cleaned = corpus._clean_spans(turns, merge_gap_s=Decimal("0.3"), overlap_floor_s=Decimal("0.1"))
    assert cleaned == {"A": [(Decimal("0.0"), Decimal("1.1")), (Decimal("5.0"), Decimal("5.2"))]}


def test_clean_spans_drops_other_speaker_overlap() -> None:
    turns = corpus.parse_rttm(
        _rttm(("m1", 5.0, 6.0, "A"), ("m1", 5.5, 0.3, "B"))
    )
    cleaned = corpus._clean_spans(turns, merge_gap_s=Decimal("0.3"), overlap_floor_s=Decimal("0.1"))
    # A [5,11] minus B [5.5,5.8]: [5,5.5] + [5.8,11]. B [5.5,5.8] fully inside A -> gone.
    assert list(cleaned) == ["A"]
    assert cleaned["A"] == [(Decimal("5.0"), Decimal("5.5")), (Decimal("5.8"), Decimal("11.0"))]


def test_clean_spans_coalesces_word_level_other_speaker() -> None:
    # A holds [0,5]; B says six contiguous 80 ms words covering [1.0,1.48]. Each word
    # alone is 0.08s (< the 0.1s floor), but coalesced they are a 0.48s continuous
    # region that MUST be cut -- the split-leakage guard for word-level RTTMs.
    rows = [("m1", 0.0, 5.0, "A")]
    t = 1.0
    for _ in range(6):
        rows.append(("m1", round(t, 2), 0.08, "B"))
        t += 0.08
    turns = corpus.parse_rttm(_rttm(*rows))
    cleaned = corpus._clean_spans(turns, merge_gap_s=Decimal("0.3"), overlap_floor_s=Decimal("0.1"))
    # A is split around the coalesced [1.0,1.48] B region; no A span contains it.
    assert cleaned["A"] == [(Decimal("0.0"), Decimal("1.0")), (Decimal("1.48"), Decimal("5.0"))]
    for start, end in cleaned["A"]:
        assert not (start < Decimal("1.48") and end > Decimal("1.0"))


def test_clean_spans_keeps_single_sub_floor_graze() -> None:
    # A single 50 ms other-speaker touch stays below the floor and does not fragment.
    turns = corpus.parse_rttm(_rttm(("m1", 0.0, 5.0, "A"), ("m1", 2.0, 0.05, "B")))
    cleaned = corpus._clean_spans(turns, merge_gap_s=Decimal("0.3"), overlap_floor_s=Decimal("0.1"))
    assert cleaned["A"] == [(Decimal("0.0"), Decimal("5.0"))]


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


def test_build_plan_segment_kept_at_exact_model_width() -> None:
    # A single-speaker turn of exactly one model window (64,600 samples = 4.0375 s)
    # must survive as a segment; the length floor is enforced in samples, not the
    # float seconds quotient (which would drop a genuinely-64,600-sample span).
    turns = corpus.parse_rttm(_rttm(("m1", 0.0, 4.0375, "A")))
    plan = corpus.build_plan(corpus.ORGANIC_SOURCES["ami"], {"m1": turns})
    assert len(plan.segments) == 1
    assert plan.segments[0].interval.n_samples == corpus.MODEL_WIDTH_SAMPLES


def test_build_plan_all_short_rejected() -> None:
    # Every span below the floor -> no clips survive -> fail closed (not an empty plan).
    turns = corpus.parse_rttm(_rttm(("m1", 0.0, 0.5, "A")))
    with pytest.raises(corpus.CorpusError, match="no clips survived"):
        corpus.build_plan(corpus.ORGANIC_SOURCES["ami"], {"m1": turns})


def test_build_plan_recording_key_mismatch_rejected() -> None:
    turns = corpus.parse_rttm(_rttm(("real", 0.0, 3.0, "A")))
    with pytest.raises(corpus.CorpusError, match="has a turn for"):
        corpus.build_plan(corpus.ORGANIC_SOURCES["ami"], {"wrongkey": turns})


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
    measured = _measured(plan.turn_clips)
    manifest = corpus.finalize_manifest(plan.turn_clips, measured)
    assert manifest.schema_version == corpus.MANIFEST_SCHEMA_VERSION
    assert manifest.corpus_kind == corpus.CORPUS_KIND_SYNTHESIS
    assert len(manifest.clips) == len(plan.turn_clips)
    clip = manifest.clips[0]
    assert clip.label == "bona_fide"
    assert clip.generator is None
    # Identity comes ONLY from the measured facts: the sha is the measured sha and
    # the duration is the measured sample count / 16000, never the planned interval.
    measured_sha, measured_samples = measured[clip.clip_id]
    assert clip.sha256 == measured_sha
    assert clip.duration_s == pytest.approx(measured_samples / corpus.CANONICAL_SAMPLE_RATE)


def test_finalize_manifest_duration_tracks_measured_not_plan() -> None:
    # A measured sample count that differs from the planned interval must drive the
    # duration; the plan's interval length must NOT leak in.
    plan = _sample_plan()
    record = plan.turn_clips[0]
    doctored = record.interval.n_samples + 8000  # 0.5 s longer than the plan
    manifest = corpus.finalize_manifest(
        [record], {record.clip_id: (hashlib.sha256(b"x").hexdigest(), doctored)}
    )
    assert manifest.clips[0].duration_s == pytest.approx(doctored / corpus.CANONICAL_SAMPLE_RATE)


def test_finalize_manifest_orphan_fact_rejected() -> None:
    plan = _sample_plan()
    measured = _measured(plan.turn_clips)
    measured["ghost-clip"] = ("c" * 64, 16000)  # a measurement for no record
    with pytest.raises(corpus.CorpusError, match="not in the record list"):
        corpus.finalize_manifest(plan.turn_clips, measured)


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


def test_cli_degrade_skips_already_degraded_parents(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    root = {
        "clip_id": "ami-m1-A-turn-0-80000",
        "rel_path": "ami/turn/ami-m1-A-turn-0-80000.wav",
        "sha256": "a" * 64, "duration_s": 5.0, "label": "bona_fide", "language": "en",
        "license_spdx": "CC-BY-4.0", "stratum": "bona_fide|organic|meetingroom",
        "source": "ami", "speaker_id": "ami-m1-A", "split": "calibration",
        "generator": None, "degradation": None, "parent_clip_id": None, "acquire": None,
    }
    child = {
        **root,
        "clip_id": "ami-m1-A-turn-0-80000-mp3-cbr48-v1",
        "rel_path": "ami/turn/degraded/ami-m1-A-turn-0-80000-mp3-cbr48-v1.wav",
        "sha256": "b" * 64,
        "stratum": "bona_fide|organic|meetingroom|mp3-cbr48-v1",
        "degradation": "mp3-cbr48-v1", "parent_clip_id": "ami-m1-A-turn-0-80000",
    }
    manifest = tmp_path / "parent.json"
    manifest.write_text(json.dumps({"schema_version": 1, "clips": [root, child]}), encoding="utf-8")
    rc = corpus.main(["degrade", "--manifest", str(manifest), "--recipe", "mp3-cbr48-v1"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    # Only the root parent is degraded; the existing child never becomes a grandchild.
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
    assert "no non-degraded parent clips" in capsys.readouterr().err
