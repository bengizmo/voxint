"""Unit tests for ``tools/eval_attribution.py`` (#113 A4).

Covers: RTTM parsing, protocol inspection, align pipeline, score (including
the frozen regression pack), report rendering, and CLI error paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from tools.eval_attribution import (
    EvalError,
    deserialize_trial,
    main,
    parse_rttm_intervals,
    render_report,
)

from voxint.harness.attribution_aligner import Interval

FIXTURES = Path(__file__).resolve().parents[1] / "parity" / "fixtures" / "attribution"


# --------------------------------------------------------------------------- #
# RTTM parsing
# --------------------------------------------------------------------------- #
class TestParseRttmIntervals:
    def test_basic(self) -> None:
        rttm = (
            "SPEAKER file1 1 1.0 2.0 <NA> <NA> alice <NA> <NA>\n"
            "SPEAKER file1 1 5.0 1.5 <NA> <NA> bob <NA> <NA>\n"
            "SPEAKER file1 1 3.5 1.0 <NA> <NA> alice <NA> <NA>\n"
        )
        result = parse_rttm_intervals(rttm)
        assert set(result) == {"alice", "bob"}
        assert len(result["alice"]) == 2
        assert result["alice"][0] == Interval(1.0, 3.0)
        assert result["alice"][1] == Interval(3.5, 4.5)
        assert result["bob"][0] == Interval(5.0, 6.5)

    def test_skips_comments_and_blank_lines(self) -> None:
        rttm = (
            ";; comment\n"
            "\n"
            "SPEAKER f 1 0.0 1.0 <NA> <NA> spk <NA> <NA>\n"
        )
        result = parse_rttm_intervals(rttm)
        assert len(result["spk"]) == 1

    def test_empty_file(self) -> None:
        assert parse_rttm_intervals("") == {}
        assert parse_rttm_intervals(";; only comments\n") == {}

    def test_non_positive_duration_rejected(self) -> None:
        rttm = "SPEAKER f 1 1.0 0.0 <NA> <NA> spk <NA> <NA>\n"
        with pytest.raises(EvalError, match="non-positive duration"):
            parse_rttm_intervals(rttm)

    def test_too_few_fields_rejected(self) -> None:
        rttm = "SPEAKER f 1 1.0 2.0 <NA> <NA>\n"
        with pytest.raises(EvalError, match="expected >=9 fields"):
            parse_rttm_intervals(rttm)

    def test_bad_float_rejected(self) -> None:
        rttm = "SPEAKER f 1 abc 2.0 <NA> <NA> spk <NA> <NA>\n"
        with pytest.raises(EvalError, match="bad start/duration"):
            parse_rttm_intervals(rttm)


# --------------------------------------------------------------------------- #
# protocol subcommand
# --------------------------------------------------------------------------- #
def _minimal_protocol() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "corpus": "ami_ihm",
        "truth_source": "corpus_gold",
        "selection_seed": "test-seed",
        "rows": [
            {
                "meeting_id": "ES2002a",
                "nxt_agent": "A",
                "channel": 0,
                "host_global_name": "MEE068",
                "base_session_id": "ES2002",
            }
        ],
        "recurrence_report": {
            "n_meetings": 1,
            "n_base_sessions": 1,
            "n_participants": 1,
            "n_cross_session_speakers": 0,
            "n_genuine_pairs": 0,
            "n_impostor_pairs": 0,
            "baseline_viable": False,
            "calibration_viable": False,
        },
        "exclusions": [],
    }


def test_cmd_protocol(tmp_path: Path) -> None:
    manifest = tmp_path / "protocol.json"
    manifest.write_text(json.dumps(_minimal_protocol()), encoding="utf-8")
    rc = main(["protocol", "--manifest", str(manifest)])
    assert rc == 0


# --------------------------------------------------------------------------- #
# align subcommand
# --------------------------------------------------------------------------- #
def _make_align_bundle(tmp_path: Path) -> Path:
    """Create a minimal self-contained align bundle and return the manifest path."""
    protocol = _minimal_protocol()
    protocol["rows"] = [
        {
            "meeting_id": "M001",
            "nxt_agent": "A",
            "channel": 0,
            "host_global_name": "speaker-alpha",
            "base_session_id": "M001",
        },
        {
            "meeting_id": "M001",
            "nxt_agent": "B",
            "channel": 1,
            "host_global_name": "speaker-beta",
            "base_session_id": "M001",
        },
    ]
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")

    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    (gold_dir / "M001.rttm").write_text(
        "SPEAKER M001 1 0.0 10.0 <NA> <NA> speaker-alpha <NA> <NA>\n"
        "SPEAKER M001 1 12.0 15.0 <NA> <NA> speaker-beta <NA> <NA>\n",
        encoding="utf-8",
    )

    hyp_dir = tmp_path / "hyp"
    hyp_dir.mkdir()
    (hyp_dir / "M001.rttm").write_text(
        "SPEAKER M001 1 0.5 9.0 <NA> <NA> SPEAKER_00 <NA> <NA>\n"
        "SPEAKER M001 1 9.5 1.0 <NA> <NA> SPEAKER_00 <NA> <NA>\n"
        "SPEAKER M001 1 12.5 14.0 <NA> <NA> SPEAKER_01 <NA> <NA>\n",
        encoding="utf-8",
    )

    align_input = {
        "schema_version": 1,
        "kind": "attribution_align_input",
        "protocol_path": "protocol.json",
        "meetings": {
            "M001": {
                "gold_rttm": "gold/M001.rttm",
                "hypothesis_rttm": "hyp/M001.rttm",
            }
        },
        "match_evidence": {
            "M001": {
                "SPEAKER_00": {
                    "run_id": "run-1",
                    "top_speaker_id": "enrolled-alpha",
                    "similarity": 0.80,
                    "margin": 0.15,
                    "vote_agreement": 0.75,
                    "eligible_turns": 3,
                    "eligible_seconds": 12.0,
                    "roster_size": 3,
                },
                "SPEAKER_01": {
                    "run_id": "run-1",
                    "top_speaker_id": "enrolled-beta",
                    "similarity": 0.72,
                    "margin": 0.10,
                    "vote_agreement": 0.68,
                    "eligible_turns": 4,
                    "eligible_seconds": 16.0,
                    "roster_size": 3,
                },
            }
        },
        "enrolled_speaker_map": {
            "speaker-alpha": "enrolled-alpha",
            "speaker-beta": "enrolled-beta",
        },
    }

    manifest_path = tmp_path / "align-input.json"
    manifest_path.write_text(json.dumps(align_input), encoding="utf-8")
    return manifest_path


def test_cmd_align(tmp_path: Path) -> None:
    manifest = _make_align_bundle(tmp_path)
    out = tmp_path / "trials.json"
    rc = main(["align", "--manifest", str(manifest), "--out", str(out)])
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["kind"] == "attribution_trials"
    assert data["schema_version"] == 1
    assert "M001" in data["alignment"]
    assert len(data["trials"]) > 0
    for trial in data["trials"]:
        assert trial["meeting_id"] == "M001"
        assert "kind" in trial
        assert "slot_label" in trial


def test_cmd_align_produces_genuine_trials(tmp_path: Path) -> None:
    manifest = _make_align_bundle(tmp_path)
    out = tmp_path / "trials.json"
    main(["align", "--manifest", str(manifest), "--out", str(out)])
    data = json.loads(out.read_text(encoding="utf-8"))
    genuine = [t for t in data["trials"] if t["kind"] == "genuine"]
    assert len(genuine) >= 1


def test_align_bad_schema(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": 99, "kind": "wrong"}))
    rc = main(["align", "--manifest", str(bad)])
    assert rc == 2


def test_align_missing_meetings(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "attribution_align_input",
                "protocol_path": "p.json",
                "match_evidence": {},
                "enrolled_speaker_map": {},
            }
        )
    )
    rc = main(["align", "--manifest", str(bad)])
    assert rc == 2


# --------------------------------------------------------------------------- #
# score subcommand
# --------------------------------------------------------------------------- #
def test_cmd_score_from_align_output(tmp_path: Path) -> None:
    manifest = _make_align_bundle(tmp_path)
    trials_path = tmp_path / "trials.json"
    main(["align", "--manifest", str(manifest), "--out", str(trials_path)])

    metrics_path = tmp_path / "metrics.json"
    rc = main(["score", "--trials", str(trials_path), "--out", str(metrics_path)])
    assert rc == 0
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert data["kind"] == "attribution_metrics"
    assert data["schema_version"] == 1
    assert "far" in data
    assert "frr" in data
    assert "coverage" in data
    assert "alignment_attrition" in data


def test_cmd_score_bad_kind(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": 1, "kind": "wrong", "trials": []}))
    rc = main(["score", "--trials", str(bad)])
    assert rc == 2


def test_cmd_score_bad_schema_version(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {"schema_version": 99, "kind": "attribution_trials", "trials": []}
        )
    )
    rc = main(["score", "--trials", str(bad)])
    assert rc == 2


# --------------------------------------------------------------------------- #
# frozen regression pack: score determinism
# --------------------------------------------------------------------------- #
def test_frozen_regression_pack() -> None:
    """Score the committed synthetic trials and verify the output matches."""
    trials_path = FIXTURES / "synthetic_trials.json"
    expected_path = FIXTURES / "expected_metrics.json"
    assert trials_path.exists(), f"fixture missing: {trials_path}"
    assert expected_path.exists(), f"fixture missing: {expected_path}"

    trials_data = json.loads(trials_path.read_text(encoding="utf-8"))
    expected = json.loads(expected_path.read_text(encoding="utf-8"))

    from voxint.harness.attribution_aligner import aggregate_trials

    trials = [deserialize_trial(d) for d in trials_data["trials"]]
    summary = aggregate_trials(trials)

    assert summary.n_genuine_trials == expected["n_genuine_trials"]
    assert summary.n_impostor_trials == expected["n_impostor_trials"]
    assert summary.n_unscoreable == expected["n_unscoreable"]
    assert summary.n_auto_correct == expected["n_auto_correct"]
    assert summary.n_auto_wrong == expected["n_auto_wrong"]
    assert summary.n_review == expected["n_review"]
    assert summary.n_abstain == expected["n_abstain"]
    assert summary.far == pytest.approx(expected["far"])
    assert summary.far_ci_upper == pytest.approx(expected["far_ci_upper"])
    assert summary.frr == pytest.approx(expected["frr"])
    assert summary.frr_ci_upper == pytest.approx(expected["frr_ci_upper"])
    assert summary.coverage == pytest.approx(expected["coverage"])
    assert summary.n_speaker_clusters == expected["n_speaker_clusters"]
    assert summary.alignment_attrition == expected["alignment_attrition"]


# --------------------------------------------------------------------------- #
# deserialize_trial round-trip
# --------------------------------------------------------------------------- #
def test_deserialize_trial_genuine() -> None:
    d = {
        "run_id": "r1",
        "label": "SPK_00",
        "similarity": 0.85,
        "margin": 0.20,
        "vote_agreement": 0.90,
        "eligible_turns": 5,
        "eligible_seconds": 25.0,
        "roster_size": 4,
        "top_speaker_id": "enrolled-A",
        "kind": "genuine",
        "truth_anchoring": "corpus_gold",
        "cluster_id": "gold-A",
        "slot_label": "SPK_00",
        "slot_classification": "genuine",
        "slot_purity": 0.95,
        "slot_coverage": 0.80,
        "slot_margin": 0.70,
        "slot_duration": 30.0,
    }
    at = deserialize_trial(d)
    assert at.trial.kind.value == "genuine"
    assert at.trial.similarity == 0.85
    assert at.slot_label == "SPK_00"
    assert at.alignment.purity == 0.95


def test_deserialize_trial_unscoreable() -> None:
    d = {
        "run_id": "r1",
        "label": "SPK_00",
        "similarity": None,
        "margin": None,
        "vote_agreement": None,
        "eligible_turns": 0,
        "eligible_seconds": 0.0,
        "roster_size": None,
        "top_speaker_id": None,
        "kind": "unscoreable",
        "truth_anchoring": "corpus_gold",
        "cluster_id": "__unscoreable__:SPK_00",
        "slot_label": "SPK_00",
        "slot_classification": "no_gold_overlap",
    }
    at = deserialize_trial(d)
    assert at.trial.kind.value == "unscoreable"
    assert at.trial.similarity is None


# --------------------------------------------------------------------------- #
# report subcommand
# --------------------------------------------------------------------------- #
def _sample_metrics() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "attribution_metrics",
        "n_genuine_trials": 5,
        "n_impostor_trials": 3,
        "n_unscoreable": 2,
        "n_auto_correct": 3,
        "n_auto_wrong": 1,
        "n_review": 2,
        "n_abstain": 2,
        "far": 0.333,
        "far_ci_upper": 0.792,
        "frr": 0.4,
        "frr_ci_upper": 0.769,
        "coverage": 0.5,
        "n_speaker_clusters": 7,
        "alignment_attrition": {
            "genuine": 8,
            "impostor": 0,
            "mixed": 0,
            "no_gold_overlap": 1,
            "unscoreable_coverage": 0,
            "unscoreable_eligibility": 1,
            "unscoreable_margin": 0,
            "unscoreable_purity": 0,
        },
        "environment": {"git_sha": "abc123", "voxint_version": "0.35.0"},
    }


def test_render_report_single_run() -> None:
    text = render_report("2026-09-05", [_sample_metrics()])
    assert "Speaker-attribution baseline" in text
    assert "FAR" in text
    assert "FRR" in text
    assert "Limitations" in text
    assert "Noise floor" not in text


def test_render_report_noise_floor() -> None:
    m1 = _sample_metrics()
    m2 = _sample_metrics()
    m2["far"] = 0.35
    m2["frr"] = 0.38
    text = render_report("2026-09-05", [m1, m2])
    assert "Noise floor (2 zero-change runs)" in text


def test_render_report_alignment_attrition() -> None:
    text = render_report("2026-09-05", [_sample_metrics()])
    assert "Alignment attrition" in text
    assert "genuine" in text


def test_render_report_empty() -> None:
    with pytest.raises(EvalError, match="no runs supplied"):
        render_report("2026-09-05", [])


def test_render_report_bad_kind() -> None:
    bad = _sample_metrics()
    bad["kind"] = "wrong"
    with pytest.raises(EvalError, match="attribution_metrics"):
        render_report("2026-09-05", [bad])


def test_cmd_report(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(_sample_metrics()), encoding="utf-8")
    out = tmp_path / "report.md"
    rc = main(
        ["report", "--run", str(metrics_path), "--date", "2026-09-05", "--out", str(out)]
    )
    assert rc == 0
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# Speaker-attribution baseline")


# --------------------------------------------------------------------------- #
# end-to-end: align -> score -> report pipeline
# --------------------------------------------------------------------------- #
def test_end_to_end_pipeline(tmp_path: Path) -> None:
    manifest = _make_align_bundle(tmp_path)
    trials_path = tmp_path / "trials.json"
    assert main(["align", "--manifest", str(manifest), "--out", str(trials_path)]) == 0

    metrics_path = tmp_path / "metrics.json"
    assert main(["score", "--trials", str(trials_path), "--out", str(metrics_path)]) == 0

    report_path = tmp_path / "report.md"
    assert (
        main(
            [
                "report",
                "--run",
                str(metrics_path),
                "--date",
                "2026-09-05",
                "--out",
                str(report_path),
            ]
        )
        == 0
    )

    report = report_path.read_text(encoding="utf-8")
    assert "Speaker-attribution baseline" in report
    assert "FAR" in report

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "gates" in metrics


# --------------------------------------------------------------------------- #
# error paths (review findings M4)
# --------------------------------------------------------------------------- #
def test_score_missing_file(tmp_path: Path) -> None:
    rc = main(["score", "--trials", str(tmp_path / "nonexistent.json")])
    assert rc == 2


def test_protocol_missing_file(tmp_path: Path) -> None:
    rc = main(["protocol", "--manifest", str(tmp_path / "nonexistent.json")])
    assert rc == 2


def test_deserialize_missing_required_fields() -> None:
    with pytest.raises(EvalError, match="missing required fields"):
        deserialize_trial({"run_id": "r1"})


def test_negative_rttm_start() -> None:
    rttm = "SPEAKER f 1 -1.0 2.0 <NA> <NA> spk <NA> <NA>\n"
    with pytest.raises(EvalError, match="invalid start"):
        parse_rttm_intervals(rttm)


def test_align_output_has_dominant_gold_speaker(tmp_path: Path) -> None:
    manifest = _make_align_bundle(tmp_path)
    out = tmp_path / "trials.json"
    main(["align", "--manifest", str(manifest), "--out", str(out)])
    data = json.loads(out.read_text(encoding="utf-8"))
    for trial in data["trials"]:
        assert "dominant_gold_speaker" in trial
