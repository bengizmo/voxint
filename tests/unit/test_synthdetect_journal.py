"""Journal parsing + join/score/compare/calibrate tests for synthdetect (#144).

Freezes the fail-closed journal contract (header identity, score/skip XOR,
duplicate rejection) and the manifest join, then exercises score / compare /
calibrate end to end on tiny in-memory fixtures so a later edit cannot silently
change how a raw-score journal becomes a metric.
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
import synthdetect_eval as se  # noqa: E402

pytest.importorskip("sklearn", reason="synthdetect-eval extra (scikit-learn) not installed")

_SHA = "b" * 64
_MANIFEST_SHA = "c" * 64


def _header(**over: Any) -> dict[str, Any]:
    h = {
        "kind": "synthdetect_journal",
        "schema_version": se.JOURNAL_SCHEMA_VERSION,
        "inference_space": "synthdetect-w2v2aasist-v1",
        "model_id": "w2v2-aasist",
        "manifest_sha256": _MANIFEST_SHA,
        "windowing": {"pooling": "logit-mean", "window_s": 4.0},
        "runtime": {"torch": "2.4.0", "cudnn": "9", "image_digest": "sha256:x"},
        "flags": {"deterministic": True, "tf32": False},
    }
    h.update(over)
    return h


def _journal_text(header: dict[str, Any], results: list[dict[str, Any]]) -> str:
    return "\n".join([json.dumps(header), *[json.dumps(r) for r in results]]) + "\n"


def _clip(clip_id: str, label: str, score: float, split: str, stratum: str = "s") -> dict[str, Any]:
    rec: dict[str, Any] = {
        "clip_id": clip_id,
        "rel_path": f"x/{clip_id}.wav",
        "sha256": _SHA,
        "duration_s": 5.0,
        "label": label,
        "language": "en",
        "license_spdx": "CC0-1.0",
        "stratum": stratum,
        "source": "test",
        "speaker_id": f"spk_{clip_id}",
        "split": split,
        "generator": None,
        "degradation": None,
        "parent_clip_id": None,
        "acquire": None,
    }
    if label == "spoof":
        rec["generator"] = {"name": "piper", "version": "1", "checkpoint_sha": None,
                            "voice": "v", "seed": "1", "text_source": "t"}
    return rec


# --------------------------------------------------------------------------- #
# parse_journal fail-closed
# --------------------------------------------------------------------------- #
def test_parse_valid_journal() -> None:
    text = _journal_text(_header(), [{"clip_id": "c1", "raw_score": 0.5}])
    j = se.parse_journal(text)
    assert j.header["model_id"] == "w2v2-aasist"
    assert j.results[0].raw_score == 0.5


def test_empty_journal_rejected() -> None:
    with pytest.raises(se.EvalError, match="empty"):
        se.parse_journal("")


def test_header_missing_kind_rejected() -> None:
    bad = _header()
    del bad["kind"]
    with pytest.raises(se.EvalError, match="header"):
        se.parse_journal(_journal_text(bad, [{"clip_id": "c1", "raw_score": 0.1}]))


def test_header_bad_manifest_sha_rejected() -> None:
    with pytest.raises(se.EvalError, match="manifest_sha256"):
        se.parse_journal(_journal_text(_header(manifest_sha256="short"),
                                       [{"clip_id": "c1", "raw_score": 0.1}]))


def test_header_missing_windowing_rejected() -> None:
    bad = _header()
    del bad["windowing"]
    with pytest.raises(se.EvalError, match="windowing"):
        se.parse_journal(_journal_text(bad, [{"clip_id": "c1", "raw_score": 0.1}]))


def test_result_score_and_skip_both_rejected() -> None:
    with pytest.raises(se.EvalError, match="exactly one"):
        se.parse_journal(_journal_text(
            _header(), [{"clip_id": "c1", "raw_score": 0.1, "skip_reason": "too short"}]))


def test_result_neither_score_nor_skip_rejected() -> None:
    with pytest.raises(se.EvalError, match="exactly one"):
        se.parse_journal(_journal_text(_header(), [{"clip_id": "c1"}]))


def test_result_non_finite_score_rejected() -> None:
    # NaN is not valid JSON via dumps(allow_nan default True) -> emit token by hand.
    text = json.dumps(_header()) + '\n{"clip_id": "c1", "raw_score": NaN}\n'
    with pytest.raises(se.EvalError):
        se.parse_journal(text)


def test_duplicate_clip_id_rejected() -> None:
    with pytest.raises(se.EvalError, match="duplicate"):
        se.parse_journal(_journal_text(
            _header(),
            [{"clip_id": "c1", "raw_score": 0.1}, {"clip_id": "c1", "raw_score": 0.2}],
        ))


def test_header_only_no_results_rejected() -> None:
    with pytest.raises(se.EvalError, match="no clip results"):
        se.parse_journal(json.dumps(_header()) + "\n")


def test_skip_reason_result_parses() -> None:
    j = se.parse_journal(_journal_text(_header(), [{"clip_id": "c1", "skip_reason": "too short"}]))
    assert j.results[0].skip_reason == "too short"
    assert j.results[0].raw_score is None


def test_blank_lines_ignored() -> None:
    text = json.dumps(_header()) + "\n\n" + json.dumps({"clip_id": "c1", "raw_score": 0.3}) + "\n"
    j = se.parse_journal(text)
    assert len(j.results) == 1


# --------------------------------------------------------------------------- #
# join_scores
# --------------------------------------------------------------------------- #
def test_join_unknown_clip_rejected() -> None:
    manifest = corpus.load_manifest(
        {"schema_version": 1, "clips": [_clip("c1", "bona_fide", 0.1, "eval")]}
    )
    journal = se.parse_journal(_journal_text(_header(), [{"clip_id": "ghost", "raw_score": 0.5}]))
    with pytest.raises(se.EvalError, match="not in the manifest"):
        se.join_scores(journal, manifest, split=None)


def test_join_filters_by_split_and_counts_skips() -> None:
    clips = [
        _clip("c1", "bona_fide", 0.1, "eval"),
        _clip("c2", "spoof", 0.9, "eval"),
        _clip("c3", "bona_fide", 0.2, "calibration"),
    ]
    manifest = corpus.load_manifest({"schema_version": 1, "clips": clips})
    journal = se.parse_journal(_journal_text(_header(), [
        {"clip_id": "c1", "raw_score": 0.1},
        {"clip_id": "c2", "raw_score": 0.9},
        {"clip_id": "c3", "raw_score": 0.2},
    ]))
    scored, skipped = se.join_scores(journal, manifest, split="eval")
    assert {s.clip_id for s in scored} == {"c1", "c2"}
    assert skipped == []


# --------------------------------------------------------------------------- #
# score_report / compare / calibrate end to end
# --------------------------------------------------------------------------- #
def _eval_corpus() -> tuple:
    clips = []
    results = []
    # 20 bona fide (low scores), 20 spoof (high scores), all in eval.
    for i in range(20):
        clips.append(_clip(f"b{i}", "bona_fide", 0.0, "eval", stratum="clean"))
        results.append({"clip_id": f"b{i}", "raw_score": -2.0 - i * 0.05, "n_windows": 1})
    for i in range(20):
        clips.append(_clip(f"g{i}", "spoof", 0.0, "eval", stratum="clean"))
        results.append({"clip_id": f"g{i}", "raw_score": 2.0 + i * 0.05, "n_windows": 1})
    manifest = corpus.load_manifest({"schema_version": 1, "clips": clips})
    journal = se.parse_journal(_journal_text(_header(), results))
    return journal, manifest


def test_score_report_end_to_end() -> None:
    journal, manifest = _eval_corpus()
    report = se.score_report(journal, manifest, split="eval")
    assert report["overall"]["n"] == 40
    assert report["overall"]["eer"] == pytest.approx(0.0, abs=1e-6)
    assert report["skip_rate"] == 0.0
    assert "clean" in report["per_stratum"]


def test_score_report_with_skips() -> None:
    _journal, manifest = _eval_corpus()
    # Append a skipped clip to the manifest + journal.
    clips = list(manifest.clips)
    extra = corpus.load_manifest({"schema_version": 1, "clips": [
        _clip(c.clip_id, c.label, 0.0, "eval", stratum=c.stratum) for c in clips
    ] + [_clip("short1", "bona_fide", 0.0, "eval")]})
    text = _journal_text(_header(), [
        *[{"clip_id": c.clip_id, "raw_score": (2.0 if c.label == "spoof" else -2.0)}
          for c in clips],
        {"clip_id": "short1", "skip_reason": "too short to score"},
    ])
    report = se.score_report(se.parse_journal(text), extra, split="eval")
    assert report["n_skipped"] == 1
    assert report["skip_rate"] > 0.0


def test_single_class_stratum_reported_not_scored() -> None:
    clips = [
        _clip("b1", "bona_fide", 0.0, "eval", stratum="only_bona"),
        _clip("b2", "bona_fide", 0.0, "eval", stratum="only_bona"),
        _clip("g1", "spoof", 0.0, "eval", stratum="mixed"),
        _clip("b3", "bona_fide", 0.0, "eval", stratum="mixed"),
    ]
    manifest = corpus.load_manifest({"schema_version": 1, "clips": clips})
    journal = se.parse_journal(_journal_text(_header(), [
        {"clip_id": "b1", "raw_score": -1.0},
        {"clip_id": "b2", "raw_score": -1.1},
        {"clip_id": "g1", "raw_score": 1.0},
        {"clip_id": "b3", "raw_score": -1.0},
    ]))
    report = se.score_report(journal, manifest, split="eval")
    assert report["per_stratum"]["only_bona"]["single_class"] is True


def test_compare_journals_paired() -> None:
    j1 = se.parse_journal(_journal_text(_header(), [
        {"clip_id": "c1", "raw_score": 1.0}, {"clip_id": "c2", "raw_score": 2.0}]))
    j2 = se.parse_journal(_journal_text(_header(), [
        {"clip_id": "c1", "raw_score": 1.1}, {"clip_id": "c2", "raw_score": 1.9}]))
    result = se.compare_journals(j1, j2, decision_threshold=0.0)
    assert result["n_common"] == 2
    assert result["max_abs_delta"] == pytest.approx(0.1)
    assert result["decision_agreement"] == 1.0


def test_compare_no_common_rejected() -> None:
    j1 = se.parse_journal(_journal_text(_header(), [{"clip_id": "c1", "raw_score": 1.0}]))
    j2 = se.parse_journal(_journal_text(_header(), [{"clip_id": "c2", "raw_score": 1.0}]))
    with pytest.raises(se.EvalError, match="no scored clips"):
        se.compare_journals(j1, j2, decision_threshold=0.0)


def test_calibrate_policy_end_to_end() -> None:
    clips = []
    results = []
    for i in range(20):
        clips.append(_clip(f"b{i}", "bona_fide", 0.0, "calibration"))
        results.append({"clip_id": f"b{i}", "raw_score": -2.0 - i * 0.05})
    for i in range(20):
        clips.append(_clip(f"g{i}", "spoof", 0.0, "calibration"))
        results.append({"clip_id": f"g{i}", "raw_score": 2.0 + i * 0.05})
    manifest = corpus.load_manifest({"schema_version": 1, "clips": clips})
    journal = se.parse_journal(_journal_text(_header(), results))
    policy = se.calibrate_policy(journal, manifest, policy_id="synthdetect-cal-v1")
    assert policy["calibration_policy_id"] == "synthdetect-cal-v1"
    assert 0.0 <= policy["brier"] <= 1.0
    assert policy["primary_threshold"]["target_fpr"] == se.PRIMARY_FPR
    assert len(policy["cohort_sha256"]) == 64


def test_calibrate_empty_split_rejected() -> None:
    clips = [_clip("c1", "bona_fide", 0.0, "eval"), _clip("c2", "spoof", 0.0, "eval")]
    manifest = corpus.load_manifest({"schema_version": 1, "clips": clips})
    journal = se.parse_journal(_journal_text(_header(), [
        {"clip_id": "c1", "raw_score": -1.0}, {"clip_id": "c2", "raw_score": 1.0}]))
    with pytest.raises(se.EvalError, match="calibration split"):
        se.calibrate_policy(journal, manifest, policy_id="p")


# --------------------------------------------------------------------------- #
# render_report (pure)
# --------------------------------------------------------------------------- #
def test_render_report_has_license_class_and_no_emdash() -> None:
    journal, manifest = _eval_corpus()
    report = se.score_report(journal, manifest, split="eval")
    md = se.render_report(report, date="2026-08-24")
    assert "license class: **shippable**" in md
    assert "—" not in md  # no emdashes (house style)
    assert "EER:" in md


def test_render_report_single_class_stratum_row() -> None:
    clips = [
        _clip("b1", "bona_fide", 0.0, "eval", stratum="only_bona"),
        _clip("g1", "spoof", 0.0, "eval", stratum="mixed"),
        _clip("b2", "bona_fide", 0.0, "eval", stratum="mixed"),
    ]
    manifest = corpus.load_manifest({"schema_version": 1, "clips": clips})
    journal = se.parse_journal(_journal_text(_header(), [
        {"clip_id": "b1", "raw_score": -1.0},
        {"clip_id": "g1", "raw_score": 1.0},
        {"clip_id": "b2", "raw_score": -1.0},
    ]))
    md = se.render_report(se.score_report(journal, manifest, split="eval"), date="2026-08-24")
    assert "single-class (not scored)" in md


# --------------------------------------------------------------------------- #
# CLI dispatch (main)
# --------------------------------------------------------------------------- #
def _write_corpus(tmp_path: Path, split: str = "eval") -> tuple[Path, Path]:
    clips = []
    results = []
    for i in range(15):
        clips.append(_clip(f"b{i}", "bona_fide", 0.0, split))
        results.append({"clip_id": f"b{i}", "raw_score": -2.0 - i * 0.1})
    for i in range(15):
        clips.append(_clip(f"g{i}", "spoof", 0.0, split))
        results.append({"clip_id": f"g{i}", "raw_score": 2.0 + i * 0.1})
    m_path = tmp_path / "m.json"
    j_path = tmp_path / "j.jsonl"
    m_path.write_text(json.dumps({"schema_version": 1, "clips": clips}), encoding="utf-8")
    j_path.write_text(_journal_text(_header(), results), encoding="utf-8")
    return m_path, j_path


def test_cli_score_writes_metrics(tmp_path: Path) -> None:
    m_path, j_path = _write_corpus(tmp_path)
    out = tmp_path / "metrics.json"
    rc = se.main(["score", "--journal", str(j_path), "--manifest", str(m_path),
                  "--split", "eval", "--out", str(out)])
    assert rc == 0
    metrics = json.loads(out.read_text())
    assert metrics["kind"] == "synthdetect_score_report"
    assert metrics["overall"]["eer"] == pytest.approx(0.0, abs=1e-6)


def test_cli_calibrate_and_report(tmp_path: Path) -> None:
    m_path, j_path = _write_corpus(tmp_path, split="calibration")
    pol = tmp_path / "cal.json"
    rc = se.main(["calibrate", "--journal", str(j_path), "--manifest", str(m_path),
                  "--policy-id", "synthdetect-cal-v1", "--out", str(pol)])
    assert rc == 0
    assert json.loads(pol.read_text())["method"] == "platt"

    # score -> report round trip
    m2, j2 = _write_corpus(tmp_path, split="eval")
    metrics = tmp_path / "metrics.json"
    se.main(["score", "--journal", str(j2), "--manifest", str(m2), "--split", "eval",
             "--out", str(metrics)])
    md = tmp_path / "report.md"
    rc = se.main(["report", "--metrics", str(metrics), "--date", "2026-08-24", "--out", str(md)])
    assert rc == 0
    assert "Synthdetect eval report" in md.read_text()


def test_cli_compare(tmp_path: Path) -> None:
    j1 = tmp_path / "j1.jsonl"
    j2 = tmp_path / "j2.jsonl"
    j1.write_text(_journal_text(_header(), [
        {"clip_id": "c1", "raw_score": 1.0}, {"clip_id": "c2", "raw_score": 2.0}]),
        encoding="utf-8")
    j2.write_text(_journal_text(_header(), [
        {"clip_id": "c1", "raw_score": 1.05}, {"clip_id": "c2", "raw_score": 1.95}]),
        encoding="utf-8")
    out = tmp_path / "cmp.json"
    rc = se.main(["compare", "--left", str(j1), "--right", str(j2), "--out", str(out)])
    assert rc == 0
    assert json.loads(out.read_text())["n_common"] == 2


def test_cli_error_returns_2(tmp_path: Path, capsys: Any) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text("", encoding="utf-8")
    m_path, _ = _write_corpus(tmp_path)
    rc = se.main(["score", "--journal", str(bad), "--manifest", str(m_path), "--split", "eval"])
    assert rc == 2
    assert "error:" in capsys.readouterr().err
