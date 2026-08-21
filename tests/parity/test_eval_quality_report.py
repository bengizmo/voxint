"""Report-renderer tests for the #97 eval-quality harness (parity lane).

The renderer is a pure metrics-JSON -> Markdown step, so these tests build
report dicts by hand (no worker, no pyannote scoring) and assert the house-style
contract the plan pins: per-corpus sections with no combined AMI+VoxConverse
number, strict primary + diagnostic cross-check, a zero-change noise band for
K > 1 runs, and the emdash-free / emoji-free copy rules. Loading the tool still
imports pyannote at module scope, so the file lives on the parity lane and skips
cleanly without the ``eval-quality`` extra.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pyannote.metrics", reason="eval-quality extra not installed")

REPO = Path(__file__).resolve().parents[2]


def _load_tool():
    path = REPO / "tools" / "eval_quality.py"
    spec = importlib.util.spec_from_file_location("eval_quality", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


ev = _load_tool()


def _per_rec(der: float, jer: float, evaluated: float, speakers: float, overlap: float) -> dict:
    return {
        "der": der,
        "jer": jer,
        "confusion_s": 0.0,
        "missed_s": 0.0,
        "false_alarm_s": 0.0,
        "total_s": evaluated,
        "uem_applied": True,
        "speaker_count": speakers,
        "reference_overlap_pct": overlap,
        "evaluated_s": evaluated,
    }


# Distinct synthetic cohort identities. `report` never recomputes the hash (that
# is `score`'s job from the scored bytes); it only requires every run of a corpus
# to be cohort-bound and to share ONE identity, so the fixtures set the hash by
# hand. Runs that form a noise-floor pair carry the same value.
AMI_COHORT = "a" * 64
VC_COHORT = "b" * 64


def _report(
    *,
    strict_der: float,
    diag_der: float,
    per_recording: dict[str, dict[str, Any]],
    wer: dict[str, Any] | None = None,
    git_sha: str = "abc123",
    manifest_sha: str = "d" * 64,
    corpus: str = "ami",
    cohort_sha256: str | None = AMI_COHORT,
    scorer_versions: dict[str, str] | None = None,
    normalizer_version: str = "openai-whisper@deadbeef/voxint-wrapper-v1",
) -> dict:
    strict = {
        "collar": 0.0,
        "skip_overlap": False,
        "pooled_der": strict_der,
        "global_jer": strict_der + 0.01,
        "per_recording": per_recording,
    }
    diagnostic = {
        "collar": 0.5,
        "skip_overlap": True,
        "pooled_der": diag_der,
        "global_jer": diag_der,
        "per_recording": per_recording,
    }
    environment: dict[str, Any] = {
        "git_sha": git_sha,
        "manifest_sha256": manifest_sha,
        "corpus": corpus,
        "diarization_cohort": sorted(per_recording),
        "wer_cohort": sorted(wer["per_recording"]) if wer else [],
        "scorer_versions": scorer_versions
        or {"pyannote.metrics": "4.1", "pyannote.core": "5.0.0", "jiwer": "3.0"},
        "normalizer_version": normalizer_version,
        "normalizer_runtime": "py3.12",
    }
    # A cohort-less run is possible from `score` but unreportable; None omits the
    # key so the fail-closed negative tests can exercise that path.
    if cohort_sha256 is not None:
        environment["cohort_sha256"] = cohort_sha256
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "eval_quality_report",
        "diarization": {
            "jer_mapping": ev.JER_MAPPING,
            "strict": strict,
            "diagnostic": diagnostic,
        },
        "environment": environment,
    }
    if wer is not None:
        report["wer"] = wer
    return report


AMI_RUN1 = _report(
    strict_der=0.120,
    diag_der=0.080,
    per_recording={"EN2002c": _per_rec(0.120, 0.20, 2970.0, 4.0, 12.1)},
    wer={"pooled_wer": 0.099, "per_recording": {"EN2002c": {"wer": 0.099, "ref_words": 12665}}},
)
AMI_RUN2 = _report(
    strict_der=0.126,
    diag_der=0.082,
    per_recording={"EN2002c": _per_rec(0.126, 0.21, 2970.0, 4.0, 12.1)},
    wer={"pooled_wer": 0.101, "per_recording": {"EN2002c": {"wer": 0.101, "ref_words": 12665}}},
)
VC_RUN1 = _report(
    strict_der=0.085,
    diag_der=0.060,
    per_recording={"abjxc": _per_rec(0.085, 0.15, 900.0, 3.0, 46.7)},
    corpus="voxconverse",
    cohort_sha256=VC_COHORT,
)


class TestRenderReport:
    def test_no_combined_grand_number(self) -> None:
        # The gate: AMI and VoxConverse are separate sections; there is no pooled
        # figure that mixes both corpora.
        text = ev.render_report("2026-08-20", {"ami": [AMI_RUN1], "voxconverse": [VC_RUN1]})
        assert "## ami" in text
        assert "## voxconverse" in text
        assert "no combined AMI plus VoxConverse figure" in text

    def test_per_corpus_pooled_and_metadata_rendered(self) -> None:
        text = ev.render_report("2026-08-20", {"ami": [AMI_RUN1]})
        assert "12.00%" in text  # pooled DER 0.120
        assert "9.90%" in text  # pooled WER 0.099
        assert "49:30" in text  # 2970s evaluated -> mm:ss
        assert "12.1%" in text  # reference overlap
        assert "| EN2002c |" in text

    def test_noise_band_only_for_multi_run(self) -> None:
        single = ev.render_report("2026-08-20", {"ami": [AMI_RUN1]})
        assert "Noise floor" not in single
        assert "single run, no noise band" in single
        multi = ev.render_report("2026-08-20", {"ami": [AMI_RUN1, AMI_RUN2]})
        assert "Noise floor (2 zero-change runs)" in multi
        # DER spread 0.126 - 0.120 = 0.006 -> 0.60 pp
        assert "pooled DER 0.60 pp" in multi
        # per-file worst DER spread also 0.60 pp on EN2002c
        assert "DER EN2002c 0.60 pp" in multi

    def test_voxconverse_has_no_wer_column(self) -> None:
        text = ev.render_report("2026-08-20", {"voxconverse": [VC_RUN1]})
        vc = text.split("## voxconverse", 1)[1]
        assert "WER" not in vc
        assert "Pooled WER" not in vc

    def test_header_stamps_date_sha_and_versions(self) -> None:
        text = ev.render_report("2026-08-20", {"ami": [AMI_RUN1]})
        head = text.splitlines()[0]
        assert head.startswith("> Eval-quality baseline. Generated 2026-08-20.")
        assert "abc123" in head
        assert "pyannote.metrics 4.1" in text

    def test_mixed_git_sha_raises_a_warning_line(self) -> None:
        other = _report(
            strict_der=0.13,
            diag_der=0.08,
            per_recording={"EN2002c": _per_rec(0.13, 0.2, 2970.0, 4.0, 12.1)},
            wer={
                "pooled_wer": 0.10,
                "per_recording": {"EN2002c": {"wer": 0.10, "ref_words": 12665}},
            },
            git_sha="different99",
        )
        text = ev.render_report("2026-08-20", {"ami": [AMI_RUN1, other]})
        assert "do not share one pipeline git sha" in text

    def test_house_style_no_emdash_or_emoji(self) -> None:
        text = ev.render_report(
            "2026-08-20", {"ami": [AMI_RUN1, AMI_RUN2], "voxconverse": [VC_RUN1]}
        )
        assert "\u2014" not in text  # em dash
        assert "\u2013" not in text  # en dash
        assert text.isascii()  # emoji-free / ascii copy

    def test_empty_raises(self) -> None:
        with pytest.raises(ev.EvalError):
            ev.render_report("2026-08-20", {})


class TestFailClosed:
    """report must refuse to render runs that are not genuinely comparable."""

    def test_rejects_a_cohort_less_run_even_at_k1(self) -> None:
        loose = _report(
            strict_der=0.12,
            diag_der=0.08,
            per_recording={"EN2002c": _per_rec(0.12, 0.2, 2970.0, 4.0, 12.1)},
            cohort_sha256=None,
        )
        with pytest.raises(ev.EvalError, match="not cohort-bound"):
            ev.render_report("2026-08-20", {"ami": [loose]})

    def test_rejects_unequal_cohort_hashes(self) -> None:
        other = _report(
            strict_der=0.126,
            diag_der=0.082,
            per_recording={"EN2002c": _per_rec(0.126, 0.21, 2970.0, 4.0, 12.1)},
            wer={"pooled_wer": 0.1, "per_recording": {"EN2002c": {"wer": 0.1, "ref_words": 12665}}},
            cohort_sha256="f" * 64,
        )
        with pytest.raises(ev.EvalError, match="do not share one cohort_sha256"):
            ev.render_report("2026-08-20", {"ami": [AMI_RUN1, other]})

    def test_rejects_corpus_label_mismatch(self) -> None:
        # A voxconverse-labelled metrics JSON supplied as --run ami=...
        with pytest.raises(ev.EvalError, match="corpus label mismatch"):
            ev.render_report("2026-08-20", {"ami": [VC_RUN1]})

    def test_rejects_non_identical_diarization_sets(self) -> None:
        other = _report(
            strict_der=0.126,
            diag_der=0.082,
            per_recording={"IS1009a": _per_rec(0.126, 0.21, 1800.0, 4.0, 10.0)},
            wer={"pooled_wer": 0.1, "per_recording": {"IS1009a": {"wer": 0.1, "ref_words": 9000}}},
        )
        with pytest.raises(ev.EvalError, match="different diarization recording sets"):
            ev.render_report("2026-08-20", {"ami": [AMI_RUN1, other]})

    def test_rejects_mixed_wer_presence(self) -> None:
        no_wer = _report(
            strict_der=0.126,
            diag_der=0.082,
            per_recording={"EN2002c": _per_rec(0.126, 0.21, 2970.0, 4.0, 12.1)},
        )
        with pytest.raises(ev.EvalError, match="some runs carry WER and some do not"):
            ev.render_report("2026-08-20", {"ami": [AMI_RUN1, no_wer]})

    def test_rejects_strict_diagnostic_set_disagreement(self) -> None:
        run = _report(
            strict_der=0.12,
            diag_der=0.08,
            per_recording={"EN2002c": _per_rec(0.12, 0.2, 2970.0, 4.0, 12.1)},
        )
        # Corrupt only the diagnostic per_recording so it disagrees with strict.
        run["diarization"]["diagnostic"]["per_recording"] = {
            "OTHER": _per_rec(0.0, 0.0, 1.0, 1.0, 0.0)
        }
        with pytest.raises(ev.EvalError, match="diarization sets disagree"):
            ev.render_report("2026-08-20", {"ami": [run]})

    def test_rejects_multi_run_scorer_version_mix(self) -> None:
        other = _report(
            strict_der=0.126,
            diag_der=0.082,
            per_recording={"EN2002c": _per_rec(0.126, 0.21, 2970.0, 4.0, 12.1)},
            wer={"pooled_wer": 0.1, "per_recording": {"EN2002c": {"wer": 0.1, "ref_words": 12665}}},
            scorer_versions={"pyannote.metrics": "4.2", "pyannote.core": "5.0.0", "jiwer": "3.0"},
        )
        with pytest.raises(ev.EvalError, match="mixes scorer versions"):
            ev.render_report("2026-08-20", {"ami": [AMI_RUN1, other]})

    def test_git_sha_mix_is_a_warning_not_an_error(self) -> None:
        # git_sha alone (same scorer_versions + normalizer) must NOT be fatal.
        other = _report(
            strict_der=0.126,
            diag_der=0.082,
            per_recording={"EN2002c": _per_rec(0.126, 0.21, 2970.0, 4.0, 12.1)},
            wer={"pooled_wer": 0.1, "per_recording": {"EN2002c": {"wer": 0.1, "ref_words": 12665}}},
            git_sha="different99",
        )
        text = ev.render_report("2026-08-20", {"ami": [AMI_RUN1, other]})
        assert "do not share one pipeline git sha" in text


class TestReportCommand:
    def _write(self, tmp_path: Path, name: str, report: dict) -> Path:
        p = tmp_path / name
        p.write_text(json.dumps(report))
        return p

    def test_cmd_report_writes_file(self, tmp_path: Path) -> None:
        a1 = self._write(tmp_path, "ami1.json", AMI_RUN1)
        a2 = self._write(tmp_path, "ami2.json", AMI_RUN2)
        vc = self._write(tmp_path, "vc.json", VC_RUN1)
        out = tmp_path / "report.md"
        rc = ev.main(
            [
                "report",
                "--date",
                "2026-08-20",
                "--run",
                f"ami={a1}",
                "--run",
                f"ami={a2}",
                "--run",
                f"voxconverse={vc}",
                "--out",
                str(out),
            ]
        )
        assert rc == 0
        text = out.read_text()
        assert "Noise floor (2 zero-change runs)" in text
        assert "## voxconverse" in text

    def test_run_spec_without_equals_rejected(self, tmp_path: Path) -> None:
        p = self._write(tmp_path, "a.json", AMI_RUN1)
        assert ev.main(["report", "--date", "2026-08-20", "--run", str(p)]) == 2

    def test_run_spec_empty_corpus_rejected(self, tmp_path: Path) -> None:
        p = self._write(tmp_path, "a.json", AMI_RUN1)
        assert ev.main(["report", "--date", "2026-08-20", "--run", f"={p}"]) == 2

    def test_non_report_json_rejected(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"kind": "something_else"}))
        assert ev.main(["report", "--date", "2026-08-20", "--run", f"ami={p}"]) == 2
