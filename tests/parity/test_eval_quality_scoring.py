"""Scorer-core tests for the #97 eval-quality harness (parity lane).

Needs the ``eval-quality`` extra (pyannote.metrics) and the ``parity`` extra
(the frozen WER stack), so it is skipped on a plain dev lane and runs under
``uv run --extra parity --extra eval-quality pytest``. Covers the load-bearing
numerics: RTTM re-keying, UEM parsing, micro-average pooling, the ASR reuse,
and — the invariant most likely to be silently wrong — collar semantics
(``collar`` is the TOTAL centered width, so NIST +/-250 ms is ``collar=0.5``).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

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


def _rttm(recording: str, rows: list[tuple[float, float, str]], file_id: str = "run-uuid") -> str:
    """RTTM text; file_id column is deliberately a run-UUID, not the recording."""
    return (
        "\n".join(
            f"SPEAKER {file_id} 1 {start:.3f} {dur:.3f} <NA> <NA> {label} <NA> <NA>"
            for start, dur, label in rows
        )
        + "\n"
    )


class TestParseRttm:
    def test_rekeys_to_recording_id_ignoring_file_id_column(self) -> None:
        ann = ev.parse_rttm(_rttm("EN2002c", [(0.0, 10.0, "spk0")], file_id="run-uuid"), "EN2002c")
        assert ann.uri == "EN2002c"
        assert ann.labels() == ["spk0"]

    def test_rejects_non_positive_duration(self) -> None:
        with pytest.raises(ev.EvalError):
            ev.parse_rttm("SPEAKER r 1 0.0 0.0 <NA> <NA> A <NA> <NA>\n", "r")

    def test_ignores_blank_and_non_speaker_lines(self) -> None:
        text = ";; comment\n\nSPEAKER r 1 0.0 1.0 <NA> <NA> A <NA> <NA>\n"
        assert ev.parse_rttm(text, "r").labels() == ["A"]


class TestParseUem:
    def test_keeps_only_matching_recording(self) -> None:
        tl = ev.parse_uem("OTHER 1 0 9\nEN2002c 1 1.0 5.0\n", "EN2002c")
        assert len(tl) == 1
        seg = next(iter(tl))
        assert (seg.start, seg.end) == (1.0, 5.0)

    def test_missing_region_raises(self) -> None:
        with pytest.raises(ev.EvalError):
            ev.parse_uem("OTHER 1 0 9\n", "EN2002c")


class TestDiarizationScoring:
    def _item(self, rec: str, ref_rows, hyp_rows, uem="r 1 0 20\n"):
        return ev.DiarItem(
            recording_id=rec,
            reference=ev.parse_rttm(_rttm(rec, ref_rows), rec),
            hypothesis=ev.parse_rttm(_rttm(rec, hyp_rows), rec),
            uem=ev.parse_uem(uem.replace("r ", f"{rec} "), rec),
        )

    def test_perfect_hypothesis_scores_zero(self) -> None:
        item = self._item("r", [(0.0, 10.0, "A")], [(0.0, 10.0, "spk0")])
        res = ev.score_diarization_set([item], protocol="strict", **ev.STRICT)
        assert res.pooled_der == pytest.approx(0.0)

    def test_strict_counts_a_small_boundary_miss(self) -> None:
        # ref A[0,10]; hyp ends 0.2s early -> 0.2s missed / 10s total = 0.02.
        item = self._item("r", [(0.0, 10.0, "A")], [(0.0, 9.8, "spk0")])
        res = ev.score_diarization_set([item], protocol="strict", **ev.STRICT)
        assert res.pooled_der == pytest.approx(0.02, abs=1e-6)
        assert res.per_recording["r"]["missed_s"] == pytest.approx(0.2, abs=1e-6)

    def test_collar_half_forgives_a_sub_250ms_boundary_error(self) -> None:
        # The same 0.2s miss lies within the +/-250ms collar around t=10, so the
        # diagnostic protocol (collar=0.5) does not score it. This is the exact
        # NIST +/-250ms == collar=0.5 semantics the plan flags as easy to get wrong.
        item = self._item("r", [(0.0, 10.0, "A")], [(0.0, 9.8, "spk0")])
        strict = ev.score_diarization_set([item], protocol="strict", **ev.STRICT)
        diag = ev.score_diarization_set([item], protocol="diagnostic", **ev.DIAGNOSTIC)
        assert strict.pooled_der > 0.0
        assert diag.pooled_der == pytest.approx(0.0)

    def test_collar_half_still_counts_a_large_boundary_error(self) -> None:
        # A 1.0s miss extends well past the 0.25s collar half-width, so it is
        # still scored at collar=0.5 -> the collar is not swallowing everything.
        item = self._item("r", [(0.0, 10.0, "A")], [(0.0, 9.0, "spk0")])
        diag = ev.score_diarization_set([item], protocol="diagnostic", **ev.DIAGNOSTIC)
        assert diag.pooled_der > 0.0

    def test_pooling_is_micro_average_over_components(self) -> None:
        # rec1 perfect, rec2 0.2s miss -> pooled = (0 + 0.2) / (10 + 10) = 0.01,
        # NOT the mean of per-file rates (0 and 0.02 -> 0.01 here by coincidence
        # of equal totals; use unequal totals to prove micro-average).
        rec1 = self._item("a", [(0.0, 30.0, "A")], [(0.0, 30.0, "s")], uem="r 1 0 40\n")
        rec2 = self._item("b", [(0.0, 10.0, "A")], [(0.0, 9.0, "s")], uem="r 1 0 40\n")
        res = ev.score_diarization_set([rec1, rec2], protocol="strict", **ev.STRICT)
        # micro-average: (0 + 1.0) / (30 + 10) = 0.025
        assert res.pooled_der == pytest.approx(0.025, abs=1e-6)
        # mean of per-file rates would be (0 + 0.1)/2 = 0.05 -> confirm we are NOT that
        assert res.pooled_der != pytest.approx(0.05, abs=1e-6)


class TestWer:
    def test_reuses_frozen_pooled_wer(self) -> None:
        out = ev.score_wer([("r", "the quick brown fox", "the quick brown fox")])
        assert out["pooled_wer"] == pytest.approx(0.0)
        out2 = ev.score_wer([("r", "the quick brown fox", "the quick brown dog")])
        assert out2["pooled_wer"] == pytest.approx(0.25, abs=1e-6)  # 1 sub / 4 words


class TestScoreCommand:
    def test_end_to_end_manifest(self, tmp_path: Path) -> None:
        (tmp_path / "ref.rttm").write_text(_rttm("r", [(0.0, 10.0, "A")]))
        (tmp_path / "hyp.rttm").write_text(_rttm("r", [(0.0, 9.8, "s")]))
        (tmp_path / "f.uem").write_text("r 1 0 20\n")
        (tmp_path / "ref.txt").write_text("hello world")
        (tmp_path / "hyp.txt").write_text("hello world")
        manifest = {
            "diarization": [
                {
                    "recording_id": "r",
                    "reference_rttm": str(tmp_path / "ref.rttm"),
                    "hypothesis_rttm": str(tmp_path / "hyp.rttm"),
                    "uem": str(tmp_path / "f.uem"),
                }
            ],
            "wer": [
                {
                    "recording_id": "r",
                    "reference_text": str(tmp_path / "ref.txt"),
                    "hypothesis_text": str(tmp_path / "hyp.txt"),
                }
            ],
        }
        mpath = tmp_path / "manifest.json"
        mpath.write_text(json.dumps(manifest))
        out = tmp_path / "metrics.json"
        assert ev.main(["score", "--manifest", str(mpath), "--out", str(out)]) == 0
        report = json.loads(out.read_text())
        assert report["diarization"]["strict"]["pooled_der"] == pytest.approx(0.02, abs=1e-6)
        assert report["diarization"]["diagnostic"]["pooled_der"] == pytest.approx(0.0)
        assert report["wer"]["pooled_wer"] == pytest.approx(0.0)

    def test_empty_manifest_errors(self, tmp_path: Path) -> None:
        mpath = tmp_path / "m.json"
        mpath.write_text("{}")
        assert ev.main(["score", "--manifest", str(mpath)]) == 2
