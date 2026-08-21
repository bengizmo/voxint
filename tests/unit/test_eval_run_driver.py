"""Unit tests for the eval-quality ``run`` driver's pure seams (issue #97).

The live driver (submit -> poll -> read DB -> export bundle) needs an idle worker
and is validated by the maintainer host runbook, but every seam it stands on is
pure and covered here WITHOUT a worker or the scoring stack:

* the Step-0 import guard (``import eval_quality`` + ``run``/``report`` load with
  pyannote/jiwer absent), so the lazy-import split cannot silently regress;
* the export shaping (ordered per-segment word lists with strict NULL-word
  rejection, monotonic-index guards);
* the cohort-input builder reused for the journal and the manifest;
* the self-contained bundle writer (files copied in, manifest paths relative);
* the reconcile-by-source_path join's exactly-one-run invariant;
* the host-side fingerprint refusal logic;
* the write-ahead journal driving crash-safe resume decisions at each boundary.

``eval_quality`` is imported with no scoring extras, so this runs in the dev lane
(the bundle round-trip THROUGH the real scorer lives in the parity lane).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
import wave
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load(name: str, rel: str):
    path = REPO / rel
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


eq = _load("eval_quality", "tools/eval_quality.py")
er = _load("eval_run", "tools/eval_run.py")


def _row(**kw) -> types.SimpleNamespace:
    return types.SimpleNamespace(**kw)


def _write_wav(path: Path, seconds: float, rate: int = 16000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nframes = round(seconds * rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * nframes)
    return path


def _resolved(root: Path, rid: str, *, ami: bool) -> types.SimpleNamespace:
    """A ResolvedItem-shaped record with real files on disk."""
    audio = _write_wav(root / f"{rid}.wav", 1.0)
    ref = root / f"{rid}.reference.rttm"
    ref.write_text("SPEAKER x 1 0.000 1.000 <NA> <NA> S0 <NA> <NA>\n", encoding="utf-8")
    if ami:
        uem = root / f"{rid}.uem"
        uem.write_text(f"{rid} 1 0.00 1.00\n", encoding="utf-8")
        wer = root / f"{rid}.words.txt"
        wer.write_text("hello world\n", encoding="utf-8")
    else:
        uem = None
        wer = None
    return types.SimpleNamespace(
        recording_id=rid, corpus="ami" if ami else "voxconverse", split="test",
        audio=audio, reference_rttm=ref, uem=uem, wer_reference=wer,
    )


# --------------------------------------------------------------------------- #
# Step 0: import guard
# --------------------------------------------------------------------------- #
class TestImportGuard:
    def test_eval_quality_imports_without_the_scoring_stack(self) -> None:
        # Step 0's claim: `import eval_quality` succeeds with pyannote.metrics
        # absent (it used to import pyannote.core at module top). Only meaningful
        # when the scoring stack is genuinely absent, so skip if it is installed.
        try:
            import pyannote.core  # noqa: F401

            pytest.skip("scoring stack present in this lane; the guard is vacuous here")
        except ImportError:
            pass
        # eq was imported at module top with pyannote absent — that is the proof.
        assert eq.build_parser() is not None

    def test_run_and_report_parse_without_scoring_stack(self) -> None:
        parser = eq.build_parser()
        run_ns = parser.parse_args(
            ["run", "--corpus", "ami", "--subset", "s.json", "--out-dir", "o",
             "--pipeline-env", "e.json"]
        )
        assert run_ns.fn is eq.cmd_run
        rep_ns = parser.parse_args(["report", "--run", "ami=x.json", "--date", "2026-08-21"])
        assert rep_ns.fn is eq.cmd_report

    def test_scoring_globals_are_lazy_until_loaded(self) -> None:
        # Before any score, the pyannote globals are None (proof the top-level
        # import was removed). A fresh module load keeps them None.
        fresh = _load("eval_quality_probe", "tools/eval_quality.py")
        assert fresh.Annotation is None
        assert fresh.score_pooled is None


# --------------------------------------------------------------------------- #
# Export shaping
# --------------------------------------------------------------------------- #
class TestExportShaping:
    def test_monotonic_unique_guard(self) -> None:
        eq.assert_monotonic_unique([0, 1, 2, 3], "x")
        with pytest.raises(eq.EvalError):
            eq.assert_monotonic_unique([0, 1, 1], "x")  # duplicate
        with pytest.raises(eq.EvalError):
            eq.assert_monotonic_unique([0, 2, 1], "x")  # out of order

    def test_words_flattened_in_segment_order(self) -> None:
        segs = [
            _row(segment_index=0, raw_text="a", words=[{"start": 0.0, "end": 0.1, "word": "a"}]),
            _row(segment_index=1, raw_text="b", words=[{"start": 0.2, "end": 0.3, "word": "b"}]),
        ]
        assert eq.segments_to_word_lists(segs) == [
            [{"start": 0.0, "end": 0.1, "word": "a"}],
            [{"start": 0.2, "end": 0.3, "word": "b"}],
        ]

    def test_null_words_with_text_is_rejected(self) -> None:
        segs = [_row(segment_index=0, raw_text="lost timing", words=None)]
        with pytest.raises(eq.EvalError, match="NULL word timing"):
            eq.segments_to_word_lists(segs)

    def test_null_words_empty_segment_is_ok(self) -> None:
        segs = [_row(segment_index=0, raw_text="   ", words=None)]
        assert eq.segments_to_word_lists(segs) == [[]]

    def test_out_of_order_segments_rejected(self) -> None:
        segs = [
            _row(segment_index=1, raw_text="", words=[]),
            _row(segment_index=0, raw_text="", words=[]),
        ]
        with pytest.raises(eq.EvalError):
            eq.segments_to_word_lists(segs)


# --------------------------------------------------------------------------- #
# Cohort inputs + bundle
# --------------------------------------------------------------------------- #
class TestCohortInputs:
    def test_ami_builds_four_roles_and_reuses_one_path(self, tmp_path: Path) -> None:
        r = _resolved(tmp_path, "EN2002c", ami=True)
        observed = eq.observe_role_files([r])
        split_by_id, inputs = eq.build_cohort_inputs([r], observed)
        assert split_by_id == {"EN2002c": "test"}
        by_role = {ci.role: ci for ci in inputs}
        assert set(by_role) == {"audio", "reference_rttm", "uem", "wer_reference"}
        assert by_role["uem"].sha256 is not None and by_role["wer_reference"].byte_len is not None

    def test_voxconverse_null_roles_are_explicit(self, tmp_path: Path) -> None:
        r = _resolved(tmp_path, "vc1", ami=False)
        observed = eq.observe_role_files([r])
        _, inputs = eq.build_cohort_inputs([r], observed)
        by_role = {ci.role: ci for ci in inputs}
        assert by_role["uem"].sha256 is None and by_role["uem"].byte_len is None
        assert by_role["wer_reference"].sha256 is None
        assert by_role["audio"].sha256 is not None


class TestBundleWriter:
    def test_bundle_copies_inputs_and_writes_relative_manifest(self, tmp_path: Path) -> None:
        r = _resolved(tmp_path / "corpus", "EN2002c", ami=True)
        out = tmp_path / "bundle"
        cohort = {"schema_version": 1, "corpus": "ami", "cohort_sha256": "deadbeef", "inputs": []}
        manifest_path = eq.write_bundle(
            out, [r], {"EN2002c": "SPEAKER u 1 0.0 1.0 <NA> <NA> S0 <NA> <NA>\n"},
            {"EN2002c": "hello world"}, cohort,
        )
        manifest = json.loads(manifest_path.read_text())
        diar = manifest["diarization"][0]
        # Every manifest path is RELATIVE (portability), and resolves under out-dir.
        for key in ("reference_rttm", "hypothesis_rttm", "uem"):
            assert not Path(diar[key]).is_absolute()
            assert (out / diar[key]).is_file()
        assert manifest["wer"][0]["reference_text"].startswith("inputs/")
        assert (out / manifest["wer"][0]["hypothesis_text"]).read_text() == "hello world"
        assert manifest["cohort"]["cohort_sha256"] == "deadbeef"

    def test_voxconverse_bundle_has_null_uem_and_no_wer(self, tmp_path: Path) -> None:
        r = _resolved(tmp_path / "corpus", "vc1", ami=False)
        out = tmp_path / "bundle"
        manifest_path = eq.write_bundle(
            out, [r], {"vc1": "SPEAKER u 1 0.0 1.0 <NA> <NA> S0 <NA> <NA>\n"}, {},
            {"schema_version": 1, "corpus": "voxconverse", "cohort_sha256": "x", "inputs": []},
        )
        manifest = json.loads(manifest_path.read_text())
        assert manifest["diarization"][0]["uem"] is None
        assert "wer" not in manifest


# --------------------------------------------------------------------------- #
# Reconcile-by-source_path (the idempotency join's exactly-one invariant)
# --------------------------------------------------------------------------- #
class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return self._rows


class _FakeQuery:
    """A chainable stand-in for a SQLAlchemy select (join/where are no-ops)."""

    def join(self, *a, **k):
        return self

    def where(self, *a, **k):
        return self


class _FakeSession:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, _query):
        return _FakeResult(self._rows)


def _fake_driver():
    return types.SimpleNamespace(
        select=lambda *a, **k: _FakeQuery(),
        PipelineRun=types.SimpleNamespace(media_item_id=None, id=None),
        MediaItem=types.SimpleNamespace(id=None, source_path=None),
    )


class TestReconcile:
    def test_exactly_one_run_is_adopted(self) -> None:
        run = _row(id="RUN-1")
        adopted = eq._reconcile_run(_fake_driver(), _FakeSession([run]), "eval/b/x.wav")
        assert adopted.id == "RUN-1"

    def test_zero_or_many_runs_refuse(self) -> None:
        for rows in ([], [_row(id="a"), _row(id="b")]):
            with pytest.raises(eq.EvalError, match="reconcile"):
                eq._reconcile_run(_fake_driver(), _FakeSession(rows), "eval/b/x.wav")


# --------------------------------------------------------------------------- #
# Fingerprint refusal logic (pure)
# --------------------------------------------------------------------------- #
def _fp(mode="full", whisper="sha256:img", gpu_name="RTX 3060"):
    return {
        "mode": mode,
        "images": {"whisper": whisper},
        "gpu": {"name": gpu_name, "uuid": "GPU-x", "driver": "550"},
        "cuda_visible_devices": "0",
        "probe_status": {},
    }


def _static_env(image="sha256:img", gpu="RTX 3060"):
    return {"code": {"image_digest": image}, "gpu": {"name": gpu}}


class TestFingerprintRefusal:
    def test_full_consistent_agreeing_passes(self) -> None:
        eq.require_verified_fingerprints(_fp(), _fp(), _static_env())

    def test_degraded_before_or_after_refuses(self) -> None:
        with pytest.raises(eq.EvalError, match="degraded before"):
            eq.require_verified_fingerprints(_fp(mode="degraded"), _fp(), _static_env())
        with pytest.raises(eq.EvalError, match="degraded after"):
            eq.require_verified_fingerprints(_fp(), _fp(mode="degraded"), _static_env())

    def test_mid_batch_change_refuses(self) -> None:
        with pytest.raises(eq.EvalError, match="changed during"):
            eq.require_verified_fingerprints(_fp(gpu_name="RTX 3060"), _fp(gpu_name="RTX 5090"),
                                             _static_env())

    def test_static_disagreement_refuses(self) -> None:
        other = _fp(whisper="sha256:other")
        with pytest.raises(eq.EvalError, match="image_digest"):
            eq.require_verified_fingerprints(other, other, _static_env(image="sha256:img"))
        gpu4090 = _fp(gpu_name="RTX 4090")
        with pytest.raises(eq.EvalError, match=r"gpu\.name"):
            eq.require_verified_fingerprints(gpu4090, gpu4090, _static_env(gpu="RTX 3060"))


# --------------------------------------------------------------------------- #
# Write-ahead journal drives crash-safe resume at each boundary
# --------------------------------------------------------------------------- #
def _env() -> dict:
    return {
        "schema_version": 1,
        "code": {"git_sha": "abc", "image_digest": "sha256:d"},
        "model_weights": {"whisper_ct2_dir_sha256": "w", "pyannote_pipeline_sha256": "p",
                          "titanet_sha256": "t"},
        "gpu": {"name": "RTX 3060", "driver": "550", "cuda": "12.4"},
        "runtime": {"ctranslate2": "4.0", "torch": "2.3", "pyannote_audio": "3.1.1"},
        "decode": {"beam_size": 5, "batch_size": 4, "word_timestamps": True},
        "flags": {"tf32": False, "deterministic": True},
    }


class TestJournalCrashSafety:
    def _journal(self, out: Path) -> dict:
        j = er.new_journal("ami", "cohorthash", _env())
        er.write_json_atomic(eq._journal_path(out), j)
        return j

    def test_crash_after_submitting_resumes_to_submit_then_reconcile(self, tmp_path: Path) -> None:
        # Write-ahead 'submitting' with no run_uuid (crash between stage and the
        # recorded run): a resume must SUBMIT, which submit_if_new makes
        # idempotent by the unique source_path (reconcile adopts the one run).
        j = self._journal(tmp_path)
        eq._record(tmp_path, j, "EN2002c", {"status": "submitting", "source_path": "eval/b/x.wav"})
        reloaded = json.loads(eq._journal_path(tmp_path).read_text())
        [d] = er.plan_resume(reloaded, ["EN2002c"], resume=True, retry_failed=False)
        assert d.action == er.ACTION_SUBMIT

    def test_crash_after_run_recorded_resumes_to_poll(self, tmp_path: Path) -> None:
        j = self._journal(tmp_path)
        eq._record(tmp_path, j, "EN2002c", {"status": "queued", "run_uuid": "u"})
        reloaded = json.loads(eq._journal_path(tmp_path).read_text())
        [d] = er.plan_resume(reloaded, ["EN2002c"], resume=True, retry_failed=False)
        assert d.action == er.ACTION_POLL

    def test_crash_after_export_resumes_to_skip_done(self, tmp_path: Path) -> None:
        j = self._journal(tmp_path)
        eq._record(tmp_path, j, "EN2002c", {
            "status": "completed",
            "artifacts": {"hypothesis_rttm_sha256": "h", "wer_text_sha256": "w"},
        })
        reloaded = json.loads(eq._journal_path(tmp_path).read_text())
        [d] = er.plan_resume(reloaded, ["EN2002c"], resume=True, retry_failed=False)
        assert d.action == er.ACTION_SKIP_DONE

    def test_partial_export_ami_missing_wer_stops(self, tmp_path: Path) -> None:
        j = self._journal(tmp_path)
        eq._record(tmp_path, j, "EN2002c", {
            "status": "completed", "artifacts": {"hypothesis_rttm_sha256": "h"},
        })
        reloaded = json.loads(eq._journal_path(tmp_path).read_text())
        [d] = er.plan_resume(reloaded, ["EN2002c"], resume=True, retry_failed=False)
        assert d.action == er.ACTION_STOP
