"""End-to-end bundle round-trip for the #97 ``run`` -> ``score`` seam (parity lane).

The dev-lane driver tests cover ``write_bundle``'s structure; this proves the
portability contract the whole self-contained-bundle design exists for: a bundle
written by ``write_bundle`` (references + hypotheses copied in, manifest paths
RELATIVE) scores through the REAL ``cmd_score`` from an UNRELATED working
directory, and its recomputed cohort hash matches the one ``run`` stamped. If
``score`` ever regressed to resolving manifest paths against the process CWD
instead of the manifest directory, this test fails.

Needs the scoring stack (pyannote + jiwer), so it lives in the parity lane:
``uv run --isolated --extra dev --extra parity --extra eval-quality``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
import wave
from pathlib import Path

import pytest

pytest.importorskip("pyannote.metrics", reason="eval-quality extra not installed")

REPO = Path(__file__).resolve().parents[2]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


eq = _load("eval_quality", "tools/eval_quality.py")
er = _load("eval_run", "tools/eval_run.py")


def _env() -> dict:
    return {
        "schema_version": 1,
        "code": {"git_sha": "abc", "image_digest": "sha256:d"},
        "model_weights": {
            "whisper_ct2_dir_sha256": "w",
            "pyannote_pipeline_sha256": "p",
            "titanet_sha256": "t",
        },
        "gpu": {"name": "RTX 3060", "driver": "550", "cuda": "12.4"},
        "runtime": {"ctranslate2": "4.0", "torch": "2.3", "pyannote_audio": "3.1.1"},
        "decode": {"beam_size": 5, "batch_size": 4, "word_timestamps": True},
        "flags": {"tf32": False, "deterministic": True},
    }


def _write_wav(path: Path, seconds: float, rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * round(seconds * rate))


def _ami_resolved(root: Path, rid: str) -> types.SimpleNamespace:
    _write_wav(root / f"{rid}.wav", 10.0)
    ref = root / f"{rid}.reference.rttm"
    ref.write_text(
        "SPEAKER x 1 0.000 4.000 <NA> <NA> A <NA> <NA>\n"
        "SPEAKER x 1 5.000 4.000 <NA> <NA> B <NA> <NA>\n",
        encoding="utf-8",
    )
    uem = root / f"{rid}.uem"
    uem.write_text(f"{rid} 1 0.00 10.00\n", encoding="utf-8")
    wer = root / f"{rid}.words.txt"
    wer.write_text("the quick brown fox\n", encoding="utf-8")
    return types.SimpleNamespace(
        recording_id=rid, corpus="ami", split="test",
        audio=root / f"{rid}.wav", reference_rttm=ref, uem=uem, wer_reference=wer,
    )


def _build_cohort_block(resolved: list, corpus: str) -> tuple[dict, str]:
    observed = eq.observe_role_files(resolved)
    split_by_id, inputs = eq.build_cohort_inputs(resolved, observed)
    descriptor = er.cohort_descriptor(
        corpus, split_by_id, inputs, er.pipeline_environment_hash(_env()), eq.HARNESS_PROTOCOL
    )
    cohort_hash = er.cohort_sha256(descriptor)
    block = {
        "schema_version": er.COHORT_SCHEMA_VERSION,
        "corpus": corpus,
        "pipeline_environment": _env(),
        "inputs": descriptor["inputs"],
        "cohort_sha256": cohort_hash,
        "batch_id": "deadbeefbatch",
    }
    return block, cohort_hash


def test_written_bundle_scores_from_an_unrelated_cwd(tmp_path: Path, monkeypatch) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    r = _ami_resolved(corpus_dir, "EN2002c")
    block, cohort_hash = _build_cohort_block([r], "ami")

    out_dir = tmp_path / "bundle"
    hyp_rttm = (
        "SPEAKER run-uuid 1 0.000 4.000 <NA> <NA> SPEAKER_00 <NA> <NA>\n"
        "SPEAKER run-uuid 1 5.000 4.000 <NA> <NA> SPEAKER_01 <NA> <NA>\n"
    )
    manifest_path = eq.write_bundle(
        out_dir, [r], {"EN2002c": hyp_rttm}, {"EN2002c": "the quick brown fox"}, block
    )

    # Score from an unrelated cwd: relative manifest paths MUST resolve against
    # the manifest directory, not here. Delete the corpus dir first to prove the
    # bundle is self-contained (score reads only the copies under out_dir).
    for f in corpus_dir.iterdir():
        f.unlink()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.chdir(scratch)

    metrics_path = out_dir / "metrics.json"
    args = types.SimpleNamespace(manifest=str(manifest_path), out=str(metrics_path))
    assert eq.cmd_score(args) == 0

    metrics = json.loads(metrics_path.read_text())
    assert metrics["kind"] == "eval_quality_report"
    # The scorer recomputed the cohort identity from the copied bytes and it
    # matches what the driver stamped (the round-trip is byte-faithful).
    assert metrics["environment"]["cohort_sha256"] == cohort_hash
    assert metrics["environment"]["corpus"] == "ami"
    # A perfect hypothesis over this reference scores DER 0 and WER 0.
    assert metrics["diarization"]["strict"]["pooled_der"] == 0.0
    assert metrics["wer"]["pooled_wer"] == 0.0


def test_bundle_with_tampered_reference_fails_cohort_binding(tmp_path: Path) -> None:
    # If a reference byte changes after the cohort was stamped, score's recompute
    # must reject it (accidental-drift integrity), not silently score.
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    r = _ami_resolved(corpus_dir, "EN2002c")
    block, _ = _build_cohort_block([r], "ami")
    out_dir = tmp_path / "bundle"
    manifest_path = eq.write_bundle(
        out_dir,
        [r],
        {"EN2002c": "SPEAKER u 1 0.000 4.000 <NA> <NA> SPEAKER_00 <NA> <NA>\n"},
        {"EN2002c": "the quick brown fox"},
        block,
    )
    # Tamper with the copied reference inside the bundle.
    ref_copy = out_dir / "inputs" / "EN2002c.reference.rttm"
    ref_copy.write_text(
        "SPEAKER x 1 0.000 9.000 <NA> <NA> A <NA> <NA>\n", encoding="utf-8"
    )
    args = types.SimpleNamespace(manifest=str(manifest_path), out=None)
    old = os.getcwd()
    try:
        raised = False
        try:
            eq.cmd_score(args)
        except eq.EvalError:
            raised = True
        assert raised, "score must reject a bundle whose reference bytes no longer match the cohort"
    finally:
        os.chdir(old)
