"""Self-parity gate (#33 Slice 2b, step 4): prove ``ct2 ≈ ct2-legacy``.

The shared-window ``ct2`` engine must reproduce the byte-faithful ``ct2-legacy``
transcript to **≤0.5pp pooled WER per vad mode** before Slice 2 can close. This
is the numerics-doctrine artifact for the seam: measured equivalence, not
reasoning. It is also the exact pattern the future mlx gate will reuse — swap
``Ct2Backend`` for the mlx backend and the gate is unchanged.

Design:

* Reference is ``ct2-legacy`` run **live** (not the frozen oracle) — the gate
  measures the two engines against each other on identical inputs, so a shared
  model/runtime drift cancels and only the decode-path difference is under test.
  (``test_whisper_ct2_legacy_replay.py`` separately anchors ``ct2-legacy`` to the
  frozen oracle.)
* Both engines are built with the SAME ``batch_size`` (4, the metal baseline):
  ``batch_size`` is not a parity variable but the two MUST match, or word-time
  threading across the batch boundary would diverge for >``batch_size`` windows.
* The two large models are resident **one at a time** (legacy fully, then ct2)
  to fit maintainer RAM; each engine transcribes the whole corpus in both vad
  modes within its single residency.
* Metric is a **micro average**: integer S/D/I/N pooled across files, then
  ``WER = (S+D+I)/N`` — per vad mode, NO averaging across modes. Empty-reference
  fixtures are excluded from the WER denominator and separately held to a
  zero-insertion invariant.

Maintainer-run, Apple-Silicon-only (mirrors ``test_whisper_metal.py``); plain
SKIP everywhere else. Needs the whisper metal venv (faster-whisper + jiwer +
the pinned large-v2 weights).
"""

from __future__ import annotations

import gc
import os
import platform
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
CORPUS_DIR = REPO / "tests" / "parity" / "fixtures" / "corpus"

# The whisper HF snapshot the metal launcher pins (must match
# scripts/metal/voxint-metal.sh + test_whisper_metal.py).
WHISPER_HF_REVISION = "f0fe81560cb8b68660e564f55dd99207059c092e"

# Both engines run at the metal-tier batch size (Gate M baseline). The value is
# not a parity variable but the two engines MUST share it.
BATCH_SIZE = 4

# Pooled-WER ceiling per vad mode. Loosening this is a numerics decision.
MAX_POOLED_WER_PP = 0.5

# Curated AMI subset (of the 15 staged clips) chosen to span the packed-window
# count from a single batch to three batches at batch_size=4 — so multi-batch
# word-time threading is exercised, not just the trivial single-window case:
#   ES2002a=2  ES2005d=3  EN2002c=5  IS1004d=8  ES2009a=10 windows.
# The full 15-clip corpus is available; set VOXINT_SELF_PARITY_FULL=1 to run it
# (slower: ~4 CPU transcribes per clip). Excluded clips add wall-clock without
# adding window-count coverage the subset lacks.
_CURATED_AMI = (
    "ES2002a.A.wav",
    "ES2005d.A.wav",
    "EN2002c.A.wav",
    "IS1004d.A.wav",
    "ES2009a.A.wav",
)

_IS_APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"


def _model_root() -> Path | None:
    metal_home = Path(
        os.getenv("VOXINT_METAL_HOME", str(Path.home() / ".voxint-metal"))
    )
    root = Path(
        os.getenv("WHISPER_DOWNLOAD_ROOT", str(metal_home / "models" / "whisper"))
    )
    snapshot = (
        root
        / "models--Systran--faster-whisper-large-v2"
        / "snapshots"
        / WHISPER_HF_REVISION
    )
    return root if (snapshot / "model.bin").is_file() else None


_MODEL_ROOT = _model_root() if _IS_APPLE_SILICON else None

pytestmark = [
    pytest.mark.skipif(
        not _IS_APPLE_SILICON,
        reason="metal self-parity gate runs on Apple Silicon macOS only",
    ),
    pytest.mark.skipif(
        _IS_APPLE_SILICON and _MODEL_ROOT is None,
        reason="pinned large-v2 model not downloaded — run "
        "scripts/metal/voxint-metal.sh setup",
    ),
]

if _IS_APPLE_SILICON:
    pytest.importorskip("faster_whisper", reason="whisper metal venv required")
    pytest.importorskip("jiwer", reason="jiwer (parity extra) required")


def _corpus() -> list[Path]:
    wavs = [CORPUS_DIR / "transcribe-short.wav"]
    ami_dir = Path(
        os.getenv("VOXINT_METAL_HOME", str(Path.home() / ".voxint-metal"))
    ) / "bakeoff" / "work" / "ami"
    if ami_dir.is_dir():
        if os.getenv("VOXINT_SELF_PARITY_FULL") == "1":
            wavs += sorted(ami_dir.glob("*.wav"))
        else:
            wavs += [ami_dir / name for name in _CURATED_AMI]
    return [w for w in wavs if w.is_file()]


def _run_engine(transcriber: Any, files: list[Path]) -> dict[tuple[str, bool], Any]:
    """Transcribe every file in both vad modes within one model residency."""
    transcriber.load_model()
    try:
        out: dict[tuple[str, bool], Any] = {}
        for wav in files:
            for vad in (True, False):
                out[(wav.name, vad)] = transcriber.transcribe(
                    str(wav), language="en", vad_filter=vad
                )
        return out
    finally:
        transcriber.cleanup_memory()


@pytest.fixture(scope="module")
def _engine_outputs() -> dict[str, dict[tuple[str, bool], Any]]:
    """Run ct2-legacy fully, unload, then ct2 — one large model at a time."""
    files = _corpus()
    if not files:
        pytest.skip("no corpus fixtures available")

    saved = {k: os.environ.get(k) for k in ("WHISPER_DOWNLOAD_ROOT", "WHISPER_REVISION")}
    os.environ["WHISPER_DOWNLOAD_ROOT"] = str(_MODEL_ROOT)
    os.environ["WHISPER_REVISION"] = WHISPER_HF_REVISION
    try:
        from tests.contracts.conftest import service_package

        with service_package("whisper"):
            from app.backends.ct2 import Ct2Backend
            from app.transcription import WhisperTranscriber

            legacy = WhisperTranscriber(
                model_name="large-v2",
                device="cpu",
                compute_type="int8",
                batch_size=BATCH_SIZE,
            )
            legacy_out = _run_engine(legacy, files)
            del legacy
            gc.collect()

            ct2 = WhisperTranscriber(
                backend=Ct2Backend(
                    model_name="large-v2",
                    device="cpu",
                    compute_type="int8",
                    batch_size=BATCH_SIZE,
                )
            )
            ct2_out = _run_engine(ct2, files)
            del ct2
            gc.collect()

        return {"files": files, "ct2-legacy": legacy_out, "ct2": ct2_out}
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.mark.parametrize("vad_filter", [True, False], ids=["vad_true", "vad_false"])
def test_ct2_matches_ct2_legacy_pooled_wer(
    _engine_outputs: dict[str, Any], vad_filter: bool
) -> None:
    """ct2 reproduces ct2-legacy to ≤0.5pp pooled WER for this vad mode."""
    from tests.parity.whisper_bakeoff_score import score_pooled

    files: list[Path] = _engine_outputs["files"]
    legacy_out = _engine_outputs["ct2-legacy"]
    ct2_out = _engine_outputs["ct2"]

    items = [
        (
            wav.name,
            legacy_out[(wav.name, vad_filter)].transcript,
            ct2_out[(wav.name, vad_filter)].transcript,
        )
        for wav in files
    ]
    pooled = score_pooled(items)

    # Per-file outliers reported separately (diagnostic, not a gate) so a single
    # bad clip is visible even when the pooled number passes.
    outliers = [
        f"{s.name}: S={s.substitutions} D={s.deletions} I={s.insertions} "
        f"N={s.ref_words} wer={s.wer * 100:.3f}pp"
        for s in pooled.files
        if not s.reference_empty and s.wer > 0
    ]

    # Empty-reference (non-speech) fixtures must produce no insertions in EITHER
    # engine's transcript — a zero-insertion invariant, not a WER contribution.
    for s in pooled.files:
        if s.reference_empty:
            assert s.insertions == 0, (
                f"{s.name}: ct2 inserted {s.insertions} words where ct2-legacy "
                f"produced no speech"
            )

    assert pooled.ref_words > 0, "no speech reference words scored"
    assert pooled.wer_pp <= MAX_POOLED_WER_PP, (
        f"[{'vad_true' if vad_filter else 'vad_false'}] pooled WER "
        f"{pooled.wer_pp:.4f}pp > {MAX_POOLED_WER_PP}pp "
        f"(S={pooled.substitutions} D={pooled.deletions} I={pooled.insertions} "
        f"N={pooled.ref_words}); per-file: {outliers}"
    )
