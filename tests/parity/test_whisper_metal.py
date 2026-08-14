"""Metal-tier ASR smoke: native CT2 large-v2 on macOS arm64 vs the CUDA reference.

v1 metal runs whisper on host CPU (plan decision 1) with the same pinned
model revision and int8 compute type as the images — but the macOS arm64
CTranslate2 wheel is a DIFFERENT BUILD than the Linux images use, so
equivalence is measured, not assumed. Smoke-level comparison against
``references/cuda/transcribe.json`` on the committed short fixture:
transcript similarity, segment count, and confidence drift, through the real
service ``WhisperTranscriber`` (same code path as production).

Pre-registered smoke bounds; the post-measurement pass (plan slice 9)
ratchets them from recorded numbers. Maintainer-run (Gate M in
docs/release-process.md): prerequisites are plain SKIPs everywhere else —
``VOXINT_PARITY_REQUIRED`` is never applied to metal lanes. Needs
faster-whisper in the running interpreter (the metal whisper venv +
``pytest``) and the pinned model already downloaded
(``voxint-metal.sh setup``).
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
CORPUS_DIR = REPO / "tests" / "parity" / "fixtures" / "corpus"
CUDA_TRANSCRIBE_JSON = (
    REPO / "tests" / "parity" / "fixtures" / "references" / "cuda" / "transcribe.json"
)

# --- Pre-registered smoke bounds (ratcheted post-measurement, plan slice 9).
TRANSCRIPT_MIN_SIMILARITY = 0.95  # difflib ratio on normalized transcripts
SEGMENT_COUNT_MAX_DIFF = 1
CONFIDENCE_MAX_DRIFT = 0.15


def _model_root() -> Path | None:
    """The pinned large-v2 snapshot, wherever this machine caches it."""
    metal_home = Path(
        os.getenv("VOXINT_METAL_HOME", str(Path.home() / ".voxint-metal"))
    )
    root = Path(os.getenv("WHISPER_DOWNLOAD_ROOT", str(metal_home / "models" / "whisper")))
    snapshots = root / "models--Systran--faster-whisper-large-v2" / "snapshots"
    if snapshots.is_dir() and any(snapshots.glob("*/model.bin")):
        return root
    return None


_IS_APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"
_MODEL_ROOT = _model_root() if _IS_APPLE_SILICON else None

pytestmark = [
    pytest.mark.skipif(
        not _IS_APPLE_SILICON, reason="metal gate runs on Apple Silicon macOS only"
    ),
    pytest.mark.skipif(
        not CUDA_TRANSCRIBE_JSON.exists(), reason="CUDA transcribe reference missing"
    ),
    pytest.mark.skipif(
        _IS_APPLE_SILICON and _MODEL_ROOT is None,
        reason="pinned large-v2 model not downloaded — run "
        "scripts/metal/voxint-metal.sh setup",
    ),
]

if _IS_APPLE_SILICON:
    pytest.importorskip("faster_whisper", reason="whisper metal venv required")


@pytest.fixture(scope="session")
def cuda_reference() -> dict[str, Any]:
    ref = json.loads(CUDA_TRANSCRIBE_JSON.read_text())
    wav = CORPUS_DIR / "transcribe-short.wav"
    bound = ref["meta"]["corpus_files_sha256"]["transcribe-short.wav"]
    assert hashlib.sha256(wav.read_bytes()).hexdigest() == bound, (
        "corpus wav does not match the committed CUDA reference"
    )
    assert ref["meta"]["service_healthz"]["device"] == "cuda"
    return ref["variants"]


@pytest.fixture(scope="session")
def native_output() -> Any:
    """Run the real service transcriber exactly as the metal launcher
    configures it: large-v2, device cpu, int8, pinned local model cache."""
    from tests.contracts.conftest import service_package

    saved = os.environ.get("WHISPER_DOWNLOAD_ROOT")
    os.environ["WHISPER_DOWNLOAD_ROOT"] = str(_MODEL_ROOT)
    try:
        with service_package("whisper"):
            from app.transcription import WhisperTranscriber

            transcriber = WhisperTranscriber(
                model_name="large-v2", device="cpu", compute_type="int8"
            )
            transcriber.load_model()
            return transcriber.transcribe(
                str(CORPUS_DIR / "transcribe-short.wav"),
                language="en",
                vad_filter=True,
            )
    finally:
        if saved is None:
            os.environ.pop("WHISPER_DOWNLOAD_ROOT", None)
        else:
            os.environ["WHISPER_DOWNLOAD_ROOT"] = saved


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


class TestTranscribeSmoke:
    def test_transcript_similarity(
        self, native_output: Any, cuda_reference: dict[str, Any]
    ) -> None:
        ref_text = _normalize(cuda_reference["vad_true"]["transcript"])
        got_text = _normalize(native_output.transcript)
        ratio = difflib.SequenceMatcher(None, ref_text, got_text).ratio()
        print(f"\ntranscript similarity vs CUDA reference: {ratio:.4f}")
        assert ratio >= TRANSCRIPT_MIN_SIMILARITY, (
            f"native arm64 CT2 transcript diverges (similarity {ratio:.4f}); "
            "diff:\n"
            + "\n".join(
                difflib.unified_diff(
                    ref_text.split(), got_text.split(), lineterm="", n=2
                )
            )
        )

    def test_segment_count_close(
        self, native_output: Any, cuda_reference: dict[str, Any]
    ) -> None:
        ref_n = len(cuda_reference["vad_true"]["segments"])
        got_n = len(native_output.segments)
        assert abs(ref_n - got_n) <= SEGMENT_COUNT_MAX_DIFF, (
            f"segment count {got_n} vs reference {ref_n}"
        )

    def test_confidence_drift_bounded(
        self, native_output: Any, cuda_reference: dict[str, Any]
    ) -> None:
        ref_conf = cuda_reference["vad_true"]["confidence"]
        drift = abs(native_output.confidence - ref_conf)
        print(f"\nconfidence: native {native_output.confidence:.4f} vs "
              f"reference {ref_conf:.4f} (drift {drift:.4f})")
        assert drift <= CONFIDENCE_MAX_DRIFT
