#!/usr/bin/env python3
"""Generate NeMo mel-spectrogram references for the titanet mel-level gate.

Runs INSIDE the pinned titanet image (NeMo 1.22.0) — same invocation pattern as
``tools/export_titanet_onnx.py``:

    docker run --rm -v $PWD:/repo -w /repo ghcr.io/bengizmo/voxint-titanet:0.3.0 \
        python3 tools/generate_mel_references.py

For a diverse subset of the golden corpus's embeddable windows, this applies the
exact ``titanet-large-v2`` runtime chain up to the model input (slice →
noise-reduce → LUFS → peak, via ``services/titanet/app/preprocess.py``) and then
records the checkpoint's own ``AudioToMelSpectrogramPreprocessor`` output. The
reimplemented front-end (``services/titanet/app/mel.py``) is measured against
these arrays in ``tests/parity/test_titanet_onnx.py``.

Mel references are computed on CPU: NeMo applies dither only in training mode,
so eval-mode mel extraction is deterministic, and the ONNX engine's front-end
runs on CPU — CPU-to-CPU is the honest comparison. (The vector/decision gates
compare against the real CUDA service references instead.)

Writes ``tests/parity/fixtures/references/mel/mel-references.npz`` (float32
arrays, one per window id) + ``mel-references.json`` (metadata binding corpus
checksums, preprocessor config, and NeMo version).
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CORPUS_DIR = REPO / "tests" / "parity" / "fixtures" / "corpus"
MEL_DIR = REPO / "tests" / "parity" / "fixtures" / "references" / "mel"

sys.path.insert(0, str(REPO / "services" / "titanet"))

# Keep the committed fixture small: a spread of lengths, SNR regimes, and both
# synthetic-speaker sexes rather than all ~92 embeddable windows.
SELECTED_WINDOW_COUNT = 12


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _select_windows(windows: list[dict]) -> list[dict]:
    """Deterministic diverse pick: sort embeddable windows by duration and take
    an even spread, which naturally mixes short/long and clean/noisy regions."""
    ok = [w for w in windows if w.get("expected_skip_reason") is None]
    ok_sorted = sorted(ok, key=lambda w: (w["end_seconds"] - w["start_seconds"], w["id"]))
    if len(ok_sorted) <= SELECTED_WINDOW_COUNT:
        return ok_sorted
    idx = np.linspace(0, len(ok_sorted) - 1, SELECTED_WINDOW_COUNT).round().astype(int)
    return [ok_sorted[i] for i in sorted(set(int(i) for i in idx))]


def main() -> None:
    import nemo
    import nemo.collections.asr as nemo_asr
    import soundfile as sf
    import torch
    from app.preprocess import (
        normalize_audio_for_embedding,
        window_sample_bounds,
    )

    windows = json.loads((CORPUS_DIR / "embed-windows.json").read_text())["windows"]
    selected = _select_windows(windows)
    print(f"selected {len(selected)} windows: {[w['id'] for w in selected]}")

    audio, sample_rate = sf.read(CORPUS_DIR / "embed-corpus.wav", dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != 16000:
        raise SystemExit(f"corpus must be 16 kHz, got {sample_rate}")

    model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
        model_name="nvidia/speakerverification_en_titanet_large", map_location="cpu"
    )
    model.eval()
    preprocessor = model.preprocessor
    if preprocessor.training:
        raise SystemExit("preprocessor unexpectedly in training mode (dither would fire)")

    arrays: dict[str, np.ndarray] = {}
    meta_windows = []
    for w in selected:
        start_sample, end_sample = window_sample_bounds(
            w["start_seconds"], w["end_seconds"], sample_rate, len(audio)
        )
        segment = audio[start_sample:end_sample]
        normalized = normalize_audio_for_embedding(segment, sample_rate).astype(np.float32)

        signal = torch.from_numpy(normalized).unsqueeze(0)
        length = torch.tensor([signal.shape[1]])
        with torch.no_grad():
            mel, mel_len = preprocessor(input_signal=signal, length=length)

        wid = str(w["id"])
        # Also persist the mel INPUT (post-preprocess audio) so the mel gate
        # measures app/mel.py in isolation — a drift in noisereduce/pyloudnorm
        # versions must not masquerade as a mel front-end mismatch (the vector
        # gate covers the full chain separately).
        arrays[f"audio_{wid}"] = normalized
        arrays[f"mel_{wid}"] = mel.squeeze(0).cpu().numpy().astype(np.float32)
        meta_windows.append(
            {
                "id": w["id"],
                "start_seconds": w["start_seconds"],
                "end_seconds": w["end_seconds"],
                "mel_shape": list(arrays[f"mel_{wid}"].shape),
                "mel_valid_frames": int(mel_len.item()),
            }
        )
        print(f"window {wid}: mel {arrays[f'mel_{wid}'].shape}, valid frames {int(mel_len.item())}")

    MEL_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(MEL_DIR / "mel-references.npz", **arrays)
    meta = {
        "generated": date.today().isoformat(),
        "generator": "tools/generate_mel_references.py",
        "nemo_version": nemo.__version__,
        "device": "cpu",
        "note": (
            "NeMo AudioToMelSpectrogramPreprocessor outputs (eval mode, dither inert) on "
            "titanet-large-v2-preprocessed corpus windows; comparator for services/titanet/"
            "app/mel.py in tests/parity/test_titanet_onnx.py"
        ),
        "corpus_files_sha256": {
            "embed-corpus.wav": _sha256(CORPUS_DIR / "embed-corpus.wav"),
            "embed-windows.json": _sha256(CORPUS_DIR / "embed-windows.json"),
        },
        "windows": meta_windows,
    }
    (MEL_DIR / "mel-references.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {MEL_DIR / 'mel-references.npz'} + metadata")


if __name__ == "__main__":
    main()
