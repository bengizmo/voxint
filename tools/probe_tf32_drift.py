#!/usr/bin/env python3
"""Drift-attribution probe: is ONNX-vs-CUDA vector drift explained by TF32?

Runs INSIDE the pinned titanet image WITH the GPU:

    docker run --rm --gpus '"device=1"' -v $PWD:/repo -w /repo \
        ghcr.io/bengizmo/voxint-titanet:0.3.0 python3 tools/probe_tf32_drift.py

For a handful of the worst-drifting corpus windows, embeds each with the NeMo
engine on CUDA twice — cuDNN TF32 as shipped (Ampere default: ON) and TF32
forced off — and prints cosines against the committed CUDA reference. If the
TF32-off embedding moves materially away from the (TF32-on) reference, the
committed reference carries TF32 numeric noise of the same order as the
ONNX-vs-CUDA drift, which attributes the drift to CUDA-side numerics rather
than the ONNX transcription. Maintainer diagnostics only; prints, writes nothing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CORPUS_DIR = REPO / "tests" / "parity" / "fixtures" / "corpus"
CUDA_REF = REPO / "tests" / "parity" / "fixtures" / "references" / "cuda" / "embed.json"

sys.path.insert(0, str(REPO / "services" / "titanet"))

PROBE_WINDOWS = ["gap_straddle_1", "lenb_1.001", "lenb_1.0", "sub_utt_13_0", "clean_utt_00"]


def main() -> None:
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("probe needs the GPU (--gpus)")

    from app.embedding import create_embedder

    windows = json.loads((CORPUS_DIR / "embed-windows.json").read_text())["windows"]
    by_id = {w["id"]: w for w in windows}
    ref = {
        w["id"]: np.array(w["embedding"])
        for w in json.loads(CUDA_REF.read_text())["windows"]
        if w["embedding"] is not None
    }
    probes = [(by_id[i]["start_seconds"], by_id[i]["end_seconds"]) for i in PROBE_WINDOWS]

    def cos(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    embedder = create_embedder()
    embedder.load_model()
    print(
        f"cudnn.allow_tf32 default: {torch.backends.cudnn.allow_tf32}, "
        f"matmul.allow_tf32 default: {torch.backends.cuda.matmul.allow_tf32}"
    )

    results: dict[str, dict[str, float]] = {i: {} for i in PROBE_WINDOWS}
    for label, tf32 in (("tf32_on", True), ("tf32_off", False)):
        torch.backends.cudnn.allow_tf32 = tf32
        torch.backends.cuda.matmul.allow_tf32 = tf32
        outcomes = embedder.embed_windows(str(CORPUS_DIR / "embed-corpus.wav"), probes)
        for wid, o in zip(PROBE_WINDOWS, outcomes, strict=True):
            if o.embedding is not None and wid in ref:
                results[wid][label] = cos(np.array(o.embedding), ref[wid])

    for wid, row in results.items():
        print(f"{wid}: vs committed CUDA ref — {row}")


if __name__ == "__main__":
    main()
