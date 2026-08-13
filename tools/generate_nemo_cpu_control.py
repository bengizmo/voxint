#!/usr/bin/env python3
"""NeMo-on-CPU control run for the parity verdict (maintainer tool).

Runs INSIDE the pinned titanet image WITHOUT ``--gpus`` so the reference
engine itself executes on CPU:

    docker run --rm -v $PWD:/repo -w /repo ghcr.io/bengizmo/voxint-titanet:0.3.0 \
        python3 tools/generate_nemo_cpu_control.py

Purpose: attribute ONNX-vs-CUDA vector drift. The space doctrine says
bit-identity is already false across hardware; this control measures how far
the REFERENCE engine (NeMo, unchanged code path via app.engine_nemo) moves
when only the compute device changes. If NeMo-CPU shows the same drift vs the
CUDA references as ONNX-CPU does — and NeMo-CPU agrees closely with ONNX-CPU —
the drift is CUDA-side numerics (e.g. Ampere TF32 convolutions), not an ONNX
transcription defect.

Writes ``tests/parity/fixtures/references/nemo-cpu/embed.json`` (same shape as
the CUDA reference: window id → embedding/snr_db/skip_reason).
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CORPUS_DIR = REPO / "tests" / "parity" / "fixtures" / "corpus"
OUT_DIR = REPO / "tests" / "parity" / "fixtures" / "references" / "nemo-cpu"

sys.path.insert(0, str(REPO / "services" / "titanet"))


def main() -> None:
    import torch

    if torch.cuda.is_available():
        raise SystemExit("run WITHOUT --gpus: this control must execute NeMo on CPU")

    from app.embedding import create_embedder

    embedder = create_embedder()  # EMBED_ENGINE default: nemo
    embedder.load_model()
    assert embedder.device_name == "cpu"

    windows = json.loads((CORPUS_DIR / "embed-windows.json").read_text())["windows"]
    outcomes = embedder.embed_windows(
        str(CORPUS_DIR / "embed-corpus.wav"),
        [(w["start_seconds"], w["end_seconds"]) for w in windows],
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "generated_on": date.today().isoformat(),
            "generator": "tools/generate_nemo_cpu_control.py",
            "engine": embedder.engine,
            "engine_version": embedder.engine_version,
            "device": embedder.device_name,
            "note": "NeMo reference engine on CPU — drift-attribution control, not a serving path",
        },
        "embedding_space": "titanet-large-v1",
        "windows": [
            {
                "id": w["id"],
                "embedding": o.embedding,
                "snr_db": o.snr_db,
                "skip_reason": o.skip_reason,
            }
            for w, o in zip(windows, outcomes, strict=True)
        ],
    }
    (OUT_DIR / "embed.json").write_text(json.dumps(payload) + "\n")
    embedded = sum(1 for o in outcomes if o.embedding is not None)
    print(f"wrote {OUT_DIR / 'embed.json'} ({embedded}/{len(outcomes)} embedded)")


if __name__ == "__main__":
    main()
