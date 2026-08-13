#!/usr/bin/env python3
"""In-container A/B: service embed path vs direct forward (drift attribution).

    docker run --rm -v $PWD:/repo -w /repo ghcr.io/bengizmo/voxint-titanet:0.3.0 \
        python3 tools/probe_roundtrip_ab.py

For the worst-drifting corpus windows, embeds each three ways INSIDE the pinned
image on CPU and prints pairwise cosines vs the committed CUDA reference:

  A. the actual service path (app.engine_nemo.NemoEmbedder.embed_windows —
     torchaudio decode, temp-wav round trip, get_embedding);
  B. direct: soundfile decode → app.preprocess chain → model.preprocessor →
     model.forward (no temp wav);
  C. B's audio but passed through A's temp-wav save/load
     (torchaudio.save → sf.read) before the forward.

Whichever leg diverges is the chain stage responsible. Maintainer diagnostics
only; prints, writes nothing.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CORPUS_DIR = REPO / "tests" / "parity" / "fixtures" / "corpus"
CUDA_REF = REPO / "tests" / "parity" / "fixtures" / "references" / "cuda" / "embed.json"

sys.path.insert(0, str(REPO / "services" / "titanet"))

PROBE = ["gap_straddle_1", "lenb_1.001", "lenb_1.0", "sub_utt_13_0", "clean_utt_00"]


def main() -> None:
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    from app.embedding import create_embedder
    from app.preprocess import (
        l2_normalize,
        normalize_audio_for_embedding,
        window_sample_bounds,
    )

    windows = {
        w["id"]: w for w in json.loads((CORPUS_DIR / "embed-windows.json").read_text())["windows"]
    }
    ref = {
        w["id"]: np.array(w["embedding"])
        for w in json.loads(CUDA_REF.read_text())["windows"]
        if w["embedding"] is not None
    }

    def cos(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    embedder = create_embedder()
    embedder.load_model()
    model = embedder.model

    corpus = str(CORPUS_DIR / "embed-corpus.wav")
    outcomes = embedder.embed_windows(
        corpus, [(windows[i]["start_seconds"], windows[i]["end_seconds"]) for i in PROBE]
    )
    audio_all, sr = sf.read(corpus, dtype="float32")

    for wid, a_out in zip(PROBE, outcomes, strict=True):
        if a_out.embedding is None:
            print(f"{wid}: service skipped ({a_out.skip_reason})")
            continue
        w = windows[wid]
        s, e = window_sample_bounds(w["start_seconds"], w["end_seconds"], sr, len(audio_all))
        normalized = normalize_audio_for_embedding(audio_all[s:e], sr).astype(np.float32)

        def fwd(audio_np: np.ndarray) -> np.ndarray:
            signal = torch.from_numpy(audio_np.astype(np.float32)).unsqueeze(0)
            length = torch.tensor([signal.shape[1]])
            with torch.no_grad():
                _logits, emb = model.forward(input_signal=signal, input_signal_length=length)
            return l2_normalize(emb.cpu().numpy().squeeze())

        b_emb = fwd(normalized)

        with tempfile.NamedTemporaryFile(suffix=".wav") as tf:
            torchaudio.save(tf.name, torch.from_numpy(normalized).unsqueeze(0), sr)
            rt, rt_sr = sf.read(tf.name)
            info = sf.info(tf.name)
        c_emb = fwd(np.asarray(rt, dtype=np.float32))

        a_emb = np.array(a_out.embedding)
        print(
            f"{wid}: A-vs-ref {cos(a_emb, ref[wid]):.6f}  B-vs-ref {cos(b_emb, ref[wid]):.6f}  "
            f"C-vs-ref {cos(c_emb, ref[wid]):.6f}  A-vs-B {cos(a_emb, b_emb):.6f}  "
            f"tempwav={info.subtype}@{rt_sr}"
        )


if __name__ == "__main__":
    main()
