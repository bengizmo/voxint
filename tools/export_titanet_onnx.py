#!/usr/bin/env python3
"""Export the pinned TitaNet-Large checkpoint to ONNX with full provenance.

Runs INSIDE the voxint titanet service image (NeMo 1.22.0, torch cu118,
Python 3.10) so the export comes from the exact runtime that defines the
``titanet-large-v1`` embedding space — never from an ad-hoc environment:

    docker run --rm --gpus '"device=1"' \
        -v $PWD:/repo -w /repo \
        ghcr.io/bengizmo/voxint-titanet:0.3.0 \
        bash -c "pip3 install --no-cache-dir onnx==1.16.2 onnxruntime==1.18.1 && \
                 python3 tools/export_titanet_onnx.py"

The .onnx artifact itself is NOT committed (≈90 MB; users and CI regenerate it
with this script, or fetch a checksummed release asset). What IS committed:

* ``tests/parity/fixtures/onnx/provenance.json`` — versions, opset, sha256 of
  the source .nemo and the exported .onnx, license note, dither semantics.
* ``tests/parity/fixtures/onnx/preprocessor-config.json`` — the checkpoint's
  ``model.cfg.preprocessor`` dump; the pinned reference the reimplemented mel
  front-end (``services/titanet/app/mel.py``) is written against.

The exported graph is the ACOUSTIC MODEL ONLY (NeMo #7245): inputs are mel
features ``audio_signal [B, n_mels, T]`` + ``length [B]``; the mel front-end
lives outside the graph and must be reimplemented parity-exact.

The script also runs a graph-level sanity check: the checkpoint's own
preprocessor output fed through (a) the in-process NeMo encoder+decoder and
(b) onnxruntime on the exported graph must agree (cosine on the embedding
output). This is NOT the parity gate — just proof the export is not corrupt.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "services" / "titanet" / "models"
FIXTURES_DIR = REPO / "tests" / "parity" / "fixtures" / "onnx"
ONNX_NAME = "titanet-large.onnx"
MODEL_NAME = "nvidia/speakerverification_en_titanet_large"

# Export sanity threshold — graph-transcription fidelity only, far looser than
# the real parity gate (which compares full pipelines on the golden corpus).
SANITY_MIN_COSINE = 0.9999


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_nemo_checkpoint() -> Path:
    """Locate the cached .nemo file the image baked at build time."""
    candidates = sorted(Path("/app/models").rglob("*.nemo")) if Path("/app/models").exists() else []
    if not candidates:
        candidates = sorted(Path.home().rglob("*.nemo"))
    if len(candidates) != 1:
        raise SystemExit(f"expected exactly one cached .nemo checkpoint, found: {candidates}")
    return candidates[0]


def _dither_semantics() -> dict[str, Any]:
    """Record, from the installed NeMo source, when dither is actually applied.

    The plan requires verifying whether NeMo applies dither in eval mode — the
    mel reimplementation must match inference-time behaviour, not training.
    """
    from nemo.collections.asr.parts.preprocessing.features import FilterbankFeatures

    src = inspect.getsource(FilterbankFeatures.forward)
    training_guarded = "self.training" in src and "dither" in src
    return {
        "filterbank_forward_dither_lines": [
            line.strip() for line in src.splitlines() if "dither" in line or "training" in line
        ],
        "dither_applied_only_in_training": training_guarded,
    }


def _stub_tts_submodules() -> bool:
    """Make ``replace_for_export``'s TTS import inert without installing TTS deps.

    NeMo 1.22's ``nemo.utils.export_utils.replace_for_export`` unconditionally
    imports ``MaskedInstanceNorm1d`` from the TTS collection, which drags in
    einops/torchvision/... — none of which the pinned titanet image ships. The
    symbol is only used to *look for* MaskedInstanceNorm1d instances inside the
    exported model, and TitaNet contains none, so a stub class that can never
    match is behavior-identical for this export. Returns True if stubbing was
    needed (recorded in provenance).
    """
    try:
        from nemo.collections.tts.modules.submodules import MaskedInstanceNorm1d  # noqa: F401

        return False
    except Exception:
        import sys
        import types

        import torch.nn as nn

        class _StubMaskedInstanceNorm1d(nn.InstanceNorm1d):
            pass

        mod = types.ModuleType("nemo.collections.tts.modules.submodules")
        mod.MaskedInstanceNorm1d = _StubMaskedInstanceNorm1d  # type: ignore[attr-defined]
        for name in ("nemo.collections.tts", "nemo.collections.tts.modules"):
            sys.modules.setdefault(name, types.ModuleType(name))
        sys.modules["nemo.collections.tts.modules.submodules"] = mod
        return True


def main() -> None:
    import nemo
    import nemo.collections.asr as nemo_asr
    import onnx
    import onnxruntime as ort
    import torch
    from omegaconf import OmegaConf

    ckpt = _find_nemo_checkpoint()
    print(f"checkpoint: {ckpt}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
        model_name=MODEL_NAME, map_location=device
    )
    model.eval()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = OUT_DIR / ONNX_NAME

    preproc_cfg = OmegaConf.to_container(model.cfg.preprocessor, resolve=True)
    if not isinstance(preproc_cfg, dict):
        raise SystemExit(f"unexpected preprocessor config shape: {type(preproc_cfg)}")
    (FIXTURES_DIR / "preprocessor-config.json").write_text(
        json.dumps(preproc_cfg, indent=2, sort_keys=True) + "\n"
    )
    print(f"preprocessor config: {json.dumps(preproc_cfg, sort_keys=True)}")

    tts_stubbed = _stub_tts_submodules()
    model.export(str(onnx_path))
    graph = onnx.load(str(onnx_path))
    onnx.checker.check_model(graph)
    opsets = {imp.domain or "ai.onnx": imp.version for imp in graph.opset_import}
    inputs = [i.name for i in graph.graph.input]
    outputs = [o.name for o in graph.graph.output]
    print(f"exported: {onnx_path} opsets={opsets} inputs={inputs} outputs={outputs}")

    # --- graph-level sanity: same mel features through NeMo vs onnxruntime ---
    torch.manual_seed(0)
    sr = int(preproc_cfg.get("sample_rate", 16000))
    wav = torch.randn(1, sr * 3, device=device) * 0.1
    lengths = torch.tensor([wav.shape[1]], device=device)
    with torch.no_grad():
        feats, feat_len = model.preprocessor(input_signal=wav, length=lengths)
        _logits_ref, emb_ref = model.forward(input_signal=wav, input_signal_length=lengths)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {
        sess.get_inputs()[0].name: feats.cpu().numpy(),
        sess.get_inputs()[1].name: feat_len.cpu().numpy(),
    }
    ort_outs = sess.run(None, ort_inputs)
    # EncDecSpeakerLabelModel exports (logits, embs) — map by output name.
    out_names = [o.name for o in sess.get_outputs()]
    emb_onnx = ort_outs[out_names.index("embs")] if "embs" in out_names else ort_outs[-1]

    import numpy as np

    a = emb_ref.cpu().numpy().ravel()
    b = np.asarray(emb_onnx).ravel()
    cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    print(f"export sanity cosine (NeMo vs ORT embs): {cosine:.8f}")
    if cosine < SANITY_MIN_COSINE:
        raise SystemExit(f"export sanity FAILED: cosine {cosine} < {SANITY_MIN_COSINE}")

    provenance = {
        "model_name": MODEL_NAME,
        "exported": date.today().isoformat(),
        "nemo_version": nemo.__version__,
        "torch_version": torch.__version__,
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "python_version": sys.version.split()[0],
        "export_device": device,
        "opset_import": opsets,
        "graph_inputs": inputs,
        "graph_outputs": outputs,
        "nemo_checkpoint_file": ckpt.name,
        "nemo_checkpoint_sha256": _sha256(ckpt),
        "onnx_sha256": _sha256(onnx_path),
        "onnx_size_bytes": onnx_path.stat().st_size,
        "onnx_committed": False,
        "onnx_regeneration": (
            "run this script inside the pinned titanet image (see module docstring)"
        ),
        "license": (
            "NVIDIA titanet_large is published under CC-BY-4.0 (NGC / Hugging Face model card). "
            "Redistribution with attribution is permitted; the .onnx is nevertheless kept out of "
            "git for repo-size reasons and regenerated (or fetched as a checksummed release asset)."
        ),
        "export_sanity_cosine": cosine,
        "export_sanity_note": (
            "same checkpoint-preprocessor mel features through in-process NeMo forward vs "
            "onnxruntime CPU EP; graph fidelity only, not the parity gate"
        ),
        "dither_semantics": _dither_semantics(),
        "tts_submodules_stubbed_for_export": tts_stubbed,
        "mel_frontend_in_graph": False,
    }
    (FIXTURES_DIR / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"provenance: {FIXTURES_DIR / 'provenance.json'}")


if __name__ == "__main__":
    main()
