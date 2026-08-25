> **Status:** S3 pre-registration evidence. Dated weight receipt for the
> synthdetect DF reproduction checkpoint `w2v2-aasist-df` (issue #144, Milestone 1
> Session S3). Records the real downloaded bytes that the registry pins now
> reference. Qualification state after this receipt is `pinned_unqualified`; it
> advances to `qualified` only after this checkpoint's own GPU determinism plus
> smoke verdict passes (see `docs/gpu-contracts.md`). This receipt is a byte
> fact, not a reproduction or determinism claim.

# synthdetect weight receipt: w2v2-aasist-df (2026-08-25)

## Summary

The DF-tuned checkpoint `Best_LA_model_for_DF.pth` and its XLS-R front end were
hashed with the harness receipt tool on 2026-08-25 and their sha256 plus byte
size frozen into `tools/synthdetect_sources.py` (`MODELS['w2v2-aasist-df']`). Both
files verify `match` against the frozen pins, and `weights_pinned()` is `True` for
the entry. This checkpoint, not the production default `w2v2-aasist`
(`LA_model.pth`), is the one the upstream 2.85 % ASVspoof 2021 DF EER is achieved
with, so it carries the hard DF reproduction stop-gate (S3 decision; see the S3
pre-registration in `docs/gpu-contracts.md`).

The `Best_LA_model_for_DF.pth` bytes were retrieved in the same 2026-08-24
maintainer-storage acquisition as the S2b default weights (see
`docs/reports/synthdetect-weight-receipt-2026-08-24.md`, which set the DF
checkpoint aside for exactly this S3 decision). The XLS-R base `xlsr2_300m.pt` is
byte-identical to the default's base (same sha256).

## Files

| File | Role | Size (bytes) | sha256 | License |
|---|---|---|---|---|
| `Best_LA_model_for_DF.pth` | `aasist_checkpoint` | 1271642081 | `1cf904f1d84c867c278cd42161df5367939d61cc28bfefd239bc995af59c2804` | MIT |
| `xlsr2_300m.pt` | `xlsr_ssl_base` | 3808868242 | `b08927597f2c9eb2ebd7dcc3ac78ee4b5f6021cbac4b3a6c5a9deec445d80ed9` | MIT |

## Provenance

| Field | Value |
|---|---|
| Retrieval date | 2026-08-24 (hashed and frozen 2026-08-25) |
| `Best_LA_model_for_DF.pth` source | Google Drive file `1JHBClArVdM-Cr1b8In1iakTV_TvC3HvG`, the upstream README's designated DF checkpoint, in the same folder as `LA_model.pth` |
| `xlsr2_300m.pt` source | `https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt` |
| Model repository | `TakHemlata/SSL_Anti-spoofing` |
| Model repository commit | `4acaa61dcef5f7610f43aa4d0b29c4559b970cd2` (same vendored `model.py` as the default) |
| fairseq runtime commit | `a54021305d6b3c4c5959ac9395135f63202db8f1` (the revision the upstream README pins) |
| Eval base image digest | `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04@sha256:8f9dd0d09d3ad3900357a1cf7f887888b5b74056636cd6ef03c160c3cd4b1d95` |
| Receipt tool | `tools/synthdetect_infer.py verify-sources`, runner at commit `2ce748f137af6c6df8dcd1e7c81bc2bdce02bfa6` |

## License disposition

`Best_LA_model_for_DF.pth` is distributed in the same upstream
`TakHemlata/SSL_Anti-spoofing` repository (MIT `LICENSE`, Copyright (c) 2022
Hemlata) as the default `LA_model.pth`, from the same author Google Drive folder
linked in the README. It inherits the MIT terms recorded in the registry
(`weights_license_spdx="MIT"`), and the `w2v2-aasist-df` entry is
`license_class="shippable"`. No Hugging Face account or token is required.

## Verification

`tools/synthdetect_infer.py verify-sources --model-id w2v2-aasist-df` against a
weights directory holding both files reports `all_present: true`, both files
`verdict: match`, `registry_state: pinned`, `inference_space:
synthdetect-w2v2aasistdf-v1`, and `weights_pinned: true`. The vendored upstream
model definition `tools/synthdetect_vendor/ssl_antispoofing_model.py` is
byte-identical to `model.py` at the pinned model-repository commit (sha256
`08b2b99b9cc0e90732746471325185f2eb144795ee35338e0a02951015a856c6`), and is shared
with the default detector: the two models differ only in the aasist checkpoint.

## Not yet qualified

This receipt freezes bytes. It does not run the checkpoint. Promoting
`w2v2-aasist-df` to `qualified` needs its own dated GPU evidence: a strict
state-dict load with no missing or unexpected keys (a fairseq checkpoint may carry
optimizer or `cfg` state, so the load is proven cold, not assumed from the
default), an all-modules-in-eval assertion, finite one-score-per-window outputs
with correct counts and polarity, a resume that does not duplicate, a deliberate
mismatch that fails closed, and a determinism spike that is bit-for-bit identical
across cold container starts. That verdict is the next GPU step in S3.
