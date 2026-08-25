> **Status:** S3 pre-registration evidence. Dated weight receipt for the
> synthdetect DF reproduction checkpoint `w2v2-aasist-df` (issue #144, Milestone 1
> Session S3). Records the real downloaded bytes that the registry pins now
> reference. This receipt records a byte fact only. The checkpoint's own GPU
> determinism plus smoke verdict has since passed
> (`synthdetect-gpu-smoke-df-2026-08-25.md`), advancing it to `qualified`; see the
> "Qualification" section below and `docs/gpu-contracts.md`.

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

## Qualification (verdict passed 2026-08-25)

This receipt freezes bytes; it does not run the checkpoint. Promotion to
`qualified` needed its own dated GPU evidence, which has now passed and is recorded
in `docs/reports/synthdetect-gpu-smoke-df-2026-08-25.md`: a strict state-dict load
with no missing or unexpected keys, an all-modules-in-eval assertion, finite
one-score-per-window outputs with correct counts and polarity, a resume that does
not duplicate, deliberate mismatches that fail closed, and a determinism spike that
is bit-for-bit identical across four cold container starts.

Proving the load cold was warranted. Both the default and DF checkpoints are bare
state dicts (no optimizer or `cfg` wrapper), but the DF checkpoint's 674 keys are
each `module.`-prefixed (it was saved from an `nn.DataParallel`-wrapped model),
while the default's are unprefixed. The registry declares that prefix as data on
the checkpoint's `WeightFile` (`state_dict_key_prefix="module."`,
`SOURCES_VERSION=synthdetect-sources-v3`), and the runner strips it from the keys
and the `_metadata` map before the strict load; on a single GPU this is numerically
identical to upstream's DataParallel evaluation. The registry entry is now
QUALIFIED.
