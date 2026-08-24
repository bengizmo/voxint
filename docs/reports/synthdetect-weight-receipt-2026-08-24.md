> **Status:** S2b freeze evidence. Dated weight receipt for the synthdetect
> default detector `w2v2-aasist` (issue #144, Milestone 1 Session S2b). Records
> the real downloaded bytes that the registry pins now reference. Qualification
> state after this receipt is `pinned_unqualified`; it advances to `qualified`
> only after the GPU determinism spike passes (see `docs/gpu-contracts.md`).

# synthdetect weight receipt: w2v2-aasist (2026-08-24)

## Summary

The two weight files for the default candidate detector `w2v2-aasist` were
retrieved to maintainer storage on 2026-08-24, hashed with the harness receipt
tool, and their sha256 + byte size frozen into `tools/synthdetect_sources.py`
(`MODELS['w2v2-aasist']`) and `services/synthdetect/provenance.eval.json`. Both
files verify `match` against the frozen pins, and `weights_pinned()` is now
`True`.

## Files

| File | Role | Size (bytes) | sha256 | License |
|---|---|---|---|---|
| `LA_model.pth` | `aasist_checkpoint` | 1271633441 | `bd6f36097259fe54e7004eb983651e5304d807be81156dbd04faccb70d91e10c` | MIT |
| `xlsr2_300m.pt` | `xlsr_ssl_base` | 3808868242 | `b08927597f2c9eb2ebd7dcc3ac78ee4b5f6021cbac4b3a6c5a9deec445d80ed9` | MIT |

## Provenance

| Field | Value |
|---|---|
| Retrieval date | 2026-08-24 |
| `LA_model.pth` source | Google Drive folder `1c4ywztEVlYVijfwbGLl9OEa1SNtFKppB`, linked from the upstream repository README |
| `xlsr2_300m.pt` source | `https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt` |
| Model repository | `TakHemlata/SSL_Anti-spoofing` |
| Model repository commit | `4acaa61dcef5f7610f43aa4d0b29c4559b970cd2` |
| fairseq runtime commit | `a54021305d6b3c4c5959ac9395135f63202db8f1` (the revision the upstream README pins) |
| Eval base image digest | `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04@sha256:8f9dd0d09d3ad3900357a1cf7f887888b5b74056636cd6ef03c160c3cd4b1d95` |
| Receipt tool | `tools/synthdetect_infer.py verify-sources`, runner at commit `a0ac52edde28af7213b5eab3a0c77d3117843743` |

## License disposition

The upstream `TakHemlata/SSL_Anti-spoofing` repository carries an MIT `LICENSE`
file (Copyright (c) 2022 Hemlata), confirmed at acquisition against the GitHub
license metadata and the file at the pinned commit. Both weight files inherit the
MIT terms recorded in the registry (`weights_license_spdx="MIT"`). The `w2v2-aasist`
entry is `license_class="shippable"`: permissive code and weights. No Hugging Face
account or token is required for either file.

The Drive folder also contains a second checkpoint, `Best_LA_model_for_DF.pth`,
the epoch upstream selects for the ASVspoof 2021 DF track. The registry pins
`LA_model.pth` as the S2b default. Which checkpoint reproduces the DF EER target
(2.85%, provisional) is a reproduction question deferred to S3/S4, not a byte or
determinism question, so this receipt freezes the registry-named file and leaves
the DF checkpoint set aside for that later decision.

## Verification

`tools/synthdetect_infer.py verify-sources --model-id w2v2-aasist` against the
maintainer weights directory reports both files present, `verdict: match`,
`registry_state: pinned`, and `weights_pinned: true`. The vendored upstream model
definition `tools/synthdetect_vendor/ssl_antispoofing_model.py` is byte-identical
to `model.py` at the pinned model-repository commit (sha256
`08b2b99b9cc0e90732746471325185f2eb144795ee35338e0a02951015a856c6`; see
`tools/synthdetect_vendor/provenance.json`).
