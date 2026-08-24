> **Status:** S2b qualification evidence. GPU smoke and determinism spike for the
> synthdetect default detector `w2v2-aasist` (issue #144, Milestone 1 Session
> S2b), run on maintainer hardware (one RTX 3060, SM 8.6). Together with the
> weight receipt (`synthdetect-weight-receipt-2026-08-24.md`) this advances the
> eval runtime to `qualified`.

# synthdetect GPU smoke and determinism spike: w2v2-aasist (2026-08-24)

## Runtime under test

| Field | Value |
|---|---|
| Eval image | `voxint-synthdetect-eval:s2b`, id `sha256:d631e02156245c6a2245c32376d260fa8c8624608f590b7fc82de0107f4e6595` |
| Base image | `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04@sha256:8f9dd0d0...` |
| torch / CUDA / cuDNN | `2.1.0+cu118` / `11.8` / `8700` |
| fairseq | `1.0.0a0+a540213` (commit `a54021305d6b3c4c5959ac9395135f63202db8f1`) |
| Python | `3.10.12` |
| GPU | RTX 3060, device capability `[8, 6]` |
| Determinism env | `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONHASHSEED=0`, deterministic algorithms on (`warn_only=false`), cuDNN benchmark off, TF32 off |
| Weights | `LA_model.pth` + `xlsr2_300m.pt`, sha-verified against the frozen pins, mounted read-only |

## GPU smoke (functional acceptance)

Corpus: three deterministic canonical-PCM clips (16 kHz mono s16le): `short`
(30000 samples, repeat-padded), `exact` (64600 samples, one full window), `long`
(150000 samples, upstream-prefixed to one window). Windowing: upstream (64,600
prefix + repeat-pad, one window per clip).

| Check | Result |
|---|---|
| Functional run exits 0 | PASS |
| Journal header valid; `flags.model_eval=true`; `flags.inference_mode=true`; polarity `higher-is-more-synthetic` | PASS |
| Three finite raw scores, `n_windows=1` each | PASS (`short -6.978631`, `exact -7.000616`, `long -6.971259`) |
| Strict checkpoint load (no missing/unexpected keys) | PASS (a non-strict load would have raised) |
| Every module out of training mode after `eval()` | PASS (measured `model_eval=true`) |
| Resume adds no duplicate lines | PASS (4 lines before and after `--resume`) |
| Fail closed on a wrong clip sha256 | PASS (nonzero exit, payload-sha error) |
| Fail closed on a tampered weight (wrong `LA_model.pth`) | PASS (nonzero exit, `sha256 mismatch`) |
| Fail closed on a header-identity change on `--resume` (windowing changed) | PASS (nonzero exit) |

The scores are negative because the fixtures are random noise, which reads as
bona fide under the negated bona-fide-logit polarity. The smoke asserts the
pipeline mechanics and fail-closed behavior, not accuracy; accuracy is the S3/S4
reproduction gates.

## Determinism spike

Four cold container starts on the same cohort, GPU, order, batch, and seed
(`--seed 1234`, upstream windowing). Because upstream windowing is one window per
clip, each journaled raw score is its single per-window logit, so the per-clip
comparison is also the per-window comparison.

| Metric | Value |
|---|---|
| Cold runs | 4 |
| `execution_identity_sha256` across runs | identical (`95e16ec56f9fbfc3112ddabc5b875fd5163db90da5c9883550140b441effc34e`) |
| Max abs diff of per-clip scores vs run 1 | `0.0` |
| Bit-identical `repr` for every clip across all runs | yes |
| NaN / Inf | none |

Per-clip scores (all four runs identical): `short -6.978631496429443`, `exact
-7.000616073608398`, `long -6.971259117126465`.

### One benign warning, pre-declared

Each run emits a single torch `UserWarning` at model construction:
`torch.nn.utils.weight_norm is deprecated in favor of
torch.nn.utils.parametrizations.weight_norm`. This is an API-naming deprecation
from the upstream wav2vec2 layers, emitted deterministically every run, and it
does not bear on numeric reproducibility. Determinism is enforced with
`torch.use_deterministic_algorithms(True, warn_only=False)`, which raises rather
than warns if any op lacks a deterministic implementation; the runs completed, so
every op used has one. The vendored upstream model file is kept byte-identical to
its pinned commit, so the deprecated call is not rewritten here.

## Verdict

The `w2v2-aasist` eval runtime is `qualified` on RTX 3060 (SM 8.6) hardware:
strict load, eval-mode, correct polarity and window counts, fail-closed on every
tampering path, and bit-exact scores across four cold starts. This is a
determinism and smoke claim for the frozen runtime, GPU class, and batch
configuration, not a benchmark-reproduction claim. Reproducing the ASVspoof 2021
DF EER target is S3/S4.
