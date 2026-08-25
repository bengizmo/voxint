> **Status:** S3 qualification evidence. GPU smoke and determinism spike for the
> synthdetect DF reproduction anchor `w2v2-aasist-df` (issue #144, Milestone 1
> Session S3), run on maintainer hardware (one RTX 3060, SM 8.6). Together with
> the weight receipt (`synthdetect-weight-receipt-df-2026-08-25.md`) this advances
> the DF anchor from `pinned_unqualified` to `qualified`. It is a determinism and
> smoke claim for the frozen runtime, GPU class, and batch configuration, not a
> benchmark-reproduction claim: reproducing the 2.85% ASVspoof 2021 DF EER is the
> S3 compare step and S4.

# synthdetect GPU smoke and determinism spike: w2v2-aasist-df (2026-08-25)

## Why this checkpoint needed its own verdict

`w2v2-aasist-df` shares the vendored model definition and XLS-R base with the
already-qualified default `w2v2-aasist`, but it is a different checkpoint
(`Best_LA_model_for_DF.pth`), so its load was proven cold rather than assumed from
the default. That caution was warranted: the default's checkpoint is a bare state
dict with unprefixed keys, while the DF checkpoint is a bare state dict whose 674
keys are each `module.`-prefixed (it was saved from an `nn.DataParallel`-wrapped
model). The registry now declares that prefix as data on the checkpoint's
`WeightFile` (`state_dict_key_prefix="module."`), and the runner strips it from
the tensor keys and the `_metadata` map before a strict load, leaving a state dict
structurally identical to a natively-saved one. On a single GPU the unwrapped
forward is numerically identical to the DataParallel forward, so this matches the
upstream single-GPU evaluation. The strip fails closed unless every key uniformly
carries the declared prefix, and the load stays `strict=True`.

## Runtime under test

| Field | Value |
|---|---|
| Eval image | `voxint-synthdetect-eval`, id `sha256:d631e02156245c6a2245c32376d260fa8c8624608f590b7fc82de0107f4e6595` (the pinned runtime is unchanged from S2b: same `Dockerfile.eval` and base digest, no app code or weights baked) |
| Base image | `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04@sha256:8f9dd0d0...` |
| torch / CUDA / cuDNN | `2.1.0+cu118` / `11.8` / `8700` |
| fairseq | `1.0.0a0+a540213` (commit `a54021305d6b3c4c5959ac9395135f63202db8f1`) |
| Python | `3.10.12` |
| GPU | RTX 3060, device capability `[8, 6]` (the flex GPU, exposed to the container as its only visible device) |
| Determinism env | `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONHASHSEED=0`, deterministic algorithms on (`warn_only=false`), cuDNN benchmark off, TF32 off |
| Weights | `Best_LA_model_for_DF.pth` + `xlsr2_300m.pt`, sha-verified against the frozen pins, mounted read-only |
| provenance.eval.json sha256 | `3009933f664785672a2824869803d238d4be580dff2279768ea8d533ece91a27` |

## GPU smoke (functional acceptance)

Corpus: three deterministic canonical-PCM clips (16 kHz mono s16le), noise from a
fixed host seed: `short` (30000 samples, repeat-padded to one window), `exact`
(64600 samples, one full window), `long` (150000 samples, upstream-prefixed to one
window). Windowing: upstream (64,600 prefix, one window per clip). Manifest sha256
`f848fd8d1b9c598f2123972287614e3474c4d5075a611042e99ac3e772a5481f`.

| Check | Result |
|---|---|
| Functional run exits 0 | PASS (`scored=3`) |
| Journal header valid; `flags.model_eval=true`; `flags.inference_mode=true`; polarity `higher-is-more-synthetic` | PASS |
| Three finite raw scores, `n_windows=1` each | PASS (`short -4.420719`, `exact -4.294983`, `long -4.283033`) |
| Strict checkpoint load after the declared `module.` unwrap (no missing/unexpected keys) | PASS (a non-strict or un-stripped load would have raised) |
| `checkpoint_loading.state_dict_key_prefix_removed` recorded in the header | PASS (`"module."`, and it flows into `execution_identity_sha256`) |
| Every module out of training mode after `eval()` | PASS (measured `model_eval=true`) |
| Resume adds no duplicate lines | PASS (4 lines before and after `--resume`, `resumed=3`) |
| Fail closed on a wrong clip sha256 | PASS (nonzero exit, payload-sha error) |
| Fail closed on a tampered weight (default `LA_model.pth` bytes under the DF name) | PASS (nonzero exit, `sha256 mismatch: mounted bd6f3609... != pinned 1cf904f1...`) |
| Fail closed on a header-identity change on `--resume` (windowing upstream to production) | PASS (nonzero exit, execution-identity error) |

The scores are negative because the fixtures are random noise, which reads as bona
fide under the negated bona-fide-logit polarity. The smoke asserts the pipeline
mechanics, the DataParallel-unwrap load, and fail-closed behavior, not accuracy;
accuracy is the S3 compare step and S4 reproduction gates.

## Determinism spike

Four cold container starts on the same cohort, GPU, order, batch, and seed
(`--seed 1234`, upstream windowing). Because upstream windowing is one window per
clip, each journaled raw score is its single per-window logit, so the per-clip
comparison is also the per-window comparison.

| Metric | Value |
|---|---|
| Cold runs | 4 |
| `execution_identity_sha256` across runs | identical (`93ff606bcd3b370fe5b7a073758cb24f37cfae8808fbd5e3a5760d488fbdb3ca`) |
| Max abs diff of per-clip scores vs run 1 | `0.0` |
| Bit-identical `repr` for every clip across all runs | yes |
| NaN / Inf | none |

Per-clip scores (all four runs identical): `short -4.420718669891357`, `exact
-4.29498291015625`, `long -4.28303337097168`.

The `execution_identity_sha256` here differs from the default's S2b value, which is
correct: this is a different checkpoint and its header additionally carries the
`checkpoint_loading` record, so a resume can never mix the two load semantics.

### One benign warning, pre-declared

Each run emits a single torch `UserWarning` at model construction:
`torch.nn.utils.weight_norm is deprecated in favor of
torch.nn.utils.parametrizations.weight_norm`. This is an API-naming deprecation
from the upstream wav2vec2 layers, emitted deterministically every run, and it does
not bear on numeric reproducibility. Determinism is enforced with
`torch.use_deterministic_algorithms(True, warn_only=False)`, which raises rather
than warns if any op lacks a deterministic implementation; the runs completed, so
every op used has one. The vendored upstream model file is kept byte-identical to
its pinned commit, so the deprecated call is not rewritten here.

## Verdict

The `w2v2-aasist-df` eval runtime is `qualified` on RTX 3060 (SM 8.6) hardware:
strict load after the declared `module.` DataParallel unwrap, eval-mode, correct
polarity and window counts, fail-closed on every tampering path, and bit-exact
scores across four cold starts. This is a determinism and smoke claim for the
frozen runtime, GPU class, and batch configuration, not a benchmark-reproduction
claim. Reproducing the ASVspoof 2021 DF EER target is the S3 compare step (against
the unmodified upstream runner on a seeded subset) and S4 (the full DF cohort).
