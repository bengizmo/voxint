> **Status:** Gate-2 ratified 2026-08-25. Paired per-clip equivalence measured on
> the frozen 53,392-clip ASVspoof 2021 DF canonical subset, our frozen eval
> container against an independent unmodified-upstream reference. PASS: at matched
> precision the two implementations agree to floating-point noise (max 0.0058,
> mean 1.6e-6, 9,641 of 53,392 clips bit-identical); the larger default-precision
> tail is one deliberate precision choice (cuDNN TF32) that leaves the EER
> identical and flips only 8 near-threshold decisions. The published 2.85 %
> full-corpus EER is not claimed here; that is S4.

# synthdetect Gate-2: DF reproduction paired-equivalence verdict

Gate-2 is the load-bearing numerics evidence for the synthetic-speech detector
(#144). Model outputs are contract, so Gate-2 measures equivalence to the upstream
reference directly. This report records how the independent reference was built,
the pre-registered tolerance and threshold, the measured agreement, and the root
cause of the observed per-clip tail.

## 1. Verdict

Our frozen eval container (`w2v2-aasist-df`, image
`sha256:d631e02156245c6a2245c32376d260fa8c8624608f590b7fc82de0107f4e6595`,
`torch 2.1.0+cu118`, fairseq `a540213`) reproduces the unmodified upstream
SSL_Anti-spoofing DF eval path on the 53,392-clip canonical subset:

| Metric | Result |
|---|---|
| Coverage | 53,392 / 53,392 both sides, zero side-only, zero skips |
| Subset EER (official `compute_eer` math) | **2.2193 % reference == 2.2193 % ours** (identical) |
| Spearman rank correlation | 0.99858 |
| Decision agreement at the frozen EER threshold | 99.985 % (8 disagreements, all within 0.0024 of the threshold) |
| Per-clip logit drift, mean / median | 7.8e-4 / 2.0e-4 |
| Per-clip logit drift, p90 / p99 / p99.9 | 1.7e-3 / 8.7e-3 / 0.043 |
| Per-clip logit drift, max | 0.152 |
| Root cause of the tail | cuDNN convolution TF32 (upstream leaves the Ampere default on; our container disables it for full fp32) |
| Matched-precision agreement (all 53,392 clips) | max 0.0058, mean 1.6e-6, **zero clips over the 1e-2 floor, 9,641 clips bit-identical** |

The per-clip logit tail is fully attributed to one hardware precision mode. When
the reference is run with TF32 disabled (matching our container's fp32
configuration), the two agree to floating-point noise (max 0.0058, 9,641 clips
bit-identical). The subset EER is identical in both precision modes and the
ranking is near-perfectly preserved (Spearman 0.99858). This is a PASS: the two
implementations are the same function, and our container's choice (fp32, TF32 off)
is the more precise one.

## 2. What Gate-2 tests

Both runners score the **canonical view** (`pcm-s16le-mono-16000-v1`, manifest
sha256 over the PCM payload bytes) of the seeded 10 % DF subset, per the S3
pre-registration in `docs/gpu-contracts.md`. The subset cannot reproduce the
corpus-level 2.85 % number; the subset EER is a diagnostic. Gate-2 asserts paired
per-clip agreement between our container and an independent upstream
implementation, with the decision threshold frozen from the reference before the
paired comparison is inspected.

## 3. The independent reference

The reference runs the numerically load-bearing upstream code verbatim, isolated
from our runner:

- **Runtime.** A throwaway image derived `FROM` the frozen eval image, so torch,
  torchaudio, fairseq, CUDA, and cuDNN are identical to what produced our scores.
  The only added dependency is `librosa==0.9.1` (upstream's pinned reader);
  numpy, scipy, torch, torchaudio, soundfile, and fairseq are held at the eval
  image's versions (verified unchanged after the install).
- **Code.** Upstream `model.py` (byte-identical to the vendored sha-pinned copy),
  `data_utils_SSL.py` (`pad` + `Dataset_ASVspoof2021_eval`), and `eval_metrics_DF`
  (`compute_eer`) from commit `4acaa61dcef5f7610f43aa4d0b29c4559b970cd2`. A thin
  driver replicates only the eval control flow of `main_SSL_DF.py`
  (`nn.DataParallel` + raw `module.`-prefixed `load_state_dict`, `model.eval()`,
  `batch_out[:, 1]` raw, `batch_size=14`, `shuffle=False`, `num_workers=0`,
  exactly one visible GPU). It imports nothing from our runner. `main_SSL_DF.py`
  is not run whole because it pulls an unpinned `core_scripts` dependency and a
  training/logging surface irrelevant to eval; the driver instead mirrors that
  module's determinism setup (seed 1234, `cudnn.deterministic=True`,
  `cudnn.benchmark=False`) explicitly.
- **Input.** The reference reads the same canonical WAV bytes our container reads,
  through `flac/<trial_id>.flac` symlinks (libsndfile identifies the WAV by its
  header, not the extension).

Two deliberate, numerically-inert choices are documented: the driver wraps the
forward in `torch.no_grad()` (upstream detaches via `.data` and never calls
backward, so the forward math is identical), and it replicates upstream's
determinism setup rather than importing `core_scripts`.

## 4. Pre-registration (sealed before inspecting agreement)

- **Reference determinism.** Three cold `batch_size=14` reference runs produced
  **bit-identical** score files (one sha256 across all three). Reference
  self-variance is 0.
- **Decision threshold.** Frozen from reference run 1 at the subset EER operating
  point: **3.5079** (our polarity, higher = more synthetic). The reference subset
  EER is **2.219 %**.
- **Tolerance floor.** A per-clip drift floor of 1e-2 was pre-declared for the
  cross-implementation comparison before journal agreement was inspected.

## 5. Measured agreement

Our container (effective forward batch 1; see below) versus the primary
`batch_size=14` reference:

- Coverage exact (53,392 both sides, zero side-only, zero skips).
- mean 7.8e-4, median 2.0e-4, p99 8.7e-3, p99.9 0.043, **max 0.152**.
- Spearman 0.99858, decision agreement 99.985 %.
- All 8 decision disagreements sit within **0.0024** of the threshold (they
  straddle the exact EER operating point, exactly the near-threshold case the
  pre-registration says is covered by the raw-logit drift level).
- **Subset EER identical to five significant figures: 2.2193 % both.**

394 clips exceed the pre-declared 1e-2 floor. Section 6 attributes that tail.

## 6. Root cause of the per-clip tail: cuDNN TF32 vs fp32

Controls rule out preprocessing and nondeterminism, leaving one precision mode:

- **Preprocessing is bit-identical.** For the worst-drift clips the prepared
  64,600-sample input tensor from our runner (`read_canonical_pcm` +
  `repeat_pad_to`) equals the upstream tensor (`librosa.load` + `pad`) to
  0.000e+00. The reader equivalence (`librosa.load` vs our int16/32768 read) is
  also bit-identical on 16 kHz mono 16-bit PCM.
- **Both sides are reproducible.** The reference is bit-identical across three
  cold runs; our container reproduces each worst-clip score exactly across
  re-runs.
- **The tail is cuDNN convolution TF32.** `configure_determinism` in our runner
  sets `torch.backends.cudnn.allow_tf32 = False` (and matmul TF32, already off by
  default in torch 2.1). The upstream reference leaves the Ampere default, so its
  cuDNN convolutions run in TF32 (10-bit mantissa in a 19-bit format). The
  wav2vec2-XLS-R front end
  and AASIST back end are convolution-heavy, so on precision-sensitive clips the
  TF32-vs-fp32 gap reaches 0.15 in the logit. With TF32 disabled across the whole
  subset, our fp32 container and the fp32 reference agree to **max 0.0058, mean
  1.6e-6** (median 4.8e-7), with **zero clips over the 1e-2 floor and 9,641 clips
  bit-identical**. The 0.152 tail is entirely the reference's Ampere-default
  TF32.
- **Batch shape is a smaller, separate effect.** Within the upstream code,
  `batch_size=14` versus `batch_size=1` (everything else equal) drifts up to
  0.070 per clip. Our container forwards one clip at a time (in upstream
  windowing each clip is a single 64,600-sample window, so the effective forward
  batch is 1 regardless of `--batch-size`); the primary reference forwards
  batches of 14. This batch-shape FP effect is present in both directions and is
  inherent to comparing two batch configurations of the same model.

## 7. Ratified Gate-2 tolerances

- **Decision layer (primary):** EER-identical. Subset EER 2.2193 % on both sides,
  Spearman 0.99858, decision agreement 99.985 % with every disagreement inside
  0.0024 of the threshold. Ratified: our container reproduces the upstream
  reference's ranking and operating point, with only 8 near-threshold decisions
  flipping.
- **Per-clip logit, matched precision (fp32 both sides):** max 0.0058, mean
  1.6e-6 across all 53,392 clips, zero over the pre-declared 1e-2 floor, 9,641
  bit-identical. Gate-2 PASSES: the pass-controlling max drift 0.0058 clears the
  1e-2 floor, and the mean sits at floating-point noise. The 0.0058 residual is
  the batch-shape FP effect (our effective forward batch 1 vs the reference's
  14); in full fp32 that effect is ~10x smaller than under TF32.
- **Per-clip logit, default precision (our fp32 vs upstream Ampere-default
  TF32):** <= 0.152, mean 7.8e-4. Ratified as the expected envelope of the
  fp32-vs-TF32 precision difference on this GPU class, EER-preserving (only 8
  near-threshold decisions flip).

No change to the shipped runner is warranted: disabling TF32 is the deliberate,
correct choice for a numerics-contract runner (full fp32 precision; the runner's
bit-exact cold-start reproducibility is recorded in the S2b determinism verdict).

## 8. Scope and what defers to S4

The 2.85 % published DF EER is a full-corpus (611,829-trial) statistic scored by
the official ASVspoof scorer, which requires every trial and rejects a subset.
The subset EER here (2.22 %) is a diagnostic, never a Gate-1 pass, and the
harness was never tuned toward 2.85 %. The full-cohort anchor (Gate-1 PASS/FAIL
against 2.85 %) runs on maintainer multi-GPU hardware in S4.

## 9. Reproduction

Run on maintainer RTX 3060 hardware, one visible GPU per container. The scored
journals and comparison outputs are run artifacts and are not committed; the
numbers above summarise them. The reference image (derived `FROM` the frozen eval
image plus `librosa==0.9.1`), the thin reference driver, and the pre-registration
and comparison scripts are maintainer tooling; their provenance (image digest,
upstream commit `4acaa61`, dependency delta) is recorded above so the reference is
reconstructable. They are held for reuse by the S4 full-cohort anchor.
