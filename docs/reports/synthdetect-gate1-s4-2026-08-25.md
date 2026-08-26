> **Status:** Gate-1 PASS 2026-08-25. The full official ASVspoof 2021 DF eval
> cohort (611,829 trials) was scored end to end with the verbatim upstream
> SSL_Anti-spoofing model and data modules (under a thin, audited reference
> driver) on the native FLAC tree, then pooled with the official DF scorer math.
> Measured pooled EER over the 533,928 phase-`eval` trials is **2.8650 %**, against
> the published **2.85 %** and the pre-registered **±0.3 pp** tolerance (a
> +0.015 pp miss, well inside). This is the corpus-level reproduction that Gate-2
> deliberately deferred. It proves the upstream stack reproduces the published
> number; Gate-2's ratified subset equivalence is then strong evidence that the
> shipped fp32 runtime reproduces it too, but the full-cohort fp32 EER itself is
> not measured here (a deferred pass, see section 7).

# synthdetect Gate-1 (S4): full-cohort DF benchmark reproduction

Gate-1 is the benchmark-reproduction gate for the synthetic-speech detector
(#144): run the unmodified upstream eval stack on the full official corpus and hit
the published number, proving the weights, data, and protocol are right. This
report records the cohort, the runner, the scorer, and the measured pooled EER.
It is the S4 anchor that the S3 pre-registration and the Gate-2 verdict deferred
to the full official cohort.

## 1. Verdict

The unmodified upstream runner reproduces the published ASVspoof 2021 DF number
on the full official eval cohort:

| Metric | Result |
|---|---|
| Trials scored (official length check) | 611,829 / 611,829, zero skips |
| Pooled scope (phase `eval`) | 533,928 trials: 14,869 bona-fide, 519,059 spoof |
| Pooled DF-eval EER (official `compute_eer` math) | **2.8650 %** |
| Independent EER cross-check | 2.8650 % (identical to four decimals) |
| EER threshold (upstream raw polarity, higher is bona-fide) | -3.533004 |
| Published target | 2.85 % EER |
| Pre-registered tolerance | provisional ±0.3 pp |
| Result | **PASS** (+0.015 pp from target, inside ±0.3 pp) |

The +0.015 pp miss is one twentieth of the allowed ±0.3 pp tolerance (0.015 / 0.3
= 0.05). The tolerance is labelled provisional in the pre-registration, but that
does not affect this verdict: it is a reproduction-closeness bound on matching a
fixed published number, not a rerun-variance envelope, and the reference is
deterministic (Gate-2 measured three cold `batch_size=14` reference runs
bit-identical, so the EER has zero rerun variance). A 2.8650 % result clears any
defensible closeness bound with wide margin.

This is a benchmark-reproduction claim for the DF-tuned `w2v2-aasist-df`
checkpoint on the full official cohort, run as the upstream stack publishes
(Ampere-default TF32, the upstream `batch_size` default). It is not a determinism
claim (that is the S2b/DF-qualification verdicts) and not a paired-equivalence
claim (that is Gate-2).

## 2. What Gate-1 tests, and why it is the full cohort

Gate-1 answers one question: does the pinned checkpoint, run through the
unmodified upstream eval code on the official data, produce the published EER?
The published 2.85 % is a statistic over the entire official DF trial list, and
the official DF scorer asserts full-metadata coverage, so a subset cannot
reproduce it. The S3 pre-registration therefore delivered Gate-1 readiness and
ratified Gate-2 on the seeded subset, and explicitly deferred the Gate-1 PASS to
the full cohort here.

Gate-1 uses upstream's own data loading (native FLAC decode plus the
64,600-sample crop rule) on an untouched native tree, per the "two corpus views,
never conflated" pre-registration. It does not use the canonical PCM view; that
view belongs to Gate-2's paired comparison.

## 3. The cohort

The official CM keys archive (`DF-keys-full.tar.gz`, sha256
`426f93e1…`, the value pinned in `tools/synthdetect_df_import.py`) was verified by
sha and extracted. Its `trial_metadata.txt` holds 611,829 rows of 13 space-
delimited columns. The columns Gate-1 reads, confirmed by measuring the real file
rather than assuming the tokens:

- column 2: the trial id (`DF_E_...`).
- column 6: the label, tokens `bonafide` (22,617 across all phases) and `spoof`
  (589,212).
- column 8: the phase, tokens `eval` (533,928), `progress` (59,325), and `hidden`
  (18,576).

The full protocol is column 2 of every row (611,829 ids, all unique). The native
FLAC tree materialised by the emit acceptance run
(`docs/reports/synthdetect-df-emit-acceptance-2026-08-25.md`) holds exactly
611,829 files, and the protocol ids and the on-disk ids form an exact bijection:
zero protocol ids missing a FLAC, zero FLAC files absent from the protocol.

Pooled EER is computed over the phase-`eval` trials only (533,928), matching the
official scorer's inner merge on `phase == 'eval'`. All 611,829 trials are scored
so the scorer's length check passes; the phase filter then selects the eval
partition.

## 4. The runner and the weights

Scoring used the same independent upstream reference driver as Gate-2
(`ref_score.py`), which imports the verbatim upstream `model.py` and
`data_utils_SSL.py` (`pad` plus `Dataset_ASVspoof2021_eval`) at commit
`4acaa61dcef5f7610f43aa4d0b29c4559b970cd2`, wraps the model in
`nn.DataParallel`, loads the raw `module.`-prefixed checkpoint strict, and emits
the raw column-1 bona-fide logit (higher is bona-fide). The driver is not
literally upstream `main_SSL_DF.py` (that module drags an unpinned dependency and
a training and logging surface irrelevant to eval); it owns only the DataLoader
wiring and the score emission, and every numerically load-bearing element (reader,
windowing, model, checkpoint load, score) is the verbatim upstream code. It runs
in the Gate-2 reference image (id
`sha256:03891ac1b090…`, built `FROM` the frozen S2b eval image with only
`librosa==0.9.1` added; torch 2.1.0+cu118, torchaudio, fairseq, numpy, and scipy
held identical to S2b).

Gate-1 runs the upstream stack as published: TF32 left at the Ampere default (the
`--no-tf32` control is a Gate-2-only diagnostic) and the upstream `batch_size`
default of 14. Weights are the DF anchor, mounted read-only:

| File | sha256 |
|---|---|
| `Best_LA_model_for_DF.pth` | `1cf904f1d84c867c278cd42161df5367939d61cc28bfefd239bc995af59c2804` |
| `xlsr2_300m.pt` | `b08927597f2c9eb2ebd7dcc3ac78ee4b5f6021cbac4b3a6c5a9deec445d80ed9` |

The checkpoint sha matches the QUALIFIED `w2v2-aasist-df` pin in the registry and
the S3 pre-registration. The vendored upstream module shas match the baked
provenance (`model.py` `08b2b99b…`, `data_utils_SSL.py` `8a32c93b…`,
`eval_metrics_DF.py` `db993ad0…`).

## 5. Execution

The 611,829-trial protocol was split into two disjoint shards (305,915 and
305,914 ids, no overlap, union equal to the full protocol) and scored
concurrently on two RTX 3060 GPUs on maintainer hardware, one visible GPU per
container (the driver asserts exactly one, so `nn.DataParallel` cannot split a
batch across devices). Both shards ran at `batch_size=14`, the identical upstream
configuration, so per-clip scores do not depend on which shard a trial landed in.
Each container verified full coverage of its shard and preserved order before
writing its score file; the two files were then concatenated for scoring. Total
wall-clock was about 2 hours 37 minutes for the two shards running concurrently.

Score-file provenance (run artifacts, not committed):

| Artifact | sha256 |
|---|---|
| shard A scores (305,915) | `ec8ead13…` |
| shard B scores (305,914) | `643b5342…` |
| concatenated (A then B) | `88786ae5…` |

Score range across the full cohort was [-3.820, 4.814]; the EER threshold
-3.533004 sits inside it.

## 6. Scoring

The pooled EER used the official ASVspoof DF math directly: the vendored,
sha-tracked `eval_metrics_DF.compute_eer` (`db993ad0…`), the same implementation
Gate-2 used. The Gate-1 scorer reads the official keys and the concatenated score
file, enforces the length check (scored row count equals the 611,829 keys rows)
and full coverage (every keys id has exactly one score, no extras), filters to
phase `eval`, splits scores into bona-fide (target) and spoof (nontarget) by the
column-6 label, and calls `compute_eer(target, nontarget)`.

The merge and phase filter, the only non-vendored logic, were implemented two
independent ways in the scorer and asserted to produce identical bona-fide and
spoof arrays before scoring. An independently written EER routine (a distinct
threshold-sweep implementation, not the vendored one) was computed alongside the
official math as a cross-check; the two agree to four decimals (2.8650 % both).

## 7. What this closes, and what remains

Gate-1 proves the upstream stack reproduces the published DF benchmark on the
full official cohort: the published number is real for the pinned checkpoint and
data. Gate-2 separately ratified that our frozen eval container reproduces that
same upstream stack per clip on the 53,392-clip subset (subset EER identical, and
the same function at matched fp32 precision). Read together they are strong
evidence that the shipped fp32 runtime reproduces the benchmark, but that last
step is an inference, not a measurement: the shipped container's own full-cohort
EER on the canonical PCM view is not scored here.

That measurement is the one optional step deliberately not run: scoring the full
611,829-clip cohort with our own container on the canonical PCM view (a
full-cohort Gate-2 parity pass). It needs the full canonical transcode, which the
emit run performed only for the subset, and it is not required for the Gate-1
anchor. Gate-2's ratified subset equivalence and its EER-preservation result are
what make the full-cohort carryover a well-supported expectation rather than a
proven number. If that proven number is later wanted, it is a separate, longer
job.

## 8. Reproduction

The upstream reference driver, the Gate-1 scorer, and the vendored upstream eval
files live in the maintainer reference harness (not committed; provenance shas
above). To reproduce: verify the official keys archive by its pinned sha, extract
`trial_metadata.txt`, build the full protocol from column 2, materialise the
native FLAC tree with the committed importer's `emit` verb, score every trial with
the reference driver on the native tree at `batch_size=14` with TF32 at the Ampere
default, and pool with `eval_metrics_DF.compute_eer` over the phase-`eval`
bona-fide and spoof scores. The pinned checkpoint and the official cohort should
land within ±0.3 pp of 2.85 %.
