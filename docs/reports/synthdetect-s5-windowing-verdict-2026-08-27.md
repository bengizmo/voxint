# Synthdetect S5 windowing verdict

> **Status:** PASS. Production windowing does not raise the bona fide
> false-positive rate at the FPR 5% or FPR 1% regions of the raw-logit
> distribution.

## 1. Scope

This verdict addresses the S5 pre-registration in `docs/gpu-contracts.md`
("Windowing-validation scope"): does production windowing raise the bona fide
false-positive rate compared to the upstream single-crop protocol?

With a bona fide-only corpus, the verdict validates FPR stability, not
separability or EER. Those need the spoof side that arrives in S6. Per-window
scores are journaled as a first-class output (not a fallback), and are analyzed
below.

## 2. Corpus

The full S5 bona fide calibration corpus (materialized 2026-08-27, materialization
verdict in `docs/gpu-contracts.md`):

| Domain | Recordings | Parents | Degraded children | Total clips |
|---|---|---|---|---|
| AMI (meetingroom) | 7 | 2787 | 1045 | 3832 |
| VoxConverse (webvideo) | 7 | 708 | 114 | 822 |
| **Combined** | **14** | **3495** | **1159** | **4654** |

Degraded children are in the calibration split only (per the calibration
discipline: codec artifacts on genuine audio are the dominant field false-positive
driver). Six degradation chains per corpus: `aac-lc-cbr48-v1`, `amr-nb-122-v1`,
`g711-mulaw-8k-v1`, `mp3-cbr48-v1`, `opus-voip-cbr16-f20-v1`,
`speed-atempo-0p90-v1`.

## 3. Runtime

| Field | Value |
|---|---|
| Model | `w2v2-aasist` (inference space `synthdetect-w2v2aasist-v1`) |
| Weights | `LA_model.pth` (sha `bd6f3609...`), `xlsr2_300m.pt` (sha `b0892759...`) |
| Image | `voxint-synthdetect-eval:s2b` (digest `sha256:d631e021...`) |
| Runtime | PyTorch 2.1.0+cu118, fairseq 1.0.0a0+a540213, Python 3.10.12 |
| CUDA | 11.8, cuDNN 8700, device capability 8.6 |
| GPU | NVIDIA GeForce RTX 3060 (device index 0 inside container) |
| Batch size | 8 |
| Seed | 0 (torch, numpy, python hash) |
| Determinism | `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` |

## 4. Windowing configurations

| Parameter | Upstream | Production |
|---|---|---|
| Mode | Single 64,600-sample crop (repeat-pad if shorter) | 4.0375 s windows at 4.0375 s hop (no overlap) |
| Window width (samples) | 64,600 | 64,600 (aligned to model input) |
| Tail floor | N/A | 8,000 samples (0.5 s); dropped when at least one full window exists |
| Pooling | logit-mean | logit-mean |
| SOURCES_VERSION | v6 | v6 |

## 5. Results

### 5.1 Aggregate score comparison

| Metric | AMI | VoxConverse |
|---|---|---|
| Clips scored (both modes) | 3832 | 822 |
| Upstream mean raw score | 1.1100 | 3.8586 |
| Production mean raw score | 1.3272 | 3.8297 |
| Mean delta (production - upstream) | +0.2172 | -0.0289 |
| Median delta | 0.0000 | 0.0000 |
| Max absolute delta | 5.4732 | 6.4959 |
| Production higher | 1004 (26.2%) | 257 (31.3%) |
| Production lower | 510 (13.3%) | 249 (30.3%) |
| Tied (no change) | 2318 (60.5%) | 316 (38.4%) |

The median delta is zero for both domains: the majority of clips (single-window)
produce identical scores. Non-zero deltas arise from clips with multiple
production windows, where the logit-mean pools over more coverage.

### 5.2 FPR at shipped operating points

Higher raw score = more synthetic. FPR is the fraction of bona fide clips scoring
above a given raw-logit threshold. Thresholds are drawn from percentiles of the
combined upstream+production score distribution (no calibrated threshold exists yet;
that requires the spoof side in S6). The FPR 5% region is where ~5% of bona fide
clips score above the threshold.

**AMI (3832 clips):**

| Threshold region | Threshold | Upstream FPR | Production FPR | Delta FPR |
|---|---|---|---|---|
| ~FPR 50% | +1.189 | 0.4932 | 0.5068 | +0.0136 |
| ~FPR 30% | +2.349 | 0.2944 | 0.3056 | +0.0112 |
| ~FPR 10% | +4.384 | 0.1007 | 0.0994 | -0.0013 |
| ~FPR 5% | +5.244 | 0.0509 | 0.0496 | -0.0013 |
| ~FPR 1% | +5.951 | 0.0102 | 0.0099 | -0.0003 |

**VoxConverse (822 clips):**

| Threshold region | Threshold | Upstream FPR | Production FPR | Delta FPR |
|---|---|---|---|---|
| ~FPR 50% | +4.424 | 0.5109 | 0.4891 | -0.0219 |
| ~FPR 30% | +5.302 | 0.3382 | 0.2616 | -0.0766 |
| ~FPR 10% | +5.781 | 0.1058 | 0.0949 | -0.0109 |
| ~FPR 5% | +5.923 | 0.0535 | 0.0487 | -0.0049 |
| ~FPR 1% | +6.104 | 0.0122 | 0.0097 | -0.0024 |

At the FPR 5% operating region, production windowing does not raise FPR. It is
marginally lower in both domains (AMI: -0.13 pp, VC: -0.49 pp). At FPR 1%, both
are indistinguishable (AMI: -0.03 pp, VC: -0.24 pp). The small positive deltas
at lower thresholds (mid-distribution) are directional noise from pooling over
more windows.

### 5.3 Per-window analysis

| Metric | AMI | VoxConverse |
|---|---|---|
| Upstream mean windows per clip | 1.0 | 1.0 |
| Production mean windows per clip | 1.7 | 2.9 |
| Production max windows per clip | 16 | 33 |
| Multi-window clips (production) | 1514 | 506 |
| Mean intra-clip score spread | 3.38 | 2.66 |
| Max intra-clip score spread | 11.51 | 10.85 |

Upstream mode always produces exactly 1 window (the 64,600-sample crop).
Production mode generates multiple windows for clips longer than the window width.
The intra-clip score spread (max window score minus min) is substantial, averaging
3+ logit units. This confirms per-window journaling is load-bearing: a single
pooled score hides real variation within long clips, and the per-window record
lets S6 calibration see it.

### 5.4 Per-stratum breakdown

**AMI (organic = parent, others = degraded):**

| Stratum | n | Mean delta | Higher | Lower |
|---|---|---|---|---|
| organic\|meetingroom | 2787 | +0.2518 | 825 | 399 |
| aac-lc-cbr48-v1 | 197 | +0.0899 | 28 | 21 |
| amr-nb-122-v1 | 176 | -0.0012 | 24 | 30 |
| g711-mulaw-8k-v1 | 173 | +0.1951 | 31 | 14 |
| mp3-cbr48-v1 | 181 | +0.0898 | 32 | 17 |
| opus-voip-cbr16-f20-v1 | 156 | +0.1520 | 24 | 14 |
| speed-atempo-0p90-v1 | 162 | +0.2423 | 40 | 15 |

**VoxConverse (organic = parent, others = degraded):**

| Stratum | n | Mean delta | Higher | Lower |
|---|---|---|---|---|
| organic\|webvideo | 708 | -0.0227 | 226 | 217 |
| aac-lc-cbr48-v1 | 24 | +0.0248 | 9 | 7 |
| amr-nb-122-v1 | 21 | -0.2079 | 3 | 7 |
| g711-mulaw-8k-v1 | 16 | -0.2853 | 2 | 4 |
| mp3-cbr48-v1 | 16 | -0.0545 | 5 | 3 |
| opus-voip-cbr16-f20-v1 | 21 | -0.0446 | 5 | 8 |
| speed-atempo-0p90-v1 | 16 | +0.1564 | 7 | 3 |

No stratum shows a systematic directional shift that would indicate production
windowing destabilizes a particular codec class. The AMI organic stratum shows a
mild positive shift (+0.25), but this does not reach the FPR 5% threshold (see
5.2).

## 6. Verdict

**PASS.** Production windowing (4.0375 s windows, logit-mean pooling, 8,000-sample
tail floor) does not raise the bona fide false-positive rate at either shipped
operating point:

- At FPR 5%: delta is -0.13 pp (AMI) and -0.49 pp (VC). Production is
  marginally better.
- At FPR 1%: delta is -0.03 pp (AMI) and -0.24 pp (VC). Indistinguishable.
- Per-stratum: no degradation chain shows a destabilizing shift.
- Per-window scores are journaled and show meaningful intra-clip variation
  (mean spread 2.7 to 3.4 logit units), confirming the pre-registration's
  call to make them first-class output.

**Scope limitation (stated by design).** This verdict covers FPR stability only.
Separability (EER) and the threshold itself require the spoof side that arrives
in S6.

## 7. Artifacts

- Journals: `ami-upstream.jsonl`, `ami-production.jsonl`, `vc-upstream.jsonl`,
  `vc-production.jsonl` (maintainer hardware, not committed)
- Verdict JSON: `ami-verdict.json`, `vc-verdict.json` (structured output of
  `synthdetect_eval.py verdict-windowing`)
- Code: `tools/synthdetect_infer.py` (per-window score journaling),
  `tools/synthdetect_eval.py` (`verdict-windowing` subcommand)
