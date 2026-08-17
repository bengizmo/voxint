# Whisper Metal bakeoff — Slice 3 decode-quality diagnostic (mlx) — 2026-08-17

**Status: FINAL · negative result for the current mlx candidate** (verdict
block landed in `docs/gpu-contracts.md`; issue #33 closed 2026-08-17).
Experiment ran on Apple-Silicon maintainer hardware (M1 Pro, 16 GB). All
numbers below are measured, produced by the committed scoring harness
(`tests/parity/whisper_bakeoff_score.py`, jiwer 4.0.0 + frozen normalizer)
against the frozen CT2-CPU baseline
(`tests/parity/fixtures/references/ct2-cpu-metal/transcribe.json`) and the
committed AMI gold references.

## Verdict

**`mlx-whisper==0.4.3` + `mlx-community/whisper-large-v2-mlx` (fp16, greedy)
fails the pre-registered compatibility gate by an order of magnitude, under
every decode configuration tested.** Pooled WER-diff vs CT2 `vad_true` is
19–21 pp against a ≤2.0 pp gate; worst-file disagreement is 115–149 pp against
a ≤5 pp bound. Per the pre-registered decision rule this is a negative result:
the candidate is **documented-ineligible** in its current form. The gate is not
weakened.

The blocker is **not** a fixable decode-config problem. The two hypotheses
carried into this session — (1) the missing temperature-fallback ladder caused
the EN2002c hallucination blowup, (2) per-window vs concatenated feeding drives
the failure — were both tested and **both falsified** as sufficient fixes.

## Experiment design

- **Corpus (diagnostic screen, not the full gate):** four 240 s AMI IHM files
  spanning 3–10 VAD windows: `ES2005d` (3), `EN2002c` (5), `IS1004d` (8),
  `ES2009a` (10). Windows produced by the repo's own `build_vad_plan` via
  `service_package("whisper")`, i.e. byte-identical to what the `ct2` backend
  receives.
- **Matrix:** feeding {`P` per-window, `C` concatenated single-call} × decode
  {`naive` = `temperature=0.0` scalar + `condition_on_previous_text=False`
  (the prior session's config), `ladder` = mlx-default temperature fallback
  `(0.0…1.0)` + `compression_ratio_threshold=2.4`, `logprob_threshold=-1.0`,
  `no_speech_threshold=0.6`, `cond=False`}.
- **Scoring:** direct candidate↔CT2 edit disagreement (never gold-WER
  subtraction); accuracy guardrail vs gold; per-segment diagnostics
  (`compression_ratio`, `avg_logprob`, `no_speech_prob`, kept `temperature`);
  warm wall time, median of 3.
- **Ship-eligibility note:** only `P` is ship-eligible under the seam contract
  (`docs/gpu-contracts.md` — engines decode *identical pre-cut windows*). `C`
  re-chunks audio and was run as an explanatory diagnostic only.
- Weights sha-verified before decode (`weights.npz`, 3,083,149,424 B, sha256
  `c9888a2d03b4e9906c2864151f56dc21e58d617938da0eb818152e3f99adc3f1`).

## Results

### Compatibility — WER-diff vs frozen CT2 `vad_true` (gate: pooled ≤2.0 pp, max file ≤5 pp)

| file | P:naive | P:ladder | C:naive | C:ladder |
|---|---|---|---|---|
| ES2005d | 13.43 | 17.91 | 21.64 | 21.64 |
| EN2002c | **148.57** | **148.57** | 115.00 | 115.00 |
| IS1004d | 11.49 | 11.49 | 12.07 | 12.07 |
| ES2009a | 4.32 | 4.32 | 6.02 | 6.02 |
| **pooled** | **20.45 FAIL** | **20.83 FAIL** | **19.17 FAIL** | **19.17 FAIL** |

### Accuracy vs gold (guardrail: ≤ CT2 + 1.0 pp)

Only `EN2002c` (gold N=158, ratio 0.89 vs CT2) and `ES2009a` (N=810, 0.94) have
trustworthy gold; `ES2005d` (N=65) and `IS1004d` (N=80 vs CT2's 522 words) are
known gold fragments and excluded from the accuracy read.

| file | CT2 | best mlx | delta |
|---|---|---|---|
| EN2002c | 20.89 | 104.43 (C) | **+83.5 FAIL** |
| ES2009a | 12.84 | 13.70 (C) | +0.9 (borderline pass) |

### Normalized word counts

| file | gold | CT2 | P:naive | C:naive |
|---|---|---|---|---|
| ES2005d | 65* | 134 | 134 | 133 |
| EN2002c | 158 | 140 | **339** | **292** |
| IS1004d | 80* | 522 | 542 | 543 |
| ES2009a | 810 | 764 | 771 | 754 |

\* fragment gold. On three of four files mlx word counts sit within ±4 % of
CT2 — the disagreement there is word-level substitution drift, not volume.
`EN2002c` is a distinct failure mode (below).

### Performance (warm medians of 3; speedup = CT2_wall / mlx_wall, gate ≥1.5×)

CT2-CPU walls measured after the mlx matrix on an idle machine (no
contention); mlx walls from the matrix run.

| file | CT2 wall | P:naive wall | speedup | C:naive wall | speedup |
|---|---|---|---|---|---|
| ES2005d | 23.7 s | 10.6 s | 2.23× | 8.7 s | 2.74× |
| EN2002c | 30.5 s | 24.5 s | 1.24× | 16.8 s | 1.82× |
| IS1004d | 71.9 s | 40.2 s | 1.79× | 29.8 s | 2.41× |
| ES2009a | 96.0 s | 54.1 s | 1.77× | 44.2 s | 2.17× |
| **pooled** | 222.1 s | 129.4 s | **1.72× PASS** | 99.4 s | 2.23× |

**Performance is not the blocker.** The ship-eligible P:naive path clears the
pooled ≥1.5× gate (1.72×), missing per-file only on EN2002c (1.24× — and
partly *because* it decodes 199 extra crosstalk words there). The prior
2-file impression that perf "straddles the gate" does not survive the larger
sample: window-dense files (IS1004d, ES2009a) slow CT2 down more than they
slow mlx. Quality is the sole blocker. (`P:ladder` on EN2002c additionally
showed 33.6–1078 s instability — see mechanism 3 below.)

## Why it fails — three measured mechanisms

### 1. EN2002c is confident crosstalk transcription, not hallucination

Per-window diagnostics on the naive decode show **no window crosses any
fallback threshold**: max `compression_ratio` 1.82 (thr 2.4), min
`avg_logprob` −0.89 (thr −1.0), max `no_speech_prob` 0.571 (thr 0.6). The
"extra" 199 words are **coherent multi-speaker meeting speech** — the AMI
headset channel picks up other participants, and mlx (greedy, fp16)
transcribes that crosstalk while CT2 (beam 5, int8) suppresses it. Example
(window 1): *"Data processing is fine, but I don't particularly want to do the
GUI for it. — You don't? Okay, I'll do it."* — clearly two speakers.

Consequence: **no threshold or temperature knob can remove this output**,
because it is real speech decoded confidently below every trigger. The
temperature ladder is arithmetically a no-op on this file (confirmed: ladder
transcript byte-identical to naive — 326 raw / 339 normalized words).

### 2. Where the ladder does fire, mlx 0.4.3 keeps the wrong attempt

On `ES2005d` window 1 the t=0 decode lands at `avg_logprob = −1.021` — just
under the −1.0 threshold — so the ladder escalates to t=1.0 and then **keeps
the final sampled attempt: 30 segments with compression_ratio 9.11**
(wildly repetitive), replacing a near-threshold but sane greedy decode. This
is upstream `mlx-examples` issue #1427 (fallback returns *last* attempt, not
*best*) observed live: the ladder made the file **worse** (13.43 → 17.91 pp)
and 2.1× slower.

### 3. The ladder is also a stability risk in long-running processes

In the long-running matrix process, `P:ladder` on EN2002c produced warm walls
of **33.6 s / 957.5 s / 1078.0 s on identical input** (30×+ blowup), yet
returned all-temp-0 segments and a transcript identical to naive. In an
isolated process the same config runs at naive speed (per-window 4.4–6.7 s,
two runs) with zero fallback firing. The blowup pattern is consistent with
process-state accumulation (Metal buffer/memory pressure on a 16 GB host)
around the sampling path. Regardless of root cause, a decode path whose wall
time varies 30× on identical input cannot meet the determinism/perf gates.

### Residual drift on well-behaved files

Even excluding EN2002c, disagreement runs 4.3–13.4 pp (P:naive) — far above
the 2.0 pp pooled gate — with word counts near-identical to CT2. This is
systematic decode divergence between **greedy fp16** (mlx has no beam search)
and **beam-5 int8** (CT2), i.e. a search-strategy + quantization confound
inherent to the current mlx stack, not a configuration error.

## Supporting observations

- **Zero-insertion (ship path):** the voxint VAD emits **zero** windows on the
  committed silence fixtures and 4 of 5 hallucination-bait fixtures
  (`bait_02_tone` yields one 2.2 s window). On the ship path, non-speech
  robustness is owned by VAD; the engine never sees that audio. On the one
  bait window that passes VAD, mlx emits exactly what the CT2 baseline emits
  (`'you'`) — no growth, so this sub-gate passes.
- **C-arm (diagnostic only):** concatenated feeding trims EN2002c
  over-transcription (339 → 292 words) and is 1.2–1.5× faster than per-window
  (no 30 s padding per window), but still fails every gate and is not
  ship-eligible under the identical-pre-cut-windows seam contract. Its perf
  benefit is real and would justify a **future contract amendment discussion**
  only if a decode stack existed that passed quality.
- The `naive` and `ladder` C-arm transcripts are byte-identical on all four
  files (no fallback ever fires on the concatenated stream).

## Confounds (recorded, not resolved)

1. **fp16 (mlx) vs int8 (CT2)** quantization.
2. **Greedy (mlx) vs beam-5 (CT2)** search: mlx-whisper has no beam search
   implementation as of 0.4.3.
3. AMI IHM headset crosstalk makes "correct" output genuinely ambiguous at the
   channel level; CT2's suppression behavior is itself emergent, not designed.

These three move together in every measured delta; none is isolated by this
experiment. What *is* established: no decode-config change available in
mlx-whisper 0.4.3 closes the gap.

## What would change the verdict (future work)

- **Beam search lands upstream in mlx-whisper** — removes the largest
  suspected driver of residual drift; re-run this diagnostic as-is.
- **Upstream #1427 fixed** (best-of-attempts fallback) — makes the ladder
  safe, though it would not fix EN2002c-class crosstalk.
- A **different candidate stack** on Apple Silicon (e.g. whisper.cpp Metal,
  which supports beam search and int8-family quantization closer to the CT2
  reference) — a separate bakeoff arm with its own diagnostic. *(Since
  measured, same day: also ineligible — see
  `whisper-metal-bakeoff-whispercpp-arm-2026-08-17.md`.)*
- Revisiting the **windowing/feeding contract** (C-arm-style decode) — only
  worthwhile after a quality-passing decode stack exists.

## Gate obligations NOT evaluated here

This was a 4-file diagnostic screen. The full pre-registered gate
(`docs/gpu-contracts.md`) additionally requires: token agreement ≥97 %,
per-file p95 over the full corpus, segment-boundary drift, zero-insertion on
the full synthetic set, confidence conformance (ρ ≥0.90, MAE ≤0.05), memory
ceiling, cold-start, and TED + synthetic strata. Moot for this candidate given
the compatibility failure, but binding on any future candidate.

## Reproduction

Experiment scripts and raw outputs (matrix runner, per-window telemetry
pre-flight, CT2 timing, scorer) are preserved alongside the session artifacts;
inputs are the committed bakeoff corpus (`tests/parity/fixtures/bakeoff/`),
the frozen CT2 baseline, and the sha-pinned mlx weights snapshot
(`mlx-community/whisper-large-v2-mlx` @ `cce86229…`). Environment:
`mlx-whisper==0.4.3`, `mlx 0.32.0`, Python 3.12, Apple-Silicon (arm64).
