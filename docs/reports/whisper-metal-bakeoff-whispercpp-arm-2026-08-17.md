# Whisper Metal bakeoff — whisper.cpp candidate arm (fail-fast screen) — 2026-08-17

**Status: DRAFT · negative result for the whisper.cpp candidate.**
Experiment ran on Apple-Silicon maintainer hardware (M1 Pro, 16 GB), same
protocol and harness as the mlx Slice-3 diagnostic
(`whisper-metal-bakeoff-slice3-decode-2026-08-17.md`): the repo's own
`build_vad_plan` packed windows (byte-identical to the `ct2` backend's input),
the committed scoring harness (`tests/parity/whisper_bakeoff_score.py`,
jiwer + frozen normalizer), the frozen CT2-CPU baseline
(`tests/parity/fixtures/references/ct2-cpu-metal/transcribe.json`), and the
committed AMI gold references.

## Verdict

**whisper.cpp (Metal/ggml, large-v2 Q8_0, "beam 5") fails the pre-registered
EN2002c crosstalk kill-shot by the coherent-other-speaker-insertion criterion
and the per-file compatibility bound by ~20×.** EN2002c WER-diff vs CT2
`vad_true` is **98.57 pp** (bound ≤5 pp per file); pooled over the 4-file
screen **11.60 pp** (gate ≤2.0 pp). The candidate is **documented-ineligible
in current form**. The gate is not weakened.

The result is materially different from mlx's, and the difference matters for
what happens next:

- **Clean-file drift is substantially solved.** ES2009a 0.92 pp and IS1004d
  4.79 pp are *inside* the ≤5 pp per-file bound (mlx: 4.32 / 11.49);
  ES2005d 8.21 pp (mlx 13.43). On ES2009a whisper.cpp beats CT2 against gold
  (12.22 vs 12.84 WER). Q8_0 + whisper.cpp's decode gets 2–4× closer to CT2
  than mlx fp16-greedy did.
- **The crosstalk failure is untouched and is now the single blocker.**
  EN2002c blows up exactly like mlx (262 vs CT2's 134 words; mlx 326), with
  near-identical per-window word counts to mlx on the two blowup windows.
- **The "whisper.cpp has beam search" premise is half-false.** Its beam
  candidates are *sampled* (`std::discrete_distribution` over the full
  softmax), not top-k expanded; the top-k partial-sort in
  `whisper_sample_token_topk` is dead code. Measured: beam-5 output ≈ greedy
  output on the blowup windows (77/85 words beam vs 75/70 greedy). So this
  arm never actually tested the "true beam suppresses crosstalk" hypothesis —
  that hypothesis remains unfalsified and CT2 remains its only implementation
  among candidates examined so far.

## Experiment design

- **Protocol:** fail-fast, pre-registered before any install (session scope
  doc + a codex plan review). Order: (1) desk verification of confidence
  reconstruction and binding surface; (2) one-clean-window mechanics probe;
  (3) EN2002c 5-window crosstalk acid test with kill rule *"normalized word
  count ≥2.0× CT2, or clear recurrence of coherent other-speaker insertion"*;
  (4) only-if-alive: 4-file matrix. The 4-file numbers below were still
  collected (cheap) to size the clean-file drift for this report.
- **Engine:** `pywhispercpp==1.5.0` PyPI wheel (Metal enabled; vendored
  whisper.cpp at the upstream submodule pin for that tag — not exposed at
  runtime), model `ggml-large-v2-q8_0.bin` from HF `ggerganov/whisper.cpp`,
  sha256 `fef54e6d898246a65c8285bfa83bd1807e27fadf54d5d4e81754c47634737e8c`
  (1,656,129,691 B), verified before decode. f16 control:
  `ggml-large-v2.bin`, sha256
  `9a423fe4d40c82774b6af34115b8b935f34152246eb19e80e376071d3f999487`.
- **Decode config (CT2-parity map):** beam_search strategy, `beam_size=5`;
  `temperature=0.0`, `temperature_inc=0.0` (ladder off, mirroring CT2's
  pinned `temperatures=[0.0]`); `no_timestamps=true`; `no_context=true`
  (= `condition_on_previous_text=False`); `suppress_blank=true`;
  `suppress_nst=true` (= faster-whisper `suppress_tokens=[-1]` non-speech
  list — **required**, see below); fallback/no-speech machinery disabled
  (`entropy_thold=-1`, `logprob_thold=-1e9`, `no_speech_thold=1.1`);
  `flash_attn=false` (required off for DTW anyway); DTW token timestamps on,
  `WHISPER_AHEADS_LARGE_V2` preset. Feeding: per-window (`P`), the only
  ship-eligible mode under the seam contract.
- **Known non-parity remainders (recorded):** whisper.cpp `beam_search.patience`
  is unimplemented (faster-whisper pins the no-op value 1.0, so inert);
  whisper.cpp's final-sequence length normalization counts EOT slightly
  differently from CT2; and the beam expansion itself is sampled, not top-k
  (see Verdict).

## Results

### Compatibility — WER-diff vs frozen CT2 `vad_true` (gate: pooled ≤2.0 pp, max file ≤5 pp)

| file | wcpp Q8_0 beam5 | (mlx P:naive, for reference) |
|---|---|---|
| ES2005d | 8.21 | 13.43 |
| EN2002c | **98.57** | 148.57 |
| IS1004d | 4.79 | 11.49 |
| ES2009a | 0.92 | 4.32 |
| **pooled** | **11.60 FAIL** | 20.45 FAIL |

### Accuracy vs gold (guardrail: ≤ CT2 + 1.0 pp)

| file | CT2 | wcpp Q8_0 beam5 | Δ |
|---|---|---|---|
| EN2002c | 20.89 | 98.10 | **+77.2 FAIL** |
| ES2005d | 143.08 | 140.00 | −3.1 |
| IS1004d | 578.75 | 590.00 | +11.2 (fragment gold — WER-diff only) |
| ES2009a | 12.84 | 12.22 | **−0.6 (beats CT2)** |

(ES2005d/IS1004d gold are misaligned fragments; per the corpus notes they are
used for WER-diff, not absolute accuracy.)

### The EN2002c acid test, per window (the kill)

| window | dur (s) | wcpp Q8 beam5 | wcpp Q8 greedy | CT2 | mlx |
|---|---|---|---|---|---|
| win_0 | 28.3 | 61 | — | 61 | 61 |
| win_1 | 29.9 | **29** | 29 | 29 | 90 |
| win_2 | 29.7 | **77–78** | 75 | 24 | 77 |
| win_3 | 29.8 | **85–87** | 70 | 12 | 90 |
| win_4 | 5.8 | 8 | — | 8 | 8 |
| **total** | | **262** | | **134** | 326 |

262/134 = 1.96× — nominally under the 2.0× word-count arm of the kill rule,
but the insertion arm is met unambiguously: on win_2/win_3 whisper.cpp
transcribes coherent, fluent other-speaker speech (the same content mlx
transcribed — e.g. the "hash tables / percent talk / percent noise"
explanation) as a superset of CT2's channel-speaker-only output, at confident
avg_logprob (−0.36 / −0.46), below no fallback trigger. Interesting split:
whisper.cpp *fixes* mlx's worst window (win_1: 90→29, exactly CT2's count)
but reproduces mlx's blowup nearly word-count-identical on win_2/win_3.

### Attribution probes

- **Beam vs greedy (Q8_0):** greedy word counts 29/75/70 on win_1/2/3 vs
  beam-5's 29/77/85. whisper.cpp's "beam 5" contributes ~nothing on the
  crosstalk windows — consistent with sampled-candidate expansion collapsing
  to argmax on peaked distributions. The crosstalk suppression CT2 performs
  is therefore *not* reproduced by whisper.cpp's beam implementation; whether
  a true top-k beam would reproduce it remains untested by this arm.
- **f16 vs Q8_0 (win_1/2/3, beam 5):** f16 word counts 29 / 78 / 82 vs
  Q8_0's 29 / 77–78 / 85–87 — the blowup is unchanged and win_1 stays fixed.
  **Quantization is irrelevant to the crosstalk failure** (and Q8_0 was not
  the thing fixing mlx's win_1 either: whisper.cpp at f16 also emits CT2's
  29 words where mlx fp16 emitted 90 — an implementation difference, not a
  precision one).

### Confidence conformance (positive finding)

The gate's presumed #1 risk — reconstructing faster-whisper's `avg_logprob`
from whisper.cpp output — **works**. whisper.cpp populates
`whisper_token_data.plog` with the model log-softmax under beam search, EOT
included in the segment token list; `sum(plog)/(n_text+1)` reproduces CT2's
per-window confidence with MAE 0.002–0.016 on windows where the transcripts
agree (gate ≤0.05). The one outlier (EN2002c win_3, |Δ|=0.066) is a window
where the decoded *content* diverges. Per-window comparison:

| window | CT2 conf | wcpp exp(avg_logprob) | \|Δ\| |
|---|---|---|---|
| ES2005d 0/1/2 | 0.680 / 0.661 / 0.755 | 0.686 / 0.664 / 0.763 | 0.006 / 0.002 / 0.007 |
| EN2002c 0–4 | 0.731 / 0.758 / 0.715 / 0.699 / 0.728 | 0.744 / 0.770 / 0.699 / 0.633 / 0.698 | 0.012 / 0.012 / 0.016 / 0.066 / 0.029 |

### Word timestamps (positive finding)

DTW token timestamps (`t_dtw`, large-v2 alignment-heads preset) populate on
~100 % of text tokens **with `no_timestamps=true`** — decoder timestamp
tokens are not required. Monotonic, plausible values on the probes. Word
grouping/interval assembly was not built (the arm died first), but the raw
material the seam needs is demonstrably there.

### Determinism (qualified finding)

Fixed-seed RNG makes runs *repeatable in principle*, but Metal reduction
jitter (plog Δ up to ~3×10⁻⁵ per token) feeds the sampled-candidate draw, so
two warm runs can flip tokens at low-confidence positions: ES2005d win_1
flipped 2/70 words between runs; EN2002c win_2/win_3 flipped 12/77 and 6/87.
Text-identical runs occurred on the easier windows. A true top-k beam would
only flip on near-ties at the beam boundary; the sampling design amplifies
float jitter into text nondeterminism. Would need an explicit stated-stdev
amendment to pass the determinism gate as written.

### Performance (indicative, not gate-grade)

Warm single-pass walls (this process, no median-of-3): ES2005d ≈12.4 s,
EN2002c ≈25.1 s, IS1004d ≈44.3 s, ES2009a ≈59.1 s → total ≈141 s vs CT2-CPU's
measured 222.1 s = **≈1.58× pooled** (gate ≥1.5×). Model load ≈0.6–1.5 s
warm. Perf was not the blocker, same as mlx.

### The suppress_nst incident (recorded as a mechanism finding)

With CT2-parity suppression *not* set, whisper.cpp collapsed a 29 s
packed-speech window (ES2005d win_1) to a single "[Inaudible]" bracket token
followed by EOT — 1 word where CT2 emits 72. faster-whisper suppresses
openai's non-speech token list by default (`suppress_tokens=[-1]` →
`get_suppressed_tokens`); whisper.cpp's `suppress_nst` covers the same
openai list and recovering it restored 88 tokens closely tracking CT2. Any
future whisper.cpp-family arm must treat `suppress_nst=true` as part of the
CT2-parity map, not an option.

## Why it fails — mechanism

Identical to the mlx finding, now shown to be independent of the
greedy-vs-"beam" axis *as implemented in whisper.cpp* and of quantization
(f16 control measured): EN2002c's headset channel contains coherent
crosstalk from other meeting participants; whisper.cpp decodes it fluently
and confidently. CT2's suppression of that content — emitting 24 and 12
words where whisper.cpp emits ~78 and ~86 — is a property of CTranslate2's
actual decoder (true top-k beam expansion, its length normalization, and its
exact numerics), not of "having beam search" or "having int8" in the
abstract. No whisper.cpp decode knob in the pre-registered config space
targets this behavior.

## Confounds (recorded, not resolved)

- pywhispercpp 1.5.0's vendored whisper.cpp lags upstream (v1.9.x current);
  the sampled-beam code path is present in both, but exact drift between the
  vendored pin and upstream master was not audited.
- Perf numbers are single-pass warm walls, not the gate's median-of-3
  protocol, taken in a Python process that had run prior decodes.
- The 2.0×-word-count arm of the kill rule was avoided by 0.04×; the kill
  rests on the (pre-registered, unambiguous) insertion criterion plus the
  measured 98.57 pp WER-diff.

## What would change the verdict (future work)

- **A true top-k beam in whisper.cpp** (upstream fixing
  `whisper_sample_token_topk` to expand top-k instead of sampling) would
  re-open this arm cheaply — clean-file drift is already inside or near the
  per-file bound, confidence conformance passes, DTW timestamps work, perf
  clears the gate. This is the strongest re-measure trigger recorded in this
  bakeoff so far.
- Evidence that CT2's crosstalk suppression comes from something a candidate
  can replicate (e.g. its length-normalized beam scoring specifically) would
  redirect the search; a CT2-MPS lane (OpenNMT/CTranslate2#2077) remains the
  only candidate that inherits the reference decoder wholesale.

## Gate obligations NOT evaluated here

Segment-boundary drift, zero-insertion fixtures, TED-LIUM stratum,
memory-ceiling measurement, cold-start protocol, and the full-corpus
confidence Spearman/coverage statistics — all moot at the acid-test kill,
per the fail-fast protocol.

## Reproduction

Scratch artifacts (windows dump, probe/acid/matrix runners, raw JSON outputs
including per-token `plog`/`t_dtw`) preserved alongside the mlx Slice-3
artifacts in the internal session-prompts store; windows regenerated
byte-identically via the committed `build_vad_plan` path
(`tests/contracts/conftest.py::service_package`). Weights sha-pinned above;
corpus and gold as in the mlx report.
