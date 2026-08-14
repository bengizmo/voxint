# GPU service contracts (v1)

The three GPU model services — **whisper** (ASR), **pyannote** (diarization), and
**titanet** (speaker embedding) — are self-contained FastAPI images under `services/`.
The pipeline talks to them only through the versioned HTTP contracts below; the
client protocols in `src/voxint/clients/base.py` are the consuming side of the
same contract.

## Shared conventions

- **Path-based requests.** Every request names an audio file by its path
  **relative to `MEDIA_ROOT`** (the shared media volume mounted into every
  service at the same location, read-only for the services). No multipart
  uploads, no absolute paths. Resolution follows symlinks, then requires the
  result to stay inside `MEDIA_ROOT`; a path that escapes (or is absolute,
  empty, or not a regular file) → `400`. A conforming path with no file → `404`.
  Path validation happens before the request waits on the inference lock.
- **Synchronous processing.** One request = one inference run, response returned
  when done. There are no job queues, background tasks, or result caches inside
  the services — retries, leases, and idempotency live in the pipeline's stage
  engine, not here. Set client timeouts accordingly (hours for long media).
- **Single-flight with bounded admission.** Each service serializes GPU
  inference internally; concurrent requests queue. Admission is bounded
  (`MAX_PENDING_REQUESTS`, default 8): a saturated service refuses new work
  with a retryable `503` instead of building an unbounded queue that outlives
  the caller's stage lease. Scale by running more replicas.
- **Versioning.** Four separate identities, never conflated:
  - *contract version* — the `/v1/` path prefix. Additive response fields are
    allowed within `v1`; renames/removals/semantic changes require `/v2`.
  - *service version* — the image's own semver (`version` in `/healthz`).
  - *model identity* — the model actually loaded (`model` in `/healthz`).
  - *embedding space* (titanet only) — versions the vector semantics; any
    model **or preprocessing** change requires a new space id even if
    `/v1/embed` stays wire-compatible.
- **Schema discipline.** Request models reject unknown fields (`422`). Response
  consumers must tolerate additive fields.
- **Units.** All times are float **seconds** (fields say so: `*_seconds`).
- **Input format.** Services expect **16 kHz mono WAV** — producing it is the
  pipeline normalize stage's invariant, not the services'. Non-conforming
  input that the decoder can still read is processed after resampling/downmix
  with a logged warning; undecodable input → `400 invalid_media`.
- **Errors.** Every error body is `{"detail": {"code": ..., "message": ...,
  "retryable": bool}}`. Stable codes: `path_violation` (400), `invalid_media`
  (400), `file_not_found` (404), `validation_error` (422, FastAPI-native body),
  `model_unavailable` (503, retryable), `saturated` (503, retryable),
  `inference_failed` (500, not retryable — the same input will fail again;
  the pipeline's failure lane owns it). Inference failures are request-fatal:
  no silent partial results.
- **Health.** `GET /healthz` on every service is a *readiness* probe:

  ```json
  {
    "status": "ok",
    "service": "whisper" | "pyannote" | "titanet",
    "version": "<service semver>",
    "contract_version": "v1",
    "model": "<model identifier actually loaded>",
    "device": "cuda" | "cpu" | "mps" | "rocm" | "metal",
    "engine": "<inference engine>",
    "engine_version": "<engine version>",
    "runtime": "<compute runtime>" | null,
    "runtime_version": "<runtime version>" | null,
    "model_loaded": true
  }
  ```

  `status` is `"ok"` only when the model is loaded and usable; otherwise `503`
  with the same shape, `status: "degraded"`, and `model: null`. `/healthz`
  never triggers model loading and never touches the GPU beyond a cheap
  availability check.

  - `device` is the compute device actually used for inference. **Honest
    reporting is required**: torch built for ROCm masquerades as CUDA
    (`torch.cuda.is_available()` is true and the device type is `"cuda"`), so
    services must report `"rocm"` whenever `torch.version.hip` is set. The
    CTranslate2 ROCm build masquerades the same way (`device="cuda"` selects
    the AMD GPU) and carries no torch signal — the torch-free whisper `-rocm`
    image detects the loaded HIP runtime library instead (`libamdhip64` in
    `/proc/self/maps`, checked after model construction) and reports
    `"rocm"`. `"mps"`
    is torch Metal Performance Shaders (host adapters); `"metal"` is
    non-torch Metal backends (e.g. whisper.cpp/ggml, or onnxruntime's CoreML
    EP — the titanet engine reports `"metal"` only when the CoreML EP is
    verified active in the session, never merely requested).
  - `engine` / `engine_version` identify the **inference engine only** — e.g.
    `faster-whisper`, `pyannote.audio`, `nemo`, `onnxruntime`, `whisper.cpp` —
    never the driver or userspace stack. `model` stays the weights identity:
    large-v2 is large-v2 regardless of engine.
  - `runtime` / `runtime_version` identify the compute userspace the engine
    runs on (e.g. `torch` / `2.5.0+cu118`, `torch` / `2.8.0+rocm7.2`,
    `ctranslate2` / `4.4.0`, `onnxruntime` / `1.20.1`), `null` when the
    engine has no separable runtime. Best-effort diagnostics — host-driver
    provenance is **not** readiness truth and is never required for `"ok"`.
  - The four fields are additive within `v1`; consumers must tolerate their
    absence (older services) and any future additive values.

## whisper — `POST /v1/transcribe` (port 8022)

Model: faster-whisper **large-v2**, int8 (policy pin — v3/turbo trade
quiet-audio robustness for speed and hallucinate; override via `WHISPER_MODEL`
at your own risk). The model is baked into the image at build time.

Request (unknown fields rejected):

```json
{
  "path": "items/1234/audio.16khz.wav",
  "language": "en",
  "initial_prompt": null,
  "vad_filter": true
}
```

- `language` (default `"en"`): target language code. `null` → auto-detect;
  the response reports the effective language either way.
- `initial_prompt` (optional, ≤2000 chars): vocabulary/context prompt.
- `vad_filter` (default `true`): Silero VAD via `BatchedInferencePipeline`.
  `false` bypasses the batched pipeline entirely and calls the raw
  `WhisperModel.transcribe` — for audio VAD misclassifies as silence.

Response:

```json
{
  "language": "en",
  "duration_seconds": 3712.4,
  "transcript": "full flattened text ...",
  "confidence": 0.93,
  "segments": [
    {
      "start_seconds": 0.0,
      "end_seconds": 7.4,
      "text": "…",
      "confidence": 0.95,
      "suspect": false,
      "suspect_score": null,
      "suspect_span": null
    }
  ],
  "words": [
    {"start_seconds": 0.0, "end_seconds": 0.4, "word": "Hello", "confidence": 0.99}
  ],
  "suspect_segment_count": 0
}
```

- Word timestamps are always requested from the model; `words` may still be
  empty when the model yields none (e.g. pure silence).
- `transcript` is the segment texts joined with single spaces, verbatim.
- `confidence` values are `exp(avg_logprob)` clamped to [0, 1]; segment
  `confidence` is `null` when the model provides no logprob; the top-level
  `confidence` is the mean over segments that have one (0.8 if none do).
  Confidence is **engine-calibrated**: values are comparable across requests
  served by the same engine (`engine` in `/healthz`), not across engines —
  the reference calibration (and the suspect detector's tuning) is
  faster-whisper/CTranslate2 output. An alternative engine must reproduce the
  same `exp(avg_logprob)` algorithm and pass a measured conformance gate
  before shipping; a deviation is a documented contract amendment, never a
  silent recalibration.
- `suspect` — hallucination soft-tag from the repetition detector (run-length +
  n-gram-density rules). Text is preserved verbatim; downstream gates decide
  how to weight flagged spans. `suspect_score` ∈ [0, 1] is a severity hint
  (the two rules score on different scales), `suspect_span` is an example
  offending substring (≤120 chars). Both are `null` when `suspect` is false.

## pyannote — `POST /v1/diarize` (port 8024)

Model: **pyannote/speaker-diarization-3.1** on pyannote.audio **3.1.1**
(pinned — 4.x drops the clustering hyperparameters this service tunes).
Weights: vendored into the image (sha256-pinned from the `pyannote-models-v1`
asset release; see `services/pyannote/models/provenance.json`) and loaded by
default — no Hugging Face token involved. Setting `DIARIZER_MODEL_NAME` to an
HF repo id restores the online path (`HF_TOKEN` required for gated repos).

Request (unknown fields rejected; `min_speakers ≤ max_speakers` enforced):

```json
{
  "path": "items/1234/audio.16khz.wav",
  "min_speakers": 1,
  "max_speakers": 10,
  "min_turn_seconds": 0.5
}
```

- `min_speakers`/`max_speakers` (defaults 1/10, each 1–20): speaker-count bounds.
- `min_turn_seconds` (default 0.5, 0.1–10): raw turns shorter than this are
  dropped before post-processing.

Response:

```json
{
  "duration_seconds": 3712.4,
  "num_speakers": 3,
  "turns": [
    {
      "start_seconds": 12.1,
      "end_seconds": 18.9,
      "label": "SPEAKER_00",
      "overlap": false,
      "overlap_seconds": 0.0
    }
  ],
  "speakers": [
    {"label": "SPEAKER_00", "total_seconds": 1800.2, "num_turns": 214}
  ]
}
```

- Post-processing order (fixed): drop turns `< min_turn_seconds` → merge
  adjacent same-speaker turns separated by less than the configured
  `min_duration_off` gap → mark overlap → compute speaker summaries.
- `label` is local to the file (`SPEAKER_NN`); global identity is the
  pipeline's job. After filtering, the label sequence may contain gaps.
- `overlap: true` marks a turn that intersects any different-speaker turn;
  `overlap_seconds` is the summed intersection time with all other speakers'
  turns, so callers can distinguish a grazing 0.2 s overlap from a
  fully-overlapped turn instead of discarding on the boolean alone.
  Overlapped speech embeds poorly — embedding extraction should skip or
  trim heavily-overlapped turns.
- `speakers[].total_seconds` / `num_turns` and `num_speakers` describe the
  **returned** turns (post-filter, post-merge). Overlapping time is counted
  for every speaker involved (no de-duplication).
- Tuning (clustering threshold, min cluster size, batch sizes, merge gap) is
  env-only (see `services/pyannote/README.md`); the request never carries
  hyperparameters beyond the speaker bounds.

## titanet — `POST /v1/embed` (port 8021)

Model: **TitaNet-Large** (`nvidia/speakerverification_en_titanet_large`
weights), 192-dim. Embedding space id: **`titanet-large-v1`** — persisted with
every vector; changing the model weights *or any parameter of the space
definition below* means a new space id, never a silent swap.

### The `titanet-large-v1` space definition (normative)

The space is defined **parametrically** — by the written parameters below, not
by any single engine or hardware. Every implementation (NeMo/CUDA, ONNX
Runtime, or future backends) must realize these parameters and pass the
measured-equivalence gate to serve under this space id.

Per-window processing chain, in order:

1. **Resample / downmix (whole file)**: if the source is not 16 kHz, the whole
   file is resampled to 16 kHz *before any slicing*; multi-channel audio is
   mean-downmixed to mono (resample first, then downmix — consistent across
   engines). All subsequent sample arithmetic is at 16 kHz. Both engines log a
   warning when this fallback fires. **Documented deviation:** the resample
   kernel is per-engine (NeMo engine: torchaudio sinc; ONNX engine:
   librosa/soxr) and is NOT covered by the parity gate — the golden corpus is
   16 kHz mono by construction, and conforming voxint deployments normalize
   all media to 16 kHz mono in the prepare stage before the services ever see
   it. Cross-engine equivalence on non-16 kHz input is unmeasured; if a
   deployment feeds non-conforming media directly, embeddings from different
   engines may diverge beyond the gate's bounds. (Follow-up: add multi-rate /
   stereo fixtures to the vector gate if this path ever becomes load-bearing.)
2. **Slice**: `[start_seconds, end_seconds)` at sample precision in the 16 kHz
   timeline — `start = int(start_seconds × 16000)`,
   `end = min(int(end_seconds × 16000), len)` (truncating int conversion, not
   rounding).
3. **Skip gates** (checked in this order, before any normalization):
   `too_short` for slices `< 1.0 s` (SNR not measured); `low_snr` for
   estimated SNR below the threshold (default 5 dB, `TITANET_SNR_THRESHOLD_DB`).
   SNR estimator: full-window RMS over the noise floor = mean of the quietest
   10% of 2048-sample frame RMS energies, clamped to [0, 60] dB, with the
   documented silence (RMS < 1e-6 → 0 dB) and digital-silence-floor
   (< 1e-10 → 40 dB) special cases. Frame tiling is normative as implemented:
   non-overlapping 2048-sample frames starting at 0, iterated while
   `start < len(window) - 2048` — the tail partial frame (and a final frame
   that would fit exactly) is excluded. The CUDA references were generated
   with this exact tiling; changing it changes `snr_db` values and gate
   outcomes.
4. **Noise reduction**: stationary spectral gating (`noisereduce`,
   `prop_decrease=0.75`).
5. **Loudness**: integrated-loudness normalization to −16 LUFS (BS.1770;
   skipped only when the meter returns non-finite loudness).
6. **Peak**: peak normalization to 0.95.
7. **Model**: TitaNet-Large forward pass on the processed 16 kHz mono window;
   mel-spectrogram front-end per the pinned NeMo 1.22 preprocessor config
   (dither 1e-5, per-feature normalization, log with zero-guard,
   n_fft/hop/window per the checkpoint's config) — implementations that do
   not embed NeMo must reproduce this front-end and prove it at the mel level.
8. **Output**: L2 normalization of the 192-dim vector.

The reference implementation of steps 1–6 is
`services/titanet/app/preprocess.py` (shared by every engine); step 7's
reference is the pinned NeMo checkpoint.

### Equivalence policy (measured, not bit-identical)

Bit-identity is not the bar — it is already false across CUDA hardware
generations. An alternative implementation may keep `titanet-large-v1` **iff**
it passes the 3-level parity gate (`tests/parity/test_titanet_onnx.py`)
against reference outputs produced by the NeMo/CUDA implementation
(fixtures: `tests/parity/fixtures/`):

- **mel level** — the reimplemented front-end matches the NeMo-internal
  mel features within tolerance on the golden corpus;
- **vector level** — per-window cosine similarity above the ratcheted
  threshold (≥ 0.999 baseline), identical `skip_reason` per window, `snr_db`
  within ±0.5 dB, on amd64 and arm64;
- **decision level** — replaying voxint's matching gates (0.60/0.70
  thresholds) on labeled same/different-speaker pairs produces no
  merge/split changes, no threshold crossings, and stable top-1/top-2
  margins, within percentile and worst-case tolerances recorded in the
  harness.

A failed gate means a new space id (`titanet-large-v2`) plus a re-embed
migration — never shipping a drifted implementation under the old id.

#### Verdict — ONNX Runtime engine: PASS (2026-08-13, amd64)

The ONNX engine (`app/engine_onnx.py`: self-exported opset-16 graph, see
`tests/parity/fixtures/onnx/provenance.json`, + the reimplemented mel
front-end `app/mel.py`) **keeps `titanet-large-v1`**. Measured on a maintainer workstation
(amd64, onnxruntime CPU EP) against the 0.3.0 CUDA references, full golden
corpus (92 embedded / 107 windows, 465 labeled pairs):

| Level | Measured | Ratcheted gate |
|---|---|---|
| mel max abs diff | 2.2e-4 | 1e-3 |
| vector cosine (min / p50) | 0.9999966 / 0.9999993 | ≥ 0.9995 |
| `skip_reason` mismatches | 0 | 0 |
| `snr_db` max diff | 0.0 dB | ±0.5 dB |
| pair-cosine drift (max) | 4.6e-4 | ≤ 2e-3 |
| 0.60/0.70 gate crossings | 0 | 0 |
| top-1 flips / margin drift (max) | 0 / 3.5e-4 | 0 / ≤ 2e-3 |

arm64 is not yet measured — the same harness must pass there before any
arm64 image ships (Phase 2 release gate). Release/CI invocations must set
`VOXINT_PARITY_REQUIRED=1`, which turns every missing prerequisite (graph,
references, deps) into a hard failure — a fully-skipped parity suite exits
green otherwise. The harness also binds the graph under test to the
committed export provenance by sha256, and the window/pair definitions to
the CUDA reference's recorded corpus hashes.

Two normative findings from the spike, binding on every non-NeMo runtime:

1. **Exact-length mel features.** `model.export()` replaces NeMo's masked
   convolutions with regular convolutions ("Turned off 25 masked
   convolutions"), so the exported graph does not mask activations past
   `length`. NeMo's `pad_to: 16` zero-padding therefore must NOT be
   reproduced: padded frames leak into every convolution near the window's
   end (measured: 0.988 cosine on a 1 s window with padding vs ≥ 0.999999
   without). Non-masking runtimes feed exactly `floor(samples/160) + 1`
   frames.
2. **Dither is training-only.** NeMo applies the config's `dither: 1e-5`
   only under `self.training`; eval-mode extraction is deterministic and the
   reimplemented front-end omits dither entirely (verified against the
   installed NeMo 1.22 source; recorded in the export provenance).

Attribution controls recorded for reference: NeMo-on-CPU reproduces the CUDA
references at cosine ≥ 0.9999992 (hardware variance is negligible for this
model), TF32 on/off changes nothing, the temp-wav round-trip in the NeMo
serving path is lossless in effect (PCM_16 quantization does not move any
window ≥ 1e-6), and the self-exported graph agrees with sherpa-onnx's
independently published export at cosine 0.99999994 on identical features.

Request (unknown fields rejected):

```json
{
  "path": "items/1234/audio.16khz.wav",
  "windows": [
    {"start_seconds": 12.1, "end_seconds": 18.9}
  ]
}
```

- `windows`: 1–512 entries. Each window requires finite values,
  `start_seconds ≥ 0`, and `end_seconds > start_seconds` (`422` otherwise).
  An empty list is a `422`. Windows extending past the end of the media are
  not an error — the slice is simply shorter and typically lands in
  `too_short`.

Response — **exactly one entry per requested window, same order** (guaranteed;
callers index results by window position):

```json
{
  "embedding_space": "titanet-large-v1",
  "results": [
    {
      "embedding": [0.013, "... exactly 192 finite floats, L2-normalized ..."],
      "snr_db": 21.4,
      "skip_reason": null
    },
    {
      "embedding": null,
      "snr_db": 3.1,
      "skip_reason": "low_snr"
    }
  ]
}
```

- Invariant: `embedding` is non-null **iff** `skip_reason` is null.
- `skip_reason` ∈ `null | "too_short" | "low_snr"`, checked in that order:
  windows under 1.0 s skip as `too_short` (with `snr_db: null` — SNR is not
  measured); windows with SNR below the threshold (default 5 dB) skip as
  `low_snr` with the measured `snr_db`.
- Per-window processing follows the normative `titanet-large-v1` space
  definition above (resample 16 kHz mono → slice → noise reduction → LUFS −16
  → peak 0.95 → TitaNet → L2); reference code in
  `services/titanet/app/preprocess.py`.

## Metal tier (bare-metal Apple Silicon) — measured status

The `metal` compute tier runs the three services natively on macOS
(`scripts/metal/voxint-metal.sh`; core stack stays in Docker via
`compose.metal.yaml`). Same `/v1` + `/healthz` contracts; heterogeneous
devices BY DESIGN:

| Service | Engine | Device | Notes |
|---|---|---|---|
| whisper | faster-whisper / CT2 | `cpu` | Same pinned large-v2 revision + int8 as the images, but the macOS arm64 CT2 wheel is a different build — measured, not assumed (`tests/parity/test_whisper_metal.py`). A Metal ASR engine is a tracked follow-up behind a pre-registered gate. |
| pyannote | pyannote.audio / torch | `mps` | `DIARIZER_DEVICE=mps` is FORCED by the launcher: backend must exist and pass the tensor-op probe or the service refuses to start — no silent CPU fallback. |
| titanet | onnxruntime | `cpu` (default) | Exactly the graph + `requirements.cpu.txt` chain the amd64 verdict measured; the macOS arm64 measurement pays down the arm64 debt flagged above. `TITANET_ORT_PROVIDERS=CoreMLExecutionProvider` enables the CoreML experiment (`device: "metal"`); the default flips only via an evidence-linked commit. |

**No committed metal reference oracle** (plan decision 3): MPS and CoreML are
not run-to-run or cross-chip (M1–M4) stable, and macOS toolchain updates
reschedule kernels unpinnably. Canonical references stay CPU/CUDA — the
spaces are defined parametrically, device-independent — and metal outputs
gate AGAINST them:

- `tests/parity/test_pyannote_metal.py` — forced-MPS vs forced-CPU vs
  `references/cuda/diarize.json`: speaker-count equality, boundary drift,
  greedy-mapped label agreement, MPS repeat stability, and a
  clustering-threshold sweep (MPS and CPU must flip speaker counts at the
  same knife edges).
- `tests/parity/test_whisper_metal.py` — native CT2 transcript vs
  `references/cuda/transcribe.json` (similarity / segments / confidence).
- `tests/parity/test_titanet_onnx.py` — the FULL 3-level gate re-run on
  arm64; `VOXINT_PARITY_ORT_PROVIDERS=CoreMLExecutionProvider` re-runs it
  under the CoreML EP, plus a same-window repeat-determinism probe.

These lanes are maintainer-run on Apple Silicon (Gate M,
docs/release-process.md) — `VOXINT_PARITY_REQUIRED` is never applied to them.
The nightly `metal-lane` workflow additionally runs them on GitHub's
`macos-15` arm64 runners (real MPS) as a regression net; its junit guard —
fail if an expected module ran zero non-skipped tests — is that lane's
substitute for the strict mode these modules deliberately lack.
Evidence lands as committed per-chip verdict reports
(chip, macOS + library versions, margins vs every bound, repeat runs) —
generalizing the ONNX verdict-table pattern above. Reference data for such
reports comes from `tools/generate_parity_references.py --tier metal
--out-dir <scratch>` (refuses the committed reference dir).

#### Verdict — metal tier: PASS (M1 Pro 16 GB, macOS 26.5.2, 2026-08-14)

Measured on maintainer Apple Silicon hardware (Apple M1 Pro, 16 GB, macOS
26.5.2), working tree `v0.7.0-12-gc6d6a72`, torch 2.5.0 / pyannote.audio
3.1.1 / faster-whisper 1.2.1 / CT2 4.8.1 / onnxruntime 1.28.0. Service-lane
numbers come from `tools/generate_parity_references.py --tier metal`
(3 repeat runs against the running native services); test-lane numbers from
the three parity modules, all green.

| Gate | Pre-registered smoke bound | Measured | Ratcheted to (slice 9) |
|---|---|---|---|
| pyannote speaker count (mps = cpu = cuda ref) | equal | equal (3 = 3 = 3, all runs) | equal (unchanged) |
| pyannote turn boundary drift vs cuda ref | ≤ 0.25 s | 0.000 s (turns identical to ref) | ≤ 0.10 s |
| pyannote mapping agreement (mps vs cpu / vs ref) | ≥ 0.99 / ≥ 0.95 | 1.00 / 1.00 (lane pass; service turns exactly match ref) | ≥ 0.995 / ≥ 0.97 |
| pyannote MPS repeat agreement | ≥ 0.999 | 1.000 (3× runs bit-identical) | ≥ 0.999 (unchanged) |
| pyannote threshold sweep (0.50 / 0.60) | counts agree per device pair | agree (lane pass) | unchanged |
| whisper transcript similarity vs cuda ref | ≥ 0.95 | 0.9907 (vad_true) / 1.0000 (vad_false) | ≥ 0.96 |
| whisper segment count / confidence drift | ± 1 / ≤ 0.15 | 0 (2 = 2, 7 = 7) / 0.0015 | ± 1 (unchanged) / ≤ 0.05 |
| titanet 3-level gate on arm64 CPU EP | existing ratcheted bounds | 7/7 pass; min window cosine 0.9999966 vs ref (floor 0.9995), repeat min 0.99999982 | unchanged (already ratcheted) |

The ratchets deliberately keep cross-chip margin rather than pinning to the
measured (mostly exact) values: the evidence is one chip (M1 Pro) on one
macOS build, MPS kernels are tuned per Apple GPU family, and the arm64 CT2
wheel is a distinct build whose VAD-path variance the ~67-word fixture
amplifies (difflib ratio drops points per word). Cross-backend agreement
(mps vs cpu, ≥ 0.995) is intentionally looser than same-device repeat
stability (≥ 0.999) — different statistical objects. Panel consult recorded
in the slice-9 commit; the split (0.97 vs 0.98 vs-ref agreement, 0.96 vs
0.965 whisper similarity) resolved toward the looser value each time because
loosening later is a formal numerics decision while re-tightening on new
multi-chip evidence is cheap.

CoreML EP experiment (`VOXINT_PARITY_ORT_PROVIDERS=CoreMLExecutionProvider`,
provider honored — validated post-construction): the full 3-level gate also
passes 7/7, with wall time indistinguishable from the CPU EP (~15 s either
way) — consistent with ORT partitioning this dynamic-length graph back to
CPU. Nothing here argues for flipping the CoreML default.

**CoreML EP default: CLOSED (slice 9) — stays off.** The measured basis: no
wall-time benefit over the CPU EP on this graph, identical 7/7 gate results,
and the CoreML path adds a cross-chip variance surface with nothing to pay
for it. Re-opening requires new measured evidence (a different graph export
or an ORT release that stops partitioning it back to CPU).

**Metal timeout factor: CLOSED (slice 9) — none needed.** Measured metal
stage speeds (transcribe 0.38–0.45× RT, diarize_embed ~0.105× RT) sit
comfortably inside the GPU-class budgets the tier inherits (4 h per-call
HTTP timeout ⇒ a 6 h recording transcribes in ~2.7 h). `metal` stays out of
the `_apply_compute_tier_profile` scaling on purpose; see
docs/timeouts-and-leases.md.

**No-metal-references call: re-affirmed by data.** 3× MPS repeats were
bit-identical on this chip, but the cross-chip argument (M1–M4 kernel
families, macOS toolchain churn) is unchanged — canonical references stay
CPU/CUDA.

Pipeline wall-times on the same hardware (console-submitted VoxConverse
clips, worker → native services): 80.2 s clip → transcribe 30.1 s +
diarize_embed 8.4 s (0.49× RT total); 122.2 s clip → 55.2 s + 12.8 s
(0.56× RT) — vs ~2.5× RT for the Docker CPU tier on the same class of
machine. These figures feed the slice-9 timeout/ratchet decisions.

Basis for the pre-registration: the Phase 0 MPS spike (2026-08-14, M1 Pro
16 GB, torch 2.5.0, vendored 3.1 pipeline) measured warm MPS diarization
~5× native-CPU speed with DER/speaker-counts/turns **identical** to CPU on
all three spike files and 3× repeats bit-stable, with zero MPS op fallbacks
(`PYTORCH_ENABLE_MPS_FALLBACK` unset). The slice-9 post-measurement pass
ratcheted the bounds from the Gate M numbers above (see "Ratcheted to"
column); loosening any bound afterwards is a numerics decision, not a test
fix.

## Contract tests

`tests/contracts/` validates — CPU-only, no model deps — that:

1. each service's pydantic schemas accept the canonical fixtures in
   `tests/contracts/fixtures/` and reject malformed variants (unknown fields,
   cross-field violations, invariant breaks);
2. service responses remain convertible to the client result types in
   `src/voxint/clients/base.py` (field names, ordering guarantees, skip
   entries);
3. path validation rejects absolute paths, traversal, and symlink escape
   outside `MEDIA_ROOT`.

Each service keeps its schemas + path validation in torch-free modules
(`services/<name>/app/schemas.py`, `services/<name>/app/paths.py`) precisely so
these tests can import them (via `importlib`, under unique module names)
without GPU dependencies. Each service's `app/schemas.py` is the wire-schema
source of truth for its endpoint; the client dataclasses are a separate
representation kept compatible by these tests.
