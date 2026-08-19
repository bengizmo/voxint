# GPU service contracts (v1)

The three GPU model services are self-contained FastAPI images under
`services/`: **whisper** (ASR), **pyannote** (diarization), and **titanet**
(speaker embedding).
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
  the services. Retries, leases, and idempotency live in the pipeline's stage
  engine, not here. Set client timeouts accordingly (hours for long media).
- **Single-flight with bounded admission.** Each service serializes GPU
  inference internally; concurrent requests queue. Admission is bounded
  (`MAX_PENDING_REQUESTS`, default 8): a saturated service refuses new work
  with a retryable `503` instead of building an unbounded queue that outlives
  the caller's stage lease. Scale by running more replicas.
- **Versioning.** Four separate identities, never conflated:
  - *contract version*: the `/v1/` path prefix. Additive response fields are
    allowed within `v1`; renames/removals/semantic changes require `/v2`.
  - *service version*: the image's own semver (`version` in `/healthz`).
  - *model identity*: the model actually loaded (`model` in `/healthz`).
  - *embedding space* (titanet only): versions the vector semantics; any
    model **or preprocessing** change requires a new space id even if
    `/v1/embed` stays wire-compatible.
- **Schema discipline.** Request models reject unknown fields (`422`). Response
  consumers must tolerate additive fields.
- **Units.** All times are float **seconds** (fields say so: `*_seconds`).
- **Input format.** Services expect **16 kHz mono WAV**. Producing it is the
  pipeline normalize stage's invariant, not the services'. Non-conforming
  input that the decoder can still read is processed after resampling/downmix
  with a logged warning; undecodable input → `400 invalid_media`.
- **Errors.** Every error body is `{"detail": {"code": ..., "message": ...,
  "retryable": bool}}`. Stable codes: `path_violation` (400), `invalid_media`
  (400), `file_not_found` (404), `validation_error` (422, FastAPI-native body),
  `model_unavailable` (503, retryable), `saturated` (503, retryable),
  `inference_failed` (500, not retryable: the same input will fail again;
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
    the AMD GPU) and carries no torch signal. The torch-free whisper `-rocm`
    image detects the loaded HIP runtime library instead (`libamdhip64` in
    `/proc/self/maps`, checked after model construction) and reports
    `"rocm"`. `"mps"`
    is torch Metal Performance Shaders (host adapters); `"metal"` is
    non-torch Metal backends (e.g. whisper.cpp/ggml, or onnxruntime's CoreML
    EP; the titanet engine reports `"metal"` only when the CoreML EP is
    verified active in the session, never merely requested).
  - `engine` / `engine_version` identify the **inference engine only** (e.g.
    `faster-whisper`, `pyannote.audio`, `nemo`, `onnxruntime`, `whisper.cpp`),
    never the driver or userspace stack. `model` stays the weights identity:
    large-v2 is large-v2 regardless of engine.
  - `runtime` / `runtime_version` identify the compute userspace the engine
    runs on (e.g. `torch` / `2.5.0+cu118`, `torch` / `2.8.0+rocm7.2`,
    `ctranslate2` / `4.4.0`, `onnxruntime` / `1.20.1`), `null` when the
    engine has no separable runtime. Best-effort diagnostics: host-driver
    provenance is **not** readiness truth and is never required for `"ok"`.
  - The four fields are additive within `v1`; consumers must tolerate their
    absence (older services) and any future additive values.
  - **whisper only** additionally carries a cached *decode identity* (#33
    Slice 2b), populated once the model is loaded (`null` while `degraded`):
    `decode_config_hash` (digest of the effective decode config: engine,
    model, compute_type, batch_size, engine/runtime versions, VAD params +
    plan version), `vad_plan_version`, `vad_params`, and `model_revision`
    (the pinned HF snapshot). It never hashes weights per request; it exists so
    two deployments are distinguishable and a numerics change is visible.

## whisper: `POST /v1/transcribe` (port 8022)

Model: faster-whisper **large-v2**, int8 (policy pin: v3/turbo trade
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
  `WhisperModel.transcribe` (for audio VAD misclassifies as silence).

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
  empty when the model yields none (e.g. pure silence). Since #59 the pipeline
  consumes this flat list (it was previously dropped at the ASR client),
  buckets each word into its segment by maximum temporal overlap, and stores
  the result as a nullable `words` JSONB column on `transcript_segments`. This
  is derived detail beside the immutable ASR text/interval, not a numerics
  contract of its own, so it carries no parity gate.
- `transcript` is the segment texts joined with single spaces, verbatim.
- `confidence` values are `exp(avg_logprob)` clamped to [0, 1]; segment
  `confidence` is `null` when the model provides no logprob; the top-level
  `confidence` is the mean over segments that have one (0.8 if none do).
  Confidence is **engine-calibrated**: values are comparable across requests
  served by the same engine (`engine` in `/healthz`), not across engines.
  The reference calibration (and the suspect detector's tuning) is
  faster-whisper/CTranslate2 output. An alternative engine must reproduce the
  same `exp(avg_logprob)` algorithm and pass a measured conformance gate
  before shipping; a deviation is a documented contract amendment, never a
  silent recalibration.
- `suspect`: hallucination soft-tag from the repetition detector (run-length +
  n-gram-density rules). Text is preserved verbatim; downstream gates decide
  how to weight flagged spans. `suspect_score` ∈ [0, 1] is a severity hint
  (the two rules score on different scales), `suspect_span` is an example
  offending substring (≤120 chars). Both are `null` when `suspect` is false.

## pyannote: `POST /v1/diarize` (port 8024)

Model: **pyannote/speaker-diarization-3.1** on pyannote.audio **3.1.1**
(pinned: 4.x drops the clustering hyperparameters this service tunes).
Weights: vendored into the image (sha256-pinned from the `pyannote-models-v1`
asset release; see `services/pyannote/models/provenance.json`) and loaded by
default; no Hugging Face token involved. Setting `DIARIZER_MODEL_NAME` to an
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
  Overlapped speech embeds poorly. Embedding extraction should skip or
  trim heavily-overlapped turns.
- `speakers[].total_seconds` / `num_turns` and `num_speakers` describe the
  **returned** turns (post-filter, post-merge). Overlapping time is counted
  for every speaker involved (no de-duplication).
- Tuning (clustering threshold, min cluster size, batch sizes, merge gap) is
  env-only (see `services/pyannote/README.md`); the request never carries
  hyperparameters beyond the speaker bounds.

## titanet: `POST /v1/embed` (port 8021)

Model: **TitaNet-Large** (`nvidia/speakerverification_en_titanet_large`
weights), 192-dim. Embedding space id: **`titanet-large-v1`**, persisted with
every vector; changing the model weights *or any parameter of the space
definition below* means a new space id, never a silent swap.

### The `titanet-large-v1` space definition (normative)

The space is defined **parametrically**: by the written parameters below, not
by any single engine or hardware. Every implementation (NeMo/CUDA, ONNX
Runtime, or future backends) must realize these parameters and pass the
measured-equivalence gate to serve under this space id.

Per-window processing chain, in order:

1. **Resample / downmix (whole file)**: if the source is not 16 kHz, the whole
   file is resampled to 16 kHz *before any slicing*; multi-channel audio is
   mean-downmixed to mono (resample first, then downmix, consistent across
   engines). All subsequent sample arithmetic is at 16 kHz. Both engines log a
   warning when this fallback fires. **Documented deviation:** the resample
   kernel is per-engine (NeMo engine: torchaudio sinc; ONNX engine:
   librosa/soxr) and is NOT covered by the parity gate. The golden corpus is
   16 kHz mono by construction, and conforming voxint deployments normalize
   all media to 16 kHz mono in the prepare stage before the services ever see
   it. Cross-engine equivalence on non-16 kHz input is unmeasured; if a
   deployment feeds non-conforming media directly, embeddings from different
   engines may diverge beyond the gate's bounds. (Follow-up: add multi-rate /
   stereo fixtures to the vector gate if this path ever becomes load-bearing.)
2. **Slice**: `[start_seconds, end_seconds)` at sample precision in the 16 kHz
   timeline: `start = int(start_seconds × 16000)`,
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
   `start < len(window) - 2048`. The tail partial frame (and a final frame
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
   n_fft/hop/window per the checkpoint's config). Implementations that do
   not embed NeMo must reproduce this front-end and prove it at the mel level.
8. **Output**: L2 normalization of the 192-dim vector.

The reference implementation of steps 1–6 is
`services/titanet/app/preprocess.py` (shared by every engine); step 7's
reference is the pinned NeMo checkpoint.

### Equivalence policy (measured, not bit-identical)

Bit-identity is not the bar. It is already false across CUDA hardware
generations. An alternative implementation may keep `titanet-large-v1` **iff**
it passes the 3-level parity gate (`tests/parity/test_titanet_onnx.py`)
against reference outputs produced by the NeMo/CUDA implementation
(fixtures: `tests/parity/fixtures/`):

- **mel level**: the reimplemented front-end matches the NeMo-internal
  mel features within tolerance on the golden corpus;
- **vector level**: per-window cosine similarity above the ratcheted
  threshold (≥ 0.999 baseline), identical `skip_reason` per window, `snr_db`
  within ±0.5 dB, on amd64 and arm64;
- **decision level**: replaying voxint's matching gates (0.60/0.70
  thresholds) on labeled same/different-speaker pairs produces no
  merge/split changes, no threshold crossings, and stable top-1/top-2
  margins, within percentile and worst-case tolerances recorded in the
  harness.

A failed gate means a new space id (`titanet-large-v2`) plus a re-embed
migration, never shipping a drifted implementation under the old id.

#### Verdict: ONNX Runtime engine PASS (2026-08-13, amd64)

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

arm64 is not yet measured. The same harness must pass there before any
arm64 image ships (Phase 2 release gate). Release/CI invocations must set
`VOXINT_PARITY_REQUIRED=1`, which turns every missing prerequisite (graph,
references, deps) into a hard failure. A fully-skipped parity suite exits
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
  not an error. The slice is simply shorter and typically lands in
  `too_short`.

Response, **exactly one entry per requested window, same order** (guaranteed;
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
  windows under 1.0 s skip as `too_short` (with `snr_db: null`, SNR is not
  measured); windows with SNR below the threshold (default 5 dB) skip as
  `low_snr` with the measured `snr_db`.
- Per-window processing follows the normative `titanet-large-v1` space
  definition above (resample 16 kHz mono → slice → noise reduction → LUFS −16
  → peak 0.95 → TitaNet → L2); reference code in
  `services/titanet/app/preprocess.py`.

## Metal tier (bare-metal Apple Silicon): measured status

The `metal` compute tier runs the three services natively on macOS
(`scripts/metal/voxint-metal.sh`; core stack stays in Docker via
`compose.metal.yaml`). Same `/v1` + `/healthz` contracts; heterogeneous
devices BY DESIGN:

| Service | Engine | Device | Notes |
|---|---|---|---|
| whisper | faster-whisper / CT2 | `cpu` | Same pinned large-v2 revision + int8 as the images, but the macOS arm64 CT2 wheel is a different build: measured, not assumed (`tests/parity/test_whisper_metal.py`). A Metal ASR engine is a tracked follow-up behind a pre-registered gate. |
| pyannote | pyannote.audio / torch | `mps` | `DIARIZER_DEVICE=mps` is FORCED by the launcher: backend must exist and pass the tensor-op probe or the service refuses to start, with no silent CPU fallback. |
| titanet | onnxruntime | `cpu` (default) | Exactly the graph + `requirements.cpu.txt` chain the amd64 verdict measured; the macOS arm64 measurement pays down the arm64 debt flagged above. `TITANET_ORT_PROVIDERS=CoreMLExecutionProvider` enables the CoreML experiment (`device: "metal"`); the default flips only via an evidence-linked commit. |

**No committed metal reference oracle** (plan decision 3): MPS and CoreML are
not run-to-run or cross-chip (M1–M4) stable, and macOS toolchain updates
reschedule kernels unpinnably. Canonical references stay CPU/CUDA (the
spaces are defined parametrically, device-independent), and metal outputs
gate AGAINST them:

- `tests/parity/test_pyannote_metal.py` compares forced-MPS vs forced-CPU vs
  `references/cuda/diarize.json`: speaker-count equality, boundary drift,
  greedy-mapped label agreement, MPS repeat stability, and a
  clustering-threshold sweep (MPS and CPU must flip speaker counts at the
  same knife edges).
- `tests/parity/test_whisper_metal.py`: native CT2 transcript vs
  `references/cuda/transcribe.json` (similarity / segments / confidence).
- `tests/parity/test_whisper_ct2_legacy_replay.py` (#33 Slice 2a/2b): the
  `ct2-legacy` engine, driven in-process at the pinned oracle `batch_size=4`,
  must replay the frozen CT2-CPU baseline
  (`references/ct2-cpu-metal/transcribe.json`) with **zero drift** on every
  committed entry × both decode paths. Originally the anchor proof that the
  `WHISPER_ENGINE` seam refactor moved the shipped path mechanically (Slice 2a);
  after Slice 2b it also guards the shared `assemble_transcription_output`, the
  result-assembly loop deduplicated into the front layer and now called by
  `ct2-legacy` too, so zero drift here proves that dedup is byte-faithful. The
  AMI windows need the prepared work-dir corpus; the committed synthetic clips
  run anywhere on Apple Silicon.
- `tests/parity/test_titanet_onnx.py`: the FULL 3-level gate re-run on
  arm64; `VOXINT_PARITY_ORT_PROVIDERS=CoreMLExecutionProvider` re-runs it
  under the CoreML EP, plus a same-window repeat-determinism probe.

These lanes are maintainer-run on Apple Silicon (Gate M,
docs/release-process.md). `VOXINT_PARITY_REQUIRED` is never applied to them.
The nightly `metal-lane` workflow additionally runs them on GitHub's
`macos-15` arm64 runners (real MPS) as a regression net; its junit guard
(fail if an expected module ran zero non-skipped tests) is that lane's
substitute for the strict mode these modules deliberately lack.
Evidence lands as committed per-chip verdict reports
(chip, macOS + library versions, margins vs every bound, repeat runs),
generalizing the ONNX verdict-table pattern above. Reference data for such
reports comes from `tools/generate_parity_references.py --tier metal
--out-dir <scratch>` (refuses the committed reference dir).

#### Verdict: metal tier PASS (M1 Pro 16 GB, macOS 26.5.2, 2026-08-17, #33 Slice 2a/2b whisper engine)

Gate M re-run for the **v0.16.0 release**, triggered by #33 Slice 2a/2b landing the
`WHISPER_ENGINE` compatibility seam and the shared-VAD `ct2` decode engine, a
substantial refactor of `services/whisper` (non-empty `git diff v0.15.0..dee0f65
-- services/`), so the whisper metal lanes must be re-measured before the tag.
Measured on maintainer Apple Silicon hardware (Apple M1 Pro, 16 GB, macOS 26.5.2
build 25F84), working tree `main` @ `dee0f65`, the pinned metal stack (python
3.11 per-service venvs, torch 2.5.0 / pyannote.audio 3.1.1 / faster-whisper 1.2.1
/ CT2 4.8.1 / onnxruntime 1.28.0). Campaign 2026-08-16 23:05 → 2026-08-17 00:14
ADT (~1h08m). **All six lanes green** (the two #33 lanes below are new on Apple
Silicon; they were flagged as absent at `dca6c06` in the verdict just below and
are added here as that note required):

| Lane | Assertion | Result |
|---|---|---|
| `test_whisper_ct2_legacy_replay.py` (#33 Slice 2a/2b, full sweep) | `ct2-legacy` replays the frozen CT2-CPU oracle (`references/ct2-cpu-metal/`) with **zero drift** on every committed entry × both decode paths (full 30 synthetic + 30 AMI corpus at oracle `batch_size=4`) | **60 passed** (39m07s), zero drift |
| `test_whisper_ct2_self_parity.py` (#33 Slice 2b) | shared `ct2` ≈ `ct2-legacy` to **≤ 0.5pp pooled WER per vad mode** (micro-avg S/D/I/N; empty-ref clips held to a zero-insertion invariant) over synthetic + curated AMI 2–10 window subset | **2 passed** (27m33s) |
| `test_whisper_metal.py` | native CT2 transcript vs cuda ref (similarity / segments / confidence) | 3 passed |
| `test_pyannote_metal.py` | full mps=cpu=cuda-ref speaker/turn/mapping gate | 7 passed |
| `test_titanet_onnx.py` | full 3-level gate on arm64 (default EP) | 7 passed |
| `test_titanet_onnx.py` (`VOXINT_PARITY_ORT_PROVIDERS=CoreMLExecutionProvider`) | same gate under the CoreML EP + repeat-determinism probe | 7 passed |

**Scope of what this run re-measures.** Only `services/whisper` changed since
v0.15.0, so the whisper lanes (the two #33 lanes above plus `test_whisper_metal`)
are the numerics under test; the granular pyannote and titanet per-bound margins
carry over unchanged from the 2026-08-16 `batch_size=4` verdict below (those
services are byte-identical since v0.15.0). Their lanes were re-run here anyway
and stayed green as a regression net. The pytest lanes assert silently against
their bounds (pass/fail, no printed margins); the two #33 lanes prove the seam
refactor and the shared-front `assemble_transcription_output` dedup are
byte-faithful (`ct2-legacy` zero drift) and that the new shared `ct2` engine
holds equivalence to it (≤ 0.5pp WER), while `ct2-legacy` remains the shipped
default (`WHISPER_ENGINE` unset → no behavior change).

Combined with **Gate A (CUDA, `transcribe.json` byte-identical to the committed
reference)** and **Gate R (ROCm / RX 9060 XT, `device: rocm` + correct
transcription at GPU speed)**, all three maintainer GPU gates PASS at `dee0f65`
for the v0.16.0 tag.

#### Verdict: v0.17.0, Gate A/R/M carry from `dee0f65`; Gate E run fresh (2026-08-18)

Cut at `d91eadc` (72 commits after v0.16.0: the #47 settings/setup arc plus the
review-console epic). **`services/`, `tests/parity/`, all `Dockerfile*`, and every
`provenance.json` are unchanged since v0.16.0** (`git diff v0.16.0..d91eadc` over
that scope is empty), so the model-service numerics gates **carry**: Gate A (CUDA
byte-parity), Gate R (ROCm / RX 9060 XT), Gate M (Metal tier) all carry their
`dee0f65` verdicts unchanged, with no maintainer GPU re-run. CI's parity + smoke jobs
still run unconditionally on the release digests.

**Gate E (whole-pipeline E2E) does NOT carry** and was **run fresh**: its
carry-over scope is pipeline-aware (`services/`, `src/voxint/{pipeline,clients,
enrichment,db,api}`, `frontend/`, `tests/e2e/`, `tools/e2e_browser_lifecycle.py`),
all heavily touched this range, and the `tests/e2e/` suite is itself new since
v0.16.0 (first release with a Gate E suite). Both lanes PASS at `d91eadc`:
- **Pipeline lane** (`tests/e2e/test_real_pipeline.py`, real ROCm whisper `0.16.0-rocm`
  + pyannote/titanet `0.16.0-cpu` on the maintainer's AMD box, RX 9060 XT, render
  gid 990): `2 passed`, i.e. COMPLETED runs, all stages, real `titanet-large-v1`
  embeddings, no restarts.
- **Browser review lane** (#53/#57/#58 islands via `tools/e2e_browser_lifecycle.py` +
  Playwright on maintainer hardware): all island behaviours asserted (verify/skip/replay,
  click-to-edit, unsaved-edit discard warning, keymap suppression on focused controls,
  the low-confidence chips, and the #57 waveform strip: single peaks fetch, region
  click → selection+seek, keymap↔strip playhead sync), then `RECONCILE PASS` against
  `segment_review_states`.

The optional real-LLM enrichment sub-lane (`test_enrich_assets_real_llm.py`) was not
run (no maintainer endpoint configured; enrichment covered mocked in unit/contracts).

#### Verdict: v0.18.0, Gate A/R/M carry from v0.17.0; Gate E run fresh (2026-08-18)

Cut at `20f3b42`, the #67 optional bundled-LLM release (`voxint-llm`,
Qwen3-4B-Instruct-2507 Q5_K_M fetched from Hugging Face at a pinned revision and
baked into the image), plus the #79/#80/#81 deterministic-corrector work.
`git diff v0.17.0..20f3b42` over the numerics scope (`services/whisper`,
`services/pyannote`, `services/titanet`, `tests/parity/`, their `Dockerfile*` +
`provenance.json`) is **empty**, so **Gate A (CUDA byte-parity), Gate R (ROCm /
RX 9060 XT), Gate M (Metal tier) all carry their v0.17.0 verdicts**, with no
maintainer GPU re-run. The only new `services/` image is `voxint-llm`, which ships
a **serving profile, not a numerics contract**, and therefore has **no parity gate**
(CI's `publish-llm` is build-only). CI's parity + smoke jobs still run
unconditionally on the release digests.

**Gate E (whole-pipeline E2E) does NOT carry** (the #66/#67 enrichment + bundled-LLM
arc touched `enrichment/db/api/frontend`) and was **run fresh** at `20f3b42`. Both
lanes PASS:
- **Pipeline lane** (`tests/e2e/test_real_pipeline.py`, real ROCm whisper `0.16.0-rocm`
  + pyannote/titanet `0.16.0-cpu` on the maintainer's AMD box, RX 9060 XT; isolated
  worktree + disposable DB to dodge a concurrent session): **`2 passed in 132.62s`**,
  i.e. COMPLETED runs, all six stages, real `titanet-large-v1` embeddings, no restarts.
- **Browser review lane** (#53/#57/#58 islands via `tools/e2e_browser_lifecycle.py` +
  Playwright on maintainer hardware): all island behaviours asserted (2 low-confidence
  chips at indexes 1 & 3; verify-and-advance; replay teardown-guard (audio survives the
  verify re-render); skip; click-to-edit; unsaved-edit discard warning; edit+save;
  keymap suppression on a focused `<select>`; and the #57 waveform strip (single peaks
  fetch, region click → selection+seek, keymap↔strip playhead sync). Final network sweep:
  exactly 2 `/verify`, 1 `/text`, 1 `/peaks`, no stray writes. `RECONCILE PASS` against
  `segment_review_states`.

The optional real-LLM enrichment sub-lane (`test_enrich_assets_real_llm.py`) was not
run (no maintainer endpoint configured; enrichment covered mocked in unit/contracts).

#### Verdict: v0.19.0, Gate A/R/M carry from v0.18.0; Gate E run fresh, both lanes PASS (2026-08-19)

Cut at `bd702aa`, the epic-#78 deterministic-corrections arc (#82 dual-pass
composition, #83 console provenance, #84 console authoring), #85 bundled-LLM
name-hint hardening, #90 console design-token foundation (no-op refactor), the
native-macOS media-serving fix, and the native install-path remediation.
`git diff 20f3b42..bd702aa` (v0.18.0..HEAD) over the numerics scope
(`services/{whisper,pyannote,titanet}`, `tests/parity/`, their `Dockerfile*` +
`provenance.json`) is **empty**, so **Gate A (CUDA byte-parity), Gate R (ROCm /
RX 9060 XT), and Gate M (Metal tier) all carry their v0.18.0 verdicts**, with no
maintainer GPU re-run required for the model numerics. CI's parity + smoke jobs
still run unconditionally on the release digests.

**Gate E (whole-pipeline E2E) does NOT carry** (the corrections epic touched
`src/voxint/{pipeline,enrichment,db,api}` and `frontend/`) and was **run fresh** at
`bd702aa`. Both lanes PASS:
- **Pipeline lane** (`tests/e2e/test_real_pipeline.py`, real ROCm whisper `0.16.0-rocm`
  + pyannote/titanet `0.16.0-cpu` on the maintainer's AMD box, RX 9060 XT; isolated
  worktree + disposable DB to dodge a concurrent session): **`2 passed`**, i.e. COMPLETED
  runs, all six stages, real `titanet-large-v1` embeddings, no restarts.
- **Browser review lane** (islands via `tools/e2e_browser_lifecycle.py` + Playwright
  on maintainer hardware): all island behaviours asserted (2 low-confidence chips at
  indexes 1 & 3; verify-and-advance; replay teardown-guard (audio survives the verify
  re-render); skip; click-to-edit; unsaved-edit discard warning; edit+save; keymap
  suppression on a focused `<select>`; the #57 waveform strip, single peaks fetch,
  region click → selection+seek, keymap↔strip playhead sync) **plus the new #83
  correction-provenance affordances** (per-segment "corrected by domain pack" marker
  distinct from "edited", expandable rule trace, raw disclosure + reset-to-raw with no
  write, copy-raw, the "1 of 2 applied, 1 never fired" reconciliation panel, provenance
  absent on an untouched segment, and operator-edit-supersedes-provenance). Final network
  sweep: exactly 2 `/verify`, 1 `/text`, 1 `/peaks`, no stray writes. `RECONCILE PASS`
  against `segment_review_states` (2 of 5 verified at `[2,4]`, correction on segment 0).
  The #84 corrections-editor path additionally has its own unit+integration coverage
  (green in CI).

All maintainer GPU/E2E gates green at `bd702aa`; clear to tag v0.19.0.

#### Verdict: metal tier PASS (M1 Pro 16 GB, macOS 26.5.2, 2026-08-16, batch_size=4 refresh)

Gate M re-run for the **v0.15.0 release**, triggered by #33 Slice 1 flipping the
metal whisper launcher to **`BATCH_SIZE=4`** (mirroring `Dockerfile.cpu`; commit
`ece6656`), a numerics-affecting change to the metal whisper lane that must be
re-measured before a tag. Measured on maintainer Apple Silicon hardware (Apple
M1 Pro, 16 GB, macOS 26.5.2 build 25F84), working tree `main` @ `dca6c06`, torch
2.5.0 / pyannote.audio 3.1.1 / faster-whisper 1.2.1 / CT2 4.8.1 / onnxruntime
1.28.0. Numbers are from the three committed parity modules run against their
matching arm64 metal venvs (short committed fixtures), all green.

| Gate | Ratcheted bound (slice 9) | Measured (2026-08-16) |
|---|---|---|
| pyannote speaker count (mps = cpu = cuda ref) | equal | equal (3 = 3 = 3) |
| pyannote turn boundary drift vs cuda ref | ≤ 0.10 s | turns identical to ref (0.000 s) |
| pyannote mapping agreement (mps vs cpu / vs ref) | ≥ 0.995 / ≥ 0.97 | 1.000000 / 1.000000 |
| pyannote MPS repeat agreement | ≥ 0.999 | 1.000000 (bitwise-identical) |
| pyannote threshold sweep (0.50 / 0.60) | counts agree per device pair | agree |
| whisper transcript similarity vs cuda ref | ≥ 0.96 | 0.9953 (vad_true) |
| whisper segment count / confidence drift | ± 1 / ≤ 0.05 | 0 (2 = 2) / 0.0014 |
| titanet vector-level window cosine vs ref (92 windows) | ≥ 0.9995 | min 0.999997 / p50 0.999999 / max 1.000000 |
| titanet decision top-1 changes / margin drift | 0 crossings | 0 changes; margin drift p50 4.31e-05 / max 3.52e-04 |
| titanet mel-level max abs diff | existing gate | max 2.156e-04 |
| titanet same-window repeat determinism | bit-stable | max abs diff 0.000e+00 |

**On `batch_size=4` (the change this refresh gates).** The metal whisper service
ships `BATCH_SIZE=4`; the CUDA reference oracle (GPU `Dockerfile`, `BATCH_SIZE=16`)
and the in-process parity lane (`WhisperTranscriber` ctor default 16) both run at
16, a same-batch comparison. Measured directly on the Gate M short fixture,
`batch_size=4` and `batch_size=16` produce **identical** margins (both 0.9953
similarity / 2 = 2 segments / 0.0014 confidence drift): the fixture resolves to
≤ 2 VAD speech segments, so BatchedInferencePipeline packs them the same way at
either batch size. Batching is a numerical no-op here and the launcher flip does
not move the gate. A fixture with > 4 VAD segments would be needed to exercise a
genuine batch-boundary difference; the committed short fixture does not, so this
lane confirms the flip is safe without proving batch-size invariance in general.

All slice-9 ratchets from the 2026-08-14 verdict below still hold with margin
(whisper 0.9953 vs the 0.96 floor; titanet min cosine 0.999997 vs the 0.9995
floor; every pyannote agreement exact). No bound is loosened. The
`WHISPER_ENGINE` compatibility seam (#33 Slice 2a) and its `ct2-legacy` replay
lane (`tests/parity/test_whisper_ct2_legacy_replay.py`) are **not** on `main` at
`dca6c06`, so they are not part of this run; the next Gate M after 2a merges
should add that lane.

#### Verdict: metal tier PASS (M1 Pro 16 GB, macOS 26.5.2, 2026-08-14)

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
stability (≥ 0.999): different statistical objects. Panel consult recorded
in the slice-9 commit; the split (0.97 vs 0.98 vs-ref agreement, 0.96 vs
0.965 whisper similarity) resolved toward the looser value each time because
loosening later is a formal numerics decision while re-tightening on new
multi-chip evidence is cheap.

CoreML EP experiment (`VOXINT_PARITY_ORT_PROVIDERS=CoreMLExecutionProvider`,
provider honored, validated post-construction): the full 3-level gate also
passes 7/7, with wall time indistinguishable from the CPU EP (~15 s either
way), consistent with ORT partitioning this dynamic-length graph back to
CPU. Nothing here argues for flipping the CoreML default.

**CoreML EP default: CLOSED (slice 9), stays off.** The measured basis: no
wall-time benefit over the CPU EP on this graph, identical 7/7 gate results,
and the CoreML path adds a cross-chip variance surface with nothing to pay
for it. Re-opening requires new measured evidence (a different graph export
or an ORT release that stops partitioning it back to CPU).

**Metal timeout factor: CLOSED (slice 9), none needed.** Measured metal
stage speeds (transcribe 0.38–0.45× RT, diarize_embed ~0.105× RT) sit
comfortably inside the GPU-class budgets the tier inherits (4 h per-call
HTTP timeout ⇒ a 6 h recording transcribes in ~2.7 h). `metal` stays out of
the `_apply_compute_tier_profile` scaling on purpose; see
docs/timeouts-and-leases.md.

**No-metal-references call: re-affirmed by data.** 3× MPS repeats were
bit-identical on this chip, but the cross-chip argument (M1–M4 kernel
families, macOS toolchain churn) is unchanged. Canonical references stay
CPU/CUDA.

Pipeline wall-times on the same hardware (console-submitted VoxConverse
clips, worker → native services): 80.2 s clip → transcribe 30.1 s +
diarize_embed 8.4 s (0.49× RT total); 122.2 s clip → 55.2 s + 12.8 s
(0.56× RT), vs ~2.5× RT for the Docker CPU tier on the same class of
machine. These figures feed the slice-9 timeout/ratchet decisions.

Basis for the pre-registration: the Phase 0 MPS spike (2026-08-14, M1 Pro
16 GB, torch 2.5.0, vendored 3.1 pipeline) measured warm MPS diarization
~5× native-CPU speed with DER/speaker-counts/turns **identical** to CPU on
all three spike files and 3× repeats bit-stable, with zero MPS op fallbacks
(`PYTORCH_ENABLE_MPS_FALLBACK` unset). The slice-9 post-measurement pass
ratcheted the bounds from the Gate M numbers above (see "Ratcheted to"
column); loosening any bound afterwards is a numerics decision, not a test
fix.

### Whisper Metal ASR engine (issue #33): pre-registered bakeoff gate

Metal tier v1 runs whisper on **CPU** (row above); a Metal-accelerated ASR
engine is the single biggest end-to-end speedup left (runs are
transcribe-bound). This is the gate a candidate engine must pass **before** the
launcher default flips. It was designed and recorded here BEFORE any candidate
output was measured, per the numerics doctrine, and it remains the standing
gate for any future candidate. **Issue #33 closed 2026-08-17 with every named
candidate carrying its measured-or-ineligible row** (the dated verdict blocks
at the end of this section): **mlx** and **whisper.cpp** are
measured-ineligible; **CT2-MPS** is deferred upstream (an open
source-build-only PR, OpenNMT/CTranslate2#2077, where MPS-int8 can be slower
than CPU-int8). No default flip occurred; `ct2` remains the default and only
shipped engine. Each verdict block records the upstream change that would
trigger a re-measure against this same gate and frozen baseline.

**Seam contract.** Engine is selected by `WHISPER_ENGINE`
(`ct2-legacy` | `ct2` | `mlx`), resolved through a **fail-closed** registry.
An unknown or unavailable value refuses to start, never silently falls back to
CPU. A shared, engine-agnostic front layer owns audio decode → 16 kHz mono PCM,
the voxint-owned Silero **VADPlan** (speech intervals + pad/merge + decode
windows + source-time offsets), window→file timestamp remapping, the
`exp(avg_logprob)` confidence transform, and repetition soft-tagging; engines
only decode identical pre-cut windows. `/healthz` identity gains a cached decode
identity: `decode_config_hash` (digest of the effective decode config: engine,
model, compute_type, batch_size, engine/runtime versions, VAD params + plan
version), `vad_plan_version`, `vad_params`, and `model_revision` (the pinned HF
snapshot; weights are never hashed per request). Device selection is
verified fail-closed so a requested Metal engine cannot silently execute on CPU.

**Two denominators, named explicitly.** `ct2-legacy` is the untouched shipped path
(`BatchedInferencePipeline` + its internal VAD); `ct2` is the same engine on the
lifted voxint VAD. Because lifting VAD is a numerics-touching change to the
shipped engine, the frozen v1 baseline is captured from **unmodified**
`transcription.py` first, and candidate drift is reported as **two attributed
deltas**: `legacy → ct2` (segmentation contribution) and `ct2 → candidate` (pure
decode drift). All WER/CER use one **frozen, versioned normalizer** (vendored
Whisper `EnglishTextNormalizer` + provenance) applied identically to hypothesis,
gold, and baseline; raw and normalized both reported. Unit of analysis = file
(paired bootstrap CI). Decode config is pinned per engine; `language=en` is
pinned for the gate (auto-detect is a separate subset).

Pre-registered bounds (ratcheting any afterwards is a numerics decision):

| Gate | Bound |
|---|---|
| CT2 self-parity (`legacy` vs `ct2`, `vad_filter` true & false) | normalized WER-diff ≤ 0.5 pp; segmentation delta reported (implemented in `tests/parity/test_whisper_ct2_self_parity.py`, #33 Slice 2b), pooled micro-average per vad mode, empty refs held to zero-insertion |
| Contract: disagreement vs frozen CT2 baseline | normalized WER-diff ≤ 2.0 pp pooled; p95 per-file ≤ 5 pp; token agreement ≥ 97 % |
| Guardrail: accuracy vs gold | candidate normalized WER ≤ CT2 normalized WER + 1.0 pp (per-stratum and pooled) |
| Segment boundary drift (text-aligned, matched non-empty only) | p95 ≤ 0.5 s **and** p99/max ≤ 1.5 s; unmatched-segment rate ≤ 2 %; word-timestamp drift reported |
| Zero-insertion (true non-speech fixtures) | absolute 0 chars; where CT2 already emits, ≤ baseline + 0 (no growth) |
| Confidence conformance | per-file Spearman ρ(candidate, CT2) median ≥ 0.90 **and** top-level confidence MAE ≤ 0.05 **and** logprob coverage within ±5 % |
| Performance (the point of Metal) | warm long-form speedup = CT2-CPU wall / candidate wall ≥ 1.5× on the maintainer box; peak unified memory ≤ ceiling (one engine resident); cold-start recorded |
| Determinism | two warm runs identical or within a stated stdev |

A confidence miss is an explicit amendment to the Confidence contract above,
never a silent recalibration. The winner is chosen by pass/fail quality first,
then ship-eligibility, then warm long-form RTF, with memory / cold-start as
tie-breakers.

**Corpus** (`tests/parity/fixtures/bakeoff/`, pre-registered strata in a
committed `manifest.json`; audio never in git, fetched + checksum-verified by
`tools/prepare_bakeoff_corpus.py`): **AMI IHM** (CC BY 4.0: gold + committed CT2
baseline transcripts, ~15 multi-minute slices) + **TED-LIUM 3** (CC BY-NC-ND:
metrics-only, transcripts fetched-not-committed, ~15 talks) + the existing
synthetic espeak-ng fixtures for controlled silence, 30 s-window seam, and
hallucination-bait. File selection is a deterministic script (not hand-picked),
and boundary drift is scored only on files with word/tight-segment gold.

**No default flip** until this gate passes; the verdict lands **here** (a metal
verdict row + report under `docs/reports/`) in a separate change from the flip,
which ships as a MINOR release with `WHISPER_ENGINE=ct2`/`ct2-legacy` as the
rollback path and a diarization canary on a diarized AMI slice beforehand.

**Measured verdict for mlx (2026-08-17): documented-ineligible in current form.**
A 4-file diagnostic screen (AMI, 3–10 VAD windows each, committed scoring
harness vs the frozen CT2 baseline) measured `mlx-whisper==0.4.3` +
`whisper-large-v2-mlx` (fp16; greedy, no beam search exists upstream) at
pooled WER-diff **19–21 pp** vs the ≤2.0 pp bound and worst-file **115–149 pp**
vs ≤5 pp, under every decode configuration tested (temperature-fallback ladder
on/off, per-window and concatenated feeding). Not a fixable decode-config
problem: the dominant failure is confident transcription of headset crosstalk
(real speech decoded below every fallback trigger), plus systematic
greedy-fp16 vs beam5-int8 drift on clean files; where the fallback ladder did
fire it kept a degenerate last attempt (upstream `mlx-examples` #1427) and made
output worse. Performance was **not** the blocker (pooled warm speedup 1.72×,
gate ≥1.5×). Full evidence:
`docs/reports/whisper-metal-bakeoff-slice3-decode-2026-08-17.md`. Re-measure
only if upstream lands beam search and/or the #1427 fallback fix; `mlx` stays
out of `KNOWN_ENGINES`. The next measured candidate arm was **whisper.cpp
Metal** (nominally addressing both measured root causes via beam search and
int8-family quantization); see the following verdict block, also
measured-ineligible. CT2-MPS remains deferred (upstream PR
OpenNMT/CTranslate2#2077).

**Measured verdict for whisper.cpp (2026-08-17): documented-ineligible in
current form.** The same 4-file screen (same harness, frozen baseline, and
per-window feeding) measured whisper.cpp Metal (`pywhispercpp==1.5.0`,
`ggml-large-v2-q8_0`, beam_size 5, CT2-parity decode map incl.
`suppress_nst`) at worst-file WER-diff **98.57 pp** (EN2002c) vs the ≤5 pp
bound and pooled **11.60 pp** vs ≤2.0 pp. Clean-file drift is largely solved
(ES2009a 0.92 pp, IS1004d 4.79 pp, both inside the per-file bound; ES2009a beats
CT2 against gold) and reconstructed `avg_logprob` confidence matches CT2 with
MAE 0.002–0.016 where transcripts agree, but the EN2002c failure is the same
confident-crosstalk transcription that killed mlx, and measured attribution
shows no decode knob reaches it: whisper.cpp's `beam_size=5` output ≈ its
greedy output (its "beam" *samples* candidates via `std::discrete_distribution`
rather than expanding top-k, so the true-beam-suppression hypothesis was
never actually exercised), and an f16 control reproduces the Q8_0 blowup
word-for-word-scale (quantization irrelevant). Performance again not the
blocker (≈1.58× pooled, indicative). Re-measure only if upstream whisper.cpp
lands true top-k beam expansion, the strongest recorded re-measure trigger,
since every other gate dimension screened is passing or near-passing. Full
evidence: `docs/reports/whisper-metal-bakeoff-whispercpp-arm-2026-08-17.md`.
With mlx and whisper.cpp both measured-ineligible and CT2-MPS deferred
upstream, no Metal `WHISPER_ENGINE` candidate is currently eligible; `ct2`
remains the default and only shipped engine.

## Bundled local LLM: llama.cpp server (optional; issue #67)

Unlike the three model services above, this is **not** part of the transcription
pipeline and has **no numerics parity gate**; it is an optional, opt-in
enrichment endpoint an operator can turn on to get transcript enhancement and
run-asset summaries/entities with no external API key (`compose.llm.yaml`; see
`docs/operations.md`). It is documented here because it, too, is a vendored,
sha-pinned, digest-pinned model whose serving profile is fixed by measurement.

- **Image**: `services/llama-cpp/Dockerfile`, `FROM
  ghcr.io/ggml-org/llama.cpp:server@sha256:092d1291…e12625` (digest-pinned),
  baking the vendored GGUF with a build-time `sha256sum -c -` gate against
  `services/llama-cpp/provenance.json` (contract-tested, mirroring titanet).
- **Weight**: Qwen3-4B-Instruct-2507, Q5_K_M GGUF (~2.89 GB, Apache-2.0),
  `sha256 5bde5e9d…658b4ecb`, from `unsloth/Qwen3-4B-Instruct-2507-GGUF`.
- **Endpoint**: OpenAI-compatible `POST /v1/chat/completions` + `GET /health`
  on port 8080 (worker-reachable by service DNS; **no host port published**).
  Model alias `qwen3-4b-instruct-2507`.
- **Pinned serving profile** (baked as the image's default command):
  `-c 32768 -np 1 -fa on -ctk q8_0 -ctv q8_0 --jinja --reasoning off`.
  `--reasoning off` is load-bearing: the client ignores `reasoning_content`, so
  reasoning must not leak into `message.content`.
- **Sampling** is pinned to **greedy** (`temperature 0`) **client-side**
  (`SamplingProfile`, `src/voxint/clients/llm.py`), not as a server flag, so the
  BYO path's bytes are unchanged and the bundled path is deterministic.
- **Scope** (enforced by the Phase B routing, not by prose): enhancement +
  run-asset summary/entity_mentions only. Web research, LLM name attribution,
  and run-asset topics stay on the BYO endpoint and never fall back here.
- **Hints not parsed on the bundled path** (#85): because the bundled model does
  no attribution, the enhancement pass **skips parsing** its `name_hints`: a weak
  model's hallucinated out-of-range hint can no longer fail the batch for a channel
  that is discarded anyway. The **prompt is unchanged** (identical on every path):
  a measured A/B showed that removing the `name_hints` block perturbs the 4B
  model's greedy output and regresses segment faithfulness, so only the reply
  parsing differs, never the qualified prompt.
- **Device**: CPU by default (a slow backstop for a dense 4B model, which is why
  the bundled run-asset input is clamped to 16k chars); GPU strongly
  recommended (uncomment the `-ngl 99` + device-reservation block in
  `compose.llm.yaml`). Qualified against the #66 frozen corpus under the shipped
  enhancement prompt, which #85 leaves byte-for-byte unchanged (parse-only), so
  the qualification carries; see
  `docs/reports/local-llm-qualification-granite-2026-08-18.md` (with the #85
  re-measure addendum, which records the measured regression that motivated
  keeping the prompt intact).

## Contract tests

`tests/contracts/` validates (CPU-only, no model deps) that:

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

## See also

- [gpu-smoke.md](gpu-smoke.md): the manual GPU smoke procedure that exercises
  these contracts against real images.
- [quality-gates.md](quality-gates.md): how the pipeline weights the confidence,
  embedding, and diarization values these services return.
- [architecture.md](architecture.md), [release-process.md](release-process.md),
  and the [docs index](README.md).
