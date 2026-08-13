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
    services must report `"rocm"` whenever `torch.version.hip` is set. `"mps"`
    is torch Metal Performance Shaders (host adapters); `"metal"` is
    non-torch Metal backends (e.g. whisper.cpp/ggml).
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
Gated weights: the image contains **no weights**; users supply `HF_TOKEN` and
must have accepted the conditions of **both**
`pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0` on
Hugging Face.

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

1. **Slice**: `[start_seconds, end_seconds)` at sample precision —
   `start = int(start_seconds × sr)`, `end = min(int(end_seconds × sr), len)`
   (truncating int conversion, not rounding).
2. **Resample / downmix**: whole-file resample to 16 kHz (torchaudio-equivalent
   sinc interpolation) before slicing if the source is not 16 kHz; channel
   mean-downmix to mono.
3. **Skip gates** (checked in this order, before any normalization):
   `too_short` for slices `< 1.0 s` (SNR not measured); `low_snr` for
   estimated SNR below the threshold (default 5 dB, `TITANET_SNR_THRESHOLD_DB`).
   SNR estimator: full-window RMS over the noise floor = mean of the quietest
   10% of 2048-sample frame RMS energies, clamped to [0, 60] dB, with the
   documented silence (RMS < 1e-6 → 0 dB) and digital-silence-floor
   (< 1e-10 → 40 dB) special cases.
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
it passes the 3-level parity gate in `tests/parity/test_titanet_onnx.py`
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
  definition above (slice → resample 16 kHz mono → noise reduction → LUFS −16
  → peak 0.95 → TitaNet → L2); reference code in
  `services/titanet/app/preprocess.py`.

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
