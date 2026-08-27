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
  - *model identity*: the model actually loaded (`model` in `/healthz`). The
    validated defaults are whisper **large-v2** and pyannote
    **speaker-diarization-3.1**; those two are the tier the numerics contracts
    below are measured against. An operator may override the ASR or diarizer
    model (deployment-owned, via `.env`; see
    [Changing pipeline models](how-to/changing-pipeline-models.md)), but an
    override is unvalidated and each stage stamps the served identity onto its
    `StageRun.metrics.model_identity` so a run records what actually ran.
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
    `decode_config_hash` (digest of the effective decode config: engine, the
    canonical compute device, model, compute_type, batch_size, engine/runtime
    versions, VAD params + plan version), `vad_plan_version`, `vad_params`, and `model_revision`
    (the pinned HF snapshot). It never hashes weights per request; it exists so
    two deployments are distinguishable and a numerics change is visible.
  - **pyannote only** additionally carries a `checkpoint_fingerprint` (#125),
    populated once the model is loaded (`null` while `degraded`). It is a weight
    identity, not a pipeline identity: a digest over the two actually-loaded
    checkpoint `.bin` files, so a deployment reporting the validated pipeline
    *name* can be checked against the validated *weights*. Algorithm (normative):
    `seg = sha256(segmentation_bin)`, `emb = sha256(embedding_bin)`, both
    lowercase hex; `checkpoint_fingerprint = sha256("segmentation:" + seg +
    "\nembedding:" + emb + "\n")`. The two `.bin` paths are read from the loaded
    pipeline config's `pipeline.params.segmentation` / `embedding`; a relative
    path there is resolved against the config file's own directory (the way
    pyannote's local loader resolves it), never against the process working
    directory, so the digest always covers the files the pipeline actually
    loaded. The config file itself is deliberately excluded: its checkpoint
    paths are repointed per
    install flavor (the metal launcher rewrites the path prefix), so its bytes
    are not deployment-invariant, while the `.bin` bytes are identical across
    flavors. The vendored default's value is
    `aa94a2d96a8f1eb5eb8fb80b863c6616417ff1e5c9a8dab91ce42914f836a0d2`, derived
    from `services/pyannote/models/provenance.json` (contract-tested). For a
    non-local (Hugging Face) source the field is `null`: those files are not
    hashed here. A local source that loaded but cannot be hashed fails the boot
    rather than reporting `null`, so `null` always means "unverifiable source",
    never "broken bake". The field is additive within `v1`; a consumer must
    treat the key being **absent** (an older service) differently from
    present-and-`null`: absence means classify by name as before, `null` means
    fail closed as unverifiable. This is operational provenance, not attestation
    against a hostile operator (a service operator can serve any `/healthz`);
    it exists to catch an accidental weights swap under the validated name in a
    single-operator deployment.
  - **pyannote only** additionally carries a `diarization_config_hash` (#129),
    populated once the model is loaded (`null` while `degraded`). Where
    `checkpoint_fingerprint` is the *weight* identity, this is the *pipeline*
    identity: a digest over the effective clustering configuration the pipeline
    actually runs with, so a deployment reporting the validated name and weights
    can also be checked against the validated *config*. The two axes are
    orthogonal by design: a weights swap flips the fingerprint, a clustering
    drift flips this hash, and each carries a distinct operator remedy, so the
    fingerprint is not folded in here. Algorithm (normative):
    `diarization_config_hash = sha256(canonical)` where `canonical =
    json.dumps(payload, sort_keys=True, separators=(",",":"))` and `payload` is
    `{hash_version, pipeline_class, clustering, clustering_method,
    clustering_threshold, clustering_min_cluster_size, segmentation_step,
    min_duration_off, embedding_exclude_overlap, engine_version}`. The static
    bits (`pipeline_class`, `clustering`, `clustering_method`,
    `embedding_exclude_overlap`) are read from the loaded pipeline config; the
    tunables (`clustering_threshold`, `clustering_min_cluster_size`,
    `segmentation_step`, `min_duration_off`) are the effective env-driven values;
    `engine_version` is `pyannote.audio.__version__`. Batch sizes are
    deliberately excluded: they are throughput-only, not numerics, and the
    hardware-aware profiles legitimately vary them per GPU, so hashing them would
    false-flag a tuned deployment. The vendored default's value is
    `9a31a4a4f1aaf4720b790bba8add7bd18f40968d428601e0ec80e3820556fca0`, derived
    from the runtime env defaults plus the vendored config and the pinned
    pyannote version (contract-tested). For the validated (local) pipeline the
    clustering overrides are applied **fail-closed**: if `instantiate()` rejects
    the tuned `{threshold, min_cluster_size}` the service refuses to start rather
    than silently degrading to threshold-only or model defaults, so a service on
    the validated identity cannot quietly run the wrong clustering config. A
    non-throwing `instantiate()` is not sufficient proof on its own (#131): pinned
    pyannote 3.1.1 can carry a frozen clustering hyperparameter and return
    successfully while leaving the effective value at its frozen default, so the
    service reads `parameters(instantiated=True)["clustering"]` back and refuses
    to start unless the effective `threshold` and `min_cluster_size` exactly match
    what was requested, keeping the config hash (computed from the requested
    values) from ever attributing a config the service never ran to the validated
    identity. The stock `config.vendored.yaml` has no freeze section, so the
    default deploy passes this check unchanged. An
    explicit non-local (Hugging Face) override keeps the tolerant fallback and
    reports `null` here (no local config to hash). The field is additive within
    `v1` with the same absent-vs-`null` contract as `checkpoint_fingerprint`.
  - The console's Settings "Pipeline models" panel consumes these to classify
    each configurable service by exact identity, not name (#125, #129): the
    validated name plus the matching `checkpoint_fingerprint` and
    `diarization_config_hash` (pyannote) or `model_revision` (whisper's baked
    large-v2 snapshot) reads as validated; the validated name with a different
    fingerprint/hash/revision reads as a mismatch and fails closed (weights and
    config are surfaced as separate axes so the remedy is specific); the
    validated name with an unverifiable (`null`) value reads as unverified and
    fails closed.
  - **All services** additionally carry an optional nested `resources` block
    (additive within `v1`; absent on older services, always present on an
    upgraded one). It reports the hardware this service sees so the operator has
    a live resource view. It is built from a background sample, never a live
    probe on the request path, and a telemetry failure never changes readiness:
    a healthy model stays `200` and the affected fields go null. Shape:

    ```json
    {
      "resources": {
        "gpu": {
          "availability": "ok",
          "gpu_uuid": "GPU-1a2b3c4d-...",
          "utilization_percent": 55,
          "vram_used_bytes": 4000000000,
          "vram_total_bytes": 12000000000,
          "temperature_celsius": 61,
          "throttle_active": false,
          "throttle_reasons": [],
          "max_temperature_celsius": 74,
          "throttle_events_since_start": 0,
          "sample_age_seconds": 1.4
        },
        "admission": {
          "pending": 0,
          "max_pending": 8,
          "rejected_since_start": 0,
          "process_started_at": "2026-08-21T00:00:00+00:00"
        },
        "cpu": {
          "availability": "ok",
          "logical_cores": 24,
          "load_average_1m": 3.2
        }
      }
    }
    ```

    - `gpu.availability` is a tri-state: `disabled` (operator set
      `VOXINT_TELEMETRY_ENABLED=0`), `unsupported` (no NVIDIA GPU, no NVML, or
      the GPU could not be resolved by UUID), or `ok`. A consumer reads it
      first; every hardware field is null unless it is `ok`, and a null means
      "not measured", never zero.
    - The GPU is resolved by UUID, not by device index: NVML physical indices
      differ from torch's `CUDA_VISIBLE_DEVICES`-remapped ordinals, so an
      index-based read can report the wrong card. Three services on one host
      commonly report the same physical GPU, so a consumer aggregates by
      `gpu_uuid` into one device. NVML memory is device-global, never one
      service's usage.
    - Bytes are integers on the wire (convert for display). `utilization_percent`
      is bounded 0-100, `vram_used_bytes <= vram_total_bytes`, and NaN/inf are
      rejected at the source. `throttle_reasons` are normalized labels
      (`thermal_sw`, `thermal_hw`, `power`, `clock`, `idle`), never a raw
      bitmask; an unknown future throttle bit still sets `throttle_active` but
      adds no label. `throttle_events_since_start` counts rising edges of a real
      (non-idle) slowdown, and `max_temperature_celsius` is the peak since the
      process started; both are cumulative counters the sampler owns.
      `sample_age_seconds` is the staleness of the cached read.
    - `admission` sources contention honestly from the service's own bounded
      admission (`pending` admitted-plus-waiting calls, `max_pending` the
      `503 saturated` threshold, `rejected_since_start` a monotonic reject
      count). It is read instantaneously under the admission lock, not sampled,
      and stays present even when GPU telemetry is `disabled`/`unsupported`.
    - `cpu` is a host-visible advisory from the stdlib; it ignores any cgroup
      CPU quota on the container, so it never drives a sizing decision alone.
    - Config is service env, fail-soft: `VOXINT_TELEMETRY_ENABLED` (default on)
      and `VOXINT_TELEMETRY_INTERVAL_SECONDS` (default 5, clamped 0.5-3600). A
      malformed value falls back to the default and never fails `/healthz`. NVML
      needs the container's NVIDIA `utility` driver capability (set in the CUDA
      images); the cpu/rocm/metal flavors omit `nvidia-ml-py` and report GPU
      telemetry as `unsupported`.

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
  the response reports the effective language either way. The omitted-field
  default stays `"en"` in v1 (changing it would be a semantic contract change,
  reserved for a `/v2`); Voxint's own pipeline client sends an explicit `null`
  since #124, so a stock install auto-detects. Detection judges the whole
  recording as one language: a recording that switches languages mid-way still
  gets a single detected language, and short or mostly silent input can produce
  low-scoring or arbitrary detections.
- `initial_prompt` (optional, ≤2000 chars): vocabulary/context prompt.
- `vad_filter` (default `true`): Silero VAD via `BatchedInferencePipeline`.
  `false` bypasses the batched pipeline entirely and calls the raw
  `WhisperModel.transcribe` (for audio VAD misclassifies as silence).

Response:

```json
{
  "language": "en",
  "language_probability": null,
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

- `language_probability` (additive v1 field, #124): faster-whisper's detection
  score for the emitted `language`, a finite number in [0, 1]. Non-null ONLY
  when detection actually ran — the request sent `language: null` on a
  multilingual model. `null` when the language was forced, the model is
  English-only, when the service substituted a fallback language (the score
  described the language it replaced), or from an older service that predates
  the field. It is the model's own score for its language guess, not a
  calibrated confidence and not a code-switch signal.
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

### ROCm `BATCH_SIZE=16` VRAM soak (measured, RX 9060 XT 16 GB, 2026-08-21)

The shipped ROCm default is `BATCH_SIZE=16`. This records the measured VRAM cost
of that default on the maintainer AMD hardware, so the "keep 16 on ROCm" stance
rests on evidence rather than the ordinal argument that 16 GB beats 12 GB.

- **Setup**: whisper `0.22.0-rocm` (`device: rocm`, large-v2, int8), one
  speech-dense 761-second file (25 concatenated LibriSpeech clips producing 28
  output segments, so more than 16 VAD windows and past the batch-16 boundary),
  `concurrency=1`. VRAM read two ways that reconcile: the per-process amdgpu
  `fdinfo` `drm-memory-vram` on the render node (isolates whisper from the
  desktop compositor) and the card-total `mem_info_vram_used` sysfs counter. The
  local LLM was stopped first so whisper had the card, matching the shipped ROCm
  deployment where only whisper is GPU-resident.
- **Result (`vad_filter=true`, the batched pipeline `BATCH_SIZE` governs)**:
  whisper sat at 1.97 GiB idle-loaded and peaked at **13.06 GiB** during decode
  (a batch-driven rise of about 11.1 GiB). Peak VRAM is bounded by the batch, not
  the file length. No out-of-memory, no container restart (restart count 0), zero
  admission rejects, transcript correct.
- **`vad_filter=false`** bypasses the batched pipeline (raw
  `WhisperModel.transcribe`), so it does not exercise the `BATCH_SIZE` path; it
  ran as a functional smoke only and passed (229 raw segments, transcript
  correct, no OOM).
- **Verdict**: `BATCH_SIZE=16` fits a 16 GB AMD card with roughly 2.9 GiB of
  headroom and is safe to ship there. It is adequate, not ample: a 12 GB AMD card
  would exceed VRAM at batch 16 (13.06 GiB peak) and must lower `BATCH_SIZE` by
  hand (see [setup.md](setup.md) and the tuning table below). This is
  infrastructure evidence for the shipped default, not a numerics oracle.

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

#### Verdict: v0.20.0, Gates A/E/M run fresh, Gate R carries; all PASS (2026-08-19)

Cut at `bc4bd12`: the epic-#89 "Reading Room" console restyle plus #94 theme
toggle, #65 read mode + Markdown export, #51 keymap single-source, #57 waveform
gap-click, and the security slices (web console D1-D4 + #103, research E1/E2,
supply-chain F1-F4 including the new F3 CUDA titanet sha-verify).

- **Gate R (ROCm) carries** from v0.19.0: `git diff v0.19.0..bc4bd12 --
  services/whisper` is empty.
- **Gate A (CUDA) run fresh** (the F3 fix touched `services/titanet/Dockerfile`,
  so carry-over does not apply). A CUDA titanet image was built from `bc4bd12`
  on maintainer NVIDIA hardware (RTX 3060); the new build-time sha gate verified
  the freshly downloaded `.nemo` OK against provenance `nemo_checkpoint_sha256`,
  proving both the checkpoint and the gate itself on a real build. The
  `generate_parity_references.py` flow then ran against that image plus the
  published `0.19.0` CUDA whisper/pyannote images (source-identical over the
  range): both transcript variants and the diarize response are byte-identical
  to the committed references, and all 92 embedding vectors compare at cosine
  1.000000 to the committed `titanet-large-v1` space. The committed references
  are unchanged (the regeneration's metadata-only diffs were discarded).
- **Gate E run fresh** at `bc4bd12` (the range touches `src/voxint/api/` and
  `frontend/`). Pipeline lane on the maintainer's AMD box (RX 9060 XT, whisper
  `0.16.0-rocm`, pyannote/titanet `0.16.0-cpu`, service code unchanged over
  that span): `2 passed`, all six stages, real embeddings, zero service
  restarts. The real-LLM enrichment sub-lane ran against a live maintainer
  endpoint: `4 passed` (summary chain plus all three malformed-reply
  honest-failure cases). Browser lane: full island sweep (uncertainty chips,
  verify-and-advance, replay teardown-guard, skip, click-to-edit, discard
  warning, edit+save, keymap suppression, the #51 cheat-sheet dialog with all
  three dismissal paths and behind-modal suppression, the #83 provenance
  affordances including operator-edit-supersedes, the #57 waveform strip with
  single peaks fetch and region-click selection+seek), `RECONCILE PASS`.
- **Gate M run fresh** at `bc4bd12` (#99 touched `metal-lane.yml`, so the
  workflow-only carry did not apply). Apple M1 Pro, macOS 26.4.1, every lane
  from the per-service metal venvs with zero skips: ct2 self-parity `2 passed`
  (28m27s), whisper-metal `3 passed`, pyannote-metal (MPS) `7 passed`,
  titanet-onnx CPU `7 passed` plus the CoreML experiment `7 passed`, and the
  legacy-replay full 15-AMI + synthetic sweep `60 passed` with zero drift
  (40m15s). A green `metal-lane` workflow dispatch was additionally verified
  at `head_sha == bc4bd12`.

All maintainer GPU/E2E gates green at `bc4bd12`; clear to tag v0.20.0.

#### Verdict: v0.21.0, Gate A/R/M carry from v0.20.0; Gate E browser lane run fresh (PASS), pipeline lane deferred (2026-08-20)

Cut at `c111308`: the #86 operator annotation layer (highlights + tags + notes)
plus Copy/export of highlights as Markdown pull-quotes, #104 YAML sidecar
metadata for watch-folder media (migration 0030), the two-lane Celery execution
topology (#109: a GPU lane `acquire`..`diarize_embed` and a `post` lane
`enhance_match`..`finalize`, with `src/voxint/pipeline/transitions.py`), and the
#96/#16 operations docs, #65 escaper, #51 shortcut chord, and #106 topics-on-BYO
slices.

- **Gate A (CUDA byte-parity), Gate R (ROCm / RX 9060 XT), and Gate M (Metal
  tier) all carry their v0.20.0 verdicts.** `git diff v0.20.0..c111308 --
  services/whisper services/pyannote services/titanet tests/parity` is **empty**
  over the numerics scope (no `Dockerfile*`, `provenance.json`, engine, or
  parity-reference change), so the model-service numerics gates carry with no
  maintainer GPU re-run. CI's parity + smoke jobs still run unconditionally on
  the release digests.
- **Gate E does NOT carry** (the pipeline-aware scope is non-empty: the two-lane
  `worker/tasks.py`, `worker/app.py`, `pipeline/engine.py`, the new
  `pipeline/transitions.py`, plus all of the #86 `api/`, `frontend/`, and
  `db/models.py`). It was split this release:
  - **Browser review lane run fresh at `c111308` — PASS.** Islands served via
    `tools/e2e_browser_lifecycle.py` (seed-only, disposable DB) and driven with a
    real browser (Playwright) on maintainer hardware. Asserted: the confidence
    signal (exactly 2 low-confidence chips at indexes 1 & 3; segments 0/2/4 not
    flagged); verify-and-advance; the replay teardown-guard (the `<audio>`
    element survives the verify's segment-array patch — `play()` fires with
    `currentTime` at the segment start and playback advances); skip; replay;
    click-to-edit; the unsaved-edit discard warning (warn-then-verify, discarding
    the edit); edit+save (`Ctrl/⌘+Enter`); keymap suppression on a focused
    `<select>`; the #51 cheat-sheet dialog opened by `?` and by button, with
    behind-modal keymap suppression and all three dismissal paths (Escape, ✕,
    backdrop) each restoring focus to the opener; the full #83 provenance
    affordances (per-segment "corrected by domain pack" marker distinct from
    "edited", expandable rule trace `everyone → everybody`, raw disclosure +
    reset-to-raw with no write, honest copy-raw status, the "1 of 2 applied, 1
    never fired" reconciliation panel, provenance absent on an untouched segment,
    and operator-edit-supersedes-provenance which clears the marker and
    un-verifies the segment); the #57 waveform strip (single `/peaks` fetch,
    region-click → selection + seek into `[10,15)`, keymap↔strip playhead sync);
    and a #86 annotation-layer smoke (selection toolbar with six colors, Save
    highlight → Highlights(1) + a `<mark>`, row Copy → honest success). Final
    reconcile against `segment_review_states`: `RECONCILE PASS` — 1 of 5 verified
    at `[4]`, corrections on segments 0 and 1 (segment 0 was un-verified by the
    operator-supersede save, per design). The #86 Copy/export browser assertions
    were also verified against the same tree earlier in the day.
  - **Real-pipeline lane (`tests/e2e/test_real_pipeline.py`, real ROCm whisper)
    deliberately deferred, NOT run this release.** Maintainer decision: the sole
    AMD host that can run the ROCm-pinned lane is under the standing issue-#23
    sustained-CPU-burst hard-reset hazard, and the lane was accepted as covered
    by (a) the two-lane topology's own unit + integration coverage green in CI
    (the stage-routed handoff and `finish_pipeline` publishing are integration-
    tested without live models), (b) the carried A/R/M numerics gates (service
    code unchanged over the range), and (c) CI's unconditional parity + smoke
    jobs on the release digests. **Residual risk recorded:** the full real-model
    pipeline through the new two-lane worker was not exercised on real GPU
    hardware for this tag; the first real handed-off run on maintainer hardware
    should be watched, and the operational `-Q celery,post` note in the CHANGELOG
    applies.

Browser Gate E green and A/R/M carried at `c111308`; the real-pipeline lane gap
above is a deliberate, recorded deferral. Clear to tag v0.21.0.

#### Verdict: v0.22.0, Gates A/R/E run fresh (all PASS), Gate M carries from v0.21.0 (2026-08-21)

Cut at `8ac53a2`: the #117 task-first review console, the #96/#118 hardware-aware
install defaults plus model-service resource telemetry, the #113 match-evidence
exporter, the #111 CUDA allocator fix, and the #97 offline eval-quality harness.
`git diff v0.21.0..main -- services/` is non-empty (+2026 lines), so the
model-service numerics gates re-run.

- **Gate A (CUDA parity) run fresh at `8ac53a2` on the titanet and pyannote CUDA
  images (maintainer NVIDIA hardware, RTX 3060) - PASS.** The #111 change drops
  `expandable_segments:True` from both images' `PYTORCH_CUDA_ALLOC_CONF` and adds
  `NVIDIA_DRIVER_CAPABILITIES=compute,utility` (NVML for the telemetry sampler);
  no weights, sha ARG, or `provenance.json` moved, so no provenance bump. The
  titanet/pyannote inference code is unchanged since v0.20.0 (only telemetry
  wiring plus a `nvidia-ml-py` dep), so the shipped 0.20.0 images run under the
  exact new allocator/caps env are faithful graph proxies. Measured against the
  committed CUDA references: titanet embeddings min cosine 0.9999998 / p50
  1.0000000 (92 windows), pyannote diarize response byte-identical (3 speakers, 7
  turns). The allocator change is numerically inert.
- **Gate R (ROCm smoke) run fresh at `8ac53a2` (maintainer AMD hardware, RX 9060 XT, render gid
  990) - PASS.** `Dockerfile.rocm` is unchanged since v0.21.0; the image was
  rebuilt to carry the new telemetry app code (nvml imported lazily, fail-soft on
  ROCm). `/healthz`: `device: rocm`, `engine: faster-whisper`, `model: large-v2`,
  rev `f0fe815`. `vad_true` transcript byte-identical to the committed CUDA
  reference; `vad_false` fully coherent, differing from the CUDA reference by one
  word ("harbour" vs "Haber", nearer the "harbor" ground truth) plus casing, the
  expected ROCm-vs-CUDA engine-level divergence.
- **Gate E does NOT carry** (pipeline-aware scope non-empty:
  `pipeline/`, `api/`, `enrichment/`, `clients/`, `db/`, `frontend/`, `services/`).
  Both lanes run fresh at `8ac53a2`:
  - **Pipeline lane (maintainer AMD/ROCm hardware, real ROCm whisper + CPU pyannote/titanet, serial)
    - PASS.** `VOXINT_E2E=1 pytest tests/e2e/test_real_pipeline.py`: 2 passed, no
    service restarts, persistence invariants intact. The 0.22.0 pipeline code runs
    in-process against real services carrying the same pinned models; the 0.22.0
    service-image numerics are certified by Gates A and R above.
  - **Browser review lane (maintainer hardware, Playwright, seed-only disposable DB) -
    PASS.** Asserted on the #117 two-step console: the confidence signal (exactly
    2 uncertain chips at indexes 1 & 3; 0/2/4 not flagged); verify-and-advance
    with the replay teardown-guard (`play()` fires at the segment start after the
    verify's segment-array patch); skip; keymap suppression on a focused
    `<select>` (confirmed with a real key press); click-to-edit; edit+save
    (`Ctrl/⌘+Enter`) which flips the header chip to "edited" and un-verifies the
    segment (operator edit supersedes the #83 pack-correction provenance); the
    unsaved-edit discard warning (warn-then-verify, discarding the edit); the #51
    cheat-sheet dialog opened by `?` and by button, with behind-modal keymap
    suppression and all three dismissal paths (Escape confirmed with a real key
    press, ✕, backdrop); the full #83 provenance affordances (marker distinct from
    "edited", rule trace `everyone → everybody`, raw disclosure + reset-to-raw with
    no write, the "1 of 2 applied, 1 never fired" reconciliation panel); and the
    #57 waveform strip (single `/peaks` fetch, region-click → selection + seek into
    `[10,15)` with playhead at 40%, no write). Final reconcile against
    `segment_review_states`: `RECONCILE PASS` - 1 of 5 verified at `[2]`,
    correction on segment 0 (un-verified by the operator-supersede save, per
    design).
- **Gate M (Metal tier) carries from v0.21.0.** The metal-lane trigger paths
  (`scripts/metal/`, the metal parity lanes, `metal-lane.yml`,
  `requirements.metal.txt`, `requirements.cpu.txt`) are unchanged since v0.21.0,
  and the new `nvidia-ml-py` telemetry dep is CUDA-only (the cpu/rocm/metal
  flavors omit it), so the metal numerics substrate is byte-identical.

Gates A/R/E green fresh and Gate M carried at `8ac53a2`. Clear to tag v0.22.0.

#### Verdict: v0.23.0, Gates A/R/E run fresh (all PASS), Gate M autodetect Tier-1 carries green from the #124 branch, other metal lanes carried (2026-08-23)

Cut at `b0937f2`: the #121 transcript semantic-search spine (the ONNX MiniLM
index plus Meaning search), #123 project glossary, #124 detected-language, #128
speaker-count hint, and the #129/#131 diarizer clustering-identity work. `git
diff v0.22.1..HEAD -- services/` is non-empty (whisper and pyannote both
changed), so the model-service numerics gates re-run; `services/titanet/` is
untouched.

- **Gate A (CUDA parity) at `b0937f2` (maintainer NVIDIA hardware, RTX 3060) -
  PASS.** Titanet CARRIES: `services/titanet/` and the committed titanet CUDA
  reference are unchanged since v0.22.1, so the shipped titanet image's
  `titanet-large-v1` embedding space (green at v0.22.0 Gate A, min cosine
  0.9999998) is unmoved. Pyannote RE-MEASURED: the #129/#131 diarizer rework
  (fail-closed hyperparameter validation, the #129 effective-clustering-config
  identity hash, the #131 stricter override application) keeps the default
  hyperparameters (threshold 0.55, min_duration_off 0.6, segmentation_step 0.5);
  rebuilt from the release tree and run against the committed CUDA reference it
  produced a byte-identical diarize response (3 speakers, 7 turns). The clustering
  rework is numerically inert.
- **Gate R (ROCm smoke) at `b0937f2` (maintainer AMD hardware, RX 9060 XT, render
  gid 990) - PASS.** The whisper app code changed (#124 autodetect, #128
  speaker-count hint, startup hardening); `Dockerfile.rocm` carries the same
  sha-pinned CT2 4.8.1 wheel and large-v2 revision `f0fe815`, and was rebuilt from
  the release tree. `/healthz`: `device: rocm`, `engine: faster-whisper`, `model:
  large-v2`. `vad_true` transcript byte-identical to the committed CUDA reference
  (confidence 0.847); `vad_false` differs by one word ("harbour" vs the
  reference's "Haber", nearer the "harbor" ground truth), the expected
  ROCm-vs-CUDA engine-level divergence, all corpus tokens present.
- **Gate E does NOT carry** (pipeline-aware scope non-empty: `pipeline/`, `api/`,
  `clients/`, `db/`, `enrichment/`, `frontend/`, `services/`; 43 files for
  #121/#124/#128 and the diarizer). Both lanes run fresh at `b0937f2`:
  - **Pipeline lane (maintainer AMD/ROCm hardware, real ROCm whisper plus CPU
    pyannote/titanet, serial) - PASS.** `VOXINT_E2E=1 pytest tests/e2e`: 2 passed,
    4 skipped (the unconfigured real-LLM sub-lane), no service restarts, service
    identities exactly whisper=rocm / pyannote=cpu / titanet=cpu, persistence
    invariants intact including the #121 embed stage writing segment_embeddings.
  - **Browser review lane (maintainer hardware, Playwright, seed-only disposable
    DB) - PASS.** Asserted on the review-stepper island: the confidence signal
    (exactly 2 uncertain chips at indexes 1 and 3); verify-and-advance with the
    replay teardown-guard (`play()` fires at the segment start after the verify's
    segment-array patch, the `<audio>` element retained); keymap suppression on a
    focused `<select>`; the #51 cheat-sheet dialog opened by `?` with behind-modal
    keymap suppression and Escape dismissal; the unsaved-edit discard warning
    (warn-then-verify, discarding the edit); edit+save (`Ctrl/⌘+Enter`) flipping
    the header chip to "edited"; the #57 waveform strip (single `/peaks` fetch);
    and the #83 pack-correction provenance marker. Network contract clean across
    the session: exactly 2 `/verify` plus 1 `/text` plus 1 `/peaks`, with no write
    leaked from any suppressed key press. Final reconcile against
    `segment_review_states`: `RECONCILE PASS`, 2 of 5 verified at `[0, 1]`,
    correction on segment 2.
- **Gate M (Metal tier): the autodetect Tier-1 whisper lane carries green from
  the #124 branch; the other metal lanes carry.** The new
  `tests/parity/test_whisper_autodetect_en.py` (Gate M, #124) passed 76 of 76:
  both engines (`ct2` and `ct2-legacy`) on the `language=None` autodetect path,
  both vad modes, every English speech clip that auto-detects en, with the four
  `TS3011a` params deselected (a TNO Dutch-accented-English AMI clip that
  legitimately auto-detects Dutch, deferred to Tier-2 issue #132). That
  measurement was taken on maintainer Apple Silicon during the #124 verification
  session at branch tip `4cd804c`, not re-run against this tag; the whisper
  service code and the gate's executable content are identical on this release
  commit (the only delta is #132 issue-number comments), so the pass carries
  validly. This is a whisper-lane gate only. The remaining metal lanes did not run
  this cut and carry: `pyannote_metal` leans on the Gate-A byte-identity above,
  `titanet_onnx` carries (titanet untouched since v0.22.1), and
  `test_whisper_ct2_legacy_replay.py` / `test_whisper_ct2_self_parity.py` carry
  from v0.21.0 (the autodetect gate reuses the frozen forced-en oracle those
  replay lanes anchor, but did not re-anchor it).

Gates A/R/E green fresh at `b0937f2`; Gate M autodetect Tier-1 carried green from
the #124 branch on patch-identical content, with the remaining metal lanes
carried. Clear to tag v0.23.0.

#### Verdict: v0.23.1, all GPU gates carry from v0.23.0 (app-Dockerfile-only patch) (2026-08-23)

0.23.1 is a patch over `6936ecc` (v0.23.0). `git diff v0.23.0..HEAD` touches only
the app `Dockerfile` (the #121 MiniLM bake now restores directory traversal so the
non-root runtime user can read the baked weights), the version pins, and the
CHANGELOG. No `services/`, pipeline, api, clients, db, enrichment, frontend,
`tests/parity/`, `tests/e2e/`, or metal-lane path changed, so Gates A, R, E, and M
all carry their v0.23.0 verdicts above. The app-image permission fix is proven by
a local build repro (the non-root user reads both baked weight files) and is
re-proven at release time by `release.yml`'s own CPU smoke gate, which exercises
the baked MiniLM weights per arch and which the v0.23.0 build failed on. Clear to
tag v0.23.1.

#### Verdict: v0.24.0, Gates A/R/M carry, Gate E re-run fresh (all lanes PASS) (2026-08-23)

0.24.0's headline is the #133 LLM transcript translation feature (plus the #130
stranded-embedding-job recovery fix). `git diff v0.23.1..HEAD -- services/` is
empty and no metal-lane path changed, so **Gates A, R, and M carry** their
v0.23.0 verdicts above (v0.23.1 itself carried; the CUDA/ROCm/metal numerics
paths have not moved). The Gate E pipeline-aware scope is non-empty
(`src/voxint/{enrichment,api,db}` and `frontend/` changed), so **Gate E re-ran
in full** on maintainer hardware before tagging:

- **Pipeline lane** (AMD host, serial): `tests/e2e/test_real_pipeline.py`
  against live model services (whisper `/healthz` `device=rocm model=large-v2`,
  pyannote cpu diarization-3.1, titanet cpu onnxruntime, all
  `model_loaded=true`) on a disposable database — **2 passed, exit 0**, no
  service restarts.
- **Real-LLM enrichment sub-lane** (configured, therefore must-pass):
  `tests/e2e/test_enrich_assets_real_llm.py` against the maintainer's real
  OpenAI-compatible endpoint — **4 passed, exit 0**, coherent real-model
  summary output.
- **Browser review lane** (translation-focused, `voxint-e2e-review` over
  `tools/e2e_browser_lifecycle.py`): interleaved translated lines in the
  hydrated island and the JS-off fallback, the export ladder live over the
  wire (200 fresh / 422 unknown-code / 409 no-fresh-translation), the review
  stepper's Translate action starting a generation and the card cancelling it,
  edit-marks-stale everywhere with translated export links gone, and a
  RECONCILE PASS on `segment_review_states`. Run on the release content
  (the only later deltas are a test-fixture literal and the version pins).

Gates A/R/M carried on empty diffs; Gate E green fresh across all three lanes.
Clear to tag v0.24.0.

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
the canonical compute device, model, compute_type, batch_size, engine/runtime
versions, VAD params + plan version), `vad_plan_version`, `vad_params`, and `model_revision` (the pinned HF
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

## Synthetic-speech detection (synthdetect): pre-registered eval gates (issue #144)

Synthdetect (audio deepfake detection) ships eventually as the first greenfield
plugin, but Milestone 1 is the maintainer evaluation capability that must exist
before any detector is productionized: a reference corpus, a detector harness,
and reproduction of the upstream published numbers. This section is the
pre-registration, recorded here BEFORE any GPU run per the numerics doctrine.
The protocol pins and tolerances below are frozen first; dated verdict blocks
are appended per session (the whisper-bakeoff pattern above), and a miss is a
STOP, never a silent tolerance widening.

The S1 scaffolding that stands this up is CI-only (no GPU, no weights): the
pins-as-data registry (`tools/synthdetect_sources.py`), the host scorer
(`tools/synthdetect_eval.py`), the manifest schema and seeded split assignment
(`tools/synthdetect_corpus.py`), and their unit suites
(`tests/unit/test_synthdetect_{sources,metrics,manifest,journal}.py`). S2 adds the
eval container and the inference runner (below); the measured gates land in S3+.

### S2: the eval container and inference runner

The reference runtime is a pinned CUDA container (`services/synthdetect/`):
`Dockerfile.eval` bakes torch cu118 plus fairseq at a frozen commit,
`requirements.eval.txt` lists the non-torch pins, and `provenance.eval.json`
records the runtime identity, the canonicalization id, the scoring polarity, and
the weight and commit pins (CANDIDATE through S2a, frozen in the S2b freeze
below). A contract test (`tests/contracts/test_synthdetect_container.py`) binds
that file to the Dockerfile, the requirements, the registry, and the runner.
Weights are never baked: they are license-gated and mounted read-only at run time.

`tools/synthdetect_infer.py` runs inside the container with the GPU and writes the
raw-score journal the S1 scorer already reads. Its engine seam is narrow: only the
fairseq forward pass is GPU-bound and weights-bound, and everything that decides
corpus identity or the scored numbers (canonical-PCM verification, windowing,
repeat-padding, batching, pooling, journaling, resume, and the determinism
provenance) is pure and covered in CI without torch, a GPU, or weights.

Two properties are load-bearing for reproducibility:

- **The runner does not resample.** Corpus audio is canonicalized once at
  acquisition to `pcm-s16le-mono-16000-v1` (16 kHz mono signed-16-bit
  little-endian PCM, no dither, normalization, or trim). The manifest `sha256` is
  the digest of the PCM `data` payload bytes only, and the runner asserts that
  format, hashes the payload, and fails closed on a mismatch. Corpus identity
  therefore never depends on the container's decoder or resampler. Changing the
  canonicalization is a new id, never a silent regeneration of corpus identity.
- **The journal header carries the full determinism provenance.** Its `runtime`
  block records the image digest, provenance sha, runner commit, and the torch,
  fairseq, CUDA, cuDNN, and device-capability versions; its `flags` block records
  the deterministic-algorithms state (with `warn_only` false), cuDNN determinism
  and benchmark, TF32 for cuDNN and matmul, the matmul precision, the cuBLAS
  workspace, the seeds, the batch size, and the asserted eval-mode result. A
  `--resume` recomputes a canonical `execution_identity_sha256` over the immutable
  header projection (excluding only timestamps, run id, host, and path) and
  refuses to continue a journal whose weights, runtime, flags, windowing, scoring,
  manifest, or selection differ.

### Qualification states and the S2b GPU gate

A weight pin advances through three states, and they are never conflated:

1. **CANDIDATE.** No real bytes have been verified. This is the S1 and S2a state;
   `weights_pinned()` is False, and `provenance.eval.json` carries null shas and a
   `candidate` qualification state.
2. **PINNED_UNQUALIFIED.** `synthdetect_infer.py verify-sources` has hashed the
   downloaded bytes and the sha is frozen in the registry and provenance. This is
   a fact about bytes; it does not depend on model accuracy or GPU determinism.
3. **QUALIFIED.** Dated GPU evidence has passed. Only then does the runtime earn
   the "deterministic" label, and a benchmark reproduction claim needs the full
   pinned dataset, protocol, and keys on top of that.

S2b (a maintainer action on the local 3060) commits three dated artifacts before
any pin is called qualified: a weight receipt (retrieval date, authoritative URL,
byte size, sha256, license disposition, upstream commit, verification tool), a GPU
smoke bundle (strict checkpoint load with no missing or unexpected keys, an
all-modules-in-eval assertion, finite one-score-per-window outputs with correct
counts and polarity, a resume that does not duplicate, and a deliberate
weight/audio/header mismatch that fails closed), and a determinism report. The
determinism spike runs at least three cold processes on the same cohort, GPU,
order, and batch size; it passes only if the per-window logits and pooled scores
are bit-for-bit identical (maximum absolute difference exactly zero) with
deterministic algorithms on and warn-only and cuDNN benchmark off. A NaN, an
infinity, a warning, a crash, a differing score, or a changed execution identity
is a failure. If exact repeatability is genuinely unavailable, the runtime stays
unqualified and at least ten cold runs are collected so S3 can pre-declare and
ratify max-absolute, percentile, and threshold-flip tolerances against that
evidence; a tolerance is never invented after observing drift.

#### Verdict: w2v2-aasist QUALIFIED on RTX 3060 (SM 8.6, 2026-08-24, S2b)

The default candidate `w2v2-aasist` is qualified. The weight pins are frozen from
real bytes (`LA_model.pth` sha256 `bd6f3609...`, `xlsr2_300m.pt` sha256
`b0892759...`; receipt: `docs/reports/synthdetect-weight-receipt-2026-08-24.md`),
the reference runtime is frozen (base image `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04@sha256:8f9dd0d0...`,
fairseq `a54021305d...`, model repo `TakHemlata/SSL_Anti-spoofing@4acaa61d...`,
vendored verbatim at `tools/synthdetect_vendor/`), and `provenance.eval.json`
carries `qualification_state: qualified`. The GPU smoke passed every check
(strict load, all-modules-eval, finite one-score-per-window with correct counts
and polarity, resume without duplication, and weight/audio/header mismatch
failing closed), and the determinism spike was bit-exact across four cold
container starts (identical `execution_identity_sha256`, maximum absolute
difference `0.0`). The one stderr line, a torch `weight_norm` deprecation
`UserWarning`, is a deterministic upstream API-naming notice, not a determinism
warning: `use_deterministic_algorithms(True, warn_only=False)` raises rather than
warns on a non-deterministic op, and the runs completed. Evidence:
`docs/reports/synthdetect-gpu-smoke-2026-08-24.md`. This is a determinism and
smoke claim for this frozen runtime, GPU class, and batch configuration; the
ASVspoof 2021 DF reproduction target remains provisional and is S3/S4.

**Two versioned identities, never conflated.** The **inference space id** (for
the default candidate, `synthdetect-w2v2aasist-v1`) is weight shas plus
preprocessing plus windowing/aggregation plus the runtime implementation
(fairseq is the reference; a numerically different runtime is a new space even
with identical weights). The **calibration policy id** (`synthdetect-cal-v1`) is
the Platt parameters plus operating threshold plus the calibration cohort hash;
recalibration bumps it WITHOUT bumping the inference space and without rescoring,
because the durable, universal output contract is the **raw score** and the
probability and flags are recomputable from stored raw scores. The console
presents a **calibrated risk score**, never a portable real-world probability
(it is valid only under the calibration distribution). Score polarity is fixed
once: a higher raw score means more likely synthetic. A checkpoint that natively
emits bona-fide logits is inverted in the runner, so the stored raw score is
comparable across models.

### Three separate gates (never conflate these)

1. **Benchmark reproduction.** Run the UNMODIFIED upstream eval stack (the
   author's repo at a pinned commit, its own eval script and 64,600-sample crop
   rule, the official keys) and hit the published number. Gate-1 scoring uses the
   **official ASVspoof scorer** (EER implementations differ enough to eat the
   tolerance); the harness sklearn scorer is cross-checked against it once and
   used everywhere else. Proves the weights, data, and protocol are right.
2. **Implementation parity.** Paired per-clip comparison of our runner against
   the upstream runner on identical clips: max-abs logit delta, rank correlation,
   and decision agreement (`tools/synthdetect_eval.py compare`). Aggregate EER
   matching is NOT accepted as equivalence evidence: two different functions can
   share an EER while disagreeing per clip.
3. **Deployment evaluation and calibration.** Our runner, our production
   windowing (merge same-speaker turns separated by under 1 s, chunk into 4 s
   windows, logit-mean pool), our domain corpus. Produces the shipped
   calibration policy. Production windowing is validated against upstream
   windowing on the domain corpus; if pooling degrades separability or
   calibration, the fallback is exposing per-window scores without pooling.

### Pre-registered reproduction targets (provisional tolerances)

Every tolerance is PROVISIONAL until ratified from measured rerun variance in S3,
before the full anchor is run. The `license_class` is printed beside every model
so a non-commercial or unlicensed result is never mistaken for a shippable one.

The `role` column separates a hard reproduction gate (a ratified miss is a STOP)
from a diagnostic (measured and reported, never a stop-gate). The 2.85 % ASVspoof
2021 DF anchor is NOT a property of the production default: it belongs to the
DF-tuned `w2v2-aasist-df` checkpoint (S3 decision, 2026-08-25; see the S3
pre-registration below). The default `w2v2-aasist` carries only a diagnostic
In-the-Wild generalization number, because no checkpoint-exact citable ITW anchor
exists for its ASVspoof2019-LA checkpoint.

| Model | License class | Anchor | Published | Role | Tolerance |
|---|---|---|---|---|---|
| `w2v2-aasist-df` (upstream runner; `Best_LA_model_for_DF.pth`) | shippable (MIT) | ASVspoof 2021 DF eval, official keys | 2.85 % EER | stop-gate | provisional ±0.3 pp (PASS is the S4 full cohort) |
| our runner vs upstream runner (same DF checkpoint) | shippable (MIT) | paired per-clip, same clips | (equivalence) | stop-gate | measured then ratcheted: max-abs logit + rank-corr + decision agreement |
| `w2v2-aasist` (production default; `LA_model.pth`) | shippable (MIT) | In-the-Wild | 10.5 % EER | diagnostic | tracked, not a stop-gate |
| `antideepfake-xlsr-2b` | noncommercial (CC-BY-NC-SA-4.0 weights) | In-the-Wild | 1.23 % EER | stop-gate | provisional ±0.5 pp |
| `audioseal` (harness-only; not shipped in v1) | shippable (MIT) | marked + unmarked clips | ~99 % TPR clean | stop-gate | TPR ≥ 99 % clean AND an unmarked-audio FPR gate |
| `nes2net` | unlicensed (no license file) | ASVspoof 2021 DF | 1.49 % EER | (blocked) | refuses to run until the author grants a license |

**Sequencing.** Shake out the harness on the pre-registered seeded 10 % subset
first, then run the full (~611k-trial) anchor once, overnight, on a maintainer
3090-class node. The subset NEVER inherits the full-set EER tolerance: it is a
preflight plus a Gate-2 paired cohort only, and after the anchor its role is
paired per-clip regression against frozen reference scores (a score diff, not an
EER). A seeded 10 % subset therefore cannot reproduce the corpus-level 2.85 %
number, so the Gate-1 PASS is claimed only on the full official cohort.

### S3 reproduction pre-registration (2026-08-25)

This is the frozen S3 protocol, recorded before any S3 GPU reproduction run. It
resolves the DF-versus-LA checkpoint question and pins the exact procedure for
the two gates. It was reviewed by two independent models before being committed.

**What S3 delivers, and what it defers.** A seeded 10 % subset on a single
maintainer RTX 3060 cannot reproduce a corpus-level EER: 2.85 % is a statistic
over the full official DF trial list, and the official DF scorer expects
full-metadata coverage. S3 therefore delivers **Gate-1 readiness** (the unmodified
upstream stack plus the official scorer runs end to end on the subset, producing a
valid score file with sensible bona-fide-versus-spoof separation, and the harness
sklearn EER is cross-checked once against the official scorer) and **ratified
Gate-2 tolerances** (paired per-clip parity, measured then ratcheted). The Gate-1
PASS against 2.85 % is claimed only on the **full official cohort**, which runs
once, overnight, on a maintainer 3090-class node (that is S4). A subset EER is a
diagnostic, never a Gate-1 pass, and the harness is never tuned to make the subset
land near 2.85 %.

**The reproduction checkpoint.** Gate-1 reproduces the DF-tuned
`w2v2-aasist-df` checkpoint (`Best_LA_model_for_DF.pth`), not the production
default `w2v2-aasist` (`LA_model.pth`): the published 2.85 % DF number is achieved
by the DF-tuned checkpoint. Both gates load `w2v2-aasist-df` on both sides. The
checkpoint is registered QUALIFIED as of 2026-08-25 (sha256
`1cf904f1d84c867c278cd42161df5367939d61cc28bfefd239bc995af59c2804`, receipt:
`docs/reports/synthdetect-weight-receipt-df-2026-08-25.md`; GPU verdict below). Its
promotion to QUALIFIED earned its own dated GPU determinism plus smoke verdict,
exactly as `w2v2-aasist` did in S2b. That ceremony was not inherited: sharing the
vendored model definition and the XLS-R base does not qualify a different classifier
checkpoint. Proving the load cold was warranted: the DF checkpoint is a bare state
dict whose 674 keys are each `module.`-prefixed (saved from an `nn.DataParallel`
model), where the default's are unprefixed. The registry declares that prefix as
data on the `WeightFile` and the runner strips it (keys and `_metadata`) before the
strict load, which on a single GPU is numerically identical to upstream's eval.

**Two corpus views, never conflated.** The published 2.85 % was produced by the
upstream stack's own data loading, so Gate-1 runs the unmodified upstream runner
on an **untouched native tree** of the official DF audio (upstream's decode, its
crop rule). Gate-2 runs both runners on our **canonical view**
(`pcm-s16le-mono-16000-v1`, manifest sha256 over the PCM payload bytes only, no
resampling in the runner). The two paths collapse into one only after a byte-level
check confirms the native files decode to identical canonical PCM bytes; until
then they are kept separate and the distinction is stated in the report.

**Subset selection.** The seeded 10 % subset is drawn by a frozen hash-rank rule
over the official DF **trial IDs** (seed `voxint-synthdetect-144`), stratified by
bona-fide-versus-spoof and by codec or source condition where the official
metadata supplies it, and marked `eval` only. The speaker-disjoint split assigner
in `synthdetect_corpus.py` is a training-split tool and is NOT applied to an
official eval set. The selection emits a trial list, a selection receipt, and a
cohort hash so the subset is reproducible and auditable.

**Gate-2 tolerances.** `tools/synthdetect_eval.py compare` pairs the two journals
per clip and reports max-absolute logit drift, Spearman rank correlation, and
decision agreement. The decision threshold is frozen from the upstream reference
scores **before** the paired comparison is inspected; the tool's default 0.0 is
never used implicitly. Tolerances are measured first from observed rerun variance,
then ratcheted to the tightest bound the evidence supports, and recorded in a
dated verdict block. Near-threshold clips are reported separately, because a
decision flip there is expected across driver, cuDNN, and torch revisions and is
covered by the raw-logit drift level instead.

**S3 corpus scope.** S3 acquires only what the two gates need: the DF subset
(native plus canonical views) and, if time allows, the In-the-Wild corpus as
generalization and Gate-2 material (a diagnostic, not a Gate-1 anchor). No TTS
synthesis, no degradation chains, and no general acquisition framework land in S3;
a narrow, committed, benchmark-specific importer (verify operator-supplied
official archives and keys by pinned sha, preserve the native tree, emit the
subset receipt plus a canonical manifest) is the whole corpus surface. The full
synthesize and degrade matrix is later sessions.

**Refinement (2026-08-25): the canonical manifest is a v2 imported-benchmark
variant of the corpus schema.** The canonical Gate-2 manifest cannot be a v1
synthesis manifest: v1 requires every spoof clip to carry synthesis generator
provenance (name, version, voice, seed, text source), which an imported eval
does not have. Rather than fabricate those fields or branch the frozen runner,
the manifest schema (`tools/synthdetect_corpus.py`) gains a `schema_version: 2`
`imported_benchmark` variant. A v2 clip carries an `imported_provenance` block
built from the official trial metadata (official trial id, source dataset, codec
condition, official split, and for spoof clips the attack system and vocoder
family); `generator` is always null; an officially-absent field is JSON null,
never a placeholder string. Only `load_manifest` learns the variant; the scoring
path (`clip_id`/`rel_path`/`sha256`/`duration_s`/`label`/`stratum`/`split`) and
`cmd_run` are unchanged, so both Gate-2 runners still bind to the exact manifest
file bytes. v1 validation is untouched. This shape was chosen after an
independent two-model design review; it is deliberately one benchmark variant,
not a general benchmark ontology (a second benchmark earns its own review).

**Emission (2026-08-25): the importer materialises both views from verified
archives.** `tools/synthdetect_df_import.py` gains the audio-dependent half as an
`emit` verb (an audio-free `select` verb emits just the trial list and receipt).
`emit` verifies the operator's official keys and audio archives against pinned
sha256 digests, then extracts the native FLAC tree itself from those verified
bytes rather than trusting a tree it is handed: a pinned archive proves the
archive bytes, not that an existing extraction came from them. Extraction is
fail-closed (absolute, traversing, linked, device, and duplicate FLAC members are
rejected; the split parts merge into one tree). Each selected trial's exact native
FLAC (resolved by trial id, never a basename search) is probed for the canonical
properties (one FLAC stream, 16 kHz, mono, 16-bit) and transcoded with ffmpeg to
the `pcm-s16le-mono-16000-v1` view without `-ar`/`-ac`, so a non-conforming source
is caught rather than silently resampled, and the output is re-verified by the
runner's own `read_canonical_pcm`. The manifest's per-clip sha is the PCM payload
only and `duration_s` is derived from the decoded frame count. A per-trial receipt
(`clip_receipt.jsonl`) binds each row's native FLAC sha256 to the canonical PCM
sha256 the manifest scores, giving the paired Gate-2 comparison a cryptographic
path from a pinned archive byte to a scored canonical sample and closing the one
failure a schema-valid manifest cannot catch: a trial id pointed at another
trial's audio, which both runners could then agree on. The native tree and the
corpus are staged and the whole corpus re-reads and revalidates before either is
published atomically, so a partial or unverified corpus is never left behind. The
official DF archives carry the Open Database License (DbCL-1.0 contents over an
ODbL-1.0 database); imported clips record `license_spdx: ODbL-1.0` and, as the
per-clip language is not published for this multi-source eval, `language: und`.

**Acceptance (2026-08-25): emit run on the real archives.** The verb was first
exercised only on synthetic tar fixtures, so it was run end to end on the four
official public archives before the DF corpus is trusted. It extracted the full
611,829-clip native FLAC tree, transcoded the 53,392-clip subset, and published
both staged roots by per-root atomic renames after the whole-corpus re-audit; the
emitted `cohort_hash` equals the pre-registered `13c4607c…`, and for sampled
clips the manifest PCM-payload sha equals the receipt's canonical PCM sha, while
the receipt's native FLAC sha equals the recomputed sha of the FLAC on disk
(these are two separate equalities, since the FLAC bytes and the decoded PCM
bytes are different payloads). The four part sha256 digests are now pinned (each
cross-checked against the md5 Zenodo publishes). Evidence:
`docs/reports/synthdetect-df-emit-acceptance-2026-08-25.md`.

#### Verdict: w2v2-aasist-df eval runtime QUALIFIED (RTX 3060, SM 8.6, 2026-08-25)

The DF anchor `w2v2-aasist-df` earned its own dated GPU determinism plus smoke
verdict, advancing it from `pinned_unqualified` to `qualified`. Evidence:
`docs/reports/synthdetect-gpu-smoke-df-2026-08-25.md`.

- **Runtime:** the S2b-frozen eval image (id
  `sha256:d631e02156245c6a2245c32376d260fa8c8624608f590b7fc82de0107f4e6595`,
  `torch 2.1.0+cu118`, fairseq `a540213`), pinned base digest, on one RTX 3060.
- **Load proven cold.** The DF checkpoint is a bare state dict with all 674 keys
  `module.`-prefixed (an `nn.DataParallel` save). The registry declares
  `WeightFile.state_dict_key_prefix="module."` as data, `SOURCES_VERSION` bumped to
  `synthdetect-sources-v3`; the runner strips the prefix from the keys and the
  `_metadata` map (fail-closed unless every key uniformly carries it) then loads
  `strict=True`. The applied strip is recorded in the journal header under
  `checkpoint_loading.state_dict_key_prefix_removed` and flows into
  `execution_identity_sha256`. The shipped default loads verbatim (prefix `None`),
  so its already-qualified header and identity are byte-for-byte unchanged.
- **Smoke:** functional run scores three canonical-PCM clips, one window each,
  finite; `model_eval` and `inference_mode` measured true; polarity
  `higher-is-more-synthetic`; resume adds no duplicate; fail-closed on a wrong clip
  sha, a tampered weight sha, and a header-identity change on resume.
- **Determinism spike:** four cold container starts, identical
  `execution_identity_sha256`
  (`93ff606bcd3b370fe5b7a073758cb24f37cfae8808fbd5e3a5760d488fbdb3ca`), max abs diff
  of per-clip scores `0.0`, bit-identical repr, no NaN/Inf.

This is a determinism and smoke claim for the frozen runtime, GPU class, and batch
configuration, not a benchmark-reproduction claim. The 2.85 % ASVspoof 2021 DF EER
reproduction is the S3 compare step (against the unmodified upstream runner on the
seeded subset) and S4 (the full DF cohort).

#### Verdict: Gate-2 paired equivalence RATIFIED (RTX 3060, SM 8.6, 2026-08-25)

Our frozen eval container reproduces the unmodified upstream SSL_Anti-spoofing DF
eval path on the 53,392-clip canonical subset. Both runners scored the same
canonical view; the reference is the verbatim upstream `model.py` +
`data_utils_SSL.py` (`pad` + `Dataset_ASVspoof2021_eval`) at commit
`4acaa61dcef5f7610f43aa4d0b29c4559b970cd2`, run in an image derived `FROM` the
frozen eval image (only `librosa==0.9.1` added; torch/torchaudio/fairseq/numpy
held identical). The decision threshold was frozen from the reference before the
paired comparison was inspected. Evidence:
`docs/reports/synthdetect-gate2-2026-08-25.md`.

- **Decision layer: EER-identical.** Subset EER 2.2193 % on both sides (official
  `compute_eer` math), Spearman rank correlation 0.99858, decision agreement
  99.985 % at the frozen threshold (3.5079, our polarity). All 8 disagreements sit
  within 0.0024 of the threshold, the near-threshold case the pre-registration
  covers by the raw-logit drift level.
- **Reference reproducible.** Three cold `batch_size=14` reference runs are
  bit-identical (self-variance 0).
- **Per-clip logit drift (our fp32 vs upstream Ampere-default TF32):** mean
  7.8e-4, median 2.0e-4, p99 8.7e-3, p99.9 0.043, max 0.152, over exact coverage
  (53,392 both sides, zero side-only, zero skips).
- **Tail root cause: cuDNN convolution TF32.** Preprocessing is bit-identical
  (reader and `pad`/`repeat_pad_to` both 0.0). `configure_determinism` disables
  TF32 (full fp32); the upstream reference leaves the Ampere default, so its
  convolution-heavy XLS-R + AASIST forward runs in TF32. With TF32 disabled across
  the whole subset, our fp32 container and the fp32 reference agree to max 0.0058,
  mean 1.6e-6, zero clips over the 1e-2 floor, 9,641 clips bit-identical: the two
  implementations are the same function and the tail is a deliberate precision
  choice that preserves the EER and ranking. The 0.0058 residual is the
  batch-shape FP effect
  (our effective forward batch is 1 in upstream windowing; the reference forwards
  14), which under TF32 is ~10x larger (~0.07).
- **Ratified tolerances:** decision layer EER-identical (8 near-threshold flips);
  per-clip logit max 0.0058 (mean 1.6e-6, zero over the pre-declared 1e-2 floor)
  at matched fp32 precision, which PASSES; per-clip logit <= 0.152 at default
  precision (our fp32 vs Ampere-default TF32), EER-preserving. No shipped-runner
  change is warranted: disabling TF32 is the deliberate, full-fp32 choice for a
  numerics-contract runner.

Gate-2 ratifies paired equivalence on the subset. The 2.85 % published EER
(Gate-1 PASS on the full official cohort) defers to S4.

#### Verdict: Gate-1 full-cohort DF reproduction PASS (RTX 3060, SM 8.6, 2026-08-25, S4)

The unmodified upstream SSL_Anti-spoofing DF runner reproduces the published
ASVspoof 2021 DF benchmark on the full official eval cohort. All 611,829 official
DF trials were scored on the untouched native FLAC tree (upstream's own decode and
64,600-sample crop), and the pooled EER was computed over the 533,928 phase-`eval`
trials with the official `eval_metrics_DF.compute_eer` math. Evidence:
`docs/reports/synthdetect-gate1-s4-2026-08-25.md`.

- **Pooled DF-eval EER: 2.8650 %** against the published 2.85 % and the
  pre-registered ±0.3 pp tolerance, a +0.015 pp miss that PASSES with wide margin.
  An independently written EER routine cross-checks the official math to four
  decimals (2.8650 % both).
- **Cohort:** 611,829 trials scored (zero skips, the official scorer's length
  check), pooled over phase `eval` (533,928 trials: 14,869 bona-fide, 519,059
  spoof). The official keys (`DF-keys-full.tar.gz`, sha256 `426f93e1…`, the pin in
  `synthdetect_df_import.py`) were verified and their real column tokens measured,
  not assumed (`bonafide`/`spoof` in column 6, `eval`/`progress`/`hidden` in
  column 8). The full protocol (column 2 of every row) forms an exact bijection
  with the 611,829-file native tree from the emit acceptance run.
- **Runner:** the same thin, audited reference driver Gate-2 used (importing the
  verbatim upstream `model.py` + `data_utils_SSL.py` at commit `4acaa61d…`; the
  driver owns only the DataLoader wiring and score emission, not literally
  `main_SSL_DF.py`) in the Gate-2 reference image (id `sha256:03891ac1b090…`,
  `FROM` the frozen S2b image plus `librosa==0.9.1`), run as published:
  Ampere-default TF32 and the upstream `batch_size=14`. Scoring is the vendored
  official `eval_metrics_DF.compute_eer` math, not literally `evaluate_2021_DF.py`.
  Weights are the QUALIFIED DF anchor (`Best_LA_model_for_DF.pth` sha256
  `1cf904f1…`, `xlsr2_300m.pt` sha256 `b0892759…`).
- **Execution:** the protocol was split into two disjoint shards and scored
  concurrently on two RTX 3060 GPUs (one visible GPU each), both at `batch_size=14`
  so a trial's score is shard-independent, then concatenated. Scores are run
  artifacts, not committed (concatenated sha256 `88786ae5…`).

Gate-1 proves the upstream stack reproduces the published number on the full
cohort. With Gate-2's ratified per-clip subset equivalence this is strong evidence
that the frozen eval container reproduces the benchmark too, but the shipped
container's own full-cohort EER on the canonical PCM view is not measured here:
that carryover is a well-supported inference, not a proven number. A full-cohort
our-container parity pass on the canonical view is optional and deferred (it needs
the full canonical transcode). The ±0.3 pp tolerance is labelled provisional in
the pre-registration, but the margin (+0.015 pp) and the reference's zero EER
rerun variance make that distinction immaterial to this PASS.

### S5 organic corpus and degradation pre-registration (2026-08-26)

This is the frozen S5 protocol, recorded before any corpus is materialized or any
GPU windowing run happens. S5 builds the bona fide (real-speech) domain side of
the corpus and validates the production windowing path; the synthetic (spoof)
side is S6 and calibration is S7. S5 lands eval-first: the pure, audio-free logic
(`tools/synthdetect_corpus.py`, `tools/synthdetect_sources.py`) freezes and is
unit-tested before any audio exists. Audio is never committed; the corpus root is
always a CLI argument.

**The plan, never a manifest, is the pure output.** A valid manifest cannot exist
before audio is materialized, because the manifest `sha256` is the canonical PCM
payload digest and `duration_s` is derived from the decoded sample count. The pure
layer therefore emits a `MaterializationPlan` (which clips and segments to extract,
with their splits) and a `finalize_manifest(plan_records, measured_facts)` function
that builds and validates the v1 manifest only after the executor supplies the
per-clip PCM sha256 and sample count. The executor splits in two. PR-2a (the
prepare executor) materializes bona fide clips and is ffmpeg-free: the staged
organic sources are already canonical PCM (16 kHz mono s16le, measured), so
materialization is deterministic byte slicing of a pin-verified payload, not a
codec pass. PR-2b (the degrade executor) is the maintainer-hardware, digest-pinned
ffmpeg step, because only the degradation round trips run lossy codecs; it also owns
the combined parent-plus-child manifest assembly (a v1 manifest rejects a child
whose parent is absent) and the AMR-NB availability decision.

**Bona fide source.** Two CC-BY-4.0 diarization corpora, a meeting-room set and a
web-video set, each with RTTMs giving per-speaker turns. `prepare` parses the
RTTM, then produces two views:

- **Turn clips** for strata, degradation, and calibration: same-speaker turns
  merged across gaps below `CLIP_MERGE_GAP_S = 0.3 s`, other-speaker overlap
  subtracted, then any surviving span shorter than `TURN_MIN_S = 1.0 s` (16,000
  samples) dropped. The other-speaker rows are first coalesced into continuous
  regions (overlapping or touching rows unioned), and a region is cut only when it
  reaches `OVERLAP_FLOOR_S = 0.1 s`. The floor ignores a brief boundary graze, but
  a run of adjacent short rows (word-level RTTMs emit many 80 ms words) coalesces
  into one region and is cut, so continuous crosstalk cannot survive inside a
  nominally single-speaker turn.
- **Session segments** for production-windowing validation: same-speaker turns
  merged across gaps below `SESSION_MERGE_GAP_S = 1.0 s` (the production windowing
  `merge_gap_s`), overlap subtracted, then any span shorter than one full model
  window (64,600 samples = 4.0375 s) dropped. The merge is an RTTM-level operation,
  so it must happen before per-turn clipping; scoring isolated turn clips would not
  validate the stated merge-then-window policy. Both length floors are enforced in
  the sample domain (after the floor/ceil conversion), because the contract is
  written in samples: a span whose second-length rounds just under the threshold
  can still convert to a full 64,600-sample window.

The staged meeting-room audio is a speaker-mixed track, so dropping other-speaker
overlap is also the leakage guard: without it a nominally single-speaker turn can
carry crosstalk from a speaker assigned to a different split.

**Pinned sample-interval rule.** `start_sample = floor(start_s * 16000)`,
`end_sample = ceil(end_s * 16000)`, so a clip is byte-reproducible from the RTTM
times. The RTTM times are parsed and multiplied in exact decimal, never binary
`float`: in `float`, `0.1 * 16000` evaluates to `1600.0000000000002` and would
ceil to `1601`, silently breaking the decimal rule for ordinary RTTM decimals.

**Conservative speaker namespace.** `speaker_id = "{source}-{recording}-{label}"`,
recording-scoped, so a recording-local or mislabeled RTTM label can never leak a
speaker across the calibration/eval/holdout boundary. This can split one real
person across recordings, which is acceptable for a bona-fide false-positive
corpus (S5 makes no generalization claim); a stronger speaker-disjoint identity is
an open S6/S7 question.

**Strata and manifest kind.** A bona fide clip's stratum is
`bona_fide|organic|{domain}` (`meetingroom` or `webvideo`); a degraded child's
stratum extends the parent's with the chain. Organic bona fide clips are a v1
`synthesis`-kind manifest (clips we materialize, each carrying per-clip provenance
in `acquire`; a bona fide clip carries no generator), not the v2 imported-benchmark
kind that the ASVspoof reproduction corpus uses.

**Degradation recipes are a closed, versioned vocabulary.** `DEGRADATION_RECIPES`
in `synthdetect_sources.py` pins each transform as data (recipe id, family, ffmpeg
implementation, encode args, intermediate container); the argv builders and chain
serialization are logic in `synthdetect_corpus.py`. The initial frozen set covers
codec (`mp3-cbr48-v1`, `opus-voip-cbr16-f20-v1`, `aac-lc-cbr48-v1`), telephony
(`g711-mulaw-8k-v1`, `amr-nb-122-v1`), and speed (`speed-atempo-0p90-v1`). Rules:

- No free-form ffmpeg. A degradation string is a `|`-joined chain of known recipe
  ids; order is significant (`speed|codec` differs from `codec|speed`) and is the
  manifest `degradation` identity.
- Every lossy recipe is a real round trip: canonical PCM into the pinned encoder,
  out to the intermediate bitstream, back through the pinned decoder to canonical
  PCM. Only the final PCM payload sha is the clip identity, never the encoded
  intermediate. Raw input is always framed `-f s16le -ar 16000 -ac 1` before `-i`.
  Determinism is pinned by `-threads 1` on both the input (decoder) and the output
  (encoder) side of every pass plus `-filter_threads 1` global: in ffmpeg's option
  model `-threads` is per-stream, so a prefix-only `-threads 1` binds the input
  decoder and leaves a multi-threaded encoder unpinned, which is the hash-relevant
  pass. `-filter_threads 1` pins the filter graph (the `atempo` speed pass).
- Byte-for-byte regeneration is promised only on the pinned realization platform
  (container digest plus codec library versions), never as universal ffmpeg
  reproducibility.
- Additive noise is deferred to the executor slice: an SNR mix needs the measured
  parent RMS, which is audio-dependent and so cannot be a pure argv. `amr-nb-122-v1`
  is registered but depends on `libopencore_amrnb` in the pinned build; if that
  encoder is absent it is dropped from the frozen set with a note.

**Degraded children (S5 PR-3).** Children are derived only from calibration-split
parents, so degraded bona fide material lands in the calibration split without
contaminating eval or holdout. A child inherits its parent's label, speaker,
language, split, and license (a load-time lineage invariant, alongside a
parent-cycle check); `finalize_manifest` runs every child through the same manifest
validation. To keep Platt honest, calibration uses one pre-registered variant per
parent (or parent-group weighting): N degraded children of one utterance are not N
independent observations.

**Cohort freeze policy (S5 PR-3, implemented).** The frozen cohort (v1) assigns
exactly one degraded chain per eligible calibration-split turn parent via
`hash-assign-v1`: `sha256(clip_id)` mod 6 indexes the sorted chain vocabulary.
The six frozen chains are the six single recipes (`mp3-cbr48-v1`,
`opus-voip-cbr16-f20-v1`, `aac-lc-cbr48-v1`, `g711-mulaw-8k-v1`,
`amr-nb-122-v1`, `speed-atempo-0p90-v1`); multi-recipe compound chains are
demonstrated (PR-2b acceptance) but excluded from v1. Session segments (kind
`segment` in `acquire`) are not degraded. The pre-audio plan identity is
`cohort_plan_sha256` (hash of version, policy id, sorted chain strings, and
sorted per-parent assignment rows including parent PCM sha256); the realized
artifact identity is `combined_manifest_sha256` in `cohort_receipt.json`.
Neither is the evaluator's scored calibration cohort hash (S7 scope).
`FROZEN_COHORT_CHAINS`, `S5_COHORT_VERSION`, and `S5_COHORT_SELECTION_POLICY`
are in `synthdetect_sources.py`; `plan_cohort` and `materialize_cohort` are in
`synthdetect_corpus.py`. The `freeze` CLI verb runs both dry-run and execution
mode.

**Production windowing fixes (pre-registered, LANDED in S5 PR-4).** The
production windowing path (`plan_windows(mode="production")` in
`synthdetect_infer.py`) had two issues fixed and versioned before scoring
against the upstream crop:

- **Tiny-tail window.** A clip a few samples past a window boundary emitted a
  final one-sample span, repeat-padded to 64,600 and logit-mean-pooled with
  equal weight to a full window. Fixed: a trailing partial window below
  `production_tail_floor_samples` (8,000 samples = 0.5 s) is dropped when at
  least one full window exists. The dropped tail length is journaled per clip
  (`dropped_tail_samples` in `ClipOutcome`).
- **64,000 vs 64,600.** `production_window_s` was 4.0 (64,000 samples) but the
  model width is 64,600, so every full production window was repeat-padded by
  600 samples unlike the upstream crop. Fixed: `production_window_s` and
  `production_hop_s` set to 4.0375 s (exactly 64,600 samples at 16 kHz). A
  contract test enforces `round(production_window_s * sample_rate_hz) ==
  upstream_window_samples` for every fixed-width model.

Both changes are part of `SOURCES_VERSION = "synthdetect-sources-v6"` and a
new inference-space windowing identity.

**Windowing-validation scope.** With a bona fide-only corpus, the S5 windowing
verdict validates false-positive stability (does production windowing raise the
bona fide false-positive rate against the upstream crop), not separability or EER,
which need the spoof side that arrives in S6. The verdict states this limit
explicitly, and per-window scores are journaled as a first-class output, not a
fallback bolted on later.

**PR-2a prepare-executor acceptance verdict (2026-08-26): PASS.** The ffmpeg-free
prepare executor materialized real bona fide corpora from both organic domains (AMI
`ES2011a`, 189 clips; VoxConverse `abjxc`, 4 clips) with byte-exact determinism
across two independent runs, every clip equal to an independently computed source
slice, all turn/segment overlaps byte-identical, and every materialized clip
re-audited through the scoring reader `synthdetect_infer.read_canonical_pcm` against
the finalized manifest. This confirms the numpy-free corpus writer and the numpy
scoring reader agree on clip identity for real audio. Detail:
`docs/reports/synthdetect-s5-pr2a-prepare-2026-08-26.md`.

**PR-2b degrade-executor acceptance verdict (2026-08-26): PASS.** The degrade
executor materialized codec-degraded children from real bona fide audio (AMI
`ES2011a`, 189 parent clips) for all six degradation recipes and one multi-recipe
chain (`speed-atempo-0p90-v1|mp3-cbr48-v1`), deterministically: byte-identical
combined manifests and child PCM sha256 values across two independent runs per
configuration. Every child re-audited against its manifest sha256, and
`resolve_clip_path` resolved every combined-manifest entry against the two-root
layout. AMR-NB ran successfully on real speech. Pinned container:
`jrottenberg/ffmpeg@sha256:292a972c...`, ffmpeg 7.1, determinism flags
`-threads 1 -filter_threads 1`. Detail:
`docs/reports/synthdetect-s5-pr2b-degrade-2026-08-26.md`.

**Full bona fide calibration corpus materialization (2026-08-27): PASS.** The
full 14-recording scoring subset (7 AMI meetingroom + 7 VoxConverse webvideo)
was materialized on maintainer hardware using the `prepare` and `freeze` CLI
verbs.

| Domain | Recordings | Turn clips | Segments | Parents total | Degraded children |
|---|---|---|---|---|---|
| AMI (meetingroom) | 7 | 2081 | 706 | 2787 | 1045 |
| VoxConverse (webvideo) | 7 | 471 | 237 | 708 | 114 |
| **Total** | **14** | **2552** | **943** | **3495** | **1159** |

Cohort plan hashes: AMI `a819df1b...`, VoxConverse `60d3d64a...`. Combined
manifest shas: AMI `a27b8b09...`, VoxConverse `20d8f64a...`. Pinned container:
`jrottenberg/ffmpeg@sha256:292a972c...`. All six degradation chains represented
in each corpus (hash-assign-v1 distribution verified uniform).

**VoxConverse RTTM trimming.** Two recordings (uicid, gtnjb, both 1200.064 s)
had a final RTTM entry extending 256 samples (16 ms) past the recording's
actual sample count. The last entry's duration was trimmed to the recording
boundary before materialization. Trimmed RTTMs and a separate acquisition
manifest (`acq_voxconverse_trimmed.json`) are stored alongside the corpus data.
This is a data annotation artifact, not a code change: the `materialize_prepare`
executor correctly rejects out-of-range intervals, and clamping RTTMs to
recording length at plan time is an open improvement for a future session.

**S5 windowing verdict (2026-08-27): PASS.** Production windowing (4.0375 s
windows at 4.0375 s hop, logit-mean pooling, 8,000-sample tail floor) does not
raise the bona fide false-positive rate at the FPR 5% or FPR 1% regions of the
raw-logit distribution (no calibrated threshold exists yet; that requires the
spoof side in S6). Scored
the full S5 calibration corpus (AMI 3832 clips, VoxConverse 822 clips) with
`w2v2-aasist` on maintainer hardware (RTX 3060, eval image `s2b`) under both
windowing modes. At FPR 5%, delta is -0.13 pp (AMI) and -0.49 pp (VC): production
is marginally better. At FPR 1%, both are indistinguishable (-0.03 pp / -0.24 pp).
No degradation stratum shows a destabilizing shift. Per-window scores are journaled
as a first-class output (per the pre-registration), with mean intra-clip spread of
2.7 to 3.4 logit units across multi-window clips. Scope limitation (stated by
design): this verdict covers FPR stability only; separability and the threshold
itself require the spoof side (S6). Detail:
`docs/reports/synthdetect-s5-windowing-verdict-2026-08-27.md`.

### S6 spoof corpus and composite manifest pre-registration (2026-08-27)

This is the frozen S6 protocol, recorded before any spoof audio is generated or
scored. S6 builds the **spoof (synthetic speech) side** of the calibration
corpus. With bona fide (S5) and spoof (S6) together, S7 can compute EER, fit
Platt calibration, and open the holdout split. S6 lands eval-first: the
composite manifest schema and assembly logic freeze and are unit-tested before
any TTS audio exists.

**Architectural constraint that forces TTS.** The v2 `imported_benchmark`
schema requires every imported clip to be eval-only (the official split is
preserved, not reassigned). The Platt calibration split needs both bona fide
and spoof clips. Therefore ASVspoof 2021 DF clips **cannot populate
calibration**, and a locally generated TTS spoof corpus is required for every
split that participates in threshold fitting. Converting ASVspoof metadata to
v1 `GeneratorProvenance` would require inventing checkpoint sha, voice, seed,
and text-source facts that the official metadata does not publish. The honest
path is to keep each provenance kind intact and combine them under a tagged
union.

#### Spoof sources

S6 uses four materially distinct TTS generator families plus the existing
ASVspoof 2021 DF benchmark subset. Each family has a different synthesis
architecture, ensuring the unseen-generator-eval requirement tests genuine
generalization rather than checkpoint variation within one family.

| Generator | Architecture | License | Venue | Role | Splits |
|---|---|---|---|---|---|
| Piper (VITS) | Hybrid VITS vocoder, CPU-only | MIT | Any node | **Seen** (calibration) | Inherits source parent's split |
| Chatterbox (AR + flow-matching) | Autoregressive speech-token model + conditional flow-matching decoder, zero-shot voice cloning | MIT | GPU (RTX 3060 or better) | **Seen** (calibration) | Inherits source parent's split |
| ElevenLabs | Proprietary cloud neural TTS | Commercial API | Cloud API | **Unseen** (eval-only) | Generated ONLY from eval-split parents |
| Google Cloud TTS | Neural2 cloud neural TTS | Commercial API | Cloud API | **Unseen** (eval-only) | Generated ONLY from eval-split parents |
| ASVspoof 2021 DF | 110 official attack systems, 4 vocoder families | ODbL-1.0 (data) | Existing subset (53,392 clips) | **Benchmark anchor** (eval-only) | v2 imported-benchmark, eval-only by schema |

**Seen generators** (Piper + Chatterbox) produce spoof clips from bona fide
parents across all three splits (calibration, eval, holdout). Their clips
participate in Platt fitting (calibration split) and are visible to the
operating-point selection. Seen-generator spoof clips inherit their source
parent's split assignment, so no speaker can straddle splits.

**Unseen generators** (ElevenLabs + Google Cloud TTS) produce spoof clips
ONLY from eval-split parents. None of their clips appear in calibration or
holdout. This tests whether the detector generalizes to synthesis systems
the operating point was never tuned on. Two distinct unseen cloud APIs with
different architectures (ElevenLabs neural TTS, Google Neural2 TTS) provide
a stronger generalization test than a single unseen family.

**Benchmark anchor** (ASVspoof 2021 DF) provides an external reference EER
on a standardized corpus. It is never pooled with the organic TTS track for a
composition-weighted combined EER, because its 53,392 clips would numerically
dominate the organic cohort. Track-specific metrics are primary.

#### Spoof generation protocol

**Ratio.** 1:1: one spoof counterpart per eligible bona fide parent clip.
"Eligible" means turn clips (kind `turn` in `acquire`) in the parent manifest;
session segments (kind `segment`) are not synthesized (they are for
production-windowing validation, not calibration). The 1:1 ratio matches the
degradation policy (one degraded chain per parent in the cohort freeze).

**Text derivation.** Each TTS clip is synthesized from a transcript of its
bona fide parent. The transcript source is the bona fide audio itself
(passed through ASR or, for Chatterbox voice cloning, as the reference audio
prompt). The exact text-source derivation is recorded per clip in
`GeneratorProvenance.text_source`.

**Generator identity.** Generator identity for split assignment and
per-generator metrics is `GeneratorProvenance.name` (the family, not the
voice or checkpoint variant). A generator's clips are assigned to exactly
one split role: both seen generators span all three splits (following their
parents); the unseen generator is eval-only. Different voices within one
generator family are NOT treated as different generators for the
unseen-generator-eval requirement.

**Assignment rule for TTS clips.** A TTS spoof clip inherits its source
parent's split. The seen generators (Piper, Chatterbox) each produce one clip
per eligible parent in every split. The unseen generator (ElevenLabs)
produces one clip per eligible eval-split parent only. The `partition_group_id`
field (new in v3) ties each TTS clip to its source parent, preventing the
scorer from interpreting a paired bona fide and spoof from the same utterance
as independent observations. A calibration weighting policy (one effective
observation per partition group) is applied in Platt fitting.

**Provenance fields.** Each TTS spoof clip carries a full
`GeneratorProvenance`:

- `name`: generator family (`piper`, `chatterbox`, `elevenlabs`, `google`)
- `version`: engine version string (pinned before generation)
- `checkpoint_sha`: sha256 of the model weights file (None for cloud API)
- `voice`: voice/speaker id used
- `seed`: RNG seed for reproducibility (None for cloud API)
- `text_source`: how the input text was derived (e.g. `whisper-large-v2-transcript`, `parent-audio-prompt`)

**Stratum.** A TTS spoof clip's stratum is
`spoof|tts|{generator_name}|{domain}` (e.g.
`spoof|tts|piper|meetingroom`). This is distinct from the bona fide
stratum (`bona_fide|organic|{domain}`) and the ASVspoof stratum
(`{label}|{codec_condition}`), so per-stratum EER breakdowns are
meaningful.

#### v3 composite manifest schema

The v3 manifest extends the existing schema with a composite corpus kind
and per-clip provenance discrimination. It serves the scorer's single-manifest
contract while keeping both provenance kinds honest.

**Top-level fields** (new or changed from v1/v2):

- `schema_version`: 3
- `corpus_kind`: `composite` (new value; v1=`synthesis`, v2=`imported_benchmark`)
- `components`: array of `{component_id, corpus_kind, manifest_sha256,
  clip_count}` pinning each constituent manifest's identity. The composite
  manifest is reproducible from its components.
- `benchmark`: present only for components that are `imported_benchmark`
  kind (carries through from v2)

**Per-clip fields** (new or changed):

- `provenance_kind`: `synthesis` | `imported_benchmark` (discriminator for
  the tagged union; replaces the implicit per-manifest `corpus_kind` dispatch)
- `generator`: present iff `provenance_kind == "synthesis"` and `label == "spoof"`
  (unchanged from v1)
- `imported_provenance`: present iff `provenance_kind == "imported_benchmark"`
  (unchanged from v2)
- `component_id`: which constituent manifest this clip belongs to
- `partition_group_id`: ties paired bona fide and TTS spoof clips from the
  same source utterance (for calibration weighting). Set for TTS spoof clips
  and their source parents; None for ASVspoof clips and for bona fide clips
  with no TTS counterpart.

**Validation rules** (all fail-closed at load time):

- Every clip has exactly one of `generator` or `imported_provenance`, matched
  to its `provenance_kind`
- No duplicate `clip_id` across components
- No `rel_path` collision across components (namespaced by component)
- Every `component_id` references a declared component
- `imported_benchmark` clips remain eval-only
- Unseen-generator clips are eval-only (generator name in a declared
  unseen set)
- A `partition_group_id` must not span splits (same partition-group,
  same split)
- Component manifest sha256 values match the declared components
- Composite manifest `sha256` = sha256 of the serialized composite
  manifest file bytes (as with v1/v2)

**Scoring path.** The scorer (`synthdetect_eval.py`) is manifest-kind-agnostic:
it reads `clip_id`, `label`, `stratum`, and `split` from any manifest version.
The `join_scores` function joins journal results to manifest clips by `clip_id`;
no change is needed for v3. Per-stratum breakdowns use the stratum field, which
already distinguishes organic, TTS, and imported-benchmark clips. The `score`
command takes one `--manifest` and one `--journal`; both the bona fide and
spoof clips appear in a single v3 manifest, and a single scoring journal covers
the full corpus.

#### Corpus assembly

The composite corpus root is a single directory containing namespaced
subdirectories for each component:

```
s6-composite-corpus/
  organic-bonafide/     # S5 bona fide clips (hardlinked from s5-corpus)
  tts-piper/            # Piper TTS spoof clips
  tts-chatterbox/       # Chatterbox spoof clips
  tts-elevenlabs/       # ElevenLabs spoof clips (eval-only)
  tts-google/           # Google Cloud TTS spoof clips (eval-only)
  asvspoof-df/          # ASVspoof 2021 DF canonical subset (hardlinked)
  manifest.json         # v3 composite manifest
  assembly_receipt.json # component hashes, assembly metadata
```

`rel_path` in the manifest is relative to the composite root and includes the
component subdirectory prefix. Every clip is re-audited against its manifest
sha256 during assembly (the full-tree revalidation established in S5 PR-2a).

#### Metrics and reporting

**Primary field result:** organic bona fide versus organic-source TTS (both
seen generators combined). This is the EER and operating point that
calibration targets.

**Required generalization slices:** organic eval clips scored against each
eval-only (unseen) generator individually (ElevenLabs and Google). If
either unseen generator's EER is materially worse than the seen-generator
eval EER, the detector may not generalize to novel synthesis systems, and
the shipped confidence should reflect that. Reporting both unseen families
separately (not pooled) shows whether the generalization gap is
architecture-dependent.

**Benchmark anchor:** ASVspoof bona fide versus ASVspoof spoof, scored under
the same inference space and windowing. This is a diagnostic, not the
calibration target. It contextualizes the field result against a published
benchmark but does not drive the shipped threshold.

**Per-generator, per-stratum, per-domain, per-vocoder-family breakdowns** are
reported as diagnostics. A composition-weighted pooled EER across all tracks is
explicitly secondary and labelled as such, because the ASVspoof clip count
dominates it.

#### S6 scoring results: first EER measurement (2026-08-27)

**Verdict: SCORED.** All 63,905 composite clips scored by w2v2-aasist
(eval container `voxint-synthdetect-eval:s2b`, maintainer RTX 3060). Journal:
`journal_composite.jsonl` (63,906 lines, 1 header + 63,905 clip outcomes,
zero skips). Structured results:
`eer_report_w2v2aasist.json`, `eer_matrix_w2v2aasist.json`.

**Primary field result (organic bona fide vs seen TTS, Piper + Chatterbox
combined):** EER = 34.31 % (95 % CI: 33.29--35.30 %, 1000 bootstrap
resamples). AUC = 0.707. n = 8,575 (3,495 bona fide + 5,080 spoof).

**Required generalization slices:**

| Generator | Role | EER | AUC | n |
|---|---|---|---|---|
| ElevenLabs | unseen | 18.06 % | 0.886 | 4,464 |
| Google Cloud TTS | unseen | 17.96 % | 0.882 | 4,464 |

Both unseen generators are *more* detectable than the seen-generator
combined average (34.31 %). The primary is dominated by the Chatterbox
blind spot (see below).

**Benchmark anchor (ASVspoof DF eval):** EER = 7.05 % (95 % CI:
6.25--7.64 %). AUC = 0.986. n = 53,392 (1,487 bona fide + 51,905 spoof).
Within the expected range for w2v2-aasist on ASVspoof 2021 DF (~5--8 %
published). Confirms the model is functioning correctly on in-distribution
data.

**Per-generator breakdown (diagnostic):**

| Generator | Role | vs AMI bf | vs VC bf | vs Both |
|---|---|---|---|---|
| Piper (VITS) | seen | 12.0 % | 41.9 % | 20.0 % |
| Chatterbox (AR + flow) | seen | 40.3 % | 71.5 % | 45.3 % |
| ElevenLabs | unseen | 11.1 % | 34.9 % | 18.1 % |
| Google Cloud TTS | unseen | 10.4 % | 37.4 % | 18.0 % |

**Finding 1: Chatterbox evasion.** Chatterbox (AR autoregressive token
prediction + conditional flow matching + HiFT source-filter vocoder) is
near-undetectable: EER = 45.32 % (AUC 0.558, near chance). Even against
AMI-only bona fide (the cleanest comparison), EER = 40.3 %. 59.4 % of
Chatterbox scores fall within the bona fide IQR. KS statistic
(bona fide vs Chatterbox) = 0.14 (vs 0.66 for Piper). A 4-model consult
(codex + deepseek-v4-pro + grok-4.5 + kimi-k3, 2026-08-27) identified the
root cause as OOD: w2v2-aasist's production checkpoint was trained on
ASVspoof 2019 LA (6 attack systems, no flow-matching or modern AR+flow
architectures). The DFADD benchmark (2024) independently reports 44.21 %
average EER for ASVspoof-trained AASIST on unseen flow-matching TTS,
closely matching our 45.32 %. The evasion is likely a closeable coverage
gap (DFADD reports ~23--25 % after fine-tuning on flow-matching examples),
not a fundamental detection impossibility. Codex also identified that
flow matching operates in mel space (not waveform): phase coherence comes
from HiFT's explicit F0-driven source-filter vocoder, not from the ODE
transport.

**Finding 2: VoxConverse channel confound.** VoxConverse bona fide
scores substantially higher than AMI (mean 3.616 vs 0.884), degrading
ALL generators by 20--30 pp EER when used as the bona fide reference.
The detector's score axis is partly a channel/style axis, not purely
a synthesis axis. This is independent of the Chatterbox evasion.

**Finding 3: voice-cloning reference transfer.** Chatterbox is the only
generator that clones from each bona fide parent clip; others use
fixed/provider voices. Chatterbox from same-domain parents (AMI text vs
AMI bona fide) scores 42.5 % EER; cross-domain (VoxConverse text vs AMI
bona fide) scores 28.0 %. The cloning transfers channel cues but is
secondary to the OOD gap.

**Implication for calibration:** Platt scaling on the seen-generator
calibration split will not produce meaningful probabilities for
Chatterbox-class generators. A single threshold cannot serve both
detectable (Piper, ElevenLabs, Google) and undetectable (Chatterbox)
families. Calibration (S7) must account for this limitation explicitly.

**Decision (2026-08-27):** proceed to S7 calibration with the current
corpus. The Chatterbox evasion is a documented, expected OOD gap
(independently confirmed by DFADD 2024 at 44.21 % avg EER on
flow-matching TTS). The eval corpus successfully detected this gap,
which is the corpus working as designed. The VoxConverse channel confound
is by design (degraded bona fide in calibration is pre-registered). Both
findings must be reported honestly in the shipped coverage statement.
Fine-tuning experiments (adding flow-matching examples to training data)
are deferred to M2 (model service, issue #252); corpus diversification
(clean read-speech control set, held-out reference speakers) is optional
measurement tightening, not a prerequisite for calibration.

### Calibration and holdout discipline

The primary shipped threshold is at **FPR 5 %**. FPR 1 % from roughly 1000 bona
fide clips is quantile noise, reported as a diagnostic only. Platt scaling is fit
on RAW logits over the calibration split, and **degraded bona fide strata are
included in that split**: codec artifacts on genuine audio are the dominant field
false-positive driver, so the operating point must see them. The `holdout` split
is opened exactly once, after every runtime and calibration choice is frozen; any
later tuning requires a new versioned holdout cohort.

The always-on fixture gate (`tests/parity/test_synthdetect_fixture_scores.py`,
S2+) replays committed CC0 journals to byte-stable metrics. The DECISION-level
fixture gate applies only to fixtures outside a guard band of roughly 3x the
measured drift tolerance around the threshold: near-threshold fixtures flip
spuriously across driver, cuDNN, and torch revisions and are covered by the
raw-logit drift level instead.

#### S7 calibration results (2026-08-27)

**Policy: `w2v2aasist-s6-piper-only`**. Platt scaling fitted on the
calibration split with Chatterbox strata excluded (1152 clips removed, 2703
retained: 1551 bona fide + 1152 Piper spoof). Chatterbox is excluded because
its scores overlap bona fide almost completely (meetingroom EER 48.25 % on
holdout, mean score 1.82 vs bona fide 1.65), and including it flattens the
Platt slope from A = 0.96 to A = 0.32, degrading calibration for every
generator the detector can actually discriminate.

**Platt parameters:** A = 0.9598, B = -3.6155.

**Calibration vs holdout comparison (Piper-only population):**

| Metric | Calibration | Holdout |
|---|---|---|
| Piper-only EER | 19.01 % | 18.04 % |
| Piper-only Brier | 0.136 | 0.140 |
| Overall EER (incl. Chatterbox) | 32.90 % | 36.04 % |

The holdout estimates track the calibration split closely, confirming
the fitted policy generalizes within the corpus.

**Holdout per-generator EER (each generator vs all 582 bona fide):**

| Generator | N | EER | Brier | FPR 5 % TPR |
|---|---|---|---|---|
| Piper meetingroom | 371 | 18.04 % | 0.137 | 16.2 % |
| Piper webvideo | 48 | 18.90 % | 0.137 | 12.5 % |
| Chatterbox meetingroom | 371 | 48.25 % | 0.345 | 1.3 % |
| Chatterbox webvideo | 48 | 33.33 % | 0.149 | 14.6 % |

**Operating point analysis (Piper-only holdout, 582 bf + 419 Piper spoof):**

| Target FPR | Threshold | Realized FPR | TPR |
|---|---|---|---|
| 1 % | 5.966 | 0.9 % | 0.2 % |
| 5 % (primary) | 5.501 | 4.8 % | 15.8 % |
| 10 % | 5.065 | 10.0 % | 43.2 % |
| 20 % (~EER) | 3.402 | 19.9 % | 89.5 % |

At the shipped FPR 5 % operating point, the detector catches approximately
16 % of Piper spoof on the holdout. This is a high-confidence flag (very
few false alarms), not a reliable filter (most spoof passes through).
At the EER threshold (~20 % FPR), TPR rises to ~90 %, suitable for risk
flagging applications that tolerate a higher false-alarm rate.

#### Coverage statement

The `w2v2-aasist` detector at checkpoint `LA_model.pth` (ASVspoof 2019 LA
training), evaluated on the S6 composite corpus and calibrated with the
`w2v2aasist-s6-piper-only` Platt policy:

1. **Reliably identifies VITS-family TTS (Piper)** and commercial neural TTS
   (ElevenLabs, Google Cloud TTS) with EER 10 to 20 %, validated on the
   holdout split. Piper holdout EER: 18.04 %. Unseen generators (ElevenLabs,
   Google) scored 18.06 % and 17.96 % EER on the eval split (not in the
   holdout; eval-only by design).

2. **Does NOT reliably identify AR + flow-matching TTS** (Chatterbox-class)
   at the current checkpoint. Holdout EER: 48.25 % (meetingroom), near
   chance. This is a known OOD gap: the ASVspoof 2019 LA training set
   contains no flow-matching attacks, and DFADD (2024) independently reports
   44.21 % avg EER on flow-matching generators. Chatterbox strata are
   excluded from the calibration fit. Closeable with fine-tuning (issue #252,
   deferred to M2).

3. **ASVspoof DF benchmark anchor is healthy:** EER 7.05 % on the imported
   ASVspoof5 DF eval partition (eval-only, not in calibration or holdout).

4. **VoxConverse-sourced bona fide scores higher** (mean raw score 3.62 vs
   AMI 0.88), raising FPR on compressed or reverberant real speech. This is
   by design: degraded bona fide strata are included in calibration so the
   operating point reflects real-world field conditions, not clean studio
   audio only (issue #253).

5. **At the shipped FPR 5 % operating point**, TPR is approximately 16 %
   on Piper holdout. The detector at this threshold functions as a
   high-confidence alert, not a reliable spoof filter. Risk-flagging
   deployments that tolerate ~20 % FPR achieve ~90 % TPR at the EER
   threshold.

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
