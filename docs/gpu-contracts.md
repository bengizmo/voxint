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
  - **titanet only** additionally carries `embedding_space` and
    `window_cap_seconds`. They report the space id and effective per-window cap
    used by the loaded service. Both fields are additive and optional;
    consumers tolerate their absence on older services.
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
weights), 192-dim. Embedding space id: **`titanet-large-v2`**, persisted with
every vector; changing the model weights *or any parameter of the space
definition below* means a new space id, never a silent swap.

### The `titanet-large-v2` space definition (normative)

`titanet-large-v2` supersedes `titanet-large-v1` on 2026-09-02. It adds the
step 3a capped sub-windows and step 9 pooling defined below. v1 and v2 vectors
must never be compared or mixed under one space id.

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
3. **Skip gates** (checked in this order, before any normalization, once on
   the whole requested slice):
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
   outcomes. A successful result reports this whole-window `snr_db`.
3a. **Capped sub-windows**: let `cap = 30.0` s (the `WINDOW_CAP_SECONDS`
   constant in `preprocess.py`) and `cap_samples = int(cap * 16000) = 480000`.
   The cap is a fixed parameter of this space definition. A gated slice longer than
   `cap_samples` is cut into contiguous, non-overlapping pieces
   `[i, min(i + cap_samples, n))` for `i = 0, cap_samples, 2 * cap_samples, ...`.
   A final piece shorter than 1.0 s (16000 samples) is dropped. The first piece
   is always a full `cap_samples`, so at least one piece survives and this step
   adds no skip reason. The cap applies to every engine, so the parity gate
   compares identical piece boundaries.
4. **Noise reduction**: stationary spectral gating (`noisereduce`,
   `prop_decrease=0.75`), applied per piece.
5. **Loudness**: integrated-loudness normalization to −16 LUFS (BS.1770;
   skipped only when the meter returns non-finite loudness), applied per piece.
6. **Peak**: peak normalization to 0.95, applied per piece.
7. **Model**: TitaNet-Large forward pass on each processed 16 kHz mono piece;
   mel-spectrogram front-end per the pinned NeMo 1.22 preprocessor config
   (dither 1e-5, per-feature normalization, log with zero-guard,
   n_fft/hop/window per the checkpoint's config). Implementations that do
   not embed NeMo must reproduce this front-end and prove it at the mel level.
8. **Piece output**: L2 normalization of each 192-dim vector.
9. **Window output**: when more than one piece survives, average the per-piece
   unit vectors elementwise (equal weight per piece, not per sample) and
   L2-normalize the result again. A single piece returns its vector unchanged.
   Every window at or below the cap therefore remains byte-identical to the
   v1 chain.

The reference implementation of steps 1–6 is
`services/titanet/app/preprocess.py` (shared by every engine), including
`subwindow_bounds` for step 3a. Step 7's reference is the pinned NeMo
checkpoint. `TitanetEmbedderBase._embed_one_window` implements the per-piece
steps 4–8, and `TitanetEmbedderBase.embed_windows` implements step 9 in
`services/titanet/app/embedding.py`.

### Equivalence policy (measured, not bit-identical)

Bit-identity is not the bar. It is already false across CUDA hardware
generations. An alternative implementation may keep `titanet-large-v2` **iff**
it passes the 3-level parity gate (`tests/parity/test_titanet_onnx.py`)
against reference outputs produced by the NeMo/CUDA implementation
(fixtures: `tests/parity/fixtures/`):

- **mel level**: the reimplemented front-end matches the NeMo-internal
  mel features within tolerance on the golden corpus;
- **vector level**: per-window cosine similarity above the ratcheted
  threshold (≥ 0.999 baseline), identical `skip_reason` per window, `snr_db`
  within ±0.5 dB, on amd64 and arm64. The golden corpus includes windows at
  the cap through the direct path and above the cap through the pooled path,
  so this level measures step 9;
- **decision level**: replaying voxint's matching gates (0.60/0.70
  thresholds) on labeled same/different-speaker pairs produces no
  merge/split changes, no threshold crossings, and stable top-1/top-2
  margins, within percentile and worst-case tolerances recorded in the
  harness.

A failed gate means a new space id (`titanet-large-v3`) plus a re-embed
migration, never shipping a drifted implementation under the old id.

#### Verdict: ONNX Runtime engine PASS for `titanet-large-v2` (2026-09-02, amd64)

The ONNX engine keeps `titanet-large-v2`. Measured on a maintainer workstation
(amd64, onnxruntime CPU EP) against 2026-09-02 CUDA references (RTX 3090,
NeMo 1.22.0). Full golden corpus: 99 embedded / 114 windows, 465 labeled
pairs, 7 long windows (30 s cap exact, 30.5 s runt, 31 s, 60 s, 90 s,
175 s, 220 s pooled).

| Level | Measured | Ratcheted gate |
|---|---|---|
| mel max abs diff | 1.76e-4 | 1e-3 |
| vector cosine (min / p50) | 0.999997 / 0.999999 | ≥ 0.9995 |
| `skip_reason` mismatches | 0 | 0 |
| `snr_db` max diff | 0.0 dB | ±0.5 dB |
| pair-cosine drift (max) | 4.6e-4 | ≤ 2e-3 |
| 0.60/0.70 gate crossings | 0 | 0 |
| top-1 flips / margin drift (max) | 0 / 3.53e-4 | 0 / ≤ 2e-3 |
| repeat determinism (max abs diff) | 0.0 | 0.0 |

The 7 long windows exercise the 30 s cap and pooled sub-window path
introduced in v2. `long_cap_runt` (30.5 s, 0.5 s tail dropped) produces
the same embedding as `long_cap_exact` (30 s) on both engines, confirming
the runt-discard path. Pooled windows (31 s through 220 s) fall within the
same cosine floor as the single-window corpus.

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
  "embedding_space": "titanet-large-v2",
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
- Per-window processing follows the normative `titanet-large-v2` space
  definition above; reference code is in `services/titanet/app/preprocess.py`
  and `services/titanet/app/embedding.py`.

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

#### Verdict: v0.32.0, Gates A/R/M carry, Gate E browser lane PASS (2026-09-02)

v0.32.0 ships the UX audit remediation epic (#369, seventeen slices), the
media library search and status filter (#380), the viewer role (#363), the
evidence pack and bundled quote export (#331, #281), and GPU resource awareness
in the installer. Releases v0.28.0 through v0.31.0 recorded no verdict block
here; this entry resumes the record.

`git diff v0.31.0..HEAD -- services/` is empty and no metal-lane path
(`scripts/metal/`, `tests/parity/`, `metal-lane.yml`) changed, so the three
inference model services are byte-identical to v0.31.0 and **Gates A (CUDA),
R (ROCm), and M (Metal) carry** their standing verdicts.

Gate E scope: the pipeline-aware diff is non-empty, but every changed path is
under `src/voxint/api/` (templates, routers, presentation helpers), `frontend/`,
and one additive migration (`0057_viewer_role`); `src/voxint/pipeline/`,
`src/voxint/clients/`, `src/voxint/enrichment/`, and `tests/e2e/` are unchanged.
That triggers the browser acceptance lane and not the pipeline lane.

- **Browser acceptance lane** (maintainer hardware, serial, headless Chrome via
  Python Playwright over `tools/e2e_browser_lifecycle.py`, the three CPU model
  services running so readiness probes were live): the full `voxint-e2e-review`
  sequence on the seeded 5-segment run. Two uncertain chips and one peaks fetch
  on load; verify-and-advance (one `POST /verify` 200, counter 0 to 1 of 5,
  cursor to the next unverified segment) followed by replay with an
  instrumented `play()`; skip with no network; click-to-edit; the discard
  warning (warned `v` fires nothing, second `v` verifies); edit and save
  (one `POST /text` 200); keymap suppression on a focused select; the
  shortcuts dialog opened by key and by button and dismissed by Escape, close
  control, and backdrop, with `v` suppressed behind it; domain-pack provenance
  (chip present on segment 0 and absent on segment 2, raw transcript reveal,
  client-only reset-to-raw, reconciliation panel "1 of 2 applied, 1 never
  fired"). 39 assertions green, no page errors. **RECONCILE PASS** (2 of 5
  verified, one correction matched).
- **Labelled rail and theme control (#384)**: the three rail tiers (168px
  labelled, 52px icon-only with tooltips, narrow-screen disclosure), the
  70rem boundary, theme cycle order and label/tooltip/aria sync, localStorage
  clearing on System, and Settings radio agreement all green on the release
  content. The low-data widget thresholds and island hydration (#385) were
  browser-verified at the landing commit `af60c45`, which is the release
  content minus version pins, changelog, docs, and screenshots.

#### Verdict: v0.34.0, Gates A/R/M carry, Gate E skipped (2026-09-04)

v0.34.0 ships speaker roster tools (dedup CLI #432, reconcile CLI #430),
auto-enroll near-miss evidence (#434), restart preflight checks (#422),
feature flag dependency nesting (#406), operator invariant feedback (#404),
and watch-folder batch cap with saturation-aware retry (#418).

`git diff v0.33.0..v0.34.0 -- services/` is **empty**. All three inference
model services (whisper, pyannote, titanet) and the synthdetect container are
**byte-identical** to v0.33.0.

- **Gate A (CUDA titanet regression)**: services unchanged. **Carries** the
  v0.33.0 PASS verdict (99/99 cosine 1.000000 against `v2-dev` references).
- **Gate R (ROCm)**: services unchanged. **Carries** the standing ROCm verdict.
- **Gate M (Metal)**: no metal-lane paths changed. **Carries** the standing
  Metal verdict.
- **Gate E (whole-pipeline E2E)**: the pipeline-aware diff is non-empty
  (`src/voxint/api/routers/editor.py`, `src/voxint/api/templates/editor/`,
  `src/voxint/db/models.py`). The changes are restart preflight rendering and
  the `auto_enroll_evidence` table, neither of which alter pipeline data flow
  or review-console island behavior. The pipeline lane requires ROCm hardware
  (test hardcodes `device: rocm`); the browser acceptance lane was not re-run.
  **Skipped** (low blast radius, no island or pipeline code changes).

Gates A/R/M carried on byte-identical services; Gate E skipped (editor template
and schema additions only, no functional pipeline or island change).

#### Verdict: v0.33.0, Gate A PASS, Gate R/M carry, Gate E deferred (2026-09-03)

v0.33.0 ships the TitaNet v2 window-cap and embedding-space bump (#424), the
`voxint speakers re-embed` migration tool (#425), and the auto-enroll
standard-gate fix for cross-episode speaker matching (#433, #431).

`git diff v0.32.0..HEAD -- services/` is **non-empty**: `services/titanet/`
changed (window-cap embedding, preprocess refactor, embedding-space bump to
`titanet-large-v2`). Whisper and pyannote service code are unchanged.

- **Gate A (CUDA titanet regression): PASS.** Formal re-run completed
  2026-09-03 against the tagged `ghcr.io/bengizmo/voxint-titanet:0.33.0`
  image on maintainer CUDA hardware (RTX 3090). 99/114 windows
  embedded (15 skipped by SNR/duration gates, matching committed corpus
  expectations). All 99 embeddings are cosine 1.000000 against the
  committed `v2-dev` references at `bd7d7b2`. References updated in-place
  to record the `0.33.0` tag (embeddings byte-identical, only metadata
  changed). Embedding space: `titanet-large-v2`, dim 192.
- **Gate R (ROCm)**: whisper and pyannote services unchanged from v0.32.0.
  **Carries** the standing ROCm verdict.
- **Gate M (Metal)**: no metal-lane paths changed. **Carries** the standing
  Metal verdict.
- **Gate E (whole-pipeline E2E)**: the pipeline-aware diff is non-empty
  (`services/titanet/`, `tests/e2e/`). The pipeline lane requires ROCm
  hardware (test hardcodes `device: rocm`). The browser acceptance lane is
  unchanged from v0.32.0 (no `frontend/` or console-path changes). **Pipeline
  lane deferred** (no AMD GPU available); browser lane carries from v0.32.0.

Gates A PASS, R/M carried on byte-identical services; Gate E browser lane
carries from v0.32.0.

#### Verdict: v0.27.0, Gates A/R/M carry, Gate E browser lane PASS (2026-08-27)

v0.27.0 ships Ops Console R4 (speakers overview + detail refresh, #213) and
R5 (project detail refresh, #214), completing the visual-refresh epic #205.
All changes are Jinja templates, `chip_semantics.py`, `base.html`, and
integration tests. No pipeline, service, or inference code changed.

`git diff v0.26.0..HEAD -- services/` is empty. The three inference model
services (whisper, pyannote, titanet) and the synthdetect eval container are
**byte-identical** to v0.26.0. **Gates A (CUDA), R (ROCm), and M (Metal)
carry** their v0.24.0 verdicts (unchanged since v0.26.0).

Gate E scope: R4/R5 changed speaker overview/roster/profile and project
detail templates (observable console behavior), triggering the browser
acceptance lane. The pipeline lane is not triggered (no pipeline/service
changes).

- **Browser acceptance lane** (maintainer hardware, serial): full
  `voxint-e2e-review` skill run. Seeded 5-segment run, exercised
  verify-and-advance, skip, replay (instrumented play()), click-to-edit,
  discard warning, edit+save, keymap suppression, cheat-sheet modal (open
  via key/button, dismiss via Escape/close/backdrop, v-suppression behind
  modal), domain-pack correction provenance (chip present/absent, raw
  transcript, reconciliation panel with 1 applied + 1 never fired), waveform
  strip (canvas, playhead, cursor-index sync, single peaks fetch). All
  assertions green. RECONCILE PASS (2/5 verified, 1 correction match).

Gates A/R/M carried on byte-identical services; Gate E browser lane green.
Clear to tag v0.27.0.

#### Verdict: v0.26.0, Gates A/R/M carry, Gate E re-run fresh (pipeline PASS, browser deferred) (2026-08-27)

0.26.0 is the largest release since initial: 272 commits since v0.24.0, covering
the Console 2.0 epic (P0a/P0b through P2c, P4, P6a/P6b, P7, Jobs #160),
Ops Console visual refresh (V1-V3, COPY, R1/R2, R6), Synthdetect M1 S1-S5,
plugin framework (#137/#138, dormant), audio-clip extraction (#88), navigable
outline (#87), translation (#133), and CI parallelization (#187).

`git diff v0.24.0..HEAD -- services/` shows 4 files changed, all in
`services/synthdetect/` (the eval container — a maintainer-only scoring tool,
not an inference service). The three inference model services (whisper,
pyannote, titanet) and their Dockerfiles are **byte-identical** to v0.24.0.
The synthdetect eval container is separately qualified (Gates 1-2 ratified
above, `w2v2-aasist-df` checkpoint frozen and GPU-qualified) and does not
participate in the pipeline inference path. No metal-lane path changed.
**Gates A (CUDA), R (ROCm), and M (Metal) carry** their v0.24.0 verdicts.

Live service health confirmed on release day (2026-08-27):
- **ROCm host**: whisper `device=rocm model=large-v2 engine=faster-whisper`,
  pyannote `device=cpu model=speaker-diarization-3.1`, titanet `device=cpu
  engine=onnxruntime` — all `model_loaded=true`, `contract_version=v1`.
- **CUDA host**: whisper `device=cuda model=large-v2`, pyannote
  `device=cuda model=speaker-diarization-3.1`, titanet `device=cuda
  engine=nemo` — all `model_loaded=true`, `contract_version=v1`.

The Gate E pipeline-aware scope is massive (105 files changed, +17K/-7K in
`src/voxint/{api,db,enrichment,media,pipeline}`, `frontend/`, `tests/`), so
**Gate E re-ran fresh** on maintainer hardware before tagging:

- **Pipeline lane** (AMD host, serial): `tests/e2e/test_real_pipeline.py`
  against live model services (whisper ROCm large-v2, pyannote cpu
  diarization-3.1, titanet cpu onnxruntime, all `model_loaded=true`) on a
  disposable database — **2 passed, exit 0, 113s**, no service restarts.
- **Real-LLM enrichment sub-lane**: SKIP (documented). `LLM_ENABLED=true` is
  configured on the reporting host but `ENRICHMENT_RUN_ASSETS_ENABLED` is not
  set, so the sub-lane skips by design (it requires both flags). This is the
  optional sub-lane; the skip is a legitimate unconfigured state, not a masked
  failure.
- **Browser review lane**: deferred to pre-tag; will run via `voxint-e2e-review`
  on maintainer hardware.

**Security audit glance** (not a new full audit): the standing audit at
`139ebe3` (2026-08-18) has 7 open findings (E3-E6 research/enrichment, F5
supply chain cache, M1-M2 media/normalize.py ffmpeg hardening). The v0.26.0
diff touches `media/executor.py` and `api/media_query.py` (P2c operations) but
not `normalize.py`, `research/agent.py`, or `release.yml` — the open findings
are neither worsened nor intersected. None are release-blocking.

PR #228 (R3 media overview refresh, `f94ba74`) merged after the pipeline lane
ran; its changes are presentation-only (`media.html` template, `media_query.py`
query helpers, test assertions) and do not touch the pipeline, services, or
inference paths. The Gate E pipeline result at `4f6cc1a` covers the same
inference code; the release commit adds the CHANGELOG stamp and this verdict.

Gates A/R/M carried on byte-identical services; Gate E pipeline lane green
fresh. Clear to tag v0.26.0.

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

## Synthetic-speech detection (synthdetect)

Synthdetect is Voxint's audio deepfake detection capability (issue #144). It
uses a fine-tuned wav2vec2-AASIST classifier (`w2v2-aasist`, MIT-licensed) to
score audio windows as bona fide or synthetic, with Platt-calibrated risk
scores. The detector is qualified on an RTX 3060 (SM 8.6) with bit-exact
determinism across cold starts.

### Eval tooling

The evaluation infrastructure (46 files, 21k lines) has been extracted to a
private repository. This includes the inference runner, scoring harness,
corpus management, fine-tuning pipeline, and all associated tests and reports.
Full eval-gate protocols, dated verdicts, and reproduction evidence live
there. The extraction has zero runtime dependencies on voxint app code.

### Model limitations and coverage

**Chatterbox evasion (#252).** The baseline w2v2-aasist checkpoint achieves
only 37.77% EER on Chatterbox-generated speech (eval subset). M2 fine-tuning
(#268) reduced this to 25.90% EER, a 11.87 pp improvement but still well
above the sub-5% range for other generators (Piper 8.04%, ElevenLabs 11.20%,
Google 10.28%). Progressive XLS-R unfreezing (Phase 3) is available if
further improvement is needed.

**VoxConverse channel confound (#253).** The VoxConverse bona-fide corpus
occupies a distinct score region from AMI bona-fide speech, inflating false
positive rates when both are mixed. At the EER threshold (2.66), AMI BF FPR
is 1.1% while the blended eval BF FPR is 23.6% (driven primarily by
ASVspoof DF anchor clips at 42.4%). Calibration targets in-domain corpora
(AMI + VoxConverse), not the ASVspoof anchor.

### M2 fine-tuning outcomes

Threshold recalibration after Chatterbox fine-tuning (seed 0, dev partition):
EER threshold 2.66, dev BF FPR 1.1%, holdout BF FPR 0.86%. Platt parameters:
A = 4.28, B = -11.42; Brier score 0.0066.

### M3 eval protocol (#252)

Pre-registration for the Chatterbox improvement effort (GitHub #252). This
section is binding: no training may begin until every artifact hash, baseline
value, and gate threshold below is committed. Protocol version 1.4
(v1.1, 2026-08-29: two-sample acceptance probe band, pinned Brier population,
explicit paired-delta percentile; v1.2, 2026-08-29: AMI speaker identity
repair and re-baselining rule, domain-matched bonafide comparison
populations; v1.3, 2026-08-30: eval-only partition sanitization, unseen-FM
slice disclosures, pre-registered slice-regeneration trigger; v1.4,
2026-08-30: single-clip integrity exclusion after a fail-closed hash
mismatch during the aborted first M2 scoring run). Designed via
4-model consult (codex, deepseek-v4-pro, grok-4.5, kimi-k3); the v1.2 repair
design, the v1.3 sanitization, and the v1.4 exclusion were independently
reviewed.

> ⚠ The re-baselining pass ran on 2026-08-30: the frozen M2 checkpoint was
> re-scored on the frozen 68,015-clip manifest with the locked evaluator
> revision, every to-be-determined value in this section is now filled,
> and the provisional
> pre-repair baselines are replaced. Gate limits kept their pre-registered
> formulas (baseline plus fixed margin); only the measured M2 inputs changed.
> Every change is itemized in "M2 re-baselining measurement (2026-08-30)"
> below.

#### v1.2 amendment: AMI speaker identity repair (2026-08-29)

The corpus tooling namespaced AMI speakers per meeting, but AMI's annotation
metadata assigns corpus-global person keys and one person attends several
meetings. Five of the 19 canonical AMI persons in the composite corpus
therefore straddled splits (three calibration/eval, two calibration/holdout),
covering 63% of AMI clips and roughly 80% of the AMI eval clips. Because M2
fine-tunes on the calibration split, the affected eval clips were scored by a
model trained on those voices. The pre-repair M2 eval numbers on AMI strata
are optimistic and are not valid gate baselines.

The repair migrates the existing composite manifest: audio bytes, clip ids,
and provenance are unchanged, and only speaker identities and split
assignments are rewritten.

- AMI speaker ids are canonicalized to the person key; synthetic-clone
  variants keep their generator suffix on the canonical base. VoxConverse ids
  stay recording-scoped (their labels are genuinely recording-local).
- A canonical person whose clips agree on one split keeps it. A person
  touching calibration in any meeting moves wholly to calibration: M2 already
  trained on that voice, so it can never again serve as clean eval or holdout
  material for M2-baseline comparisons. A person straddling only eval and
  holdout moves wholly to holdout.
- There is deliberately no fresh hash-rank re-assignment. Re-ranking could
  move other M2-trained calibration voices into eval, recreating the leakage
  elsewhere. The repaired eval is therefore a leakage-free interim cohort;
  the challenge partition remains the unbiased cohort for ship decisions.
- The pre-repair manifest is archived beside the repaired one, and the repair
  tool emits a JSON audit (per-person actions, before/after counts, input and
  output hashes). The repaired manifest hash becomes the binding one in the
  frozen-artifacts table.
- Split assignment for all future corpora (the challenge partition included)
  uses canonical person identity from the start, and challenge speaker
  freshness is defined against canonical identity.

Re-baselining rule (ordering is binding): first freeze the repaired manifest
and its audit; then re-score the frozen M2 checkpoint on the repaired splits
to fill every to-be-determined value and replace the provisional baselines;
then freeze the
updated gate matrix. Only after the gate matrix is frozen may any candidate
model be scored on the repaired eval. Gate limits keep their pre-registered
margins (for example, per-generator non-regression stays baseline plus
2.00pp); only the measured M2 inputs change. The pre-repair baselines remain
in this document's history as invalid, with this amendment as the reason.

The repaired AMI eval cohort is small (about 690 clips) until the challenge
partition lands; per-generator EER confidence intervals on AMI strata must be
read accordingly, and the effective component counts reported beside every
number are the honest sample-size signal.

#### v1.3 amendment: eval-only partition sanitization and unseen-FM slice disclosures (2026-08-30)

**Eval-only partition sanitization (data-integrity correction).** Four
generators are registered eval-only in the corpus assembler (ElevenLabs,
Google TTS, Matcha-TTS, F5-TTS): their clips may exist only in the eval
split. ElevenLabs and Google TTS are core gate generators that no model
trains or calibrates on; Matcha-TTS and F5-TTS are the unseen-FM
diagnostics. The v1.2 identity repair moved calibration-contacted persons
wholly to calibration, and three of those persons were among the identities
used to generate the eval-only slices. The repair therefore carried 2,140
eval-only clips (535 per generator, all AMI-sourced) into the calibration
split. Those clips have no permitted use anywhere: not Platt fitting
(eval-only rule), not candidate training (the no-train guarantee), and not
eval or holdout (M2 trained on those voices).

A second migration removes them, by predicate rather than by enumerated
person list: every clip whose generator is registered eval-only and whose
post-repair split is not `eval` is dropped from the composite manifest, and
the affected component clip counts are updated so the manifest stays
internally truthful. Audio bytes and the sha-pinned source slice manifests
are untouched; the pre-sanitization manifest is archived beside the
sanitized one, and the migration emits a JSON audit listing every dropped
clip id and sha256 with per-component, per-generator, and per-person
counts. The sanitized manifest hash becomes the binding one in the
frozen-artifacts table. Like the v1.2 repair, there is no fresh hash-rank
re-assignment.

**Surviving eval-only slice sizes (disclosure).** After sanitization each
eval-only generator retains 125 AMI eval clips from 4 canonical persons,
plus its VoxConverse eval slice (ElevenLabs 309, Google TTS 309, Matcha-TTS
309, F5-TTS 302 clips). The AMI side of every eval-only generator is
underpowered on its own: per-domain (AMI-only or VoxConverse-only) numbers
for these generators are exploratory diagnostics, never gates. The binding
metrics remain the pooled in-domain EERs already defined in this protocol,
with grouped-bootstrap CIs and effective component counts reported beside
every number.

**Pre-registered regeneration trigger.** Whether the surviving slices are
adequate is decided by precision, not by looking at the numbers. At the M2
re-baselining pass: if the grouped-bootstrap 95% CI half-width of any
eval-only generator's pooled eval EER exceeds 5.00pp, that generator's
slice must be expanded or regenerated on clean post-repair eval identities
(as a new corpus version with pinned generator versions and seeds) before
any gate that consumes the metric may bind. For ElevenLabs and Google TTS
that means their non-regression gates; for Matcha-TTS the P2+ unseen-FM
gate. Phase 1 unseen-FM reporting stays diagnostic-only either way. An
under-precision baseline is reported with its CI and marked non-binding
until the slice is regenerated and the baseline re-measured.

**F5-TTS reference-fallback disclosure (post-data amendment).** The F5-TTS
slice generator selects one same-speaker reference clip per parent in
deterministic seeded digest order and, when synthesis yields a non-finite
output, retries with the next-ranked reference for up to 5 attempts; the
first finite output wins. On the AMI slice, 41 of 660 parents (6.2%)
required at least one fallback, which exceeded the roughly 2% rate at which
the pre-generation consult said to stop and reassess; the VoxConverse slice
required none (0 of 309 eligible parents; 7 parents were skipped for lacking
any eligible reference, and 3 AMI parents for lacking a transcript, so the
VoxConverse slice is 302 clips). The maintainer decision, recorded here
before any M2 scoring of these slices, is to accept the slice with full
disclosure rather than regenerate: fallbacks substitute a different
same-speaker reference, deterministically, and every substitution is
logged. The per-domain generation receipts (checkpoint and vocoder sha256s,
synthesis parameters, environment pins, selection seed, reference-map
hashes) and the per-parent fallback log are retained beside the slice
manifests, and the used-reference maps are hash-bound in the receipts. If
F5-TTS scoring behaves anomalously relative to the other eval-only slices,
the fallback log is the first place to look.

#### v1.4 amendment: single-clip integrity exclusion (2026-08-30)

The first authoritative M2 scoring run aborted fail-closed at clip 57,126 of
68,016: the canonical-PCM sha256 of one calibration-split Piper spoof clip,
`ami-EN2002d-MEE073-turn-2345920-2632160--piper`, no longer matched its
manifest row (expected `9f9da460...`, observed `6b5c6598...`; two independent
decoders reproduce the observed digest). A full sweep of all 68,016 clips
found exactly this one mismatch. The clip's duration matches its manifest row
to the sample and the audio is intact-sounding speech, so the bytes appear to
be the described synthesis with silently altered content; silent storage
corruption and a generation-time hash-versus-write discrepancy cannot be
distinguished from the surviving evidence. The recorded digest is consistent
across every archived manifest generation back to the original Piper
component manifest, so the v1.2 and v1.3 migrations are ruled out as causes.

The original bytes are unrecoverable: every on-disk path to the file is a
hardlink to a single inode, no backup exists, and Piper synthesis is
non-deterministic (verified by regeneration), so the clip cannot be
reproduced. Adopting the unexplained disk bytes into the manifest would
launder an integrity failure into the frozen record, and a regenerated
replacement would be a new, adaptively created clip. The clip is therefore
structurally dropped, the most conservative of the three options.

- The drop is a guarded migration in the evaluator repo
  (`synthdetect_integrity_drop.py`): it verifies the frozen input-manifest
  file sha, requires the on-disk hash to actually mismatch (it refuses to
  drop an intact clip), removes exactly the one named row, decrements only
  that component's clip count, revalidates the output row for row, and emits
  a forensic audit (both digests, whole-file sha, file metadata, before and
  after counts). The pre-drop manifest and split-hash report are archived
  beside their successors; the referenced WAV is quarantined outside the
  corpus root with an incident note.
- Corpus impact: 68,016 clips become 68,015. Only the calibration split
  changes (6,063 to 6,062 clips; one Piper spoof removed). The eval, holdout,
  and challenge split hashes are byte-identical before and after, verified
  mechanically. No evaluation-gate cohort loses a row; the Platt fit uses one
  fewer calibration observation.
- After the drop, a fresh canonical-PCM verification of all 68,015 retained
  clips passed with zero mismatches.
- Timing disclosure: this amendment is post-scoring-abort and pre-metric,
  not pre-scoring. The aborted run wrote raw per-clip scores for the 57,126
  clips preceding the failure. No metrics or aggregates were computed from
  that journal and the maintainer did not inspect any scores. The aborted
  journal is sealed with a recorded sha256, labeled invalid for metrics and
  resume, and retained as evidence only. The exclusion trigger is a
  deterministic input-integrity invariant, independent of model performance.
- The authoritative M2 scoring run restarts from scratch against the amended
  manifest under a new journal identity. The provisional baselines in this
  section are carried forward as pre-registered expectations only; final
  values come from the restarted run per the v1.2 re-baselining rule.

#### Scope and precedence

This protocol governs all model candidates trained under #252 (Phases 1
through 4 of the approved plan). It takes precedence over experiment notes,
ad-hoc analysis, and verbal agreements. Amendments may tighten gates or add
metrics but never loosen or remove them. Every amendment is timestamped,
version-bumped, and committed before the relevant training results are
unblinded.

#### Frozen artifacts

The following artifacts must be hashed and committed before Phase 1 training.

| Artifact | Role | Hash / version | Permitted use |
|---|---|---|---|
| M2 checkpoint (`finetuned_aasist.pth`) | Reference model | `e178446b640b8e9f9cf6dd359428b2243f49e24e613e1ae952cd706216b8111e` | Baseline scoring only |
| XLS-R 300M (`xlsr2_300m.pt`) | Frozen frontend | `b08927597f2c9eb2ebd7dcc3ac78ee4b5f6021cbac4b3a6c5a9deec445d80ed9` | Feature extraction; shared across all candidates |
| Selection seed | Partition assignment | `voxint-synthdetect-144` | Immutable; shared with bootstrap seeding |
| Evaluator revision | Metric computation | `8c3f36a3c7791848ceb2d90235e8a2bdddb72350` (evaluator repo commit) | Locked after golden-test validation (742-test suite green at this revision) |
| ffmpeg version | Codec pipeline | `7.1`, digest-pinned container `sha256:292a972c60356abd651d9a4f9c808c13e7473f65ad400b7eb99215f4e571931d` | Locked for all codec materialization |
| Calibration manifest | Platt fitting | `0a6fa284c6af00dfbea31b8bdda923479c173376aea97fb4f5d9a807b024384d` | Calibration split only; no model selection |
| Eval manifest | Iteration gates | `47a5fff1a76a053621f3129784c447217504d0063f2a8709485a1adb3d1f7560` | Phase gates; every touch logged |
| Holdout manifest | Phase-exit confirmation | `10300125f17628686c8e7e3e9530d9c7eaaec0c002511e315fc9724121e3db11` | At most 1 touch per phase |
| Challenge manifest | Ship decision | `528eaeaa19bb5988d1bddc7e77a8473d19d57adf2104b1668b4f5557c4414bf7` (RETIRED: the 2026-08-30 acceptance probe FAILED and the cohort was unblinded by the probe investigation; a reconstructed cohort under a new hash must replace it) | See Challenge procedure below |

Per-split manifest hash definition: take every clip record in the frozen
composite manifest carrying that split, sort the records by `clip_id`,
serialize them as one JSON array with sorted keys and compact separators,
and hash the UTF-8 bytes with SHA-256. The evaluator ships the tool that
computes these hashes; the four values above must be reproducible from the
frozen composite manifest alone. The frozen composite manifest (post-repair,
post-sanitization, post-v1.4-exclusion, 68,015 clips) has file sha256
`2c0717bb68b4802290fc033964ad3746e3dbf40ba66816b3fd832234173c5851`.

Codec recipes (deterministic, reproducible via the existing degradation
executor):

| Name | Codec | Bitrate | Mode | Notes |
|---|---|---|---|---|
| `clean` | None | n/a | n/a | Original waveform |
| `mp3-cbr48-v1` | MP3 | 48 kbps | CBR | `-c:a libmp3lame -b:a 48k -ar 16000 -ac 1` |
| `opus-voip-cbr16-f20-v1` | Opus | 16 kbps | VoIP, 20 ms frames | `-c:a libopus -b:a 16k -vbr off -application voip -frame_duration 20 -ar 16000 -ac 1` |
| `aac-lc-cbr48-v1` | AAC-LC | 48 kbps | CBR | `-c:a aac -b:a 48k -profile:a aac_low -ar 16000 -ac 1` |

#### Partitions and access budget

Four speaker-disjoint partitions. A speaker's clips never straddle two
partitions. Assignment uses SHA-256 hash-rank with `SELECTION_SEED`.

| Partition | Role | Access rule |
|---|---|---|
| **Calibration** | Fit Platt A, B; derive operating thresholds (5% FPR, 1% FPR) | No model selection. Thresholds frozen per seed before eval scoring. |
| **Eval** | Iteration feedback and phase gates | Every touch logged. Semi-public by design. |
| **Holdout** | Phase-exit confirmation | At most 1 touch per phase. Not for model selection. |
| **Challenge** | Binding ship decision | Opened once per phase after all choices frozen. See Challenge procedure. |

The challenge partition is a new 4th split created for this protocol. It is
not carved from the existing eval set. Speaker-disjoint and prompt-disjoint
from all other partitions and from all training data. Created and hash-frozen
before any Phase 1 training.

#### Metrics

**Score polarity.** Higher raw score always means more likely synthetic.
Monotonic Platt scaling preserves rank order and does not change EER.

**Windowing.** All protocol scoring uses production windowing (4.0375 s
windows, logit-mean pooling, 8,000-sample tail floor), the mode validated by
the S5 windowing verdict and used by the shipped service. Upstream
windowing remains a diagnostic mode only.

**Per-generator EER.** For each synthetic generator g, compute EER against
the frozen bonafide comparison population on the target partition. The
comparison population is domain-matched (v1.2): core TTS and unseen FM
generators score against in-domain bonafide (AMI and VoxConverse); the
ASVspoof DF anchor scores against ASVspoof bonafide clips. The two bonafide
pools are never mixed in one EER. Use the
existing linear-interpolation EER implementation (FPR/FNR crossing via
`eer_from_roc`). Report: EER point estimate, bootstrap 95% CI, effective
group count, clip count.

**Macro-average EER.** Equal-weight average of per-generator EERs across the
core generators (Chatterbox, Piper, ElevenLabs, Google TTS). ASVspoof DF is
an anchor reported separately; it is not included in the macro average.
Unseen FM generators (Matcha-TTS, F5-TTS) are also reported separately.

**TPR@5%FPR (primary operating metric).** On the calibration split, select
the lowest threshold satisfying grouped, weighted bonafide FPR at most 5%.
Freeze that threshold per seed. Apply it unchanged to eval, holdout, and
challenge. Report: target FPR, threshold, realized FPR, TPR, effective
bonafide group count, bootstrap CI. Per-generator TPR at the frozen threshold
is also reported.

**TPR@1%FPR (diagnostic).** Same procedure at 1% FPR. Not a binding gate. If
calibration has fewer than 1,000 independent bonafide clusters, label this
metric as underpowered in the report.

**Codec-stratified EER.** For each codec c, compare synthetic codec-c
descendants with bonafide codec-c descendants. Clean is its own stratum. All
descendants of one parent remain one effective group. Minimum 50 parent groups
per stratum to report; strata below this threshold are marked "underpowered"
and excluded from the guard rail.

**Out-of-sample Platt Brier score.** Fit Platt A and B on the calibration
split using one effective observation per parent/speaker group. Compute Brier
on the eval split with identical weighting. The binding Brier population is
fixed to the four core generators (Chatterbox, Piper, ElevenLabs, Google TTS)
and in-domain bonafide from AMI and VoxConverse. Exclude ASVspoof DF anchor
clips. Do not expand this population when later phases add generators. Report
full-eval Brier, including unseen FM generators and anchors, as a diagnostic.
Require A > 0 (higher score must mean more synthetic). The M2 baseline Brier
of 0.0066 was computed in-sample; the out-of-sample M2 value was required to
be measured before the gate limit became final, and on 2026-08-30 it was:
0.0347 on the binding population, fixing the gate limit at 0.0447.

**Paired delta.** For M2-vs-candidate comparisons, define
`delta = EER_candidate - EER_M2` and use identical bootstrap component draws
for both models. The gate passes iff `quantile(delta, 0.95) < 0`. A second
opening under the access budget uses `quantile(delta, 0.975) < 0`. Neutralize
`model-seed` in the bootstrap-seed derivation for paired comparisons so both
models draw identical component streams. Report the paired-delta distribution.

#### Bootstrap procedure

Connected-component resampling, 1,000 replicates, 95% percentile CI.

1. Build connected components from shared speaker identity and
   `partition_group_id`. A component includes every clean and codec-degraded
   descendant and all paired bonafide/synthetic items.
2. Resample components with replacement within each partition. Do not mix
   partitions in one replicate.
3. Within each replicate, retain generator and corpus strata composition.
4. Compute the target metric per replicate.
5. Report percentile CI using numpy `method="linear"` for cross-version
   stability.
6. For paired comparisons (M2 vs candidate), use the same component draws for
   both models in each replicate.

Thresholds (Platt parameters, operating-point thresholds) are calibration
artifacts. They are not refitted inside eval/holdout/challenge bootstrap
replicates.

Bootstrap seed:
`SHA-256(SELECTION_SEED || metric-schema-version || model-seed || cohort-hash || metric-context)`.

#### M2 re-baselining measurement (2026-08-30)

Run identity: the frozen M2 checkpoint (`e178446b...`) scored on all 68,015
clips of the frozen composite manifest (`2c0717bb...`) with the locked
evaluator revision (`8c3f36a3`), production windowing, deterministic CUDA
settings, journal `journal_m2_r2.jsonl` (0 clip errors, 0 skips). Platt
policy `m2-252-step6` fit on the 6,062-clip calibration split: A = 1.2222,
B = -1.6236 (A > 0 as required), in-sample Brier 0.0414. Frozen operating
thresholds from calibration: primary 5%-FPR threshold 1.7939 (realized
weighted BF FPR 4.08%, TPR 95.00%); diagnostic 1%-FPR threshold 2.2556
(realized 1.00%, TPR 90.42%), labeled underpowered because calibration has
35 bona fide components, far below 1,000.

Measured eval values (grouped bootstrap, 1,000 replicates, 95% percentile
CI; effective component counts beside every number):

| Metric | M2 measured | 95% CI | Components | Replaces |
|---|---|---|---|---|
| Piper EER | 1.84% | 0.00% to 2.56% | 18 | provisional 8.04% |
| ElevenLabs EER | 3.28% | 0.00% to 4.30% | 18 | provisional 11.20% |
| Google TTS EER | 4.69% | 0.75% to 5.68% | 18 | provisional 10.28% |
| Chatterbox EER (primary endpoint) | 10.37% | 2.52% to 12.26% | 18 | ship-gate M2 reference |
| ASVspoof DF anchor EER | 41.02% | 36.06% to 45.25% | 90 | newly measured |
| Unseen FM (Matcha) EER | 14.98% | 2.15% to 18.36% | 18 | newly measured; under-precision, see trigger below |
| Unseen FM (F5-TTS) EER | 11.48% | 2.19% to 12.76% | 18 | diagnostic only |
| Macro-average core EER | 5.05% | n/a (mean of 4 cores) | n/a | reported |
| AMI BF FPR at frozen 5%-FPR OP | 2.23% | n/a (point value at frozen threshold) | 4 (188 clips) | newly measured |
| VoxConverse BF FPR at frozen 5%-FPR OP | 4.95% | n/a (point value at frozen threshold) | 14 (452 clips) | newly measured |
| Out-of-sample Brier (binding population, n=2,376) | 0.0347 | n/a | n/a | newly measured |
| Full-eval Brier (diagnostic, anchors included) | 0.1955 | n/a | n/a | reported |

Per-source BF FPRs use the same inverse-component-size weighting as the
calibration realized FPR. The ASVspoof DF anchor bona fide population sits
at 63.26% weighted FPR at the frozen primary threshold; this is the
documented cross-corpus confound (#253), reported for transparency and
consumed by no gate.

**Regeneration trigger evaluation (v1.3).** Half-width is computed as
`(CI_hi - CI_lo) / 2` on the unrounded percentile bounds; this pins down a
term the trigger left ambiguous for asymmetric intervals, and no
classification below changes under either reading. CI half-widths of the
eval-only generators: ElevenLabs 2.15pp and Google TTS 2.47pp, both within
the 5.00pp precision bound, so their non-regression gates bind. Matcha-TTS 8.10pp
exceeds the bound: the trigger fires, the P2+ unseen-FM gate does not bind,
and the Matcha slice must be expanded or regenerated on clean post-repair
eval identities before that gate can bind. The Matcha baseline above is
recorded as under-precision and non-binding. F5-TTS at 5.28pp also exceeds
the bound; no gate consumes F5-TTS (it is diagnostic-only in every phase),
so this is a disclosure with no gate consequence.

**Holdout confirmation (single touch, logged 2026-08-30).** One holdout
scoring pass: Chatterbox EER 7.35% (CI 3.49% to 13.25%), Piper EER 0.34%,
out-of-sample Brier 0.0412. No further holdout touches this phase.

**Acceptance probe result (challenge partition): FAIL.** M2 Chatterbox EER
on eval 10.37% (grouped-bootstrap SE 2.99pp) versus challenge 19.54% (SE
1.17pp). The absolute delta of 9.17pp exceeds the pre-registered band of
6.28pp (1.96 times the root sum of squared SEs), so the probe fails and the
challenge manifest hash stays provisional. Per-source diagnostics: the
challenge partition is entirely AMI-sourced (the VoxConverse challenge
speakers remain deferred), and the shift is present within AMI itself
(eval AMI-only Chatterbox EER 4.00% on 125 spoof clips, an underpowered
exploratory number, versus 19.54% on the challenge cohort). Because this
subsection discloses the cohort's numeric results, the cohort is unblinded:
it is permanently retired and can never serve as a ship-decision cohort.
Reconstruction must select a fresh challenge cohort (new speaker selection,
new clips, new manifest hash) after the confounding investigation; the
challenge procedure's blinding rules apply to that new cohort from scratch.
No ship decision may consume any challenge cohort until then.

#### Binding gate matrix

**Non-regression gates (worst-of-3-seeds):**

| Phase | Cohort | Metric | M2 baseline | Limit | Consequence |
|---|---|---|---|---|---|
| All | Eval | Piper EER | 1.84% | ≤3.84% | Blocks ship |
| All | Eval | ElevenLabs EER | 3.28% | ≤5.28% | Blocks ship |
| All | Eval | Google TTS EER | 4.69% | ≤6.69% | Blocks ship |
| All | Eval | ASVspoof DF EER | 41.02% | ≤45.02% | Blocks ship |
| All | Eval (AMI) | BF FPR at 5%-FPR OP | 2.23% | ≤5.00% | Blocks ship |
| All | Eval (VoxConverse) | BF FPR at 5%-FPR OP | 4.95% | ≤10.00% | Blocks ship |
| All | Eval | Out-of-sample Brier | 0.0347 | ≤0.0447 | Blocks ship |
| P1 | Eval | Unseen FM (Matcha) EER | 14.98% | Report only | Diagnostic |
| P2+ | Eval | Unseen FM (Matcha) EER | 14.98% (under-precision, non-binding) | ≤25.00% AND gap ≤10pp | Blocks ship once the Matcha slice is regenerated |
| P2+ | Eval | Codec guard rail | n/a | No codec EER > min(2x pooled, pooled+10pp) | Blocks ship |

Unseen FM gap condition: for each seed s,
`Matcha_EER_s - Chatterbox_EER_s ≤ 10pp`. The maximum gap across seeds must
satisfy this condition.

The codec guard rail requires minimum 50 parent groups per stratum. Strata
below this threshold are excluded from the guard rail but still reported.

**Ship gates (median-of-3-seeds):**

| Phase | Cohort | Metric | Limit | Notes |
|---|---|---|---|---|
| P1 (interim) | Eval | Chatterbox EER | ≤18.00% | Plus ≥5pp absolute gain over M2 |
| P2 (primary) | Eval AND challenge | Chatterbox EER | ≤15.00% | Must pass on each cohort independently |
| P3 (stretch) | Eval AND challenge | Chatterbox EER | ≤10.00% | Aspirational; not a blocker for lower-tier ships |

All ship gates also require: every non-regression gate passes AND the
paired-delta bootstrap CI for Chatterbox EER (candidate vs M2) excludes zero
in the improving direction.

#### Seed discipline and instability veto

Three seeds per training configuration: seeds 0, 1, and 2.

- **Non-regression gates**: evaluated on the worst (maximum) value across
  seeds.
- **Ship gates**: evaluated on the median value across seeds.
- **Reporting**: all three seed values, median, and worst are always reported.

**Instability veto.** If `worst - median > 5pp` on Chatterbox EER (the
primary endpoint), the run is INVALID regardless of individual gate outcomes.
Pre-specified remediation: train 2 additional seeds (3 and 4) under the same
frozen configuration, then re-evaluate all gates as worst-of-5
(non-regression) and median-of-5 (ship). If instability persists at 5 seeds,
the configuration fails the phase.

**Checkpoint selection.** Fixed rule per phase, committed before training:
select the checkpoint at the epoch with the lowest Chatterbox EER on the
calibration split. This rule does not vary per seed.

#### Exclusion criteria

- **Minimum clip duration**: 1.0 second after silence trimming. Clips below
  this are excluded before scoring; the exclusion count is reported.
- **Failed decodes**: codec materialization failures are logged and the parent
  group is excluded from that codec stratum (not from clean). If more than 5%
  of parent groups fail any single codec, the recipe is invalid.
- **NaN or infinite scores**: any clip producing a NaN or infinite raw score
  makes the entire seed's evaluation INVALID.
- **Score polarity**: if Platt slope A is not positive after calibration
  fitting, the seed is INVALID.

#### Challenge procedure

1. **Assembly.** Create the challenge partition from Chatterbox clips whose
   speakers and prompts do not appear in calibration, eval, holdout, or
   training data. Include matched bonafide speakers from AMI and VoxConverse,
   also speaker-disjoint from all other partitions. Target: 130 Chatterbox
   speakers (~90 AMI, ~40 VoxConverse, mirroring the eval source mix), 6
   clips each. Select speakers via SHA-256 hash-rank with `SELECTION_SEED`
   and context `"challenge-speakers-v1"`. Keep at least 40 AMI speakers as
   reconstruction reserve. Final count confirmed by empirical power pilot
   before freezing.

2. **Acceptance probe.** Before hash-freezing, score the challenge partition
   with the frozen M2 model. Let `E_challenge` and `E_eval` be M2 Chatterbox
   EER on challenge and eval, with `SE_challenge` and `SE_eval` estimated from
   each partition's own grouped bootstrap. Accept iff
   `|E_challenge - E_eval| <= 1.96 * sqrt(SE_challenge^2 + SE_eval^2)`;
   otherwise investigate confounding and reconstruct the partition. The
   probe's detection floor is approximately 8 to 10 percentage points. Run
   separate AMI and VoxConverse diagnostics alongside the pooled probe to
   detect subtler corpus-specific shifts.

3. **Freezing.** Hash the challenge manifest (speaker IDs, clip IDs, codec
   variants, bonafide pairings). Expose only: manifest hash, total counts,
   and acceptance-probe pass/fail. Withhold individual scores and per-stratum
   outcomes.

4. **Opening.** Score the challenge once per phase, after architecture,
   augmentation policy, stopping rule, three training seeds, checkpoint
   selection, and calibration procedure are all frozen for that phase. No
   parameter may be changed between opening and verdict.

5. **After opening.** Report all metrics per seed. If the ship gate fails, the
   phase fails. Training may continue to the next phase, but the failed
   result is recorded and the same challenge partition is reused (with
   Bonferroni adjustment if consulted more than once for the same phase).

6. **Budget.** One unblinding per phase. A second use of the same challenge
   partition for the same phase requires halving the effective alpha
   (equivalent to requiring the upper 97.5% CI to pass instead of 95%).

#### Verdict definitions

- **PASS**: every non-regression gate passes (worst-of-3) AND the applicable
  ship gate passes (median-of-3) AND `quantile(delta, 0.95) < 0` for the
  paired Chatterbox EER delta AND no instability veto fires.
- **FAIL**: any binding gate does not pass after all pre-specified remediation
  is exhausted.
- **INVALID**: instability veto fired, NaN or infinite scores, Platt slope not
  positive, or underpowered partition. An invalid run is not a pass or a fail;
  it must be remediated before a verdict.

All gate comparisons use unrounded values. 18.004% does not pass a ≤18.00%
gate. Report all seed values, all metrics, all strata, and all exclusion
counts whether the run passes or fails. A gate outcome is reported as
computed and never retroactively reclassified.

### Service contract

The synthdetect model service runs as a standalone FastAPI container
(`voxint-synthdetect`), deployed via the `compose.plugin-synthdetect.yaml`
overlay.

| Property | Value |
|---|---|
| **Image** | `ghcr.io/bengizmo/voxint-synthdetect:{tag}` |
| **Port** | 8025 (compose publishes `127.0.0.1:8025` for local debug; app traffic uses service DNS) |
| **Health endpoint** | `GET /healthz` |
| **GPU** | 1x NVIDIA GPU (tested on RTX 3060 12 GB, SM 8.6) |
| **Restart policy** | `unless-stopped` |
| **Weights** | Baked into the image at build time (sha-verified; not downloaded at startup) |
| **Media volume** | `MEDIA_ROOT` shared read-only with api/worker |

Environment variables (set in `.env`, passed through the overlay):

- `SYNTHDETECT_ENABLED` (default `false`): master switch. When false, no new
  scoring jobs are created (manual or automatic). Jobs already queued before
  the flag was cleared may still complete.
- `SYNTHDETECT_AUTOGENERATE` (default `false`): when true, completed pipeline
  runs are automatically scored without operator action.
- `SYNTHDETECT_URL` (default `http://localhost:8025`): the overlay overrides
  this to the compose service DNS name (`http://synthdetect:8025`).
- `SYNTHDETECT_HTTP_TIMEOUT_SECONDS` (default `120`): HTTP timeout for
  scoring requests to the service.

The service is optional. When it is not running or the plugin is disabled,
submissions complete normally; they are not scored. The pipeline is never
blocked by synthdetect availability.

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
