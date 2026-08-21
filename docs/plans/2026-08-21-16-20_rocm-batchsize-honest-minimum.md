# Plan: AMD/ROCm whisper BATCH_SIZE — honest minimum (fix hash + measure + document)

## Context

The task began as "add an AMD/ROCm whisper `BATCH_SIZE` hardware profile" on the
already-shipped hardware-aware feature, targeting the maintainer's AMD box (Radeon
9060 XT, 16GB, ROCm tier). A deep read plus a 3-way model consult (codex,
deepseek-v4-pro, kimi-k3) unanimously reframed it: building an AMD BATCH_SIZE profile
now would manufacture a numerics change nothing demands, and would add speculative
bloat (AMD detection, a standing ROCm parity lane) to a single-operator tool.

Why a reduced BATCH_SIZE is not warranted:

- The #96 OOM concern was a **12GB NVIDIA card with three GPU-resident services**
  competing for VRAM. On ROCm only whisper is GPU-resident (pyannote/titanet run
  the `-cpu` images because MIOpen convolutions fail on consumer AMD GPUs; titanet's
  CPU path is already faster than real-time — `compose.rocm.yaml` header, issue #4).
  So a single model sits alone on a **16GB** card. The pressure does not transfer.
- The ROCm image already ships `BATCH_SIZE=16`. Keeping it means **no decode-identity
  change and no parity obligation**. Cutting to 8 would spend the one measured AMD
  win (4.8x over CPU baseline, `compose.rocm.yaml:9-10`), fork the numerics identity
  per tier, and create a gate that cannot run in GitHub CI (no GPU runner). Doctrine:
  "conservative = smallest delta from the *measured* baseline," and that baseline is 16.

The genuinely valuable finding from the consult is a **real, live provenance bug**,
independent of the AMD question: `decode_config_hash` omits `device`. That is the
core deliverable here.

Scope chosen by the operator: **honest minimum** (below). Explicitly NOT building
the installer AMD plumbing or a standing ROCm parity lane.

## Deliverable 1 (primary, hardware-independent): fix the `decode_config_hash` device blind spot

`decode_identity()` (`services/whisper/app/transcription.py:380-418`) hashes a
`config` dict that folds in `engine`, `runtime`, `runtime_version`, `model_name`,
`compute_type`, `batch_size`, and the VAD plan/params — but **not `device`**. Its
docstring claims it "Digests everything that moves numerics deployment-to-deployment,"
which is false: both the CUDA and ROCm images report `faster_whisper.__version__`
= 1.2.1 and `ctranslate2.__version__` = 4.8.1 (verified — the ROCm CT2 build carries
no HIP suffix), with identical engine/model/compute/batch/VAD. So **a CUDA box and a
ROCm box at batch 16 produce a byte-identical `decode_config_hash` today**, even though
the repo's own doctrine states CT2 output differs across backends. The provenance
artifact cannot name the two platforms it would compare — a live collision.

Change (~3 lines + docstring):

- In the `config` dict (`transcription.py:397-408`), add
  `"device": resolve_device_name(self._backend.device)`. Use the **resolved** label
  (`resolve_device_name`, `transcription.py:44-65`, returns "rocm"/"cuda"/"cpu"/...),
  NOT the raw CT2 `backend.device` string (which is "cuda" on ROCm). This is safe here
  because `decode_identity()` is already documented as call-after-`load_model`, and the
  HIP maps-probe inside `resolve_device_name` is only valid post-construction.
- Fix the docstring (`transcription.py:383-386`) to state device/backend is included.

Doctrine note: this changes the reported hash for **every** tier (cuda/cpu/metal too),
by design — distinct platforms must hash distinctly. It is a decode-**identity** change,
NOT a numerics-**output** change: transcription output is untouched, so the committed
transcript/segment parity references are unaffected. Blast radius on tests is tiny:
the only test referencing the field (`tests/contracts/test_routes.py:101`) asserts a
**stubbed** `"deadbeef"` from a fake backend, not a real computed hash.

Test to add (`tests/contracts/` or `tests/unit/`): construct two identities that differ
only in resolved `device` and assert their `decode_config_hash` values differ; assert a
single-device identity is stable/cached. Reuse the existing fake-backend pattern from
`tests/contracts/test_routes.py`.

Adjacent (note, do not act now): the committed CUDA reference predates this field and
pins no batch_size; on the next reference regeneration, record the full decode config.

## Deliverable 2 (maintainer, on the AMD box): host-side VRAM soak at BATCH_SIZE=16

Prove the shipped config is safe on the real AMD card, since ordinal "16GB > 12GB"
reasoning is directionally reassuring but doctrinally inadmissible (ROCm allocator /
hipBLASLt workspace behavior differs from CUDA).

- Bring up the ROCm whisper container on the maintainer AMD box (`compose.yaml` +
  `compose.rocm.yaml`).
- Transcribe one **speech-dense file with >16 VAD windows** (long-form) at `concurrency=1`
  — peak VRAM at fixed batch is roughly file-length-independent (batch-bounded), so this
  is the whole soak. Both VAD modes.
- Measure VRAM with **host `rocm-smi` / `amd-smi`**, NOT `/healthz` (the service telemetry
  is NVML-only and reports "unsupported" on ROCm — a service-telemetry gap, not a
  measurement excuse).
- Pass condition: no OOM, no container restart, correct transcription, recorded headroom.
- Record the number and verdict in `docs/gpu-contracts.md` next to the 4.8x claim
  (this is infrastructure evidence, not a numerics oracle).

First check kimi's open question: did the original 4.8x RDNA4 measurement already run at
batch 16 on long files with no OOM observed? If yes, this soak is informally done and
only needs writing down; if not, run it fresh. (This step requires GPU hardware and runs
after plan approval — it cannot run in plan mode.)

## Deliverable 3 (docs): state the honest AMD stance

Using the `voxint-docs` skill (lay-reader lane; no emdashes, no LLM-isms, emoji-free):

- `docs/setup.md:177` currently says a tuned per-GPU `BATCH_SIZE` profile is "still to
  come." Reword to the honest stance: the ROCm tier keeps the image default
  `BATCH_SIZE=16`; there is no per-tier batch matrix, because only whisper is
  GPU-resident on ROCm and 16GB is ample for a single model.
- `docs/gpu-contracts.md`: record the soak verdict (Deliverable 2) in the ROCm/Gate R area.
- Optional escape hatch: document `BATCH_SIZE=8` as an **unmeasured, at-your-own-risk**
  override for a hypothetical smaller AMD card, mirroring the `WHISPER_MODEL` disclaimer
  pattern (`transcription.py:5-7`). Only if the operator wants it surfaced.
- Update `CHANGELOG.md` under `[Unreleased]` for the hash fix.

## Deliverable 4 (docs, spec-only): the future batch-invariance gate

Write down — as a trigger-gated obligation, NOT built now — what would be owed IF a
measured OOM at 16 ever appears on AMD and forces a lower batch. Put this in
`docs/gpu-contracts.md` (or an internal maintainer note). The spec, distilled from the
consult:

- Same host, one config resident at a time (mirror `_run_engine`,
  `test_whisper_ct2_self_parity.py:123-135`); compare baseline `BATCH_SIZE=16` vs candidate.
- Corpus MUST cross the batch-16 boundary: clips with **>16 VAD windows** (>=17 crosses
  once, >=33 twice). The curated AMI subset (max 10 windows) is vacuous at batch 16 and
  must not be reused unchanged.
- Metrics: pooled WER (`score_pooled`) **plus** alignment coverage (>=99% of words align
  1:1, to defeat survivorship bias) **plus** matched-word |dstart|/|dend| distribution
  (p50/p95/max, floored at the ~20ms whisper token quantum) **plus** segment-boundary
  drift **plus** the zero-insertion invariant on non-speech fixtures. WER alone certifies
  the wrong property.
- Rationale to record: the VAD path risk is `restore_speech_timestamps`
  (`transcription.py:483-488`) — a small local timestamp perturbation near a speech-chunk
  boundary can jump by the full removed-silence gap. And a committed `references/rocm/`
  oracle is rejected: CT2 picks ISA/BLAS dynamically, so a frozen cross-machine oracle
  conflates machine/driver drift with the batch variable (self-parity cancels it).

## Explicitly NOT in scope (rejected as speculative bloat)

- No AMD GPU detection in `scripts/install.sh` (no `rocm-smi`/`amd-smi`/PCI probe): the
  operator already selected `compose.rocm.yaml`, so the tier is known; detection would
  also weaken the documented host-driver-only requirement.
- No ROCm profile arm in `select_hardware_profile`; no un-gating the hardware-override
  path from gpu-only to rocm.
- No `references/rocm/` committed oracle.
- No standing ROCm whisper parity lane in CI (permanent maintainer cost for an unobserved
  problem).
- No `rocm-smi` telemetry path added to the whisper service.

If ever un-gated in future, the load-bearing hazard the consult flagged: a stale
NVIDIA-generated `compose.hardware.yaml` folding into the ROCm chain after a tier switch.
Mitigation would be to bind each managed override to its owning tier and refuse folding on
mismatch. (Recorded for the future; not built now.)

## Critical files

- `services/whisper/app/transcription.py` — `decode_identity()` (380-418), `resolve_device_name()` (44-65).
- `tests/contracts/test_routes.py` — fake-backend pattern; add the device-hash test near here.
- `docs/setup.md:177`, `docs/gpu-contracts.md` (ROCm/Gate R area, ~561/581/815-819), `CHANGELOG.md`.
- `compose.rocm.yaml` (hybrid-tier rationale; the 4.8x claim at :9-10) — reference only.

## Verification

- `uv run ruff check services/voxint tests && uv run mypy services/whisper/app/transcription.py`
  (types mandatory; ruff/mypy clean before landing).
- `uv run pytest tests/contracts/test_routes.py tests/unit -q` plus the new device-hash test:
  assert two identities differing only in resolved device produce different
  `decode_config_hash`, and that the existing "deadbeef" stub test still passes.
- Full `uv run pytest -q` green; the whisper parity lanes still SKIP off Apple Silicon
  (unchanged — we add no ROCm lane).
- Deliverable 2 soak: on the maintainer AMD box, `rocm-smi` shows headroom and no
  OOM/restart across the >16-window run; verdict recorded in `docs/gpu-contracts.md`.
- Release hygiene before landing: `gitleaks git .` and `gitleaks dir .` (no internal
  hostnames — the plan doc and any commit must say "maintainer AMD hardware", never the
  machine name). Non-trivial change gets a multi-model review before landing; record
  applied fixes / deliberate skips in the commit message. Push both remotes if a private
  origin exists on this checkout (currently only `origin`).
