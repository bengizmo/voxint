# Plan: Bare-metal Apple Silicon ("metal") compute tier — v1

> Status: approved 2026-08-14. Phase 0 (MPS spike) in progress.

## Context

Voxint 0.6.0 runs on Apple Silicon only via the CPU tier in Docker — validated
(SMOKE PASS, titanet cosine 0.999999 vs the CUDA reference, VoxConverse 4-file
DER 0.105) but CPU-bound at ~2.5× real-time, because Docker Desktop on macOS
has no GPU passthrough. The heavy stages are transcribe (84–134 s) and
diarize_embed (78–110 s) on ~70–84 s clips. Goal: a new `metal` compute tier
running the three model services natively on macOS so pyannote can use the
Apple GPU (MPS), analogous to gpu/rocm — with measured parity evidence per the
numerics doctrine. Follows the 2026-08-14 Apple Silicon CPU-tier validation.

## Decisions

1. **v1 scope: package all three services natively; accelerate only pyannote
   (MPS).** Whisper = native CT2 large-v2 on CPU (packaging byproduct — expect
   ≤~1.4× on transcribe; honest v1 outcome ~1.5–1.8× RT overall, still
   transcribe-bound; docs must say so). Titanet = ONNX CPU EP by default —
   this also discharges the arm64 3-level parity measurement
   `docs/gpu-contracts.md` flags as owed. CoreML EP lands as an env flag +
   measurement harness only; the default flips only via a later
   evidence-linked commit.
2. **Whisper Metal engine: defer + file a tracking issue.** v1 reserves an
   `ENGINE` seam in the whisper service (documented, not built). The issue
   captures: mlx-whisper as leading candidate (ports openai-whisper's
   `avg_logprob` semantics — the contracted confidence field), whisper.cpp
   fallback, CTranslate2's experimental MPS PR (OpenNMT/CTranslate2#2077)
   tracked; pre-registered gate = WER/CER vs committed CT2 baseline
   transcripts on an expanded corpus (≥40 files incl. long-form for
   30 s-window seam artifacts), text-aligned boundary tolerances
   (p95 ≤ ~0.5 s), zero-insertion silence/hallucination fixtures, confidence
   correlation or an explicit contract amendment (gpu-contracts.md
   "Confidence" section); VAD stays voxint-owned to isolate decode drift.
3. **Parity strategy: NO committed `references/metal/` oracle.** MPS/CoreML
   are not run-to-run or cross-chip (M1–M4) stable, and macOS toolchain
   updates reschedule kernels unpinnably. Canonical references stay CPU/CUDA
   (the embedding space is defined parametrically, device-independent). Metal
   outputs gate against existing references with tier-specific **measured**
   tolerances; per-chip verdict/drift reports (chip, macOS, library versions,
   margins vs every floor, 3× repeat runs) are committed as evidence —
   generalizing the existing ONNX verdict-table pattern. Tooling still
   supports writing a metal reference set to a scratch dir so this call can
   be revisited with data.
4. **Supervision: launchd user agents.** launchd `KeepAlive` is the native
   equivalent of the `restart: unless-stopped` doctrine the contract tests
   enforce for containers; a pidfile supervisor either lacks crash-restart or
   reimplements launchd badly. A `run <svc> --foreground` mode covers
   debugging.
5. **Installer stays Docker-only**; the native side is a separate launcher
   script the installer hands off to. Merging multi-GB downloads + venv
   builds into the Bash-3.2 installer doubles its failure domain.
6. **Explicit device control, no silent fallback:**
   `DIARIZER_DEVICE=auto|cuda|mps|cpu` (a forced device must exist AND pass
   the probe, else hard error) and `TITANET_ORT_PROVIDERS` (requested EPs
   must be what the ORT session actually uses, else fail at load). Healthz
   honesty per gpu-contracts.md (`mps` = torch Metal; `metal` = CoreML EP).

## Architecture

Docker keeps core (postgres/redis/migrate/api/worker/beat) via `compose.yaml`
+ new `compose.metal.yaml` that only rewires api/worker:
`ASR_URL/DIARIZER_URL/EMBEDDER_URL → http://host.docker.internal:{8022,8024,8021}`,
`COMPUTE_TIER: metal`, `extra_hosts: host.docker.internal:host-gateway`.
Native services bind **127.0.0.1 only**, run from per-service uv venvs
(Python 3.11, matching the images) under
`${VOXINT_METAL_HOME:-$HOME/.voxint-metal}/{venvs,models,logs,run}`, launched
by `scripts/metal/voxint-metal.sh` (Bash 3.2, `set -eu`, library seam
`VOXINT_METAL_LIB=1` mirroring install.sh). Subcommands: `setup / up / down /
status / logs / doctor / run <svc> --foreground`.

Weights (token-free, sha-verified): pyannote + titanet fetched via `curl -fL`
from the public release assets (`pyannote-models-v1`, `titanet-onnx-v1`),
verified against the committed provenance JSONs; whisper pre-downloaded in the
venv exactly like `Dockerfile.cpu` (pinned HF revision `f0fe815…`), with
post-download sha256s recorded to a local manifest for drift detection.

Pyannote vendored config: recreate the tree under
`$VOXINT_METAL_HOME/models/pyannote/vendored/pyannote/` (path keeps the
load-bearing `"pyannote"` substring) and **generate** a local config from the
committed `config.vendored.yaml` — rewrite only the two checkpoint paths,
assert the embedding path contains `"pyannote"` and that non-path params still
match `provenance.json pipeline_params`; export `VOXINT_VENDORED_PIPELINE` to
it (the missing-file hard-fail in `diarizer.py` then protects it).

`MEDIA_ROOT` for native services = the physically resolved (`pwd -P`) host dir
the worker mounts as `/data/media` (clients send MEDIA_ROOT-relative paths);
`doctor` asserts agreement with `.env`, detects port collisions (e.g. a
leftover cpu-tier stack on 8021/8022/8024), and checks ORT providers + the
MPS probe.

## Phase 0 — MPS spike (go/no-go, before any build-out)

Throwaway venv, no tier code: run pyannote.audio 3.1.1 on torch 2.5.0 with
device=mps on VoxConverse files. Measure wall time + DER vs CPU; census MPS op
fallbacks with `PYTORCH_ENABLE_MPS_FALLBACK` **unset** so unsupported ops fail
loudly; verify actual device placement (healthz-says-mps while the hot path
runs CPU is a false green); 3× repeat for run-to-run variance. Kill or
continue. Also verify CT2/faster-whisper arm64 macOS wheels resolve under uv.

## Implementation slices (each commit green; contract tests land with their change)

1. **Config tier** — `src/voxint/config.py` `compute_tier` Literal +=
   `"metal"` (keeps GPU-class timing; scaling is `== "cpu"` only).
   `tests/unit/test_config.py` metal row; `docs/timeouts-and-leases.md`
   tier-table row.
2. **Compose overlay** — `compose.metal.yaml` (api/worker only, no images).
   Contract tests: add to the restart-policy enumeration; refactor
   `test_compose_default_pins_identical` to a glob where zero-pin files are
   exempt but any found pins must match; new
   `test_metal_overlay_is_rewiring_only` (services ⊆ {api,worker}, no
   image/volumes/ports keys, URLs → host.docker.internal, COMPUTE_TIER set).
3. **titanet providers** — `TITANET_ORT_PROVIDERS` env replacing the
   hardcoded `["CPUExecutionProvider"]` in `engine_onnx.py`; validate
   requested ⊆ available before construction AND assert
   `session.get_providers()` honors the request after (ORT silently
   degrades); `device_name="metal"` when CoreML EP is active. Unit tests for
   parsing/mismatch/mapping. Default unchanged → shipped images behaviorally
   identical.
4. **pyannote device forcing** — `DIARIZER_DEVICE` consumed by
   `select_device()`; a forced device must pass `probe_device` else
   RuntimeError. Extend `TestDeviceCascade`/`TestResolveDeviceName`. Needed
   for same-host MPS-vs-CPU A/B measurement.
5. **Requirements** — `services/whisper/requirements.metal.txt` (torch-free,
   modeled on the rocm flavor which proves torch is optional; numpy 1.24.3
   for py3.11; pin `ctranslate2==<measured>` once known — the macOS wheel is
   otherwise un-sha-pinned) and `services/pyannote/requirements.metal.txt`
   (= requirements.txt + `torch==2.5.0`/`torchaudio==2.5.0`, matching
   Dockerfile.cpu; extend `test_torch_pins_match_across_flavors`). titanet
   reuses `requirements.cpu.txt` verbatim (the parity verdict binds to
   exactly that chain). Mirror contract tests land in the same commit.
6. **Launcher** — `scripts/metal/voxint-metal.sh` + launchd plist heredocs
   (`plutil -lint`; explicit env dict — launchd inherits no shell env;
   StandardOut/ErrorPath to logs; KeepAlive SuccessfulExit=false) +
   `tests/unit/test_metal_launcher.py` via the library seam (env assembly,
   plist gen to tmp, sha-verify logic against fixture provenance, vendored-
   config generator, `pwd -P` MEDIA_ROOT resolution).
7. **Installer** — `[M]` option: `normalize_tier`,
   `compose_file_args_for_tier` (`-f compose.yaml -f compose.metal.yaml`),
   `prompt_compute_tier` (default `m` on Darwin/arm64), `print_handoff`
   (core up, model services NOT running — hand off to
   `voxint-metal.sh setup && up`, honest that submissions fail until then),
   `.env.example` `VOXINT_COMPOSE_TIER` values. Installer tests: fixture copy
   list + parametrized rows + `m` prompt case.
8. **Parity tooling + docs** — extend `tools/generate_parity_references.py`
   with `--tier metal` (heterogeneous expected healthz: whisper cpu, pyannote
   mps, titanet cpu|metal; the committed-dir guard stays cuda-only; stamp
   chip/macOS/library-version metadata). New
   `tests/parity/test_pyannote_metal.py` (no pyannote parity module exists
   today): forced-MPS vs forced-CPU vs `references/cuda/diarize.json` —
   speaker-count equality, boundary-drift bound, mapping agreement; include a
   near-threshold fixture (the clustering threshold 0.7045 is a knife edge).
   Thread `VOXINT_PARITY_ORT_PROVIDERS` into `test_titanet_onnx.py` session
   construction so the full 3-level gate runs against CUDA refs under CoreML
   EP, plus a same-window repeat-determinism probe. Whisper:
   transcribe-fixture comparison in smoke style (macOS arm64 CT2 is a
   different wheel build — measure anyway). All skip-clean in CI (dev-mode
   `_PREREQS` pattern; `VOXINT_PARITY_REQUIRED=1` never set for metal lanes).
   Docs: gpu-contracts.md metal section (verdict table mirroring the ONNX
   one), release-process.md "Gate M" (maintainer-run on Apple Silicon,
   mirroring the ROCm gate), operations.md, onboarding.md, README,
   CHANGELOG [Unreleased].
9. **Post-measurement (separate PR, evidence in the commit message)** —
   ratchet measured floors into constants; confirm the no-metal-references
   call with data; decide the CoreML default (likely stays off — ORT
   partitions dynamic-length graphs back to CPU, may be slower AND noisier);
   decide whether metal needs a timeout factor (v1 runs CPU whisper under
   GPU-class timeouts — a 6 h recording could hit lease reclaim; a
   one-constant change in `_apply_compute_tier_profile`).

## Follow-up issues to file (not in this effort)

- Whisper Metal engine bakeoff (decision 2 — full gate design in the issue).
- macOS CI: GitHub `macos-14+` arm64 runners have MPS — could automate part
  of Gate M later (unlike ROCm, hardware exists in CI).

## Verification (end-to-end, maintainer Apple Silicon hardware, 16 GB)

Memory budget: whisper int8 ~4 GB + pyannote/MPS ~3 GB + titanet ~1 GB
native; cap the Docker Desktop VM at ~4 GB (core stack only). Steps:

1. `./scripts/install.sh` → `[M]` → core stack up with the metal overlay.
2. `voxint-metal.sh setup` (~3.2 GB downloads, sha-verified) → `up`.
3. **Loopback check first**: from the worker container,
   `httpx.get('http://host.docker.internal:8022/healthz')` — verifies the
   Docker Desktop loopback proxy before anything else (OrbStack differs).
4. `voxint-metal.sh status` → whisper `device: cpu`, pyannote `device: mps`,
   titanet `device: cpu, engine: onnxruntime`.
5. Smoke `tools/smoke_cpu_services.py` against the native services with the
   parity fixtures as media root.
6. Full pipeline run via the console (tutorial clip); record per-stage
   wall-times (feeds the timeout-factor decision) and the diarize_embed
   pyannote/titanet split (feeds the CoreML-worth-it decision).
7. Parity measurement: pyannote MPS-vs-CPU A/B; titanet CoreML-vs-CPU under
   the 3-level gate; 3× repeats; record chip metadata → verdict report.
8. Success bar: DER within measured ε of the 0.105 baseline, wall-clock
   beats 2.5× RT materially, zero runtime network fetches, honest healthz.

## Key risks (carried into implementation)

- MPS silent CPU fallback / op gaps → spike-first, fallback unset during
  validation, placement check.
- pyannote decision-level fragility: near-threshold clustering (0.7045) can
  flip speaker counts under small MPS drift — the near-threshold fixture is
  load-bearing, and the 4-file DER corpus is a smoke test, not a floor.
- `host.docker.internal`→loopback is Docker-Desktop-specific (verify early;
  a 0.0.0.0 fallback has LAN exposure — document only if ever needed).
- Version skew: native services run from the working tree, core from pinned
  images — `status` prints `git describe` + `VOXINT_IMAGE_TAG` as a skew
  warning; docs pair tag X.Y.Z with image X.Y.Z.
- APFS case-insensitivity + `/tmp`→`/private/tmp` symlinks vs the path
  contract → `pwd -P` everywhere; exercise path-validation tests on APFS.
- Thread oversubscription (CT2 × torch × uvicorn × Docker) → pin thread envs
  during benchmarks; warm before timing; `caffeinate` for long batches.

## Review notes

- Reviewed pre-approval by a three-model panel (codex via clink; two
  OpenRouter flagships) plus a repo-grounded planning agent that verified
  every file/line claim. Unanimous panel outcomes adopted: no committed
  metal reference oracle; pyannote-MPS-only acceleration in v1; whisper
  engine deferred with a pre-registered gate. Splits recorded rather than
  hidden: engine preference (mlx-whisper vs whisper.cpp — deferred to the
  bakeoff), CoreML-in-v1 (resolved: flag + harness only), supervision
  (resolved: launchd, per the restart-policy doctrine).
