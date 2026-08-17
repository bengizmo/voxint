# Release process (maintainers)

How a Voxint release is cut. Development happens on the private origin; the
public mirror is `github.com/bengizmo/voxint`. Releases are cut there because
the image workflow only runs on GitHub (`release.yml` is guarded by
`github.server_url` and stays inert elsewhere).

## What a release produces

| Artifact | Where | How |
|---|---|---|
| `ghcr.io/bengizmo/voxint:X.Y.Z` (app: api/worker/beat/migrate, multi-arch amd64+arm64) | GHCR | `release.yml` `build-multiarch` → `merge-multiarch` |
| `ghcr.io/bengizmo/voxint-{whisper,pyannote,titanet}:X.Y.Z` (CUDA, amd64) | GHCR | `release.yml` `publish-images` matrix |
| `ghcr.io/bengizmo/voxint-{whisper,pyannote,titanet}:X.Y.Z-cpu` (multi-arch amd64+arm64) | GHCR | `release.yml` `build-multiarch` → `merge-multiarch` |
| `ghcr.io/bengizmo/voxint-whisper:X.Y.Z-rocm` (AMD GPU, amd64, build-only in CI; see Gate R) | GHCR | `release.yml` `publish-whisper-rocm` |
| `voxint X.Y.Z` sdist + wheel | PyPI | manual `uv build` + `uv publish` (not in CI) |
| Release notes | GitHub Releases | manual `gh release create` |

Tag semantics (immutable exact-semver, per the non-NVIDIA plan): unsuffixed
model-service tags are **CUDA**; `-cpu` is the multi-arch CPU flavor;
`-rocm` is the AMD whisper flavor (amd64). `docker/metadata-action` also
emits mutable `X.Y` / `X.Y-cpu` / `X.Y-rocm`
tags for full releases; compose files must keep referencing the exact `X.Y.Z`.
Pre-release versions (e.g. `v0.0.0-test`) get only their exact tag, useful for
exercising the workflow without polluting `X.Y`.

### The titanet model asset (release dependency)

The ~100 MB `titanet-large.onnx` is **not in git**. The workflow's parity gate
and the titanet `-cpu` build fetch it from the standing model-asset release
(**`titanet-onnx-v1`**) and verify its sha256 against
`tests/parity/fixtures/onnx/provenance.json`; the Dockerfile re-verifies at
build time. A re-export (new checkpoint or export pins) must publish a **new**
asset release (`titanet-onnx-v2`, …), update `provenance.json` and
`services/titanet/Dockerfile.cpu`'s `TITANET_ONNX_SHA256` default (a contract
test pins them together), and bump `TITANET_ONNX_RELEASE` in `release.yml`. Assets under an existing tag
are immutable by policy, never replaced.

### The pyannote model asset (release dependency)

Same pattern: the diarization checkpoints (~33 MB, `segmentation-3.0.bin` +
`wespeaker-voxceleb-resnet34-LM.bin`) are **not in git**. Every pyannote image
build fetches them from the standing **`pyannote-models-v1`** asset release
and verifies their sha256s against `services/pyannote/models/provenance.json`
(the Dockerfiles re-verify at build time; a contract test pins the Dockerfile
ARG defaults to the provenance file). A weights refresh publishes a **new**
asset release (`pyannote-models-v2`, …), updates the provenance file and both
Dockerfiles' sha ARGs, and bumps `PYANNOTE_MODELS_RELEASE` in `release.yml`.

### Release gates wired into the workflow

- **`frontend`** (CI, issue #48) runs `npm ci` → lint → typecheck → `npm run
  build` (which includes `tsc --noEmit`, `vite build`, and the no-CDN
  `check-no-cdn-urls.mjs` offline-self-host check over the built `dist/` bytes)
  → `npm audit --omit=dev --audit-level=high`. It gates merge on the same
  footing as `ruff`/`mypy`/`pytest` and runs independently (no Postgres
  coupling). It is a **required status check** on `main`. The Dockerfile runs
  the identical frontend build stage as part of the single image build, so no
  standalone release job is needed — building in both CI and the Dockerfile is
  intentional (CI fails fast without a full Docker build; the Dockerfile stage
  is what ships). A version bump needs **no** frontend rebuild unless
  `frontend/` changed. Frontend dependency provenance: capture
  `npm ls --all --json` (or `npm sbom`) as a release artifact, mirroring the
  ONNX/pyannote provenance discipline; each new npm dep is justified in its PR.
- **`parity-gate`** runs the strict titanet ONNX parity harness
  (`VOXINT_PARITY_REQUIRED=1`) natively on both amd64 and arm64 runners, and
  blocks every multi-arch build. The NeMo/CUDA reference side stays a
  maintainer-run gate: re-run the `tools/generate_parity_references.py` flow on
  an NVIDIA box against the new tag before releasing (Gate A, the NVIDIA
  regression gate).
- **`smoke-cpu`** runs before any tag exists, per arch, against the
  untagged digest images (`image@sha256:…` is pullable without tags), so a
  failed smoke leaves nothing public. `tools/smoke_cpu_services.py` asserts
  the `/healthz` identity fields (whisper `device: cpu`, titanet
  `engine: onnxruntime`), a real corpus transcription, and a titanet
  embedding within cosine 0.999 of the committed CUDA reference, proving the
  ONNX graph executes on the shipped numerical stack. A `low_snr` skip
  response counts as failure, not success. The app image is booted too
  (`import voxint`). pyannote smokes unconditionally (its weights are
  vendored into the image, fetched from the `pyannote-models-v1` asset
  release at build) and must produce a real 3-speaker diarization.
  `merge-multiarch` then tags only smoke-passed digests and verifies each
  manifest list exposes exactly `linux/amd64` + `linux/arm64`.
- **`publish-whisper-rocm`** builds the AMD whisper image (`-rocm`, amd64
  only) as build-only in CI: GitHub has no AMD-GPU runners, so its inference
  path cannot smoke there. The compensating gate is a maintainer-run
  real-GPU smoke on AMD hardware BEFORE tagging (Gate R): build
  `services/whisper/Dockerfile.rocm` on an AMD box, run it via the
  `compose.rocm.yaml` passthrough stanza against the parity corpus, and
  assert `/healthz device: rocm` plus a correct transcription at GPU speed.
  The CT2 ROCm wheel is sha256-pinned in the Dockerfile; the ROCm userspace
  debs are suite-pinned (`apt/7.0.2`) and the ubuntu:24.04 base floats, so
  the CI build is engine-identical (not byte-identical) to what was
  smoked. After the release publishes, optionally re-run the smoke against
  the published `X.Y.Z-rocm` tag on the AMD box.
- **Metal tier (Gate M)**: the metal tier ships no images at all (native
  services from the working tree plus the standard core images), so like
  ROCm it cannot smoke in shared CI. Unlike ROCm, the hardware *does* exist
  in GitHub's macOS arm64 runner pool, so the `metal-lane` workflow
  (`.github/workflows/metal-lane.yml`, nightly + manual dispatch on
  `macos-15`) automates the regression half: launcher unit tests on real
  macOS plus the three parity modules from the launcher's own per-service
  venvs, with an MPS tensor-op probe and a junit guard that fails the lane
  if an expected module green-boards fully-skipped. That lane catches drift
  *between* releases; the release gate itself stays a maintainer-run gate
  on Apple Silicon BEFORE tagging a release that touches the metal lane
  (CI runners are one chip generation, and the per-chip verdict report is
  the release artifact): with the
  tag checked out, `voxint-metal.sh setup && up && doctor`, then run the
  metal parity lanes from the metal venvs
  (`tests/parity/test_pyannote_metal.py`, `test_whisper_metal.py`,
  `test_titanet_onnx.py`, plus the two whisper-engine lanes from the
  `WHISPER_ENGINE` seam (#33): `test_whisper_ct2_legacy_replay.py`, which must
  replay the frozen CT2-CPU baseline with zero drift — run the full 15-AMI +
  synthetic sweep here, not just the fast synthetic subset — and, once shipping
  the shared `ct2` engine, `test_whisper_ct2_self_parity.py`, which must hold
  `ct2 ≈ ct2-legacy` to ≤0.5pp pooled WER per vad mode; on arm64; see
  docs/gpu-contracts.md "Metal tier"), and record/refresh the per-chip verdict
  report. `VOXINT_PARITY_REQUIRED`
  is deliberately never set for these lanes; the compensating control is
  this gate being listed here and the PENDING verdict table in
  gpu-contracts.md, which a release must not leave stale.

### E2E gate (Gate E — whole pipeline, maintainer-run)

The per-service smokes (`smoke-cpu`, Gate R) prove each model service in
isolation; the parity gates prove numerics. Neither proves the **whole
pipeline** — submit → PREPARE → transcribe → diarize → embed → persist — holds
together against the real services. `tests/e2e/` is that gate. It is
**maintainer-run and never wired into CI** (GitHub has neither GPUs nor the
weights), so it runs on maintainer hardware BEFORE tagging.

Before tagging a release that touches `services/` or the pipeline stages, bring
up the three model services on a lane the host supports (the maintainer's
host-specific bring-up — compose overlays and CPU limits — lives outside this
public repo) and run the real-pipeline lane against a disposable database:

```bash
export VOXINT_TEST_DATABASE_URL="postgresql+psycopg://voxint:voxint@127.0.0.1:5432/voxint_e2e"
VOXINT_E2E=1 uv run --extra dev pytest tests/e2e -q
```

Expect COMPLETED runs with the persistence invariants intact and no
model-service restarts. Keep it serial. `VOXINT_E2E=1` makes a missing
prerequisite a hard failure, not a skip — see
[`testing.md`](testing.md#automated-e2e-testse2e). (The LLM and browser lanes of
this suite are still being built; this gate currently covers the pipeline lane.)
Gate E follows the same carry-over rule as Gates A/R below: an empty
`git diff vPREV..main --stat -- services/` carries the previous evidence.

### Gate-evidence carry-over

The maintainer-run gates re-verify the *model services*, so they re-run only
when what they measure could have changed. Before tagging, check
`git diff vPREV..main --stat -- services/`:

- **Empty** → Gates A (CUDA reference regeneration), R (ROCm smoke), and E
  (whole-pipeline E2E) carry over from the previous release's evidence; the new
  images are rebuilds of the same numerics (CI's parity + smoke jobs still run
  unconditionally and prove the rebuild). Record the carry-over and the commit
  range it rests on
  in the release-commit message (v0.10.0 is the precedent: `services/`
  untouched since v0.9.0, A/R carried, Gate M satisfied by the committed
  per-chip verdict plus a green `metal-lane` run on the pre-bump commit).
- **Non-empty** → the affected gates re-run in full before tagging. Same
  conditional already stated for Gate M above ("touches the metal lane",
  which includes `scripts/metal/`, the metal parity lanes, and
  `metal-lane.yml`, not just `services/`).

Carry-over is an evidence judgment, not a loophole: anything that shifts the
numerics outside `services/` (parity fixtures, reference payloads, pinned
model assets) voids it for the gate it feeds.

## Cutting a release

1. **Release commit** on `main`: bump the version in `pyproject.toml` AND
   `src/voxint/__init__.py`, and bump the `VOXINT_IMAGE_TAG` default pin in
   `compose.yaml` + `compose.gpu.yaml` + `compose.cpu.yaml` +
   `compose.rocm.yaml` (and its
   `.env.example` comment) to the new version, so the default stack always runs
   the release this checkout documents. Run the gates (`ruff` / `mypy` /
   `pytest` with the pgvector test DB) and both gitleaks scans
   (`gitleaks dir .` and `gitleaks git .` with `.gitleaks.toml`).
2. Push to both remotes; wait for `ci` to go green on GitHub.
3. **Tag**: `git tag -a vX.Y.Z -m "Voxint vX.Y.Z" && git push github vX.Y.Z`
   (push the tag to the private origin too). The tag must point at the release
   commit so images are built from exactly what the compose files pin.
4. **Watch `release.yml`** (3 CUDA matrix jobs + 1 rocm build + 2 parity
   runs + 8 per-arch multi-arch builds + 2 per-arch smokes + 4 merges;
   smoke runs BEFORE merge, on digests; whisper builds are the slow ones,
   25–45 min each).
   `fail-fast` is off, so one failed matrix entry leaves the others published.
   A failure in `docker/metadata-action` *before* the build step has been
   transient GitHub infrastructure: re-run failed jobs
   (`gh api -X POST repos/…/actions/runs/<id>/rerun-failed-jobs`; this `gh`
   version's `run rerun` has no `--failed` flag).
5. **Verify anonymous pull** of all published images, with no login: app
   (both arches), three CUDA, three `-cpu` (both arches), whisper `-rocm`.
   The packages inherit public visibility from the repo via the
   `org.opencontainers.image.source` label, but confirm it: fetch each
   manifest with an anonymous GHCR token and expect 200. Optionally
   `docker run --rm ghcr.io/bengizmo/voxint:X.Y.Z python -c "import voxint; print(voxint.__version__)"`.
   Frontend smoke (issue #48): the review console loads with Tailwind styling
   and the transcript-player island hydrates over its server-rendered fallback;
   `docker run --rm ghcr.io/bengizmo/voxint:X.Y.Z sh -c 'command -v node || echo NO-NODE'`
   prints `NO-NODE`, proving no Node ships in the runtime image.
6. **PyPI**: `rm -rf dist && uv build && uv publish --token <pypi-token> dist/*`,
   then check `https://pypi.org/pypi/voxint/json` reports the new version.
7. **GitHub Release**: `gh release create vX.Y.Z --title "Voxint vX.Y.Z" --notes …`
   and update `CHANGELOG.md` in the next commit if it wasn't part of the
   release commit.
8. Optional: mirror the app image to Docker Hub
   (`docker tag … bengizmo/voxint:X.Y.Z && docker push …`). GHCR is canonical;
   the multi-GB GPU images are GHCR-only.

## Gotchas

- **Workflow smoke-testing**: a pre-release tag like `v0.0.0-test` runs the
  whole pipeline safely (no `X.Y` mutable tag). GHCR versions of public
  container packages can NOT be deleted through the REST API (422), only via
  the web UI, so don't mint test tags casually.
- The compose pin means a release is *self-referential*: the images the tag
  builds are the ones the tagged compose files pull. Step 1's pin bump is what
  keeps that true; tagging without it ships compose files that run the
  previous release.
- PyPI publishing is deliberately manual (no long-lived token in CI). If that
  changes, prefer PyPI trusted publishing over a stored secret.
