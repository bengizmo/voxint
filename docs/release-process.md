# Release process (maintainers)

How a Voxint release is cut. Development happens on the private origin; the
public mirror is `github.com/bengizmo/voxint` — releases are cut there,
because the image workflow only runs on GitHub (`release.yml` is guarded by
`github.server_url` and stays inert elsewhere).

## What a release produces

| Artifact | Where | How |
|---|---|---|
| `ghcr.io/bengizmo/voxint:X.Y.Z` (app: api/worker/beat/migrate) — **multi-arch amd64+arm64** | GHCR | `release.yml` `build-multiarch` → `merge-multiarch` |
| `ghcr.io/bengizmo/voxint-{whisper,pyannote,titanet}:X.Y.Z` (CUDA, amd64) | GHCR | `release.yml` `publish-images` matrix |
| `ghcr.io/bengizmo/voxint-{whisper,pyannote,titanet}:X.Y.Z-cpu` — **multi-arch amd64+arm64** | GHCR | `release.yml` `build-multiarch` → `merge-multiarch` |
| `voxint X.Y.Z` sdist + wheel | PyPI | manual `uv build` + `uv publish` (not in CI) |
| Release notes | GitHub Releases | manual `gh release create` |

Tag semantics (immutable exact-semver, per the non-NVIDIA plan): unsuffixed
model-service tags are **CUDA**; `-cpu` is the multi-arch CPU flavor; a future
`-rocm` is AMD. `docker/metadata-action` also emits mutable `X.Y` / `X.Y-cpu`
tags for full releases; compose files must keep referencing the exact `X.Y.Z`.
Pre-release versions (e.g. `v0.0.0-test`) get only their exact tag — useful for
exercising the workflow without polluting `X.Y`.

### The titanet model asset (release dependency)

The ~100 MB `titanet-large.onnx` is **not in git**. The workflow's parity gate
and the titanet `-cpu` build fetch it from the standing model-asset release
(**`titanet-onnx-v1`**) and verify its sha256 against
`tests/parity/fixtures/onnx/provenance.json`; the Dockerfile re-verifies at
build time. A re-export (new checkpoint or export pins) must publish a **new**
asset release (`titanet-onnx-v2`, …), update `provenance.json` and
`services/titanet/Dockerfile.cpu`'s `TITANET_ONNX_SHA256` default (a contract
test pins them together), and bump `TITANET_ONNX_RELEASE` in `release.yml` —
assets under an existing tag are immutable by policy, never replaced.

### Release gates wired into the workflow

- **`parity-gate`** — the strict titanet ONNX parity harness
  (`VOXINT_PARITY_REQUIRED=1`) runs natively on **both** amd64 and arm64
  runners and blocks every multi-arch build. The NeMo/CUDA reference side
  stays a maintainer-run gate: re-run the
  `tools/generate_parity_references.py` flow on an NVIDIA box against the new
  tag before releasing (Gate A, the NVIDIA regression gate).
- **`smoke-cpu`** — after the manifests merge, each arch boots the `-cpu`
  images, asserts the `/healthz` identity fields (whisper `device: cpu`,
  titanet `engine: onnxruntime`), and exercises a short-clip transcribe/embed.
  pyannote's smoke needs a repo `HF_TOKEN` secret (weights are HF-gated):
  present → real boot smoke; absent → **explicit SKIP** in the step summary,
  never a silent pass.

## Cutting a release

1. **Release commit** on `main`: bump the version in `pyproject.toml` AND
   `src/voxint/__init__.py`, and bump the `VOXINT_IMAGE_TAG` default pin in
   `compose.yaml` + `compose.gpu.yaml` + `compose.cpu.yaml` (and its
   `.env.example` comment) to the new version — the default stack always runs
   the release this checkout documents. Run the gates (`ruff` / `mypy` /
   `pytest` with the pgvector test DB) and both gitleaks scans
   (`gitleaks dir .` and `gitleaks git .` with `.gitleaks.toml`).
2. Push to both remotes; wait for `ci` to go green on GitHub.
3. **Tag**: `git tag -a vX.Y.Z -m "Voxint vX.Y.Z" && git push github vX.Y.Z`
   (push the tag to the private origin too). The tag must point at the release
   commit so images are built from exactly what the compose files pin.
4. **Watch `release.yml`** (3 CUDA matrix jobs + 2 parity runs + 8 per-arch
   multi-arch builds + 4 merges + 2 smokes; whisper builds are the slow ones,
   25–45 min each).
   `fail-fast` is off, so one failed matrix entry leaves the others published —
   a failure in `docker/metadata-action` *before* the build step has been
   transient GitHub infrastructure: re-run failed jobs
   (`gh api -X POST repos/…/actions/runs/<id>/rerun-failed-jobs`; this `gh`
   version's `run rerun` has no `--failed` flag).
5. **Verify anonymous pull** of all published images — app (both arches),
   three CUDA, three `-cpu` (both arches) — with no login:
   the packages inherit public visibility from the repo via the
   `org.opencontainers.image.source` label, but confirm — fetch each manifest
   with an anonymous GHCR token and expect 200. Optionally
   `docker run --rm ghcr.io/bengizmo/voxint:X.Y.Z python -c "import voxint; print(voxint.__version__)"`.
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
  container packages can NOT be deleted through the REST API (422) — only via
  the web UI, so don't mint test tags casually.
- The compose pin means a release is *self-referential*: the images the tag
  builds are the ones the tagged compose files pull. Step 1's pin bump is what
  keeps that true — tagging without it ships compose files that run the
  previous release.
- PyPI publishing is deliberately manual (no long-lived token in CI). If that
  changes, prefer PyPI trusted publishing over a stored secret.
