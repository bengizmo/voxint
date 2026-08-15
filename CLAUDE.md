# Voxint — Project Instructions

> **Docs index**: [docs/README.md](docs/README.md) · **Release process**: [docs/release-process.md](docs/release-process.md) · **Changelog**: [CHANGELOG.md](CHANGELOG.md)

## Project Overview

**Voxint** is a self-hosted, open-source audio-intelligence pipeline: media in →
transcription (faster-whisper) → speaker diarization (pyannote) → speaker
embedding/identification (TitaNet) → a single-operator review console for
adjudicating results. Apache-2.0, single-operator deployments by design —
enterprise-scale security and scalability are explicitly not primary concerns;
correctness, numerics stability, and non-technical onboarding are.

Public repo: `github.com/bengizmo/voxint` (releases are cut there — `release.yml`
only runs on GitHub). Development also pushes to a private origin: **always push
both remotes**. Agents may commit and push in this repo.

## Layout

```
src/voxint/            # app: API + review console, pipeline, DB (alembic), ingest
services/{whisper,pyannote,titanet}/   # model-service containers (own Dockerfiles per flavor)
compose.yaml           # core stack; overlays: compose.{gpu,cpu,rocm}.yaml + *.build.yaml
scripts/install.sh     # guided installer (Bash 3.2-compatible; test seam: VOXINT_INSTALL_LIB=1)
tools/                 # maintainer tools (parity reference generation, smokes, exports)
tests/{unit,contracts,integration,parity}/
docs/                  # architecture, gpu-contracts, operations, release-process, ...
```

## Working in this repo directly

Development sessions root HERE (`~/dev/voxint`), not in a parent project.

- **Internal notes** (session prompts, runbooks with machine names, reports,
  plans) live in `internal/` — a nested, gitignored, SEPARATE git repo pushed
  only to the private origin host. Nothing in `internal/` is ever part of
  this public repo, and committed content here must never reference it by
  URL or hostname.
- **Session prompts**: `docs/session-prompts` is a gitignored symlink into
  `internal/session-prompts/`, so `/next-session-prompt` and
  `/start-next-session` work unchanged from this root. After writing one,
  commit + push the `internal/` repo (its own git) so it follows you across
  machines.
- On a fresh clone, restore the pair:
  `git clone <private-origin>/voxint-internal internal` then
  `ln -s ../internal/session-prompts docs/session-prompts`.

## Hard Rules — public clean-room repo

- **Never** commit internal hostnames, IPs, credentials, tokens, or `.env` —
  gitleaks rules enforce this; run both `gitleaks dir .` and `gitleaks git .`
  before a release.
- In public issues/releases, never name internal machines — say "the reporting
  host" or "maintainer hardware".
- No `--force` pushes, no `--amend` on pushed commits, no `--no-verify`.

## Numerics Doctrine (the load-bearing invariant)

Model outputs are contract, not implementation detail. Any change that could
touch inference numerics needs **measured equivalence** evidence, not reasoning:

- **Parity gates**: `tests/parity/` (strict in CI via `VOXINT_PARITY_REQUIRED=1`)
  plus committed CUDA references (`tests/parity/fixtures/references/`). The
  titanet ONNX engine holds the `titanet-large-v1` embedding-space id on a
  measured 3-level gate (mel/vector/decision) — see `docs/gpu-contracts.md`.
- **Contract tests** (`tests/contracts/`) pin invariants that would otherwise
  rot silently: version-pin parity across pyproject/compose/.env.example,
  Dockerfile sha ARGs ↔ provenance files, restart policies, vendored-config
  paths and hyperparameters. When you add an invariant, add the contract test
  in the same commit.
- Whisper is pinned to **large-v2** (v3/turbo hallucinate); pyannote to
  **speaker-diarization-3.1 on pyannote.audio 3.1.1** (4.x drops the tuned
  clustering hyperparameters). These are deliberate pins, not lag.

## Model Weights — vendored, sha-pinned, immutable

Weights are **not in git**. Standing GitHub asset releases hold them:
`titanet-onnx-v1` and `pyannote-models-v1`, each with a provenance file
(`tests/parity/fixtures/onnx/provenance.json`,
`services/pyannote/models/provenance.json`) recording per-file sha256s,
upstream revisions, and license attribution (CC-BY-4.0/MIT — the attribution
ships inside the images too). Rules:

- Assets under a published tag are **immutable** — a weights refresh publishes
  a new release (`…-v2`), updates the provenance file + Dockerfile sha ARGs +
  the `*_RELEASE` env in `release.yml` together.
- ⚠ The vendored pyannote checkpoints must live under a **"pyannote"-named
  path** (`/app/vendored/pyannote/…`): pyannote.audio 3.1.1 dispatches
  embedding loaders on path substrings, and a wespeaker-named path without
  "pyannote" routes to the uninstalled ONNX loader. Contract-tested — never
  rename without re-running the offline smoke.
- No Hugging Face account/token in the default install; `DIARIZER_MODEL_NAME`
  (+ optional `HF_TOKEN`) restores the online path.

## Quality Standards

- **Python**: type hints mandatory; `uv` for all environments (never bare
  pip/virtualenv); `ruff` + `mypy` clean before landing.
- **Tests**: ≥85% coverage for new code; pytest layout above. Never weaken an
  assertion to make a test pass — a failing gate is information.
- **No mocks/stubs in production code**; no fallbacks that mask root causes;
  no hardcoded configurable values (env vars, documented in `.env.example`).
- **Honest UX copy**: installer/handoff/error text states what is actually
  true (e.g. a down service means submissions fail — say so; don't claim
  "downloading weights" when weights are baked).
- **Reviews**: non-trivial changes get a multi-model review before landing;
  never dismiss a finding without verifying it — record applied fixes and
  deliberate skips (with reasons) in the commit message.
- Commit work before dispatching background reviewers (they can stash-wipe
  uncommitted changes).

## Versioning & Releases

- Pre-1.0: **bump MINOR often** (0.5.0 → 0.6.0 …); PATCH is for pure fixes.
  Never propose 1.0.0 — that needs far more field validation.
- Version bumps are **atomic**: pyproject + `__init__.__version__` + all four
  compose `VOXINT_IMAGE_TAG` defaults + `.env.example` comment move together
  (pin-parity contract tests enforce this). Floating `X.Y` image tags exist —
  don't ship behavior changes under an already-published minor.
- Full checklist in `docs/release-process.md`: gates (ruff/mypy/pytest/
  gitleaks), maintainer GPU gates for the lanes CI can't smoke (CUDA and
  `-rocm` — GitHub has no GPU runners), tag → `release.yml` (smoke runs
  BEFORE tags exist, on digests) → verify anonymous pulls, PyPI, GitHub
  Release, optional Docker Hub mirror.
- Image naming: exact `X.Y.Z` tags in compose; no `-fixed`/`-optimized`-style
  suffixes anywhere.
- Build wall-clock: `release.yml`'s matrix jobs already run fully in
  parallel on GitHub-hosted runners — the critical path is the slowest
  single whisper image build, not serialization. If that ever needs to be
  faster, the sanctioned approach is switching the workflow's layer cache
  from `type=gha` to `type=registry` and pre-warming it from maintainer
  hardware before tagging (donates layers only — CI still builds, smokes,
  and publishes its own digests, so the gate ordering and provenance are
  unchanged). Details and the per-machine fan-out live in the maintainer's
  internal build runbook, not in this repo. Self-hosted runners are
  deliberately NOT used: this is a public repo, and runner boxes would
  enter the release supply chain.

## Development Workflow

- Feature branches, FF-merge to `main`; `main` is always releasable.
- CHANGELOG in Keep a Changelog format — update under `[Unreleased]` as part
  of the change, stamp on release.
- Update `docs/` in the same change that alters behavior (installer text,
  contracts, operations); stale docs are treated as bugs.
- The installer is Bash 3.2-compatible (macOS ships it) and sourced in
  library mode by `tests/unit/test_installer.py` — keep pure logic testable
  without Docker.
- Ask for help / stop and reassess when stuck >15 minutes; never claim "done"
  without checking the actual artifact (service health, DB rows, logs,
  anonymous image pulls).
