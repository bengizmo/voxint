# Contributing to Voxint

Thanks for your interest. Voxint is pre-alpha, so expect churn through the 0.x
series. The full documentation index is [docs/README.md](docs/README.md).

## Development setup

```bash
uv sync --extra dev          # install everything (Python >= 3.11)
uv run ruff check .          # lint
uv run mypy                  # strict type-checking
uv run pytest tests/unit     # unit tests
uv run pytest tests/contracts  # GPU-service contract tests (CPU-only, no GPU deps)
```

Integration tests need the compose stack: `docker compose up -d postgres redis`.

The default compose files run the pinned *release* images, not your working
tree. To run your checked-out code as the full stack, layer the build overlays.
See "Release images vs. building from source" in
[docs/operations.md](docs/operations.md).

The contract tests exercise the GPU services' schemas, path containment, error
mapping, and route behavior without torch or model weights. They load each
service's torch-free modules straight from `services/*/app/`. The main package
targets Python >= 3.11; the service images pin their own interpreters
independently (currently Python 3.10, dictated by the CUDA base images).

## Ground rules

- Voxint serves individuals and small teams (non-technical researchers,
  journalists, and educators) running locally hosted audio intelligence on their
  own hardware. Avoid bloat: every new dependency, feature, or configuration
  knob must earn its place for that audience. When in doubt, leave it out.
- Type hints are mandatory; `mypy` runs in strict mode.
- New code needs unit tests (CI enforces 85% coverage).
- No hardcoded endpoints, paths, or credentials. Everything enters via
  `voxint.config.Settings`.
- Secrets never land in the repo; CI runs gitleaks with repo-specific rules
  (`.gitleaks.toml`).
- Model weights are never committed to git. They are vendored into the service
  images at build time from sha256-pinned standing release assets, with
  provenance files recording upstream revisions and license attribution (see
  NOTICE and the provenance JSONs referenced there).

## Documentation

Docs are part of the change, not an afterthought. Update `docs/` in the same
commit that alters the behavior it documents; stale docs are treated as bugs.

Voxint docs follow a two-lane house style. User-facing docs (README, `docs/setup.md`,
`docs/onboarding.md`, `docs/how-to/*`, installer and first-run copy) are written
for a non-technical reader: warm, jargon-free, outcome-focused. Technical docs
(this file, `docs/architecture.md`, the contract and operations references) are
written for a developer: direct, code-first, with in-line parameter tables.

Two rules hold in both lanes: no emdashes, and no LLM-isms (negative
parallelism, rule-of-three padding, empty summary closers, over-signposting).
Prose stays emoji-free apart from a `⚠️`/`✅` callout glyph. The full guidance,
including per-lane structure and a pre-publish checklist, lives in the
`voxint-docs` skill; agents should load it before writing or editing docs.

## Reporting security issues

Do not open a public issue for a vulnerability. Report it privately via GitHub
Security Advisories, as described in [SECURITY.md](SECURITY.md).

## License

By contributing you agree your contributions are licensed under Apache-2.0.
