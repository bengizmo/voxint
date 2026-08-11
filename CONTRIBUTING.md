# Contributing to Voxint

Thanks for your interest! Voxint is pre-alpha; expect churn until `v0.1.0`.

## Development setup

```bash
uv sync --extra dev          # install everything (Python >= 3.11)
uv run ruff check .          # lint
uv run mypy                  # strict type-checking
uv run pytest tests/unit     # unit tests
uv run pytest tests/contracts  # GPU-service contract tests (CPU-only, no GPU deps)
```

Integration tests need the compose stack: `docker compose up -d postgres redis`.

The contract tests exercise the GPU services' schemas, path containment, error
mapping, and route behavior without torch or model weights — they load each
service's torch-free modules straight from `services/*/app/`. The main package
targets Python >= 3.11; the service images pin their own interpreters
independently (currently Python 3.10, dictated by the CUDA base images).

## Ground rules

- Type hints are mandatory; `mypy` runs in strict mode.
- New code needs unit tests (CI enforces 85% coverage).
- No hardcoded endpoints, paths, or credentials — everything enters via `voxint.config.Settings`.
- Secrets never land in the repo; CI runs gitleaks with repo-specific rules (`.gitleaks.toml`).
- Model weights are never vendored — they download at build/run time under the user's own
  credentials (see NOTICE).

## License

By contributing you agree your contributions are licensed under Apache-2.0.
