# Contributing to Voxint

Thanks for your interest! Voxint is pre-alpha; expect churn until `v0.1.0`.

## Development setup

```bash
uv sync --extra dev          # install everything (Python >= 3.11)
uv run ruff check .          # lint
uv run mypy                  # strict type-checking
uv run pytest tests/unit     # unit tests
```

Integration tests need the compose stack: `docker compose up -d postgres redis`.

## Ground rules

- Type hints are mandatory; `mypy` runs in strict mode.
- New code needs unit tests (CI enforces 85% coverage).
- No hardcoded endpoints, paths, or credentials — everything enters via `voxint.config.Settings`.
- Secrets never land in the repo; CI runs gitleaks with repo-specific rules (`.gitleaks.toml`).
- Model weights are never vendored — they download at build/run time under the user's own
  credentials (see NOTICE).

## License

By contributing you agree your contributions are licensed under Apache-2.0.
