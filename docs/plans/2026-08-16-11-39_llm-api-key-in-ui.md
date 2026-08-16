# Plan — #10 slice 1: In-UI LLM API key (system-wide)

> Final home on approval: `docs/plans/2026-08-16-11-39_llm-api-key-in-ui.md`
> (copied verbatim, including Review notes). Branch: `feat/llm-in-ui-config`
> (already cut from `main @ 76275bf`). Issue: #10.

## Context

Voxint's optional LLM enhancement layer is configurable from the UI for base
URL and model, but the **API key is env-only** (`LLM_API_KEY`): the setup wizard
deliberately never stores it, and `AppSettings` has no column for it. For the
target audience — non-technical, single-operator, local-first — enabling LLM
enhancement therefore still requires hand-editing `.env` and restarting the
worker, the one piece the wizard cannot manage. This slice closes that gap: let
the operator enter/replace/remove the LLM API key **from the UI**, persist it on
the `app_settings` row, and have every LLM client construction resolve the
effective key with **DB row wins, env `LLM_API_KEY` as seed/fallback** (mirroring
how `llm_enabled` is already taken hard from the row).

**Scope decision (user, this session): close ALL callers now.** The stored key
becomes truly system-wide — it reaches transcript enhancement *and* the LLM
enrichment producers (names / web-research / run-assets) *and* `voxint doctor`.
This avoids the footgun of a "saved key" that silently works for enhancement but
not for enrichment/doctor. (Codex flagged the labeled-boundary alternative as the
only other defensible option; the user chose full consistency.)

**Anti-bloat:** no encryption-at-rest, no new deps. Plaintext-at-rest in Postgres
is explicitly accepted for this single-operator, local-first deployment; SQL
dumps/backups necessarily contain it — documented, not defended against.

## Precedence & secret rules (load-bearing)

- Effective value: `stored` iff the row column is **non-blank**, else env. NULL is
  the *only* representation of "no override" — clear/blank input canonicalizes to
  NULL, never `""` (codex #2).
- The key is a credential: **never** prefilled, rendered, logged, exported, or put
  in an error/validation message. Validation messages are **fixed strings that
  never interpolate the submitted value**; `redaction.redact(..., extra_secrets=…)`
  is only a backstop if an exception could carry it (codex #6).
- Keep the key **out of `RunPreferences`** and out of any dataclass with a repr /
  `asdict` / Celery serialization; thread it as a keyword-only `str` (codex #7).
  Never snapshot it into a job row (`job.config` / `job.budget`) — resolve it
  **live** at execution.

## Approach

A single pure resolver, then thread the resolved value through every LLM client
construction site. Each site already has a DB session (worker run, enrichment
`execute_job`, API asset route) or an engine (`voxint doctor`) to load the row.

### A. Schema + config
- **Migration `0016_app_settings_llm_api_key`** (`down_revision = "0015"`): add
  `llm_api_key` nullable `TEXT`; downgrade drops it. Pattern:
  `alembic/versions/0013_run_archived_at.py`.
- **`db/models.py` `AppSettings`** (~:1021): add
  `llm_api_key: Mapped[str | None] = mapped_column(Text)`. Rewrite the class
  docstring (currently "the LLM API key is never stored here"): key **may** be
  stored, row wins, NULL = env fallback, plaintext-at-rest accepted per charter,
  never rendered/logged/exported.

### B. Precedence helpers (single source) — `app_settings.py`
- `resolve_effective_llm_api_key(row, settings) -> str`:
  `(row.llm_api_key or "").strip() or settings.llm_api_key.strip()` → canonical
  stripped effective key (trim surrounding ws; empty ⇒ env; codex #4).
- `resolve_effective_llm_endpoint(row, settings) -> tuple[str, str]` (base_url,
  model) — non-secret; DRY the base_url/model precedence already inlined in
  `context.resolve_run_preferences` and `app.py:_setup_context` to call this.
- `effective_llm_key_source(row, settings) -> "stored" | "environment" | "none"`
  for honest UI copy (`stored` only when the **row** value is non-blank).

### C. Wizard field semantics (`api/setup_wizard.py`, pure)
- `normalize_llm_api_key(raw) -> str | None`: strip; blank ⇒ `None` (**no change**
  sentinel); reject remaining whitespace/control chars; `maxlength`/cap at
  `MAX_LLM_KEY_CHARS = 512`; else return canonical stripped value.
- `validate_llm_enable(effective_api_key: str, settings) -> None`: presence check
  on the **passed effective** key (post-save: submitted-or-stored-or-env) + the
  unchanged budget check via `llm_budget_fits_stage_lease`. Update both callers.
- Form inputs: password `llm_api_key` (never prefilled, `autocomplete="new-password"`,
  `maxlength=512`) + explicit `remove_llm_api_key` checkbox labeled **"Remove
  saved key (revert to environment)"**. **Reject `remove` + non-blank key** as a
  contradictory submission (codex #3).

### D. Atomic route logic (`api/app.py`) — codex #1 (HIGH)
API sessions commit on every successful response *including error re-renders*, so
mutation-before-return persists. Both routes therefore **compute a candidate
state, validate, then perform ONE deliberate mutation**:
1. Parse: `base_url`/`model` via existing normalizers; key via
   `normalize_llm_api_key` + the remove flag → candidate key = new value / remove
   (→NULL) / unchanged (existing row value).
2. Compute effective key from the candidate; if `enabled`, run
   `validate_llm_enable(effective_key, settings)`.
3. On validation failure: **persist the valid non-secret overrides and the valid
   candidate key** (a good key the operator typed is not thrown away), force
   `llm_enabled = False`, re-render with the fixed reason. On success: persist
   candidate key + overrides + `llm_enabled`. Every success/failure combination
   has explicit, tested persistence behavior.
- `_setup_context` (:726): `llm_key_present` becomes effective via the resolver;
  add `llm_key_source`. **Never pass the value to a template.**
- **New `POST /settings/llm`** (CSRF `CSRF_SETTINGS`) + an LLM section in
  `settings.html` (today it has none), reusing the same normalize/validate/atomic
  helpers so the key/enable/base_url/model are editable **after** onboarding.
  Extend `settings_page` context with effective present/source + enabled/base_url/
  model. 303-redirect on success.
- `setup.html` LLM step: replace "env-only, set it and restart" copy with the
  password field, remove checkbox, and masked `set (stored|environment) / not set`
  status.

### E. Thread the effective key through every client construction (system-wide)
Resolver from B, row loaded at each site:
- **enhance_match**: `apply_run_preferences(base, settings, prefs, *, llm_api_key)`
  uses the passed key for both the presence check and `HttpLLMClient` build
  (`context.py` ~:237/:241). `tasks.run_pipeline` resolves it in the existing
  `with factory() as session` block (:149) and passes it.
- **run-asset jobs** (`enrichment/asset_jobs.py:313`): resolve effective key
  **live** from the row in the existing `execute_job` session; base_url/model come
  from `job.config` — so make `create_jobs` snapshot the **row-resolved** endpoint
  at enqueue (it already has a session) so the operator's UI endpoint/model is the
  enqueue contract. Key never enters the snapshot.
- **web-research jobs** (`enrichment/research_jobs.py:349`): resolve key +
  base_url/model **live** from the row in the `execute_job` session (research
  already reads these live, not from its budget snapshot).
- **names LLM pass** (`enrichment/producers/names_llm.py:194`): caller resolves
  effective config from the row and injects it (extend the existing `client`
  injection seam, or pass effective key/base_url/model).
- **doctor** (`diagnostics.check_llm`): `run_diagnostics` opens a short session
  from its `engine` to read the row; `check_llm` gates on **effective** enabled
  and sends `Bearer <effective key>`. The base URL is still never printed.

### F. Ancillary
- `.env.example`: note `LLM_API_KEY` is now also settable in-UI (DB row wins;
  env is the seed/fallback).
- CHANGELOG `[Unreleased]`; docs touch (`docs/` operations/setup) in the same
  change per the stale-docs-are-bugs rule.

## Affected files
- `alembic/versions/0016_app_settings_llm_api_key.py` — new migration
- `src/voxint/db/models.py` — column + docstring
- `src/voxint/app_settings.py` — resolver helpers
- `src/voxint/pipeline/stages/context.py` — `apply_run_preferences` key param; DRY endpoint resolution
- `src/voxint/worker/tasks.py` — resolve + pass key in `run_pipeline`
- `src/voxint/api/setup_wizard.py` — `normalize_llm_api_key`, `validate_llm_enable` signature
- `src/voxint/api/app.py` — atomic `/setup/llm`; new `/settings/llm`; `_setup_context`, `settings_page`
- `src/voxint/api/templates/{setup,settings}.html` — password field, remove checkbox, masked status
- `src/voxint/enrichment/asset_jobs.py`, `research_jobs.py`, `producers/names_llm.py` — effective key/endpoint
- `src/voxint/diagnostics.py` — row-aware `check_llm`/`run_diagnostics`
- `.env.example`, `CHANGELOG.md`, `docs/`

## Implementation order (each step individually reviewable, commit per step)
1. Migration 0016 + model column + docstring. (`alembic heads` == single 0016.)
2. Resolver helpers in `app_settings.py`; DRY existing base_url/model resolution.
3. Worker/enhance_match: `apply_run_preferences` key param + `run_pipeline` wiring.
4. Wizard pure helpers (`normalize_llm_api_key`, `validate_llm_enable` signature).
5. Atomic `/setup/llm` + `setup.html`; then new `/settings/llm` + `settings.html`.
6. Close remaining callers: asset_jobs (enqueue endpoint snapshot + live key),
   research_jobs, names_llm, diagnostics/doctor.
7. `.env.example` + CHANGELOG + docs.

## Testing strategy (gates)
- **Unit**: `normalize_llm_api_key` (blank→None/no-change, reject ws/control, cap
  512, canonical passthrough); `resolve_effective_llm_api_key` precedence (non-blank
  row wins, blank row→env, no row→env, surrounding ws trimmed); `effective_llm_key_source`;
  `validate_llm_enable(effective_key, …)` incl. **stored key present + env absent →
  enable OK**, and budget fit/non-fit with env-disabled + stored key (codex #12).
- **Migration** (codex #9): upgrade **preserves an existing populated singleton**
  and sets the new column NULL; key round-trips; downgrade drops; reflected
  TEXT/nullable parity; single Alembic head. Pattern: `tests/integration/test_migration_00XX.py`.
- **Routes** (codex #10): `/setup/llm` and `/settings/llm` — store key; remove→env
  fallback; **invalid replacement / budget fail preserves the prior stored key**
  and forces `llm_enabled=False`; `remove`+non-blank rejected; missing/invalid CSRF
  and unauthorized preserve state; success 303 redirect; GET/POST HTML never echoes
  a sentinel key.
- **Acceptance — real wire** (codex #13): via httpx `MockTransport`/injected client,
  assert the actual request carries `Bearer <stored-key>` with env unset; a
  replacement takes effect on the **next** run/job without worker restart; removing
  the stored key switches to `Bearer <env-key>` live. Cover all five client sites
  (enhance_match, asset job, research job, names pass, doctor).
- **Secret-absence** (codex #5): a sentinel stored/submitted key is absent from
  GET/POST HTML, validation-error responses, **captured logs**, run/transcript
  **exports**, and **`voxint doctor` output**. Plus `repr(AppSettings(...))`.
- **Gates**: `ruff check .`, `mypy src/`, `pytest` (≥85% on changed files), both
  `gitleaks dir .` and `gitleaks git .` before any release. Full suite on a
  throwaway DB (never live `voxint`), per the #12 harness pattern.

## Risks / open questions
- **Enqueue-time endpoint snapshot for asset jobs** (E, bullet 2) is the subtlest
  change — verify `create_jobs` resolves the row-endpoint at enqueue while the key
  stays live at execution; a test must prove the key is *not* written to `job.config`.
- **`_settings_from_snapshot`** overlays snapshot onto env `Settings`; confirm the
  overlaid `exec_settings.llm_base_url/model` reflect the row and only the key is
  resolved separately (no double source).
- Plaintext-at-rest accepted per charter — restate in migration + model docstring.
- Browser hygiene (`autocomplete=new-password`, `maxlength`) is best-effort, not a
  guarantee (codex #11).

## Verification (end-to-end, after implementation)
1. `uv run --extra dev alembic upgrade head` on a throwaway DB → column present, NULL.
2. Start API; walk `/setup` LLM step: with **no env key**, enter a key, enable →
   persists, status shows "set (stored)". Restart worker not required.
3. Submit a run needing enhancement; confirm the worker's `HttpLLMClient` uses the
   stored key (log/inspect via MockTransport in tests; manually via a local
   OpenAI-compatible endpoint if available).
4. `voxint doctor` → LLM check reflects the stored (row) key/enabled, not env.
5. `/settings` LLM section: replace key, then remove key → reverts to env; both
   take effect on the next run without restart.
6. Grep captured logs / rendered HTML / an export for the sentinel key → absent.

## Review notes (codex planner critique — clink, 2026-08-16)
- **#1 atomic POST (HIGH)** — accepted: candidate-state-then-single-mutation; every
  success/failure persistence path defined and tested (§D).
- **#8 scope (HIGH)** — surfaced to user; user chose **close all callers now**
  (§E) over the labeled enhancement-only boundary.
- **#2 NULL-only** — accepted (§Precedence). **#3 remove vs checkbox** — accepted:
  labeled "Remove saved key", reject remove+replacement (§C). **#4 whitespace** —
  accepted: trim surrounding, reject remaining, canonical stripped (§B/C).
- **#5 secret tests / #6 fixed messages / #7 keyword-only str** — accepted
  (§Secret rules, §Testing). **#9 migration tests / #10 route tests / #13 real
  header** — accepted (§Testing). **#11 browser hygiene / #12 budget cases** —
  accepted.
- Confirmed by codex: no application path serializes `AppSettings` today (narrow
  leak surface); no correctness conflict with the env-time `_llm_budget_fits_stage_lease`
  validator (it stays env-`llm_enabled`-gated; the per-run worker guard is
  authoritative). Also noted: the "per-migration test" convention is uneven
  (no 0013/0015 test) — we still add a 0016 test.
