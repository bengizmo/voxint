# Plan — Voxint end-to-end (E2E) test suite

> Reviewed by a second model (planner role) — feedback folded in: reordered
> around a host-safety canary, corpus generator deferred, LLM gate tightened to
> the chain (not duplicated shape validators), browser boundary corrected,
> fail-not-skip gate semantics. Host-specific invocation (which maintainer box,
> CPU limits, the LLM proxy URL) is kept out of this public repo, in a
> maintainer-only runbook.

## Context

Two whole classes of Voxint regression escape **every** current gate:

1. **Frontend island runtime behaviour** — the #53/#58 verify-and-advance
   keyboard loop, click-a-line-to-move-cursor, unsaved-edit discard warning,
   keymap guards. Covered only by Python *route-level* integration tests (which
   never render the islands) plus a **hand-run manual browser pass**.
2. **Enrichment against a real LLM** — every LLM path is tested with
   `FakeLLM`/`FailingLLM`. Endpoint/proxy wiring, real response-schema
   compatibility, durable job success/failure, and staleness are unproven.

This suite closes both as a **maintainer-run, opt-in gate** — never public CI,
never operator ceremony. No-bloat: reuse the skip-when-unconfigured pattern, the
**existing committed tutorial WAV**, and the synchronous CLI enrichment seams;
add no frontend test runner; defer a fixture generator until a gap demands it.

Decisions locked: full real pipeline; real LLM via an internal LiteLLM proxy to
a local model (base URL from a gitignored maintainer config). Model-service host
chosen by measurement (Phase 0 canary), not assumption.

## Guiding corrections from the review

- **`restart: unless-stopped` recovers a container crash, not a host reset;
  concurrency=1 does not cap pyannote/TitaNet CPU parallelism or package power.**
  Host safety must be *measured*, not assumed (Phase 0).
- **A remote model host is not just a URL swap** — model requests carry paths
  relative to `MEDIA_ROOT`; a remote host must see identical audio at the same
  relative path (shared mount). A local model host avoids this entirely.
- **Most shape invariants (summary length, topic bounds, quote grounding,
  name-hint evidence) are already enforced deterministically** in
  `run_assets.py`/`run_assets_llm.py`/`names_llm.py` and covered by integration
  tests. Re-asserting them against a live model just checks one lucky sample.
  Gate the **chain**, characterize the **semantics**.
- **Browser: keep interaction + immediate DOM/network assertions together**; use
  DB only for durable-state reconciliation. The Claude skill is a **thin wrapper**
  around a tool-neutral workflow, not the canonical spec.
- **CLI names (verified):** `voxint enrich assets <run>`,
  `voxint enrich names <run> --llm`, `voxint research speaker <id>` (NOT
  `enrich research`). The canary uses `voxint tutorial seed` +
  `voxint submit`/`status`.

## Phased plan (each phase gates the next)

### Phase 0 — Host-safety + topology canary (do FIRST) — DONE

- Brought up the real services; asserted health identities
  (`whisper.device=rocm`, `pyannote.device=cpu`, `titanet.device=cpu`,
  `model_loaded=true`), no device fallback.
- Ran `sample-3speaker.wav` through all three real services twice, serially,
  recording CPU load, temps, wall time, restarts.
- **Result: PASS** — peak load well below saturation, cool temps, 0 restarts, no
  host instability. CPU caps on the CPU-only services (pyannote + titanet) live
  in a maintainer-only compose override. Disposable `voxint_e2e` DB + disposable
  media root, never live data.

### Phase 1 — Minimal real-pipeline vertical slice — DONE

- Reuse `sample-3speaker.wav` as the sole input (no new corpus yet).
- Submit + execute the real pipeline **serially** against the disposable DB/media
  (in-process, not the live Celery worker).
- Assert **persistence invariants**: run COMPLETED; exactly one
  `preprocessed_audio` artifact normalized to 16 kHz mono; non-empty transcript
  segments; diarization turns all embedded in `titanet-large-v1`;
  `duration_seconds` populated by the real PREPARE stage. Ranges/shape, never
  exact transcript text.
- **Gate:** one full-pipeline run repeats **twice** with no leak. Committed as
  `tests/e2e/{__init__,conftest,test_real_pipeline}.py`.

### Phase 2 — One real-LLM protocol smoke (summary only)

- Run **summary** generation through the configured LiteLLM endpoint via
  `voxint enrich assets` (real `HttpLLMClient`, `llm=None` auto-build).
- **Hard-gate the chain, not the prose:** endpoint reachable through the real
  adapter; durable job reaches **succeeded**, `error` NULL; a **current** asset
  persists with expected kind, producer/prompt version, model alias,
  `config_snapshot`, and `source_content_hash`; asset **non-stale** immediately
  after generation; **one real operator correction** via the review route → asset
  becomes **stale** (`kinds_needing_generation` reappears); malformed model
  output → honest **failed** job (not partial success).
- **Semantic quality = reported characterization, never a blocking assertion.**
  Env gotcha: `enrich names --llm` gate is env-only
  (`enrichment_names_llm_enabled AND llm_enabled`); set `LLM_ENABLED=true` +
  `ENRICHMENT_NAMES_LLM_ENABLED=true` + `ENRICHMENT_RUN_ASSETS_ENABLED=true`.
- **Defer web research** from the gate (adds uncontrolled internet/search/DNS).

### Phase 3 — Browser runtime acceptance

- One **canonical tool-neutral** setup/seed/verify/cleanup workflow (a committed
  script or docs page — the single lifecycle source of truth).
- **Playwright MCP drives interaction + immediate assertions:** v/e/n/p advance,
  click-a-line moves cursor, discard warning on first attempt + advance only on
  second, focused `<select>`/textarea suppresses the global keymap, no verify
  request emitted while a guarded control has focus.
- **DB reconciles durable state:** the browser was the only writer; expected
  `segment_review_states` rows + corrected text persisted; progress agrees; a
  correction changes `source_content_hash` and marks assets stale.
- **The `.claude/skills/voxint-e2e-review/SKILL.md` skill is a thin adapter** that
  invokes the canonical workflow — public repo (clean-room), not the spec. Swap
  the interaction layer for a Python Playwright harness later if agent runs prove
  inconsistent, keeping the same seed + verifier.

### Phase 4 — Gate semantics + failure classification — DONE (pipeline lane)

- `tests/e2e/` **skipped only when `VOXINT_E2E` is absent** (mirror the
  `VOXINT_TEST_DATABASE_URL` dir-scoped skip).
- When `VOXINT_E2E=1`, **fail (not skip)** if DB, model services, LLM config,
  model identity, or browser capability is missing. Optional per-lane markers
  (pipeline / llm / browser) for partial diagnostics.
- Add **negative tests for the gate behaviour** before documenting it as a
  release requirement. (LLM/browser lanes still to come.)

### Phase 5 — Evidence-driven expansion (only after a demonstrated gap)

- Add fixtures **only** for regressions the earlier slices cannot detect; each
  new fixture must name a unique failure mode. Likely next: **silence** and a
  **sample-accurately mixed overlap** clip. Then — if justified — build
  `tools/generate_e2e_corpus.py` (extend the espeak-ng pattern of
  `tools/generate_tutorial_audio.py`: distinct voices → 16 kHz mono → measured
  timings + provenance sha, CC0, no runtime espeak dep).
- **Cut from the initial matrix:** correction-re-mine and claim-takeover are
  *workflow scenarios*, not audio classes; low-confidence-heavy and
  long/many-segment are unauthorable/already-covered. Keep workflow scenarios
  separate from audio characteristics; don't run every fixture through every
  pillar.

## Real-LLM flake mitigations (bake in from Phase 2)

- Temperature-0 does **not** guarantee deterministic local inference — never
  assert exact strings; never retry assertions invisibly.
- **Pin the LiteLLM alias to a concrete backend revision** and record the
  resolved model identity in the run report; a silent alias reroute is a flake.
- **Separate ASR/diarization acceptance from LLM acceptance** so a failure names
  the responsible boundary.
- Entity mentions / name hints may be legitimately empty for a valid response —
  don't gate on their presence.

## Run / gate / docs / split

- `tests/e2e/` = **maintainer gate** documented in `docs/release-process.md`
  ("E2E: real pipeline + real LLM + browser"), never public CI.
- Update `docs/testing.md` — the real thing replaces the "Planned" placeholder.
- **CHANGELOG untouched** (maintainer harness; no user-facing behaviour change).
- **Public vs internal:** generic harness + `tests/e2e/` + skill + (eventual)
  generator are public; host-specific invocation (which maintainer box, the
  CPU-limit compose override, the LLM proxy base URL) stays in a maintainer-only
  runbook. Never commit hostnames/IPs/keys.

## Files

- New (public): `tests/e2e/{__init__.py,conftest.py,test_real_pipeline.py,
  test_enrich_assets_real_llm.py}`, the canonical browser workflow, the review
  skill, this plan.
- Edit (public): `docs/testing.md`, `docs/release-process.md`.
- Internal-only: host-specific bring-up + CPU-limit override, LLM proxy URL,
  host runbook.
- Reuse (do not reinvent): `tests/integration/conftest.py`
  (skip/engine/truncate/`seed_onboarded`), `voxint tutorial seed` +
  `sample-3speaker.wav`, `src/voxint/clients/llm.py` (`HttpLLMClient`),
  the `voxint enrich …` / `voxint research speaker` CLI.

## Verification (end to end)

1. Phase 0 canary green — DONE.
2. `VOXINT_E2E=1 uv run pytest tests/e2e -q` → real-pipeline pass;
   **`VOXINT_E2E` unset → skips; bare `pytest` stays green**; **`VOXINT_E2E=1`
   with a missing dep → fails, does not skip**. Pipeline lane DONE.
3. Skill drives the verify loop in a real browser against disposable
   `voxint_e2e`; DB confirms durable state. (Phase 3.)
4. Landing gates: `cd frontend && npm run typecheck && lint && build`;
   `uv run ruff check src tests`; `uv run mypy`; full pytest. Multi-model review;
   commit; push both remotes.
