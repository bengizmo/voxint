# Voxint: LLM transcript translation (preferred language + auto-translate + translate button)

## Context

Voxint transcribes, diarizes, and identifies speakers, but a finished transcript is only useful in the language it was spoken in, and Voxint's audience (researchers, journalists) routinely works across languages. #124 just landed language *detection*; enrichment epic #41 explicitly deferred translation as out of scope while establishing the exact pattern it needs (immutable generations, source-content hashing, stale detection, regenerate-supersedes). Competitor research: Descript treats translation as a generated rendition of the finished composition requiring re-translation after edits; HappyScribe's differentiator is side-by-side original/translated review. The already-configured LLM (BYO or bundled, #10/#67) is the engine; no new dependencies.

**Decisions made with Ben:** segment-aligned per-line translation (not a document blob); auto-translate fires on pipeline completion with hash-based staleness + one-click re-translate; "preferred language" is a per-installation setting (Voxint is single-operator; `app_settings` singleton).

**Multi-model consult done (codex planner + grok-4.5, both reviewed the draft + code):** strong convergence; all agreed changes folded in below (see "Review notes" at bottom).

## Design

### 1. Settings: preferred language + auto-translate toggle

- `AppSettings` (src/voxint/db/models.py:1430): add `translation_target_language: Mapped[str | None]` (nullable-Text override, like `llm_model`) and `translation_autogenerate: Mapped[bool | None]` (tri-state nullable-Boolean; template = `semantic_index_autogenerate`).
- Env fallbacks in `Settings` (src/voxint/config.py): `VOXINT_TRANSLATION_TARGET_LANGUAGE` (unset), `VOXINT_TRANSLATION_AUTOGENERATE` (false). Document in `.env.example`.
- `src/voxint/app_settings.py`: `resolve_effective_translation_target_language` / `resolve_effective_translation_autogenerate` via existing `_resolve_str_flag`/`_resolve_bool_flag`; invariant validator modeled on `semantic_index_flags_ok`: autogenerate requires a target language AND an LLM path (`byo_llm_configured` or `llm_bundled_active`). Validate the language code against `LANGUAGE_NAMES`.
- Settings UI: new partial `settings/_translation.html` (include from settings.html), modeled on `_semantic.html` (tri-state radio) + a `<select>` from `LANGUAGE_NAMES` (src/voxint/api/languages.py). Route `POST /settings/translation` cloned from `settings_semantic` (app.py:6068). Honest copy: "Translates the transcript as it is when the pipeline finishes. Re-translate after you finish corrections."

### 2. Storage: TWO tables (migration 0038), lines as versioned JSONB

Clone the `RunAssetJob` pattern (do NOT reuse its tables — translation has target-language identity and its own gates; both consult models agreed):

- **`translation_jobs`** (`TranslationJob`): run FK, `target_language`, status enum (queued/running/succeeded/failed/cancelled), `job_config_snapshot` (freeze LLM config at enqueue, like asset_jobs.py `job_config_snapshot`), `source_content_hash` (frozen at enqueue), `error`, timestamps; partial unique index = one active job per (run, target_language). CAS claim (`claim_job` pattern), `request_cancel`, single-attempt.
- **`run_translations`** (`RunTranslation`): run FK, `target_language`, `generation` (monotonic per run+language), `source_language` (from `detected_language` at generation time, nullable), `model`, `producer_version`, `payload_schema_version`, `source_content_hash`, `superseded_by_id`, created_at; partial unique = one current head per (run, target_language) where `superseded_by_id IS NULL`. **`lines` JSONB**: `[{"i": line_index, "segment_id", "word_start", "word_end", "source_text", "text"}, ...]` with count/byte bounds enforced at record time.

**Identity/freshness contract** (both consult models, unanimous):
- `line_index` is the identity *within* a generation; `segment_id`/`word_start`/`word_end`/`source_text` are frozen provenance + diagnostics, never live join keys.
- The **run-level `source_content_hash` is the only freshness authority**: hash over the ordered structural + effective-text fields of `attributed_transcript(session, run_id, text=CORRECTED)` (src/voxint/adjudication/transcript.py:229) plus a schema version and source-language hint. **Exclude speaker names** from the hash (speakers are not sent to the LLM; reassigning a speaker must not stale a translation).
- Fresh (hash matches current transcript) → zip lines by order. Stale → the whole generation is stale: one banner + Re-translate; never partially align old lines to a changed transcript. Regeneration supersedes; rows are never edited.

### 3. Generation: producer + Celery task

- Producer `src/voxint/enrichment/producers/translation_llm.py` beside `run_assets_llm.py`:
  - Source = frozen in-memory snapshot of `attributed_transcript(..., CORRECTED)` lines taken once at job start.
  - Batch by the existing `llm_batch_max_segments`/`llm_batch_max_chars` knobs (budgeting output expansion conservatively). Empty source lines get empty translations deterministically, never sent to the model.
  - **`chat_json` requires a JSON object** (src/voxint/clients/llm.py:275, verified) — prompt sends numbered lines and demands `{"translations": [{"i": <index>, "text": "..."}]}`. Validate an **exact index set** (no missing/duplicate/unknown, ints not bools, strings only) — catches merges, drops, AND reorders, which a positional count check cannot. NUL check; per-line growth ceiling `max(3 x source_len, source_len + 500)` + the existing aggregate `MAX_CHAT_REPLY_CHARS`; non-empty source requires non-empty translation.
  - Failure ladder: retry per `llm_attempts_per_batch`, then **bisect the batch** (halve recursively down to 1 line; one retry at size 1, then fail the job). No fuzzy realignment ever; no partial generations persisted.
  - **Source-changed race guard**: recompute the current transcript hash immediately before recording/superseding; on mismatch finish the job as failed with a `source_changed` error and leave the previous generation current (an operator edited during generation).
  - Prompt names the source language from `detected_language` when present ("from the source language" otherwise), instructs preservation of names/numbers (prompt rule only; no NER guardrail).
- Job module `src/voxint/enrichment/translation_jobs.py` cloned from `asset_jobs.py` (`create_job`, `claim_job`, `execute_job`, `request_cancel`, `_finish`, config snapshot). `ChatJsonLLM` Protocol reused as the test seam.
- Celery task `translate_run(job_id)` in src/voxint/worker/tasks.py (thin wrapper like `generate_run_asset`, tasks.py:449).
- **Auto hook**: `_autogenerate_translation(factory, run_id, settings)` beside `_autogenerate_run_assets` in the post-segment COMPLETED branch (tasks.py:230-236). Gates: effective autogenerate on, target language set, LLM available, and skip when normalized `detected_language == target_language`.

### 4. Console surfacing + exports (fail closed)

- **Run detail card**: htmx fragment `fragments/run_translation.html` cloned from `fragments/run_assets.html` — current translation (language, model, generated-at, fresh/stale banner), Translate/Re-translate button with a language select defaulting to the preferred language (`POST /runs/{id}/translation/generate`), self-poll while active, cancel. Routes cloned from `run_assets_fragment`/generate (app.py:6689-6696).
- **Transcript page** (`transcript.html` + `TranscriptPlayer`): when a **fresh** generation exists, a view toggle renders the translated line beneath each original (interleaved side-by-side; no dual-column layout). Island props gain optional `translation: {language, lines: string[] (by line order), stale: bool}`. When stale: generation-level banner + Re-translate, translated lines NOT shown against the changed transcript. Server-rendered fallback mirrors this.
- **Review stepper** terminal block (`ReviewStepper.tsx` ~line 924): "Translate" action beside "Open the transcript to export" firing the same generate endpoint via `apiFetch`, then linking to the transcript view. Editing surfaces stay source-language only; translated text is never hand-editable (regenerate, don't edit).
- **Exports**: `?lang=<code>` on the existing `/review/{id}/export.{txt,md,srt,vtt,json}` wrappers substitutes translated text before `render_transcript` — **only when the current generation is fresh and complete; otherwise 409** (browser links disabled/absent when stale). Never per-line fallback to original; no mixed-language srt/vtt ever. `?lang` combined with `text=raw|enhanced` → 422 (translation is a rendition of corrected text). Subtitle cue text is substituted with no reflow (documented: translated subs may read fast). RTTM unaffected.

### 5. Not doing (v1)

Dubbing/TTS/lip-sync; hand-editing translations; multi-language UI (schema supports via target_language); incremental/copy-forward re-translation (defer until whole-run regen is measurably painful on the bundled model); per-line stale chips (generation-level banner only); cue reflow; fuzzy batch repair; mixed-language exports; parity gate (LLM output is non-deterministic generated content, same class as #41 summaries — provenance not numerics); new dependencies.

## Files touched (representative)

- `alembic/versions/0038_translation.py` (+ head-test bump)
- `src/voxint/db/models.py` (AppSettings cols, TranslationJob, RunTranslation)
- `src/voxint/config.py`, `.env.example`, `src/voxint/app_settings.py`
- `src/voxint/enrichment/translation_jobs.py`, `src/voxint/enrichment/producers/translation_llm.py`
- `src/voxint/worker/tasks.py` (task + auto hook)
- `src/voxint/api/app.py` (settings route, fragment + generate/cancel routes, export `?lang`, transcript props)
- `src/voxint/api/templates/settings/_translation.html`, `fragments/run_translation.html`, `transcript.html`, `run_detail.html`
- `frontend/src/components/TranscriptPlayer.tsx`, `ReviewStepper.tsx`
- CHANGELOG under [Unreleased]; docs: `docs/` how-to page for translation (voxint-docs house style)

## Testing

- Unit: producer batching/validation (fake ChatJsonLLM: merge, drop, duplicate/unknown index, refusal, fenced prose, oversized, NUL), bisection ladder, hash construction (speaker excluded; edits/splits change it), settings resolvers + validator.
- Contracts: routes registered, migration 0038 head + model parity, settings partial included, `.env.example` documents new vars.
- Integration: job lifecycle (create/claim/execute/cancel/supersede), source_changed race (edit mid-job leaves prior generation current), auto-hook gating (flag off / language match / LLM absent / no target), export fail-closed (409 stale, 422 lang+raw, fresh substitution correct in all 5 formats), stale-after-edit banner state.
- Browser lane (voxint-e2e-review, opt-in): translate button → poll → translated lines interleave; edit a segment → stale banner, translated export link gone/409.

## Implementation slices + GitHub issues (after approval)

1. Create feature issue "LLM transcript translation: preferred language, auto-translate, translate button" referencing #41's explicit deferral and #10/#67 (LLM config), with the two slices as a checklist. Comment on #41 linking it. (Also check #66 relevance — local-LLM qualification.)
2. **Slice A** (backend): settings + schema/migration 0038 + hash module + job lifecycle + producer + Celery task + auto-hook + run-detail card. Includes the source-changed race test and export fail-closed policy tests (so Slice B can't soft-mix).
3. **Slice B** (console): transcript interleaved view + island props + exports `?lang` + stepper Translate action + docs how-to.
4. Each slice: /unit-testing → /code-review (3-engine) → land per repo workflow (feature branch, FF-merge, both remotes). No version bump on the branches; translation ships in the next minor.

## Review notes (multi-model consult audit trail)

Panel: codex (clink planner, read plan + 11 files) + grok-4.5 (zen chat, high thinking). Convergent findings, all adopted:
- `chat_json` rejects non-object JSON → object envelope with echoed indices + exact-index-set validation (codex's ID-echo variant chosen over grok's positional count: catches reorders).
- Collapse 3 tables → 2 (lines as versioned JSONB on `run_translations`).
- `line_index` = within-generation identity; structural fields provenance-only; run-level hash is the sole freshness authority (my original per-line live-join keying was overfit to "survive edits").
- Whole-generation supersede only; no incremental re-translate v1 (grok: copy-forward as a later optimization; codex: defer until measured — deferred).
- Exports fail closed (409/absent when stale); never mixed-language output (both flagged my per-line fallback as a journalist-facing footgun).
- Batch bisection ladder; translation-specific growth ceiling; empty lines short-circuited.
- Codex-only adds adopted: source_changed pre-commit race guard; speaker names excluded from hash; 422 on `lang`+`text=raw|enhanced`; normalize codes before source==target skip.
Disagreements: none material.
