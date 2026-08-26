# Voxint UX Refactor: Descript-Inspired IA ("Console 2.0")

## Context

Voxint's console grew screen-by-screen around the pipeline (flat top nav: Dashboard, Resources, Runs, Review, Speakers, Settings, all in one 7.5k-line `app.py`). Ben wants a full IA refactor modeled on Descript's shell (reference screenshots in `internal/mockups/descript-reference/`): left sidebar **Home / Media / Projects / Speakers / Jobs / Settings**, library-style list views, and a full-transcript editor that replaces the two-step review flow. This supersedes the Aug-21 gap analysis's non-goals (no-projects, don't-merge-review) while keeping its findings (vocabulary pass, progressive disclosure, honest states, export-as-end-state — they bind every new screen).

Decisions settled with Ben: editor **replaces** review (with a stepper walk mode retained); **projects = group of media folders** (folder in ≤1 project) with project-scoped vocabulary/corrections; **plugins UI consumes epic #136's framework** (substrate landed on main); **Reading Room palette kept**, Descript structure adopted; **full file management** (move + delete, trash-guarded); **speaker confidence as qualitative tiers** + verified badge (numbers behind a reveal; upgrade path to #114); **hardware settings read + guided edit** (no self-restart); **one epic, phased slices, parallel agent tracks on separate branches**; extras in scope: global search, activity/toasts, drag-drop upload, command palette, export center, bulk actions, undo+trash, needs-attention nudges. **After approval, the first execution step is creating the GitHub issue set.**

A 3-model panel reviewed the draft (codex planner; deepseek-v4-pro; claude-sonnet clink — kimi-k3 unavailable, OpenRouter monthly key limit). Their corrections are folded in below; the consultation record is at the end.

## Current-state anchors

- Routes: single `src/voxint/api/app.py`; templates `src/voxint/api/templates/`; `base.html` holds the token layer + flat top nav; 4 React islands (`frontend/`, Vite multi-entry): `transcript-player`, `review-stepper`, `workbench-player`, `corrections-editor`, shared `WaveformStrip`/`AnnotationLayer`/`KeymapHelp`/`keymap.ts`.
- DB (alembic head 0038): `media_items` (**identity = unique `source_path`, ~126 refs across 20+ files; `sha256` nullable**), `pipeline_runs`/`stage_runs`, `transcript_segments`+`segment_review_states`, global `speakers` + embeddings/assignments/adjudications/candidates, enrichment tables, `app_settings` singleton (media_folders/vocabulary/corrections/flags).
- No project entity, no folder table, no move support, no per-speaker aggregation, no Celery broker introspection, no install-type detection. Plugin framework (`src/voxint/plugins/`) on main; #138–#142 + synthdetect in flight — every slice rebases on fresh origin/main.
- Tutorial banner is keyed to a hardcoded `STEP_PAGE` route map in app.py; 18 committed screenshots under `docs/images/` referenced from README/how-to docs.

## Target IA

```
/            Home: needs-attention first (continue review, unidentified speakers, failed runs),
             quick actions (Add media, New project, Review speakers), summary stats
             (media added / jobs run / speakers identified × hour/day/week/all), recent activity
/media       Library: folders + files, card/table toggle, sort, multi-select bulk actions, upload
/media/{id}  Editor (flagship): full transcript + speaker rail + verify/edit/split/annotate + export sheet
/projects    Project list        /projects/{id}  vocabulary + corrections, member folders, speakers
/speakers    Stats + reminders; roster card/table (name, files, minutes, tier, verified)
/speakers/{id}  Stats header; profile (manual + enrichment-accepted, provenance-tracked); media table
/jobs        Pipeline component status (DB-derived queued/active per stage + service admission) + runs table
/jobs/{id}   Run detail (stage ledger, provenance, requeue/cancel, assets, translation)
/settings    Hub → /settings/{status,hardware,database,plugins,plugins/{id}} + regrouped sections
```

Home is attention-first, not a stats dashboard — Jobs owns the operational detail (panel finding: don't build the same status twice). Legacy routes redirect as each phase lands; **a phase's redirect activation is that phase's version-bump trigger** (never retroactive under a published minor), and the tutorial `STEP_PAGE` map is remapped in the same commit as each redirect.

## Architecture decisions (post-consult)

1. **Server-rendered + islands stays.** No SPA. New list pages = Jinja + htmx; the `media-editor` island composes existing pieces (TranscriptPlayer, WaveformStrip, AnnotationLayer, keymap, verify/split/relabel/text endpoints).
2. **Media identity vs location (panel-corrected).** `media_items.id` is the anchor; `source_path` **stays the immutable acquisition identity**. Physical location becomes `media_folder_id` + mutable `current_path` (backfilled = source_path). Byte-openers (acquire, serving, reclaim, watch) switch to `current_path`; the other ~120 `source_path` reads stay valid. A P0 audit confirms the byte-opener set. sha256 backfill job for integrity, never identity.
3. **Crash-safe file ops.** Moves/trash go through a durable `media_operations` journal (planned → fs_applied → db_applied → completed/failed) with a startup/sweep reconciler and operator-visible recovery; moves rejected while a run is queued/executing; EXDEV = copy+fsync+verify+unlink; destination-collision refusal. **Trash = physical move into a managed `.voxint-trash/` subtree** (excluded from watch sweeps) recording origin + deadline; restore = journaled move back; empty-trash removes bytes + derived artifacts and leaves an honest degraded state on runs pages (transcript text and run history survive; playback/waveform honestly gone). Manual empty-trash only (no auto-GC v1).
4. **Projects membership invariant.** Media belongs through exactly one `media_folders` row (`media_items.media_folder_id` FK, set at ingest, updated by move); no path-prefix inference; overlapping folder registrations refused. Config resolution for NEW runs: project → folder pack → global, resolved from relations not path ancestry; effective config shown before rerun; explicit conflict rule when a packed folder joins a project (project wins, with a warning at assign time). Existing frozen run snapshots untouched. **Setup wizard is a named line item**: its folder registration is re-pointed to the new table in the same slice as the migration cutover (no dual-write window).
5. **Migrations land with their first consumer** (P0 is schema-free). `media_folders` migrates from `app_settings` via expand → dual-read equivalence check on real data → cutover → drop columns a release later, with a preflight normalization report (dupes, nested paths, dead dirs).
6. **Editor run + claim contract** (`/media/{id}`): canonical run selection = latest completed, `?run=` override + version chooser; claim is `(media, run)`-scoped, **never acquired on GET** — acquired on first edit intent via POST, same-operator claims reused not rotated (multi-tab safe), bounded renewal heartbeat, clear read-only state on claim 409. `Cache-Control: no-store` handling generalized off the `/review` prefix before any token-bearing `/media` page ships. Default entry for an unreviewed file = **verify-walk mode** (the stepper's completeness guarantee survives inside the editor); free-scroll for completed files. Undo for enroll/merge = compensating rulings on the append-only ledger (no row deletion); generalized undo deferred.
7. **Parallelization (panel-corrected).** Ownership is by symbol matrix, not just filenames: `db/models.py`, `app_settings.py`, `setup_wizard.py`, CSRF wiring, contract-test files, integration fixtures, base.html, frontend entries, alembic, docs each have ONE named owner. Contracts land before consumers; dependent tracks use stacked branches / an integration branch; every new page ships behind a feature flag so each PR stays releasable. Track E splits into **E1 editor-backend** (`routers/editor.py`, run/claim contract) and **E2 island**. A CSRF contract test asserts every new router's POST routes are covered.
8. **Speaker aggregation** from **effective resolver output** (canonicalized through merges, later rulings override earlier assigns, split/range overrides not double-counted), not raw joins; `verified` = currently-effective human assign. `speaker_profiles` carries per-field provenance (manual vs accepted enrichment candidate) so acceptance materializes without losing the draft-claim history; `speakers.notes` is not duplicated.
9. **Editor performance gates**: transcript-size budget, benchmark on a large seeded run, progressive rendering/virtualization if needed, keyboard/focus/a11y gates — all before legacy retirement.
10. **Numerics untouched.** No parity gates needed. Contract tests added in P0 and extended per phase: route inventory + redirect map, CSRF coverage, settings-section ↔ plugin-registry parity, dark-mode block identity (existing).

## Phases

Each phase = child issue(s) → feature branch → GitHub PR (lint-test + secrets-scan) → sync Forgejo; CHANGELOG per slice; docs + screenshots (grep for stale references, not just regenerate images) per shipping phase; voxint-docs house style on all copy (no "adjudication"/"cosine"/"ASR" on the happy path).

- **P0a — Contracts + tests (serial).** ADRs (media identity/location, project membership, run selection, claim lifecycle, profile provenance); route-inventory + redirect-map + CSRF contract tests characterizing current behavior; `source_path` byte-opener audit; sha256 backfill.
- **P0b — Router decomposition (serial).** Behavior-preserving extraction of app.py into `api/routers/{home,media,projects,speakers,jobs,settings,editor,legacy_review,legacy_runs}.py` + shared deps, one route family per commit, characterization tests green after each; templates into per-area dirs.
- **P1 — Shell + Home (serial).** Sidebar/topbar in base.html + library primitives (`.lib-table`, `.lib-cards`, view toggle, right-rail tokens); Home (attention-first + stats windows via extended `stats_query.py`); feature-flag plumbing; tutorial STEP_PAGE remap mechanism; `/dashboard` redirect (this phase's bump trigger).
- **P2a — Projects + folders schema, read-only library (Track A).** `projects` + `media_folders` (+ `media_items.media_folder_id`, `current_path`) with preflight + dual-read migration; setup wizard + settings folder panels re-pointed in the same slice; read-only `/media` and `/projects` pages; project vocabulary/corrections editors (reuse CorrectionsEditor) + submit-freeze resolution change + precedence contract test.
- **P2b — Upload + organization (Track A).** Upload on `/media` (+ drop zone on that page), assign-to-project/folder, non-destructive bulk actions (re-run, assign, archive).
- **P2c — Journaled move/trash (Track A, after storage ADR field-checked).** `media_operations` journal + reconciler + recovery UI, on-disk move, trash subtree, restore, manual empty-trash; crash-injection + EXDEV + collision + move-then-rerun-pack tests.
- **P3a — Editor backend contract (Track E1).** Run-selection + claim lifecycle endpoints under `routers/editor.py`, no-store generalization, export-sheet endpoints (existing 6 formats + options).
- **P3b — Editor island (Track E2).** `media-editor`: document view, speaker rail (assign/enroll/exclude/unknown/merge, name hints), inline reassign/edit/split/verify, walk mode (default for unreviewed), annotations, waveform, keymap, export sheet, translate; perf/a11y budgets; **coexists with legacy review for one full release**; browser E2E via the voxint-e2e-review lane extended to the editor.
- **P3c — Legacy retirement.** `/review*`, `/runs/{id}/transcript` redirects + removal after the coexistence release proves parity; enroll/merge compensating-undo.
- **P4 — Speakers (Track B).** Effective-resolution aggregation module + EXPLAIN-checked indexes; `/speakers` overview (reminders incl. possible-duplicate merges) + card/table roster; `/speakers/{id}` with provenance-tracked profile + enrichment acceptance + media drill-through.
- **P5 — Jobs (Track C).** `/jobs` + `/jobs/{id}` absorbing `/runs*`; DB-derived stage counts + service admission; redirects.
- **P6 — Settings (Track D).** Sub-pages: status (install-kind marker + doctor + hardware snapshot), hardware (read + copyable guided edit, "restart pending" detection), database (retention/GC/DB size), plugins from registry (`settings_sections()`); regrouped existing sections; scoped to the already-landed plugin interface (not blocked on #138–#142).
- **P7 — Extras (kept per Ben's selection; panel recommends this ships last and stays cuttable).** Global search bar (exact over media/speakers/projects first, semantic integration second), activity indicator + toasts (small `activity_events` outbox table, not heterogeneous-table polling), drag-drop-anywhere, command palette.

Integration order: P0/P1 serial → A, B, C, D parallel (A first among equals; B/C/D independent) → E1/E2 (consumes A's routes + P0 primitives) → P3c → P7.

## Verification

- Per slice: ruff/mypy/pytest green; frontend lint/typecheck/build; contract tests (route inventory, redirects, CSRF, precedence, pin-parity) extended not forked; upgrade tests from a populated 0038 DB for every migration; each PR releasable behind its flag.
- P2c: crash-injection at every fs/DB boundary; move/trash/restore/empty-trash integration tests over a tmp MEDIA_ROOT (sha256 continuity, run linkage, watcher exclusion, honest degraded states).
- P3: two-tab, claim expiry/renewal, rerun selection, back/forward/refresh, legacy deep links, export-after-correction; large-transcript benchmark; full browser E2E before P3c.
- Per release: seeded tutorial run end-to-end through the new flow; screenshots + doc-reference grep.

## First execution step after approval

Create the GitHub issue set: epic "Console 2.0" + children per phase above (P0a, P0b, P1, P2a–c, P3a–c, P4, P5, P6, P7), each carrying: scope, file/symbol ownership + do-not-touch list, endpoints consumed, acceptance criteria + verification gates, redirect/bump trigger, and cross-links (epic #136 for P6; #114 for the confidence upgrade path; supersedes note on the Aug-21 report).

## Consultation record (2026-08-24)

- **codex (clink planner)**: source_path can't be both mutable location and identity (anchor = media_items.id; sha256 nullable/non-unique); fs+DB needs an operation journal + reconciler; trash must be a physical managed move; project membership needs an FK invariant; P0 must be schema-free with expand/dual-read/backfill/contract; run-selection + claim lifecycle + no-store generalization before any token-bearing /media page; tracks not file-disjoint (models/app_settings/setup/CSRF/fixtures/docs need owners); editor coexists a release; defer palette/drag-drop/toasts/undo/guided-.env; speaker stats from effective resolution; large-transcript budgets. → all adopted (deferred extras kept as P7 per Ben).
- **deepseek-v4-pro (zen chat)**: sha256 backfill prerequisite + source_path audit (verified: nullable, ~126 refs); media_folders dual-write window (fixed: wizard/settings cutover in-slice); /media/{id} run ambiguity; Track E needs its own backend router; route-inventory test needs one owner; P2 split; move changes future pack resolution (explicit rule added); cut semantic search from v1 polish (staged inside P7).
- **claude-sonnet (clink, kimi-k3 substitute)**: Home-vs-Jobs duplication (Home is attention-first); walk mode must be the default entry or completeness regresses; setup wizard is an unscoped write path (now a named P2a item); tutorial STEP_PAGE breaks silently on redirects (remap same-commit); redirect activation = version-bump trigger under the no-behavior-changes-under-published-minor rule; CSRF coverage contract test; docs screenshot-reference grep.
