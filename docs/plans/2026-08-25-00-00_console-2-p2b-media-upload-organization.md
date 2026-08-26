# Console 2.0 P2b (#154) — Upload + organization + bulk actions on `/media`

## Context

Console 2.0 (epic #149) refactors Voxint's console to a Descript-style IA. Track A's
schema foundation and read-only library landed in P2a (#153, main `223cca9`): the
`media_folders`/`projects` tables, `media_items.media_folder_id`/`current_path`, per-field
config precedence frozen at submit (`config_resolution_version:2`), and read-only `/media`
+ `/projects` pages. **P2b makes `/media` operable**: upload/URL ingest move there, files
get organized into folders/projects, and a multi-select drives non-destructive bulk actions
(re-run, assign, archive/unarchive). No migration this slice; Track A files only.

Everything ships dark behind the existing `console_media_enabled` flag (default off). Flag
off: `/media` 404s and legacy `/runs` upload is byte-identical to today. This unblocks the
`/runs`-retirement path (P5) and the future `/media/{id}` detail links the speakers profile
(#159) is waiting on.

This plan was reviewed by a codex (planner) + deepseek-v4-pro panel; their findings and the
three decisions Ben settled are folded in below and recorded under **Review notes**.

## Decisions settled with Ben (post-consult)

1. **Amend ADR 0002: `media_folder_id` becomes a logical config-scope pointer.** The consult
   found a real contradiction: ADR 0002 Decision 1 currently says membership is "set at
   ingest and updated by a move," and treats uploads/URLs as unmapped/global — but #154
   asks for a "registration-level assign, no disk mutation." Ben's call: amend the ADR.
   `media_folder_id` is redefined as the folder whose settings apply, which MAY differ from
   the file's physical location; upload-picker and manual assign set it WITHOUT moving bytes.
   A future P2c physical move must PRESERVE an explicit override, never silently re-derive
   membership from `current_path`. UI copy must never imply a disk move.
2. **Add an archived view to `/media`.** `media_library()` shows only each item's latest
   NON-archived run, so bulk "unarchive" has no target as-is. Add a `?archived=1` toggle
   (reusing the existing `/runs` archived-view pattern) so archived items are discoverable;
   support both bulk archive and unarchive over the latest run in the current view. Label the
   action honestly ("Archive latest run" — archive is per-run).
3. **Atomic + prevalidate bulk semantics.** Validate the whole selection up front (reject
   stale/ineligible with ZERO writes), then apply in one transaction, commit once, publish
   after commit, and render a per-item result summary. (The panel split here — deepseek
   preferred per-item "do what you can"; Ben chose codex's atomic approach for cleaner
   double-submit/idempotency reasoning.)

## Ground truth from exploration (file:line anchors)

- **Ingest backends are UI-agnostic** but hardcode `media_folder_id=None`: `submit_upload`
  (`ingest/service.py:912`, freeze `:940`), `submit_url` (`:1022`, freeze `:1053`).
  `submit_media_item` (`:368`, freeze `:398`) is the folder-aware fresh-run creator (resolves
  precedence off the media row's persisted `media_folder_id`) — the ONLY re-run path that
  re-freezes config. `requeue_failed_run` (`:533`) does NOT re-freeze and is FAILED-only.
- **Membership is durable on reuse** (`_get_or_create_media`, `service.py:494`, docstring
  `:506-509`): "assignment happens once, at creation; a reused row keeps the membership it
  was created with." So a manual assign survives a later re-run/scan. The FK is
  `ON DELETE SET NULL` (`db/models.py:337`) — unregistering a folder nulls its media's
  `media_folder_id` rather than erroring.
- **One precedence engine**: `_run_domain_pack_snapshot(session, media_folder_id, *, settings,
  domain_pack_name, extra_name_seeds)` (`service.py:217`), per-field replacement
  explicit→project→folder→global, stamps `config_resolution_version:2` last. Reads only; runs
  inside the write txn today, no read-only preview caller exists.
- **Media page is read-only**: router `api/routers/media.py` (GET `/media` only, gated
  `require_onboarded` + `require_media_enabled`); query `api/media_query.py`
  (`media_library()`, `MediaLibraryRow` exposes `folder_path` string but NOT `media_folder_id`
  or project; `MEDIA_LIBRARY_LIMIT=500`); template `api/templates/media/media.html` (card|table
  on `.lib-*`, NO checkbox/multi-select/dropzone).
- **Assign pattern to imitate**: `projects.py::assign_folder` (`:201`) — CSRF-first,
  404-missing, `_reject` re-render closure, load, mutate, 303 PRG.
- **Folder registration service** (reuse verbatim): `media/registration.py`
  `register_folder`/`unregister_folder`/`set_folder_pack` (each returns error-string|None, takes
  the advisory lock, re-checks; `MAX_MEDIA_FOLDERS=64`); already HTTP-wired in
  `settings.py::settings_folders` (`:2231`) via `_apply_folder_mutation` (`:351`).
- **Archive is per-RUN** (`PipelineRun.archived_at`): `archive_run`/`unarchive_run`
  (`service.py:642`/`:667`, `_ARCHIVABLE_STATUSES={COMPLETED,FAILED,CANCELLED}`), routes
  `POST /runs/{id}/archive|unarchive` (`legacy_runs.py:1138`/`:1161`). The `/runs` page already
  has the `?archived=1` view pattern (`runs_query.py:429`, `legacy_runs.py:604`) to mirror.
- **Bulk selection precedent**: annotation pull-quote export ("a POST body is how the client
  ships a large selection", `test_console2_characterization.py:74`). CSRF constants in
  `api/csrf.py`; `_require_csrf` verify at `deps.py:385`; contract tests force every mutating
  route to expose+verify a defense field (`test_console2_characterization.py:230`/`:241`).
- **Goldens** (hand-regenerated, no script): `route_inventory.json`,
  `console2_route_characterization.json`, `console2_route_order.json` under
  `tests/contracts/fixtures/`. **Correction from the consult:** the new routes are
  ALWAYS registered (they 404 via the gate, not by being absent), so all three goldens GAIN
  rows even flag-off. What stays byte-identical is the EXISTING rows/order/gates — not "no new
  rows."
- **ADR verification**: ADR 0002 Decision 3 already says "the effective config is shown before
  a rerun so the operator sees what will apply" — the re-run preview is sanctioned. Decision 5:
  frozen run snapshots are untouched by membership changes.

## Design decisions (settled)

- **`media_folder_id` decoupled from path is now blessed** (Decision 1 above) via an ADR 0002
  addendum. UI copy: "Settings folder" / "Use settings from…", never "move". The library shows
  the physical `source_path` alongside the settings-folder so location stays visible.
- **"Assign to folder/project" = set `media_folder_id` to a folder.** Folders are the only
  thing media can belong to; the picker lists registered folders grouped/labelled by project,
  plus a "(no folder — global settings)" option that clears it. The folder `<select>` is
  server-rendered inside the plain form so the no-JS bulk path works (deepseek MUST-FIX); it
  must render even when `console_projects_enabled` is off (media/projects flags are independent).
- **Re-run preview is ADVISORY, not a guarantee.** It calls the real `_run_domain_pack_snapshot`
  read-only, but preview and confirm are separate READ COMMITTED txns, so config can change
  between them. The confirm page says config is re-resolved at confirmation; the minted run
  reflects confirm-time config. `preview_effective_config` returns a summary dataclass carrying
  the resolved pack name AND which layer supplied vocabulary/corrections (project/folder/global),
  not bare counts (two same-size configs differ).
- **Bulk re-run is double-submit safe.** The preview carries an expected latest-run baseline per
  item (incl. a no-run sentinel). Confirm row-locks the selected media in sorted order,
  re-verifies baselines, and skips/refuses any item that gained a newer run since preview — so a
  double-confirm creates at most one new run per item.
- **Re-run drops run-specific inputs, honestly.** A fresh run re-resolves current config and
  re-reads the on-disk sidecar if present; it does NOT carry the prior run's `operator_notes` or
  manual diarization hints (those belong to the old run). Documented in the confirm copy.
- **Multi-select = progressive enhancement over a plain form.** Native `<input type=checkbox
  name="media_id">` per row (table + card), repeated field ships the selection; JS adds
  select-all + a sticky action bar. Selections are deduped, capped at `MEDIA_LIBRARY_LIMIT`,
  malformed IDs rejected uniformly, loaded in one query with a count-match check.
- **Separate named routes, no `/media/bulk` dispatcher** (both models agreed): each mutation
  gets its own route + action-bound CSRF token, matching `projects.py`/`legacy_runs.py`. The one
  action-field route is `POST /media/folders` (add|remove), mirroring `settings_folders`.
- **Whole slice dark behind `console_media_enabled`**, including every "Add media" pointer.

## Route set (all on the media router, gated `require_media_enabled`, action-bound CSRF)

- `POST /media/submit` — file upload → `submit_upload(..., media_folder_id=picker)` → 303 `/media`
- `POST /media/fetch` — URL ingest (ytdlp gate) → `submit_url(..., media_folder_id=picker)` → 303
- `POST /media/folders` — register/unregister (`action=add|remove`) via registration service → 303
- `POST /media/assign` — bulk set `media_folder_id` over the selection (incl. clear) → 303
- `POST /media/archive`, `POST /media/unarchive` — bulk over each item's latest run in view → 303
- `POST /media/rerun` — advisory preview (no mutation) → renders confirm page
- `POST /media/rerun/confirm` — atomic mint of fresh runs → result-summary page

## Progress

- **Commit 1 DONE** — `072204d` on `feat/154-media-upload-organization`, pushed to BOTH
  remotes, gate-green (ruff check + mypy 180 + unit/contracts + targeted integration).
  ADR 0002 P2b addendum; `submit_upload`/`submit_url` optional `media_folder_id`;
  `_resolve_run_config` split + read-only `preview_effective_config`; `media_library()`/
  `MediaLibraryRow` expose `media_folder_id`+project + `archived` filter. Tests added.
- **Commit 2 IN PROGRESS** — concrete decisions locked from reading the code:
  - New routes `POST /media/submit`, `POST /media/fetch` on `api/routers/media.py`,
    modeled on `legacy_runs.py:734`/`:782` (CSRF-first; same error→status map
    413/422/409 + `DomainPackError`→422 via `deps._submit_domain_pack_detail`;
    commit-before-publish via `deps._publish_or_defer`). Redirect to **`/media`** (not
    `/runs/{id}`): `303` to `/media?submitted=1`, `?submitted=deferred` when the publish
    deferred — a small notice on the page (mirrors legacy `_run_redirect`'s
    `?enqueue=deferred` honesty).
  - New CSRF actions `CSRF_MEDIA_SUBMIT = "media-submit"`, `CSRF_MEDIA_FETCH =
    "media-fetch"` in `api/csrf.py` (own tokens, not the legacy `submit`/`fetch`).
  - Optional picker: form field `media_folder_id` (empty ⇒ None). Handler PREVALIDATES a
    non-empty value (parse UUID → 400; load `MediaFolder` → 400 if missing) BEFORE calling
    the service, so a stale pick never reaches the savepoint as an ambiguous IntegrityError.
  - New `media_query.folder_options(session) -> list[FolderOption(id, path,
    project_name)]` (LEFT outerjoin Project, ordered by lower(path)) feeds a native
    `<select>` labelled by project; renders with `console_projects_enabled` OFF.
  - GET `/media` context gains `csrf_media_submit`/`csrf_media_fetch` (via
    `mint_csrf_token`), `folder_options`, `ytdlp_enabled`
    (`resolve_effective_ytdlp_enabled(get_app_settings(session), settings)`), a fresh
    `submission_id`/`fetch_submission_id` per render, and the `submitted` notice flag.
  - Template `templates/media/media.html`: upload + URL forms (native `<select>` picker,
    hidden csrf + submission_id, URL form disabled when `ytdlp_enabled` is false) + a
    progressive drop zone, slotted above the toolbar; empty-state also gets the upload.
  - Regenerate 3 goldens (GAIN rows for the 2 new POSTs; existing rows/order/gates
    unchanged) + satisfy the two CSRF contracts. AC-1 integration test: same bytes/URL
    through `/media` and legacy ⇒ identical `MediaItem`/`PipelineRun` columns.

## Implementation outline (ordered, gate-green)

1. **Semantics + service/query seams (no routes).** ADR 0002 addendum (logical config-scope
   membership; move must preserve an override; uploads/URLs may now carry a chosen folder).
   Thread a validated optional `media_folder_id` into `submit_upload`/`submit_url` (default None
   ⇒ byte-identical legacy), preserving first-write membership on idempotent replay (a replayed
   submission_id returns the original run and does NOT re-assign/re-freeze). Add read-only
   `preview_effective_config(session, media_folder_id, *, settings) -> summary` built from
   `_run_domain_pack_snapshot`. Extend `MediaLibraryRow` + `media_library()` with
   `media_folder_id`, project id/name (LEFT outerjoin), and an `archived` filter param. Tests:
   assign changes only `media_folder_id` (source_path/current_path/bytes/frozen snapshots
   unchanged); `submit_media_item` reuses stored assignment, never re-resolves from path;
   scan/`submit_media_item_if_new`/registration/reconcile never clobber an override; replay
   preserves first membership; stale folder id fails before any write.
2. **Upload/fetch on `/media`.** `POST /media/submit`,`/media/fetch`; server-rendered folder
   `<select>` grouped by project (renders with projects flag off); ytdlp gate; new CSRF actions;
   drop zone (progressive). Regenerate 3 goldens (new rows; existing rows/order/gates unchanged);
   CSRF contracts. Integration: same bytes/URL through `/media` and the legacy path yield
   identical `MediaItem`/`PipelineRun` columns (`source_path`, `sha256`, `size_bytes`,
   `domain_pack`) — AC-1.
3. **Selection UI + bulk assign + folder panel.** Multi-select baseline lands HERE, before any
   bulk route (consult MUST-FIX on 3/4 ordering): checkboxes (table+card), select-all, sticky bar
   (JS-enhanced, plain-form fallback with the native folder `<select>` in-form). `POST
   /media/assign` (bounded/deduped/capped, prevalidate whole selection → atomic set, incl.
   clear-to-none). `POST /media/folders` register/unregister reusing the service; define + test
   unregister-with-assigned-media: FK `SET NULL` nulls those rows — surface it honestly ("N files
   reverted to global settings"), never a silent orphan or 500. Tests: assign no-fs-touch;
   register/unregister happy + overlap/cap refusal; unregister-with-assigned behavior.
4. **Bulk re-run + advisory preview + result page.** `POST /media/rerun`: prevalidate selection,
   resolve per-item via `preview_effective_config`, carry expected latest-run baseline per item,
   render advisory confirm page. `POST /media/rerun/confirm`: row-lock selected media in sorted
   order, re-verify baselines, skip/refuse items with a newer run, create ALL runs in one
   transaction, commit once, publish each, render per-item result summary. Tests: same-state
   preview==dispatch parity; changed-config-between-requests dispatches confirm-time config;
   double-confirm ⇒ ≤1 new run/item; any pre-commit failure ⇒ zero new runs; precedence follows
   the assigned folder/project (AC-2); sidecar re-read / hints-dropped documented behavior.
5. **Archived view + bulk archive/unarchive.** `media_library` `archived` filter + `/media`
   `?archived=1` toggle (mirror `/runs`). `POST /media/archive`,`/media/unarchive`: prevalidate,
   atomic over each item's latest run in the current view; not-archivable ⇒ reported "skipped",
   not "failed". Tests + archive→unarchive round-trip from the UI.
6. **Flag-aware pointers + docs + CHANGELOG.** Audit ALL "Add media" pointers under both flag
   states — `legacy_runs/runs.html`, `home/home.html`, `media/media.html` (consult MUST-FIX):
   flag-on → `/media`, flag-off → legacy forms byte-identical. Flag-off byte-identity test on the
   RENDERED `/runs` HTML (not just status). ADR 0002 addendum + architecture/operations docs;
   CHANGELOG `[Unreleased]`. Screenshots deferred to the release batch (convention).

Review fixes fold into the owning commits (no separate review commit). Gate per commit with
ruff + mypy + unit + contracts + TARGETED integration; run the FULL integration suite + the
browser lane ONCE before the PR (the full suite is >10 min; per-commit full runs are
disproportionate — consult note).

## Verification

- **AC-1 (identical rows)**: submit same bytes/URL through `/media` and legacy, diff
  `MediaItem`/`PipelineRun` columns. Model on `test_submit_api.py::test_upload_creates_namespaced_media`.
- **AC-2 (re-run honesty)**: assert precedence follows the assigned folder/project; assert the
  minted run's frozen `domain_pack` matches confirm-time resolution; double-confirm idempotency.
- **AC-3 (no fs touch)**: assign/archive tests assert `current_path` and on-disk state unchanged.
- **Browser lane** (headless Chrome, `uv run --with playwright`, `channel="chrome",
  headless=True` — Playwright MCP has no X here): ONE flow covering upload → select → assign →
  re-run preview → confirm → double-submit; plus archive round-trip. (Consult: avoid redundant
  browser coverage.)
- **Flag-off byte-identity**: `/media` 404s; `/runs` rendered HTML + POST behavior unchanged;
  existing golden rows/order/gates retained (new rows added for the new routes).
- Integration lane needs `VOXINT_TEST_DATABASE_URL` (the `voxint-test-pg` container); ONE pytest
  invocation at a time (shared DB deadlocks on DROP SCHEMA).

## Tail

Branch from fresh `origin/main` (synthdetect + the `feat/160-console2-jobs` track land in
parallel — merge, never rebase a pushed branch). Gate-green commits → full suite + browser lane
→ `/code-review` (3-engine) → GitHub PR (lint-test + secrets-scan) → merge on green → `git push
forgejo origin/main:main` → #154 close-out + epic #149 box. No release cut (Ben: land more
phases before 0.25.0). Hand the `/runs/{run_id}` → `/media/{id}` link switch in
`speakers/profile.html` to a LATER slice — a media detail page still does not exist (P3b editor
/ a future media-detail route, NOT P2b).

## Review notes (consult record, 2026-08-25)

Panel: **codex** (zen clink, planner role) + **deepseek-v4-pro** (zen chat). Both read the draft
plan and the P2a source.

- **Agreements**: keep separate named routes (no `/media/bulk`); the two-step re-run preview is
  right, not overbuild; progressive-enhancement multi-select is the right call; prevalidate the
  whole selection before any write; UI copy must never imply a disk move ("Settings folder");
  the no-JS assign target must be a server-rendered `<select>`; audit ALL Add-media pointers, not
  just runs.html; fix the commit 3/4 ordering so multi-select lands before bulk-assign.
- **The decisive catch (codex, verified in source)**: my draft's premise that ADR 0002 already
  blessed logical/decoupled membership was FACTUALLY WRONG — ADR 0002 Decision 1 says membership
  is "set at ingest and updated by a move," uploads/URLs unmapped/global. deepseek missed this
  and called the decoupling "sound." Resolved by Ben → amend ADR 0002 (Decision 1 above). Lesson
  re-confirmed: verify a plan's ADR/contract claims in source before trusting either model.
- **codex-only catches**: bulk re-run needs an expected-latest-run baseline for double-submit
  safety; bulk unarchive has no target given the latest-non-archived query (→ archived view);
  goldens gain rows even flag-off; re-run silently drops sidecar/operator_notes/hints unless
  specified; bound/dedup/cap the selection.
- **Genuine split surfaced to Ben**: atomic-all-or-nothing (codex) vs commit-per-item-and-report
  (deepseek) for partial bulk failure. Ben chose atomic + prevalidate + result summary.
- **deepseek-only emphasis**: unregister-with-assigned-media must be handled honestly (FK is SET
  NULL — tell the operator); result page must report per-item outcomes, never a bare 303.
