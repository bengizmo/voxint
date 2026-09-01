# Changelog

All notable changes to Voxint. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning: [SemVer](https://semver.org/) (0.x; expect breaking changes between minors).


## [Unreleased]

### Added
- **Media library search and status filter** (#380). Server-side search
  (case-insensitive substring over display name, source path, and folder
  name) and a status filter (Needs review / Failed / Reviewed / All).
  Search and filter compose with each other and with the existing sort
  and view controls. An empty-result state with a Clear link replaces the
  generic empty message when a filter is active. Row actions are now
  state-dependent: Review (needs adjudication), Retry (failed), or Open
  (everything else). The "File missing" chip is replaced with plain
  language ("Original file not found") showing the expected path.
- **GPU resource awareness in the installer**. VRAM-aware tier recommendation
  with per-device compose targeting, interactive device selection, `--gpu-check`
  diagnostics, and enhanced `--hardware-dry-run`. The installer now inventories
  all NVIDIA and AMD GPUs, measures free VRAM, and recommends the compute tier
  that fits. When a GPU is occupied (e.g. by a local LLM), the installer
  explains why and defaults to CPU.

- **Runs canonical lifecycle surface** (#381, part 1). `/runs` becomes the
  canonical lifecycle view: four lifecycle tabs (Needs attention / Active /
  Failed / All), title-first rows with the run ID demoted to a copyable
  secondary, a collapsible filter bar (auto-opens when any filter is active,
  power filters like transcript search and date range under "More filters"),
  a one-line pipeline health summary linking to Settings Status, and
  auxiliary jobs behind a closed disclosure. State-dependent row actions
  (Review / Retry / View) and degraded-service banners carried over from
  the Jobs page.
- **Grouped failures with bulk retry** (#381, part 2). On the Failed tab,
  runs with identical errors are grouped into a single row showing the
  error label, count, and a "Retry all" button. The bulk retry endpoint
  uses per-item CAS with savepoints for partial-failure safety, CSRF
  protection, and a per-item result page.

### Changed
- `/jobs` now redirects to the canonical `/runs` surface; old bookmarks and
  links continue to work (#382)

### Removed
- `CONSOLE_JOBS_ENABLED` dark-ship flag; the sidebar Jobs entry now always
  points at `/runs` (#382)
- **Settings control model** (#379). Every tri-state feature setting
  (On / Off / Use installation setting) is now a toggle switch showing
  the effective state. Overridden settings show a "Changed" badge and a
  "Reset to default" affordance. Each settings tab has one Save button
  instead of one per section. Feature dependencies are enforced by
  disabling the dependent switch with an inline reason, not prose
  notices. The LLM checkbox is restyled as a visual switch (no
  inheritance — the underlying model is unchanged). The three persisted
  states (on / off / inherit) are preserved on the wire; saving without
  changes never converts an inherited flag to a pinned override (save
  idempotency).
- **Settings IA: tabs** (#378). The settings hub is split into tabbed
  sub-pages: General (appearance, features, tutorial), Media (folders,
  sources), and AI (LLM, translation, corrections, glossary, semantic
  search). All five existing sub-pages (Status, Hardware, Database,
  Plugins, Users) now share a single tab strip include instead of
  duplicating it. Old fragment links like `/settings#llm` redirect to
  the correct tab. All POST routes unchanged. The legacy flat page
  (flag off) is unaffected.
- **Run detail restructure** (#377). The run detail page (shared by `/runs`
  and `/jobs`) puts the summary card first (status, readable stage name,
  error with recovery hint, and the primary action). The Manage card moves
  up, right after notices. Technical details (source path, revision, pipeline
  models, glossary, detected language, stage ledger) fold into a collapsed
  section. Speaker timeline shows readable durations. "Restart from scratch"
  becomes "Run again from the beginning".
- **Error normalization** (#376). Raw pipeline error strings are replaced with
  plain-language labels and recovery hints across all operator surfaces (home,
  editor, run detail, jobs). The original error is preserved in a collapsible
  details fold. The home Recent feed groups consecutive identical failures
  into one row with a count. New patterns: GPU memory, GPU errors, deferred
  stages, paused runs.
- **Editor content reorder** (#375). The outline panel (topics and entities)
  moves below the transcript and starts collapsed with a count badge, putting
  the work surface first. Transcript rows show one speaker display name
  instead of duplicating the raw diarization label and display name. The
  export section collapses into a single "Download transcript" button with a
  compact format picker (reviewed text default; other variants in a nested
  disclosure). The breadcrumb shows run status instead of duplicating the
  live progress counter. A new Run card shows status, error, and
  Retry/Re-run actions directly on the editor page.

### Added
- **Single-operator auto-claim** (#374). When multi-user mode is off, the
  editor and workbench auto-claim on mount, removing the manual "Claim for
  editing" / "Claim for review" friction. The review queue hides the
  "Claimed by" column. A non-rotating `refresh` endpoint replaces the
  heartbeat's previous re-claim, so a stale tab receives 409 and drops to a
  manual "Resume editing here" action with no two-tab heartbeat fight.
  Visibility-aware: the heartbeat pauses while the tab is hidden and fires
  an immediate refresh on re-visible. Multi-user behavior is unchanged.

### Changed
- **Type foundation** (#373). Base font raised from 13px to 15px for reading
  surfaces; grid-tables and HTML tables use a new dense token (13px) so they
  keep their compact density.

### Fixed
- **Queue progress strikethrough** (#372). The review queue's "N of M
  resolved" label appeared struck-through because the progress column
  (10rem) was too narrow for the track (`min-width: 8rem`) plus the
  label text; the overflow caused the track to visually interfere with
  adjacent text. Widened the column to 14rem.
- **Invisible "Set up" button on Status page** (#372). The primary
  action button next to "Your own AI endpoint" rendered as a solid teal
  rectangle with no visible text. Cause: `.component-row .cr-action a`
  (specificity 0,2,1) set `color: var(--accent)`, overriding
  `.cb-btn-primary`'s `color: var(--accent-contrast)` (0,1,0). Fixed
  with a `:not(.cb-btn)` guard on the action link rule.
- **Editor speaker-rail collapse** (#370). The media-editor page nested the
  island's `.me-layout` grid inside `.lib-two-col`, causing the speaker rail
  to collapse to a sliver at desktop widths. Restructured: metadata cards
  (media info, run chooser) now sit in a horizontal bar above the island,
  and `.me-layout` owns the full page width. The stacking breakpoint drops
  from 72rem to 48rem since the grid is no longer width-constrained by an
  outer grid.
- **Contradictory empty-state copy** (#370). "No runs for this media file
  yet." appeared alongside listed failed runs. Now reads "No completed runs
  yet." when runs exist but none completed.

### Added
- **Viewer role** (#363). A new read-only `viewer` role: viewers can browse
  transcripts, results, and exports but cannot submit media, adjudicate,
  correct, annotate, or change any settings. Enforced server-side with a
  blanket write gate on all mutation routes (console and API); the gate
  allowlists only auth routes (login/logout). Template context exposes a
  `can_write` boolean for UI affordance hiding. CLI `user create --role`
  and `user set-role` accept `viewer`. Migration 0057 widens the DB CHECK
  constraint. Dark-shipped behind the existing `console_users_enabled` flag
  (user management page only).
- **Evidence pack** (#331, Phase 7 remainder). A printable page of a run's
  highlights with their provenance (speakers, timing, tags, notes, source
  text hash, clip references, pipeline model identity), linked from the
  review transcript. Print or save it as a PDF from the browser; no PDF
  library added. Stale highlights print with a visible warning instead of
  blocking the document.
- **Bundled quote export** (#281). One click downloads a highlight's Markdown
  pull-quote, JSON provenance manifest, and audio clip (when one has been
  extracted) as a single ZIP, per highlight or for every highlight matching
  the current tag filter. The bundled files match the standalone exports
  byte for byte (the manifest differs only in its export timestamp). No new
  dependencies (stdlib zipfile).
- **Highlight tag rollup on Explore** (#331, Phase 7 remainder). The Explore
  page now shows how many highlights carry each tag, corpus-wide or narrowed
  by the project filter. Computed live with one SQL query (no cache, no new
  dependencies); archived tags and deleted highlights are excluded, and
  highlights on runs still under review count from the moment they exist.

### Fixed
- **Copy and behavior bug batch** (#371). Explore header now pluralizes
  correctly ("1 hour" not "1 hours"). `/login` redirects to `/` with a 302
  when multi-user mode is off (was a raw 404). Backups copy on the Database
  settings page now mentions the native launcher's backup command alongside
  `pg_dump`. Database largest-tables estimates suppress zero-row entries
  (Postgres stats lag makes those misleading).


## [0.31.0] - 2026-08-31

### Added
- **Media editor island** (#157, Console 2.0 P3b). In-browser transcript
  editor with memoized row rendering, editor-native claim lifecycle (claim,
  heartbeat, release), progressive-disclosure speaker rail, translate action,
  per-segment correction provenance, and accessibility hardening (roving
  tabindex, aria-current, cursor announcer, prefers-reduced-motion,
  responsive narrow viewports). E2E harness extended with editor and
  benchmark fixtures, split and annotation reconciliation.
- **More like this passage** (#357, Phase 7 partial). Every KWIC row on the
  Explore page expands an inline panel of the corpus passages nearest in
  meaning to that segment, re-embedded at request time over the existing
  MiniLM index (exact cosine scan, no new dependencies). The originating
  paragraph and its overlap span are excluded; other passages from the same
  recording are kept but capped. Honest off/unavailable/indexing states.
- **Semantic meaning map** (#357, Phase 7 partial). Collapsible canvas
  scatter on the Explore page: every indexed passage placed by a
  deterministic 2D PCA projection of its embedding (pure numpy, not UMAP),
  colored by recording, with hover previews and click-to-jump transcript
  links. Computed on read and cached as a `semantic_layout` corpus-analysis
  artifact (the unused `umap_layout` kind was renamed before first use);
  corpora over 3,000 passages are sampled evenly across recordings.
- **Quote board** (#338, Phase 6). Save KWIC concordance rows as
  project-scoped evidence quotes from the Explore page. Saved quotes appear
  on the project detail page with inline note editing, deletion, and CSV
  export. Deduplicated by (project, segment, search query). Migration 0056.
- **Project temporal trends** (#337, Phase 5). Interactive D3 line chart on
  the project detail page showing term and entity frequency over recording
  dates. Adaptive calendar bucketing (day/week/month), date provenance
  tracking (source upload date preferred, ingestion date fallback), top-20
  series with selectable top-5 display, responsive SVG, and keyboard-
  accessible data points. Compute-on-read with advisory-locked fingerprint
  short-circuit, stored as a TEMPORAL_TRENDS corpus analysis artifact.
- **Run speaker timeline** (#337, Phase 5). Visual speaker timeline on the
  run detail page showing who spoke when as colored horizontal lanes. Built
  from diarization turns with canonical label_states() resolution, sub-50ms
  interval merging, overlap hatching, unresolved/excluded visual treatment,
  and keyboard-focusable blocks linking to transcript timestamps.
- **Explore page: corpus-wide evidence browser** (#331, #333, #334). New
  top-level "Explore" section with KWIC concordance search across every
  finished transcript (speaker, project, date, confidence, and suspect
  filters; pagination; CSV export), plus a TF-IDF word cloud and ranked
  distinctive terms for the corpus or a single project, cached and refreshed
  after each run completes. `/search` redirects to `/explore`. Migration 0055.
- **Speaker insights on the profile page** (#335). Distinctive vocabulary
  (log-odds with an informative Dirichlet prior), conversation partners
  (who speaks before and after them), and speaking pace, computed for all
  eligible speakers after each run completes and rendered from cache.
- **Project overview** (#336). The project page now opens with an Overview
  section built from run-assets enrichment across the project's recordings:
  summary stat tiles, most-mentioned entities with click-through to a
  filtered Explore search, topic chips, and a speakers-by-recordings
  coverage matrix. Computed on read behind a fingerprint cache, so it
  self-heals after enrichment, adjudication, merges, and folder moves;
  honest notes when some or all recordings have no enrichment yet.

### Fixed
- **Explore term-stats advisory lock now varies per scope** (#338 follow-up).
  The lock key was built from the first 8 bytes of the raw key string, which
  are the constant prefix "term_sta", so every scope serialized on one
  advisory lock. Now hashed (sha256) before truncation, matching the meaning
  map's lock. Over-serialization only; behavior was never incorrect.
- **Project speaker counts no longer double-count re-runs** (#336). The
  project page's speaker list now counts each recording once (its newest
  completed run) instead of once per historical run. Membership follows the
  same rule: a speaker who only appears in an older, superseded run of a
  recording no longer shows on the project — the newest completed run is
  the operative transcript, matching how the rest of the console resolves
  speakers.
- **Media file operations no longer stall for up to five minutes after two
  cleanup passes collide** (#346). When two reconciler passes raced for the
  same pending file operation (a trash, restore, or purge), the winner could
  skip its own claimed work, leaving the operation parked until its lease
  expired. The winner now waits out the moment of contention and finishes
  the job. Found via a CI flake; a deterministic regression test pins the
  interleaving.
- **Top-bar controls no longer overlap on narrow screens.** Below ~800px
  CSS width, the fixed-height command bar let the Table/Cards view toggle
  slip under the page action button ("+ Add media", "+ New speaker"),
  which then swallowed its clicks. The bar now wraps on narrow screens so
  every control stays visible and clickable.
- **Explore filters no longer 422 on empty selection** (#356). The HTML
  `<select>` sends `project=&speaker=` (empty strings) when "All" is
  selected; the explore endpoints now accept empty strings gracefully
  instead of rejecting them as invalid UUIDs.


## [0.30.0] - 2026-08-30

### Added
- **Pause/resume and restart-from-scratch pipeline controls** (#324). The run
  detail page now has a **Pause** button for queued and running runs
  (cooperative — the current stage finishes first, then no further stages
  start until you resume), **Resume** and **Cancel** for paused runs, and
  **Restart from scratch** in the Manage section for completed, failed, or
  cancelled runs (re-queues from the first stage; prior results are preserved
  in the stage ledger). New `PAUSED` status with amber pill; migration 0053.

### Changed
- **Decorative search chrome removed** (#162). The inert "Search everything
  ⌘K" / "Find a person" / "Filter media" affordances in the topbar and on the
  media and speakers pages are gone until search actually exists; the layout
  seam and styles stay for the real implementation.

### Fixed
- **Settings status reports the two AI lanes separately and honestly**
  (#316). The single "Local AI model" row only ever probed the
  bring-your-own endpoint, so a healthy bundled-only install showed
  "rejected (HTTP 401)" and an amber banner. The status page now shows
  "Bundled AI model" (probed when the bundle is active) and "Your own AI
  endpoint" (probed only when a deliberate endpoint is configured; the
  untouched install default reads "not configured" instead of a false
  alarm). `voxint doctor` and the setup wizard report the same two lanes;
  exit codes are unchanged (both checks stay advisory). A deliberately
  configured endpoint that rejects still warns - a broken deliberate
  config is real information.
- **Settings status knows how Voxint was installed** (#317). A
  `VOXINT_INSTALL_KIND` marker stamped by the packaging (baked into the app
  image for Docker installs, set by the native launcher's service
  environment) replaces the hardcoded "Install type unknown". Existing
  installs pick it up on their next `docker compose up` image pull or
  `voxint-native.sh up` (the launcher re-renders service definitions every
  start). Running from source still honestly reports unknown.
- **Home "voices without a name" card dead-ended** (#315). The card's arrow
  now opens the review queue sorted by unresolved voices (where those voices
  can actually be named) instead of /speakers, which never lists them, and
  its note says "across recordings waiting for review" (the old "across
  reviewed recordings" was false).
- **Topbar section label on review pages** (#319). The review workbench and
  its transcript page now label the topbar "review / <media name>" (they fell
  through to "home"), and the legacy transcript page crumbs to "jobs /" or
  "runs /" following which run browser the console exposes.
- **Singular/plural copy errors** (#318). "1 voice need you" is now "1 voice
  needs you" everywhere (home activity feed, media library, jobs board, via a
  shared chip macro that also fixes the jobs board's unstyled needs-you chip),
  the speakers header says "1 person" instead of "1 people", the home
  attention card says "voice without a name" at a count of one, and
  name-evidence snippets no longer start or end mid-word.
- **Console favicon** (#320). The console now ships a favicon (the teal "V"
  brand mark) served from an authenticated `/favicon.ico` route, so browser
  tabs show an icon instead of a 404 on every page.


## [0.29.0] - 2026-08-29

### Changed
- **Speakers overview and profiles activated** (#159). The new speakers overview
  (aggregation numbers, named/unnamed grouping, tier chips, verified badges) and
  per-speaker profile pages are now on by default. Set
  `CONSOLE_SPEAKERS_ENABLED=false` to roll back to the legacy roster.
- **Jobs page: TOOK column shows elapsed time for running runs** (#244). Running
  runs now display their wall-clock elapsed instead of "...".
- **Jobs page: degraded pipeline cells** (#244). When a model service is down,
  the corresponding pipeline board cell renders with an amber border, amber dot,
  and "paused" subtitle instead of appearing idle.
- **Jobs page: estimated wait time** (#244). Pipeline board cells with queued
  runs show "~Xm left" based on historical stage durations.
- **Jobs page: plain-language failure reasons** (#244). Failed runs in the REVIEW
  column now show a human-readable label ("worker timed out", "file not found",
  etc.) instead of raw Python exception text.
- **Jobs page: GPU busy in pipeline summary** (#244). The command bar summary
  includes "GPU busy" when GPU telemetry reports high utilization.
- **Deferred**: Pause queue button (#244 delta 1, needs Celery control surface)
  and progress percentage (#244 delta 5, needs stage-level progress callbacks).

### Security
- **Open-redirect bypass in login redirect hardened** (#304). `_validate_next()`
  now rejects backslash normalization (`/\evil.com`), control-character injection
  (response splitting), and scheme-bearing paths via `urlparse`. Previously only
  `//` was blocked. 20 unit tests cover the bypass vectors.
- **Admin gate added to synthdetect mutation routes** (#305). `POST
  /synthdetect/settings` and `POST /synthdetect/score/{run_id}` now require the
  `AdminDep` dependency. In multi-user mode, only admins can toggle synthdetect
  settings or trigger manual scoring. Route characterization golden updated.
  Three integration tests.

### Added
- **Multi-user authentication and roles** (#9). Opt-in via
  `VOXINT_MULTI_USER=true`. Adds a local user table (Argon2id password
  hashing), DB-backed session cookies, a login/logout form, and two roles
  (reviewer/admin). The single-operator HTTP Basic path is unchanged when the
  flag is off (the default). Per-user attribution: a nullable `user_id` FK on
  `adjudication_decisions` links human rulings to the authenticated user (system
  decisions stay `user_id=NULL`). Settings area is admin-only. User management
  via CLI: `voxint user create/list/delete/set-role/set-password`. First user is
  forced to admin. Migration 0051.
- **Quote provenance manifest** (#122). Pull-quote and clip exports now have a
  JSON sidecar (`export.json`) carrying per-line speaker attribution, timecodes,
  clip integrity digest, per-stage model identity from the pipeline run, and
  input-media SHA-256. Single-annotation and bulk (tag-filtered) endpoints.
- **Benchmark feature** (CLI + DB + scorer + console). Shipped 12-clip benchmark
  corpus (~3 MB, CC-BY-4.0 / CC0) with `voxint benchmark run` (serial
  submission, pooled micro-WER, hallucination resistance scoring),
  `voxint benchmark list`, and `voxint benchmark compare`. DB-backed results
  (migration 0048) for cross-run comparison. Benchmark runs excluded from the
  review queue via reserved `benchmark/` source path prefix. Minimal Settings
  page integration (recent runs table). See `docs/benchmark.md`.
- **Auto-enrollment: cross-run voice clustering** (#275). When a pipeline run
  completes, unmatched diarization labels with sufficient speaker audio are
  automatically enrolled as unnamed roster speakers ("Voice 1", "Voice 2", ...).
  Future runs match against these centroids naturally through the existing
  pipeline. Uses grounding-tier gates (cosine >= 0.70, margin >= 0.08, vote
  agreement >= 0.67, min 3 eligible turns, 10s eligible speech). Controlled by
  `AUTO_ENROLL` env var (default: true). Auto-enrolled speakers are exempt from
  the 5-minute unverified floor so they always appear in the "Unnamed voices"
  section for operator review. New `Decision.AUTO_ENROLL` and
  `Resolution.AUTO_ENROLL` types keep system-initiated enrollments distinct from
  human rulings (migration 0049).
- **Auto-enrollment backfill** (#280). `voxint speakers auto-enroll-backfill`
  runs auto-enrollment against all completed runs with unresolved labels,
  letting existing deployments benefit without reprocessing. Supports
  `--dry-run`, `--run-id`, and `--limit`.
- **Speakers overview: named/unnamed grouping** (#245). The roster splits into
  two visual groups: "Known speakers" (verified) shown in full, and "Unnamed
  voices" (unverified) collapsed by default in a `<details>` disclosure.
- **Speakers overview: heard-name annotation** (#245). Unverified speakers with
  an LLM-proposed name show `"Sam?" heard name, unconfirmed` and a "Rule" link
  instead of "Open".
- **Speakers overview: TO DO action buttons** (#245). The TO DO strip now
  includes `[Review them]` and `[Compare]` action links.
- **Speakers overview: "+ New speaker" button** (#245). Creates a bare roster
  entry from the overview page (name only, no audio).
- **Speakers overview: search placeholder** (#245). Shows "Find a person"
  instead of the generic "Search everything".
- **Speaker detail: "verified by you" chip** (#246). The profile page chip now
  reads "verified by you" for confirmed speakers.
- **Speaker detail: profile panel redesign** (#246). Aggregate provenance chip,
  3-column ROLE/ORGANIZATION/LINKS grid, and a hide/show collapsible.
- **Speaker detail: command bar actions** (#246). "Edit profile" and "Merge"
  dropdown in the command bar.
- **Speaker detail: FOLDER column** (#246). The Heard In table now shows which
  folder each recording belongs to.
- **Project detail: rename** (#247). "Rename" button in the command bar with
  inline form.
- **Project detail: folder unlink** (#247). Each assigned folder now has an
  "unlink" action to remove it from the project.
- **Project detail: vocabulary "+ add term" chip** (#247). Dashed inline chip
  that opens the vocabulary editor.
- **Project detail: speaker links** (#247). Speaker names in the sidebar now
  link to their individual profile pages.
- **Project detail: "+ add speaker" and "+ link folder" links** (#247).
  Navigation links in the sidebar section headers.
- **Synthetic-speech detection plugin** (#145, epic #143). Opt-in audio
  deepfake detection using a fine-tuned w2v2-AASIST classifier with
  Platt-calibrated risk scores. Ships as a plugin with four pieces: a
  standalone GPU inference service (`voxint-synthdetect`, compose overlay
  `compose.plugin-synthdetect.yaml`), a plugin backend (DB tables, Celery
  task, Platt calibration, job lifecycle), console integration (tri-state
  settings toggles, run-detail score panel with all job states and risk
  chips, standalone report page with known-limitations disclosure), and
  manual scoring for existing runs. Enable via Settings or
  `SYNTHDETECT_ENABLED=true`. Known limitations: Chatterbox evasion (#252,
  25.90% EER after M2 fine-tuning) and VoxConverse channel confound (#253).

### Changed
- **Synthdetect eval tooling extracted to private repo.** 46 files (21k lines)
  of maintainer evaluation infrastructure moved. Zero runtime dependencies on
  voxint app code. The compact public summary in `docs/gpu-contracts.md`
  retains model limitations, coverage gaps, and M2 fine-tuning outcomes.
- **Translation integrity gaps closed** (ADR 0008, Phase 2 Step 3).
  `run_translations` now has an `idempotency_key` (nullable UNIQUE) and
  `replay_digest` (canonical sha256 of the `lines` JSONB for efficient replay
  comparison without loading multi-MB payloads). `record_translation` is wired
  through `savepoint_adopt_or_conflict` with full replay matching (minus
  `completed_at`, which is unstable across retries). A database immutability
  trigger rejects UPDATE (except the write-once supersession stamp) and DELETE,
  matching the `run_enrichment_assets` pattern. Migration 0047.

## [0.28.0] - 2026-08-27

### Added
- **ADR 0008: enrichment persistence simplification scope** (Phase 2, finding
  H7). Preserves the append-only generation model (proportionate to real
  concurrency and decision immutability constraints); scopes implementation to
  narrow transaction-choreography extraction and two translation integrity gaps
  (idempotency key, immutability trigger). See `docs/adr/0008-*`.
- **Media detail page** (`GET /media/{id}/editor`, #156, epic #149). The editor
  backend contract: run selection (latest completed by default, `?run=`
  override validated against the media item), claim-token verification
  (stale/absent tokens degrade to read-only), transcript display with
  speaker palette and verified-progress counter, run chooser, and media
  metadata rail. Existing `/review/{run_id}/*` endpoints remain the only
  mutation surface. The media library's "View" links now open the detail
  page instead of the legacy run page.
- **Security: token-sensitive path classifier** (#156). `Cache-Control:
  no-store` now covers both `/review/*` and `/media/{uuid}/editor*` paths, so
  claim tokens embedded in the editor detail page are never cached. Library
  routes (`/media`, `/media/submit`, etc.) are unaffected.
- **Possible-duplicates reminder card** (#181). The speakers overview page shows
  a reminder card when two or more roster speakers share a high cosine
  similarity, prompting the operator to review whether they should be merged.

### Changed
- **Idempotent insert helper extracted** (Phase 2, finding M10). The
  savepoint-adopt-or-conflict skeleton that was duplicated across ledger,
  annotations, drafts, and run assets is now a single
  `savepoint_adopt_or_conflict()` in `voxint.idempotency`. Each site calls
  it with its own lookup, match, and persist callbacks. Behavior-preserving:
  same key, fingerprint, savepoint, and advisory-lock semantics.
- **Stage graph contract tests and runtime guard** (Phase 1 H6). Twelve
  contract tests lock the pipeline stage invariants: enum/STAGE_ORDER
  agreement, GPU/POST lane partition completeness and contiguity,
  build_stage_fns coverage, and lane routing correctness. A runtime guard in
  `build_stage_fns` fails at startup if a new Stage member is missing a
  function mapping.
- **Ingest: SubmissionResult makes commit-before-publish visible** (Forgejo #7,
  finding H5). All submit functions (`submit_upload`, `submit_url`,
  `submit_media_item`, `submit_media_item_if_new`) now return a
  `SubmissionResult` carrying `run_id` and a `publish()` method instead of a
  bare `PipelineRun`. Callers commit, then call `result.publish()`. The three
  duplicate publish-or-defer implementations (API deps, CLI, worker) are
  consolidated into the single `publish()` method. The watch sweep no longer
  needs an injected publish callback. `deps._publish_or_defer` is retained for
  the requeue handler (which uses a stage parameter).
- **Worker hardening: uniform broker fault tolerance** (Forgejo #8, finding M7).
  Post-finalize autogeneration functions now use `.apply_async(ignore_result=True)`
  instead of `.delay()`, and the recovery sweep redispatches stale QUEUED
  run-asset and translation jobs alongside the existing embedding lane.
- **Centralized config-resolution-version parsing** (Forgejo #8, finding M11).
  The inline parsing in `worker/tasks.py` is now
  `parse_config_resolution_version()` in `pipeline/stages/context.py`.
- **Global HTML error renderer** (Forgejo #5, finding H2). HTTP errors and
  unhandled exceptions now content-negotiate between a styled HTML error page
  (for browsers) and JSON (for API/CLI clients). The standalone error template
  shows the status code, a friendly title, and recovery advice for 5xx. Server
  errors never leak exception details or headers to the client.
- **Home feed: unresolved voice count on completed runs** (#249, epic #149).
  Run-completed events in the Home activity feed now show an amber "X voices
  need you" chip when unresolved speaker labels exist for that run.
- **Media overview: child-row review state** (#249, epic #149). Child rows and
  ungrouped rows now render review-oriented chips ("X voices need you" when
  unresolved labels exist, "reviewed" when all resolved) instead of the raw
  run status, matching the folder-row treatment.
- **Media overview: selected-row teal tint** (#249, epic #149). Checking a media
  row now highlights it with the existing teal-tint background rule.
- **Jobs page: TOOK column, subfolder line, failure reason** (#244, epic #149).
  Completed and failed runs now show actual elapsed time (e.g. "3m40s") instead
  of a placeholder. Runs from a registered settings folder show the folder path
  below the filename. Failed runs show their error message in the REVIEW column.
- **Settings status: host-level hardware gauges** (#248, epic #149). The status
  page now shows five gauges (Processor, Memory, Graphics card, Graphics memory,
  Disk) instead of two GPU-only gauges. CPU and memory are read from the host
  via `/proc/meminfo` and `os.getloadavg`; disk usage is the media root
  partition via `shutil.disk_usage`. No new runtime dependencies.
- **Settings status: GPU acceleration in install summary** (#248, epic #149).
  The banner line now includes "GPU acceleration on (16 GB)" (or "off" for
  CPU-only installs) between the install type and the version number.
- **Settings status: LLM component description** (#248, epic #149). The Local
  AI model row shows "off, used for polish & profiles" with a primary "Turn on"
  button instead of a plain text link.
- **Settings sub-page nav: Features tab** (#248, epic #149). All four settings
  sub-pages (Status, Hardware, Database, Plugins) now include a Features tab
  that links back to the hub page's Features section.
- **Settings status: Check for updates** (#248, epic #149). The banner button
  now reads "Check for updates" and links to the GitHub releases page instead
  of the hardware sub-page.

### Added
- **Synthetic-speech detection: feature-cached AASIST fine-tuning pipeline**
  (#252, M2). `tools/synthdetect_finetune.py` freezes the XLS-R-300M backbone
  and fine-tunes the AASIST backend (~18 M params) on precomputed frame
  embeddings to close the Chatterbox flow-matching TTS evasion gap. Fits on a
  12 GB GPU. Four subcommands: `cache-features` (extract and persist XLS-R
  embeddings), `train` (constrained Pareto checkpoint selection: minimize
  Chatterbox dev EER subject to Piper regression and bona-fide FPR gates),
  `evaluate`, and `recalibrate` (Platt scaling + operating-point selection for
  the fine-tuned checkpoint). Includes speaker-disjoint hash partitioning,
  balanced batch sampling (70/30 Chatterbox/Piper, 25/75 VoxConverse/other),
  and multi-seed support. 12 unit tests.

### Removed
- **Legacy `app_settings` folder columns dropped** (#177, epic #149). Migration
  0046 drops `app_settings.media_folders` and `app_settings.folder_domain_packs`,
  which were retained for one release as rollback inputs after #153 moved
  registered folders into the `media_folders` relation. The CLI
  `voxint media folders preflight` subcommand and the unused
  `resolve_folder_pack_name` registry function are also removed.

### Fixed
- **Corrections editor: inherit-to-empty silent conversion** (#176, epic #149).
  On the project detail page, opening the corrections editor while a project
  was inheriting corrections (NULL) and saving without adding rules silently
  wrote an explicit empty list, overriding the folder pack or global baseline.
  The editor now gates on an `inheriting` prop: when corrections are NULL, a
  "Set corrections for this project" button must be clicked before the editor
  and its Save become reachable. Regression tests cover the prop serialization.

## [0.27.0] - 2026-08-27

### Changed
- **Ops Console R4: speakers overview and detail refresh** (#213, epic #205).
  Both `/speakers` and `/speakers/{id}` rewritten with V3 primitives. The
  overview moves its people count and verified summary into the command bar,
  adds an amber TO DO strip for unverified voices and pending name suggestions,
  and replaces the card-based roster with a grid-table showing initials badges,
  status chips (verified/needs you/unknown), voice match percentages
  (green >=85%, amber <85%), and Open links. The detail page adds a profile
  header with a large avatar badge, six stat tiles (recordings, speaking time,
  segments, avg match, first heard, last heard), and a heard-in grid-table.
  New "active" chip semantic added to both the Python mapping and Jinja macro.
- **Ops Console R5: project detail refresh** (#214, epic #205). The
  `/projects/{id}` page rewritten with V3 primitives: command-bar breadcrumb
  with an "active" chip, two-column layout (vocabulary + corrections on the
  left, media folders + project speakers sidebar on the right, stacking on
  narrow viewports). Vocabulary terms render as bordered chip pills; the edit
  form is in a collapsible disclosure. The corrections grid-table shows the
  existing React editor island. Speakers in the sidebar show initials badges
  with verified/needs-you status chips. Completes epic #205 (10/10 checklist
  items done).
- **Error copy refactor** (#232). Internal-vocabulary error strings rewritten
  for operators: replaces technical jargon (adjudication, diarization, embedding)
  with plain-language equivalents throughout user-facing error messages.

## [0.26.0] - 2026-08-27

### Added
- **Synthdetect M1 S7: Platt calibration and holdout evaluation** (#144).
  Piper-only calibration policy (`w2v2aasist-s6-piper-only`): Chatterbox
  strata excluded from Platt fitting because near-chance EER (48.25% on
  holdout) poisons the slope for detectable generators (A flattens from 0.96
  to 0.32). Holdout opened exactly once (1420 clips): Piper-only EER 18.04%
  (calibration 19.01%), Brier 0.140 (calibration 0.136), confirming the
  policy generalizes. Coverage statement documents five findings: VITS-family
  and commercial TTS detectable at 10-20% EER; Chatterbox-class TTS
  uncalibrable; ASVspoof benchmark healthy; VoxConverse channel confound; FPR
  5% operating point is a high-confidence flag (TPR ~16%), not a reliable
  filter. `calibrate_policy` gains `exclude_strata` parameter for fitting on
  a generator subset.
- **Synthdetect M1 S6: spoof corpus, composite assembly, and first EER**
  (#144). Four TTS generators materialized (Piper 2540, Chatterbox 2540,
  ElevenLabs 969, Google Cloud TTS 969 clips) plus ASVspoof 2021 DF
  benchmark anchor (53,392 clips). v3 composite manifest (63,905 clips).
  First EER measurement: primary 34.31%, benchmark anchor 7.05% (healthy).
  Per-generator and per-domain EER matrix reveals Chatterbox AR+flow TTS
  is near-undetectable (EER 45.32%, independently confirmed by DFADD 2024
  at 44.21% for flow-matching TTS vs AASIST). VoxConverse channel confound
  identified as a separate issue. Scoring results, 4-model analysis, and
  proceed-to-calibration decision documented in `gpu-contracts.md`.
- **Synthdetect M1 S5 PR-5: windowing verdict** (#144). Per-window score
  journaling (`window_scores` field in `ClipOutcome`) and `verdict-windowing`
  subcommand in `synthdetect_eval.py` for comparing upstream vs production
  windowing FPR on bona fide clips. GPU verdict on the full S5 calibration
  corpus (4654 clips): production windowing does not raise false-positive rate
  at FPR 5% or FPR 1%. Verdict report and `gpu-contracts.md` updated.
- **Synthdetect M1 S5 PR-3: cohort freeze** (#144). Frozen calibration cohort
  policy: exactly one degraded child per eligible calibration-split turn parent,
  assigned via stable `sha256(clip_id)` hash from the six single-recipe chains.
  Multi-recipe compound chains are excluded from v1. New constants
  `FROZEN_COHORT_CHAINS`, `S5_COHORT_VERSION`, `S5_COHORT_SELECTION_POLICY` in
  `synthdetect_sources.py` (SOURCES_VERSION bumped to v5). New `plan_cohort`,
  `CohortPlan`, `materialize_cohort`, `CohortResult` in `synthdetect_corpus.py`.
  New `freeze` CLI verb (dry-run prints plan + `cohort_plan_sha256`; execution
  materializes atomically with `cohort_receipt.json`). 37 new tests (30 unit +
  7 contract).
- **Ops Console COPY: vocabulary contract and stale-term audit** (#209, epic
  #205). Canonical plain-language vocabulary mapping (`vocabulary.py`): maps
  12 internal terms to user-facing equivalents (diarization -> "separate
  voices", adjudication -> "review", embedding -> "voice sample", etc.). A
  grep-based contract test (`test_vocabulary_contract.py`) catches prohibited
  terms in rendered template text, stripping Jinja2 constructs, HTML comments,
  CSS, and scripts before scanning. Legacy templates, Settings detail pages,
  and the RTTM export disclosure carry explicit allowlists until their R-issue
  refreshes apply the vocabulary.
- **Console 2.0 P2c: journaled media operations primitives and executor**
  (#155, epic #149, ADR 0007). Concurrency primitives for the media operations
  journal: `claim_operation` (row-lock serialized), `cas_transition` and
  `cas_pointer` (CAS state machine), `has_active_operation` and `has_active_run`
  guards, `lock_media_row` (`FOR UPDATE`). Run admission in ingest rejects
  trashed/purged/active-op items. Prepare stage defers when a media item has an
  active operation. `openable_source` renamed to `openable_path` (generic
  media-root-relative resolver); `openable_current` resolves `current_path` with
  `source_path` fallback. Reclaim alias guard checks both `current_path` and
  `source_path`. The filesystem executor (`media/executor.py`) drives move,
  trash, and restore through the ADR 0007 state machine: plan, claim, publish
  (same-device link+rename or cross-device durable sequence), CAS pointer,
  metadata, cleanup. Destination-collision handling per R3 (replay detection,
  owned-temp cleanup). Transient-error retry with exponential backoff. Purge
  refused (step 5). `_trash` added to setup wizard reserved trees.
  Library UI (step 6): trash, restore, and empty-trash routes behind the
  `console_media_enabled` flag. Active view excludes trashed/purged items; new
  trash view shows trashed-not-yet-purged items with trashed-at timestamps.
  Each operation is planned durably, then executed inline for immediate feedback;
  the reconciler beat is the crash-safety net. Missing-file warning: the library
  row surfaces a visual indicator when a media file's `current_path` does not
  resolve to a regular file, rather than crashing.
- **Synthdetect M1 S5 PR-2b: degrade executor** (#144). Materializes
  codec/telephony/speed-degraded children of bona fide clips via the frozen
  `build_recipe_argv` ffmpeg round trips inside a digest-pinned container
  (`jrottenberg/ffmpeg:7.1-ubuntu`). Resolves the three PR-2a deferred blockers:
  B1 (raw/WAV framing through headerless s16le), B2 (combined parent+child
  manifest validated through `load_manifest`), and B5 (separate immutable degrade
  root published by atomic rename). New functions: `materialize_degrade`,
  `_run_containerized_ffmpeg`, `_degrade_one_clip`, `_assemble_combined_manifest`,
  `resolve_clip_path`, `_write_degrade_artifacts`, `DegradeResult`. The `degrade`
  CLI verb gains `--corpus-root`/`--parent-root`/`--container-image` flags for
  execution mode (unchanged dry-run without them).
- **Console 2.0 P7: run-completion activity indicator and toasts, dark-shipped**
  (#162, epic #149). Behind a new `CONSOLE_ACTIVITY_ENABLED` flag (off by
  default), a completed pipeline run writes one row to a new `activity_events`
  outbox in the same transaction as the completion, and the console polls
  `/activity/events` to raise a toast ("Transcription finished: file. Review
  speakers.") and show a live-jobs count on the Jobs entry. The badge reads the
  same query the Jobs page renders, so the two always agree; the outbox is kept
  to its newest 500 rows by an opt-in retention sweep. Nothing is reachable until
  the flag is on (the endpoint answers 404, the shell renders no chrome), and
  activation additionally requires Jobs discovery so the badge and links stay
  honest. No redirect or version bump ships here; that activation is a later slice.
- **Console 2.0 P7: speaker-identification activity events, dark-shipped** (#162,
  epic #149). Behind the same `CONSOLE_ACTIVITY_ENABLED` flag, naming a
  diarization label now raises a "Speaker identified: name." toast. The event is
  written to the `activity_events` outbox in the same transaction as the ledger
  ruling, so it appears only if the ruling commits. It fires from the four
  adjudication paths (assign a label, override one segment, enroll a new speaker,
  merge labels), and only for a positive identification: excluding a label,
  marking it unknown, resetting a segment, or re-asserting a label's current
  speaker stays silent. A merge raises one toast for the whole consolidation, not
  one per label. The Jobs badge is unchanged (it stays a live-jobs count; speaker
  events affect toasts only). No version bump.
- **synthdetect S5 organic-corpus tooling, pure layer** (#144, M1 S5). The
  audio-free half of the organic (real-speech) corpus lands frozen and unit-tested
  before any audio exists. `tools/synthdetect_corpus.py` gains a strict RTTM
  parser, a pinned decimal-to-sample rule (`floor(start*16000)`,
  `ceil(end*16000)`), an overlap-cleaning and same-speaker merge planner that emits
  two views (per-turn clips for strata and calibration, merged session segments for
  production-windowing validation), a `MaterializationPlan` schema, and a
  `finalize_manifest(plan, measured_facts)` that builds the v1 manifest only from
  the executor's measured PCM sha256 and sample count. It also gains a closed,
  versioned degradation-recipe vocabulary (`DEGRADATION_RECIPES` in
  `tools/synthdetect_sources.py`, `SOURCES_VERSION` bumped to
  `synthdetect-sources-v4`: codec, telephony, and speed families), deterministic
  ffmpeg argv builders (real encode-decode-canonical-PCM round trips, `-threads 1`,
  final PCM payload sha as identity), degraded-child derivation with lineage
  inheritance, and hardened manifest lineage invariants (parent-cycle rejection,
  child inherits its parent's label/speaker/language/split/license, and a
  degradation string must name known recipe ids). New `prepare` and `degrade` CLI
  subcommands emit a plan in dry-run mode only (they extract no audio). Additive
  noise is deferred to the executor slice (its SNR mix needs a measured parent
  RMS). A multi-model implementation review hardened the numerics before landing:
  RTTM times parse and scale in exact decimal (not binary `float`), so the pinned
  sample rule is byte-exact for ordinary decimals such as `0.1`; the overlap floor
  applies to coalesced continuous other-speaker regions, so word-level crosstalk
  cannot survive inside a turn; both length floors are enforced in the sample
  domain; the ffmpeg argv pins `-threads 1` on the encoder output as well as the
  input, plus `-filter_threads 1`; `finalize_manifest` rejects measured facts for
  clips not in the record list; manifest lineage also inherits `source`; and
  `degrade` skips already-degraded parents so a re-run never plans grandchildren.
  Pre-registration in `docs/gpu-contracts.md`.
- **synthdetect S5 PR-2a prepare executor** (#144, M1 S5). `prepare` gains an
  execution mode (`--corpus-root`, with `--audio-dir` and `--acquisition-manifest`)
  that materializes bona fide turn and segment clips from the staged organic
  recordings; without a corpus root it stays the dry-run planner. Because the staged
  sources are already canonical PCM (16 kHz mono s16le, measured), materialization is
  ffmpeg-free: the executor pin-verifies every RTTM and recording against the
  acquisition manifest on the exact bytes it reads (open-once, no reopen race),
  slices each clip at the plan's integer sample offsets from the validated payload,
  wraps each slice in a deterministic canonical WAV, finalizes and validates the v1
  manifest, and publishes the whole tree by atomic rename (refusing a populated
  root). New numpy-free canonical-WAV primitives (`read_canonical_wav_payload`,
  `write_canonical_wav`, `payload_sha_and_count`) share the data-chunk payload
  identity with `synthdetect_infer.read_canonical_pcm`; a contract test pins their
  canonical constants equal so a corpus materialized by one and scored by the other
  can never disagree. The degrade executor and its codec toolchain are a separate
  later slice (PR-2b).
- **synthdetect Gate-1: full-cohort DF benchmark reproduction PASS** (#144, M1
  S4). The verbatim upstream SSL_Anti-spoofing model and data modules (under a
  thin, audited reference driver) were run over the full official ASVspoof 2021 DF
  eval cohort (611,829 trials) on the native FLAC tree,
  and the pooled EER over the 533,928 phase-`eval` trials was computed with the
  official `eval_metrics_DF.compute_eer` math and cross-checked against an
  independent EER routine. Result: **2.8650 % EER** against the published 2.85 %
  and the pre-registered ±0.3 pp tolerance, a +0.015 pp miss that PASSES with wide
  margin. The DF anchor `w2v2-aasist-df` checkpoint (sha256 `1cf904f1…`) and the
  official keys (sha256 `426f93e1…`) are pinned; every trial was scored (the
  official length check) and the protocol forms an exact bijection with the native
  tree. With Gate-2's ratified per-clip subset equivalence this is strong evidence
  that the shipped fp32 container reproduces the benchmark too, though the
  container's own full-cohort EER is not measured here (a deferred pass). Verdict
  in
  `docs/gpu-contracts.md`; evidence in
  `docs/reports/synthdetect-gate1-s4-2026-08-25.md`.
- **Console 2.0 P6a: settings sub-pages and plugins UI, dark-shipped** (#161,
  epic #149 Track D). Behind a new `CONSOLE_SETTINGS_ENABLED` flag (off by
  default), `/settings` becomes a hub that keeps every existing section inline
  (its anchors intact, so deep-links keep working) grouped into everyday and
  advanced categories, and links out to four new read-only sub-pages: `/settings/
  status` (install kind, live component health, hardware snapshot), `/settings/
  hardware` (effective configuration with the exact env key and restart to change
  it, and a restart-pending badge when the environment changed after boot),
  `/settings/database` (size, per-table row estimates, retention settings, backup
  guidance), and `/settings/plugins` plus `/settings/plugins/{id}` (the plugin
  registry with an honest empty state today and per-plugin pages that render a
  plugin's own settings section). The sub-pages are registered unconditionally
  and reachable by URL regardless of the flag, render honestly when the model
  services or Redis are unavailable, and never edit the environment or restart a
  service. No `/resources` redirect or version bump ships here: that activation
  is a later slice.
- **synthdetect Gate-2: DF reproduction paired equivalence RATIFIED** (#144, M1
  S3). The frozen `w2v2-aasist-df` eval container was measured against an
  independent, unmodified-upstream reference (verbatim upstream `model.py` +
  `data_utils_SSL.py` at the pinned commit, in an image derived from the frozen
  eval image with only `librosa` added) on the 53,392-clip canonical DF subset.
  The decision threshold was frozen from the reference before agreement was
  inspected. Result: subset EER identical (2.2193 % both), Spearman 0.99858,
  decision agreement 99.985 % (every disagreement within 0.0024 of the
  threshold), exact coverage. The per-clip logit tail (max 0.152) is attributed
  entirely to cuDNN convolution TF32 (upstream leaves the Ampere default on; our
  runner disables it for full fp32); with TF32 matched the reference reproduces
  our container's scores exactly, so the two are the same function and the tail
  preserves the EER and ranking. No shipped-runner change is warranted. Verdict in
  `docs/gpu-contracts.md`; evidence in
  `docs/reports/synthdetect-gate2-2026-08-25.md`. The 2.85 % full-corpus EER
  (Gate-1) defers to S4.
- **Console 2.0 Jobs area, dark-shipped** (#160, epic #149 Track C). Two new
  console pages: `/jobs` shows a per-stage pipeline-activity strip
  (queued/active), the per-service hardware-health strip, a compact recent-runs
  table, and a normalized list of the auxiliary job families (asset, translation,
  embedding, research); `/jobs/{run_id}` absorbs the `/runs/{id}` run-detail page,
  rendering the same sections from a shared body partial so the two never drift.
  The strip's queued totals break the queued run count `voxint stats` reports
  down by stage (they sum to it exactly), so the page and the CLI cannot silently
  disagree. `/jobs/{id}` reuses
  the existing `/runs/{id}/...` action endpoints rather than minting aliases. Both
  pages are registered unconditionally and reachable directly; the new
  environment flag `console_jobs_enabled` (default off) gates only the sidebar's
  Jobs entry, which stays on the `/runs` placeholder until the flag is on. No
  redirect, tutorial remap, or version bump ships here: the `/runs` retirement is
  a separate coordinated slice, so `/runs` and its detail page are unchanged.
- **Console 2.0 P4: speakers overview and profile pages** (#159, epic #149).
  Behind a new `CONSOLE_SPEAKERS_ENABLED` flag (off by default), `/speakers`
  becomes an overview of everyone Voxint has heard: per-speaker files, spoken
  time, first and last file added, a Verified badge (at least one human
  assignment that still wins after later rulings), and a strong/moderate/weak
  voice-match chip graded against the same gates the matcher enforces, with the
  numbers behind a "why this match?" reveal. Card and table views, sortable,
  with the existing rename/merge/archive/restore actions, plus an action
  reminders strip (pending speaker-name suggestions linking each recording's
  review workbench; frequently heard voices not yet confirmed). Each name links
  a new `/speakers/{id}` profile page: stats header, a per-field profile
  (bio, affiliation, link) with provenance chips and manual edit/clear, the
  web-research block (accepting a claim updates the profile panel in place),
  and the recordings the speaker appears in. A merged-away speaker's URL
  redirects to the speaker that absorbed it; archived speakers read-only. All
  numbers come from effective resolution: one canonical (newest completed) run
  per recording, human rulings winning over automatic matches, so reprocessing
  a file never inflates anyone's minutes. With the flag off, `/speakers` keeps
  the current roster page unchanged and the profile routes 404.
- **Speaker profiles are now stored rows** (migration 0041, applies flag-off).
  Accepting a bio, affiliation, or link research claim now also materializes
  the value into a new `speaker_profiles` table (one current value per field,
  with provenance and the accepted claim's id); existing accepted decisions are
  backfilled. Draft-claim history is untouched, a replayed accept can never
  overwrite a later manual edit, and merging speakers carries profile rows to
  the surviving speaker (its own values win a per-field conflict).
- **Synthdetect M1 S3: ASVspoof 2021 DF importer core + v2 imported-benchmark
  manifest** (#144). Groundwork for the DF reproduction gates. The corpus manifest
  schema (`tools/synthdetect_corpus.py`) gains a `schema_version: 2`
  `imported_benchmark` variant so an acquired evaluation benchmark can be scored
  by the frozen runner without fabricating synthesis provenance: a v2 clip carries
  an `imported_provenance` block (official trial id, source dataset, codec
  condition, official split, vocoder family, and for spoof clips the attack
  system), `generator` is always null, and an officially-absent field is null
  rather than a placeholder (sentinel strings are rejected on the identity fields;
  the vocoder family is recorded as-is because the official metadata uses the
  literal `unknown` as a real family). A v2 clip's
  provenance is bound to its scoring identity: the official trial id must equal
  the clip id, the clip is eval-only, and the stratum must match the official
  label and codec. Only `load_manifest` learns the variant; the scoring fields
  and `cmd_run` are unchanged, and v1 acceptance is unchanged for every valid v1
  manifest (both v1 and v2 now also reject unexpected top-level keys, consistent
  with the module's existing per-clip unexpected-key rule). A new
  `tools/synthdetect_df_import.py` holds the audio-free importer core:
  a fail-closed parser for the official `trial_metadata.txt` and the frozen,
  seeded (`voxint-synthdetect-144`), stratified (label by codec condition, `eval`
  split only) hash-rank selection of the 10 % subset, emitting a canonically
  ordered trial list bound by a cohort hash. The same tool now also carries the
  audio-dependent emission verb: `emit` verifies the operator's official archives
  by pinned sha, safely extracts an untouched native FLAC tree from those verified
  bytes (rejecting absolute, traversing, linked, or duplicate members), transcodes
  each selected trial's exact native FLAC to the canonical
  `pcm-s16le-mono-16000-v1` view with ffmpeg (no resample: a non-conforming source
  is rejected before and after), and writes a v2 `imported_benchmark` manifest
  whose per-clip sha is the PCM payload only, alongside a per-trial receipt that
  binds each pinned-archive FLAC to the exact canonical PCM the manifest scores.
  Nothing is published until the whole corpus re-reads and revalidates, so a
  partial or misbound corpus is never left behind; an audio-free `select` verb
  emits just the trial list and selection receipt. The shape was chosen after an
  independent two-model design review; see the S3 pre-registration refinement note
  in `docs/gpu-contracts.md`. The four official ASVspoof 2021 DF eval archives
  are now pinned by sha256 (each cross-checked against the md5 Zenodo publishes),
  and the emission verb has an acceptance run on the real archives: it extracted
  the full 611,829-clip native tree, transcoded the 53,392-clip subset, and
  reproduced the pre-registered cohort hash, with each sampled clip's manifest
  PCM sha equal to its receipt canonical sha and its receipt native FLAC sha
  equal to the recomputed FLAC sha (see
  `docs/reports/synthdetect-df-emit-acceptance-2026-08-25.md`).
- **Attributed audio-clip extraction** (#88, completes the operator-output-layer
  arc). A `word`-timed highlight can be extracted as a standalone WAV clip: the
  exact span of the run's normalized 16 kHz audio the highlight covers, cut with
  the stdlib `wave` frame-copy (no re-encode, no editing). The Highlights panel
  gains an **Extract clip** action beside Copy, shown only for `word`-timed
  highlights; it downloads the clip via `Content-Disposition` without buffering
  bytes in the page. Two routes back it: `POST
  /review/{run_id}/annotations/{annotation_id}/clips` (CSRF-gated, no claim)
  generates or adopts the clip, and `GET|HEAD /runs/{run_id}/clips/{clip_id}`
  serves it with byte-range support and an attachment disposition. Clips are a
  content-addressed, idempotent, reclaimable cache: identical requests adopt one
  canonical row (advisory-locked on the content digest), a re-anchor makes a new
  clip, the stored snapshot is an immutable attribution record, and the clip
  serves independently of the source audio while a cache miss regenerates only
  while that source is live. The media GC sweep (issue #15) reclaims clip files
  too, aged by the clip's own `created_at`. Preconditions are enforced
  server-side: a foreign or soft-deleted highlight is 404, a stale one 409, and a
  non-`word`-timed one 422. Migration `0039` adds the `audio_clip` artifact kind,
  a content-addressed `idempotency_key`, and the live-only partial-unique index.
  See `docs/annotations.md`.
- **Synthdetect M1 S3: DF anchor GPU-qualified** (#144). The DF reproduction
  checkpoint `w2v2-aasist-df` (`Best_LA_model_for_DF.pth`) advances from
  `pinned_unqualified` to `qualified` on its own dated maintainer GPU verdict (one
  RTX 3060): strict checkpoint load, all-modules-eval, correct polarity and window
  counts, fail-closed on a wrong clip sha / a tampered weight / a header-identity
  change on resume, and a bit-exact determinism spike across four cold container
  starts (`docs/reports/synthdetect-gpu-smoke-df-2026-08-25.md`, plus a dated
  verdict in `docs/gpu-contracts.md`). The load was proven cold and found to
  differ from the default: the DF checkpoint was saved from an `nn.DataParallel`
  model, so all its keys carry a `module.` prefix. `WeightFile` gains a declared
  `state_dict_key_prefix` (data, not inferred; a closed vocabulary of `None` or
  `"module."`) that the runner strips from the keys and the state-dict `_metadata`
  before a strict load (rebuilt in two phases so the result never depends on dict
  order, and failing closed on any `_metadata` entry that does not fit the
  DataParallel layout), which on a single GPU is numerically identical to
  upstream's evaluation. The applied strip is recorded in the journal header
  (`checkpoint_loading`) and flows into the execution identity; the shipped default
  loads verbatim so its already-qualified path is untouched. The sources schema
  version moves to `synthdetect-sources-v3`.
- **Synthdetect M1 S3 reproduction pre-registration** (#144). The audio-deepfake
  eval harness gains its S3 protocol, frozen before any reproduction run and
  reviewed by two independent models. The registry now pins a second detector,
  `w2v2-aasist-df` (the DF-tuned `Best_LA_model_for_DF.pth`, PINNED_UNQUALIFIED,
  sha frozen from real bytes with a dated receipt): the published 2.85 % ASVspoof
  2021 DF EER is achieved with that checkpoint, so it, not the production default
  `w2v2-aasist`, carries the hard DF stop-gate. Reproduction targets grow a
  `gate_role` (`stop_gate` versus `diagnostic`), and the default's In-the-Wild
  number is reclassified as a diagnostic (no checkpoint-exact citable ITW anchor
  for its ASVspoof2019-LA checkpoint). The pre-registration in
  `docs/gpu-contracts.md` pins the checkpoint decision, the native-versus-canonical
  corpus split, the seeded subset selection rule, the Gate-2 tolerance procedure,
  and the fact that S3 delivers Gate-1 readiness plus ratified Gate-2 tolerances
  while the full-cohort Gate-1 PASS is deferred to S4. The registry's import-time
  integrity rails are extended to match the doctrine they enforce: weight shas
  and model commits are format-checked (not merely non-null), `gate_role` is
  required with no silent default, the shipped default may carry no stop-gate,
  and at most one runnable model may claim the DF anchor. The sources schema
  version moves to `synthdetect-sources-v2` for the new `gate_role` field.
- **Navigable outline on the read-only transcript** (#87 follow-up). The
  `/runs/{id}/transcript` page now shows the same navigable outline the review
  workbench does: grounded people, organizations, and products become
  click-to-seek jump targets above the transcript, with summary and topics as
  inert context. A jump reveals the line (and plays from it when audio seek is
  available), so it works even on a media-less run. Without JavaScript the panel
  degrades to a readable, non-interactive list, which both surfaces now render
  from one shared template so the two fallbacks cannot drift. No new server
  data: the outline already rode in the shared transcript island props.
- **Console 2.0 groundwork, P2a: the projects and media-folders schema** (#153,
  epic #149). Additive schema with no operator-visible change on its own: new
  `projects` and `media_folders` tables and a media location split
  (`media_items.current_path`, backfilled to `source_path`, and a
  `media_folder_id` folder membership FK), the relational foundation the console
  library, projects, and project-scoped configuration build on. The migration
  carries each registered folder over from the `app_settings` list, seeds folder
  membership by deepest-ancestor, and refuses to change any run's effective
  domain pack (an install with nested folder registrations is asked to reconcile
  them first). The `app_settings.media_folders` / `folder_domain_packs` columns
  are kept for one release as a rollback input.
- **Console 2.0 P2a: the media library page** (#153, epic #149). A new `/media`
  page lists every recording Voxint knows about with its folder, length, size,
  and the status of its latest run (one aggregate query, not one per file;
  archived runs are excluded so a file reads as "not processed yet" honestly).
  It offers a card or table view and sorts by newest, name, longest, or largest,
  with an unknown sort or view degrading to the default rather than erroring. The
  area ships dark behind `CONSOLE_MEDIA_ENABLED` (off by default): the route is
  always registered so the console route inventory is stable, but returns 404
  until the flag is on. Reachable by URL for now; the sidebar's Media link is
  repointed in a later phase.
- **Console 2.0 P2b: the media library becomes operable** (#154, epic #149).
  With `CONSOLE_MEDIA_ENABLED` on, `/media` gains everything needed to run the
  library from one page. File upload and URL fetch move onto it (each can pick a
  settings folder whose vocabulary and corrections apply to the run without moving
  the file), and a folder panel registers or unregisters folders. A multi-select
  drives non-destructive bulk actions over the chosen files: assign a settings
  folder (or clear it), re-run transcription, and archive or restore the latest
  run. Re-run is a two-step, advisory flow: a preview shows the config each fresh
  run would freeze and the config is re-resolved when you confirm, so a
  double-confirm mints at most one new run per file and a file whose config or
  sidecar cannot be resolved is skipped, not failed. An archived view
  (`/media?archived=1`) lists files whose latest run is archived so they can be
  restored. Every bulk action validates the whole selection first and reports a
  per-item or counted outcome, never a silent partial change; assigning a settings
  folder never moves the file on disk. The sidebar Media link and the Home "Add
  media" quick action point at `/media` once the flag is on, and at the legacy
  `/runs` upload otherwise; the legacy `/runs` page is unchanged. Track A files
  only, no schema migration; the whole slice stays dark until the flag is on.
- **Console 2.0 P2b: the projects pages** (#153, epic #149). New `/projects` and
  `/projects/{id}` pages let an operator create a project, assign registered
  folders to it, and see its folders and the speakers its finished recordings
  resolve to. A project's speakers are derived, not stored: they come from the
  shared adjudication resolver, so a human ruling wins over a grounded automatic
  match exactly as it does in the review workbench. Assigning a folder that
  carries a domain pack states the precedence plainly, so a project's own
  vocabulary later taking over is never a surprise. The area ships dark behind
  the existing `CONSOLE_PROJECTS_ENABLED` flag (off by default), which also
  governs whether the sidebar shows the Projects link; the routes are always
  registered so the console route inventory is stable but return 404 until the
  flag is on.
- **Console 2.0 P2a: project-scoped vocabulary and corrections** (#153, epic
  #149). The project detail page gains two editors: a vocabulary list and the
  corrections editor (the same one Settings uses, with the same validation).
  Once set, a project's vocabulary or corrections apply to every new run under
  the project's folders in place of the folder pack's and the global settings'.
  Each editor can inherit (use the folder pack or global settings) or be set for
  the project, where an empty list means explicitly none. This applies to new
  runs only; existing runs keep the configuration they were frozen with.
- **Console 2.0 P1: app shell and Home** (#152, epic #149). The console gains
  its new frame: a collapsible left sidebar (Home, Media, Projects behind a
  dark-ship flag, Speakers, Jobs, Settings, with Review and Hardware kept as
  interim links so nothing loses its path), a top bar with a theme cycle
  button, and shared library primitives for the area pages later phases
  build. `/` is now a real Home page: needs-attention cards (continue
  review, unidentified voices, failed runs) whose counts derive from the
  queues they link to, quick actions, activity counts (media added, runs
  started, speakers enrolled) with an hour/day/week/all switch that shares
  its query functions with `voxint stats`, and a recent-activity feed
  derived from existing tables. `voxint stats` and `GET /metrics`'s JSON
  sibling gain the two new windowed counts. The guided tutorial's
  page-to-route map is data-driven now, and unfinished console areas ship
  dark behind environment flags (`CONSOLE_PROJECTS_ENABLED`).
- **Navigable outline: grounded entity-mention jump targets** (#87, epic #47).
  The review transcript gains an Outline panel that turns grounded entity
  mentions into clickable jump targets: each seeks the player to the mention's
  segment timestamp. Summaries and topics have no source spans, so they render
  as inert context, never as click-to-seek navigation. The panel is honest by
  construction: the seek target is the segment's immutable start time, so a jump
  stays valid even when the quoted text has drifted; an occurrence whose segment
  no longer exists is dropped and counted; an outline built from an earlier
  version of the transcript shows a single "may be out of date" banner while
  jumps keep working; and the empty state distinguishes "none generated yet" from
  the feature being off. Gated on LLM enhancement plus the run-assets feature.
  The server ships each target's start time only and the client resolves it to
  the current line at click time, so a jump lands correctly after a segment
  split. Includes a JS-off read-only fallback that lists the same grounded
  entities as inert text.
- **Plugin seam wiring** (#138, epic #136). Wires the (still empty) plugin
  registry into every core seam so a future plugin activates through generic
  loops instead of a hand-wired copy: API router mounting with route collision
  rejection, the settings-section and run-detail-panel render loops, the
  Features-section flag merge, namespaced plugin templates, the Celery task
  include and routing merge with a guard against a plugin shadowing a core task,
  the post-completion hook fan-out, the stale-job recovery sweep generalized from
  the embedding lane, the CLI subcommand loop, and a boot-time plugin invariant
  check. This change is dormant: the registry is empty, so behavior is
  byte-identical and nothing an operator sees changes yet. The framework is for
  future greenfield plugins; the existing optional features stay native (epic
  #136, ADR 0006).
- **Console 2.0 groundwork, P0a: contracts, characterization tests, and a
  content-hash backfill** (#150, epic #149). Schema-free foundation for the
  review-console information-architecture refactor. It adds architecture
  decision records under `docs/adr/` (media identity versus location with a
  grep-verified byte-opener audit, project membership, editor run selection,
  claim lifecycle, speaker profile provenance), characterization contract tests
  that pin the current route surface (path, method, onboarding gate, operator
  auth), the CSRF coverage of every mutating route, and a declarative redirect
  map for later phases to extend. Nothing an operator sees changes yet. One new
  maintenance command ships: `voxint media backfill-hashes` computes and stores
  `sha256` for media rows that lack it (an integrity aid, never identity;
  idempotent, and a row whose bytes are gone is reported and retried later).
- **Plugin framework substrate** (#137, epic #136). Internal groundwork for
  building self-contained, flag-gated plugins on top of Voxint. This change ships
  the framework only: an empty registry, the plugin interface, and the guard
  tests that keep future plugins safe (route and task inventories,
  import-direction checks). Nothing an operator sees changes yet. One new setting
  is documented: `VOXINT_PLUGINS_DISABLED`, a comma-separated kill switch for
  turning built-in plugins off in a process (inert while the registry is empty);
  `voxint doctor` reports active plugins and flags an unknown id in the switch.
  The framework is for future greenfield plugins; the existing optional features
  stay native (ADR 0006).
- **Groundwork for synthetic-speech (audio deepfake) detection (#143, #144).**
  This change adds the maintainer evaluation harness only; nothing is exposed to
  users yet. The first milestone is a CI-only scoring toolkit: a pins-as-data
  model registry that gates each detector by license class (a model with no
  license refuses to run), a manifest schema with seeded, speaker-disjoint corpus
  splits, and a host-side scorer for equal-error rate, paired per-clip
  comparison, and Platt calibration. The pre-registered evaluation gates and
  tolerances are recorded in `docs/gpu-contracts.md` before any measurement runs.
  The detector itself, its model service, and the review-console plugin come in
  later milestones (#145, #146).
- **Synthetic-speech detection eval container and inference runner (#144).**
  Still maintainer-only, nothing user-facing. This adds the pinned CUDA eval
  container (`services/synthdetect/`, torch cu118 plus fairseq at a frozen
  commit, with weights mounted rather than baked) and the inference runner
  (`tools/synthdetect_infer.py`) that scores a corpus into the raw-score journal
  the host scorer reads. The runner verifies each clip against a fixed
  canonical-PCM digest without resampling, records the full determinism
  provenance in the journal header, and resumes an interrupted run only when the
  execution identity matches. A `verify-sources` pass produces a dated weight
  receipt for the S2b freeze. The weight pins stay CANDIDATE until a maintainer
  freezes real bytes and commits GPU determinism evidence.
- **Synthetic-speech detection: default detector frozen and qualified on GPU
  (#144).** Still maintainer-only, nothing user-facing. This wires the real
  fairseq forward pass behind the runner's engine seam (the upstream
  wav2vec2-XLS-R plus AASIST model, vendored verbatim and sha-pinned under
  `tools/synthdetect_vendor/`), freezes the default detector's weight, runtime,
  and model-repository pins from real downloaded bytes, and advances the eval
  runtime to `qualified`. The engine verifies each weight against its frozen
  sha256 before loading, loads the checkpoint strictly, and asserts every module
  is in eval mode so the run is deterministic. On maintainer RTX 3060 hardware the
  GPU smoke passed (strict load, correct per-window counts and score polarity,
  resume without duplication, and weight/audio/header mismatch failing closed) and
  scores were bit-for-bit identical across four cold container starts. Evidence is
  recorded under `docs/reports/` and `docs/gpu-contracts.md`. Contract tests now
  bind the vendored model file to its pinned upstream hash and require the
  `qualified` state to name existing GPU evidence, so neither the numerics-defining
  bytes nor the qualification claim can drift silently. Reproducing the published
  ASVspoof 2021 DF error rate remains a later milestone.

### Changed
- **Ops Console R3: media overview refresh** (#212, epic #205). The `/media`
  page is rewritten with the V3 design system: a command bar with breadcrumb,
  summary stats (folder/file count, total hours), filter field, table/cards
  toggle, and an add-media dropdown (upload, pick from server, fetch from link)
  replacing the old stacked forms. Folder rows expand in place with a disclosure
  triangle showing aggregate stats (item count, duration, review chip) and
  child file rows indented underneath. File rows show a status chip, duration,
  size, and a single action link. Trash and restore views use the same grid
  table with state-appropriate tints. Default view changes from cards to table.
- **Ops Console V3: shared primitives** (#208, epic #205). Semantic chip system
  (`.chip` + 6 semantic variants: ok/warn/danger/info/accent/neutral) with a
  canonical label-to-semantic mapping (`chip_semantics.py` + Jinja2 macro
  `_chips.html`). Ops Console stat tiles (`.oc-stat-tile` with mono numerals,
  micro-labels, and a time-window segmented control). Grid-row data views
  (`.grid-table` with micro-headers, subtle row dividers, right-aligned mono
  numerals, teal action links, row-state tints, and a multi-select action bar).
  Command-bar action styles (`.cb-btn` secondary/primary/danger variants).
  Contract test pins the chip mapping and verifies CSS class existence.
- **Ops Console V2: icon rail and command-bar framework** (#207, epic #205).
  The 13rem collapsible sidebar is replaced by a 52px icon rail with inline SVG
  glyphs (Home square, Media circle, Projects diamond, Speakers ring, Jobs
  bars), a teal "V" brand square, CSS tooltips with aria-labels, and a
  settings entry with a system-health dot at the bottom. The topbar is replaced
  by a command-bar framework: a 40px strip with Jinja2 block slots for
  breadcrumb (`cb_breadcrumb`), summary (`cb_summary`), search (`cb_search`),
  and actions (`cb_actions`). Pages that don't fill a slot get sensible
  defaults (page name as breadcrumb, placeholder search, theme toggle). On
  narrow viewports the rail collapses to a horizontal disclosure behind a
  hamburger in the command bar. Shell-less pages (setup wizard) unaffected.
  Print excludes shell chrome. No new runtime dependencies.
- **Ops Console V1: tokens, typography, and theme contract** (#206, epic #205).
  Dark-theme palette updated from the warm "Reading Room" values to the Ops
  Console direction: darker canvas (#121316), cooler ink, alpha-based semantic
  soft fills, new `--line-subtle` token. Typography switches from the system-ui
  stack to self-hosted IBM Plex Sans (400/500/600) and IBM Plex Mono (400/500)
  as latin-subset woff2 (101 KB total), loaded via the Vite build pipeline with
  system-ui fallback when unbuilt. Type scale tightened (body 13px, new
  `--t-micro` for 10.5px micro-headers). Light theme and print-stays-light
  contract preserved. No layout or shell changes.
- **Console 2.0 P6b: Settings activated; the hardware page folds into it** (#161,
  epic #149 Track D). The settings hub and its read-only sub-pages
  (status, hardware, database, plugins) are now the default `/settings`, so the
  console opens on the grouped hub rather than the single long page
  (`CONSOLE_SETTINGS_ENABLED` still gates the content and defaults on; set it off
  to fall back to the flat page). The old `/resources` hardware view is now
  `/settings/status`: `/resources` issues a permanent redirect there, the sidebar
  "Hardware" shortcut and the Jobs "Full hardware view" link point at it, and the
  status page keeps the 15-second live hardware refresh (it answers the poll with
  just the hardware snapshot, without re-running the component-health checks).
  Bookmarks and an already-open Resources tab keep working through the redirect.
- **Test loop and PR CI parallelized.** The DB-backed integration suite now runs
  under `pytest-xdist` with one disposable database per worker
  (`tests/integration/conftest.py`), keyed on the xdist run id and worker so
  concurrent invocations no longer deadlock on a shared `DROP SCHEMA`. Measured
  ~6x faster at `-n 8` (13m24s serial to 2m14s) on a multi-core box, and the
  serial (no-`-n`) path is byte-identical to before. The PR gate (`ci.yml`) drops
  coverage instrumentation from the fast required lane (a measured ~33% wall-clock
  tax) into a separate, parallel, still-required `coverage` job; the fast lane
  runs the unit/contract suites parallel and the integration suite at `-n 8`. No
  change to the parity gate, the contract goldens, or the secrets scan. See
  `docs/testing.md`.
- **Pure driver tests moved to the unit lane; new coverage for the read-time
  attribution helpers.** The 27 database-free tests in
  `tests/integration/test_export_match_evidence_driver.py` (manifest parsing,
  serialization, atomic writes, git helpers, and the CLI paths that fail closed
  before any database access) moved to a same-named unit module, each assertion
  unchanged; the six tests that seed real runs stay in the integration lane. The
  CLI error-path tests now also assert the database is never reached on those
  paths. Added direct unit coverage for the pure read-time attribution helpers
  (`winning_attribution`, `display_name`, `segment_speaker`,
  `parse_transcript_text`), previously exercised only through database-backed
  walks. First slice of an incremental effort to keep the integration lane focused
  on what needs Postgres; the relocation rubric is documented in `docs/testing.md`.
- **Console 2.0 P2a: folder registration writes the media_folders relation**
  (#153, epic #149). The setup wizard folder step and the Settings folder panel
  now register folders as first-class `media_folders` rows through one shared,
  serialized write service instead of the `app_settings` list, and the
  submit-time domain-pack snapshot, the folder scan, and the watch-folder sweep
  all read the folder set from that relation (no dual-write). Each run submitted
  under a registered folder records its folder membership
  (`media_items.media_folder_id`). One operator-visible change: registering a
  folder that nests inside (or contains) an already-registered folder is now
  refused, so every file belongs to exactly one folder; register only the parent
  or only the child. The `app_settings.media_folders` / `folder_domain_packs`
  columns are no longer written and remain only as a one-release rollback input.
- **Console 2.0 P2a: per-field config resolution and the submit-time freeze**
  (#153, epic #149). A new run's effective vocabulary and corrections now resolve
  by first present layer, each independently: an explicit per-run pack (a CLI or
  sidecar name), then the project, then the folder pack, then the global baseline
  (the default pack plus the operator's glossary and corrections). Both fields
  are frozen at submit into the run's snapshot, tagged
  `config_resolution_version: 2`, so a later settings edit cannot change a run
  already queued. One behavior change worth noting: a run that resolves to an
  explicit folder pack, or to an explicit CLI or sidecar pack, no longer also
  inherits the global glossary and corrections; before, both were added to every
  run. A run with no project and no folder pack still resolves to the default
  pack plus the global settings exactly as before, so an install that never set a
  per-folder pack sees no change. Existing runs are untouched; no migration
  rewrites their snapshots.
- **The dashboard folded into Home** (#152). `/dashboard` now issues a
  permanent redirect to `/`, and `/` no longer redirects to `/review`. The
  dashboard's task cards and stat figures live on Home; the hardware strip
  stays on the Resources page (sidebar: Hardware); the per-status and
  stage-timing detail tables are retired from HTML (the same figures remain
  on `GET /metrics` and `voxint stats`). The pre-onboarding setup wizard
  renders without the new shell.
- **Console 2.0 groundwork, P0b: the console API is decomposed into per-area
  routers** (#151, epic #149). Internal refactor with no operator-visible
  change: the single 7.5k-line `api/app.py` module is split into
  `api/routers/{speakers,settings,legacy_review,legacy_runs}.py` plus shared
  dependency, triage, transcript-island, and tutorial-banner modules, with
  empty scaffolds for the areas later phases fill (home, media, projects,
  jobs, editor) and template files grouped into matching per-area
  directories. `app.py` keeps only the application factory, middleware, and
  route registration. Every route's path, method, status codes, onboarding
  gate, operator auth, CSRF verification, and registration order are pinned
  unchanged by the P0a characterization contracts plus a new
  registration-order golden added in this change.
- **Legacy submit redirect when Media library is enabled** (#9, refactoring
  batch 0F). When `console_media_enabled` is on, the `/runs` page replaces
  its upload and URL-fetch forms with a banner linking to `/media`. POST
  `/submit` and POST `/fetch` redirect to `/media` (303) instead of
  processing the upload, so a direct POST or stale browser tab never
  bypasses the banner.
- **Requeue refuses archived runs** (#7, refactoring batch 0D M9). A
  `RunArchivedError` now fires before the FAILED status check in
  `requeue_failed_run`, so an archived run cannot be driven live from
  any surface (the API's route-level guard was already there; the
  service-level guard protects the CLI path too).
- **Startup reconciler for orphaned incoming/ files** (#7, refactoring
  batch 0D M4). `reconcile_orphaned_incoming` scans `media_root/incoming/`
  at app startup and removes files that have no committed MediaItem row
  (crash orphans from the os.replace-before-commit window).

### Fixed
- **Synthdetect M1 S5 PR-4: production windowing fixes** (#144). Two
  pre-registered fixes to `plan_windows(mode="production")`: (1) production
  window width aligned to model input width (4.0 s / 64,000 samples changed to
  4.0375 s / 64,600 samples, eliminating 600-sample repeat-padding on every
  full window); (2) trailing partial windows below 8,000 samples (0.5 s) are
  dropped when at least one full window exists, preventing tiny tails from
  pooling with equal weight. `production_tail_floor_samples` added to
  `WindowingPolicy`; `dropped_tail_samples` journaled per clip in
  `ClipOutcome`. SOURCES_VERSION bumped to v6. Contract tests enforce
  production window == model width invariant.
- **Integration tests could touch the live database when run in isolation.** The
  `enrich` CLI tests invoked the app's own DB path without depending on the
  `engine`/`session_factory` fixtures, so in isolation `DATABASE_URL` was unset
  and the CLI fell back to the default live DSN and queried real data. They only
  passed in the full serial suite by inheriting an earlier test's fixture side
  effect. An autouse fixture now pins every integration test to its disposable
  per-worker database, closing both the data hazard and the flake the new
  parallel runner surfaced.

## [0.24.0] - 2026-08-23

### Added
- **Transcript translation, slice A: settings, jobs, and the run-page card**
  (#133). Voxint can now translate a finished transcript into a preferred
  language with the configured LLM (bring-your-own or bundled). A new
  Settings → Translation section holds the installation's preferred target
  language (picked from the vendored language list) and an opt-in
  auto-translate toggle that translates each run as its pipeline completes,
  skipping runs whose detected language (#124) already matches. The run detail
  page gains a Translation card: translate or re-translate into any language
  (defaulting to the preferred one), watch progress, cancel, and see honest
  freshness — a translation is an immutable generation of the corrected
  transcript, line-aligned and hash-stamped, so a later edit or split marks the
  whole rendition "out of date" for one-click re-translation rather than ever
  showing old lines against a changed transcript (speaker renames deliberately
  do not invalidate it). Generation is fail-closed: the model must echo every
  line index exactly (merged, dropped, or reordered lines fail the batch, which
  retries and then bisects down to single lines), runaway output is bounded,
  partial generations never persist, and an edit landing mid-generation fails
  the job while keeping the previous translation current. Console rendering of
  translated lines and translated exports land in slice B. Migration 0038.
- **Transcript translation, slice B: interleaved view, translated exports, and
  the review-stepper Translate action** (#133). The transcript page gains a
  Translation switcher: picking a translated language shows the translated
  line beneath each original (in the hydrated player and the JS-off fallback
  alike), with the model, source language, and generation time stated above.
  Every transcript export format (`txt`/`md`/`srt`/`vtt`/`json`) accepts
  `?lang=<code>` and serves the reviewed transcript with translated text
  substituted, fail closed: 409 when no fresh translation exists (missing,
  still generating, or out of date after an edit), 422 for an unknown code or
  when combined with `text=raw|enhanced`, and never mixed-language output.
  Subtitle cue timing is untouched, so translated captions can read fast (the
  export menu and docs say so). The export menu lists translated downloads for
  fresh generations only, and the review stepper's terminal block offers a
  one-click Translate to the preferred language once every line is checked.
  New how-to: `docs/how-to/translating-transcripts.md`.

### Fixed
- A transcript could be left permanently out of semantic search if its
  embedding job was recorded but never handed to the worker (a crash or broker
  outage in the moment between the two). The stranded job held the run's
  one-active slot, so neither automatic indexing nor `voxint embed backfill`
  would retry it. The background recovery pass now re-dispatches such a job once
  it has sat unclaimed past the staleness grace, and `voxint embed backfill`
  adopts and runs it instead of reporting the run as already active (#130).

## [0.23.1] - 2026-08-23

### Fixed
- The Docker app image baked the transcript-search MiniLM weights (#121) into
  directories the non-root runtime user could not enter, so a container built
  from the 0.23.0 tree failed the moment it first touched the semantic index. The
  weight files were copied read-only, which also stamped that read-only mode onto
  the directories holding them and removed their traversal bit; the image now
  restores directory traversal so the baked weights load. The 0.23.0 release build
  failed on exactly this at the CPU smoke gate, so it never published an app image
  or a PyPI package; 0.23.1 is the first shipped 0.23.x.

## [0.23.0] - 2026-08-23

### Added
- **Voxint now detects each recording's language and records it on the run**
  (#124). The pipeline previously told the transcription service to use English
  on every request, so multilingual recordings were force-decoded as English.
  The ASR client now requests auto-detection (the contract's documented
  `language: null` path; the service's v1 omitted-field default stays `"en"`
  for other callers), and the transcribe stage stamps the detected language and
  whisper's detection score on the run after a successful decode. The runs
  browser gains a "Detected language" filter and column showing names with
  codes ("Spanish (es)"), and the run detail page shows the language with the
  detection score, framed as the model's own score for its guess rather than a
  calibrated confidence. Legacy runs stay unrecorded (no backfill: a
  reconstructed language would fabricate provenance). The transcribe response
  gains an additive nullable `language_probability` field within v1. Detection
  covers the whole recording as one language; a low-confidence fallback
  guardrail is a recorded non-goal for this release.
- **Transcript semantic search (#121).** Voxint now finds a passage by what it
  means, across every transcript, not only by its exact words. A new **Meaning**
  tab beside the runs search box opens a ranked `/search` page: type what you are
  looking for and get the closest passages from the whole corpus, each with the
  run, speaker, and time range, and a link that opens the transcript scrolled to
  that passage. A `"quoted phrase"` is matched exactly and floated to the top. The
  search runs fully on-device with no LLM and no network, combining semantic
  closeness with word-for-word overlap.
- **The semantic index that powers it (#121).** An in-process embedder (the
  multilingual MiniLM model, run as a vendored ONNX graph with no torch and no
  network at runtime) reads each completed run's resolved transcript, splits it
  into passages, and stores their vectors. Indexing runs automatically when a
  run finishes, and `voxint embed backfill` indexes the existing back catalogue
  (stale-only by default; `--force` reindexes everything). Correcting a
  transcript reindexes that run on the next pass, so the index tracks the
  transcript as you fix it. Semantic search and automatic indexing are two
  toggles under **Settings > Semantic search** (and the `SEMANTIC_INDEX_ENABLED`
  and `SEMANTIC_INDEX_AUTOGENERATE` installation defaults), both on by default.
  The Docker images bake the model files; a native install fetches them from the
  pinned `minilm-onnx-v1` release asset, and `voxint-native.sh doctor` reports
  when they are missing.
- **The models panel now also checks the diarizer's clustering configuration,
  not just its weights (#129).** A deployment could report the validated diarization
  name and the validated weights while running a different clustering
  configuration (the tuned threshold, minimum cluster size, segmentation step,
  or merge gap), which changes the result. The diarization service now reports a
  hash of its effective clustering configuration on `/healthz`, and the
  "Pipeline models" panel classifies it as a second identity axis alongside the
  weights: a drifted configuration reads as a mismatch with its own remedy (reset
  the `PYANNOTE_CLUSTERING_*` values), separate from a weights mismatch (re-pull
  or rebuild). The two axes are independent, so each points at the right fix.
  Relatedly, on the built-in (validated) pipeline the service now refuses to
  start if it rejects the tuned clustering settings, where before it logged a
  warning and quietly ran with default values under the validated name; it also
  refuses to start in the subtler case where the pipeline accepts the settings
  but silently keeps a frozen default, by reading the effective clustering
  parameters back and confirming they match what was requested (#131). An
  explicitly overridden pipeline keeps the previous tolerant behavior. The
  default install is unaffected: it ships the validated configuration, starts
  normally, and reads as validated. Batch sizes are excluded from the hash: they
  are throughput-only and expected to vary per GPU. See `docs/gpu-contracts.md`
  for the hash definition.
- **The models panel now checks the actual model files, not just the model
  name (#125).** A deployment could report a validated model name while running
  different weights: the diarization service does not distinguish the built-in
  checkpoint from a re-fetched one by name alone, and the transcription service
  would accept the validated `large-v2` name paired with a different revision.
  The diarization service now reports a fingerprint of its loaded checkpoint
  files on `/healthz`, and the "Pipeline models" panel classifies each service
  by exact identity: the validated name with matching files reads as validated;
  the validated name with different files reads as a weights mismatch and is not
  trusted; the validated name whose files cannot be verified (a model loaded from
  an online source) reads as unverified and is not trusted. The default install
  is unaffected: it ships the validated files and reads as validated. See
  `docs/gpu-contracts.md` for the fingerprint definition.
- **Settings has a Glossary editor for expected proper nouns.** The settings page
  now has a Glossary section that edits the operator's list of expected names,
  places, organizations, and acronyms without re-running the setup wizard. Voxint
  already feeds these terms to transcription as a decoding hint on every run; this
  makes the list manageable after onboarding. The section reuses the wizard's
  validation exactly (one term per line, deduplicated, at most 500 terms of up to
  120 characters), replaces the whole list on save, and applies to runs that start
  after you save, including any already queued. Terms are a hint, not a guarantee,
  and a rejected save keeps your text and explains what to fix. The copy points a
  recurring, identical mistranscription at the Corrections section instead, and
  notes that a name split across a pause is handled here because the glossary
  steers transcription before the words are grouped into lines. Decoding behaviour
  is unchanged; only where the same list is edited moves.
- **Run detail records the glossary a run decoded with.** The run-detail page now
  shows the exact bounded vocabulary hint whisper saw for that run, the operator's
  glossary unioned with the run's domain-pack words. The frozen domain-pack
  snapshot only recorded the pack's words, so the operator's live-applied glossary
  was previously unrecoverable per run; a new nullable `initial_prompt` column on
  `pipeline_runs`, stamped by the transcribe stage once the decode succeeds, closes
  that gap. A run not yet transcribed, one with no vocabulary, or one that finished
  before this existed shows an honest empty state.
- **Settings shows which models are running ("Pipeline models").** The settings
  page now has a read-only panel that reads each model service live as the page
  loads and shows the transcription, diarization, and speaker-embedding model
  actually running, with its engine and revision where the service reports them.
  A service running the validated default is marked as such; a transcription or
  diarization service running anything else is flagged as an unvalidated override
  with an honest accuracy caveat, distinct from a service that is simply
  unavailable. The speaker-embedding model is fixed and shown without a warning.
  The panel changes nothing itself: model selection is deployment-owned, so it
  lists the `.env` keys to change and notes that only the affected service
  restarts. It never displays `HF_TOKEN`.
- **Run detail shows which models ran ("Pipeline models").** The run-detail page
  now renders the per-attempt model identity recorded for the transcription and
  diarization stages, taken from each stage's latest completed attempt, so a
  retried stage shows the model that produced the result rather than a failed
  attempt's stamp. Each service shows its model, engine, revision, and decode
  configuration where the service reports them. A run that finished before this
  provenance existed, or a stage whose service could not be reached, reads "Not
  recorded" or "Not observed" rather than implying the model is unknown for a
  subtle reason. The copy states plainly that this is observed immediately before
  the attempt, not read back from the output.
- **Per-attempt model-identity provenance on each stage run.** Every stage that
  calls a model service now records, on its own `StageRun`, which model answered
  it: the identity (`model`, `revision`, `engine`, `decode_config_hash` where the
  service reports them) is read from the service `/healthz` immediately before the
  attempt runs and stored under `StageRun.metrics.model_identity`. The read is
  best-effort and never blocks a run: an unreachable or not-yet-loaded service
  records a `reachable: false` marker and the stage proceeds. Because it lives on
  the per-attempt claim row and is written in the same transaction that completes
  the claim, a failed or lease-expired attempt can never overwrite a later
  successful attempt's recorded identity. This is provenance observed just before
  the call, not response-carried proof of the exact build; the groundwork for the
  operator-facing pipeline-models surfaces that follow.
- **Opt-in alternate transcription model (whisper), fully gated.** The shipped
  `large-v2` stays the only validated model and its baked, offline path is
  unchanged. An operator can now select an alternate model, but only by opting in
  explicitly: a non-default `WHISPER_MODEL` requires both `WHISPER_ALLOW_DOWNLOAD=1`
  and `WHISPER_REVISION` set to that model's full 40-character commit SHA, or the
  whisper service refuses to start with a message naming exactly what to set.
  Alternate weights download into a separate cache volume that never shadows the
  baked default. Alternate models are an unvalidated mechanism (v3 and turbo
  hallucinate); the startup warns, and the numerics guarantees cover `large-v2`
  only.
- **Reproducible pin for an overridden diarization model (`DIARIZER_REVISION`).**
  Setting `DIARIZER_MODEL_NAME` to an alternate Hugging Face pipeline can now be
  paired with `DIARIZER_REVISION` to pin that pipeline to an exact commit, so the
  recorded provenance is reproducible instead of floating with the repo's default
  branch. The pin is applied across pyannote's incompatible loader forms (3.1.x
  pins via the `repo@revision` form with `use_auth_token`; 4.x via `revision=`
  with `token=`) and surfaces in the service `/healthz` as `model_revision`. It
  is not applicable to the vendored default (whose config is itself the pin) and
  is ignored there with a warning. The validated default load path is unchanged.
- **Installer can record an alternate pipeline model.** The guided installer
  (`scripts/install.sh`) now offers one skippable advanced prompt for pointing
  transcription or diarization at a non-default model. It defaults to skip and
  writes nothing, so a normal install is unchanged. Opting in collects the
  whisper model plus its required 40-character revision (and records
  `WHISPER_ALLOW_DOWNLOAD=1` automatically), the diarizer model plus an optional
  revision pin, and a Hugging Face token for a gated pipeline that is read
  hidden, never echoed, and single-quoted when written. The prompt states
  honestly that only the shipped models are validated. A new how-to,
  [Changing pipeline models](docs/how-to/changing-pipeline-models.md), documents
  the full procedure, the validated-versus-unvalidated tradeoff, and the
  timeout note for slower models; the new keys are documented in `.env.example`.
- **Operator-supplied speaker count for diarization (issue #128).** You can now
  tell the pipeline how many speakers a recording has, so pyannote stops
  over-segmenting one voice into many clusters on hard audio. Supply either a
  bound (the most speakers to expect) or an exact count three ways: the
  `voxint submit --max-speakers N` / `--num-speakers N` flags, the matching
  `max_speakers:` / `num_speakers:` keys in a media file's YAML sidecar, or the
  install-wide `DIARIZATION_MAX_SPEAKERS` ceiling (unset by default, so a stock
  install sends no bound and the service's own default of 10 still applies). A
  per-recording hint is frozen onto the run at submit and reused on requeue. The exact count is sent to the service as equal `min`/`max` bounds, so
  it needs no change to the pyannote request contract. See
  [docs/interpreting-diarization.md](docs/interpreting-diarization.md).

### Fixed
- **The built-in diarizer now confirms its clustering settings actually took
  effect before reporting them as validated.** On the built-in pipeline the
  service applied the tuned clustering settings and, if that call did not raise,
  treated them as applied and folded the requested values into the configuration
  hash. A pinned pyannote build can hold a clustering setting fixed and still
  accept the call, which would leave the service running one configuration while
  reporting the hash of another under the validated identity. The service now
  reads the effective clustering settings back after applying them and refuses to
  start unless the threshold and minimum cluster size match what was requested.
  The stock configuration pins no setting, so the default install applies its
  settings, starts normally, and reads as validated.
- **The run page no longer attributes a result to an older attempt's model
  stamp.** When a stage was retried and the newest completed attempt carried no
  recorded model identity, the "Pipeline models" panel fell back to an older
  attempt's stamp, which could name a model that did not produce the shown
  result. The panel now reads the identity only from the attempt that actually
  completed the stage and says "Not recorded" when that attempt has none.
  Display only; the worker stamps every completed attempt on the normal path,
  so this surfaces only with hand-edited or partially migrated data.
- **The transcription service can no longer silently load a local directory in
  place of its model.** Any accepted `WHISPER_MODEL` value, a Hub id such as
  `org/name` or the validated default's own spellings, can also name a real
  directory next to the service, which faster-whisper loads verbatim, skipping
  the pinned weights. The service now refuses to start when the resolved model
  value names an existing local directory. Requires a colliding directory in
  the service working directory; the stock container deployment was
  unaffected.
- **The transcription service no longer loads different weights under the
  validated model name.** Setting `WHISPER_MODEL=large-v2` (the validated name)
  together with a different valid `WHISPER_REVISION` previously took the default
  path and loaded that other snapshot with no download gate and no warning. The
  startup resolver now refuses to start in that case: the validated name loads
  only the built-in snapshot, and an alternate build must go through the explicit
  `WHISPER_ALLOW_DOWNLOAD=1` path with its own model id and commit SHA. Reachable
  only in non-default setups (a custom image or a cleared offline flag); the
  stock offline deployment was already safe.
- **Whisper model services no longer fail to start under the configurable-model
  compose pass-throughs.** The whisper overlays forward
  `WHISPER_MODEL: ${WHISPER_MODEL:-}` and `WHISPER_REVISION: ${WHISPER_REVISION:-}`,
  which set those container variables to the empty string when an operator does
  not override them, shadowing the image's baked values. An empty model made the
  fail-closed startup resolver refuse to start ("Invalid model size"), and an
  empty revision made the offline load resolve the absent "main" ref and raise
  `LocalEntryNotFoundError`. The model overlay now defaults to `large-v2`, and on
  the default path the resolver restores the baked revision from
  `WHISPER_BAKED_REVISION` (baked from the same ARG, never compose-forwarded), so
  a stock deployment stays byte-identical and offline while an explicit override
  still works. The default `-cpu`/`-rocm` deployments were unaffected only by
  accident of an older compose; this closes the gap for all three overlays.

## [0.22.1] - 2026-08-21

### Fixed
- **CUDA model-service images crash-looped on startup (regression in 0.22.0).**
  The hardware-telemetry code added in 0.22.0 imported `datetime.UTC`, which
  exists only on Python 3.11 and later. The `voxint-whisper`, `voxint-pyannote`,
  and `voxint-titanet` CUDA images build on `nvidia/cuda:*-ubuntu22.04` (system
  Python 3.10), so every one of them failed to import and never served. The
  `-cpu` (Python 3.11) and `-rocm` (Python 3.12) images were unaffected. The
  three vendored `resource_probe.py` files now use `timezone.utc`, ruff no longer
  rewrites them back to `datetime.UTC` under the py311 target (a per-file `UP017`
  ignore for `services/*/app/*.py`), and a contract test rejects 3.11+-only
  constructs in the model-service code so this cannot regress silently.
- **The whisper decode identity now includes the compute device.** The
  `decode_config_hash` reported on the whisper service `/healthz` folds the
  effective decode config into one digest so a numerics-relevant change is
  visible deployment to deployment. It omitted the device, so a CUDA
  deployment and a ROCm deployment produced the same hash even though they run
  different backends. The hash now distinguishes them. This is a provenance
  fix, not a transcription change: model output is untouched. One-time effect:
  after this update every deployment reports a new `decode_config_hash` on the
  next restart, on every tier (CUDA, ROCm, CPU, Metal).

## [0.22.0] - 2026-08-21

### Added
- **A task-first "first 30 minutes" flow for the review console (#117).** The
  console now guides a first-run operator instead of only presenting controls.
  The dashboard leads with three task cards (add audio, continue review with the
  count of recordings still needing a ruling, and the last finished run) and
  tucks the throughput and stage metrics behind a "Show run details" disclosure,
  with the window control and any error kept in view. Review reads as an explicit
  two-step sequence, "who is speaking" then "check the words", with one dominant
  Continue on each step and an honest end state that never claims a completion
  the system does not track. Happy-path wording drops maintainer jargon: the nav
  says "Review", a voice match reads "Strong voice match" or "Possible voice
  match" with the raw similarity tucked into a "Why this match?" reveal, and the
  setup step is "Text clean-up and name hints (optional)". The guided tutorial
  now walks both review steps (five steps: run, review, attribute the voices,
  check the words, export). No numerics change and no new configuration.
- **Optional reasoning-off switch for LLM calls (`LLM_DISABLE_THINKING`).** When
  enabled, every LLM request (BYO and bundled, across enhancement, the name pass,
  run assets, and web research) carries the vLLM chat-template switch
  `chat_template_kwargs.enable_thinking=false`, which turns off a reasoning
  model's chain-of-thought. Off by default so BYO/OpenAI request bodies are
  unchanged. This fixes read-timeouts on the heavy entity-mention and
  research calls when pointing at a thinking model such as Qwen3 on vLLM, where
  the reasoning traces alone could exceed `LLM_TIMEOUT_SECONDS`. See
  `docs/operations.md` and `.env.example`.
- **Model-service hardware telemetry on `/healthz`.** Each GPU model service
  (whisper, pyannote, titanet) now samples its hardware in a background thread
  and reports an additive, optional nested `resources` block on `/healthz`:
  GPU utilization, VRAM, temperature, throttle state and decoded reasons,
  cumulative peak-temperature and throttle-event counters, plus an `admission`
  block (pending, max-pending, rejects-since-start) and a host-visible CPU
  advisory. The GPU is resolved by UUID rather than device index, so a shared
  card is reported honestly and three services aggregate into one device. It is
  fail-soft by construction: telemetry is served from the cache, never probed on
  the request path, and any NVML or driver failure degrades the affected fields
  to null with a tri-state `availability`, never changing a service's readiness.
  Off or a bad interval falls back safely (`VOXINT_TELEMETRY_ENABLED`,
  `VOXINT_TELEMETRY_INTERVAL_SECONDS`; see `docs/gpu-contracts.md`). This is the
  telemetry foundation for the operator resource view and safe hardware-aware
  defaults.
- **App-side resource-telemetry aggregation.** The app now collects the three
  services' `/healthz` telemetry into one operator view: it parses each block
  defensively (an older service or a malformed value degrades to "unavailable",
  never an error), deduplicates a shared GPU by UUID into a single device
  (freshest reading wins; cumulative counters take the max), and reads the
  degraded 503 body too so a struggling service still reports its numbers.
  Probes run concurrently behind a short single-flight cache
  (`RESOURCE_STATUS_TTL_SECONDS`, default 10) so a browser poll across tabs
  never fans out live probes. `voxint doctor` now prints the aggregated GPU and
  per-service admission state.
- **Operator resource visibility.** The aggregated hardware snapshot now renders
  everywhere from one cached source: a curated **Dashboard** strip, a dedicated
  **Resources** page (`GET /resources`), `voxint stats` (text and, under a
  `resources` key, `--json`), and `voxint_gpu_*` / `voxint_service_admission_*`
  gauges on `GET /metrics`. The strip is deliberately quiet to avoid alarm
  fatigue: it shows each GPU's activity (idle / working / busy, never an alarm,
  since full utilization during a transcription is healthy) and warns, with one
  plain remedy each, only when the driver reports thermal throttling or a service
  queue is currently full. High VRAM and cumulative-since-restart counts are not
  warnings; the resource page shows them as instantaneous or cumulative context.
  When no service reports telemetry the surfaces say so rather than claiming
  all-clear. Warnings are advisory in v1: the driver already protects the
  hardware, so Voxint never pauses work on its own.
- **Speaker-matching decision evidence (#113).** Every pipeline run now records,
  for each diarized voice, what the matcher decided and the numbers behind it:
  the top roster candidate, cosine similarity, top-1 vs top-2 margin,
  vote-agreement, and why a voice was accepted, rejected, or set aside (too few
  turns, too little clear speech, no enrolled speaker to compare against). Until
  now only accepted matches were kept, so a voice the matcher narrowly declined
  left no trace. This is observational only: it changes nothing an operator sees
  and does not alter a single match. It is the measurement groundwork for making
  speaker attribution trustworthy enough to widen (epic #112).
- **Scoring-harness exporter (#113).** A maintainer-facing library
  (`voxint.harness_export`) renders live pipeline runs into the file shapes the
  offline `voxint score` harness consumes: name-accuracy items (the matcher's
  automatic attribution scored against the human ruling) and acoustic-agreement
  enrollment plus per-voice slots (the exact voiceprints and centroids
  production compares), alongside an evidence snapshot recording the code
  version, matching gates, and roster identity at export time. This closes the
  measurement loop so a first speaker-attribution baseline can be produced; it
  reads the database and never re-runs the models or the matcher.
- **Match-evidence export driver (#113).** A maintainer tool
  (`tools/export_match_evidence.py`) reads a small run-selection manifest, calls
  the exporter, and writes the score-harness input files plus the evidence
  snapshot into a directory. Writes are atomic and deterministic, so a repeated
  export from an unchanged database is byte-for-byte identical, and it refuses to
  run on a working tree with uncommitted tracked changes so the recorded code
  version means what it says.
- **Hardware-aware conservative install defaults (#96).** On the GPU tier,
  `scripts/install.sh` now reads the host GPU with `nvidia-smi`, matches it
  against a tested-profile table, and writes a generated, marker-owned
  `compose.hardware.yaml` that it merges into the stack it launches. For a GPU
  with no measured profile yet, it applies a safe fallback that only serializes
  scheduling: worker `--concurrency=1` and whisper `MAX_PENDING_REQUESTS=1`. It
  deliberately leaves `BATCH_SIZE` untouched, since that value moves whisper's
  output and so must come from a parity-gated profile plus a real-GPU
  out-of-memory soak, not the installer. The generated file is regenerated each
  run and refreshed on a GPU swap; a hand-written one (no marker) is left alone.
  The installer also now merges an operator's own `compose.override.yaml` last of
  all, so it wins over the base stack, the tier overlay, and the hardware
  baseline. `./scripts/install.sh --hardware-dry-run` previews the detection and
  the file without writing or starting anything. See `docs/operations.md` (#96).
- **Offline eval-quality harness (#97).** A maintainer-facing scorer
  (`tools/eval_quality.py`, `tools/eval_run.py`) measures transcription and
  diarization quality against a reference: word error rate, diarization and
  Jaccard error rates, and concatenated-minimum-permutation WER for
  speaker-attributed text. A companion tool
  (`tools/build_ami_wer_reference.py`) freezes an AMI WER reference so scores are
  reproducible across runs. Its dependencies live in an isolated `eval-quality`
  extra so a normal install does not carry them, and its parity and contract
  tests keep the metrics pinned.

### Changed
- **A visual-polish pass over the review console: consistent spacing, clearer
  button hierarchy, and completed interactive states.** Card padding, section
  margins, and gaps now come from one spacing scale instead of ad-hoc values, so
  surfaces share a rhythm, and the eight hand-written card recipes now share a
  single `.card` primitive recipe. Each screen now has one
  dominant primary action: Assign on the speaker workbench, Submit on the runs
  page, Review on a finished run, Confirm on the merge prompt, with the quieter
  alternatives de-emphasized. Genuinely destructive actions (delete derived
  audio, merge a speaker, remove an enrollment) get a distinct danger style,
  measured to pass WCAG AA contrast in both themes. The run detail page, the only
  screen with no card grouping, now matches the rest: its sections sit in cards
  and its metadata reads as an aligned two-column list. Buttons gained a pressed
  state, several pointer elements gained a hover cue, and the input focus ring is
  wired to the same token as everything else. No numerics change and no new
  configuration.

### Fixed
- **The dashboard's Review backlog now counts the recordings actually waiting
  for review (#117).** It previously counted runs in a transient
  `awaiting_adjudication` status that a finished pipeline never ends in, so the
  figure sat at zero even while recordings were waiting. It now shares the
  review queue's own definition (a finished recording with at least one speaker
  still unconfirmed), so the headline can no longer disagree with the queue it
  links to.
- **The titanet and pyannote CUDA images no longer enable the PyTorch
  `expandable_segments` allocator mode (#111).** Under heavy VRAM pressure
  (observed with titanet on a GPU shared with another CUDA workload), that mode
  can trip an upstream PyTorch allocator bug
  (`!block->expandable_segment_ INTERNAL ASSERT FAILED`), hard-failing the
  request and the run's current stage; retries and service restarts do not
  clear it while the mode is enabled. Each image keeps the rest of its allocator tuning.
  For older already-pulled images, the same fix works as a compose environment
  override; see the GPU memory section of
  [docs/operations.md](docs/operations.md).

## [0.21.0] - 2026-08-20

### Added
- **Operator annotation layer for the review console (#86).** An operator can now
  select a span of transcript text and save it as a highlight in one of six
  colors, with optional flat tags and a margin note. A Highlights panel lists
  every annotation in transcript order with a tag filter (any selected tag
  matches), and each row jumps to and plays its place in the transcript. The `h`
  shortcut highlights the current selection, and says so honestly when nothing is
  selected. Speaker and timing on each highlight are read live from the current
  transcript so they stay correct as the review changes; timing is labeled
  approximate when a span cannot be tied to exact word times. When an edit moves
  the text a highlight covered, the highlight is marked stale (no colored mark,
  just an approximate locator where it was) and can be refreshed if the wording
  returns or re-anchored to a fresh selection. Creating and editing highlights
  needs the console; with JavaScript off the review page shows a read-only list of
  the existing highlights.
- **Copy highlights as pull-quotes (#86).** Each Highlights row has a Copy button,
  and a Copy-all action lifts every highlight that matches the current tag filter,
  in transcript order. A copied highlight is Markdown: the quoted text (byte-for-byte
  what a transcript export would produce), then a source line with the recording name
  and the timestamp range, and the highlight's tags and note. Copy works while just
  viewing a run, without holding the review slot. When the browser blocks automatic
  copying (which happens on a plain-http local network), the text appears in a box to
  select and copy by hand, so a copy never silently fails. A stale highlight cannot be
  copied until it is refreshed or re-anchored, so a quote is never built from text
  that has since changed.
- **YAML sidecar metadata for watch-folder media (#104).** A recording dropped
  into a watched folder can arrive with a companion sidecar file
  (`interview.wav.yaml`, or `interview.yaml` when the stem is unambiguous; the
  full-name form wins) whose fields feed the run at submit time: `title`
  (display name in the queue and run detail), `speakers` (trusted name hints,
  unioned into the run's frozen domain-pack `name_seeds`), `domain_pack` (wins
  over the folder mapping), and `notes` (saved as the run's operator notes).
  Keys Voxint does not recognize are kept with the run for reference and
  otherwise ignored, so sidecars written by other tooling ingest as-is. A
  sidecar with a problem (bad YAML, a bad value on a known key, an ambiguous
  stem name, an unknown pack) holds its recording un-submitted with a
  plain-language reason on the Settings status line, retried every check until
  fixed; the pair also settles together, so a half-written sidecar is never
  applied. Sidecar content is frozen at submit and stored write-once on the
  run (`pipeline_runs.sidecar`, migration 0030); a sidecar arriving after its
  recording was picked up deliberately does nothing. The sweep's net-new file
  cap now applies to actual submissions rather than scan candidates, so held
  recordings can never crowd new ones out of a check.
- **Operator guidance for GPU memory on a single, modest GPU (#96).**
  `docs/operations.md` now documents why the stock GPU overlay can exhaust VRAM
  on one small card and cascade into an `invalid device ordinal` from a poisoned
  CUDA context, and how to tune it down: worker `--concurrency`, whisper
  `BATCH_SIZE`, and `MAX_PENDING_REQUESTS`, with a conservative
  `compose.override.yaml` profile. Documents existing knobs; the safe-by-default
  sizing itself is still tracked in #96.
- **Operator guidance for restricted URL-ingestion egress (#16).**
  `docs/operations.md` now lays out the four egress controls in order (submit gate,
  resolved-host gate, proxy overlay, network policy) and states plainly that the
  two userland gates do not close the SSRF residual on their own. It adds a
  ready-to-adapt Kubernetes `NetworkPolicy` example for the egress layer, bound to
  the worker pod (where yt-dlp and any helper run) with explicit allows for the
  worker's private-address dependencies, plus honest caveats about CNI enforcement,
  DNS and Service scoping, and the coarse `ipBlock` list versus the precise
  `ip_is_public` policy. Documentation only; no behavior change.

### Changed
- **The pipeline now runs as two execution lanes, so a GPU-serialized
  deployment no longer idles the card during LLM enhancement.** The six stages
  are partitioned into a GPU lane (`acquire` through `diarize_embed`, the
  default `celery` queue, task `voxint.run_pipeline`) and a post lane
  (`enhance_match` and `finalize`, new `post` queue, new task
  `voxint.finish_pipeline`). The engine hands a run between lanes with a
  validated `running -> queued(next stage)` transition committed atomically
  with the finished stage, so the durable queued row survives any crash or
  broker outage between the handoff and its publication and the recovery
  sweep re-publishes it to the correct queue by `current_stage`. The LLM-bound
  post-run jobs (`voxint.generate_run_asset`, `voxint.research_speaker`) and
  the beat sweeps (recovery, GC, notify, watch) are routed to the `post` queue
  as well, so crash recovery and housekeeping never wait behind GPU work. Default deployments are unchanged: both
  queues are declared on the Celery app and a worker started without `-Q`
  consumes both. Deployments that pin the worker to `--concurrency=1` for a
  shared GPU can now run a second worker on the `post` queue so one run's LLM
  enhancement overlaps the next run's transcription; see the override recipe
  in `docs/operations.md`. Existing worker commands with an explicit `-Q` must
  drop that flag or use `-Q celery,post`, or handed-off runs remain queued at
  `enhance_match` while recovery republishes them to an unconsumed `post` queue.

### Fixed
- **The Markdown transcript export can no longer let transcript text forge
  document structure (#65 follow-up).** The `.md` escaper now defuses a `=`
  setext underline and a bare carriage return (each could forge a heading, and
  the carriage return could break out of the quote to a top-level heading),
  escapes `~` and `|` (tilde code fences and GFM table pipes), measures line-
  leading markers over tabs as well as spaces (a leading tab previously slipped
  a marker past the defuse), strips per-line indentation (four leading spaces or
  a tab would otherwise open an indented code block inside the quote), and folds
  Unicode line separators (U+2028/U+2029/NEL) to real physical lines for
  non-CommonMark preview tools. On-screen read mode was never affected, since it
  renders through Jinja autoescape; the plain `.txt` and `.json` exports are
  unchanged.
- **The review console's Save-edit shortcut (Ctrl/⌘+Enter) is now part of the
  keyboard-shortcut source of truth (#51 follow-up).** Its key chord, its two
  on-screen hints, and the `?` cheat-sheet now read one shared definition, so
  they can no longer drift apart; the save chord is also listed in the
  cheat-sheet for the first time.
- **Topics enrichment now runs on a distinct BYO endpoint even when a scoped
  bundle is active (#106).** Previously any active bundle dropped the `topics`
  run-asset kind at enqueue, on the assumption that the bundle was the only (and
  too weak) LLM. When the bundle serves enhancement AND a separate BYO endpoint
  is configured (for example a capable LAN model), topics now flow to that BYO
  endpoint instead of being refused. The default bundled install (no BYO
  endpoint configured) still shows the same guidance, unchanged.

## [0.20.0] - 2026-08-19

### Added
- **A System / Light / Dark theme control for the review console (#94).** A new
  **Appearance** section on the Settings page lets the operator follow the
  device's light/dark preference (System, the default) or force either look.
  The choice is stored per device in the browser (localStorage), applied by an
  inline script before first paint so no page ever flashes the wrong theme, and
  propagated to every open console tab. Native controls (form fields,
  scrollbars, the audio player) follow the chosen theme via `color-scheme`, and
  the waveform strip repaints in the new palette the moment the theme changes.
  Without JavaScript the console keeps following the device setting.
- **On-screen read mode and a Markdown export for finished transcripts (#65).** A
  new reading view (`/runs/{id}/transcript?read=1`, reachable from the Download
  transcript menu) renders the transcript as prose: one heading per speaker over a
  merged paragraph of their words, with a plain-anchor toggle for timestamps and
  no JavaScript required. A new `.md` export writes the same layout to a file
  (`## Speaker` headings, one `>` blockquote per contiguous run, per-paragraph
  time ranges gated by `?timestamps=false`), on both the HTTP route
  (`/review/{id}/export.md`) and the CLI (`voxint export --format md`). Read mode
  and the Markdown export share one grouping helper (`paragraphize_transcript`)
  with the existing presentation seam, so they can never drift from what the other
  exports show. Markdown output escapes inline specials, raw HTML, and
  line-leading block markers (headings, lists, thematic breaks), and folds a
  speaker name to one line, so transcript content cannot forge document structure;
  read mode renders through Jinja autoescape for the same guarantee in HTML. The
  plain `.txt` default is unchanged
  (timestamps stay on); the timestamp-free reading copy is now the prominent first
  choice in the picker, which also surfaces the reviewed (operator-effective) text
  variant it previously hid.

### Changed
- **The `export.json` envelope `schema_version` is now 2.** If you feed
  `export.json` into your own scripts, check that field before parsing: version 2
  marks the reduction of the uploader, channel, canonical, and `raw.webpage_url`
  fields to host-only values (the D4 item under Security below has the full
  policy). The envelope shape and every other field are unchanged.
- **Review-console keyboard shortcuts now have one source of truth, and the
  cheat-sheet is easier to find (#51).** The eight review shortcuts used to be
  written out in three places — the key handler, the `?` cheat-sheet, and the
  inline hints on the buttons — and kept in sync by hand. They now share a single
  `keymap` definition, so the cheat-sheet and the on-screen hints can never
  disagree with the keys that actually fire. Discoverability nudges, no new
  behaviour: the **⌨ Shortcuts** button now shows its `?` accelerator, and the
  **Assign speaker** control shows a `1–9` cue when the run has a speaker roster
  (and stays quiet when it doesn't, since those keys have nothing to assign). The
  cheat-sheet's copy is corrected to say every action has a *clickable equivalent*
  (not a button — walking with `j`/`k` is done by clicking a line, which is why
  that pair stays cheat-sheet-only rather than gaining page clutter). Visual
  restyling of the dialog is intentionally left to the console-modernization epic
  (#89). Browser behaviour is covered by the maintainer E2E lane, extended to open
  and dismiss the cheat-sheet.
- **Docs: aligned the documentation set with the `voxint-docs` house style.**
  Retrofitted the README, contributor docs, and every reference and how-to guide
  under `docs/` to the two audience lanes: removed emdashes, cut LLM-isms,
  tightened cross-links, and added lay-reader subtitles and technical "See also"
  footers. Dated reports, plans, and the security audit were left as historical
  artifacts. Prose only, no behaviour change.
- **Waveform strip: a click with no transcript there now says so (#57).** Clicking
  a silent gap or a stretch of speech that was never transcribed used to do
  nothing, which reads as a broken control. The strip now shows a brief, local
  note ("No transcript text at this point") and marks where you clicked. It is
  presentational only: no seeking, no playback, no change to the fail-closed seek
  gate. A new contract test pins the speaker-palette size across all four places
  it lives (the backend index map, the waveform probe count, and the light and
  dark `--spk-*` CSS tokens plus their `.spk-*` class mappings) so a partial edit
  can no longer silently drop a speaker's colour to the neutral bar.
- **Review-journey restyle (#92, epic #89)**: the workbench and review-transcript
  screens are styled as one continuous "Reading Room" flow. The workbench gains a
  **review header card** (run identity, a claim-state indicator, and the journey's
  actions — with **Review transcript →** / **Claim for review** as the single
  primary accent action per surface); the review transcript gains a **verified
  progress track** (the N-of-M count stays the visible signal; the bar is
  decoration) in both the island and the JS-off fallback. The native `<audio>`
  control, speed selector, speaker-colored waveform and capability banner are now
  framed together in one **player surface** panel — the native control is wrapped,
  never replaced, so playback, media keys, capability gating and the JS-off
  fallback behave exactly as before (a custom transport remains a separately-gated
  follow-up). Transcript lines pick up a quiet hover surface and aligned padding;
  keyboard-shortcut hints render as `kbd` chips; adjudication card actions are
  grouped into one row. All colors come from the existing token layer and meet
  WCAG AA in both themes; keyboard review, htmx swaps, island-failure and JS-off
  fallbacks verified in a real browser.
- **Per-screen visual rollout (#93, epic #89)**: the "Reading Room" treatment now
  reaches the remaining console screens. The adjudication queue keeps its scannable
  table but promotes each row's **Review** action to the primary accent and slims
  its progress cell to the shared review-journey track, with the resolved-of-total
  count sitting beside the bar as the primary signal (an overlaid label measured
  below WCAG AA across the filled and unfilled halves). The Speakers roster renders
  as proper cards with a grouped action row. Settings frames each section as a card
  while leaving the tutorial banner and every **Save** button neutral. The setup
  wizard frames each step as one panel and reserves the teal accent for its forward
  navigation (Get started, Continue, Finish and start tutorial), styles the
  readiness-check rows and step markers, and gives the shared watch-folder panel its
  first real layout. All color comes from the existing token layer and meets WCAG AA
  in both themes. The obsolete-literal inventory is clean: no screen references a
  pre-token color.

### Security

- **`X-Content-Type-Options: nosniff` on every console response (#103).** The
  console serves user-controlled transcript text as downloadable exports (`.txt`,
  `.md`, `.srt`, `.vtt`, `.json`, `.rttm`) and the built frontend as first-party
  assets. The header, stamped in the same shared seam as the D1 headers (so a new
  route cannot miss it and it survives an unhandled 500), forces the browser to
  honor the server-declared `Content-Type` instead of sniffing the bytes, so a
  transcript carrying crafted markup cannot be reinterpreted as HTML and executed.
  Every asset the frontend build emits already resolves to a correct type, so
  nothing legitimate is blocked. Scope is this one header: `X-Frame-Options` and a
  content-security policy stay out, keeping the minimal posture calibrated for a
  single-operator, loopback console.
- **Web-console hardening (audit findings D1–D4).** A calibrated pass over the
  review console for the single-operator, loopback threat model — no database
  migration, no change to how the islands talk to the server:
  - **D1 — contain the claim token in the URL.** The per-claim review token rides
    in `?token=`; a new response-header layer stamps `Referrer-Policy: no-referrer`
    on every response (so a followed link or subresource never leaks it in a
    `Referer` header) and `Cache-Control: no-store` on every `/review` response (so
    a token-bearing page or redirect is never cached). This is a *mitigation* with
    a consciously accepted residual (browser history, screenshots, access logs),
    not token removal — proportionate for a loopback, Basic-authed console.
  - **D2 — CSRF tokens now expire.** The stateless, action-bound CSRF token gains a
    signed mint timestamp and a fixed 24 h TTL (plus a small clock-skew allowance),
    so a captured token is no longer valid indefinitely. A form left open past the
    TTL simply re-mints on refresh.
  - **D3 — authoritative request-body cap.** The size middleware now counts body
    bytes as they stream, so a chunked request with no (or an understated)
    `Content-Length` is bounded at the cap and refused with 413 instead of being
    fully spooled first — closing the residual the `Content-Length`-only check left.
  - **D4 — run export URL fields reduced to host-only.** `export.json` now runs the
    uploader / channel / canonical / `raw.webpage_url` fields through the same
    host-only provenance policy the console UI uses; descriptive metadata (title,
    names, tags, description) is retained. The export envelope `schema_version` is
    bumped **1 → 2**. (`raw` was already an allowlisted subset with no signed URLs,
    so this aligns URL/identity surface, not a secret leak.)
- **CI supply-chain hardening (audit findings F1, F2, F4).** Hardens the GitHub
  Actions release pipeline that builds and publishes the public images:
  - **F1, pin every action to a commit SHA.** All third-party actions across
    `ci.yml`, `release.yml`, and `metal-lane.yml` now reference a full 40-character
    commit SHA (with a `# vX.Y.Z` comment) instead of a mutable major tag, so a
    moved tag can no longer swap an action's code into a `packages: write` job. A
    new `.github/dependabot.yml` proposes grouped action updates once a month to
    keep the pins current.
  - **F2, least-privilege token.** `ci.yml` and `metal-lane.yml` declare a
    top-level `permissions: contents: read`, dropping the over-broad default
    token; `release.yml` already scoped its jobs per job.
  - **F4, verify the gitleaks download.** The secrets-scan job downloads the
    gitleaks tarball to a file and checks its sha256 against a repo-held digest
    before extracting, replacing the previous unverified `curl | tar`.
  - A contract test (`tests/contracts/test_workflow_supply_chain.py`) pins all
    three invariants so a later edit cannot quietly undo them. This scopes the
    GitHub lane only: the Forgejo Actions mirror ignores `permissions`, and other
    mutable inputs (base images, the `pgvector` service image, the CUDA weight
    bake) are out of scope for this change.
- **CUDA titanet weight integrity (audit finding F3).** The CUDA titanet image
  baked the TitaNet-Large `.nemo` at build time via `from_pretrained` with no
  revision pin and no checksum, the one weights path in the repo that floated
  while the ONNX, pyannote, and LLM weights were all sha-pinned. A build-time
  `sha256sum -c` gate now pins the downloaded checkpoint to
  `nemo_checkpoint_sha256` in `tests/parity/fixtures/onnx/provenance.json` (via a
  `TITANET_NEMO_SHA256` Dockerfile ARG), so a re-published upstream checkpoint
  fails the build instead of shipping weights the parity references were never
  measured against. The image is also now offline-bound at runtime
  (`HF_HUB_OFFLINE=1`, matching the whisper service), because the titanet service
  calls `from_pretrained` again at startup; without it an online host could
  re-resolve a re-published checkpoint that never passed the build gate. Contract
  tests pin the ARG to provenance, keep the checksum gate wired, and assert the
  offline bind. The build sha was confirmed against a real NeMo download, so the
  gate passes on the correct weights and fails on drift. This closes the
  drift-detection hole; the
  behavioral CUDA parity gate stays a maintainer-run precondition to tagging
  (Gate A, no GPU runner in CI), now recorded as such in the release process.
- **Research-agent hardening (audit findings E1, E2).** Two calibrated fixes to
  the web-research tool loop for the single-operator threat model:
  - **E2, charset-decode denial of service.** A hostile page could set the
    `Content-Type` charset to one of several registered codecs that pass
    `codecs.lookup` yet still abort `bytes.decode`: the non-text codecs (`rot13`,
    `base64`, `hex`) raise `LookupError`, and text codecs such as `undefined`,
    `idna`, and malformed `punycode` raise `UnicodeError` even under
    `errors="replace"`. Any of them crashed the fetch worker. `decode_bytes` now
    falls back to UTF-8 (a codec that with `errors="replace"` cannot re-raise for
    any byte string) whenever the declared decode raises, honouring the never-raise
    guarantee the docstring already promised.
  - **E1, `web_search` as a residual disclosure channel (documented).** A hostile
    page cannot steer `read_url`, but it can still influence a free-form
    `web_search` query, which discloses private context to the configured search
    provider. This is accepted for the single-operator deployment (search volume
    is budget-bounded) and is now recorded in the agent's injection-posture
    docstring rather than filtered, since a query-content filter would refuse
    legitimate searches.

### Fixed
- **Neutral typed buttons now get their intended surface background (#100).** The
  Tailwind Preflight reset for `[type='submit'|'button'|'reset']` outranks the
  console's bare-button rule, so a neutral `<button type="submit">` rendered
  transparent-backed. Invisible while every such button sat on a same-coloured
  card, it would have shown through on the page canvas or a tinted notice. A
  same-specificity tie-break in the console stylesheet re-declares the surface
  background for typed buttons (browser-verified in both themes).

## [0.19.0] - 2026-08-19

### Added
- **Author domain-pack corrections from the console (#84, epic #78 — final child)**:
  **Settings → Corrections** is a list editor for the operator's own deterministic
  correction rules — add / edit / remove / reorder, toggle *match case* and *whole
  word only*, leave the id blank to auto-generate one from the `match`. No more
  hand-editing a pack's `manifest.yaml` to fix a recurring mis-hear. Every rule is
  validated through the **same #80 gate** a pack gets (length bounds, invisible-
  character rejection, unique ids, the boundary-aware idempotence check) with
  **plain-language errors pinned to the offending row**, so a bad rule is refused
  when you save it, not when a run fails. Rules live **per deployment** (new
  `app_settings.corrections` JSONB column, migration **0029**), so they **survive
  pack upgrades** and apply on top of whichever pack a run resolves to: at submit
  time they are **unioned onto the selected pack's own corrections and frozen** into
  that run's snapshot (pack rules first, then the operator's), so the corrector
  (#82) applies them and the review console shows their provenance (#83) with no
  code changes downstream. Editing them affects the **next** run, never one already
  submitted; a rule that would collide with the selected pack's own rules (a
  duplicate id, or a replacement that re-fires another rule) is refused with the
  reason — at author time for the default pack, and visibly at submit-freeze for a
  differently-scoped pack (never a silent drop). That submit-freeze refusal is a
  plain-language message on **every** ingest path — the upload, URL-fetch and
  folder-scan submit routes and the CLI all surface it instead of an opaque 500, and
  the background watch-folder sweep logs-and-skips the offending file instead of
  stalling. Saving corrections while the pack registry is unreadable is refused with
  guidance rather than erroring. A React island with a server-rendered read-only
  fallback; the editor itself needs JavaScript, stated honestly.
  *(Not a regex editor; a dry-run preview and seeding rules from operator edit
  signals are deliberate follow-ups.)*
- **Deterministic-correction provenance in the review console (#83, epic #78)**:
  the review console now **shows** deterministic domain-pack corrections. Each
  corrected segment carries a distinct **"corrected by domain pack"** marker (never
  conflated with an operator's own "edited" change) that expands to the exact pack +
  rule (`match → replace`) behind every edit; the **immutable raw text** is one
  action away for compare / copy / **reset-to-raw** (reset populates the edit box
  only — the operator still saves, so the unsaved-edit discard protection stays
  intact); an operator edit **supersedes** the marker (stale spans never show against
  operator-authored text); and a run-level **"declared but never fired"** panel
  reconciles every declared rule as `applied` / `no_raw_match` / `growth_rejected`
  with plain-language remediation. All resolved **read-time** from the persisted
  `#82` `correction_trace` + the run's frozen `domain_pack` snapshot, with **no
  migration**. Honest by construction: a rule recorded by a different corrector
  version reads as "unavailable" (never replayed with mismatched semantics); a
  NULL/corrupt snapshot yields **no** provenance rather than a fabricated default
  pack; an unresolved rule id stays visible; and provenance keys off the canonical
  `trace_has_entries` predicate, never a text diff. *(LLM-enforcement-pass growth
  rejection and cross-segment matching are honest, deferred v1 gaps — steer such
  terms to pack `vocabulary`.)* Authoring these rules in the console is #84.
- **Domain-pack corrections composed into enhancement (#82, epic #78)**: the
  deterministic corrector (#81) is now wired **inside** the `enhance_match` stage
  via a **raw-gated dual pass** — rules run on the raw ASR text first to fix which
  rules matched *in the evidence*, the LLM enhancement runs as usual, then **only
  those raw-matched rules are re-enforced on the LLM output**, so a term the model
  *invents* can never be corrected into a domain phrase and mistaken for
  operator-authored (and a term the model *undoes* is restored). Each segment gains
  a durable, versioned provenance trail (new migration **0028**): `correction_trace`
  (either `[]` or the `{version, input_base, entries}` envelope, with `input_base`
  recording a `raw` vs `llm` base) and `corrector_version` — written **only when the
  final text materially differs from raw** (a no-op reads back byte-identical), reset
  atomically on every re-enhance, and never recomputed at read time (legacy pre-#82
  `enhanced_text` reads as "enhanced (unversioned)"). A materially-corrected segment
  is now **unsplittable** (#59), read from the stored trace so the console renders it
  whole instead of deriving children at stale offsets. The `generic` pack declares no
  corrections, so a **default install's pipeline output is unchanged**; authoring/
  provenance UI follows in #83/#84.

### Changed
- **Console typography + shell restyle (#91, epic #89 — console visual refresh)**:
  the first visible step of the "Reading Room" refresh. A warm paper-and-ink canvas
  replaces the pure black/white, with a calm teal accent kept distinct from the
  success-green, subdued speaker hues, and a real type hierarchy (larger, clearer
  headings; monospace tabular figures for timecodes, IDs and counters). Nav,
  buttons, form fields, cards, status pills, notices and tables are all restyled
  from the #90 design tokens — so the page chrome **and** the React islands move
  together — and the dashboard gains **summary stat cards** (backlog, runs,
  completed, roster) above the detail tables plus an inline relative **mini-bar**
  on stage timing (the exact numbers stay visible). Theming still follows the OS
  light/dark setting; an in-app theme toggle is a later step. Accessibility is
  preserved: the keyboard focus ring (now teal), skip-link, high-contrast
  (forced-colors) support, table scroll-containment, non-color-only status cues,
  and a reduced-motion guard.
- **Console design-token foundation (#90, epic #89 — console visual refresh)**:
  the review console's colors now come from one semantic set of CSS design tokens
  in `base.html` that the React islands alias through Tailwind (never duplicating
  values), replacing four ad-hoc color literals in the transcript player and
  waveform. This is an internal foundation with **no visible change** — every
  token maps to the color that already rendered — that lets the upcoming visual
  refresh restyle the whole console from a single source.
- **Native: launcher now runs under `set -o pipefail` (#11)**: a defensive
  hardening so a mid-pipe failure can no longer be masked by a later command's
  success. The pipeline inventory found no silent-loss path, so three benign
  mid-pipe exits are explicitly guarded to keep `pipefail` from turning them into
  spurious aborts: the log-archive prune (`grep` legitimately finds nothing to
  prune) and the two `launchctl`-state captures (a `head -1` could SIGPIPE `sed`).
  Rotation and `status` behaviour are unchanged.
- **Native: `upgrade-db --rehearse` is now listed in `--help` (#13)**: the accepted
  maintainer self-test flag was parsed but undocumented in the built-in usage
  block. The `doctor` help line also now names what it checks (datastore
  reachability & managed-cluster identity) rather than an over-promising "ports".

### Fixed
- **Review console: audio playback and the waveform strip now work on the native
  macOS install**: the media-serving gate probed the just-opened file descriptor
  through `/proc/<pid>/fd` (a Linux-only interface). On the native, docker-free
  macOS path — which has no `/proc` — ffprobe could not open that path, so **every
  valid audio file was rejected as unservable**: `GET /media/<run>` 404'd and the
  review console showed a false "the processed audio file could not be opened"
  banner with playback and the #57 waveform strip both dead. The gate now names
  the descriptor per platform (Linux `/proc/<pid>/fd`, macOS `fcntl(F_GETPATH)`),
  keeping the probe-the-exact-descriptor anti-TOCTOU property on both. Docker and
  Linux deployments were never affected.
- **Bundled local LLM no longer risks batch-poisoning on hallucinated name hints
  (#85, follows #67)**: the scoped bundled model (Qwen3-4B) powers transcript
  enhancement text only — never speaker attribution — yet its reply was strictly
  parsed for `name_hints` that the pipeline discards anyway. A weak model
  hallucinating an out-of-range hint (unknown speaker label / bad `kind`) raised
  an error that failed the **whole enhancement batch** for output that was thrown
  away. Enhancement now carries a `want_name_hints` seam: on the bundled path the
  reply is **not parsed for hints**, so a hallucinated hint is ignored rather than
  fatal. The enhancement **prompt is unchanged on every path** — a measured A/B
  against the #66 frozen corpus (`tools/qualify_local_llm.py`) showed that removing
  the `name_hints` block from the prompt perturbs the 4B model's greedy output and
  **regresses segment faithfulness**, so per the numerics doctrine the qualified
  prompt is kept byte-for-byte and only the parse differs. BYO (bring-your-own
  capable model) enhancement and the BYO name producer still parse hints exactly
  as before.
- **Native: `status` no longer reports a crash-looping worker/beat as healthy
  (#6)**: `worker`/`beat` have no `/healthz`, so `status` printed a bare
  `[supervised]` for them purely from `launchctl print` exit 0 — a job stuck in a
  KeepAlive restart loop read as fine. `status` now parses launchd's own
  bookkeeping (best-effort; that output is non-API) and appends a liveness word:
  `running`, `restarting (last exit N)`, or `state unknown` (never a misleading
  bare healthy state; a stale non-zero exit on a currently-running job is not
  flagged).
- **Native: the DB connection string now percent-encodes the password (#7)**:
  `native_database_url` (and the acceptance tool's composer) interpolated the DB
  password into the DSN raw, so a password containing RFC-3986 reserved
  characters (`@ : / ? # & % +` …) — which the launcher otherwise allows — would
  corrupt the URL psycopg/SQLAlchemy parses. Both composers now RFC-3986
  percent-encode the userinfo (Bash side under `LC_ALL=C`, matching Python's
  `urllib.parse.quote(safe="")` byte-for-byte).
- **Native: `doctor` now detects a foreign Postgres squatting the port (#10)**:
  when a managed cluster exists and Postgres is reachable, `doctor` asserts the
  reachable postmaster is ours (`SHOW data_directory == $NATIVE_PGDATA`) instead
  of trusting bare reachability, reported through the aggregating `doctor_report`
  so later checks still run.
- **Native: the `upgrade-db` old-cluster bindir override is now validated (#12)**:
  `VOXINT_NATIVE_OLD_PG_BINDIR` bypassed `validate_native_inputs`; it now passes
  the same control-character gate as the other operator-settable path knobs.
- **Native: plain `restore` now takes a pre-restore safety backup too**: the
  docker-free launcher's `restore <file>` (in-place replace) previously took **no**
  safety backup, while the scarier `restore --fresh` did — an inversion. The
  single-transaction guarantee only rolls back a *failed* restore; a *successful*
  restore of a valid-but-wrong or older dump silently overwrites live data with no
  fallback. Both restore paths now share one collision-safe
  `pre_restore_safety_backup` helper that dumps the current database to a `0600`
  `pre-restore-<stamp>.dump` / `pre-fresh-restore-<stamp>.dump` under `backups/`
  **before any mutation** and **aborts before touching anything** if that dump
  fails. The shared helper also fixes a latent same-second filename collision in
  `--fresh` (two restores in the same wall-clock second could let `mv` clobber the
  earlier backup). Recover with `restore --fresh` on the printed `SAFETY_BACKUP`
  path.

## [0.18.0] - 2026-08-18

### Added
- **Optional bundled local LLM — no API key required (#67)**: a new opt-in
  overlay [`compose.llm.yaml`](compose.llm.yaml) ships a vendored, Apache-2.0
  **Qwen3-4B-Instruct-2507** (Q5_K_M, ~2.9 GB) served by llama.cpp, so a
  single-operator install gets working enrichment with **no external API key**.
  Turn it on in **Settings → Features → "Use the bundled local model"** (or
  `LLM_BUNDLED_ENABLED=true`). Deliberately **scoped**: the bundle powers **only
  transcript enhancement and run-asset summaries + entity mentions**. Web
  research, LLM speaker-name suggestions, and run-asset *topics* are **not** run
  on it — #66 measured that a small local model isn't reliable at those — and
  they never silently fall back to it; they still need a bring-your-own endpoint
  and key. The bundled model's own name suggestions are dropped so enhancement
  can't attribute speakers through the back door, and its run-asset input is
  clamped to 16k chars (`LLM_BUNDLED_RUN_ASSETS_MAX_INPUT_CHARS`) since a dense
  4B model is a slow backstop on CPU — a GPU is recommended for anything but
  short clips (see [`compose.llm.yaml`](compose.llm.yaml) and
  [`docs/gpu-contracts.md`](docs/gpu-contracts.md) for the pinned serving
  profile). The weight is sha-pinned + provenance-tracked
  (`services/llama-cpp/provenance.json`); because the ~2.9 GB GGUF exceeds
  GitHub's 2 GiB release-asset limit, the image build fetches it from Hugging
  Face at the sha-pinned upstream revision and verifies its sha256 before baking
  it in — the whisper large-v2 pattern — so end users pull
  `ghcr.io/bengizmo/voxint-llm` with the weight baked in and need no Hugging Face
  account, token, or network access.
- **Deterministic corrector engine + faithfulness gate (#81, epic #78)**: the pure,
  `stdlib`-only, versioned engine (`CORRECTOR_VERSION`) that **applies** a pack's
  `corrections:` rules to a segment — a single left-to-right, non-cascading pass with
  **leftmost-longest** overlap resolution (manifest order as the final tie-break),
  exact-literal replacement, and a `{id, from, to, span}` trace whose spans address the
  final corrected string. Reuses #80's matcher unchanged, so boundary/case semantics
  can't fork. A would-be segment growth past a caller-supplied limit is rejected whole
  and atomically (never truncated). Proven by a two-part gate: the six frozen
  enhancement fixtures replay byte-identically under an empty rule set (with an explicit
  NFC assertion), and a new stricter-than-LLM corpus (`tests/fixtures/rules_correct/`)
  pins positive substitutions, substring/collision safety, possessive/hyphen/NFD
  boundary edges, regex-metachar literals, leftmost-longest determinism, idempotence
  over a validated set, atomic growth rejection, and full trace faithfulness — **zero
  unauthorized edits, one failure blocks release**. **No pipeline wiring or persistence
  yet** (that is #82); the `generic` pack still declares no rules, so the default
  pipeline stays byte-preserving.
- **Domain-pack `corrections:` schema (#80, epic #78)**: a new frozen, per-run
  domain-pack field declaring deterministic **literal** substitution rules
  (`id`/`match`/`replace`/`case_sensitive`/`whole_word`) that fix recurring
  domain mis-hears offline, with no model or prompt. Strict load-time validation —
  literal-only, unique ids, an explicit whole-word boundary predicate (apostrophes,
  hyphens, and combining marks are intra-word, so `it→IT` never breaks `it's` and
  a rule never splits `Zoë`), a boundary-aware replacement-contains-match
  idempotence check, and hard bounds (256 rules / 256 match / 512 replace / 128 KiB)
  — surfaces a malformed pack loudly before a run is submitted. Ships the pure
  single-rule matcher the apply engine (#81) will reuse. **Schema + validation
  only; no runtime text change** — the `generic` pack declares none, so the
  default pipeline stays byte-preserving.
- **Design recommendation: deterministic, non-LLM transcript correction (#79,
  epic #78)**: a research-spike report
  (`docs/reports/nonllm-transcript-correction-design-2026-08-18.md`) recommending a
  deliberately narrow v1 — a pure-`stdlib`, versioned, per-segment
  **literal-substitution** engine driven only by a new frozen domain-pack
  `corrections:` field, refusing general homophone/casing/number/disfluency rules
  and executable regex, and gated by a stricter-than-LLM faithfulness corpus. Built
  by one AI panel and then **adversarially reviewed by a second 3-model panel**
  whose convergent findings hardened the design (§12): a **raw-gated dual-pass**
  composition with the optional LLM path (so a hallucinated term can't be entrenched
  as operator-authored), an explicit whole-word boundary predicate, a
  replacement-contains-match idempotence validation, a persisted
  `correction_trace`/`corrector_version`, and a required console
  authoring surface. Sequences the follow-up issues (#80 schema → #81 engine → #82
  composition+migration → #83 provenance → #84 authoring). **No runtime change** —
  analysis only.
- **Local-LLM qualification harness + frozen corpus (#66)**: a maintainer tool
  (`tools/qualify_local_llm.py`) and a hand-annotated, clean-room fixture corpus
  (`tests/fixtures/llm_qual/`, 19 fixtures + a frozen six-gate manifest) that
  drive Voxint's *unmodified* enhancement / run-asset / research code paths
  against a candidate local model and score structural validity, faithfulness,
  semantic usefulness, grounding, latency, and bounded-failure per-fixture across
  ≥3 reps. Verdict for the first candidates in
  `docs/reports/local-llm-qualification-granite-2026-08-18.md`: neither IBM
  Granite 4.0 H-Tiny nor Qwen3-4B-Instruct-2507 (Q5_K_M) qualifies as an
  unrestricted bundled default (Granite obeys prompt injection; both are weak at
  the agentic research loop). The corpus + harness are reused as #67's acceptance
  gate. No change to shipped runtime behaviour.
- **Watch-folder ingest (#60, console-UX arc #47)**: drop a batch of recordings
  into a registered media folder and Voxint picks them up on its own — no per-file
  submitting. An opt-in beat sweep walks the operator's registered folders
  (Settings → Media folders), starts a run for each **new** recording, and **skips
  files it already knows** (dedupe on the media `source_path`, the same predicate
  the setup-wizard scan uses). It reuses the existing bounded, containment-safe
  scan (registered folders only, `incoming`/`artifacts` pruned, symlinks never
  followed, entry/file caps) and the race-safe submit primitive, so a re-scan or an
  overlapping sweep can never duplicate a run. A **settle window** (a file must sit
  unchanged, by newest of mtime/ctime, for `WATCH_FOLDER_SETTLE_SECONDS`, default
  60 s) keeps a file that is still being copied in from being ingested mid-write —
  the reliable way to add files is an atomic move/rename into the folder.
  **Off by default**, toggled from a tri-state control beside the folders panel
  (On / Off / use the installation setting) that applies with **no restart**; a
  plain-language status line shows the last check ("picked up 3 new files;
  12 already known; 2 waiting to settle", with a warning when a very large folder
  was only partially scanned). New settings `WATCH_FOLDER_ENABLED`,
  `WATCH_FOLDER_SWEEP_SECONDS` (cadence, default 5 min), `WATCH_FOLDER_SETTLE_SECONDS`;
  migration `0026` adds the runtime-override + last-sweep columns to `app_settings`.
  Skipping is honest — a file whose earlier run *failed* is "already known", not
  re-tried by the watcher (requeue it from the run detail); a renamed/moved file is
  treated as new.
- **Keyboard shortcuts + in-app cheat-sheet (#51, console-UX arc #47)**: the
  review-stepper island extends its verify-and-advance keymap so a solo operator
  can drive the whole adjudication loop from the keyboard. Beyond the shipped
  `v`/`n`/`p`/`e` (verify+advance / skip / replay / edit), **`j`/`k`** walk to the
  next/previous segment (the "go back" the forward-only skip lacked), **`1`–`9`**
  assign the focused segment to the Nth roster speaker and **`0`** resets it to its
  detected label, and **`?`** opens a cheat-sheet listing every shortcut. The
  cheat-sheet is a focus-trapped, theme-aware modal dialog reachable by mouse too
  (a **"⌨ Shortcuts"** button), and digit-assign has a visible **whole-segment
  speaker picker** as its clickable twin — so every shortcut stays clickable and
  none is the sole path to its action. Unmodified keys only (no browser-shortcut
  collisions), never firing while a text box or menu is focused or while the
  cheat-sheet is open; **Space** (play/pause) and the **arrow keys** (scroll) stay
  with the native audio player as before. Digit-assign reuses the existing
  whole-segment `/relabel` scope (no new backend); a React island change only.
- **Native tier: guided Postgres major-version upgrade (`upgrade-db`, #71)**: the
  native macOS launcher can now move real data forward one Postgres major at a
  time (first certified edge 17 → 18) with a **dump/restore** upgrade. It runs the
  old cluster briefly on a private Unix socket (needs the old `postgresql@NN`
  binaries, or `VOXINT_NATIVE_OLD_PG_BINDIR`), dumps `voxint` with the **new**
  `pg_dump` (`--exclude-extension=vector --quote-all-identifiers`), and **proves
  the dump restorable before touching the data directory**, then renames the old
  cluster aside as a rollback (`pgdata.pg<old>-<stamp>`), `initdb`s the new major,
  and rebuilds via the tested `restore --fresh` path (pgvector-safe,
  single-transaction, `alembic upgrade head`). Fail-closed throughout: same-major
  is a no-op, downgrades and skipped majors are refused, and the stack must be
  fully down (no api/worker/beat, no supervised datastores, nothing on the PG
  port). A source-inventory gate refuses a cluster carrying extra databases or
  unexpected extensions a single-database dump can't preserve; a disk-headroom
  gate refuses if there isn't room for a second cluster + the dump. On any failure
  after the cutover it **auto-rolls-back** — the partial new cluster is set aside
  as `pgdata.failed-<stamp>` (never deleted) and the old cluster restored — and
  the same recovery is exposed as `upgrade-db --rollback`; `up` refuses (pointing
  at `--rollback`) if it finds a set-aside cluster but no live one. A maintainer
  `--rehearse` flag forces the full cycle at the same major for a mechanical
  proof. The old cluster is kept for you to delete once a good run is confirmed.
- **Responsive + accessibility polish (#64, console-UX arc #47)**: the
  server-rendered console gets a baseline of responsive and accessible behaviour
  so it's usable beyond a desktop developer's screen. A **skip-link** to a real
  `<main>` landmark (focusable via `tabindex="-1"`, so it moves focus rather than
  only scrolling) and a visible, theme-aware **`:focus-visible` ring** (with a
  forced-colors/high-contrast fallback) make keyboard and assistive-tech
  navigation coherent. The wide data tables — the runs list, the 8-column stage
  ledger, and the review queue — now scroll inside their own **keyboard-reachable,
  labelled `role="region"` container** instead of forcing horizontal *page*
  overflow on a phone, and their column headers carry `scope="col"`; the small
  dashboard metric tables get a plain scroll container (no needless focus stop).
  Long unbroken **source paths and URLs** rendered outside a table (run detail,
  run summary) now **break instead of forcing page overflow** (`overflow-wrap`),
  closing the last horizontal-scroll gap at phone widths. The primary nav wraps,
  and a single `max-width: 40rem` breakpoint tightens the layout on narrow
  screens. Status is confirmed **never conveyed by colour alone** — every status
  pill already renders its state word as text, now locked by a test — and the
  light/dark theming is preserved, with a darker-legible error colour added for
  the dark scheme. Pure CSS + template markup (the inline `base.html` stylesheet
  and the table pages); no Python, no new dependency.

### Changed
- **Docs: non-NVIDIA install-path audit remediation (docs pass)**: closed the
  documentation findings from the 2026-08-18 audit of the CPU / Apple-Silicon
  metal / docker-free native paths
  ([`docs/reports/native-cpu-mac-audit-2026-08-18.md`](docs/reports/native-cpu-mac-audit-2026-08-18.md)).
  [`docs/setup.md`](docs/setup.md): honest prerequisites (the metal tier needs
  Homebrew + `uv`; not everything is "just Docker"), a hardware→tier decision aid
  that separates deployment path from compute tier, the CPU-tier memory floor and
  its opaque-OOM footgun (16 GB comfortable), the Docker-Desktop-only caveat for
  metal, a `status` verification step, and a clearer metal-vs-native signpost.
  [`docs/operations.md`](docs/operations.md): the same metal `uv`/Docker-Desktop
  prerequisites. [`docs/native-macos-preview.md`](docs/native-macos-preview.md): a
  "next: first-run walkthrough" bridge into onboarding/how-to, and `doctor`
  port-reachability wording. The terminal `docker compose exec … voxint` recipes in
  the how-to and onboarding guides are now labelled Docker-only (they don't apply to
  the native preview). No behaviour change.
- **Docs: accessible README, a Setup guide, and a how-to series**: the README
  was rewritten for a non-technical audience — a plain-language intro, an honest
  "local by default" privacy note (URL fetch and remote LLM enhancement are
  called out as opt-in network features), a short maturity notice, a fresh
  dark-theme screenshot gallery of the current console, and a Quickstart that
  leads with the guided installer and links the deeper material. A new
  [`docs/setup.md`](docs/setup.md) consolidates per-OS Docker install pointers
  and every compute tier (CPU / NVIDIA / AMD ROCm / Apple metal), and a new
  [`docs/how-to/`](docs/how-to/README.md) series covers the day-to-day tasks
  (add media & manage runs, review & adjudicate, manage speakers & export,
  settings & troubleshooting). The console screenshots (`docs/images/`) were
  regenerated against v0.17.0 in dark theme — the waveform strip, keyboard
  cheat-sheet, split/reassign, the setup wizard, and the Settings pages that the
  two prior images predated. Docs only; no code change.
- **Transcript enhancement is hardened against prompt injection (#67 groundwork)**:
  the enhancement system prompt now instructs the model to treat every segment's
  text strictly as content to edit, never as instructions — a segment that reads
  like a command ("ignore previous instructions", "reply with a single word", "you
  are now a translator", "drop the other segments") is returned unchanged, not
  obeyed. Transcripts are untrusted input and small local models otherwise follow
  instructions embedded in speech; the clause measurably stops that against the
  local-LLM qualification corpus (obedience 8/8 → 0/8 with no regression). It is a
  best-effort guard backstopped by the structural batch-integrity gate, not a
  sandbox. The LLM client also gained a fixed **sampling profile** (default greedy,
  `temperature 0`, byte-identical to before; a bundled model can pin its own), and
  the qualification harness a `--sampling greedy|qwen` flag. Measured evidence
  corrected the #66 verdict's claim that Qwen "resists injection" and settled the
  #67 serving-profile questions — see the correction in
  [`docs/reports/local-llm-qualification-granite-2026-08-18.md`](docs/reports/local-llm-qualification-granite-2026-08-18.md).

### Fixed
- **CI: the native `upgrade-db` launcher tests now run on Linux (#71)**: 17 tests
  in `tests/unit/test_native_launcher.py` (the `upgrade-db` version-gate, arg-parse,
  and rollback-shape tests) invoke `cmd_upgrade_db`/`cmd_up`, which start with
  `require_macos` — so on the Linux CI runner they failed at the OS gate before
  reaching the portable logic they assert (green on maintainer macOS, red on CI
  since the slice-2b landing). `require_macos` now no-ops under the existing
  `VOXINT_NATIVE_LIB=1` library/test seam, so the 16 portable-logic tests are
  exercisable on Linux while a real (non-library) invocation on a non-macOS host
  still gets the clean "macOS-only" error. The one end-to-end rung
  (`test_upgrade_happy_path`) brings the cluster up and `plutil`-lints the generated
  plists, a macOS-only step, so it now carries the same `plutil` skipif the file's
  other plist tests use. Verified by reproducing the failures under a Linux `uname`
  shim (and a plutil-less runner) and confirming green after the change, with the
  guard still firing for a real non-macOS user.
- **PyPI wheel/sdist re-include the prebuilt frontend island bundles**: #69's
  `.gitignore` rule for `src/voxint/api/static/app/*` (clean-tree hygiene) made
  hatchling's VCS-ignore drop the built island bundles, so `uv build` produced a
  wheel containing only `static/app/.gitkeep` — a PyPI install could not hydrate
  the review-console islands (v0.16.0's wheel, built before that rule, included
  them). A global `[tool.hatch.build] artifacts` entry now re-includes
  `static/app/{assets,.vite}` in both the sdist and the wheel. Docker images were
  never affected (the Node stage COPYs `dist` into the image directly).

### Security
- **Native launcher hardening (#71)**: closes findings from the 2026-08-18
  repo security audit (`docs/security/audit-2026-08-18.md`), calibrated to the
  single-operator threat model. The launchd plists (which embed `DATABASE_URL`
  with the DB password, `VOXINT_PASSWORD`, and `CSRF_SECRET`) and `pg_dump`
  backups are now created mode `0600` and the `~/.voxint-native` tree `0700`,
  instead of the umask-default `0644`/`0755` that exposed credentials to other
  local accounts. A new `validate_native_inputs` gate (run for every subcommand
  and after `load_state`) fails closed on any unsafe operator-settable
  `VOXINT_NATIVE_*` value before it can reach a shell (`pg_ctl -o` — ports
  restricted to `1..65535`), superuser SQL (DB role/name restricted to a safe
  identifier grammar; the `CREATE ROLE` password now passed as a psql variable
  rather than an inline literal), a launchd plist env record (CR/LF rejected in
  every serialized value, so a newline cannot forge a second `PYTHONPATH` entry),
  or bash arithmetic (log sizes must be positive integers). `restore --fresh`
  now takes an automatic pre-drop safety backup (mode `0600`) and aborts before
  any destruction if it fails, so an incomplete restore can be recovered. No
  change to the normal happy path; verified live end-to-end (setup + backup +
  `restore --fresh`) on the native macOS lane.

## [0.17.0] - 2026-08-18

### Added
- **Setup wizard: honest first-run readiness checks (#61, settings-overhaul arc
  #47)**: the wizard's model-services step now surfaces the full `voxint doctor`
  readiness checks in the browser — **Postgres, Redis, the three model services,
  and (when LLM enhancement is on) the LLM endpoint** — instead of probing the
  model services alone. Each dependency renders in one of three honest states:
  **ready**, **failed** (a required dependency is down — the pipeline can't run
  until it's fixed), or **unverified** (an optional/advisory check couldn't be
  confirmed), each failure paired with a plain-language fix, never a stack trace
  and never a false all-good. The LLM check honours the **effective** (#74
  row-over-env) endpoint/key and only appears when enhancement is enabled. The
  Hugging Face token check is deliberately **cut** from the wizard — the default
  install runs on vendored weights, so it's noise (and skipping it also avoids a
  live huggingface.co call the step has no reason to make). No secrets, endpoints,
  or DSNs are ever rendered, and the step renders even when Postgres itself is down
  (the failed-database row is exactly what it must show, never a 500). Pure app +
  template; the checks reuse the existing `diagnostics.py` functions with no logic
  fork. *(Known behaviour: enhancement also needs a key and a fitting budget to
  actually run, so the LLM row can read "enabled but unreachable" when the master
  toggle is on without a usable key — surfaced honestly rather than hidden.)*
- **Folder browser + per-folder domain-pack picker (#63, settings-overhaul arc
  #47)**: the setup wizard's media step and a new **Settings → Media folders**
  section replace the raw newline-path textarea with an **htmx directory browser**
  confined to `MEDIA_ROOT`. Click into sub-folders, register the ones Voxint should
  watch, and assign each a **domain pack** via a `<select>` (a "Default" leaves it
  unmapped) — completing the in-UI half of #11. The server re-validates containment
  on every request and never trusts a client-supplied path (traversal, symlinks,
  and the reserved `incoming`/`artifacts` trees are all rejected; a bad path
  recovers to the media root with an honest notice, never disclosing what was
  attempted). Every mutation serialises on the singleton `app_settings` row, so
  overlapping edits cannot lose an update or leave a `folder_domain_packs` mapping
  whose folder is no longer registered; removing a folder prunes its mapping. A
  stored pack the registry no longer offers renders as **"(unavailable)"** rather
  than a false "Default", and a registry that cannot be listed disables pack
  selection with a plain-language message instead of failing the page. No schema
  change (reuses the `media_folders` and `folder_domain_packs` columns). The old
  `POST /setup/media` textarea route was removed.
- **Native tier: Postgres major-version skew detection (#71)**: the native macOS
  launcher now reads the managed cluster's on-disk major (`PG_VERSION`) and the
  installed binaries' major (parsed from `postgres --version`, tolerating the
  Homebrew vendor suffix) and refuses `up` — **before** starting anything — when
  they differ, instead of letting the postmaster fail with a cryptic "database
  files are incompatible with server". `doctor` reports the same check as a
  PASS/FAIL naming both versions. The message is actionable: install the matching
  major **and** repoint `VOXINT_NATIVE_PG_BINDIR` at it (installing the formula
  alone does not change which binaries the launcher uses). A guided,
  data-preserving major-version *migration* (`upgrade-db`) is the next #71 slice.
- **Settings → Sources & research: in-UI web-research config (#76, settings-overhaul
  arc #47)**: a new **Sources & research** section on the Settings page lets a
  non-technical operator configure web research entirely from the browser — the
  web-research **master** and **enrichment producer** toggles as **tri-state**
  controls (On / Off / use installation setting), the search-provider **endpoint**
  and **API key** (a secret, stored like the LLM key: never rendered back,
  blank-keeps-stored, a remove-checkbox reverts to the environment), and a
  **trusted-domains** editor that raises a draft's review priority for citing
  those domains (it boosts scoring only — it never blocks any site). Backed by
  #74's `resolve_effective_*` resolvers, so edits take effect on the next job with
  **no `.env` edit and no restart** (DB-row-wins-over-env; blank writes `NULL` to
  inherit the installation setting). `POST /settings/web-research` validates the
  whole submission — the cross-flag invariants through the single shared
  `validate_effective_flags`, plus the endpoint and each domain token — and on any
  violation re-renders with a plain-language message and the operator's non-secret
  choices preserved, **writing nothing**. This delivers the standing
  `source_authority_domains` "move to the settings UI" promise.
- **Start the guided tutorial from the UI (#75, settings-overhaul arc #47)**: the
  bundled three-speaker walkthrough no longer needs the `voxint tutorial seed` CLI
  step — a non-technical operator can stage and start it entirely from the browser.
  The setup wizard's Finish step now offers **"Finish setup & start tutorial →"**
  (alongside a plain **"Finish setup →"**), and the Settings page offers **"Set up
  & start the guided tutorial →"** when it has not been staged yet; both seed the
  sample idempotently and drop the operator straight into the run. The `start_tutorial`
  intent — not mere availability — drives the launch, so a plain finish never starts
  a tutorial and the button label never lies. A concurrency-safe advisory lock inside
  the shared seeder serialises seeds (CLI and web) so no duplicate tutorial run can
  be built, and a failed seed (media folder not writable, or missing/unreadable
  bundled data) rolls back and re-renders **bounded, non-secret** guidance without
  completing onboarding — never a stack trace or a path. The CLI seed still exists
  for scripted/maintainer setups.
- **Settings → Features: in-UI runtime toggles (#62, settings-overhaul arc #47)**:
  a new **Features** section on the Settings page exposes the live-read capability
  flags as real **tri-state** controls — **On / Off / use installation setting** —
  backed by #74's `resolve_effective_*` resolvers, so a non-technical operator
  turns speaker-name suggestions, the LLM name pass, run assets (+ auto-generate),
  and URL downloads on or off from the browser with **no `.env` edit and no
  restart** (DB-row-wins-over-env; "use installation setting" writes `NULL` so an
  override never permanently pins). The `POST /settings/features` route validates
  the effective combination through the single shared `validate_effective_flags`
  and, on an invariant violation (e.g. the LLM name pass without LLM enhancement),
  re-renders with the plain-language message and the operator's choices preserved,
  **writing nothing** — it never silently enables or disables an unrelated setting.
  The Settings page is now decomposed into self-contained per-section partials
  (`templates/settings/`), each with its own CSRF-guarded POST, so later arc
  children slot a section in without disturbing another's save.
- **Native macOS/arm64 core-stack technical preview** (#69, the MVP of epic
  #68 "run without Docker"): a `launchd`-supervised launcher
  (`scripts/native/voxint-native.sh`) that runs the whole stack on Apple Silicon
  **without Docker** — a launcher-managed private PostgreSQL 17 + pgvector,
  Redis, the API server, and the Celery worker/beat, all under
  `~/.voxint-native/` on collision-free ports and isolated from any Docker or
  brew Postgres. `up` provisions the role/db/extension and runs
  `alembic upgrade head` before starting the app (compose's migrate gate);
  `setup` builds and stages the review-console islands and, by default, drives
  `scripts/metal/voxint-metal.sh` so **one command brings up the whole preview**
  (core + whisper/pyannote/titanet). `--no-models` skips that for operators
  running the models elsewhere. Includes backup/restore, `copytruncate` log
  rotation (a daily `launchd` job), a `doctor` that checks every prerequisite,
  and the offline unit + contract tests that pin the launcher's env/plist logic
  and its command/port parity with compose and the metal launcher. Technical
  preview, **not** the signed non-technical release (#73); on the metal tier
  long recordings take real compute time. See `docs/native-macos-preview.md`.
- **Native launcher `restore --fresh` disaster-recovery** (#69): a destructive
  `scripts/native/voxint-native.sh restore --fresh <dump>` that, with the app
  services down, drops the database, proves it is genuinely empty (OID flip +
  zero public tables), and rebuilds it from a dump as the sole schema source —
  the vendored pgvector extension is preinstalled by the superuser and excluded
  from the restore so the unprivileged role never recreates a non-trusted
  extension. It **fails closed before touching your data**: it refuses while
  api/worker/beat are supervised, verifies the archive is a voxint dump
  (`alembic_version` in its TOC) and that the postmaster on the port is the
  managed cluster, all **before** the drop; the restore runs in a single
  transaction (no `--clean`). Recovery scope is DB-only (not media/weights).
- **Word-boundary segment splits** (#59, slice 2): an operator can
  split a mis-split diarization segment at a word boundary. A split is stored as
  an append-only *cut* ("split before word i") in a new `segment_split_boundaries`
  table (migration 0023) — never a new transcript row and never a mutable overlay
  the append-only ledger points at; the parent segment row stays immutable ASR
  evidence. Children are *derived* at read time from the parent's word tokens
  through the one shared read path (`attributed_transcript`), so the transcript
  export now reflects splits too, and each child inherits the parent's resolved
  speaker (per-child reassignment is a later slice). Verification stays
  parent-scoped (children never double-count the N-of-M queue). A split is
  claim-gated and structurally idempotent (the cut's UNIQUE key makes a replay a
  no-op — no nonce). Conservative by doctrine: a segment is splittable only when
  its stored words exactly reconcatenate to `raw_text` and its text was not
  materially enhanced — otherwise the console shows an honest "unsplittable"
  affordance rather than inventing offsets. Splitting a corrected segment (and
  correcting a split one) are mutually refused. New routes:
  `POST /review/{run}/segments/{seg}/split` and a lazy
  `GET /review/{run}/segments/{seg}/words` (words are fetched only when split mode
  engages — never bloating the shared read payload). In the review console a
  **⎇ Split at a word** toggle turns the focused segment's words into clickable
  cut points; the words load lazily on first use. A split segment's derived
  children keep their own word-derived text (a parent-scoped verify never clobbers
  them), and editing is disabled on a split segment with a plain note — splitting
  and free-form correction are mutually exclusive. An unsplittable segment says so
  rather than offering a control that would fail. Splitting needs the browser
  island; with JavaScript off the transcript still lists any already-derived
  child lines.
- **Sub-segment speaker reassignment** (#59, slice 3, backend): a derived split
  child (or any immutable word-range of a segment) can be reassigned to a
  different speaker, so the two halves of a mis-split segment can carry the two
  real speakers instead of sharing the parent's one. The scope is an append-only
  ledger ruling keyed on the *immutable parent segment id + a half-open
  `[start, end)` word-range* (nullable `start_word_index`/`end_word_index` on
  `adjudication_decisions`, migration 0025) — never a foreign key to a disposable
  split-boundary row — so a reassignment survives re-split/un-split. Read-time
  precedence is most-specific-wins: a word-range override beats a whole-segment
  override beats the label, and an `inherit` on the exact range removes it live
  (append-only, never a frozen copy). Applied through the one shared read path
  (`attributed_transcript`), so the transcript export reflects a reassigned child
  too. The `POST /review/{run}/segments/{seg}/relabel` route gains an optional
  `start_word_index`/`end_word_index`; a range is accepted only when it matches a
  *current* split child (never an arbitrary span the read path would ignore) and
  is validated against the parent's word count. Same claim-lock + nonce
  idempotency as every other review mutation. The **review-stepper island now
  renders a per-child speaker picker**: each derived child line carries its
  `[wordStart, wordEnd)` and a `<select>` of the active roster; choosing a
  speaker reassigns just that child, choosing "inherit" resets it to follow the
  label. The picker POSTs the range to `/relabel` (which gains a JSON-Accept
  response path returning the whole-run reconcile, leaving the htmx labels
  workbench's HTML fragment byte-identical) and the console adopts server truth
  wholesale. Splitting an already-split segment into more than two parts is now
  refused server-side (it would re-derive the children and orphan a reassignment
  keyed on the old coordinates); the existing cut still replays idempotently.
  (Un-split/re-split of an already-reassigned range, and ranged *correction* of a
  split child, remain later work.)
- **Per-word timings captured from ASR** (#59, foundation): the whisper service
  already computes word-level timestamps (`word_timestamps=True`) but voxint
  dropped them at the client seam; they now flow through and are stored as a
  nullable `words` JSONB column on `transcript_segments` (migration 0022),
  bucketed into their segment by maximum temporal overlap. No numerics change
  (word timing was already part of the decode config), no UI change yet, and no
  backfill — runs transcribed before this keep `words = NULL`. This is the
  groundwork for click-a-word-to-split in the review console.
- **Waveform strip with per-speaker regions** (#57): the transcript and review
  pages draw a compact amplitude strip under the audio player, tinted per
  speaker from the **diarization turns** (the honest who-spoke-when record —
  overlapping speech gets a hatched marker; diarized-but-untranscribed speech
  still appears), using the same palette as the segment list. Clicking the
  strip selects that segment in the list and — only when the fail-closed seek
  gate (#55) trusts the timeline — plays it; the review surface also shows the
  cursor position, and a playhead tracks playback. Rendering is a hand-rolled
  canvas (no new frontend dependency — a contract test now pins the runtime
  npm dependencies to exactly `react`+`react-dom`; deliberate deviation from
  the issue's wavesurfer.js suggestion, rationale on #57). Peaks are computed
  lazily on first view from the normalized WAV by a new
  `GET /media/{run_id}/peaks` route and cached as a `waveform_peaks` artifact
  (migration 0021) with a source fingerprint so a re-prepared run can never
  show a stale envelope; the cached strip keeps rendering (statically) after
  the WAV is reclaimed. Fully offline; ~14 KB payload regardless of duration.
- **Runtime feature-flag foundation (#74, settings-overhaul arc #47)**: the
  singleton `app_settings` row gains one nullable column per in-UI-editable
  feature flag (`enrichment_names_enabled`, `enrichment_names_llm_enabled`,
  `enrichment_run_assets_enabled`, `enrichment_run_assets_autogenerate`,
  `voxint_web_research`, `enrichment_web_research_enabled`, `ytdlp_enabled`,
  `source_authority_domains`, `web_search_base_url`, `web_search_api_key`;
  migration `0024`). Each resolves **DB-row-wins-over-env** (the `llm_*`
  tri-state precedent): NULL/blank inherits the environment default, a stored
  value overrides it — so once the console lands (later arc children), an
  operator toggle applies at the next job with no restart and no `.env` edit.
  Every runtime gate now routes through a `resolve_effective_<flag>` resolver
  and the five cross-flag invariants live in one `validate_effective_flags`
  shared with the boot-time config validator. `web_search_api_key` is a
  credential (plaintext at rest, like `llm_api_key`): resolved only through its
  resolver, never rendered or logged. Purely additive and behavior-preserving —
  with every column NULL the environment still governs exactly as before. No UI
  in this change.
- **Whisper Metal bakeoff (#33) — mlx candidate measured ineligible**: the
  Slice-3 decode diagnostic
  (`docs/reports/whisper-metal-bakeoff-slice3-decode-2026-08-17.md`) measured
  mlx-whisper large-v2 (fp16, greedy) at 19–21 pp pooled disagreement vs the
  frozen CT2 baseline against a ≤2.0 pp gate, under every decode configuration
  tested — a negative result recorded per the numerics doctrine (the gate is
  unchanged). `docs/gpu-contracts.md` gains the dated verdict block, names
  whisper.cpp Metal as the next measured candidate arm (since measured — see
  the entry below), and corrects the
  performance-gate wording to the intended `speedup = CT2 wall / candidate
  wall ≥ 1.5×` form. No behavior change: `mlx` was never registered in
  `WHISPER_ENGINE`.
- **Whisper Metal bakeoff (#33) — whisper.cpp candidate measured ineligible**:
  the fail-fast diagnostic
  (`docs/reports/whisper-metal-bakeoff-whispercpp-arm-2026-08-17.md`) measured
  whisper.cpp Metal (`pywhispercpp==1.5.0`, `ggml-large-v2-q8_0`, beam_size 5,
  CT2-parity decode map) at 98.57 pp worst-file disagreement vs the frozen CT2
  baseline against a ≤5 pp gate — the same confident headset-crosstalk
  transcription that killed mlx. Measured attribution: whisper.cpp's "beam"
  samples candidates rather than expanding top-k, so beam-5 ≈ greedy on the
  failing windows, and an f16 control reproduced the Q8_0 blowup (quantization
  irrelevant). Clean-file drift, reconstructed confidence, and performance all
  pass or near-pass; re-measure only if upstream lands true top-k beam
  expansion. `docs/gpu-contracts.md` gains the dated verdict block: with mlx
  and whisper.cpp measured-ineligible and CT2-MPS deferred upstream, no Metal
  `WHISPER_ENGINE` candidate is eligible and `ct2` remains the default and
  only shipped engine. No behavior change.
- **Draft triage: multi-signal review-priority scoring** (#42): enrichment drafts
  now carry an explainable **review priority** that orders them and populates the
  `unresolved` bucket, so an operator sees the strongest name and profile
  suggestions first. It fuses, with visible components, per-producer name-match
  strength (via adapters — a producer's raw score is never treated as comparable
  to another's), grounded voice support that must match the proposed identity
  (a cosine naming someone else is shown as a **voice conflict**, never a boost),
  the count of **distinct evidence domains** (not raw URLs), an operator-editable
  source-authority allowlist (`SOURCE_AUTHORITY_DOMAINS`, empty by default — no
  built-in list), and cross-producer agreement. The score is **derived at read
  time** (no schema change), **capped below certainty**, and **never
  auto-accepts** — auto-accept thresholds need a deployment's own adjudicated
  history and stay out of scope. To make the domain/authority/agreement signals
  real, the web-research producer now **keeps every independently grounded
  source** for a value (previously it dropped duplicate sources) as separate
  evidence rows. Documented in `docs/enrichment-triage.md`; the LLM-key and
  research config knobs are unchanged. Off-by-default features stay off.
- **Review-queue & dashboard ergonomics** (#56): the adjudication queue and the
  operator dashboard now read for a non-technical operator instead of an
  engineer. Each **queue** row shows a **friendly title** (the acquisition
  metadata title when present, else a cleaned, percent-decoded filename), the
  recording **duration**, its **age** ("3 hours ago", with the exact UTC time on
  hover), and a **resolved-of-total progress bar** that fills toward done — so a
  glance says both *what* a recording is and *how much* is left to adjudicate.
  The queue can be **sorted** "Oldest first" (default, unchanged FIFO order) or
  "Most voices to resolve". The `/runs` listing shows the same friendly title and
  relative age. On the **dashboard**, the time window is now a **24h / 7d / 30d
  picker on the page** (no longer URL-only) and pipeline **stage names are
  human-readable** ("Diarize & embed"). The machine-facing `/metrics`, JSON, and
  `voxint stats` outputs are unchanged — humanization is display-only, and the
  status pills keep their colours. No new dependency, config, or schema change.
- **Restricted URL-download egress overlay** (#16): a new opt-in
  `compose.ytdlp-egress.yaml` productizes the previously docs-only "run the worker
  with restricted egress" guidance for the URL-ingestion SSRF residual. It routes
  yt-dlp's always-passed `--proxy` through a small filtering forward proxy
  (`voxint.media.egress_proxy`, shipped in the same image — no extra download)
  that re-applies the **same** public-address policy the worker gate uses
  (`ip_is_public`) **at the connection boundary** and connects only to the vetted
  public IP. Because the proxy makes the outbound connection, this closes the
  DNS-rebind window and refuses redirect / extractor destinations that resolve to
  a private address — the part yt-dlp's independent re-resolution reopens. The
  worker keeps its normal network (Postgres/Redis/model-services/LLM unaffected).
  Stated honestly, it is **not** a sandbox: a helper yt-dlp spawns that ignores
  the proxy, or the worker's own routable network, still wants a host egress
  firewall (`docs/operations.md`, "Restricted URL-download overlay"). A passive
  notice beside the URL-fetch form points operators to it — no config knob, no
  status badge. Default deployments are unchanged.
- **Per-segment verify + transcript text correction** (#53 #58, #47): the
  review console can now record that a segment has been **checked** and let the
  operator **fix the words** the model got wrong — the two halves of "adjudicate
  the results" that were missing. Both are stored as mutable, per-segment
  operator state (`segment_review_states`), kept separate from the immutable ASR
  evidence: `raw_text` is never overwritten, a correction is written beside it,
  and `?text=raw` always shows the untouched original. The default transcript
  view and every text export now render corrections (`corrected → enhanced →
  raw`); `?text=enhanced` still gives the pipeline text without them. Editing a
  segment's text clears its verified mark (edited text must be re-checked).
  Reverting to the pipeline wording (or clearing the box) removes the
  correction. Claim-gated like every other review write. **Corrections are also
  full-text-searchable** (#58): a word you fixed is findable from the `/runs`
  search alongside the raw and enhanced renderings — never coalesced, so a term
  is found in whichever rendering contains it, and the hit snippet shows the
  corrected wording. **Name enrichment and generated summaries now read
  corrections too** (#58): the offline/LLM name miners and the run-asset
  generators consume the same effective text the console shows, so fixing a
  proper name honestly re-mines names and marks the affected summaries stale for
  regeneration (operator-triggered — no rerun storm). **A keyboard-first
  verify-and-advance loop** (#53) makes this usable end-to-end: a claim-gated
  review surface (`/review/{id}/transcript`, reached from the workbench) walks
  the operator through the transcript one unverified segment at a time — press
  **`v`** to confirm a segment and jump to the next (uncertain segments are
  surfaced with the "uncertain" chip), **`e`** to edit its words in place, **`n`**
  to skip, **`p`** to replay — with a live "N of M verified" readout. Editing
  posts a correction (clearing the verified mark); every keystroke is
  typing-guarded so the keys never fire while you are typing, and Space/scroll
  keys stay with the native player. It **degrades to plain HTML**: with
  JavaScript off the same page lists every segment, with a plain "Verify" form
  on each one still unverified (editing needs the island, stated honestly), and
  it renders read-only with a prompt to claim when this tab does not hold the
  run's claim.
- **Low-confidence highlight in the transcript** (#53, #47): faster-whisper
  already reports how sure it was of each segment, but Voxint used to throw that
  away. The transcript now flags segments the model was **uncertain** about — a
  dashed underline and a small "uncertain" chip — so a non-technical operator is
  drawn straight to the parts worth a listen instead of re-reading everything.
  The label is deliberately honest: **"uncertain, not necessarily wrong"** — the
  score is a transformed likelihood, not a probability of error, and it never
  claims a percentage. Segments the model reported no confidence for (older runs)
  are never flagged. The cutoff is a configurable setting
  (`REVIEW_LOW_CONFIDENCE_THRESHOLD`, default 0.6), intentionally not a UI
  slider. Reads existing model output only — inference and the parity gates are
  untouched.

### Changed
- **`voxint doctor` LLM check now distinguishes reachable from ready (#61)**: a
  `/models` response is only reported healthy on a **2xx**; a 4xx/5xx (a wrong key →
  401, a wrong endpoint path → 404, a broken provider → 5xx) is reported as an
  advisory miss rather than "reachable", because a real enhancement call would be
  rejected the same way. Still advisory — the LLM endpoint's state never changes the
  doctor exit code — and the base URL and key are never printed. This makes both
  `voxint doctor` and the new setup-wizard readiness step refuse to paint a bad key
  green.
- **The LLM settings form refuses a disable that would strand a dependent feature
  (#77, settings-overhaul arc #47)**: turning LLM enhancement off from the LLM
  section (setup wizard or Settings) while a feature that needs it is still on —
  run assets, the LLM name pass, or web-research enrichment — is now rejected with
  a plain-language message naming the blocker(s), and **nothing is written**, so
  LLM stays on. Previously the disable saved a combination the boot validator
  (`config.py`) would reject on restart (runtime stayed safe because the gates
  already fail closed). The form never auto-disables the dependent (that would
  silently flip an unrelated setting — #62); the operator turns the dependent off
  in the Features or Sources & research section first. The decision reuses the
  single shared `validate_effective_flags` and acts only on the invariants that
  *disabling LLM* newly introduces, so an unrelated pre-existing issue never blocks
  an LLM save. (One narrow edge remains by design: the fail-closed *enable* path —
  requested-on but no usable key/budget — still forces LLM off and can leave a
  dependent stranded; it returns the primary key error and is out of #77 scope.)
- **Gated-feature panels remediate in plain language, not env vars (#62)**: the
  "run assets are off" and "web research is off" blocks on the run and speaker
  pages no longer instruct a non-technical operator to set raw environment
  variables (`ENRICHMENT_RUN_ASSETS_ENABLED`, `VOXINT_WEB_RESEARCH`, …); they name
  the in-UI Settings toggle to turn the feature on instead. A contract test keeps
  the copy from regressing.
- **CLI honors the effective (row-over-env) capability gates (#74)**: `voxint
  fetch`, `voxint research search|read`, and `voxint enrich names` now resolve
  their enablement from the database (row-over-env) instead of a bare
  environment flag, so a future in-UI disable governs the CLI too. On an
  unavailable database these commands **fail honestly** (exit non-zero) rather
  than silently falling back to the environment, which could otherwise bypass a
  console disable. No change when no override is stored.

### Fixed
- **Native plain `restore` no longer trips the pgvector ownership footgun** (#71,
  epic #68): `scripts/native/voxint-native.sh restore <file>` (the non-`--fresh`
  path) used to run `pg_restore --clean --if-exists` as the unprivileged `voxint`
  role against a dump whose TOC carries the `EXTENSION vector` entry — so `--clean`
  tried to drop and recreate the non-trusted extension as `voxint` and failed
  (the same footgun `restore --fresh` was built to avoid). The plain path now
  reuses the fresh path's fail-closed preflight — services-down gate, archive
  integrity + `alembic_version` identity gate, managed-postmaster (`data_directory`)
  check, and vector-TOC filtering — preinstalls `vector` as the superuser, and
  restores inside a single transaction (`-L <filtered> --clean --if-exists
  --no-owner --single-transaction --exit-on-error`), so a failure rolls back and
  leaves the database unchanged and the dump untouched. It remains *replacement in
  place* (objects absent from the archive survive); `restore --fresh` stays the
  *exact rebuild* / disaster-recovery path. `backup` now takes new dumps with
  `pg_dump --exclude-extension=vector`, so fresh archives omit the extension
  entirely (legacy archives are still filtered at restore time). The two restore
  paths now share a `restore_preflight`/`restore_postmigrate` helper pair; the
  fresh path's destructive core is unchanged.
- **Saving LLM settings no longer silently pins the endpoint** (#46): the setup
  and settings LLM forms used to prefill the base-URL/model inputs with the
  *effective* value, which is the environment default when no override is stored
  — so saving any LLM change (even just adding a key) re-submitted that value and
  quietly converted the `LLM_BASE_URL`/`LLM_MODEL` fallback into a pinned
  database override, after which later environment changes stopped applying with
  no visible cue. The inputs now render **blank when inheriting** (the
  environment default is shown as the placeholder), and a save stores **NULL**
  when the field is blank *or* merely equals the environment default, so the row
  keeps inheriting; a genuinely different value is stored as a deliberate
  override. Same "revert to the installation setting" behaviour the *remove saved
  key* checkbox already gives for the API key.
- **Enrichment jobs no longer strand on a malformed LLM endpoint** (#46): the
  LLM client is now built *inside* each enrichment worker's failure boundary, so
  a malformed `LLM_BASE_URL` set only in the environment (never validated by the
  setup wizard) fails the run-asset, web-research, or LLM-name job cleanly —
  status `FAILED` with the plain message "LLM endpoint could not be initialized
  (check LLM_BASE_URL)" — instead of raising out of the worker and leaving the
  job stuck `RUNNING` forever (there is no recovery sweep). The catch is narrow,
  around client construction only, so a genuine HTTP error from web research's
  own fetches is never mislabeled an endpoint problem.

### Security
- **The LLM API key can no longer leak into logs via an echoed error body**
  (#46): when an LLM endpoint returns an HTTP error whose body echoes the request
  back (including the `Authorization` header), the key is now scrubbed from the
  error before it is raised and logged. Status and a redacted body are still
  surfaced for debugging.
- **Optional in-product close for the URL-ingestion SSRF residual** (#16): the
  `compose.ytdlp-egress.yaml` overlay (see Added) lets an operator constrain
  yt-dlp's egress — including its redirects and extractor-constructed URLs — to
  vetted public addresses without any external firewalling, closing the
  rebind/redirect residual for yt-dlp's own HTTP(S) traffic.

## [0.16.0] - 2026-08-17

### Added
- **Export picker — every format, one menu** (#52, #47): the console already
  produced SubRip (`.srt`), WebVTT (`.vtt`), JSON, and RTTM alongside plain
  text, but only the `.txt` download was ever linked — the subtitle and
  diarization formats a journalist or researcher actually needs were invisible.
  The workbench and the transcript page now show a single **Download transcript**
  menu listing all five formats with a plain-language description of each, plus
  the raw/enhanced text choice (RTTM carries the raw diarization labels, so it
  has no variant). Plain text additionally offers a **timestamp-free** copy —
  just the words with speaker labels, for pasting into a document — mirrored on
  the command line as `voxint export --format txt --no-timestamps`; the download
  is byte-for-byte identical to the CLI. Built as plain HTML links (no new
  frontend dependency; works with JavaScript off). Subtitle timing and the JSON
  key layout are left untouched — a subtitle without timing or a JSON with
  missing keys would be a broken file, not an option.
- **Inline speaker merge in the workbench** (#54, #47): the most common
  diarization fix — one person split across labels (`SPEAKER_00` + `SPEAKER_03`)
  — is now a one-click action where the operator is actually reviewing, instead
  of a trip to the separate roster page. A **"Same speaker across labels?"**
  panel lets the operator tick the labels that are one voice, choose who they
  are (an existing roster speaker or a newly enrolled one), and **preview the
  exact, server-computed change** (turns and transcript segments affected) before
  applying. The merge is **run-local**: it records one `assign` ruling per label
  to a single survivor within this recording and **never** performs a
  roster-wide identity merge (that stays the explicit `/speakers` action — the
  preview says so when two labels already map to distinct roster identities, and
  points there). Applies are **atomic** (all labels move together or none do,
  under the run's claim lock) with **optimistic-concurrency** safety: if any
  label's ruling changed since the operator previewed, the confirm is rejected
  with a 409 rather than silently overriding a decision they never saw, and
  replaying a confirmed merge returns the original outcome (deterministic child
  idempotency keys). Every ruling lands in the existing append-only decision
  history; nothing is destructive and any label can be re-ruled afterwards. Built
  as progressively-enhanced htmx (no new frontend dependency).
- **Two-scope relabel — "this segment only"** (#54, #47): assigning a speaker in
  the workbench now offers both scopes. The existing per-label controls rule
  **all** of a label; a new per-segment control (on each previewed line)
  reassigns **just that one transcript segment** without touching the rest of its
  label, with a one-click **"reset to label"** that returns the segment to its
  label's resolution. Segment overrides live in the same immutable decision
  ledger (a nullable `transcript_segment_id`, not a parallel store) and are
  resolved at read time with a strict precedence — a segment override beats its
  label's ruling beats the machine proposal — so a later whole-label ruling never
  clobbers an explicit segment correction, and a reset **tracks the label live**
  rather than freezing a copy. The HTML transcript and every text export share
  the one resolver, so they can never disagree. Known v1 limitation (documented):
  the run **search** facet and the review queue stay label-scoped — a speaker
  present only via a segment override does not surface there.
- **Whisper `WHISPER_ENGINE` compatibility seam** (#33, Slice 2a): the whisper
  service now selects its decode engine through a fail-closed registry
  (`services/whisper/app/backends/`, mirroring titanet's `EMBED_ENGINE`
  factory) instead of a single hard-wired class. `WhisperTranscriber` becomes an
  engine-agnostic facade that dispatches by a typed backend descriptor
  (`legacy_file` vs `shared_windows`); the default engine `ct2-legacy` is a
  **byte-faithful mechanical move** of the shipped whole-file CT2 path (decode
  branches + result assembly), so the frozen #33 CT2-CPU baseline replays with
  zero drift — proven by a new Apple-Silicon-only maintainer gate
  (`tests/parity/test_whisper_ct2_legacy_replay.py`, plain SKIP elsewhere). An
  unknown `WHISPER_ENGINE` raises rather than silently degrading to CPU, and a
  new CT2 `verify_device()` hook fails closed on a device CTranslate2 cannot run
  (the whisper analogue of pyannote's `probe_device`; a no-op for the shipped
  cpu/cuda/rocm paths). The metal launcher pins `WHISPER_ENGINE=ct2-legacy`.
  `/healthz` identity, the public `transcribe` signature, and every existing
  test are unchanged. This is the structural half of Slice 2; the shared-VAD
  `ct2` backend and its self-parity gate (Slice 2b) fail closed until they land.
- **Whisper shared-VAD `ct2` decode engine** (#33, Slice 2b): the real
  `shared_windows` engine now decodes. A shared front layer
  (`app.backends.vad_plan.build_vad_plan` + the `WhisperTranscriber` facade)
  owns VAD, packing, packed→source time restoration and result assembly by
  reusing faster-whisper 1.2.1's OWN primitives (`get_speech_timestamps`,
  `collect_chunks`, `restore_speech_timestamps`) with the exact
  `BatchedInferencePipeline` parameters; the backend (`Ct2Backend`) does only
  the raw batched CT2 forward on the packed windows, so a future mlx backend
  consumes identical windows. The decode vehicle (direct `pipeline.forward` vs
  the public `transcribe(clip_timestamps=…)` path) was **chosen by
  measurement** — both are byte-exact on the comparison fixtures; `forward`
  wins because it feeds integer-exact audio and reconstructs no metadata. The
  batched decode reproduces `_batched_segments_generator`'s exact
  `Segment`/`Word` materialization (global ids, three-decimal rounding,
  `last_speech_timestamp` threading). The result-assembly loop is
  deduplicated into a shared front helper used by BOTH engines
  (`ct2-legacy` byte-identity still guarded by the frozen-oracle replay gate).
  Equivalence is proven by a new self-parity gate
  (`tests/parity/test_whisper_ct2_self_parity.py`, Apple-Silicon-only, plain
  SKIP elsewhere): `ct2 ≈ ct2-legacy` to **≤0.5pp pooled WER per vad mode**
  (micro-averaged S/D/I/N; empty-reference clips held to a separate
  zero-insertion invariant) over the committed synthetic fixture + a curated
  AMI subset spanning 2–10 packed windows. `/healthz` gains a cached decode
  identity (`decode_config_hash`, `vad_plan_version`, `vad_params`,
  `model_revision`) so two deployments are distinguishable and a numerics
  change is visible. **Behind-seam and off by default** (`WHISPER_ENGINE`
  stays `ct2-legacy`): zero behavior change until a deployment opts in.

## [0.15.0] - 2026-08-16

### Added
- **Follow-along highlight + per-speaker colors** (#50, #47): the transcript
  player now keeps the currently-playing line in view as playback advances
  (scroll-into-view, no smooth animation, no focus stealing). Following starts
  on and stops the moment the operator scrolls by any means (wheel, touch,
  keyboard, scrollbar); a single **"Resume following"** control — shown only
  while paused-from-following — turns it back on and re-centers the active line.
  A programmatic-scroll guard keeps the auto-scroll from being misread as a
  manual take-over. Each speaker also gets a **deterministic identity color**,
  assigned from one canonical per-run label universe so the transcript page, the
  JS-off fallback, and the workbench label cards all agree for the same label.
  Color is **supplemental only**: a raw-label badge is the primary, non-color
  identity cue on every surface (never color alone), and the curated,
  contrast-verified 8-color palette repeats past eight speakers by design, with
  the badge disambiguating repeats. The palette lives in CSS with light/dark
  variants.
- **Per-turn audio playback + honest seek gating** (#49, #55): the review
  console can now play just one transcript line, or preview a single speaker,
  from both the transcript page and the workbench. On transcript.html the
  `transcript-player` island grows per-line ▶ buttons and click-to-seek; on
  run.html a new `workbench-player` island owns the `<audio>` and drives
  server-rendered per-segment / "preview this speaker" buttons via
  document-level event delegation that survives every htmx swap of the decision
  cards. "Preview this speaker" seeks a clean diarization turn (longest
  non-overlap, fallback longest), never the longest transcript segment (which
  can contain other voices). A shared playback rate control (0.5×–2×) persists
  in `localStorage`. Seeking is **fail-closed**: a new backend capability
  predicate offers it only when the media is genuinely servable (reusing the
  exact `GET /media` servability seam, so it can never diverge), the duration is
  finite and positive, and every transcript timestamp is well-formed and inside
  the recording (within a fixed 0.05s tail tolerance). When any check fails the
  buttons stay disabled and a **visible banner** lists every reason in plain
  language — no false affordance, and manual scrubbing still works with JS off.
- **Frontend island foundation** (#48): prebuilt Vite/React/Tailwind bundles
  served through the auth-aware `/static/app` route (never a `StaticFiles`
  mount — the operator auth invariant stays absolute), a read-only audio-synced
  transcript island that highlights the playing segment over its server-rendered
  fallback, and the shared `data-island` + dynamic-import mount convention plus
  `api-client.ts` error seam for #49+. Node exists only in the Dockerfile build
  stage; no Node ships at runtime. A new CI `frontend` job gates
  lint/typecheck/build/audit and the offline no-CDN check.
- **Per-run / per-folder domain pack selection** (#11, backend): a run now
  freezes the resolved domain pack it was submitted with as a JSON snapshot on
  the run (`pipeline_runs.domain_pack`, migration 0017), stamped write-once at
  submit. Packs are selected **per watched folder** via a
  `{media_folder → pack_name}` map on `app_settings` (`folder_domain_packs`) —
  point a *podcast* folder and an *interview* folder at different packs — with an
  optional explicit override at submit; an unmapped folder uses the default pack
  (`DOMAIN_PACK_PATH`, else the bundled `generic`). Multiple named packs can live
  under a new `DOMAIN_PACKS_DIR` (one child folder per pack, resolved by manifest
  `name`). The **pipeline worker and the offline name producer both read the
  run's frozen snapshot**, not the live global env, so late enrichment can never
  diverge from what transcription used and a manifest edited on disk afterward
  never changes a past run's result. `DomainPack` gained strict, round-trippable
  serialization (`to_mapping`/`from_mapping`); a corrupt snapshot degrades to the
  default pack with a warning rather than wedging the run. Legacy runs
  (pre-migration, `NULL` snapshot) reproduce the prior global-pack behavior.
  _(The default pack, `DOMAIN_PACK_PATH`, is the operator-facing control in this
  release; the in-console UI to edit the per-folder map ships with the
  review-console overhaul, #63.)_
- **Domain packs shape more of the pipeline** (#11): two additional
  `prompt_fragments` keys are now consumed from the run's frozen pack, each with
  a single documented consumer (fragments are never concatenated). A
  `summary_context` fragment is appended to the run-asset LLM producer's system
  prompt, so the summary/topics/entity-mention analysis gets domain framing; a
  `name_attribution_context` fragment is added as a second labeled block on the
  transcript-enhancement call that harvests speaker-name hints (e.g. anchoring a
  recurring host). Both are fenced as advisory so a pack can guide but never
  override the strict reply schemas, and an absent fragment leaves the prompt
  byte-for-byte unchanged.
- **In-UI LLM API key** (#10): the optional LLM API key can now be set, replaced,
  and removed from the setup wizard and the Settings page — no more hand-editing
  `.env` and restarting the worker just to enable enhancement. The key is stored on
  the singleton `app_settings` row (migration 0016); precedence mirrors the other
  LLM settings: a value saved in the UI **wins**, and env `LLM_API_KEY` is the
  seed/fallback. A single resolver threads the effective key through **every** LLM
  client — transcript enhancement, the enrichment producers (names / web-research /
  run-assets), and `voxint doctor` — so a saved key is truly system-wide, and a
  changed key takes effect on the next run/job with no restart. The endpoint
  (`base_url`/`model`) a UI action enqueues is snapshotted per job while the key is
  resolved **live** at execution (never written into a job row). The key is a
  credential: **plaintext at rest** in Postgres — an accepted trade-off for this
  single-operator, local-first deployment (a SQL dump necessarily contains it) —
  and it is never prefilled, rendered back, logged, put in an error/validation
  message, or exported. Enabling still fails closed (an unusable key or an
  over-lease budget refuses to enable and shows why); a blank key field leaves the
  saved key untouched, and an explicit "Remove saved key" checkbox reverts to env.
  LLM **enablement** is resolved the same row-over-env way system-wide — including
  the enrichment producers, not just transcript enhancement — so turning LLM off in
  the UI stops enrichment jobs (and auto-generated run assets) with no restart, and
  the recorded web-research provenance names the endpoint that actually served the
  request.
- **Whisper Metal bakeoff corpus** (#33, Slice 1): `tools/prepare_bakeoff_corpus.py`
  now implements `generate` (fetch/synthesize every stratum, write a candidate
  `manifest.json`) and `prepare` (re-fetch + verify every file, fail closed) for
  the pre-registered whisper-engine bakeoff. Corpus = 15 AMI IHM word-gold
  windows (CC-BY-4.0, committed gold) + 15 TED-LIUM 3 windows (CC-BY-NC-ND-3.0,
  transcript hash only) + 15 synthetic CC0 fixtures (silence / hallucination-bait
  / short-clean, committed audio). All windows are a fixed content-independent
  240 s @ 120 s slice; selection is a seeded hash-rank (pre-registration-safe);
  AMI fetches only the window via HTTP Range. Prepared audio is byte-canonical
  16 kHz mono s16le with a committed per-file `sha256`; a manifest-schema contract
  test (`tests/contracts/test_bakeoff_manifest.py`) binds the committed gold and
  synthetic audio to the manifest and enforces the licensing doctrine (no TED
  transcript/audio committed). No new Python deps (stdlib + `soundfile`/`numpy`);
  synthetic regeneration uses `espeak-ng` + `ffmpeg` (versions pinned in the
  manifest provenance).
- **Whisper Metal bakeoff — frozen CT2-CPU baseline** (#33, Slice 1):
  `tools/generate_bakeoff_baseline.py` captures the load-bearing numerics oracle
  every Metal candidate is measured against, from the **unmodified** shipped
  `transcription.py` decode path (fails closed if it has uncommitted changes).
  For each corpus window it records both decode variants the frozen engine
  exposes — `vad_true` (production `BatchedInferencePipeline`) and `vad_false`
  (raw `model.transcribe`) — with per-segment/word text, timestamps, and
  `exp(avg_logprob)` confidence, over two warm passes that must agree
  (determinism gate). AMI (CC-BY-4.0) + synthetic (CC0) baselines are committed
  to `tests/parity/fixtures/references/ct2-cpu-metal/transcribe.json`; TED-LIUM 3
  (CC-BY-NC-ND) stays metrics-only (per-variant hypothesis hash, never text). The
  committed oracle pins the full runtime identity (CT2/faster-whisper/ORT/PyAV/
  ffmpeg versions, model revision, host, code SHA) — deterministic run-to-run on
  one runtime, not asserted byte-identical across machines. A contract test
  (`tests/contracts/test_bakeoff_baseline.py`) binds each committed entry to its
  manifest `sha256` and enforces the no-TED-leakage doctrine.

### Fixed
- **Metal launcher whisper batch size** (#33): the native launcher
  (`scripts/metal/voxint-metal.sh`) now sets `BATCH_SIZE=4`, mirroring the CPU
  image (`Dockerfile.cpu`) it reproduces, instead of silently inheriting the
  GPU/ROCm app default of 16. `batch_size` feeds the `vad_filter=True` batched
  pipeline and is numerics-affecting, so the CT2-CPU tier had been running at a
  batch size no shipped CPU deployment uses; the frozen #33 baseline oracle is
  captured at 4. Pinned by `tests/unit/test_metal_launcher.py`.

## [0.14.0] - 2026-08-16

### Added
- **Run notifications / webhooks** (#12): an opt-in, signed webhook POST when a
  run reaches a **notifiable transition** (`completed` / `failed`), delivered
  **at-least-once** via a transactional outbox so pipeline correctness is never
  held hostage to remote latency or failure. The notification is recorded (new
  `notification_deliveries` table, migration 0015) **in the same transaction as
  the run's state change** — atomic intent, no commit-to-broker loss window, and
  a rolled-back transition takes its row with it. A beat-scheduled sweep
  (`voxint.notify_sweep`) then claims due rows under a lease (`FOR UPDATE ...
  SKIP LOCKED`, safe under overlapping sweeps; a crashed sweep's lease is
  reclaimed) and POSTs each **outside any DB transaction**: deterministic JSON
  body, `X-Voxint-Signature` = HMAC-SHA256 over `timestamp + "." + body`, with
  `X-Voxint-Delivery` (the receiver's dedup key) and `X-Voxint-Timestamp`
  headers. Egress is hardened like URL ingestion — the endpoint must be a
  **public** http/https URL, the host is re-resolved and address-pinned every
  attempt (DNS-rebind safe), redirects are refused, and `HTTP(S)_PROXY` is
  ignored; no URL, secret, or payload is ever logged or stored in an error. A
  `failed` arrival whose run was requeued before delivery is **suppressed**
  (after a short settle delay) rather than sent as stale news; non-2xx/timeout
  retries with capped exponential backoff + jitter, then `dead` after
  `NOTIFY_MAX_ATTEMPTS`. The payload is minimal (`schema_version`, `event`,
  `run_id`, `transition_revision`, `occurred_at`, `delivery_id`) and **omits the
  run's error text** (leak-safe). Settled (`delivered`/`suppressed`) rows are
  purged after `NOTIFY_RETENTION_SECONDS`; `dead` rows are kept for inspection.
  **Off by default** (`NOTIFY_ENABLED=false`) — enabling requires a public
  `NOTIFY_WEBHOOK_URL` and a `NOTIFY_WEBHOOK_SECRET` (≥ 16 chars), and never
  back-fills runs that finished while it was off. Setup, the at-least-once
  contract, and a receiver signature-verification snippet are in
  [docs/operations.md](docs/operations.md).
- **Media retention / garbage collection** (#15): an opt-in, beat-scheduled GC
  sweep (`voxint.gc_sweep`) that reclaims the large normalized-audio
  intermediate (`artifacts/{run_id}/normalized.wav`) for **old terminal runs** —
  unlinking the file and stamping the `audio_artifacts` row (new
  `reclaimed_at`/`reclaimed_bytes`, migration 0014; the row is kept as an audit
  record). File reclamation only: the **source media**, transcript, diarization,
  and the immutable adjudication ledger are always kept, so a reclaimed run
  stays re-processable from its source. Eligibility is `completed`/`cancelled`
  runs untouched for `MEDIA_RETENTION_SECONDS` (archived runs included — archive
  is a visibility flag, orthogonal to reclamation); the tutorial run and any
  file also registered as a source are excluded; missing files are tolerated.
  Rows are claimed oldest-first with `FOR UPDATE ... SKIP LOCKED` (safe under
  overlapping sweeps), one bounded `GC_BATCH_LIMIT` batch per run. **Off by
  default** (`MEDIA_RETENTION_ENABLED=false`) — nothing is reclaimed until an
  operator opts in. The console shows a "Media reclaimed on `<date>`" notice
  instead of the audio link, and `GET /media/{run_id}` returns `410 Gone`. This
  scheduled sweep is complementary to #5's manual **Delete derived audio files**
  action: the sweep keeps the row and stamps `reclaimed_at` (audit), while the
  manual action deletes the `AudioArtifact`/`AudioChunk` rows and files outright.

### Changed
- **Run enrichment assets read attributed speaker names** (#41 follow-up):
  the summary / topics / entity-mention generators now see each transcript
  segment's *adjudicated* speaker — resolved through the same `display_name`
  the review console and export use — instead of the raw `SPEAKER_00`
  diarization label. Generated summaries name real speakers, and because the
  attributed name is part of the hashed source snapshot, re-adjudicating (or
  renaming/merging) a speaker now correctly marks the run's assets **stale** so
  they regenerate. An unadjudicated run's hash is unchanged, so nothing
  regenerates for free (`SOURCE_SCHEMA_VERSION` stays 1); the `run_assets.llm`
  producer/prompt versions bump to 2 for honest provenance. Operator-set
  speaker names are sanitized before entering the prompt (the `: [ ]`
  delimiters are flattened and control/format characters dropped so a name
  cannot forge a line or a field), and the entity-mention instruction now tells
  the model the speaker prefix is not part of the transcript text.

## [0.13.0] — 2026-08-16

### Added
- **Console run cancellation** (#5): a Cancel button on the run detail page for
  any *live* run (`QUEUED` / `RUNNING` / `AWAITING_ADJUDICATION`), backed by a new
  `POST /runs/{id}/cancel` route and `cancel_run` service, an exact-revision
  (CAS) mutation mirroring `/requeue`, so a stale tab 409s. Cancellation is
  **cooperative and pure DB state** (drives the existing `→ CANCELLED`
  transition, publishes nothing): a `QUEUED` run never starts; a `RUNNING` run's
  currently executing stage finishes first (not an immediate kill), then no
  further stages run and the worker stops cleanly at its next CAS; the engine
  now resolves a cancel-lost advance/complete/failure CAS by stopping only when
  the run is confirmed `CANCELLED` (a genuine race still raises) and closes the
  abandoned stage claim `SKIPPED` rather than leaving it "running". Re-cancelling
  an already-cancelled run is an idempotent success. Cancel leaves media and
  partial results in place: **delete/archive is a separate action** (below).
- **Console run archive + derived-media deletion** (#5): finishes the run
  lifecycle beyond append-only. **Soft-archive** hides a terminal run
  (`COMPLETED` / `FAILED` / `CANCELLED`) from `/runs` and the `/review` queue via
  a new nullable `pipeline_runs.archived_at` stamp (migration `0013`) while
  keeping every row (including the append-only adjudication ledger) intact;
  it is fully reversible (**Un-archive**). Archive is operator-visibility
  metadata: last-write-wins, orthogonal to `status`, no CAS/revision bump
  (mirrors operator notes), and idempotent. Live runs refuse archive (cancel
  first); an archived run refuses requeue/claim so a stale tab can't drive a
  hidden run live. `/runs` hides archived by default with a `?archived=1` view,
  and dashboard/`/metrics`/`voxint stats` exclude archived runs. **Delete derived
  audio files** is a separate, destructive, terminal-only action removing only a
  run's own `AudioArtifact`/`AudioChunk` rows and files (post-commit unlink,
  path-confined, idempotent); it **never** touches the shared original
  `MediaItem.source_path`; deleting the shared source is a future refcount-guarded
  action. New routes `POST /runs/{id}/archive`, `/unarchive`, `/media/delete`.

### Changed
- **`LLM_TIMEOUT_SECONDS` default raised 90 s → 300 s**: entity-mention
  extraction on a local ~35B model routinely needs 180–300 s per call, so the
  old default made run assets fail on exactly the self-hosted local-model
  deployments Voxint targets. Cloud endpoints are unaffected on healthy
  connections (they answer in seconds; connection establishment keeps its own
  short cap). New `docs/operations.md` section covers the trade-off (slower
  hung-endpoint detection), proxy-side ceilings the client timeout cannot
  override (an OpenAI-compatible proxy observed 408ing at its own 180 s
  ceiling), and sizing
  `RESEARCH_DEADLINE_SECONDS` for slow local models.

- **Docs state the project's audience and anti-bloat principle**: README,
  CLAUDE.md, CONTRIBUTING, and onboarding now say who Voxint is for:
  individuals and small teams (non-technical researchers, journalists,
  educators) needing locally hosted audio intelligence, and that new
  dependencies, features, and configuration surface must earn their place for
  that audience. Third-party proxy products are no longer named in docs as if
  part of the stack (generic "OpenAI-compatible proxy" phrasing).

### Fixed
- **Research jobs run under their snapshotted LLM timeout**: the worker's
  LLM client was built from live settings while the cancel path's
  stale-RUNNING bound used the job's enqueue-time snapshot, so a settings
  change between enqueue and execution could force-cancel a still-live
  request. Both sides now read the snapshot through one helper (falling back
  to the shared default for pre-0.11 snapshots; the hard-coded `90.0`
  fallbacks in both job modules are gone). The stale bound also now allows
  **two** post-deadline LLM calls (the forced conclude plus its single repair
  attempt) instead of one, matching what the research loop legitimately does.
- **Research-job finalization guards** (#40 follow-up): `research_jobs`
  now carries the same terminal-state protections `run_asset_jobs` shipped
  with in 0.12.0. The success stamp refuses a job with a cancel pending (a
  cancel landing during the final LLM call previously stamped SUCCEEDED and
  kept its drafts); finalization runs under the worker's failure umbrella,
  so a DB error while recording the outcome lands as an honest FAILED row
  instead of a forever-RUNNING job; `_finish` is a guarded active→terminal
  CAS; a force-cancelled row is never overwritten by a late worker verdict,
  and a FAILED verdict racing an operator cancel resolves to CANCELLED. The
  stale-RUNNING force-cancel cutoff now compares DB clock to DB clock
  (`now() - make_interval(...)`), closing the clock-skew window the claim
  path already avoided.

## [0.12.0] - 2026-08-15

### Added
- **In-console run/throughput dashboard** (#13): a new authenticated
  `GET /dashboard` page (first in the top nav) renders the same aggregates
  the Prometheus `/metrics` endpoint and `voxint stats` already expose:
  runs by status, the review backlog (runs awaiting adjudication), per-stage
  average timing and failure counts, roster size, and runs created in the
  window, as a read-only HTML page for a human at the console. It reuses
  `stats_query.collect_stats` verbatim (no new aggregation), so all three
  surfaces agree. An `?since=` query param overrides the default 24 h
  throughput window (same span/ISO-8601 syntax as `voxint stats --since`),
  degrading to 24 h if malformed rather than erroring; the page auto-refreshes
  every 15 s via an htmx fragment poll with no external assets.
- **Run-level enrichment assets** (#41): on-demand LLM-generated **summary**,
  **topics**, and **entity mentions** per run: three independently
  versioned, independently failing whole-document assets, distinct from the
  #37 per-field claim/review model. New `run_enrichment_assets` table
  (migration 0012): success-only, immutable rows (append-only trigger; the
  one permitted mutation is the write-once supersession stamp), keyed
  `(run, kind, generation)` with a monotonic per-kind generation allocated
  under an advisory lock; regenerate supersedes, never edits, and failed
  attempts consume no generation. Every asset records producer + version,
  the exact model, a schema-versioned config snapshot (including whether the
  input was truncated), and a **source-content hash** over the canonical
  serialization of everything the generator read (transcript with
  attribution, #36 metadata, operator notes); staleness is detectable by
  recomputing it, and the console/export flag stale assets explicitly.
  Entity mentions are grounded spans: offsets are never trusted from the
  model. Each quote is located verbatim in its referenced segment
  (word-boundary, case-insensitive fallback), unlocatable or out-of-run
  spans are dropped with recorded diagnostics, and a reply whose every span
  fails grounding fails the job rather than recording an authoritative
  empty. Topic entries reserve null `vocabulary`/`term_id` fields for the
  future domain-pack vocabularies (#11) without a schema change. Durable
  `run_asset_jobs` rows carry queued→running→terminal status per kind (one
  active job per (run, kind), DB-enforced by a partial unique index) with
  deadline-aware cancel; one Celery task per asset, no automatic retries.
  Run detail page gets a "Run assets" block (generate all / per-kind
  regenerate, 3 s polling while active, stale badges, machine-generated
  labeling); the `/runs/{id}/export.json` envelope gains an additive
  `enrichment_assets` key (schema_version stays 1). CLI:
  `voxint enrich assets <run_id> [--kind …]` runs inline without a broker.
  **Off by default**: `ENRICHMENT_RUN_ASSETS_ENABLED=false` requires
  `LLM_ENABLED` (validated at startup, re-checked in the worker). Optional
  post-finalize step `ENRICHMENT_RUN_ASSETS_AUTOGENERATE` enqueues the
  three kinds when a run completes, skipping kinds whose current asset
  already matches the source; best-effort, never fails the run. New
  setting `RUN_ASSETS_MAX_INPUT_CHARS` bounds the rendered prompt document
  (head+tail truncation, recorded on the asset).

## [0.11.0] - 2026-08-15

### Added
- **Web-research speaker profile enrichment** (#40): the `web_researcher`
  producer, an operator-initiated, per-speaker research job driving a
  budgeted LLM tool loop over exactly three tools (#39's `web_search` +
  `read_url`, plus a read-only roster lookup) and quarantining findings as
  #37 drafts for field-by-field review. The loop is a strict-JSON action
  protocol over plain `/chat/completions` (new `HttpLLMClient.chat_json`;
  no provider function-calling, no framework): unknown/malformed replies
  get one repair attempt then the job fails, never a silent "not found".
  Server-side evidence gate: a claim survives only when its source is a
  server-issued id of a page the job actually fetched AND its snippet
  locates verbatim (NFKC+casefold+whitespace-collapsed) in that page's
  kept text; generic values ("the host", "Speaker 2") and non-URL `link`
  values are dropped. Retrieved page text reaches the model only as a
  4k-char untrusted-marked excerpt, and `read_url` accepts only URLs from
  the job's own search results or operator-stored seed URLs; an injected
  page cannot steer fetches. Durable `research_jobs` rows (migration 0011)
  carry status, live counters, and a cooperative cancel flag; the speaker
  card on `/speakers` gets a research block with a budget preview, 3 s
  polling while active, cancel, and per-draft accept/reject (the review
  decision surface now serves bio/affiliation/link; NAME stays on the
  workbench). `found=false` records an authoritative `outcome='none'`
  generation; failures/cancellation record nothing. Per-job idempotency
  (`web_researcher:speaker:{id}:{job_id}`): a rerun is a new superseding
  generation, and there are deliberately no automatic retries or recovery
  sweeps. CLI: `voxint research speaker <id> [--note …]` runs one inline.
  **Off by default**: `ENRICHMENT_WEB_RESEARCH_ENABLED=false` requires
  both `VOXINT_WEB_RESEARCH` and `LLM_ENABLED` (validated at startup,
  re-checked in the worker). New settings: `RESEARCH_MAX_SEARCHES`,
  `RESEARCH_MAX_READS`, `RESEARCH_MAX_ROUNDS`,
  `RESEARCH_MAX_ACTIONS_PER_ROUND`, `RESEARCH_DEADLINE_SECONDS`.
- **Controlled web retrieval** (#39): a new `voxint.research` package:
  `web_search` (pluggable `SearchProvider` protocol, SearxNG built in,
  normalized title/url/snippet results with every result URL pre-filtered
  through the shared egress string gate) and `read_url` (hardened
  single-page fetcher: per-redirect-hop revalidation, DNS answers vetted
  fail-closed, and the connection **pinned** to the vetted address with the
  canonical hostname kept in `Host` + TLS SNI on a fresh client per attempt,
  closing the redirect/rebinding residual for this path; identity-encoding
  only, streamed byte cap, MIME allowlist, stdlib-only text extraction with
  invisible-instruction-character stripping). **Off by default**
  (`VOXINT_WEB_RESEARCH=false`) and fully independent of `LLM_ENABLED`.
  Both tools enforce an atomic per-invocation budget (structured
  `budget_exhausted` outcomes, the contract the future research loop, #40,
  builds on; quota charged only after validation + concurrency slot),
  require a bounded `Attribution`, and log one host-only attribution line
  per outbound request; no error detail or log line ever carries a URL,
  query string, or credential (`final_url` on a successful read is explicit
  provenance, printed query-stripped by the CLI). Operator surface:
  `voxint research search|read` (feature-gated, refuses before any DNS
  when off). New settings: `VOXINT_WEB_RESEARCH`, `WEB_SEARCH_PROVIDER`,
  `WEB_SEARCH_BASE_URL`, `WEB_SEARCH_API_KEY`, `WEB_SEARCH_MAX_RESULTS`,
  `WEB_SEARCH_TIMEOUT_SECONDS`, `WEB_READ_MAX_BYTES`,
  `WEB_READ_MAX_REDIRECTS`, `WEB_READ_TIMEOUT_SECONDS`,
  `WEB_READ_TOTAL_SECONDS`, `WEB_READ_MAX_TEXT_CHARS`.

### Changed
- **CLI `submit`/`fetch`/`requeue` degrade cleanly on a broker outage** (#31):
  the three commands now mirror the HTTP API's commit-before-publish contract.
  Each prints the run id (or `requeued <id>`) to stdout *before* publishing, so
  a Redis outage never costs the operator the id; the publish is wrapped in the
  same `OperationalError`-only guard the API uses (via
  `apply_async(ignore_result=True)`, so a dead broker surfaces as
  `OperationalError` rather than a vague `RuntimeError`), warning on stderr and
  exiting `0` with the run left `QUEUED` for the beat recovery sweep. A genuine
  bug in the publish path still raises. `submit --wait` notes when polling is
  waiting on a deferred enqueue. Previously a broker outage produced an uncaught
  traceback (and, for `submit`, lost the run id from stdout).
- The string-level URL gate moved from `ingest.service` into
  `media.netcheck.parse_http_url` (shared by ingestion and web research,
  one egress policy module to audit); `validate_ingest_url` delegates and
  its error messages are preserved byte-for-byte.
  `assert_host_resolves_public` now wraps a new `resolve_public_addresses`
  core that returns the vetted address set for connection pinning.

### Fixed
- **Whisper startup is offline-clean** (#30): the whisper images set
  `HF_HUB_OFFLINE=1` and pin `WHISPER_REVISION` to the baked snapshot, so
  the service no longer makes an unadvertised Hugging Face revision check
  at startup (which stalled/failed on air-gapped hosts and could have
  re-downloaded a different revision than the one baked). The CUDA image's
  build-time bake is now sha-pinned like the CPU/ROCm flavors, the metal
  launcher exports the same offline guard, and a contract test holds all
  four deployment flavors to one revision. Documented in
  `docs/operations.md` ("Offline / air-gapped hosts").

## [0.10.0] - 2026-08-15

The enrichment foundation (#36, #37, #38): write-once source-metadata
capture at acquisition, the reviewable evidence-backed draft schema, and
the first producers: offline + optional-LLM speaker-name suggestions with
their adjudication-workbench review surface. Plus the macOS arm64 CI lane
(partial Gate M automation, #34), metal-tier log rotation, and metal parity
bounds ratcheted from Gate M evidence.

### Added
- **Offline speaker-name suggestions** (#38): a new `names.offline`
  enrichment producer mines evidence-backed name candidates from stored
  source metadata (title/description/channel/tags) and transcript text
  (self- and host-introductions), fully offline, deterministic regex with
  explicit false-positive guards, no LLM and no network. Cluster-level
  (per-diarization-label) suggestions come **only** from self-introductions
  inside that cluster's own segments; everything else stays a run-level
  hint, so a title mention can never masquerade as cluster identity.
  Scoring is explainable (max pattern reliability + small corroboration/
  diversity/domain-pack-seed bonuses, capped at 0.95) with the full
  component breakdown stored per candidate. Reruns supersede cleanly via an
  input-signature idempotency key. Invoke with `voxint enrich names
  <run_id>` (or `--all-completed`), or from the workbench. New settings:
  `ENRICHMENT_NAMES_ENABLED` (default true), `ENRICHMENT_NAMES_LLM_ENABLED`
  (default false, requires `LLM_ENABLED`).
- **Name-suggestion review surface** (#38): the adjudication workbench now
  shows a "Name hints" block (run-level) and per-label "Self-introduced
  (unverified)" suggestions, each with its evidence snippet and score.
  Operators can trigger/re-run the sweep (claim-gated, synchronous) and
  accept or reject each suggestion: accepting records a profile-review
  decision only, never a speaker, assignment, or adjudication ruling. An
  accepted per-label suggestion prefills the Enroll input (editable, never
  auto-submitted). Rerun duplicates group under their decided history
  instead of re-presenting as new.
- **Additive LLM name pass** (#38): a second producer, `names.llm`, mines
  name hints from the transcript via the configured enhancement LLM
  (`voxint enrich names <run_id> --llm`; CLI-only, never in the console
  request path). Strictly additive: its own supersession lineage, and the
  offline path never depends on it. Model output obeys the same evidence
  discipline: a hint survives only when the name is located verbatim in a
  real segment (in the hinted label's own segments for self-intros, which
  alone may become cluster-level claims); unlocatable names are dropped.
  Fixed uncalibrated score 0.5. Gated behind `ENRICHMENT_NAMES_LLM_ENABLED`
  + `LLM_ENABLED`; an LLM failure aborts rather than recording a false
  authoritative "found nothing".
- **Enrichment draft schema** (#37): machine-derived claims about speakers
  and runs now live as reviewable, evidence-backed drafts. Four new tables
  (migration 0010): `enrichment_producer_runs` (one row per completed
  producer invocation: scope, covered fields, monotonic generation, and an
  explicit `outcome='none'` when a producer looked and found nothing),
  immutable `enrichment_candidates` (claim field/value, producer-local score
  with visible components, write-once supersession stamp), normalized
  `enrichment_candidate_evidence` (one claim can cite a metadata field,
  transcript segments, and several URLs together), and the append-only
  `profile_review_decisions` human trail, deliberately separate from the
  attribution ledger. Review state is derived at read time (decision >
  superseded > proposed), never stored. Single sanctioned writers in
  `voxint.enrichment` (atomic per-scope finalization under an advisory lock;
  terminal accept/reject with idempotent replay). Invariant unchanged: drafts
  are suggestions *about* identity: accepting a name claim never touches
  `speakers.display_name`, machine proposals, or attribution resolution.
  Schema + writer layer only; the producers (#38, #40) and their console
  surface come separately.
- **Source media metadata capture** (#36, schema slice): new write-once
  `media_source_metadata` table (1:1 with `media_items`) holding normalized
  extractor context: title, uploader/channel (+URLs), description, upload
  date, source-claimed duration, tags, canonical URL, extractor
  name/version, plus a bounded, allowlisted, schema-versioned `raw` JSONB
  subset and the acquisition timestamp. Metadata is context, not identity:
  a MediaItem is per-acquisition, so a snapshot can never rewrite the
  context a past adjudication was made against. New nullable
  `pipeline_runs.operator_notes` keeps human input structurally apart from
  scraped metadata. Migration 0009 (additive, clean downgrade).
- **Metadata capture at acquisition** (#36): the yt-dlp invocation now also
  writes a clean info-JSON (`--write-info-json --clean-info-json
  --no-write-playlist-metafiles`, typed `infojson:` output, same invocation,
  no extra network exposure); ACQUIRE sanitizes it through a strict allowlist
  (secret-bearing keys: `formats`, `http_headers`, `cookies`, signed URLs
  are never copied), publishes a hash-addressed replay sidecar before the
  media file, and inserts the write-once snapshot row. Best-effort: bad
  metadata logs a warning, never fails an acquisition.
- **Operator notes + surfacing** (#36): run detail gains a Source-metadata
  section and an editable Operator-notes form (`POST /runs/{id}/notes`,
  CSRF-gated, 10K-char cap); the runs browser shows the source title with
  media-path fallback; new `GET /runs/{id}/export.json` returns a versioned
  envelope (run + source_metadata + operator_notes + segments) while the
  pinned bare-array `/review/{id}/export.json` contract stays frozen.
- **macOS arm64 CI lane** (`.github/workflows/metal-lane.yml`, issue #34):
  nightly + manual-dispatch partial Gate M automation on `macos-15` runners
  (real MPS): launcher unit tests on real macOS, then the whisper/pyannote/
  titanet parity modules from the launcher's own sha-verified per-service
  venvs, with provenance-keyed weight caches, an MPS tensor-op probe, and a
  junit guard that fails the lane if an expected module green-boards
  fully-skipped. Maintainer Gate M (per-chip verdict refreshes) is
  unchanged; this catches regressions between refreshes.
- **Metal-tier log rotation** (metal review follow-up): `voxint-metal.sh up`
  now installs a daily launchd job (`com.voxint.metal.logrotate`) that
  copy-truncates any service log over 50 MB to a timestamped archive,
  keeping the newest 5; launchd's `StandardOutPath` never rotates and
  `KeepAlive` keeps services up for months. `VOXINT_METAL_LOG_MAX_MB` /
  `VOXINT_METAL_LOG_ARCHIVES` override; new `rotate-logs` subcommand runs a
  pass by hand; `logs -f` now follows with `tail -F`.
- **Parity references now record the exact request payloads** they were
  generated with (`tools/generate_parity_references.py` writes a
  `meta.request` block per reference): parity lanes replay hardcoded
  "service-default" params, and a regenerated reference could otherwise pair
  silently with different params than the lanes measure. Takes effect on the
  next reference regeneration; committed references predate the field
  (metal review follow-up).
- **Contract test binding `compose.metal.yaml` ports to the metal launcher's
  `service_port()`**: the overlay's `host.docker.internal:<port>` URLs and
  the native services' bind ports were each pinned to their own literals;
  a port moved in only one place would have kept both tests green while the
  worker called a dead port (metal review follow-up).

### Changed
- **Metal parity bounds ratcheted from Gate M evidence** (slice 9, panel
  consult recorded in the commit): pyannote boundary drift ≤ 0.10 s (was
  0.25), agreement vs reference ≥ 0.97 (was 0.95), MPS-vs-CPU ≥ 0.995 (was
  0.99); whisper transcript similarity ≥ 0.96 (was 0.95), confidence drift
  ≤ 0.05 (was 0.15). Repeat/segment/count bounds unchanged. Three deferred
  decisions closed as measured no-ops: CoreML EP default stays off (no
  speedup), no metal timeout factor (0.38–0.45× RT transcribe fits
  GPU-class budgets), no committed metal reference oracle (re-affirmed).
  See docs/gpu-contracts.md metal verdict table.

## [0.9.0] - 2026-08-14

The Apple Silicon "metal" compute tier (#1): native macOS model services
under launchd with diarization on the Apple GPU via torch-MPS, measured
against the committed CUDA references (maintainer Gate M PASS on an M1 Pro),
plus the tier-independent device-control contracts (`DIARIZER_DEVICE`,
`TITANET_ORT_PROVIDERS`) and multi-model review hardening of the launcher.

### Added
- **Apple Silicon "metal" compute tier**: the core stack stays in Docker
  (`compose.metal.yaml` rewires api/worker to `host.docker.internal`) while
  the three model services run natively on macOS, set up, sha-verified,
  and supervised under launchd by the new `scripts/metal/voxint-metal.sh`
  (`setup / up / down / status / logs / doctor / run --foreground`), so
  diarization runs on the Apple GPU via torch-MPS (~5× native-CPU
  diarization measured on an M1 Pro, identical outputs; transcription stays
  on host CPU in v1 and remains the bottleneck). The installer grew an `[M]`
  option (default on Apple Silicon) that starts the core and hands off.
  New device-control contracts, both tier-independent:
  `DIARIZER_DEVICE=auto|cuda|mps|cpu` (a forced device must pass the sanity
  probe or the service refuses to start) and `TITANET_ORT_PROVIDERS`
  (requested ONNX EPs must be verifiably active, no silent fallback
  anywhere). Metal parity lanes gate against the committed CUDA references
  (no metal oracle by design): `tests/parity/test_pyannote_metal.py`,
  `test_whisper_metal.py`, `VOXINT_PARITY_ORT_PROVIDERS` threading for the
  titanet 3-level gate, and `tools/generate_parity_references.py --tier
  metal`. Maintainer-run Gate M documented in the release process.

### Fixed
- **Metal tier review hardening** (pre-landing multi-model review): the
  installer's metal handoff no longer claims model services "were started";
  whisper's runtime load is pinned to the same HF revision setup downloads
  (`WHISPER_REVISION`, launcher-set; unset keeps image behavior), the local
  manifest records that revision and excludes HF cache bookkeeping, and a
  stale/corrupt cache is cleared before re-download instead of being
  re-blessed; `voxint-metal.sh up` preflights venvs/weights/config and waits
  out the launchd bootout-vs-bootstrap race instead of crash-looping under
  KeepAlive; `VOXINT_METAL_DIARIZER_DEVICE` accepts only `mps`/`cpu` (`auto`
  would re-open silent CPU fallback); vendored-config generation escapes sed
  metacharacters in the destination path and fails explicitly under
  `PYTHONOPTIMIZE`; doctor now verifies whisper weights; sha verifiers
  distinguish unreadable provenance from weight mismatch; metal parity lanes
  fail closed on empty diarizations, pin the whisper snapshot, and shed
  ambient `TITANET_ORT_PROVIDERS` / `PYANNOTE_*` env.
- **Metal launcher `.env` reading**: `voxint-metal.sh` read `MEDIA_ROOT`
  verbatim from `.env`, but the installer writes it single-quoted; the
  launcher hard-failed on every installer-generated file ("does not resolve
  to an existing directory"). Values are now normalized exactly like the
  installer reads them back (strip CR, blanks, and one matched pair of
  quotes), matching what Compose interpolation passes to the containers.

## [0.8.0] - 2026-08-14

Runs search (#8) plus CLI/observability ergonomics (#25, #32): the runs
browser gains transcript full-text search and facets, and the CLI grows
export, list, doctor, stats, and watch alongside a Prometheus `/metrics`
endpoint. Also carries the cross-platform / dev-experience hardening bundle
(#26, #27, #28, #29).

### Added
- **Search on the runs browser** (`/runs`, #8): transcript full-text search
  (`q=`, Postgres `websearch_to_tsquery` syntax: quotes, `-word`, `OR`) with
  a highlighted first-hit snippet per run, a speaker facet (runs whose
  read-time attribution (human ruling or grounded cosine, merge tombstones
  canonicalized) is the selected speaker; archived speakers stay listed,
  marked), a source-path substring facet, and UTC date-range bounds. All
  facets AND-compose with the existing status/review filters and keyset
  pagination. Backed by two GIN expression indexes (migration 0008) over
  `raw_text` AND `enhanced_text` separately; enhancement never makes the raw
  rendering of a term unfindable, and vice versa. Dictionary is `english`
  (stemming recall); a stopword-only query matches nothing by design. Results
  stay newest-first (no relevance ranking pre-1.0) and the search document
  is one segment (terms split across segments of a run don't AND-match).
- **Structured & subtitle transcript exports.** The review console now offers
  SubRip (`.srt`), WebVTT (`.vtt`), JSON, and diarization RTTM (`.rttm`)
  alongside the existing plain-text export, at
  `GET /review/{run_id}/export.{srt,vtt,json,rttm}` (all accept `?text=raw|
  enhanced`, default enhanced; RTTM carries raw diarization labels). SRT/VTT/
  JSON/TXT share one set of pure formatters (`voxint.export`) with the CLI, so a
  downloaded file and a piped export are byte-identical.
- **`voxint export <run_id> --format srt|vtt|json|rttm|txt`**: headless
  transcript export to stdout or `-o PATH` (refuses to overwrite without
  `--force`); `--text raw|enhanced` selects the transcript variant.
- **`voxint list`**: a CLI run browser (newest first) mirroring the `/runs`
  query, with `--status`, `--limit` (1–500, default `runs_page_size`), and
  `--json`.
- **`voxint doctor`**: read-only preflight diagnostics. Postgres, Redis, and
  each model service's `/healthz` (reporting the compute `device`) are hard
  checks (exit 1 if any is down); the Hugging Face token and LLM endpoint are
  advisory (reported, never fail the exit). Credentials are never printed.
- **`voxint stats`**: an aggregate, read-only system summary. Run counts by
  status, failed stage attempts by stage, average per-stage duration (over
  finished attempts), roster size, and runs created in a window (`--since`,
  accepting `<n>h`/`<n>d`/ISO-8601, default 24h). `--json` emits a stable object.
- **`GET /metrics`**: a Prometheus text-exposition endpoint (format 0.0.4)
  built on the same query module, on the authenticated router (scrape it with
  `basic_auth`, keeping the "everything but `/healthz` authenticates" invariant).
  Every `RunStatus`/`Stage` series is zero-filled so a series never disappears
  between scrapes; the one windowed gauge bakes its window into its name
  (`voxint_runs_created_24h`).
- **`voxint watch <run_id>`**: follow a run until it stops advancing, with a
  live progress line on stderr. Exit codes: `0` completed, `1` failed/cancelled,
  `2` missing run, `3` awaiting adjudication (paused, needs a human ruling),
  `124` timeout. `--interval` (default 2s) and `--timeout` (default 3600s) tune
  the poll.
- **`voxint submit --wait`**: enqueue, then follow the new run to a stop state
  with the same poll loop and exit codes (the run id stays alone on stdout;
  progress goes to stderr).

### Fixed
- **macOS/BSD media-download teardown raised the wrong error (#26).** On a
  download timeout, if the yt-dlp process-group leader had already been reaped
  and the survivor was a zombie reparented to launchd, `killpg` returns `EPERM`
  (not `ESRCH`); the raw `PermissionError` escaped the teardown and replaced the
  intended redacted `AcquisitionError`. Both teardown signals now suppress
  `PermissionError` alongside `ProcessLookupError`. (Linux returns `ESRCH`, so
  this was macOS/BSD-only; validated by a new monkeypatched unit test; a real
  Mac run is the true confirmation.)
- **Installer could offer the busy port as its own "alternate" (#27).** On
  macOS/BSD a listener with a full accept-backlog refuses further connects, so
  the `/dev/tcp` probe can misread a bound port as free; `resolve_port` then
  re-scanned starting *at* the known-busy default and could suggest it right
  back. It now searches strictly above the busy port, so the offered alternate
  is always distinct. The probe stays advisory (Compose remains the collision
  authority); its residual limitation is now documented in-script.

### Changed
- **Fresh `uv sync --extra dev` checkout is green again (#28).** The loopback
  default-credentials test is now hermetic (`_env_file=None`, so an on-disk
  `.env` can't override the code default), and the two librosa-dependent mel
  contract tests `importorskip("librosa")` (it ships only in the `parity`
  extra); they still run in the parity lane, and no assertions were weakened.
- **Documented the CPU-tier host-RAM floor (#29).** The CPU tier holds the
  models in RAM (~6 GiB idle; whisper alone ~4.8 GiB) and needs **≥ 8 GB**
  available to the container host (on Docker Desktop the VM's memory limit, not
  the physical machine) or services are OOM-killed with an opaque exit. Noted
  in `docs/operations.md`, `docs/onboarding.md`, and the installer's tier prompt.

## [0.7.0] - 2026-08-14

Speaker roster management (#7): the roster is no longer write-only.

### Added
- **Speaker roster page** (`/speakers`, #7): view every enrolled speaker with
  its enrollment provenance, machine-proposal count, and a deterministic
  voiceprint strip derived from its own centroid. Curation actions: rename,
  merge duplicates, archive/restore, and remove a bad enrollment embedding,
  all without ever rewriting the append-only decision ledger. Merges keep the
  source speaker as a tombstone (`merged_into_id`, migration 0007) and readers
  canonicalize at read time, so historical rulings render under the merge
  target while the ledger rows stay byte-identical.

### Changed
- Speaker matching, the workbench assign dropdown, and the decide route now
  consider **active** speakers only; merged and archived speakers stop
  attracting proposals and decisions (archiving also removes the speaker's
  machine proposals; restore does not resurrect them).

### Fixed
- Enrollment replay now validates against durable provenance (run, label,
  operator) instead of the current display name, so renaming a speaker can no
  longer make a replayed enrollment POST falsely conflict.

## [0.6.0] - 2026-08-14

Token-free onboarding: the diarization weights are vendored (#24). No
numerical changes; vendored-vs-HF diarization verified byte-identical.

### Changed
- **No Hugging Face account or token needed** (#24): the
  `speaker-diarization-3.1` pipeline weights are now vendored into the
  pyannote images, sha256-pinned from the standing `pyannote-models-v1`
  asset release (`services/pyannote/models/provenance.json`; segmentation-3.0
  MIT, WeSpeaker embedding CC-BY-4.0, redistributed with attribution) and
  loaded offline by default. Vendored-vs-HF parity verified byte-identical on
  the parity clip. `HF_TOKEN` is demoted to an optional override for a custom
  `DIARIZER_MODEL_NAME`; the installer no longer prompts for a token, the
  compute overlays start without one (the `${HF_TOKEN:?}` guard is gone), the
  setup wizard drops its token row, and pyannote's CI smoke runs
  unconditionally (the secret-absent SKIP lane is deleted).
- `DIARIZER_MODEL_NAME` is now interpolated from `.env` by every compute
  overlay, so the documented override works without editing compose files.

### Fixed
- **CUDA pyannote image**: `setuptools` pinned `>=70,<81` with a build-time
  `pkg_resources` canary; the unpinned upgrade would have shipped an image
  that crashes on boot at the next rebuild (setuptools 81 removed
  `pkg_resources`, which pyannote.database imports; the CPU flavor already
  carried the pin).
- `/healthz` keeps reporting the canonical `pyannote/speaker-diarization-3.1`
  identity for the vendored default; an explicitly configured
  `VOXINT_VENDORED_PIPELINE` that does not exist now fails fast instead of
  silently degrading to a gated network fetch.

## [0.5.1] - 2026-08-14

Burst-load resilience patch (#23). No inference or contract
changes.

### Fixed
- **All long-running services now carry `restart: unless-stopped`** (core
  stack + every model-service overlay; `migrate` keeps its deliberate
  `"no"`): a transient model-service crash self-heals instead of staying
  down until a human runs `up -d` (#23).
- **Connection failures to a model service now say what they mean**: when
  the service DNS name stops resolving or the connection is refused (inside
  the compose network this almost always means the container is
  down), the worker's ledger error names the service host and says the
  service is likely down or restarting (pointing compose deployments at
  `docker compose ps`), instead of surfacing a raw resolver error that
  reads as a network problem (#23).

## [0.5.0] - 2026-08-14

AMD-GPU acceleration for ASR (#4). The ROCm tier is a hybrid: whisper runs on
the AMD GPU, pyannote/titanet stay on CPU. No numerical changes to existing
flavors.

### Added
- **whisper `-rocm` image** (`services/whisper/Dockerfile.rocm`, amd64):
  same faster-whisper 1.2.1 / CTranslate2 engine and code path as CUDA:
  the CTranslate2 4.8.1 **ROCm build** (GitHub release wheel, sha256-pinned;
  not on PyPI) on ubuntu:24.04 with the minimal measured ROCm 7.0.2
  runtime-library set. Torch-free (the 1.2.x Silero VAD is
  onnxruntime-based). Measured on RDNA4 (RX 9060 XT, gfx1200): warm
  transcription 4.8× the CPU baseline on the parity corpus clip (this
  image's smoke measured faster still); host needs only the amdgpu kernel
  driver.
- **`compose.rocm.yaml` overlay**: whisper on the GPU (`/dev/kfd` +
  `/dev/dri` passthrough + the owning host gid via `VOXINT_RENDER_GID`;
  no `video` group, no `seccomp:unconfined`; both verified unnecessary on
  real hardware), pyannote/titanet on the `-cpu` images,
  `COMPUTE_TIER=rocm` timing profile. Pin-parity contract test now covers it.
- **Installer AMD tier**: `[A]` in the compute-tier prompt (suggested when
  `/dev/kfd` exists and no NVIDIA driver is), records
  `VOXINT_COMPOSE_TIER=rocm` and auto-detects + records the gid owning
  `/dev/kfd` in `.env` (`VOXINT_RENDER_GID`); kept-`.env` re-runs re-detect
  and refresh it (the gid is per-host).
- **Honest `/healthz` device reporting without torch**: the CT2 ROCm build
  masquerades as CUDA and the `-rocm` image carries no torch, so
  `resolve_device_name` now also detects the loaded HIP runtime
  (`libamdhip64` in `/proc/self/maps`) and reports `device: "rocm"`.
- **`release.yml` `publish-whisper-rocm` lane**: build-only in CI (GitHub
  has no AMD-GPU runners); the real-GPU inference gate is a maintainer step
  on AMD hardware before tagging (Gate R, `docs/release-process.md`).
- Docs: `docs/operations.md` ROCm-tier section (incl. why pyannote/titanet
  stay CPU: MIOpen convolutions fail on current AMD consumer GPUs in both
  shipping torch-ROCm wheel lines), README AMD callout,
  `docs/gpu-contracts.md` device-reporting note, whisper README image matrix.

### Changed
- `cleanup_memory` in the whisper service tolerates a torch-free image
  (guarded import; CT2 manages its own device memory).

## [0.4.1] - 2026-08-14

Onboarding patch: closes the v0.4.0 first-run traps (#17–#22). No model
service, pipeline, or numerical changes; images rebuild, numerics untouched.

### Added
- **Installer compute-tier selection** (GPU / CPU / none-for-now; suggests GPU
  when `nvidia-smi` is present), remembered in `.env` as
  `VOXINT_COMPOSE_TIER`; one helper owns the tier → compose-file mapping and
  every installer Compose invocation goes through it, so the pull/up/status
  commands can never disagree about the active overlay (#18).
- **Installer Hugging Face token prompt** (hidden input, both pyannote gate
  URLs explained) with an advisory two-stage check: token validity, then
  access to each gated repo (terms accepted). Warnings only, never blocks;
  the token reaches curl via stdin config, never argv. Skipping the token
  records the tier but starts the core stack only (both compute overlays
  refuse to interpolate without `HF_TOKEN`), and the completion notice spells
  out the three steps to finish (#17).
- **Setup wizard SERVICES step**: a Hugging Face token presence row (never
  the value) and guidance covering both compute tiers, not just GPU (#17, #18).
- **Run page**: static guidance when a run failed at a model stage: start a
  compute tier, wait for it, requeue (#18).
- **`docs/interpreting-diarization.md`**: segment labels are a
  dominant-overlap projection and can under-report speakers (the turn ledger
  is the source of truth); short clips can over-split; honest note that
  `min/max_speakers` is service-API-only today (#22).
- **Offline installer test suite** (33 tests) driving the
  `VOXINT_INSTALL_LIB=1` seam with fake `docker`/`curl` on PATH: tier
  mapping, port-collision handling (#21), `.env` render/update/backup/0600,
  dotenv normalization, and secret non-disclosure (token never in
  stdout/stderr/argv).

### Fixed
- **Installer port-collision prompts were invisible**: after the first
  detected collision, a stray `exec … 2>/dev/null` in `port_in_use`
  permanently redirected the whole script's stderr to /dev/null; every
  later prompt and message vanished (#21).
- Installer re-runs that switch tier (or defer on a removed token) no longer
  strand the previous overlay's model containers
  (`docker compose up --remove-orphans`).
- Kept-`.env` reads now match Compose dotenv semantics (trailing CR,
  surrounding blanks, matched single/double quotes); a hand-edited
  `HF_TOKEN=""` no longer defeats the skip-token deferral or produces a
  false "token rejected" warning.
- `.env` backups are forced to mode 0600 (`cp -p` had preserved a loose
  source mode).
- The false "a run simply waits on any service it needs" claim (wizard +
  onboarding docs) replaced with the real behavior: retry with backoff
  (about five attempts over roughly an hour and a half), then FAILED, then
  requeue from the run's page.

### Changed
- README leads non-NVIDIA users to the CPU tier from the top of the
  quickstart ("No NVIDIA GPU? Start here too"), and the CPU section is a
  linkable heading (#20).
- README and `voxint score --help` now state exactly what the harness
  scores: speaker attribution (name accuracy / agreement / ensemble); ASR
  accuracy / WER is out of scope (#19).
- The installer handoff is honest about readiness: only the API is
  health-checked; model services are reported as *started* with the ps
  command to check them, not "enabled".

## [0.4.0] - 2026-08-13

CPU tier: run Voxint's full pipeline with **no NVIDIA GPU**, on plain
servers, AMD boxes, and Apple Silicon (Docker Desktop). Closes the
container-path ask of #1 (Apple Silicon) and #4 (AMD); accelerated ROCm and
native-Metal tiers are tracked separately.

### Added
- **Multi-arch (amd64 + arm64) `-cpu` image flavor** for all three model
  services (`voxint-{whisper,pyannote,titanet}:X.Y.Z-cpu`), built natively
  per arch (no QEMU) and merged into one manifest list. Unsuffixed
  model-service tags remain CUDA, unchanged.
- **`compose.cpu.yaml`** overlay: the whole stack on CPU with
  `docker compose -f compose.yaml -f compose.cpu.yaml up -d`. Sets
  `COMPUTE_TIER=cpu`, which scales default inference timeouts, stage leases,
  and the Celery visibility horizon so slow-but-healthy CPU runs are never
  reclaimed as hung. Honest expectation: long recordings take **hours** on
  CPU.
- **titanet ONNX Runtime engine in the shipped `-cpu` image**
  (`EMBED_ENGINE=onnx`, torch- and NeMo-free): same embedding space id
  (`titanet-large-v1`), kept on the measured three-level parity gate
  (mel / vector / decision) against the CUDA engine; verdict recorded in
  `docs/gpu-contracts.md`. The build verifies the model artifact's sha256
  against the committed export provenance; the ~100 MB `.onnx` ships via the
  standing `titanet-onnx-v1` model-asset release, never git.
- **pyannote device cascade** (`cuda → mps → cpu`) with a real-tensor-op
  startup probe that checks device output against a CPU reference: a backend
  that computes silently-wrong results (the historical MPS failure mode) is
  rejected, not trusted. MPS is inert in containers; the branch serves the
  future Apple host-process path.
- **Release gates in `release.yml`**: the strict titanet parity harness
  (`VOXINT_PARITY_REQUIRED=1`) runs on amd64 **and** arm64 runners and blocks
  the multi-arch builds; the per-arch smoke (`tools/smoke_cpu_services.py`)
  runs against the **untagged digest images before any tag exists** and
  requires healthz identity fields, a real corpus transcription, and a
  titanet embedding within cosine 0.999 of the committed CUDA reference
  (pyannote's smoke needs an `HF_TOKEN` secret and SKIPs explicitly when
  absent); tags are only ever attached to smoke-passed digests, and each
  manifest list is verified to expose exactly amd64+arm64.

### Changed
- **The app image (`voxint`) is now multi-arch** (amd64 + arm64).
- The whisper CUDA image's engine, pins, and behavior are untouched; the
  `-cpu` flavor runs the same faster-whisper/CTranslate2 int8 engine with
  CPU-appropriate defaults (`BATCH_SIZE=4`).

## [0.3.0] - 2026-08-13

Non-technical onboarding: get from a fresh clone to a first successful,
adjudicated run without editing config by hand.

### Added
- **Guided installer** (`scripts/install.sh`): one command that takes a fresh
  clone to a running core stack for non-technical users. Prompts only for an
  admin password and a media folder; auto-generates `CSRF_SECRET`, detects
  host-port collisions and offers a free alternate, and renders `.env` from
  `.env.example` (never overwriting an existing one without a timestamped
  backup). Preflights Docker + the Compose plugin (≥ 2.24), pulls the pinned
  images, starts the stack, and polls the API container's healthcheck, then
  prints the console URL and states plainly that the core stack is the control
  plane only (audio processing needs the GPU overlay). Bash 3.2+, macOS/Linux,
  no runtime dependency beyond Docker. (#2)
- **First-run setup wizard** (`/setup`): a guided, operator-authenticated flow
  that takes a fresh install to a configured state. Choose media folders (with
  an optional bounded scan that previews and batch-registers existing media),
  define a domain vocabulary that feeds both the Whisper `initial_prompt` and
  the LLM enhancement context, toggle optional LLM transcript enhancement, and
  check GPU service health: core-only when the GPU overlay is absent,
  with no silent fallback. Preferences apply per run with no worker restart. An
  onboarding gate holds the console at the wizard until setup is finished, then
  releases the full app. Backed by an `app_settings` singleton (alembic
  revision 0006). (#3)
- **Guided 3-speaker tutorial**: a bundled synthetic 3-speaker sample and an
  idempotent `voxint tutorial seed` command that stages a ready-to-adjudicate
  run. Server-rendered `?tutorial=<step>` banners walk through the
  run → review → transcript flow on the real console pages, and a new
  **Settings** page re-runs the wizard and starts, replays (non-destructively),
  or completes the tutorial. (#3)

## [0.2.0] - 2026-08-12

### Added
- **Browser console** served from the same FastAPI app: a keyset-paged `/runs`
  execution-history browser (orthogonal `status=` / `review=` filters), a
  `/runs/{id}` run-detail page with the per-stage attempt ledger, and a
  resolver-attributed transcript view (`raw`/`enhanced`).
- **File upload** (`POST /submit`): bounded, streamed enforcement of
  `UPLOAD_MAX_BYTES` (default 5 GiB); each upload lands under a server-issued,
  uuid-namespaced immutable path, with idempotent form replay.
- **URL ingestion** via yt-dlp: `voxint fetch <url>` and `POST /fetch` register a
  `MediaItem.source_url` and enqueue a run. A new **ACQUIRE** stage
  (`STAGE_ORDER[0]`, a no-op for local/uploaded media) downloads it on the worker
  (alembic revision 0005 adds `source_url` and the `acquire` stage). Toggle with
  `YTDLP_ENABLED` (default on).
- **CAS requeue route** (`POST /runs/{id}/requeue`): the browser equivalent of
  `voxint requeue`, guarded by exact-revision compare-and-swap.

### Security
- **Two-gate SSRF model** for URL ingestion: a string-level check at submit and a
  host re-resolution check in the worker before download, sharing one
  public-address rule that unwraps IPv4-in-IPv6 embeddings and rejects site-local.
  Documented as authenticated admin egress with a residual that needs network
  policy (see `docs/architecture.md`).
- **yt-dlp lockdown**: `--no-config`, `--no-plugin-dirs`, `--no-exec`,
  `--no-playlist --max-downloads 1`, a size cap, hard wall-clock timeouts, and
  explicit proxy handling; proxy/cookies are treated as credentials and scrubbed
  from surfaced errors.
- **CSRF** on the mutation forms (`POST /submit`, `/fetch`, `/runs/{id}/requeue`,
  and `POST /review/{id}/claim`): a stateless, action-bound HMAC token keyed by a
  dedicated `CSRF_SECRET`.

### Changed
- Submission mutations are **durable-first**: the run is committed before the
  Celery task is published, so a broker outage leaves the run `QUEUED` (never
  `FAILED`) for the recovery sweep instead of failing the request.

## [0.1.0] - 2026-08-12

First public release.

### Added
- End-to-end pipeline: preprocess → transcribe (faster-whisper) + diarize
  (pyannote) + embed (TitaNet) → optional LLM transcript enhancement →
  speaker matching → human adjudication.
- Compare-and-swap run/stage state machine in Postgres with leased stage
  claims, retry budgets, and a beat-scheduled crash-recovery sweep.
- Adjudication web console (review queue, guarded slot workbench,
  decision-resolved transcript export) served as Jinja + htmx from the API.
- pgvector cosine speaker matching with a strict *named ≠ grounded* invariant;
  machine proposals kept separate from human rulings (append-only ledger).
- Scoring harness `voxint score` (name-accuracy, acoustic agreement, ensemble
  fusion): DB-free, installable standalone from PyPI; synthetic walkthrough
  under `examples/`.
- Three GPU model services with frozen v1 HTTP contracts
  (`/v1/transcribe`, `/v1/diarize`, `/v1/embed`).
- Compose-first deployment: pinned GHCR release images by default,
  build-from-source overlays (`compose.build.yaml`, `compose.gpu.build.yaml`),
  one-shot `migrate` gate, swappable domain pack.

[Unreleased]: https://github.com/bengizmo/voxint/compare/v0.26.0...HEAD
[0.27.0]: https://github.com/bengizmo/voxint/compare/v0.26.0...v0.27.0
[0.26.0]: https://github.com/bengizmo/voxint/compare/v0.24.0...v0.26.0
[0.24.0]: https://github.com/bengizmo/voxint/compare/v0.23.1...v0.24.0
[0.23.1]: https://github.com/bengizmo/voxint/compare/v0.23.0...v0.23.1
[0.23.0]: https://github.com/bengizmo/voxint/compare/v0.22.1...v0.23.0
[0.22.1]: https://github.com/bengizmo/voxint/compare/v0.22.0...v0.22.1
[0.22.0]: https://github.com/bengizmo/voxint/compare/v0.21.0...v0.22.0
[0.21.0]: https://github.com/bengizmo/voxint/compare/v0.20.0...v0.21.0
[0.20.0]: https://github.com/bengizmo/voxint/compare/v0.19.0...v0.20.0
[0.19.0]: https://github.com/bengizmo/voxint/compare/v0.18.0...v0.19.0
[0.18.0]: https://github.com/bengizmo/voxint/compare/v0.17.0...v0.18.0
[0.17.0]: https://github.com/bengizmo/voxint/compare/v0.16.0...v0.17.0
[0.16.0]: https://github.com/bengizmo/voxint/compare/v0.15.0...v0.16.0
[0.15.0]: https://github.com/bengizmo/voxint/compare/v0.14.0...v0.15.0
[0.14.0]: https://github.com/bengizmo/voxint/compare/v0.13.0...v0.14.0
[0.13.0]: https://github.com/bengizmo/voxint/compare/v0.12.0...v0.13.0
[0.12.0]: https://github.com/bengizmo/voxint/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/bengizmo/voxint/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/bengizmo/voxint/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/bengizmo/voxint/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/bengizmo/voxint/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/bengizmo/voxint/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/bengizmo/voxint/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/bengizmo/voxint/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/bengizmo/voxint/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/bengizmo/voxint/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/bengizmo/voxint/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/bengizmo/voxint/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/bengizmo/voxint/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/bengizmo/voxint/releases/tag/v0.1.0
