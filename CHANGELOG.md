# Changelog

All notable changes to Voxint. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning: [SemVer](https://semver.org/) (0.x; expect breaking changes between minors).

## [Unreleased]

### Added
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

[Unreleased]: https://github.com/bengizmo/voxint/compare/v0.14.0...HEAD
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
