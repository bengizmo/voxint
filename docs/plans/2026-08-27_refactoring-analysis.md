# Voxint Refactoring Analysis

_4-model consultation, 2026-08-27. Analysis only, no code changes._

**Models**: codex (structural complexity), deepseek-v4-pro (fragility/coupling),
grok-4.5 (overengineering), kimi-k3 (UX/operator experience).

**Codebase**: `main` at `4f6cc1a`, v0.26.0-dev. ~60K lines production Python
(176 files), ~102K lines tests (333 files).

**Grounding principle**: Voxint serves individuals and small teams who need
locally hosted audio intelligence. Non-technical researchers, journalists,
educators. Single-operator deployments. Enterprise-scale security and
scalability are explicitly not primary concerns.

---

## Consolidated Findings

Findings are deduplicated across models, grouped by category, and sorted by
impact. Agreement count shows how many models independently flagged the issue.

### High Priority

#### H1. Configuration sprawl: 157 params across three representations
**Agreement: 3/4** (codex, grok, kimi) |
**Impact: high** | **Effort: medium-high**

126 env-sourced settings + 31 DB columns + 25 resolver functions + 10 cross-field
validators. The DB-over-env model (tri-state NULL/true/false with row-wins-over-env
precedence) is warranted for UI-togglable preferences but has expanded beyond a
narrow boundary: booleans, endpoint strings, plaintext credentials, watch-sweep
state, onboarding lifecycle, and tutorial progress share one row. Adding a
UI-controlled setting requires synchronized changes in env fields, resolver logic,
form metadata, ORM migration, and consumers across worker/API.

The operator sees: a settings page where "inherit" references an env layer they
may not know exists, cross-flag dependency errors phrased in snake_case env names
("enrichment_names_llm_enabled requires llm_enabled=true"), and a constraint
network that makes configuration changes hard to reason about.

- `config.py:70-859` -- 126 fields + 10 validators
- `app_settings.py:27-594` -- 31 columns, 25 resolvers, tri-state logic
- `app_settings.py:542-594` -- 5 cross-flag invariants in env-var vocabulary
- `models.py:1727` -- mixed-purpose singleton row

**Quick wins**: (a) Rewrite invariant errors with section labels ("Name
suggestions requires the LLM to be enabled") and offer one-click "enable
prerequisites"; (b) show effective value + source beside each radio ("On --
inherited from installation"); (c) progressive disclosure: short "Common"
section vs collapsed "Advanced".

---

#### H2. Internal machinery vocabulary leaks into operator-facing errors
**Agreement: 1/4** (kimi, but exceptionally well-evidenced) |
**Impact: high** | **Effort: medium**

When something goes wrong, the operator sees messages like "idempotency key
'abc123' was already used with a different payload", "word-range [7, 12) is not
a current split child of this segment", "text-range anchor is stale; re-anchor
to recover", or that their tag name "is 42 code points". These describe the
system's model, not the operator's situation or next action. Half-open interval
notation and field names like `speaker_id` appear verbatim in 4xx detail strings.

More broadly: legacy action routes (requeue, cancel, archive, submit) raise
`HTTPException` from browser form POSTs; FastAPI renders `{"detail": "..."}` as
a bare JSON page. A stale-tab requeue -- an entirely expected event -- lands on
a JSON blob with no way forward except the back button. The newer media router
already solved this: `_reject` re-renders the page with a banner, preserving
selection state.

- `legacy_review.py:1075-1082` -- half-open interval + "split child" jargon
- `legacy_review.py:846-847` -- `detail=str(exc)` for ledger replay errors
- `annotations.py:1028` -- "text-range anchor is stale; re-anchor to recover"
- `legacy_runs.py:1119-1123` -- StaleRevisionError as bare 409 from form POST
- `api/app.py:264-276` -- 500 handler returns `text/plain`
- `media.py:621-637` -- the fix pattern already in-tree: `_reject` re-renders

**Quick wins**: (a) Rewrite the 10 worst strings via a route-boundary message
map (no behavior change); (b) add a global HTML error renderer for
`Accept: text/html` requests (friendly 4xx/500 template with "reload and try
again"); (c) port legacy routes to the media `_reject` pattern.

---

#### H3. Plugin framework is dormant ceremony
**Agreement: 3/4** (codex, grok, deepseek) |
**Impact: high** | **Effort: medium**

~1,040 production lines implement manifests, route panels, settings sections,
CLI registration, Celery task routing, completion hooks, job-lane recovery,
collision detection, boot validation, and a kill switch. The authoritative
builtin tuple is empty. Core API, worker, CLI, settings, templates, and
diagnostics carry plugin seams. At present this is an abstraction layer with
operational and testing cost but no operator-visible extension.

The enrichment subsystem, which plugins were intended to absorb, remains
active with hard-coded producers running alongside the plugin hooks -- creating
dual paths (`_autogenerate_*` AND plugin hooks) that double maintenance.

- `plugins/base.py:1-26` -- full contract surface
- `plugins/registry.py:1-14` -- fail-loud load, kill switch, collision checks
- `api/app.py:294-304, 474-537` -- registry load, validate_boot, collision at startup
- `worker/tasks.py:275-285, 435-457` -- dispatch + plugin lane redispatch

**Recommendation**: Treat plugins as aspirational. Until 2+ real builtins land,
freeze or delete unused registry/doctor/boot surface. Do not convert more
features until one conversion proves net LOC/clarity win.

---

#### H4. Legacy/new console coexistence creates dual submission surfaces
**Agreement: 2/4** (codex, kimi) |
**Impact: high** | **Effort: low (interim) / high (full retirement)**

The Console 2.0 migration has created overlapping route families. Production
mounts all 6 legacy routers (4 from legacy_runs, 2 from legacy_review) alongside
new areas. The new Jobs detail page imports `build_run_detail_context` from
`legacy_runs` -- new code depending on legacy code. With `CONSOLE_MEDIA_ENABLED`
on, the operator has two working submission surfaces: /runs forms and /media
forms, with divergent capabilities (only /media offers folder assignment).

- `api/app.py:48, 396, 457` -- all legacy routers mounted alongside new
- `jobs.py:40` -- cross-import from legacy_runs
- `legacy_runs.py:677-691` -- legacy still mints submission forms
- `media.py:498-591` -- parallel submission adding folder assignment
- `deps.py:297-306` -- sidebar retargets on flag but legacy forms remain

**Quick win**: When `console_media_enabled` is on, render a "New submissions
live in Media" banner on /runs forms and consider disabling the legacy submit
forms server-side. Strategic: accelerate P5 retirement.

---

#### H5. Implicit commit-before-publish contract has no enforcement seam
**Agreement: 1/4** (deepseek) |
**Impact: high** | **Effort: low**

`ingest/service.py` is deliberately broker-free: every function documents that
the caller owns the commit boundary and must later publish `voxint.run_pipeline`.
But nothing in function signatures or return values enforces or reminds callers of
that second step. A future caller can commit the transaction and forget
publication, leaving durable `QUEUED` rows that never get dispatched.

- `ingest/service.py:3-8` -- "the caller owns the commit boundary"
- `ingest/service.py:501-540` -- submit returns PipelineRun, no publish instruction
- `ingest/service.py:1077-1080, 1203-1206` -- upload/URL paths reiterate contract

**Recommendation**: Introduce a result object or context manager that exposes the
run id and a `publish_after_commit()` hook, or a high-level wrapper guaranteeing
commit-then-publish. Keep the broker-free property structural.

---

#### H6. Stage graph is duplicated across engine, context, and worker
**Agreement: 1/4** (deepseek) |
**Impact: high** | **Effort: medium**

Pipeline stage ordering exists in 4+ places: the `Stage` enum, `build_stage_fns`
in context.py, `next_stage()` in engine.py, and `GPU_SEGMENT`/`POST_SEGMENT`
lane sets + `pipeline_task_for_stage()` in worker/tasks.py. Adding or reordering
a stage requires coordinated edits across all of them.

- `pipeline/stages/context.py:369-376` -- explicit `{Stage: partial(...)}` mapping
- `pipeline/engine.py:49, 316, 398, 410-423` -- next_stage usage
- `worker/tasks.py:40-41, 313-316` -- lane sets + routing decision

**Recommendation**: Model the stage graph once (ordered stages + lane membership +
next-stage derivation) and derive all consumers from that model.

---

#### H7. Enrichment draft/asset pipeline is enterprise event-sourcing
**Agreement: 1/4** (grok) |
**Impact: high** | **Effort: high**

Producer runs use advisory locks, monotonic generations, full-payload
idempotency fingerprints, supersession of undecided candidates, evidence
ordinals, and scope XOR kinds. This machinery is appropriate for multi-writer
suggestion systems, applied to offline LLM drafts one operator triages.

The enrichment package is the widest-reaching non-API package (21 files,
18+ external import sites), so every change touches a heavy contract.

- `enrichment/drafts.py:1-32, 475-599` -- advisory lock, generation, supersede
- `enrichment/run_assets.py:1-28, 538-669` -- same pattern for assets

**Recommendation**: Keep fail-closed validation and human-decision immutability.
Simplify persistence toward: one current draft/asset per (scope, field/kind) +
optional history. Drop composite fingerprint equality where "same key = same
logical job" suffices.

---

### Medium Priority

#### M1. Tri-state feature flags + dark-ship route inventory
**Agreement: 3/4** (codex, grok, kimi) |
**Impact: medium** | **Effort: medium**

~20 nullable DB overrides create inherit/on/off where operators want on/off.
Console areas appear and disappear via flags, 404 without explanation, and
flag dependencies (activity depends on jobs) are invisible. Valuable for
zero-restart toggles and safe Console 2.0 rollout; excessive once flags
stabilize.

- `app_settings.py:143-187, 439-456` -- tri-state resolvers
- `deps.py:207-250` -- area gates raising undifferentiated 404
- `deps.py:307-318` -- activity ANDs console_jobs_enabled (undocumented)

**Post-Console 2.0**: collapse to bool env + optional DB override only for flags
operators actually toggle. Remove dark-ship when areas graduate.

---

#### M2. Settings/onboarding god file (2,578 lines)
**Agreement: 1/4** (codex) |
**Impact: medium** | **Effort: high**

Two routers, first-run wizard, folder browsing, tutorial seeding, diagnostics,
service status, credentials, feature invariants, and every settings mutation.
Because this controls onboarding and recovery, unrelated settings work has a
large regression radius. `app.py` imports the router's private
`_settings_context`.

- `settings.py:140, 471, 773, 1136, 1296, 1649` -- major sections
- `api/app.py:66` -- imports private `_settings_context`

---

#### M3. ORM schema god file (2,943 lines)
**Agreement: 1/4** (codex) |
**Impact: medium** | **Effort: high**

The entire schema in one file: core pipeline, media, adjudication, enrichment,
research, translations, embeddings, notifications, activity, annotations,
media-operation journals. 86 production modules import it. Schema changes create
broad import coupling across otherwise separate domains.

- `models.py:249` (media), `476` (pipeline), `1088` (adjudication), `1727`
  (app settings), `1879` (research), `2487` (notifications), `2626`
  (annotations), `2864` (operations)

---

#### M4. Upload finalizes filesystem before transaction commits
**Agreement: 1/4** (deepseek) |
**Impact: medium** | **Effort: medium**

`submit_upload` streams to a temp path and `os.replace()`s into the final
location before the caller commits the transaction. The DB transaction can roll
back after the file is durably published, leaving orphan files.

- `ingest/service.py:1104-1139` -- os.replace before caller's commit

---

#### M5. Split is append-only with no undo; recovery message is a dead end
**Agreement: 1/4** (kimi) |
**Impact: medium** | **Effort: medium**

One misclick in split mode is permanent (append-only CUTs). The multi-cut
refusal tells the operator to "re-transcribe to clear the split" but no
re-transcribe action exists for COMPLETED runs; the actual recovery path
(Media re-run) mints a new run and lives on a possibly flag-gated page.

- `legacy_review.py:1270-1276` -- "re-transcribe to clear the split"
- `adjudication/splits.py:230-270` -- append-only, no delete path

---

#### M6. Claim ceremony surfaces single-operator self-contention
**Agreement: 1/4** (kimi) |
**Impact: medium** | **Effort: medium**

CAS claim (one reviewer, token in URL, TTL expiry, takeover) is sound integrity
machinery for what is actually a tab-coordination problem. The operator opens the
workbench in a second tab and the first tab's next save 409s. The queue shows
runs "claimed by ben" when ben is the only user.

- `legacy_review.py:545-547` -- takeover acknowledged
- `legacy_review.py:501-513` -- claim loss as 409
- `resolver.py:609-619` -- queue surfaces claimed_by to one-person roster

**Recommendation**: Reframe in operator language ("You're editing this recording
here" / "This recording is open in another tab"). Make re-claim a single click
from the 409 toast. Suppress self-warnings.

---

#### M7. Post-finalize jobs not uniformly broker-fault tolerant
**Agreement: 1/4** (deepseek) |
**Impact: medium** | **Effort: low**

`_publish_finish_or_defer()` catches broker outages. But `_autogenerate_*`
for run assets, translation, and embeddings commit a job row then `.delay()`
inside `except Exception`. Broker failure after commit leaves stuck rows.
Stale-job redispatch exists only for embeddings.

- `worker/tasks.py:615-741` -- autogenerate paths with broad exception handling
- `worker/tasks.py:416-434` -- stale redispatch only for embeddings

---

#### M8. CSRF form expiry after restart hits a bare 403
**Agreement: 1/4** (kimi) |
**Impact: medium** | **Effort: low**

With no `csrf_secret` configured, the app mints a per-process secret. After
container restart, every open form fails with bare "invalid or missing CSRF
token" -- no hint that reloading fixes it.

- `api/app.py:309-315` -- per-process fallback with logger.warning only
- `deps.py:418-424` -- uniform 403

**Quick win**: Change 403 copy to "This form expired -- reload the page and
try again." Persist generated secret to data dir on first run.

---

#### M9. Archive/requeue lifecycle rules split and partly unenforced
**Agreement: 1/4** (deepseek) |
**Impact: medium** | **Effort: low**

`requeue_failed_run` does not check `archived_at`, even though archived failed
runs should stop being requeueable. The guard may live elsewhere, but the rule
is cross-module and invisible at the call site.

- `ingest/service.py:669-703` -- requeue checks status/stage/revision, not archived

---

#### M10. Idempotency ceremony duplicated everywhere
**Agreement: 1/4** (grok) |
**Impact: medium** | **Effort: medium**

The same "nonce + fingerprint + savepoint adopt + conflicting replay" pattern
appears in ledger, annotations, enrichment drafts, run assets, merge, uploads.
Each new write path copies ~80 lines of adopt-or-conflict.

- `ledger.py:114-164`, `annotations.py:736-882`, `drafts.py:443-598`,
  `run_assets.py:505-668`, `merge.py:126-131`, `ingest/service.py:675-703`

**Recommendation**: Extract a small shared `idempotent_insert(key, payload_hash,
build_row)` helper. Relax enrichment/annotation replay to key-only where
double-submit is the only threat. Do not weaken pipeline CAS.

---

#### M11. Config resolution version distributed across 3 modules
**Agreement: 1/4** (deepseek) |
**Impact: medium** | **Effort: low**

Version stamped in ingest, parsed with fallback in worker, interpreted in
StageContext. Fallback behavior is inconsistent (unknown = live-union in one
place, malformed = version 1 in another).

- `ingest/service.py:380-382` -- stamps version 2
- `worker/tasks.py:202-216` -- parses, falls back to v1
- `pipeline/stages/context.py:280-283` -- interprets

---

#### M12. Route assembly depends on registration order and path strings
**Agreement: 1/4** (deepseek) |
**Impact: medium** | **Effort: low**

Dark-ship discovery flags are derived from exact `route.path` comparisons after
mounting. The onboarding gate is structural (routes outside the `console`
aggregator bypass auth). Registration order is comment-stated. A path rename or
route reordering can silently break onboarding or UI discovery.

- `api/app.py:358-472` -- "keep registration order"
- `api/app.py:539-570` -- exact path comparisons for dark-ship stamps

---

### Low Priority

#### L1. Web research agent as default codebase weight
**Agreement: 1/4** (grok) |
**Impact: low** (disabled by default) | **Effort: low (containment)**

724-line agent with budget management, protocol repair, snippet grounding, and
injection posture is a "second product" bolted onto transcription review.
Disabled by default. Security burden (SSRF, untrusted text) is permanent.

**Recommendation**: Keep disabled. Treat as experimental extra. Separate package
docs, avoid new invariants on the hot settings path.

---

#### L2. StageContext broad dependency bundle
**Agreement: 2/4** (deepseek, grok) |
**Impact: low** | **Effort: low**

20+ fields in a frozen dataclass. Every stage receives the full context.
Manageable at current stage count. Both models agree this is not the primary
concern.

**Disagreement**: Deepseek suggests splitting into lane-specific contexts; Grok
says leave as-is and avoid adding more.

---

#### L3. CLI oversized composition root (1,702 lines)
**Agreement: 1/4** (codex) |
**Impact: low** | **Effort: medium**

19 handlers + parser builder, but lazy imports protect startup. Score harness
confirmed NOT dead code (isolated, DB-free, covered by tests).

---

#### L4. Residual folder-setting code
**Agreement: 1/4** (codex) |
**Impact: low** | **Effort: low**

`AppSettings.media_folders` and `folder_domain_packs` are no longer written.
Only consumed by migration-preflight CLI.

---

#### L5. Home windowing inconsistency, ghost pipeline states in copy
**Agreement: 1/4** (kimi) |
**Impact: low** | **Effort: low**

All-time figures (failed_runs, review backlog) sit beside windowed counts (24h)
with no labeling distinction. Cancel-refusal message lists "awaiting_adjudication"
-- an unreachable state.

---

## Key Disagreement

### Adjudication append-only architecture: overkill or load-bearing?

**Grok's position**: Mostly load-bearing. Single-operator still double-submits,
refreshes stale tabs, needs undo without destroying history. The append-only
ledger with CAS claims matches real single-operator failure modes (double-click,
network retry, soft-delete idempotency, split-stable anchors), not enterprise
fantasy. The overkill is at the margins: three attribution grains (label /
segment / word-range), dual SQL/Python resolvers, and 1,574 lines of pure
annotation coordinate math.

**Deepseek's position (tangential)**: The CAS/revision patterns in the pipeline
engine are appropriate and well-implemented. The concern is that they're implicit
and scattered, not that they're unnecessary.

**Consensus**: Do NOT replace ledger with mutable state for v1. The correctness
guarantees are real. Consider deferring finer grains (word-range reassignment,
annotation anchors) behind "advanced" if usage data shows they're rarely
exercised. Simplify the review queue SQL rather than materializing full
`label_states` per completed run.

---

## Cross-Cutting Themes

1. **Correctness culture applied uniformly, including to dormant and rare paths.**
   The codebase's strongest quality -- rigorous CAS, idempotency, append-only
   discipline -- is also its primary complexity source. The machinery is load-bearing
   in the pipeline and adjudication; it is proportionally expensive in enrichment,
   plugins, and disabled features.

2. **The newer code is better than the legacy code.** The media router's `_reject`
   pattern, honest skip-not-abort banners, contextual merge conflict copy, and
   translation's plain-language guidance are all good. The fix for many UX findings
   is "apply the pattern the newer code already uses."

3. **The mid-flight legacy migration is the largest single structural risk.** 3,797
   lines of "legacy" routes ARE production. New code cross-imports from them. Two
   submission surfaces coexist. Completing the Console 2.0 migration naturally
   resolves H4, contributes to H2 and M1-M2, and enables flag retirement (M1).

---

## Quick-Win Summary (low effort, high payoff)

| # | What | Effort | Addresses |
|---|------|--------|-----------|
| 1 | Rewrite 10 worst error strings via route-boundary message map | low | H2 |
| 2 | Global HTML error renderer for `Accept: text/html` (friendly 4xx/500) | low | H2, M8 |
| 3 | "New submissions live in Media" banner on /runs when flag is on | low | H4 |
| 4 | Commit-before-publish enforcement wrapper | low | H5 |
| 5 | Centralize config-resolution-version parsing | low | M11 |
| 6 | Uniform broker OperationalError handling for post-finalize jobs | low | M7 |
| 7 | Add archived-state check to requeue_failed_run | low | M9 |
| 8 | CSRF 403 copy: "This form expired -- reload" + persist secret | low | M8 |
| 9 | Label home cards: "all time" vs "last 24h" | low | L5 |
| 10 | Change split dead-end message to name real recovery path | low | M5 |

---

## Methodology

Each model received the same project briefing (overview, file sizes, architecture,
configuration surface, test structure) plus a distinct analytical lens:

- **Codex** (via clink): structural complexity, god files, dead code, config proliferation.
  Attached: db/models.py, config.py, app_settings.py, cli.py, settings.py.
- **Deepseek-v4-pro** (via analyze): fragility, coupling, implicit dependencies,
  change amplification. Attached: ingest/service.py, engine.py, context.py,
  tasks.py, app.py.
- **Grok-4.5** (via analyze): overengineering for the audience, enterprise patterns
  that add complexity without value. Attached: adjudication/ (11 files), enrichment/,
  plugins/, research/.
- **Kimi-k3** (via analyze): UX and operator experience, where complexity leaks
  into the workflow. Attached: legacy_review.py, legacy_runs.py, media.py, home.py,
  deps.py.

Findings were deduplicated by topic, agreement counted, and disagreements surfaced
explicitly. Severity ratings reflect consensus where models agreed; the higher
rating was preserved where they diverged on severity for the same finding.
