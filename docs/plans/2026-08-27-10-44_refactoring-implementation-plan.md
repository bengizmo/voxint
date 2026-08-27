# Refactoring implementation plan

## Context

A 4-model refactoring analysis (docs/plans/2026-08-27_refactoring-analysis.md)
identified 7 high-priority and 12 medium/low-priority findings across the
Voxint codebase. This plan turns those findings into actionable work by:

1. Right-sizing the project's development rules so refactoring sessions don't
   over-classify routine structural moves
2. Creating a phased implementation plan that allows parallel sessions where
   file ownership doesn't conflict
3. (After approval) Creating a Forgejo epic with child issues, then a
   next-session-prompt for the first implementing session

**Baseline warning**: The analysis was performed at `4f6cc1a`. Main has since
advanced to `9a989df` (~1,103 insertions, 266 deletions). A revalidation pass
against current main is required before creating issues (Step 0 below).

## Part 1: Right-size development rules

### Problem

The worked-example table in docs/testing.md has 11 rows but none explicitly
covering internal restructuring (god-file splits, helper extraction) or
operator-facing copy rewrites. Without a row, an agent following the rules
could over-classify these as "public contract or seam" changes.

### Proposed changes to docs/testing.md

Add 2 rows to the worked-example table:

| Example change | Impact class | Code-review depth | Browser lane | Why |
|---|---|---|---|---|
| Localized internal restructuring (helper extraction, module split) preserving all public imports, with no ORM registration, migration, concurrency, or numerics implications | routine | single-model review | no | Structural move, not a contract change. Existing coverage verifies behavior is preserved. Escalate if the move touches ORM mapper registration, creates circular imports, or changes import-time side effects. |
| Operator-facing error/UX copy rewrite that changes message text but not HTTP status, error conditions, or behavior | routine | single-model review | yes, if the affected error path is console-visible | Copy is observable behavior. Single-model review verifies no information-hiding or recovery-instruction regressions. |

**No Gate E batching note** (codex correctly flagged this as misstating the
existing carry-over rule, which already works as cumulative-diff-before-tagging).

**No changes to CLAUDE.md** (the existing text is correct and the worked
examples in docs/testing.md are the operative reference).

### What this does NOT relax

- 85% coverage floor, all 3 CI required checks
- Numerics doctrine, parity gates
- Contract-test-in-same-commit rule
- "When unsure, pick the deeper tier"
- Mandatory triggers for security, auth/CSRF, concurrency, migrations, deps
- Final-diff reclassification

## Part 2: Phased refactoring plan

### Parallelization model

Phases are NOT fully independent. File-level conflicts exist between some
items. The plan assigns **file ownership** per item so parallel sessions avoid
merge conflicts. Items touching the same files are sequenced or grouped.

### Phase 0: Quick wins

Low-risk, high-payoff changes. Each is its own PR. Items within Phase 0 are
parallelizable EXCEPT where noted.

**Batch 0A: Error copy rewrites** -- DONE (PR #232, merged)
- H2-qw: Rewrite 10 worst internal-vocabulary error strings at the route
  boundary. Files: `legacy_review.py`, `legacy_runs.py` (error-handling blocks
  only). Review: single-model. Browser lane: yes (console-visible errors).
- M5: Split dead-end message: name real recovery path. File:
  `legacy_review.py:1270-1276`. Review: single-model. Browser: yes.
- L5: Label home cards "all time" vs "last 24h"; scrub ghost pipeline state
  from cancel copy. Files: `home.py`, `ingest/service.py:126-130`. Review:
  single-model. Browser: yes (home).

**Batch 0B: Global HTML error renderer** (depends on 0A landing first)
- H2-renderer: Add a global HTML error template for `Accept: text/html`
  requests (friendly 4xx/500 with "reload and try again"). Adopt the media
  router's `_reject` pattern. File: `api/app.py` (exception handlers).
  Review: **multi-model** (new cross-cutting response seam; must preserve JSON
  content negotiation, status codes, security headers, and information-hiding).
  Browser lane: **yes**.

**Batch 0C: CSRF persistence** -- DONE (PR #234, merged)
- M8: Persist generated CSRF secret to data dir on first run; change 403 copy
  to "This form expired -- reload the page." Files: `api/app.py:309-315`
  (secret generation), `deps.py:418-424` (403 handler). Review: **full panel**
  (CSRF/security trigger per existing rules). Browser lane: **yes**.

**Batch 0D: Ingest service hardening** (H5 DONE, M9+M4 PR #239 open)
- H5: **DONE** (PR #251). Make the commit-before-publish contract visible. Add a SubmissionResult
  that carries the run id and a `publish()` method. Callers are migrated
  atomically. This makes the contract structural and visible, though it does
  not mechanically prevent out-of-order calls. Files: `ingest/service.py`,
  all callers (media.py, legacy_runs.py, cli.py, watch.py, tutorial/seed.py).
  Review: **multi-model** (cross-cutting seam, return contract change).
  Browser lane: no.
- M9: Add `archived_at` check to `requeue_failed_run`. File:
  `ingest/service.py:669-703`. Review: single-model. Browser: no.
- M4: Document the upload path's pre-commit filesystem exception; add a
  startup reconciler that removes orphaned `incoming/` files with no committed
  MediaItem. Files: `ingest/service.py`, `api/app.py` (startup hook). Review:
  single-model. Browser: no.

**Batch 0E: Worker hardening** (items share `worker/tasks.py`)
- M7: Uniform broker OperationalError handling for all `_autogenerate_*`
  post-finalize jobs. Pair with lane-specific stale-job redispatch. File:
  `worker/tasks.py`. Review: **full panel** (concurrency/reliability trigger).
  Browser lane: no.
- M11: Centralize config_resolution_version parsing into one helper. Files:
  `ingest/service.py:380`, `worker/tasks.py:202-216`,
  `pipeline/stages/context.py:280-283`. Review: single-model if pure
  extraction; multi-model if fallback semantics change. Browser: no.

**Batch 0F: Legacy submit redirect** (PR #238 open)
- H4-qw: When `console_media_enabled` is on, render a "New submissions live in
  Media" banner on /runs forms. Consider disabling the legacy submit forms
  server-side. Files: `legacy_runs.py` (submission form rendering), templates.
  Review: single-model. Browser lane: **yes**.

### Phase 1: Pipeline internals

**H6: Stage graph consolidation**
- Identify the exact remaining duplication beyond the existing STAGE_ORDER,
  GPU_SEGMENT, POST_SEGMENT in db/models.py. Add exhaustive equivalence
  contract tests. Derive consumer mappings where safe; avoid placing executable
  pipeline registration in the ORM module.
- Files: `db/models.py` (Stage enum area only), `pipeline/transitions.py`,
  `pipeline/stages/context.py`, `worker/tasks.py` (lane routing only).
- Review: **multi-model** (cross-cutting seam spanning enums, lanes, Celery
  routing). Browser lane: no.
- Can run in parallel with Phase 0 batches that don't touch worker/tasks.py
  (i.e., after Batch 0E lands).

### Phase 2: Enrichment simplification

**Step 1: M10 helper extraction ONLY** (no semantic relaxation)
- Extract the adopt-or-conflict pattern into a shared helper. Byte-for-byte
  behavior-preserving: same key + fingerprint + savepoint logic.
- Files: `adjudication/ledger.py`, `annotations.py`, `enrichment/drafts.py`,
  `enrichment/run_assets.py`, `merge.py`, `ingest/service.py`.
- Review: single-model (behavior-preserving extraction with existing tests).
- Any key-only relaxation (weakening fingerprint checks) is a **separate**
  follow-up requiring full panel review and concurrent-insert test evidence.

**Step 2: H7 design spike** (ADR, not implementation)
- Before implementation, produce a focused ADR answering:
  - What is the target persistence model? ("One current row, optional history"
    is unresolved.)
  - Prove the writer topology: can concurrent writers race on generation
    allocation or supersession? (Advisory locks currently prevent this.)
  - How do existing installations with multiple generations and partially
    completed jobs migrate?
  - How does ordering/provenance work without generation counters?
  - What is the rollback/refusal policy?
- Require representative pre-migration fixtures, concurrent database tests,
  and upgrade/downgrade validation.
- Only after the ADR is reviewed should implementation be scheduled.
- Review: multi-model for the ADR; **full panel** for any implementation PR
  (migration, locking, concurrency, data-integrity contracts).

### Phase 3: Console migration + config architecture (blocked)

Depends on Console 2.0 P5 (legacy route retirement). Documented for
sequencing; will be planned in detail when P5 lands.

**P5-dependent:**
- H4-full: Retire legacy_review.py and legacy_runs.py routes
- M1: Collapse tri-state flags to bool where operators don't need inherit

**Potentially independent of P5** (evaluate when scheduling):
- H1-full: Configuration architecture consolidation
- M2: Split settings.py into wizard + diagnostics + per-section routers
- M12: Replace path-string route discovery with explicit router metadata

Review: full panel for each DB migration, auth/security change, or public
contract retirement. Browser lane: yes for route changes.

## Finding disposition (complete)

Every finding from the analysis has an explicit decision:

| Finding | Decision | Rationale |
|---|---|---|
| H1 config sprawl | Phase 0 quick wins + Phase 3 full | Quick wins (copy) are independent; structural work depends on Console 2.0 |
| H2 error vocabulary | Phase 0A + 0B | Copy + renderer, no structural deps |
| H3 plugin framework | **Keep** | User decision + ADR 0006 (greenfield plugin commitment) |
| H4 dual surfaces | Phase 0F quick win + Phase 3 full | Interim signposting now; full retirement with P5 |
| H5 commit-before-publish | Phase 0D | Makes contract visible; bundled with M4/M9 |
| H6 stage graph | Phase 1 | Localized pipeline work |
| H7 enrichment event-sourcing | Phase 2 (design spike first) | Needs ADR before implementation |
| M1 tri-state flags | Phase 3 | Depends on flag graduation after P5 |
| M2 settings god file | Phase 3 | Best with H1 full |
| M3 ORM god file | **Defer** | Declarative; 86 import sites = high conflict risk for low payoff. Revisit based on measured change friction. |
| M4 upload pre-commit filesystem | Phase 0D | Bundled with H5 (same file, related durability) |
| M5 split dead-end message | Phase 0A | Copy-only fix |
| M6 claim ceremony | **Defer** | Real write-integrity guarantee. Reframe when Console 2.0 review surface replaces legacy. |
| M7 broker fault tolerance | Phase 0E | Bundled with M11 (same file) |
| M8 CSRF 403 | Phase 0C | Security-classified, independent |
| M9 archive check in requeue | Phase 0D | Bundled with H5 (same file) |
| M10 idempotency helper | Phase 2 step 1 | Extraction only; no semantic relaxation |
| M11 config version centralization | Phase 0E | Bundled with M7 (overlapping files) |
| M12 route assembly | Phase 3 | Evaluate P5 dependency per-item |
| L1 web research | **Defer** | Already disabled. Retain security audit trigger. |
| L2 StageContext | **Defer** | Manageable at current scale (2-model agreement) |
| L3 CLI root | **Defer** | Lazy imports mitigate; score harness confirmed live |
| L4 residual folder code | **Defer** | Intentional transition debt, low impact |
| L5 home cards + ghost state | Phase 0A | Copy-only fix |
| Adjudication append-only | **Preserve** | Load-bearing for real single-operator failure modes (4-model agreement) |

## Part 3: Implementation deliverables (this session)

After plan approval:

1. **Edit docs/testing.md** with the 2 new worked-example rows
2. **Revalidate** findings against current main (9a989df) -- check file
   locations, caller counts, and line references are still accurate
3. **Create Forgejo epic** linking to the analysis report, with child issues:
   - One issue per batch in Phase 0 (0A through 0F)
   - One issue for Phase 1 (H6)
   - One issue for Phase 2 step 1 (M10 extraction)
   - One issue for Phase 2 step 2 (H7 design spike)
   - One umbrella issue for Phase 3 (details deferred)
   - Each issue includes: description, files touched, review tier, browser
     lane decision, dependencies, acceptance criteria
4. **Write next-session-prompt** for the first implementing session

## Verification

- docs/testing.md changes pass lint and don't conflict with CLAUDE.md
- All findings have explicit disposition
- Issues have file ownership to prevent merge conflicts
- Review tiers respect existing mandatory triggers (codex-verified)
- Phase 2 H7 requires ADR before implementation (not just tests)

## Review notes (codex critique resolution)

Codex critique received via zen clink (planner role). Key points and resolution:

| Codex feedback | Resolution |
|---|---|
| **Analysis baseline is stale** (4f6cc1a vs 9a989df) | ACCEPT. Added revalidation step before issue creation. |
| **H3 conflicts with ADR 0006** | ACCEPT. User also said keep plugins. H3 dropped. |
| **Rule changes too permissive** (ORM split = public seam, error copy = observable behavior, dead-code = dynamic callers, Gate E note misleading) | ACCEPT. Narrowed restructuring row to "localized, no ORM/migration/concurrency." Upgraded error copy to single-model + browser lane. Dropped dead-code row and Gate E note. |
| **Phases 0-2 not truly independent** (file overlaps in ingest/service.py, worker/tasks.py, api/app.py) | ACCEPT. Replaced independence claim with batch-level file ownership and sequencing constraints. |
| **M4, M6, L4 undispositioned** | ACCEPT. Added complete finding disposition table. M4 bundled with H5, M6 deferred, L4 deferred. |
| **H5 doesn't enforce ordering** | ACCEPT. Acknowledged in description: makes contract visible, does not mechanically prevent out-of-order calls. |
| **M10 extraction + semantic relaxation conflated** | ACCEPT. Split into extraction-only (Phase 2 step 1) and optional relaxation (separate follow-up with full panel). |
| **H7 needs design spike** (advisory locks protect generation allocation, generations encode ordering, target model unresolved, migration strategy missing) | ACCEPT. H7 elevated to ADR-first approach with concurrent-db proof. |
| **Review tiers under-classified** (H2 renderer = multi-model, M8 = full panel, H5 = multi-model, M7 = full panel, H6 = multi-model) | ACCEPT. All tier assignments corrected per codex's analysis, cross-checked against CLAUDE.md mandatory triggers. |
| **Phase 3 P5 dependency not justified for all items** | PARTIALLY ACCEPT. M2/M12 flagged as "evaluate P5 dependency per-item" rather than blanket block. |
| **Characterization tests needed, not just coverage** | ACCEPT. Added to acceptance criteria for each batch. |
