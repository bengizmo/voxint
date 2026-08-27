# ADR 0008: Enrichment persistence simplification scope

> **Status:** Proposed (refactoring plan Phase 2 Step 2, finding H7)

## Context

A 4-model refactoring analysis (finding H7) identified the enrichment
persistence model as "enterprise event-sourcing" machinery applied to offline
LLM drafts that a single operator triages. The enrichment subsystem uses
advisory locks, monotonic generations, full-payload idempotency fingerprints,
supersession chains, evidence ordinals, and scope-XOR kinds across three
persistence families: draft candidates (`drafts.py`), run assets
(`run_assets.py`), and translations (`translations.py`). At 21 files and 18+
external import sites, it is the widest-reaching non-API package.

The question: is this proportionate to a single-operator, locally hosted system?

Three constraints make most of the machinery load-bearing:

1. **Real concurrency.** The Celery worker, the API server, and the operator can
   all trigger enrichment producers at the same time. Row locks cannot guard a
   scope with no prior rows (the first invocation for a new speaker or run), so
   advisory locks fill that gap.

2. **Human decision immutability.** An operator accept or reject is terminal. A
   producer rerun must never touch a decided candidate. The supersession
   mechanism stamps only still-proposed candidates and uses a two-statement
   `FOR UPDATE` then `UPDATE` pattern to defeat a READ COMMITTED snapshot
   anomaly where a concurrent decision would otherwise be invisible to the
   supersession UPDATE.

3. **Finalization-order guarantees.** Generations are allocated at finalization
   time under the advisory lock: `MAX(generation) + 1`. Supersession scopes to
   `generation < run.generation`, so a late-arriving completion cannot supersede
   claims from a run that finalized more recently. This is finalization order,
   not invocation order or causal order. If invocation A starts before B but B
   finalizes first (generation 1) and A finalizes last (generation 2), A
   supersedes B. This is the intended policy for a single-operator system: the
   most recent completed analysis wins.

## Decision

### 1. The persistence model stays

The append-only generation chain with supersession stamps is the simplest
correct model satisfying the three constraints above. A "one current row plus
history table" alternative would require moving superseded rows to a history
table (an extra write per supersession), checking the history table for
decided-candidate immunity (an extra join), and synchronizing the move with the
insert (an extra serialization point). For draft candidates, where decisions and
evidence must retain stable foreign-key identities, the cross-table split is
strictly more complex.

"Latest" continues to be derived at read time via `superseded_by IS NULL` for
all consumer surfaces. No schema change, no data migration.

### 2. Generations stay

Generation counters cost one `MAX+1` query under an already-held lock and
provide clean write-time ordering independent of wall-clock. Removing them
would require replacing the supersession scope with `created_at` comparisons
(sensitive to clock skew between workers) or supersession-chain traversal (O(n)
in chain length, fragile). The `latest_producer_run()` query is the only
read-time consumer of generation ordering; everywhere else, supersession stamps
are the filter.

### 3. Full-payload replay-conflict detection stays

The current replay check compares every stored field when an idempotency key
matches. This catches key-construction bugs that would otherwise silently adopt
a row with different results, source hash, model, or configuration. The cost is
one JSON comparison on the uncommon replay path. Reducing to key-only
first-write-wins adoption would remove this diagnostic protection for no
measurable performance gain.

For translations, where payloads can reach several megabytes, the replay check
should use a stored canonical digest rather than loading and comparing the full
`lines` JSONB.

### 4. Transaction choreography is extracted

The advisory-lock, generation-allocation, idempotency-lookup, and
savepoint-protected-persist dance repeats across all three families (~86 lines
in drafts, ~72 in assets, ~42 in translations). M10 already extracted the
savepoint skeleton (`savepoint_adopt_or_conflict` in `idempotency.py`). The next
step extends or complements that helper with advisory-lock acquisition and
post-lock idempotency re-lookup.

The extraction boundary is narrow by design: only the invariant transaction
choreography moves into the shared module. Validation, payload construction, row
building, fingerprint definition, and supersession SQL stay in each family
module. The drafts two-statement `FOR UPDATE` supersession is materially
different from the simple asset/translation head retirement; turning those into
generic predicates would hide concurrency invariants without removing much code.

Net-complexity gate: if the extracted helper saves fewer than ~30 net lines or
needs more than three callbacks, keep the current family-local code and
standardize only comments and tests.

### 5. Translation integrity gaps are closed

`run_translations` has two gaps relative to `enrichment_candidates` and
`run_enrichment_assets`:

**Missing idempotency key.** `record_translation` inserts directly with no
savepoint, no idempotency key, and no replay detection. The job system's
`claim_job()` CAS prevents LLM re-invocation on Celery redelivery (a succeeded
job returns no row to the claimer), so this is not a crash-retry data-loss
scenario. The gap is defense-in-depth and API consistency: every other
enrichment writer has replay protection, and translations should too.

One migration adds a nullable `idempotency_key TEXT UNIQUE` column.
Existing rows receive no backfill (they are completed generations needing no
replay protection). The writer populates it for all new rows. The key derives
from the translation job ID, matching the asset-job convention.

**Missing immutability trigger.** `enrichment_candidates` and
`run_enrichment_assets` both have database triggers rejecting UPDATE (except the
write-once supersession stamp) and DELETE. `run_translations` documents the same
immutability contract but enforces it only at the application layer. The same
migration adds the trigger, bringing translations to parity.

### 6. Validation layering stays

Python-side validation in the writer modules mirrors the DB CHECK constraints.
This is defense-in-depth, not duplication: the Python layer gives earlier,
clearer error messages before the advisory lock is acquired and the expensive
work is committed. Removing it would push all error reporting to opaque
`IntegrityError` exceptions from the database.

## Consequences

- No data migration for the main refactoring. Code-only helper extraction.
- One Alembic migration for `run_translations`: nullable `idempotency_key`
  column with UNIQUE constraint, plus an immutability/supersession-integrity
  trigger matching the asset pattern.
- Reduced code duplication across the three persistence families, contingent on
  the net-complexity gate.
- Translation integrity brought to parity with assets and candidates.
- The finalization-order-wins policy is explicitly documented and accepted.
- Future enrichment producers follow the extracted protocol rather than copying
  the advisory-lock/generation/idempotency/supersession pattern by hand.
- Consumer surfaces (review workbench, speakers page, run detail, CLI) are
  unaffected. They read through `queries.py`, `latest_assets()`, and
  `current_translations()`, none of which change.

## Rejected alternatives

**One current row plus history table.** More complex for candidates (decisions
and evidence reference candidate rows by FK; moving superseded rows breaks those
references or requires duplicating them). Plausible for assets and translations
alone via trigger-maintained history, but not worth losing the uniform
append-only model across all three families.

**Removing generation counters.** Saves one `MAX+1` query per finalization and
loses write-time ordering safety. The alternatives (wall-clock comparison or
chain traversal) are either clock-skew-sensitive or O(n).

**Key-only first-write-wins idempotency.** Removes the diagnostic check that
catches key-construction bugs. Silent adoption of a row with different results
is harder to diagnose than an explicit conflict error.

## Verification

The refactoring plan requires concurrent test scenarios and validation plans.
Since there is no data migration for the main refactoring, these are test
scenarios for the translation gap closure and protocol extraction:

1. **Finalization-order supersession.** Invocation A starts, B starts, B
   finalizes (generation 1), A finalizes (generation 2). Assert A's candidates
   supersede B's proposed candidates.
2. **Decision beats supersession.** Covered by the existing integration test
   `test_decision_beats_concurrent_supersession` in
   `tests/integration/test_enrichment_drafts.py`.
3. **Translation idempotency replay.** Insert a translation, call
   `record_translation` again with the same idempotency key and payload. Assert
   the existing row is returned, not a new generation.
4. **Translation immutability trigger.** Insert a translation row. Attempt
   UPDATE of content columns and DELETE. Assert both are rejected. Verify that
   stamping `superseded_by_translation_id` once succeeds and stamping it again
   is rejected.
5. **Crash-after-claim recovery.** Document that `claim_job()` CAS handles
   Celery redelivery (a succeeded job returns no row), so result-row
   idempotency is defense-in-depth, not the primary retry mechanism.
6. **Translation migration rollback.** The migration is a single `ALTER TABLE
   ADD COLUMN` plus `CREATE TRIGGER`. Postgres DDL is transactional:
   `alembic downgrade` drops both atomically.
