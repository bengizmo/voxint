# Plan: Console soft-archive runs + derived-media deletion (issue #5, slice 2)

**Written:** 2026-08-16 · **Branch:** `feat/console-archive-runs` · **Issue:** #5

Finishes issue #5 (Cancel half already landed). Gives the single operator a safe,
reversible way to hide finished/junk runs, plus a separate destructive control to
reclaim a run's derived audio bytes. Console is append-only by doctrine, so this
is soft-archive (a timestamp column), not row deletion.

## Locked decisions (maintainer, 2026-08-15) — not re-litigated
- Soft-archive via nullable `archived_at`; ledger + all rows stay intact;
  reversible. No new `ARCHIVED` RunStatus (keeps status orthogonal).
- Media deletion is a SEPARATE action; archive never touches MEDIA_ROOT.

## Finalized decisions (after codex + grok review — see audit trail)

1. **`archive_run` / `unarchive_run`**: terminal-only guard
   ({COMPLETED, FAILED, CANCELLED}); live runs → `RunNotArchivableError`
   ("cancel first"). Last-write-wins (mirrors `save_operator_notes`), NO revision
   bump, idempotent (already-archived / already-active → no-op success).
2. **Media deletion — DERIVED-ONLY in v1.** `delete_run_derived_media` removes
   THIS run's `AudioArtifact` + `AudioChunk` rows and their files only.
   **Never touches `MediaItem` / `source_path`.** Terminal-only (never unlink
   under a live worker); NOT archived-first (orthogonal actions). Shared-source
   deletion (refcount lock + `incoming/`-only allowlist + confirm copy) is
   deferred to a **v2 follow-up issue** — both reviewers flagged it as the footgun.
3. **Post-commit unlink**: service mutates rows + returns `MediaDeleteResult`
   with the list of confined paths; the route commits, THEN unlinks best-effort
   (mirrors publish-after-commit). Never unlink before commit. Idempotent on
   already-missing files. Path-confined under `media_root` (defense in depth —
   don't trust DB path strings).
4. **Filtering**: default-exclude archived from `/runs` (`list_runs`) and
   `/review` (`adjudication_queue`). `?archived=1` = archived-only view on /runs,
   preserved through keyset pagination URLs. Predicate `archived_at IS NULL`
   before the cursor clause.
5. **Stats**: exclude archived from `run_status_counts` + `runs_created_since`
   (consistent across dashboard/#13, /metrics, `voxint stats`). StageRun attempt
   telemetry left alone (historical). Prometheus loop unaffected (no new status).
6. **Archived-run mutation guard**: refuse `requeue` and review-`claim` on an
   archived run (cheap `archived_at is not None` check) — a stale tab must not
   drive a hidden run live. (Codex point; keeps visibility and mutability aligned.)
7. **Placement**: detail-page-only buttons (mirror Cancel). runs.html gets the
   archived toggle + an "archived" pill only — no per-row mutation, no
   RunListItem.revision plumbing. Button copy honest: "Delete derived audio
   files" (NOT "delete media"), "Archive (hide; keeps all data)".
8. **No CLI** this slice (Cancel added none either). Deferred.
9. CSRF: `CSRF_RUN_ARCHIVE`, `CSRF_RUN_UNARCHIVE`, `CSRF_RUN_MEDIA_DELETE`
   (per-action; distinct blast radii).

## Implementation order (TDD)
1. Migration 0013 (down_rev 0012): `pipeline_runs.archived_at TIMESTAMPTZ NULL`.
2. Model: `PipelineRun.archived_at`.
3. Service (red→green): `archive_run`, `unarchive_run`, `RunNotArchivableError`,
   `delete_run_derived_media` + `MediaDeleteResult`. Export from `ingest/__init__`.
4. Read filtering: `list_runs` archived predicate + `archived` param;
   `adjudication_queue` exclusion; `stats_query` exclusions.
5. Mutation guards: requeue + claim refuse-if-archived.
6. CSRF actions + routes: `POST /runs/{id}/archive|unarchive|media/delete`;
   mint tokens in run_detail context; `runs()` gains `archived` param.
7. Templates: run_detail buttons + result banner; runs.html toggle + pill.
8. Tests ≥85%: service unit, API integration, runs_query filter, stats exclusion.
9. Docs: operations.md console-actions §, README, CHANGELOG [Unreleased].

## Review notes (audit trail)
- **codex (clink, planner role)**: endorsed soft-archive; pushed archived-first +
  Voxint-owned-namespace + source tombstone + broad mutation guards. Count ALL
  runs as referrers. Structured skip-result. No numerics risk; append-only risk
  is in media deletion, not archived_at.
- **grok-4.5 (zen chat)**: agreed archive design; DECISIVE call to ship media
  deletion DERIVED-ONLY in v1, deferring shared-source entirely — removes the
  refcount race, library-deletion hazard, and tombstone need. Terminal-only, NOT
  archived-first. Post-commit unlink ordering. Name it `delete_run_derived_media`
  so a future maintainer doesn't extend it to source.
- **Reconciliation**: took grok's derived-only scope cut (resolves codex's
  biggest hazards by omission) + codex's post-commit-unlink and requeue/claim
  guard. Deferred: archived-first coupling, source tombstone, Voxint-owned check
  (all move to the v2 "delete original source media" follow-up issue).
