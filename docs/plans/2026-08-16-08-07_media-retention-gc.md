# Plan — Media retention / garbage collection (issue #15)

**Created:** 2026-08-16 08:07 · **Repo:** `~/dev/voxint` · **Issue:** #15 · **Author:** planning session (codex-reviewed)

---

## Goal

Voxint storage grows unbounded — every run leaves a normalized 16 kHz WAV
intermediate on disk under `artifacts/{run_id}/normalized.wav`, and nothing
ever reclaims it. Add an **opt-in, beat-scheduled GC sweep** that, for **old
terminal runs**, unlinks that intermediate WAV and records the reclamation on
the artifact row (audit trail). Everything else — the **source media** (runs
stay re-processable), transcripts, diarization turns, speaker assignments, and
the immutable `adjudication_decisions` ledger — is **always kept**. No run-row
deletion (that is issue #5's roadmap). This is **file reclamation only**.

## Decisions locked with the maintainer

1. **Intermediates only** — reclaim the `preprocessed_audio` WAV; never the
   source under `incoming/{uuid}/`.
2. **`reclaimed_at` audit column** — stamp the artifact row, don't delete it.
3. **Retention basis = `run.updated_at`** (no schema/state-machine change).
   Documented semantics: "age since the run was last modified." Adjudication
   and enrichment write to *separate* tables and do **not** bump it; an
   operator editing `operator_notes` on a completed run *does* bump it, which
   benignly delays reclamation for actively-touched runs. A dedicated
   `completed_at` column was considered and **deferred** (costs a migration +
   backfill + touching the CAS state machine — out of proportion for a
   single-operator tool where `updated_at` is a good-enough proxy).
4. **OFF by default** — `media_retention_enabled=false`. No data is ever
   reclaimed until the operator consciously enables it and sets a TTL.
   `docs/operations.md` makes enabling loud. (Accepted tradeoff: unbounded
   growth persists until opt-in — matches the project's honest/safe-default
   doctrine over surprise deletion.)

## Assumptions & constraints

- Terminal statuses = `completed`, `cancelled` (`RunStatus`). `failed` is
  semi-terminal (requeue-able) → **excluded**. Only these two are eligible.
- Exactly one `preprocessed_audio` row per completed run (the `/media`
  read-path predicate depends on this; we preserve that invariant).
- All `AudioArtifact.path` values are **relative** to `media_root`.
- **Storage layout invariant:** intermediates live under `artifacts/…`,
  sources under `incoming/…` — they never alias. We nonetheless add a
  defensive source-alias guard (below) rather than trust the invariant alone.
- Single-box deployment, but Celery may run >1 worker and beat can overlap a
  long sweep → **concurrency must be correct**, not assumed away.
- `pipeline_runs.status` is indexed; `updated_at` is not. `audio_artifacts`
  has no index beyond `pipeline_run_id`. We add a partial index for the sweep.
- Repo enforces a **global** `--cov-fail-under=85` (not per-changed-file);
  target high coverage on the new modules and lean on `/unit-testing`.
- No inference numerics are touched → **no parity gate** (confirmed).

## Proposed approach

A pure, Celery-free reclamation core + a thin beat task, mirroring the
`recovery_sweep` pattern.

### Reclamation core — `src/voxint/media/reclaim.py`

`reclaim_expired_intermediates(session, *, media_root, cutoff, batch_limit, tutorial_run_id) -> ReclaimSummary`

Per-row, **claim-then-act** to be safe under overlapping sweeps and duplicate
Celery delivery:

1. **Select eligible rows** with a row lock that skips contended rows:
   ```sql
   SELECT a.id, a.path, a.pipeline_run_id
   FROM audio_artifacts a
   JOIN pipeline_runs r ON r.id = a.pipeline_run_id
   WHERE a.kind = 'preprocessed_audio'
     AND a.reclaimed_at IS NULL
     AND r.status IN ('completed','cancelled')
     AND r.updated_at < :cutoff
     AND r.id <> :tutorial_run_id            -- never reclaim the tutorial run
     AND NOT EXISTS (                          -- defensive source-alias guard
       SELECT 1 FROM media_items m WHERE m.source_path = a.path)
   ORDER BY r.updated_at ASC, a.id ASC        -- oldest-first, stable tie-break
   LIMIT :batch_limit
   FOR UPDATE OF a SKIP LOCKED
   ```
2. For each locked row, **path-safe resolve + unlink**, then **stamp** inside
   the same transaction so the lock is held across the whole reclaim:
   - `resolved = (media_root / path).resolve()`; fail closed (log, leave
     `reclaimed_at` NULL, skip) unless `resolved.is_relative_to(media_root.resolve())`.
   - `lstat` the resolved path; if it is a **symlink** or a **directory**,
     fail closed (never unlink). This tightens the reused serving guard for a
     *deletion* context (TOCTOU/symlink-replacement).
   - `size = resolved.stat().st_size` (before unlink) for `reclaimed_bytes`.
   - `resolved.unlink()`.
     - `FileNotFoundError` → **tolerated**: the file was already gone (orphan
       from prepare's delete-without-unlink, or an interrupted prior reclaim).
       Stamp `reclaimed_at=now`, `reclaimed_bytes=0`.
     - `PermissionError` / other `OSError` (read-only mount, races) → **fail
       closed**: log at WARNING, leave `reclaimed_at` NULL (retried next
       sweep), continue the batch. **Per-row isolation** — one bad row never
       aborts the batch.
   - On success stamp `reclaimed_at=now`, `reclaimed_bytes=size`.
3. Commit per row (or per small chunk) so a mid-batch crash keeps completed
   reclaims durable.

**Crash-window semantics (documented, not eliminable across fs+db):** a crash
after `unlink` but before commit leaves an absent file + unreclaimed row; the
next sweep re-selects it, hits `FileNotFoundError`, and stamps
`reclaimed_bytes=0`. So **`reclaimed_bytes` = bytes measured at the moment of a
clean reclaim; 0 if the file was already absent.** It is an audit/observability
counter, not an accounting guarantee. Tested explicitly.

`ReclaimSummary` = counts: `selected, reclaimed, missing, failed, bytes` —
returned *and* emitted as a structured log line, plus a per-row WARNING on each
failure (a returned dict alone is invisible for a periodic Celery task).

### Beat task — `src/voxint/worker/tasks.py`

```python
@app.task(name="voxint.gc_sweep")
def gc_sweep() -> dict[str, int]:
    settings = get_settings()
    if not settings.media_retention_enabled:
        return {"selected": 0, "reclaimed": 0, "missing": 0, "failed": 0, "bytes": 0}
    factory, _ = _runtime()
    cutoff = datetime.now(tz=UTC) - timedelta(seconds=settings.media_retention_seconds)
    tutorial_run_id = <resolve from app_settings.tutorial_run_id>
    with factory() as session:
        summary = reclaim_expired_intermediates(
            session, media_root=settings.media_root, cutoff=cutoff,
            batch_limit=settings.gc_batch_limit, tutorial_run_id=tutorial_run_id,
        )
    logger.info("gc_sweep %s", summary)
    return summary.as_dict()
```

The task double-checks the enable gate (belt-and-suspenders with the beat
registration) so a stale beat entry can't act when disabled.

### Beat registration — `src/voxint/worker/app.py`

Register `"gc-sweep"` **only when enabled**:
```python
if settings.media_retention_enabled:
    app.conf.beat_schedule["gc-sweep"] = {
        "task": "voxint.gc_sweep", "schedule": settings.gc_sweep_seconds}
```

### Console read-path — reclaim-aware, `normalized_audio_path` untouched

`normalized_audio_path` is shared by pipeline stages (transcribe,
diarize_embed), the `/media` route, run_detail, and tutorial seed repair.
**Do not change its contract** — stages must still resolve the path mid-run.
Instead add reclaim-awareness only at the read/console layer:

- `api/app.py` run_detail: `audio_available` becomes False when the sole
  `preprocessed_audio` row has `reclaimed_at IS NOT NULL`; pass
  `media_reclaimed=True` + the reclaimed date to the template.
- `run_detail.html`: render "Media reclaimed on `<date>`" (muted notice)
  instead of a dead **Audio** link.
- `/media/{run_id}`: when the artifact is reclaimed, return a clear
  **410 Gone** ("media reclaimed") rather than the incidental 404 that
  MediaGate would raise on the missing file — an honest, distinguishable
  status. (Small helper reused by both the route and run_detail.)
- **Pipeline stages:** no change — reclaimed rows are always terminal, and
  stages only touch non-terminal runs, so a stage can never meet a reclaimed
  artifact.
- **Tutorial run:** excluded from GC entirely (query guard above), so seed
  repair never faces a reclaimed WAV.

### Config — `src/voxint/config.py`

Three new settings (heavy doc-comments, mirroring the existing sweep block;
**none** added to `TIER_SCALED_TIMING_FIELDS`):
- `media_retention_enabled: bool = False`
- `media_retention_seconds: int = Field(default=2592000, ge=3600)` — 30-day
  default *value* (only used when enabled), 1-hour floor.
- `gc_sweep_seconds: int = 3600` — beat cadence.
- `gc_batch_limit: int = Field(default=500, ge=1)` — rows per sweep,
  oldest-first; documents the backlog drain rate (`batch_limit` per
  `gc_sweep_seconds`). One batch per sweep (simple; drains over time).

### Migration 0014 — `alembic/versions/0014_audio_artifact_reclamation.py`

- `audio_artifacts.reclaimed_at TIMESTAMPTZ NULL`
- `audio_artifacts.reclaimed_bytes BIGINT NULL`
- **Paired-nullability CHECK** (mirrors `pipeline_runs_review_claim_shape_check`):
  `(reclaimed_at IS NULL) = (reclaimed_bytes IS NULL)` — no half-stamped rows.
- **CHECK** `reclaimed_bytes IS NULL OR reclaimed_bytes >= 0`.
- **Partial index** for the sweep, validated with EXPLAIN on production-shaped
  data:
  `CREATE INDEX ix_audio_artifacts_reclaimable ON audio_artifacts (pipeline_run_id) WHERE kind = 'preprocessed_audio' AND reclaimed_at IS NULL`
  (small — only unreclaimed intermediates; the run-side filter rides
  `pipeline_runs.status` index + the join).
- Downgrade drops index, checks, columns.
- Matching columns on the `AudioArtifact` model.

## Affected files / components

| Path | Change |
|---|---|
| `src/voxint/config.py` | +4 settings (enabled gate, retention seconds+floor, sweep cadence, batch limit); doc-comments; NOT tier-scaled |
| `src/voxint/media/reclaim.py` | **new** — pure `reclaim_expired_intermediates` + `ReclaimSummary`; safe-unlink helper |
| `src/voxint/worker/tasks.py` | +`gc_sweep` task (gated), structured log |
| `src/voxint/worker/app.py` | conditional `"gc-sweep"` beat entry |
| `src/voxint/db/models.py` | +`reclaimed_at`, `reclaimed_bytes` on `AudioArtifact` |
| `alembic/versions/0014_audio_artifact_reclamation.py` | **new** migration |
| `src/voxint/api/app.py` | reclaim-aware `audio_available`; `/media` 410-on-reclaimed |
| `src/voxint/api/templates/run_detail.html` | "Media reclaimed on `<date>`" notice |
| `.env.example` | document the 4 new env vars |
| `docs/operations.md` | retention section (what's reclaimed/kept, how to enable, drain rate) |
| `docs/architecture.md` | GC sweep + `reclaimed_at` note |
| `CHANGELOG.md` | `[Unreleased]` entry |
| tests (below) | unit + integration + migration |

## Step-by-step implementation

1. **Config** + `.env.example` + unit tests (`tests/unit/test_config.py`):
   defaults, `ge` floor rejection, env override, enabled-gate boolean parsing.
2. **Migration 0014** + model columns + `tests/integration/test_migration_0014.py`
   (upgrade adds columns/checks/index; both CHECKs reject bad rows;
   downgrade removes cleanly; existing rows preserved with NULLs).
3. **`media/reclaim.py`** core + unit tests with a tmp `media_root` and a
   fake/real session (mostly integration — see below).
4. **`gc_sweep` task** + **beat registration** + `tests/unit/test_worker.py`
   (assert `"gc-sweep"` present iff enabled; disabled → task returns zeros
   and never touches the DB).
5. **Console**: reclaim-aware `audio_available`, `/media` 410, template
   notice.
6. **Docs** (`operations.md`, `architecture.md`) + **CHANGELOG** `[Unreleased]`.
7. Run gates: `ruff`, `mypy src/`, unit+contract (deselect the two known
   non-hermetic installer tests), then the integration suite against the
   throwaway DB.

## Testing strategy

**Unit** — config (defaults/floor/env/gate); worker beat-schedule presence &
disabled-path zeros.

**Integration** (`tests/integration/test_gc_sweep.py`, throwaway DB + tmp
`media_root`), covering codex's gap list:
- Happy path: terminal run + WAV on disk older than TTL → file unlinked,
  `reclaimed_at` set, `reclaimed_bytes == real size`.
- **Non-terminal excluded**: `queued`/`running`/`awaiting_adjudication`/
  `failed` runs untouched.
- **Both terminal states** reclaimed (`completed` *and* `cancelled`).
- **Cutoff boundary**: `updated_at` exactly at / just under / just over cutoff.
- **Already-reclaimed skipped** (idempotent second run is a no-op).
- **Missing file tolerated** → stamped, `reclaimed_bytes == 0`.
- **Crash-window reconciliation**: pre-stamp file removed → next sweep stamps
  bytes 0, doesn't error.
- **Source-alias guard**: an artifact row whose `path` also appears as a
  `media_items.source_path` is **not** reclaimed.
- **Tutorial run excluded**: `app_settings.tutorial_run_id`'s WAV survives.
- **Path safety**: a symlink at the artifact path is **not** unlinked (fail
  closed); a path escaping `media_root` is rejected.
- **Permission failure** on a row → that row left unreclaimed, batch still
  reclaims the others (per-row isolation).
- **Batch limit + oldest-first ordering**: with limit N and >N eligible, the
  N oldest are taken; a second sweep drains the rest.
- **Concurrency**: two overlapping `reclaim_*` calls (or `FOR UPDATE SKIP
  LOCKED` behavior) don't double-count / don't clobber `reclaimed_bytes` with
  0 — assert via two sessions on the same row set.

**Console/integration**: run_detail shows the reclaimed notice (not a dead
Audio link); `GET`/`HEAD /media/{id}` returns **410** after reclamation;
non-reclaimed run still links + serves 200.

**Contract**: no version/compose/pin change in this feature → the pin-parity
contract tests should stay green untouched (confirm; add none).

## Rollout / risks / open questions

- **Risk — data deletion.** Mitigated: OFF by default; intermediates only;
  source+transcript+decisions always kept; audit row survives; runs
  re-processable from source.
- **Risk — concurrency clobber.** Mitigated by `FOR UPDATE SKIP LOCKED` +
  claim-then-act; tested.
- **Risk — crash window** loses a byte count (never a file that should live).
  Documented semantic; tested.
- **Out of scope (accurately named):** true **filesystem orphans** (a file
  with *no* DB row — e.g. prepare's delete-without-unlink leaves the file but
  removes the row) are invisible to this row-walk and are a **separate**
  future filesystem-orphan sweep. This feature only reclaims files still
  referenced by an unreclaimed row.
- **Deferred:** dedicated `completed_at` column (see Decision 3); run
  delete/archive (#5).
- **Version bump:** none now — additive feature; bump `0.11.0 → 0.12.0`
  atomically at release time per `docs/release-process.md`.
- **Open (minor, decide at implementation):** exact default numbers
  (30-day TTL / 1-h cadence / 500 batch) are conservative starting points,
  tunable via env; confirm they read sensibly in `.env.example`.

## Review notes (codex critique → resolution)

Codex (planner role, 3-way-quality single pass) flagged the draft as
"directionally sound but not implementation-ready." Resolutions:

- **[critical] False idempotence under overlapping sweeps** → **accepted**:
  `FOR UPDATE SKIP LOCKED` claim-then-act; concurrency test added.
- **[critical] Undefined fs/db crash window** → **accepted**: defined
  `reclaimed_bytes` as measured-at-clean-reclaim / 0-if-absent; reconciliation
  test added.
- **[critical] No guard against deleting a file also registered as source**
  → **accepted**: `NOT EXISTS (media_items.source_path = path)` guard +
  documented layout invariant; test added.
- **[high] Retention-clock semantics** → **decided** (maintainer):
  `updated_at`, documented; `completed_at` alternative deferred with rationale.
- **[high] Path-safety underspecified for deletion** → **accepted**: resolve +
  `is_relative_to` + `lstat` symlink/dir rejection + fail-closed on ambiguous;
  tests added.
- **[high] `batch_limit` unspecified** → **accepted**: `gc_batch_limit`
  setting, oldest-first stable ordering, documented drain rate.
- **[high] Only `FileNotFoundError` handled** → **accepted**: per-row failure
  isolation, fail-closed leaving `reclaimed_at` NULL on non-missing errors.
- **[medium] Schema invariants** → **accepted**: paired-nullability +
  `>= 0` CHECKs; migration tests exercise both.
- **[medium] Off-by-default may not fix growth** → **decided** (maintainer):
  keep OFF by default; operations doc makes enabling loud.
- **[medium] `normalized_audio_path` shared contract** → **accepted**: leave
  the function unchanged; add reclaim-awareness only at the read layer; audited
  callers (stages safe, tutorial run excluded, `/media` → 410).
- **[medium] Observability** → **accepted**: structured sweep summary log +
  per-row failure WARNINGs, not just a counter dict.
- **[medium] Index underspecified** → **accepted**: exact partial-index
  predicate stated; EXPLAIN-validate.
- **[medium] Orphan-scope conflation** → **accepted**: `FileNotFoundError`
  (row w/ absent file) vs true filesystem orphan (no row) named distinctly;
  latter out of scope.
- **Accepted as-is by codex:** row-retention audit design; eligibility limited
  to `preprocessed_audio` + `completed`/`cancelled`; orphan discovery
  out-of-scope; no parity gate.
