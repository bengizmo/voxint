# Plan: Speaker-identification activity events (#162 P7)

_Console 2.0 P7, issue #162. Extends the activity outbox (run-completion shipped
in PR #193) to announce speaker identifications. Drafted 2026-08-26, codex
second opinion folded in (see Review notes)._

## Goal

Announce **speaker identifications** in the console activity indicator (aria-live
toasts) when an operator assigns a diarization label to a speaker, enrolls a new
speaker, or merges labels into one. Dark-shipped behind the **existing**
`CONSOLE_ACTIVITY_ENABLED` flag (default off), reusing the shipped
`activity_events` outbox, poll endpoint, client poller, and retention beat. A
merge fans out N label decisions; the operator sees **one** announcement, not N.
No version bump (matches #160 / #162-item-1 precedent; still dark-shipped).

## Assumptions & constraints

- Reuse everything already shipped: one table, one endpoint, one poller, one flag,
  one retention beat. No new table, endpoint, or flag (anti-bloat).
- Emit is persistence-only, in the caller's transaction, idempotent via
  `occurrence_key` ON CONFLICT DO NOTHING — mirror `record_run_completed`.
- `pipeline_run_id` stays **NOT NULL** (every speaker decision names a run), so
  migration 0043 adds **no** nullable change and **no** typed provenance columns.
  The frozen `title`/`href` snapshot is enough for a toast. (A stale docstring in
  0042 foreshadowed "nullable + typed provenance"; that guess is dropped and the
  docstrings are corrected.)
- The Jobs **badge stays a live-jobs count** (`jobs_badge_count`, nonterminal
  runs). Speaker identifications do **not** change the badge — they affect toasts
  only. An "unread activity" badge would need per-event read-state, which is
  bloat for a single operator. Pinned by a regression test.
- Invalidating conditions: if the badge must reflect unread speaker activity, or
  if typed provenance gains a real reader, this plan is wrong and the schema
  changes.

## Proposed approach

### 1. What announces (event semantics)

Only **positive identifications** that **change effective attribution**:

| Path | Emits? | Rule |
|---|---|---|
| Label-scope `ASSIGN` | yes, guarded | skip if the label's current effective `speaker_id` already equals the new one (a cheap guard: `LabelState.speaker_id` is loaded pre-write in the route). Re-assign to a **different** speaker announces. |
| Segment-scope `ASSIGN` | yes | a per-segment override is always a deliberate, specific action; emit one event with the loaded speaker name. |
| Standalone enroll (new speaker) | yes | a brand-new speaker is by definition a new identification. |
| Merge (N labels → survivor) | yes, **one** event | one deliberate consolidation = one announcement (server-side coalescing). |
| `EXCLUDE` / `UNKNOWN` / `INHERIT` | **no** | corrections, not identifications — silent (keeps the feed meaningful). |

Rationale for the label-scope guard: a fresh nonce re-asserting the
already-effective speaker writes a new ledger row; without the guard it would
toast despite no attribution change (codex medium). The guard is free at label
scope. Segment scope emits on ASSIGN (re-affirming one segment via a fresh nonce
is rare and low value to suppress; no cheap pre-write effective read there).

### 2. Emit placement — orchestration layer, ledger stays pure

Emit where `settings` is already in hand (the routes), never inside
`record_decision`. This mirrors `record_run_completed` being called from
`cas_update_run`, not from the low-level writer. `record_decision` keeps taking no
`settings`. The completeness obligation (a future `record_decision` caller could
forget to emit) is handled by a documented invariant note near `record_decision`
and a call-site inventory in the tests / review notes (codex low).

### 3. Schema — migration 0043 (CHECK widen only)

- Add `SPEAKER_IDENTIFIED = "speaker_identified"` to `ActivityKind` (the **model**
  CheckConstraint uses `_enum_values(ActivityKind)` so it auto-tracks; the literal
  **migration** CHECK is edited by hand).
- `upgrade()`: drop `activity_events_kind_check`, add it back as
  `kind IN ('run_completed','speaker_identified')`.
- `downgrade()`: **delete only `speaker_identified` rows** (run_completed rows
  survive), then restore the narrow CHECK. Documented as a destructive,
  speaker-event-losing downgrade.
- Single Alembic head becomes `0043`.

### 4. New helpers in `activity/__init__.py`

- `record_speaker_identified(session, *, run_id, decision_id, speaker_name)`:
  `occurrence_key = f"decision:{decision_id}:identified"` (decision UUID is
  globally unique and stable under replay), `title = speaker_name` (clamped to
  500), `href = f"/jobs/{run_id}"`.
- `record_speaker_merge(session, *, run_id, occurrence_decision_id, survivor_name, label_count)`:
  `occurrence_key = f"merge:{occurrence_decision_id}"` where
  `occurrence_decision_id = min(MergeResult.decision_ids.values())` — a stable,
  globally-unique per-merge key (**not** `merge:{nonce}`: a nonce is not
  collision-safe across runs / label sets, codex high). `title = f"{survivor_name} ({label_count} labels)"`,
  `href = f"/jobs/{run_id}"`. `label_count = len(MergeResult.labels)` (apply_merge
  deduplicates, so use the resolved set, not raw form input).

Both go through the existing `record_activity_event` (same ON CONFLICT path).

### 5. Emit call sites (all gated `if settings.console_activity_enabled`, same tx)

- `legacy_review.py` label-scope route (~823): capture `row = record_decision(...)`;
  if `decision is Decision.ASSIGN` and the pre-write `LabelState.speaker_id` for
  this label != `speaker_id`, emit `record_speaker_identified(decision_id=row.id,
  speaker_name=<loaded Speaker.display_name>)`.
- `legacy_review.py` segment-scope route (~1062): capture `row`; if
  `decision is Decision.ASSIGN`, emit with the FOR-SHARE-loaded speaker name.
- enroll route (~1281): capture `enrollment = enroll_new_speaker(...)`; load the
  authoritative `Speaker` by `enrollment.speaker_id` (**not** the submitted
  `display_name` — enroll ignores submitted names on replay, codex medium), emit
  `record_speaker_identified(decision_id=enrollment.decision_id, speaker_name=<loaded>)`.
- `merge_apply` route (~909): capture `result = apply_merge(...)`; emit exactly one
  `record_speaker_merge(...)`. Enrollment inside a merge does **not** double-emit
  (only the merge route emits for a merge; the enroll route emits for standalone
  enroll).

### 6. Client (`base.html` poller) — kind-aware copy

- `renderToast(event)` (pass the event, not `title,href`): branch on `event.kind`.
  `run_completed` keeps "Transcription finished: {title}. Review speakers." +
  "View"; `speaker_identified` → "Speaker identified: {title}." + "View" (→ href).
- `renderCoalesced` becomes kind-aware: homogeneous run-only keeps
  "N transcriptions finished"; homogeneous speaker-only → "N speakers identified";
  mixed → neutral "N updates. Open Jobs." All link to `/jobs`.
- `safeHref` unchanged (relative single-slash only). `flush` still coalesces at
  `> MAX_TOASTS (3)`.

## Affected files

- `alembic/versions/0043_widen_activity_kind.py` — new migration (CHECK widen +
  speaker-row-only downgrade).
- `src/voxint/db/models.py` — add `SPEAKER_IDENTIFIED`; correct the
  `ActivityKind` / `ActivityEvent` docstrings (drop the "nullable + typed
  provenance" foreshadow).
- `src/voxint/activity/__init__.py` — two new helpers; update module docstring
  (speaker events are no longer "deferred").
- `src/voxint/api/routers/legacy_review.py` — 3 emit sites (label ASSIGN guarded,
  segment ASSIGN, enroll) + capture return values.
- `src/voxint/api/routers/activity.py` — docstring note only (endpoint unchanged;
  already returns `kind`).
- `src/voxint/adjudication/ledger.py` — a comment documenting the orchestration
  emit invariant (no behavior change; ledger stays settings-free).
- `src/voxint/api/templates/base.html` — kind-aware `renderToast` + coalesce copy.
- `CHANGELOG.md` — `[Unreleased]` entry.
- `docs/` — update `docs/` note on the activity feature if one exists (check
  `docs/operations` / console docs); stale docs are bugs.

## Step-by-step implementation (TDD, contract tests in the same commit)

1. **Migration + enum (gate 1).** Add enum value; write 0043; write
   `tests/integration/test_migration_0043.py` (accepts both kinds; rejects a
   genuinely-unknown kind e.g. `"bogus_kind"`; up/down/up roundtrip; downgrade
   deletes only speaker rows and preserves run_completed rows). Fix
   `test_migration_0042.py` (its "unknown kind" example currently **is**
   `speaker_identified` — change to `"bogus_kind"`). Bump the single-head pin in
   `test_migration_0016.py` from `["0042"]` to `["0043"]`.
2. **Helpers (gate 2).** `record_speaker_identified`, `record_speaker_merge`.
   Extend `test_activity_events.py`: title clamp at 500, replay → one row,
   collision-safe merge key (same-nonce different runs/label-sets → distinct
   events), and a transaction-rollback test (roll the caller's tx back → the
   event is gone).
3. **Route wiring (gate 3).** Wire the 4 sites; capture return values. Add
   route/integration tests: assign / segment-assign / enroll / merge each emit
   correctly; EXCLUDE / UNKNOWN / INHERIT stay silent; flag-off emits nothing;
   exact replay (same nonce) → one decision + one event; merge emits exactly one;
   enroll-inside-merge does not double-emit; the label-scope no-op guard (re-assign
   same speaker → no event, re-assign different speaker → event);
   enroll rename-then-replay freezes the **current** name, not stale form input;
   a same-tx rollback at the route removes both decision and event; a speaker
   identification leaves `jobs_badge_count` unchanged (badge regression).
4. **Client + browser (gate 4).** Kind-aware `renderToast` / coalesce copy. Run
   the browser acceptance lane (`voxint-e2e-review` skill) covering: one toast of
   each kind, homogeneous speaker coalescing (4 events → speaker copy), mixed-kind
   coalescing (neutral copy), the 3-vs-4 `MAX_TOASTS` boundary, `safeHref`
   rejection, live-region insertion, and cursor advancement. (Codex high: a
   template-substring assertion cannot validate kind dispatch or coalescing —
   these ARE the feature.)
5. **Docs + changelog + docstring corrections.**

## Testing strategy / gates

- Gates: `uv run ruff check .`, `uv run mypy src`, full suite `-n 8`, gitleaks,
  and the standing internal-hostname/IP/credential grep over the branch diff
  (expect empty).
- **Review depth: multi-model** (new emit seam across 4 sites + a migration
  widening a CHECK = high blast radius per CLAUDE.md).
- **Browser acceptance lane: run it** (the poller copy/dispatch is observable
  review-console behavior; codex high pushed back on skipping).

## Rollout / risks / open questions

- **Deployment order** (codex medium): an already-open old page would render a new
  speaker event as "Transcription finished" (its `renderToast` ignores kind).
  Mitigation: the flag stays default-off through migration + app rollout; refresh
  the operator session before enabling. Sufficient for a single-operator install.
- **Open question for Ben (non-blocking):** the label-scope no-op guard makes a
  re-affirm of the same speaker silent. If you'd rather every positive ASSIGN
  toast (simpler, slightly noisier), say so and I drop the guard. Default: keep
  the guard.
- **Completeness obligation:** future `record_decision` callers must remember to
  emit. Mitigated by a documented invariant note + a call-site inventory in tests.

## Review notes (codex planner critique, 2026-08-26)

Codex assessed the direction "sound … with several correctness and test gaps."
Resolutions:

- **`merge:{nonce}` not collision-safe (high) — ACCEPTED.** Merge ledger keys
  include a label-set digest; a nonce can span distinct valid merges. Switched to
  `merge:{min(decision_ids)}` (a stable per-merge decision UUID); added a
  same-nonce-different-runs/label-sets test.
- **JS-shape assertion too weak (high) — ACCEPTED.** Upgraded to the full browser
  acceptance lane covering kind dispatch + homogeneous/mixed coalescing + safeHref
  + live-region + cursor.
- **Badge semantics undefined (high) — ACCEPTED/ADJUDICATED.** Badge stays a
  live-jobs count; speaker events affect toasts only; pinned by a regression test.
- **No-op ASSIGN noise (medium) — ACCEPTED, scoped.** Announcements track
  effective-attribution change at label scope (cheap guard via loaded
  `LabelState.speaker_id`); segment/enroll/merge emit on the deliberate action.
- **Capture return values (medium) — ACCEPTED.** All four sites assign the
  returned row/result.
- **Enroll display_name unsafe on replay (medium) — ACCEPTED.** Resolve the
  authoritative Speaker by `EnrollmentResult.speaker_id`; added a
  rename-then-replay test.
- **Homogeneous speaker coalescing copy (medium) — ACCEPTED.** `renderCoalesced`
  made kind-aware (run-only / speaker-only / mixed); 3-vs-4 boundary tested.
- **Atomicity/replay test matrices (medium) — ACCEPTED.** Added same-tx rollback +
  exact-replay tests for all four seams.
- **Downgrade contract (medium) — ACCEPTED.** Delete only speaker rows; assert
  run_completed survive.
- **Deployment order (medium) — ACCEPTED.** Documented (flag off through rollout).
- **Merge label copy ambiguity (low) — ACCEPTED.** "{name} ({label_count} labels)"
  using `len(MergeResult.labels)`.
- **Stale docstrings (low) — ACCEPTED.** Corrected across models / activity /
  migration commentary (0042 code left immutable except its test fixture).
- **Schema decision (low) — CONFIRMED.** No nullable change, no provenance columns.
- **Route-emit completeness (low) — ACCEPTED.** Documented invariant + call-site
  inventory in tests.

Rejected alternatives (codex agreed): emit inside `record_decision` (needs config
in the ledger, cannot coalesce a merge); add provenance columns (no reader);
client-side merge coalescing (no reliable grouping field without new API surface).
