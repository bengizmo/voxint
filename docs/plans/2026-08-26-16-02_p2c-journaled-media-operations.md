# Console 2.0 P2c: journaled move, trash, delete (issue #155)

Plan authored 2026-08-26. Track A finale of the Console 2.0 epic (#149). This is
the only slice that mutates or deletes source bytes, so crash safety and honest
degraded UX are the load-bearing requirements, not feature breadth.

## Goal

Give the operator file management over their media library: move a file between
folders, trash and restore it, and empty the trash to reclaim disk. Every
byte-touching step is recorded in a durable `media_operations` journal so a crash
at any filesystem or database boundary leaves the system reconcilable, with no
orphaned file and no lost file. `source_path` stays the immutable acquisition
identity; a move rewrites only the mutable `current_path` (and, on explicit
operator request, `media_folder_id`). Empty-trash is manual only.

## Assumptions and constraints

Taken as given (fixed by prior ADRs and the code survey):

- `media_items.source_path` is the immutable, unique acquisition identity, never
  rewritten. `current_path` (nullable today, seeded equal to `source_path` by the
  ORM default `_seed_current_path` and the 0040 backfill) plus the
  `media_folder_id` FK carry mutable location and logical config scope. Byte reads
  resolve `media_root / current_path` after the split; identity, dedup, and
  display stay on `source_path`. See ADR 0001, ADR 0002.
- ADR 0002 P2b forward constraint: a move updates `current_path`; for a
  path-derived membership it may update `media_folder_id`, but it must never
  clobber an explicit operator override. There is no stored flag today that
  distinguishes path-derived from explicit membership.
- Production is Postgres only. SQLite is a dev and test single-writer engine, so
  advisory locks are dialect-guarded and no-op there. Concurrency correctness is a
  Postgres property and must be tested against Postgres.
- Single operator, but concurrency is still real: two browser tabs, a reconciler
  beat, and a worker can all touch one media row at once.
- Alembic head is 0042, so this slice adds 0043. The single-head test at
  `tests/integration/test_migration_0016.py:129` asserts `["0042"]` and carries an
  inline revision catalog that both need a 0043 line.

What would invalidate the plan: a decision to make trash a pure database flag with
no physical move (rejected below), or a requirement that a move automatically
re-home a path-derived `media_folder_id` (would force a membership-provenance
column; see open question O1).

Verified during survey, correcting the draft:

- `current_path` is written but read by nothing yet. All three byte-openers still
  resolve `source_path`: `prepare.py:29`, `media/integrity.py:124`,
  `acquire.py:71`. The switch to `current_path` is prospective and lands here.
- The location audit is wider than prepare plus integrity. `reclaim.py:200` keys
  its source-alias exclusion guard on `MediaItem.source_path == AudioArtifact.path`
  (ADR 0001 case B); left on `source_path`, GC could delete an artifact path that
  is now a relocated live source.
- `current_path NOT NULL` cannot be tightened in 0043. Integration tests issue raw
  `INSERT INTO media_items (id, source_path)` at head schema (for example
  `test_activity_events.py:39`, `test_notify_delivery.py:90`), and the ORM default
  is not a server default, so a NOT NULL column with no server default fails those
  inserts. A retained purged row also has no live source path. NOT NULL is deferred.
- No contract test pins the byte-opener set, so the switch has no golden to update,
  but ADR 0001's status and audit table must be refreshed and a location-audit
  contract test added (O5).

## Proposed approach

A `media_operations` journal row records the intent and progress of one
byte-touching operation. A dedicated reconciler beat drives interrupted rows to a
consistent terminal state by comparing the recorded intent against filesystem and
database ground truth. Operations serialize per media item through a claim that is
held across the filesystem phase, not just a transaction.

### Concurrency protocol (the correctness core)

Transaction-scoped advisory and row locks release at each commit, so they cannot
protect the filesystem gap between one commit and the next. Two requests or two
reconcilers could otherwise execute the same row. The protocol:

1. A partial unique index enforces at most one non-terminal `media_operations` row
   per `media_id`. A second concurrent request to move the same item fails at
   insert and is reported as "an operation is already in progress", not executed
   twice.
2. Each operation carries a claim lease: `claimed_by` (a per-process or per-request
   token) and `lease_expires_at`. The executor and the reconciler both acquire a
   row by a compare-and-set that also checks the lease, mirroring the StageRun
   lease pattern already proven in `test_cas_and_restart.py`. An expired lease is
   stealable, so a dead executor does not wedge the row.
3. Every state transition is a compare-and-set on the prior state, so a stale actor
   cannot advance a row another actor already moved.
4. Run admission and move share ordering on the `MediaItem` row. Move, trash, and
   purge take `SELECT ... FOR UPDATE` on the media row and refuse if a non-terminal
   run exists (`status NOT IN {completed, failed, cancelled}`, the archivable
   inverse; stricter than the literal "queued or executing" because a requeue
   re-decodes from `current_path`). Run admission (`submit_media_item`,
   `submit_media_item_if_new`, the library rerun) takes the same row lock and
   refuses to admit while a non-terminal operation exists, or while the item is
   trashed or purged. The PREPARE stage defers (re-queues) rather than decodes when
   an unresolved operation exists for its item.

Rationale: this is the minimal set that closes both the executor-versus-executor
race and the move-versus-run race that codex flagged as critical. It reuses the
lease and CAS idioms the codebase already tests, rather than inventing a
session-level advisory lock whose lifetime is the database connection.

### Filesystem publication (durable, no-clobber)

Preflight collision checks followed by `os.replace` are TOCTOU-prone and
`os.replace` clobbers a destination created after the check. Checksum equality at
the destination does not prove this operation created it, so unlinking the origin
on a checksum match could destroy the operator's only copy. The publication
sequence:

- Same device: create the destination with no-clobber semantics (open the parent
  directory, `os.link` or `renameat2` with no-replace where available, else an
  `O_EXCL` create of an operation-owned temp name derived from the operation id
  followed by an atomic rename onto the final name only if the final name is
  absent). Fsync the destination directory. For a move the origin and destination
  are distinct names, so the rename cannot lose the file. Fsync the origin
  directory after unlink.
- Cross device (EXDEV): copy through an opened source descriptor into an
  operation-owned temp path (deterministic from the operation id, so recovery can
  find and clean it), fsync the temp file, verify size and digest, publish
  no-clobber onto the final name, fsync the destination directory, unlink the
  origin, fsync the origin directory.

Voxint never unlinks the origin until it owns and has verified a durable
destination. That is the invariant tests assert, in place of an absolute
no-lost-file claim (which is unassertable, since an external process can always
delete a file).

### State machine

Five states, each with a durable invariant (a state with no invariant is dropped):

- `planned`: journal row committed, no operation-owned destination published yet.
- `fs_applied`: the destination is durably published and the origin disposition is
  known (still present for pre-unlink, or removed). This is the single filesystem
  crash window.
- `db_applied`: `current_path` has been updated by a CAS that required the expected
  prior path. For a same-device move this transition is committed in the same
  transaction as `fs_applied` to remove a redundant database-only crash window
  (codex Q4). `db_applied` is retained as a distinct label only because `completed`
  performs real remaining work.
- `completed`: cleanup and verification finished (temp files removed, origin
  directory verified, purge child-file inventory fully resolved).
- `failed`: an unrecoverable or operator-action-required stop, left at a consistent
  filesystem and database state and surfaced in the recovery panel.

`failed` is split by an `error_code` and a retriable flag, with `attempt_count`,
`last_attempt_at`, and `next_attempt_at`, so a transient permission or mount error
schedules a retry rather than becoming permanently terminal.

Operations chain: a restore records `restores_operation_id` (or validates that
`current_path` equals the trash destination of a completed trash), so a stale move
or restore cannot complete after a newer operation.

### Reconciliation decision table (priority-one deliverable)

Before any 0043 code, ADR 0007 specifies the full decision table. The reconciler
never decides from two path existence checks alone. Inputs: journal state,
`current_path` classified as one of {origin, destination, other}, origin identity
and digest, destination identity and digest, operation-owned temp state, media_root
and mount availability, and whether a newer operation for the item exists. The
`current_path` update is always a CAS with the expected prior pointer, so a stale
reconciler cannot overwrite a later pointer. If media_root or the relevant mount is
unavailable, the reconciler aborts that row (following the
`MediaRootUnavailableError` precedent) rather than classifying an absent file as
lost.

### Trash, restore, purge

- Trash is a move whose destination is a managed subtree inside media_root, named
  by operation id so the operation owns it. `current_path` follows into the trash
  tree, so existing playback (which serves normalized `AudioArtifact` bytes, not
  the source) keeps working while trashed. The trash tree name is added to
  `_RESERVED_TREES` (`setup_wizard.py:58`), the single choke point that excludes it
  from every scan, so the watcher never re-ingests trashed files. A deletion
  deadline is recorded for display only; there is no auto-purge.
- Restore is a journaled move back to the recorded original location, refusing and
  reporting operator action required on a destination collision rather than
  overwriting.
- Purge (empty-trash, manual) is designed separately from move because a single
  origin and destination row cannot track multi-file deletion. Purge builds a
  durable per-file manifest (a `media_operation_files` child table) enumerating the
  source and every derived target across the item's runs (AudioArtifact,
  AudioChunk, peaks, clips), commits it, then unlinks each target and marks that
  child row done, missing, or failed, then, only after every child is resolved,
  deletes the artifact database rows and sets `media_items.purged_at`. The existing
  `delete_run_derived_media` is not reused as-is, because it deletes the artifact
  rows before the best-effort unlink, which would lose the retry inventory on a
  crash. Run pages keep transcript text and run history and return an honest 410
  Gone from the source, peaks, and clip endpoints, with capability banners driven
  off the same `purged_at` seam.

### Sidecar as a bundled secondary file (O4)

A sidecar YAML sits beside a media file in a watched folder and carries per-file
configuration. The operator decision is that it belongs to the media bundle, so a
move, trash, and restore relocate it with the file, and `_reread_sidecar` reads
beside `current_path`, not `source_path`.

Crash safety: two files cannot move atomically, so the sidecar is a bundled
secondary target, not a co-equal of the media. The media is the critical file and
is published and verified first through the full no-clobber, fsync sequence. The
sidecar is then relocated best-effort, with its own origin and destination recorded
in the operation's `media_operation_files` manifest (the same child table purge
uses), so the reconciler can complete or roll back the sidecar move independently.
A missing or unmovable sidecar is a warning, never a failure of the media
operation, because the sidecar's effect is already frozen into any past run
snapshot and a rerun that finds no sidecar behaves exactly as an unsidecar'd file
does today. This keeps the media-loss invariant strictly about the media file while
still moving the sidecar with it in the normal case.

### Rejected alternatives

- Database-first ordering (update `current_path`, then move bytes). Rejected:
  `current_path` is the pointer byte-openers use, so a crash after the pointer
  update but before the move leaves the pointer aimed at a nonexistent
  destination. Filesystem-first keeps the pointer valid until the destination is
  durably published, and the reconciler rolls forward from filesystem truth.
- Trash as a pure database soft-delete flag, no physical move. Rejected: the issue
  requires bytes to leave the watched folders so the sweep stops re-emitting them
  and so disk is reclaimable. A flag alone leaves bytes where the watcher re-ingests
  them.
- Killing a real process for crash tests. Rejected: the house pattern
  (`test_cas_and_restart.py`) constructs the intermediate state deterministically
  and runs the recoverer, which is CI-friendly and reproducible. This slice extends
  that pattern with executor hooks at each filesystem boundary.

## Affected files and components

New:

- `alembic/versions/0043_media_operations.py`: the journal table, the
  `media_operation_files` child table (per-file progress for both the purge
  inventory and a move's bundled sidecar), `media_items.purged_at`, CHECK
  constraints, the partial unique index (one active op per item), and a reconciler
  index on (state, next_attempt_at). No `current_path NOT NULL`.
- `src/voxint/media/operations.py`: the executor (move, trash, restore, purge), the
  claim and CAS helpers, the filesystem publication primitives, and the pure
  helpers (trash-path builder, filesystem-reality classifier, decision-table
  function, transition guards).
- `src/voxint/media/reconcile.py` (or a function in operations.py): the reconciler
  driving non-terminal rows to a terminal state per the decision table.
- `docs/adr/0007-media-operations-journal.md`: the state machine, the concurrency
  protocol, the reconciliation decision table, the EXDEV durable sequence, the
  purge manifest, and the purged-media serving contract.
- `tests/unit/test_media_operations.py`, `tests/integration/test_media_operations.py`,
  and crash-injection and Postgres two-session concurrency tests.

Changed:

- `src/voxint/db/models.py`: `MediaOperation` and `MediaOperationFile` models,
  `MediaItem.purged_at`.
- `src/voxint/pipeline/stages/prepare.py`, `src/voxint/media/integrity.py`,
  `src/voxint/media/reclaim.py`: switch the byte-location reads and the reclaim
  alias guard from `source_path` to `current_path`, through one shared
  `openable_current(media_root, media)` helper.
- `src/voxint/ingest/service.py`: run admission shares the `MediaItem` row lock and
  refuses admission under a non-terminal operation, trash, or purge; new refusal
  error types in the `IngestError` hierarchy.
- `src/voxint/worker/app.py`, `src/voxint/worker/tasks.py`: register
  `voxint.media_reconcile` in the beat schedule and task routes.
- `src/voxint/api/setup_wizard.py`: add the trash tree to `_RESERVED_TREES`.
- `src/voxint/api/routers/media.py` and templates: move, trash, restore, and
  empty-trash routes; the trash view; the recovery panel; CSRF actions; honest copy.
- `src/voxint/api/playback.py`, `src/voxint/media/serving.py`,
  `src/voxint/media/peaks.py` consumers: 410 Gone when `purged_at` is set.
- `src/voxint/api/routers/media.py` rerun sidecar read (`_reread_sidecar`): read
  beside `current_path` (O4), and the executor relocates the sidecar as a bundled
  secondary file on move, trash, and restore.
- `docs/adr/0001-media-identity-vs-location.md`: status and audit refresh.
- `tests/integration/test_migration_0016.py`: single-head bump and catalog line.
- `CHANGELOG.md`, `docs/architecture.md`, `.env.example` (trash tree name, any new
  reconciler cadence setting).

## Step-by-step implementation

Ordered so each commit is individually reviewable and green. This is a multi-session
slice; do not attempt it in one sitting.

0. ADR 0007, docs only. Write the state machine, concurrency protocol, full
   reconciliation decision table, EXDEV durable sequence, purge manifest, and the
   purged-media serving contract. Refresh ADR 0001 (re-grep the `source_path`
   audit, add reclaim aliasing and the sidecar read). This is the priority-one
   deliverable and gates the code.
1. Migration 0043 plus models plus pure helpers. `media_operations`,
   `media_operation_files`, `media_items.purged_at`, CHECKs, partial unique index,
   reconciler index. Single-head bump. The pure helpers in operations.py with unit
   tests. No executor, no routes.
2. Concurrency and location primitives. The claim and CAS helpers; run admission
   sharing the `MediaItem` row lock and refusing under a non-terminal operation or
   trash or purge; PREPARE deferral; the `openable_current` helper; the byte-opener
   switch in prepare, integrity, and reclaim.
3. Move, trash, restore executor. No-clobber publication, the EXDEV fsync protocol,
   CAS transitions, the lease, predecessor chaining, and the refusal error types.
4. Reconciler and beat. The decision table implementation with CAS pointer updates,
   lease stealing, and the root-availability guard; the dedicated
   `voxint.media_reconcile` task and route; the trash tree added to
   `_RESERVED_TREES`; the recovery-panel query.
5. Purge. The durable per-file manifest flow; honest 410 serving for source, peaks,
   and clips; blocking runs while trashed or purged.
6. Library UI. Move, trash, restore, and empty-trash routes; the trash view; the
   recovery panel; CSRF actions; honest copy; route goldens regenerated
   additions-only; the browser acceptance lane.
7. Watcher missing-file warning (a read-time regular-file check that surfaces a
   library warning, never a crash); docs; CHANGELOG.
8. Crash-injection, Postgres two-session concurrency, and adversarial path tests,
   then a full three-engine review.

## Testing strategy

- Unit (`tests/unit/test_media_operations.py`): the pure helpers, the
  filesystem-reality classifier over every {origin, destination, temp} combination,
  the decision-table function, transition guards, and digest-mismatch handling.
- Integration over a `tmp_path` media root (`tests/integration/test_media_operations.py`):
  move updates `current_path` and leaves `source_path` unchanged; move refused while
  a non-terminal run exists; destination collision refused; EXDEV forced by patching
  the transfer primitive selectively (not every `os.replace`) exercises the copy
  path; trash then restore is byte-identical by digest; empty-trash removes source,
  derived artifacts, peaks, and clips, sets `purged_at`, and leaves an honest 410
  and a readable degraded run page; the watcher skips the trash tree; a missing
  `current_path` surfaces a warning, not a crash. A sidecar present beside a moved
  or trashed file is relocated with it and re-read beside `current_path`; a missing
  or unmovable sidecar is a warning, not an operation failure; a restore brings the
  sidecar back; empty-trash removes it via the manifest.
- Crash-injection (deterministic executor hooks after journal commit, temp fsync,
  destination publish, destination-directory fsync, origin unlink, origin-directory
  fsync, pointer commit, and purge-child completion): run restart recovery through
  each hook and assert convergence to a consistent terminal state, with the
  origin-never-unlinked-before-verified-destination invariant holding at every
  boundary. Assert fail-closed on media_root unavailability rather than
  classifying loss.
- Postgres two-session concurrency: duplicate execution attempts converge to one
  physical operation; move versus rerun cannot admit a run into the pointer gap;
  reconciler versus reconciler; and a stale CAS is rejected.
- Adversarial paths: symlink leaf and parent swaps, destination directories,
  absolute and `..` paths, canonicalization and case collisions, hard links,
  permission errors, source mutation during hashing, and a pre-existing
  operation-owned temp file.
- Contract and migration: a new location-audit contract test pinning that the
  byte-openers and the reclaim alias resolve `current_path`; the single-head test;
  the `media_operations` CHECK and enum constraints; and a migration test covering
  upgrade with a NULL `current_path`, raw head-schema insert behavior, the purge
  representation, indexes, and downgrade from a journal in every state.
- Gates each commit: ruff check, mypy, and the full pytest suite at `-n 8` against
  the Postgres test container. The browser acceptance lane on commit 6 (this alters
  observable console behavior and the delivery contract islands depend on). A full
  three-engine `/code-review` on the whole slice: this is unambiguously high blast
  radius (a migration, byte mutation and deletion, concurrency, crash safety, and a
  location contract), so the deepest review tier is mandatory.

## Rollout, risks, and open questions

Rollout: the library UI stays dark behind the existing `console_media_enabled`
flag, so the operation routes and the reconciler beat are inert until an operator
opts in. No release is cut by this slice.

Risks:

- The concurrency protocol is the highest-risk surface. It must be validated on
  Postgres, not SQLite. The partial unique index plus lease plus CAS plus shared
  row lock must all be present; dropping any one reopens a race.
- The byte-opener switch is behavior-changing but byte-identical until a real move,
  because `current_path` equals `source_path` for every existing row. The parity
  references are unaffected (same bytes, same resolved path). Still, the reclaim
  alias switch must be tested to confirm GC does not delete a relocated live source.
- Purge is irreversible. The manifest-first design is what makes a partial purge
  recoverable and honest rather than silently orphaning files.

Operator decisions, resolved 2026-08-26 (O1, O3, O4 answered; O2 and O5 keep their
recommended defaults):

- O1 (resolved: preserve membership). A move updates `current_path` only and never
  changes `media_folder_id`, honoring the ADR 0002 MUST-NOT-clobber rule without a
  provenance column. An explicit "move and reassign to the destination folder" is a
  separate operator action that reuses the P2b bulk assign. This is a deliberate
  deviation from the literal issue wording ("updating current_path and
  media_folder_id"), recorded here and to be noted in the PR.
- O2 (default: yes). Purged items are hidden from the active and trash views while
  their runs stay reachable under /runs with degraded media.
- O3 (resolved: 410 Gone). The purged source, peaks, and clip endpoints return 410
  Gone, and capability banners key off the same `purged_at` seam.
- O4 (resolved: the sidecar moves with the file). A sidecar is treated as part of
  the media bundle, not fixed acquisition metadata. A move or trash relocates the
  sidecar alongside the media, a restore brings it back, and `_reread_sidecar`
  reads beside `current_path`, not `source_path`. See "Sidecar as a bundled
  secondary file" below for the crash-safety handling.
- O5 (default: defer NOT NULL). `current_path NOT NULL` stays deferred (raw
  head-schema inserts and retained purged rows both violate it). Tighten later only
  after adding a server default, updating the raw inserts, and deciding how a
  purged row represents no live source.

## Review notes

A codex planner second opinion (zen clink, role planner) reviewed the draft and
returned a structured critique. Its findings and their resolution:

- Lock lifetime (critical). Transaction-scoped locks release before the filesystem
  work, so they cannot serialize execution. Accepted. Added the partial unique
  index plus claim lease plus CAS transitions, and made run admission share the
  `MediaItem` row lock.
- Run race (critical). The active-run preflight was racy against run admission.
  Accepted. Run admission and move now share row-lock ordering, and PREPARE defers
  under an unresolved operation.
- Destination safety (critical). `os.replace` clobbers, and checksum equality does
  not prove destination ownership. Accepted. Switched to no-clobber publication and
  operation-owned destinations, and the reconciler never unlinks an origin on a
  bare digest match.
- Cross-device durability (critical). EXDEV needs an operation-owned temp path and
  file plus directory fsyncs. Accepted. Specified the full durable sequence.
- Purge journaling (critical). Deleting artifact rows before the best-effort unlink
  loses the retry inventory. Accepted. Purge now builds a durable per-file manifest
  child table first and reuses `delete_run_derived_media` only after unlink
  convergence, not before.
- Reconciliation ground truth (critical). Filesystem presence alone is
  insufficient. Accepted. The reconciler decision table now includes `current_path`,
  journal state, digests, ownership, root availability, and newer-operation
  existence, with a CAS pointer update. Writing the table is the priority-one
  deliverable.
- State machine. `db_applied` was declared but never committed; `failed` was
  underspecified; chaining and both-absent were unmodeled. Accepted. Each state now
  has a durable invariant, same-device folds `fs_applied` and the pointer update
  into one transaction, `failed` gained retry scheduling and an error code,
  restore chains to its trash operation, and the both-absent test asserts
  fail-closed with a mount-availability check rather than impossibility.
- Byte-opener blast radius. reclaim aliasing and the rerun sidecar read were
  missing from the audit, and a centralized live-path resolver was absent. Accepted.
  Added reclaim to the switch set, added the `openable_current` helper, and raised
  the sidecar decision as O4. Verified in source that playback serves
  `AudioArtifact` bytes, so trash playback is independent of the source-opener
  switch, and that a location-audit contract test is needed.
- Q1 (media_folder_id): preserve by default, explicit reassign instead of
  provenance. Accepted as O1.
- Q2 (purged_at): keep as denormalized authoritative state, set at convergence.
  Accepted.
- Q3 (NOT NULL): defer. Accepted and independently verified: raw
  `INSERT INTO media_items (id, source_path)` exists in head-schema integration
  tests, so a NOT NULL column with no server default would break them.
- Q4 (transaction collapse): combine `current_path` with the `fs_applied` commit.
  Accepted.
- Q5 (reconciler home): a dedicated task, block or defer PREPARE, and avoid a
  worker-boot stampede by using the same claim protocol. Accepted.

Nothing from the critique was rejected outright. The remaining judgment calls are
carried as O1 through O5 for the operator.
