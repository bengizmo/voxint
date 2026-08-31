# ADR 0007: Media operations journal

> **Status:** Accepted (Console 2.0 P2c, issue #155). Migration 0044, the
> executor, the reconciler, and the library UI land in the same slice.

## Context

Console 2.0 Track A adds file management: the operator can move a media file
between folders, trash it, restore it from trash, and empty the trash to reclaim
disk. These are the first operations that mutate or delete source bytes. A crash
at any filesystem or database boundary can leave the system in a state where a
file exists at neither its old nor its new location, or where the database
pointer disagrees with the filesystem. Without a durable record of what was
intended, recovery requires manual inspection.

The pins that constrain the design:

- `source_path` is the immutable acquisition identity, never rewritten
  (ADR 0001). Physical location is the mutable `current_path`, seeded equal to
  `source_path` at ingest. Byte-opening reads switch to `current_path` in this
  slice.
- `media_folder_id` is a logical config scope (ADR 0002 P2b addendum), not a
  physical location tracker. A move updates `current_path` only and never
  clobbers an explicit operator membership override (operator decision O1).
- Single operator, but concurrency is real: two browser tabs, a reconciler beat,
  and a pipeline worker can all touch one media row at once.
- Production is Postgres only. Advisory locks and partial unique indexes are
  Postgres features; SQLite is a dev and test single-writer engine.

## Decision

Every byte-touching operation is recorded in a `media_operations` journal table.
The journal row carries the full intent (operation type, origin path, destination
path, digests) and tracks progress through a state machine. A dedicated
reconciler beat drives interrupted rows to a consistent terminal state by
comparing the recorded intent against filesystem and database ground truth.

The load-bearing invariant: **Voxint never unlinks the origin until it owns and
has verified a durable destination.** That is the invariant tests assert. An
absolute "no file was ever lost" claim is unassertable (an external process can
always delete a file), but the system never creates a window where its own
operations leave both the origin absent and the destination unverified.

### 1. State machine

Five states, each with a durable invariant:

| State | Invariant | Transition |
|---|---|---|
| `planned` | Journal row committed. No operation-owned destination published yet. Origin bytes are at `current_path` (unchanged). | On successful filesystem publication, advance to `fs_applied`. |
| `fs_applied` | The destination is durably published (fsynced to disk) and its digest verified. The origin disposition is known (still present for pre-unlink, or already removed by a same-device rename). This is the single filesystem crash window. | On successful `current_path` CAS update, advance to `db_applied`. For a same-device move, `fs_applied` and the pointer update commit in one transaction (no redundant database-only crash window). |
| `db_applied` | `current_path` has been updated by a CAS that required the expected prior path. The database and filesystem agree on the live location. Origin may still be present (pending unlink and cleanup). | On cleanup completion (origin unlinked, temps removed, directory fsynced), advance to `completed`. |
| `completed` | Terminal. Cleanup finished: temp files removed, origin directory verified, purge child-file inventory fully resolved. | None. |
| `awaiting_retry` | Non-terminal. A transient error (permission, mount, I/O) occurred and the operation is waiting for a scheduled retry. The filesystem and database are at a consistent state. The row stays inside the partial unique index, so no new operation can be admitted for this item while a retry is pending. | When `next_attempt_at` has passed, the reconciler steals the lease and re-enters from the last durable state (`planned` or `fs_applied`). |
| `failed` | Terminal. An unrecoverable or operator-action-required stop. The filesystem and database are at a consistent state: either the operation was rolled back (origin intact, no destination) or rolled forward (destination published, pointer updated). | None. The operator must resolve the condition and issue a new operation. |

`awaiting_retry` carries an `error_code` enum, `attempt_count`,
`last_attempt_at`, and `next_attempt_at`, so a transient permission or mount
error schedules a retry with exponential backoff. The partial unique index
covers non-terminal states including `awaiting_retry`, so no second operation
or run admission can slip in while a retry is pending. `failed` is strictly
terminal and excluded from the index, freeing the item for a new operation.

Operations chain: a restore records `restores_operation_id` pointing at the
completed trash operation it reverses, so the reconciler can validate that the
restore's origin (the trash destination) is consistent. A stale operation whose
item has a newer non-terminal operation cannot complete: the CAS on
`current_path` rejects the stale expected-prior value.

### 2. Concurrency protocol

Transaction-scoped advisory and row locks release at each commit, so they cannot
protect the filesystem gap between one commit and the next. Two requests or two
reconcilers could otherwise race on the same row. The protocol uses four
interlocking mechanisms:

**2a. Partial unique index (at most one active operation per item).**
A partial unique index on `media_operations(media_id)` filtered to non-terminal
states (`state NOT IN ('completed', 'failed')`) enforces at the database level
that a second concurrent request for the same item fails at insert. This covers
`planned`, `fs_applied`, `db_applied`, and `awaiting_retry`. The caller reports
"an operation is already in progress", never executes a second operation.

**2b. Claim lease (stale executor detection).**
Each operation carries `claimed_by` (a per-process or per-request token) and
`lease_expires_at`. The executor sets these at insert. Nothing renews the lease
during long-running filesystem work yet: a pass that outlives its lease is
stealable mid-flight, and adding renewal plus per-action rechecks is tracked in
issue #353. The reconciler steals an expired lease by a CAS on
`claimed_by` that also checks `lease_expires_at < now()`. A dead executor
(crashed process, killed request) does not wedge the row: once its lease
expires, the reconciler takes over. This mirrors the `StageRun` lease pattern
proven in `test_cas_and_restart.py`.

**2c. Compare-and-set on state (no stale advancement).**
Every state transition is a `WHERE state = :expected_state AND claimed_by = :me`
update. A stale actor whose CAS returns zero rows affected knows it lost the
race and aborts without further side effects. No actor can advance a row another
actor already moved past.

**Lease fencing limitation and mitigation.** The lease detects stale ownership
but does not fence filesystem side effects retroactively: a paused executor can
resume after its lease expires, perform a filesystem write (a no-clobber
publish or a temp-file create), and only discover at the subsequent CAS that it
lost ownership. No-clobber publication limits the damage (it cannot overwrite a
destination the new owner created), but a stale executor could create an orphaned
temp file. Mitigation: (a) the executor re-checks `claimed_by` and
`lease_expires_at` immediately before every destructive filesystem action
(origin unlink, temp cleanup), aborting if the lease is no longer held; (b) the
reconciler cleans operation-owned temp files (deterministic naming) as part of
its convergence pass; (c) origin unlinks happen only after the pointer CAS
succeeds, so a stale executor whose CAS fails never unlinks. This is weaker
than a session-level advisory lock held across the filesystem phase, but avoids
the connection-lifetime coupling that the plan explicitly rejected.

**2d. Shared row lock on MediaItem (move-versus-run serialization).**
Move, trash, restore, and purge take `SELECT ... FOR UPDATE` on the `MediaItem`
row and refuse to proceed if a non-terminal pipeline run exists for that item
(`status NOT IN ('completed', 'failed', 'cancelled')`). Run admission
(`submit_media_item`, `submit_media_item_if_new`, the library rerun action) takes
the same row lock and refuses to admit while a non-terminal operation exists, or
while the item is trashed or purged. The PREPARE stage defers (re-queues) rather
than decodes when an unresolved operation exists for its item. This closes both
the executor-versus-executor race and the move-versus-run race.

### 3. Filesystem publication (durable, no-clobber)

Preflight collision checks followed by `os.replace` are TOCTOU-prone, and
`os.replace` clobbers a destination created after the check. Checksum equality at
the destination does not prove this operation created it, so unlinking the origin
on a checksum match could destroy the operator's only copy.

**Same-device sequence:**

1. Open the parent directory of the destination.
2. Create the destination with no-clobber semantics. The preferred primitive is
   `os.link` (hard link from origin to destination, atomic, fails if destination
   exists). If the filesystem does not support hard links across the relevant
   paths, use `renameat2` with `RENAME_NOREPLACE` (Linux 3.15+, atomic
   no-clobber rename). The implementation must not fall back to a
   check-then-rename sequence, which is TOCTOU-prone.
3. `os.fsync` the destination directory.
4. For a move: unlink the origin, then `os.fsync` the origin directory.

For a same-device move, steps 1 through 3 and the `current_path` CAS commit
happen in one transaction: the state advances directly from `planned` to
`db_applied` (skipping `fs_applied`), because no crash window exists between
publication and pointer update when they share a commit. Step 4 (origin unlink)
happens after the commit, and the state advances to `completed` on success.

**Cross-device (EXDEV) sequence:**

1. Open the source file by descriptor (pinning its inode).
2. Copy through the opened descriptor into an operation-owned temp path
   (deterministic from the operation id: `.voxint-op-{op_id}.tmp`), so recovery
   can find and clean it.
3. `os.fsync` the temp file.
4. Verify the temp file's size and sha256 digest against the source.
5. Publish no-clobber onto the final destination name (same as same-device step
   2), then `os.fsync` the destination directory.
6. Commit `state = 'fs_applied'`. This is the EXDEV crash window: the
   destination is durable but the pointer has not moved.
7. CAS the pointer (`current_path = destination WHERE current_path = origin`).
   Commit `state = 'db_applied'`.
8. Unlink the origin, `os.fsync` the origin directory.
9. Remove the operation-owned temp file (if distinct from the published
   destination).
10. Commit `state = 'completed'`.

The temp file name is deterministic so the reconciler can find and clean orphaned
temps without a directory scan. A crash at any step is recoverable: the
reconciler classifies the filesystem state (section 4) and drives to a terminal
state.

### 4. Reconciliation decision table

The reconciler processes non-terminal `media_operations` rows on a beat
schedule. It never decides from two path-existence checks alone. For each row,
it classifies the filesystem reality and applies the decision table below.

**Inputs to classification:**

| Input | Source |
|---|---|
| `journal_state` | The `state` column of the `media_operations` row. |
| `current_path_class` | Where `media_items.current_path` points: `origin` (equals `op.origin_path`), `destination` (equals `op.destination_path`), or `other` (neither, meaning a later operation moved it). |
| `origin_exists` | Whether `media_root / op.origin_path` is a regular file. |
| `origin_digest` | sha256 of the origin file, if it exists. |
| `dest_exists` | Whether `media_root / op.destination_path` is a regular file. |
| `dest_digest` | sha256 of the destination file, if it exists. |
| `temp_exists` | Whether an operation-owned temp file (`.voxint-op-{op_id}.tmp`) exists in the destination directory. |
| `root_available` | Whether `media_root` and the relevant mount points are accessible. |
| `newer_op_exists` | Whether a newer non-terminal `media_operations` row exists for this `media_id`. |

**Pre-check: root availability.** If `root_available` is false, the reconciler
skips this row without changing its state (following the
`MediaRootUnavailableError` precedent). A long NAS outage must not cause the
reconciler to classify an absent file as lost.

**Pre-check: newer operation.** If `newer_op_exists` is true and the current
operation is non-terminal, the current operation is marked `failed` with
`error_code = 'superseded'` (not retriable). A stale operation must not
overwrite a later operation's pointer.

**Decision table (after pre-checks pass):**

| # | `journal_state` | `current_path_class` | `origin_exists` | `dest_exists` | `temp_exists` | Action |
|---|---|---|---|---|---|---|
| R1 | `planned` | `origin` | yes | no | no | Normal: execute from the beginning. Acquire lease, proceed with filesystem publication. |
| R2 | `planned` | `origin` | yes | no | yes | Interrupted EXDEV copy. Delete the orphaned temp, then execute from the beginning. |
| R3 | `planned` | `origin` | yes | yes | any | Destination collision. If `dest_digest` matches `origin_digest` and the destination path is inside the operation-owned namespace (trash: `_trash/{op_id}/`; move: the recorded `destination_path`), this is a replay of a completed operation that lost its DB commit: CAS the pointer to destination, advance to `completed` (do NOT unlink origin until the CAS succeeds, then unlink). If digests differ or the destination is not operation-owned, mark `failed` with `error_code = 'destination_exists'` (operator action required). |
| R4 | `planned` | `origin` | no | no | no | Origin vanished before execution. Mark `awaiting_retry` with `error_code = 'origin_missing'` (the mount may return). |
| R4a | `planned` | `origin` | no | no | yes | Origin vanished and an incomplete EXDEV temp exists. Delete the orphaned temp, then mark `awaiting_retry` with `error_code = 'origin_missing'`. |
| R5 | `planned` | `origin` | no | yes | any | Origin gone, destination present with matching `origin_digest`. Roll forward: CAS the pointer to destination, advance to `completed`. If `dest_digest` differs from `origin_digest`, mark `failed` with `error_code = 'ambiguous_state'` (not retriable). |
| R6 | `planned` | `destination` | any | any | any | Pointer already at destination. Another actor or the executor completed the pointer update but not the state transition. Verify destination exists and digest matches, then advance to `completed`. If destination is absent or digest mismatches, mark `failed` with `error_code = 'pointer_dangling'`. |
| R7 | `planned` | `other` | any | any | any | Pointer moved by a later operation. Mark `failed` with `error_code = 'superseded'`. |
| R8 | `fs_applied` | `origin` | any | yes | any | Destination published but pointer not yet updated. Verify `dest_digest` matches `origin_digest`. If match: CAS the pointer to destination, then unlink origin (only after CAS succeeds), clean up temps, advance to `completed`. If mismatch: mark `failed` with `error_code = 'digest_mismatch'`. |
| R9 | `fs_applied` | `destination` | any | yes | any | Pointer already updated (state label lagged). Verify `dest_digest` matches `origin_digest`. Clean up origin (unlink if present, after digest verification) and temps, advance to `completed`. |
| R10 | `fs_applied` | `destination` | any | no | any | Pointer updated but destination is now absent. Mark `failed` with `error_code = 'pointer_dangling'` (not retriable). |
| R11 | `fs_applied` | `origin` | any | no | yes | Destination absent but temp exists. The `fs_applied` state means publication was committed, so the destination was removed after publication (external or concurrent). Verify temp digest matches `origin_digest`. If match: re-publish no-clobber from temp, then proceed as R8. If mismatch: delete temp and mark `failed` with `error_code = 'temp_corrupt'`. |
| R12 | `fs_applied` | `origin` | any | no | no | Destination lost after publication. If `origin_exists` with matching digest, roll back pointer to origin, mark `awaiting_retry` with `error_code = 'destination_lost'`. If origin also absent, mark `failed` with `error_code = 'both_absent'` (operator action required). |
| R13 | `fs_applied` | `other` | any | any | any | Superseded. Mark `failed` with `error_code = 'superseded'`. Clean up any temp. |
| R14 | `db_applied` | `destination` | any | yes | any | Normal post-pointer state. Clean up origin (unlink if present, fsync), remove temp, advance to `completed`. |
| R15 | `db_applied` | `destination` | any | no | any | Destination absent after pointer update. Mark `failed` with `error_code = 'pointer_dangling'`. |
| R16 | `db_applied` | not `destination` | any | any | any | Pointer diverged (superseded or rolled back externally). Mark `failed` with `error_code = 'superseded'`. |

Every pointer update in the table is a CAS: `UPDATE media_items SET current_path = :dest WHERE id = :id AND current_path = :expected`. A CAS that returns zero rows means another actor updated the pointer; the reconciler marks the row `failed` with `error_code = 'cas_conflict'`.

**Purge reconciliation.** The decision table above applies to move, trash, and
restore operations (single origin, single destination). Purge follows a
different flow driven by its child manifest. The reconciler handles purge
operations by state:

| State | Action |
|---|---|
| `planned` | The manifest may or may not be built. If `media_operation_files` has no children for this operation, build the manifest and commit. Then attempt each `pending` child (unlink, mark `done`/`missing`/`failed`). |
| `fs_applied` | All children are resolved (`done` or `missing`). Delete artifact rows, set `purged_at`, advance to `db_applied`. |
| `db_applied` | Artifact rows deleted, `purged_at` set. Advance to `completed`. |
| `awaiting_retry` | Re-attempt `failed` children. If all children reach `done` or `missing`, advance to `fs_applied`. |

A purge operation transitions from `planned` to `fs_applied` only when every
child is `done` or `missing`. If any child is `failed`, it moves to
`awaiting_retry`. The partial unique index and lease protocol apply identically.

### 5. Trash, restore, purge

**Trash** is a journaled move whose destination is a managed subtree inside
`media_root`, named `_trash/{op_id}/{filename}` so each trash operation owns its
destination directory. `current_path` follows the file into the trash tree, so
existing playback (which serves the derived `AudioArtifact` preprocessed-WAV
bytes, not the source) keeps working while the item is trashed. The trash tree
name (`_trash`) is added to `_RESERVED_TREES` (`setup_wizard.py`), the single
choke point that excludes it from every watch-folder scan, so the watcher never
re-ingests trashed files. `media_items.trashed_at` is set when the trash
operation completes. A deletion deadline is recorded for display only; there is
no auto-purge.

**Restore** is a journaled move back to the recorded original location (the
trash operation's `origin_path`). If the destination (the original location) is
occupied, the restore refuses with an honest "destination occupied" error and
tells the operator which file is there, rather than overwriting. The restore
records `restores_operation_id` pointing at the completed trash operation so the
reconciler can validate the chain.

**Purge** (empty-trash, manual only) is designed separately from move because a
single origin-destination pair cannot track multi-file deletion. Purge builds a
durable per-file manifest using the `media_operation_files` child table,
enumerating the source file and every derived target across the item's runs
(`AudioArtifact` preprocessed WAV, `AudioChunk` clips, peaks files). The
sequence:

1. Build the manifest: query the item's runs for all derived artifacts and
   record each file path as a child row in `media_operation_files` with
   `status = 'pending'`. Commit.
2. For each child row: attempt unlink, then mark the child row `done` (file
   removed), `missing` (file was already absent), or `failed` (permission or I/O
   error). Commit after each child.
3. Convergence check: only when every child row is `done` or `missing` (no
   `pending` or `failed` remaining) may the purge proceed to step 4. If any
   child is `failed`, the parent operation moves to `awaiting_retry` with
   `error_code = 'partial_purge'`, preserving the manifest for retry. The
   reconciler re-attempts failed children on the next pass.
4. Delete the artifact database rows (`AudioArtifact`, `AudioChunk`, peaks
   metadata) and set `media_items.purged_at`. Commit.
5. Advance the parent operation to `completed`.

The existing `delete_run_derived_media` is not reused as-is because it deletes
artifact rows before the best-effort unlink, which loses the retry inventory on
a crash. Purge reuses its file enumeration logic but defers the row deletion
until the manifest confirms convergence.

Run pages for purged items keep transcript text and run history. The source,
peaks, and clip endpoints return 410 Gone (not 404), with an honest capability
banner driven off `purged_at`.

### 6. Sidecar as a bundled secondary file

A sidecar YAML sits beside a media file and carries per-file configuration. It
belongs to the media bundle (operator decision O4), so a move, trash, and restore
relocate it with the file, and `_reread_sidecar` reads beside `current_path`, not
`source_path`.

Crash safety: two files cannot move atomically. The media file is the critical
file and is published first through the full no-clobber, fsync sequence. The
sidecar is then relocated best-effort, with its own origin and destination
recorded in the `media_operation_files` manifest (the same child table purge
uses), so the reconciler can complete or roll back the sidecar move
independently. A missing or unmovable sidecar is a warning, never a failure of
the media operation: the sidecar's effect is already frozen into any past run
snapshot, and a rerun that finds no sidecar behaves exactly as an unsidecar'd
file does today. This keeps the media-loss invariant strictly about the media
file.

### 7. Purged-media serving contract

When `media_items.purged_at IS NOT NULL`:

| Endpoint | Response |
|---|---|
| `GET /media/{run_id}` (preprocessed WAV) | 410 Gone, `X-Voxint-Purged: true` header. |
| `GET /media/{run_id}/peaks` | 410 Gone, `X-Voxint-Purged: true` header. |
| `GET /media/{run_id}/clips/{clip_id}` | 410 Gone, `X-Voxint-Purged: true` header. |
| Run detail page | Renders with a capability banner: "Source media has been permanently deleted. Transcript text and run history are preserved." Audio player and waveform are hidden. |
| Library list (active view) | Row is hidden (purged items do not appear in the active library). |
| Library list (trash view) | Row is hidden (purged items have left trash). |
| `/runs` list | Row is visible. The run's transcript text and metadata are intact. A badge indicates the source is purged. |

The 410 responses use `X-Voxint-Purged` so the console islands can distinguish
"purged" from "broken" and render the appropriate banner without a second round
trip.

### 8. Operation-specific metadata transitions

Each operation type sets metadata atomically with its completion CAS:

| Operation | On `completed` | On rollback |
|---|---|---|
| Trash | Set `media_items.trashed_at = now()`. | Clear `trashed_at` (if the trash pointer CAS was never committed, `trashed_at` was never set). |
| Restore | Clear `media_items.trashed_at`. | No change (item stays trashed). |
| Purge | Set `media_items.purged_at = now()`. Clear `current_path` (the file no longer exists). | No change (item stays trashed, manifest preserved for retry). |
| Move | No metadata beyond the `current_path` CAS. | No change. |

These writes are part of the completion CAS transaction, so they are atomic with
the state transition and idempotent under reconciliation (the reconciler applies
the same write if the state transition was interrupted).

## Consequences

- Every byte-touching operation is recoverable from a crash at any boundary.
  The reconciler drives interrupted rows to a consistent terminal state using
  the decision table, which relies on digests and the recorded intent, not on
  path-existence checks alone.
- The concurrency protocol (partial unique index, claim lease, CAS transitions,
  shared row lock) closes both executor-versus-executor and move-versus-run
  races with mechanisms the codebase already tests, rather than inventing
  session-level advisory locks.
- Purge is irreversible but safe: the manifest-first design makes a partial
  purge recoverable and honest rather than silently orphaning files. The 410
  contract gives the UI a clean seam to degrade gracefully.
- The trash tree is excluded from watch-folder scans by the single
  `_RESERVED_TREES` choke point, so no new exclusion wiring is needed.
- The `current_path` split (ADR 0001) becomes load-bearing: byte-openers,
  the reclaim alias guard, and the sidecar read all switch to `current_path`
  in this slice.
- `media_folder_id` is untouched by a move (operator decision O1, honoring the
  ADR 0002 MUST-NOT-clobber rule). An explicit "move and reassign" is a separate
  operator action.

## Rejected alternatives

- **Database-first ordering** (update `current_path`, then move bytes).
  Rejected: a crash after the pointer update but before the move leaves the
  pointer aimed at a nonexistent destination, and existing byte-openers would
  fail. Filesystem-first keeps the pointer valid until the destination is
  durably published.
- **Trash as a pure database soft-delete flag** (no physical move). Rejected:
  the issue requires bytes to leave the watched folders so the sweep stops
  re-emitting them and so disk is reclaimable.
- **Kill-a-real-process crash testing.** Rejected: the house pattern
  (`test_cas_and_restart.py`) constructs intermediate states deterministically
  and runs the recoverer, which is CI-friendly and reproducible. This slice
  extends that pattern with executor hooks at each filesystem boundary.
- **`os.replace` for destination publication.** Rejected: `os.replace` clobbers
  an existing destination, and a checksum match does not prove this operation
  created it.
- **Session-level advisory lock across the filesystem phase.** Considered for
  lease fencing. Rejected: the lock lifetime is the database connection, and a
  connection drop during a long EXDEV copy would release the lock while the
  filesystem work is still in progress, recreating the same race. The
  claim-lease + CAS + pre-action recheck protocol is weaker but has no
  connection-lifetime coupling.

## Review notes

A codex planner-role review (zen clink, 2026-08-26) examined the ADR for
protocol correctness, decision-table completeness, and internal consistency.
Eight findings were accepted and folded in:

1. **Retriable `failed` contradiction** (HIGH). `failed` was simultaneously
   terminal (excluded from the unique index) and retriable. Added an
   `awaiting_retry` non-terminal state, included in the unique index, so a
   pending retry blocks new operations and the reconciler can re-enter.

2. **Purge convergence with failed children** (HIGH). The original wording let
   purge complete with `failed` child deletions, orphaning files. Convergence
   now requires every child to be `done` or `missing`; `failed` children move
   the parent to `awaiting_retry`.

3. **No purge reconciliation table** (HIGH). R1-R16 assume a move
   (origin/destination). Added a purge-specific reconciliation table keyed on
   the manifest state.

4. **Lease fencing limitation** (HIGH). Documented that the lease detects stale
   ownership but does not fence filesystem side effects retroactively. Added
   mitigations: pre-action lease recheck, origin unlink only after CAS success,
   reconciler cleans orphaned temps.

5. **Hard-link fallback TOCTOU** (HIGH). Removed the check-then-rename fallback.
   Publication requires `os.link` or `renameat2(RENAME_NOREPLACE)`.

6. **R3 ownership rule** (HIGH). R3 now requires the destination to be in the
   operation-owned namespace and re-verifies digest before origin unlink.
   R8/R9 now verify `dest_digest` before origin cleanup.

7. **`db_applied` reachability** (MEDIUM). Clarified that same-device moves
   skip `fs_applied` and go directly to `db_applied`; EXDEV moves commit
   `fs_applied` after publication and `db_applied` after the pointer CAS,
   making both states reachable.

8. **Metadata transitions unspecified** (MEDIUM). Added section 8 specifying
   `trashed_at`/`purged_at`/`current_path` writes per operation type, atomic
   with the completion CAS.

Session-level advisory locks (finding 4 alternative) were considered and
rejected (see "Rejected alternatives"). R4a (origin vanished + temp exists)
was independently identified before the review arrived. R11 was reframed as a
post-publication destination loss with temp recovery.
