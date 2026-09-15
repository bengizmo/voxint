# #478 Watch-folder pickup feed marker

Status: done

## Goal

Add "N files picked up from watched folder X" entries to the Home feed
when the watch sweep ingests new files. Today the feed shows run and
speaker events but watch pickups are invisible. A prior 4-model review
rejected synthesizing this from `media_folder_id` alone because manual
uploads assigned to a watched folder would produce false copy. The fix
is a durable ingest-origin marker set only by the sweep.

## Assumptions and constraints

- Column on `MediaItem`, not a separate event table. The Home feed is
  "derived entirely from existing tables (no event/outbox table; that is
  a P7 concern)" per the `home_query` module docstring; enriching an
  existing row is the proportionate approach.
- Single-operator deployment. No concurrent sweeps from different
  workers in practice; batch-cap splits producing separate feed entries
  is acceptable.
- Historical sweep pickups are unknowable (sweep vs wizard vs CLI
  cannot be distinguished retroactively); no backfill.
- The existing `WatchSweepSummary` is not modified. It stays a
  newest-wins aggregate with no per-folder breakdown.

## Proposed approach

**Two columns on `MediaItem`**:
- `picked_up_by_sweep_at` (nullable `DateTime(tz)`) -- the sweep's
  batch timestamp. Acts as both occurrence time and natural grouping
  key (all files in one sweep share the same value).
- `picked_up_from_folder` (nullable `Text`) -- the MEDIA_ROOT-relative
  folder path frozen at ingest time. Immutable display label that
  survives folder reassignment, rename, and deletion.

**Write semantics**: The marker is set inside `submit_media_item_if_new`
AFTER the run is successfully created (after domain-pack snapshot +
`submit()` succeed), not at MediaItem construction. This ensures a
DomainPackError, sidecar error, or race loss leaves the marker NULL --
"picked up" means a durable QUEUED run exists. Broker publication
failure still counts (the run is committed and recoverable).

**One `sweep_ts`** computed once in `sweep_watch_folders` before the
batch loop (derived from the existing `now`, not a second
`datetime.now()` call). Passed as a keyword to
`submit_media_item_if_new`. All files in one sweep invocation share the
same timestamp so they group naturally by
`(picked_up_from_folder, picked_up_by_sweep_at)`.

**Feed query**: New `_watch_pickup_rows(session, *, limit)` in
`home_query.py`. Groups `media_items` rows where
`picked_up_by_sweep_at IS NOT NULL` by
`(picked_up_from_folder, picked_up_by_sweep_at)` -- no join to
`media_folders` needed since the path is frozen. Returns one
`ActivityItem` per group with `kind="watch_pickup"`,
`pickup_count=count(*)`, `at=picked_up_by_sweep_at`,
`title=picked_up_from_folder`, `source_path=""`.

**ActivityItem**: Add `pickup_count: int = 0`. Non-zero only for
`watch_pickup` kind.

**Sort-key fix**: Extend the merge sort key so watch-pickup items
(which have `run_id=None`, `speaker_id=None`) break ties
deterministically via `title` (the frozen folder path):
`(at, kind, str(run_id or speaker_id or title or ""))`.

**Template**: New `{% elif item.kind == "watch_pickup" %}` branch with
pluralized copy ("1 file" / "N files") and no action link.

**Partial index**: `(picked_up_by_sweep_at DESC)` WHERE
`picked_up_by_sweep_at IS NOT NULL` to keep the feed query off the
full-table scan as `media_items` grows.

### Why not a separate `watch_folder_pickups` table

Codex advocated for this. I reject it for #478 because: (a) the feed's
design philosophy is derived-from-tables, not event/outbox; (b) the two
columns are simpler -- one migration, no new ORM model, no write-path
coupling beyond one keyword; (c) the frozen folder path addresses
every attribution concern the table would solve. If a future event
system (P7 toasts) needs a proper outbox, the marker column survives
alongside it and is not wasted.

### Feed flooding: accepted, not suppressed

A sweep of N files also produces N `run_started` entries. With
`limit=10`, a large sweep's pickup entry competes with its own starts.
This is accepted: the individual starts are useful (they link to runs),
the pickup is supplementary context, and suppressing `run_started` for
swept items has a re-submit caveat (a manual re-run of a swept file
would also lose its started entry because the marker lives on the
MediaItem). Documented, not fixed.

## Acceptance criteria

### Requirement: Watch-folder pickup feed entries

The system SHALL display grouped watch-folder pickup entries in the
Home feed when the watch sweep successfully ingests new files.

#### Scenario: Sweep picks up files from a watched folder

WHEN the watch sweep ingests 3 new files from a registered, watched
folder "interviews-2026"
THEN the Home feed shows an entry "3 files picked up from watched
folder interviews-2026"
AND the entry's timestamp matches the sweep's batch timestamp
AND the entry is sorted chronologically among other feed events

#### Scenario: Sweep picks up files from multiple folders

WHEN the watch sweep ingests 2 files from folder "interviews-2026"
and 1 file from folder "meetings"
THEN the Home feed shows two separate entries, one per folder, each
with its correct count

#### Scenario: Manual upload to a watched folder does not appear

WHEN an operator uploads a file and assigns it to a watched folder
THEN no "picked up from watched folder" entry appears for that file

#### Scenario: Failed submission does not appear

WHEN the watch sweep attempts to ingest a file but
`submit_media_item_if_new` raises DomainPackError
THEN no pickup marker is set on the resulting MediaItem
AND no pickup entry appears in the feed

#### Scenario: Race loss does not appear

WHEN the watch sweep loses a UNIQUE(source_path) race
THEN `submit_media_item_if_new` returns None
AND no pickup marker is set on the existing MediaItem

#### Scenario: Feed limits apply

WHEN the Home feed is trimmed to its display limit
THEN watch-pickup entries compete with run and speaker entries on
recency, not separately capped

#### Scenario: Single-file pluralization

WHEN 1 file is picked up from a folder
THEN the feed shows "1 file picked up" (not "1 files")

### Requirement: Durable ingest-origin marker

The system SHALL mark each MediaItem successfully ingested by the
watch sweep with an immutable sweep timestamp and folder path.

#### Scenario: Marker set on successful sweep ingest

WHEN the watch sweep successfully submits a new file
THEN the resulting MediaItem has `picked_up_by_sweep_at` set to the
sweep's batch timestamp
AND `picked_up_from_folder` set to the MEDIA_ROOT-relative folder path

#### Scenario: Marker not set on non-sweep ingest

WHEN a file is submitted via CLI, browser upload, URL ingest, or
wizard scan
THEN the resulting MediaItem has both pickup columns NULL

#### Scenario: Frozen path survives folder changes

WHEN a MediaItem was picked up from folder "interviews-2026"
AND the folder is later unregistered or the media is bulk-reassigned
THEN the feed entry still shows "interviews-2026" unchanged

## Spec deltas

Spec deltas: none (no living spec declared)

## Affected files / components

| File | Change | Why |
|---|---|---|
| `alembic/versions/0063_watch_pickup_marker.py` | New migration | Add two nullable columns + partial index |
| `src/voxint/db/models.py` | Add columns + index to `MediaItem` | Durable marker |
| `src/voxint/ingest/service.py` | Add params, set after run creation | Mark only successful submissions |
| `src/voxint/ingest/watch.py` | Compute `sweep_ts`, pass to submissions | Thread timestamp + folder path |
| `src/voxint/api/home_query.py` | New `_watch_pickup_rows()`, extend `ActivityItem`, update `recent_activity()`, docstrings | Feed entries |
| `src/voxint/api/templates/home/home.html` | New template branch | Render pickup entries with pluralization |
| `tests/unit/test_watch.py` | Extend | Marker threading |
| `tests/integration/test_watch_sweep.py` | Extend | Marker set on swept items, not on failures |
| `tests/unit/test_home_query.py` (new or extend) | New | Query grouping, limits, sort determinism |
| `tests/integration/test_home_api.py` | Extend | Feed renders pickup entries |
| `tests/contracts/test_watch_folder_config.py` | Extend | Column + index existence |
| `CHANGELOG.md` | Entry under [Unreleased] | |

## Implementation slices

### Slice 1: Migration + ingest marker (write path)

- Migration 0063: add `picked_up_by_sweep_at` (nullable
  `DateTime(tz)`) and `picked_up_from_folder` (nullable `Text`) to
  `media_items`. Add partial index
  `ix_media_items_pickup(picked_up_by_sweep_at DESC)` WHERE
  `picked_up_by_sweep_at IS NOT NULL`. No backfill.
- Model: add both mapped columns and index to `MediaItem.__table_args__`
  (keep in lockstep with migration).
- `submit_media_item_if_new`: add `picked_up_by_sweep_at: datetime |
  None = None` and `picked_up_from_folder: str | None = None` keywords.
  Set on the MediaItem AFTER successful run creation (after
  `_run_domain_pack_snapshot` + `submit()` return), not at construction.
  Document in docstring: the marker means a durable QUEUED run exists.
- `sweep_watch_folders`: compute `sweep_ts =
  datetime.fromtimestamp(now, tz=UTC)` from the existing `now` (line
  153). For each submitted file, resolve the folder path from the
  MediaItem's `media_folder_id` (it is still fresh at this point) and
  pass both to `submit_media_item_if_new`.
- Tests:
  - Sweep sets identical `picked_up_by_sweep_at` for all items in one
    batch.
  - Second sweep produces a different timestamp.
  - Non-sweep callers (wizard, CLI, upload, URL) leave both columns
    NULL.
  - Race loss (`submit_media_item_if_new` returns None): no marker on
    existing item.
  - DomainPackError: marker stays NULL.
  - Migration up/down.
- Verify: `uv run pytest -n 8 -o addopts="" -q` passes, `ruff check`,
  `mypy`.

### Slice 2: Home feed watch-pickup entries (read path + UI)

- `ActivityItem`: add `pickup_count: int = 0`. Update kind docstring
  to list `watch_pickup`.
- `home_query.py`:
  - New `_watch_pickup_rows(session, *, limit)`: query groups
    `media_items` by `(picked_up_from_folder, picked_up_by_sweep_at)`
    WHERE `picked_up_by_sweep_at IS NOT NULL`, returns `count(*)` and
    `max(picked_up_by_sweep_at)`, ordered by timestamp desc, limited.
    No join to `media_folders`. Normalize `.astimezone(UTC)`.
  - Append to `merged` in `recent_activity()`.
  - Extend sort key: `str(i.run_id or i.speaker_id or i.title or "")`.
  - Update module docstring: "three" to "four" slices,
    `4 * limit` bound.
  - Update `recent_activity` docstring.
  - Add note in `_watch_pickup_rows` docstring: trashed media keep
    their pickup entry (historical activity); feed competes with
    run_started entries (accepted).
- `home.html`: new `{% elif item.kind == "watch_pickup" %}` branch.
  Pluralized: `{{ item.pickup_count }} file{{ "s" if item.pickup_count
  != 1 else "" }} picked up from watched folder
  {{ item.title }}`. No action link (blank `recent-action` cell).
  Jinja autoescaping handles folder paths with special characters.
- `group_activity`: no change needed. Watch-pickup items are not
  `run_failed`, so they pass through. Note this in the plan so nobody
  over-touches it.
- Tests:
  - Grouping: N files/one folder/one sweep = one item, count=N.
  - Split: two folders/one sweep = two items.
  - Limit respected.
  - NULL `picked_up_by_sweep_at` media excluded.
  - Sort determinism: two folders at the same sweep_ts order stably.
  - Merge with run/speaker items: correct chronological interleave.
  - Template renders correctly including N=1 singular.
  - Manual upload with `media_folder_id` set does NOT produce a
    pickup entry (the honesty regression test).
- Verify: full suite passes.

### Slice 3: Contract + docs + cleanup

- Contract test: `picked_up_by_sweep_at` and `picked_up_from_folder`
  columns exist. Partial index exists. Model and migration in lockstep.
- CHANGELOG entry under `[Unreleased]`: "Home feed shows watched-folder
  pickup entries. Pickup history begins after this upgrade (no backfill
  of historical sweeps)."
- Verify: full suite passes, ruff/mypy clean.

## Testing strategy

| Gate | What it covers |
|---|---|
| Migration tests (slice 1) | Up adds columns + index, down removes. No backfill. |
| Unit: write path (slice 1) | Marker set only after successful run creation. Race loss and DomainPackError leave NULL. Non-sweep callers leave NULL. |
| Unit: feed query (slice 2) | Grouping, counting, limit, sort determinism, NULL exclusion, UTC normalization. |
| Integration: sweep (slice 1) | End-to-end: sweep submits files, markers set, correct timestamps and folder paths. |
| Integration: Home API (slice 2) | Feed includes/excludes pickup entries correctly. Template renders pluralized copy. Manual upload honesty test. |
| Contract (slice 3) | Column + index existence. Model-migration parity. |
| Regression | Full suite green. Existing feed behavior unchanged. |

## Rollout / risks / open questions

### Resolved (all 4 models agreed)

- **Link target**: No link. The pickup describes a batch, not a single
  run. No `/runs?folder=` filter exists. Blank `recent-action` cell,
  consistent with how the template handles id-less kinds. Can add a
  link later if a folder-filter view ships.
- **Copy**: Folder path (the frozen `picked_up_from_folder`). It is the
  precise, recognizable subject the operator configured. Project
  membership is mutable and may be absent.
- **Historical backfill**: None. Historical sweep-vs-upload provenance
  is unknowable; backfilling would fabricate provenance. Same posture as
  `detected_language`. Note in CHANGELOG: "history begins after this
  upgrade."

### Risks

- **Feed flooding**: A sweep of N files produces N `run_started` entries
  that compete with the single pickup entry for the feed's limit=10
  display slots. Accepted: individual starts are useful, the pickup is
  supplementary context.
- **Batch-cap splits**: If `watch_folder_batch_size` cuts a sweep, the
  remainder is picked up by the next sweep invocation with a different
  timestamp, producing two feed entries. Documented as expected.
- **Migration number**: 0063 is the successor of the current head
  (0062). Verify `alembic heads` before implementation; re-number if
  parallel work lands first.

## Review notes

### 4-model panel (codex, deepseek-v4-pro, grok-4.5, kimi-k3)

**Accepted (folded in):**

| Finding | Raised by | Resolution |
|---|---|---|
| Mutable `media_folder_id` breaks feed history | All 4 | Added frozen `picked_up_from_folder` column. Feed groups by the snapshot, not the live FK. |
| Sort-key determinism breaks for watch_pickup | All 4 | Extended sort key to fall back to `title` when both ids are None. |
| Missing partial index | All 4 | Added to migration and model `__table_args__`. |
| DomainPackError can stamp a pickup with no run | Codex | Moved marker to after successful run creation. |
| Docstring drift (3 slices to 4, kind enum) | DeepSeek, Grok, Kimi | Explicit step in slice 2. |
| Template pluralization (1 file vs N files) | Codex, Grok, Kimi | Explicit in template spec and tests. |
| UTC normalization in query | DeepSeek, Grok, Kimi | Mirror existing `.astimezone(UTC)` pattern. |
| Compute sweep_ts once from existing `now` | DeepSeek, Grok, Kimi | Derive from `watch.py` line 153's `now`. |
| Feed flooding accepted, not suppressed | Kimi | Documented; suppressing run_started has re-submit caveats. |
| Trashed media keep pickup entry | Kimi | Documented in query docstring. |

**Rejected:**

| Finding | Raised by | Reason |
|---|---|---|
| Use a separate `watch_folder_pickups` table | Codex | Over-engineered for a single-operator tool. The feed's design philosophy is derived-from-tables. The frozen path column addresses every attribution concern. |
| Use a UUID `sweep_id` instead of timestamp | Codex | Timestamp microsecond resolution is sufficient for single-operator beat schedule. Two columns (UUID + occurred_at) add complexity with no practical benefit. |
| `pickup_count` default 0 permits invalid state | Codex | Acceptable trade-off for dataclass simplicity. Tests assert non-zero for emitted rows. |
| Suppress `run_started` for swept items | Kimi | Scope creep; re-submit caveat makes it imprecise. |
| Reserve a pickup quota for feed fairness | Grok | Chronological ordering intentionally permits bursts. Over-engineering for this audience. |

## Completion notes

Closed 2026-09-15. PR #483 merged to main at `cf26d7d`.

**Verified**: all 3 slices complete, all 7 acceptance scenarios covered by
tests (68 tests across 5 files, all green). Sort-key determinism, template
pluralization, contract tests for columns + partial index, migration up/down
all confirmed. No rejected proposals reintroduced. No drift detected.

**Spec sync**: skipped (no living spec declared).

**Follow-ups**: none.
