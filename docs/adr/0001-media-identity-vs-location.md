# ADR 0001: Media identity versus location

> **Status:** Accepted (Console 2.0 P0a, issue #150). The `current_path` and
> `media_folder_id` columns landed in P2a (migration 0040). P2c (issue #155,
> ADR 0007) makes `current_path` load-bearing: the byte-opener switch,
> the reclaim alias guard, and the sidecar read all move to `current_path`.

## Context

Today `media_items.source_path` carries two jobs at once. It is the identity of a
piece of acquired media (a unique, deduplicating key, written once at ingest) and
it is the physical location the pipeline reopens to read bytes. Console 2.0 adds
file management: an operator can move a file between folders and trash or restore
it. The moment a file can move, a single column cannot be both the stable
identity a past adjudication was made against and the mutable path the decode
stage opens.

The pins that constrain the design:

- `media_items.source_path` is `Text`, `unique=True`, and non-null. It is stored
  relative to `MEDIA_ROOT` (byte reads resolve as `media_root / source_path`).
- `media_items.sha256` is nullable, non-unique. Rows predating the content-hash
  work, and some local or uploaded media, carry NULL.
- URL submission mints a fresh uuid-namespaced `source_path`; the published bytes
  are immutable. Re-acquiring a URL creates a new `MediaItem`, so identity is
  already per-acquisition.
- Frozen run snapshots (domain packs, provenance) reference media through the
  run, and must not shift when a file moves.

## Decision

1. **`media_items.id` is the identity anchor.** Every relation that means "this
   piece of media" keys on the uuid primary key, not on a path.

2. **`source_path` stays the immutable acquisition identity.** It keeps its
   uniqueness and dedup role and is never rewritten after ingest. It records
   where the media was acquired, not where its bytes live now.

3. **Physical location moves to new columns.** A future migration adds a mutable
   `current_path` (backfilled to `source_path`) plus a `media_folder_id` foreign
   key (ADR 0002). Byte-opening reads switch to `current_path`; identity, dedup,
   search, and display reads stay on `source_path`.

4. **`sha256` is an integrity aid, never identity.** It detects a silent byte
   change under an unchanged path and lets tooling deduplicate by content. It is
   not made unique and does not gate acquisition. P0a ships an idempotent backfill
   (`voxint media backfill-hashes`) so existing rows can be filled in.

5. **The byte-opener set is fixed by audit, not assumption.** The switch to
   `current_path` touches only the reads enumerated below. Every other
   `source_path` read is left on the identity column.

## Byte-opener audit

Grep-verified enumeration of every `source_path` reference in `src/voxint/`,
as of P0a, classified by the operation it performs. The base audit covered the
130 references across 24 files present before this slice; P0a itself adds one
new opener (`media/integrity.py`, the sha256 backfill), listed below.

| Category | Count | Meaning |
|---|---|---|
| IDENTITY/WRITE | 68 | Written at ingest, or used as a uniqueness/dedup/equality key. Stays on `source_path`. |
| DISPLAY | 41 | Rendered to a human, logged, or returned as a JSON field. Stays on `source_path`. |
| SCHEMA/DEF | 14 | Column, ORM mapping, or DTO field declaration. Structural. |
| BYTE-OPENER | 7 | Derives a filesystem path to touch bytes (pre-P0a). Two are executable; five are comments annotating them. P0a adds a third executable opener in `media/integrity.py`. |
| LOCATION-SEMANTIC | 2 | Not a byte open, but keyed on the physical location and must follow a move. Added in the P2c audit refresh: the reclaim alias guard and the sidecar read. |

The load-bearing finding: **three executable statements** derive a byte path
from the column, two of them post-ingest reads (PREPARE and the P0a backfill).
Line numbers are as of P0a and will drift; the symbol column is the durable
anchor. Re-derive with `rg -n 'source_path' src/voxint` and reclassify.

### Byte-opener list (executable statements)

| Symbol (location as of P2c) | Nature | Migration action |
|---|---|---|
| `prepare.run` (`pipeline/stages/prepare.py:30`) | `source = media_root / media.source_path`, then `source.is_file()`, `normalize_to_wav(source, ...)` ffmpeg input, `source.stat()` | **Switch to `current_path` (P2c).** The canonical post-ingest read: PREPARE reopens the acquired file to normalize it to WAV. |
| `acquire.run` (`pipeline/stages/acquire.py:72`, URL media) | `dest = (media_root / media.source_path).resolve()`, then `dest.is_file()` / `dest.stat()` / `_sha256(dest)` / `os.link(produced, dest)` | **Acquisition write, see ambiguous case A.** This materializes the bytes and their hash at the `source_path` location; it is the ingest, not a post-ingest read. |
| `backfill_sha256` (`media/integrity.py:124`, via `openable_source`) | `openable_source(media_root, media.source_path)` then `sha256_file(path)` | **Switch to `current_path` (P2c).** A post-ingest maintenance read that hashes the media content. After a move, the live bytes are at `current_path`, so the backfill must open there. |

### Location-semantic references (added in P2c audit refresh)

These are not byte openers, but they key on the physical location and must
follow a move to stay correct. Both switch to `current_path` in P2c.

| Symbol (location as of P2c) | Nature | Migration action |
|---|---|---|
| `_reclaimable_artifacts` (`media/reclaim.py:200`) | `MediaItem.source_path == AudioArtifact.path` exclusion guard: prevents GC from deleting an artifact file that is also a media source. After a move, the live source is at `current_path`, not `source_path`, so the guard must compare against `current_path` to avoid deleting a relocated live source. | **Switch to `current_path` (P2c).** The guard is belt-and-suspenders today (`source_path` is `incoming/{uuid}/source` and the query is gated to `artifacts/%`), but the split makes it load-bearing. |
| `_reread_sidecar` (`api/routers/media.py:760`) | Resolves `media_root / source_path` to find the paired YAML sidecar for a rerun. After a move, the sidecar sits beside `current_path`, not `source_path` (operator decision O4, ADR 0007). | **Switch to `current_path` (P2c).** The sidecar moves with the file as a bundled secondary (ADR 0007 section 6). |

No other code opens, stats, hashes, serves, or deletes bytes via `source_path`.
Verified negatives worth recording, because they look like byte-openers and are
not:

- `GET /media/{run_id}` and `GET /media/{run_id}/peaks` serve and read the
  derived `AudioArtifact` preprocessed-WAV path, never `source_path`.
- Media delete plans from `AudioArtifact` rows, not `source_path`.
- `media/reclaim.py` deletes `AudioArtifact.path` bytes; its `source_path` use is
  an exclusion guard (a WHERE clause), not an open. Promoted to a location-semantic
  reference in the P2c audit refresh (see above).
- The watch sweep stats `media_root / rel` from the freshly scanned relative
  path, never `media.source_path`.
- The upload path writes bytes through a local `dest`/`rel` variable
  (`os.replace`), not by re-reading `media.source_path`.

### Ambiguous cases

**A. `acquire.py:72` (and its upload twin) is the acquisition write, not a
post-ingest read.** Under the split, `source_path` is the acquisition identity,
never reopened after ingest, and ACQUIRE is the ingest. It writes the bytes, then
reads them back for the idempotent-replay hash and size at that same location.
Whether it switches to `current_path` depends on how the migration seeds the
split at ingest. The clean reading: ACQUIRE (and the upload publish) is the one
place that legitimately writes the acquisition location, and it should also seed
`current_path := source_path` at completion. The alternative, treating ACQUIRE's
target as `current_path` from the outset, leaves `source_path` as a pure recorded
label no code opens. The migration ADR picks one; both are internally consistent.

**B. `reclaim.py:200` is a physical-location safety guard keyed on the path
string** (`MediaItem.source_path == AudioArtifact.path`). It excludes any file
physically located at a `source_path` from reclamation. It is not a byte open,
but it is location-semantic: when a file moves, the live source is at
`current_path`, not `source_path`, so the guard must compare against
`current_path` to avoid deleting a relocated live source. In practice source
paths are `incoming/{uuid}/source` and the query is already gated to
`artifacts/%`, so it is belt-and-suspenders today, but the split makes it
load-bearing. **Switches to `current_path` in P2c** (see the location-semantic
table above).

**C. `runs_query.py:492` (`source_path.ilike(...)`) is the operator search
filter.** It queries the stored path text, not the filesystem, so it is not a
byte open. At split time it becomes a UX choice: search the acquisition path, the
live location, or both.

## Consequences

- A file move updates `current_path` and `media_folder_id` only. Identity, dedup,
  frozen snapshots, and past adjudications are untouched, because they key on
  `id` and `source_path`.
- The migration surface is small and known: two post-ingest reads (PREPARE in
  `prepare.py` and the `backfill_sha256` maintenance read in `media/integrity.py`)
  plus two location-semantic references (the reclaim alias guard in `reclaim.py`
  and the sidecar read in `media.py`) switch to `current_path` in P2c. The
  acquisition write (case A) stays on `source_path`. The other roughly 120
  reads are proven not to open bytes.
- `sha256` becomes populated corpus-wide via the backfill, enabling later
  integrity and content-dedup features without ever becoming identity.
- The audit is a point-in-time snapshot (P0a, refreshed P2c). Any new
  `source_path` reader must be classified against this list; the byte-opener
  and location-semantic sets are the contract the `current_path` switch depends
  on. P2c adds a location-audit contract test pinning the switch set.

## Appendix: full reference classification

The complete grep-verified table (every reference, including comments and
docstrings, tagged by the operation it performs or documents) is preserved with
this ADR as the audit of record. It is reproduced here so the byte-opener set can
be re-derived without re-running the audit.

### `pipeline/stages/acquire.py`

| Line | Category | Note |
|---|---|---|
| 4, 74, 147, 260 | IDENTITY | docs/comments on identity provenance and row/bytes coherence |
| 10, 14, 67, 80, 142 | BYTE-OPENER | comments annotating the acquisition byte operations |
| 18 | IDENTITY/WRITE | doc: the acquisition write |
| 72 | BYTE-OPENER | EXEC: derives the download destination; see byte-opener list |
| 77 | DISPLAY | error message text |

### `pipeline/stages/prepare.py`

| Line | Category | Note |
|---|---|---|
| 30 | BYTE-OPENER | EXEC: the post-ingest normalize read; primary migration target |
| 35, 38 | DISPLAY | error message text |

### `ingest/service.py`

| Line | Category | Note |
|---|---|---|
| 196, 220, 280, 288, 291, 305, 310, 327, 335, 339, 343, 357, 362, 368, 374, 393, 394, 396, 406, 410, 419, 786, 835, 852, 853, 863, 868, 891, 897, 923, 927, 949, 955, 960 | IDENTITY/WRITE | dedup lookups, UNIQUE-conflict handling, and identity writes at ingest |
| 178 | SCHEMA/DEF | `UploadConflictError` ctor param |
| 179, 180, 793, 902 | DISPLAY | conflict error surface carrying the identity value |
| 206, 622, 829, 941 | IDENTITY | docs/comments |

### `ingest/watch.py`, `config.py`

| Line | Category | Note |
|---|---|---|
| watch.py:11, config.py:375 | IDENTITY | docs: the sweep stats the scanned relative path, never `media.source_path` |

### `media/reclaim.py`, `media/ytdlp.py`

| Line | Category | Note |
|---|---|---|
| reclaim.py:186 | IDENTITY | doc |
| reclaim.py:200 | LOCATION-SEMANTIC | EXEC: exclusion guard (WHERE equality), ambiguous case B. Switches to `current_path` in P2c. |
| ytdlp.py:5 | IDENTITY/WRITE | doc: downloads into a caller temp dir, never touches the column |

### `adjudication/resolver.py`

| Line | Category | Note |
|---|---|---|
| 551 | SCHEMA/DEF | `QueueEntry` DTO field |
| 616 | DISPLAY | populates the queue-view DTO |

### `domain_packs/registry.py`

| Line | Category | Note |
|---|---|---|
| 90, 92, 102, 116, 125, 127, 133, 134 | IDENTITY | pure string/path-prefix compare to pick a folder pack; no filesystem access |

### `api/presentation.py`

| Line | Category | Note |
|---|---|---|
| 44, 45, 51, 52, 54, 78, 84, 86, 98 | DISPLAY | derives a readable display label from the path string; no file access |

### `api/meaning_query.py`

| Line | Category | Note |
|---|---|---|
| 113, 139 | SCHEMA/DEF | DTO fields |
| 283, 310, 426 | DISPLAY | SELECT column and DTO population for result rows |

### `cli.py`

| Line | Category | Note |
|---|---|---|
| 449 | IDENTITY/WRITE | comment |
| 998, 1020 | DISPLAY | JSON export field and console print |

### `api/runs_query.py`

| Line | Category | Note |
|---|---|---|
| 183, 611 | SCHEMA/DEF | DTO fields |
| 192, 414, 535, 614, 654, 677 | DISPLAY | SELECT columns, DTO population, template fallback comments |
| 492 | IDENTITY | EXEC: operator search `ilike` over stored path text; ambiguous case C |

### `api/setup_wizard.py`

| Line | Category | Note |
|---|---|---|
| 329, 360 | IDENTITY | comments |
| 431, 432 | IDENTITY | EXEC: dedup lookup and membership test (skip already-known) |

### `tutorial/seed.py`

| Line | Category | Note |
|---|---|---|
| 27, 84, 86, 198 | IDENTITY | comments (seed row is pre-COMPLETED; ACQUIRE/PREPARE never read it) |
| 159, 167, 172, 184, 187 | IDENTITY/WRITE | dedup lookups and the seeded identity write |

### `media/integrity.py` (added in P0a)

| Symbol | Category | Note |
|---|---|---|
| `backfill_sha256` / `openable_source` (line 124) | BYTE-OPENER | EXEC: post-ingest maintenance read; switches to `current_path` in P2c |
| module docstring, `sha256_file` doc | IDENTITY | notes that the hash is integrity, never identity |

### `config.py`, `db/models.py`

| Line | Category | Note |
|---|---|---|
| models.py:215 | SCHEMA/DEF | the `unique=True` column definition |
| models.py:218, 219 | SCHEMA/DEF | column comments |
| models.py:240 | IDENTITY/WRITE | doc: URL submission mints a fresh uuid `source_path` |

### `api/app.py`

| Line | Category | Note |
|---|---|---|
| 1585 | DISPLAY | derives the run display title; no file access |
| 4019, 4154 | IDENTITY / IDENTITY-WRITE | comments on shared-path collision and write ordering |
| 4143 | DISPLAY | comment on conflict error text |

### Templates (`api/templates/`)

| File | Category | Note |
|---|---|---|
| transcript.html, run_detail.html, review_journey.html, runs.html, dashboard.html, queue.html, search.html | DISPLAY | rendered labels, tooltips, and code spans |
| runs.html:32 | IDENTITY | template comment on idempotent double-submit |
