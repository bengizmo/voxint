# ADR 0002: Project membership invariant

> **Status:** Accepted (Console 2.0 P0a, issue #150). The `projects` and
> `media_folders` tables and the `media_items.media_folder_id` foreign key land in
> a later phase (P2a) with their first consumer.

## Context

Console 2.0 introduces projects: a project groups media folders and carries
project-scoped vocabulary and corrections. Today there is no project entity, no
folder table, and no move support. Media is discovered by scanning registered
folders held in the `app_settings` singleton (`media_folders`), and a file's
folder is inferred from its path.

Path-prefix inference is fragile once files move and projects nest. If two
registered folders overlap, or a file moves between them, "which folder does this
belong to" has no single answer derivable from the path. Configuration resolution
for a run (which vocabulary and corrections apply) must be deterministic and must
not shift silently when a file is relocated.

## Decision

1. **Membership is a foreign key, not an inference.** A `MediaItem` belongs to
   exactly one `media_folders` row through `media_items.media_folder_id`, set at
   ingest and updated by a move. Folder membership in a project is likewise a
   relation, not a path prefix.

2. **A folder joins at most one project.** Overlapping folder registrations are
   refused at registration time rather than resolved heuristically.

3. **Config resolution walks relations, not path ancestry.** For a new run the
   effective vocabulary and corrections resolve project, then folder pack, then
   global, following the membership relations. The effective config is shown
   before a rerun so the operator sees what will apply.

4. **A packed folder joining a project has an explicit conflict rule.** The
   project wins, and the operator is warned at assign time. This is a stated rule,
   not an emergent outcome.

5. **Existing frozen run snapshots are untouched.** Membership changes and config
   resolution apply only to new runs. A past run keeps the snapshot it was frozen
   against.

6. **The setup wizard is re-pointed in the same slice as the schema.** The
   wizard's folder registration writes to the new `media_folders` table in the
   same migration cutover, with no dual-write window that could leave the two
   sources of truth disagreeing.

## Consequences

- Every media row has a definite folder, and every folder a definite project (or
  none), independent of where the bytes currently sit. Location moves (ADR 0001)
  update the FK, not identity.
- The migration from the `app_settings.media_folders` list carries a preflight
  normalization report (duplicate registrations, nested paths, dead directories)
  so the operator resolves ambiguity before cutover rather than inheriting it.
- Refusing overlapping registrations is a real constraint an operator can hit; the
  error copy states the reason and the fix, honestly, rather than silently picking
  a folder.
- Config resolution is auditable: given a run, the applied pack is a walk of
  stored relations, reproducible without reasoning about path strings.

## Addendum (P2a, issue #153): resolution semantics and snapshot versioning

The P2a slice makes decisions 3 and 4 concrete. Two points needed pinning
before implementation:

1. **Per-field replacement, not union.** Vocabulary and corrections each resolve
   independently by first present layer, in the order: an explicit per-run pack
   override (CLI or sidecar), then the project field, then the folder pack field,
   then the global baseline. A folder layer is present when
   `media_folders.domain_pack` is non-NULL. Project fields are nullable: NULL
   means inherit the layer below, an empty list means "explicitly none" and wins.
   This is what "the project wins" (decision 4) means in practice: a project with
   its own vocabulary replaces, rather than adds to, what the folder pack or the
   global baseline would have contributed.

   The observable consequence, worth a release note: a media item that resolves
   to an explicit folder pack now stops inheriting the global vocabulary and
   corrections. Under the pre-P2a behavior both were unioned onto every run. A
   media item with no project and no folder pack still resolves to the default
   pack composed with the global settings exactly as before, so existing installs
   that never set a per-folder pack see no change.

2. **Snapshot versioning keeps frozen runs byte-identical.** Vocabulary was
   applied live at run start (`pipeline/stages/context.py` unioned
   `app_settings.vocabulary` onto the pack). Deterministic project-scoped
   resolution requires freezing the effective vocabulary at submit alongside the
   corrections. To avoid rewriting or reinterpreting any existing
   `pipeline_runs.domain_pack` row, new snapshots carry a
   `config_resolution_version: 2` key. The worker branches on it: version 2 uses
   the frozen vocabulary and does not live-union `app_settings.vocabulary`; a
   missing key (every pre-P2a row) keeps the exact live-union path. No migration
   rewrites `pipeline_runs`, so requeuing an old run reproduces its original
   behavior.

## Addendum (P2b, issue #154): membership is a logical config scope

P2a set `media_folder_id` only from the file's path (the deepest registered folder
that contains `source_path`), and Decision 1 above framed the column as "set at
ingest and updated by a move". P2b's library adds two operations that need
membership to change without the bytes changing location: an upload or URL fetch
where the operator picks a folder, and a bulk "use this folder's settings" assign.
Uploads and URLs live under `incoming/{uuid}/...`, which is inside no registered
folder, so neither can be expressed by path inference.

This addendum widens Decision 1: **`media_folder_id` is the folder whose settings
apply, a logical scope that MAY differ from where the bytes physically sit.** It is
set at ingest from the path when the file is discovered by a scan, and it may be set
or cleared explicitly by the operator (the upload/URL picker, the bulk assign). An
explicit assignment never moves or copies the file; `current_path` and `source_path`
are untouched. The library labels the control as a settings folder, never as a move,
so the operator is not told the bytes went somewhere they did not.

The invariant that makes this safe: **an explicit assignment is authoritative and is
never silently re-derived from the path.** Path inference runs at one moment only,
the creation of a new `MediaItem` row (`_get_or_create_media`,
`submit_media_item_if_new`); a reused row keeps its stored membership, so a later
scan, re-run, or folder registration cannot overwrite an operator's pick. Config
resolution still walks the stored relation, not the path, so a run under an assigned
folder freezes that folder's (and its project's) vocabulary and corrections exactly
as a path-resolved one would. Decision 5 still holds: frozen run snapshots do not
move when membership changes; only new runs see the new scope.

Forward constraint for P2c (journaled physical move): a move updates `current_path`
and, for a file whose membership was path-derived, may update `media_folder_id` to
match the destination. It MUST NOT clobber an explicit override by re-deriving
membership from the new `current_path`. Whether a move offers to re-home an override
is a UI choice P2c owns; the default is to preserve the operator's pick. Until P2c
lands there is no move, so an assignment is stable.

Unregistering a folder that still has media assigned relies on the FK
`ON DELETE SET NULL` (ADR 0001): those rows revert to `media_folder_id = NULL` (the
global baseline) rather than erroring or orphaning. The library states how many
files reverted so the change is not silent.
