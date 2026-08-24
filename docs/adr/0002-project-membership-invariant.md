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
