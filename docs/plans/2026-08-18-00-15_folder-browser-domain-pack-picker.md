# Plan — #63: MEDIA_ROOT folder browser + per-folder domain-pack picker

**Created:** 2026-08-18 00:15 · **Issue:** #63 (settings/first-run arc child of #47) ·
**Branch (to cut):** `feat/63-folder-browser` (unmerged until complete) ·
**Base:** `main` @ `df1f4d7` · **Migration:** none (reuses existing columns).

## Goal

Replace the wizard media step's raw newline-path **textarea** with an htmx
server-rendered **directory browser** confined to `MEDIA_ROOT`, and add an
equivalent **folders section to the Settings page** (currently absent). The
browser lets the single non-technical operator click into subdirectories under
`MEDIA_ROOT`, register a folder for watching, and assign each registered folder a
**domain pack** via a `<select>` persisted to the existing `folder_domain_packs`
JSONB column (a "Default" option = no mapping). This folds in the UI half of #11
(the read side — `resolve_folder_pack_name` → run snapshot — already ships). Every
request re-validates containment server-side and **never trusts a client-supplied
path**. No DB migration.

## Assumptions & constraints

- **Reuse the canonical validator.** All registered paths still go through
  `normalize_media_folders` (`src/voxint/api/setup_wizard.py:112`): resolve +
  `is_relative_to(root)` containment, reject absolute/traversal/symlink-escape,
  reject `_RESERVED_TREES` (`incoming`/`artifacts`), require an existing dir,
  dedup, cap `MAX_MEDIA_FOLDERS`=64. Root `.` registration is a supported
  capability we must not silently drop.
- **Pack names** come from `available_domain_packs(settings)`
  (`src/voxint/domain_packs/registry.py:40`) — filesystem-touching, may raise
  `DomainPackError` (name collision / `DOMAIN_PACKS_DIR` not a dir / bad manifest).
  Its docstring already says it "Backs the Settings folder→pack picker".
- **Dirty tracking:** neither `media_folders` (ARRAY) nor `folder_domain_packs`
  (JSON/JSONB) uses `MutableDict`/`MutableList`; every mutation must **reassign a
  fresh object** (the existing `/setup/media` does `row.media_folders = folders`).
- **Singleton row, no version column.** `app_settings` is id=1. `get_or_create`
  only guards the initial insert race — concurrent mutations to the JSON/ARRAY
  state can lose updates. Double-clicks / multiple tabs / overlapping htmx are
  normal single-user behaviour, so mutations MUST serialize (see approach).
- **htmx is already a hard dependency** (scan, review islands). Non-JS degradation
  is via ordinary `<form>`/`<a>` full-page responses, not a second bulk editor.
- **Trusted local-filesystem threat model.** `resolve()`-then-`scandir` has a
  TOCTOU window only under hostile concurrent filesystem mutation, which is out of
  scope for a single-operator self-hosted box; documented, not defended with
  `openat`/`O_NOFOLLOW`. What IS defended: traversal, symlink-escape, reserved
  trees — on every request.
- Numerics doctrine untouched (no inference path changes).

## Proposed approach

### 1. Pure listing function (setup_wizard.py) — transport-agnostic, unit-tested

```python
MAX_BROWSE_ENTRIES = 500  # immediate child dirs listed per directory

@dataclass(frozen=True)
class DirEntry:
    name: str          # leaf display name
    rel: str           # MEDIA_ROOT-relative POSIX path
    registered: bool   # already in media_folders

@dataclass(frozen=True)
class BrowseListing:
    current: str                        # "." at root, else relative POSIX
    current_registered: bool            # is the browsed dir itself registered
    current_reserved: bool              # browsed dir is a reserved tree (no Add)
    parent: str | None                  # parent's relative POSIX, None at root
    breadcrumbs: list[tuple[str, str]]  # (label, rel), root → current
    entries: list[DirEntry]             # child dirs, sorted by name, ≤ cap
    truncated: bool                     # hit MAX_BROWSE_ENTRIES
    invalid_path: bool                  # submitted path bad → recovered to a valid dir
    root_missing: bool                  # MEDIA_ROOT itself absent / not a dir

def list_media_subdirs(media_root: Path, rel_path: str, registered: set[str]) -> BrowseListing
```

- Resolve `(root / rel_path).resolve()`. If it escapes containment / is reserved /
  is not a dir → **recover to root** and set `invalid_path=True` (honest, NOT a
  silent snap; the template shows a bounded generic warning — **never echoes the
  submitted path**). `root_missing=True` is reserved for `MEDIA_ROOT` itself being
  absent.
- List immediate children with `os.scandir`; keep only `is_dir(follow_symlinks=
  False)`; **skip all symlinks** and reserved trees; per-entry `OSError` → skip that
  entry (don't fail the listing); the browsed dir being unreadable → empty
  `entries` (containment already passed). Sort by name, then apply the cap →
  `truncated`. Mark each `registered` by set membership.
- No `settings` parameter (the cap is a module const, like `MAX_MEDIA_FOLDERS`).

### 2. Persistence — serialized, shared setup/settings helpers (app.py)

Three helpers mirroring `_persist_llm_settings` (return `str | None` error;
caller commits on success, re-renders panel with the message on error). **Each
acquires the singleton row `FOR UPDATE`, then rereads and validates under the
lock**, so overlapping requests serialize and no update is lost / no orphan
mapping survives:

- `_add_media_folder(session, settings, raw_path)` — `get_or_create` (insert-race)
  → `SELECT … WHERE id=1 FOR UPDATE` (guarded to postgresql) → normalize **only the
  submitted path** via `normalize_media_folders([raw_path], media_root)` (so one
  stale/vanished *existing* folder can't block a new add) → merge/dedup with the
  reread `media_folders`, enforce `MAX_MEDIA_FOLDERS` → reassign. Idempotent
  (re-adding an existing folder is a no-op).
- `_remove_media_folder(session, settings, folder)` — require `folder` ∈ the reread
  stored list (don't trust an arbitrary string; no filesystem touch). Drop it from
  `media_folders` **and** drop `folder_domain_packs[folder]`; reassign both.
  Idempotent.
- `_set_folder_pack(session, settings, folder, pack)` — require `folder` ∈ stored
  list. `pack == ""` (the **sole** Default sentinel) → delete the key. Else `pack`
  must be in `available_domain_packs(settings)` → set the key; reassign. Last-write
  wins. `DomainPackError` → honest error message, writes nothing.

**Invariant enforced by these helpers (the feature's contract):** every
`folder_domain_packs` key is a currently-registered `media_folders` entry, and
every non-empty value resolved against `available_domain_packs` at write time.

### 3. Routes — 2 per mount (consolidated), shared panel fragment

Consolidating to one browse + one mutate route per mount (vs 8 fine-grained
routes) cuts the onboarding-exempt + CSRF + inventory surface for a small feature.

- `GET  /setup/folders/browse?path=<rel>` → panel fragment. **Read-only, no CSRF**
  (authenticated, bounded, must not create the row; `Cache-Control: no-store`).
- `POST /setup/folders` — `_require_csrf(CSRF_SETUP)`; `action: Literal["add",
  "remove","pack"]` (422 on unknown) dispatched to the helpers; on `HX-Request`
  returns the panel fragment, else `303` → `/setup?step=media&path=<current>`.
- Mirror on `protected`: `GET /settings/folders/browse`, `POST /settings/folders`
  (`CSRF_SETTINGS`; non-HX `303` → `/settings#folders`).
- **Form bounds:** `path`/`folder` `Form(max_length=4096)`, `pack`
  `Form(max_length=200)` — small-form bounds (the app request limit is upload-sized
  and useless here). Oversized → validation error, no write.
- `_setup_context` (MEDIA step **only** — not welcome/vocab/llm/services/finish) and
  `_settings_context` read an optional `?path=` and build the panel via a shared
  `_folder_panel_context(session, settings, *, action_prefix, csrf, path)` that
  calls `list_media_subdirs` + `available_domain_packs` **once**.

### 4. Templates — one shared panel, outerHTML swap

- `templates/fragments/folder_panel.html` (NEW, shared) — root element
  `id="folder-panel"`; consumers target `#folder-panel` with `hx-swap="outerHTML"`
  (so the returned root replaces it — **no nested duplicate IDs**). Renders:
  - **Registered folders**: each with a Remove button and a pack `<select>`
    (options: "Default" = `""`, then sorted available pack names; a stored pack no
    longer available renders as an explicit **"(unavailable)"** selected option and
    the select is disabled; a registry-wide `DomainPackError` disables all selects
    with a visible message — never a false "Default"). A stored folder missing on
    disk renders with a "(missing)" note, still removable.
  - **Browser**: breadcrumbs (root→current), a "↑ up" link, child-dir rows (each an
    `<a>`/`hx-get` to browse + an Add button), an "Add this folder" action for the
    current dir (incl. root `.`), a bounded **"go to folder"** path input
    (server-canonicalized navigation — doubles as truncation recovery so no folder
    is ever unreachable), and an honest `truncated`/`invalid_path` notice.
  - All URLs `| urlencode` the path; Jinja autoescapes names.
  - Parameterized by `folders_browse_action`, `folders_mutate_action`,
    `folders_csrf` so setup (CSRF_SETUP) and settings (CSRF_SETTINGS) share it.
  - Non-JS: child links are real `?path=` page links; Add/Remove/Pack are ordinary
    `<form method=post>` (full-page 303) htmx-enhanced to swap the panel.
- `templates/settings/_folders.html` (NEW) — wraps the panel; included in
  `settings.html` after `_sources.html`.
- `templates/setup.html` — media step: **delete the textarea + its "Registered:"
  line**; render the panel. Keep the existing "Scan for existing media" block and
  the Continue link.

## Affected files

| File | Change |
|---|---|
| `src/voxint/api/setup_wizard.py` | + `list_media_subdirs`, `DirEntry`, `BrowseListing`, `MAX_BROWSE_ENTRIES`. |
| `src/voxint/api/app.py` | + `_folder_panel_context`; + `_add_media_folder`/`_remove_media_folder`/`_set_folder_pack` (row-locked); + 4 routes; media-step + settings context read `?path=`; **remove `POST /setup/media` textarea route** (folder reg now via the panel). |
| `src/voxint/api/templates/fragments/folder_panel.html` | NEW shared fragment. |
| `src/voxint/api/templates/settings/_folders.html` | NEW settings partial. |
| `src/voxint/api/templates/settings.html` | `{% include %}` the folders partial. |
| `src/voxint/api/templates/setup.html` | Media step: textarea → panel. |
| `tests/integration/test_onboarding_gate.py` | `EXEMPT_PATHS`: drop `/setup/media`; add `/setup/folders/browse`, `/setup/folders`. |
| `tests/unit/test_setup_wizard.py` | Unit tests for `list_media_subdirs`. |
| `tests/integration/test_setup_wizard.py` | Retire textarea-route tests; add browse/add/remove/pack (setup mount) + the mapping-prune invariant. |
| `tests/integration/test_settings_folders.py` | NEW — settings mount: browse/add/remove/pack, onboarding-gated, CSRF scope, honest degradation, + ingest-read-path proof. |
| `docs/onboarding.md`, `docs/architecture.md` | Media step = browser + pack picker; folder→pack resolution now UI-driven. |
| `CHANGELOG.md` | `[Unreleased]` entry; remove the "#11 folder→pack UI deferred" note. |

> ⚠ **Removing `POST /setup/media`** deletes a currently-tested route. Its tests
> (`test_post_media_*`) get replaced by the panel tests. If the maintainer would
> rather keep a bulk textarea path, the fallback (see Open questions) is to retain
> it but make it **atomically prune `folder_domain_packs` to the new folder set** —
> otherwise a textarea save orphans mappings. Default plan: remove it.

## Step-by-step implementation

1. **Contracts first (no code):** confirm — textarea removed (Open Q1); root `.`
   registration kept; truncation recovery = "go to folder" input; invalid-path =
   recover-to-root + bounded warning; non-HX = full-page 303/`?path=`. Express each
   as an assertion before writing templates.
2. **Pure listing** (`setup_wizard.py`) + unit tests (`tests/unit`). Gate: unit
   tests green (containment recover, reserved prune, symlink skip, per-entry OSError
   skip, sort+cap→truncated, root_missing, breadcrumbs/parent, root `.`).
3. **Row-locked persist helpers** + the consolidated routes (both mounts) + context
   threading (MEDIA step only). Gate: integration tests prove idempotency, no
   lost-update, no orphan mapping, CSRF scope, auth/onboarding gating, dirty
   tracking, form bounds.
4. **Shared panel + partials**; remove the textarea; wire `_folders.html`. Gate:
   both mounts render/mutate/navigate/degrade with exactly one `#folder-panel` id
   and no false "Default"/selected value; special-char + non-ASCII names
   escaped/encoded.
5. **Close contracts + docs:** update `EXEMPT_PATHS`; land the mapping-invariant
   contract test in this commit; onboarding + architecture + CHANGELOG. Gate: full
   `pytest tests --cov` (global fail-under 85), `ruff check .`, `mypy src/voxint`,
   `tests/contracts`.

## Testing strategy

- **Unit (`list_media_subdirs`):** absolute/`..`/symlink-escape/reserved/non-dir all
  recover to root with `invalid_path` and disclose **no** child entries; reserved
  trees pruned from a valid listing; symlinked dirs skipped; per-entry OSError
  skipped; `root_missing` when MEDIA_ROOT absent; sort + `MAX_BROWSE_ENTRIES`
  truncation; breadcrumbs/parent; root `.` and `current_registered`.
- **Integration (both mounts):** GET browse requires auth and does not create
  `app_settings`; setup routes exactly exempt / settings routes onboarding-gated;
  wrong-CSRF-scope rejected on both; add persists + dedups + `MAX_MEDIA_FOLDERS`
  boundary + duplicate-add no-op; remove drops folder **and** its mapping + is
  idempotent + rejects a non-registered string; pack sets/validates + `""` removes
  key + rejects unknown pack + rejects folder-not-registered; oversized form fields
  rejected; a stale mapping renders "(unavailable)" and total `DomainPackError`
  disables selects without hiding stored state; **two concurrent distinct Adds
  preserve both** and Remove-racing-Pack leaves no orphan (two sessions,
  FOR-UPDATE); fragment has exactly one `#folder-panel` and preserves the current
  path after every action; non-HX add/remove/pack returns a usable 303.
- **End-to-end read path:** after setting a folder→pack mapping via the route,
  submitting a media item under that folder freezes the selected pack in the run
  snapshot (`resolve_run_domain_pack` / ingest), proving the UI write reaches the
  existing consumer — not just DB storage.
- **Contract:** the keys-subset-of-`media_folders` + values-resolvable invariant
  (named test, lands same commit); `EXEMPT_PATHS` enumeration stays exact.
- **Final gate:** full `pytest tests --cov` (deselect the two host-only installer
  tests `test_detect_render_gid_*` and macOS `plutil` autoskips per the arc's
  session notes), ruff, mypy, contracts.

## Rollout / risks / open questions

- **Risk — TOCTOU** on `resolve`→`scandir`: accepted under the documented
  trusted-local-filesystem model (single-operator box); `openat`/`O_NOFOLLOW` is
  out of scope.
- **Risk — truncation unreachability:** mitigated by the server-canonicalized "go
  to folder" input + breadcrumb nav; `MAX_BROWSE_ENTRIES`=500 is generous for
  personal media trees, and truncation is shown honestly.
- **Risk — concurrent singleton writes:** mitigated by `FOR UPDATE` + post-lock
  reread; covered by concurrency tests.
- **Q1 — RESOLVED (maintainer, 2026-08-18):** honor "replace" literally —
  **remove the textarea** and `POST /setup/media`; the browser panel is the only
  registration UI; non-JS preserved via ordinary forms + full-page redirects. No
  bulk textarea baseline on Settings, so no `POST /settings/media`.
- **Q2 — RESOLVED (maintainer, 2026-08-18):** `MAX_BROWSE_ENTRIES = 500`.

## Review notes (codex second opinion via zen clink, planner role)

Codex verdict: *"sound direction, but revise before implementation."* Endorsed the
core architecture (bounded server-side listing, canonical validation, immediate
persistence, one shared panel, no migration, honest pack-failure handling).
Findings and resolutions:

- **[HIGH] Concurrency** — ACCEPTED. Added `SELECT … FOR UPDATE` on the singleton
  row for every mutation, reread+validate under the lock, idempotent Add/Remove,
  Pack = last-write; added concurrency tests. (Draft's read-modify-reassign could
  lose an Add or leave a Remove-vs-Pack orphan.)
- **[HIGH] Replacement semantics** — ACCEPTED. Remove the textarea (also resolves
  the draft's open question); this deletes the orphan-mapping obligation a
  bulk-save would carry. Non-JS preserved via ordinary forms + full-page redirect.
- **[HIGH] Honest invalid-path** — ACCEPTED. No silent snap-to-root; recover to
  root with an explicit bounded `invalid_path` warning that never echoes the
  submitted path; `root_missing` reserved for an actually-absent MEDIA_ROOT.
- **[HIGH] Truncation unreachability** — ACCEPTED. Added a server-canonicalized "go
  to folder" input as bounded recovery; truncation shown honestly.
- **[HIGH] Contract test** — ACCEPTED. `EXEMPT_PATHS` alone is insufficient; added a
  named keys-subset/values-resolvable invariant test in the same commit.
- **[MED] Route surface** — ACCEPTED. Consolidated 8 routes → 2 per mount (browse
  GET + mutate POST with an action enum); 2 new exempt paths instead of 4.
- **[MED] Context on every step** — ACCEPTED. Panel context built only on
  `WizardStep.MEDIA`; `available_domain_packs` loaded once per panel response.
- **[MED] Stale/failed pack rendering** — ACCEPTED. Stored-but-unavailable pack
  renders "(unavailable)" selected + disabled; registry failure disables selects
  with a message; both tested.
- **[MED] Mutation validation** — ACCEPTED. Add normalizes only the submitted path
  (a stale existing folder can't block it); Remove/Pack require membership in the
  stored list; stale registered folders surfaced separately ("(missing)").
- **[MED] Root `.` parity** — ACCEPTED. "Add this folder" for the current dir incl.
  root preserves `normalize_media_folders`' existing `.` capability.
- **[MED] htmx swap contract** — ACCEPTED. `id="folder-panel"` root + `hx-swap=
  outerHTML`; validated current path preserved every mutation.
- **[MED] Input bounds** — ACCEPTED. `Form(max_length=…)` on path/folder/pack;
  dropped `"__default__"` — empty string is the sole Default sentinel.
- **[LOW] Filesystem errors** — ACCEPTED. Per-entry OSError skips the entry;
  unreadable browsed dir → empty listing; no host-path leakage; tested.
- **[LOW] Docs** — ACCEPTED. Also update `docs/architecture.md`; remove the "UI
  deferred" CHANGELOG note.
- **[LOW] Coverage gate** — ACCEPTED. Final gate is the full `pytest tests --cov`
  (global fail-under 85), not changed-file targeting.
- **Q1 GET-no-CSRF for browse** — CONFIRMED (authenticated, read-only, bounded, no
  row creation, `Cache-Control: no-store`, honest errors).
- **Q4 symlinks** — CONFIRMED skip-all; canonical target still reachable via its
  real location; removing the textarea eliminates the alias inconsistency.
