# #477 Project archive and restore

Status: done

## Goal

Let the operator archive a project from the detail page's "..." overflow menu
(beside Rename) and restore it later. An archived project keeps its folders,
history, derived speakers, insights and quotes viewable, but becomes inert:
its settings are read-only, new recordings in its folders no longer inherit
its vocabulary or corrections, learned-corrections capture stops, new quote
saves are refused, and it drops out of the active list into a collapsed
"Archived projects" section. Today the projects router has no lifecycle at
all, so the overflow menu has nothing to hold. Part of epic #149.

## Assumptions and constraints

- **Maintainer decisions (binding, 2026-09-15):** folders stay linked on
  archive. The issue's "folders revert to unassigned" is rejected because
  every project view reaches the project through the folder chain, so
  unlinking hollows out history and makes restore return an empty shell.
  Archive/restore only; hard delete is a follow-up issue.
- **What "inert" means (precise):** no project configuration inheritance for
  new runs, no new learned-corrections capture, no new saved quotes. It does
  NOT mean frozen membership or frozen insights: a recording added to a
  retained folder while archived still appears in the archived project's
  views, and insight/trend/semantic caches keep computing. Retraction of
  invalidated learning evidence still runs on edits (so stale evidence does
  not survive restore). Existing saved quotes remain editable and deletable
  (curation of history, not configuration).
- Mirror the speaker roster lifecycle (`speakers/roster.py`
  `archive_speaker`/`restore_speaker`): FOR UPDATE lock, idempotent flag
  flip, flush not commit, per-action CSRF, 303 back to the page, archived
  surface read-only with a `notice error` banner, mutation routes 409.
- Column name is `archived_at`, matching `PipelineRun.archived_at`, not the
  speaker table's `deleted_at` naming divergence.
- Runs have no `project_id`; `PipelineRun.domain_pack` is a frozen snapshot
  taken at submit time. Archiving never touches existing runs. A submit
  whose config resolved while the project was active may commit after the
  archive; that is accepted (resolution-time semantics). The same overlap
  window is accepted for settings writes and quote saves in a single-operator
  tool; only learning capture gets a recheck because its locking read
  already exists.
- Project names stay globally unique including archived projects (no
  constraint change).
- No living spec declared. Single-operator tool: no query knobs, no index on
  a tiny table.
- Migration `0064` succeeds head `0063`; verify `alembic heads` first.

## Proposed approach

**Schema.** `Project.archived_at: Mapped[datetime | None]`
(`DateTime(timezone=True)`, nullable, no index), with a model comment naming
the three inertness choke points. Migration
`alembic/versions/0064_project_archived_at.py` adds only the column. Its
docstring records that downgrade drops archive state: archived projects come
back active under the old code, and a re-upgrade leaves `archived_at` NULL.

**Lifecycle service.** New `src/voxint/projects/lifecycle.py` (the
`projects_query.py` module is documented as side-effect free, and CLI or
import callers should not import from `voxint.api`):

- `ProjectNotFoundError`, `ProjectArchivedError`
- `archive_project(session, project_id) -> Project`: `select ... with_for_update()`,
  raise not-found, return unchanged if already archived, else stamp
  `datetime.now(UTC)` and `flush()`.
- `restore_project(session, project_id) -> Project`: symmetric.
- `require_active_project(session, project_id) -> Project`: raises the two
  errors; the router maps them to 404 / 409 in one helper
  (`_active_project_or_error`) that replaces the repeated
  `session.get(Project, ...)` / 404 prelude in rename, vocabulary,
  corrections, learning, suggestions and assign. The guard sits immediately
  after `_require_csrf` and before any branch, including the same-name
  rename early return and the corrections reset/JSON branches.
- `describe_project_name_owner(owner) -> str`: "A project named X is
  archived. Restore it instead of creating it again." vs "... already
  exists." (mirrors `roster.describe_name_owner`).

**Routes.** `POST /projects/{id}/archive` and `POST /projects/{id}/restore`
with new `CSRF_PROJECT_ARCHIVE` / `CSRF_PROJECT_RESTORE` constants; 303 to the
detail page. Six mutation routes gain the 409 guard (rename, vocabulary,
corrections including reset and the JSON path, learning, suggestions
accept/dismiss, assign folder). Unlink stays allowed so folders are never
stuck; moving a folder out of an archived project is unlink then assign (the
assign picker only lists unassigned folders, so no new "move" branch). Unlink
and reassignment change live historical membership and restore does not undo
them; the archived banner says so in one clause.

**Inertness at the three choke points that walk folder to project:**

1. `ingest/service.py` `_folder_and_project`: after the single joined read,
   return `(folder, None)` when the joined project is archived. The docstring
   redefines the second element as "the configuration-eligible project".
   Still one relational read, so the READ COMMITTED argument in
   `_resolve_run_config` holds; `provenance.project_name` becomes `None` for
   free, so the re-run preview honestly shows no project. Covers CLI submit,
   watch pickup, re-run and URL fetch (all go through
   `_run_domain_pack_snapshot`).
2. `adjudication/learned_corrections.py`: `_resolve_project` adds
   `Project.archived_at.is_(None)` to its join, AND the per-project locking
   read (`select(Project.id).where(...).with_for_update()`) gains the same
   predicate and bails when no row comes back, closing the resolve-then-lock
   window. `_retract_evidence` stays before the resolve, deliberately.
3. `api/saved_quotes.py` `save_quote`: after `resolve_project_id`, refuse
   when the project is archived by raising a dedicated
   `ProjectArchivedError` (from the lifecycle module) so the router can tell
   it apart from "not in a project"; `routers/quotes.py` returns 422 with
   `"This recording's project is archived. Restore it to save quotes."`.
   The ExploreIsland currently maps every 422 to the fixed title "Recording
   not in a project", which would be false here; the save button reads the
   JSON `error` on 422 and shows it as the title, falling back to the
   current copy. Small island change, island rebuild, no new prop.

**Detail page when archived.** `chip("archived")` (new CHIP_MAP entry,
`chip-neutral`) replaces `chip("active")`; banner "This project is archived:
its settings are read-only, and new recordings in its folders no longer use
its vocabulary or corrections. Existing quotes can still be edited. Restore
it to edit settings again."; the command bar shows only a "..." menu with
"Restore project". Audit of controls and copy, not just forms:

- Hidden: Rename, the vocabulary editor `details` and its "+ add term" chip
  (the chip opens that editor by DOM lookup and would be orphaned), the
  learning toggle, suggestion accept/dismiss, reset-to-inherited, the assign
  `details` and "+ link folder", the no-JS add-rule guidance.
- Copy: "Applies to the next run" becomes "Inactive while archived" on the
  stored vocabulary/corrections; the folder-row "superseded by this project"
  note is suppressed (the project no longer supersedes anything).
- Kept: unlink forms, all read-only content, quote board fully working
  (delete, note, export), temporal trends.
- The corrections-editor island is not mounted (omit the `data-island`
  attributes; its no-JS grid is the static rendering). Render the archived
  page with populated vocabulary, learning on, and folder packs, with and
  without JavaScript, in the browser lane.

**Detail page when active.** Keep the Rename dropdown and add a second
`<details class="cb-overflow">` with `aria-label="More actions"` and a
`class="danger"` "Archive project" form, copied from
`speakers/profile.html`.

**Projects list.** Roster pattern, not a `?archived=1` knob: `ProjectSummary`
gains `archived_at`; `list_projects` orders archived last; `_list_context`
splits `projects` (active) and `archived_projects`, mints `csrf_restore`.
Archived rows render inside a collapsed "Archived projects (N)" `<details>`
with an inline Restore form. Summary shows active and archived counts
("2 projects, 1 archived"); exact sentence variants are not acceptance
criteria.

**Pickers.** Explore project filter (`explore.py` `_filter_projects`) keeps
archived projects, labelled "(archived)", active first. The media folder
picker (`media_query.folder_options`) keeps its path ordering and only adds
"(archived)" to the project label, since the folder's project config no
longer applies there.

### Why not unlink folders on archive

See constraints. It is lossy, makes restore non-reversible, and the frozen
run snapshot already protects existing runs without it.

### Why no read-only quote board

Deleting or annotating an existing saved quote is curation of history, not
project configuration. Blocking only new saves (choke point 3) keeps the
archived project inert for new work without a `readOnly` island prop. The
banner names the exception so it does not read as an inconsistency.

## Acceptance criteria

### Requirement: Project archive and restore

The system SHALL let the operator archive an active project and restore an
archived one, idempotently, from the project detail page.

#### Scenario: Overflow menu on an active project

WHEN the operator opens an active project's detail page
THEN the command bar shows the `active` chip, the Rename dropdown, and a
"..." menu containing "Archive project" with its own CSRF token

#### Scenario: Archive or restore with a wrong-action token

WHEN "Archive project" or "Restore project" is submitted with a token minted
for another action
THEN the response is 403 and `archived_at` is unchanged

#### Scenario: Archive succeeds

WHEN "Archive project" is submitted with its token
THEN the response is 303 to the detail page
AND `archived_at` is set
AND the page shows the `archived` chip, the read-only banner, no rename,
vocabulary, add-term, learning, suggestion, reset, assign or link-folder
controls, no `data-island="corrections-editor"`, the unlink forms, the quote
board, and a "..." menu containing only "Restore project"

#### Scenario: Restore preserves state

GIVEN an archived project with two linked folders, an explicit empty
vocabulary, own corrections, learning on, one suggestion, one saved quote,
and a run frozen before the archive
WHEN "Restore project" is submitted
THEN `archived_at` is NULL, the detail page renders as active, and every
listed item is unchanged when read through a fresh session
AND a subsequent segment edit captures a new suggestion
AND a subsequent Explore save creates a `saved_quotes` row
AND the frozen run's `domain_pack` is byte-identical

#### Scenario: Idempotent replay and missing project

WHEN archive is replayed on an archived project, or restore on an active one
THEN the response is 303 and `archived_at` is unchanged
WHEN either is posted for a nonexistent id
THEN the response is 404

### Requirement: Archived project settings are read-only

The system SHALL refuse configuration changes to an archived project.

#### Scenario: Mutation routes refuse on every branch

WHEN rename (including the same-name case), vocabulary (set and inherit),
corrections (set, inherit, and the JSON path), learning (on and off),
suggestion (accept and dismiss), or assign-folder is POSTed with a valid
token against an archived project
THEN the response is 409 and the project row, learning flag, suggestion rows
and folder links are unchanged

#### Scenario: Unlink still works

WHEN unlink is POSTed against an archived project's folder
THEN the response is 303 and the folder's `project_id` is NULL

#### Scenario: Archived name blocks re-creation with guidance

WHEN a project is created or renamed to an archived project's name
THEN the response is 409 with "is archived. Restore it instead" guidance

### Requirement: Archived project is inert for new work

The system SHALL treat an archived project as absent when resolving config
for new runs, capturing learned corrections, and saving quotes, while
leaving history, membership and derived caches live.

#### Scenario: New run does not inherit archived config

GIVEN a folder whose project is archived and has its own vocabulary and
corrections
WHEN a run is submitted for media in that folder
THEN the frozen snapshot's vocabulary and corrections sources are `folder`
(or `global` with no folder pack) and `provenance.project_name` is None
AND the re-run preview and the dispatched snapshot agree
AND after restore, a new submit resolves the project again

#### Scenario: Learned corrections are not captured but stale evidence is retracted

GIVEN a run under an archived project with `learn_corrections` on,
corrections set, and one previously observed segment with evidence
WHEN that segment's edit is verified again with a different substitution
THEN its old evidence rows are retracted and no new `learned_corrections`
or evidence rows are written

#### Scenario: Quote save is refused with the reason

WHEN a KWIC row from a run under an archived project is saved from Explore
THEN the response is 422 with the archived message, no `saved_quotes` row
is written, and the Explore save button's title shows that message rather
than "Recording not in a project"

#### Scenario: Existing quotes stay editable

WHEN a note update, delete, or export is requested for a quote on an
archived project
THEN it succeeds as it does for an active project

#### Scenario: Membership stays live

WHEN a recording is added to a folder linked to an archived project
THEN the archived project's detail page lists it and its insights include it

### Requirement: Archived projects are listed separately

The system SHALL hide archived projects from the active list while keeping
them reachable.

#### Scenario: List page partitions

WHEN the projects list renders with two active and one archived project
THEN the summary shows two active and one archived
AND the archived project appears only inside the collapsed "Archived
projects (1)" section with an inline Restore form
AND submitting that form restores it into the active list

#### Scenario: Pickers label archived projects

WHEN the Explore project filter renders
THEN the archived project is labelled "(archived)" and sorts after active
projects
WHEN the media folder picker renders
THEN folders keep path order and a folder in an archived project shows
"(archived)" after its project name

## Spec deltas

Spec deltas: none (no living spec declared)

## Affected files / components

| File | Change | Why |
|---|---|---|
| `alembic/versions/0064_project_archived_at.py` | New migration, docstring on downgrade semantics | Add nullable column |
| `src/voxint/db/models.py` | `Project.archived_at` + comment naming the three choke points | Durable flag |
| `src/voxint/projects/__init__.py`, `src/voxint/projects/lifecycle.py` | New service | archive/restore/require_active/name guidance/errors |
| `src/voxint/api/csrf.py` | `CSRF_PROJECT_ARCHIVE`, `CSRF_PROJECT_RESTORE` | Per-action tokens |
| `src/voxint/api/routers/projects.py` | Two routes, `_active_project_or_error` placed after CSRF on six routes, name guidance, list/detail context | Lifecycle + read-only |
| `src/voxint/api/projects_query.py` | `archived_at` on `ProjectSummary` and `ProjectDetail`, list ordering | Partition + chip |
| `src/voxint/ingest/service.py` | `_folder_and_project` returns `None` project when archived; docstring | Choke point 1 |
| `src/voxint/adjudication/learned_corrections.py` | `_resolve_project` filter + archived predicate on the locking read | Choke point 2 |
| `src/voxint/api/saved_quotes.py`, `src/voxint/api/routers/quotes.py` | `ProjectArchivedError` on save, distinct 422 message | Choke point 3 |
| `frontend/src/components/ExploreIsland.tsx` | Read `error` from the 422 body for the button title | Honest copy |
| `src/voxint/api/routers/explore.py`, `src/voxint/api/media_query.py` | `archived` flag on picker rows | Labels |
| `templates/projects/project_detail.html` | Overflow menus, chip, banner, hidden controls, copy audit, island not mounted | Read-only surface |
| `templates/projects/projects.html` | Archived section + counts | Partition |
| `templates/fragments/_chips.html` | `archived` entry | Chip |
| `templates/explore/explore.html`, `templates/media/media.html` | "(archived)" labels | Pickers |
| `tests/integration/test_migration_0064.py` | New; seeds archived rows before downgrade | Up/down with data |
| `tests/contracts/test_project_archive.py` | New | Column + CSRF distinctness |
| `tests/contracts/fixtures/route_inventory.json`, `console2_route_order.json`, `console2_route_characterization.json` | Regenerate; diff must add exactly two POST entries | Two new routes |
| `tests/integration/test_projects.py`, `test_config_resolution_freeze.py`, `test_review_api.py`, `test_quotes_api.py`, explore and media page tests | Extend | Scenarios above |
| `CHANGELOG.md`, `docs/operations.md`, `docs/architecture.md` | Entry; project archive bullet beside run archive + downgrade note; `projects` row in data-model table, revision range 0064, precedence sentence | Docs in same change |

## Implementation slices

### Slice 1: Schema

- Model column with comment; migration 0064 with downgrade docstring.
- `test_migration_0064.py`: copy the 0063 shape, then seed one active and
  one archived project with linked folders via SQL, downgrade, assert rows
  and links survive, re-upgrade, assert `archived_at` is NULL.
- Contract test for column presence on `Project.__table__`.
- Verify: `alembic heads` is `0064`, full suite green.

### Slice 2: Lifecycle service, routes, detail page

- `projects/lifecycle.py`; CSRF constants; archive/restore routes;
  `ProjectDetail.archived_at`; `_detail_context` mints both tokens.
- Template: chip, banner, overflow menus, hidden controls and copy audit per
  the approach section, corrections-editor not mounted when archived.
- `_active_project_or_error` on the six mutation routes, placed after
  `_require_csrf` and before any branch; duplicate-name guidance in create
  and rename `IntegrityError` branches.
- Regenerate the three route goldens with the existing enumerators;
  inspect the diff and require exactly the two new POST entries (removing
  them from the order fixture must reproduce the previous order). Run
  inventory, order, characterization and CSRF contracts together.
- Tests in `test_projects.py`: overflow scrape (model:
  `test_profile_overflow_archive_posts_with_its_own_csrf_token`), wrong
  token 403 for both actions, archive round trip, populated restore round
  trip read through a fresh session, idempotent replay, 404 on missing id,
  parameterized 409 matrix over every branch with persisted-state
  assertions, unlink allowed, name guidance, archived page with populated
  vocabulary/learning/folder packs asserting hidden controls and copy.
- Verify: full suite green, ruff, mypy.

### Slice 3: Inertness

- `_folder_and_project` docstring + archived check; `_resolve_project`
  filter and locking-read predicate; `save_quote` raises
  `ProjectArchivedError`; quotes router message; ExploreIsland reads the 422
  error; rebuild and stage islands.
- Tests: `test_config_resolution_freeze.py` (extend `_seed_project_folder`
  with `archived_at`; folder fallback, global fallback, explicit-pack
  precedence, preview/dispatch agreement, then restore and re-submit),
  `test_review_api.py` (beside `test_learning_toggle_off_produces_no_rows`:
  retraction runs, no new rows; capture resumes after restore),
  `test_quotes_api.py` (422 message, no row, save works after restore;
  note/delete/export on an archived project's quote succeed).
- Verify: full suite green; `tests/contracts/test_frontend_build.py` green
  after the island rebuild.

### Slice 4: List page and pickers

- `ProjectSummary.archived_at`, ordering, `_list_context` split +
  `csrf_restore`; `projects.html` archived section, counts, empty state
  "No active projects. Restore one below or create a new one."
- `_filter_projects` gains `archived` and active-first order;
  `folder_options` gains `project_archived` with unchanged path order; label
  sites in `explore.html` and `media.html`.
- Tests: list partition + inline restore in `test_projects.py`; one
  assertion each in the explore and media page tests (label present, order
  preserved); membership-stays-live assertion on the detail page.
- Verify: full suite green.

### Slice 5: Docs, changelog, browser lane

- CHANGELOG `[Unreleased]` Added entry (name the downgrade caveat);
  `docs/operations.md` bullet beside the run archive block plus the
  rollback note; `docs/architecture.md` `projects` row, revision range
  0064, precedence sentence: "An archived project is skipped: its folders
  resolve as if unassigned for new runs, learned-corrections capture and
  new quote saves are off, history and membership stay live."
- Browser acceptance lane: detail-page archive/restore round trip (menu
  opens, chip and banner flip, controls hidden, restore returns them),
  archived page with populated config with and without JavaScript, the
  list-page collapsed section, and an Explore save on an archived project
  showing the archived message.
- Verify: full suite green, `gitleaks dir .`, browser lane notes recorded in
  the PR.

## Testing strategy

| Gate | What it covers |
|---|---|
| Migration test (slice 1) | Up adds nullable column, down with archived rows present keeps projects and folder links, re-up leaves NULL |
| Contract (slices 1, 2, 3) | Column on `Project.__table__`; CSRF constants distinct; route goldens gain exactly two POSTs with CSRF enforced; frontend build after island change |
| Integration: routes (slice 2) | Every scenario under the first two requirements, including the populated restore round trip and the parameterized 409 matrix |
| Integration: inertness (slice 3) | The three choke-point scenarios, retraction-still-runs, restore re-enabling inheritance, capture and saves; existing quote edits unaffected |
| Integration: list + pickers (slice 4) | Partition, counts, inline restore, labels, ordering, live membership |
| Browser lane (slice 5) | Observable console behaviour on detail, list and Explore |
| Regression | Full suite green; existing project tests unchanged |

Not tested: the two-session deterministic ordering of archive against an
in-flight settings write or quote save. The overlap window is accepted by
design for a single operator and recorded in the constraints; the learning
path is the one place with a lock, and its recheck is covered by the
predicate on the locking read.

## Rollout / risks / open questions

### Risks

- **Route golden churn**: regenerate the three fixtures in the slice 2
  commit with a constrained diff, or the contract suite goes red or blesses
  unrelated drift.
- **Stale-tab JSON path**: a corrections-editor tab opened before archival
  can still POST; it gets a bare 409 rather than the island's
  `{"ok": false}` shape and shows a generic error. Accepted; do not lift the
  guard to "fix" it.
- **Inertness by omission**: future code that walks folder to project for
  new work outside the three choke points would silently re-inherit.
  Mitigation: the model comment names them and `docs/architecture.md`
  records the invariant. Read-side consumers (`project_insights`,
  `temporal_trends`, `explore_query`, `semantic_layout`) are deliberately
  untouched.
- **Chip fallthrough**: a missing CHIP_MAP entry degrades to `chip-neutral`
  with the raw label, which renders correctly anyway. The entry is added for
  explicitness; no test claims to guard it.
- **Downgrade loses archive state**: documented in the migration docstring,
  operations doc and changelog; requires a matching application rollback.
- **Migration number**: verify `alembic heads` before slice 1.

### Open questions

- None blocking. Hard delete is a follow-up issue (cascade semantics for
  learned corrections, saved quotes, corpus-analysis artifacts by
  `scope_id`).

## Review notes

### Codex (planner role, 2026-09-15)

Verdict: sound approach with correctness and validation gaps. Eleven
findings; three concrete code claims were verified against the tree before
acceptance (Explore ignores the 422 body, `_retract_evidence` runs before
`_resolve_project`, the "+ add term" chip dereferences the editor).

**Accepted (folded in):**

| Finding | Resolution |
|---|---|
| Explore discards the archived-project 422 explanation | Island reads the `error` field on 422 for the button title; dedicated `ProjectArchivedError` so the router can emit a distinct message |
| Learning filter does not stop all learning-state mutation | Redefined: archive stops capture only; retraction still runs (stale evidence must not survive restore). Scenario rewritten accordingly |
| Archive lock is not a universal cutoff | Resolution-time semantics accepted for submits; overlap accepted for settings and quotes; learning's locking read gains the archived predicate |
| Archived views stay live, including caches | Inertness defined precisely in constraints; membership-stays-live scenario added; read-side consumers explicitly untouched |
| Hidden forms leave orphaned controls and misleading copy | Copy and control audit added ("+ add term", "+ link folder", "Applies to the next run", "superseded" note, no-JS guidance); browser lane renders populated archived page with and without JS |
| Restore coverage too thin | Populated restore round trip through a fresh session; capture and save resume after restore; frozen run unchanged; 403 and 404 on restore |
| 409 guard placement and branch coverage | Guard after CSRF before any branch; parameterized matrix with persisted-state assertions |
| Downgrade loses archive state | Migration test seeds archived rows; docstring, operations doc and changelog note it |
| Golden regeneration needs a constrained diff | Exactly two new POST entries; order fixture round-trip check; run all four contracts together |
| Chip assertion cannot catch the failure | Claim removed; entry kept for explicitness |
| List copy over-specified, picker reorder changes navigation | Counts only; media folder picker keeps path order; Explore active-first |

**Confirmed by codex, kept:** filtering inside `_folder_and_project` with a
"configuration-eligible project" docstring; writable quote board with
refused new saves; unlink allowed with a banner clause that restore does not
undo it.

**Rejected:** none. The two-session deterministic ordering test suggested
under finding 3 is deferred as disproportionate for a single-operator tool
(recorded under Testing strategy).

## Completion notes

**Completed:** 2026-09-15. PR #486 merged at `22db4d0`, all 5 slices landed.

**Verification:** 4572 contract + unit tests pass, 113 integration tests for
the touched files pass (migration 0064, projects, config resolution, quotes
API, project archive contract). Three-model code review (Codex, Grok, Kimi)
with 4 findings fixed and 3 dismissed. Browser acceptance lane passed all 8
assertions (archive/restore round trip, read-only controls, suggestions
read-only, list-page partition, detail-page restore, Explore picker labels,
Explore save-on-archived error message).

**Scenario coverage:** 12 of 15 acceptance scenarios have dedicated automated
tests. Three lack a dedicated test, all accepted in code review as non-blocking
test-only gaps (no code defects): learning inertness retract-without-capture,
membership stays live, picker label assertions. The first two are WHERE-predicate
additions with no new code path; the third was verified in the browser lane.

**Spec files updated:** none (no living spec declared).

**Drift:** `test_migration_0016.py` received a 1-line import adjustment; no
feature drift.

**Follow-ups:** hard delete (cascade semantics for learned corrections, saved
quotes, corpus-analysis artifacts by `scope_id`). The three test gaps above
are candidates for a test-debt pass.
