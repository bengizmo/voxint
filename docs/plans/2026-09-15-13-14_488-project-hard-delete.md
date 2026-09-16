# Plan: #488 Project hard delete

Status: done

## Context

#477 landed archive/restore for projects (PR #486, v0.38.0). Archive is soft
and reversible: it stamps `archived_at`, makes the project inert (no new runs
inherit it, no new learned corrections, no new saved quotes), but preserves
all data and links. Issue #488 adds the next step: permanent deletion of an
archived project. This is destructive and not undoable. The existing DB
cascade rules already handle most of the work; the main implementation effort
is the confirmation UX, the explicit cleanup of soft-referenced artifacts, and
correct transaction boundaries.

## Assumptions and constraints

- **Single-operator deployment** -- concurrent access to the same project from
  multiple browser tabs is unlikely but handled gracefully (FOR UPDATE
  serializes lifecycle operations).
- **No alembic migration needed** -- the existing FK cascade rules are correct:
  `learned_corrections` CASCADE, `saved_quotes` CASCADE, `media_folders` SET
  NULL. The unique constraint index on `corpus_analysis_artifacts(scope_kind,
  scope_id, artifact_kind, generation)` serves the cleanup query.
- **No filesystem cleanup** -- all project-scoped data is DB-only (JSONB
  payloads in `corpus_analysis_artifacts`, FK'd evidence rows, JSONB corrections
  on the project row). No on-disk caches, exports, or attachments keyed by
  project ID.
- **No ORM delete hooks** -- no `before_delete`/`after_delete` event listeners
  exist on `Project`. Bulk DML (`delete(Project).where(...)`) is safe.
- **`scope_id` is UUID** -- name reuse after delete cannot collide with orphaned
  artifact rows because the new project gets a fresh UUID.
- **`projects.name` has `unique=True`** -- deleting the row frees the name
  automatically; no partial index or archived-name reservation logic needed.

## Proposed approach

Server-rendered confirmation page showing the blast radius (what will be
destroyed and what will survive), with a POST that executes the delete in a
single transaction. Matches the existing `rerun_confirm.html` pattern.

### Service layer: `lifecycle.delete_project(session, project_id)`

1. `SELECT ... FOR UPDATE` on the project row (serializes against concurrent
   archive/restore/delete).
2. If missing: raise `ProjectNotFoundError`.
3. If `archived_at IS NULL`: raise `ProjectNotActivelyArchivedError` (new, or
   reuse an appropriate exception).
4. Delete `corpus_analysis_artifacts` WHERE `scope_kind='project'` AND
   `scope_id=project_id` (explicit cleanup -- no FK).
5. Capture the project name (for flash/redirect messaging) before the delete.
6. Delete the project row via bulk DML: `delete(Project).where(Project.id ==
   project_id)`. DB cascades handle `learned_corrections` (+ evidence),
   `saved_quotes`, and `media_folders` SET NULL.
7. Flush (do not commit -- caller owns the transaction, following the existing
   lifecycle pattern).

### Confirmation page: GET `/projects/{project_id}/delete`

- Loads the project row and verifies archived (redirect to detail if active,
  404 if missing).
- Runs lightweight count queries: learned corrections, saved quotes, linked
  folders.
- Renders `project_delete_confirm.html` with two-sided blast radius:
  - **Will be permanently deleted**: the project and its settings
    (vocabulary, corrections, learning config), N learned corrections (with
    evidence), N saved quotes, cached analysis artifacts.
  - **Will be kept**: N folders (become unassigned), all recordings,
    transcripts, pipeline runs, and embeddings.
- Cancel button returns to the project detail page. "Permanently delete
  project" button POSTs with CSRF token.
- GET is read-only (no side effects, safe for prefetch/back-button).

### POST `/projects/{project_id}/delete`

- Requires valid `CSRF_PROJECT_DELETE` token (new: `"project-delete"`).
- Calls `lifecycle.delete_project()`.
- On success: 303 redirect to `/projects`.
- On active project: 409.
- On missing project (including double-submit): 404.

### Detail page integration

- Archived project's overflow "..." menu gains a "Delete permanently" link
  pointing to the GET confirmation page.
- Active projects do not see the delete option (archive-first gate).

### Artifact writer race (accepted, documented)

The four artifact cache writers (project_insights, temporal_trends, term_stats,
semantic_layout) use `pg_advisory_xact_lock` with artifact-specific keys. These
do not conflict with the project row's `FOR UPDATE` lock. A theoretical race
exists: a writer that started before the delete transaction committed could
insert an orphan artifact after the cleanup.

This is accepted because: (a) the project must be archived, and the only way
to trigger a writer is to view the archived project's detail or explore page
from another tab simultaneously; (b) `scope_id` is a UUID, so a new project
with the same name will not collide; (c) orphaned artifacts are bounded
(at most ~5 rows per project, one per artifact kind) and are never read again
because no route serves the deleted project; (d) acquiring all four
artifact-specific advisory locks in `delete_project` would couple the delete
path to every writer's key derivation, which is invasive for a near-zero risk.

A post-delete cleanup pass (second DELETE of artifacts, same transaction) is
included to narrow the window: it catches artifacts committed by a concurrent
writer between the first cleanup and the project row delete within the same
snapshot. Artifacts committed after our entire transaction commits are the only
true orphans.

Note: `term_stats` has a Celery background task trigger
(`compute_term_stats`, dispatched post-run-completion), so the race is not
purely interactive. However, archived projects do not produce new runs, so the
Celery path is naturally excluded for archived projects. The interactive path
(viewing the archived project's explore page) is the only realistic trigger.

**Rejected alternatives:**
- JS `confirm()` only: doesn't show blast radius, insufficient for an
  irreversible cascading delete.
- Soft-delete tombstone: archive already fills this role; adding `deleted_at`
  touches every query for no benefit.
- Typed-name confirmation gate: overengineered for single-operator where the
  project is already archived and named on the confirmation page.
- DB trigger for archive-before-delete: CHECK can't inspect deleted rows; a
  trigger is overkill for app-level enforcement.

## Acceptance criteria

### Requirement: archive-before-delete guard

The system SHALL refuse to delete an active (non-archived) project.

#### Scenario: attempt to delete an active project via POST
WHEN a DELETE POST targets a project with `archived_at IS NULL`
THEN the system responds 409 and the project is unchanged

#### Scenario: attempt to delete an active project via GET confirmation
WHEN the operator navigates to the delete confirmation page for an active project
THEN the system redirects to the project detail page

#### Scenario: attempt to delete a non-existent project
WHEN a DELETE request targets a UUID that does not match any project
THEN the system responds 404

#### Scenario: double-submit (second POST after successful delete)
WHEN a DELETE POST targets a project that was already deleted
THEN the system responds 404

### Requirement: cascade on delete

The system SHALL cascade a project deletion to its learned corrections,
evidence, saved quotes, and corpus analysis artifacts, and SHALL SET NULL the
project_id on linked folders, while preserving pipeline runs, segments, and
embeddings.

#### Scenario: project with full related data is deleted
GIVEN a project with learned corrections (with evidence), saved quotes, linked
folders, and corpus analysis artifacts (multiple kinds)
WHEN the project is permanently deleted
THEN learned_corrections rows for the project are deleted
AND learned_correction_evidence rows for those corrections are deleted
AND saved_quotes rows for the project are deleted
AND corpus_analysis_artifacts with scope_kind='project' and scope_id matching
the project are deleted
AND media_folders previously linked to the project have project_id=NULL and
remain usable (can be reassigned to another project)
AND pipeline_runs, transcript_segments, and segment_embeddings are unchanged
AND corpus_analysis_artifacts for other projects and other scope_kinds are
unchanged

### Requirement: confirmation page with blast radius

The system SHALL display a confirmation page showing both what will be deleted
and what will be retained before executing a project delete.

#### Scenario: confirmation page displays counts
GIVEN an archived project with 3 learned corrections, 5 saved quotes, and 2
linked folders
WHEN the operator navigates to the delete confirmation page
THEN the page displays the project name, the deletion counts (3 corrections,
5 quotes), what will be kept (2 folders become unassigned, recordings and
transcripts remain), and a warning that this cannot be undone

#### Scenario: confirmation page for empty project
GIVEN an archived project with no corrections, no quotes, and no folders
WHEN the operator navigates to the delete confirmation page
THEN the page displays zero counts and still warns this cannot be undone

### Requirement: CSRF protection

The system SHALL require a valid, action-scoped CSRF token for the delete POST.

#### Scenario: invalid CSRF token
WHEN the delete POST carries an invalid CSRF token
THEN the system responds 403 and the project is unchanged

#### Scenario: missing CSRF token
WHEN the delete POST carries no CSRF token
THEN the system responds 403 and the project is unchanged

#### Scenario: cross-action CSRF token
WHEN the delete POST carries a valid CSRF token minted for a different action
(e.g., project-archive)
THEN the system responds 403 and the project is unchanged

### Requirement: name release

The system SHALL release the project's unique name on delete, making it
available for new projects.

#### Scenario: reuse deleted project name
GIVEN a project "Field Study A" was permanently deleted
WHEN the operator creates a new project named "Field Study A"
THEN the project is created successfully with a new UUID, no inherited config,
and no automatic relinking of old folders

## Spec deltas

Spec deltas: none (no living spec declared)

## Affected files / components

| File | Change |
|---|---|
| `src/voxint/projects/lifecycle.py` | Add `delete_project(session, project_id)` |
| `src/voxint/api/csrf.py` | Add `CSRF_PROJECT_DELETE = "project-delete"` |
| `src/voxint/api/projects_query.py` | Add `project_delete_counts(session, project_id)` returning a dataclass |
| `src/voxint/api/routers/projects.py` | Add GET + POST `/projects/{project_id}/delete` routes |
| `src/voxint/api/templates/projects/project_delete_confirm.html` | New: confirmation page |
| `src/voxint/api/templates/projects/project_detail.html` | Add "Delete permanently" to archived overflow menu |
| `tests/integration/test_projects.py` | Integration tests for all scenarios |
| `CHANGELOG.md` | Entry under [Unreleased] |

## Implementation slices

### Slice 1: Service layer + cascade integration tests

- Add `delete_project(session, project_id)` to `lifecycle.py`: FOR UPDATE
  lock, verify archived, delete artifacts (two passes), delete project row via
  bulk DML, flush.
- Add `CSRF_PROJECT_DELETE` to `csrf.py`.
- Add integration tests in `test_projects.py`:
  - **Full cascade test**: seed project with learned corrections + evidence,
    saved quotes, linked folders, corpus_analysis_artifacts (multiple kinds),
    plus a control project with its own data. Delete the target project. Assert
    all target data is gone, folders have `project_id=NULL` and remain usable
    (all other columns intact, can be reassigned to another project),
    runs/segments/embeddings survive, control project's data is untouched.
  - **Archive guard**: POST on active project returns 409, project unchanged.
  - **Missing project**: POST on nonexistent UUID returns 404.
  - **Double submit**: POST twice, second returns 404.
  - **Idempotent evidence cascade**: evidence rows delete through
    learned_corrections cascade (not direct project FK).
  - **FK-inventory schema-contract test**: query `information_schema` for all
    FKs referencing `projects.id` and assert the expected set (3 FKs:
    learned_corrections CASCADE, saved_quotes CASCADE, media_folders SET NULL).
    Catches future tables added without a cascade decision.
- Gate: `VOXINT_TEST_DATABASE_URL=... uv run pytest tests/integration/test_projects.py -k delete -q`

### Slice 2: Counts query + confirmation template + GET route

- Add `project_delete_counts(session, project_id)` to `projects_query.py`:
  independent COUNT queries for learned_corrections, saved_quotes,
  media_folders. Use the same predicates as the delete (project_id match), not
  the detail-page's filtered suggestion queries.
- Add `project_delete_confirm.html` template: extends base, two-sided blast
  radius (deleted vs retained), cancel and delete buttons with CSRF token.
- Add GET `/projects/{project_id}/delete` route: verify archived (redirect to
  detail if active, 404 if missing), load counts, render template.
- Integration tests:
  - GET returns 200 with correct counts for a seeded archived project.
  - GET for active project redirects to detail.
  - GET for missing project returns 404.
  - Counts match seeded data (zero case, multiple corrections with evidence).
  - GET is read-only (no mutations, no cache writes).
- Gate: integration tests pass.

### Slice 3: POST route + detail page menu + CHANGELOG

- Add POST `/projects/{project_id}/delete` route: verify CSRF, call
  `delete_project`, 303 redirect to `/projects`. Same operator/auth/feature
  guards as existing project routes.
- Add "Delete permanently" link in archived project's overflow menu in
  `project_detail.html` (links to GET confirmation page, not a direct POST).
- Integration tests:
  - Invalid CSRF → 403.
  - Missing CSRF → 403.
  - Cross-action CSRF (archive token) → 403.
  - Successful delete → 303 redirect, project gone from list.
  - Name reuse after delete: create new project with same name, new UUID, no
    inherited config, old folders not auto-relinked.
  - POST for active project → 409.
  - Archived detail page shows "Delete permanently" in overflow menu.
  - Active detail page does NOT show "Delete permanently".
- Add CHANGELOG entry under [Unreleased].
- Gate: full test suite passes. Browser acceptance lane if running.

## Testing strategy

- **Integration tests** (primary gate): all scenarios verified against a real
  Alembic-migrated PostgreSQL, using the existing `session_factory` pattern
  (TRUNCATE between tests).
- **One comprehensive cascade test** seeds the full graph (project with
  corrections + evidence, quotes, folders, artifacts for all 4 kinds, plus a
  control project) and asserts the complete postcondition matrix in a single
  test. Seeded evidence rows verify the transitive cascade through
  learned_corrections.
- **Count fidelity**: counts query uses the same predicates as the delete, and
  a test verifies counts match seeded data.
- **CSRF**: three negative paths (invalid, missing, cross-action) following the
  existing test pattern.
- **Browser acceptance lane**: this changes observable console behavior (new
  confirmation page, new menu option). Per the review gating policy, the E2E
  browser lane should run. Key assertions: overflow menu visibility
  (archived-only), confirmation page renders counts, Cancel returns to detail,
  successful delete removes project from list, name reuse works.
- **No parity/numerics impact**: CRUD operation, no inference changes. Gate E
  not needed. Verify that other projects' insights/explore pages still work
  after a delete (no cross-project side effects).

## Risks and open questions

1. **Artifact writer orphans**: accepted and documented. Post-delete cleanup
   pass narrows the window. Orphaned rows are bounded, never re-read, and
   cannot collide with new projects (UUID scope_id). See "Artifact writer race"
   section above.
2. **ORM session state after bulk DML**: after `delete(Project).where(...)`,
   any in-process `Project` objects are stale. All tests use fresh sessions or
   HTTP-stack assertions, so this is not a concern in practice.
3. **Restore-then-delete race**: operator restores a project while the delete
   confirmation page is open. The POST's FOR UPDATE + archived_at check catches
   this: it will see the restored project and return 409.
4. **Large artifact tables**: the unique constraint index on
   `(scope_kind, scope_id, artifact_kind, generation)` serves the cleanup
   DELETE efficiently. Artifact rows per project are bounded (~5, one per kind).

## Review notes

### 5-model consult summary (codex + deepseek + grok + glm + kimi-k3)

**Theme 1: Transaction boundary must be explicit (4/4 agree)**
All four models flagged that the draft plan didn't specify a single transaction.
**Resolution**: ACCEPTED. Plan now explicitly states: single transaction, lock +
verify + artifact cleanup + project delete + flush, caller commits.

**Theme 2: Confirmation page must distinguish deleted vs retained (4/4)**
All models said "blast radius counts" is insufficient; must also show what
survives (folders unassigned, media/runs kept).
**Resolution**: ACCEPTED. Two-sided blast radius in the template.

**Theme 3: CSRF testing too narrow (4/4)**
Missing token and cross-action token replay need testing, not just wrong token.
**Resolution**: ACCEPTED. Three CSRF negative paths added to acceptance criteria.

**Theme 4: Idempotency / double submit (3/4)**
Second POST after successful delete should return 404, not 500.
**Resolution**: ACCEPTED. Explicit scenario added.

**Theme 5: Evidence cascade needs explicit verification (4/4)**
Evidence deletes transitively through learned_corrections, not directly from
project. Needs its own assertion.
**Resolution**: ACCEPTED. Full cascade test seeds evidence rows explicitly.

**Theme 6: Advisory lock alignment to close orphan race (3/4)**
Codex, Grok, DeepSeek suggested taking the same advisory locks as the four
writers. GLM suggested documenting the gap.
**Resolution**: REJECTED as invasive (couples delete to every writer's key
derivation for a near-zero-probability single-operator race). ACCEPTED the
alternative: post-delete cleanup pass + explicit documentation + note that
orphans are bounded and harmless. `scope_id` is UUID, so no collision.

**Theme 7: Audit for non-DB project state (3/4)**
Grok, DeepSeek, GLM suggested checking for filesystem artifacts.
**Resolution**: VERIFIED absent. All project-scoped data is DB-only (JSONB
payloads, FK'd evidence rows). No file cleanup needed. Documented in
assumptions.

**Theme 8: Typed-name confirmation gate (2/4)**
Grok and GLM suggested requiring the operator to type the project name.
**Resolution**: REJECTED. Overengineered for single-operator where the project
is already archived and named prominently on the confirmation page.

**Theme 9: Raw SQL vs ORM delete (2/4)**
GLM questioned necessity; DeepSeek said audit hooks first.
**Resolution**: VERIFIED no ORM delete hooks exist. Bulk DML
(`delete(Project).where(...)`) is cleaner than `session.delete()` because it
delegates entirely to DB cascades without loading children. Terminology
corrected per Codex: "bulk DML" not "raw SQL".

**Theme 10: Index on (scope_kind, scope_id) (1/4)**
DeepSeek raised performance concern for artifact cleanup.
**Resolution**: VERIFIED covered. The unique constraint on
`(scope_kind, scope_id, artifact_kind, generation)` creates an index with
`scope_kind, scope_id` as leading columns. Sufficient for the cleanup DELETE.

**Theme 11: Folder post-delete state (2/4)**
Grok and GLM noted the test should verify folders remain usable after SET NULL.
**Resolution**: ACCEPTED. Cascade test asserts folders have `project_id=NULL`
and can be reassigned.

**Theme 12: GET route applies same guards as POST (2/4)**
DeepSeek and Codex noted GET should enforce archive/404 checks.
**Resolution**: ACCEPTED. GET redirects active projects to detail, 404s for
missing.

**Theme 13: FK-inventory schema-contract test (1/5 -- K3)**
K3 proposed a test that queries `information_schema` for all FKs referencing
`projects.id` and asserts the expected set. Catches future tables added without
a cascade decision.
**Resolution**: ACCEPTED. Added to slice 1 -- high value relative to effort.

**Theme 14: Startup janitor for orphaned artifacts (1/5 -- K3)**
K3 noted that `term_stats` has a Celery background trigger, making the race
not purely interactive, and suggested a startup cleanup query.
**Resolution**: NOTED but not added to this plan. Archived projects don't
produce new runs, so the Celery path is naturally excluded. A janitor is a
reasonable future addition but not blocking for #488.

**Theme 15: TOCTOU on name-addressed routes (1/5 -- K3)**
K3 flagged that name-based routes could cause delete-the-wrong-project if the
name is reused between confirmation and POST.
**Resolution**: NON-ISSUE. Routes are UUID-based (`{project_id}`), not
name/slug.

**Theme 16: Capture project name before delete (1/5 -- K3)**
For flash/redirect messaging. `ObjectDeletedError` risk if accessed after bulk
DML.
**Resolution**: ACCEPTED. Capture name before the delete statement.

**Theme 17: Deadlock surface (1/5 -- K3)**
Delete locks project then cascades to folders; concurrent folder reassignment
is a potential AB-BA deadlock.
**Resolution**: NOTED. Surfaces as a retryable 500 with no corruption.
Acceptable for single-operator; not worth lock-order refactoring.

**Theme 18: Counts predicate must equal cleanup predicate (2/5 -- K3 + Codex)**
Count queries and delete must use identical filters to avoid divergence.
**Resolution**: ACCEPTED. Plan specifies using the same predicates.
