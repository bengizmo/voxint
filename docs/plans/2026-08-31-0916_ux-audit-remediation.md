# UX audit remediation plan

> **Status:** APPROVED 2026-08-31. Execution is sliced across sessions; each
> slice is a GitHub issue under the remediation epic and lands as one PR.

## 1. Background

A full visual audit of the review console (v0.31.0) ran on 2026-08-31: every
user-facing page captured full-page at 1440px plus a 768px subset on
maintainer hardware, cataloged control-by-control, and reviewed by the
maintainer plus a four-model design panel (independent AI design reviews with
differing visual grounding). The consolidated findings: the media editor's
desktop layout is broken and mis-ordered, the settings hub is a wall of text
with a confusing tri-state control pattern, maintainer telemetry (paths,
hashes, worker ids, lease expiries, raw confidence decimals) renders on
operator pages, base type is 13px everywhere, and the Jobs/Runs/Review trio
presents one lifecycle three ways. The audience mandate makes these
correctness-of-experience issues, not polish: Voxint serves non-technical
researchers, journalists, and educators.

The screenshots and the raw audit are maintainer working artifacts and are
not committed to this repo.

## 2. Code findings that reshaped the audit

Exploration corrected three audit conclusions before planning:

- **The editor already ships full speaker assignment.**
  `frontend/src/components/SpeakerRail.tsx` (decide/enroll/merge against
  `/review/{run}/labels/{label}/...`), a per-segment assign select
  (`MediaEditor.tsx`), and digit-key assignment all exist. The audit's "no
  naming control" was the collapsed rail plus the unclaimed read-only state
  hiding them. The remedy is layout, claim friction, and discoverability,
  not new naming UI.
- **The rail collapse has a located root cause**: double-nested grids.
  `src/voxint/api/templates/editor/detail.html` wraps the page in
  `.lib-two-col` (`minmax(0,1fr) 20rem`, `base.html:329`) while the island
  renders `.me-layout` (`minmax(0,1fr) minmax(14rem,18rem)`,
  `base.html:334`) inside its first column; `.lib-sidebar` carries
  `min-width: 0`.
- **Two reported "bugs" are rendering artifacts needing live diagnosis.**
  The review-queue "N of M resolved" strikethrough has no strike markup or
  CSS anywhere (suspected progress-track overlap,
  `legacy_review/queue.html:45-46`). The "unlabeled" teal Status-page button
  has labels ("Turn on" / "Set up", `routers/settings.py:878-896`) whose
  text renders invisible (suspected color-token defect).

Other load-bearing anchors: `--t-base: .8125rem` (13px) at `base.html:115`;
the settings hub includes ~12 partials, each with its own POST route and
`csrf_settings` token; the settings sub-page tab strip is duplicated markup
in four files, not a shared include; `/jobs/{id}` and `/runs/{id}` share
`legacy_runs/_run_detail_body.html`; `/login` deliberately 404s when
`voxint_multi_user` is off (`routers/auth_pages.py:62-66`); the explore
header pluralization is hardcoded (`explore/explore.html:6`); the media
library has no search input; there is no auto-claim seam (`claim_run()` in
`src/voxint/adjudication/slots.py`, same-reviewer reuse per ADR 0004, with a
60-second heartbeat reclaim in `MediaEditor.tsx`).

## 3. Standing gates (every slice)

- ruff + mypy + pytest green; frontend CI job (lint, typecheck, build,
  audit) green.
- Route changes regenerate the route-inventory and console2
  characterization/order goldens in the same commit, with the golden diff
  inspected, never blind-regenerated. Every new mutating route gets CSRF
  wiring (contract-enforced).
- Browser acceptance lane (`voxint-e2e-review` skill +
  `tools/e2e_browser_lifecycle.py`) for island-behavior changes.
- CHANGELOG entry under `[Unreleased]` per slice.
- No new frontend component test runner (repo policy: lib-layer vitest
  only).
- One PR per slice on `feat/<issue>-<desc>`; review tier per
  `docs/testing.md` (multi-model for S4/S9/S10/S12; single-model
  elsewhere; none minimal).

## 4. Slices, in execution order

**S1. Editor layout bug fix** (bug; browser lane). Un-nest the double grid
in `editor/detail.html` (move the Media/Runs info cards out of
`.lib-two-col`, or drop that wrapper on the editor page); add min-width
guards so the speaker rail can never collapse; verify at
900/1024/1152/1280/1440px with no overlap or horizontal overflow. Fix the
contradictory "No runs for this media file yet." copy to distinguish "no
successful run" from listed failed runs.

**S2. Deterministic copy and behavior fixes** (bug). Explore header
pluralization; `/login` redirects to `/` when multi-user is off (goldens
regenerated and diff-inspected); Backups copy made universally truthful
(keep pg_dump guidance, add that native-launcher installs include a backup
command; no install-type detection exists, so no conditional copy); DB
largest-tables estimates get a staleness caveat or zero-row suppression.

**S3. Rendering diagnostics** (bug; browser lane). Live DOM and
computed-style investigation, then fix: (a) the queue progress label
overlapping its progress track; (b) the invisible "Turn on"/"Set up" label
on the Status page primary action. Verify in light, forced dark, and system
dark.

**S4. Type foundation** (multi-model review; screenshot pass). Raise
`--t-base` from 13px to 15px; introduce a dense token (~13px) applied to
grid-tables so lists keep density; fix immediate wrapping/overflow fallout
only; re-verify the S1 editor breakpoints at the new scale. Broad label
de-capsing is deliberately NOT here; it distributes into later surface
slices so each stays reviewable.

**S5. Single-operator claim friction** (concurrency design first). When
multi-user is off: editor and workbench auto-claim on mount; the queue's
claimed-by column hides. Constraint: same-reviewer reuse plus the 60-second
heartbeat means two auto-claiming tabs would invalidate each other's tokens
in a loop, so the slice specifies arbitration before code (auto-claim on
mount only; on claim-lost show a "Resume editing here" action instead of
auto-reclaiming). Acceptance: two tabs, alternating heartbeats,
unload/release race, expired token, editor-to-workbench switch. Multi-user
behavior unchanged.

**S6. Editor content reorder** (browser lane). Outline collapsed by default
with a count and moved below the transcript; one speaker label per
transcript row; the six download links become one button with a
keyboard-accessible format menu; the duplicated verified-progress copy
deduplicated; the editor's Runs card gains run status plus Retry/Re-run
actions (reusing the existing retry endpoint); sentence-case labels for
this surface.

**S7. Error normalization helper.** Backend mapping of known raw errors to
a plain sentence with the raw string in a details fold (a structured
error-code system was considered and rejected as bloat for a
single-operator product). Group consecutive identical failures in the home
Recent feed with counts and a one-line cause. Applied to home and run
detail here; reused by S12.

**S8. Run detail restructure.** One shared partial makes this one edit:
summary card first (status, current stage, plain-language error, primary
recovery action); Manage card hoisted; pipeline models, stage ledger, and
archive JSON folded into a collapsed "Technical details"; speaker timeline
as name plus readable duration; "Restart from scratch" reworded.

**S9. Settings IA: tabs** (multi-model review). Extract the duplicated
sub-page tab strip into one shared include; redistribute the hub partials
into tab pages (approximately Everyday / Media and ingest / AI and
advanced, plus the existing Status/Hardware/Database/Plugins), reusing the
existing per-section POST routes unchanged; benchmark CLI text, compose
file names, and endpoint internals move under Advanced.

**S10. Settings control model.** Separate from S9 so either can regress
alone: the tri-state radios become a switch presenting effective state, a
"customized" badge, and a "reset to default" affordance, keeping the three
persisted states (on/off/inherit) on the wire honestly; covers the
`_features.html` loop AND the standalone copies in `_semantic`,
`_translation`, `_folders`, `_sources`; one visible Save per tab; feature
dependencies enforced by grouping/disabling controls rather than prose.
Tests: all three stored values against both environment defaults,
dependency-validation failure, and value preservation after a failed save.

**S11. Media library: search and actions.** Server-side search
(case-insensitive substring over display name and folder name, composing
with a new status filter and the existing view/sort params; defined
empty-result state); state-dependent row action (Review/Retry/Open); file
rows renamed "Open" (they open the editor; folders keep "Open"); "File
missing" becomes plain language with Locate/Remove actions.

**S12. Runs canonical surface** (multi-model review; may split into UI and
bulk-retry PRs at execution). `/runs` becomes the one lifecycle surface:
tabs (Needs attention / Active / Failed / All), title-first rows with the
run id demoted to a copyable secondary, the filter bar collapsed behind a
Filter control with power options under More filters, local time, a
one-line pipeline health summary replacing the stage tiles, the
background-jobs table behind a disclosure, and grouped identical failures
with a bulk Retry (a new mutating workflow: eligibility, CSRF, idempotency,
and partial-failure reporting specified in the issue). Review stays a
separate destination.

**S13. Jobs compatibility migration.** After S12 proves out: `/jobs`
becomes a redirect/alias with an explicit query-vocabulary mapping,
`/jobs/{id}` retained as an alias; nav updated; goldens diff-inspected.
Route goldens do not protect redirect semantics, so integration tests
cover destination and query preservation, and the browser pass checks old
bookmarks.

**S14. Queue, speakers, explore polish.** Review-queue rows differentiated
by folder plus added date with raw paths removed; minimal workbench
name-hint humanization (confidence bands, no raw decimals; deeper
workbench investment deliberately rides the #158 convergence decision);
speakers page: unnamed voices expanded by default and listed first, one
profile form with one Save, insights polling bounded (a few retries with
backoff, then a manual check control); explore empty state gains example
searches and a capability hint.

**S15. Icon rail labels and theme access.** Labeled rail at desktop width
(wider rail or labels under icons; the mobile hamburger unchanged); the
theme control moved to one consistent location. A universal page-header
template refactor was considered and dropped as bloat; header consistency
lands opportunistically in slices that already touch a page.

**S16. Low-data thresholds.** Project detail: entity bars render as ranked
count lists below five distinct entities (boundary-tested), the coverage
matrix becomes a speaker list below 3x3, temporal trends hide below two
distinct dates; truncated labels get title attributes. Simpler
representation, never hiding data that exists.

**S17. Terminology sweep (final).** The glossary (Run not job; Recording;
Separate voices; Needs review; Open) is defined up front in the epic body
and applied opportunistically in every slice; this final slice audits and
sweeps the remainder.

**Issues-only actions.** The editor/workbench convergence decision is
recorded on #158; #149 gets a note that this epic is independent
remediation, not Console 2.0 scope.

## 5. Traceability: audit findings to slices

| Audit finding | Slice |
|---|---|
| Rail collapse | S1 |
| Copy/render bug batch | S2 + S3 |
| Speaker naming | S1 + S5 (existing UI unhidden) + S6 (discoverability) |
| Editor content order | S6 |
| Run controls on the recording | S6 (editor Runs card) + S11 (library actions) |
| Settings tabs and radios | S9 + S10 |
| Type ramp | S4 (+ distributed labels) |
| Error presentation | S7 + S12 |
| Jobs/Runs merge | S12 + S13 |
| Workbench friction | S5 + S14 |
| Speakers defaults | S14 |
| Run detail folding | S8 |
| Rail labels and chrome | S15 |
| Low-data widgets + explore empty state | S16 + S14 |
| Terminology | S17 (glossary up front) |
| Editor/workbench convergence | #158 comment (decision, not code) |
| Stage tiles | S12 (replaced by a health line) |
| Library search | S11 |

Deliberate deferments, recorded in the epic: the bulk-action-bar
discoverability hint (panel rated it minor); a media detail page (the panel
unanimously recommended against adding one; `/media/{run_id}` serves media
bytes and stays that way); the universal page-header template.

## Review notes

A codex second opinion reviewed the 14-slice draft; the final 17-slice plan
folds in its critique. Accepted: the two-tab auto-claim token race (S5 now
requires explicit arbitration design and concurrency acceptance tests);
tri-state switch honesty (S10 separated from S9, wire model kept, standalone
copies included); /jobs redirect compatibility (S13 separated, query mapping
plus integration tests, alias retained); the bug batch split into
deterministic fixes versus live rendering diagnostics with a real review
tier; the base-token foundation split from the broad label restyle and moved
early, with labels distributed to surface slices; no polishing /jobs before
S12 replaces it (stage-tile and background-jobs decisions moved into S12);
the page-header refactor cut as bloat; bounded insights polling; the
traceability table; explicit #149/#158 relationships; inspected golden
diffs; defined search semantics; defined low-data thresholds with boundary
tests. Rejected: replacing raw-error details folds with a structured
error-code system (heavier than a single-operator product needs). Deferred:
the exact settings tab taxonomy wording (decided in S9's issue) and the S12
split decision (made at execution if the PR grows).
