# Plan: the "first 30 minutes" flow slice

> **Status:** implementation plan (no production code yet). Converts the accepted
> subset of [ux-ui-gap-analysis-2026-08-21.md](../reports/ux-ui-gap-analysis-2026-08-21.md)
> into a phased, reviewable build. Tracked as #117. Scope: three console moves
> for the non-technical first-run operator. Native packaging (#73) is a noted
> parallel track, not in this plan. Reviewed by a codex second opinion (zen
> clink, planner role); see **Review notes** at the end.

## 1. Goal

Voxint's review console is correct and now visually warm, but it explains itself
like a maintainer and offers freedom like a power tool. This slice gives a
first-run operator a task-first spine: where to start, what is next, and (as far
as the system honestly tracks it) when they are done. It bundles three moves
from the gap-analysis report:

1. **#1 Critical-path vocabulary pass** (`copy`): rename happy-path display copy
   only. Docs, API, CLI, enum values, and module names stay precise.
2. **#2 Explicit two-step review sequence** (`flow`): keep both review routes
   (workbench for people, stepper for words), do not merge them, and wrap them in
   one "1. Who is speaking, then 2. Check the words" framing with one dominant
   Continue per step. Retire "either order."
3. **#3 Task-first dashboard and entry** (`IA`): open the dashboard on Add audio,
   Continue review, and the last finished run; demote throughput and stage
   metrics behind disclosure. Add an Add-media affordance to the Runs header.

The audience mandate governs every choice: prefer hide or defer behind
progressive disclosure over remove, and never strip a capability the single
power operator relies on. No numerics are touched, so there is no parity impact.

## 2. Assumptions and constraints

- **The console is server-rendered (FastAPI + Jinja) with React islands.** All
  routes live in `src/voxint/api/app.py`; templates in
  `src/voxint/api/templates/`; islands in `frontend/src/`.
- **The claim already lands on the workbench first.** `POST /review/{id}/claim`
  redirects to `/review/{id}` (the workbench), so "people first" is already the
  default entry. The sequence work is framing, not routing.
- **Display-copy renames are cheap on tests.** No contract test pins the
  user-facing strings this pass touches. The one literal pin is
  `tests/unit/test_presentation.py:172-173` (stage labels). Integration tests
  assert `aria-label="Review queue"` (structural) and functional "model services"
  copy (not the wizard step labels).
- **"Claimed by you." on the workbench is contract text** integration tests
  assert verbatim (`run.html`). Any header work must preserve it exactly.
- **What would invalidate this plan:** a decision to make transcript verification
  part of run-level completion (that is a schema and lifecycle change, deliberately
  out of scope here, tracked as report #7); or a decision to promote native
  packaging (#73) ahead of console work.

### The two blocking semantics (resolved in Phase 0, verified against the code)

A codex review surfaced two blockers. Both were confirmed by reading the source,
and both are resolved before any UI code by Phase 0 below.

- **Backlog counts the wrong status.** The dashboard computes
  `review_backlog = stats.status_counts[AWAITING_ADJUDICATION]` (`app.py:5493`),
  but the real review queue is `adjudication_queue()`
  (`resolver.py:574`): `status == COMPLETED` and `archived_at IS NULL` and at
  least one unresolved label. A successful pipeline terminates a run as
  `COMPLETED`, not `AWAITING_ADJUDICATION`, so the existing "Review backlog" stat
  card is already semantically wrong. Promoting that number to a primary
  first-run affordance would amplify the bug.
- **The two-step sequence implies a completion the system does not track.** Queue
  membership ends when speaker labels are resolved. Transcript verification
  (Step 2) does not affect `adjudication_queue()`, `PipelineRun.status`, or any
  durable run-level flag. After Step 1, a run can drop off Review with Step 2
  untouched. The sequence copy must therefore be honest navigation, not a
  completion claim.

## 3. Proposed approach

Ship the three moves as **one atomic slice**, implemented in order A then B then
C (friendly naming from A feeds B and C; the dashboard should advertise Continue
review only once the sequence exists). Commits stay independently reviewable, but
the full first-run journey is validated together before release. Precede all
three with a **Phase 0 decision gate** that fixes the two semantics above and
specifies the one new query, so the copy and task cards can be validated
honestly.

**Alternatives considered and rejected:**

- *Ship #3 (dashboard) first, since it is the entry point.* Rejected: it would
  route first-run users prominently into the old vocabulary and the unsequenced
  flow, and it would advertise "Continue review" before the sequence exists.
- *Unify the review backlog as unresolved-speakers-OR-unverified-segments, so the
  dashboard can report complete review across both steps.* Rejected for this
  slice: it expands query, legacy-data, and tutorial scope and edges into a real
  definition-of-done (report #7), which the report deliberately excludes here. We
  keep label-resolution as the queue predicate and frame Step 2 as recommended.
- *Merge the two review screens.* Rejected by both design panels in the report:
  it compounds the workbench overload it aims to cure.

## 4. Phase 0: decision gate (no UI code)

Resolve and land these before any template work. Each is a small, testable unit.

1. **One canonical review-eligibility count.** Add
   `review_backlog_count(session) -> int` beside `adjudication_queue()` in
   `src/voxint/adjudication/resolver.py`, using the identical predicate
   (`COMPLETED`, not archived, `unresolved_labels > 0`). Point the dashboard route
   (`app.py:5493`) at it. This fixes the existing bug and gives Phase C a truthful
   number. Unit test asserts the count equals `len(adjudication_queue(session))`
   across fixtures (zero, some-resolved, all-resolved, archived).
2. **Completion contract for Step 2.** Decide: transcript verification is
   **recommended, not a queue-completion gate**. Queue membership stays
   label-resolution-only. The sequence header and Continue are orientation and
   navigation; they never assert a durable "run complete" state. Record this in
   the plan and in the how-to so copy stays inside it.
3. **`latest_completed_run` specification.** `PipelineRun` has no `finished_at`
   (only `created_at` / `updated_at`); `StageRun` has `finished_at`
   (`models.py:396`). Define the query as: most recent `COMPLETED`, non-archived
   run, ordered by the run's terminal-stage (`finalize`) `StageRun.finished_at`,
   tie-broken by `(created_at, id)`; fall back to `updated_at` for seeded or
   legacy runs that lack stage rows (and cover that case in a test). Destination
   is the run-detail page (which already carries the export menu); a first-class
   "export as an end state" surface is report #9, out of scope. Return `None`
   cleanly for the empty case.
4. **Tutorial scope (decided: align).** The guided tutorial
   (`src/voxint/tutorial/steps.py`) runs `RUN -> REVIEW -> ADJUDICATE -> EXPORT`,
   and both `ADJUDICATE` and `EXPORT` bind to the **workbench** page; it never
   visits the transcript stepper, and its EXPORT copy calls "submit, review,
   attribute, export" the "whole loop." The maintainer chose to **align the
   tutorial to the new sequence**: add a transcript-page traversal and make the
   "whole loop" wording include checking the words, scheduled inside Phase B. This
   is more than copy, but leaving the flagship first-run surface teaching users to
   skip Step 2 would contradict the whole slice.

**Gate 0 fails (stop) if** the eligibility count, the Step 2 contract, the
`latest_completed_run` spec, or the tutorial decision is not settled: the copy and
task cards cannot be validated honestly without them.

## 5. Phase A: critical-path vocabulary pass (#1)

Display-only string changes plus the one unit-test update and doc alignment.
Reuse the existing title-precedence helper `_run_source_title(run)`
(`app.py:1501`: sidecar title, then acquisition metadata, then cleaned filename)
everywhere a friendly run name is shown, so the queue, workbench, transcript, and
dashboard never disagree. Register it as a Jinja global (or pass it in context)
rather than calling `friendly_media_label(source_path)` directly.

Affected surfaces:

- `templates/base.html` (nav): "Review queue" -> "Review". Route stays `/review`;
  `active_nav` unchanged.
- `templates/queue.html`: `<title>` and H1 "Adjudication queue" -> "Review";
  subtitle "voices still needing a human ruling" -> plain language ("Recordings
  that finished and still need you to confirm who is speaking"). Keep the empty
  state honest.
- `templates/fragments/labels.html:145-154` (the cosine line): replace the raw
  `Cosine suggestion: {name} ({conf}, grounded)` with a plain match label, and
  move the raw float and grounded flag into a `<details><summary>Why this
  match?</summary></details>` reveal.
  - The plain label is **conditional on `s.cosine_grounded`**: grounded ->
    "Strong voice match: {name}"; ungrounded -> a neutral phrasing that never says
    "strong" (for example "Possible voice match: {name}"). The template can hold a
    `cosine_speaker_name` with a false grounded flag, so unconditional "strong"
    would overclaim evidence.
  - Render the evidence summary and the `<details>` as **sibling block elements**,
    not inside the existing `<p class="muted">`: `<details>` is flow content and
    would produce invalid paragraph nesting.
  - Render the reveal **only when cosine evidence exists**, to avoid a "Why this
    match?" control on every card.
  - Keep the honest taxonomy verbatim: the grounded machine-match, the "Heard name
    (unverified)" line, and the no-name case are unchanged. Only the raw float is
    hidden.
- `templates/run.html` (workbench H1): raw `source_path` -> `_run_source_title(run)`
  plus the recording date; keep the short run hash secondary and copyable. Keep
  "Claimed by you." verbatim.
- `templates/review_transcript.html`: H1 "Review transcript {hex}" ->
  `_run_source_title(run)` plus "Check the words". Fix the read-only link copy on
  line 20: the destination `/runs/{id}/transcript` defaults to the **corrected**
  variant and is variant-selectable, so "See the original, unedited transcript"
  would be false. **Decided:** link stays on the default view with copy "Read the
  full transcript" plus the honest note that versions are selectable; do not force
  `?text=raw` and do not claim "unedited".
- `templates/dashboard.html:5-8` (subtitle): drop "the same figures as GET
  /metrics and voxint stats" and fix the emdash on that line; plain "A quick look
  at your recordings."
- `templates/setup.html` (`labels` dict + copy): "LLM enhancement" -> **"Text
  clean-up and name hints (optional)"** (decided: it also surfaces speaker-name
  hints, so this stays honest without the "LLM" jargon); "Model services" ->
  "Readiness" (the H1 is already "Readiness checks"). Fix the emdashes on
  `setup.html:29` (Welcome) and `setup.html:138` (services).
- **Deferred out of Phase A:** the `_STAGE_LABELS` rename
  (`presentation.py:175-178`, "Diarize & embed" / "Enhance & match" -> outcomes).
  Those labels appear only in the stage-timing and stage-failure tables, which
  Phase C demotes behind disclosure, so renaming them adds churn (and the one
  test pin) without improving the first-30-minutes path. Revisit only if user
  evidence shows they matter.

Docs alignment (same phase, per the stale-docs-are-bugs rule):
`docs/how-to/reviewing-and-adjudicating.md` and `docs/onboarding.md` copy that
quotes the renamed strings.

Tests:

- Update `tests/unit/test_presentation.py` only if the stage labels change (they
  do not in this phase).
- Structural (not substring) integration asserts: parse the queue HTML and assert
  the H1 is "Review" and "Adjudication queue" is absent; parse a label card and
  assert the cosine float text is a descendant of a **closed** `<details>` whose
  `<summary>` has a useful accessible name, and is **not** in the visible evidence
  line; assert grounded renders "Strong" and ungrounded never does (two separate
  fixtures); assert the grounded/heard/no-name taxonomy still renders.

## 6. Phase B: explicit two-step review sequence (#2)

Add a shared, **static** run-level identity-and-sequence fragment, and let each
step keep its own live progress. Do not stack a second sticky card on top of the
existing `.review-head` surface; extend or refactor `.review-head` on the
workbench and add the matching identity block to the transcript page. **Decided:**
ship the header as a normal (non-pinned) block first; add sticky positioning only
if a browser check shows orientation suffers without it, to keep layout risk and
viewport crowding down.

- **New fragment** `templates/fragments/review_journey.html`: friendly run name
  (`_run_source_title`), the step name ("Step 1 of 2 - Who is speaking" /
  "Step 2 of 2 - Check the words"), and the one dominant Continue. It carries
  **only static identity and sequence markup**; it does not own either progress
  counter.
- **Step 1 progress stays inside the HTMX-swapped `#labels` region** (or is
  updated out-of-band on the decision responses); **Step 2 progress stays inside
  the `ReviewStepper` island and its server fallback**. A header rendered outside
  those owners would go stale immediately after an action, so no second
  server-owned transcript counter is rendered.
- **Workbench Continue** = the existing `run.html:18` link reframed "Continue to
  checking the words". Keep "Release claim". **Drop the vaguely specified "skip for
  now"** unless a real destination and persisted deferral state are added: the
  queue already lets the operator leave, and a redundant link would imply saved
  state that does not exist.
- **Transcript back** = "Back to the people" (reframe "workbench"); do not promote
  a sideways jump. **Define the Step 2 terminal action:** at all-lines-verified,
  show a plain "You have checked every line" state with a clear next action (export
  via the existing menu, or back to Review). It never sets a durable whole-run
  completion flag (Phase 0 contract).
- **Retire "either order":** rewrite `reviewing-and-adjudicating.md:11` to sequence
  the two halves ("Review has two steps: start with the people, then check the
  words"), keeping both workflows documented. Leave line 184 (keyboard-vs-buttons
  "whichever you prefer") untouched; it is unrelated.
- **Tutorial alignment** (from the Phase 0 decision): add a transcript-page
  binding to `src/voxint/tutorial/steps.py` (`TutorialPage`, `STEP_PAGE`, step
  copy) so the guided walkthrough traverses attribute then check-the-words then
  export, and update the "whole loop" wording. Update the tutorial banner fragment
  and the tutorial integration tests.

Tests: both routes work directly, claimed, unclaimed, stale-token, JS-off, and
hydrated; "Claimed by you." still verbatim; the `review-stepper` island still
mounts and its fallback still renders; Step 1 progress updates after an HTMX label
decision and Step 2 after a React verify/edit/split/relabel; the tutorial
traverses (or truthfully describes) both steps; the how-to no longer says "either
order."

## 7. Phase C: task-first dashboard and entry (#3)

- **Three task cards render first on the dashboard**, above the metrics: **Add
  audio** (links to the Runs Add-media section), **Continue review (N)** (N from
  the Phase 0 `review_backlog_count`, links to `/review`), and **Last finished
  run** (from `latest_completed_run`, links to run detail; honest empty state when
  there is none).
- **Freshness:** render the count on load and refresh it on page reload; do not
  style it as live. The metrics fragment keeps its 15s HTMX poll, but the Continue
  count must not look auto-updating while sitting outside that poll (a live-looking
  but stale number is worse than a plainly static one).
- **Demote metrics behind disclosure**, with two carve-outs codex flagged: keep the
  invalid-`?since=` notice and the time-window `<select>` **outside** the
  `<details>` (errors and inputs must never be hidden); place the window selector
  with the disclosed tables; task cards stay first. The 4 stat cards plus the
  runs-by-status, stage-timing, and stage-failure tables move inside "Show run
  details".
- **Runs header Add-media affordance:** give the existing upload form
  (`runs.html:15-22`) a clear "Add media" heading and anchor, and keep the URL
  fetch form (when `ytdlp_enabled`) alongside it: elevating only the upload would
  demote the URL and video workflow. No new top-level nav item.

Tests: unit for `latest_completed_run` (ordering, deterministic ties, archived
excluded, missing terminal-stage timestamp, empty); integration for the three task
cards (including the empty last-run state), the metrics-behind-`<details>`
structure with the since-notice and window control kept outside it, and the Runs
Add-media section exposing both upload and enabled URL fetch; the dashboard count
equals `adjudication_queue` eligibility.

## 8. Testing strategy (across the slice)

- **Gates every phase:** `ruff`, `mypy`, `pytest` (unit, contracts, integration),
  frontend typecheck and unit, `gitleaks`. No parity impact.
- **Structural DOM assertions, not substring locks:** assert hierarchy, link
  targets (exact hrefs, token propagation, stale-token behavior), disclosure
  structure, and accessible names.
- **Progress ownership:** one live owner per counter, asserted after a real action
  on each side.
- **Completion boundaries:** 0/N, N/N, zero segments, zero labels,
  labels-resolved-but-transcript-incomplete and the reverse.
- **Browser lane** (the `voxint-e2e-review` skill, seed-only lifecycle, no real
  pipeline per the reporting-host hazard): `<details>` keyboard behavior, React
  hydration preservation, HTMX swaps, narrow-viewport obstruction, focus order,
  and one end-to-end first-run journey (dashboard -> Add audio -> Step 1 -> Step 2
  -> export). Serial on maintainer hardware only.
- **Screenshots and CHANGELOG:** regenerate the affected `docs/images/*.png`
  **once** after the full integrated flow passes (B and C change the same
  first-run surfaces, so per-phase regeneration is wasted churn); one coherent
  `[Unreleased]` CHANGELOG entry, with commits kept independently reviewable.

## 9. Rollout, risks, open questions

- **Rollout:** one atomic slice, A then B then C, after Phase 0. No version bump;
  a later release session bundles the six-compose-file bump. Native packaging
  (#73) is untouched.
- **Risks:** the shared header competing with the tutorial banner and playback
  controls on a narrow viewport (mitigated by keeping it static and refactoring
  review-head rather than stacking); hydration replacing the transcript fallback
  (mitigated by keeping Step 2 progress inside the island); the sidecar-vs-metadata
  title precedence drifting between pages (mitigated by the single
  `_run_source_title` helper).
- **Decisions settled with the maintainer (2026-08-21):**
  1. **Tutorial: align.** The guided tutorial is aligned to traverse both steps
     (Phase B), not narrowed.
  2. **Read-only transcript link:** copy is "Read the full transcript" on the
     default view (versions selectable); the link is not changed to `?text=raw` and
     never claims "unedited".
  3. **LLM setup step label:** "Text clean-up and name hints (optional)".
  4. **Header:** static first; sticky only if a browser check shows it is needed.
- **No open questions remain.** Implementation can begin at Phase 0.

## Review notes

A codex second opinion (zen clink, planner role) inspected the routes, models,
resolver, `ReviewStepper.tsx`, the tutorial, and the tests. Every substantive
finding was verified against the source before folding in.

- **Blocker: dashboard backlog counts the wrong status.** Verified
  (`app.py:5493` vs `resolver.py:574`). Accepted: Phase 0 adds one canonical
  `review_backlog_count` mirroring the queue predicate, used by the card and the
  existing stat card.
- **Blocker: two-step framing implies untracked completion.** Verified
  (transcript verification gates nothing durable). Accepted: Phase 0 fixes Step 2
  as recommended navigation, not a completion gate; the header never claims
  whole-run "done."
- **High: the tutorial skips Step 2.** Verified (`tutorial/steps.py`; ADJUDICATE
  and EXPORT both bind the workbench). Accepted, and the maintainer confirmed the
  align option: Phase B aligns the tutorial and its tests.
- **High: a shared server header cannot own live progress.** Verified (HTMX-owned
  Step 1, React-owned Step 2). Accepted: the fragment carries static identity and
  sequence only.
- **High: the read-only link copy would become false.** Verified
  (`/runs/{id}/transcript` defaults to corrected, variant-selectable). Accepted:
  the maintainer chose "Read the full transcript" on the default view.
- **High: `latest_completed_run` assumed a `finished_at` that does not exist.**
  Verified (`PipelineRun` has none; `StageRun` does). Accepted: Phase 0 specifies
  terminal-stage `StageRun.finished_at` ordering with a legacy fallback.
- **High/medium: "Strong voice match" must be conditional on grounded; do not nest
  `<details>` in `<p>`; "skip for now" is undefined; Step 2 needs a terminal
  action; Add-media must retain URL fetch; task-card freshness must not look
  live; keep the since-notice and window control outside disclosure.** All
  accepted and folded into Phases A, B, C.
- **Defer the stage-label rename.** Accepted: moved out of Phase A (churn behind
  demoted metrics).
- **Screenshots and CHANGELOG once, not per phase.** Accepted for an atomic slice.
- **Rejected nothing outright.** The one point held at arm's length is the
  "unified backlog" alternative (unresolved-speakers-OR-unverified-segments): it
  is the cleaner long-term model but expands into a real definition-of-done, which
  the report scopes out of this slice, so it is recorded as an alternative, not
  adopted.
