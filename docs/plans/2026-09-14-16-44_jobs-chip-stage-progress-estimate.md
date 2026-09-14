# Plan: per-run stage progress estimate in the Jobs STATUS chip (#475)

Status: done

## Goal

On the Jobs page (`/runs`, which `/jobs` redirects to), a running row's STATUS
chip says only "Running". Issue #475 (moved from the #244 polish deltas, mockup
`2f-jobs.png`) wants the chip to say which stage the run is in and roughly how
far along that stage is, for example "Transcribing ~62%". The pipeline exposes
no real per-stage progress: the worker's transcribe call is one blocking HTTP
POST and the whisper service consumes the segment generator fully before
replying; diarization and embedding expose nothing. This change derives an
estimate from the same per-stage typical durations the progress strip already
uses for its "~Xm left" cells. Real transcribe progress from the whisper service
(the issue's option b) is deferred.

## Assumptions and constraints

- `pipeline_dashboard_query.pipeline_dashboard_state` is already computed on
  every `/runs` page load outside the archived view and carries
  `StageProgress.avg_seconds` per stage: a 90-day average of successful
  attempts with at least 3 samples, else a per-stage heuristic scaled by
  compute tier. The chip reuses that map and adds no second average.
- The grid renders once per page load; only the strip polls. The chip is a
  page-load snapshot, like the TOOK column for running rows ("as of page
  load"). No client-side ticking and no new polling endpoint: a ticking chip
  can contradict the polled strip on the same page after a stage transition,
  and a per-row poll is ceremony this audience does not need.
- The "active attempt" predicate matches the strip and `stage_activity`:
  `StageRun.status = running AND StageRun.stage = PipelineRun.current_stage`.
  Per run, the attempt with the greatest `started_at` is used. This is a
  defensive selection among anomalous concurrent RUNNING rows, not a liveness
  proof: lease validity is deliberately ignored to match the existing
  activity semantics (the strip's `_active_started_at` ignores it too).
- Only rows with `status = running` get the estimate. Queued, paused,
  awaiting_adjudication, completed, failed, and cancelled rows keep their
  labels. Archived-view rows (`?archived=1`) keep "Running" because the
  dashboard read model is not built for that view.
- No living spec is declared (`CLAUDE.md` has no `Living specs:` line).
- No migration, no service change, no numerics.
- Would invalidate the plan: a decision to want real progress now (then the
  service contract work comes first), or a decision that the grid should poll
  (then the freshness design changes).

## Proposed approach

1. **Read model.** `RunListItem` gains `current_stage: str | None = None` and
   `stage_started_at: datetime | None = None` (defaults keep the keyword
   factory in `tests/unit/test_runs_search.py` valid). `list_runs` selects
   `PipelineRun.current_stage` and a correlated scalar subquery
   `max(StageRun.started_at)` over running attempts at the run's current
   stage, explicitly correlated to `PipelineRun`, alongside the existing
   `processing_seconds` subquery. It changes no row cardinality, ordering, or
   cursor construction, so keyset pagination is untouched. It is a second
   indexed `stage_runs` scan per selected row, bounded by page size.
2. **Estimate.** A pure function in `pipeline_dashboard_query.py` next to
   `compute_stage_eta`:
   `estimate_run_stage_progress(stage, started_at, avg_seconds, now) -> RunStageProgress | None`
   with `RunStageProgress(stage: str, percent: int | None, overrun: bool)`.
   Rules: `None` when stage, started_at, or avg is missing or avg <= 0;
   elapsed clamped at 0 (clock skew); `elapsed >= avg` gives `overrun=True,
   percent=None`; otherwise `percent = floor(elapsed / avg * 100)`, which is
   at most 99 by construction. No cross-module helper: the dashboard module
   stays ignorant of `RunListItem`.
3. **Route.** `/runs` builds `stage_progress: dict[uuid.UUID, RunStageProgress]`
   with a small comprehension over running items and
   `{s.stage: s.avg_seconds for s in dashboard.stages}` when `dashboard` is
   not None (empty dict otherwise), and passes it to the template.
4. **Template.** The `run_row` macro takes `progress=none` as a second
   argument; the standard call site passes `stage_progress.get(it.run_id)`,
   the grouped failed path passes nothing. For a running row with an estimate
   the chip keeps `class="pill running"` and its text becomes
   `<Stage-ing> ~NN%` or `<Stage-ing> · taking longer`. The title names the
   source honestly: with a learned average, "Estimated from recent completed
   attempts of this stage; as of page load"; with the heuristic (fewer than 3
   samples), "Estimated from a default stage duration, there is not enough
   history yet; as of page load". `RunStageProgress.using_heuristic` carries
   the flag from `StageProgress`. Without an estimate the chip stays "Running".
5. **Wording.** `presentation.humanize_stage_progress(value)` returns a
   present-participle label per stage (Acquiring, Preparing, Transcribing,
   Diarizing & embedding, Enhancing & matching, Finalizing), falling back to
   `humanize_stage` for an unknown value so a new stage never breaks the page.
   Registered as a template global in `routers/deps.py` next to
   `humanize_stage`. Display only; the raw stage value never enters a class
   or machine output.
6. **CHANGELOG** entry under `[Unreleased]` / `### Added`.
7. **Issue follow-up.** One sentence on #475 when the PR opens: estimate
   shipped, real whisper progress deferred.

Alternatives considered and rejected:
- Real whisper progress now: service API contract, released images, mid-stage
  writes outside the claim transaction, only one of six stages covered; a
  full-panel change disproportionate to a single-operator product.
- Client-side ticking from `data-started-at` and `data-avg-seconds`: cheap,
  but the chip would climb to "taking longer" while the polled strip already
  shows the run at the next stage. Contradiction on one page is worse than a
  static snapshot.
- Out-of-band chip refresh piggybacked on the strip poll: a viable follow-up
  if staleness bothers the operator in use; not needed to satisfy the issue.
- Pinning at 99% on overrun (the issue's literal suggestion): a run at 150% of
  the average reading "~99%" for ten minutes claims a near-completion it
  cannot back. "taking longer" matches the strip footer's existing overrun
  language.
- Attaching the estimate to `RunListItem`: `list_runs` has no dashboard or
  settings context, so the item would carry a half-computed field. Raw
  `current_stage` and `stage_started_at` on the item, estimate in the route.

## Acceptance criteria

Exact-boundary values (0, 99, exactly the average) are pinned in pure-function
tests with an explicit `now`. Page-render scenarios use interior values where
several seconds of test latency cannot move the floored percentage.

### Requirement: Running rows show an estimated stage progress chip

The system SHALL render, for each Jobs-page row whose run status is `running`
and for which an active attempt at the run's current stage and a typical
duration for that stage are both known, a STATUS chip reading the stage's
progress label followed by an estimated percentage, and SHALL mark the chip as
an estimate in its title text, naming whether the estimate comes from recent
completed attempts or from a default duration.

#### Scenario: within the typical duration (page)
- GIVEN a running run whose current stage is `transcribe`, with one running
  attempt at that stage started 300 seconds before page load, and no learned
  history so the GPU heuristic (600 s) applies
- WHEN the operator loads `/runs`
- THEN the row's STATUS chip reads "Transcribing ~NN%" with NN in 50..55
  (a few seconds of test latency cannot move it further), keeps the
  `pill running` class, has seven cells in its row, and carries the
  default-duration title

#### Scenario: learned average beats the heuristic (page)
- GIVEN at least three completed `transcribe` attempts within 90 days of
  1000 s each, and a running run at `transcribe` started 250 s before page
  load (1000 s gives 10 s of latency headroom per percent; the original
  100 s / 25 s pairing had under 1 s)
- WHEN the operator loads `/runs`
- THEN the chip reads "Transcribing ~NN%" with NN in 25..27 and the page
  carries the recent-completed-attempts title

#### Scenario: percent never claims done (pure)
- GIVEN stage `transcribe`, avg 600 s, and `now` exactly 599.9 s after
  `started_at`
- WHEN `estimate_run_stage_progress` runs
- THEN it returns percent 99 and overrun False

#### Scenario: start of stage and clock skew (pure)
- GIVEN `now` equal to `started_at`, or 5 s before it
- WHEN `estimate_run_stage_progress` runs
- THEN it returns percent 0 and overrun False

### Requirement: overrun replaces the percentage

The system SHALL replace the percentage with "taking longer" when the active
attempt's elapsed time reaches or exceeds the stage's typical duration.

#### Scenario: exactly the average (pure)
- GIVEN avg 600 s and `now` exactly 600 s after `started_at`
- WHEN `estimate_run_stage_progress` runs
- THEN it returns overrun True and percent None

#### Scenario: elapsed exceeds the estimate (page)
- GIVEN a running run at `transcribe` started 900 s before page load under
  the 600 s heuristic
- WHEN the operator loads `/runs`
- THEN the chip reads "Transcribing · taking longer" and contains no "%"

### Requirement: rows without an honest estimate keep the plain status

The system SHALL render the plain status label when a running row has no
running attempt at its current stage, when the dashboard read model is not
built (archived view), or when the row's status is not `running`.

#### Scenario: no active attempt matches (page)
- GIVEN a running run whose only running attempt is at a stage other than
  `current_stage`, another running run with no stage rows at all, and (in a
  separate request, page size 2) a running run with a running attempt but a
  NULL `current_stage`
- WHEN the operator loads `/runs`
- THEN every such chip reads "Running"

#### Scenario: latest attempt wins (page)
- GIVEN a running run at `transcribe` with two anomalous concurrent RUNNING
  attempts at that stage, one started 1800 s before page load and one
  started 60 s before page load
- WHEN the operator loads `/runs`
- THEN the chip reads "Transcribing ~NN%" with NN in 10..15, because
  selection is by greatest `started_at`, not by attempt number or lease
  expiry

#### Scenario: non-running statuses are untouched (page, split across
requests because the test client pages at 2 rows)
- GIVEN a `queued` row and an `awaiting_adjudication` row, each with a
  running attempt at its `current_stage`
- WHEN the operator loads `/runs`
- THEN each STATUS chip reads its existing humanized status and contains no
  "%"
- GIVEN a `paused` row and a `completed` row
- WHEN the operator loads `/runs`
- THEN each STATUS chip reads its existing humanized status

#### Scenario: archived view (page)
- GIVEN an archived run in status `running` with a running attempt at its
  current stage
- WHEN the operator loads `/runs?archived=1`
- THEN the chip reads "Running"

### Requirement: read model exposes the raw inputs

`list_runs` SHALL return `current_stage` and, as `stage_started_at`, the
greatest `started_at` among running attempts at that stage, or None when no
attempt matches.

#### Scenario: pagination unchanged with multiple attempts
- GIVEN three runs newer-to-older, the middle one carrying two running
  attempts at its current stage
- WHEN `list_runs` walks pages of size 2 with the cursor
- THEN every run appears exactly once, in the same order as before, and the
  middle run's `stage_started_at` is the later attempt's start

### Requirement: grid shape unchanged

The system SHALL keep seven cells per row (the existing seven-cells test
still passes) and the `pill running` class on the chip.

Spec deltas: none (no living spec declared)

## Affected files

- `src/voxint/api/runs_query.py`: `RunListItem` fields `current_stage`,
  `stage_started_at`; correlated subquery in `list_runs`.
- `src/voxint/api/pipeline_dashboard_query.py`: `RunStageProgress`,
  `estimate_run_stage_progress`.
- `src/voxint/api/presentation.py`: `humanize_stage_progress` and its label
  map.
- `src/voxint/api/routers/deps.py`: register `humanize_stage_progress` as a
  template global.
- `src/voxint/api/routers/legacy_runs.py`: `/runs` builds and passes
  `stage_progress`.
- `src/voxint/api/templates/legacy_runs/runs.html`: `run_row(it, progress)`
  macro and both call sites; chip rendering.
- `tests/unit/test_pipeline_dashboard_query.py`: pure estimate tests.
- `tests/unit/test_presentation.py`: label tests plus the every-Stage-has-a-
  label invariant and the unknown-stage fallback.
- `tests/integration/test_runs_api.py`: page-render and read-model scenarios.
- `CHANGELOG.md`: `[Unreleased]` / Added.

## Implementation slices

1. **Estimate function and labels** (pure): `RunStageProgress`,
   `estimate_run_stage_progress`, `humanize_stage_progress`, unit tests for
   every boundary above. Tree works; nothing rendered yet.
2. **Read model**: `RunListItem` fields and subquery; integration tests for
   the read-model requirement including the pagination scenario.
3. **Route and template**: wire the map, render the chip, page scenarios
   (within, learned, overrun, no attempt, latest wins, non-running split in
   two, archived), seven-cells invariant still holds. CHANGELOG.
4. **Browser pass** (short): run the canonical lifecycle (`setup`, `seed
   --fixture rail`, `serve`), then add one running run with a running
   `transcribe` attempt started about 5 minutes earlier through a small ORM
   snippet against the e2e DSN (the seeder only creates COMPLETED runs and is
   not extended for this), load `/runs` in Playwright, assert the chip text,
   title, and class, and that the strip still renders. Teardown.

## Testing strategy

- Unit: estimate boundaries (0, floor, 99.9-case floors to 99, exactly-avg
  overrun, beyond, missing inputs, avg 0, NaN and inf avg, negative elapsed,
  heuristic flag pass-through); label map covers every `Stage` value and
  falls back for an unknown value.
- Integration: each page scenario via `make_run` (accepts `StageRun` kwargs)
  and `_set_current_stage`; the grid parser `_runs_grid_rows` reads the
  STATUS cell text; the title and class are asserted directly on the HTML;
  the learned-average scenario seeds three completed attempts to cross
  `_MIN_HISTORY_SAMPLES`. Scenarios are split so each request's first page
  (2 rows) holds the rows under test.
- Standard gates: ruff, mypy, pytest; CI `lint-test`, `secrets-scan`,
  `coverage`.
- Review: multi-model code review (design choices, new read-model seam);
  browser lane yes (observable Jobs-page behaviour, same classification the
  TOOK column received). Codex would cut the browser pass as duplicative of
  the TestClient HTML tests; kept, scoped to one short pass, because the
  repo policy binds once assigned and the precedent on this page ran it.

## Rollout, risks, open questions

- Fresh installs use heuristics, so the first runs may show percentages that
  are off by a lot; the tilde and title say "estimated", and the strip already
  exposes the same numbers as "~Xm left". Accepting.
- The correlated subquery adds one more per-row indexed scan to an already
  subquery-heavy list statement; bounded by page size. Accepting; no measured
  cost expected on a single-operator database. No lateral join or composite
  index without measured need.
- Resolved: the chip uses present participles ("Diarizing & embedding",
  "Enhancing & matching") while the strip keeps its imperative labels
  ("Diarize & embed"). The chip describes an activity in progress, the strip
  names a stage; both forms stay.
- Resolved: the chip says "· taking longer", not "· taking longer than
  usual". With fewer than 3 samples the comparison is against a default, not
  this installation's history, so "than usual" would overclaim.
- Known asymmetry: the chip anchors on the latest running attempt per run
  (max `started_at`), the strip's `_active_started_at` on the earliest per
  stage (min). They diverge only when a run carries two RUNNING rows at the
  same stage, which the engine never produces; documented in the query
  comment.

## Review notes

Codex (planner role, two rounds) critiqued the option set and this draft.
Resolution of each point:

1. Subquery semantics: accepted. Documented that `max(started_at)` is a
   defensive selection, not lease-authoritative, and added the pagination
   regression with multiple attempts.
2. Timing-sensitive exact percentages in page tests: accepted. Boundaries
   moved to pure tests with explicit `now`; page scenarios use interior
   values.
3. Cross-module helper coupling: accepted. Dropped
   `run_stage_progress_by_run`; the route builds the map; the estimate stays
   off `RunListItem`.
4. Page size 2 and the keyword factory: accepted. Non-running scenario split
   across requests; new fields default to None.
5. Cut the browser lane: rejected, with the reason recorded in the testing
   strategy. Scoped to one short pass with an ORM-seeded running run.
6. The 99 scenario and "latest attempt wins" wording: accepted. Moved to the
   pure contract; the retry scenario now states the selection rule.
7. Process bloat: accepted in part. The speculative real-progress
   architecture is dropped from the plan; the issue comment is one sentence.
   The presentation helper and CHANGELOG stay.

### Code review (three reviewers: Codex lead, Grok, Kimi)

No Critical or High findings from any reviewer. All three confirmed every
acceptance scenario is covered by a test and reported no drift beyond the
learned-scenario values above.

- Medium, Codex (1/3), accepted: the title said "typical time" even when the
  number was a fixed default. `using_heuristic` now flows into the chip and
  the title names the source; CHANGELOG reworded.
- Medium, Grok and Codex (2/3), accepted: exact page percentages under a
  600 s heuristic have only 6 s of latency headroom. Page tests now assert a
  small range; exact boundaries stay in the pure tests. Codex's alternative
  (a patchable clock seam in the route) was rejected as a production seam
  added for a test.
- Medium, Grok (1/3), accepted: the archived test's raw-HTML substring was
  replaced by parsed-cell and "no estimate title in body" assertions.
- Medium, Grok (1/3), skipped: chip versus strip overrun wording. Resolved
  above in favour of the short form for honesty.
- Medium, Kimi (1/3), accepted as a comment fix: the query comment claimed
  parity with the strip; it now states the max-versus-min asymmetry.
- Low, Grok and Kimi (2/3), accepted: `StageStatus.RUNNING.value` replaces
  the string literal.
- Low, Kimi, accepted: `math.isfinite` guard, tz-aware contract in the
  docstring, redundant conditional in the route removed (mypy rejects a None
  key, so the route skips NULL `current_stage` explicitly), NULL
  `current_stage` page test added.
- Low, Codex, accepted: the seven-cells test only rendered grouped failure
  rows, so the standard-row test now asserts seven cells too.
- Low, Grok, accepted: the read-model field comment says it is not a
  liveness flag.


## Completion notes

- Closed 2026-09-14. Verified against the final diff on
  `feat/475-stage-progress-chip` (commits `ceaf40c`, `77dc575`).
- Slices 1 to 3: every acceptance scenario has a named test in
  `tests/unit/test_pipeline_dashboard_query.py`,
  `tests/unit/test_presentation.py`, or
  `tests/integration/test_runs_api.py`; scoped run 238 passed, ruff and
  mypy clean.
- Slice 4: browser pass run twice on maintainer hardware (before and after
  the review fixes) with two ORM-seeded running runs; both chips, titles,
  class, seven cells, and the strip verified in Playwright. PASS.
- Spec files touched: none (no living spec declared).
- Drift: none. The learned-scenario values and the range assertions were
  folded into the plan during review.
- Follow-ups: real transcribe progress from the whisper service stays
  deferred (issue #475 comment); an out-of-band chip refresh on the strip
  poll is available if page-load staleness bothers the operator.
