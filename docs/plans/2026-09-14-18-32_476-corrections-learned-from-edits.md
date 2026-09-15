# Plan: corrections learned from operator edits (#476)

Status: done

## Goal

Issue #476 (from the Console 2.0 polish deltas, epic #149): let Voxint watch
operator transcript edits in a project, detect the same literal substitution
recurring across segments, and surface it as a suggestion that one click turns
into an ordinary rule. Suggestions never apply on their own.

Spec deltas: none

## Assumptions and constraints

- Provenance lives in a separate table, never inside the rule JSON (the
  `{id, match, replace, case_sensitive, whole_word}` shape is shared with pack
  manifests, console authoring, and the frozen `pipeline_runs.domain_pack`
  snapshot, with a strict allowed-key set).
- Suggest-and-accept, not auto-activation. The count threshold gates visibility
  of a suggestion, never application.
- On any re-edit, retract that segment's earlier evidence before re-deriving;
  never read-modify-write counts (the edit route's FOR UPDATE lock is per run,
  not per project).
- The corrector is raw-gated (`_compose_correction` only enforces rules that
  fired on `raw_text`), so a candidate must match `raw_text`, not only the
  enhanced text the operator saw.
- An accepted suggestion becomes an ordinary rule appended to
  `projects.corrections` through the normal validator. The freeze walk
  (`_resolve_run_config`) and snapshot are untouched. Learning therefore
  requires the project to own its corrections list (`corrections IS NOT NULL`).
- Case-only pairs ("it" -> "IT") are learned; learned rules are always
  `case_sensitive=True, whole_word=True`.
- No `active` state, no `promote` route, no slug ids, no `first_seen_at`.
  Dismiss deletes the row; the pair can only come back after three new
  segments carry the same edit.
- Would invalidate the plan: a decision to auto-apply suggestions, or a
  decision to carry learning state inside the rule JSON.

## Acceptance criteria

### Toggle

- WHEN an operator enables "Learn from my edits" on a project that owns its
  corrections list THEN `projects.learn_corrections` is true and future edits
  are observed.
- WHEN the project inherits its corrections THEN the toggle is disabled and
  enabling it returns 422 "Set corrections for this project first".
- WHEN the operator switches to inherit THEN `learn_corrections` is set false.

### Evidence collection

- WHEN an operator edits a segment's corrected text in a project with learning
  on THEN substitutions are extracted and each matching pair's evidence is
  recorded.
- WHEN the same segment is re-edited THEN its previous evidence is retracted
  before the new edit is observed.
- WHEN corrected text is reverted (set to None) THEN evidence is removed and
  orphaned suggested rows are deleted.
- WHEN a pair appears in base text but not raw text THEN it is not learned.
- WHEN a pair has more than 4 words on either side THEN it is not learned.
- WHEN the replaced word count exceeds max(4, base_words // 2) THEN the entire
  edit is skipped.

### Suggestions

- WHEN a suggested pair has evidence from at least 3 distinct segments THEN it
  appears on the project detail page's suggestion grid.
- WHEN evidence count is below the threshold THEN the pair is not visible.
- WHEN the project has 256 suggested rows THEN no new suggestions are created
  (existing ones still accumulate evidence).

### Accept and dismiss

- WHEN the operator accepts a suggestion THEN the pair is appended to
  `projects.corrections` as a validated rule with case_sensitive=True,
  whole_word=True, and the row is marked accepted with the rule's id.
- WHEN accepting would create a self-refiring or colliding rule THEN a 422 is
  returned with the operator message and the suggestion stays.
- WHEN the operator dismisses a suggestion THEN the row is deleted and can
  only return after 3 new segments carry the same edit.
- WHEN a corrections save removes an accepted rule's id from the list THEN the
  accepted row is deleted (the pair may be re-suggested later).

### Display

- WHEN a rule was learned THEN the project detail no-JS grid shows
  "learned xN" in the SOURCE column and the island CorrectionsEditor shows a
  SOURCE chip with the live count.
- WHEN the Settings corrections editor renders THEN no learned counts are
  passed and it displays as before.

### Regression

- WHEN a run is submitted in a project with an accepted learned rule THEN the
  frozen `pipeline_runs.domain_pack.corrections` contains the rule and the
  enhance stage fires it on matching raw text.

## Slices

1. Migration + ORM models + migration test
2. Pure extraction module (`learning.py`) + unit and contract tests
3. Observation service + edit-route hook + review API integration tests
4. Read model, routes, template, island chip + project integration tests +
   frontend build
5. Docs, changelog, gates, browser lane, PR

## Review notes

4-model consult before implementation:
- Codex (planner via zen clink): raw-gating requirement, rewrite cut-off
  formula, evidence-per-segment retraction design.
- DeepSeek V4 Pro: agreed on all points; suggested the max(4, base_words//2)
  formula over a flat 50% cut-off.
- Grok 4.5: agreed; flagged the inheriting-project edge case (which is
  handled by the toggle guard).
- Kimi K3: agreed; initially proposed an `active` intermediate state, which
  was cut for simplicity.
- Split 2/2 on whether learning should auto-propagate to inheriting projects.
  Ben chose: no propagation, an accepted suggestion becomes an ordinary rule
  in `projects.corrections` only.

Review tier: high-risk (migration, new routes with CSRF, island data contract).
Full 3-way `/code-review` panel + browser acceptance lane.
