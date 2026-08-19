# Plan — #83 Expose deterministic-correction provenance in the review console

**Issue:** #83 (4th child of the #78 non-LLM-correction epic). READ-side provenance
+ reversion affordance + docs only; authoring is #84 (out of scope).
**Branch:** `feat/83-console-correction-provenance` off `main` @ `0c4935e`.
**Alembic head:** `0028`. **Status at plan time:** ruff+mypy clean, both remotes synced.
**Reviewed by:** codex (zen clink, planner role) — see Review notes at the end.

---

## Goal

A non-technical single operator, in the review console, can: (1) see that
deterministic domain-pack corrections fired on a segment and **which pack + rule**
produced each edit; (2) view/export the immutable `raw_text` one action away for
comparison/reversion; (3) read honest "splitting disabled because a material
correction applied" messaging; (4) see **declared-rule reconciliation** — rules
declared in the pack that never materially fired for this run ("declared but never
fired"). Nothing changes silently, and deterministic **pipeline** provenance is
never conflated with a later **operator** edit.

## Assumptions & constraints

- `_island_segment` (`src/voxint/api/app.py:1213`) is the ONE per-segment builder
  shared by hydration + `/split` + `/relabel` reconcile (`_run_island_segments`
  :1263, `_run_reconcile_response` :1272). A field added there flows everywhere.
- A run freezes **exactly one** pack into `pipeline_runs.domain_pack` (JSON
  snapshot); rule ids are uniqueness-enforced within a pack (`validate_corrections`).
  So a fired trace-entry `id` resolves unambiguously — **no migration for fired-rule
  provenance display** (Decision A, settled).
- `correction_trace` envelope (`{version, input_base:"raw"|"llm",
  entries:[{id,from,to,span:[s,e]}]}` or `[]`) is a **frozen contract**:
  `trace_has_entries` + splittability depend on `[] == not materially corrected`.
  **Do not extend the envelope shape.** `CORRECTOR_VERSION == 1` today.
- `raw_text`/`words`/intervals are immutable ASR evidence; read-time precedence is
  `corrected → enhanced → raw`. Trace spans address the **persisted enhanced text
  in its `input_base` coordinate space** — never the operator-effective text.
- Doctrine: public clean-room (no internal hosts/IPs); type hints + `uv` + ruff +
  mypy clean; ≥85% new-code coverage; **NEVER weaken an assertion**; contract-test
  any new invariant **in the same commit**; update docs in the same change; no
  unnecessary bloat.
- **Invalidated if:** a second pack per run becomes possible, or the envelope shape
  changes, or `CORRECTOR_VERSION` bumps without a version-dispatch strategy.

## Decision B (reconciliation) — RESOLVED for v1, with an open confirm

**Chosen: read-time reconstruction, NO migration, with hard correctness guards.**
The design report (§6 L207-212, §10-D, §12-F5) explicitly permits *not* computing
`cross_segment`/growth exhaustively in v1 and steering cross-segment terms to
`vocabulary`. Read-time keeps this read-side issue read-side and avoids a
write-path change to the frozen #82 numerics stage.

v1 reconciliation statuses, all **exactly** derivable read-time by re-running
`apply_corrections(raw_text, declared_rules, max_output_chars=…)` over the
immutable `raw_text` (independent of the persisted trace, so `input_base:"llm"`
spans are irrelevant):

| Status | v1 treatment |
|---|---|
| `applied` | Declared rule fired on ≥1 segment's raw (exact). |
| `no_raw_match` | Declared rule matched **no** segment's raw (exact). |
| `growth_rejected` (raw pass) | Re-run with the growth cap → `growth_rejected=True` (exact). |
| `growth_rejected` (LLM-enforcement pass) | **Not computed in v1** — honest UI + docs boundary. |
| `cross_segment` | **Not computed in v1** — surfaced as a *concept* with remediation ("declare as `vocabulary`"). |

**Guards codex flagged (all folded in):**
1. **Version gate.** Reconstruct provenance/reconciliation only when the row's
   `corrector_version == CORRECTOR_VERSION`. Otherwise emit an honest
   `provenanceUnavailable` state ("recorded by corrector v{n}; this console reads
   v{cur}") — never silently replay with mismatched semantics.
2. **No snapshot fabrication.** Read the `run.domain_pack` **dict directly**; do
   **not** call `domain_pack_from_snapshot` (it degrades NULL/corrupt snapshots to
   the *current default pack* — `registry.py:157-165` — which would fabricate
   declarations). NULL/malformed snapshot → no provenance + honest "unavailable".
3. **Unresolved ids stay visible.** A persisted trace `id` absent from the declared
   snapshot renders as "unresolved rule `id`", never dropped.
4. **Provenance ≠ edit.** Distinct labels: "corrected by domain pack" vs "edited by
   you". A raw-matched rule can legitimately vanish from final text (LLM already
   canonicalized) — reconciliation and final-text provenance are *different concepts*.
5. **Span safety.** Never highlight trace spans against operator-effective
   `current.text`; spans are display-only against the matching `input_base` text.

**Rejected (with reason):** codex's Option 2 (persist an observational ledger +
change the corrector's observational contract to expose candidates on atomic
rejection + migration 0029). Its *correctness concerns* are accepted (guards 1-5
above apply to read-time too); its *mechanism* is rejected for v1 as scope
expansion of a read-side issue into the frozen numerics engine, for the rarest case
(LLM-enforcement growth rejection). **Deferred** to a follow-up issue if field use
shows exact LLM-pass growth-rejection reconciliation matters.

> **RESOLVED (user, 2026-08-18):** read-time v1 confirmed. No migration; the
> LLM-enforcement-growth and `cross_segment` cases are honest, documented v1 gaps,
> deferred to a follow-up issue if field use warrants. (codex had recommended the
> persistence path; user chose read-time — "read-side stays read-side".)

## Proposed approach

Read-time provenance threaded through the existing single builder; run-level
reconciliation computed once server-side. Reuse `trace_has_entries`; never re-diff
effective text. Pure, DB-free helpers carry the logic (and the coverage).

**Alternatives considered:**
- *Persist per-segment status arrays (codex Option 2 lite).* Rejected for v1 (bloat
  + frozen-stage write-path change); deferred.
- *Fetch provenance lazily per segment from a new endpoint.* Rejected: the single
  shared builder already carries every render path; a second endpoint duplicates
  auth/claim plumbing. **But** revisit for `rawText` specifically if payload
  measurement (Step 0b) shows material growth — a lazy raw endpoint is the fallback.

## Affected files / components

**Backend**
- `src/voxint/adjudication/transcript.py` — add `correction_trace`,
  `corrector_version`, `raw_text` to the `TranscriptLine` dataclass; set at the 2
  construction sites (unsplit + split children). Split children are **parent-scoped**:
  carry parent trace but the payload marks provenance as parent-level (no child-local
  span claims).
- `src/voxint/adjudication/corrections_view.py` — **new** pure module:
  `build_declared_rule_index(snapshot_dict) -> dict[str, RuleDisplay]` (declared, not
  fired); `resolve_segment_provenance(trace, corrector_version, index) -> …`
  (version-gated, unresolved-id-preserving); `run_reconciliation(declared_rules,
  raw_texts) -> list[ReconEntry]`. No Session, no I/O.
- `src/voxint/api/app.py` —
  - thread a per-run declared-rule index + a small DB-load wrapper into
    `_island_segment(ln, palette, index)`; emit `corrections` (`{version, inputBase,
    entries:[{id,from,to,span,pack,match,replace}]}` or null) + `rawText` +
    `provenanceUnavailable`;
  - attach `island_props["reconciliation"]` at :3369 (counts + per-rule status +
    affected-segment refs);
  - verify/text response path: ensure an operator edit **supersedes** provenance
    display (clear/flag stale spans in `applyResult`, don't patch text alone);
  - split-probe reason (:4232) may be enriched with the pack/rule name (optional).
- `CHANGELOG.md` — `[Unreleased]` entry (repo convention).

**Frontend** (`frontend/src/`)
- `components/TranscriptPlayer.tsx` — extend the inline `Segment` interface (:21-63)
  with `corrections`/`rawText`/`provenanceUnavailable`; pure/prop-driven only.
- `components/ReviewStepper.tsx` — "corrected by domain pack" marker + expandable
  pack/rule list beside the "edited" badge (:760-762); "reveal/compare/copy(+fallback)/
  open raw" affordance near the edit textarea (:168-174, 764-786) — reset-to-raw
  **populates the textarea only**, preserves undo/discard, requires the existing Save;
  run-level "declared but never fired" collapsible in the header (:726-748), typed on
  `ReviewStepperProps` (:14-35); safe truncation + accessible expansion; keyboard/focus/
  screen-reader/empty states.

**Docs**
- `docs/domain-packs.md` — extend the #82 dual-pass section: provenance display,
  reconciliation status meanings, the **explicit v1 boundary** (LLM-enforcement
  growth + cross_segment not exhaustively computed; use `vocabulary` for
  cross-segment terms), precedence `corrected→enhanced→raw`, regex unsupported v1.
- The review how-to — the new provenance/raw/reconciliation affordances.

## Step-by-step implementation

- **Step 0a.** Write the reconciliation **truth table** (status × phase × data
  state) + resolved status/precedence semantics into the plan/docs BEFORE coding
  (codex's "immediate next action"). Fixes: cardinality (per-rule/run for the panel;
  per-segment for the marker), precedence when a rule is `applied` on one segment but
  `no_raw_match` on another (→ `applied` wins for the run panel; per-segment marker is
  literal), and the `applied` definition (= matched raw; distinct from "present in
  final text").
- **Step 0b.** Measure payload delta of adding `rawText` to every segment on a
  representative large run; if material, switch `rawText` to a lazy per-segment
  endpoint (decision recorded, not silent).
- **Step 1.** Branch off `main`.
- **Step 2.** Pure `corrections_view.py` + its unit tests (TDD-friendly; no DB).
- **Step 3.** Thread trace/version/raw onto `TranscriptLine` (2 sites); wire the
  index + `_island_segment` fields; version-gate + snapshot-direct-read + unresolved-id
  handling; run-level `reconciliation`. Backend contract + integration tests.
- **Step 4.** Frontend: interface + marker/expander + raw affordance + reconciliation
  panel + a11y; behavioral tests.
- **Step 5.** Operator-edit-supersedes-provenance wiring + its test.
- **Step 6.** Docs + CHANGELOG (same commits as the behavior).
- **Commit strategy (stacked, each buildable+tested, one branch):** (i) readers
  (`corrections_view` + line threading + payload) + tests; (ii) frontend +
  behavioral tests; (iii) docs + CHANGELOG. Commit BEFORE any background reviewer.

## Testing strategy

≥85% new-code coverage; pure helpers are the backbone (table-driven). Dimensions
(from codex's gap list, folded in):
- **Envelope/data states:** `[]` vs envelope-with-empty-entries vs envelope-with-
  entries; legacy-NULL `corrector_version`; unknown/unresolved id; malformed
  entry/span; future-version envelope (→ `provenanceUnavailable`).
- **Snapshot states:** NULL snapshot, corrupt snapshot, duplicate-id (can't happen
  post-validation but assert the guard), name+corrections round-trip.
- **Reconciliation:** rule applied-in-some/no-match-in-others (pin run-panel
  precedence); raw-pass growth rejection (single rule; multiple collectively
  overflowing); raw-matched rule absent from final (LLM canonicalized) surfaced as a
  *distinct* concept, not a failure.
- **Split-parent** rows carrying trace (parent-scoped assertion; no child-local span).
- **Live paths:** hydration + both whole-run reconcile paths after split and relabel
  agree (same builder).
- **Operator edit** layered over provenance → stale spans suppressed.
- **Frontend behavioral:** expand/reveal; reset-to-raw preserves unsaved-edit
  discard protection; copy fallback (non-secure context); focus + accessible status.
- **Mandatory browser E2E** (`.claude/skills/voxint-e2e-review/`, serial, this host)
  for the core operator flow — NOT optional; the user-facing goal warrants it.

**Gates:** ruff+mypy; pytest unit+contracts (expect only the 2 known `render:990`
installer fails); disposable-Postgres integration; frontend lint/typecheck/build;
representative reconciliation performance check; secret grep (gitleaks not installed
→ grep `192.168.`/hosts/tokens); mandatory core-flow browser test.

## Rollout / risks / open questions

- **Risk — payload growth** from per-segment `rawText` (Step 0b measures; lazy
  endpoint is the fallback).
- **Risk — read-time replay cost** (≤256 rules × segments on page load) — benchmark
  gate; reconciliation is per-run computed once, not per-segment-per-request.
- **Risk — version drift** — handled by the version gate + honest fallback (only v1
  exists today; forward-safe).
- **Open question (user): RESOLVED** — read-time v1 confirmed (2026-08-18); see
  Decision B box.
- **Deferred to a follow-up issue:** exact LLM-enforcement growth-rejection
  reconciliation and `cross_segment` detection, if field use warrants persistence.

## Review notes (codex, zen clink planner — 2026-08-18)

Codex inspected all cited files and produced a 3-step critique. Disposition:

- **Decision B — codex recommended Option 2 (persist a versioned observational
  ledger + change the corrector to expose candidates on atomic rejection).**
  *Partially accepted:* all five correctness concerns folded in as guards (version
  drift, snapshot-fallback fabrication, unresolved ids, provenance-vs-edit, span
  safety). *Mechanism rejected for v1* as scope expansion into the frozen numerics
  engine for the rarest case; deferred + surfaced to the user as the one open
  decision. Codex correctly noted Option 1-as-drafted omitted required statuses — the
  plan now surfaces `applied`/`no_raw_match`/raw-pass `growth_rejected` exactly and
  is explicit+honest about the two v1 gaps (design report sanctions this).
- **`domain_pack_from_snapshot` default-pack fallback fabricates provenance on
  NULL/corrupt snapshots** — *accepted*; plan reads the snapshot dict directly.
- **`_run_corrections_index(session, run_id)` isn't pure; it maps declared not fired
  ids** — *accepted*; split into a DB-load wrapper + pure
  `build_declared_rule_index`; renamed; unresolved ids preserved.
- **Split-parent children reusing parent trace / parent-coordinate spans are
  misleading** — *accepted*; provenance is parent-scoped, no child-local span claims;
  tested.
- **Change propagation: verify/text `applyResult` patches text only, risking stale
  spans** — *accepted*; operator edit supersedes provenance (Step 5).
- **Missing CHANGELOG + migration up/down verification** — *accepted* (CHANGELOG
  added; migration verification only applies if the deferred persistence path is
  chosen).
- **Performance (read-time replay + rawText payload)** — *accepted* as gates
  (Step 0b measurement, reconciliation perf check, lazy-raw fallback).
- **Reset-to-raw ambiguity, clipboard-failure, truncation, a11y, empty/unavailable
  states** — *accepted*; specified in the frontend section.
- **Mandatory (not optional) browser test** — *accepted*.
- **Write the reconciliation truth table before coding** — *accepted* as Step 0a.
- **Alternatives (hybrid reconstruction / run-level summary / engine preflight)** —
  *noted*; the run-level summary shape informs the deferred persistence issue.
