# Transcript text correction — raw-vs-corrected provenance (issue #58)

**Status:** design decision, settling the provenance model before code.
**Context:** implemented alongside the #53 verify-and-advance triage loop, so the
loop's "fix" can edit words, not only relabel speakers. Gated by the doctrine
rule "#58 raw-vs-corrected provenance must be settled in a `docs/` note first —
product-identity fork, voxint is adjudication-only."

## The fork

Voxint has, until now, been adjudication-only: an operator resolves *speaker
identity* (assign / exclude / unknown / merge / per-segment relabel) but never
edits the *words*. `raw_text` is immutable ASR evidence; `enhanced_text` is
pipeline-authored (LLM punctuation/spacing), never operator-authored. Adding
human text correction makes voxint, in a bounded way, a transcript editor. This
note fixes the boundaries so that capability does not erode the numerics /
provenance doctrine.

## Invariants (non-negotiable)

1. **`raw_text` stays immutable.** It is the ASR observation of record. A
   correction is *never* written over `raw_text` (contract-tested today at
   `models.py:452`; the prose is extended, the guarantee is not weakened).
2. **`enhanced_text` is not repurposed.** It remains pipeline-authored; the
   enhancement stage may reset/rewrite it on a re-run. Corrections must survive a
   re-enhancement, so they cannot live in `enhanced_text`.
3. **Correction is operator-authored, single-operator, latest-wins.** No
   correction history is kept (a single local operator's working copy); the last
   edit is the truth. This mirrors how verified-state and per-segment overrides
   are treated — mutable operator workflow state, not an append-only evidence
   ledger.

## One choke point: `effective_text`

All four D-decisions rest on a **single shared selector**, so display, exports,
search, and enrichment can never drift:

```
effective_text(segment, review_state) =
    review_state.corrected_text   if corrected_text IS NOT NULL
    else segment.enhanced_text    if enhanced_text  IS NOT NULL
    else segment.raw_text
```

**`IS NOT NULL`, never truthiness** — an intentionally-empty correction (should
one ever be allowed) must not silently fall through. In v1 the write path coerces
empty/whitespace-only corrections to `NULL` (see write semantics), so an empty
string never persists; `IS NOT NULL` is still the coded rule (defensive + future-
proof). This helper replaces the ad-hoc `enhanced_text or raw_text` at every read
site (`transcript.py:102`, `run_assets.py:234`, the name producers).

## D1 — Storage  *(FINAL — both panelists endorse)*

A new **mutable** table, one row per reviewed segment, carrying *all* per-segment
operator workflow state introduced by #53/#58:

```
segment_review_states(
  transcript_segment_id  PK / FK -> transcript_segments.id  ON DELETE CASCADE,
  verified_at    timestamptz NULL,   -- #53 verified mark (NULL = unverified)
  corrected_text text        NULL,   -- #58 operator text (NULL = no correction)
  corrected_at   timestamptz NULL    -- non-NULL ⇔ corrected_text non-NULL
)
CHECK ((corrected_text IS NULL) = (corrected_at IS NULL))
CHECK (char_length(corrected_text) <= <BOUND>)   -- bound operator input
```

- **UPSERT keyed on `transcript_segment_id`** → naturally idempotent; no nonce,
  no append-only trigger, no audit machinery (a lost/duplicate write costs one
  re-click for a single operator).
- **`verified_by` / `corrected_by` CUT** (both panelists): a single-operator local
  tool would store one constant forever; the claim token already gates the write,
  and the UI "edited" badge keys on `corrected_at IS NOT NULL`, not an identity.
  *(Reinstate both `_by` columns only if multi-operator review is ever on the
  roadmap — the one condition that flips this.)*
- **`ON DELETE CASCADE`**: re-transcription mints new segment ids; cascade keeps
  re-ingest from leaking orphan review rows.
- **Not** on `transcript_segments` (keeps the immutable-observation row clean).
- **Not** the append-only `AdjudicationDecision` ledger: its CHECKs restrict
  segment-scope rows to `assign`/`inherit` and its resolver counts "resolved"
  labels — a verified/corrected row would violate the grammar and pollute
  resolution/queue counts. Verified/corrected is orthogonal to *speaker
  attribution*.
- One table for verified + corrected: both are per-segment, operator-authored,
  latest-wins state keyed by the same segment; nullable columns cover every
  combination (corrected, verified, both, neither). No-bloat: one table, one
  overlay load, one writer.

### Write semantics (both panelists — these are not optional)

- **Editing text atomically clears verification** (same transaction): a
  correction invalidates any prior `verified_at`, so you never have "verified"
  sitting on since-changed text. The explicit verify action follows the edit.
- **No-op guard**: if submitted text equals the current `effective_text`
  (enhanced-or-raw), store `NULL` — never badge unchanged text as operator-edited
  (same honesty doctrine as "uncertain, not necessarily wrong").
- **Empty/whitespace-only → `NULL`** (revert-to-pipeline-text). The PATCH route
  MUST support clearing a correction (`corrected_text → NULL`), else latest-wins
  UPSERT is accidentally append-only with no undo. (Deleting a hallucinated
  segment's content is served by the existing `suspect`/exclude mechanisms, not by
  persisting empty corrected text — avoids empty-cue/empty-line edge cases across
  the five export formats. Intentional-empty-as-deletion is a deferred option.)

## D2 — Enrichment DOES consume corrected text  *(REVERSED after panel — see below)*

**The panel split here.** Kimi (no-bloat) argued *exclude*: keep operator-mutable
data out of `source_content_hash` so a typo fix can't churn finished adjudication
or re-mine non-deterministic names. Codex argued *include*: `source_content_hash`
is defined to cover *what regeneration reads*, so excluding corrections makes
enrichment silently operate on different text than the console/export show.

**Resolved by code evidence → INCLUDE.** `run_assets.py:18-21,170-172` states its
own invariant: run assets must render **"the same name the review console and
export show,"** and it already routes names through the shared `display_name`
resolver *specifically so assets never disagree with the console/export*. Run
assets are operator-facing deliverables (summaries/transcript assets), not the
purely-advisory layer kimi's argument assumed. Once the console/export render
`corrected > enhanced > raw`, run assets **must** too, or they break their own
contract. Excluding is therefore not internally consistent.

**Decision:** every enrichment read site uses the shared `effective_text` helper
(`corrected > enhanced > raw`). `source_content_hash` then covers the corrected
text, so a correction to a proper name **honestly** marks name-suggestions /
summaries stale.

- **No re-run storm** (verified): enrichment is **operator-triggered** — the
  `enrich_names` route (`app.py:2504`) and CSRF-gated asset generation
  (`kinds_needing_generation` only *surfaces* "regenerate available", never
  auto-runs). Multiple edits collapse to one stale state, then one deliberate
  operator regeneration.
- Non-deterministic re-mining on regeneration is a pre-existing property of *any*
  `enhanced_text` change, not new to corrections; a stale *suggestion* flag never
  un-resolves a completed adjudication (accepting a suggestion is a separate
  explicit action).

## D3 — Corrected text IS full-text-searchable  *(FINAL — both panelists: keep in v1)*

The whole point of the triage loop is to fix uncertain words; an operator will
reasonably search for a word they just corrected. Search queries columns directly
(not the resolver): `runs_query.py:263-272` ORs the `raw_text` and `enhanced_text`
tsvectors — **deliberately never coalesced**, so a term is findable in *any*
rendering. Shipping a third rendering that search silently skips would break that
invariant and produce a *silent completeness lie* (operator corrects "GCHQ" in six
places, searches it, gets only the segments ASR already got right → concludes the
corpus is clean). For a correctness-doctrine tool that is worse than a missing
feature.

**Decision:** add a third tsvector rendering for corrected text, in lockstep with
the FTS contract test (`tests/contracts/test_fts_expressions.py`) and its migration.

- **Cheap because corrections are sparse.** `corrected_text` lives on
  `segment_review_states` (rows only for reviewed segments) and is NULL for most →
  a **partial GIN index `WHERE corrected_text IS NOT NULL`**, created in the *same*
  migration as the table.
- **Per-rendering headline, never coalesced.** `ts_headline` over
  `coalesce(corrected, enhanced, raw)` can pick a rendering that doesn't contain
  the matched term (match came from raw, snippet shows corrected) → "why did this
  match?" confusion. Keep the existing per-rendering headline branching; add a
  corrected branch with priority corrected → enhanced → raw among matches.
- **Join, don't denormalize.** `corrected_text` stays on `segment_review_states`;
  never copy it onto `transcript_segments` to simplify the query (re-blurs the D1
  boundary). Because it's a LEFT JOIN + 3-way OR, add an **EXPLAIN integration
  test** proving the partial corrected index actually participates in the plan;
  restructure with UNION branches if the planner ignores it.
- If it genuinely must slip under budget: degrade to "corrected not searchable in
  v1", stated loudly in release notes + the search docstring — never silently. (At
  ~50 lines riding an existing migration, slipping is unlikely to be worth it.)

## D4 — Display / export precedence and default  *(FINAL — both panelists)*

Precedence, applied at the single choke point `attributed_transcript()`
(`adjudication/transcript.py:102`) via the `effective_text` helper (**`IS NOT
NULL`, not truthiness**):

```
corrected  >  enhanced  >  raw
```

This isn't really a choice: the resolver *already* overlays operator speaker
overrides by default (default = "current adjudicated truth"), with `?text=raw` as
the evidence path. Making operator text edits the one thing the default *hides*
would contradict the tool's own semantics **and silently drop the operator's work
from exports** (data loss by default). The `TranscriptText` enum gains `CORRECTED`
and the resolver reads the `segment_review_states` overlay (mirroring per-segment
speaker overrides via `segment_states`). Variant semantics:

| `?text=` | renders | corrections applied? |
|---|---|---|
| *(default)* ≡ `corrected` | `effective_text` (corrected → enhanced → raw) | **yes** |
| `enhanced`  | `enhanced → raw`            | no |
| `raw`       | `raw` (immutable evidence)  | no |

- **`?text=corrected` IS the default** — one *named* variant, not a second
  switcher entry that no-ops. The variant switcher offers a three-rung ladder:
  my-corrections (default) / pipeline-text / asr-evidence. `?text=raw` stays the
  untouched ASR evidence path (contract-tested immutable, `test_review_api.py:272`).
- Formatters (`voxint.export`) need **no change** — they consume the already-
  resolved `TranscriptLine.text`, so all four line-based formats (TXT/SRT/VTT/JSON)
  and the CLI inherit precedence for free. **RTTM carries no text → N/A.**
- Update the `app.py:2587-2588` variant comment + `parse_transcript_text` default
  policy + CLI `choices`/help to the three-variant grammar.
- JSON export keeps its frozen `{start_seconds,end_seconds,speaker,text}` shape
  (no new `corrected` provenance key) — `text` simply carries the resolved value.

## Doctrine framing (honest UX copy)

- ASR **confidence** (`exp(avg_logprob)`, persisted by #53) is documented in
  `quality-gates.md` under "Confidence is not probability" as a *transformed
  likelihood, not a calibrated probability* — the UI says segments are
  **"uncertain, not necessarily wrong"**, never "N% correct".
- A **corrected** segment is visibly attributed as operator-edited in the review
  UI; the raw ASR text remains one click away (`?text=raw`) so evidence is never
  hidden.

## Consumers touched (the precedence rule, threaded)

| Area | File(s) | Change |
|---|---|---|
| **Shared selector** | new `effective_text(segment, review_state)` helper | the one place precedence lives; `IS NOT NULL` |
| Display/resolver | `adjudication/transcript.py:26,44,102` | `CORRECTED` enum + `effective_text` + overlay load |
| Island props | `api/app.py:1808-1828` | inherits (fed from resolver) |
| HTML | `templates/transcript.html` | inherits; variant switcher gains a 3rd rung |
| Exports | `api/app.py:2589-2640`, `cli.py:637-677` | accept `corrected` variant; update comment/help; RTTM unaffected |
| Search (D3) | `db/search.py`, `api/runs_query.py:263-317`, `models.py` partial index, new migration, `test_fts_expressions.py` + EXPLAIN test | 3rd tsvector (partial GIN) + OR-clause + per-rendering headline |
| Enrichment (D2) | `producers/names.py:115,251`, `names_llm.py:87,110`, `run_assets.py:234` | use `effective_text`; `source_content_hash` now covers corrected (honest staleness) |
| Write path | new claim-gated `PATCH /review/{run}/segments/{id}/text` | UPSERT; clears verification same-tx; no-op→NULL; empty→NULL; **not** wiped by `enhance_match.py:53-54` reset |
| Immutability docs | `models.py:452`, `docs/architecture.md` | extend prose: corrected written beside, never over, raw |

## Phase 3 — verify-and-advance UI surface decision (2026-08-17)

Settled with a 2-flagship consult (codex + kimi-k3); both independently chose the
same option, and both flagged the same single risk.

**Decision: a dedicated claim-gated route `GET /review/{run_id}/transcript?token=…`**
mounting one island (`review-stepper`) that *composes* the pure `TranscriptPlayer`.
Rejected: (B) mounting the review player onto the existing `/review/{id}` workbench
page — two `<audio>` on one page (keyboard-focus ambiguity); (C) a brand-new
stepper island duplicating playback/scroll/highlight.

- **Compose, don't mode-flag** (kimi): `TranscriptPlayer` stays pure (playback,
  highlight, auto-scroll, `playTurn`) and gains only a `forwardRef` imperative
  handle (`playSegment`). The loop — flag queue, typing-guarded keymap, verify/
  correct POSTs, N-of-M counter, current-segment edit textarea — lives in the thin
  `ReviewStepper` wrapper. The read-only `/runs/{id}/transcript` page ships
  byte-identical (it passes no ref, no review props).
- **The claim risk, and why A is safe** (both models): `claim_run` is per-reviewer
  with *takeover* — re-claiming mints a fresh token and kills the old one, NOT
  idempotent-per-holder. So the transcript review route must **reuse the existing
  claim token** carried in `?token=` (the same convention the workbench redirect
  already uses: `/review/{id}?token=…`), never acquire a fresh claim. On a stale
  token the GET renders read-only (mirrors the workbench GET); a 409 from a verify/
  correct POST stops the loop and preserves position + edit — never advances
  optimistically.
- **JS-off fallback**: the new page still owes a server-rendered flagged-segment
  list with plain verify form POSTs carrying the token.
