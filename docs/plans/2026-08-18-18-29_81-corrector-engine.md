# Plan: Voxint #81 — deterministic corrector engine + faithfulness contract & frozen corpus

## Context

Voxint (Apache-2.0, local-only, single-operator audio intelligence for non-technical
users) is building a deterministic, offline, **non-LLM** transcript-correction path
(epic #78) — on-ethos: fast, reproducible, no model, no prompt-injection surface,
complementing the optional LLM enhancement.

**#80 already landed** (`main` @ `94870c2`): it owns the frozen rule *type* + strict
load-time validation + the single-rule *matcher* in
`src/voxint/domain_packs/corrections.py`. Nothing applies corrections to text yet.

**#81 (this plan):** the pure, stdlib-only, versioned **multi-rule apply engine** that
consumes #80's `CorrectionRule`/matcher and turns `(segment_text, rules)` into corrected
text + a structured applied-rule trace, proven by a *stricter-than-LLM* faithfulness
gate. **No pipeline wiring, no DB migration** — those are #82.

Frozen design authority (do NOT re-litigate):
`docs/reports/nonllm-transcript-correction-design-2026-08-18.md` §6/§7/§8/§10/§12.

This plan was reviewed by codex (zen clink, planner role); its critique caught a
**critical correctness bug** in the first-draft algorithm — see Review notes.

## Assumptions & constraints

- **Reuse #80's matcher primitives unchanged** — `compile_match` (public) and the
  boundary predicate `_boundary_ok`/`_is_word_char`. Do NOT re-implement boundary/case
  logic; do NOT fork the idempotence guard's semantics.
- **Stdlib only** (`re` + PyYAML via the existing pack parser). No new runtime dep.
- Pure module: no DB/ORM/I/O/torch — import hygiene mirrors `src/voxint/enrichment/triage.py`.
- **No existing production enhanced-text growth constraint exists** (verified by grep;
  `reply_size_max_chars` lives only in fixture gold). The pure engine has no policy
  number to inherit — #82 supplies it.
- ≥85% coverage on new code (measured with `--cov`); `ruff` + `mypy` clean; type hints
  mandatory; never weaken an assertion to pass a gate.
- Invalidated if #80's matcher primitives change, or if #82 needs the engine to own
  persistence/policy (it does not — kept clean here).

## Key decisions

1. **Module location:** `src/voxint/domain_packs/corrector.py` — next to its only
   dependency `corrections.py` (pure same-package import, no cycle). *Rejected* the
   issue's `src/voxint/transcript/correct.py`: `src/voxint/adjudication/transcript.py`
   already exists (name confusion) and there is no `transcript` package. Codex concurs.

2. **Public entry point:** `apply_corrections(text: str, rules: Sequence[CorrectionRule],
   *, max_output_chars: int | None = None) -> CorrectionResult`. `rules` is an **ordered
   Sequence** (manifest order is observable — the final leftmost-longest tie-break).
   Renamed from `correct` (clearer, no builtin/collision ambiguity — codex).

3. **Growth seam:** the pure engine owns the growth *mechanism*; #82 owns the *policy
   value*. When `max_output_chars` is set and the **transformation** would exceed it,
   reject the whole segment atomically → `CorrectionResult(text=original, trace=(),
   growth_rejected=True)`; never truncate/partial-apply. Precedence & validation
   (codex-hardened):
   - **No-op precedence:** if no rule fires (empty rules / no winners), return
     `growth_rejected=False` **even if the untouched input already exceeds the limit** —
     the engine only rejects a *transformation* it would introduce.
   - **Projected-length preflight:** compute projected final length
     (`len(text) - Σ match_len + Σ replace_len` over winners) **before** building the
     output; reject without allocating the doomed string. Exactly-at-limit is allowed;
     one-over is rejected.
   - **Limit validation:** `max_output_chars` must be `None` or a positive `int`; reject
     `bool` (guard `isinstance(x, bool)` first) and non-positive with a `ValueError`.

4. **Result shape:** `@dataclass(frozen=True) CorrectionResult{text: str,
   trace: tuple[AppliedCorrection, ...], growth_rejected: bool}` and
   `@dataclass(frozen=True) AppliedCorrection{id: str, from_text: str, to_text: str,
   span: tuple[int, int]}`, with `to_mapping()` emitting JSON keys `id`/`from`/`to`/`span`
   for #82's persisted trace. `from_text` avoids the `from` keyword (codex: fine).

5. **`CORRECTOR_VERSION = 1`** (module-level int), contract-pinned like `TRIAGE_VERSION`,
   **plus** one exact pinned `to_mapping()` example (Unicode + shifted spans + a
   growth-rejection result) so trace serialization compatibility is pinned, not just the
   int (codex).

6. **`enhance_match` identity test → formally reassigned to #82.** The spec §7A's "one
   separate no-LLM/no-rules `enhance_match` identity test" exercises the stage's
   `enhanced_text` write/JSON-serialization path, which #81 does not wire. #81 delivers
   the corrector-only Gate A; the stage-level test moves to #82 (documented here + in the
   #82 scope so the design report's acceptance item is not silently dropped). **Surfacing
   to the user** as the one scope call (see Open questions).

## Core algorithm — cursor-relative leftmost-longest, single non-cascading pass

The first draft eagerly materialized all `iter_matches` output then selected; codex
proved this **misses cross-rule overlapping matches** (see Review notes). Corrected
design:

Helper `_first_match_from(rule, text, pos) -> tuple[int, int] | None`: reuses
`compile_match(rule)` and `_boundary_ok`, looping `search(text, p)` and advancing
`p = m.start() + 1` past boundary-invalid candidates until the first boundary-valid hit
at/after `pos` (identical per-rule semantics to `iter_matches`, but seeded at a live
cursor rather than restarting at 0).

Main loop:
1. `pos = 0`, `winners = []`.
2. For each rule, `_first_match_from(rule, text, pos)` → its earliest eligible candidate
   **rediscovered from the current cursor** (this is what surfaces a match another rule
   previously hid).
3. Winner = leftmost `start`; tie-break longest (`max end`); tie-break min manifest
   index. If no candidate anywhere ≥ pos, stop.
4. Append winner; `pos = winner.end`; repeat from 2.
5. Preflight projected length; if over `max_output_chars` → atomic reject (decision 3).
6. Build final text from **original** slices (gaps between winners) + each winner's
   `replace`; compute each `span` in the FINAL string via a running output offset.
7. Trace per winner: `{id, from_text = text[start:end], to_text = rule.replace,
   span = (final_start, final_end)}`.

Non-cascading is intrinsic: candidates come only from the ORIGINAL `text`; a replacement's
own characters are never searched. Complexity ≈ O(rules × winners × search) — bounded and
avoids the eager approach's O(rules × all-matches) storage (codex risk note).

## Affected files / components

- **NEW** `src/voxint/domain_packs/corrector.py` — `apply_corrections`, `_first_match_from`,
  `CorrectionResult`, `AppliedCorrection`, `CORRECTOR_VERSION`. Imports only
  `corrections`/stdlib.
- **NEW** `tests/fixtures/rules_correct/*.json` — Gate B corpus (schema: `rules`, `input`,
  and either `expected_text` + `expected_trace`, or `expect_load_error` /
  `expect_growth_rejected`).
- **NEW** `tests/unit/test_corrector.py` — engine unit tests + Gate B corpus driver +
  the bidirectional trace walker.
- **NEW** `tests/unit/test_corrector_faithfulness.py` — Gate A (six reused fixtures,
  empty rule set, byte-identical + NFC), reusing `tools/qualify_local_llm.py` scorers.
- **NEW** `tests/contracts/test_corrector_config.py` — pin `CORRECTOR_VERSION == 1` + the
  exact `to_mapping()` example + growth/no-op invariants, mirroring `test_triage_config.py`.
- **EDIT** `docs/domain-packs.md` — engine-semantics subsection (leftmost-longest,
  exact-literal replace, growth rejection, ordering, **limited** idempotence).
- **EDIT** `CHANGELOG.md` — `[Unreleased] → Added`.

## Step-by-step implementation

1. `corrector.py`: dataclasses + `CORRECTOR_VERSION` + `_first_match_from` + the
   cursor-relative loop + `to_mapping()`.
2. Engine unit tests, **including the `xababa` cross-rule regression first**
   (leftmost-longest, earlier-start-shorter beats later-start-longer, same-start longest,
   identical-span manifest-order tie, adjacent winners, expanding+shrinking replacements,
   empty text/rules/no-match, non-cascading, exact-literal replace under `IGNORECASE`).
3. Growth tests: exactly-at-limit (allowed), one-over (rejected), shrinking output,
   multi-winner atomic rollback (empty trace on reject), no-op precedence
   (over-limit untouched input → not rejected), limit-arg validation.
4. Gate A (`test_corrector_faithfulness.py`).
5. Gate B corpus JSON + driver + bidirectional trace walker.
6. Contract test (version + pinned mapping + invariants) — invariant and its test land in
   the same commit (repo rule).
7. Docs + CHANGELOG.
8. Tail: `uv run ruff check .`, `uv run mypy`, targeted pytest **with**
   `--cov=voxint.domain_packs.corrector --cov-report=term-missing` (≥85%), then the
   **full** `uv run pytest` suite (this module joins domain-pack imports); 3-model
   `/code-review` → apply Critical/High/Medium → `/commit` BOTH remotes → FF-merge `main`.

## Testing strategy

**Gate A — regression, empty rule set.** For each of the six
`tests/fixtures/llm_qual/enhancement/*.json`, per segment call `apply_corrections(text, ())`
and assert `result.text == text` byte-identical (decoded strings), `result.trace == ()`,
`growth_rejected is False`; assert every fixture segment is already NFC
(`unicodedata.is_normalized("NFC", …)`) so the "no NFC normalization" contract is
explicit. Retain one `score_segment` cross-check per fixture (protected-tokens /
no-merge-split-reorder, the spec's enumerated §7A checks) — codex flags this as
*largely redundant with byte-identity*; kept deliberately because the spec/prompt name
those scorers as the reuse substrate and the check is cheap. (Stage-level identity test:
**#82**, decision 6.)

**Gate B — new `tests/fixtures/rules_correct/`**, one-failure-blocks (assert-per-case, no
aggregate). Valid-pack fixtures round-trip through `parse_corrections` (so a test can't
smuggle an invalid set as production-valid — codex); isolated engine-mechanics tests may
construct `CorrectionRule` directly. Cases:
- positive required substitution (multiword + Unicode term)
- negative collision (`catalog` not matched by a `cat` whole-word rule) → zero changes
- boundary/case: possessive `it's` under `it→IT`, hyphenated compound,
  punctuation-adjacent, NFD `Zoë` (decomposed) — no corruption; case-sensitive vs
  `IGNORECASE`; exact-literal (non-case-inheriting) replace
- leftmost-longest determinism incl. the **`xababa` cross-rule overlap** case
- idempotence over a *validated* set `apply(apply(t)) == apply(t)` + reuse #80's
  load-rejection of `[a→b, b→c]`; **plus** a documenting test that a **cross-boundary**
  chain (`ab→x`, `xc→y` on `abc`) is NOT guaranteed idempotent and is outside #80's guard
  (honest limit — codex)
- regex-metachar literal `C.D.B.G.` treated literally
- growth rejection (small `max_output_chars` → unchanged + `growth_rejected=True`)
- **trace completeness via a bidirectional walker**: spans ordered, half-open, in-bounds,
  non-overlapping; `result.text[start:end] == entry.to`; `entry.to` == that rule's exact
  `replace`; walking source and result together, output gaps equal source gaps
  byte-for-byte and the next source slice equals `entry.from`; suffixes match after the
  last entry. Plus **exact pinned `expected_text` + `expected_trace`** for representative
  frozen fixtures (don't use engine internals as the oracle — codex).

**Faithfulness gate = ZERO unauthorized edits** (every changed char covered by an applied
rule; every unchanged gap byte-identical). Deterministic ⇒ single run (no `--reps`).

## Rollout / risks / open questions

- **Risk (resolved):** cross-rule leftmost-longest correctness — the cursor-relative
  rediscovery fixes the eager-materialization bug; the `xababa` regression guards it.
- **Risk:** importing the module-private `_boundary_ok` from `corrections.py`. Mitigation:
  same-package reuse of the frozen predicate is intended (codex endorses reusing it
  unchanged). If ruff/hygiene objects, fallback (b): add a thin **public**
  `first_match_from(rule, text, start=0)` to `corrections.py` that `iter_matches`/
  `find_first` delegate to (behavior-preserving; existing 40 tests stay green) and consume
  that instead. Default = direct reuse; escalate to (b) only if lint demands.
- **Open question for the user:** confirm decision 6 — ship the `enhance_match` stage-level
  identity test in **#82** (needs the stage write path #81 doesn't own), with #81
  delivering corrector-only Gate A. Alternative: attempt a minimal stage baseline now.
- **Note:** on approval the final plan is also persisted to
  `docs/plans/{timestamp}_{title}.md` per repo convention (plan-mode currently restricts
  writes to this file).

## Review notes (codex, zen clink · planner)

Codex inspected `corrections.py`, `enhance_match.py`, the design report, the scorers,
`test_domain_packs.py`, `test_triage_config.py`, the six fixtures, `pyproject.toml`,
`CONTRIBUTING.md`, `CHANGELOG.md`. Overall: "mostly sound, but the candidate enumeration
is incorrect and the growth/test contracts need tightening." Disposition:

- **ACCEPTED (critical):** eager `iter_matches` materialization misses cross-rule
  overlapping matches (`xababa` counterexample). → Rewrote to cursor-relative rediscovery;
  `xababa` regression added first.
- **ACCEPTED:** growth no-op precedence + projected-length preflight + limit-arg
  validation (zero/negative/bool/non-int). → Folded into decision 3.
- **ACCEPTED:** trace oracle must walk both coordinate spaces + pin exact expected traces;
  reconstruction-only is insufficient. → Bidirectional walker + pinned fixtures.
- **ACCEPTED:** don't claim `validate_corrections` gives universal idempotence
  (cross-boundary refiring possible). → Documenting test + precise docs wording.
- **ACCEPTED:** `rules` is an ordered `Sequence`, not a set; rename to `apply_corrections`;
  pin an exact `to_mapping()` example in the contract test; use `--cov` + full suite; use
  `parse_corrections` for valid-pack fixtures. → All folded in.
- **ACCEPTED (scope):** deferring the `enhance_match` identity test must be *formal*, not
  silent. → Decision 6 reassigns it to #82 explicitly + surfaces to the user.
- **NOTED / kept with rationale:** codex called the Gate A `score_segment` cross-check
  "largely redundant" with byte-identity and flagged importing private tool helpers. Kept
  a single cross-check per fixture because the spec/prompt name those scorers as the reuse
  substrate and it's cheap; primary Gate A assertion remains byte-identity.
- **ACCEPTED (naming, minor):** `domain_packs/corrector.py` + `from_text/to_text` fine.
