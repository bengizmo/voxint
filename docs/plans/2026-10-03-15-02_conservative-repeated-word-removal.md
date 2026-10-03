# Plan: conservative repeated-word removal in the Markdown turns export (#755)

Status: done

Spec deltas: none (this project declares no living spec).

## Goal

Add a separate, opt-in export option, `repeats=drop` (HTTP) / `--drop-repeats`
(CLI), that removes an immediately repeated one- or two-word run of function
words inside one speaker turn: `go to the the store` becomes `go to the store`,
`we were we were going` becomes `we were going`. It belongs to epic #752
(transcript clean-up closer to Descript). The maintainer chose automatic narrow
rules on 2026-10-03 (recorded on #755); confirm-only review waits for #757.

Like fillers, this is a render-time projection: stored transcript data never
changes, and an export without the option is byte-identical to today's. The
rules are deliberately narrow. When a candidate is ambiguous, the rule declines
it rather than adding grammar machinery or a setting.

## Assumptions and constraints

- **Scope matches `fillers=drop`:** Markdown, `turns` style, no translation.
  Read mode and plain text belong to #754, which will pick up both options
  through the same turn pipeline.
- **Honesty (#755: "same honesty report as fillers"):** fillers have no runtime
  report. Their honesty is the documented rules, "nothing stored changes", and
  a plain 422 or exit-2 message for each invalid combination. This plan matches
  that. A runtime "what was removed" report is #753's item for both options.
- **Language:** the word lists are English. Like fillers today, the rule applies
  to whatever text is exported, and the docs say it is meant for English
  transcripts. Language-aware lists belong to #756 for both options
  (maintainer decision 2).
- **Footprint:** no new dependency, configuration knob, DB change or migration.
  No numerics impact (render-time text only), so no parity gate applies.
- **What would invalidate this plan:** the maintainer wanting repeats in read
  mode or plain text now, or a runtime report now.

## Proposed approach

A new pure module `src/voxint/export/repeats.py`, shaped like `fillers.py`:
`drop_repeats(turns: Sequence[SpeakerTurn]) -> list[SpeakerTurn]` rewrites
piece text only and keeps attribution, order, piece timing and flags. The
module docstring carries the rules R1 to R9 the way `fillers.py` carries F1 to
F6.

### Rules

- **R1 Tokens and comparison.**
  - **Word:** a maximal run of letters with optional internal apostrophes
    (`'` or `’`), bounded on both sides by whitespace, turn start or turn end.
  - **What blocks a copy:** any other attached character, including a digit,
    hyphen, dash, slash, underscore, bracket, quote mark or punctuation mark.
    This holds before the first copy, between the copies, and after the last
    copy: `the the.` stays, `the-the` stays, `the the2` stays.
  - **Gap between copies:** whitespace by `str.isspace()`.
  - **Comparison key:** `casefold()` with `’` normalised to `'`. The original
    characters are always kept.
  - **Case differences:** allowed only when the first copy starts a sentence
    (turn start, or after `.`, `?` or `!`) and is not all caps. `The the cat`
    matches. `US us` and `May may` in mid-sentence do not.
- **R2 One-word repeat.** The pattern is `W W`, where W is in the one-word
  eligible set (below). The pair must be a maximal run of exactly two equal
  tokens. The second copy goes; the first copy, with its casing and timing,
  stays.
- **R3 Two-word repeat.** The pattern is `A B A B`, where A ≠ B and both are in
  the function-word list. The pattern must not extend on either side: the token
  before must not be B and the token after must not be A. So `we were we were
  we` and `were we were we were` stay. The second `A B` goes.
- **R4 Three or more copies stay.**
  - `I I I`, `no no no`, `we were we were we were`, `that that that that`
    and `no no no no` all stay unchanged.
  - A token inside an equal-token run longer than two can never be part of an
    R2 or R3 candidate.
  - R3 needs A ≠ B, so four equal words never become a two-word repeat. That
    keeps R4 and the one-word exclusions from being bypassed through R3.
- **R5 Overlapping candidates cancel.** Candidates are found on the input text
  only, and two candidates that share a token both decline: `we were we were
  were` stays. Non-overlapping candidates all apply, in one pass with no
  rescan: `we we were were` becomes `we were`.
- **R6 Boundaries.** Both copies and the gap must sit inside one speaker turn
  as produced by the attribution, and inside one transcript segment (no piece
  with `segment_start` after the first copy's first character).
  - When timed pieces are involved, removing the copy must not leave a gap of
    `PARAGRAPH_PAUSE_SECONDS` (3.0 s, `export/reading.py:29`) or more between
    the surviving neighbours.
  - Such a candidate declines, so layout never gains a paragraph break because
    a stutter was removed.
  - Cross-speaker echoes stay because matching is per turn.
- **R7 Quotes.**
  - A candidate any of whose tokens sits inside a quoted span declines.
  - **Double quotes:** `“` opens and `”` closes. ASCII `"` toggles.
  - **Single quotes:** a `'` or `‘` directly before a word's first letter, and
    preceded by whitespace or turn start, opens a span. The next `'` or `’`
    directly after a word's last letter, followed by whitespace, punctuation or
    turn end, closes it. An internal apostrophe neither opens nor closes.
  - **Unmatched marks:** an unmatched opening mark quotes the rest of the
    turn; an unmatched closing mark opens nothing.
  - Over-protection only means fewer removals. Quote state is computed per
    turn, on the text as it reaches this filter.
- **R8 Separators.**
  - Removing a copy also removes all whitespace on both sides of it.
  - When text survives on both sides, exactly one space is added back, owned
    by the next surviving character's piece (the F3 technique,
    `fillers.py:99-102`). At turn end nothing is added back.
  - A piece whose text becomes empty or whitespace-only is dropped. No piece
    is left holding only punctuation, because R1 blocks punctuation attached to
    a copy.
  - Untouched text keeps its exact whitespace.
- **R9 No-op and composition.**
  - With the option absent, the filter is not called at all.
  - With it given and no candidate (or only declined ones), output bytes are
    identical, and unchanged turns and piece tuples are reused rather than
    rebuilt.
  - **With fillers:** filler cleaning runs per turn first, then repeats, then
    the filler seam merge (F5).
    - So `the um the cat` becomes `the cat` with both options. With repeats
      alone it stays.
    - An A `the` / B `um` / A `the cat` sequence keeps both `the` copies. The
      merge seam is the original interruption, which repeats do not cross
      (maintainer decision 1).
    - Composition: `drop_fillers_with_seams` then `drop_repeats(turns,
      seams)`, which treats a seam like a segment boundary. The existing
      filler goldens pin `drop_fillers`' own behaviour byte for byte.

### Word lists (module constants, mirrored in `docs/operations.md`)

- **Function words** (eligible inside R3 and, minus the exclusions, R2):
  - **Pronouns:** `i me my we us our you your he him his she her it its they
    them their what who`.
  - **Auxiliaries and modals:** `am is are was were be been being have has had
    do does did would shall should could may might must`.
  - **Contractions** (generated as a complete paradigm, not hand-picked):
    - pronoun + `'m 're 's 've 'll 'd` for `i we you he she it they that`;
    - `n't` forms of `is are was were do does did have has had could should
      would` plus `can't won't`.
  - **Determiners:** `a an the this that these those some any each every no`.
  - **Prepositions:** `in on at to of for from with by about into over under
    after before between through up down out off`.
  - **Conjunctions:** `and or but so if because as than then when while`.
- **One-word exclusions** (never removed by R2; still allowed inside R3):
  - Grammatical doubles: `that` (I know that that is true), `had` (she had had
    enough), `is` and `was` (what it is is; what it was was), `do` and `did`
    (what I do do; the work they did did nothing), `her` (I gave her her keys).
  - Words with a common content reading: `can`, `will` (the will will be read).
    Neither is in the function-word list at all.
  - Answer and discourse words, where two copies can carry meaning: `no`, `so`.
- **Deliberately absent** (recorded so they are not relitigated): `well like
  right just there yes yeah okay how where why`.
- **Recall cost (documented, not fixed):** content-word restarts (`we went we
  went`), `you know you know`, `I mean I mean`, and any repeat with a comma
  between copies (`We were, we were going`) stay. Whisper often punctuates a
  restart pause, so the two-word rule will fire less often than the one-word
  rule.

### Option plumbing (mirrors fillers one for one)

- **`export/service.py`:** `parse_repeats(raw, fmt, style) -> bool`, validated
  exactly like `parse_fillers` (`service.py:56-68`):
  - empty means absent;
  - `keep` and `drop` are both rejected outside md turns, with "repeats
    applies to the md turns style only";
  - an unknown value gets "unknown repeats value 'x'; valid: keep, drop".
- **`render_run_transcript(..., drop_repeats: bool = False)`:**
  - rejects a translation ("repeats cannot be combined with a translation")
    before loading data, including when `translated_texts == []`;
  - validation precedence is style, then fillers, then repeats;
  - with both options and a translation, the fillers message wins.
- **`cli.py`:** `--drop-repeats`, parsed before the DB is touched, with exit 2
  and the same message.
- **Console routes** (`adjudication_api.py` `_export_transcript` and all six
  `export.*` routes):
  - each takes `repeats`;
  - the console keeps its existing order: run 404 first, then option 422;
  - the translation conflict is checked before translation lookup.
- **`/api/v1/runs/{run}/transcript`:** takes `repeats`, keeping its existing
  order (option 422 before the run lookup) and its error envelope.

### Alternatives considered

1. **Token-list rewrite (split, compare, rejoin).** Loses piece ownership and
   normalises untouched whitespace, so rejected for the byte guarantee.
2. **Folding repeats into `fillers.py` behind a flag.** The issue requires a
   separate option that is never part of `fillers=drop`.
3. **Removing the first copy.** It carries the sentence-start capital and the
   earliest timing.
4. **Relocating trailing punctuation onto the survivor** (`it's the the.`
   becomes `it's the.`). glm proposed it. Rejected for this first cut in
   favour of blocking (R1), which avoids the orphaned-punctuation and
   timestamp failures codex reproduced. It can be revisited with #757.
5. **Matching across segments, gated only on temporal continuity.** codex's
   alternative to R6. Rejected as more machinery for a rare case: whisper
   segments run 25 to 50 s, so a stutter rarely straddles one.

## Acceptance criteria

### Requirement: Repeat removal is opt-in and separate from fillers
The system SHALL leave Markdown turns exports unchanged unless `repeats=drop`
(HTTP) or `--drop-repeats` (CLI) is given, and SHALL never remove repeats as
part of `fillers=drop`.

#### Scenario: option absent
- WHEN a Markdown turns export is requested without the repeats option
- THEN the bytes equal the fixed pre-change golden for that run and options

#### Scenario: keep and empty mean absent
- WHEN `repeats=keep` or `repeats=` is given
- THEN the bytes equal the export without the option

#### Scenario: fillers only
- GIVEN a turn `the the um cat`
- WHEN exported with `fillers=drop` only
- THEN the text reads `the the cat`

#### Scenario: no candidate
- GIVEN a run whose turns contain no candidate, or only declined candidates
- WHEN exported with `repeats=drop`, with and without timestamps
- THEN the bytes equal the export without the option

### Requirement: Narrow automatic repeat rules
The system SHALL remove the second copy of an immediately repeated one- or
two-word run of eligible function words, separated only by whitespace, inside
one speaker turn and one segment, in one pass.

#### Scenario: one word
- WHEN a turn reads `go to the the store`
- THEN the export reads `go to the store`

#### Scenario: two words
- WHEN a turn reads `we were we were going`
- THEN the export reads `we were going`

#### Scenario: sentence-start case difference
- WHEN a turn reads `The the cat sat.`
- THEN the export reads `The cat sat.`

#### Scenario: adjacent independent repeats
- WHEN a turn reads `we we were were going`
- THEN the export reads `we were going`

#### Scenario: contractions and curly apostrophes
- WHEN a turn reads `It’s it's fine` or `I'm I'm here`
- THEN the export reads `It’s fine` or `I'm here`

#### Scenario: turn made of one repeat
- WHEN a turn reads only `the the`
- THEN the export reads `the`

#### Scenario: separators survive every piece shape
- GIVEN word-level pieces `["we ", "we ", "go"]`, `[" we", " we", " go"]`,
  `["we", " we", " go"]`, a coarse piece `we  we go`, or a mixed coarse and
  word-level turn
- WHEN exported with `repeats=drop`
- THEN the text reads `we go`, never `wego` or a double space beside the
  removal, and whitespace away from the removal keeps its original bytes

#### Scenario: after filler removal
- WHEN a turn reads `the um the cat` with `fillers=drop` and `repeats=drop`
- THEN the export reads `the cat`

### Requirement: Repeat removal declines every guarded case
The system SHALL keep the text unchanged for:
- punctuation or symbols attached to either copy;
- three or more copies, and overlapping candidates;
- quoted text;
- a repeat spanning a segment or an original speaker turn;
- a removal that would open a 3-second gap;
- content words, the one-word exclusions, and non-sentence-start case
  differences.

#### Scenario: punctuation
- WHEN a turn reads `No, no, no`, `We were, we were going`, `the. The`,
  `the-the`, `the the.`, `the the2`, or `(the the)`
- THEN the text is unchanged

#### Scenario: runs and overlaps
- WHEN a turn reads `I I I think`, `no no no no`, `we were we were we were`,
  `we were we were we`, `that that that that`, or `we were we were were`
- THEN the text is unchanged

#### Scenario: quotes
- WHEN a turn reads `he said "the the" twice`, `he said “use the the label”`,
  `he said 'use the the label'`, `the "the" store`, or `he said "the the`
  (unbalanced)
- THEN the text is unchanged

#### Scenario: boundaries
- GIVEN a repeat whose copies fall in two segments of one turn, a repeat
  across a speaker change, or A `the` / B `um` / A `the cat` with both options
- THEN both copies stay

#### Scenario: pause guard
- GIVEN timed pieces `we` 0.0 to 1.0 s, `we` 3.0 to 3.5 s, `go` 4.0 to 5.0 s
- THEN the text is unchanged and the paragraphing equals the default export

#### Scenario: words that stay
- WHEN a turn reads `going going gone`, `she had had enough`, `I know that
  that is true`, `what it was was a joke`, `I gave her her keys`, `the will
  will be read`, `no no`, `so so`, `you know you know`, `we went we went`,
  or mid-sentence `in May may be`
- THEN the text is unchanged

### Requirement: One contract across CLI, console and /api/v1
The system SHALL produce byte-identical successful output for the same run
and options from `voxint export --format md --drop-repeats`, the console
`/review/{run}/export.md?repeats=drop`, and
`/api/v1/runs/{run}/transcript?format=md&repeats=drop`. It SHALL reject
invalid repeats options with each surface's existing error contract.

#### Scenario: three-surface parity
- WHEN the three surfaces export one run with `repeats=drop`, alone and with
  `fillers=drop`, with and without timestamps
- THEN the three byte strings are identical and match the golden

#### Scenario: refusal matrix
- WHEN `repeats` (keep or drop) is given with txt, srt, vtt, json, rttm or
  `style=blocks`, or an unknown value, or (console only) with a translation
  `lang`
- THEN the console and `/api/v1` return 422 with the documented message in
  their own envelopes
- AND the CLI exits 2 with the same message before opening the database
- AND a missing run keeps today's console 404 and `/api/v1` 422 precedence

### Requirement: Stored data never changes
The system SHALL NOT modify stored transcript text, words, corrections or
review state when exporting with `repeats=drop`.

#### Scenario: stored data intact
- WHEN a repeats export completes
- THEN a fresh session reads the same segment `raw_text`, `words`, review
  states and corrections as before the export

## Affected files / components

- `src/voxint/export/repeats.py` (new): rules R1 to R9, word lists and
  exclusions, `drop_repeats`.
- `src/voxint/export/fillers.py`: a `drop_fillers_with_seams` variant that
  also reports, per output turn, the piece indexes where F5 joined two input
  turns. `drop_fillers` is defined through it, so its bytes cannot change.
  (Implementation note: a clean-then-merge split would have changed filler
  output, because F5 cleans the merged turn as one.)
- `src/voxint/export/service.py`: `parse_repeats`, the `drop_repeats` param,
  composition order and validation precedence.
- `src/voxint/cli.py`: the `--drop-repeats` flag, parsed before the DB is
  touched.
- `src/voxint/api/routers/adjudication_api.py`: `repeats` on
  `_export_transcript` and the six export routes.
- `src/voxint/api/routers/api_v1/transcript.py`: the `repeats` param.
- `tests/unit/test_repeats.py` (new): rule goldens, the decline matrix, piece
  shapes, metadata retention, the pause guard through `layout_turns` and
  `to_markdown_turns`, and no-op reuse.
- `tests/unit/test_fillers.py`: unchanged assertions, plus coverage of the
  split steps.
- `tests/unit/test_export_service.py`: `parse_repeats`, precedence and
  composition order.
- `tests/integration/test_turn_exports.py`:
  - fixed pre-change golden for the default output;
  - repeats-only and combined three-surface goldens;
  - the refusal matrix;
  - stored-state checks from a fresh session.
- `docs/operations.md`: the `--drop-repeats` flag, `?repeats=drop`, and a
  "Repeats" rule entry beside "Fillers" with the word lists and the recall
  cost.
- `CHANGELOG.md` `[Unreleased]` Added.

## Implementation slices

1. **Rule engine with goldens.** `repeats.py` plus `tests/unit/test_repeats.py`.
   Covers every rule and decline case on synthetic turns (word-level, coarse,
   mixed, segment and pause boundaries), verified through `layout_turns` and
   `to_markdown_turns`. Not wired anywhere yet; the tree stays green.
2. **Fillers seams.** Add `drop_fillers_with_seams`, with existing filler
   tests unchanged and green. Add the interruption case: A `the` / B `um` /
   A `the cat` keeps both copies.
3. **Wire all three surfaces.**
   - `parse_repeats`, `render_run_transcript`, the CLI flag, the console
     routes and `/api/v1`.
   - A fixed default golden captured before wiring, then the three-surface
     goldens, the refusal matrix and the fresh-session stored-state check.
   - A maintainer spot check: run `drop_repeats` over two or three real local
     runs and read the diff. The results stay in internal notes, never in the
     repo.
4. **Docs and changelog.** `docs/operations.md` and CHANGELOG, in the
   `voxint-docs` house style.

## Testing strategy

- **Unit goldens:** one or more per rule (R1 to R9) and per decline case in
  the acceptance list, on coarse, word-level and mixed piece shapes.
  Whitespace variants: double space, tab, newline, trailing-space ownership,
  synthetic segment separator.
- **Rendered checks, not just dataclass fields:** the pause guard,
  minute-marker placement and paragraph starts are asserted on
  `to_markdown_turns(layout_turns(...))` output with timestamps on.
- **Metadata:** surviving pieces keep `start_seconds`, `end_seconds`, `timed`,
  `segment_start`, `rule` and `mapping`. A word-level second copy disappears as
  a whole piece.
- **No-op:** candidate-free and declined-only turns yield byte-identical
  renders and reuse the input piece tuples.
- **Integration:**
  - default output against a fixed pre-change golden, not against `keep`;
  - three-surface byte parity for repeats-only and repeats plus fillers, with
    and without timestamps, across raw, enhanced and corrected text variants;
  - the refusal matrix with each surface's precedence;
  - stored-state reread through a fresh session.
- **Gates:**
  - ruff, mypy, the full unit and contract suite, and integration on the local
    test DB;
  - CI `lint-test`, `coverage`, `secrets-scan` and `frontend`;
  - review: **High** tier (export contract);
  - no browser lane, since no island or template changes.

## Rollout, risks, open questions

**Risks**
- **Grammatical doubles outside the exclusion list:** mitigated by the narrow
  list and the decline-first rules; new doubles are added to the list.
- **ASR punctuation:** whisper punctuating restart pauses lowers recall
  (documented, accepted).
- **Quote heuristics:** these fail toward over-protection.

**Decisions (maintainer, 2026-10-03, recorded on #755)**
1. **A repeat across a filler-only interruption** (A `the` / B `um` /
   A `the cat`): both copies stay. The original speaker turn is the unit;
   filler removal reports its merge seams and repeats never match across one.
2. **Language:** apply regardless of the run's language, documented as built
   on English word lists, matching fillers. #756 owns language-aware lists.
3. **`no no` and `so so`:** excluded from one-word removal.
4. **Console export menu:** no entry. URL parameter only, matching fillers.

## Review notes

High-tier consult, 2026-10-03. Seats: codex (clink planner), z-ai/glm-5.3,
qwen/qwen3.8-max-prime. 3 of 3 answered.

| Theme | Raised by | Resolution |
|---|---|---|
| Removing `<ws> copy` and dropping whitespace-only pieces can glue words (`wego`) or leave double spaces; the segment-boundary synthetic space can be owned by the dropped piece | codex (reproduced), qwen; 2 | Accepted: R8 adds separator collapse and re-insertion owned by the next survivor (the F3 technique), plus piece-shape goldens |
| Allowing trailing punctuation leaves a punctuation-only piece with the deleted word's timing (`the .`, timestamped `.` paragraph) | codex (reproduced), glm, qwen; 3 | Accepted, blocking variant (codex): R1 blocks any punctuation attached to either copy. glm's relocation variant recorded as Alternative 4 |
| One-copy lookahead lets `I I I` lose a suffix pair; four equal words read as a two-word double; R3 can bypass the exclusions | codex, qwen; 2 | Accepted: R2 maximal-run-of-two, R3 needs A ≠ B and non-extension, R4 run ineligibility, R5 overlap cancel |
| Exclusion set misses `her her`, `was was`, `did did`; `can` and `will` have content readings | glm, codex; 2 (glm on `her`/`was`/`did`, codex on `her`/`can`/`will`) | Accepted: all added to the exclusions; `can`/`will` dropped from the list entirely |
| Partial contraction paradigm creates "why this one but not that" | glm; 1 | Accepted: complete generated paradigm |
| Interrogatives `what who how where why` absent | glm; 1 | Partly: `what`, `who` (pronouns) added; `how where why` (adverbs) listed as deliberately absent |
| Fillers merge (F5) lets repeats cross an original interruption | codex (high), qwen (asked for a decision), glm (wants it caught); split | Maintainer: do not cross (decision 1). Fillers report merge seams (`drop_fillers_with_seams`) |
| Removing a piece changes paragraphing, timestamps and minute markers; layout must be asserted on rendered output | codex (reproduced), qwen; 2 | Accepted: R6 within-segment plus 3 s pause guard; rendered-output goldens. Codex's temporal-continuity cross-segment option rejected as Alternative 5 |
| The "identity fast path like fillers" claim is wrong (`drop_fillers` rebuilds turns) | codex, qwen; 2 | Accepted: R9 now states byte identity, plus reuse of unchanged turns and pieces |
| Refusal matrix overstated equivalence; console 404-first versus API 422-first; precedence with both options and translation; `keep` validated | codex, qwen; 2 | Accepted: requirement rewritten to keep each surface's contract; precedence defined |
| Tokenisation and normalisation underspecified (curly apostrophe keys, casefold, `US us` / `May may`, digits and hyphens, whitespace definition) | codex, qwen; 2 | Accepted: R1 rewritten; case-difference only at sentence start; decline matrix extended |
| Quote handling: single quotes, unmatched closing marks, closing quote after a copy, post-filler state | codex, glm, qwen; 3 | Accepted: R7 defines double and single spans; attached quotes block via R1; computed on post-filler text |
| Add pins: comma two-word restart, `you know you know`, `we went we went`, `no no`, unbalanced quote, a turn of only `the the`, seam cases | glm, qwen; 2 | Accepted into the scenarios |
| Real-run spot check, since synthetic goldens can't show whether whisper's punctuation starves the two-word rule | glm; 1 | Accepted: slice 3, results in internal notes only |
| Fixed pre-change goldens; `keep` versus absent proves nothing on its own; fresh-session stored-state check | codex; 1 | Accepted into the testing strategy |
| Rule numbering gap (R7 missing) and R4's inline question mark | glm; 1 | Fixed: rules renumbered R1 to R9, decisions stated |
| Draft question: runtime report | all 3: no | No runtime report |
| Draft question: language | glm, qwen: apply regardless; codex: reject known non-English | Maintainer: apply regardless (decision 2) |
| Draft question: `no`, `so` | codex, qwen: exclude; glm: keep | Maintainer: exclude (decision 3) |
| Draft question: menu entry | glm, qwen: no; codex: minimal entry | Maintainer: no entry (decision 4) |

## Completion notes

Closed 2026-10-03 against `main` at `7e85c1df` (PR #780).

- **Slices:** all four landed. Slice 1 and 2 in `cd80b07c`, slices 3 and 4 in
  `fc8bf3dc`, review fixes in `c70ae637` and `91c4a9b7`. The slice 3 spot check
  over real local runs is recorded in internal notes: 20 turns changed, 36 words
  removed, no false positives.
- **Scenarios:** every acceptance scenario has a test.
  - Rule, decline, piece-shape, boundary, pause-guard and composition scenarios:
    `tests/unit/test_repeats.py`.
  - Three-surface parity, the fixed default golden, the refusal matrix with each
    surface's precedence, text variants, declined-only byte identity, the
    Alex / Sam / Alex seam and the stored-data rereads:
    `tests/integration/test_turn_exports.py`.
  - The "fillers only" scenario is pinned by the integration golden
    `("drop", None)`, where `the the` survives `fillers=drop`.
- **Checks run at close:** the four covering test files, 300 passed;
  `repeats.py` and `fillers.py` at 100% line coverage; ruff and mypy clean.
- **Design decisions:** all four maintainer decisions are in the code.
  Rejected alternatives 1 to 5 were not reintroduced.
- **Drift:** none in behaviour. One structural note: fillers clean and merge in
  one call (`drop_fillers_with_seams`), then repeats run with the merge seams,
  rather than the clean, repeats, merge order R9 first described. The
  Affected files note explains why, and the observable result matches R9.
- **Spec files touched:** none. This project declares no living spec.
- **Follow-ups:** #754 should carry both options into read mode and plain text,
  and #753 owns the runtime report for both. A console export-menu entry stays
  deferred (decision 4).

