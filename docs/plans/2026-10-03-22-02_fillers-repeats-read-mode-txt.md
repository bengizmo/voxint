# Plan: fillers and repeats in read mode and the plain-text export (#754)

Status: done

Spec deltas: none (this project declares no living spec).

## Goal

Epic #752 (transcript clean-up closer to Descript) has two render-time filters,
both limited to the Markdown turns export today:

- `fillers=drop` / `--drop-fillers` (#741 slice 4, `src/voxint/export/fillers.py`,
  rules F1 to F6);
- `repeats=drop` / `--drop-repeats` (#755, `src/voxint/export/repeats.py`, rules
  R1 to R9).

#754 carries both into two more surfaces, through the same turn pipeline, so no
two surfaces can disagree about which words were left out:

1. **Read mode** (`GET /runs/{id}/transcript?read=1`). This server-rendered,
   JavaScript-free page already shows the Markdown turns paragraphs. It gains
   `fillers` and `repeats` query parameters, a toggle link for each, and a
   one-line note saying how many words were left out. On that page, the export
   menu's existing reading-copy links carry the active filters, so the operator
   can download what they see.
2. **The plain-text export**, on the CLI, the console
   (`/review/{run}/export.txt`) and `/api/v1/runs/{run}/transcript?format=txt`.
   It gains a paragraph layout, `style=turns`, which takes both filters. The
   default txt bytes never change.

Subtitles (srt, vtt), json, rttm and translations keep rejecting both filters.

**Maintainer decisions (2026-10-03).**
- On #754: #754 goes before #753 and starts on the fixed filler list; both
  filters are in scope.
- At plan review:
  - A filter on txt needs an explicit `style=turns`. A filter never changes the
    txt layout on its own.
  - On the read page, the export menu's existing reading-copy links carry the
    active filters. No new menu entries.
  - The 422 text "X applies to the md turns style only" is reworded, because it
    would no longer be true. This amends #754's "same message as today"
    acceptance line and is recorded on the issue.

**Decisions from #755 that still bind:**
- filters apply whatever the language, on English lists;
- stored data never changes;
- repeats never cross a filler seam.

## Ground truth (surveyed 2026-10-03, main `1e427a76`)

- **`src/voxint/export/service.py`**
  - `parse_style(raw, fmt)` is md only. It defaults to `turns` for md and
    `None` otherwise. Any style on another format gets 422 "style applies to
    the md format only".
  - `_parse_turn_filter(name, raw, fmt, style)` checks the layout first ("X
    applies to the md turns style only") and the value second ("unknown X value
    'v'; valid: keep, drop").
  - `render_run_transcript` runs md turns as `attributed_turns` (or
    `translated_turns`) → `drop_fillers_with_seams` → `drop_repeats(turns,
    seams)` → `layout_turns` → `to_markdown_turns`. Every other format goes
    `attributed_transcript` → `render_transcript`.
  - A translation conflict is raised only when a filter is `drop`. `keep` plus
    `lang` returns 200, which `test_repeats_refusal_matrix` pins.
- **`src/voxint/export/__init__.py`**
  - `to_txt` writes one line per transcript **segment**,
    `[{start:9.2f} {end:9.2f}] {speaker}: {text}`, with no header. Speakers are
    attributed per segment.
  - `to_markdown_turns` writes `[HH:MM:SS] **Speaker:** text` paragraphs.
    Continuations omit the speaker, minute markers sit inline, and there is a
    `# title` header and Markdown escaping.
  - The module docstring says every transport renders through
    `render_transcript`, which has not been literally true since md turns.
- **Turn attribution** (`attributed_turns`) is per word, so it can split a
  segment between speakers. Turn layout and segment layout can therefore
  attribute text differently.
- **`src/voxint/api/routers/legacy_runs.py`, `run_transcript`**
  - A request without `read` redirects to the media editor, and other query
    parameters are ignored.
  - In read mode, `text` is parsed first (422 when invalid), then rows are built
    from `layout_turns(attributed_turns(...))`.
- **`src/voxint/api/templates/legacy_runs/transcript.html`**
  - Variant tabs and the "Reading view options" nav build hrefs by hand with
    literal `&` (pinned by `test_transcript_read_mode_preserves_query_in_toggles`).
  - An empty `read_rows` shows "No transcript segments for this run."
  - The page includes `export_menu(run, text, translation_ctx.fresh)`.
- **`src/voxint/api/templates/fragments/export_menu.html`**
  - The menu is shared with `editor/detail.html`.
  - Primary links use `text=corrected`. "Read on screen" carries the current
    variant. "Other text variants" holds enhanced and raw copies, and the
    translated links use `lang`.
- **`src/voxint/cli.py`**: `--style` has `choices=["turns","blocks"]` and its
  help says md only. The `--drop-fillers` and `--drop-repeats` help also says md
  only.
- **Browser lane**: `tools/e2e_browser_lifecycle.py` seeds fixtures from text
  with faithful word timings (`review`, `editor`, `benchmark`, `rail`,
  `voices`). `tests/unit/test_e2e_browser_lifecycle.py:290` pins
  `FIXTURE_CHOICES`, and the seed depends on segment 0's correction rule firing.

## Assumptions and constraints

- Render-time only: no DB change, migration, dependency, setting or numerics
  impact.
- The fixed filler list stays. #753 owns presets, operator lists and an export
  "what was removed" report. The read-mode note is the only runtime count here.
- Read mode never shows a translation, so no translation conflict can arise
  there.

## Proposed approach

### A. One shared turn pipeline with honest counts

New pure module `src/voxint/export/turn_filters.py`:

```python
@dataclass(frozen=True)
class FilteredTurns:
    turns: list[SpeakerTurn]
    fillers_removed: int
    repeats_removed: int

def apply_turn_filters(turns, *, drop_fillers: bool, drop_repeats: bool) -> FilteredTurns
```

- **Composition** is today's: `drop_fillers_with_seams` → `drop_repeats(turns,
  seams)`. With neither flag, the input turns come back unchanged and both counts
  are zero.
- **Counts are word tokens: whitespace-separated tokens holding at least one
  letter or digit.**
  - `fillers_removed` is the count before the filler step minus the count after
    it. `repeats_removed` is the count after fillers minus the count after
    repeats, so it is measured on filler-cleaned text by design.
  - A raw whitespace-token diff overcounts. Codex reproduced `um ...` reporting
    2 and `um , !` reporting 3, because F5 drops punctuation-only remnants.
  - The word-token diff is exact: neither filter ever creates a token or joins
    two surviving tokens, since F3, R8 and `_join_at_seam` each leave a separator.
  - A filler-only turn dropped by F5 counts its filler. A seam space counts
    nothing.
  - Counts are never clamped. A negative count would be a bug, and a unit test
    asserts it cannot happen.
- **Parser split.** `parse_filter_value(name, raw) -> bool` checks only the value:
  `None`, `""` and `keep` mean off, `drop` means on, and anything else gets
  "unknown X value 'v'; valid: keep, drop". `_parse_turn_filter` keeps its
  layout-first order and calls it. Read mode calls `parse_filter_value` directly,
  so both surfaces share the error strings.
- `render_run_transcript` (md turns and txt turns) and read mode all call
  `apply_turn_filters`.

### B. A paragraph layout for txt: `style=turns`

- `parse_style` accepts `style=turns` on txt. No style keeps today's per-segment
  lines, so default bytes cannot change.
  - txt accepts **only** `turns`. Calling the default layout `blocks` would be
    false for txt (qwen).
  - `MarkdownStyle` keeps its name; its docstring says txt uses `TURNS` only. A
    rename would churn every test for no behaviour change.
- **New renderer `to_txt_turns(paragraphs, *, timestamps)`** in
  `export/__init__.py`, the same shape as `to_markdown_turns` minus Markdown:
  - one paragraph per physical line, a blank line between paragraphs;
  - `[HH:MM:SS] Speaker: text`, with continuations omitting the speaker;
  - with timestamps on, inline `[HH:MM:SS]` minute markers before a run;
  - line breaks inside a speaker name or a run collapse to single spaces, using
    the md normaliser; no escaping;
  - **no header** (txt never had one; all three seats agreed);
  - a trailing newline only when there is a paragraph, so an all-filtered
    export is an empty file.
- **`render_run_transcript`** sends md turns and txt turns through one branch:
  turns → `apply_turn_filters` → `layout_turns` → the format's renderer.
  - txt turns plus a console translation (`lang`, no filters) therefore renders
    the translated turns as plain text, like md turns does. This is a deliberate
    new capability, named and golden-tested.
- **Filters** are accepted on md turns and txt turns. A filter on txt without
  `style=turns` gets a 422, so a filter never silently changes the txt layout or
  its speaker attribution.
- **Error copy** (final strings; the tests pin them):
  - style on srt, vtt, json or rttm: `style applies to the md and txt formats only`
  - txt style other than turns: `unknown style 'v' for txt; valid: turns`
  - md style unknown: unchanged, `unknown style 'v'; valid: turns, blocks`
  - filter outside a turn layout: `fillers applies to md turns and txt turns only`
    (the same for `repeats`)
  - unknown filter value, translation conflict: unchanged.
- **Unchanged behaviour:**
  - `keep` with a translation stays allowed (today's behaviour);
  - the console checks run 404 first and `/api/v1` checks option 422 first;
  - the CLI validates before touching the DB (exit 2).
- **CLI:** `--style` choices stay `turns|blocks`, with validation in
  `parse_style`. The help text for `--style`, `--drop-fillers` and
  `--drop-repeats` changes to name txt turns.
- **Docs:** a txt export with `style=turns` attributes speakers word by word and
  carries paragraph start clocks, not segment start and end times. Diffing it
  against the default txt shows layout and attribution changes on top of any
  removed words.
- Update the stale `export/__init__.py` docstring: line formats go through
  `render_transcript`, turn layouts through the service's turn branch.

### C. Read mode

- **Parameters.** `run_transcript` gains `fillers: str | None` and
  `repeats: str | None`.
  - Precedence stays: run 404 first, then the non-read redirect (filters
    ignored, like `timestamps`), then the `text` 422, then a filter 422 from
    `parse_filter_value`.
  - The 422 is rendered the same way as today's invalid `text`.
- **Rows** come from `layout_turns(apply_turn_filters(attributed_turns(...)).turns)`.
- **Links.** Every read-mode link carries the other current parameters: the
  variant tabs, the timestamps toggle and both filter toggles.
  - The canonical query is `text, read=1, timestamps`, plus `fillers=drop` and
    `repeats=drop` only when they are on. `keep` is canonicalised away.
  - "Exit reading view" drops all of them.
  - The suffix is built in the template from fixed tokens, keeping the existing
    literal-`&` href convention. Tests pin exact hrefs in all four filter
    states.
- **Toggles** sit in the "Reading view options" nav as plain anchors:
  - "Remove filler words" or "Show filler words";
  - "Remove repeated words" or "Show repeated words".

  They mirror "Hide/Show timestamps".
- **Note.** When at least one filter is on, the page shows one muted line naming
  only the active filters, with singular and plural forms and the same verb at
  zero:
  - "Left out 4 filler words and 1 repeated word. The saved transcript is
    unchanged."
  - "Left out no filler words. The saved transcript is unchanged."
- **Everything filtered.**
  - The rule: `read_rows` is empty **and** `fillers_removed + repeats_removed >
    0`. A count above zero proves there were words before filtering, and with
    zero removals the filters were a no-op, so empty rows mean an empty run.
  - The page then shows "The filters left out every word. The saved transcript
    is unchanged." in place of both the note and "No transcript segments for
    this run."
  - An empty run with filters on keeps the existing message and shows no note.
- **Export menu (download what you see).** `export_menu` gains an optional
  `filters` argument; the editor page passes nothing and is unchanged. When the
  read page has a filter on:
  - **md links carry the active filters:** primary and variant, reading copy and
    with times.
  - **Reading-copy txt links carry `style=turns` plus the filters:** primary and
    variants.
  - **"Read on screen"** keeps the filters too.
  - **Left unchanged:** timed txt (its label promises start and end times),
    srt, vtt, json, rttm and the translated links.
  - **One added line** in the dropdown: "Reading copies leave out the same words
    as this page."
  - The primary links stay on reviewed text as they are today; this plan does
    not change the menu's variant behaviour.

### Alternatives considered

1. **A filter on txt implies the turn layout** (codex). Less friction and closer
   to the issue's wording, but a diff would show layout and attribution changes
   beyond the removed words. The maintainer chose explicit `style=turns`.
2. **Clean each txt segment line separately.** This keeps the txt shape, but F5
   merges and the R6 guards work on turns, so txt and md would disagree on
   filler-only turns and cross-segment repeats. Rejected.
3. **A client-side toggle in read mode.** It needs JavaScript on a page that is
   deliberately JavaScript-free. Rejected.
4. **Leave the export menu unchanged** (qwen). Then the filtered text could not
   be downloaded from the console at all. The maintainer chose carrying.
5. **Count actual filter matches inside the filter modules** (codex's first
   option). This is exact but needs instrumentation through both F-passes. The
   word-token diff gives the same number with no change to the filters.

## Acceptance criteria

### Requirement: Read mode applies the same filters as the Markdown export
The system SHALL render read-mode paragraphs through the same filter composition
and layout as the Markdown turns export when `fillers=drop` or `repeats=drop` is
given.

#### Scenario: response-level paragraph parity
- GIVEN an escaping-free run with fillers, a one-word repeat, a filler-only turn
  between two turns of one speaker, and a word-timed speaker change inside one
  segment
- WHEN read mode and the Markdown export are requested with the same text
  variant, timestamps flag and filters (each alone, and both)
- THEN the paragraphs parsed from both responses (speaker or continuation, clock,
  markers, text) are identical and equal a small fixed expectation

#### Scenario: defaults unchanged
- WHEN read mode is requested with no filters, or with `keep` or empty values
- THEN the reading paragraphs equal today's for that run and no note is shown

#### Scenario: toggles keep state
- WHEN read mode has any combination of the two filters on
- THEN the variant tabs, the timestamps toggle and the other filter's toggle keep
  the current filters
- AND each filter's own toggle flips only that filter
- AND "Exit reading view" carries none of them (exact hrefs)

#### Scenario: removed-words note
- WHEN at least one filter is on
- THEN one line names only the active filters with their word counts (zero and
  singular forms included) and says the saved transcript is unchanged

#### Scenario: everything filtered versus empty run
- GIVEN a run whose only words are fillers
- WHEN read mode drops fillers
- THEN the page says the filters left out every word and that the saved transcript is
  unchanged
- AND a run with no segments, with filters on, keeps "No transcript segments for
  this run." with no note

#### Scenario: invalid value
- WHEN read mode gets `fillers=bogus` or `repeats=bogus`
- THEN the response is 422 with "unknown fillers value 'bogus'; valid: keep,
  drop" (or the repeats form)

#### Scenario: stored data intact
- WHEN read mode renders with both filters
- THEN a fresh session reads the same segment text, words, review states and
  corrections

### Requirement: The read page's reading-copy downloads match the page
The system SHALL add the read page's active filters to the export menu's md
links and reading-copy txt links, and to "Read on screen", and SHALL leave all
other menu links unchanged.

#### Scenario: menu hrefs
- WHEN the read page has `fillers=drop` (and, separately, both filters) on
- THEN the md links carry the filters
- AND the reading-copy txt links carry `style=turns` plus the filters
- AND the timed txt, srt, vtt, json, rttm and translated hrefs equal today's
- AND a one-line note says reading copies leave out the same words as the page

#### Scenario: download what you see
- WHEN the md reading-copy link from a filtered read page is fetched
- THEN its paragraphs equal the page's paragraphs

#### Scenario: other pages untouched
- WHEN the editor page, or the read page with no filters, renders the menu
- THEN the hrefs equal today's

### Requirement: A paragraph layout for plain text that takes the filters
The system SHALL offer `style=turns` on the txt export, rendering the reading
paragraphs as plain text with no header, and SHALL accept `fillers` and
`repeats` on it.

#### Scenario: default txt unchanged
- WHEN txt is exported without `style`, with and without timestamps
- THEN the bytes equal a fixed pre-change golden

#### Scenario: txt turns golden and md equivalence
- WHEN an escaping-free run is exported as txt turns and as md turns with the
  same filters and timestamps flag
- THEN the txt bytes equal a fixed golden
- AND each txt paragraph line equals the md paragraph line with `**` and the
  header removed

#### Scenario: three-surface parity
- WHEN the CLI, the console and `/api/v1` export txt turns with each filter alone
  and both, with and without timestamps, across raw, enhanced and corrected text
- THEN the three byte strings are identical and match the goldens

#### Scenario: translated txt turns
- WHEN the console exports `txt?style=turns&lang=es` for a fresh translation
- THEN the translated turns render as plain-text paragraphs (golden)

#### Scenario: all filtered
- WHEN every word of a run is a filler and txt turns drops fillers
- THEN the export is empty (zero bytes)

### Requirement: Filters stay out of timed and structural formats
The system SHALL reject `fillers` and `repeats` (keep or drop) on srt, vtt, json,
rttm, txt without `style=turns`, and md `style=blocks`. It SHALL reject `drop`
with a console translation and keep allowing `keep` with one.

#### Scenario: refusal matrix
- WHEN a filter is sent with srt, vtt, json, rttm, txt (no style) or md
  `style=blocks`
- THEN the console and `/api/v1` return 422 in their own envelopes with
  "fillers applies to md turns and txt turns only" (or the repeats form)
- AND the CLI exits 2 with the same message before opening the database
- AND a missing run keeps the console's 404 and `/api/v1`'s 422 precedence

#### Scenario: style errors
- WHEN `style=blocks` (or any value other than `turns`) is sent with txt, or any
  style with srt, vtt, json or rttm
- THEN the response or exit carries "unknown style 'blocks' for txt; valid:
  turns" or "style applies to the md and txt formats only"

#### Scenario: translation
- WHEN the console gets `drop` with `lang`
- THEN it returns 422 "X cannot be combined with a translation"
- AND `keep` with `lang` returns 200

## Affected files / components

- `src/voxint/export/turn_filters.py` (new): `apply_turn_filters`,
  `FilteredTurns`, the word-token counter.
- `src/voxint/export/__init__.py`: `to_txt_turns`; the module docstring.
- `src/voxint/export/service.py`: `parse_filter_value`; `parse_style` for txt
  turns; the reworded messages; one turn branch for md and txt;
  `MarkdownStyle` docstring.
- `src/voxint/cli.py`: help text for `--style`, `--drop-fillers` and
  `--drop-repeats`.
- `src/voxint/api/routers/legacy_runs.py`: read-mode parameters, rows through the
  shared pipeline, counts and the everything-filtered flag in the context.
- `src/voxint/api/templates/legacy_runs/transcript.html`: the toggles, link
  suffixes, note and everything-filtered message; passing filters to the menu.
- `src/voxint/api/templates/fragments/export_menu.html`: an optional `filters`
  argument, the carried hrefs and the one-line note.
- `src/voxint/api/routers/adjudication_api.py`, `api_v1/transcript.py`: no
  wiring change expected (they already pass `style`, `fillers` and `repeats`);
  verify.
- `tools/e2e_browser_lifecycle.py`: a `cleanup` fixture (fillers, a repeat, a
  filler-only turn between two turns of one speaker, a repeat across that seam),
  keeping segment 0's correction-rule invariant.
- Tests:
  - `tests/unit/test_turn_filters.py` (new);
  - the export renderer unit tests, for `to_txt_turns`;
  - `tests/unit/test_export_service.py`;
  - `tests/unit/test_e2e_browser_lifecycle.py` (`FIXTURE_CHOICES`);
  - `tests/integration/test_turn_exports.py`: the txt goldens, the refusal
    matrix and the messages, with the fillers and repeats matrices updated for
    the new copy;
  - `tests/integration/test_runs_api.py`: read mode and menu hrefs.
- Docs: `docs/operations.md` (txt `style=turns` and its attribution and timing
  model, filters on txt turns, read-mode toggles, the new error strings); any
  lay-reader doc that describes reading view or the download menu;
  `CHANGELOG.md` `[Unreleased]` Added, plus Changed for the error copy.

## Implementation slices

1. **Shared pipeline, counts and parser split.**
   - `turn_filters.py` and its unit tests (counts, including the
     `um ...` / `um , !` reproductions, F5-dropped turns and the
     repeat-after-filler case; no-op identity; non-negative counts).
   - `parse_filter_value`.
   - The md export switches to `apply_turn_filters`, and every existing md
     golden stays byte-identical.
2. **txt turns.**
   - Capture the fixed txt default golden first.
   - Then `parse_style` for txt, `to_txt_turns`, the merged turn branch, the
     new messages, CLI help, txt turns goldens on three surfaces, translated txt
     turns, the all-filtered empty file, the updated refusal matrices, and the
     stored-data reread.
3. **Read mode.** Parameters and precedence, rows, toggles with exact hrefs,
   the note, everything-filtered versus empty, response-level parity with md,
   the escaping smoke, the 422, and the stored-data reread.
4. **Menu carry, browser lane, docs.**
   - The `export_menu` `filters` argument and its href tests (read page
     filtered and unfiltered, editor page).
   - The `cleanup` fixture.
   - The Playwright lane: all four filter states, state across variant and
     timestamps links, the note and everything-filtered copy, menu hrefs, and
     a fetched md reading copy matching the page. Then reconcile.
   - operations.md, lay docs and CHANGELOG, in the `voxint-docs` house style.

## Testing strategy

- **Unit.**
  - Word-token counts: punctuation remnants, a filler-only turn dropped by F5,
    a seam, a repeat eligible only after filler removal, and the no-op.
  - The count equals the word tokens visibly missing from a before/after text
    pair.
  - `to_txt_turns` goldens: timestamps on and off, continuation, minute marker,
    line breaks in run text and in the speaker name, empty input, `*`, `#` and
    `_` left literal.
- **Integration.**
  - Fixed txt default golden; txt turns goldens on three surfaces and three
    variants; txt and md equivalence on an escaping-free fixture; translated txt
    turns.
  - Read-mode parity built from **parsed responses** (read HTML and the md
    export) against a small fixed expectation. The route's rows are also
    asserted equal to a direct `layout_turns(apply_turn_filters(...))` call made
    in the test, so a dropped flag or reordered filter is caught.
  - A separate hostile-text read-mode fixture (`*`, `#`, `<`, `&`).
  - Exact hrefs for the toggles and menu in every filter state; the note and
    everything-filtered rule; the 422; the refusal matrix with the new messages
    and each surface's precedence; fresh-session stored-state rereads after
    exports and read requests.
- **Gates.**
  - Local: ruff, mypy, the unit and contract suites, and integration on the
    local test DB.
  - CI: `lint-test`, `coverage`, `secrets-scan` and `frontend`.
  - Review: **High** tier (public export contract, error copy, and a console
    page plus a menu shared with the editor).
  - **Browser lane: yes.** The read page and the shared export menu change, and
    #754's acceptance asks for it.

## Rollout, risks, open questions

- **Risk: scripts that match the old 422 text break.** Pre-1.0 and single
  operator; recorded under CHANGELOG Changed and on #754.
- **Risk: the reading-copy txt link changes layout when a filter is on**
  (segment lines become paragraphs). The label still fits, the dropdown note
  explains it, and the layout change is the maintainer's chosen trade-off.
- **Risk: shared-menu regression on the editor page.** Pinned by the
  unchanged-hrefs scenario and the browser lane.
- **Follow-up, out of scope:** the menu's primary links always download reviewed
  text, even from a raw or enhanced read view. That existing gap in
  download-what-you-see is not changed here.

## Review notes

High-tier consult, 2026-10-03. Seats: codex (clink planner), z-ai/glm-5.3,
qwen/qwen3.8-max-prime. 3 of 3 answered. The maintainer settled the three splits
after the panel.

| Theme | Raised by | Resolution |
|---|---|---|
| Explicit `style=turns` on txt versus a filter implying the turn layout | codex: implicit (the issue says txt takes the filters; CLI default txt makes `--drop-fillers` fail); glm, qwen: explicit (a filter must not silently change layout and attribution); split | **Maintainer: explicit.** qwen's refinement accepted: txt takes only `turns`, no `blocks` |
| Read page export menu carrying the active filters | glm (High: otherwise the cleaned text cannot be downloaded from the console), codex: carry into reading-copy links only; qwen: not now; split | **Maintainer: carry** into md and reading-copy txt links plus "Read on screen"; structural, timed and translated links unchanged (codex's scoping) |
| Reworded 422 copy versus #754's "same message as today" | glm, qwen: change it, it becomes untrue; codex: it violates acceptance unless amended | **Maintainer: new copy**, recorded on #754 as an acceptance amendment; exact strings fixed in the plan (glm) |
| Whitespace-token diff overcounts | codex (reproduced `um ...` → 2, `um , !` → 3), qwen; glm thought it honest | Accepted: count tokens holding a letter or digit (qwen); codex's reproductions become unit tests. Codex's match-instrumentation alternative recorded as Alternative 5. qwen's clamp rejected: a negative count is a bug and is tested instead |
| Parity tests were tautological against `layout_turns` | codex, glm, qwen; 3 | Accepted: response-level parity against fixed expectations, a direct-pipeline equality check, escaping-free fixtures, a separate hostile-text fixture |
| Whole-HTML equality impossible once toggles are added | codex, glm; 2 | Accepted: pin paragraphs and the absent note instead |
| Translation refusal criterion wrong (`keep` + `lang` is allowed today; CLI and `/api/v1` take no `lang`) | codex, qwen; 2 | Accepted: `drop` conflicts, `keep` allowed, translation tests console only |
| txt turns plus translation slips in unannounced | glm, qwen; 2 | Accepted as a named capability with a golden |
| Read mode cannot reuse the export parser as is | glm; 1 | Accepted: `parse_filter_value` split |
| Everything-filtered detection unspecified | glm, qwen; 2 | Accepted: empty rows and removals above zero (glm's rule); replaces the note; keeps "saved transcript is unchanged" (glm) |
| Link preservation and HTML 422 underspecified | qwen; 1 | Accepted: canonical query, exact-href tests in all states, 422 like today's invalid `text` |
| Adversarial fixtures: seams, word-timed speaker change, divergent variants, empty versus filtered | codex, qwen; 2 | Accepted into scenarios and the `cleanup` fixture |
| `MarkdownStyle` name misleads once txt uses it | glm, qwen; 2 | Partly: docstring updated, no rename (churn for no behaviour change) |
| Stale `render_transcript` docstring | qwen; 1 | Accepted |
| CLI help for both filter flags still says md only; normalise speaker line breaks in txt turns | codex; 1 | Accepted |
| Browser fixture: `FIXTURE_CHOICES` pin, correction invariant, driver steps | codex, glm (cost), qwen; 3 | Accepted: fixture listed with its test update and invariant; Playwright steps listed in slice 4 |
| txt turns header | all 3: none | No header |
| Toggle and note copy | glm: Remove/Keep and the same verb at zero; qwen: Include instead of Keep, plurals | "Remove / Show" (mirrors Hide/Show timestamps), "Left out no …" at zero, singular and plural forms |
| "Non-read requests redirect to the editor" claim questioned | glm; 1 | Rejected: verified in `run_transcript`, which returns a 302 to `/media/{id}/editor` before any read-mode work |
| Docs must state the attribution and timing model of txt turns | glm, qwen; 2 | Accepted |

## Completion notes

Closed 2026-10-04 against `main` at `c9cf7651` (PR #782).

- **Slices:** all four landed. Slice 1 in `136128c4`, slice 2 in `17b8a5f0`,
  slice 3 in `94b3c25f`, slice 4 with the docs in `7364ba6f`, review fixes in
  `f9936701`.
- **Scenarios:** every acceptance scenario has a test.
  - Counts, composition, seams and the no-op: `tests/unit/test_turn_filters.py`.
  - `to_txt_turns` goldens: `tests/unit/test_export_formatters.py`. Parser and
    error copy: `tests/unit/test_export_service.py`.
  - The fixed default txt golden, txt turns goldens on three surfaces and three
    variants, md equivalence, translated txt turns, the empty all-filtered
    file, the refusal and style-error matrices with each surface's precedence,
    and the stored-data reread: `tests/integration/test_turn_exports.py`.
  - Read-mode parity with the md export, defaults, exact toggle hrefs, the
    note, everything-filtered versus empty, the 422 and its precedence, hostile
    text, the stored-data reread, the menu hrefs, download-what-you-see, and the
    unchanged editor and unfiltered menus (against HTML captured from the
    pre-change template): `tests/integration/test_runs_api.py`.
- **Checks run at close:** CI green on PR #782 (`lint-test`, `coverage`,
  `frontend`, `secrets-scan`); ruff and mypy clean locally; `gitleaks git` clean
  on the branch commits.
- **Review:** High tier, codex, deepseek-v4-pro and qwen/qwen3.8-max-prime, 3 of
  3. Applied fixes and deliberate skips are listed in the PR body.
- **Browser lane:** run on the `cleanup` fixture. All four filter states, state
  across the variant tabs and the timestamps link, the note copy, every menu
  href, and the fetched md and txt reading copies equal to the page. Reconcile
  passed.
- **Deviation:** slice 4 listed the everything-filtered copy for the browser
  lane. The `cleanup` fixture keeps ordinary words in every state, so that
  message was not driven in a browser. It is server-rendered with no script and
  is pinned at response level by `test_read_cleanup_all_filtered_or_empty`.
- **Design decisions:** the three maintainer decisions are in the code
  (explicit `style=turns` on txt, the menu carries the filters, the reworded
  422 copy). Rejected alternatives 1 to 5 were not reintroduced.
- **Left open:** the menu's main links download reviewed text from a raw or
  enhanced read view (the follow-up named above). The docs now say so.
