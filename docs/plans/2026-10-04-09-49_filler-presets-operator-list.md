# Plan: filler presets in tiers plus an operator list (#753)

Status: in-progress

Spec deltas: none (this project declares no living spec). Q1 below reconciles
two acceptance lines on #753. The reconciliation is recorded on the issue, not
in a spec.

## Goal

Epic #752 brings transcript clean-up closer to Descript's "Remove filler
words". Today `fillers=drop` removes a fixed English set, `um uh umm uhh uhm
erm`: the regex `_FILLER` in `src/voxint/export/fillers.py` applies rules F1 to
F6. Three surfaces share one pipeline, `apply_turn_filters` in
`src/voxint/export/turn_filters.py`: md turns, txt turns and read mode. #753
makes the list:

- **Tiered presets, in code, versioned.**
  - Tier 1 applies whenever fillers are dropped. It is exactly today's six.
  - Tier 2 is available but off. It holds `hm mm mmm` and the phrases
    `you know`, `kind of`, `sort of`, `I mean`, `I guess`, `I suppose`,
    `or something`, `you know what I mean` and `you see`.
  - Never offered: `like so right well`.
- **Operator-extendable.** The `app_settings` row holds language-keyed `add` and
  `keep` lists, edited on the console Settings page. `VOXINT_FILLERS_ADD` and
  `VOXINT_FILLERS_KEEP` are an environment fallback, and the row wins. The
  effective list is (tier 1 + add) - keep. One resolver serves the CLI, the
  console download, `/api/v1` and read mode.
- **Honest.**
  - A customised md export reports which list removed how many words.
  - The CLI prints the count to stderr.
  - `voxint fillers show` prints the effective list.
  - `doctor` names where the list came from.

**Binding maintainer decisions.**
- 2026-10-02, #752:
  - stored text never changes;
  - `like`, `so`, `right` and `well` are never removed by a preset rule.
- 2026-10-03, #755 plan:
  - filters apply whatever the run language, on English lists;
  - repeats never cross a filler seam.
- 2026-10-04, comment on #753: `hm mm mmm` are tier 2. Tier 1 stays today's six,
  so default output cannot change.
- 2026-10-04, at plan review (recorded on #753):
  - Q1: the md report comment appears only for a customised list.
  - Q2: phrases use the conservative both-sides rule (F1p) in v1.
  - Q3: tier 2 is a row of checkboxes that write into the add list.
  - An explicit "empty, overriding the environment" state is deferred.

## Ground truth (surveyed 2026-10-04, main `811e663b`)

- **`export/fillers.py`.**
  - `_FILLER` (lines 30 to 34) matches one word: `(?<!\S)`, captured opening
    wrappers (group 1), the word, an optional `(?P<mark>[,.?!;:])` and
    `(?=$|\s)`, then trailing `\s*`.
  - `_clean_indexed` runs it over a char-owned concatenation of the turn's
    pieces. Each char keeps its source piece index, and a synthetic segment
    separator is owned by the right-hand piece.
  - The loop then applies F2 (mark moving, using `core`, the text so far with
    wrappers stripped) and F4 (capitalising).
  - `drop_fillers_with_seams` runs F5:
    - it pre-screens filler-only turns with `_clean_pieces` (line 152);
    - it merges same-identity neighbours;
    - it re-cleans each merged turn and reports seams for repeats.
- **`export/turn_filters.py`.** `apply_turn_filters(turns, *, drop_fillers: bool,
  drop_repeats: bool) -> FilteredTurns`. Counts are word tokens taken from
  `join_pieces(...).split()`, so a removed two-word phrase counts as 2.
- **`export/service.py`.** `render_run_transcript(...) -> str` is the single
  render entry for three callers:
  - the CLI (`cli._export`);
  - the console (`adjudication_api._export_transcript`, which takes no `Request`
    today);
  - `/api/v1` (`api_v1/transcript.py`, which takes no `Request`; the sub-app
    shares `app.state.settings`, see `api/app.py:776`).

  Read mode (`legacy_runs.run_transcript`) calls `apply_turn_filters` directly.
- **Settings row precedents.**
  - `vocabulary` is an `ARRAY(Text)` edited in a textarea. Saves go through
    `normalize_vocabulary`, replace the whole list, and re-render with the
    submitted text on a 422.
  - Nullable "NULL or blank inherits env" strings resolve through
    `app_settings.resolve_effective_*`.
  - JSON columns use `JSON().with_variant(JSONB(), "postgresql")`.
  - The latest migration is `alembic/versions/0070`.
- **Settings console.**
  - General renders `settings/hub.html` when `console_settings_enabled` is on,
    and `settings/settings.html` when it is off.
  - `_settings_page_template` (`settings.py:2427`) maps POST paths to the tab
    that renders validation errors. An unknown path falls through to Plugins.
- **Environment.** `config.Settings` fields map to the uppercase name with no
  prefix. Collection-like values are plain strings that the app parses.
  `.env.example` lines are contract-tested per feature.
- **Goldens.** The existing tests:
  - `test_fillers_three_surface_golden_and_default`
    (`tests/integration/test_turn_exports.py:219`) pins the CLI, console and
    `/api/v1` bytes for md `fillers=drop`;
  - `test_read_cleanup_response_parity` (`tests/integration/test_runs_api.py`)
    covers read mode;
  - `tests/unit/test_fillers.py` covers F1 to F6, including the negatives
    `mm-hmm` and `hmm`.

## Assumptions and constraints

- **English only.** The row stores lists keyed by language (`{"en": [...]}`), so
  #756 can add languages without a migration. #753 reads and edits only `en`
  and applies it to every run. The settings page says the list is English. The
  environment variables are English-only comma lists. They are not
  language-keyed.
- **Not in scope:** a per-recording override, an API for editing the list, and
  stored state for tier 2.
- Q1 to Q3 below are decided (2026-10-04). Reopening any of them invalidates
  the matching part of this plan.

## Proposed approach

### A. Presets and the effective list (`src/voxint/export/filler_lists.py`, new, pure)

- **Constants.**
  - `PRESET_VERSION = "en-1"`.
  - `TIER_1` holds today's six. `TIER_2` holds the twelve entries listed in the
    Goal.
  - `NEVER = ("like", "so", "right", "well")` is used for the UI caution and for
    a contract check that it shares nothing with either tier.
- **`FillerList`** is frozen and holds:
  - `language`;
  - `preset_version`;
  - `words`: single-word entries in their first spelling, sorted by casefold;
  - `phrases`: multi-word entries in their first spelling, sorted longest first;
  - `added` and `kept`: the configured entries, each carrying its source;
  - `kept_without_effect`;
  - `add_source` and `keep_source`, each one of `settings`, `environment` or
    `none`;
  - `env_overridden`: which environment lists the row is overriding.

  `is_default` compares the canonical effective set with the canonical tier 1
  set. It does not compare tuple order.
- **`normalize_entries(lines) -> tuple[str, ...]`.**
  - Each entry is trimmed, with internal whitespace collapsed to one space.
    Identity is casefolded; display keeps the first spelling. Duplicates are
    dropped.
  - Blank lines and empty comma items are skipped.
  - An entry may contain letters, apostrophes (`'` and `’`) inside or at the
    end of a word, and single hyphens between letters. Anything else is
    refused with one plain-language `FillerListError`. That rules out digits,
    other punctuation and a leading or doubled hyphen.
  - Bounds: at most 5 words per entry, at most 40 characters, and at most 100
    entries per list after dedupe.
  - The same function gates the console save, the environment parse and
    `fillers show`.
- **`effective_filler_list(add, keep, ...)`** computes (TIER_1 ∪ add) − keep.
  - `keep` excludes exact entries only. `keep = you know` does not protect
    those words inside an added `you know what I mean`, and the help text says
    so.
  - A keep entry that matches nothing is listed in `kept_without_effect`.
- **`resolve_effective_filler_list(row, settings)`** lives in `app_settings.py`
  beside the other `resolve_effective_*` helpers. It is the only resolver.
  - Each list resolves on its own. A row value (SQL NOT NULL) wins. Otherwise
    the environment string applies, and otherwise the list is empty.
  - The row stores JSON keyed by language and is always assigned whole. That
    keeps unknown language keys and avoids mutating JSON in place.
  - Whichever list a missing `en` key affects resolves as unset.

### B. Matching (`export/fillers.py`)

These rules settle what the panel flagged: span and capture semantics, F5, and
phrase meaning.

- **The tier 1 path is literally today's code.** The single-word pattern is
  generated from `words` inside today's exact scaffold:
  - `(?<!\S)`;
  - the opening-wrapper group;
  - the mark group;
  - `(?=$|\s)`;
  - trailing `\s*`;
  - `IGNORECASE`.

  For the default list, the generated pattern string must equal today's
  `_FILLER.pattern` (an alternation order such as `umm|uhh|uhm|erm|um|uh`
  puts longer words first). A unit test pins this, and a small property test
  compares old and new output on random token soups. A list with no
  single-word entries skips the pass, so an empty alternation is never
  compiled. Patterns are cached with a bounded `lru_cache(maxsize=32)` keyed
  on the entry tuple.
- **Phrases are a separate pass that runs first.** Phrases go through their own
  `_clean_indexed` pass, then the single-word pass runs on what is left. Each
  pass uses the existing F2 to F4 machinery. When there are no phrases, the
  phrase pass does not run, so default and words-only lists keep today's code
  path.
- **Phrase pattern.**
  - The scaffold is the same as the single-word one.
  - Each phrase is `re.escape`d per word, with words joined by `\s+`. A phrase
    can therefore span word pieces and the synthetic segment separator,
    because removal works on char ownership.
  - The pattern is `IGNORECASE`, with alternatives longest first.
- **F1p decides whether a phrase stands alone.** A phrase match is removed only
  if all of the following hold:
  - **Left side.** The text before it, `core` (already computed in the loop),
    is empty (turn start) or ends in a clause mark `, ; : . ? !`. Python's
    lookbehind cannot express this, so it is checked in the loop. A rejected
    match is appended back unchanged.
  - **Right side.** The captured mark is `,`, `;`, `:`, `.` or `!`, or there is
    no mark and the phrase ends the turn.
  - **Not a tag question.** A phrase followed by `?` stays (`He lied, you know?`
    is unchanged).
  - **Not a whole sentence.** A phrase with a sentence boundary on both sides
    stays: turn start or `. ? !` before, and `. !` or turn end after. So
    `Will you come? I guess.` keeps its reply.
  - **Not quoted.** A phrase directly followed by a closing quote stays, as F1
    already does for single words.
  - **Other cases that stay:** a dash or ellipsis before the phrase, or a
    conjunction with no mark (`And you know, we left.`). Q2 is about this
    conservative behaviour.
- **F5 seams.** A phrase never crosses an F5 merge seam.
  - The phrase pass receives each char's source turn and skips a match whose
    chars come from more than one source, the same rule repeats already follow.
  - This keeps "merging never makes a phrase newly eligible". The left context
    after a merge is at least as restrictive as a turn start, and the right
    context is at least as restrictive as a turn end.
  - As a backstop, an output turn left empty by the final pass is dropped, and
    a unit test proves the backstop is unreachable for single-word lists.
- **Threading the list.** `_clean_pieces`, `_clean_indexed`, the F5 pre-screen
  and the merged-turn re-clean all receive the same `FillerList`.
  `drop_fillers` and `drop_fillers_with_seams` take `fillers: FillerList`. It
  is required, so there is no hidden default.
- **Lists that remove nothing.** If keep removes all six and nothing is added,
  fillers are still "on", nothing matches, and the output equals `keep`. The
  count is 0.

### C. Pipeline and transports

- **`apply_turn_filters`** becomes `apply_turn_filters(turns, *, fillers:
  FillerList | None, drop_repeats)`. `None` means fillers are kept. This
  replaces the bool, so no caller can drop fillers without naming a list.
- **The service.**
  - `render_run_transcript_report(...)` returns
    `RenderedTranscript(content, fillers_removed, repeats_removed)`.
  - `render_run_transcript` wraps it and returns `.content`, so there is a
    single attribution and filter pass.
  - It takes `fillers: FillerList | None` in place of `drop_fillers`.
- **Explicit `Settings` threading.** One resolve per operation:
  - The console helper and route wrappers take `Request` and use
    `request.app.state.settings`.
  - `/api/v1` takes `Request` and reads the same shared state.
  - Read mode already has `Request`.
  - The CLI uses the sanitized `get_settings()` path it already relies on.
  - Each transport keeps `parse_fillers` (a bool) for the 422 matrix, so bad
    options are still refused before the DB is touched. It then sets `fillers =
    resolve_effective_filler_list(get_app_settings(session), settings) if drop
    else None`.
- **md report (Q1).** When fillers are on and the list `is_default` is false,
  md turns get exactly one final LF-terminated line:
  `<!-- Filler words left out: 12. Preset en-1; also removed: you know, I mean;
  kept: um. The saved transcript is unchanged. -->`
  - With the default list, md output is unchanged.
  - Entries cannot contain `--`, because the whitelist allows only single inner
    hyphens. The comment therefore cannot be closed early.
  - txt gets no report, because plain text has no comment syntax.
- **CLI.** Whenever fillers are on (md or txt), the CLI writes `Left out K
  filler words (preset en-1, 2 added, 1 kept).` to stderr. When streaming,
  stdout carries only the document. With `-o`, the existing `wrote …` line
  stays on stdout.
- **Read mode.** The note keeps its shape. With a non-default list it adds
  "using your filler list", linking to the settings section. With the default
  list, nothing changes. The all-filtered message is updated in the same way.

### D. Storage, settings page, environment, doctor, CLI

- **Migration 0071** adds `app_settings.fillers_add` and
  `app_settings.fillers_keep`.
  - Both are nullable `JSON().with_variant(JSONB(), "postgresql")` columns
    created with `none_as_null=True`, so Python `None` is SQL NULL.
  - Shape: `{"en": ["you know", ...]}`. NULL inherits the environment.
  - There is no backfill.
  - Upgrade and downgrade are covered by a round-trip test.
- **Console.** A new `settings/_fillers.html`, titled "Filler words", sits on
  the **General** tab in both `hub.html` and `settings.html`, outside the
  existing forms.
  - It is its own form, `POST /settings/fillers`, with `CSRF_SETTINGS`. Its path
    is added to the hub mapping in `_settings_page_template`, so validation
    errors render on General and not on Plugins.
  - **Tier 2 is a row of checkboxes that write into `add`** (Q3). A ticked box
    means the entry is present in the add list. The add textarea shows only
    entries that are not in tier 2, and a save sets add = ticked boxes +
    textarea lines. No new stored state.
  - The keep textarea has one entry per line.
  - The section shows the effective list and flags keep entries that have no
    effect.
  - The caution reads: "Voxint never removes like, so, right or well on its
    own: in 'turn right' they carry meaning. You can still add them."
  - An empty add or keep field saves NULL, which means "use the installation's
    environment list, if any". When a list is inherited, the page shows the
    environment value read-only and says it comes from the environment.
  - A 422 changes nothing in the database and re-renders with the submitted
    text.
- **Environment.** `voxint_fillers_add: str = ""` and `voxint_fillers_keep: str
  = ""` are comma-separated English entries.
  - A field validator checks them through `normalize_entries` and fails closed.
    Its message names the variable and the rule, never the raw value.
  - Both are documented in `.env.example` with the note that an HTTP process
    reads them at start-up and needs a restart after an edit.
- **doctor** adds one advisory `CheckResult("filler list", ...)` that never
  causes a hard failure.
  - Example detail: `preset en-1; additions from settings (2), overriding the
    environment; keeps: none`.
  - If the row cannot be read, it says `settings unavailable; environment only
    (provisional)` and does not claim an effective list.
- **`voxint fillers show`** prints the preset version, tier 1, tier 2 (marking
  which entries are added), add and keep with their sources, and the effective
  list. It reads the DB through the normal engine path. If the DB is
  unreachable it fails like other DB commands. There is no environment-only
  mode.

### Alternatives considered

- **Tier 2 as stored checkbox state, a third list.** Rejected. The checkboxes
  are UI over `add`, which gives the same effect with no extra state.
- **Tier 2 only by typing into `add`.** Rejected (glm). Copying phrases is
  friction, and a typo fails silently.
- **One alternation for words and phrases.** Rejected (codex and qwen). Shared
  boundaries either over-match phrases or change tier 1 bytes, and a variable
  left context cannot be a Python lookbehind.
- **Drop the environment fallback** (glm). Rejected. It is in #753's scope and
  acceptance, for headless installs.
- **An explicit "empty, overriding the environment" state** (codex). Deferred.
  Showing the inherited environment value makes the behaviour visible, and an
  operator who set the environment can unset it. It can be added later without
  changing the storage.
- **Unconditional md comment.** Rejected (3 of 3). It breaks the default golden,
  and it pollutes Markdown that people paste into CMSs.

## Acceptance criteria

### Requirement: Default filler removal is unchanged
The system SHALL remove exactly today's six words, with today's bytes, when no
operator list is configured.

#### Scenario: no list configured
- GIVEN no row lists and no environment lists
- WHEN a run is exported with `fillers=drop` (md turns or txt turns) or read
  with `fillers=drop`
- THEN the bytes equal the existing goldens: no comment, and `hm`, `mm` and
  `mmm` survive

#### Scenario: generated default pattern
- WHEN the default list is compiled
- THEN its single-word pattern string equals today's `_FILLER.pattern`

### Requirement: Added phrases are removed only where they stand alone
The system SHALL remove an added phrase only when it is set off by clause
punctuation on both sides, and never as a tag question or a whole sentence.

#### Scenario: parenthetical phrase
- GIVEN `add` = `you know`
- WHEN `It was, you know, big. You know the rules.` is exported
- THEN it reads `It was, big. You know the rules.` and the filler count is 2

#### Scenario: not standalone
- GIVEN `add` = `you know`
- WHEN `Do you know, sir?` or `And you know, we left.` is exported
- THEN the text is unchanged

#### Scenario: tag question and standalone reply survive
- GIVEN `add` = `you know`, `I guess`
- WHEN `He lied, you know? Will you come? I guess.` is exported
- THEN the text is unchanged

#### Scenario: longest phrase first
- GIVEN `add` = `you know`, `you know what I mean`
- WHEN `Fine, you know what I mean, we left.` is exported
- THEN it reads `Fine, we left.` and the filler count is 5

#### Scenario: phrase across word pieces and a segment boundary
- GIVEN `add` = `you know`, and `you` and `know,` in separate pieces or segments
- WHEN exported
- THEN both are removed, surviving pieces keep their timing and flags, and the
  separators collapse per F3

#### Scenario: phrase does not cross a speaker merge seam
- GIVEN Alex `So, you`, Sam `um`, Alex `know, yes.` and `add` = `you know`
- WHEN exported
- THEN Sam's turn is dropped, Alex's turns merge, and `you know` stays

### Requirement: Keep overrides presets
#### Scenario: keep um
- GIVEN `keep` = `um`
- WHEN `um, uh, yes` is exported
- THEN the text reads `um, yes`

#### Scenario: keep with no effect
- GIVEN `keep` = `hm`
- WHEN the settings page renders
- THEN it lists `hm` as having no effect

### Requirement: Tier 2 is opt-in through the settings page
#### Scenario: tick a suggestion
- WHEN the operator ticks `I mean` and saves
- THEN the add list holds `I mean` and the effective list shows it

### Requirement: One effective list on every surface
#### Scenario: parity with a configured list
- GIVEN a configured row list
- WHEN the run is exported over the CLI, the console download and `/api/v1`, as
  md turns and txt turns
- THEN the bytes are identical per format, and read mode shows the same
  paragraphs

### Requirement: The settings row wins over the environment
#### Scenario: both set
- GIVEN row `add` = `you see` and environment `VOXINT_FILLERS_ADD=I mean`
- WHEN exporting
- THEN `you see` is removed and `I mean` is not
- AND `doctor` and `voxint fillers show` report the additions as coming from
  settings, overriding the environment

#### Scenario: cleared field inherits
- GIVEN environment `VOXINT_FILLERS_ADD=I mean`
- WHEN the operator saves an empty add field
- THEN the row stores NULL, the page shows the inherited environment list, and
  `I mean` is removed

### Requirement: Honest reporting
#### Scenario: customised md report
- GIVEN a non-default list
- WHEN md turns are exported with `fillers=drop`
- THEN the last line is the pinned HTML comment with the count, the preset
  version and the added and kept entries

#### Scenario: CLI stderr
- WHEN the CLI streams an export with `--drop-fillers`
- THEN stderr carries the count and stdout is byte-identical to the console
  download

### Requirement: Invalid lists are refused plainly
#### Scenario: console
- WHEN the operator saves an entry with a digit or punctuation, or one that
  breaks the bounds
- THEN nothing is written, the response is a 422 on the General tab, and the
  submitted text is kept

#### Scenario: environment
- WHEN `VOXINT_FILLERS_ADD` holds an invalid entry
- THEN start-up fails closed with a message that names the variable but not the
  value

## Affected files / components

- `src/voxint/export/filler_lists.py` (new): the tiers, the version,
  `normalize_entries`, `FillerList` and `effective_filler_list`.
- `src/voxint/export/fillers.py`: the generated word pattern, the phrase pass
  with F1p, the F5 source guard, and `FillerList` threaded through every
  cleaning call.
- `src/voxint/export/turn_filters.py`: `fillers: FillerList | None`.
- `src/voxint/export/service.py`: the parameter change,
  `render_run_transcript_report` and the md report line.
- `src/voxint/app_settings.py`: `resolve_effective_filler_list`.
- `src/voxint/db/models.py` and `alembic/versions/0071_filler_lists.py`: two
  nullable JSON columns.
- `src/voxint/config.py` and `.env.example`: two environment fields and their
  validator.
- `src/voxint/cli.py`: the export resolves the list and writes stderr; the new
  `fillers show` subcommand.
- `src/voxint/api/routers/adjudication_api.py`, `api_v1/transcript.py` and
  `legacy_runs.py`: `Request` and `Settings` threading plus resolving the list.
- `src/voxint/api/templates/legacy_runs/transcript.html`: the "your list" note
  and the all-filtered copy.
- `src/voxint/api/routers/settings.py`: `POST /settings/fillers`, the
  context, and the `_settings_page_template` mapping.
- `src/voxint/api/templates/settings/_fillers.html` (new), `hub.html` and
  `settings.html`.
- `src/voxint/diagnostics.py`: the filler list line.
- **Tests.**
  - Unit: `tests/unit/test_fillers.py` and `tests/unit/test_turn_filters.py`
    (extended), plus `tests/unit/test_filler_lists.py` (new).
  - Integration: `tests/integration/test_turn_exports.py`,
    `tests/integration/test_runs_api.py` and the settings-page tests.
  - Contract: `tests/contracts/test_fillers_config.py` (new).
  - The doctor and CLI tests, and a migration round trip.
- **Docs.** `docs/operations.md`,
  `docs/how-to/managing-speakers-and-exporting.md`, `docs/architecture.md`,
  CHANGELOG `[Unreleased]`, and the browser lane
  (`.claude/skills/voxint-e2e-review/SKILL.md` and
  `tools/e2e_browser_lifecycle.py`, if the `cleanup` fixture needs a list).

## Implementation slices

Each slice leaves the tree green, and its docs land in the same slice.

1. **Presets, matcher and preset contract (pure).**
   - Covers `filler_lists.py`, the generated word pattern, the phrase pass with
     F1p, the F5 source guard, and `FillerList` threaded through
     `apply_turn_filters`, the service and the transports. Every transport
     passes the **default** list for now.
   - `tests/contracts/test_fillers_config.py` (preset part) pins the tiers,
     `PRESET_VERSION` and `NEVER` as literals.
   - **Gate:** every existing golden and parity test passes unchanged, the
     pattern-equality and property tests pass, and the phrase, keep, seam and
     count unit cases pass.
2. **Storage, resolver, environment and transports.**
   - Covers the migration, the row columns, the environment fields and
     validator, `resolve_effective_filler_list`, explicit `Settings` threading,
     and the real list in all four surfaces.
   - Adds the `.env.example` lines and the rest of the contract test.
   - **Gate:** the migration round trip, the precedence matrix, three-surface
     md and txt parity with a row list, read-mode parity, and the existing
     option and translation refusals unchanged.
3. **Honesty surfaces.** The md report line, CLI stderr, `voxint fillers show`
   and the `doctor` line, with their docs. **Gate:** the pinned comment golden,
   stdout and stderr separation, and the doctor detail with a readable and an
   unreadable row.
4. **Console.**
   - Covers the settings section and its POST, the tier 2 checkboxes, the
     inherited-environment display, the no-effect keep entries, and the
     read-mode note.
   - **Gate:** valid and invalid saves with `console_settings_enabled` on and
     off, CSRF, 422 keeping the submitted text, the browser acceptance lane, and
     the CHANGELOG.

## Testing strategy

- **Unit tests.** Every matcher scenario above, plus F1p edge cases:
  - closing quote after the phrase;
  - ellipsis or dash before it;
  - repeated punctuation;
  - adjacent phrases (`, you know, I mean,`);
  - tabs and newlines inside a phrase;
  - F2 moving a mark after a phrase (`It was, you know.` → `It was.`);
  - F4 capitalising after a sentence-start phrase;
  - a phrase at a segment boundary;
  - a speaker merge seam;
  - repeats blocked at a seam after a phrase is removed;
  - counts when filler-only turns drop or merge;
  - all six kept;
  - an empty list;
  - casing (`You Know`).

  `normalize_entries` is tested at its bounds before and after dedupe.
- **Property test.** The old regex and the generated default pattern give
  identical `drop_fillers` output on generated token soups.
- **Contract test.** Tier literals and version, `NEVER` disjoint from both
  tiers, `.env.example` lines, invalid environment values failing closed.
- **Integration tests.**
  - The existing default goldens stay unchanged.
  - New three-surface md and txt goldens with a row list.
  - Read-mode parity with a row list.
  - Settings POST in both console modes.
  - The migration round trip.
  - The doctor and CLI checks.
  - The stored transcript stays unchanged after every export.
- **Browser lane** (mandatory, because console behaviour changes): save the
  settings section, tick a tier 2 box, check the effective list, the read-mode
  note, and that the download matches the read view.
- **Gates.** ruff, mypy, the full pytest suite with coverage of at least 85% on
  new code, and `gitleaks git . --log-opts="main..HEAD"`.
- **Review.** High tier (a DB migration and the public export contract).

## Rollout, risks and open questions

- **Q1 (decided 2026-10-04: conditional; panel 3 of 3).** The md comment
  appears only for a non-default list, so the default output stays
  byte-identical. This reconciles two #753 lines ("ends with an HTML comment"
  and "byte-identical with no operator list"). Recorded on #753.
- **Q2 (decided 2026-10-04: conservative).** F1p is conservative. It misses `And you know, it was` and
  `It was kind of strange`, which have no surrounding mark. The panel split:
  - codex: ship the conservative rule;
  - glm: allow a leading coordinating conjunction, or restrict v1 to
    single-word additions;
  - qwen: noted the misses and the false positives.

  Recommendation: ship conservative and widen only on real operator feedback.
  Whisper output is usually well punctuated around these phrases, so the rule
  is not dead code.
- **Q3 (decided 2026-10-04: checkboxes).** Tier 2 as checkboxes over the add list (the synthesis of
  codex's "no tier 2 state" and glm's "copying is friction").
- **Risk: a regex built from operator input.** Mitigated by the character
  whitelist, `re.escape`, the length and count bounds, and the bounded cache.
- **Risk: the CLI and HTTP read the environment at different times.** An
  environment edit needs an HTTP restart, and the docs say so. Parity tests use
  one `Settings` for every surface.
- **Risk: phrase removal at piece boundaries changes ownership of the
  replacement space.** Covered by unit tests on timing, flags and separators.
- **Deferred:** an explicit "empty, overriding the environment" state, and
  conjunction-fronted phrases.

## Review notes

High tier consult, 3 of 3 voices: codex (planner, verified against the code at
`811e663b`), `z-ai/glm-5.3` and `qwen/qwen3.8-max-prime`.

- **The md comment is conditional (Q1). 3 of 3 agreed.** Accepted. glm added
  that people paste md into CMSs. codex added that it should list the entries,
  not just counts, and be exactly one LF-terminated line. Both accepted.
- **Keep tier 1 byte-identical with today's exact scaffold, and keep phrases
  separate. codex and qwen agreed.** Accepted: phrases are a separate first
  pass, the default pattern string is pinned equal to today's, a property test
  backs it, and an empty list skips the pass.
- **The phrase left context can't be a Python lookbehind, and the left context
  must not be consumed (codex).** Accepted: the left side is checked against
  `core` in the loop, and a rejected match is appended back unchanged.
- **F5: thread the list through the pre-screen and the re-clean; phrases across
  merge seams can empty a turn or hide a seam. codex and qwen agreed.**
  Accepted: phrases never cross a source seam, the final pass has a backstop
  that drops empty turns, and seam tests were added.
- **F1p meaning: tag questions, standalone replies and turn-initial clarifiers.
  glm and qwen agreed; codex agreed on the risk.** Accepted: a phrase followed
  by `?` and a phrase that forms a whole sentence are kept. Turn-initial
  `I mean, …` is still removed; that is the operator's opt-in. Conjunction
  recall goes to Q2 (split).
- **Tier 2 UX. Split: codex said no tier 2 state; glm said checkboxes.**
  Synthesised as checkboxes that write into `add`, with no new state (Q3).
- **Empty override and the environment. Split: codex wanted an explicit empty
  override; glm said cut the environment fallback.** The environment fallback
  is kept (it is in the issue's scope). The empty override is deferred, with
  the inherited value shown on the page.
- **Settings plumbing (codex).** The console helper and `/api/v1` lack
  `Request`, and `get_settings()` and `app.state.settings` can diverge.
  Accepted: explicit `Settings` threading, one resolve per operation, and the
  restart note.
- **Console placement (codex).** General renders `hub.html` when the console
  setting is on, and an unknown POST path falls through to Plugins. Accepted:
  both templates, plus the `_settings_page_template` mapping. glm also asked for
  the caution to say why. Accepted.
- **JSON NULL semantics (codex).** Accepted: `none_as_null`, whole-value
  assignment, unknown language keys preserved, and a round-trip test. The
  environment lists are no longer described as language-keyed.
- **`is_default` compared tuple order; keep semantics; counts (codex).**
  Accepted: canonical set comparison, exact-entry keep stated in the help
  text, and keep entries with no effect reported.
- **Honest doctor and CLI (codex).** Accepted: doctor does not claim a list
  when the row is unreadable; the stdout claim is scoped to streaming; validator
  messages never echo the value; the bounded cache; and no "no ReDoS" claim.
- **`fillers show` is bloat (glm).** Partly accepted. It stays, because #753
  names it, but it has no environment-only mode.
- **The version digest is ceremony (codex).** Accepted: literal pins instead.
- **Move the preset contract into slice 1 (codex and glm agreed).** Accepted.
  Docs now land with each slice (codex).
- **Word-count inflation if phrase removal glues tokens (qwen).** Covered by the
  count tests.

### Slice 1 code review (High tier, 3 of 3: codex, deepseek-v4-pro, qwen/qwen3.8-max-prime)

- **Backstop dropped punctuation-only turns that had no filler (codex, confirmed).**
  Fixed: a merged turn is dropped only when cleaning returned nothing.
- **Phrase alternation trusted the caller's order (deepseek, qwen).** Fixed:
  the pattern builder sorts longest first itself.
- **An expanding case mapping never matched (codex: `weiß`, then `İ`).** Fixed:
  casefold decides identity only. `words` and `phrases` keep the first spelling
  for matching, because `IGNORECASE` handles case but not an expanded fold.
- **A phrase before a same-speaker `.`-only turn stays after the merge
  (deepseek, High).** Not changed: merging may make a phrase ineligible, never
  newly eligible, and the seam space comes from `_join_at_seam`, which default
  output depends on.
- **A kept opening quote at turn end loses its space (`Hello, "you know` gives
  `Hello,"`) (qwen, Low).** Not changed: the single-word path already does this
  (`Hello, "um`), and a fix would change default bytes.
- In review, Claude found and fixed two bugs in codex's first draft: a rejected
  phrase skipped a pending capital, and trailing seam whitespace counted as a
  second source.

### Slice 2 code review (High tier, 3 of 3: codex, deepseek-v4-pro, qwen/qwen3.8-max-prime)

- **The resolver trusted the saved JSON shape (all three; codex Low, deepseek
  and qwen High).** Codex showed that `{"en": "um"}` iterated as the entries `u`
  and `m`, so a hand-edited row silently removed single letters. Fixed: a saved
  value must be a mapping, and when `en` is present it must hold a list of
  strings that passes `normalize_entries`. SQL NULL and a missing `en` still
  inherit the environment. Anything else raises a value-free `FillerListError`
  that starts "The saved filler word list is not valid. Save it again in
  settings." and, for a bad entry, appends the rule it broke. The
  console download, `/api/v1` and read mode answer 409, like the stale
  translation refusal, because the caller cannot repair server state by
  changing the request (codex follow-up; 422 was the first draft). The CLI
  exits 2. Exports without `fillers=drop` never read the list.
- **A separator-only environment value (`,,`) counted as an environment source
  (all three).** Fixed: the source and `env_overridden` come from the normalized
  entries, so it resolves as unset.
- **The CLI calls `get_settings()` a second time (all three, Low or
  acceptable).** Not changed: `_engine_or_report` does not return its
  `Settings`, `get_settings()` is the sanctioned constructor, and the call is
  guarded by the same exit-2 handler.

### Slice 3 code review (High tier, 3 of 3: codex, deepseek-v4-pro, qwen/qwen3.8-max-prime)

- Before the panel, Claude's review of codex's draft found three issues and fixed
  them with tests. `doctor` and `fillers show` printed a settings-sourced empty
  list (`{"en": []}`) as "none", hiding that it overrides the environment; they
  now show a 0 count or "(empty)" with "overriding the environment".
  `fillers show` let a database failure escape as a traceback that can carry the
  DSN; it now prints a DSN-free error and exits 2. `_export` now binds `fillers`
  before the session block.
- **`fillers show` labelled tier 1 "always removed" (codex Low, deepseek
  High).** With `keep = um` the line contradicted the effective list. Fixed: the
  label is "tier 1 (preset)".
- **`check_filler_list` could raise into `doctor` (qwen High).** Claude had
  narrowed codex's `except Exception` to `SQLAlchemyError`, which broke the
  module's "return a result, never raise" contract that every other check keeps.
  Fixed: the specific handlers stay and a final handler reports `check failed
  (TypeName)`, never the message.
- **Fixed, cheap (one reviewer each):** an empty effective list prints
  "(empty)" (deepseek); the `-o` branch flushes `wrote PATH` before the stderr
  count (qwen); tests now pin that `run_diagnostics` omits the filler line, so
  the setup wizard stays unchanged (qwen), that the DB-failure path leaves
  stderr empty (deepseek), and that an environment-sourced tier 2 addition is
  starred (qwen); the docs say the comment's "also removed" names the entries
  in effect, whether or not the recording contains them, and that the stderr
  counts match the comment (qwen).
- **Skipped (deepseek Medium):** making the comment's blank line independent of
  `to_markdown_turns` ending in LF. The renderer always ends a non-empty
  document with one LF, and the integration goldens assert the exact `"\n\n" +
  comment` suffix, so a change in that invariant fails a test instead of being
  papered over.
- Codex verified the fix delta. Its one Low, that no test pins the `flush=True`
  ordering, is skipped: pytest's capture cannot observe stdout/stderr
  interleaving, and a buffered-stream spy would outweigh a one-argument fix.

### Slice 4 code review (High tier, 3 of 3: codex, deepseek-v4-pro, qwen/qwen3.8-max-prime)

- All three seats reported the slice-4 acceptance criteria (tier 2 opt-in,
  cleared field inherits, invalid lists refused on the console, the read-mode
  note) as satisfied with tests. No Critical or High findings.
- Before the panel, Claude's review of codex's draft caught the filler context
  being merged into the setup wizard's context instead of the settings page's
  (codex had moved it by the time its tests ran), and the browser lane found the
  tier 2 checkboxes running together; a scoped `fieldset.filler-suggestions` rule
  fixes that. The textarea labels keep the existing inline look the Glossary
  section already has.
- **"Always removed (preset en-1)" contradicted a saved keep (3 of 3).** Fixed:
  "Removed by default (preset en-1): ... A word you keep below comes off this
  list."
- **"A kept word is never removed" overclaimed (codex).** Reproduced: keeping
  `you` does not protect it inside an added `you know`, because keeps match
  whole entries. Fixed: the help says keeps take an exact word or phrase off the
  list and that a single kept word is not protected inside a longer phrase.
- **Fixed, one reviewer each (qwen unless noted):** both invalid lists are now
  reported together; the fillers context is built before the benchmark query,
  whose rollback would expire the row; the read-mode "your filler list" link is
  plain text for non-admins in a multi-user install, matching the admin-only
  Settings route (red-checked); keep copy says "words or phrases"; the
  operations doc says the server validates submitted suggestions; tests cover
  `/settings/media` with a malformed row and assert the full read-mode sentence
  (deepseek).
- **Skipped (qwen Medium):** gating the all-filtered "using your filler list"
  clause on `fillers_removed > 0`. Unreachable: repeat removal keeps the first
  copy, so repeats alone can never remove every word.
- **Skipped (qwen Low):** qualifying "General tab" in the lay-reader how-to.
  Tabs are the default, and `docs/operations.md` names the single-page legacy
  mode.
- Codex verified the fix delta with no findings.
