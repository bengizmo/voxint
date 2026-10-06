# Plan: keep or omit a filler per word, through an append-only mark table (#757)

Status: done

Spec deltas: none (this project declares no living spec). Q1 and Q2 below
change the wording of #757's scope and acceptance. The maintainer decided Q1 to
Q3 on 2026-10-05 (recorded on #757, not in a spec).

## Goal

Epic #752 brings transcript clean-up closer to Descript's "Remove filler
words". Today `fillers=drop` removes the effective filler list (#753) from three
surfaces that share one pipeline, `apply_turn_filters`: md turns, txt turns and
read mode. The operator cannot overrule a single detection ("this `um` matters,
keep it") or drop one word the list does not cover ("leave out this
`basically`"). #757 adds both as text-only rulings:

- **Keep**: a detected filler survives `fillers=drop`.
- **Omit**: any one word is left out wherever `fillers=drop` applies.
- Both rulings:
  - can be reversed, by a clear ruling or by the console undo;
  - are stored append-only, with operator attribution;
  - never change stored text.
- **Console**:
  - underlines the fillers that `fillers=drop` will remove;
  - shows kept fillers distinctly and omitted words struck through;
  - gives one key to toggle keep and one to toggle omit on a focused word.
- **Unchanged output**: these stay byte-identical whether or not marks exist:
  - subtitles, `json`, `rttm`, timed txt, md blocks and translations;
  - every export with `fillers=keep`, which is the default.

**Binding maintainer decisions** (#752 and its children):
- Stored text never changes; everything is a projection or a ruling.
- `like so right well` are never removed by a preset rule. An operator omit on
  one occurrence is a ruling, not a rule, so it is allowed.
- Repeats never cross a filler seam (#755).
- No audio editing, no gap replacement, no retake detection.

## Ground truth (surveyed 2026-10-05, main `55bb6377`)

- **Turn projection** (`src/voxint/adjudication/turns.py`).
  - `TurnPiece` carries `text`, timing, `timed`, `segment_start`, `rule` (E1 to
    E6, translation) and `mapping` (T1 verbatim, T2 token-equivalent, coarse).
    It has **no segment id and no word index**.
  - Word pieces exist only for E4/E5 with a T1/T2 mapping. Split children (E1),
    whole-segment overrides (E2), no-words segments (E3) and unmapped text (E6,
    E4-coarse) are one coarse piece each.
  - For a coarse rule, `project_turns` builds no units at all
    (`turns.py:337`) and never calls `_map_text` (`:381`).
  - `_word_units` merges glued tokens and drops their original token indexes.
- **Filler pass** (`src/voxint/export/fillers.py`).
  - `_clean_pass` concatenates a turn's pieces into owned characters
    `(char, piece_index)` and iterates `pattern.finditer` over the original
    text.
  - For each match it applies F2 (mark moving, judged on `core`), F3 (separator
    collapse) and F4 (capitalise).
  - The regex consumes trailing whitespace and captures opening wrappers.
  - Phrases run first (F1p both-sides rule, with `sources` owner checks across
    F5 seams), then words, on the phrase pass's rewritten pieces.
  - F5 probes each turn's survival with a full clean, discards those results,
    merges the original pieces of one identity, and cleans the merged turn again.
  - Counts are word-count deltas (`turn_filters.py:38-43`).
- **Word coordinates.**
  - Splits (`segment_split_boundaries`) and word-range attribution rulings index
    the parent segment's immutable `words` token list, half-open.
  - A split adds a cut and never moves a token. The routes allow one cut per
    parent and have no un-split operation. A cut may fall at any interior token
    index, including inside a glued unit.
  - Splits and operator text corrections are mutually exclusive: the routes
    return 409 either way, and no DB constraint enforces it.
  - A segment with a fired pack trace can still project word pieces (T1/T2);
    only splitting is blocked by the trace.
- **The attribution ledger is not a safe home.** Readers of
  `adjudication_decisions` reduce a scope to its newest row without a decision
  filter:
  - `resolver.word_range_states`: a mark on a one-word split child's exact
    range would supersede its ASSIGN.
  - `resolver.newest_in_scope`, used by:
    - `undo._has_live_decisions`, which could then archive a speaker;
    - `naming._other_runs`;
    - restart impact.
  - `undo._scope_history`.
  - The #507 detach trigger turns a detached segment row into one that
    label-scope readers can see.
- **Undo today.**
  - Each action carries its own undo, with a 5 minute grace
    (`UNDO_GRACE_SECONDS`) and one `undoInfo` slot in `MediaEditor.tsx`.
  - `UndoToast` kinds are `enroll|decide|merge|relabel`. It posts `decision_id`
    to `/review/{run}/undo/{kind}` with the claim token, `csrf_token`
    (`CSRF_CLAIM`) and an `undo:<id>` key, then expects a labels response
    (`onLabelsChanged` iterates `result.labels`).
  - Segment undo refuses when the ruling is no longer newest in its scope
    (`undo.py:409`).
- **Console** (`MediaEditor.tsx`, `TranscriptPlayer.tsx`,
  `frontend/src/components/keymap.ts`).
  - Segment text is one string in `<span data-seg-text>`. Highlights use
    `renderAnnotatedText` over code-point offsets, and the span's text must stay
    byte-identical (`lib/selection.ts` walks it).
  - The only word-level UI is split mode: words come lazily from
    `GET /review/{run}/segments/{id}/words` and render as `tp-split-word`
    buttons.
  - Keys already taken: `v n p e j k 0 s = h w d ? @`, `1` to `9`,
    Ctrl/Cmd+Enter and Ctrl/Cmd+K.
  - Segment writes are gated by the claim token. Claims are refused on archived
    runs (`deps.py:549`). Relabel takes an 8 to 64 character nonce.
- **Exports.**
  - `render_run_transcript_report` (`export/service.py`) and read mode
    (`legacy_runs.py`) build `attributed_turns`, then `apply_turn_filters`,
    then `layout_turns`.
  - API v1 and the CLI delegate to the service.
  - Fillers are rejected together with a translation.
  - The md honesty comment appears only for a non-default list. HTTP txt has no
    comment channel. Read mode shows `read_filter_note`, and the CLI prints
    counts to stderr.
- **Restart.**
  - `RestartImpact.has_editorial_work` counts corrections and verifications.
    Five places consume it: the blocked-restart exception, the stage profiles,
    the run-detail template and its JavaScript, and the CLI restart and bulk
    preview.
  - Split boundaries and review state cascade on segment delete.
  - Native backups dump the whole database, so no table allow-list exists.
- **Users** are never deleted by application code. The ledger trigger forbids
  any `user_id` change.
- **Alembic** head is `0071`, pinned in `tests/integration/test_migration_0016.py`.

## Assumptions and constraints

- **Where marks can exist:**
  - on segments whose stored `words` validate and reconcatenate to `raw_text`,
    and whose CORRECTED text maps onto those words (T1 or T2);
  - on split children, whose text is derived from the words.
- **Where they cannot:** a segment whose shown text does not map. The console
  says why, and the operator's own text correction is the per-word tool there.
- **When marks apply:** only when `fillers=drop`. They never apply with
  `fillers=keep` or with `repeats=drop` alone, so default bytes cannot move.
- **Scale:** single-operator, so recomputing a run's projection per editor load
  and per write is acceptable until measured otherwise.
- **The plan is invalidated by either of these requirements:**
  - marks must survive a full run restart (segments get new ids and tokens);
  - omit must act as redaction. It does not: raw text and other formats keep
    the word, and the operator guide says so.

## Proposed approach

### A. Storage: a separate append-only table, `segment_word_marks` (migration 0072)

The table is modelled on `segment_split_boundaries`: per-segment editorial
state keyed on immutable token offsets. It takes the ledger's append-only,
attributed and idempotent properties.

| Column | Notes |
|---|---|
| `id` | uuid pk |
| `seq` | `BIGINT GENERATED ALWAYS AS IDENTITY`. Gives a deterministic newest-wins order that does not depend on transaction timestamps or random UUIDs |
| `pipeline_run_id` | FK `pipeline_runs`. Denormalised by the writer, which checks it equals the segment's run |
| `segment_id` | FK `transcript_segments` ON DELETE CASCADE (see Q2) |
| `start_word_index`, `end_word_index` | One word unit as a half-open token range into the parent's `words`. CHECK `start >= 0 AND end > start` |
| `action` | CHECK in `keep`, `omit`, `clear`, `undo` |
| `voids_mark_id` | FK self ON DELETE CASCADE. CHECK `(action = 'undo') = (voids_mark_id IS NOT NULL)` and `voids_mark_id <> id`. Unique partial index: one undo per mark |
| `operator`, `user_id` | As on the ledger, including the system/user CHECK pair |
| `idempotency_key` | UNIQUE. A client nonce, or `undo:<mark id>` for an undo |
| `created_at` | Server default; for display only |

Index: `(pipeline_run_id, segment_id, seq)`.

**Append-only trigger.**
- It rejects every UPDATE. `user_id` included, as on the ledger, because users
  are never deleted.
- It rejects a DELETE while the parent segment still exists:
  `EXISTS (SELECT 1 FROM transcript_segments WHERE id = OLD.segment_id)`.
- An FK cascade runs after the parent row is gone, so cascade and the self-FK
  chain succeed, and a direct DELETE of a live mark fails.
- No TRUNCATE trigger. The integration test harness truncates every table
  between tests, and no application code truncates.

**One writer**, `src/voxint/adjudication/word_marks.py`, does every insert:
- Idempotency is adopt-or-conflict, as in `ledger.record_decision`. A replay
  with the same payload returns the row; a different payload under the same key
  is a conflict.
- Only an undo may use a key in the `undo:` namespace.
- It validates that the segment belongs to the run.
- `action=undo` is accepted only from the undo function, never from the mark
  route.

**Resolution** is a pure function over a run's rows:
1. Drop `undo` rows and every row an undo voids.
2. For each unit, the remaining row with the highest `seq` decides: `keep` or
   `omit`, while `clear` means no mark.

Marks are one unit each, so ranges never partly overlap a unit. A `clear`
targets the same unit as the mark it clears.

**Undo** voids one row:
- It is refused when:
  - the target is in another run;
  - the target is itself an undo, or is already voided;
  - the grace window has passed;
  - any newer, non-voided row exists on the same unit (drift, as
    `undo.py:409`).
- It is allowed for keep, omit and clear. Undoing a clear brings back the mark
  under it.
- With the drift rule in place, "omit then undo" restores the previous state
  exactly.

**Alternative rejected:** new `Decision` values in `adjudication_decisions`,
which the issue text proposes.
- It needs loosened segment CHECKs.
- It needs a decision filter added to at least five newest-row readers, and one
  missed reader silently changes attribution or archives a speaker.
- The separate table keeps "attribution rulings and their tests are unchanged"
  true by construction, and keeps the issue's intent: append-only, attributed,
  newest wins, undo.

### B. Units and anchors on turn pieces (no output change)

- **Keep token indexes on units.** `_word_units` keeps each unit's parent token
  range.
- **Provenance-only units for coarse rules.** E1 and E2 get units without any
  change to their attribution or their coarse rendering:
  - E1 units come from the child's token range.
  - E2 units come from the whole segment and are mapped with `_map_text`.
- **New field** on `TurnPiece`: `anchors: tuple[WordAnchor, ...] = ()`, where
  `WordAnchor(segment_id, token_start, token_end, lex_start, lex_end)`.
  - `lex_start` and `lex_end` place the unit's **lexical** characters inside
    `piece.text`, so leading and trailing whitespace are excluded.
  - Anchors are an input-only coordinate. The filter pass converts them into
    per-character source identities before it rewrites anything (C below), so
    no stage ever reads an offset from rewritten text.
- **Which pieces get anchors:**
  - E4/E5 word pieces (T1/T2): one anchor each.
  - E1 children: one anchor per unit, allowing for the child text's outer trim.
  - E2 overrides that map: one anchor per unit.
  - E3, E6, E4-coarse and translation pieces: none.
- `join_pieces`, layout and every renderer ignore anchors, so every output stays
  byte-identical.
- **A cut inside a glued word** (only possible for splits made before #757
  ships, since slice 4 refuses new ones on a marked word): each child anchors its
  own part, in parent token coordinates. So a mark targets an anchor's token
  range, which is not always a whole spoken word. Pinned by
  `test_split_cut_inside_a_glued_word_anchors_each_child_part`.

### C. The filter pass: source identities, an omit pass, keep protection, a trace

- **Removal spans.** `_clean_pass` is refactored from "iterate regex matches"
  to "iterate removal spans". Each span carries its lexical range, its consumed
  separator and mark, and its opening wrappers. F2, F3 and F4 are unchanged in
  behaviour.
- **Source identity.** Each owned character also carries a source identity:
  `(segment_id, token range)` for an anchored character, or none. Identities
  survive every rewrite: the omit pass, the phrase pass, the word pass,
  separator collapse, mark moving and capitalising. A moved terminal mark keeps
  its own identity. A synthetic separator has none.
- **Order of passes**, each run on the text as it stands after the one before:
  1. The **omit pass**. Each omitted unit's lexical characters form a removal
     span. Its opening wrappers and closing quotes or brackets stay, as F1 keeps
     them, and F2 to F4 tidy around it.
  2. The **phrase pass**, as today.
  3. The **word pass**, as today.

  Applying omits first means a phrase that now stands alone after an omission
  is judged as a reader would see it. This is the one place where marks can
  create a removal, and the console shows it, because detection is the same
  trace.
- **Keep protection.** A filler match whose **lexical** range includes any
  character from a kept unit is skipped, in the same way a rejected phrase is
  skipped. Separators and wrappers do not count, so keeping `uh` in `um uh`
  still removes `um`.
- **F5** runs the full sequence (omit, phrase, word) both in its survival probe
  and in the merged clean. Counts and the trace come only from the final merged
  clean, never from the probe. The `sources` bookkeeping for the phrase owner
  check and the seam set is unchanged, because the omit pass only removes
  characters.
- **Repeats** (#755) run afterwards, unchanged. Keep protects against filler
  removal only, which matches its name. This is pinned by a test.
- **Trace.** The pass records each committed removal as `(source identity,
  cause)`, where the cause is `filler`, `phrase` or `omit`, and each protected
  match as `kept`. Detection (D) is the trace of the real export pipeline with
  every effective mark applied.
- **Counts are words, with no double counting:**
  - `fillers_removed`: units removed by a filler or phrase match.
  - `omitted`: units removed by an omit mark, including an omitted unit that is
    on the filler list.
  - `kept`: keep marks that actually blocked a match.
- **Unanchored marks.** An effective mark on a token range that no piece in
  this rendering anchors is handled as Q1 decides. The proposed rule is to
  refuse.

### D. Endpoints

**`GET /review/{run}/word-marks`**, for the CORRECTED view. For each emission
(a segment, or a split child by token range) it returns:
- `markable`, with a reason when false;
- the trace's removals and kept matches, each as source units with code-point
  offsets into the text the editor shows (a phrase may span segments, so this
  is a list);
- the effective marks;
- stale marks (effective, but unanchored now).

It carries a `version` (the run's highest mark `seq` plus a hash of the text
inputs), so the client can drop a stale response.

**`POST /review/{run}/segments/{seg}/word-marks`** takes the form fields `token`
(the claim), `nonce`, `action` and `start`/`end`.
- `keep` and `omit` require:
  - a valid claim (archived runs cannot be claimed);
  - a unit-aligned range inside the token count;
  - a segment that is markable now.

  `keep` also requires a range inside a traced filler or phrase removal or a
  kept match. A focused word inside a detected phrase therefore qualifies.
- `clear` requires only the claim and an effective mark on exactly that unit. It
  works on a stale mark, an unmarkable segment and a filler that is no longer
  on the list.
- The client always sends an explicit action, so a replayed request cannot flip
  state.
- The response is the run's word-marks payload, plus `undo` for a fresh write.

**Undo** is a new kind `word-mark` on `/review/{run}/undo/{kind}`. It sends the
claim token, `csrf_token` (`CSRF_CLAIM`) and `mark_id`, with key
`undo:<mark id>`. It returns the word-marks payload, not labels; `UndoToast`
gains a branch for that response.

**Interaction with other writes:**
- **Text correction** (`POST …/text`): when the new text no longer maps onto a
  segment's words and the segment has effective marks, the same transaction
  appends a `clear` for each of them. The response says how many were cleared,
  and the console shows that message. This keeps the CORRECTED view free of
  stale marks without refusing the edit (see Q1).
- **Split**: a cut that falls strictly inside a marked unit is refused with
  "this word has a clean-up mark; clear it first". Every other cut leaves marks
  anchored.
- **Verification**: marks do not clear `verified_at`, because they do not
  change the reviewed text.

**Restart:** `RestartImpact` gains `word_marks`, the effective keep and omit
marks, counted in `has_editorial_work`. All five consumers learn about it: the
exception text, the stage profiles, the template data attributes and their
JavaScript, and the CLI restart and bulk preview. A later-stage restart that
keeps segments but changes enhanced text can strand marks. That is the backstop
case in Q1.

### E. Console

- **Normal view.**
  - Two highlight layers added to `renderAnnotatedText`:
    - an underline for traced removals;
    - a distinct style for kept matches, and strike-through for omitted words.
  - The `data-seg-text` text stays byte-identical.
  - A stale mark shows a small notice on its segment, with **Clear**.
- **Clean-up mode** for the active segment, modelled on split mode:
  - `c` enters the mode. Words render as focusable buttons, and Tab and the
    arrow keys move focus between them.
  - `f` toggles keep on a focused detected filler; `o` toggles omit on any
    focused word; Escape leaves.
  - The mode bar prints the keys and has buttons for mouse users.
  - On an unmarkable segment the mode explains why.
- **Undo**: `UndoToast` gains kind `word-mark` with its own response handler.
- **Refresh**: the marks payload reloads after a mark write, correction, split,
  relabel, attribution undo or settings save. Older responses are discarded by
  `version`.
- **Keymap**: `REVIEW_KEY` constants, `extraShortcuts` cheat-sheet rows, the
  `case`s and `<kbd>` hints.

### Alternatives considered

- **Anchor on whitespace tokens of the displayed text, plus a text hash.**
  Every edit to the text would orphan the segment's marks, and it adds a second
  word coordinate system beside the one splits and attribution use. Rejected.
- **Sequence alignment to re-anchor after an edit.** This is positional
  guessing, which the projection doctrine forbids. Rejected.
- **Report stale marks and continue the export.** This was the first draft's
  proposal. It contradicts the issue's "refuses with a clear message", and HTTP
  txt has no channel to report on. Q1 prevents staleness at its sources
  instead, and keeps refusal as the backstop.
- **Apply omit with `fillers=keep` too.** This breaks byte-identical default
  exports and contradicts the issue. Rejected.
- **Per-segment detection.** It would disagree with exports at turn seams and
  on the phrase rule. Rejected in favour of the trace.
- **Underlines only inside clean-up mode** (a glm suggestion, to save one
  projection per editor load). The issue asks for underlines in the editor.
  Kept in the normal view; the projection cost is measured in slice 4.

## Acceptance criteria

### Requirement: Kept fillers survive filler removal
The system SHALL leave a word marked keep in md turns, txt turns and read mode
when fillers are dropped, and SHALL judge its neighbours as ordinary text.

#### Scenario: keep defeats the list
- GIVEN "Um, we start now." with `Um` marked keep
- WHEN the md turns export runs with `fillers=drop`
- THEN the output reads "Um, we start now." and the comment counts 1 kept

#### Scenario: a neighbour is still removed
- GIVEN "um uh we start" with `uh` marked keep
- WHEN fillers are dropped
- THEN the output reads "Uh we start"

#### Scenario: keep inside a phrase
- GIVEN `you know` is on the add list, and `know` in one occurrence is marked keep
- WHEN fillers are dropped
- THEN that occurrence stays whole and other set-off occurrences are removed

### Requirement: Omitted words leave human-readable exports only
The system SHALL leave out a word marked omit in md turns, txt turns and read
mode when fillers are dropped, using the same comma, terminal-mark, separator
and capitalisation rules as for a filler.

#### Scenario: omit any word
- GIVEN "So basically, the coil froze." with `basically,` marked omit
- WHEN the txt turns export runs with `fillers=drop`
- THEN it reads "So the coil froze."

#### Scenario: an omission sets off a phrase
- GIVEN `you know` is on the add list and "you, basically, know, it froze" has
  `basically,` marked omit
- WHEN fillers are dropped
- THEN the output, the console underline and the counts all agree on what was
  removed

#### Scenario: other formats are untouched
- GIVEN a run with keep and omit marks
- WHEN srt, vtt, json, rttm, timed txt, md blocks, a translated md export, and
  md turns with only `repeats=drop` are rendered
- THEN each is byte-identical to the same export before any mark existed

#### Scenario: default exports are untouched
- GIVEN a run with marks
- WHEN md turns, txt turns or read mode render with `fillers=keep`
- THEN the bytes equal those of the run with no marks

### Requirement: Marks are reversible
The system SHALL let the operator clear a mark at any time and undo a fresh
action within the undo window, restoring the previous rendering exactly.

#### Scenario: omit then undo
- WHEN a word is omitted and the omit is undone
- THEN every export returns to its previous bytes

#### Scenario: undo after a newer mark
- GIVEN a word was omitted and then, separately, marked keep
- WHEN the operator tries to undo the omit
- THEN the undo is refused as out of date, and the console refreshes

#### Scenario: clear after the window
- WHEN a kept filler is toggled off after the undo window
- THEN a `clear` row is appended and `fillers=drop` removes the filler again

### Requirement: Marks stay anchored, or the export refuses
The system SHALL anchor a mark to the parent segment's word tokens, so a split
never moves it. An export that cannot place an effective mark SHALL refuse with
a message naming the segment and the way to clear the mark.

#### Scenario: split after a mark
- GIVEN `um` is marked keep and the segment is then split between other words
- WHEN fillers are dropped
- THEN the same `um` is kept, in the child that holds it

#### Scenario: a split inside a marked word
- WHEN the operator splits inside a marked glued word
- THEN the split is refused with a reason

#### Scenario: text re-typed after a mark
- GIVEN a word is marked omit, and the segment's text is then corrected beyond a
  case or punctuation change
- WHEN the correction is saved
- THEN the segment's marks are cleared in the same transaction, and the
  response says how many

#### Scenario: a mark the rendering cannot place
- GIVEN an effective mark whose tokens the requested text variant does not map
  (for example `text=enhanced` after a later-stage restart changed the
  enhanced text)
- WHEN the export runs with `fillers=drop`
- THEN it refuses (HTTP 409, CLI non-zero exit, no file written), and the
  console lists the stale mark with **Clear**

### Requirement: Marks are only offered where they can apply
The system SHALL refuse keep or omit on a segment whose shown text does not map
to its stored words, with a reason the console shows.

#### Scenario: unmapped text
- WHEN the operator opens clean-up mode on such a segment
- THEN the mode explains that marks need the text to match the recorded words,
  and the write route returns 409 with the same reason

### Requirement: Attribution is unaffected
The system SHALL not change any attribution read, write, undo or restart count
because of marks. The attribution ledger, its readers and their tests are
unchanged.

### Requirement: Marks are append-only and attributed
The system SHALL store every mark, clear and undo as a new row naming its
operator. The database SHALL refuse any update, and SHALL refuse a deletion
unless the parent segment has been deleted.

### Requirement: Restart warns about marks
The system SHALL count effective marks as editorial work in every restart
warning surface.

### Requirement: Console shows detection and marks
The system SHALL underline the words that `fillers=drop` will remove, show
kept fillers distinctly and omitted words struck through, and offer keyboard
access to keep and omit.

#### Scenario: toggle keep by key
- GIVEN clean-up mode on a segment with a detected `uh`
- WHEN the operator focuses `uh` and presses `f`
- THEN the word shows as kept and the undo toast offers undo

## Affected files / components

| Path | Change |
|---|---|
| `alembic/versions/0072_segment_word_marks.py` | Table, CHECKs, indexes, identity column, trigger. Downgrade drops the table and its data (documented) |
| `src/voxint/db/models.py` | `SegmentWordMark`, `WordMarkAction` |
| `src/voxint/adjudication/word_marks.py` | New: writer, undo, resolver, markability, ownership checks |
| `src/voxint/adjudication/turns.py` | Unit token ranges, provenance-only units for E1/E2, `WordAnchor`, `TurnPiece.anchors` |
| `src/voxint/export/fillers.py` | Span-based `_clean_pass`, source identities, omit pass, keep protection, trace |
| `src/voxint/export/turn_filters.py` | `marks` parameter, trace, new counts |
| `src/voxint/export/service.py` | Loads the overlay centrally when fillers drop; refusal error; comment counts; `RenderedTranscript` counts |
| `src/voxint/api/routers/legacy_runs.py` | Read mode overlay, note, `all_filtered` with omit-only output, refusal page |
| `src/voxint/api/routers/api_v1/transcript.py`, `src/voxint/cli.py` | Map the refusal to 409 and a non-zero exit; stderr counts; restart warnings |
| `src/voxint/api/routers/adjudication_api.py` | Word-marks GET and POST, undo kind, clear-on-correction in the text route, split refusal inside a marked unit |
| `src/voxint/ingest/service.py`, `src/voxint/api/templates/legacy_runs/_run_detail_body.html` | `RestartImpact.word_marks`, exception, stage profiles, warning JavaScript |
| `frontend/src/components/{MediaEditor,TranscriptPlayer,UndoToast}.tsx`, `frontend/src/components/keymap.ts` | Highlight layers, clean-up mode, keys, undo response branch, refresh and versioning |
| `tools/e2e_browser_lifecycle.py`, `.claude/skills/voxint-e2e-review/SKILL.md` | `wordmarks` fixture and scenario; fix the stale `--fixture` help |
| `tests/contracts/` | Formats unchanged with marks present; trigger and CHECK presence; no attribution reader references the new table |
| `tests/integration/test_migration_0016.py` | Head pin to `0072` |
| `docs/how-to/reviewing-and-adjudicating.md`, `docs/operations.md`, `docs/architecture.md`, `CHANGELOG.md` | Operator guide (omit is not redaction), export behaviour and refusal, data model |

## Implementation slices

Q1 to Q3 were decided on 2026-10-05. Slices 1 and 2 are preparation and change
no output.

1. **Units and anchors.**
   - Unit token ranges, provenance-only units for E1/E2, anchors.
   - Unit tests per rule: E1 to E6, T1, T2, glued tokens, whitespace-only
     tokens, a child's outer trim, an E2 that maps and one that does not.
   - Gate: the existing suite passes unchanged, and md/txt turns and read mode
     are byte-identical over the e2e and parity fixtures.
2. **Span refactor and trace.**
   - `_clean_pass` iterates removal spans and carries source identities. The
     trace is recorded but has no consumer yet.
   - Gate: characterization snapshots of md/txt turns and read mode, captured
     **before** the refactor over every filler, repeat and turn test input plus
     a generated corpus, are byte-identical afterwards. The trace is checked
     against hand-written expectations, including cross-segment phrases and F5
     probes.
3. **Marks honoured by exports (High review: migration).**
   - Migration 0072, model, writer, undo function, resolver.
   - The omit pass and keep protection, counts, honesty comment, refusal on
     every surface, restart impact in all five consumers.
   - Contract tests and the head pin.
   - Integration tests write marks through the writer and render every surface.
   - The CHANGELOG entry waits for slice 4, because no operator can create a
     mark yet. The PR says this slice is infrastructure.
4. **Console delivery (API and UI together).**
   - Word-marks GET and POST, the undo kind, clear-on-correction, the split
     refusal.
   - Highlight layers, clean-up mode, keys, refresh and versioning.
   - vitest, the browser lane with the `wordmarks` fixture, the operator
     how-to, the CHANGELOG.
   - Measure the projection cost on the longest local fixture.

Each slice gets its own `/code-review`:
- Slice 3: High (schema).
- Slice 4: High, because it touches a public route contract and claim/CSRF
  gating.
- Slices 1 and 2: Medium, unless the final diff hits a High trigger.

The browser lane runs on slice 4. Each slice merges by PR.

## Testing strategy

- **Pure unit tests:**
  - Anchors.
  - The trace.
  - Keep isolation: a keep right after an unkept filler, phrase-adjacent
    whitespace, a moved terminal mark.
  - Omit: sentence start, mid-sentence comma, terminal mark moving, quotes and
    brackets around the word, an omitted filler counted once.
  - The omit-then-phrase interaction; F5 probe and merge with marks; a
    filler-only or omit-only turn between two turns of one identity.
  - A mark across a split cut; repeats after omit and after a kept filler.
  - Resolver: highest `seq` wins per unit, clear, undo of a clear, undo drift.
- **Integration tests (Postgres):**
  - Migration up, down and up again, with populated rows, the CHECKs and the
    identity column.
  - Trigger: UPDATE refused; direct DELETE refused; segment delete cascading
    through a mark and its undo row; run restart with marks.
  - Writer: adopt-or-conflict, the reserved `undo:` keys, segment ownership.
  - Routes: claim, archived run, nonce replay, the 409 reasons, the undo window
    and drift, clear on a stale mark, clear-on-correction, the split refusal.
  - Every surface with and without marks, including refusal (no file written).
  - Byte-identity of srt, vtt, json, rttm, timed txt, md blocks, translated md
    and default exports.
  - A native dump and restore round trip of a populated database, which keeps
    marks, self-FKs and the trigger.
- **Contract tests:**
  - No attribution reader module imports or queries the new table.
  - Trigger and CHECK names.
  - The head pin.
- **Frontend:**
  - vitest for the highlight layers (span text unchanged, astral characters),
    the clean-up mode reducer, the keys, the undo response branch and stale
    response dropping.
  - The browser lane for the whole loop, including the refresh after a
    correction.
- **Discipline:**
  - Run new test files with `-n 4`, and never put nonces in `parametrize` ids.
  - Run the full integration suite (in the background, to a log) before each
    PR.

## Rollout, risks and open questions

- **Q1 (decided 2026-10-05: prevent at the sources, refuse as the backstop):
  stale marks.** The issue requires that a mark
  "still anchors or the export refuses". Proposed:
  - prevent staleness where it starts (a correction clears the segment's marks
    in the same transaction and says so; a split inside a marked unit is
    refused);
  - refuse an export only as the backstop, for a non-default text variant or a
    later-stage restart.

  The panel split here. glm and the first draft preferred to report the stale
  mark and continue; codex holds that refusal is required unless the maintainer
  changes acceptance, and notes that HTTP txt cannot carry a report.
- **Q2 (decided 2026-10-05: separate table, cascade with the segment):
  storage and retention.**
  - Proposed: a separate `segment_word_marks` table, unlike the issue's wording,
    which places the rulings in `adjudication_decisions`.
  - Proposed: marks cascade with their segment, like corrections and splits,
    and the restart warning names them.
  - The alternative is #507-style history preservation. It needs SET NULL plus
    a trigger exception, and keeps rows that can never apply again.
  - All three voices endorse the separate table. qwen and codex asked for the
    retention policy to be an explicit choice.
- **Q3 (decided 2026-10-05: clean-up mode with `c`, `f`, `o`): interaction.** Proposed: clean-up mode with `c`,
  `f` (keep a filler) and `o` (omit), with keys printed in the mode bar and
  buttons for mouse users. Clicking an underlined filler directly could be added
  later.
- **Settled with the panel:** keep does not protect against repeat removal (3 of
  3 agree), and a test pins it.
- **Risk:** the span refactor touches the most-tested code in the filter stack.
  Slice 2 isolates it behind characterization snapshots taken before it starts.
- **Risk:** projection cost per editor load. The first cut recomputes the whole
  run. A neighbourhood-only shortcut is rejected, because F5 can join turns
  across any number of removed turns.
- **Risk:** an unusual whisper `words` list leaves a segment unmarkable, and the
  reason must say so plainly.

## Review notes

Plan consult, High tier (schema migration, first round), 3 of 3 seats: codex
(`gpt-6.1-sol`, planner role, read the repo), `z-ai/glm-5.3`, and
`qwen/qwen3.8-max-prime` at high effort. The second seat is glm, per the plan
rule. DeepSeek was not seated.

**Storage choice** (3 of 3 endorse the separate table)
- All three confirmed the newest-row reader hazard. glm noted that
  `_has_live_decisions` could archive a speaker.
  - Kept.
- codex: a synthetic label is not strictly needed, because labelled segments
  have one.
  - Wording corrected; the reader hazard remains the reason.

**DELETE trigger** (codex Critical, qwen Critical, glm Medium)
- codex: `pg_trigger_depth()` is already 1 inside the trigger, so `> 0` lets a
  direct DELETE through. qwen: depth does not prove a cascade, and TRUNCATE is
  not blocked.
  - Replaced with a parent-existence check.
  - TRUNCATE is left unblocked on purpose: the test harness truncates every
    table.
- qwen: the self-FK needs an explicit ON DELETE.
  - Set to CASCADE, with a test.

**Retention** (qwen High, codex High)
- Cascade discards history, unlike #507.
  - Made an explicit maintainer decision (Q2); cascade is proposed.

**Ordering** (qwen Critical)
- `created_at, id` ties are arbitrary.
  - Added the `seq` identity column.

**Undo** (glm Critical, qwen Critical, codex High)
- Undoing an older mark under a newer one did nothing visible.
  - Refused on drift, as the attribution undo does.
- Reject undo of an undo, cross-run targets, and reuse of an `undo:` key.
  - Added.
- codex: `UndoToast` expects a labels response.
  - Added a response branch.

**Idempotency and ownership** (glm High, qwen High, codex Medium)
- Adopt-or-conflict, a reserved namespace, and the segment/run ownership check.
  - Added to the writer.

**Stale marks and refusal** (codex High vs glm Low; maintainer decision Q1)
- Split surfaced in Q1. Proposed: prevent at the sources, refuse as the
  backstop.
- codex: `clear` was blocked by the markability check.
  - `clear` now has its own validation.

**Keep isolation** (codex High)
- A keep anchored over leading whitespace protected the neighbouring filler.
  - Anchors and the overlap test now use lexical ranges only, with a scenario.

**Glued units and split cuts** (codex High, glm High)
- `_word_units` drops token indexes, and a cut may fall inside a unit.
  - Units now keep token ranges, a cut inside a marked unit is refused, and
    tests are added.

**Character provenance** (codex High, glm Medium, qwen Medium)
- Offsets go stale after rewrites, and the F5 probe must not count.
  - Source identities are carried per character, anchors are input-only, and
    counts come from the final clean only.

**Omit and phrase ordering** (glm Critical, qwen High)
- The draft did not say which pass omit runs in.
  - Pinned: omit pass, then phrases, then words, each on the text as it stands,
    with a scenario.

**Detection** (codex High, qwen Medium)
- It needs a trace, not a text diff. Ignoring keeps would change F5 seams.
  - Detection is now the trace of the real pipeline with all marks applied.
- glm: drop the normal-view underlines to save cost.
  - Rejected, because the issue asks for them. Measured in slice 4.

**Counts** (codex Medium, glm Medium, qwen High)
- Word-count deltas conflate the causes.
  - Counts now come from trace causes, in words, with each unit counted once.

**Restart** (codex High, qwen High)
- Five consumers, not one.
  - All five listed; the count is effective marks.

**Refresh** (codex Medium)
- Other writes change the marks.
  - Added refresh events and a payload `version`.

**Slicing** (codex Medium, glm High)
- glm: split the refactor out. codex: ship the console API and UI together.
  - Both adopted: four slices, two of them zero-change preparation.

**Testing** (all three)
- Characterization snapshots captured before the refactor.
- New-semantics tests.
- A populated dump and restore round trip (codex: no table allow-list exists, so
  no backup change is needed).
- Migration down with data.
- glm: drop a scan-based "sole writer" contract test.
  - Replaced with a narrower contract: no attribution reader references the
    table.

**Factual corrections** (codex Low, glm High)
- The keymap path.
- No un-split route exists.
- E1/E2 have no units today.
- E4-coarse means unmapped.
- Pack-traced segments can be T2-markable.
  - All fixed in Ground truth.

**Scope trims** (codex Low)
- The neighbourhood fallback.
  - Dropped.
- Overlay loading duplicated in API v1 and the CLI.
  - Centralised in the service.

### Slice 1 code review (Medium, 2 of 2: codex, moonshotai/kimi-k3)

kimi-k3 substituted for deepseek-v4-pro, which returned 402 (insufficient
balance) after a retry. Merged as PR #785 (`86df1dfe`).

- 0 Critical or High.
- Fixed:
  - The anchor harness now requires one anchor on every word piece. Dropping
    word-piece anchors fails 45 tests.
  - The E2 raw-text mismatch case now selects text that maps, so only the
    reconcile guard refuses it (codex).
  - A test pins a split cut inside a glued word: each child anchors its own
    part.
  - A test covers an override under a review correction.
  - Anchors are hidden from repr, and hash invariance is pinned.
  - The E3 predicate is shared as `_usable_words`.
- Delta re-review clean.

### Slice 2 code review (Medium, 2 of 2: codex, deepseek-v4-pro)

A kimi-k3 substitute errored. DeepSeek was topped up mid-review and took its
seat.

**Gate.** The characterization fixture covers 1,165 cases: harvested test
inputs, each also rendered under its captured configuration, 500 generated
cases and 200 seeded soups. It was regenerated against the pre-refactor
filters and verified independently by swapping the `HEAD` filter files in
(1,166 passed). codex also ran 40,000 randomized comparisons against `HEAD`,
all matching.

**Rejected.** deepseek M1: a span skipped by the keep hook still gets F4
capitalisation. That is intended. A kept filler that becomes sentence-initial
reads "Um", and a test pins it for slice 3.

**Skipped.** deepseek L2: `_track` trusts anchor bounds. Only `project_turns`
produces anchors, and it bounds-checks them, so an IndexError is the right loud
failure.

**Fixed.**
- Slotted internal dataclasses, direct `_Char` construction and reuse of
  unchanged pieces. Filtering 24,000 words takes 382 ms against 272 ms before
  the refactor (453 ms before this fix).
- Anchors are restored only when both the text and the character identities
  are unchanged.
- The owners guard.
- The trace order is documented: per output group, phrases before words.
  Consumers key by source identity.
- A three-survivor F5 identity test.
- The harness renders each harvested case under its captured configuration,
  shares the soups with `test_fillers.py`, and names cases uniquely and
  stably across hash seeds.
- The fixture size cap is 2.5 MB.

Delta re-review clean.

### Slice 3 code review (High, 3 of 3: codex, deepseek-v4-pro, qwen/qwen3.8-max-prime)

Implemented by codex in three bounded chunks (storage, filter, surfaces), each
diff reviewed before the next. The full suite passed before review (11,931
passed; the 3 failures are the known local-only enrichment gates).

**Fixed.**
- codex Medium: the writers locked the segment and then needed the run (FK),
  while restart locks the run and then deletes segments, so the two could
  deadlock. Both writers now take `FOR KEY SHARE` on the run first. A test
  pins the order.
- codex Low: omitting a wrapped word left `()` and stray punctuation. Balanced
  wrappers owned by the unit are now removed with it, and a mark right after
  them is consumed by F2.
- qwen Medium: an omitted unit with no letters or digits was silently skipped.
  It is now removed whole.
- qwen Medium: a segment without words could be marked. The writer refuses it.
- qwen Medium: undo before the row was flushed had no `created_at`. Undo now
  flushes and refreshes first.
- qwen Low: the refusal message could read "segments at .". It now falls back
  to "some segments".
- codex, untested: a `clear` after the undo window, and undo after a concurrent
  newer mark, now have tests.

**Rejected.**
- qwen High: translated exports with fillers load marks. The service refuses
  fillers with a translation before marks load.
- qwen High, deepseek Low: TRUNCATE bypasses the trigger. Deliberate, as
  section A says: the test harness truncates every table.
- deepseek High: `clear` is blocked when a correction shrinks `words`. Stored
  `words` never change; corrections change text only.
- deepseek Medium: `omitted` counts units, not words. A unit is one
  whitespace-free word by construction.
- deepseek Medium: an attached ellipsis goes with an omitted word. Intended.
- qwen Medium: `kept` is counted before repeat removal. Keep protects only from
  filler removal (#755), which is what the count reports.

Delta re-reviews (codex): round 1 raised one Low (punctuation after removed
wrappers), fixed; round 2 clean.

### Slice 4a code review (High, 3 of 3: codex, deepseek-v4-pro, qwen/qwen3.8-max-prime)

Slice 4 was split: 4a is the API and backend writes (PR #788, merged as
`e1f97774`), and 4b is the console. Codex implemented 4a from a brief. The full
suite passed before review (11,994 passed; the 3 failures are the known
local-only enrichment gates).

**Fixed.**
- codex, deepseek, qwen: the mark and undo routes committed before building the
  response, so a failing response could hide a committed write. The response is
  now built first.
- A broken saved filler list failed the read. Detection is now empty,
  `detectionError` explains why, and omit, clear and undo still work.
- `version` now hashes focus-independent state, including each emission's shown
  text. A case-only edit moves it; changing focus does not.
- GET reads text, marks and settings in one repeatable-read snapshot.
- A text correction walks the run only when its segment has marks.
- A stale mark on a segment outside the walk raised `KeyError`. Fixed.

**Rejected.**
- qwen High: `clear` depends on `segment.words` bounds. Stored words never
  change for a segment, and a restart that re-transcribes deletes segments and
  their marks by cascade.

Delta re-reviews (codex, then deepseek): the last round was clean.

### Slice 4b code review (High, 3 of 3: codex, deepseek-v4-pro, qwen/qwen3.8-max-prime)

Codex implemented 4b from a brief. Before review, the browser lane on
maintainer hardware found and fixed four issues: a duplicate focused GET when
`j` followed a mark write (a real race; focus reads now wait for the write to
settle), the undo toast copy after a clear, an underline that matched the
speaker label's, and a cleared-marks notice that never went away. A flaky vitest
exposed a real focus race: the first word's autofocus could replace a word
picked right after entering the mode. The full suite passed before review
(11,994 passed; the 3 failures are the known local-only enrichment gates).

Projection cost, measured on the 2,000-segment browser-lane fixture with word
timings on every segment: the word-marks GET takes about 0.6 s (median of 5,
unfocused and focused alike, timed in `build_word_marks_payload`). A mark write
rebuilds the same payload. Acceptable for a single operator; no shortcut added.

**Fixed.**
- codex, deepseek, qwen (qwen High): clean-up autofocus took focus on every
  mount, including from the edit box or a dialog when words arrived late. It
  is now a one-shot request on entry and on a cursor move, skipped while focus
  is in a form control or dialog.
- codex, deepseek, qwen: a failed mark write or undo superseded a pending read
  and scheduled nothing, so the mode could stick on "Loading words…" with
  stale units. A failed write now refreshes once.
- codex: moving between split children, the navigation frame moved focus from
  the word to the row. It now leaves word, form and dialog focus alone.
- deepseek, qwen: leaving the mode by `c`, the toolbar or **Done** dropped focus
  to the page. Every exit now returns focus to the cursor row.
- deepseek Low: with a broken filler list, `f` named the wrong cause. It now
  says detection is unavailable.
- qwen: the request guard kept stale state and outlived the editor. It resets
  on invalidate, and each run's scheduler is disposed on cleanup.
- qwen: normal view split text at every word once a focused payload loaded. It
  now passes only styled units.
- qwen Low: a dead abort timeout on the undo-conflict refresh was removed.
- qwen Low: the broken-list notice showed to read-only viewers.
- qwen Low: the text-save test now asserts the save-time refresh itself.

**Rejected.**
- deepseek Medium: normal-view status is styling and `title` only, so screen
  readers miss it. Adding text would break the byte-identical `data-seg-text`
  contract that selection offsets rely on; clean-up mode's word buttons carry
  the status in their accessible names.

Delta re-review (codex): clean. A browser re-check after the fixes confirmed
entry focus, focus after `j`, focus back on the row after exit, and edit-box
focus kept when words arrive late.

## Completion notes

Completed 2026-10-05 on `main` `d2791f68`. Slices 1 to 4 merged as PRs #785,
#786, #787, #788 (slice 4a, API) and #789 (slice 4b, console).

**Verified.**
- Every slice's outcome is met. The slice 1 and 2 byte-identical gates hold
  (`tests/unit/test_filter_characterization.py`, never regenerated).
- Every acceptance scenario has a test: `tests/unit/test_word_mark_filters.py`,
  `test_word_mark_reporting.py`, `tests/integration/test_word_mark_surfaces.py`,
  `test_word_mark_api.py`, `test_word_mark_restart.py`,
  `test_word_marks_writer.py`, `test_migration_0072.py`,
  `tests/contracts/test_word_mark_storage.py`, and the frontend
  `MediaEditor.word-marks.test.tsx` and `UndoToast.test.tsx`. Three console
  halves are covered by the browser lane rather than vitest: the underline for
  an omission that sets off a phrase (the payload's `removed` flags are
  tested), the unmapped-text reason shown in clean-up mode (the 409 and reason
  are tested), and keep by key on `uh` (vitest uses `um`).
- On `d2791f68` the covering suites pass (2,522 Python tests, 96 frontend
  tests); the full suite passed before the merge (11,994, with the 3 known
  local-only enrichment-gate failures).
- The rejected alternative (marks in `adjudication_decisions`) is pinned out by
  `test_attribution_readers_never_reference_word_marks`.

**Spec files touched:** none; the project declares no living spec.

**Drift (behaviour the plan did not ask for).**
- Slice 4 shipped as two PRs (4a API, 4b console).
- The browser lane reuses the `cleanup` fixture instead of a new `wordmarks`
  fixture; its last segment stores `know.` as two timing tokens so the lane can
  exercise the interior-split refusal.
- A broken saved filler list degrades instead of failing: detection is empty,
  `detectionError` explains it, omit, clear and undo still work, and keep is
  refused (added in the slice 4a review).
- The editor's global keydown handler now ignores events another handler
  already handled (`defaultPrevented`) and IME composition.
- The cleared-marks notice clears on the next cursor move.

**Known limits (maintainer decided to keep).**
- A segment whose text changed beyond case and punctuation, including any
  domain-pack substitution, is unmarkable, and its fillers are not underlined.
- Normal-view highlight status is visual only; clean-up mode's word buttons
  carry it for screen readers.

**Follow-ups:** none required. Next in epic #752 is #758 (maintainer's choice).
