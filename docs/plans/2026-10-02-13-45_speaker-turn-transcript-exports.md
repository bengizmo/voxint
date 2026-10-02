# Plan: transcript exports as speaker turns (#741)

Status: in-progress

Spec deltas: none (this project declares no living spec).

## Goal

A transcript export merges two people into one block. Each whisper segment
(25 to 50 s) gets one speaker label from the diarization turn it overlaps most
(`_dominant_label`, `src/voxint/pipeline/stages/diarize_embed.py`), and
`paragraphize_transcript` (`src/voxint/adjudication/transcript.py`) joins
neighbouring segments that share a name with no length limit. On two real
two-speaker recordings this produced blocks of 132 s and 177 s that held both
speakers under one name. The export also shows `Voice N` until someone names
the voice in the review console, and its layout is a `##` heading plus a
blockquote with millisecond ranges.

Per-word timings (`transcript_segments.words`) and the full turn ledger
(`diarization_turns`) are already stored. Nothing joins them. This plan adds
that join as a read-time projection, makes a speaker-turn layout the default
Markdown export and the read mode, and adds the input paths that close the
remaining gaps: naming, a recording date, per-recording vocabulary and filler
removal. Nothing stored is mutated.

Target layout:

```markdown
# <Title> \| <D Mon YYYY>

[00:00:12] **Alex:** Thanks for making time today.

[00:00:15] **Sam:** Of course. Where should we start?

[00:01:40] A continuation paragraph of a long turn has a timestamp and no name.
```

Reference numbers from a commercial transcription tool on the same two
recordings (structure only): 84 and 173 paragraphs, median 22 to 24 words,
longest about 100 words, one inline marker per minute boundary, no fillers.

## Decisions (maintainer, 2026-10-02)

1. The turn layout becomes the default `md` export. The current layout stays
   available as a legacy style. `txt`, `srt`, `vtt`, `json` and `rttm` do not
   change.
2. In scope: title and date header, minute markers, opt-in filler removal,
   per-recording vocabulary.
3. A CLI naming command.
4. The console read mode adopts the same turns.
5. Not fixed: stutters and cut-off words. Restoring them would mean changing
   ASR behaviour.

## Limits the docs must state

- The diarizer drops turns shorter than 0.5 s before storage
  (`services/pyannote/app/postprocess.py`), so some one-word interjections
  cannot be recovered at export time. Changing that is a diarization-contract
  change and is out of scope.
- Word-level turns need whitespace-delimited text. Other scripts fall back to
  segment level.
- Validated on two-speaker recordings. Three or more speakers are covered by
  synthetic tests only.
- Word-level attribution is a better projection of stored evidence. It is
  still model output.

## Approach

### 1. Projection: `src/voxint/adjudication/turns.py` (new)

A pure core and a thin loader:

```python
project_turns(emissions, turns, *, text) -> list[SpeakerTurn]     # pure
attributed_turns(session, run_id, *, text) -> list[SpeakerTurn]   # loader
```

It composes with `walk_attributions()` and `winning_attribution()`
(`src/voxint/adjudication/attribution.py`), which already handle split
expansion, word-range > segment > label precedence, revocations, merge
tombstones and live roster names. `attributed_transcript`,
`attributed_intervals` (speaker aggregation reads it) and
`paragraphize_transcript` (embedding paragraphs and pull-quotes read it) are
not touched.

A `SpeakerTurn` has an `identity_key` (the canonical speaker id, else the raw
label plus the ruling kind), a display `speaker` (the same rules as
`display_name()`, so exclude and unknown rulings keep their annotation), and
ordered `pieces`. A piece is a verbatim slice of the selected text with
`start`, `end` and `timed`. Consecutive pieces with the same `identity_key`
form one turn, across segment boundaries. Grouping is by identity, never by
display string. Pieces join with the `_join_segment_texts` boundary rule.
Empty pieces are skipped.

Per-emission rules, first match wins:

| Rule | Case | Result |
|---|---|---|
| E1 | Derived split child (an operator cut) | One coarse piece: `child.text` with the child's winning attribution. Never subdivided. |
| E2 | Whole-segment human override | One coarse piece with the override's speaker. |
| E3 | No usable words: `words` is NULL or empty, structural validation fails, or the tokens do not reconcatenate to `raw_text` | One coarse piece with the existing attribution. |
| E4 | Every word resolves to one identity | That identity. Word pieces if the text maps (T1 or T2), else one coarse piece of the selected text. |
| E5 | Two or more identities and the text maps (T1 or T2) | Word-level pieces. |
| E6 | Two or more identities and the text does not map | One coarse piece with the existing attribution. This is today's behaviour for that segment. |

Word validation reuses `_validated_words` and the reconcatenation check in
`src/voxint/adjudication/splits.py` through a `min_words` parameter. The split
feature keeps its minimum of 2 and its correction guards.

**Word to identity.** A word takes the label of the diarization turn with the
greatest positive intersection. Ties go to the lowest `turn_index`, the
existing `_dominant_label` rule. A zero-length word uses point containment in
`[start, end)`. A word inside no turn takes the emission's existing
attribution, with one exception: when the nearest turn-supported words on both
sides share one identity, it takes that identity. Inferred words are never
chained into new evidence. Labels resolve through `label_states()`. There is
no smoothing: a one-word turn is the feature.

**Text to words.** Timings belong to `raw_text`. The variant selector
(`_resolve_body`) does not change.

- T1: the selected text equals the joined words, outer whitespace aside. The
  pieces are the words verbatim.
- T2: whitespace-tokenise the selected text. It maps when the token count
  equals the word count and each token's key equals its word's key. The key is
  NFKC, casefolded, with curly quotes and dashes folded and edge punctuation
  stripped, and must be non-empty. This covers punctuation and casing edits.
- Anything else falls to the coarse branch of E4, or to E6. There is no
  `difflib` alignment and no positional substitution rule.

Order is always `(segment_index, word_index)`. No global sort, no
deduplication, no clamping.

### 2. Reading layout (pure, shared by `md` and read mode)

`ReadingParagraph(speaker, continuation, start_seconds, runs)`.

- P1: a pause of 3.0 s or more between timed pieces starts a paragraph. So
  does time going backwards.
- P2: after a sentence end (`.`, `?`, `!`, closing quotes allowed), a new
  paragraph starts once the current one has 20 or more words (60 as first
  planned; see the slice 1 implementation notes). A coarse piece
  breaks only at its own boundaries. No maximum length is promised.
- Minute markers: inside a paragraph, before the first timed piece that
  crosses into a new minute, emit that minute's `[HH:MM:00]`. A paragraph
  start consumes every minute up to its own timestamp, so markers never stack
  and never go backwards. A coarse piece carries no interior marker.
- Timestamps are `[HH:MM:SS]`. `--no-timestamps` drops paragraph timestamps
  and minute markers.
- The thresholds are module constants, not settings.

### 3. Format surface

- `to_markdown` is renamed `to_markdown_blocks` with unchanged bytes. It stays
  the body of annotation pull-quotes. The "byte-identical to the transcript
  export" claim in its docstring and in `docs/annotations.md` becomes
  "identical to `--style blocks`".
- New `to_markdown_turns(paragraphs, header, *, timestamps)`, reusing
  `_md_escape`, `_normalize_line_breaks` and `_md_defuse_block_start`. One
  physical line per paragraph.
- The header is composed in the export layer: `# <Title> \| <D Mon YYYY>`, or
  the title alone when no date is known. The title comes from
  `title_from_snapshot` and `friendly_media_label`
  (`src/voxint/api/presentation.py`).
- `--style turns|blocks` on the CLI and `style=` on the console download and
  `/api/v1` routes, for `md` only. Omitting it selects the format's default.
  Passing it with another format is an error.
- One shared entry point, `src/voxint/export/service.py` (new), called by the
  CLI `_export`, the console download dispatcher in `adjudication_api.py` and
  `api_v1/transcript.py`, so CLI and HTTP byte parity holds by construction.
- Translated `md` (`lang=`): the freshness check and the original line
  attribution are kept, and each translated line becomes one coarse piece.
  Word projection is bypassed on this branch.

### 4. Read mode

The read branch of `run_transcript` in `legacy_runs.py` and
`legacy_runs/transcript.html` render the same reading paragraphs. It stays
server-rendered with no island. It lands in the same PR as the default change
so the two never disagree.

### 5. Naming: `voxint speakers name <run> [<voice> "<Name>"]`

With `<run>` alone it lists the run's voices: label, current name, how the
name was resolved, and talk time. With a voice and a name it runs one
transaction:

1. Lock the run row `FOR UPDATE`. Refuse if the run is not completed or any
   live review claim exists, the same reviewer's included, because `claim_run`
   would rotate a console tab's token. A new helper in
   `src/voxint/adjudication/slots.py` owns this check. No claim is written, so
   a crash leaves nothing behind.
2. Resolve `<voice>`: the raw label, else a unique current display name among
   the run's labels.
3. Branch:
   - the name is an active roster speaker: `record_decision(ASSIGN)`, with the
     route's speaker lock and active check;
   - the label's speaker is a placeholder (created by an `AUTO_ENROLL` ledger
     decision and still carrying its generated name): `rename_speaker()` plus
     a human ASSIGN. The command reports how many other recordings that voice
     appears in, because the rename applies there too;
   - otherwise: `enroll_new_speaker()`.
4. A fresh nonce per invocation. The operator is the configured operator
   identity. The route's activity record is mirrored. The output states each
   effect, and that undoing the assignment does not undo a rename.

### 6. Fillers: `--drop-fillers` and `fillers=drop`

For the `md` turn style only. Standalone `um`, `uh`, `umm`, `uhh`, `uhm` and
`erm` (English) are removed with a following comma, before paragraphing. A
sentence start is re-capitalised. A turn left empty is dropped without handing
its speaker to a neighbour. Stored text is untouched. Other formats, the
blocks style and `lang=` reject the option.

### 7. Recording date

- `media_items.recorded_on DATE NULL`, with a migration. PREPARE runs a
  bounded, metadata-only ffprobe of the resolved source file. Today only the
  normalized audio is probed. Only a tag with an explicit UTC offset is used
  (`com.apple.quicktime.creationdate`). A UTC-only `creation_time` can be a
  day off and is ignored. First write wins. A missing or malformed tag yields
  no date and never fails the stage. The docs call it the file's creation
  date.
- Sidecar `recorded: YYYY-MM-DD` overrides it at render time, read from the
  run's frozen snapshot. It is never written to the column.
- `voxint media backfill-recorded-dates`: NULL rows only, files still on disk,
  with `--dry-run`.

### 8. Per-recording vocabulary

A bounded sidecar `vocabulary:` list, unioned after effective vocabulary
resolution in `src/voxint/ingest/service.py` and recorded in the config
provenance. The pinned faster-whisper keeps only the last 223 prompt tokens,
while `_initial_prompt` keeps the earliest terms up to 2000 characters. So the
budget is reserved for sidecar terms first, and pack terms are serialised
before sidecar terms. With no key, the frozen snapshot and the ASR request are
byte-identical to today on every resolution path. The existing cap exceeding
the engine window is tracked in #743 and is not fixed here.

## Acceptance scenarios

**Projection**

- WHEN a segment's words fall in two speakers' turns and the selected text is
  the raw text, THEN the export has one turn per run of same-speaker words, in
  word order.
- WHEN a segment has a whole-segment or word-range human assignment, THEN its
  text appears under that speaker as one piece, however its words fall.
- WHEN the selected text differs from the raw text only in punctuation and
  casing, THEN turns are word-level and the output text is the selected text.
- WHEN the selected text was rewritten and the words span two speakers, THEN
  the segment appears whole under its existing speaker.
- WHEN a label is ruled exclude or unknown, THEN its words keep the
  `(excluded)` or `Unknown` annotation and are never shown under another name.
- WHEN any fixture is projected, THEN the joined pieces equal the selected
  text apart from outer whitespace.

**Format**

- WHEN `voxint export --format md` runs with no style, THEN the output is the
  turn layout with a title header.
- WHEN `--style blocks` is passed, THEN the bytes equal today's Markdown
  export.
- WHEN `--style` is passed with a format other than `md`, THEN the command
  fails with a clear message.
- WHEN the same run is exported over the CLI and over HTTP, THEN the bytes are
  identical for both styles.
- WHEN a paragraph crosses a minute boundary, THEN exactly one `[HH:MM:00]`
  marker appears at the crossing, and none appears at a paragraph start.

**Naming**

- WHEN a run has a live review claim, THEN `voxint speakers name` refuses and
  changes nothing.
- WHEN a voice is an auto-enrolled placeholder and the name is new, THEN the
  placeholder is renamed, the label is human-assigned to it, and the command
  reports how many other recordings carry that voice.
- WHEN a voice is named A, then B, then A again, THEN each call records a new
  ruling and the final speaker is A.

**Inputs**

- WHEN a sidecar has no `vocabulary:` key, THEN the frozen snapshot and the
  ASR request are byte-identical to today.
- WHEN a source file carries only a UTC `creation_time`, THEN no recording
  date is stored and the header shows the title alone.

## Slices

Each slice is its own PR. Every slice is the High review tier (export
contract, ledger writes, a migration, an ASR input).

| Slice | Content | Gates beyond lint, types, tests and CI |
|---|---|---|
| S1 (PR #745) | Projection, reading layout with minute markers, `to_markdown_turns`, shared export entry, `--style turns` as an opt-in with blocks still the default | Old goldens untouched, new goldens, CLI and HTTP parity for both styles |
| S2 | Default change, header title, read mode, pull-quote contract wording, docs, CHANGELOG Changed entry naming `--style blocks` | Browser acceptance lane, measured numbers on real recordings |
| S3 | `voxint speakers name` | Integration tests on a real database |
| S4 | Fillers | |
| S5 | Recording date: migration, PREPARE probe, sidecar key, backfill | Migration up and down |
| S6 | Sidecar vocabulary | Absent-key contract test in the same commit, parity gates |

## Testing strategy

Fixtures are synthetic timed tokens and turns only. Real recordings and their
text never enter the repo, an issue or a PR.

- S1 round-trip invariant on every fixture, asserted on the rendered bytes as
  well as the in-memory paragraphs. Tokens are whisper-realistic: leading
  spaces, punctuation-glued and punctuation-only tokens, contractions, curly
  quotes, hyphen splits, zero-length words on a boundary, empty word buckets,
  empty selected text, text with no whitespace.
- S1 adjudication: range > segment > label, a revoked override, merged
  speakers with different display names, exclude and unknown labels, a split
  parent with an override, three speakers, uncovered words (bracketed and
  not), overlapping segments with backwards times, a coarse piece spanning
  several minutes, a minute boundary at a paragraph start, translated `md`
  with a split parent.
- S3: any live claim refused (same reviewer included), each branch, A then B
  then A, a name collision with a merged or archived speaker, no eligible
  audio, rollback after a rename, undo leaving the name in place.
- S6: an absent key across explicit, project, folder and global resolution.

## Verification

- S2, on two real two-speaker recordings on maintainer hardware, numbers only:
  the share of words at word level against coarse, per rule and per text
  variant; paragraph counts against 84 and 173; the longest paragraph; a
  manual read that finds no paragraph holding two speakers outside E6
  segments.
- S2 browser lane through the `voxint-e2e-review` skill.
- The release is a MINOR bump, done by the release process.

## Review notes

Consulted 2026-10-02 before finalizing: codex, deepseek, glm and kimi. All
four answered.

Adopted:

- Verified and corrected segments are not frozen at segment level (4 of 4).
  Verification is workflow state, and freezing would make exports coarser the
  more an operator reviews.
- No positional substitution rule. Codex gave a counter-example in which
  reordered phrases pass boundary anchors; glm agreed, deepseek agreed
  conditionally, kimi dissented.
- E4 keeps word pieces when the text maps (codex). The shared validator keeps
  the split minimum (codex, kimi). The join rule and the round-trip invariant
  (kimi). The translation bypass (codex, kimi).
- Refusing any live claim under a row lock, a fresh nonce per invocation, and
  placeholder detection by ledger origin (codex, kimi). A deterministic
  idempotency key was rejected because it breaks A then B then A.
- Vocabulary ordering against the engine's tail window (codex verified it
  against the pinned source; kimi and glm concurred).
- S1 ships opt-in, and the default change ships together with read mode (glm,
  codex).
- Marker stacking, backwards time, empty pieces and punctuation-only tokens
  (deepseek).

Splits and the choice made:

- Words inside no turn (2 to 2). Deepseek and codex preferred the segment's
  speaker; glm and kimi preferred the neighbouring word's. Chosen: the
  segment's speaker, plus the bracketed-gap exception.
- Excluded voices. Kimi wanted an excluded voice's interjections folded into
  the surrounding speaker. Not adopted: that attributes words to the wrong
  person, and today's export already annotates excluded voices.
- A footer line when any segment fell back to segment level (glm only). Not
  adopted. The docs carry the caveat.
- The backfill command (3 to 1 to keep).
- `fillers=drop` over HTTP (glm would ship the CLI flag only). Kept for CLI
  and HTTP parity. No console menu control in this plan.

Follow-ups filed: #742 (show projected turn boundaries in the editor) and #743
(the vocabulary prompt cap against the engine's token window).

### Slice 1 implementation notes (2026-10-02)

Refinements made while implementing, all inside the rules above:

- **Word units.** A stored token with no leading whitespace is glued to the
  token before it, and the unit is attributed and rendered whole. Without
  this a speaker change, a paragraph break or a minute marker could land
  inside a word. T2 compares whitespace tokens to units, which equals the
  word count whenever every token has a leading space (the normal case).
- **Bracketed gaps.** The search for the nearest turn-supported words on both
  sides of an uncovered word may cross a segment boundary. It stops at an E1,
  E2 or E3 emission. A within-segment search would hand an uncovered word at
  a segment edge to the segment's speaker and create one-word turns there.
- **`timed`.** P1 (the pause and backwards-time rule) and minute markers use
  every piece's start and end, coarse pieces included, because a coarse
  piece's interval is real. `timed` only records whether a piece carries its
  own word timing. A marker is placed before a piece and never inside one.
- **Translated `md`.** The route validates the translation and passes the
  texts to the export entry point, which walks the emissions itself, so
  translated turns keep identity-grade grouping.
- The existing split validator rejects a segment when a word falls outside
  the segment's interval or word starts go backwards. A numbers-only check on
  the two reference recordings found 3 of 113 segments affected, so the
  validator is reused unchanged.
- **P2 threshold: 20 words, not 60.** Measured on the two reference
  recordings with the branch code (corrected text, 3.0 s pause). The reference
  tool has 173 and 84 paragraphs with a median of 22 to 24 words.

  | Minimum words | Paragraphs | Median words | Longest |
  |---|---|---|---|
  | 60 | 110 and 51 | 53.5 and 46 | 111 and 163 |
  | 40 | 129 and 60 | 43 and 41 | 111 and 118 |
  | 30 | 149 and 68 | 34 and 34 | 111 and 118 |
  | 20 | 179 and 82 | 24 and 25 | 111 and 115 |

  Shortening the pause instead (2.0, 1.5, 1.0 s) added very short paragraphs
  and did not bring the median down until 1.0 s. The pause stays at 3.0 s.
- **Early measurement** (same run): 97% and 100% of words are attributed at
  word level under raw text, 94% under enhanced text. Speaker turns rose from
  28 and 14 single-name blocks to 67 and 30. The coarse remainder is two
  segments that fail word validation and three reworded two-speaker segments
  (E6). The S2 gate repeats this on the final code.
- **Minute marker placement.** "Crosses into a new minute" is read as "starts
  in a new minute". A word that straddles the boundary stays before the
  marker, so everything after a marker starts at or after it, the same rule
  paragraph timestamps follow.
- The pull-quote contract wording moved from S2 into S1, because the rename
  to `to_markdown_blocks` lands here.
