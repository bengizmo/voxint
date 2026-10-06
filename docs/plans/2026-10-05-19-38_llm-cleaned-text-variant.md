# Plan: optional LLM clean-up as a separate `cleaned` text variant (#758)

Status: draft

Spec deltas: none (this project declares no living spec). The maintainer
decided D1 to D6 below on 2026-10-05 (to be recorded on #758).

## Goal

Epic #752's deterministic tiers are done: #753 presets, #754 read mode, #755
repeats, #757 per-word marks. #758 adds an **opt-in, LLM-proposed,
deletions-only** clean-up of the reviewed transcript, aimed at the contextual
fillers that fixed lists cannot catch: `I mean`, `you know` used as a filler,
hedges and false starts. The result is a separate text variant, `cleaned`.
It:

- is generated only when the operator asks, through the existing local LLM
  client;
- is built only from source words: the LLM chooses which words to drop and
  never contributes characters;
- carries provenance (model, prompt version, effective filler list, source
  hash) and the counts of rejected proposals by reason;
- goes stale when the reviewed transcript changes, and its export then refuses
  with the same shape of message as a stale translation;
- is never the default and never in the corrected text's fallback ladder. It
  is reachable only by an explicit `text=cleaned` on the console export routes;
- is shown in the console as a diff against the reviewed text, with removed
  words struck through.

Default exports and subtitles stay byte-identical with or without a cleaned
variant.

**Maintainer decisions (2026-10-05):**

- **D1 Validator.** The LLM returns each line cleaned. Voxint checks that the
  proposal's normalized words are an in-order subsequence of the source line's
  words. The variant is then built from the SOURCE words. Any substitution or
  insertion drops the whole line's proposal: the line stays verbatim and the
  drop is counted as rejected.
- **D2 Protected words.** These are never deleted:
  - tokens containing a digit;
  - English number words;
  - negations;
  - likely names: a token whose first letter is uppercase and that does not
    start a sentence, except `I` and its contractions.

  A deletion of one rejects the line.
- **D3 Standalone.** `text=cleaned` refuses `fillers=drop`, `repeats=drop` and
  `lang=` with 422, like a translation does. The prompt sees the effective
  filler list as hints. The list is provenance and is not part of the
  staleness hash. The operator's #757 keep and omit marks are ignored by the
  cleaned variant.
- **D4 Console.** A server-rendered Cleaned page with no new island and no new
  JSON API. It shows the reviewed text with removed words struck through, a
  stale banner, counts, Generate and Cancel, and download links.
- **D5 Turns attribution.** Deletions are anchored to #757 word identities
  `(segment_id, token_start, token_end)`. Turns exports apply them as omits
  through the normal attribution projection and filler pass, so speaker
  attribution is exactly that of the reviewed turns export. A line whose
  deletions cannot be anchored is rejected as `unanchored`. This is the same
  limit as #757's unmarkable segments.
- **D6 Language.** English only. Runs detected as another language are
  refused. Runs with no detected language are allowed, and the page says the
  language was not verified as English.

**Binding epic decisions still in force:** stored text never changes; there is
no audio editing. The rule that `like so right well` are never removed applies
to *preset rules*. The LLM may propose them in context; that is the point of
this tier, and D1 and D2 bound it.

## Ground truth (surveyed 2026-10-05, main `0ed8c4f4`)

- **LLM client** (`src/voxint/clients/llm.py`).
  - `HttpLLMClient.chat_json(messages) -> dict` (:323). The prompt demands a
    JSON object envelope; fences are stripped and replies are capped at 100k
    characters.
  - It raises the single `LLMError` for transport failures *and* for reply
    shape failures (invalid JSON, non-object, NUL, oversize). Callers cannot
    tell the two apart without parsing messages.
  - Gates come from `app_settings.resolve_effective_llm_*`; the batch knobs are
    `LLM_ATTEMPTS_PER_BATCH`, `LLM_BATCH_MAX_SEGMENTS` and
    `LLM_BATCH_MAX_CHARS`.
- **Translations (#133), the closest analogue.**
  - `enrichment/translations.py`:
    - `load_translation_source` freezes the CORRECTED `attributed_transcript`
      lines with their segment and word coordinates.
    - `translation_source_hash` hashes those lines plus the run id and the
      source language.
    - The writer `record_translation` takes an advisory lock, allocates a
      generation, supersedes the old head and replays idempotently.
  - Job module `translation_jobs.py`. Producer `producers/translation_llm.py`:
    an index-echoed reply envelope, retry and then bisect, and a line that
    still fails fails the job.
  - Tables `translation_jobs` and `run_translations` (migrations 0038, 0047).
    The source-changed race guard is documented as best-effort.
  - Celery `voxint.translate_run` runs on the default queue. Touchpoints:
    recovery sweep (`worker/tasks.py:533`), GPU-phase LLM-pending list
    (`gpu_phase/state.py:312`), orchestrator republish
    (`gpu_phase/orchestrator.py:470`), jobs page (`api/jobs_query.py:165,204`).
  - Translation controls moved from the run page to the editor in #567
    (`test_run_detail_excludes_translation_card`).
- **Translation export** (`api/routers/adjudication_api.py:1405-1525`).
  - `_export_translated_texts` refuses with 409 when no translation exists, one
    is running, or it is stale, and with 422 for a wrong variant.
  - `render_run_transcript_report(translated_texts=...)` substitutes per line.
  - The turns layout goes through `coarse_turns` (`adjudication/turns.py:555`),
    which deliberately bypasses word evidence: each line gets one speaker.
    That is why D5 exists.
- **#757 machinery.**
  - `emission_anchors(emission, text=CORRECTED)` (`turns.py:383`) gives each
    displayed line's word units as `WordAnchor(segment_id, token_start,
    token_end, lex_start, lex_end)` offsets into the shown text, or an
    unmarkable reason.
  - `apply_turn_filters(turns, fillers=, drop_repeats=, marks=)` places omits,
    raising `UnplaceableWordMarkError` as a backstop.
  - `export/fillers.py` `_clean_pass` applies F2 (mark moving), F3 (separator
    collapse) and F4 (capitalise) to arbitrary removal spans; `_omit_spans`
    builds spans from owned characters.
  - `_token_key` (`turns.py:297`) normalizes case, quotes and hyphens with
    NFKC.
- **`TranscriptText` enum** (`adjudication/transcript.py:24`). Adding a member
  would also add a read-mode tab, an `api_v1` value, a `segment_embeddings`
  precedence entry and CLI `choices` drift. The export-menu HTML is a byte
  golden (`tests/integration/test_runs_api.py:1807`).
- **Migrations.** The latest is `0072`, so the new one is `0073`; it must bump
  the alembic head pin in `tests/integration/test_migration_0016.py`.

## Assumptions & constraints

- **Source:** exactly the translation source, the CORRECTED attributed lines.
  The source hash reuses `translation_source_hash` unchanged. It includes the
  detected language, which changes only on re-transcription, and that already
  creates new segments and stales the variant.
- **Lines:** emissions from `walk_attributions` and `attributed_transcript`
  lines correspond one-to-one in order. Translations rely on this too, and a
  test pins it for cleaned.
- **Anchors:** a deletion that coincides with whole #757 anchors is placeable
  in the turns projection. #757's console and export already depend on this
  invariant (the export backstop is `WordMarkPlacementError`).
- **No new knobs:** no new env var or setting, no auto-generation after
  finalize, no CLI and no `/api/v1` support in v1 (translations have neither).
  Both keep rejecting `cleaned` as an unknown variant, and that is tested.
- **Invalidated if:** the maintainer wants stacking with the deterministic
  filters, the LLM's own punctuation, or per-word partial acceptance within a
  line.

## Proposed approach

### Pure core: `enrichment/cleanup.py` (no DB)

- **One normalization authority**, `cleanup_key(token)`, reusing `_token_key`:
  NFKC, casefold, curly to straight quotes, Unicode hyphens. Edge punctuation
  is trimmed, then every remaining non-alphanumeric character is dropped
  before matching. So `don't`, `don’t` and `dont` match, and `twenty-one`
  matches `twentyone`.
- **Source words per line are the line's #757 anchors**, using the lexical
  slice `text[lex_start:lex_end]` with its key. A line with no anchors is
  *unanchorable*: any proposed change to it is rejected as `unanchored`. It is
  still sent to the LLM as context.
- **Protection** (D2) is judged on raw token shapes:
  - a digit anywhere;
  - number words, including hyphenated parts (`twenty-one`);
  - negations (`not no never nor none nobody nothing nowhere neither cannot`,
    plus any token whose key ends in `nt` from an `n't` or `n’t` form);
  - likely names: first letter uppercase and not sentence-initial, where
    sentence-initial means the first word of the line or a word after `. ? !`;
    `I` and its contractions are exempt.
- **`validate_proposal(words, proposed_text)`** returns
  `Accepted(deleted: frozenset[int])`, `Unchanged`, or `Rejected(reason)`:
  - Proposal keys (empty keys ignored) must form an in-order subsequence of
    the source keys, otherwise `not_deletion`.
  - **Alignment is protection-preferring and deterministic.** A small DP over
    the line looks for an alignment that deletes no protected word. Ties go
    to keeping the earliest source occurrences. If every valid alignment
    deletes a protected word, the result is `protected`.
  - Deleting every word gives `whole_line` (O3, kept as rejection in v1).
  - Precedence: `not_deletion` > `protected` > `whole_line`. An empty proposal
    for a non-empty line is `whole_line`; an identical one is `Unchanged`.
- **`render_cleaned(text, words, deleted) -> str`** goes through a new public
  seam, `fillers.delete_spans(text, spans)`. It builds one owned-character run
  per line and an omit-style `_RemovalSpan` per deleted anchor, then runs
  `_clean_pass` with marks and filler lists disabled. Punctuation and
  capitalisation therefore follow the `fillers=drop` rules. The rendered text
  is stored per line, and exports never re-render it.
- **Prompt.** `CLEANUP_PROMPT_VERSION = 1`. It uses translation's index-echoed
  envelope (numbered lines in; an array of `{"index", "text"}` out, with full
  coverage). It lists the effective filler list as examples, states the rules
  (delete only; never change numbers, names or negation), and caps each reply
  line at the source length plus a small slack.

### Typed LLM reply errors (`clients/llm.py`)

- Add `LLMReplyError(LLMError)` for shape failures: an invalid envelope,
  non-JSON, a non-object, NUL, or an oversized reply. Transport, deadline and
  HTTP failures stay plain `LLMError`.
- Existing `except LLMError` callers are unaffected because it is a subclass.
- This seam lets the producer keep malformed lines verbatim without failing the
  job, and without parsing messages.

### Storage (migration 0073)

There are two new tables, mirroring translations: a mutable job row and an
immutable result row.

- **`cleanup_jobs`.**
  - Columns: id, pipeline_run_id, status (queued, running, succeeded, failed,
    cancelled), cancel_requested, error, `config` JSONB, enqueue-time
    `source_content_hash`, `cleanup_id`, and timestamps.
  - `config` is the request snapshot: endpoint, model, batch knobs, prompt
    version and filler list. It records what was *requested*.
  - A partial unique index allows one active job per run.
- **`run_cleanups`.**
  - `lines` JSONB, one entry per source line:

    ```json
    {"i": 0, "segment_id": "...", "word_start": null, "word_end": null,
     "source": "<frozen shown text>", "text": "<rendered cleaned text>",
     "deleted": [[token_start, token_end, lex_start, lex_end], ...],
     "outcome": "changed|unchanged|rejected", "reason": null}
    ```

    The frozen `source` and lexical ranges let the Cleaned page draw the diff
    from the snapshot even when the variant is stale, labelled as historical.
    Old offsets are never applied to current words.
  - `counts` JSONB holds `lines_changed`, `words_removed`, and `rejected` by
    `not_deletion | protected | whole_line | unanchored | malformed`. Counts
    are final line outcomes and never count retry attempts.
  - `config` JSONB holds `prompt_version`, the filler-list snapshot
    `{preset_version, words, phrases, kept}` and the batch knobs. **This is the
    one provenance record of the list.**
  - Translation-style columns: `generation` (unique per run),
    `payload_schema_version`, producer, producer_version, model,
    `source_content_hash`, `idempotency_key`, `replay_digest`,
    `superseded_by_cleanup_id`, started_at, completed_at.
  - Check constraints mirror `run_translations`. A trigger allows only a
    one-time set of `superseded_by_cleanup_id`.
  - Bounds: `MAX_LINES` and a payload byte cap, as for translations.
- **Writer `record_cleanup`.**
  - It mirrors `record_translation` (advisory lock per run, generation,
    supersede, idempotent replay).
  - It independently re-validates every line: deleted ranges are anchors of
    the frozen source, no protected word is deleted, and rendered text equals
    `render_cleaned(...)`. It derives `counts` itself; the producer's counts
    are never trusted.
  - Replay equality covers lines (by digest), config, counts and provenance.

**Rejected alternatives.**

| Alternative | Rejected because | It would win if |
|---|---|---|
| A: reuse `run_enrichment_assets` with a `cleanup` kind | Its source hash covers metadata, notes and diarization labels, which is the wrong freshness domain. Assets also auto-generate after finalize and render in the assets card. | never, here |
| B: reuse `run_translations` with a pseudo language, or a discriminator column | It overloads the per-language head semantics and the language check, so every reader would branch on a magic value. | never, here |
| C: generalize translations into typed "transcript renditions" (codex) | It is a larger migration that rewrites a shipped table, just to share two producers. | a third rendition becomes real |

The two tables still share narrow helpers where shape is identical: source
loading, the batching loop, claim, cancel and finalize. No generic framework.

### Producer and job (`enrichment/producers/cleanup_llm.py`, `enrichment/cleanup_jobs.py`)

- **Batching:** translation's batching, retry and bisect. Translation's
  failure rules change as follows:
  - An `LLMReplyError` or a misaligned reply that persists down to a single
    line marks that line `malformed` (kept verbatim).
  - A plain `LLMError` (transport) that survives retries fails the job.
  - A generation in which **no line received a parseable reply** fails the job:
    an all-verbatim "success" from a broken model would be dishonest. A run
    where the model parsed fine but proposed nothing, or proposed only
    rejected deletions, is an honest success that reports its counts.
- **Cancel:** checked before *and after* every LLM call and immediately before
  `record_cleanup`; a cancelled job stores nothing.
- **Language gate (D6):** checked at enqueue and rechecked at claim.
- **Race guard:** the source hash is recomputed immediately before persisting;
  a mismatch fails the job with the source-changed error. As in translations,
  this is best-effort: an edit committed between the check and the insert
  leaves a generation that is simply stale at read time. Every reader
  recomputes the hash, so no stale text is ever served.
- **Celery:** task `voxint.cleanup_run` goes on the **same queue as
  translation** (the default queue), never `post`, so minutes-long LLM work
  cannot delay `finish_pipeline` or asset jobs. It is pinned in
  `task_inventory.json`. The recovery-sweep republish of stale queued jobs, the
  GPU-phase LLM-pending list, the orchestrator republish and the jobs page
  family are all extended.

### Export: `text=cleaned` (console routes)

- `TranscriptText` is **not** extended. A contract pins it to {corrected,
  enhanced, raw}. `_export_transcript` takes `text` as a string;
  `cleaned` is recognised by an explicit allow-list before
  `parse_transcript_text`.
- **Refusals:**
  - 422 when combined with `lang=`, `fillers=drop` or `repeats=drop`; the
    wording parallels translation's.
  - 409 when no generation exists, one is still being generated, or the
    variant is stale. The stale detail follows the translation message:

    ```
    the cleaned variant is out of date — the transcript changed since it was
    generated; regenerate it from the Cleaned page and retry
    ```
- **Segment-shaped formats** (txt, md blocks, srt, vtt, json): stored per-line
  `text` is substituted through the existing `translated_texts` parameter. It
  is not renamed: an internal name is not worth churning tested translation
  paths, so the docstring gains one line instead. Cue timing is untouched.
- **Turns layout (D5):** `attributed_turns(text=CORRECTED)` plus
  `apply_turn_filters(fillers=<empty list>, drop_repeats=False, marks=<cleaned
  omits>)`. The cleaned omits are a `WordMarkKey -> "omit"` map built from the
  stored `deleted` ranges. Speaker attribution is identical to the reviewed
  turns export. `UnplaceableWordMarkError` maps to 409 as a backstop. This
  needs `apply_turn_filters` to accept marks without a non-empty filler list:
  a small, tested change that leaves today's paths untouched.
- **Freshness and rendering:** the hash check and the rendering read the same
  transaction snapshot (REPEATABLE READ, the #757 GET pattern). An edit
  between the two cannot mix generations.

### Console (D4): the Cleaned page

- **`GET /runs/{id}/cleanup`** is server-rendered (`legacy_runs/cleanup.html`):
  - the status line and the D6 language note;
  - Generate and Cancel forms, with CSRF tokens `CSRF_CLEANUP_GENERATE` and
    `CSRF_CLEANUP_CANCEL`. Cancel stays available even when the LLM has since
    been disabled;
  - the counts by reason, model, prompt version and the filler list used;
  - the stale banner;
  - the diff: each line's frozen `source` with deleted lexical ranges wrapped
    in `<del class="cleanup-removed">`. Rejected lines are shown plain, with a
    muted reason tag;
  - download links (`text=cleaned` per format), hidden when stale.
- **Polling:** a small separate status fragment, polled while the job is
  active, using the existing server-rendered polling pattern rather than an
  island.
- **Entry points:** an "LLM clean-up" row under the export menu's "Other text
  variants", plus a link from the transcript read page.
  - The row renders **only when the LLM is enabled or a generation exists**, so
    the existing byte goldens stay unchanged for the gated-off fixture.
  - It links to the Cleaned page, never straight to a download, so a stale
    variant cannot produce a dead link.
  - New goldens cover the row-present case.
- No run-detail card: #567 moved enrichment controls off the run page, and the
  Cleaned page is the single home.

## Acceptance criteria

### Requirement: deletions only

The system SHALL accept an LLM proposal for a line only when it is a pure
deletion of non-protected, anchorable source words. It SHALL build the cleaned
line from the source words.

#### Scenario: substitution rejected

- WHEN the LLM returns `we went to the stores` for source `we went to the store`
- THEN the line stays verbatim and `rejected.not_deletion` is incremented

#### Scenario: insertion rejected

- WHEN the proposal contains a word absent from the source at that position
- THEN the line stays verbatim and `rejected.not_deletion` is incremented

#### Scenario: protected word

- WHEN every valid alignment of the proposal drops `not`, `didn't`, `15`,
  `twenty-one` or a mid-sentence `Sarah`
- THEN the line stays verbatim and `rejected.protected` is incremented

#### Scenario: protection-preferring alignment

- GIVEN source `Mark said, Mark it down` with proposal `Mark it down`
  (the first `Mark` is sentence-initial and unprotected; the second is a
  protected mid-sentence name)
- THEN the alignment keeps the second `Mark` and deletes `Mark said,`, so
  the line is accepted. Earliest-first greedy would have deleted the protected
  `Mark` and rejected the line.
- AND GIVEN source `No, no, I said no` with proposal `no I said no`
- THEN every alignment deletes a negation, so the line is rejected `protected`
- AND GIVEN source `you know I know you know` with proposal `I know you know`
- THEN the first `you know` is deleted (earliest occurrences kept on ties)

#### Scenario: LLM punctuation ignored

- WHEN the proposal is `i think its fine` for `I mean, I think it's, uh, fine.`
- THEN the stored line is built from source words under `fillers=drop`
  punctuation rules, never from the LLM's characters

#### Scenario: whole line

- WHEN a proposal deletes every word of a line
- THEN the line stays verbatim and `rejected.whole_line` is incremented

#### Scenario: unanchored

- WHEN a line has no #757 anchors (for example a domain-pack substitution) and
  the proposal changes it
- THEN the line stays verbatim and `rejected.unanchored` is incremented

#### Scenario: writer refuses a forged generation

- WHEN `record_cleanup` is given a deleted range that is not an anchor, a
  protected deletion, or text that differs from `render_cleaned`
- THEN it raises and nothing is stored

### Requirement: provenance and staleness

The system SHALL store each generation with its model, prompt version,
effective filler list, source hash and final per-reason rejection counts. It
SHALL treat the generation as stale whenever the recomputed source hash
differs.

#### Scenario: stale export refusal

- GIVEN a cleaned generation
- WHEN a segment's corrected text is edited, or a split or unsplit happens
- THEN `GET /review/{id}/export.txt?text=cleaned` returns 409 with the
  translation-shaped stale message
- AND the Cleaned page shows the stale banner and the historical diff, with no
  download links

#### Scenario: speaker rename does not stale

- WHEN a speaker is renamed or reassigned
- THEN the cleaned variant stays current, and turns exports show the new
  attribution

#### Scenario: filler list or marks change does not stale

- WHEN the operator changes the filler list or adds a #757 mark
- THEN the variant stays current, and its provenance still shows the list it
  was generated with

### Requirement: explicit selection only

The system SHALL serve the cleaned variant only for `text=cleaned` on console
export routes. It SHALL never use the variant as a default or fallback.

#### Scenario: defaults byte-identical

- GIVEN one run exported in every format, style and timestamps combination:
  first with no generation, then with a current generation, a superseded one,
  and a coexisting translation
- THEN the exports for `text` omitted, `corrected`, `enhanced` and `raw`,
  subtitles included, are byte-identical across all four states

#### Scenario: turns attribution matches reviewed turns

- GIVEN a segment whose words split between two speakers
- WHEN exported as md turns with `text=cleaned`
- THEN the speaker grouping equals the `text=corrected` md turns export, minus
  the deleted words

#### Scenario: incompatible options

- WHEN `text=cleaned` is combined with `lang=`, `fillers=drop` or `repeats=drop`
- THEN the response is 422

#### Scenario: no generation, or one running

- WHEN no generation exists, or one is running
- THEN the response is 409 with a "generate one from the Cleaned page" or
  "still being generated" message

#### Scenario: API and CLI unchanged

- WHEN `/api/v1/runs/{id}/transcript?text=cleaned` or `voxint export --text
  cleaned` is used
- THEN both are refused as an unknown variant, as today

### Requirement: opt-in job

#### Scenario: gated

- WHEN the LLM is disabled
- THEN the Generate form is hidden and the generate route refuses with 409 and
  an honest message

#### Scenario: language

- WHEN the run is detected as a non-English language
- THEN generate refuses and names the English-only limit
- AND WHEN no language was detected, THEN generate works and the page says the
  language was not verified as English

#### Scenario: cancel

- WHEN cancel is requested mid-run
- THEN no further LLM call is made, a reply that is already in flight is
  discarded, and no generation is stored

#### Scenario: broken model

- WHEN no line receives a parseable reply
- THEN the job fails with an honest error and stores nothing
- AND WHEN some lines are malformed and others parse, THEN the generation is
  stored and the malformed lines are counted

#### Scenario: source changes mid-job

- WHEN the transcript changes before the final hash check
- THEN the job fails with the source-changed error
- AND WHEN it changes after that check, THEN the stored generation reads as
  stale everywhere

## Affected files / components

- **Migration and models:** `alembic/versions/0073_transcript_cleanup.py`
  (new); `db/models.py` gains `CleanupJob`, `RunCleanup` and a status enum.
  `tests/integration/test_migration_0016.py` gets the head pin bump.
- **LLM client:** `clients/llm.py` gains `LLMReplyError`.
- **New enrichment modules:**
  - `enrichment/cleanup.py`: the pure core (normalization, protection,
    alignment DP, render).
  - `enrichment/cleanups.py`: the writer, read side, and anchor-based source
    loading.
  - `enrichment/producers/cleanup_llm.py` and `enrichment/cleanup_jobs.py`.
- **Filter seams:**
  - `export/fillers.py` gains the public `delete_spans` seam.
  - `export/turn_filters.py`: marks without a filler list.
- **Worker:** `worker/tasks.py`, `gpu_phase/state.py`,
  `gpu_phase/orchestrator.py`, `api/jobs_query.py`.
- **Routes:**
  - `api/routers/adjudication_api.py`: the `text=cleaned` path, refusals and
    the snapshot read.
  - `api/routers/legacy_runs.py`: the Cleaned page, status fragment, and
    generate and cancel routes.
- **Templates and CSS:** `legacy_runs/cleanup.html`, the status fragment,
  `fragments/export_menu.html`, the `.cleanup-removed` style in `base.html`,
  and the CSRF constants.
- **Contracts:** `task_inventory.json`, `route_inventory.json`, a new
  `test_cleanup_storage.py` (constraints and trigger), and a `TranscriptText`
  set pin.
- **Docs:**
  - `docs/operations.md`: the exports table, the 422 and 409 rows, the LLM
    section, and the Celery queue list.
  - `docs/how-to/managing-speakers-and-exporting.md`.
  - A new `docs/how-to/cleaning-up-with-the-llm.md` (lay-reader lane,
    `voxint-docs` skill).
  - `docs/architecture.md` (tables) and `CHANGELOG.md`.

## Implementation slices

Each slice leaves the tree green. Review tiers follow the repo rules.

1. **Pure core and seams.** `cleanup.py` (normalization, protection,
   alignment DP, all rejection reasons), `fillers.delete_spans`,
   `LLMReplyError`, and `apply_turn_filters` marks-only.
   - Unit tests; the characterization fixture is untouched.
   - Medium review (it touches the LLM client's error contract).
2. **Storage and writer.** Migration 0073, the models, `record_cleanup` with
   independent re-validation, the read side and the anchor-based source
   loader. Also the storage contract and the head pin.
   - High review (schema).
3. **Producer, job and worker.** Prompt, batching and bisect with the
   malformed and broken-model rules, cancel before and after, the language
   gate, the race guard, and the Celery task with its four touchpoints and the
   task inventory. Tests use a scripted fake `ChatJsonLLM` plus an opt-in
   real-LLM e2e test.
   - Medium review.
4. **Export `text=cleaned`.** The refusals, segment-format substitution, the
   D5 turns path, the snapshot read, the byte-identical default matrix, the
   attribution parity test and the API/CLI negative tests.
   - High review (public export contract).
5. **Console and docs.** The Cleaned page, status fragment, generate and
   cancel routes, export-menu row and goldens, route inventory, docs and
   CHANGELOG.
   - Medium review. Run the browser lane only if the menu change touches
     island props.

## Testing strategy

- **Unit:**
  - the validator table: every scenario above, plus repeats, curly and
    straight apostrophes, hyphenated numbers, `I`-contractions, NFKC
    casefold, punctuation-only tokens, empty and identical proposals, and
    sentence-initial capitalisation after `. ? !`;
  - property tests: an accepted line's kept keys are always a subsequence of
    the source keys, and its deleted set never holds a protected word;
  - render parity with `fillers=drop` on shared single-line cases;
  - prompt and envelope parsing;
  - producer paths with a scripted fake: bisect, malformed line,
    transport-fatal, every line malformed, cancel after the final call.
  - `HttpLLMClient` raises `LLMReplyError` for each shape failure and plain
    `LLMError` for transport.
- **Integration (Postgres):**
  - migration up and down, constraints and the supersede trigger;
  - the writer's forged-input refusals, idempotent and conflicting replay;
  - the job lifecycle: a concurrent claim, cancel, the race guard, the
    recovery sweep and GPU admission;
  - export refusals; staleness after an edit, split or unsplit; speaker rename
    not staling;
  - the byte-identical default matrix (none, current, superseded, plus a
    translation);
  - md turns attribution parity on a mixed-speaker segment;
  - the Cleaned page: the historical diff when stale, and the language note;
  - export-menu goldens, both unchanged and new.
- **Contracts:** the task and route inventories, the `TranscriptText` set and
  the storage invariants.
- **Real-LLM e2e** (opt-in, mbp bundled model): one short English fixture.
  It asserts only invariants: every stored line satisfies the validator, the
  counts add up, and the export works. It never asserts specific deletions. It
  also records the reject rate for the PR notes.
- **Before each PR:** the full integration suite, `ruff check` (not
  `ruff format`) and mypy.

## Rollout / risks / open questions

- **Reject rate.** Small local models reformat, and D1 rejects any wording
  change; D5 adds `unanchored` rejections on heavily corrected lines. The
  counts make this visible. Measure it on the real-LLM lane before calling the
  feature useful, and report the measurement to the maintainer.
- **Name heuristic.** It misses lowercase and sentence-initial names, and it
  over-protects mid-sentence capitalised common words. This was accepted
  under D2; protection fails closed.
- **Greedy-versus-DP cost.** The alignment DP is O(n·m) per line on short
  lines; negligible.
- **`_clean_pass` drift.** A future filler-pass change can change how *new*
  generations render. Stored generations keep their stored text, and turns
  exports re-run the pass over omits, the same as #757 marks.
- **Resolved O1 to O3:**
  - O1 is D6.
  - O2: API and CLI are deferred (3/3 agree).
  - O3: whole-line deletion stays a rejection in v1. Codex and qwen agree;
    glm preferred a blank line. The maintainer approved rejection (2026-10-05).

## Review notes

Plan consult, High tier, 3/3 voices: codex (`gpt-6.1-sol`), `z-ai/glm-5.3`,
`qwen/qwen3.8-max-prime`. Claims about the code were verified before accepting
or rejecting them.

**Storage and diff snapshot (codex blocking, qwen; 2/3).** `keep` indexes
alone cannot draw a stale diff. *Accepted:* each line stores the frozen
`source` and deleted lexical ranges, and the stale diff is labelled
historical.

**Turns attribution (codex; verified at `turns.py:555`, `coarse_turns`
bypasses word evidence).** glm raised a related concern that cue text and the
strike-through could disagree. *Escalated to the maintainer → D5:* deletions
are anchored to #757 word identities and applied as omits in turns exports.

**Typed reply errors (codex; verified, `chat_json` raises one `LLMError` for
both classes).** *Accepted:* `LLMReplyError` subclass.

**Broken-model threshold (glm).** *Accepted:* the job fails only when no line
got a parseable reply. A model that parsed fine but proposed only rejected
deletions is an honest success with counts.

**Normalization authority (codex, qwen; 2/3).** *Accepted:* one key function
built on `_token_key`, covering curly apostrophes and hyphens.

**Alignment ambiguity (glm, qwen, codex; 3/3).**
- glm: reject when alignments disagree on protection.
- qwen: a DP that prefers keeping protected words.
- codex: keep greedy and document it.

*Accepted qwen's DP*, with earliest-occurrence ties. It accepts more valid
proposals than glm's rule and stays deterministic.

**Writer trust and replay (codex).** *Accepted:* the writer re-validates and
derives counts itself; replay compares config and counts.

**Race guard overclaim (codex).** *Accepted:* reworded to translation's
best-effort guarantee. The export check and render share one REPEATABLE READ
snapshot.

**Console placement (codex; verified #567 moved translation controls to the
editor).** *Accepted:* the dedicated Cleaned page is the single home, with no
run-detail card, a separate polling fragment, and cancel kept available while
the LLM is disabled.

**Filler list stored twice (glm).** *Accepted:* the generation's `config` is
the one provenance record; the job's is the request snapshot.

**`translated_texts` rename (glm, qwen; 2/3).** *Accepted:* not renamed.

**Queue (qwen).** *Accepted:* the same queue as translation, never `post`.

**Cancel timing (qwen).** *Accepted:* checked after each call and before the
write.

**Reply envelope (qwen).** *Accepted:* translation's index-echoed array, plus
a per-line length cap.

**Bounds and tests (qwen, glm, codex).** *Accepted:*
- `MAX_LINES` and payload caps;
- superseded and translation-coexisting defaults;
- API/CLI negatives;
- concurrency and replay tests;
- the menu row links to the page, so a stale variant leaves no dead download.

**Hash domain (qwen): own hash without the source language.** *Rejected:* the
detected language changes only on re-transcription, which replaces segments
and stales the variant anyway. Reuse keeps one freshness authority.

**Sentence-initial name protection (qwen).** *Not adopted:* it changes D2, and
protecting line-initial capitals would also protect line-initial `So` and
`Well`. Recorded as a known heuristic gap.

**Explicit `fillers=keep` refusal (codex).** *Rejected:* `keep` is the default
no-op, and translation refuses only `drop`; parity wins.

**Typed renditions table (codex alternative C).** *Rejected for now*, as in
the table above.

**O3 whole-line deletion.** Codex and qwen favour rejection; glm favours a
blank line. *Split surfaced to the maintainer;* the plan rejects for v1.
