# Plan: compare voices from other recordings (#714)

Status: in-progress

## Goal

In the review console's speaker menu, the operator can already hear any roster
speaker who has a line in the current recording (#571, #712). The editor plays
that line by seeking its own audio. A person with no line here cannot be heard,
so an operator who suspects a new voice belongs to someone known from another
recording has to assign blind or leave the page.

This plan adds a cross-recording voice sample. It is a short clip of a line
that the operator confirmed as that person in another recording, trimmed to
audio where only that person's diarization cluster is speaking. The speaker
menu plays it so the operator can compare before assigning. The audience is
single-operator local installs, so the plan adds no configuration knob, no
dependency and no disk cache.

## Assumptions and constraints

- **#712 is frontend-only.**
  - `MediaEditor.tsx:467-511` (`comparisons`) picks an in-recording line per speaker.
  - Playback goes `TranscriptPlayer.previewSegment` → `playTurn` (`lib/playback.ts`) on the shared `<audio src="/media/{run_id}">`.
  - No backend exemplar exists.
- **Editor audio** is the run's normalized 16 kHz mono PCM WAV. It is resolved only through `resolve_servable_media` (`api/playback.py`), which returns an open handle the caller must close, or raises one of:
  - `MediaReclaimed` (→ 410 on `/media`);
  - `MediaMissing` (no `preprocessed_audio` row, e.g. after purge);
  - `MediaUnservable` (confinement or ffprobe failure).
- **`/media/{run_id}` auth:** `OperatorDep` plus `require_onboarded`. Viewers may GET. There is no claim check and no CSRF (it's a GET). Multi-user mode has only operator and viewer roles, with no per-project visibility.
- **The no-store rule does not cover editor subroutes.** `_MEDIA_DETAIL_RE` (`app.py:253`) matches `/media/{uuid}/editor` only when followed by `?` or end of string, so editor subroutes (`/editor/claim`, …) don't get `Cache-Control: no-store`, despite the docstring saying "and descendants".
- **`lib/playback.ts` doesn't make playback exclusive.**
  - `cancelActiveTurn` removes the previous guard but never pauses its audio.
  - Free playback from the transport (`PlaybackControls.tsx`) bypasses `playTurn` entirely.
- **Attribution spans are transcript segments, not diarization turns.** `attributed_intervals(session, run_id)` yields transcript-segment or split-child spans with a winning resolution, canonical speaker id and raw diarization label. A segment carries only its dominant label and can contain other voices: the harm #55 guards against, which `representative_turns` (`api/playback.py`) handles by preferring the longest non-overlap turn.
- **`media/clips.py`** already has `resolve_sample_bounds` (raises `ClipBoundsError`), the PCM invariant (`ClipSourceError`) and a frame-copy `extract_clip` that writes to a file.
- **`speakers/aggregate.py`:**
  - `_canonical_runs` picks the newest completed, non-archived run per media item, newest media first. It does **not** filter trashed or purged media.
  - `aggregate_speakers` runs a full fold and is measured at about 0.9 s for a 200-file library.
- **Media lifecycle:**
  - Trash keeps `normalized.wav`, and `/media` still serves it.
  - Purge requires trash, deletes the artifact rows and sets `purged_at`.
  - Reclaim unlinks `normalized.wav` and stamps the artifact.
- **No living spec is declared.**
- **What would invalidate this plan:** the maintainer wanting machine-attributed samples, samples from trashed media, or a different clip length (the maintainer set 10 s and a 2 s floor on 2026-10-02).

## Proposed approach

### Eligibility and picking: one function

`src/voxint/speakers/voice_sample.py` (new) holds a single candidate walk that
both the availability list and the clip endpoint use, so they cannot drift:

```
voice_sample_candidates(session, *, exclude_media_id, speaker_id=None)
    -> dict[canonical_speaker_id, SampleChoice]
```

1. **Source runs.** It walks `canonical_runs(session)` (`_canonical_runs` made public, unchanged semantics) newest media first, and skips the excluded current media and any media item with `trashed_at` or `purged_at` set.
   - Rationale: the app should not resurface material the operator deleted in a new place. Trash is not a privacy boundary elsewhere in the app, and the docs will say that rather than imply it.
2. **Cheap DB pre-checks.** A run whose `preprocessed_audio` artifact is reclaimed or absent is recorded as a *gone* source, not a candidate. These checks need no file I/O and no ffprobe.
3. **Clean spans.** For each surviving run, take every `attributed_intervals` row with `resolution is HUMAN_ASSIGN` and a canonical speaker id. Intersect the row with that run's diarization turns that carry the same raw label and `overlap = false`. The resulting clean spans are the only audio that may be offered as this person's voice.
   - A word-range split child is handled the same way, so a narrower override still wins under the resolver's precedence.
   - A span shorter than `VOICE_SAMPLE_MIN_SECONDS = 2.0` is discarded: a fragment that short is too little voice to compare and is worse than nothing (maintainer decision, 2026-10-02).
4. **Choice per speaker.** The first run, newest first, that has a clean span of at least 2 s wins, and its longest clean span is used. Ties go to the earliest start.
   - The floor sets the quality bar and recency only orders runs that pass it. A 1.2 s fragment from last week is not a candidate, so it cannot beat a clean 6 s line from last month.
   - Because the first qualifying run wins, the clip route can stop walking at that run.
   - It is a deliberate new policy, not #712's in-recording ordering, which keeps the earliest line.
5. **Window.** If the span is 10 s or shorter (`VOICE_SAMPLE_MAX_SECONDS = 10.0`), the whole span is used. Otherwise the clip is the first 10 s from speech onset. Bounds go through `resolve_sample_bounds` and are capped in integer frames, so the cap is exact.
   - Both values are module constants, not settings: they describe the shape of the preview, not deployment configuration.
6. **Speaker ids.** They are canonicalized through merges. A merged-away id resolves to its active survivor (read-time identity, like the rest of the app). Archived, unknown or merged-into-archived ids get no sample. Machine attributions (grounded cosine, auto-enroll) are never sources, not even as a labelled fallback: a hedged label on a possibly wrong voice is exactly the misleading confidence the human-only rule exists to prevent.

`SampleChoice` carries the run, the span and the ordered fallbacks: the next
best spans in later runs. If the first choice's audio fails at serve time
(`MediaUnservable`, `ClipSourceError`, `ClipBoundsError`), the endpoint tries
the next fallback.

Rejected:
- **The enrollment source** (`speaker_embeddings.source_pipeline_run_id` plus label): it has no time span, and enrollment can come from a run the operator later corrected.
- **Reusing `aggregate_speakers`:** it computes far more than one bit per speaker, would need a per-caller trash filter that breaks the module's "overview and profile never disagree" doctrine, and still could not produce clean spans.
- **A raw SQL `EXISTS` over the ledger:** it cannot account for overrides that displace a label assignment.

### Endpoints (editor router, after the existing editor routes)

The `{media_id}` in both paths names the **current** recording, which is
excluded as a source. It is never the source, and `docs/architecture.md` will
say so.

1. **`GET /media/{media_id}/editor/voice-samples`** → `{"speakerIds": [...]}`.
   - These are active canonical speakers whose candidate walk found at least one non-gone candidate.
   - The editor fetches it on the first popover open and keeps it for the session. Eligibility depends only on *other* recordings, so edits in this one cannot change it.
   - Nothing is added to editor page load.
2. **`GET /media/{media_id}/editor/voice-sample/{speaker_id}`** → `200 audio/wav`, an in-memory WAV of exactly the window.
   - It is extracted from the same handle `resolve_servable_media` returned, then closed on every path. `extract_clip` is refactored to write to any `BinaryIO`; the file wrapper stays for the annotation clip cache.
   - The clip is at most 10 s × 32 kB/s ≈ 320 kB.
   - Headers: `Content-Type: audio/wav`, `Content-Disposition: inline` with no filename, and `Cache-Control: no-store`.
   - There is no range support and no HEAD. The client fetches the whole clip and plays it from a blob, so ranges would only be unused machinery, and with no cache, ranges across separate picks could splice two different clips.
   - The WAV has no metadata chunks. No source media id, run id, title or path appears in the headers or body.
3. **Errors.** Both use the editor router's existing error shape and its existing code for a bad current media id.
   - **404 `no_voice_sample`:** the speaker is unknown or archived, or has no clean confirmed span outside this recording other than in trashed or purged media.
   - **410 `voice_sample_gone`:** clean confirmed spans exist, but every candidate recording's audio is reclaimed, missing or unplayable.
4. **Auth.** Identical to `/media/{run_id}`: `OperatorDep` plus `require_onboarded`. Viewers may GET. There is no claim, because hearing is read-only, and no CSRF, because these are GETs.
   - The 200, 404 and 410 status codes reveal adjudication state to a viewer, which is within that role's existing trust. The docs will say so in one sentence.
   - If per-project visibility ever lands, `/media` and these routes must be bounded together.
5. **No-store fix.** `_MEDIA_DETAIL_RE` is widened to `/editor(?:/|\?|$)`, so every editor descendant is `no-store`, and the docstring is corrected. The existing claim, refresh and release POSTs gain the header too, which is the safe direction.

### UI

- **The menu group.** `SpeakerAssignPopover` gains `otherRecordingSpeakers` and `onHearOtherRecording(speakerId)`.
  - The group sits below the in-recording compare list as a `<details>` titled "Compare with a voice from another recording". It starts collapsed when the in-recording list has entries and expanded when that list is empty (maintainer decision, 2026-10-02), so it is never hidden behind a click when it is the only comparison on offer.
  - Each button reads `▸ Name`, with `aria-label="Hear Name from another recording"`.
  - Names are sorted, and the group lists every eligible speaker **except those already in the rendered in-recording list**. The line's current speaker is kept, so the operator can check "is this really Dana?". When seek is disabled the in-recording list is empty, so nothing playable is hidden.
- **Playback.** A small `lib/voice-sample.ts` does the following:
  - It fetches the clip with an `AbortController`, so a newer click aborts an older one, and branches on the status and `detail.code`.
  - On 200 it plays the blob on a dedicated `Audio` element at the stored playback rate, so both voices are compared at the same speed.
  - It revokes the previous object URL. The CSP already allows `media-src blob:`.
- **Exclusivity is explicit.** Starting a sample pauses the main `<audio>` and calls `cancelActiveTurn()`. A `play` event on the main `<audio>` (transport or turn preview) pauses the sample. Closing the popover does not stop a playing sample (at most 10 s); unmounting the editor does.
- **Failures** appear in the editor's existing notice area (`role="status"`), not inside the popover, which can close mid-fetch:
  - 404: "No confirmed line of {name} from another recording is available."
  - 410: "The recordings with {name}'s confirmed lines can't be played anymore."
  - Other: "Couldn't play {name}'s voice. Try again."
- **The cursor never moves,** so #732's unsaved-edit guarantees hold.

## Acceptance criteria

### Requirement: Cross-recording voice sample
The system SHALL serve an authenticated operator or viewer a WAV clip, at most
10 s long, of a span that the operator confirmed as the requested speaker in a
recording other than the current one, where only that speaker's diarization
cluster is speaking. The response SHALL contain only that clip's audio.

#### Scenario: Speaker confirmed elsewhere
- GIVEN Dana has a human-assigned 5 s line in recording B over a non-overlapping turn of the same label, and no line in A
- WHEN the operator requests A's voice sample for Dana
- THEN the response is 200 `audio/wav` whose frames equal B's frames for that 5 s span exactly, with `Cache-Control: no-store`

#### Scenario: Overlapped head is trimmed
- GIVEN Dana's confirmed segment in B opens with 1.5 s that overlaps another turn
- THEN the clip starts at the end of the overlap

#### Scenario: Long line is capped
- GIVEN the only clean span is 20 s long
- THEN the clip is exactly 10 s (160,000 frames) from its start

#### Scenario: Floor before recency
- GIVEN Dana has a clean 1.2 s span in the newest recording C and a clean 6 s span in older recording B
- THEN the clip comes from B

#### Scenario: Only machine attributions
- GIVEN Dana's only lines elsewhere are grounded-cosine or auto-enroll attributions
- THEN the response is 404 `no_voice_sample`

#### Scenario: Only trashed or purged sources
- GIVEN Dana's only confirmed lines are in trashed or purged media
- THEN the response is 404 `no_voice_sample`

#### Scenario: Source audio gone
- GIVEN Dana's only confirmed lines are in a recording whose audio was reclaimed, or whose file fails the media gate
- THEN the response is 410 `voice_sample_gone`

#### Scenario: Falls through to an older recording
- GIVEN Dana's best recording's audio was reclaimed and an older one is intact
- THEN the clip comes from the older recording

#### Scenario: Current recording excluded
- GIVEN Dana's only confirmed line is in A itself
- WHEN the request is A's voice sample
- THEN the response is 404 `no_voice_sample`

#### Scenario: Merged id
- GIVEN Dana-2 was merged into Dana
- WHEN the request names Dana-2
- THEN the response is Dana's clip

#### Scenario: Archived speaker
- GIVEN Dana is archived
- THEN the response is 404 `no_voice_sample`

#### Scenario: No leak
- THEN the response headers and body contain no source media id, run id, title or file path, and the body has no WAV chunks other than `fmt ` and `data`

#### Scenario: Auth
- WHEN the request has no credentials
- THEN the response is 401
- AND a viewer's GET succeeds, as it does for `/media/{run_id}`

#### Scenario: Error responses are not cached
- THEN the 401, 404 and 410 responses also carry `Cache-Control: no-store`

### Requirement: Voice-sample availability
The system SHALL list, for a recording, the active speakers that have a voice
sample available from another recording, using the same eligibility as the
clip endpoint.

#### Scenario: List matches picker
- GIVEN a corpus that mixes confirmed, machine-only, trashed-only, reclaimed-only, current-only and archived speakers
- THEN a speaker is listed iff the clip endpoint answers 200 or 410 for them, never 404

### Requirement: Compare voices from other recordings in the speaker menu
The speaker menu SHALL offer, for each eligible speaker not already in the
in-recording comparison list, a button that plays that speaker's voice sample.
When the sample cannot play, the menu SHALL say why.

#### Scenario: Preview
- WHEN the operator opens the menu, expands "Compare with a voice from another recording" and clicks Dana
- THEN Dana's sample plays, the main player is paused, and the cursor and edit box are unchanged, even with an unsaved edit

#### Scenario: Main player takes over
- WHEN the main player starts while a sample plays
- THEN the sample pauses

#### Scenario: Group starts expanded when it is the only comparison
- GIVEN no other voice in this recording can be compared
- WHEN the operator opens the menu
- THEN the "Compare with a voice from another recording" group is already expanded
- AND when the in-recording list has entries, the group starts collapsed

#### Scenario: Current speaker can be checked
- GIVEN the line is attributed to Dana, and Dana is eligible but not in the in-recording list
- THEN Dana appears in the group

#### Scenario: Gone media
- WHEN the endpoint answers 410
- THEN the editor shows "The recordings with Dana's confirmed lines can't be played anymore."

## Spec deltas

Spec deltas: none (no living spec declared)

## Affected files and components

- `src/voxint/speakers/voice_sample.py` (new): the candidate walk, clean-span intersection, choice and fallbacks, `VoiceSample`/`SampleChoice`, constants.
- `src/voxint/speakers/aggregate.py`: rename `_canonical_runs` → `canonical_runs` (semantics unchanged; existing callers updated).
- `src/voxint/media/clips.py`: `extract_clip` writes to any `BinaryIO`; integer-frame cap helper.
- `src/voxint/api/routers/editor.py`: two GET routes.
- `src/voxint/api/app.py`: widen `_MEDIA_DETAIL_RE` and fix its docstring.
- `frontend/src/lib/voice-sample.ts` (new): fetch, blob, play, abort, revoke, exclusivity hooks.
- `frontend/src/components/SpeakerAssignPopover.tsx`: the `<details>` group.
- `frontend/src/components/MediaEditor.tsx`: lazy availability fetch, group eligibility, notices, main-audio `play` hook.
- `tools/e2e_browser_lifecycle.py`: new `voices` fixture with two media items. A has no Dana. In B, Dana has a human-assigned line over a non-overlap turn, with distinguishable PCM (a tone) so the lane can tell which audio played. Plus the tool's unit and integration tests.
- `tests/contracts/fixtures/{route_inventory,console2_route_characterization,console2_route_order}.json`: regenerated with the two GET routes.
- Docs: `docs/how-to/reviewing-and-adjudicating.md` (compare voices); `docs/architecture.md` (the routes, the exclusion-key path id, the eligibility rules, the viewer note, no-store); `CHANGELOG.md` Added and Fixed (the no-store widening); `.claude/skills/voxint-e2e-review/SKILL.md` (the `voices` fixture and steps).

## Implementation slices

1. **No-store for editor descendants.** Widen the regex and add a test that every editor subroute, including error responses, carries `no-store`. This is a small, separable security-header change that lands first.
2. **Backend voice sample end to end.** The candidate walk, the clip route and the availability route, plus contract goldens, unit and integration tests, the architecture doc, and a measured latency of both routes on a seeded library of about 200 runs, recorded in the PR. This slice is verifiable with curl.
3. **Menu preview.** `voice-sample.ts`, the popover group, exclusivity, notices, vitest, the how-to and the CHANGELOG.
4. **Browser lane.** The `voices` fixture and its tests, a lane run on maintainer hardware with a before and after durable-state reconcile (this flow must write nothing), and the SKILL.md update.

## Testing strategy

- **Unit (picker), on distinguishable synthetic PCM:**
  - human-only; clean-span intersection (overlapped head and tail trimmed);
  - a split child that overrides a label (precedence);
  - the 2 s floor (a 1.9 s span is ineligible, 2.0 s is eligible), floor before recency, ties;
  - the exact 10 s frame cap with a fractional start;
  - skipping trashed, purged, current and archived-run media;
  - a reclaimed run → gone, then fallback to the next run;
  - `ClipBoundsError` and `ClipSourceError` → next fallback;
  - merged → survivor; archived → none.
- **Unit (extraction):** exact frame count, the PCM invariant, only the `fmt ` and `data` chunks, and the handle closed on the error paths.
- **Integration (pg17), using real trash, purge and reclaim transitions, not just file deletion:**
  - 200 with exact source frames; 401; viewer 200;
  - 404 for each cause; 410 for reclaimed and for unservable;
  - no-store on success and on errors; the no-leak grep;
  - list ⇔ picker agreement on a mixed corpus;
  - byte-identical responses to repeated requests (deterministic choice).
- **Contract:** three route goldens regenerated, with minimal diffs justified in the commit.
- **Vitest:**
  - group eligibility (the current speaker kept, the in-recording list excluded, nothing hidden with seek off);
  - the group's initial open state (expanded with an empty in-recording list, collapsed otherwise);
  - lazy fetch once per session;
  - abort on a rapid second click; object URL revoked;
  - main paused on sample start, and sample paused on main `play`;
  - notice copy per status, and a rejected `play()` promise;
  - the cursor and a dirty edit unchanged.
- **Browser lane:** real clicks. The network log shows the availability call once and a 200 `audio/wav`. The sample element plays while the main element is paused. The 410 path runs by stamping B's artifact as reclaimed through the tool, not by deleting the file, and the SKILL names which path it covers. The durable-state reconcile is unchanged before and after.
- **Review tiers:** **High** for slices 1 to 3 (security headers, a new audio-serving route that crosses recordings, a public route contract). The browser lane is mandatory for slices 3 and 4.

## Rollout, risks and open questions

- **Cost.** The candidate walk runs `attributed_intervals` over every eligible canonical run: once per editor session for the list, and once per click for a clip. Slice 2 measures both on about 200 runs. The clip route stops at the first qualifying run; the list route must still visit every eligible run once per editor session.
- **Clean spans depend on diarization overlap flags.** A speaker whose confirmed lines all overlap other speech gets no sample. That is preferable to playing a mixed clip, and the 404 copy stays true.
- **Privacy.** Playing B's audio on A's page is within what the same principal can already hear through `/media`. Trashed and purged media are excluded by design.
- **Settled by the maintainer (2026-10-02):** a 10 s cap, a 2 s floor, and the group starts expanded when the in-recording list is empty.
- **Open:** none.

## Review notes

Consult tier: **High**. Panel: codex (planner, verified against the repo),
z-ai/glm-5.3 and moonshotai/kimi-k3, 3 of 3 voices.

| Theme | Raised by | Resolution |
|---|---|---|
| Confirmed segments can contain other voices; intersect with non-overlap turns | codex, glm, kimi (3/3) | **Accepted.** Clean-span intersection, a floor (maintainer set 2 s), and a test for an overlapped head. |
| `lib/playback.ts` does not pause audio, and the transport bypasses it, so the draft's exclusivity claim was false | codex, glm, kimi (3/3) | **Accepted.** Explicit pause both ways via `voice-sample.ts` and a main `play` hook, with tests in both directions. |
| The list and the picker are two implementations that will drift; the draft's list description was inconsistent | codex, glm, kimi (3/3) | **Accepted.** One `voice_sample_candidates` walk feeds both, plus a list ⇔ picker agreement test. |
| Full aggregate fold on every editor load | codex (measure), glm (lazy), kimi (dedicated query) | **Accepted as lazy:** a list route fetched on first menu open, sharing the picker's walk. A raw `EXISTS` was rejected because it cannot see displaced overrides (codex). The cost is measured in slice 2. |
| `<audio src>` cannot read 404 vs 410; use fetch plus blob | codex, glm (2/3); kimi kept ranges for Safari | **Accepted fetch plus blob.** It makes the range responder, HEAD and the cross-pick splice hazard (glm) unnecessary, and Safari's range probe does not apply to blob URLs. **Split recorded:** kimi preferred keeping range support. |
| The no-store regex does not match editor subroutes | codex, verified at `app.py:253` (kimi asked to verify) | **Accepted.** Its own slice 1, with error-response coverage. |
| Purge-to-410 contradicted skipping purged media; needs a decision table | codex, glm (copy honesty) | **Accepted.** Trashed and purged sources → 404; reclaimed, missing or unservable → 410. The copy was reworded so neither message claims more than is known. |
| Recency-first selection lets a short blip beat a good line | glm, kimi | **Accepted.** A 2 s floor first, then recency. |
| Frame-exact cap, `ClipBoundsError`/`ClipSourceError` mapping, a single gate handle | codex, kimi | **Accepted.** Integer-frame cap, next-fallback on either error, one handle closed on every path. |
| Excluding the line's current speaker hides the most useful check | codex | **Accepted.** The current speaker is kept; only the rendered in-recording list is excluded (kimi's seek-off edge is also covered). |
| Errors inside a popover that closes are lost | kimi | **Accepted.** Errors go to the editor notice area. |
| Playback rate should match the main player | glm | **Accepted.** The sample uses the stored rate. |
| The trash rationale overstated privacy | glm | **Accepted.** Reworded to "don't resurface deleted material in a new surface". |
| A `media_id` path vs `/media/{run_id}` is confusing | glm, kimi | **Accepted as documented.** It matches the editor route family; the architecture doc names the path id as an exclusion key. |
| Merged id → survivor | codex, glm, kimi (3/3) | **Accepted.** |
| Machine-attributed fallback | codex, glm, kimi (3/3): no | **Closed:** not in this issue. |
| Multi-user visibility | codex, glm, kimi (3/3): match `/media` | **Closed.** Plus a docs note that status codes reveal adjudication state to viewers (kimi). |
| Tests: real lifecycle transitions, distinguishable PCM, determinism, a worst-case latency measurement | codex, glm, kimi | **Accepted** into the testing strategy and slice 2. |
| Cut mid-utterance at the cap | glm | **Partly accepted.** A whole span is used when it is 10 s or shorter; longer spans are cut at 10 s from onset (accepted trade-off). |
| Maintainer decisions (2026-10-02) | Ben | 10 s cap (was 8 s), 2 s floor (was 1 s, which also removes the sub-floor fallback), and the group starts expanded when the in-recording list is empty. |
