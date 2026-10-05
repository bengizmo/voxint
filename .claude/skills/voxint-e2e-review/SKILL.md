---
name: voxint-e2e-review
description: >-
  Run the maintainer, opt-in browser E2E acceptance lane for the Voxint
  review-console islands (#53/#58): build + stage the islands, seed a disposable
  database, serve a throwaway instance, drive the verify-and-advance loop with a
  real browser (Playwright MCP) asserting DOM + network behaviour, reconcile the
  durable state, then clean up. Use whenever verifying the transcript review loop
  end to end in a browser — verify/edit/skip/replay keys, click-to-edit, the
  unsaved-edit discard warning, the keymap suppression on focused form
  controls, the keyboard-shortcuts cheat-sheet modal (#51: open via `?` or the
  button, focus trap, Escape/close/backdrop dismiss), word marks and clean-up
  mode (#757: keep/omit, undo, correction clearing), or the waveform strip
  (#57: peaks fetch, region click → selection,
  playhead/cursor sync), or the speaker menu's voice from another recording
  (#714: `--fixture voices`) — or when a Gate E release check calls for the
  browser lane. Serial only on maintainer hardware; never public CI.
---

# Voxint browser E2E review lane

This is a **thin adapter** over the canonical lifecycle tool
`tools/e2e_browser_lifecycle.py`, which owns build/seed/serve/reconcile/cleanup.
This skill adds only the interaction layer: it drives the review-console islands
in a real browser via Playwright MCP and asserts each behaviour immediately,
then hands durable-state checking back to the tool. Keep interaction and its
assertion together; the tool is the source of truth for everything else.

**Audience & scope:** Voxint is a self-hosted, single-operator, local
audio-intelligence app. This lane exists to gate two runtime-only island
behaviours that every server-side test leaves green. It is **opt-in and
maintainer-run** (never public CI — GitHub has no browser/model runners), and
**serial** on maintainer hardware (a past hard-reset under concurrent CPU load —
issue #23; keep concurrency at one). If Playwright MCP is not available in the
session, this lane must **fail, not skip** — an operator who asked for the
browser gate must never get a green result because the browser was absent.

## Preconditions

- Run from the repo root, under `uv`. A disposable Postgres DSN whose database
  name contains `test` or `e2e` (the tool refuses anything else — the live
  database is named `voxint`). Export it once, e.g.
  `DSN=postgresql+psycopg://voxint:voxint@localhost:5432/voxint_e2e`.
- Playwright MCP tools present in the session (`browser_navigate`,
  `browser_snapshot`, `browser_click`, `browser_type`, `browser_press_key`,
  `browser_network_requests`, `browser_evaluate`, `browser_close`). Absent →
  stop and report the lane could not run (fail, not skip).

## 1. Bring-up (the tool does the heavy lifting)

```bash
uv run python tools/e2e_browser_lifecycle.py setup            # build + stage islands
uv run python tools/e2e_browser_lifecycle.py seed  --database-url "$DSN"   # → prints RUN_ID=<uuid>
uv run python tools/e2e_browser_lifecycle.py serve --database-url "$DSN" & # background; port 8099
```

Capture the `RUN_ID=` line from `seed`. On a fresh host, pass `--create-db` to
`seed` once (creates the database + `vector` extension). Wait for
`http://127.0.0.1:8099/healthz` to return 200 (basic-auth `admin` / `e2epass`).

## 2. Drive the islands + assert (Playwright MCP)

Navigate **once** with credentials embedded to cache basic-auth, then
**re-navigate to the clean URL** — an island `fetch()` throws "URL includes
credentials" if the *document* URL carries `user:pass@` (a harness artifact, not
a product bug):

1. `browser_navigate http://admin:e2epass@127.0.0.1:8099/` then
   `browser_navigate http://127.0.0.1:8099/runs`.
2. Open the seeded run's workbench (`/runs/<RUN_ID>`), click **Open in editor**:
   it auto-claims the run and lands directly on the media editor page at
   `/media/<MEDIA_ID>/editor?run=<RUN_ID>&token=…`. The `MediaEditor` island
   handles the transcript walk, verify/edit/skip/replay controls, speaker rail,
   waveform strip, and provenance all in one page.

First assert the seeded confidence signal renders: exactly **two**
`.tp-uncertain-chip` (segments 1 and 3, confidence 0.42 / 0.31 < the 0.6
threshold), and the high-confidence and NULL-confidence segments are **not**
flagged — a threshold-rendering regression must fail here, not pass silently
(`browser_evaluate` counting `.tp-line.tp-uncertain`).

Then exercise each behaviour, checking `browser_network_requests` filtered to
`/segments/.*/(verify|text)` right after each — only **verify** and **save**
touch the network:

- **verify-and-advance (`v`):** click the page heading (move focus off any form
  control), press `v` → exactly one `POST …/verify` (200); the counter
  ("N of 5 segments verified", the first `p` in `[aria-label="Editor controls"]`)
  advances by one and the cursor
  moves to the next unverified segment. **Then press `p` (replay)** and confirm
  the instrumented `play()` fires and `currentTime` moves — a verify patches the
  `segments` array, and a regression that tore down the `<audio>` element on that
  identity change would leave playback dead for the rest of the session while
  the DB reconcile still passed. This assertion is the guard for that class.
- **skip (`n`):** press `n` → **no** new `/verify` or `/text` request; the cursor
  advances to the next unverified segment.
- **replay (`p`):** asserting "no network + cursor unchanged" is not enough — a
  removed `p` handler would pass it. Instrument playback first: via
  `browser_evaluate`, wrap the audio element's `play()` to set a flag (and read
  `currentTime`), then press `p` and assert `play()` was invoked for the current
  segment (its `currentTime` set to the segment start). Still **no** network
  request. (The headless browser may not decode the seeded WAV, so instrument
  `play()` rather than relying on audible playback.)
- **click-to-edit:** click a transcript line
  (`p.tp-line:has-text("<line text>")`) → the edit textarea
  (`aria-label="Corrected transcript text for this segment"`) loads that
  segment's text and "Cursor on segment at X.X seconds, speaker SN" updates.
  Verified lines are re-reachable this way.
- **discard warning (warn-then-advance):** with a segment loaded, type into the
  textarea (makes it dirty), press **Escape** to blur (the keymap is suppressed
  while the textarea has focus), then press `v` → the `p[role="alert"]` "You have
  an unsaved edit…" appears and **no** `/verify` fires; press `v` again → exactly
  one `/verify` (verifies the original wording, discarding the edit) and the
  counter advances.
- **edit + save (Ctrl/⌘+Enter):** type a genuine correction into the textarea,
  press `ControlOrMeta+Enter` → one `POST …/text` (200).
- **keymap suppression:** focus the speed control
  (`select[aria-label="Playback speed"]`), press `v` → **no** new `/verify`
  (a focused `<select>`/`<textarea>`/`<input>` suppresses the single-key keymap).
- **cheat-sheet modal (#51) — open both ways, dismiss, and suppress:** move focus
  off any form control, then press `?` → a `[role="dialog"][aria-modal="true"]`
  titled "Keyboard shortcuts" appears (its `<dl>` renders from the shared
  `REVIEW_SHORTCUTS` source of truth; the media editor lists 18 rows). While it is open, press `v` → **no** new
  `/verify` (the `helpOpenRef` guard suppresses the global keymap behind the modal).
  Press **Escape** → the dialog is gone. Reopen via the **"Shortcuts ?"** button
  (`getByRole('button', { name: 'Shortcuts ?' })`; `button[aria-haspopup="dialog"]`
  also matches the palette and speaker buttons) instead of the key, then dismiss with the ✕
  close control (`aria-label="Close keyboard shortcuts"`); reopen once more and
  dismiss with a **backdrop click** (mousedown+click on the fixed overlay, not the
  panel). Each dismiss returns focus to the opener. (The seed has a roster, so the
  header's inline `Assign speaker (1–9):` digit cue is present; on a roster-less run
  it is intentionally absent — not asserted here.)

### Domain-pack correction provenance (#83)

The seed freezes a two-rule pack (`_E2E_PACK_NAME`/`_E2E_CORRECTIONS` in the
lifecycle tool) and corrects **segment 0** (`everyone` → `everybody`, an
`input_base:"raw"` trace); the second rule (`quarterly synergies`) matches nothing,
so it never fires and no line carries it (the editor lists only rules that fired;
#674). All server-side tests stay green without this, so assert it in the browser:

- **Marker present + distinct from "edited".** Navigate to segment 0 (click its
  transcript line, or step there). A `button.tp-corrected-chip` reading "corrected by
  domain pack (1)" is present in the current-segment header. It is **not** the
  `.spk-badge` "edited" chip (that one appears only after an operator save) — the two
  are different affordances. Click the chip → its `aria-expanded` flips to `true` and
  the `#editor-provenance-body` list shows the rule `everyone → everybody`.
- **Marker absent on an untouched segment.** Move to segment 2 (high-confidence, no
  correction): **no** `.tp-corrected-chip` in the header.
- **Operator edit supersedes provenance.** With segment 0 focused, type a genuine
  correction into the edit box and save (`ControlOrMeta+Enter`) → one `POST …/text`
  (200), and the `.tp-corrected-chip` marker is now **gone** from the header (the
  operator's text supersedes the pipeline trace — no stale spans). Restore/leave as
  appropriate for the reconcile expectation (this counts segment 0 as corrected).
  Saving text on an already-verified segment clears its verification, so if
  segment 0 was verified earlier it leaves `verified_segment_indexes` and the
  counter drops by one.

Note: `aria-current` on `p.tp-line` tracks the audio **playback** highlight, not
the review cursor — assert the review cursor via the edit box's value and the
"Cursor on segment at …" line, not `aria-current` (which stays put when the
headless browser cannot play the audio element).

### Waveform strip (#57)

The strip (`[data-testid="waveform-strip"]`) mounts inside the player when the
island's `peaksUrl` fetch succeeds. The seed writes quiet constant-amplitude
audio plus one diarization turn per segment, so the strip renders five colored
regions. Assert:

- **Presence + single fetch:** the strip container and its `<canvas>` exist;
  `browser_network_requests` shows exactly **one** `GET /media/<RUN_ID>/peaks`
  (200) — no retry loop, and none of the other actions below add another.
- **Region click → selection (+ seek):** via `browser_evaluate`, compute the
  midpoint x of segment 2's span
  (`(2.5 * 5.0 / duration) * canvas.getBoundingClientRect().width`) and
  dispatch a click there → "Cursor on segment at 10.0 seconds, speaker S0" appears, the
  container's `data-cursor-index` is `2`, and (instrumented as for `p` above)
  the audio element's `currentTime` lands inside `[10, 15)`. **No** `/verify`
  or `/text` request fires — a region click is selection + playback, never a
  write.
- **Keymap ↔ strip sync:** press `n` → `data-cursor-index` advances to the next
  unverified segment's index with no network request; press `p` → the playhead
  div (`[data-testid="waveform-playhead"]`) is present and its `style.left`
  moves off `0%` (instrument `play()` as usual — the headless browser may not
  actually decode).
- **Gap click is a no-op:** click the far right edge of the strip (past the
  last segment's end, if the seed leaves tail silence) — cursor and network
  both unchanged. Skip this check when the seed's segments tile the whole
  duration.
- **Peaks-absent degradation (spot-check, cheap):** `browser_evaluate`
  `fetch('/media/<RUN_ID>/peaks').then(r => r.status)` must be 200 here; the
  no-strip path (`peaksUrl: null` ⇒ no strip node, no `/peaks` request at all)
  is already pinned server-side in
  `tests/integration/test_runs_api.py::test_transcript_island_props_carry_turns_and_gate_peaks_url`
  — do not rebuild a second seed just for it.

The strip is `aria-hidden` (the list is the accessible surface): never assert
on its accessibility tree, only `data-*` attributes, the canvas, and network.

### Speaker rail (#115)

Seed with `--fixture rail` (12 segments, labels S0..S5, a four-person roster
including the auto-saved `Voice 1`). Open `/runs/<RUN_ID>` and click **Open in editor**:
it claims the run and lands on `/media/<MEDIA_ID>/editor?run=…&token=…`, where
the rail mounts as `[aria-label="Speaker rail"]`. Assert, in order:

- **Initial partition.** `.rail-summary` reads exactly "5 voices need you, 1
  with very little speech. 1 matched automatically."; the `Needs you (4)`
  section lists S1, S5, S2, S3 in that order (confirmable unresolved, confirmable
  auto-saved, ambiguous, unmatched); `Too little speech to identify (1)` (S4)
  and `Matched automatically (1)` (S0) are closed `<details>`; no `Your rulings`
  group yet.
- **Card copy.** S1: "Possibly Blair Roster" + `button "Confirm Blair Roster"`,
  pill `needs you`. S5: same headline, pill `saved automatically`, detail "Saved
  as Voice 1 for now. Confirm if this is Blair Roster." S2: "Similar voices
  found", **no** Confirm button, its disclosure is titled "Why no name?" and
  its text names nobody and carries no numbers. S3: "Who is this?" with
  disclosure "Why no match?". Open S1's "Why this match?": the text carries the
  raw numbers (0.65 / 0.12 / 0.80) and no percent sign.
- **Hear this voice.** Instrument `play()` as for `p`, click S1's button → one
  `play()` call at `currentTime` 10 (S1's first line), the `[aria-live="polite"]`
  line reads "Cursor on segment at 10.0 seconds, speaker S1", and **no** `/labels/`
  or `/segments/` request fires.
- **Confirm.** Click S1's Confirm → exactly one `POST …/labels/S1/decision`
  (200); S1 moves to `Your rulings`, the summary drops to "4 voices need you…",
  and the transcript lines for S1 now read "Blair Roster:". Repeat on S5 (the
  auto-saved path) → `POST …/labels/S5/decision`.
- **Rulings.** `Can't tell` on S3 and, after opening the too-short group,
  `Not a person` on S4 → one decision POST each; S4's row reads "Left out".
- **Add a new person.** On S2 open the picker (`aria-label="S2: choose who
  this is"`) and type a name that is not on the roster → the listbox offers a
  `Create "<name>"` row. Choosing it fires `POST …/labels/S2/enroll`; on this
  seed it returns **400** ("no speaker audio to create an identity from": the
  seed's turns carry no embeddings) and the rail shows that message in its
  `[role="alert"]` without losing the card. The picker stays open with the typed
  name kept so a retry needs no retyping; press Escape to close it. That is the
  honest error path; do not treat it as a pass for enrollment itself.
- **Finish line.** `Can't tell` on S2 → summary "Every voice has a ruling." and
  both `Matched automatically` and `Your rulings` are open; a `Change` button on
  a resolved row reveals the `Reassign to…` picker.

Reconcile with the rulings you made, for example:

```bash
--expect '{"verified_segment_indexes":[],"corrections":{},"progress":{"verified":0,"total":12},
  "label_rulings":{"S1":{"decision":"assign","speaker":"Blair Roster"},
  "S5":{"decision":"assign","speaker":"Blair Roster"},"S3":{"decision":"unknown","speaker":null},
  "S4":{"decision":"exclude","speaker":null},"S2":{"decision":"unknown","speaker":null}}}'
# → ok: 0 of 12 verified; corrections match; 5 label ruling(s) match / RECONCILE PASS
```

### Voice from another recording (#714)

Seed with `--fixture voices`. It prints `RUN_ID`/`MEDIA_ID` for the run the lane
opens (the 5-segment review run, with S1 already confirmed as Blair Roster) and
`VOICE_SOURCE_RUN_ID` for a second, 30-day-old recording the lane never opens.
In that recording Dana Roster and Blair Roster are confirmed over 5 s tones of
440 Hz and 880 Hz. This flow is read-only: reconcile both runs **before and
after** with the same expectation.

Instrument before the first click (`browser_evaluate`): wrap `play()`/`pause()`
on the main `audio[data-run-id]`, and replace `window.Audio` with a wrapper that
records the element the sampler creates and its `play`/`pause`/`playing` events.
Here the headless browser does decode the clip, so assert the `playing` event.

- **Lazy, once.** No `/editor/voice-samples` request before the first speaker
  menu opens; exactly one after, and none on later opens or other lines.
- **Collapsed beside the in-recording list.** Open the menu on an S0 line
  (`p.tp-line[data-seg-index="0"] .tp-speaker-btn`, a real click). The
  in-recording list offers `Hear Blair Roster`. `details.sp-compare-other` is
  closed and holds only `Hear Dana Roster from another recording` (Blair is not
  repeated). Click the summary to expand; the menu stays open.
- **Preview.** Type into the edit box first (an unsaved edit). Click Dana: one
  `GET …/editor/voice-sample/<id>` → 200 `audio/wav`, `no-store`, 160,044 bytes;
  the main element gets `pause()`, the sample element `play()` then `playing`
  with `duration` 5. The menu stays open, the "Cursor on segment…" line and the
  edit text are unchanged, and no unsaved-edit warning appears.
- **Main player takes over.** Click `Hear Blair Roster` (in-recording): the main
  element plays and the sample element is paused.
- **Closing the menu keeps the clip.** Click Dana, press Escape: the sample
  element is still playing.
- **Expanded when alone, current speaker kept.** Open the menu on an S1 line
  (Blair's). There is no in-recording list, the group is already open, and it
  lists `▸ Blair Roster`, `▸ Dana Roster` in that order.
- **Rate.** Set `select[aria-label="Playback speed"]` to 1.5, click a sample:
  its `playbackRate` is 1.5.
- **Whose voice.** `fetch` each listed speaker's clip in the page and count
  zero crossings of the PCM (skip the 44-byte header): Dana's is 440 Hz and
  Blair's 880 Hz, 80,000 frames each.
- **Audio gone (410).** This covers the **reclaimed** path only (the artifact
  row stamped by the product sweep), not a file deleted by hand:

  ```bash
  uv run python tools/e2e_browser_lifecycle.py reclaim-source --database-url "$DSN" \
      --run-id "<VOICE_SOURCE_RUN_ID>"      # → RECLAIM PASS
  ```

  Without reloading, click Dana → 410, and a visible `p[role="status"]` inside
  `.me-segment-actions` reads "The recordings with Dana Roster's confirmed lines
  can't be played anymore." It stays after the menu closes. **Hear this voice**
  still plays the open recording. Do not call `response.text()` on the 410 in
  `browser_run_code_unsafe`; it never resolves there. Read the status only.

Reconcile (same before and after; the only POSTs in the server log should be
the editor's `claim` and `refresh`):

```bash
--run-id "<RUN_ID>" --expect '{"verified_segment_indexes":[],"corrections":{},
  "progress":{"verified":0,"total":5},"expected_annotations":0,
  "label_rulings":{"S1":{"decision":"assign","speaker":"Blair Roster"}}}'
--run-id "<VOICE_SOURCE_RUN_ID>" --expect '{"verified_segment_indexes":[],"corrections":{},
  "progress":{"verified":0,"total":2},"expected_annotations":0,
  "label_rulings":{"S0":{"decision":"assign","speaker":"Dana Roster"},
  "S1":{"decision":"assign","speaker":"Blair Roster"}}}'
```

### Cleanup fixture (#754)

Seed with `--fixture cleanup` and open
`/runs/<RUN_ID>/transcript?text=corrected&read=1&timestamps=false`.
The four segments have faithful word timings and two speakers. Segment 0 keeps
its `everyone` to `everybody` correction. This lane only reads data.

- Drive all four filter states using **Remove/Show filler words** and
  **Remove/Show repeated words**. With both on, expect `S0: Hello everybody,
  we go we`, an unnamed continuation `we stay.`, and `S1: You know.`. The two
  occurrences of `we` across the removed S1 filler turn must both survive.
- Check counts: fillers only leaves out 3 filler words; repeats only leaves
  out 2 repeated words; both leaves out 3 filler words and 2 repeated words.
  Each note starts with `Left out` and ends with `The saved transcript is
  unchanged.` Neither filter on shows no note.
- Switch raw, enhanced and corrected tabs, then toggle timestamps. Each link
  keeps both active filter states. Raw uses `everyone`; enhanced and corrected
  use `everybody`.
- Check every menu href: Markdown copies carry the active filters, reading-copy
  txt links also carry `style=turns`, and **Read on screen** keeps the filters.
  Timed txt, subtitle, data and translated links keep their original options.
  The filtered menu says `Reading copies leave out the same words as this page.`
- On corrected text with timestamps hidden, fetch the menu's Markdown reading
  copy and compare its speaker and continuation paragraphs with the page.
  The primary menu links always use corrected text, including from other tabs.
- This fixture retains ordinary words in every state, so it cannot exercise
  `The filters left out every word. The saved transcript is unchanged.`
  That case remains covered by the integration test's filler-only run.

Reconcile with no edits: `verified_segment_indexes: []`, `corrections: {}`,
`progress: {verified: 0, total: 4}`, and `expected_annotations: 0`.

#### Filler list (#753), same `cleanup` seed

Run after the #754 checks above; it writes only the `app_settings` filler
columns, which the reconcile does not read. Check them with `psql` against the
e2e database.

- Open `/settings#fillers` (General tab). Expect 12 unchecked **Also remove**
  boxes, the "Removed by default (preset en-1)" line, the never-removed caution, and
  `Removed now: erm, uh, uhh, uhm, um, umm`.
- Tick **you know**, type `um` and `hm` (two lines) into **Words or phrases to
  keep**,
  click **Save filler words**. Expect a 303 back to `/settings#fillers`, the box
  still ticked, the add textarea empty, `Removed now: erm, uh, uhh, uhm, umm,
  you know`, and a no-effect line naming `hm`. The row holds
  `{"en": ["you know"]}` and `{"en": ["um", "hm"]}`.
- Tick **I mean**, type `um2` into the add textarea and save. Expect a 422 on the
  General tab with `Words to remove: Use letters, ...` in a `role="alert"`, both
  boxes ticked, `um2` still in the textarea, `Removed now` unchanged, and the row
  unchanged. The one console error is the 422 document load itself.
- Open the corrected read view with `&fillers=drop`. Expect `S0: Hello
  everybody, um we we go we`, `we stay.`, `S1: You you know.` (the phrase is
  not set off, so it stays) and the note `Left out 2 filler words. The saved
  transcript is unchanged. Filler words are matched using your filler list.`
  with the link going to `/settings#fillers`. Fetch the menu's corrected
  Markdown reading copy: the same paragraphs, then `<!-- Filler words left out:
  2. Preset en-1; also removed: you know; kept: um. The saved transcript is
  unchanged. -->`.
- Untick everything, clear both fields and save. Both columns are NULL, and the
  read view note is back to `Left out 3 filler words. The saved transcript is
  unchanged.` with no link.

### Word marks (#757)

Seed a fresh `--fixture cleanup` run after resetting any filler-list changes
from #753. Open `/runs/<RUN_ID>`, click **Open in editor**, and wait for the
claim token and edit textarea on `/media/<MEDIA_ID>/editor?run=…&token=…`.
The same four-segment fixture serves this lane. Its final `know.` is stored as
` kn` + `ow.` timing tokens, preserving the text while enabling an interior cut.

- **Detection on mount.** Exactly one unfocused `GET /review/<RUN_ID>/word-marks`
  (200). `[data-seg-text] .wm-removed` underlines `uh` and `erm`; each title
  says `removed by filler clean-up`. Segment 0 is unmarkable: its domain-pack
  substitution (`everyone` to `everybody`) no longer maps to the recorded words,
  so it has no unit underline. Every `[data-seg-text].textContent`
  equals the corresponding island segment's `text`, including punctuation.
- **Mode and focus.** Select segment 1 (`uh`), move focus off the textarea and
  press `c`. The **Clean-up mode** region and **Exit clean-up c** button appear,
  and a `GET …/word-marks?segment=<SEGMENT_1_ID>` returns all its units.
  `[data-word-unit="0"]` is a focusable `.tp-cleanup-word` button. On segment 3,
  use Tab and ArrowLeft/ArrowRight to reach `erm`; each moves focus without a POST.
- **Keep and undo.** Focus `uh` on segment 1, press `f`: exactly one
  `POST …/segments/<SEGMENT_1_ID>/word-marks` with `action=keep`, `start=0`,
  `end=1`, claim `token` and a fresh `nonce` (200). The button has `.wm-keep`,
  title `kept`, and a **Clean-up mark applied.** toast. Click **Undo**: one
  `POST …/undo/word-mark` with `mark_id` from the write response, claim
  `csrf_token`, `token` and `nonce=undo:<markId>` (200). The word returns to
  `.wm-removed`, and the toast disappears. Repeat keep on `uh` in segment 1
  and leave it kept for the reading-copy check.
- **Omit ordinary text.** Focus `stay.` in segment 2 (`start=1`, `end=2`), press
  `o`: one mark POST with `action=omit` (200). The button has `.wm-omit` and
  title `omitted`. Focusing an ordinary unmarked word and pressing `f` sends
  no POST and shows `Only a word the filler list removes can be kept.`
- **Reading copy.** Open a second tab at
  `/runs/<RUN_ID>/transcript?text=corrected&read=1&timestamps=false&fillers=drop`.
  Assert kept `uh` survives, `stay.` is absent, and unkept `um`/`erm` are
  absent. Fetch the Markdown reading copy and assert the same keep/omit result.
- **Interior split refusal.** Back in the editor, select segment 3, enter
  clean-up mode and omit the whole `know.` unit (`start=3`, `end=5`). Click
  **Split**: clean-up mode leaves. Click the `ow.` split token (cut index 4).
  Exactly one `POST …/segments/<SEGMENT_3_ID>/split` returns 409 with
  `This word has a clean-up mark. Clear it first, then split.`; the editor
  displays that message, and the parent remains unsplit. Re-enter clean-up,
  focus `know.`, and click the mode bar's **Clear** (`action=clear`, 3..5).
- **Correction clears marks.** Select segment 2, change its text to
  `We leave now.` and save. `POST …/text` returns
  `marksCleared=1`; the editor says
  `1 clean-up mark was cleared because the text changed.` and refreshes
  `GET …/word-marks`. The old keep/omit styles disappear; entering clean-up
  explains that the changed text cannot be marked when it no longer maps.
- **Exit and suppression.** Press Escape on a unit button: mode region gone,
  **Clean up c** has `aria-pressed=false`. With the textarea focused, c/f/o
  produce no marks requests; Escape only blurs it. With the shortcuts dialog
  open, Escape closes the dialog and leaves clean-up mode active underneath.

Check the append-only rows after each write/undo using `psql` (use the normal
Postgres DSN, without SQLAlchemy's `+psycopg` suffix):

```sql
SELECT s.segment_index, m.seq, m.start_word_index, m.end_word_index,
       m.action, m.voids_mark_id, m.operator
FROM segment_word_marks m
JOIN transcript_segments s ON s.id = m.segment_id
WHERE m.pipeline_run_id = '<RUN_ID>'
ORDER BY m.seq;
```

Expect segment 1's `keep` 0..1 followed by `undo` pointing to that mark id and
another `keep` 0..1; segment 2's `omit` 1..2 followed by `clear` 1..2 after
correction; segment 3's `omit` 3..5 followed by `clear` 3..5. The refused split
adds no mark row. Reconcile with `verified_segment_indexes: []`,
`corrections: {"2": "We leave now."}`,
`progress: {verified: 0, total: 4}`, and `expected_annotations: 0`.

### Media library (#646, #682)

The Media library is always on (#682 removed its flag), so every review entry
point must reach the editor. Run the trash checks **after** the step-3
reconcile: emptying the trash purges the seeded run.

- **Review entry points.** `/review/<RUN_ID>` and `/review/<RUN_ID>/transcript`
  land on `/media/<MEDIA_ID>/editor?run=…`, the island mounts (wait for the edit
  textarea) and claims (`&token=` appears). On `/runs/<RUN_ID>`, **Open in
  editor** and **Transcript** do the same. `/review` lands on `/media` (200).
- **Add media.** On `/media`, **+ Add media** opens a menu; **Upload from this
  computer** shows the upload panel (drop zone, file input, folder picker,
  disabled **Submit for transcription**), and **Fetch from a link** shows the URL
  input and **Fetch and transcribe**.
- **No missing-file chip.** The seeded row carries no "Original file not found"
  chip (the seed writes the original).
- **Trash and restore.** Select the row, **Move to trash** → "Moved 1 file to trash."
  With the library now empty, **Trash →** must still render; follow it, select the
  row, **Restore selected** → "Restored 1 file from trash." and the row is back.
- **Empty trash asks first.** Trash the row again, open the trash view, click
  **Empty trash permanently** with a dialog handler that **dismisses** → one
  `confirm` dialog, **no** `POST /media/empty-trash`, the row still listed. Click
  again and **accept** → exactly one POST, "Permanently deleted 1 file.", and
  "Trash is empty."

## 3. Reconcile durable state, then always clean up

Build the expectation from what you drove (segment indexes verified, index→text
corrections, and the `verified`/`total` counter), and hand it to the tool — the
fail-closed verifier over `segment_review_states` (the browser was the sole
writer):

```bash
uv run python tools/e2e_browser_lifecycle.py reconcile --database-url "$DSN" \
    --run-id "<RUN_ID>" --expect '{"verified_segment_indexes":[0,3],
    "corrections":{"4":"…the exact saved text…"},"progress":{"verified":2,"total":5}}'
# → RECONCILE PASS  (non-zero + a per-mismatch list on any drift)
```

Always tear down, even on failure (kills by **port**, never
`pkill -f "voxint serve"` — that restarts the dockerized `api`):

```bash
# browser_close, then:
uv run python tools/e2e_browser_lifecycle.py teardown --port 8099   # add --drop-db --database-url "$DSN" to drop the DB
```

⚠ `--drop-db` removes the same `voxint_e2e` database the pytest pipeline lane
(`tests/e2e/`) defaults to, and that lane expects the database to exist rather
than creating it. If the pipeline lane still has to run (a Gate E sequence),
either skip `--drop-db` here or recreate the database plus its `vector`
extension before the pytest lane. See
[`docs/testing.md`](../../../docs/testing.md#automated-e2e-testse2e).

Confirm afterward: port 8099 freed, `src/voxint/api/static/app/` holds only
`.gitkeep`, and `git status` is clean (no staged build artifacts, no `media-e2e`).

## Swapping the driver later

The seed and reconcile live in the tool and are tool-neutral, so if agent-driven
Playwright runs prove inconsistent, the interaction layer here can be replaced by
a Python Playwright harness without touching the seed or the verifier.
