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
  button, focus trap, Escape/close/backdrop dismiss), or the waveform strip
  (#57: peaks fetch, region click → selection,
  playhead/cursor sync) — or when a Gate E release check calls for the browser
  lane. Serial only on maintainer hardware; never public CI.
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
2. Open the seeded run's workbench (`/runs/<RUN_ID>`), click **Review** — it
   auto-claims the run and lands directly on the media editor page at
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
  (`p[aria-live="polite"]` in `.review-stepper`) advances by one and the cursor
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
  titled "Keyboard shortcuts" appears (the eight-row `<dl>` renders from the shared
  `REVIEW_SHORTCUTS` source of truth). While it is open, press `v` → **no** new
  `/verify` (the `helpOpenRef` guard suppresses the global keymap behind the modal).
  Press **Escape** → the dialog is gone. Reopen via the **"⌨ Shortcuts ?"** button
  (`button[aria-haspopup="dialog"]`) instead of the key, then dismiss with the ✕
  close control (`aria-label="Close keyboard shortcuts"`); reopen once more and
  dismiss with a **backdrop click** (mousedown+click on the fixed overlay, not the
  panel). Each dismiss returns focus to the opener. (The seed has a roster, so the
  header's inline `Assign speaker (1–9):` digit cue is present; on a roster-less run
  it is intentionally absent — not asserted here.)

### Domain-pack correction provenance (#83)

The seed freezes a two-rule pack (`_E2E_PACK_NAME`/`_E2E_CORRECTIONS` in the
lifecycle tool) and corrects **segment 0** (`everyone` → `everybody`, an
`input_base:"raw"` trace); the second rule (`quarterly synergies`) matches nothing,
so reconciliation carries one `applied` and one `no_raw_match`. All server-side
tests stay green without this, so assert it in the browser:

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
including the auto-saved `Voice 1`). Open `/runs/<RUN_ID>` and click **Review**:
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
- **Add a new person.** On S2 pick "Add a new person…" in the picker → the
  name box (`aria-label="Name the new person for S2"`) appears focused with
  **Add person** disabled until text is typed. Submitting fires
  `POST …/labels/S2/enroll`; on this seed it returns **400** ("no speaker audio
  to create an identity from": the seed's turns carry no embeddings) and the
  rail shows that message in its `[role="alert"]` without losing the card. That
  is the honest error path; do not treat it as a pass for enrollment itself.
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
