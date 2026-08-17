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
  controls, or the waveform strip (#57: peaks fetch, region click → selection,
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
2. Open the seeded run's workbench (`/runs/<RUN_ID>`), click **Review** →
   `/review/<RUN_ID>`, click **Claim for review** (the token lands in the URL as
   `?token=…`), then follow **Review transcript →** to
   `/review/<RUN_ID>/transcript?token=…`. The `review-stepper` island mounts
   here (`[data-island="review-stepper"]`).

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
  moves to the next unverified segment.
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
  segment's text and "Reviewing segment at X.XXs" updates. Verified lines are
  re-reachable this way.
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

Note: `aria-current` on `p.tp-line` tracks the audio **playback** highlight, not
the review cursor — assert the review cursor via the edit box's value and the
"Reviewing segment at …" line, not `aria-current` (which stays put when the
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
  dispatch a click there → "Reviewing segment at 10.00s" appears, the
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

Confirm afterward: port 8099 freed, `src/voxint/api/static/app/` holds only
`.gitkeep`, and `git status` is clean (no staged build artifacts, no `media-e2e`).

## Swapping the driver later

The seed and reconcile live in the tool and are tool-neutral, so if agent-driven
Playwright runs prove inconsistent, the interaction layer here can be replaced by
a Python Playwright harness without touching the seed or the verifier.
