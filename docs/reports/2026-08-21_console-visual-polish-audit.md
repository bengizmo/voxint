# Console visual-polish audit

> **Status:** Findings + implementation direction. Audit performed 2026-08-21
> against `main` @ `9635142` (branch `feat/console-visual-polish`). Evidence is a
> static read of `src/voxint/api/templates/base.html` and every console
> template/island, cross-checked against seeded-console screenshots of all nine
> screens in both themes. This is the micro-craft pass the prior two design
> reports deferred: [console-ux-modernization-design-2026-08-19.md](console-ux-modernization-design-2026-08-19.md)
> set the "Reading Room" direction, [ux-ui-gap-analysis-2026-08-21.md](ux-ui-gap-analysis-2026-08-21.md)
> covered information architecture and flow. Neither looked at pixel-level rhythm,
> button hierarchy, or interactive states.

## 1. Summary

The design system is coherent at the token level. It defines scales for type
(`--t-*`), radius (`--r-*`), color, and two restrained shadows, and the islands
reuse the base classes so a token edit restyles chrome and islands together.
What it lacks is a **spacing scale** and a **shared card primitive**. Because of
that gap the "elevated panel" recipe is hand-rewritten eight times with five
different padding pairs, an r-md/r-lg split, and a shadow that is present on some
cards but absent on visually adjacent ones. The rest of the findings are smaller
and independent: button hierarchy is missing on the two busiest action surfaces,
several interactive elements have no hover cue, and one screen (`run_detail.html`)
has no card grouping at all.

Priority order below is biggest craft win against lowest risk. Findings 2 and 3
are the most visible; finding 1 is the foundation that de-risks the rest.

## 2. Findings

### 2.1 No spacing scale; the card recipe is duplicated eight times (foundation)

`base.html` has no `--space-*` tokens. Every padding, margin, and gap is a raw
rem literal, and the values form a fine 21-step set rather than a 4px rhythm. The
clearest symptom is the card family, which never agrees on padding:

| Surface | Location | Radius | Shadow | Padding |
|---|---|---|---|---|
| `.setup` | base.html:276 | r-lg | shadow-1 | `1.2rem 1.4rem` |
| `.settings > section` | base.html:316 | r-lg | shadow-1 | `1rem 1.2rem` |
| `.label-card.roster-card` | base.html:326 | r-lg | shadow-1 | `.9rem 1.1rem` |
| `.review-head` | base.html:390 | r-lg | shadow-1 | `.9rem 1.1rem` |
| `.review-stepper` | base.html:430 | r-lg | shadow-1 | `.9rem 1.1rem` |
| `.player-surface` | base.html:427 | r-lg | none (deliberate) | `.6rem .9rem` |
| `.task-link` | base.html:484 | r-lg | shadow-1 | `1rem 1.1rem` |
| `.stat-card` | base.html:470 | r-md | shadow-1 | `.7rem .9rem` |

`.setup` is the lone `1.2rem 1.4rem`, even though its comment claims it matches
`.review-stepper` (which is `.9rem 1.1rem`). A `--t-*` type scale exists yet many
font sizes skip it for raw literals (`.95rem`, `.9rem`, `.85rem`, `.72rem`,
`.68rem` recur across ~20 rules).

**Direction:** add a `--space-*` scale and a shared `.card` primitive with
`.card--flat` and `.card--sm` modifiers; fold the eight surfaces onto it, keeping
each surface's unique rules (grid, hover, accent stripe) and its intentional
variation (player flat, stat-card small). Thread the type scale where it does not
shift the design. Spacing and radius tokens are theme-agnostic, so this touches
only the bare `:root`, not the dark blocks.

### 2.2 No button hierarchy on the two busiest action surfaces (highest visible win)

The primary style (`button.primary` / `.btn-primary`, accent fill) is used well
on `run.html`, `queue.html`, `review_transcript.html`, and `setup.html`. It is
absent where decisions actually happen:

- **`fragments/labels.html` (the "who is speaking" workbench).** The ruling row
  (base.html `.card-actions`, labels.html:247) puts Assign (:261), Enroll new
  (:272), Exclude (:281), and Unknown (:290) at equal neutral weight. The most
  common action, Assign, does not dominate, and the destructive-leaning Exclude
  reads identical to it. This is the console's busiest surface.
- **`run_detail.html`.** Submit-style and destructive actions are all neutral;
  "Delete derived audio files" (:123, labeled Irreversible) looks exactly like
  "Archive run" and "Save notes".
- **`runs.html`.** "Submit for transcription" (:26) is neutral, so the primary
  task on the page is not visually the primary.

No screen distinguishes destructive actions, though `--danger` / `--danger-soft`
tokens exist. Primary placement is also inconsistent: `run.html` and
`review_transcript.html` lead with the primary on the left, `setup.html` trails
it on the right. `merge_confirm.html` inverts the pattern outright: the go-action
"Confirm merge" (:61) is neutral while "Cancel" (:62) carries `.secondary`.

**Direction:** add a `button.danger` / `.btn-danger` variant on the existing
danger tokens (measure AA contrast in both themes before landing) and reserve it
for genuinely destructive, hard-to-reverse actions: `run_detail` "Delete derived
audio files", roster "Merge" (cannot be undone) and "Remove". Exclude and Unknown
on the labels row are routine classifications, not destruction, so they are
de-emphasized as `secondary` rather than colored red. Promote one dominant
primary per surface (Assign; the Review next-step on `run_detail`; Submit on
`runs`; Confirm merge on `merge_confirm`), and let the quieter alternatives
recede. Standardize action-row placement on primary-leading-left, and keep the
wizard's directional "Continue" trailing. The `merge_confirm` Cancel keeps its
lightweight client-side clear; converting it to htmx would need a server endpoint
for an empty response, which is more machinery, not less.

### 2.3 `run_detail.html` has no card grouping (highest-visible screen deficit)

Every other content screen groups its sections into cards; `run_detail.html`
renders as a flat wall of text. Three specific problems compound it:

- The metadata `<dl>` (:39-46) is unstyled. The `.settings dt/dd` rules are
  scoped to `.settings`, so Status / Created / Updated stack with default browser
  indentation and no alignment.
- The Manage section references `form.cancel` (:71), `form.archive` (:114),
  `form.media-delete` (:121), and `form.notes` (:173), none of which have a CSS
  rule. They fall back to default block layout, so those rows do not align with
  the one styled `form.requeue`.
- The action links (:88-101, Audio / Transcript / Review / Complete run archive)
  are an undifferentiated inline `<p>`; Review, the meaningful next step for a
  completed run, has no emphasis.

**Direction:** wrap the sections in the shared `.card`, give the `<dl>` a shared
definition-grid rule, add a shared `.action-row` for the Manage forms (modeled on
`form.requeue`), and apply the danger and primary treatments from 2.2.

### 2.4 Incomplete interactive states

- The base `button` has `:hover` and `:disabled` but no `:active` pressed state.
- Several `cursor: pointer` elements have no hover cue and rely only on the global
  focus outline: `.tag-pill` (546), `.hl-swatch` (534), `.tp-corrected-chip`
  (383), `details summary` (338/343/347/501), and body/`.sort-control`/breadcrumb
  links.
- Input focus (base.html:222) uses `var(--accent)` directly instead of
  `var(--focus-ring)`. Same resolved color today, but the two can drift.
- The `prefers-reduced-motion` block (216) nulls only `button` transitions.
  `.task-link` animates `border-color`/`box-shadow`/`background` on hover (488)
  and is not covered.

### 2.5 Smaller cleanups

- Three raw radius literals bypass `--r-*`: `kbd` `.3rem` (443), `mark.hl-*`
  `.15em` (518), `.skip-link` `.4rem` (592).
- Two genuine inline styles: `transcript.html:9,20` set `border:0;margin:.5rem 0`
  on `nav.top` to repurpose it as a tab strip. The `style="width:%"` progress
  spans elsewhere are dynamic values and are correct as-is.
- Repeated markup with no shared class: the media-identity block (`queue.html:27`,
  `runs.html:129`, `run_detail.html:9`) and the token+nonce decision-form header
  in `labels.html` (eight occurrences).

## 3. What is already right (do not regress)

- The token architecture, the dark two-block structure, and the RGB-channel
  aliasing for island opacity utilities. All contract-tested.
- `review_transcript.html` (Step 2) is the reference for correct hierarchy: one
  dominant teal primary leading left, kbd chips, a floating stepper card, a
  deliberately flat player and waveform below it.
- `settings.html` is the reference for correct card grouping.
- The accessibility investment from #64 (skip link, focus-visible ring,
  forced-colors support, non-color status cues) and the deliberate flat treatment
  of the player and annotation toolbar (elevation reserved for header cards).

## 4. Constraints on the fix

- **Dark parity is contract-tested.** `test_theme_toggle_css.py` requires the two
  dark blocks' declaration sets to be identical. Any themed-token edit changes the
  light `:root` and both dark blocks in lockstep. The spacing/radius tokens and
  the `.card` primitive are theme-agnostic and need no dark-block work.
- **Palette parity is contract-tested** (`--spk-*`, `--hl-*`). This pass does not
  need to touch them.
- **No automated contrast gate.** AA is verified by hand; the danger variant is
  the only new color and gets a measured check in both themes.
- **No new runtime dependency, no web font, no new config knob.** Settled in the
  prior reports.

## 5. Elevation rule (proposed)

State one rule and apply it through `.card` versus `.card--flat` rather than
per-surface decisions:

- Header and list-row cards that sit on the page canvas float with `--shadow-1`
  (`.review-head`, `.review-stepper`, `.settings > section`, roster cards,
  `.task-link`, `.stat-card`, `.setup`, and the `run_detail` section cards).
- In-context media surfaces stay flat: `.player-surface` and the annotation
  toolbar read as inset panels, not floats.
- `--shadow-2` is reserved for transient lift on hover and popovers. Its only
  current use is `.task-link:hover`.

## 6. Verification plan

Contract tests (`tests/contracts/test_theme_toggle_css.py`,
`test_speaker_palette_parity.py`, `test_highlight_palette_parity.py`,
`test_frontend_build.py`), then the full unit + contracts + integration suite run
alone (host is fragile under load). Re-run the seeded browser lane and compare
every screen in both themes against the pre-change baseline. Regenerate the
affected `docs/images/*.png` with the internal-IP placeholder neutralized.
Manual AA contrast check on the danger variant recorded here before landing.
