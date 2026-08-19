# Console UX modernization — design direction ("Reading Room") — 2026-08-19

> **Status:** research + design spike (no production code). Deliverable of the
> UX-polish spike requested after the corrections epic (#78) shipped and 0.19.0
> was cut. Mirrors the #79 spike format: a recommendation, an inspiration
> survey, a token system, a phased plan, a decisions log, and the second-panel
> review that reshaped it.
>
> **Interactive mockup (the "after" you can see and toggle):** a self-contained
> HTML mockup of the dashboard + review screens with a working light/dark/system
> toggle lives at `internal/mockups/console-ux-modernization.html` (published as a
> private Artifact for review). Every color, radius, shadow, and type size in it
> is a CSS custom property in one `:root` block — the same single source of truth
> this report proposes.

## 1. The recommendation in one paragraph

Adopt a warm **"Reading Room"** visual direction: replace the pure-black/white
canvas and the muddy translucent-gray borders with a **warm paper-and-ink neutral
palette**, a **calm teal interaction accent** kept deliberately distinct from the
success-green (so "verified/corrected" green never collides with links/focus), a
**harmonized, subdued 8-hue speaker scale**, and a **real type hierarchy** on the
existing `system-ui` stack. Deliver it through **one small semantic CSS-custom-property
token layer** in `base.html` that `tailwind.config.ts` *aliases* (never duplicates),
so the page chrome and the React islands theme from a single source. Ship it as a
**per-journey rollout** — the review flow first (workbench **and** review-transcript
together), then the rest by operator frequency — not a big-bang restyle. **Keep the
`system-ui` stack** (no bundled font) and **keep the native `<audio>` element**
(wrapped in a styled player surface, not replaced) for the first pass; both a
bundled Inter and a fully custom audio transport are deferred, evidence-gated
follow-ups, not part of the visual slice. The theme **toggle** is worth it for this
audience but is **sequenced last**, after the two-theme token conversion is proven
under the OS signal alone. This is a **zero-new-runtime-dependency** change: tokens
+ restyled bespoke CSS + a Tailwind theme alias, honoring the no-CDN and anti-bloat
doctrines.

## 2. Current-state assessment — what makes it read as a developer tool

The console works and is accessible (issue #64), but it looks like a tool built
by and for engineers. Concretely (all in `src/voxint/api/templates/base.html:12-161`):

- **Two style layers, already intertwined.** (1) One hand-authored inline
  `<style>` block (~150 lines) is authoritative for *all* page chrome — nav,
  tables, pills, cards, forms, the wizard, the review chrome. It is placed
  **after** the Tailwind `<link>` on purpose so bespoke rules win collisions.
  (2) The seven React islands (`frontend/src/components/*.tsx`) use Tailwind
  utilities, config `darkMode: "media"`. Crucially, **the islands already lean on
  the base-CSS layer**: they reuse `.muted` (×13), `.spk-badge`, `.tp-uncertain-chip`,
  `.tp-corrected-chip`, `.pill`, `.notice`, `.visually-hidden`. This intertwining
  is the redesign's biggest lever — restyle the shared classes/tokens and the
  islands move with them.
- **No design tokens, no brand.** Colors are ad-hoc, GitHub-Primer-derived. Speaker
  accents `.spk-0..7` are eight unrelated Primer hues. Almost everything else is
  **translucent gray-with-alpha** (`#8884`, `#8886`, `#8888`, `#c332`, `#2823`, …)
  for borders and pill fills — cheap and theme-agnostic, but muddy and unbranded.
- **Flat hierarchy.** `h1` is `1.4rem`, `h2` is `1.1rem`; body is `system-ui` at
  `1.45` line-height. Nothing signals importance; the dashboard reads as an
  undifferentiated wall of bold labels and thin-ruled tables (see
  `docs/images/dashboard.png`).
- **No elevation, no spacing system, no motion.** Every container is a 1px `#8886`
  border at `.5rem` radius. There are no cards that feel like cards.
- **The flagship review screen exposes the raw browser `<audio>` control**
  (`docs/images/transcript-review.png`) — the single most "unfinished" element on
  the page, and inconsistent across browsers.
- **Theme is OS-only.** `:root { color-scheme: light dark; }` — no in-app control.
- **Accessibility is already invested and must not regress (#64):** skip-link,
  `:focus-visible` ring, `forced-colors`, `.table-wrap` scroll containment, and
  deliberately non-color-only status cues (uncertain = dashed underline **plus** a
  text chip; corrected-by-domain-pack = a distinct bordered chip). **No
  `prefers-reduced-motion` handling exists yet** — any new motion must add it.

## 3. Inspiration survey — what "modern + trustworthy" looks like for this audience

Surveyed transcription/audio tools, local-first apps with praised UIs, and token
methodologies. Concrete takeaways, filtered for portability to a no-CDN,
htmx+islands app:

- **Descript** (real screenshots studied, v90 redesign). The signature move is a
  **warm off-white document surface** — *not* pure white — carrying **generous,
  editorial transcript typography**; **floating rounded cards with soft shadows**
  (real elevation) on a light chrome; a **grouped-cards properties panel** with
  quiet borders; a **restrained near-black primary action** plus a small selection
  accent (no rainbow); and **edit/state color used sparingly** as highlights
  (peach tint, strikethrough), never as the whole surface. The lesson: calm,
  document-first, elevation + warmth = "trustworthy," not flashy.
- **MacWhisper** (self-hosted Whisper GUI, closest peer). Its praise is entirely
  about feeling like a **clean native app** — simple, uncluttered, per-file model
  choice surfaced without ceremony. Trust here is *nativeness and restraint*, not
  visual ambition. Directly portable.
- **Calm-productivity / local-first (Actual Budget, Obsidian, Linear, Tailscale
  admin).** Recurring pattern: **neutral, slightly-hued backgrounds**; a **single
  restrained accent**; **light, purposeful micro-motion** (never decorative);
  clarity of numbers over cleverness. Obsidian in particular themes entirely
  through **CSS variables** — exactly the mechanism proposed here.
- **Radix Colors** (a *token methodology*, adoptable as CSS variables without
  shipping the library). Its 12-step scale assigns fixed jobs to steps — 1–2 app
  backgrounds, 3–5 component backgrounds, 6–8 borders (6 non-interactive, 7
  interactive, 8 strong/focus), 9–10 solid, 11–12 text (contrast-guaranteed).
  We do **not** need 12 steps for a console this size, but the *discipline* —
  name colors by their job, guarantee text contrast, provide a matched dark set —
  is the right backbone for our token layer.

**Synthesis:** the target is Descript's warmth + MacWhisper's restraint, delivered
through Obsidian-style CSS variables with Radix-style semantic naming. Nothing in
that requires a CDN, a framework, or a webfont.

## 4. Proposed design system — "Reading Room" tokens

One `:root` block is the source of truth. **Semantic names, not primitives** — a
color is named by its job (`--surface`, `--accent`, `--warn`), so a consumer never
hardcodes a hue. The full set below is the *vocabulary*; §6 says to introduce only
the subset the first slice actually consumes.

**Palette — warm paper & ink (light → dark):**

| Token | Light | Dark | Job |
|---|---|---|---|
| `--paper` | `#f7f4ee` | `#17191c` | app canvas (warm; *not* pure black/white) |
| `--surface` | `#fffdf9` | `#1e2126` | cards, nav |
| `--surface-2` | `#f1ece2` | `#262a30` | inset / subtle raise / row hover |
| `--ink` | `#23201b` | `#ece7dd` | primary text |
| `--ink-2` | `#5c564c` | `#b2aca2` | secondary text |
| `--ink-3` | `#8a8477` | `#807a70` | muted / captions |
| `--line` | `#e6dfd2` | `#2e333a` | hairline borders |
| `--accent` | `#2f7d76` | `#5eb8ae` | brand · links · focus · primary (calm teal) |
| `--accent-ink` | `#1f5c56` | `#7fccc3` | accent text on paper (AA) |
| `--ok` | `#1a7f37` | `#3fb950` | verified / completed / corrected |
| `--warn` | `#9a6700` | `#d29922` | uncertain / caution |
| `--danger` | `#b3261e` | `#f2685e` | failed / error |
| `--info` | `#2f6fb0` | `#6ba7dd` | running |

Each semantic status also gets a `-soft` fill for pill/chip backgrounds. **The
accent is teal, chosen to be distinct from `--ok` green** so brand/interactive
never reads as "success"; today's `--focus-ring` blue and green `--corrected`
already hint at this split — the token layer formalizes it.

**Speaker scale (`--spk-0..7`).** Replace the eight unrelated Primer hues with a
**harmonized, subdued** set at consistent lightness/chroma per theme, so no speaker
"shouts." Speaker color stays **supplemental** (the raw label text and the
`.spk-badge` remain the primary cue) — this honors the non-color-only doctrine and
avoids "confetti." The existing per-theme `@media` swap structure
(`base.html:100-107`) is reused; only the values change.

**Type.** Keep the `system-ui` stack (§7). Add a real, restrained scale as tokens
(`--t-xs .75` → `--t-2xl 2.25rem`), editorial line-heights (body 1.55, transcript
~1.65–1.7, headings 1.2), `font-variant-numeric: tabular-nums` on all timecodes/
durations/counters, and `ui-monospace` for time/IDs. **This is the single biggest
cheap win** — the current flatness, not the font, is what reads as unfinished.

**Space / radius / elevation / motion.** A 4px spacing step, three radii, and —
used *sparingly* (see the panel disagreement in §9) — one or two soft shadows for
genuinely layered/sticky surfaces. Motion tokens exist but ship **only** with a
justified animation, each gated on `prefers-reduced-motion`.

### The token delivery mechanism (the architectural crux)

- **`base.html`** defines the tokens in `:root` and consumes them directly. It can
  stay the later, authoritative stylesheet; the point of tokens is that *source
  order stops being a design mechanism* — semantic classes + variables win by
  meaning, not by cascade luck.
- **`tailwind.config.ts`** *aliases* the same variables via
  `theme.extend.colors` / `borderRadius` / `boxShadow` — it must **not** duplicate
  hex values. **Critical detail (surfaced in review):** islands use Tailwind
  **opacity modifiers** (e.g. `bg-sky-500/20`), so color tokens Tailwind consumes
  must be exposed as **RGB channels** and mapped as
  `rgb(var(--accent-rgb) / <alpha-value>)` — a plain `var(--accent)` alias breaks
  `bg-accent/20`. This is a small but load-bearing implementation constraint.
- Because islands use **zero `dark:` variant utilities** (verified) and inherit
  `color` + reuse base classes, restyling the tokens/base-classes restyles the
  islands with them, and a future `data-theme` toggle needs no `dark:`-variant
  reconfiguration.

## 5. Before / after direction

The **`internal/mockups/console-ux-modernization.html`** mockup renders the
dashboard and the review-transcript screen in the proposed system, with a live
light/dark/system toggle. The "after" images below are captured from that mockup
(neutral placeholder data only); the "before" images are the current console.

### Dashboard

| Before (current) | After — "Reading Room" |
|---|---|
| ![Dashboard — current](../images/dashboard.png) | ![Dashboard — proposed, light](images/proposed/dashboard-light.png) |

The wall-of-tables becomes **summary-first stat cards** (backlog, runs, roster,
completed) above the detail tables; status rows carry cohesive **semantic pills**;
stage timing gains an inline **mini-bar** so relative duration reads at a glance.
Warm canvas, quiet borders, clear type hierarchy. Dark:
![Dashboard — proposed, dark](images/proposed/dashboard-dark.png)

### Review transcript (flagship)

| Before (current) | After — "Reading Room" |
|---|---|
| ![Review — current](../images/transcript-review.png) | ![Review — proposed, light](images/proposed/review-light.png) |

A focused **review header card** (progress track + a **styled player surface**
wrapping the audio control + the **speaker-colored waveform**), then an
**editorial transcript list** where each line has a speaker color spine, a mono
timecode, and the existing uncertain/corrected/verified chips restyled onto
tokens. Note the single biggest visual delta from the current screen: the raw
browser `<audio>` control is replaced in the mockup by a styled transport — see
§9 for why the first pass **wraps** the native control instead of replacing it.
Dark: ![Review — proposed, dark](images/proposed/review-dark.png)

> **Screenshot provenance.** Before: `docs/images/{dashboard,transcript-review}.png`
> (regenerated 2026-08-19). After: `docs/images/proposed/{dashboard,review}-{light,dark}.png`,
> captured from the mockup at 2× and quantized. These are *proposals*, not shipped
> UI; they use neutral placeholder data (no internal hostnames/IPs).

## 6. Implementation plan (phased, narrowed after review)

The plan below is the review-revised sequencing (§9). The guiding rule: **introduce
a token only when a real consumer or a theme boundary needs it** — no speculative
design-system machinery.

- **Phase 0 — Baseline & inventory.** Regenerate current light+dark screenshots of
  every screen at desktop **and** narrow widths, capturing flagship *states*
  (unclaimed / claimed / audio-unavailable / uncertain / corrected / split / edit
  error / **JS-off fallback**). Inventory *all* color decisions to migrate — not
  just hex literals but the **stock Tailwind palette classes** `bg-sky-500/20`
  (active-segment tint) and `bg-amber-500/30` (uncertain highlight) in
  `TranscriptPlayer.tsx`, the `NEUTRAL_BAR = "#88888c"` and `border-[#8886]` in
  `WaveformStrip.tsx`. This inventory is the finite migration checklist.
- **Phase 1 — Minimal semantic token layer.** Define only the tokens the shell +
  review journey consume (surfaces, ink, line, accent, focus, active-segment,
  statuses, speaker accents, core radii, ≤2 shadows). Light defaults in `:root`;
  system-dark overrides in a guarded `@media`; **preserve `forced-colors`**. Map
  only the required Tailwind colors/shadows to the variables (RGB-channel form).
  Replace the four island color literals/palette-classes and the waveform constant
  with tokens. **Gate:** nothing in the flagship depends on a stock Tailwind color
  or a hardcoded canvas color; light+dark meet WCAG AA.
- **Phase 2 — Restyle the shell + the whole review journey.** Type scale, page
  width, nav, buttons, forms, cards, chips, notices, tables, focus states — then
  style the **workbench and review-transcript as one continuous flow** (they are
  one flagship journey; styling only one route risks a visibly broken handoff).
  Wrap the native `<audio>` + speed control + capability banner + waveform in one
  coherent player surface. Shadows sparingly; raw labels and non-color cues intact.
  **Gate:** the journey works with React, with an island failure, and **with JS
  disabled**; keyboard review + htmx swaps preserve focus and state.
- **Phase 3 — Roll out by operator frequency.** Dashboard/runs tables → queue/
  speakers cards → settings/setup wizard. Small slices, each compared to its
  baseline; remove obsolete literals only after their last consumer migrates.
  Explicit stop criteria so mixed old/new surfaces don't linger.
- **Phase 4 — Theme toggle (deferred; see §8 decision).** Only after the two-theme
  redesign is accepted under the OS signal. Local-only persistence; a guarded
  pre-paint resolver; set the **resolved** `color-scheme`; a **waveform-canvas
  repaint signal** (§9); handle the cross-tab `storage` event and OS changes while
  in system mode.
- **Phase 5 — A11y & release qualification.** Update the brittle literal-CSS
  contract assertions to the new theme contract; run frontend lint/typecheck/build
  (the no-CDN guard runs here) + `uv run pytest tests/contracts` if
  `tailwind.config.ts` or the island build changes (`test_frontend_build.py`);
  exercise keyboard review, screen-reader names, forced-colors, 200% zoom, narrow
  layout, both-theme contrast, JS-off. Add `prefers-reduced-motion` for any motion
  that actually shipped. Regenerate the docs screenshots **last**, after behavior
  and palette stabilize (recipe: memory `voxint-docs-screenshot-capture`).

## 7. Font decision — refined system stack (Inter deferred)

**Decision: keep the `system-ui` stack; do not bundle a webfont in the first pass.**
Rationale (and the second panel pushed hard for this): the current problem is the
**flat scale and lack of hierarchy**, not the typeface. `system-ui` already renders
SF Pro / Segoe / Roboto / Cantarell — all excellent. A bundled Inter adds image
bytes, subsetting + license provenance, another cross-platform rendering variable,
and — ironically — can make the console read as *generic SaaS*, contributing less
trust than hierarchy, line length, spacing, and control consistency do. Under the
no-CDN guard a webfont **must** be a self-hosted subset `woff2` (Google Fonts is not
allowlisted); Inter is OFL and self-hostable, so it remains a **documented,
evidence-gated Phase-2+ option** — revisit only if side-by-side screenshots on
macOS/Windows/Linux show a material defect the system stack cannot fix. (The mockup
shows Inter to preview that ceiling; the shipped recommendation is the system stack.)
Pair with `ui-monospace` for timecodes/IDs and `tabular-nums` everywhere digits align.

## 8. Decisions the spike was asked to make

- **Theme toggle: YES, but sequenced last (Phase 4), local-only.** It earns its
  place for a non-technical audience running long transcription sessions (many will
  want to force light for a friendlier feel), *but* it is not part of the minimum
  visual direction. Persist in **`localStorage`, not `app_settings`** — theme is
  per-device, and a DB setting would mean a migration + a round-trip for zero
  cross-device benefit. Implement via `data-theme` on `<html>`, with the token
  blocks responding to both `@media (prefers-color-scheme: dark)` (guarded
  `:root:not([data-theme="light"])`) and `:root[data-theme="dark"]`. Present it as
  a labelled **System / Light / Dark** control in Settings (a three-state icon
  cycle is too opaque for this audience), not a bare icon.
- **Token delivery: one semantic `:root` block, Tailwind aliases it (RGB-channel
  form).** CSS owns the values; Tailwind never duplicates them. `base.html` stays
  authoritative but no longer *relies* on collision ordering.
- **Custom audio transport: NO for the first pass — wrap the native control.**
  See §9; this was the sharpest disagreement.
- **Dependency verdict: zero new runtime dependencies.** Tokens + restyled bespoke
  CSS + a Tailwind theme alias. No UI framework, no webfont, no CDN. This is the
  anti-bloat-compliant answer.
- **First implementation slice:** Phase 1 (minimal tokens) + Phase 2 scoped to
  typography, nav, buttons, notices, chips, and the review-journey player surface —
  one reviewable flagship build before any secondary screen.

## 9. Second-panel review and the changes it forced (codex, planner role)

Per the `/create-plan` house style, the direction went to **codex (via `zen clink`,
planner role)** before finalizing. Codex inspected the actual repo (base.html, the
island TSX, the no-CDN guard, the a11y tests) and returned a high-confidence
"proceed, but narrow it." The disagreements are recorded here rather than quietly
resolved:

1. **Custom audio transport — codex says cut it (I initially proposed replacing the
   native control).** Its argument: the repo already relies on the native `<audio>`
   for playback-rate control, capability gating, segment seeking, teardown, media
   keys, and the **JS-off fallback**; a fully custom transport re-implements all of
   that and takes on the full a11y surface (accessible seek slider, elapsed/duration
   announcements, buffering/error states, touch targets). **Resolution: accepted for
   the first pass** — wrap the native element in a styled player surface; treat a
   custom transport as a *separately-scoped, gated* product feature (its own issue),
   not part of the restyle. Honest caveat retained in the report: a wrapper improves
   the *surroundings* but cannot fully rebrand the browser's native control chrome,
   so the "raw control" look is reduced, not eliminated — which is exactly why a
   custom transport may still be worth a dedicated issue later. The mockup keeps the
   custom transport as the visible aspiration to inform that decision.
2. **Factual correction codex forced (verified true).** My claim "islands contain
   only 2 hardcoded colors" was **incomplete**: `TranscriptPlayer.tsx` also uses the
   stock Tailwind palette classes `bg-sky-500/20` (active segment) and
   `bg-amber-500/30` (uncertain) with **opacity modifiers**. Consequences folded in:
   (a) the migration inventory must include stock palette classes, not just hex
   literals; (b) the token→Tailwind alias must use **RGB-channel** variables so
   `/20`-style utilities keep working (§4).
3. **Waveform-canvas repaint gotcha codex forced (verified true).**
   `WaveformStrip.tsx` resolves `--spk-accent` via `getComputedStyle` but only bumps
   its repaint epoch on `matchMedia("(prefers-color-scheme: dark)")` change. A
   `data-theme` toggle that doesn't move the OS signal would leave the canvas
   **stale**. Fix folded into Phase 4: also observe the `data-theme` attribute
   (MutationObserver on `documentElement`) or dispatch a theme-change event, bumping
   the same epoch. Also: set the **resolved** `color-scheme` (not a permanent
   `light dark`) so native controls follow the explicit theme, not the OS.
4. **Over-designed token framework — narrow it.** Don't build a full spacing/radius/
   elevation/motion/primitive-color taxonomy up front. **Resolution: accepted** —
   §6 now says introduce only tokens with an immediate consumer or theme purpose.
5. **Soft shadows on every card fight "calm."** Codex prefers tonal surfaces + quiet
   borders, elevation reserved for genuinely layered/sticky UI. **Partial
   disagreement, surfaced honestly:** the Descript reference *does* use soft
   elevation and reads calm, so I keep 1–2 shadow tokens — but restricted to the
   sticky review header and popovers/floating cards, with tonal surfaces + hairline
   borders as the default. This is a taste split worth the user's eye on the mockup.
6. **8-hue speaker palette risks "confetti."** **Accepted** — subdued saturation,
   optimize for stable differentiation over equal prominence; speaker color stays
   supplemental to the label text.
7. **Flagship is a journey, not a page; system font over Inter; defer the toggle;
   don't add motion just because motion tokens exist; test warm backgrounds against
   dense tables / long sessions.** All **accepted** and folded into §6–§8.

Net effect of the review: the *direction* is unchanged, but the *scope* is
tighter, the *sequencing* is safer (states + full journey before palette
expansion), and two real technical defects (RGB-channel opacity, canvas repaint)
plus one factual error were caught before any code.

## 10. Risks / open questions

- **Warmth vs. sharpness.** A warm canvas can mute status tints and reduce perceived
  sharpness in dense tables. Validate the exact `--paper`/`--surface` values against
  the runs table and a long review session before locking them.
- **The native-audio ceiling.** If the wrapped native control still looks
  unacceptably inconsistent after Phase 2, a custom transport becomes a real
  candidate — but as its own issue with the full a11y checklist, not a restyle.
- **htmx swap fidelity.** Aesthetic selectors must not override island interaction
  states (active/disabled/hover/focus-visible/error/pending) or survive-a-swap
  behavior. Explicit state coverage in Phase 2's gate.
- **Contract-test churn.** Some contract tests assert literal CSS; they'll need
  updating to the theme contract (Phase 5) — do this deliberately, never by
  weakening an assertion to pass.
- **Should this become a GitHub epic?** Likely yes once this report is accepted — a
  "console visual refresh" epic with per-phase children (tokens → review journey →
  per-screen rollout → toggle). That's a follow-up decision, not part of the spike.

## 11. Reproduction / provenance of this spike

- **Inputs read:** `src/voxint/api/templates/base.html:12-161`; the island TSX
  (`ReviewStepper`, `CorrectionsEditor`, `TranscriptPlayer`, `WaveformStrip`);
  `frontend/tailwind.config.ts`; the current `docs/images/*.png` baseline.
- **Verifications run:** islands use **zero** `dark:` utilities; islands reuse base
  classes (`.muted` ×13, `.spk-badge`, `.tp-*`, `.pill`, `.notice`); island color
  decisions are `NEUTRAL_BAR="#88888c"` + `border-[#8886]` **and** `bg-sky-500/20`
  + `bg-amber-500/30` (codex catch, confirmed); `WaveformStrip` repaints only on
  `prefers-color-scheme` change (codex catch, confirmed).
- **Inspiration survey:** WebSearch + real Descript UI screenshots (studied,
  cached under the session scratchpad); MacWhisper reviews; Actual Budget /
  local-first UX patterns; Radix Colors 12-step methodology.
- **Artifact:** `internal/mockups/console-ux-modernization.html` (self-contained,
  light/dark/system toggle, both flagship screens) — the visible "after."
- **Second panel:** codex via `zen clink` (planner role), continuation
  `5079d87a-da8a-4096-8ea1-dcaba8d1ddac`; its corrections are §9.
- **Constraints honored:** no CDN (system stack, no webfont); zero new runtime deps;
  architecture preserved (Jinja + htmx + islands, `darkMode: "media"`);
  accessibility preserved (#64). No production template was modified by this spike.

## Sources

- [MacWhisper Review 2026 (LumeVoice)](https://lumevoice.com/blog/macwhisper-review-2026/)
- [Actual Budget (GitHub)](https://github.com/actualbudget/actual) · [Budgeting-app calm-UX patterns (Appthetics)](https://www.appthetics.com/blog/budgeting-apps-ux-patterns)
- [How to use Radix Colors](https://www.radix-ui.com/colors/docs/overview/usage) · [Understanding the 12-step scale](https://www.radix-ui.com/colors/docs/palette-composition/understanding-the-scale)
- [Descript screen-recording / product UI](https://www.descript.com/screen-recording) · [New Descript v90 first look](https://www.descriptmastery.com/blog/new-descript-first-look-and-reaction)
