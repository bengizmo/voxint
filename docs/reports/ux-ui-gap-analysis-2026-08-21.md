# Voxint UX/UI gap analysis and opportunities, 2026-08-21

> **Status:** research + design analysis (no production code). Deliverable of the
> UX/UI gap-analysis session requested after the "Reading Room" visual refresh
> (epic #89) and the #86 annotation layer shipped in 0.21.0. Scope, set with the
> maintainer at the start: the **web console plus the native macOS first-run
> path**, focused on **flow, onboarding, and information architecture** on top of
> the shipped visual base, not a further visual pass. The output is this report;
> it ends with decisions to settle (§10) rather than a build plan.
>
> **Method:** studied the committed console screenshots in `docs/images/` as the
> real current state; read the prior design spike
> ([console-ux-modernization-design-2026-08-19.md](console-ux-modernization-design-2026-08-19.md)),
> the first-run docs ([onboarding.md](../onboarding.md),
> [native-macos-preview.md](../native-macos-preview.md)), and the operator how-to
> guides ([how-to/reviewing-and-adjudicating.md](../how-to/reviewing-and-adjudicating.md)
> and its siblings); then consulted two independent design panels (codex via zen
> clink, and grok-4.5) whose agreement and two disagreements are recorded in §9.

## 1. The finding in one paragraph

Voxint's review console is correct, honest, and now visually warm, but it still
**explains itself like a maintainer and offers freedom like a power tool**. The
underlying model is right: identity is a set-wide decision (a label fans out
across the whole recording), words are a line-by-line decision, and the two are
different kinds of work. What a non-technical first-run operator hits is not a
missing feature. It is **implementation vocabulary in the on-screen copy**,
**too many equally-loud choices per screen**, and **no task-first path** that
tells them where to start, what to do next, and when they are done. The
highest-leverage work is therefore a **critical-path vocabulary pass** and an
**explicit two-step review sequence (people, then words)**, both of which carry
the risk profile of copy edits rather than a rebuild. A full merge of the two
review screens is the wrong bet; it would compound the overload it aims to cure.

## 2. Current-state inventory

The console is server-rendered (FastAPI + Jinja) with React "islands" hydrated
on top for the interactive parts. Everything is local and single-operator. Top
navigation is **Dashboard, Runs, Review queue, Speakers, Settings**.

**The flagship review flow is three screens:**

1. **Review queue** (`docs/images/review-queue.png`), titled "Adjudication
   queue". Rows for each completed run: media title with the raw file path
   beneath, duration, age, a "N of M resolved" progress bar, claimed-by, and a
   **Review** button.
2. **Workbench** (`/review/{id}`, `docs/images/review-workbench.png`). One card
   per detected voice, each with **Assign / Enroll new / Exclude / Unknown**, a
   per-segment "reassign segment" control, a "**Same speaker across labels?**"
   merge panel (checkboxes plus a preview), and a "**Generate name
   suggestions**" name-hints block. Cards carry evidence: a "machine" chip, a
   "needs ruling" chip, and lines like "Cosine suggestion: Jordan Rivera (0.97,
   grounded)" or "Heard name (unverified): Priya".
3. **Review transcript** (`/review/{id}/transcript`,
   `docs/images/transcript-review.png`). A **verify-and-advance stepper** with a
   per-line edit box, a **speaker-colored waveform strip**, **split-at-a-word**,
   per-line speaker reassignment, keyboard shortcuts, and a "?" cheat-sheet.

Supporting screens: **Dashboard** (`docs/images/dashboard.png`), stat cards over
a runs-by-status table and a stage-timing table; **Speakers**
(`docs/images/speakers.png`), the durable roster with rename / merge / archive;
**Settings** (`docs/images/settings.png`), a single long page covering first-run,
appearance, features, media folders, corrections, LLM enhancement, and web
research; and the six-step **setup wizard** (`docs/images/setup-wizard.png`).

### What is already good (do not re-propose)

These are real strengths. The recommendations below must not regress them.

- **The evidence taxonomy is honest and well-labelled.** The UI distinguishes a
  *grounded machine match* (measured against enrolled voices) from a *heard
  name* (a guess someone said aloud) from *no name at all*, and never lets a
  heard name stand in for identity. This is the trust backbone of the product.
- **Uncertainty is stated without false precision.** "Uncertain is not the same
  as wrong", and no confidence percentage is invented.
- **The claim/release model and honestly-disabled controls** tell the operator
  plainly when a button will not work and why.
- **The guided tutorial** stages a synthetic three-speaker sample so the whole
  loop can be learned before real audio is involved.
- **Accessibility is invested** (#64): skip link, focus-visible ring,
  forced-colors, non-color status cues.
- **The waveform strip, keyboard shortcuts, and cheat-sheet** are genuinely
  strong review tools.

## 3. First-run friction map

Walking one first-run non-technical operator (a journalist with an interview
recording) from install to a finished transcript. Each numbered stall is a place
they would hesitate, guess, or reach for a manual.

1. **Install is a terminal path.** The Docker install is `git clone` plus
   `./scripts/install.sh`; the native macOS preview is `voxint-native.sh
   setup/up/status` with brew, uv, and node, and reading `doctor` output. For
   the stated audience this is the first and tallest wall. It is called out
   honestly in the docs as not-yet-the-packaged-release, but it still gates
   everyone who is not comfortable in a shell.
2. **Setup wizard leaks two technical step names.** "LLM enhancement" and "Model
   services" are the operator's first sight of jargon; both are optional and
   could be named by outcome. The Welcome copy also uses an emdash, which the
   house style bans and which UI copy should match.
3. **No obvious "add my own audio".** After the tutorial, the way to submit a
   real recording is not a first-class action in the navigation; it lives on the
   Runs page. The core job of the product is one click harder to find than the
   metrics are.
4. **The Dashboard answers a maintainer's question, not the operator's.** It
   opens on throughput, runs-by-status, and stage timing ("Diarize & embed",
   "Enhance & match", "Finished attempts"), and the subtitle reads "the same
   figures as GET /metrics and voxint stats". A first-run operator wants "what
   needs my attention and how do I add a recording", not telemetry.
5. **Long jobs look frozen.** On modest local hardware, transcription and
   diarization are minutes, not seconds, and the first diarization run pays a
   one-time warm-up. Without an honest "this can take a while on this computer"
   state, the operator assumes it broke.
6. **The queue speaks in legal and hash vocabulary.** "Adjudication queue",
   "voices still needing a human ruling", and rows whose media is a raw hashed
   filename (`da36a6701e2c...wav`) give the operator nothing human to recognize.
7. **The workbench presents five equally-loud jobs at once.** Per-voice
   assign/enroll/exclude/unknown, per-segment reassignment, the same-speaker
   merge panel, and name-hint generation all compete for attention with no
   "start here". The evidence line "Cosine suggestion (0.97, grounded)" exposes
   the raw measurement the docs so carefully translate.
8. **The two review screens are a fork, not a path.** Identity (workbench) and
   words (stepper) are separate routes the guide says can be done "in either
   order". That is false flexibility for this reader: it doubles the orientation
   cost and never says which to do first.
9. **"Read-only transcript (raw ASR evidence, all variants)"** on the stepper is
   three pieces of jargon in one line ("ASR", "evidence", "variants").
10. **No explicit definition of done.** Nothing tells the operator whether an
    Unknown ruling, an excluded voice, or an unverified line blocks completion.
    "The run drops off the queue" is a maintainer's completion signal, not a
    reassuring one.
11. **Recovery is unclear and therefore frightening.** A wrong Enroll teaches
    the machine a lasting mistake, a merge feels irreversible, and there is no
    visible autosave state. Non-technical users freeze at exactly the actions
    that feel permanent.
12. **The finished thing has no obvious home.** Export lives inside the review
    screens as a download menu. The operator's goal is a document to keep or
    share, and the product presents that as a sub-action of a pipeline rather
    than an end state.
13. **Settings is one long scroll with a dependency web.** The Features block is
    On/Off/"Use installation setting" tri-state radios with chained
    prerequisites ("Requires LLM enhancement and speaker name suggestions to be
    on"), and later sections expose "compose.llm.yaml", "LLM_API_KEY",
    "SearXNG", and punycode. It is the single most overwhelming page.
14. **Turning on the optional AI or web features silently changes the privacy
    promise.** For an audience that chose Voxint because "nothing leaves my
    machine", enabling LLM enhancement or web research is the moment data starts
    leaving, and the UI does not say so at the moment of the choice.

## 4. Gaps, grouped

The friction above collapses into five gaps.

- **Vocabulary gap.** Implementation terms (adjudication, ASR, cosine, stage
  names, env-var and compose-file names, hashes) surface in the copy a first-run
  operator reads, even though the reference docs translate them well.
- **Sequencing gap.** There is no task-first spine. The operator is not told
  where to start (dashboard), which review step comes first (people vs words),
  or when they are done.
- **Density gap.** The workbench and the settings page each present many
  equally-weighted controls with no progressive disclosure, so the important
  decision does not stand out.
- **Confidence-and-recovery gap.** Long-job feedback, a definition of done,
  visible autosave, and undo for the scary actions (enroll, merge) are missing
  or implicit.
- **Onboarding-reach gap.** The in-product experience is only reachable after a
  developer-shaped install, which excludes much of the stated audience before
  the console is ever seen.

## 5. Prioritized opportunities

Impact is the effect on a first-run non-technical operator's success. Effort is
rough build cost. **Tag**: `copy`, `flow`, `IA`, `polish`, `net-new`.
**Disposition** states, per the power-user balance, whether the move *hides*
(defers behind disclosure, still reachable), *defers* (out of the happy path),
*removes*, or *adds*. Nothing here removes a capability the single power operator
relies on.

| # | Opportunity | Impact | Effort | Tag | Disposition |
|---|---|---|---|---|---|
| 1 | **Critical-path vocabulary pass.** A UI string table for the happy path only (queue, workbench, transcript, dashboard subtitle, primary settings labels, the two wizard steps). "Adjudication queue" becomes "Review"; "human ruling" becomes "reviewed by you"; "Cosine suggestion (0.97, grounded)" becomes "Strong voice match" with the number behind a "Why this match?" reveal; stage names become outcomes ("Find speakers"); hex titles become "original filename, date" with the hash secondary. Docs and API stay precise. | High | Low | copy | hide |
| 2 | **Explicit two-step review sequence.** Keep both routes; wrap them in one run-level header, "1. Who is speaking, then 2. Check the words", with one dominant Continue action per screen and preserved progress. Retire "either order" from copy and affordances (back is fine, sideways is not promoted). | High | Low-Med | flow | add |
| 3 | **Task-first dashboard and entry.** Open on three things: **Add audio**, **Continue review (N)**, and the last finished run or export. Demote throughput, runs-by-status, and stage timing behind a "Show run details" section. Add an **Add audio** action to the Runs header; do not add a new nav item. | High | Med | IA | defer |
| 4 | **Workbench progressive disclosure.** Lead with one detected voice and one instruction: listen, then Known person / Add person / Not a person / Not sure. Park the same-speaker merge until at least two labels exist (or the operator opens "These sound like the same person?"), keep name hints inside the relevant card, and move segment-level reassignment to the transcript step. No new features; subtract competition. | High | Med | IA | hide |
| 5 | **Settings in two strata.** "Need to change" (language/vocabulary, quality-vs-speed, one "Suggest names from context" master toggle) versus "Advanced" (anything with a yaml, env-var, key, or endpoint). Replace tri-state radios with plain On/Off plus "uses app default" helper text on override, or a "Reset to default" link. | Med | Med | IA | hide |
| 6 | **Honest long-job state.** Plain-language current stage, elapsed time, and a candid "this can take a while on this computer", with no fabricated ETA. | Med | Low | polish | add |
| 7 | **Definition of done.** A compact completion checklist per run that distinguishes blocking items from acceptable uncertainty (Unknown and Exclude are valid endings, not errors). | Med | Low-Med | flow | add |
| 8 | **Recovery confidence.** Visible autosave state, one-click undo for enroll and merge, and plain confirmation only on genuinely destructive actions. | Med | Med | polish | add |
| 9 | **Export as an end state.** Make "here is your finished transcript, keep or share it" a first-class outcome of finishing a run, not only a download menu inside review. | Med | Low-Med | flow | add |
| 10 | **Local-data disclosure at enablement.** When the operator turns on LLM enhancement or web research, state plainly what leaves the machine and to which service, at the moment of the choice. | Med | Low | copy | add |
| 11 | **Import-failure clarity.** On an unreadable file (a video with sound, an odd `.m4a`, a missing audio track), say "we could not read this file" and list accepted types, never a stack trace, and confirm the source is untouched. | Med | Low-Med | polish | add |
| 12 | **Packaged native first-run.** A double-click or menu-bar launch and a "drop audio here" first screen, tracked as the packaging child (#73). This is the gate for much of the audience, but it is a separate release track, not an in-console UI change. | High (reach) | High | net-new | defer |

**The top three if only three ship:** #1 (vocabulary), #2 (two-step sequence),
#3 (task-first entry). They are the cheapest and the most load-bearing, and each
fails safe.

## 6. The two-screen decision (identity and words)

Both panels reached the same verdict independently: **keep the two screens,
sequence them explicitly, and do not merge.**

- **Why not merge.** Identity is a set-wide decision and words are line-local;
  the mental models genuinely differ. Stacking all speaker cards, the merge
  panel, every transcript line, the waveform, and split/reassign onto one route
  would intensify the workbench overload this report already flags. The screen
  would scroll forever and the keyboard flow would weaken.
- **Why not leave as-is.** "Either order" is false flexibility for a
  non-technical operator; it doubles orientation cost every session and never
  answers "what first".
- **The path.** One recording-level review with a sticky header (filename, step N
  of 2, progress), **people first, then words**, a dominant Continue, and a quiet
  "skip for now" for the rare case where reading the words first helps identify a
  voice. Revisit a unified screen only if real use shows operators shuttling
  between the two steps despite preserved context.

The workbench overload is an argument for progressive disclosure *inside* step 1
(opportunity #4), not for a merge.

## 7. Native first-run (in scope, separate track)

The native macOS preview keeps the pipeline and numerics identical to the Docker
stack and only changes the process supervisor, which is the right engineering
call. For UX, the honest reading is that **the native path is still a developer
path**: its own docs state it is not the packaged, signed, non-technical release,
and it asks the operator to run shell subcommands and read `doctor` output. Both
panels agree that for journalists and educators, **install and first-audio can
outrank in-console polish**, because in-product UX is academic if the gate is a
shell. The move is not to dress up the shell; it is the packaging child (#73), a
double-click launch and a "drop audio here" first screen, tracked as its own
release track (opportunity #12). Everything in §5 items 1 to 11 improves the
console for whoever reaches it; #12 is what widens who reaches it.

## 8. Leave it out (bloat to avoid)

Explicit non-goals, so a well-meant "while we are in here" does not expand the
surface this audience has to learn.

- **Do not merge the two review routes into one dense mega-screen.**
- **Do not add a wizard framework, workflow customization, roles, team
  assignment, or an audit dashboard.** Single-operator by design.
- **Do not add confidence percentages, composite scores, or more pipeline
  telemetry.** The uncertainty model is already honest; do not sugar it.
- **Do not add a second onboarding product.** The guided tutorial exists; do not
  build a coach-mark tour on top of it, and do not block normal work with
  mandatory onboarding after it.
- **Do not add a top-level nav item just for importing audio.** Promote the
  action in place instead.
- **Do not expose model selection or infrastructure dependencies** outside an
  Advanced settings stratum.
- **Do not introduce projects, workspaces, or multi-run batch UX** for a
  single-operator local tool.

## 9. Panel consult: agreement and the two disagreements

Two independent panels reviewed the same friction inventory: **codex** (via zen
clink, planner role) and **grok-4.5** (via zen chat). Their strong agreement is
recorded above as the recommendation. The disagreements are surfaced rather than
averaged:

1. **What ranks first: entry or copy.** Codex ranked the **task-first dashboard**
   as move #1 and vocabulary #3; grok ranked the **critical-path copy pass** as
   #1 and task-first #4. The split is about blast radius. Copy is the cheapest
   and lowest-risk change and removes intimidation immediately; the task-first
   dashboard has slightly more reach into layout but resolves the "where do I
   start" question copy cannot. This report sequences copy first (cheapest, fails
   safe) with the task-first entry immediately after, which honors both.
2. **How high native packaging ranks.** Grok pushed harder that install and
   first-audio may **outrank in-console work** for this audience ("if the gate is
   still git plus compose, in-app UX is academic"); codex treated packaging as an
   essential but separate release track. Both agree it is not a low-effort UI
   change. The report keeps it as opportunity #12 on its own track and names it as
   the true reach gate, which is grok's point, without letting it block the
   console improvements, which is codex's.

Both panels also surfaced friction the maintainer view is blind to, folded into
§3 and §5: no definition of done, recovery/undo anxiety, long-job silence,
import-format cliffs, local-data disclosure at enablement, export-as-ending, and
queue-row status legibility.

## 10. Decisions to settle

Per the session's framing, this report ends by putting the open choices to you
rather than committing to a build plan. These are the questions whose answers
would turn this analysis into a scoped next step.

1. **Order of attack.** Ship the vocabulary pass (#1) first as a standalone,
   low-risk change, or bundle it with the two-step sequence (#2) and task-first
   entry (#3) as one "first 30 minutes" flow slice?
2. **Vocabulary reach.** Rename only the happy-path strings a first-run operator
   sees, leaving docs, API, and CLI precise, or also revisit the reference docs
   and CLI vocabulary for consistency?
3. **How far to take the workbench.** Full progressive disclosure (one voice at a
   time, opportunity #4), or the lighter "one dominant action per card, merge and
   hints behind disclosure" version that keeps today's layout?
4. **Definition of done.** Introduce a per-run completion checklist (#7), and if
   so, what counts as blocking versus acceptable uncertainty (are Unknown and
   Exclude always valid endings)?
5. **Native packaging priority.** Treat #12 as the top-of-backlog reach gate now,
   or continue improving the console for existing operators first and schedule
   packaging as its own arc?
6. **Report to plan.** Should the next session convert the chosen subset into a
   `/create-plan` phased build, or validate first with a small number of novice
   walkthroughs (install, start, next step, recovery, done) before committing?

### Decisions settled (2026-08-21)

Resolved with the maintainer at the end of this session:

- **First slice: the "first 30 minutes" flow slice.** Bundle the vocabulary pass
  (#1), the two-step people-then-words sequence (#2), and the task-first
  dashboard (#3) into one coherent onboarding-flow slice, rather than shipping
  copy alone first.
- **Native packaging (#12 / #73) advances in parallel** with the console flow
  slice, as an independent track, rather than strictly console-first or
  packaging-first.
- **Next move: convert the chosen subset to a `/create-plan` phased build.**
- **This report is committed to the public repo.**

## 11. Cross-links and provenance

- **Prior design work:**
  [console-ux-modernization-design-2026-08-19.md](console-ux-modernization-design-2026-08-19.md)
  (the "Reading Room" visual direction this report builds on).
- **First-run docs:** [onboarding.md](../onboarding.md),
  [native-macos-preview.md](../native-macos-preview.md).
- **Operator guides (the lay-reader UX contract):**
  [how-to/README.md](../how-to/README.md),
  [how-to/reviewing-and-adjudicating.md](../how-to/reviewing-and-adjudicating.md),
  and its siblings.
- **Current-state screenshots:** `docs/images/{setup-wizard, dashboard,
  review-queue, review-workbench, transcript-review, speakers, settings}.png`.
- **Audience mandate and house style:** `CLAUDE.md` ("Who it is for"); the
  `voxint-docs` skill.
- **Panels:** codex via zen clink (planner role); grok-4.5 via zen chat.
