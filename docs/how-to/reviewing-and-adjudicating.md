# Reviewing and adjudicating a run

*How to turn a finished pipeline run into a transcript you trust: name the
voices, then check and correct the words.*

Voxint listens to your recording and makes its best guesses about who spoke,
when, and what they said. Those guesses are **proposals**. This
guide walks you through the review console, where **you have the final say**: you
confirm or overrule every proposal, and nothing is settled until you say so.

Review has two steps: start with the people, then check the words.

- **Step 1, [Identify the voices](#workflow-a-identify-the-voices):** decide who
  each detected voice really is (or that they should be left out). Opening a run
  takes you here.
- **Step 2, [Verify and correct the transcript](#workflow-b-verify-and-correct-the-transcript):**
  read through the words, confirm the ones that are right, and fix the ones that
  are wrong.

The console leads you from Step 1 to Step 2, and you can go back to the people at
any time. Checking the words is recommended, not required: your speaker rulings
are what settle a run.

New to Voxint? The bundled [guided tutorial](../onboarding.md#3-guided-tutorial)
walks this whole loop on a sample recording before you use your own audio; a
good place to start. This guide is the fuller reference for the same work.

**Related how-to guides:** [Add media and manage
runs](add-media-and-manage-runs.md) · [Managing speakers and
exporting](managing-speakers-and-exporting.md) · [Settings and
troubleshooting](settings-and-troubleshooting.md). See also
[onboarding](../onboarding.md) and, when the number of voices surprises you,
[interpreting diarization](../interpreting-diarization.md).

---

## Before you start

You review a run **after it finishes processing**. Adjudication does not run
while transcription is still working. A run only reaches you once the pipeline
has produced its transcript and split it by voice.

Everything lives on your own machine. Open the console at
**`http://127.0.0.1:8080/`** and sign in with the username and password you set
during install (`VOXINT_USER`, default `admin`, and `VOXINT_PASSWORD`). Nothing
leaves your computer.

### 1. Open Review and choose a run

Click **Review** in the sidebar. The **Review** page lists every
completed run that still has voices needing a human ruling. Each row shows:

- a **friendly title** (the recording's own title when it has one, otherwise a
  cleaned-up filename) with the folder name beneath it when the file came from
  a registered folder,
- the recording's **duration** and its **age** ("3 hours ago"; hover for the
  exact time),
- a **progress bar** that fills as you resolve voices ("2 of 4 resolved"), and
- a **Review** button.

You can sort **Oldest first** (the default) or **Most voices to resolve**.

![The Voxint adjudication queue: a table of completed runs, each row showing a
recording name with its folder and date, duration, age, a progress bar reading
"N of M resolved," and a Review button.](../images/review-queue.png)

Press **Review** on the run you want. In the default single-operator mode, the
workbench claims the run for this browser tab as it opens. The transcript editor
does the same when you open it. There is no separate claiming step.

> In multi-user mode, the queue adds a **Claimed by** column and you claim work
> manually with **Claim for review**. Use **Release claim** when you want to hand
> it to someone else.

---

## Workflow A: Identify the voices

Opening a run shows the **slot workbench** (`/review/{id}`). Voxint separated
the recording into voices and gave each one a placeholder label like
`SPEAKER_00`. Your job here is to say who each label really is.

Each voice gets a **card** showing what Voxint knows about it (how many turns
it took and how many seconds it spoke) plus whatever evidence it could gather
about the person's identity. Read the evidence carefully, because there are
**three very different kinds**, and Voxint labels them honestly:

- **A grounded machine match.** "Strong voice match: *Jordan*." Voxint compared
  this voice's sound to speakers you enrolled before and found a close match
  against real voice evidence. Open **Why this match?** to see the exact
  similarity score behind it. A weaker match reads "Possible voice match"
  instead. This is the strongest signal, but it is still a suggestion for
  **you** to accept.
- **A heard name (a guess).** "Heard name (unverified): *"Alex"*" or
  "Self-introduced (unverified): *"…this is Alex…"*." Someone in the audio said
  a name. That tells you a name is *probably* in the room; it does **not** tell
  you this voice belongs to that person. Treat it as a lead, not a fact.
- **No name at all.** Voxint has a voice but nothing to attach to it. You
  decide entirely.

**Grounded is not the same as heard.** A *grounded* match is measured from the
voice itself, against speakers you enrolled. A *heard* name is just words the
recording contained. Never let a heard name stand in for identity on its own;
confirm it by listening.

![A voice card in the review workbench showing three kinds of evidence: a
grounded cosine match to an enrolled speaker, a heard name marked unverified,
and a voice with no name, each with Assign, Enroll new, Exclude, and Unknown
buttons.](../images/review-workbench.png)

### Listen before you rule

Each card has a **preview this speaker** button that plays a clean stretch of
this voice, and each transcript line under it has its own play button for just
that line. Use them to confirm what you're about to decide. (If playback controls
are greyed out, a banner explains why, usually that the audio can't be reliably
lined up with the timeline; you can still scrub the main player by hand.)

### Rule on each voice

Every card gives you four choices:

| Action | What it does |
|---|---|
| **Assign** | This voice is an **existing** speaker on your roster. Pick them from the list. |
| **Enroll new** | This is a **new** person. Type their name; Voxint adds them to your roster and remembers this voice for next time. |
| **Exclude** | This "voice" isn't a person you want in the results (background noise, a TV, a passer-by). Leave it out. |
| **Unknown** | You genuinely can't tell who this is. A valid, honest ruling, better than a wrong guess. |

A card's pill shows where it stands: **needs ruling**, **assigned**,
**excluded**, **unknown**, or a **machine** suggestion still waiting for your
accept. A voice is "resolved" once you've ruled on it.

If Voxint offered **name hints** (from metadata or the transcript), you can
**Accept** or **Reject** each one. Accepting a hint records your review of it; it
does **not** assign a speaker. Assigning is always the explicit **Assign** or
**Enroll new** action.

### One person split across two labels ("same speaker")

Diarization sometimes splits **one** person into two labels: you'll see
`SPEAKER_00` and `SPEAKER_03` that are clearly the same voice. Fix it right here
with the **"Same speaker across labels?"** panel:

1. **Tick** the labels that are the same person in this recording.
2. Choose **who they are**: an existing speaker, or enroll a new one.
3. Press **Preview merge…** to see the **exact change** Voxint will make (how
   many turns and transcript segments move) before anything happens.
4. **Confirm**.

This is **run-local**: it records one ruling per label within *this* recording.
It does **not** merge identities across your whole roster; that stays a
deliberate action on the [Speakers page](managing-speakers-and-exporting.md),
and the preview points you there if two of the ticked labels are already
different roster people. Nothing here is destructive; you can re-rule any label
afterward.

> **Seeing fewer or more voices than you heard?** That's often correct behavior
> being misread: a quiet interjection, crosstalk, or a very short clip. See
> [interpreting diarization](../interpreting-diarization.md) before you assume
> it's wrong.

---

## Workflow B: Verify and correct the transcript

From the workbench, follow **Continue to checking the words →** to open the
transcript review page (`/review/{id}/transcript`). This is where you read the
words, mark the right ones as checked, and fix the wrong ones. It works as a
steady loop: **read a line → confirm it → move to the next.** When you have been
through every line, the page says so and offers to export or go back to Review.
To return to the speakers, use **← Back to the people** at the top.

At the top you'll always see a live count, **"7 of 32 segments verified"**, so
you know how far you are.

![The transcript review page: a stepper with a verify-and-advance count, an edit
box for the current line, a colored per-speaker waveform strip under the audio
player, and transcript lines with dashed "uncertain" chips on the low-confidence
ones.](../images/transcript-review.png)

### The verify-and-advance loop

The page starts on the first line that hasn't been checked yet. For each line:

- Listen to it (it plays as you land on it; **replay** any time).
- If the words are right, **Verify** it, and Voxint marks it checked and jumps
  you to the next unchecked line.
- If the words are wrong, **edit** them (below), then verify.
- **Skip** a line to come back later.

You can drive this entirely with the keyboard (see [Keyboard
shortcuts](#keyboard-shortcuts)) or entirely with the on-screen buttons,
whichever you prefer.

### Lines the model was unsure about

Some lines carry a small dashed **"uncertain"** chip. That means the transcriber
reported **low confidence** on that line; it's flagging the parts most worth a
listen, so you don't have to re-read everything. The label is deliberately
honest: **uncertain is not the same as wrong.** It's a nudge to check, not a
claim of an error, and Voxint never puts a percentage on it. (Older runs made
before this feature simply won't show the chip.)

### Fix a line's words

Click a line to bring it into the **edit box**, correct the text, and save with
**Ctrl+Enter** (**⌘+Enter** on a Mac). A few things to know:

- Voxint keeps your correction **beside** the original; it never overwrites what
  the model actually heard. Exports show your corrected wording by default, and
  the raw version is always still available.
- **Editing a line clears its "verified" mark**: corrected words should be
  re-checked, so the line rejoins the queue for a fresh confirm.
- Clearing the box (reverting to the model's wording) removes your correction.
- **Unsaved-edit warning:** if you have unsaved text in the box and try to
  verify or move on, Voxint warns you once rather than silently throwing the
  edit away. Save it (Ctrl/⌘+Enter), or repeat the action to discard and
  continue.

### Corrections your domain pack made

If you run with a [domain pack](../domain-packs.md) that declares corrections, some
lines are fixed **automatically** before you ever see them: a recurring
mishearing turned into the right spelling every time. When that happened on the
current line, you'll see a **"corrected by domain pack"** marker next to the line,
kept deliberately separate from the **"edited"** badge, which means a change *you*
made. Expand the marker to see exactly which rule fired: the phrase it matched and
what it became.

![A reviewed transcript line carrying a "corrected by domain pack" marker, expanded
to show the rule that fired (match → replace), with the run-level "Correction rules"
panel above reconciling which declared rules applied and which never
fired.](../images/correction-provenance.png)

- **Compare against the original.** Open **Original (raw) transcript** to see the
  exact words the model first heard, next to the corrected version. You can **copy**
  the raw text, or **Reset edit to raw** to drop it back into the edit box.
- **Reset doesn't save.** "Reset edit to raw" only fills the box; nothing is stored
  until you Save, so you stay in control (and the unsaved-edit warning still applies).
- **Your edit wins.** The moment you save your own wording for a line, the
  "corrected by domain pack" marker goes away; from then on the line shows *your*
  text, not the pack's automatic edit.
- **A corrected line can't be split.** Splitting a line at a word (below) is turned
  off once a correction has fired on it; Voxint tells you why rather than offering a
  cut that wouldn't work.

At the top of the page, **Correction rules** summarizes how the pack's rules did
across the whole run: how many **applied**, and which ones **never fired**. A rule
that never fired usually means the recording didn't contain that term, or the term
was split across a pause. For terms that get broken across pauses, add them to the
pack's **vocabulary** (which nudges the transcriber up front) instead of relying on
a correction after the fact.

### The waveform strip

Under the audio player sits a compact **waveform**, a colored strip where each
band is tinted for the speaker who was talking, using the same colors as the
transcript. It's a map of who spoke when. **Click anywhere on the strip to jump**
to that moment and select the matching line (overlapping speech is marked, and
stretches that were spoken but not transcribed still show up, so the picture
stays honest). If you click a spot with no transcript there, whether a silent
gap or speech that was never transcribed, the strip says so instead of doing
nothing. A marker tracks playback and shows where your review cursor is.

### Split a segment at a word

Sometimes one transcript segment actually contains **two speakers**: the
diarizer drew the boundary in the wrong place. You can cut it at the right word:

1. Press **Split at a word** to turn on split mode.
2. The current line's words become clickable. **Click the word where the new
   speaker starts**, and Voxint cuts the segment just before it.

The two halves then stand on their own, and you can give each one the correct
speaker (below). A few honest limits: a segment can only be split when its words
line up cleanly with what was transcribed (Voxint tells you plainly when one
can't be split), a split segment can't also be free-text edited (splitting and
editing are mutually exclusive), and a segment can be split into two parts, not
more, in this release.

### Reassign a segment (or half of one) to another speaker

Each line has a **speaker picker** so you can hand it to the right person
without leaving the transcript:

- **A whole segment:** with a line focused, pick a speaker from the **Assign
  speaker** menu, or press a number key **1–9** to assign it to the 1st–9th
  speaker on your roster. Press **0** to reset the line to its **detected**
  speaker (undo your override).
- **Half of a split segment:** after you split a segment, **each part gets its
  own picker**. Choose the speaker for each half independently, or pick
  **inherit** to send it back to following its label.

Reassigning changes *attribution only* (it never rewrites the words), and it
flows through to your exports.

---

## Keyboard shortcuts

The transcript review page is built to run from the keyboard. Press **?** at any
time, or click the **Shortcuts** button (it shows the `?` accelerator), for the
same cheat-sheet built into the console.

![The keyboard-shortcuts cheat-sheet: a modal dialog titled "Keyboard shortcuts"
listing v, n, p, e, j/k, 1–9, 0, and ? with a plain-language description of each,
and a note that Space and the arrow keys stay with the audio
player.](../images/keyboard-shortcuts.png)

| Key | Action |
|---|---|
| **v** | Verify this line and go to the next unchecked one |
| **n** | Skip to the next unchecked line |
| **p** | Replay the current line |
| **e** | Edit the current line's text |
| **j** / **k** | Go to and play the next / previous line |
| **1**–**9** | Assign this line to the 1st–9th speaker on your roster |
| **0** | Reset this line to its detected speaker |
| **?** | Show the cheat-sheet |
| **Ctrl/⌘+Enter** | Save an edit (while typing in the edit box) |

A few deliberate rules:

- **Shortcuts never fire while you're typing** in a text box or menu, so `v`
  types a "v" in the edit box, it doesn't verify.
- **Space** (play/pause) and the **arrow keys** (scroll) stay with the audio
  player, as you'd expect.
- **Every shortcut also has a visible, clickable control** on the page; the
  keyboard is a shortcut, never the only way in. The digit keys mirror the
  on-screen **Assign speaker** menu; `1`–`9` do nothing on a run that has no
  speakers yet, and the cheat-sheet says so.

---

## Finishing a run

A run leaves the **Review** list once **every voice has a ruling** (the workbench
cards all show a decision, not "needs ruling"). Checking the words in Step 2 is
recommended, and it is how you get a transcript you can fully trust, but it is
not what removes the run from Review. A run can drop off the list with some lines
still unchecked. To keep checking or to export it afterwards, reopen it from
the **Runs** page (**Media** in the sidebar).

Now read or export it. The **Download transcript** menu on either the workbench
or the transcript page offers plain text, Markdown, subtitles (SubRip / WebVTT),
JSON, and RTTM, with your corrections and speaker names baked in. The same menu
has a **Read on screen** link that opens a clean reading view of the transcript,
no download needed. See [Managing speakers and
exporting](managing-speakers-and-exporting.md) for the formats, the reading view,
and when to use each.

## Related guides

- [Add media & manage runs](add-media-and-manage-runs.md)
- [Manage speakers & export](managing-speakers-and-exporting.md)
- [Settings & troubleshooting](settings-and-troubleshooting.md)
- [Setup](../setup.md): install Voxint on your OS and hardware.
- [First-run walkthrough](../onboarding.md): the setup wizard and guided tutorial.
