# Reviewing and adjudicating a run

*How to turn a finished pipeline run into a transcript you trust: name the
voices, then check and correct the words.*

Voxint listens to your recording and makes its best guesses about who spoke,
when, and what they said. This guide walks you through the editor, where you
have the final say: you confirm the matches Voxint is unsure about, rule on the
voices it could not match, and change any automatic match that is wrong.

Review has two parts, and both happen on one page, the **editor**:

- **Step 1, [Identify the voices](#workflow-a-identify-the-voices):** decide who
  each detected voice really is (or that they should be left out). This work
  lives in the speaker rail beside the transcript.
- **Step 2, [Verify and correct the transcript](#workflow-b-verify-and-correct-the-transcript):**
  read through the words, confirm the ones that are right, and fix the ones that
  are wrong.

You can switch between the two at any time. Settling the voices first means
every line shows the right name while you check the words. Checking the words is
recommended but optional: voice matches and your speaker rulings decide when a
recording stops showing as needing review.

New to Voxint? The bundled [guided tutorial](../onboarding.md#3-guided-tutorial)
walks this whole loop on a sample recording before you use your own audio, so it
is a good place to start. This guide is the fuller reference for the same work.

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

### 1. Open a recording from the Media page

Click **Media** in the sidebar. The Media page lists every recording you have
added, and its **REVIEW** column tells you where each one stands:

- A chip such as **2 voices need review** means processing finished and some
  voices are waiting for your ruling.
- **Reviewed** means every voice has a ruling or a strong voice match.
- Anything else shows the run's current state, for example a recording that is
  still being transcribed or one whose run failed.

To see only the recordings that are waiting for you, choose **Needs review** in
the status menu next to the search box. The list updates as soon as you pick it.

![The Media page filtered to Needs review: a table of recordings with name,
length, and a REVIEW column showing a "voices need review" chip on each
row.](../images/media-needs-review.png)

Each row ends with a link that opens the recording in the editor. It reads
**Review →** while the run is waiting on your speaker decisions, **Retry →**
when the last run failed, and **Open →** otherwise. The **Home** page also
counts your **runs to review**, and its arrow brings you to the Media page with
**Needs review** already chosen. On the **Runs** page, a finished run that still
needs you has a **Review →** link that opens it in the editor.

Opening the editor claims the run for this browser tab, so there is no separate
claiming step. A few related messages you may see:

- **Read-only view** with a **Claim for editing** button: the editor is showing
  the run without a claim. Press the button to start making changes.
- **Your claim expired or was taken over.** Everything you already saved is
  safe. If the edit box held text you had not saved, the message shows it in a
  box you can copy from, and the editor stays on that line until you press
  **re-claim to keep editing it**, which puts your text back in the edit box.
  With nothing unsaved, press **Re-claim to continue editing**.

If a recording has been processed more than once, the **Run** card at the top
of the editor lists the others under **Other runs**. The **Run details** link on
that card opens the technical page where you can restart or archive a run.

---

## Workflow A: Identify the voices

The editor shows the speaker rail beside the transcript. Voxint separated the
recording into voices and gave each one a label such as `SPEAKER_00`. The
sentence at the top of the rail tells you how much work remains, for example "2
voices need you, 1 with very little speech. 3 matched automatically." When you
are done it reads "Every voice has a ruling." If matching never ran on the
recording, or you had not added any people yet when it ran, the sentence says
that instead, so an empty result is never mistaken for a clean one.

Above the transcript, a line such as "2 unidentified voices, 5 segments
affected" offers a **Start reviewing** link. It jumps to the first line spoken
by a voice that needs you and opens the speaker menu for that line.

The rail puts voices into these groups:

- **Needs you** is open. These voices need your decision.
- **Too little speech to identify** is closed at first. These voices still need
  a ruling, so open the group and review them.
- **Matched automatically** is closed while work remains. These voices are
  settled by a strong voice match or an automatic save.
- **Your rulings** is also closed while work remains. It holds decisions you
  made during review.

When no voice still needs a ruling, **Matched automatically** and **Your
rulings** open. You can open or close any group shown with a disclosure arrow.

![The speaker rail beside the transcript, with its summary sentence, an open
Needs you group, and disclosures for voices with too little speech, automatic
matches, and your rulings.](../images/review-workbench.png)

### Read a voice card

A card starts with the voice label and a pill such as **needs you** or **saved
automatically**. Its headline and reason explain what Voxint found:

- **Possibly Jordan** means the voice resembles a known person but needs your
  check. The reason reads **Not strong enough to confirm without your check.**
  Press **Confirm Jordan** if the match is right. Confirming assigns the existing
  speaker to this recording. It does not add another voice sample.
- A possible match can also say **Saved as Voice 4 for now. Confirm if this is
  Jordan.** Voxint already saved the voice under a temporary name, but the
  possible match still needs your ruling.
- **Similar voices found** means two known speakers sound close. No candidate
  name or **Confirm** button is shown. Open **Why no name?** for the explanation,
  then listen and choose.
- **Who is this?** appears when Voxint has no useful candidate. The reason says
  why in plain words, for example **Not enough clear speech to match.** or
  **No known speakers to compare against.**

Below the reason, the card shows the number of turns and seconds of speech. If
the audio can be played from the timeline, **Hear this voice** jumps to the
first transcript line for that voice and plays it. Listen before you rule.

Open **Why this match?** (or **Why no match?** on a card with no candidate) to
see the reason followed by the raw scores, such as "Voice similarity 0.72. Lead
over the next closest voice 0.05. Agreement across this voice's speech 0.75."
With one known person on your roster the lead reads "none (only one known
speaker)." These scores are never shown as percentages.

> **A voice match and a heard name are different evidence.** A voice match
> compares the sound with saved voice samples. The line **Heard name
> (unverified): "Alex".** reports a name spoken in the recording. Treat it as a
> lead and listen before assigning the voice.

### Rule on a voice

Use the actions on the card:

| Action | What it does |
|---|---|
| **Confirm Jordan** | Records your ruling that this voice is the suggested known person. It does not add a voice sample. |
| **Someone else…** or **Known person…** | This voice is a different person from your roster: a searchable speaker picker opens where you can type to filter by name, use arrow keys to navigate, and press Enter to select. Type a name that does not exist yet and choose **Create "[name]"** to add them to your roster on the fly; that keeps this voice as their sample. Press **Escape** to close the picker. |
| **Not a person** | This "voice" is background noise, music, a TV, or someone you do not want in the results. Leaves it out. |
| **Can't tell** | You genuinely cannot tell who this is. An honest ruling that settles the voice; you can change it later. |

### Check automatic matches and earlier rulings

Open **Matched automatically** to see voices already shown as a known person.
A voice match has a **voice match** pill and reads **Shown as Jordan**, with the
reason **Matched by voice.** An automatically saved voice has a **saved
automatically** pill and reads **Saved as Voice 4**. Its reason is **Saved
automatically so Voxint can recognise this voice later. Name them on the
Speakers page.** The **Speakers page** link takes you there.

Open **Your rulings** to see rows with a **your ruling** pill. They read the
person's name with **Your ruling.**, **Left out** with **Your ruling: not a
person.**, or **Could not tell** with **Your ruling.**

Each row has **Change**. Press it to reveal the same actions with a **Reassign
to…** picker. Press **Hide** to close the actions again.

### Hear a voice before assigning (popover preview)

When the speaker menu is open (click a speaker name in the transcript, or press
`s`), **Hear this voice** plays the line you opened the menu on.

Below it, **Compare with a voice named in this recording** lists the people
from your speaker list who speak elsewhere in this recording, under a different
detected voice. Click a name to hear one of their lines, then compare it with
the voice you are assigning. Voxint picks a line you assigned yourself when
there is one, and prefers a line of a couple of seconds or more. The person the
line is currently assigned to is listed too when they speak elsewhere, so you
can check a doubtful match. People with no other line in this recording are not
listed.

Neither button moves your place in the transcript or closes the menu, and each
plays one line and then stops. Both appear only when audio playback is
available.

Someone you know from an earlier recording may have no line here yet. Open
**Compare with a voice from another recording** in the same menu and click
their name. Voxint plays a short clip, ten seconds at most, of a line you
assigned to that person yourself in another recording. A name appears there
only when such a line exists, so people Voxint matched on its own are not
listed, and neither are people whose only lines are in recordings you moved to
the trash. The list opens by itself when it is the only comparison on offer.

The clip pauses the recording you are reviewing, and pressing play on the
recording stops the clip. Your place in the transcript and any text you are
editing stay as they are. If the clip cannot be played, a short message under
the edit box says why, for example because the other recording's audio was
removed to free up space.

### Undo a ruling

After assigning a speaker, excluding a voice, or ruling "can't tell," an **undo
toast** appears at the bottom of the screen. Click **Undo** within five minutes
to reverse the ruling and restore the label to its previous state.

Changing the speaker of a single line (or of one part of a split line) shows the
same toast. **Undo** takes that change back, so the line shows whatever it
showed before you changed it.

The undo window is enforced on the server: if the label or line was changed
again before you click Undo, the toast tells you so. Closing the toast, waiting
past the deadline, or making another ruling dismisses it.

### Merge suggestions

After you assign a speaker to a label, the editor checks whether other
unresolved labels have voices that match the same person. If it finds any, a
**merge suggestion toast** appears (stacked above the undo toast if both are
showing).

The toast names the suggested speaker and offers two choices:

- **Merge** opens a preview of the exact change (how many labels move), then
  confirms.
- **Dismiss** (the x button, or wait for it to expire) skips the suggestion.

The suggestion only appears for labels that have no human ruling and no grounded
match, so it will not override your earlier decisions.

### One person split across two labels ("same speaker")

Diarization sometimes splits **one** person into two labels: you'll see
`SPEAKER_00` and `SPEAKER_03` that are clearly the same voice. Fix it right here
with the **Same speaker across labels?** panel in the rail:

1. **Tick** the labels that are the same person in this recording.
2. Under **Who are they?**, pick an existing speaker, or type a new name and
   choose **Create "[name]"**.
3. Press **Preview merge…** to see the **exact change** Voxint will make (how
   many turns and transcript segments move) before anything happens.
4. Press **Confirm merge**.

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

The transcript sits beside the speaker rail on the same page. This is where you
read the words, mark the right ones as checked, and fix the wrong ones. It works
as a steady loop: **read a line, confirm it, move to the next.**

A counter at the top keeps score, for example **"7 of 32 segments verified · 25
left"**. When every line is checked it reads "You have checked every line.", and
two buttons appear: **Download transcript** and **Back to the library**.

![The editor in walk mode: a counter reading "0 of 10 segments verified", the
Exit walk mode, Split and Shortcuts buttons, a line counting the unidentified
voices with a Start reviewing link, an edit box for the current line with
Verify & next, Save edit, Skip and Replay, a colored per-speaker waveform strip
under the audio player, and the transcript lines
below.](../images/transcript-review.png)

### Walk mode

While any line is still unchecked, the editor opens in **walk mode**, which runs
the loop for you: verifying a line takes you to the next unchecked line and
plays it. Press **Exit walk mode** (or `w`) when you would rather move around
freely. **Verify** then marks the current line and stays on it. Press **Walk
mode** to turn the loop back on.

### The verify-and-advance loop

The editor starts on the first line that hasn't been checked yet. Its words sit
in the edit box above the transcript. For each line:

- Listen to it. A line plays when you move to it, and **Replay** plays it again.
- If the words are right, press **Verify & next**. Voxint marks the line checked
  and takes you to the next unchecked line.
- If the words are wrong, fix them first (below), then verify.
- **Skip** leaves a line for later and moves to the next unchecked one.

You can drive this entirely with the keyboard (see [Keyboard
shortcuts](#keyboard-shortcuts)) or entirely with the on-screen buttons,
whichever you prefer.

### Lines the model was unsure about

Some lines carry a small dashed **"uncertain"** chip. The transcriber reported
**low confidence** on that line, so the chip points you at the parts most worth
a listen and you don't have to re-read everything. Uncertain lines are often
correct: the chip is a nudge to check, and Voxint never puts a percentage on it.
(Older runs made before this feature simply won't show the chip.)

### Fix a line's words

Click any line in the transcript to make it the current line. Its words appear
in the **edit box**. Correct the text and press **Save edit** (or **Ctrl+Enter**,
**⌘+Enter** on a Mac). A few things to know:

- Voxint keeps your correction **beside** the original; it never overwrites what
  the model actually heard. Exports show your corrected wording by default, and
  the model's exact words stay available under **Other text variants** in the
  **Download transcript** menu.
- **Editing a line clears its "verified" mark**: corrected words should be
  re-checked, so the line rejoins the loop for a fresh confirm.
- Saving an empty box, or the model's original wording, removes your
  correction.
- **Unsaved-edit warning:** if you have unsaved text in the box and try to
  verify, split, or move to another line (clicking a line or the waveform, a
  navigation key, or a jump from the outline or a note), Voxint warns you once
  rather than silently throwing the edit away. Save it (Ctrl/⌘+Enter), or
  repeat the action to discard and continue. A click on another line or the
  waveform still plays it, so you can listen around the line you are editing.

### Corrections your domain pack made

If you run with a [domain pack](../domain-packs.md) that declares corrections, some
lines are fixed **automatically** before you ever see them: a recurring
mishearing turned into the right spelling every time. When that happened on the
current line, a **corrected by domain pack** marker appears next to it, kept
separate from the **edited** badge, which means a change *you* made. Press the
marker to see exactly which rule fired: the phrase it matched, what it became,
and the pack and rule it came from.

![The current line carrying a "corrected by domain pack" marker, expanded to
show the rule that fired (match → replace) with its pack and rule
name.](../images/correction-provenance.png)

- **Compare against the original.** Open the **Download transcript** menu, then
  **Read on screen**. The reading view has a tab for each version of the text:
  **raw** is the exact words the model first heard, and **corrected** is the
  reviewed version.
- **Your edit wins.** The moment you save your own wording for a line, the
  "corrected by domain pack" marker goes away; from then on the line shows *your*
  text, not the pack's automatic edit.
- **A corrected line can't be split.** Splitting a line at a word (below) is turned
  off once a correction has fired on it; Voxint tells you why rather than offering a
  cut that wouldn't work.

The editor only lists rules that fired. A rule that never matched anything in
the recording leaves no trace in the console. If a rule you expected never shows
up, the recording may not contain the term, or the term was broken across a
pause. For terms that get broken across pauses, add them to the pack's
**vocabulary** (which nudges the transcriber up front) instead of relying on a
correction after the fact.

### The waveform strip

Under the audio player sits a compact **waveform**, a colored strip where each
band is tinted for the speaker who was talking, using the same colors as the
transcript. It's a map of who spoke when. **Click anywhere on the strip to jump**
to that moment and select the matching line (overlapping speech is marked, and
stretches that were spoken but not transcribed still show up, so the picture
stays honest). If you click a spot with no transcript there, whether a silent
gap or speech that was never transcribed, the strip says so instead of doing
nothing. A marker tracks playback and shows which line is current.

You can also **click and drag** across the strip to select a time range. The
selected region gets an accent-colored overlay, and a **Play selection** button
appears below the strip for bounded playback (audio plays the selected range and
stops). A time display shows the start, end, and duration. Click **Clear** or
press **Escape** to dismiss the selection. A plain click (no drag) still seeks
as before and clears any prior selection.

### Split a segment at a word

Sometimes one transcript segment actually contains **two speakers**: the
diarizer drew the boundary in the wrong place. You can cut it at the right word:

1. Press **Split** to turn on split mode. The editor confirms with "Split mode
   on", and the button changes to **Exit split mode**.
2. The current line's words become clickable. **Click the word where the new
   speaker starts**, and Voxint cuts the segment just before it.

The two halves then stand on their own, and you can give each one the correct
speaker (below). A few honest limits: a segment can only be split when its words
line up cleanly with what was transcribed (Voxint tells you plainly when one
can't be split), a split segment can't also be free-text edited (splitting and
editing are mutually exclusive), and a segment can be split into two parts, not
more, in this release.

### Reassign a segment (or half of one) to another speaker

You can hand any line to the right person without leaving the transcript. The
pickers are searchable: type to filter by name, use arrow keys to navigate, and
press Enter to select. If you need a new speaker, type their name and choose
**Create "[name]"** to add them to the roster on the fly.

- **A whole segment:** use the **Assign speaker** picker under the edit box, or
  press a number key **1–9** to assign the current line to the 1st–9th speaker
  on your roster. Press **0** to reset the line to its **detected** speaker
  (undo your override). Clicking the speaker name on a line (or pressing `s`)
  opens the same choice as a menu next to the line.
- **Half of a split segment:** after you split a segment, **each part gets its
  own speaker picker** in the transcript. Choose the speaker for each half
  independently, or pick **inherit** to send it back to following its label.

Reassigning changes *attribution only* (it never rewrites the words), and it
flows through to your exports.

---

## Keyboard shortcuts

The editor is built to run from the keyboard. Press **?** at any time, or click
the **Shortcuts** button (it shows the `?` accelerator), for the same
cheat-sheet built into the console.

![The keyboard-shortcuts cheat-sheet: a dialog titled "Keyboard shortcuts"
listing each key with a plain-language description, and a note that Space and
the arrow keys stay with the audio player.](../images/keyboard-shortcuts.png)

| Key | Action |
|---|---|
| **v** | Verify this line (and, in walk mode, go to the next unchecked one) |
| **n** | Skip to the next unchecked line |
| **p** | Replay the current line |
| **e** | Edit the current line's text |
| **j** / **k** | Go to and play the next / previous line |
| **1**–**9** | Assign this line to the 1st–9th speaker on your roster |
| **s** / **@** | Open the speaker menu for this line |
| **0** | Reset this line to its detected speaker |
| **=** | Assign this line to the previous line's speaker |
| **h** | Highlight the transcript text you have selected |
| **d** | Open the **Download transcript** menu |
| **w** | Turn walk mode on or off |
| **?** | Show the cheat-sheet |
| **Ctrl/⌘+Enter** | Save an edit (while typing in the edit box) |

A few deliberate rules:

- **Shortcuts never fire while you're typing** in a text box or menu, so `v`
  types a "v" in the edit box instead of verifying.
- **Space** (play/pause) and the **arrow keys** (scroll) stay with the audio
  player, as you'd expect.
- **Every shortcut also has a visible, clickable control** on the page; the
  keyboard is a shortcut, never the only way in. The digit keys mirror the
  on-screen **Assign speaker** picker; `1`–`9` do nothing on a run that has no
  speakers yet, and the cheat-sheet says so.

---

## Finishing a run

A recording drops out of the **Needs review** filter on the Media page once
every voice has your ruling or a strong voice match. A voice with very little
speech still needs a ruling; **Not a person** and **Can't tell** both settle it.
Its **REVIEW** chip then reads **Reviewed**. Checking the words in Step 2 is how
you get a transcript you can fully trust, but it does not affect this chip: a
recording can show as reviewed with some lines still unchecked. To keep checking
or to export it later, open it again from the Media page.

Now read or export it. The **Download transcript** menu in the editor offers
plain text, Markdown, subtitles (SubRip / WebVTT), JSON, and RTTM, with your
corrections and speaker names baked in. The same menu has a **Read on screen**
link that opens a clean reading view of the transcript, no download needed. See
[Managing speakers and exporting](managing-speakers-and-exporting.md) for the
formats, the reading view, and when to use each.

## Related guides

- [Add media & manage runs](add-media-and-manage-runs.md)
- [Manage speakers & export](managing-speakers-and-exporting.md)
- [Settings & troubleshooting](settings-and-troubleshooting.md)
- [Setup](../setup.md): install Voxint on your OS and hardware.
- [First-run walkthrough](../onboarding.md): the setup wizard and guided tutorial.
