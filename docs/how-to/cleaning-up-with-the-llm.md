# Cleaning up a transcript with the LLM

*How to ask the language model for a tidier copy of a transcript, check every
word it took out, and download the result.*

People rarely speak in clean sentences. A transcript that is faithful to the
recording is full of "I mean", "you know" and false starts, which makes it hard
to quote or share. Voxint's filler removal already drops fixed words such as
`um` and `uh`. The **LLM clean-up** goes a step further: the language model
(LLM) you set up reads each line and suggests which words are filler in that
context, so it can catch phrases a fixed list cannot.

The clean-up is a separate copy. Your reviewed transcript, its normal
downloads and its subtitles stay exactly as they are, and the clean-up appears
only where you ask for it.

## What the model is allowed to do

The model can only suggest words to remove. Voxint checks every suggestion
against the original line before using it:

- **Removing words is the only change allowed.** If the model rewords a line,
  adds a word or swaps one word for another, Voxint ignores its suggestion for
  that line and keeps the line as it was.
- **Some words are never removed:** numbers (written as digits or as words),
  negations such as "not", "never" and "don't", and words that look like names (a
  capital letter in the middle of a sentence). A suggestion that removes one of
  these is ignored for the whole line.
- **A whole line is never removed.** A suggestion that would empty a line is
  ignored.
- **The cleaned text is built from your transcript's own words.** The model
  never contributes a character, so the copy cannot contain a word that was not
  in the transcript.

> The clean-up is the model's reading of what counts as filler. Check the
> struck-through words before you use the copy. It does not cut the audio, and
> leaving a word out is not redaction: the full transcript is still stored.

## Before you start

- **Turn on the LLM.** The clean-up uses the same LLM as transcript
  enhancement. Open **Settings**, find the **LLM** section, and make sure
  **LLM transcript enhancement** is on. See
  [Settings & troubleshooting](settings-and-troubleshooting.md) if you have not
  set up the LLM yet.
- **English recordings only.** If Voxint detected another language in the
  recording, the clean-up is not offered. If it could not detect the language,
  you can still run it, and the page notes that the language was not verified
  as English.
- **Optional: your filler words.** The words in **Settings → Filler words** are
  passed to the model as examples of what to look for. See
  [Choose your filler words](managing-speakers-and-exporting.md#choose-your-filler-words).

## Make a clean-up

1. Open the recording in the editor (from **Media**, click the recording).
2. Click **Download transcript**, expand **Other text variants**, and click
   **LLM clean-up**. The reading view has the same **LLM clean-up** link at the
   top.
3. On the clean-up page, click **Generate clean-up**. The page shows
   **Cleaning up** with the job's progress and a **Cancel** button. A long
   recording can take several minutes, depending on your LLM.

When it finishes, the page reloads with the result. (With JavaScript turned
off, reload the page yourself to see when it is done.)

## Check what was removed

The **Changes** section shows every line of the transcript with the removed
words struck through, so you can read the original and the cleaned version in
one pass. A line where Voxint ignored the model's suggestion is shown whole,
with a short note saying why, for example "(kept: reworded, not only
shortened)".

The **Summary** above it gives the totals: how many lines changed, how many
words were removed, and how many suggestions were not used, by reason. It also
records which model made the clean-up and when.

## Download the cleaned copy

The **Download** section on the clean-up page lists the cleaned copy in every
transcript format: plain text and Markdown (each as a reading copy or with
times), `.srt` and `.vtt` subtitles, and `.json`. Subtitles keep the original
timing: each caption shows the cleaned words for the same stretch of audio.

The cleaned copy is not combined with the reading view's **Remove filler
words** or **Remove repeated words** options, or with a translation. It is its
own version of the text, and the words you marked to keep or leave out in the
review editor do not apply to it.

## If you edit the transcript afterwards

A clean-up is a snapshot of the transcript at the moment it was made. If you
correct a line or split a segment afterwards, the clean-up no longer matches,
and the page says so:

- An **Out of date** notice appears at the top.
- The download links are removed, and a direct download is refused rather than
  served stale.
- The **Changes** section still shows the old clean-up against the transcript
  as it was then, so you can see what it did.

Click **Generate again** to clean up the current transcript. The new clean-up
replaces the old one. Renaming a speaker, changing your filler words, or
marking words in the review editor does not make a clean-up out of date; only
changes to the transcript text do.

## Next steps

- [Manage speakers & export](managing-speakers-and-exporting.md) for the
  regular downloads, the reading view and filler removal.
- [Translate transcripts](translating-transcripts.md), the other LLM feature
  that works on a finished transcript.
- [Settings & troubleshooting](settings-and-troubleshooting.md) for LLM setup
  and general fixes.
