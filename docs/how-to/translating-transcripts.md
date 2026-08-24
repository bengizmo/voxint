# Translating transcripts

*How to turn a finished transcript into another language, read the two side by
side, and download the translated version.*

Suppose you have interviews recorded in Spanish and colleagues who read only
English. Voxint can translate a finished transcript into the language you
choose, show the translation beneath each original line, and let you download
it in every transcript format. The translation is made by the language model
(LLM) you configured during setup, on your own hardware or through your own
LLM endpoint, so the text never goes anywhere you did not choose.

> Machine translation is a working rendition, not a certified translation.
> Voxint always keeps the original wording and shows you both, so you can
> check any line that matters.

## Before you start

Translation uses the same LLM as transcript enhancement. Open **Settings**,
find the **LLM** section, and make sure **LLM transcript enhancement** is
turned on. If it is off, the translation controls tell you so and point you
there. See [Settings & troubleshooting](settings-and-troubleshooting.md) if
you have not set up the LLM yet.

While you are in Settings, look at the **Translation** section:

- **Preferred language** is the language transcripts get translated into. It
  becomes the default for every Translate button, so you pick it once instead
  of on every recording.
- **Translate new runs automatically** translates each recording as soon as
  the pipeline finishes. It skips recordings that are already in your
  preferred language.

Both are optional. You can leave them unset and choose a language each time
instead.

## Translate a recording

1. Open the recording's run page (from **Runs**, click the recording).
2. Find the **Translation** card. Pick a language from the **Translate to**
   list. If you set a preferred language, it is already selected.
3. Click **Translate**. The card shows the job's progress and a **Cancel**
   button while it works. Translation of a long recording can take several
   minutes, depending on your LLM.

When it finishes, the card links to the transcript page, where the
translation appears.

You can also start a translation right after reviewing: once you have checked
every line on the review screen, a **Translate** action appears next to
"Open the transcript to export."

## Read the translation

Open the transcript page (from the run page, or the **Translation** card's
link). A **Translation** switcher appears above the transcript with one entry
per translated language:

- **Original only** shows the transcript as reviewed.
- Picking a language shows the translated line beneath each original line, so
  you can read them together and spot-check any line against the audio.

A note above the transcript records which language it was translated from,
which model did the work, and when.

## Download the translation

On the transcript page, open the **Download transcript** menu. Each
translated language has its own row of download links: `.txt`, `.md`, `.srt`,
`.vtt`, and `.json`, all in the translated language. The regular downloads of
the original transcript are unchanged.

> Translated subtitles (`.srt` / `.vtt`) keep the original timing. Some
> languages take more words to say the same thing, so translated captions can
> read fast. Check them in your video player before publishing.

## If you edit the transcript afterwards

A translation is a snapshot of the transcript at the moment it was made. If
you correct a line, split a segment, or edit the text afterwards, the
translation no longer matches, and Voxint says so honestly:

- The run page's **Translation** card marks it **out of date**.
- The transcript page stops showing the translated lines rather than showing
  a translation of text that no longer exists.
- The translated download links disappear, and a direct download request is
  refused rather than served stale.

To fix it, click **Re-translate** on the run page's Translation card. The new
translation replaces the old one. Renaming a speaker does not make a
translation out of date; only changes to the spoken text do.

## Next steps

- [Manage speakers & export](managing-speakers-and-exporting.md) for the rest
  of the export formats and the on-screen reading view.
- [Settings & troubleshooting](settings-and-troubleshooting.md) for LLM setup
  and general fixes.
