# Checking for AI-generated speech

*How to tell whether a recording contains speech made by an AI voice tool, and
what the scores mean.*

Some recordings may contain speech that was not produced by a real person. Voxint
can score each speaker turn for how likely it is to be AI-generated (a
"deepfake"), and show you the result as a simple risk level: **low**, **medium**,
or **high**. You decide what to do with that information.

This feature is **opt-in**. It requires a separate GPU service and must be
enabled in Settings before it does anything. If you did not install it, nothing
in this guide applies and nothing changes about how your recordings are
processed.

**Related how-to guides:** [Settings and
troubleshooting](settings-and-troubleshooting.md) · [Review and
adjudicate](reviewing-and-adjudicating.md). Technical details:
[gpu-contracts.md](../gpu-contracts.md#synthetic-speech-detection-synthdetect).

---

## Turning it on

1. Make sure the synthdetect service is running. If you followed the
   [setup guide](../setup.md#optional-synthetic-speech-detection-deepfake-scoring),
   it is already part of your stack.

2. Go to **Settings → Synthetic-speech detection**.

3. Turn on **Enabled**. This is the master switch. With it off, the feature is
   completely dormant.

4. Optionally, turn on **Autogenerate**. With this on, every recording that
   finishes processing is scored automatically. With it off, you score recordings
   one at a time (see below).

## Scoring a recording

### Automatic scoring (autogenerate on)

When autogenerate is on, there is nothing to do. Each recording is scored as
soon as it finishes the pipeline. The risk chip appears on the recording's detail
page when scoring is complete (typically a few seconds to a minute, depending on
length).

### Manual scoring

When autogenerate is off, or for recordings that were processed before you
enabled the feature:

1. Open the recording's detail page.
2. Find the **Synthetic speech detection** panel.
3. Click **Score for synthetic speech**.

Scoring runs in the background. Reload the page after a moment to see the
result.

## Reading the scores

Each recording's detail page shows a **Synthetic speech detection** panel once
scoring is complete. The panel shows:

- **Mean risk**: the average risk across all speaker turns. Shown as a colored
  chip (low, medium, or high).
- **Max risk**: the single highest-risk turn. Also shown as a chip.
- A link to the **full report**.

The **full report** page breaks the score down by individual speaker turn. Each
turn shows its raw logit (a number from the model) and its calibrated risk
percentage.

### What the risk levels mean

| Chip | What it suggests |
|---|---|
| **Low** (green) | The speech is consistent with a real human voice. |
| **Medium** (amber) | The model is uncertain. Could be real or synthetic. |
| **High** (red) | The speech is consistent with known AI voice generators. |

These are the model's estimates, not certainties. A high-risk score does not
prove the speech is fake, and a low-risk score does not guarantee it is real.

## Known limitations

The report page includes a limitations section. The two most important ones:

- **Some AI voice generators partially evade the detector.** Chatterbox is the
  most notable: the model catches it less reliably than other generators.
  (#252)
- **Recording conditions can shift scores.** Certain real-world corpora produce
  different score distributions than others, which can inflate false positives in
  some cases. (#253)

These are documented findings, not bugs. They are why the scores are presented
as risk levels for your judgment, not as verdicts.

## Turning it off

Go to **Settings → Synthetic-speech detection** and turn off **Enabled**.
No new recordings will be scored. Scores that were already produced remain
visible on the recordings that have them.

> Jobs that were already queued when you turned it off may still complete. This
> is normal and not a sign that the setting did not take effect.
