# Interpreting diarization output

Two traps routinely make correct diarization output *look* wrong. Both come up
in real first runs; neither is a bug. This page explains what each surface in
the output actually means and which one to trust.

## Segment labels are a projection; the turn ledger is the truth

Voxint records "who spoke when" in two places with different fidelity:

- **The diarization turn ledger** (`diarization_turns`; one row per diarization
  turn, with interval, local label such as `SPEAKER_00`, overlap info, and the
  window's embedding outcome). This is the **source of truth** for speakers and
  timing. The review queue's per-voice evidence and speaker matching are built
  from it.
- **Transcript segment labels** (`diarization_label` on each transcript
  segment, shown in the speaker-labelled transcript). These are a **coarse
  join**: Whisper decides the segment boundaries, and each segment gets the
  label of the *one* diarization turn with maximum temporal overlap
  (ties → earliest turn; no overlap → no label). One label per segment is a
  stated v1 simplification.

The consequence: **segment labels can under-report speakers.** If Whisper emits
a long utterance as a single segment while two people spoke inside it, that
segment carries only the dominant speaker's label — the transcript can read as
one speaker while the turn ledger correctly shows two. When the two surfaces
disagree, believe the turn ledger (and the review workbench built on it), not
the segment labels.

## Short clips can over-split speakers

On short, single-speaker clips, diarization occasionally splits one voice into
two labels (`SPEAKER_00` / `SPEAKER_01`) — a classic short-form false positive:
the model has little audio to establish a stable voice profile, so a shift in
tone or channel character can read as a second speaker. Expected model
behavior, not a Voxint bug. In measurement on bundled short clips, think
"about one in a handful may over-split", not "every split is real".

What to do about it:

- **Adjudicate it away.** The review workbench is exactly the place to rule
  that two proposed voices are the same person — assign both to one speaker.
- **Known-speaker-count constraints**: the pyannote service's `/v1/diarize`
  endpoint accepts `min_speakers` / `max_speakers` bounds (defaults 1–10).
  The pipeline does not currently set them per run — a run always uses the
  defaults — so this knob is available to direct service callers only today;
  wiring a per-run speaker-count hint through the pipeline is future work.

## Quick diagnostic checklist

| Symptom | Where to look | Verdict |
|---|---|---|
| Transcript shows 1 speaker, you know there were 2 | Review workbench / turn ledger for the run | If the ledger shows 2 voices, diarization worked; segment labels collapsed under a long Whisper segment (see above). |
| Short clip shows 2 speakers, you know there was 1 | Length of the clip; the review workbench | Short-form over-split — adjudicate the two voices to one speaker. |
| Every segment shows "(no speaker)" | Run status / stage ledger | Diarization likely never ran (service down or failed) — check the run's stage ledger, not this page. |

Related reading: [quality-gates.md](quality-gates.md) (how voice evidence is
gated before a name is proposed), [architecture.md](architecture.md) (data
model), and the run's stage ledger in the console for anything that looks like
a pipeline failure rather than an interpretation question.
