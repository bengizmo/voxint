# Interpreting diarization output

Two traps make correct diarization output *look* wrong. Both show up in real
first runs, and neither is a bug. This page explains what each surface in the
output means and which one to trust.

## Segment labels are a projection; the turn ledger is the truth

Voxint records "who spoke when" in two places, at different fidelity:

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
segment carries only the dominant speaker's label. The transcript reads as one
speaker while the turn ledger correctly shows two. When the two surfaces
disagree, trust the turn ledger and the review workbench built on it, not the
segment labels.

## Short clips can over-split speakers

On short, single-speaker clips, diarization sometimes splits one voice into two
labels (`SPEAKER_00` / `SPEAKER_01`). This is a short-form false positive: the
model has little audio to build a stable voice profile, so a shift in tone or
channel character can read as a second speaker. Expected model behavior, not a
Voxint bug. On the bundled short clips, think "about one in a handful may
over-split", not "every split is real".

What to do about it:

- **Adjudicate it away.** The review workbench is the place to rule that two
proposed voices are the same person. Assign both to one speaker.
- **Tell the pipeline how many speakers to expect** (see the next section). When
you already know a recording has two people, cap or pin the count and
diarization stops splitting one voice into many.

## Telling the pipeline the speaker count

pyannote estimates the speaker count on its own. On hard audio (distance, wind,
crosstalk) it can split one voice into several clusters, and it stops at a
ceiling of 10 speakers by default. When you know the real answer, supply it and
diarization is constrained to it.

There are two constraints:

- A **bound** (`max_speakers`): the most speakers diarization may return. This is
the safe choice, because the diarizer can still return fewer. Reach for it when
you know a recording has at most a handful of voices.
- An **exact count** (`num_speakers`): pins diarization to that many speakers.
Use it only when you are certain, because a wrong exact count merges two real
people or splits one. The exact count wins when both are supplied.

Both are best-effort at the final output. pyannote can still land on fewer
speakers when the audio is too short or sparse to support the count you asked
for, and Voxint drops sub-second turns before reporting, which can remove a
cluster. The constraint steers clustering; it does not force a minimum.

Supply the count three ways, from most to least specific:

| Where | How | Scope |
|---|---|---|
| Per recording, CLI | `voxint submit clip.wav --max-speakers 3` or `--num-speakers 2` | That one submission. |
| Per recording, sidecar | `max_speakers: 3` or `num_speakers: 2` in the media file's YAML sidecar | Every submission of that file, frozen at submit. |
| Install-wide default | `DIARIZATION_MAX_SPEAKERS=3` in `.env` | Every run with no per-recording override. |

A per-recording value overrides the install-wide default. The sidecar keys are
separate from the sidecar's `speakers:` list, which seeds speaker *names* and
never implies a count. All values are bounded 1 to 20 (the pyannote service
limit). The hint is frozen onto the run at submit, so a requeue or recovery
reuses it.

Under the hood the exact count is sent to the service as equal
`min_speakers`/`max_speakers` bounds, so no separate service field is involved
(`services/pyannote/app/schemas.py`).

## Quick diagnostic checklist

| Symptom | Where to look | Verdict |
|---|---|---|
| Transcript shows 1 speaker, you know there were 2 | Review workbench / turn ledger for the run | If the ledger shows 2 voices, diarization worked; segment labels collapsed under a long Whisper segment (see above). |
| Short clip shows 2 speakers, you know there was 1 | Length of the clip; the review workbench | Short-form over-split; adjudicate the two voices to one speaker. |
| Every segment shows "(no speaker)" | Run status / stage ledger | Diarization likely never ran (service down or failed); check the run's stage ledger, not this page. |

Related reading: [quality-gates.md](quality-gates.md) (how voice evidence is
gated before a name is proposed), [architecture.md](architecture.md) (data
model), and the run's stage ledger in the console for anything that looks like
a pipeline failure rather than an interpretation question.
