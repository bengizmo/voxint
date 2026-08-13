# Guided-tutorial sample assets

These files back Voxint's first-run guided tutorial (`voxint tutorial seed`). They
are **package data** (shipped in the wheel and the Docker image), loaded at runtime
via `importlib.resources`; the seed and the test suite consume only these committed
files and never invoke a speech synthesizer.

| File | Role |
|------|------|
| `sample-3speaker.wav` | The bundled clip the seed copies into `media_root` and `/media/{run}` streams. 16 kHz mono PCM (`pcm_s16le`). |
| `utterance.json` | Hand-authored speaker layout (labels, voices, text) with per-utterance start/end times measured from the synthesized timeline. |
| `expected-transcript.json` | The attributed transcript the seeded run reproduces — used to verify export/attribution in tests. |
| `provenance.json` | Tool versions, per-voice synthesis args, and the WAV SHA-256. |

## How the WAV was generated

Regenerated only by a human running, from the repo root:

```
python tools/generate_tutorial_audio.py
```

The script synthesizes each utterance with the **espeak-ng** binary (three distinct
voices/rates/pitches), resamples each to 16 kHz mono PCM with **ffmpeg**, and
concatenates them sample-accurately (a fixed inter-utterance silence gap) so the
recorded timings match the audio exactly. Exact tool versions, the per-speaker voice
arguments, and the current `wav_sha256` are recorded in `provenance.json`.

Three speakers, chosen to teach the three adjudication states:

- `SPEAKER_00` → a **grounded cosine** match to a roster speaker (`Jordan Rivera`).
- `SPEAKER_01` → a **heard name** (`Priya`, self-introduced) that is *unverified* —
  a suggestion, never an attribution.
- `SPEAKER_02` → **unresolved**, no name and no acoustic match.

## License and provenance

The utterance text, `tools/generate_tutorial_audio.py`, and the resulting
`sample-3speaker.wav` are original works of the Voxint project and are dedicated to
the public domain under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).
There is no copyrighted source audio and no real-person voice or personal data; the
names in the script are fictional.

The audio was **synthesized with espeak-ng**, which is licensed GPLv3. That is a
factual statement of provenance. Voxint invokes the system `espeak-ng` binary at
generation time — it does **not** vendor, modify, or redistribute espeak-ng or its
data, and none of espeak-ng's source is included here. espeak-ng's license governs
espeak-ng; it does not attach to this text or WAV, and the CC0 dedication above is a
deliberate release of the project's own rights in these authored works — **not** a
claim that "the output is CC0 because the synthesizer is GPL."
