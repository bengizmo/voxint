# Parity golden corpus

Deterministic, redistributable audio inputs for the backend-parity gates
(docs/gpu-contracts.md, "Equivalence policy") and the standing NVIDIA
regression gate. Regenerated only by a human running, from the repo root:

```
python tools/generate_parity_corpus.py
```

| File | Role |
|------|------|
| `embed-corpus.wav` | One 16 kHz mono timeline: 24 utterances across 6 synthetic espeak-ng voices, noise-mixed copies calibrated against the titanet service's own SNR estimator to straddle the 5 dB `low_snr` gate, and a dedicated digital-silence region. |
| `embed-windows.json` | ~107 windows over the timeline (clean, sub-windows, 1.0 s `too_short` straddles, SNR straddles, pure silence, gap straddles, past-end, determinism duplicates), each with its expected `skip_reason`/`snr_db` computed with the service's exact slice + SNR math. |
| `embed-pairs.json` | Labeled same/different-speaker window pairs for the decision-level parity gate. |
| `transcribe-short.wav` | Single-voice scripted passage for ASR tolerance checks (script text in `manifest.json`). |
| `diarize-3speaker.wav` | Three voices alternating with a sample-accurate turn layout for diarization tolerance checks. |
| `manifest.json` | Clip metadata, scripts/layouts, utterance placements, checksums. |
| `provenance.json` | Tool versions, noise seed, per-voice synthesis args, WAV SHA-256s. |

Reference *outputs* live in `../references/` and are produced by
`tools/generate_parity_references.py` against running CUDA services on
maintainer NVIDIA hardware.

## License and provenance

The utterance texts, the generator script, and the resulting WAVs are original
works of the Voxint project and are dedicated to the public domain under
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). There is no
copyrighted source audio and no real-person voice or personal data; all voices
are synthetic and all content is fictional.

The audio was **synthesized with espeak-ng** (GPLv3). That is a factual
statement of provenance: Voxint invokes the system `espeak-ng` binary at
generation time and does not vendor, modify, or redistribute espeak-ng or its
data. espeak-ng's license governs espeak-ng; it does not attach to these
authored texts or WAVs.
