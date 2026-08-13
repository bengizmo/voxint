#!/usr/bin/env python3
"""Regenerate the bundled guided-tutorial sample audio (run ONCE by a human).

Voxint's first-run tutorial (`voxint tutorial seed`) plays a short, controlled
three-speaker clip. This script synthesizes that clip from the hand-authored
utterance layout in ``src/voxint/tutorial/assets/utterance.json`` using the
``espeak-ng`` binary (three distinct voices), resamples each utterance to the
project's 16 kHz mono PCM invariant with ffmpeg, and concatenates them
sample-accurately so the recorded per-utterance timings match the audio exactly.

It writes four committed artifacts under ``src/voxint/tutorial/assets/``:

* ``sample-3speaker.wav``      — the clip the seed copies into ``media_root``
* ``utterance.json``           — the same layout, with measured start/end times
* ``expected-transcript.json`` — the attributed transcript the seed reproduces
* ``provenance.json``          — tool versions, per-voice args, and the WAV SHA-256

The runtime seed and the test suite consume ONLY the committed artifacts; neither
requires ``espeak-ng`` to be installed. Re-run this only to change the sample:

    python tools/generate_tutorial_audio.py

espeak-ng is GPLv3; we do not vendor or modify it — we invoke the system binary.
The utterance text, this script, and the resulting WAV are authored by the
project and dedicated to CC0 (see the assets README). "Synthesized with
espeak-ng" is a factual provenance statement, not a license claim on espeak.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import wave
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / "src" / "voxint" / "tutorial" / "assets"
UTTERANCE_JSON = ASSETS / "utterance.json"
WAV_OUT = ASSETS / "sample-3speaker.wav"
TRANSCRIPT_OUT = ASSETS / "expected-transcript.json"
PROVENANCE_OUT = ASSETS / "provenance.json"

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # pcm_s16le


def _tool_version(binary: str, *args: str) -> str:
    """First non-empty line of ``binary --version`` (best-effort provenance)."""
    # Fixed binary name, list args, no shell — safe by construction.
    out = subprocess.run(
        [binary, *args], capture_output=True, text=True, check=True
    )
    for line in (out.stdout + out.stderr).splitlines():
        if line.strip():
            return line.strip()
    return "unknown"


def _synthesize(text: str, voice: str, speed: int, pitch: int, dest: Path) -> None:
    """espeak-ng -> raw WAV, then ffmpeg -> 16 kHz mono PCM at ``dest``."""
    with tempfile.NamedTemporaryFile(suffix=".wav") as raw:
        subprocess.run(
            [
                "espeak-ng",
                "-v", voice,
                "-s", str(speed),
                "-p", str(pitch),
                "-w", raw.name,
                text,
            ],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", raw.name,
                "-ar", str(SAMPLE_RATE),
                "-ac", "1",
                "-c:a", "pcm_s16le",
                str(dest),
            ],
            check=True,
            capture_output=True,
        )


def _read_frames(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav:
        if wav.getframerate() != SAMPLE_RATE or wav.getnchannels() != 1:
            raise RuntimeError(f"{path} is not 16 kHz mono after resample")
        return wav.readframes(wav.getnframes())


def main() -> int:
    layout = json.loads(UTTERANCE_JSON.read_text())
    gap_frames = round(float(layout["gap_seconds"]) * SAMPLE_RATE)
    silence = b"\x00\x00" * gap_frames

    timeline = bytearray()
    cursor = 0  # frame position in the final 16 kHz timeline
    utterances = sorted(layout["utterances"], key=lambda u: u["index"])

    with tempfile.TemporaryDirectory() as tmp:
        for utt in utterances:
            clip = Path(tmp) / f"clip_{utt['index']}.wav"
            _synthesize(utt["text"], utt["voice"], utt["speed"], utt["pitch"], clip)
            frames = _read_frames(clip)
            n_frames = len(frames) // SAMPLE_WIDTH

            start = cursor
            timeline += frames
            cursor += n_frames
            utt["start"] = round(start / SAMPLE_RATE, 3)
            utt["end"] = round(cursor / SAMPLE_RATE, 3)

            if utt is not utterances[-1]:
                timeline += silence
                cursor += gap_frames

    with wave.open(str(WAV_OUT), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(SAMPLE_WIDTH)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(bytes(timeline))

    # Persist the measured timings back into the authored layout.
    UTTERANCE_JSON.write_text(json.dumps(layout, indent=2) + "\n")

    # The attributed transcript the seed reproduces: the grounded roster speaker
    # shows its display name; every unresolved label (heard-name or unknown)
    # renders as its raw diarization label — the heard name is a suggestion, not
    # an attribution, so it never appears in the export.
    roster_label = layout["roster_speaker"]["label"]
    roster_name = layout["roster_speaker"]["display_name"]
    segments = [
        {
            "segment_index": utt["index"],
            "diarization_label": utt["label"],
            "start_seconds": utt["start"],
            "end_seconds": utt["end"],
            "text": utt["text"],
            "speaker": roster_name if utt["label"] == roster_label else utt["label"],
        }
        for utt in utterances
    ]
    TRANSCRIPT_OUT.write_text(json.dumps({"segments": segments}, indent=2) + "\n")

    sha256 = hashlib.sha256(WAV_OUT.read_bytes()).hexdigest()
    duration = round(cursor / SAMPLE_RATE, 3)
    provenance = {
        "generator": "tools/generate_tutorial_audio.py",
        "espeak_ng_version": _tool_version("espeak-ng", "--version"),
        "ffmpeg_version": _tool_version("ffmpeg", "-version"),
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
        "encoding": "pcm_s16le",
        "duration_seconds": duration,
        "gap_seconds": layout["gap_seconds"],
        "voices": {
            utt["label"]: {"voice": utt["voice"], "speed": utt["speed"], "pitch": utt["pitch"]}
            for utt in utterances
        },
        "wav_sha256": sha256,
        "license": "CC0-1.0",
    }
    PROVENANCE_OUT.write_text(json.dumps(provenance, indent=2) + "\n")

    print(f"wrote {WAV_OUT.relative_to(ASSETS.parents[3])}  ({duration:.2f}s)")
    print(f"  sha256 {sha256}")
    print(f"  espeak-ng: {provenance['espeak_ng_version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
