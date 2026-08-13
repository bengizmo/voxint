#!/usr/bin/env python3
"""Generate the redistributable parity golden corpus (run ONCE by a human).

The non-NVIDIA GPU work (issues #1/#4) gates every alternative backend on
measured equivalence against CUDA reference outputs (docs/gpu-contracts.md,
"Equivalence policy"). This script synthesizes the *inputs* for that gate —
entirely from espeak-ng voices and seeded noise, so the corpus is original,
deterministic, and redistributable (CC0, same provenance model as the guided
tutorial's sample audio).

It writes committed artifacts under ``tests/parity/fixtures/corpus/``:

* ``embed-corpus.wav``      — one 16 kHz mono timeline: 24 utterances from 6
  synthetic speakers, plus noise-mixed copies calibrated (against the titanet
  service's own SNR estimator) to straddle the 5 dB low_snr gate.
* ``embed-windows.json``    — ~100 windows over that timeline: clean windows,
  sub-windows, 1.0 s too_short boundary straddles, SNR straddles, silence,
  past-end, and determinism duplicates. Each window records its expected
  skip_reason, computed with the service's exact slice + SNR math.
* ``embed-pairs.json``      — labeled same/different-speaker window pairs for
  the decision-level parity gate.
* ``transcribe-short.wav``  — one-voice scripted passage for ASR tolerance
  checks (script text included in the manifest).
* ``diarize-3speaker.wav``  — three voices alternating, sample-accurate turn
  layout recorded for diarization tolerance checks.
* ``manifest.json``         — clip metadata, scripts/layouts, checksums.
* ``provenance.json``       — tool versions, seeds, voice args, license.

Reference *outputs* are produced separately by
``tools/generate_parity_references.py`` against the CUDA services.

espeak-ng is GPLv3; we invoke the system binary and vendor nothing. The texts,
this script, and the resulting WAVs are original works of the Voxint project,
dedicated to CC0 (see tests/parity/fixtures/corpus/README.md).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CORPUS_DIR = REPO / "tests" / "parity" / "fixtures" / "corpus"

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # pcm_s16le
GAP_SECONDS = 0.5
NOISE_SEED = 20260813
SNR_TOLERANCE_DB = 0.05


def _load_titanet_preprocess() -> ModuleType:
    """The service's own slice + SNR math — expectations must use IT, not a copy."""
    path = REPO / "services" / "titanet" / "app" / "preprocess.py"
    spec = importlib.util.spec_from_file_location("titanet_preprocess", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["titanet_preprocess"] = mod
    spec.loader.exec_module(mod)
    return mod


preprocess = _load_titanet_preprocess()

# ---------------------------------------------------------------------------
# Authored content (original, fictional, domain-neutral). Six distinct espeak
# voice profiles = six labeled synthetic "speakers" for the decision gate.
# ---------------------------------------------------------------------------

VOICES: dict[str, dict[str, Any]] = {
    "SPK_A": {"voice": "en-us+m3", "speed": 150, "pitch": 40},
    "SPK_B": {"voice": "en-us+f4", "speed": 168, "pitch": 62},
    "SPK_C": {"voice": "en-gb+m6", "speed": 140, "pitch": 28},
    "SPK_D": {"voice": "en-us+f2", "speed": 155, "pitch": 70},
    "SPK_E": {"voice": "en-gb+m2", "speed": 132, "pitch": 48},
    "SPK_F": {"voice": "en-us+f5", "speed": 175, "pitch": 55},
}

# Four utterances per speaker: one short, two medium, one long.
UTTERANCE_TEXTS: list[str] = [
    "The harbor lights dim after midnight.",
    "Every autumn the river carries red leaves past the old stone bridge and out toward the bay.",
    "Nobody remembered who first planted the orchard, but the trees kept their rows.",
    "When the northern wind finally settled, the climbers checked their ropes twice, shouldered "
    "their packs, and started the long traverse across the glacier before sunrise could soften "
    "the ice beneath them.",
    "Fresh bread cools by the window.",
    "The observatory dome opened slowly while the astronomer aligned the mirror on the horizon.",
    "A single lantern is enough to cross the meadow if you trust the path.",
    "The archivist numbered each folder, tied the bundles with cotton string, and carried them "
    "down to the basement vault where the temperature never changed and the dust settled in "
    "even, patient layers.",
    "Rain traced lines down the glass.",
    "Two ferries pass each other every morning in the narrow channel south of the lighthouse.",
    "The chess club met on Thursdays above the bakery, and the smell of cinnamon always won.",
    "After the festival ended, volunteers folded the banners, stacked the chairs in the town "
    "hall, swept the square in overlapping circles, and argued cheerfully about where next "
    "year's stage should stand.",
    "The kettle whistled twice, then stopped.",
    "Maps of the old coastline show a harbor where the parking lot is now.",
    "She tuned the violin slowly, listening for the fifth to settle between the strings.",
    "The night train slid past the sleeping villages, its windows a ribbon of pale light, and "
    "the conductor walked the aisle counting tickets with the ease of a man who had done it "
    "ten thousand times.",
    "Snow erased the garden by noon.",
    "The printing press needed three people: one to feed, one to catch, and one to worry.",
    "Low tide uncovered a field of stones arranged in a spiral nobody could date.",
    "In the workshop behind the chandlery, coils of rope hung from oak pegs by thickness and "
    "lay, and the smell of tar and hemp followed the apprentices home through the winding "
    "streets every evening.",
    "The elevator hummed one floor too far.",
    "Morning fog kept the gliders grounded until the ridge finally showed its edge.",
    "He labeled the seed envelopes in pencil because ink fades faster than memory.",
    "The orchestra rehearsed the final movement four times, and each time the silence after "
    "the last chord lasted a little longer, as if the hall itself were learning where the "
    "music ended.",
]

TRANSCRIBE_TEXT = (
    "This recording exists to measure transcription parity. The quick grey fox counted "
    "thirty two crates of oranges on the harbor scale. Numbers matter here: seventeen, "
    "four hundred six, and nineteen fifty two. The committee approved the proposal without "
    "amendment, then adjourned for the season. A longer sentence follows, containing a few "
    "deliberately unusual words such as gyroscope, marmalade, and periwinkle, spoken at a "
    "steady pace to give the decoder an honest chance."
)

DIARIZE_SCRIPT: list[tuple[str, str]] = [
    ("SPK_A", "Welcome back everyone, let's review the survey results from last week."),
    ("SPK_B", "Thanks. The response rate was higher than expected, almost seventy percent."),
    ("SPK_C", "That is encouraging. Did the coastal group respond as well as the valley group?"),
    ("SPK_A", "They did, and their comments were longer on average."),
    ("SPK_B", "I can summarize the main themes if that helps the planning discussion."),
    ("SPK_C", "Please do, and note anything that contradicts the earlier interviews."),
    ("SPK_A", "Agreed. We will hold the open questions for the end of the session."),
    ("SPK_B", "Then I will start with the three most common suggestions we received."),
]


def _tool_version(binary: str, *args: str) -> str:
    out = subprocess.run([binary, *args], capture_output=True, text=True, check=True)
    for line in (out.stdout + out.stderr).splitlines():
        if line.strip():
            return line.strip()
    return "unknown"


def _synthesize(text: str, voice: str, speed: int, pitch: int, dest: Path) -> None:
    """espeak-ng -> raw WAV, then ffmpeg -> 16 kHz mono PCM at ``dest``."""
    with tempfile.NamedTemporaryFile(suffix=".wav") as raw:
        subprocess.run(
            ["espeak-ng", "-v", voice, "-s", str(speed), "-p", str(pitch), "-w", raw.name, text],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", raw.name,
                "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le",
                str(dest),
            ],
            check=True,
            capture_output=True,
        )


def _read_float(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        if wav.getframerate() != SAMPLE_RATE or wav.getnchannels() != 1:
            raise RuntimeError(f"{path} is not 16 kHz mono after resample")
        raw = wav.readframes(wav.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _quantize(audio: np.ndarray) -> np.ndarray:
    """Round-trip through int16 exactly as the WAV (and torchaudio.load) will."""
    clipped = np.clip(audio, -1.0, 1.0 - 1.0 / 32768.0)
    return np.round(clipped * 32768.0).astype(np.int16).astype(np.float32) / 32768.0


def _write_wav(path: Path, audio: np.ndarray) -> str:
    clipped = np.clip(audio, -1.0, 1.0 - 1.0 / 32768.0)
    pcm = np.round(clipped * 32768.0).astype(np.int16)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(SAMPLE_WIDTH)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(pcm.tobytes())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mix_to_target_snr(
    speech: np.ndarray, noise: np.ndarray, target_db: float
) -> tuple[np.ndarray, float]:
    """Bisect the noise gain until the SERVICE's estimator reads ``target_db``.

    Calibrating against ``calculate_snr_db`` on the int16-quantized mix (not a
    textbook SNR formula) makes the gate expectations exact by construction.
    """
    lo, hi = 1e-6, 4.0
    best_audio, best_snr = speech, preprocess.calculate_snr_db(_quantize(speech))
    for _ in range(60):
        gain = (lo + hi) / 2.0
        mixed = _quantize(0.85 * speech + gain * noise)
        snr = preprocess.calculate_snr_db(mixed)
        best_audio, best_snr = mixed, snr
        if abs(snr - target_db) <= SNR_TOLERANCE_DB:
            break
        if snr > target_db:
            lo = gain  # more noise -> lower measured SNR
        else:
            hi = gain
    return best_audio, best_snr


class Timeline:
    """Sample-accurate concatenation with silence gaps."""

    def __init__(self) -> None:
        self.audio = np.zeros(0, dtype=np.float32)

    def append(self, clip: np.ndarray, gap_seconds: float = GAP_SECONDS) -> tuple[float, float]:
        gap = np.zeros(int(gap_seconds * SAMPLE_RATE), dtype=np.float32)
        start = len(self.audio) / SAMPLE_RATE
        self.audio = np.concatenate([self.audio, clip, gap])
        end = start + len(clip) / SAMPLE_RATE
        return round(start, 6), round(end, 6)


def _expected_skip(
    timeline: np.ndarray, start_s: float, end_s: float
) -> tuple[str | None, float | None]:
    """Replicate the service's gate math exactly on the final timeline."""
    lo, hi = preprocess.window_sample_bounds(start_s, end_s, SAMPLE_RATE, len(timeline))
    window = timeline[lo:hi]
    if len(window) < int(preprocess.MIN_WINDOW_SECONDS * SAMPLE_RATE):
        return "too_short", None
    snr = preprocess.calculate_snr_db(window)
    if snr < 5.0:  # service default TITANET_SNR_THRESHOLD_DB
        return "low_snr", round(snr, 2)
    return None, round(snr, 2)


def main() -> int:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(NOISE_SEED)
    speakers = list(VOICES)

    # --- synthesize the 24 base utterances -------------------------------
    utterances: list[dict[str, Any]] = []
    clips: dict[str, np.ndarray] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for i, text in enumerate(UTTERANCE_TEXTS):
            speaker = speakers[i // 4]
            args = VOICES[speaker]
            dest = Path(tmp) / f"utt_{i:02d}.wav"
            _synthesize(text, args["voice"], args["speed"], args["pitch"], dest)
            clip = _read_float(dest)
            utt_id = f"utt_{i:02d}"
            clips[utt_id] = clip
            utterances.append(
                {"id": utt_id, "speaker": speaker, "text": text,
                 "duration_seconds": round(len(clip) / SAMPLE_RATE, 3)}
            )

    # --- build the embed timeline ----------------------------------------
    timeline = Timeline()
    placed: dict[str, tuple[float, float]] = {}
    for utt in utterances:
        placed[utt["id"]] = timeline.append(clips[utt["id"]])
        utt["start_seconds"], utt["end_seconds"] = placed[utt["id"]]

    # Noise-mixed copies straddling the 5 dB gate (and one clearly below,
    # one clearly above), calibrated against the service's own estimator.
    snr_targets = [3.0, 4.0, 4.7, 5.3, 6.0, 8.0, 12.0, 20.0]
    noisy_entries: list[dict[str, Any]] = []
    for j, target in enumerate(snr_targets):
        utt = utterances[(j * 5) % len(utterances)]
        speech = clips[utt["id"]]
        noise = rng.normal(0.0, 1.0, len(speech)).astype(np.float32) * 0.05
        mixed, achieved = _mix_to_target_snr(speech, noise, target)
        start, end = timeline.append(mixed)
        noisy_entries.append(
            {"id": f"noisy_{j}", "source_utterance": utt["id"], "speaker": utt["speaker"],
             "target_snr_db": target, "calibrated_snr_db": round(achieved, 2),
             "start_seconds": start, "end_seconds": end}
        )

    # A dedicated 2 s digital-silence region: pure-zero windows must hit the
    # 0 dB -> low_snr path (RMS < 1e-6), which the 0.5 s gaps are too short for.
    silence_start, silence_end = timeline.append(
        np.zeros(2 * SAMPLE_RATE, dtype=np.float32)
    )

    audio = timeline.audio
    total_seconds = len(audio) / SAMPLE_RATE

    # --- window manifest --------------------------------------------------
    windows: list[dict[str, Any]] = []

    def add(win_id: str, category: str, start_s: float, end_s: float,
            speaker: str | None, utterance: str | None) -> None:
        skip, snr = _expected_skip(audio, start_s, end_s)
        windows.append(
            {"id": win_id, "category": category,
             "start_seconds": round(start_s, 6), "end_seconds": round(end_s, 6),
             "speaker": speaker, "utterance": utterance,
             "expected_skip_reason": skip, "expected_snr_db": snr}
        )

    # 1. clean full-utterance windows (24)
    for utt in utterances:
        s, e = placed[utt["id"]]
        add(f"clean_{utt['id']}", "clean", s, e, utt["speaker"], utt["id"])

    # 2. sub-windows of every utterance longer than 4 s (3 each)
    for utt in utterances:
        s, e = placed[utt["id"]]
        dur = e - s
        if dur < 4.0:
            continue
        for k, (off, sub_dur) in enumerate([(0.2, 1.5), (0.5, 2.5), (dur - 3.6, 3.2)]):
            add(f"sub_{utt['id']}_{k}", "sub_window", s + off, s + off + sub_dur,
                utt["speaker"], utt["id"])

    # 3. 1.0 s too_short boundary straddles from a long utterance
    long_utt = max(utterances, key=lambda u: u["duration_seconds"])
    ls, _le = placed[long_utt["id"]]
    for dur in [0.9375, 0.99, 0.999, 1.0, 1.001, 1.05, 1.25, 0.5]:
        add(f"lenb_{dur}", "length_boundary", ls + 1.0, ls + 1.0 + dur,
            long_utt["speaker"], long_utt["id"])

    # 4. SNR straddles: the calibrated noisy copies (full window + one sub)
    for entry in noisy_entries:
        s, e = entry["start_seconds"], entry["end_seconds"]
        add(f"snrb_{entry['id']}", "snr_boundary", s, e, entry["speaker"],
            entry["source_utterance"])
        add(f"snrb_{entry['id']}_sub", "snr_boundary", s + 0.15, min(s + 1.9, e),
            entry["speaker"], entry["source_utterance"])

    # 5a. pure-silence windows inside the dedicated zero region (0 dB -> low_snr)
    add("silence_0", "silence", silence_start + 0.1, silence_start + 1.4, None, None)
    add("silence_1", "silence", silence_start + 0.4, silence_end - 0.1, None, None)

    # 5b. gap straddles: mostly digital-silence gap + a speech tail exercises
    # the digitally-silent noise-floor special case (40 dB path)
    for k, utt in enumerate(utterances[:2]):
        _s, e = placed[utt["id"]]
        add(f"gap_straddle_{k}", "gap_straddle", e + 0.02, e + 0.02 + 1.2, None, None)

    # 6. past-end windows (shorter slice; typically too_short)
    add("past_end_short", "past_end", total_seconds - 0.4, total_seconds + 3.0, None, None)
    add("past_end_long", "past_end", total_seconds - 2.0, total_seconds + 6.0, None, None)

    # 7. determinism duplicates: same span requested twice must embed identically
    for utt in [utterances[1], utterances[9]]:
        s, e = placed[utt["id"]]
        add(f"dup_{utt['id']}", "duplicate", s, e, utt["speaker"], utt["id"])

    # --- decision-level pairs ---------------------------------------------
    clean_by_speaker: dict[str, list[str]] = {}
    for w in windows:
        if w["category"] in {"clean", "sub_window"} and w["expected_skip_reason"] is None:
            clean_by_speaker.setdefault(w["speaker"], []).append(w["id"])
    same_pairs = [
        [ids[i], ids[j]]
        for ids in clean_by_speaker.values()
        for i in range(len(ids))
        for j in range(i + 1, len(ids))
    ]
    diff_pairs = []
    spk_list = sorted(clean_by_speaker)
    for a_idx in range(len(spk_list)):
        for b_idx in range(a_idx + 1, len(spk_list)):
            a_ids = clean_by_speaker[spk_list[a_idx]]
            b_ids = clean_by_speaker[spk_list[b_idx]]
            diff_pairs.append([a_ids[0], b_ids[0]])
            diff_pairs.append([a_ids[-1], b_ids[len(b_ids) // 2]])

    # --- transcription + diarization clips --------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        t_dest = Path(tmp) / "transcribe.wav"
        args = VOICES["SPK_C"]
        _synthesize(TRANSCRIBE_TEXT, args["voice"], args["speed"], args["pitch"], t_dest)
        transcribe_audio = _read_float(t_dest)

        d_timeline = Timeline()
        d_turns = []
        for i, (speaker, text) in enumerate(DIARIZE_SCRIPT):
            v = VOICES[speaker]
            d_dest = Path(tmp) / f"turn_{i}.wav"
            _synthesize(text, v["voice"], v["speed"], v["pitch"], d_dest)
            s, e = d_timeline.append(_read_float(d_dest), gap_seconds=0.35)
            d_turns.append({"speaker": speaker, "text": text,
                            "start_seconds": s, "end_seconds": e})

    embed_sha = _write_wav(CORPUS_DIR / "embed-corpus.wav", audio)
    transcribe_sha = _write_wav(CORPUS_DIR / "transcribe-short.wav", transcribe_audio)
    diarize_sha = _write_wav(CORPUS_DIR / "diarize-3speaker.wav", d_timeline.audio)

    (CORPUS_DIR / "embed-windows.json").write_text(
        json.dumps({"wav": "embed-corpus.wav", "windows": windows}, indent=2) + "\n"
    )
    (CORPUS_DIR / "embed-pairs.json").write_text(
        json.dumps(
            {"same_speaker_pairs": same_pairs, "different_speaker_pairs": diff_pairs},
            indent=2,
        ) + "\n"
    )
    (CORPUS_DIR / "manifest.json").write_text(
        json.dumps(
            {
                "embed": {"wav": "embed-corpus.wav", "sha256": embed_sha,
                          "duration_seconds": round(total_seconds, 3),
                          "utterances": utterances, "noisy_copies": noisy_entries},
                "transcribe": {"wav": "transcribe-short.wav", "sha256": transcribe_sha,
                               "duration_seconds": round(len(transcribe_audio) / SAMPLE_RATE, 3),
                               "speaker": "SPK_C", "text": TRANSCRIBE_TEXT},
                "diarize": {"wav": "diarize-3speaker.wav", "sha256": diarize_sha,
                            "duration_seconds": round(len(d_timeline.audio) / SAMPLE_RATE, 3),
                            "turns": d_turns},
            },
            indent=2,
        ) + "\n"
    )
    (CORPUS_DIR / "provenance.json").write_text(
        json.dumps(
            {
                "generator": "tools/generate_parity_corpus.py",
                "espeak_ng_version": _tool_version("espeak-ng", "--version"),
                "ffmpeg_version": _tool_version("ffmpeg", "-version"),
                "noise_seed": NOISE_SEED,
                "sample_rate": SAMPLE_RATE,
                "channels": 1,
                "encoding": "pcm_s16le",
                "voices": VOICES,
                "wav_sha256": {"embed-corpus.wav": embed_sha,
                               "transcribe-short.wav": transcribe_sha,
                               "diarize-3speaker.wav": diarize_sha},
                "license": "CC0-1.0",
            },
            indent=2,
        ) + "\n"
    )

    n_skip = sum(1 for w in windows if w["expected_skip_reason"])
    print(f"embed-corpus.wav: {total_seconds:.1f}s, {len(windows)} windows "
          f"({n_skip} expected skips), {len(same_pairs)} same / {len(diff_pairs)} diff pairs")
    print(f"transcribe-short.wav: {len(transcribe_audio) / SAMPLE_RATE:.1f}s")
    print(f"diarize-3speaker.wav: {len(d_timeline.audio) / SAMPLE_RATE:.1f}s, "
          f"{len(d_turns)} turns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
