"""VADPlan prototype gate (#33 Slice 2b, step 1): prove the shared front's
packed windows are byte-identical to faster-whisper 1.2.1's internal VAD.

The ``ct2`` engine's shared front (``app.backends.vad_plan.build_vad_plan``)
reuses faster-whisper's ``get_speech_timestamps`` + ``collect_chunks`` so the
packed decode windows match what ``BatchedInferencePipeline.transcribe``
computes inside the shipped ``ct2-legacy`` path. This gate measures that
equality directly — packed-PCM sha256 and the integer-sample speech-chunk list
— on real speech (committed synthetic + AMI when present) plus the no-speech
short-circuit.

The oracle is deliberately written a DIFFERENT way than ``build_vad_plan``:
it overrides ONLY the two ``VadOptions`` fields the pipeline overrides
(``max_speech_duration_s`` / ``min_silence_duration_ms``) and lets the
``VadOptions`` dataclass defaults supply the rest, whereas ``build_vad_plan``
hardcodes all six explicitly. Identical output cross-checks those literals
against faster-whisper's own defaults — so a future faster-whisper bump that
moved a default would fail here loudly.

Maintainer-run, Apple-Silicon-only (mirrors ``test_whisper_metal.py``); plain
SKIP everywhere else. Needs faster-whisper in the running interpreter (the
metal whisper venv), but NOT the large-v2 weights — this is VAD only.
"""

from __future__ import annotations

import hashlib
import os
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
CORPUS_DIR = REPO / "tests" / "parity" / "fixtures" / "corpus"

_IS_APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"

pytestmark = pytest.mark.skipif(
    not _IS_APPLE_SILICON, reason="metal VADPlan gate runs on Apple Silicon macOS only"
)

if _IS_APPLE_SILICON:
    pytest.importorskip("faster_whisper", reason="whisper metal venv required")


def _ami_wavs() -> list[Path]:
    """Real multi-interval speech (pauses + turn-taking) when the AMI corpus is
    staged on maintainer hardware; empty otherwise (the gate still runs on the
    committed synthetic fixture)."""
    metal_home = Path(
        os.getenv("VOXINT_METAL_HOME", str(Path.home() / ".voxint-metal"))
    )
    ami = metal_home / "bakeoff" / "work" / "ami"
    return sorted(ami.glob("*.wav"))[:3] if ami.is_dir() else []


def _speech_wavs() -> list[Path]:
    wavs = [CORPUS_DIR / "transcribe-short.wav"]
    return [w for w in wavs if w.is_file()] + _ami_wavs()


def _decode(wav: Path) -> Any:
    from faster_whisper.audio import decode_audio

    return decode_audio(str(wav), sampling_rate=16000)


def _fw_reference(audio: Any) -> tuple[list[dict], list[Any]]:
    """Independent oracle: mirror BatchedInferencePipeline.transcribe:398-424,
    overriding only the two fields the pipeline overrides."""
    from faster_whisper.vad import VadOptions, collect_chunks, get_speech_timestamps

    vad_options = VadOptions(max_speech_duration_s=30, min_silence_duration_ms=160)
    chunks = get_speech_timestamps(audio, vad_options)
    audio_chunks, _meta = collect_chunks(
        audio, chunks, sampling_rate=16000, max_duration=30
    )
    return chunks, audio_chunks


def _pcm_sha(arr: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.float32).tobytes()).hexdigest()


@pytest.fixture(scope="module")
def _vad_plan_module() -> Any:
    from tests.contracts.conftest import service_package

    with service_package("whisper"):
        from app.backends import vad_plan

        return vad_plan


@pytest.mark.parametrize("wav", _speech_wavs(), ids=lambda w: w.name)
def test_vad_plan_matches_faster_whisper(_vad_plan_module: Any, wav: Path) -> None:
    """build_vad_plan's packed windows == faster-whisper's own VAD+pack."""
    audio = _decode(wav)
    ref_chunks, ref_audio_chunks = _fw_reference(audio)
    plan = _vad_plan_module.build_vad_plan(audio)

    # Integer-sample speech-chunk list (the restoration source of truth).
    assert plan.speech_chunks == ref_chunks, f"speech chunks diverged for {wav.name}"

    # One packed window per fw audio chunk, byte-identical PCM.
    assert len(plan.windows) == len(ref_audio_chunks)
    for i, (win, ref_chunk) in enumerate(
        zip(plan.windows, ref_audio_chunks, strict=True)
    ):
        assert _pcm_sha(win.audio) == _pcm_sha(ref_chunk), (
            f"packed PCM diverged: {wav.name} window {i}"
        )

    assert plan.has_speech is True
    # Original file duration, never packed/post-VAD.
    assert plan.duration_seconds == pytest.approx(audio.shape[0] / 16000)


def test_no_speech_short_circuits(_vad_plan_module: Any) -> None:
    """Pure silence → VAD finds nothing → empty plan at the original duration."""
    silence = np.zeros(16000 * 2, dtype=np.float32)  # 2 s
    plan = _vad_plan_module.build_vad_plan(silence)

    assert plan.has_speech is False
    assert plan.windows == []
    assert plan.speech_chunks == []
    assert plan.duration_seconds == pytest.approx(2.0)


def test_restoration_wiring_is_faster_whisper(_vad_plan_module: Any) -> None:
    """The front restores packed→source time via faster-whisper's own
    restore_speech_timestamps against the GLOBAL chunk list (not a per-window
    map). Guard the wiring: a segment spanning the first packed window restores
    to a source start at/after the first speech chunk, monotonic and in-bounds.
    """
    from faster_whisper.transcribe import Segment, restore_speech_timestamps

    speech = _speech_wavs()
    if not speech:
        pytest.skip("no speech fixture available")
    audio = _decode(speech[0])
    plan = _vad_plan_module.build_vad_plan(audio)
    assert plan.has_speech

    first = plan.windows[0].metadata
    packed_start = float(first["offset"])
    packed_end = packed_start + float(first["duration"])
    seg = Segment(
        id=1, seek=0, start=packed_start, end=packed_end, text="x",
        tokens=[], avg_logprob=-0.1, compression_ratio=1.0, no_speech_prob=0.0,
        words=None, temperature=0.0,
    )
    restored = list(restore_speech_timestamps([seg], plan.speech_chunks, 16000))
    assert len(restored) == 1
    # The first window's packed offset restores to the first speech chunk's
    # absolute source start; monotonic and within the original duration.
    src_start = plan.speech_chunks[0]["start"] / 16000
    assert abs(restored[0].start - src_start) < 0.5
    assert restored[0].end >= restored[0].start
    assert restored[0].end <= plan.duration_seconds + 0.5
