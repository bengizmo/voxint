"""Tier-1 auto-detect gate (issue #124): ``language=None`` matches forced-en.

Since #124 Voxint's ASR client sends ``language: null``, so production runs the
auto-detect paths the frozen parity fixtures never exercised (they force
``"en"``): the ct2 shared-VAD ``detect_language`` branch and faster-whisper's
raw/legacy detection. This gate is the measured evidence for the narrowed
equivalence claim (codex C3): *when auto-detection selects en, decoding
normally matches forced-en* — not an unconditional byte-identity claim.

For every SPEECH entry of the frozen ct2-cpu oracle (the silence and
hallucination-bait clips are excluded — with nothing to detect from,
auto-detection legitimately returns arbitrary languages, Tier-2 scope), both
engines (``ct2-legacy`` and ``ct2``), both vad modes, transcribed with
``language=None``:

- the detected language is ``"en"`` and ``language_probability`` is a finite
  score in (0, 1] (detection actually ran on the multilingual model);
- the decode matches the frozen forced-``"en"`` oracle within the replay
  gate's tolerances (transcript byte-equal, timestamps ≤1e-3 s, confidence
  ≤1e-4). The forced-en oracle itself stays frozen and is separately anchored
  by ``test_whisper_ct2_legacy_replay.py``.

Tier 2 (non-English / ambiguous / silence fixtures with fresh CUDA references)
is a separate follow-up issue, deliberately not built here.

Maintainer-run (Gate M pattern): Apple-Silicon-only, needs the metal whisper
venv and the pinned large-v2 snapshot (``voxint-metal.sh setup``); the AMI
entries additionally need the prepared work-dir WAVs. Prerequisites are plain
SKIPs everywhere else. ``-k synthetic`` runs the committed short clips only.
The two engines are resident one at a time (the cache evicts the other on
switch); params order all ct2-legacy work before ct2 so exactly one swap
happens.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
_REF_DIR = REPO / "tests" / "parity" / "fixtures" / "references" / "ct2-cpu-metal"
FIXTURE = _REF_DIR / "transcribe.json"
MANIFEST = REPO / "tests" / "parity" / "fixtures" / "bakeoff" / "manifest.json"
BAKEOFF_DIR = MANIFEST.parent

# Must match the launcher / images and the recorded oracle (meta.decode_config).
WHISPER_HF_REVISION = "f0fe81560cb8b68660e564f55dd99207059c092e"
ORACLE_BATCH_SIZE = 4

# The replay gate's tolerances, reused verbatim: this gate makes the SAME
# zero-drift claim, conditioned on detection selecting en. Loosening them is a
# numerics decision.
TS_TOL_S = 1e-3
CONF_TOL = 1e-4

_ENGINES = ("ct2-legacy", "ct2")


def _model_root() -> Path | None:
    metal_home = Path(os.getenv("VOXINT_METAL_HOME", str(Path.home() / ".voxint-metal")))
    root = Path(os.getenv("WHISPER_DOWNLOAD_ROOT", str(metal_home / "models" / "whisper")))
    snapshot = (
        root / "models--Systran--faster-whisper-large-v2" / "snapshots" / WHISPER_HF_REVISION
    )
    return root if (snapshot / "model.bin").is_file() else None


def _work_dir() -> Path:
    metal_home = Path(os.getenv("VOXINT_METAL_HOME", str(Path.home() / ".voxint-metal")))
    return Path(os.getenv("VOXINT_BAKEOFF_WORK", str(metal_home / "bakeoff" / "work")))


_IS_APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"
_MODEL_ROOT = _model_root() if _IS_APPLE_SILICON else None

pytestmark = [
    pytest.mark.skipif(
        not _IS_APPLE_SILICON,
        reason="auto-detect Tier-1 gate runs on Apple Silicon macOS only",
    ),
    pytest.mark.skipif(not FIXTURE.exists(), reason="frozen CT2-CPU baseline missing"),
    pytest.mark.skipif(
        _IS_APPLE_SILICON and _MODEL_ROOT is None,
        reason="pinned large-v2 model not downloaded — run scripts/metal/voxint-metal.sh setup",
    ),
]

if _IS_APPLE_SILICON:
    pytest.importorskip("faster_whisper", reason="whisper metal venv required")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_by_key() -> dict[str, dict[str, Any]]:
    files = json.loads(MANIFEST.read_text())["files"]
    return {f"{f['dataset']}/{f['upstream_id']}": f for f in files}


def _wav_for(entry: dict[str, Any]) -> Path:
    acq = entry["acquire"]
    if acq["kind"] == "committed":
        return BAKEOFF_DIR / str(acq["path"])
    if acq["kind"] == "ami_range":
        return _work_dir() / "ami" / f"{acq['meeting']}.{acq['agent']}.wav"
    if acq["kind"] == "ted_window":
        return _work_dir() / "ted" / f"{entry['upstream_id']}.wav"
    raise AssertionError(f"unknown acquire.kind {acq['kind']!r}")


_ENTRIES: dict[str, Any] = (
    json.loads(FIXTURE.read_text())["entries"] if FIXTURE.exists() else {}
)
_MANIFEST: dict[str, dict[str, Any]] = _manifest_by_key() if MANIFEST.exists() else {}


# Non-speech oracle entries (silence + hallucination bait) are OUT of Tier-1
# scope: with nothing to detect from, auto-detection legitimately returns
# arbitrary languages, so the en-conditional equivalence claim does not apply.
# Their auto-detect behavior belongs to the Tier-2 follow-up (fresh references
# for silence/ambiguous input), deliberately not built here.
_NON_SPEECH_PREFIXES = ("synthetic/bait_", "synthetic/silence_")


def _params() -> list[tuple[str, str, str]]:
    # Engine outermost so all ct2-legacy tests run before any ct2 test — the
    # transcriber cache then swaps models exactly once.
    return [
        (engine, key, vad)
        for engine in _ENGINES
        for key in sorted(_ENTRIES)
        if not key.startswith(_NON_SPEECH_PREFIXES)
        for vad in ("vad_true", "vad_false")
    ]


class _TranscriberCache:
    """One resident engine at a time; switching evicts the other model."""

    def __init__(self) -> None:
        self._engine: str | None = None
        self._transcriber: Any = None
        self._env_saved: dict[str, str | None] = {}

    def get(self, engine: str) -> Any:
        if self._engine == engine:
            return self._transcriber
        self.close()
        os.environ["WHISPER_DOWNLOAD_ROOT"] = str(_MODEL_ROOT)
        os.environ["WHISPER_REVISION"] = WHISPER_HF_REVISION
        from tests.contracts.conftest import service_package

        with service_package("whisper"):
            from app.backends.ct2 import Ct2Backend
            from app.transcription import WhisperTranscriber

            if engine == "ct2-legacy":
                t = WhisperTranscriber(
                    model_name="large-v2",
                    device="cpu",
                    compute_type="int8",
                    batch_size=ORACLE_BATCH_SIZE,
                )
            else:
                t = WhisperTranscriber(
                    backend=Ct2Backend(
                        model_name="large-v2",
                        device="cpu",
                        compute_type="int8",
                        batch_size=ORACLE_BATCH_SIZE,
                    )
                )
            t.load_model()
        self._engine, self._transcriber = engine, t
        return t

    def close(self) -> None:
        if self._transcriber is not None:
            self._transcriber.cleanup_memory()
        self._engine, self._transcriber = None, None
        gc.collect()


@pytest.fixture(scope="session")
def transcribers() -> Any:
    saved = {k: os.environ.get(k) for k in ("WHISPER_DOWNLOAD_ROOT", "WHISPER_REVISION")}
    cache = _TranscriberCache()
    try:
        yield cache
    finally:
        cache.close()
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _assert_matches_forced_en_oracle(got: Any, ref: dict[str, Any], label: str) -> None:
    assert got.language == "en", (
        f"{label}: auto-detection selected {got.language!r}, not 'en' — the "
        f"conditional equivalence claim does not apply; investigate the clip"
    )
    prob = got.language_probability
    assert prob is not None, f"{label}: detection ran but reported no score"
    assert math.isfinite(prob) and 0.0 < prob <= 1.0, (
        f"{label}: detection score {prob!r} outside (0, 1]"
    )

    assert abs(got.duration_seconds - ref["duration_seconds"]) <= TS_TOL_S, (
        f"{label}: duration {got.duration_seconds} vs {ref['duration_seconds']}"
    )
    assert got.transcript == ref["transcript"], (
        f"{label}: transcript diverged from the frozen forced-en oracle"
    )
    assert got.suspect_segment_count == ref["suspect_segment_count"], (
        f"{label}: suspect_segment_count {got.suspect_segment_count} "
        f"vs {ref['suspect_segment_count']}"
    )
    assert abs(got.confidence - ref["confidence"]) <= CONF_TOL, (
        f"{label}: overall confidence {got.confidence} vs {ref['confidence']}"
    )
    assert len(got.segments) == len(ref["segments"]), (
        f"{label}: {len(got.segments)} segments vs frozen {len(ref['segments'])}"
    )
    for i, (gs, rs) in enumerate(zip(got.segments, ref["segments"], strict=True)):
        assert gs["text"] == rs["text"], f"{label}: segment {i} text diverged"
        for k in ("start_seconds", "end_seconds"):
            d = abs((gs[k] or 0.0) - (rs[k] or 0.0))
            assert d <= TS_TOL_S, f"{label}: segment {i} {k} drift {d:.4g}s > {TS_TOL_S}s"


@pytest.mark.parametrize("engine,entry_key,vad_mode", _params())
def test_autodetect_en_matches_forced_en_oracle(
    engine: str, entry_key: str, vad_mode: str, transcribers: Any
) -> None:
    manifest_entry = _MANIFEST.get(entry_key)
    assert manifest_entry is not None, f"{entry_key} not in bakeoff manifest"
    wav = _wav_for(manifest_entry)
    if not wav.is_file():
        pytest.skip(f"corpus WAV absent: {wav} (run prepare_bakeoff_corpus.py)")
    assert _sha256(wav) == manifest_entry["sha256"], (
        f"{entry_key}: WAV is not the frozen corpus"
    )

    ref = _ENTRIES[entry_key]["variants"][vad_mode]
    got = transcribers.get(engine).transcribe(
        str(wav), language=None, vad_filter=(vad_mode == "vad_true")
    )
    _assert_matches_forced_en_oracle(got, ref, f"{engine} {entry_key} {vad_mode}")
