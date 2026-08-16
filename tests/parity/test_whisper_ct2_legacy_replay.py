"""Slice-2a anchor proof: the ``ct2-legacy`` seam reproduces the frozen oracle.

Issue #33's numerics doctrine requires that the ``WHISPER_ENGINE`` seam refactor
move the shipped decode path *mechanically* — the frozen CT2-CPU baseline
(``tests/parity/fixtures/references/ct2-cpu-metal/transcribe.json``, captured
from the pre-seam ``transcription.py``) must replay byte-for-byte through the
new ``ct2-legacy`` backend. If it does not, the Slice-2b self-parity gate would
be measuring a candidate against an accidental new CT2 segmentation.

This drives the real service ``WhisperTranscriber`` in-process (default engine =
``ct2-legacy``, ``batch_size=4`` — the pinned oracle batch size) over the same
corpus WAVs the baseline was captured from, both decode paths (``vad_true`` via
BatchedInferencePipeline + Silero, ``vad_false`` via raw ``model.transcribe``),
and asserts zero drift against every committed entry.

Maintainer-run (Gate M, docs/release-process.md): Apple-Silicon-only, needs the
pinned large-v2 snapshot (``voxint-metal.sh setup``) and, for the AMI entries,
the prepared work-dir WAVs (``tools/prepare_bakeoff_corpus.py prepare``). The
committed synthetic entries need neither and run anywhere on Apple Silicon.
Prerequisites are plain SKIPs everywhere else — ``VOXINT_PARITY_REQUIRED`` is
never applied to metal lanes. Run it from the metal whisper venv:

    ~/.voxint-metal/venvs/whisper/bin/python -m pytest \\
        tests/parity/test_whisper_ct2_legacy_replay.py -v

The full 30-entry sweep transcribes ~123 min of audio (the AMI windows
dominate); ``-k synthetic`` runs the short committed clips in seconds.
"""

from __future__ import annotations

import hashlib
import json
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

# The baseline recorded max drift 0.0 (CT2-CPU int8 is bit-exact here); these
# are the generator's determinism tolerances, kept as a hair of float slack.
# Loosening them is a numerics decision.
TS_TOL_S = 1e-3
CONF_TOL = 1e-4


def _model_root() -> Path | None:
    """The pinned large-v2 snapshot, wherever this machine caches it."""
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
        not _IS_APPLE_SILICON, reason="ct2-legacy replay runs on Apple Silicon macOS only"
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
    """Resolve a manifest entry to its WAV on disk (mirrors the generator)."""
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


@pytest.fixture(scope="session")
def transcriber() -> Any:
    """One resident ct2-legacy transcriber at the oracle batch size, configured
    exactly as the metal launcher runs it (large-v2, cpu, int8, pinned cache)."""
    from tests.contracts.conftest import service_package

    saved = {k: os.environ.get(k) for k in ("WHISPER_DOWNLOAD_ROOT", "WHISPER_REVISION")}
    os.environ["WHISPER_DOWNLOAD_ROOT"] = str(_MODEL_ROOT)
    os.environ["WHISPER_REVISION"] = WHISPER_HF_REVISION
    try:
        with service_package("whisper"):
            from app.transcription import WhisperTranscriber

            t = WhisperTranscriber(
                model_name="large-v2",
                device="cpu",
                compute_type="int8",
                batch_size=ORACLE_BATCH_SIZE,
            )
            t.load_model()
            # The seam must not have moved the identity the baseline recorded.
            assert t.engine == "faster-whisper"
            assert t.runtime == "ctranslate2"
            yield t
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _resolve_present() -> list[tuple[str, str]]:
    """(entry_key, vad_mode) params, skipping entries whose WAV is absent so a
    machine with only the committed synthetic corpus still proves those."""
    params: list[tuple[str, str]] = []
    for key in sorted(_ENTRIES):
        params.append((key, "vad_true"))
        params.append((key, "vad_false"))
    return params


def _assert_zero_drift(got: Any, ref: dict[str, Any], label: str) -> None:
    assert got.language == ref["language"], (
        f"{label}: language {got.language!r} != {ref['language']!r}"
    )
    assert abs(got.duration_seconds - ref["duration_seconds"]) <= TS_TOL_S, (
        f"{label}: duration {got.duration_seconds} vs {ref['duration_seconds']}"
    )
    assert got.transcript == ref["transcript"], f"{label}: transcript diverged from frozen oracle"
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
        assert gs["suspect"] == rs["suspect"], f"{label}: segment {i} suspect flag diverged"
        for k in ("start_seconds", "end_seconds"):
            d = abs((gs[k] or 0.0) - (rs[k] or 0.0))
            assert d <= TS_TOL_S, f"{label}: segment {i} {k} drift {d:.4g}s > {TS_TOL_S}s"
        gc, rc = gs.get("confidence"), rs.get("confidence")
        if gc is not None and rc is not None:
            assert abs(gc - rc) <= CONF_TOL, f"{label}: segment {i} confidence drift"

    assert len(got.words) == len(ref["words"]), (
        f"{label}: {len(got.words)} words vs frozen {len(ref['words'])}"
    )
    for i, (gw, rw) in enumerate(zip(got.words, ref["words"], strict=True)):
        assert gw["word"] == rw["word"], f"{label}: word {i} text diverged"
        for k in ("start_seconds", "end_seconds"):
            d = abs((gw[k] or 0.0) - (rw[k] or 0.0))
            assert d <= TS_TOL_S, f"{label}: word {i} {k} drift {d:.4g}s > {TS_TOL_S}s"


@pytest.mark.parametrize("entry_key,vad_mode", _resolve_present())
def test_ct2_legacy_replays_frozen_baseline(
    entry_key: str, vad_mode: str, transcriber: Any
) -> None:
    manifest_entry = _MANIFEST.get(entry_key)
    assert manifest_entry is not None, f"{entry_key} not in bakeoff manifest"
    wav = _wav_for(manifest_entry)
    if not wav.is_file():
        pytest.skip(f"corpus WAV absent: {wav} (run prepare_bakeoff_corpus.py)")
    # Integrity: the frozen oracle is only meaningful against the frozen bytes.
    assert _sha256(wav) == manifest_entry["sha256"], f"{entry_key}: WAV is not the frozen corpus"

    ref = _ENTRIES[entry_key]["variants"][vad_mode]
    got = transcriber.transcribe(
        str(wav), language="en", vad_filter=(vad_mode == "vad_true")
    )
    _assert_zero_drift(got, ref, f"{entry_key} {vad_mode}")
