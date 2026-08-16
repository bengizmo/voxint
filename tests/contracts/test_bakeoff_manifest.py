"""Contract: the committed whisper-bakeoff ``manifest.json`` is internally
consistent and honors the licensing doctrine on disk.

The pure schema logic lives in ``tools/prepare_bakeoff_corpus.py``
(``validate_manifest_schema``) and is unit-tested over synthetic manifests in
``tests/unit/test_bakeoff_corpus_prepare.py``. This contract binds that logic to
the *real* committed artifacts:

  * every stratum hits its exact pre-registered count;
  * AMI (CC-BY-4.0) word gold is committed and its bytes hash to the manifest's
    ``transcript_sha256``;
  * synthetic (CC0) audio is committed and hashes to the manifest ``sha256``;
  * TED-LIUM 3 (CC-BY-NC-ND-3.0) leaks nothing — no committed transcript, gold,
    or audio.

Audio for AMI/TED is never committed; only AMI gold + synthetic audio are.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import wave
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BAKEOFF = REPO / "tests" / "parity" / "fixtures" / "bakeoff"
MANIFEST = BAKEOFF / "manifest.json"


def _load_tool():
    path = REPO / "tools" / "prepare_bakeoff_corpus.py"
    spec = importlib.util.spec_from_file_location("prepare_bakeoff_corpus", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


prep = _load_tool()
src = prep.src

pytestmark = pytest.mark.skipif(
    not MANIFEST.exists(),
    reason="bakeoff manifest not generated yet (run tools/prepare_bakeoff_corpus.py generate)",
)


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def test_schema_and_strata_invariants(manifest: dict) -> None:
    prep.validate_manifest_schema(manifest)  # exact counts + per-dataset rules


def test_selection_is_pinned(manifest: dict) -> None:
    assert manifest["selection"] == {
        "seed": src.SELECTION_SEED,
        "version": src.SELECTION_VERSION,
    }


def test_ami_gold_committed_and_binds_to_transcript_sha(manifest: dict) -> None:
    ami = [e for e in manifest["files"] if e["dataset"] == "ami_ihm"]
    assert ami, "no AMI entries"
    for entry in ami:
        gold = BAKEOFF / entry["gold_file"]
        assert gold.exists(), f"missing committed gold {entry['gold_file']}"
        payload = json.loads(gold.read_text())
        recomputed = prep.sha256_hex(prep.canonical_reference_bytes(payload))
        assert recomputed == entry["transcript_sha256"], entry["upstream_id"]


def test_synthetic_audio_committed_canonical_and_hashes(manifest: dict) -> None:
    synth = [e for e in manifest["files"] if e["dataset"] == "synthetic"]
    assert synth, "no synthetic entries"
    for entry in synth:
        wav = BAKEOFF / entry["acquire"]["path"]
        assert wav.exists(), f"missing committed synthetic {entry['acquire']['path']}"
        assert hashlib.sha256(wav.read_bytes()).hexdigest() == entry["sha256"]
        with wave.open(str(wav), "rb") as w:
            assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)


def test_no_ted_transcript_or_audio_committed(manifest: dict) -> None:
    # NC-ND: TED entries carry only a hash; nothing TED-derived is on disk.
    ted = [e for e in manifest["files"] if e["dataset"] == "tedlium3"]
    assert ted, "no TED entries"
    for entry in ted:
        assert "gold_file" not in entry and "text" not in entry
    assert not (BAKEOFF / "gold" / "ted").exists()
    # No stray audio (only manifest, README, gold/ami, synthetic are committable).
    for wav in BAKEOFF.rglob("*.wav"):
        assert wav.parent.name == "synthetic", f"unexpected committed audio: {wav}"
