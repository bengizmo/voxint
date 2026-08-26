"""Prepare executor tests for synthdetect S5 PR-2a (issue #144).

Exercises the ffmpeg-free materialization path end to end against a synthetic,
coordinate-coded source (each sample encodes its own index, so a global shift or an
off-by-one is detectable independently of the executor's own interval math). Covers
the happy path plus an independent slice oracle, reproducibility, overlap identity,
the acquisition-manifest pin verification, and the fail-closed matrix. No ffmpeg and
no real corpus file is touched.
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import synthdetect_corpus as corpus  # noqa: E402


def _coord_payload(n_samples: int) -> bytes:
    """Sample i holds ((i % 65536) - 32768) as s16le, so absolute position is checkable."""
    return b"".join(struct.pack("<h", (i % 65536) - 32768) for i in range(n_samples))


def _rttm(*rows: tuple[str, str, float, float]) -> str:
    return (
        "\n".join(
            f"SPEAKER {rec} 1 {start} {dur} <NA> <NA> {label} <NA> <NA>"
            for rec, label, start, dur in rows
        )
        + "\n"
    )


@dataclass(frozen=True)
class _Source:
    audio_dir: Path
    rttm_path: Path
    acq_path: Path
    payload: bytes


def _build_source(tmp_path: Path, *, n_samples: int = 110_000) -> _Source:
    """Stage a synthetic 'ami' source: one recording, one speaker, two turns.

    Turn gap 0.5 s (> the 0.3 s turn-merge, < the 1.0 s session-merge), so the plan
    yields two turn clips and one merged session segment spanning both.
    """
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    payload = _coord_payload(n_samples)
    rec_path = audio_dir / "rec1.wav"
    corpus.write_canonical_wav(rec_path, payload)
    rec_sha, rec_size = corpus._hash_file(rec_path)

    rttm_text = _rttm(("rec1", "S", 0.0, 3.0), ("rec1", "S", 3.5, 3.0))
    rttm_path = tmp_path / "rec1.rttm"
    rttm_path.write_text(rttm_text, encoding="utf-8")
    rttm_bytes = rttm_path.read_bytes()

    acq = {
        "source": "ami",
        "recordings": {"rec1": {"rel_path": "rec1.wav", "sha256": rec_sha, "size": rec_size}},
        "rttms": [
            {
                "rel_path": "rec1.rttm",
                "sha256": hashlib.sha256(rttm_bytes).hexdigest(),
                "size": len(rttm_bytes),
            }
        ],
    }
    acq_path = tmp_path / "acquisition.json"
    acq_path.write_text(json.dumps(acq), encoding="utf-8")
    return _Source(audio_dir, rttm_path, acq_path, payload)


def _run(src: _Source, corpus_root: Path) -> corpus.PrepareResult:
    source = corpus.ORGANIC_SOURCES["ami"]
    _, recording_pins, rttm_pins = corpus.load_acquisition_manifest(
        json.loads(src.acq_path.read_text())
    )
    turns = corpus.verify_and_read_rttms([str(src.rttm_path)], rttm_pins)
    return corpus.materialize_prepare(
        source,
        turns,
        corpus_root=corpus_root,
        audio_dir=src.audio_dir,
        recordings=recording_pins,
    )


# --------------------------------------------------------------------------- #
# happy path + independent oracle
# --------------------------------------------------------------------------- #
def test_materialize_happy_path(tmp_path: Path) -> None:
    src = _build_source(tmp_path)
    root = tmp_path / "corpus"
    result = _run(src, root)
    assert result.source == "ami"
    assert result.recordings == 1
    assert result.turn_clips == 2
    assert result.segments == 1
    # manifest.json validates through load_manifest.
    manifest = corpus.load_manifest(json.loads((root / "manifest.json").read_text()))
    assert len(manifest.clips) == 3
    assert all(c.label == "bona_fide" for c in manifest.clips)
    # receipts exist.
    assert (root / "clip_receipt.jsonl").is_file()
    assert (root / "prepare_receipt.json").is_file()


def test_each_clip_is_the_planned_source_slice(tmp_path: Path) -> None:
    """Independent oracle: every clip payload equals the source sliced at its interval."""
    src = _build_source(tmp_path)
    root = tmp_path / "corpus"
    _run(src, root)
    plan = corpus.build_plan(corpus.ORGANIC_SOURCES["ami"], _plan_turns(src))
    for record in (*plan.turn_clips, *plan.segments):
        start, end = record.interval.start_sample, record.interval.end_sample
        expected = src.payload[start * 2 : end * 2]
        got = corpus.read_canonical_wav_payload(root / record.rel_path)
        assert got == expected, record.clip_id
        # coordinate check: the first sample equals its absolute index (mod 65536).
        first = struct.unpack("<h", got[:2])[0]
        assert first == (start % 65536) - 32768


def test_position_matters_shifted_slice_differs(tmp_path: Path) -> None:
    """A one-sample-shifted slice differs, so a global shift could not pass unnoticed."""
    src = _build_source(tmp_path)
    root = tmp_path / "corpus"
    plan = corpus.build_plan(corpus.ORGANIC_SOURCES["ami"], _plan_turns(src))
    _run(src, root)
    rec = plan.turn_clips[0]
    start, end = rec.interval.start_sample, rec.interval.end_sample
    got = corpus.read_canonical_wav_payload(root / rec.rel_path)
    assert got != src.payload[(start + 1) * 2 : (end + 1) * 2]


def _plan_turns(src: _Source) -> dict[str, tuple[corpus.RttmTurn, ...]]:
    grouped: dict[str, list[corpus.RttmTurn]] = {}
    for turn in corpus.parse_rttm(src.rttm_path.read_text()):
        grouped.setdefault(turn.recording, []).append(turn)
    return {rec: tuple(turns) for rec, turns in grouped.items()}


# --------------------------------------------------------------------------- #
# reproducibility + overlap identity
# --------------------------------------------------------------------------- #
def test_two_runs_are_byte_identical(tmp_path: Path) -> None:
    src = _build_source(tmp_path)
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    _run(src, root_a)
    _run(src, root_b)
    assert (root_a / "manifest.json").read_bytes() == (root_b / "manifest.json").read_bytes()
    for wav in sorted((root_a).rglob("*.wav")):
        rel = wav.relative_to(root_a)
        assert wav.read_bytes() == (root_b / rel).read_bytes()


def test_segment_and_turn_share_overlapping_bytes(tmp_path: Path) -> None:
    """A turn clip and the merged segment slice the same payload, so they agree on overlap."""
    src = _build_source(tmp_path)
    root = tmp_path / "corpus"
    _run(src, root)
    plan = corpus.build_plan(corpus.ORGANIC_SOURCES["ami"], _plan_turns(src))
    seg = plan.segments[0]
    seg_payload = corpus.read_canonical_wav_payload(root / seg.rel_path)
    for turn in plan.turn_clips:
        lo = max(turn.interval.start_sample, seg.interval.start_sample)
        hi = min(turn.interval.end_sample, seg.interval.end_sample)
        if lo >= hi:
            continue
        turn_payload = corpus.read_canonical_wav_payload(root / turn.rel_path)
        t_off = (lo - turn.interval.start_sample) * 2
        s_off = (lo - seg.interval.start_sample) * 2
        length = (hi - lo) * 2
        assert turn_payload[t_off : t_off + length] == seg_payload[s_off : s_off + length]


# --------------------------------------------------------------------------- #
# fail-closed matrix
# --------------------------------------------------------------------------- #
def test_populated_root_is_refused(tmp_path: Path) -> None:
    src = _build_source(tmp_path)
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "existing.txt").write_text("x")
    with pytest.raises(corpus.CorpusError, match="already populated"):
        _run(src, root)


def test_recording_sha_mismatch_fails_closed(tmp_path: Path) -> None:
    src = _build_source(tmp_path)
    acq = json.loads(src.acq_path.read_text())
    acq["recordings"]["rec1"]["sha256"] = "0" * 64
    src.acq_path.write_text(json.dumps(acq))
    with pytest.raises(corpus.CorpusError, match="does not match pinned"):
        _run(src, tmp_path / "corpus")


def test_recording_size_mismatch_fails_closed(tmp_path: Path) -> None:
    src = _build_source(tmp_path)
    acq = json.loads(src.acq_path.read_text())
    acq["recordings"]["rec1"]["size"] = acq["recordings"]["rec1"]["size"] + 1
    src.acq_path.write_text(json.dumps(acq))
    with pytest.raises(corpus.CorpusError, match="does not match pinned"):
        _run(src, tmp_path / "corpus")


def test_unpinned_rttm_fails_closed(tmp_path: Path) -> None:
    src = _build_source(tmp_path)
    acq = json.loads(src.acq_path.read_text())
    acq["rttms"][0]["sha256"] = "1" * 64
    src.acq_path.write_text(json.dumps(acq))
    _, _, rttm_pins = corpus.load_acquisition_manifest(json.loads(src.acq_path.read_text()))
    with pytest.raises(corpus.CorpusError, match="not pinned"):
        corpus.verify_and_read_rttms([str(src.rttm_path)], rttm_pins)


def test_rttm_pin_coverage_gap_fails_closed(tmp_path: Path) -> None:
    src = _build_source(tmp_path)
    acq = json.loads(src.acq_path.read_text())
    acq["rttms"].append({"rel_path": "ghost.rttm", "sha256": "2" * 64, "size": 5})
    src.acq_path.write_text(json.dumps(acq))
    _, _, rttm_pins = corpus.load_acquisition_manifest(json.loads(src.acq_path.read_text()))
    with pytest.raises(corpus.CorpusError, match="never supplied"):
        corpus.verify_and_read_rttms([str(src.rttm_path)], rttm_pins)


def test_rttm_size_mismatch_fails_closed(tmp_path: Path) -> None:
    src = _build_source(tmp_path)
    acq = json.loads(src.acq_path.read_text())
    acq["rttms"][0]["size"] = acq["rttms"][0]["size"] + 1  # sha still matches, size does not
    src.acq_path.write_text(json.dumps(acq))
    _, _, rttm_pins = corpus.load_acquisition_manifest(json.loads(src.acq_path.read_text()))
    with pytest.raises(corpus.CorpusError, match="does not match pinned size"):
        corpus.verify_and_read_rttms([str(src.rttm_path)], rttm_pins)


def test_rttm_supplied_twice_fails_closed(tmp_path: Path) -> None:
    src = _build_source(tmp_path)
    _, _, rttm_pins = corpus.load_acquisition_manifest(json.loads(src.acq_path.read_text()))
    with pytest.raises(corpus.CorpusError, match="supplied twice"):
        corpus.verify_and_read_rttms([str(src.rttm_path), str(src.rttm_path)], rttm_pins)


def test_unpinned_recording_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(corpus.CorpusError, match="not pinned"):
        corpus._read_verified_recording(tmp_path, "ghost", {})


def test_surplus_recording_pin_fails_closed(tmp_path: Path) -> None:
    """A pinned recording the RTTMs never declare must fail closed (coverage both ways)."""
    src = _build_source(tmp_path)
    acq = json.loads(src.acq_path.read_text())
    acq["recordings"]["ghost"] = {"rel_path": "ghost.wav", "sha256": "a" * 64, "size": 10}
    src.acq_path.write_text(json.dumps(acq))
    with pytest.raises(corpus.CorpusError, match="pin coverage mismatch"):
        _run(src, tmp_path / "corpus")


def test_duplicate_recording_rel_path_fails_closed() -> None:
    with pytest.raises(corpus.CorpusError, match="share rel_path"):
        corpus.load_acquisition_manifest(
            {
                "source": "ami",
                "recordings": {
                    "a": {"rel_path": "same.wav", "sha256": "a" * 64, "size": 1},
                    "b": {"rel_path": "same.wav", "sha256": "b" * 64, "size": 1},
                },
                "rttms": [{"rel_path": "r.rttm", "sha256": "c" * 64, "size": 1}],
            }
        )


def test_recording_substituted_after_hash_is_caught(tmp_path: Path) -> None:
    """Open-once: the bytes hashed for the pin are the bytes decoded (no reopen race).

    A same-size canonical WAV whose bytes differ from the pin must be rejected, since
    the pin sha is recomputed from the exact bytes that are decoded.
    """
    src = _build_source(tmp_path)
    # Flip one sample and rewrite the staged recording: same byte length, different
    # content, so its whole-file sha no longer matches the pin. Materialization must
    # abort rather than slice these unpinned bytes.
    rec_path = src.audio_dir / "rec1.wav"
    payload = bytearray(corpus.read_canonical_wav_payload(rec_path))
    payload[0] ^= 0xFF
    corpus.write_canonical_wav(rec_path, bytes(payload))
    with pytest.raises(corpus.CorpusError, match="does not match pinned"):
        _run(src, tmp_path / "corpus")


def test_interval_out_of_range_is_plan_drift(tmp_path: Path) -> None:
    """_materialize_record fails closed when a plan interval exceeds the recording."""
    payload = _coord_payload(1000)  # 1000 samples
    rec = corpus.IngestRecord(
        clip_id="ami-rec1-S-turn-0-2000",
        rel_path="ami/turn/clip.wav",
        source="ami",
        recording="rec1",
        speaker_id="ami-rec1-S",
        label="bona_fide",
        language="en",
        license_spdx="CC-BY-4.0",
        stratum="bona_fide|organic|meetingroom",
        interval=corpus.SampleInterval(0, 2000),  # past the 1000-sample recording
        split="eval",
        acquire={},
        kind="turn",
    )
    with pytest.raises(corpus.CorpusError, match="out of range"):
        corpus._materialize_record(rec, payload, tmp_path)


# --------------------------------------------------------------------------- #
# acquisition manifest shape
# --------------------------------------------------------------------------- #
def test_acquisition_manifest_rejects_bad_shape() -> None:
    with pytest.raises(corpus.CorpusError, match="must be a JSON object"):
        corpus.load_acquisition_manifest([])
    with pytest.raises(corpus.CorpusError, match="'recordings'"):
        corpus.load_acquisition_manifest({"source": "ami", "rttms": [{}]})
    with pytest.raises(corpus.CorpusError, match="sha256"):
        corpus.load_acquisition_manifest(
            {
                "source": "ami",
                "recordings": {"r": {"rel_path": "r.wav", "sha256": "nope", "size": 1}},
                "rttms": [{"rel_path": "r.rttm", "sha256": "a" * 64, "size": 1}],
            }
        )


def test_acquisition_manifest_rejects_traversal() -> None:
    with pytest.raises(corpus.CorpusError, match="contain no"):
        corpus.load_acquisition_manifest(
            {
                "source": "ami",
                "recordings": {"r": {"rel_path": "../escape.wav", "sha256": "a" * 64, "size": 1}},
                "rttms": [{"rel_path": "r.rttm", "sha256": "b" * 64, "size": 1}],
            }
        )


def test_acquisition_manifest_rejects_duplicate_rttm_sha() -> None:
    with pytest.raises(corpus.CorpusError, match="duplicate rttm"):
        corpus.load_acquisition_manifest(
            {
                "source": "ami",
                "recordings": {"r": {"rel_path": "r.wav", "sha256": "a" * 64, "size": 1}},
                "rttms": [
                    {"rel_path": "one.rttm", "sha256": "c" * 64, "size": 1},
                    {"rel_path": "two.rttm", "sha256": "c" * 64, "size": 1},
                ],
            }
        )


# --------------------------------------------------------------------------- #
# CLI: dry-run unchanged + equivalence, execution wiring
# --------------------------------------------------------------------------- #
def test_cli_dry_run_matches_build_plan(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src = _build_source(tmp_path)
    rc = corpus.main(["prepare", "--source", "ami", "--rttm", str(src.rttm_path)])
    assert rc == 0
    printed = json.loads(capsys.readouterr().out)
    plan = corpus.build_plan(corpus.ORGANIC_SOURCES["ami"], _plan_turns(src))
    assert printed == corpus.plan_to_dict(plan)


def test_cli_execute_requires_audio_dir_and_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = _build_source(tmp_path)
    rc = corpus.main(
        [
            "prepare",
            "--source", "ami",
            "--rttm", str(src.rttm_path),
            "--corpus-root", str(tmp_path / "c"),
        ]
    )
    assert rc == 2
    assert "requires --audio-dir" in capsys.readouterr().err


def test_cli_execute_happy(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src = _build_source(tmp_path)
    root = tmp_path / "corpus"
    rc = corpus.main(
        [
            "prepare",
            "--source", "ami",
            "--rttm", str(src.rttm_path),
            "--corpus-root", str(root),
            "--audio-dir", str(src.audio_dir),
            "--acquisition-manifest", str(src.acq_path),
        ]
    )
    assert rc == 0
    result = json.loads(capsys.readouterr().out)
    assert result["turn_clips"] == 2
    assert result["segments"] == 1
    assert (root / "manifest.json").is_file()


def test_cli_execute_source_mismatch_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = _build_source(tmp_path)
    acq = json.loads(src.acq_path.read_text())
    acq["source"] = "voxconverse"
    src.acq_path.write_text(json.dumps(acq))
    rc = corpus.main(
        [
            "prepare",
            "--source", "ami",
            "--rttm", str(src.rttm_path),
            "--corpus-root", str(tmp_path / "corpus"),
            "--audio-dir", str(src.audio_dir),
            "--acquisition-manifest", str(src.acq_path),
        ]
    )
    assert rc == 2
    assert "does not match" in capsys.readouterr().err
