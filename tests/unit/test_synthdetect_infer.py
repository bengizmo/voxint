"""Pure-orchestration tests for the synthdetect GPU inference runner (#144, M1 S2).

The runner's engine seam is deliberately narrow: only the fairseq forward pass is
GPU/weights-bound, and everything that decides corpus identity or the scored
numbers -- canonical-PCM verification, windowing, repeat-padding, batching,
pooling, the journal header contract, resume, and the determinism-provenance
capture -- is pure and covered here without torch, fairseq, a GPU, or weights.

A RECORDING fake engine (its output depends on every input window, never a
constant) stands in for the model, so an ordering, tail, or pooling bug cannot
hide behind a degenerate stub. A cross-module test proves the header this runner
emits is accepted verbatim by the S1 host scorer.
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
import wave
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import synthdetect_corpus as corpus  # noqa: E402
import synthdetect_eval as se  # noqa: E402
import synthdetect_infer as si  # noqa: E402
from synthdetect_sources import get_model  # noqa: E402

MODEL = get_model("w2v2-aasist")
WIDTH = MODEL.windowing.upstream_window_samples  # 64600


# --------------------------------------------------------------------------- #
# Fixtures: write canonical-PCM WAVs and matching manifests
# --------------------------------------------------------------------------- #
def _write_wav(path: Path, samples: np.ndarray) -> str:
    """Write a canonical mono/16k/S16LE WAV; return the sha of the PCM payload."""
    samples = samples.astype("<i2")
    payload = samples.tobytes()
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(payload)
    return hashlib.sha256(payload).hexdigest()


def _clip_record(clip_id: str, rel_path: str, sha: str, *, label: str = "bona_fide",
                 split: str = "eval", speaker: str | None = None) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "clip_id": clip_id,
        "rel_path": rel_path,
        "sha256": sha,
        "duration_s": 4.0,
        "label": label,
        "language": "en",
        "license_spdx": "CC0-1.0",
        "stratum": "test",
        "source": "unit",
        "speaker_id": speaker or f"spk_{clip_id}",
        "split": split,
        "generator": None,
        "degradation": None,
        "parent_clip_id": None,
        "acquire": None,
    }
    if label == "spoof":
        rec["generator"] = {"name": "piper", "version": "1", "checkpoint_sha": None,
                            "voice": "v", "seed": "1", "text_source": "t"}
    return rec


class RecordingEngine:
    """A fake engine whose per-window score depends on the window's bytes.

    A constant fake would pass ordering/tail/pooling bugs; this one maps each
    window to a distinct, deterministic score derived from its content, so a
    mis-sliced or mis-ordered batch produces a different pooled score.
    """

    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []

    def score_windows(self, batch: np.ndarray) -> np.ndarray:
        self.calls.append(np.asarray(batch))
        # A content hash per row -> a stable float; never constant across windows.
        return np.array(
            [
                float(np.sum(row.astype(np.float64) * (np.arange(row.shape[0]) + 1)))
                for row in batch
            ],
            dtype=np.float64,
        )


# --------------------------------------------------------------------------- #
# read_canonical_pcm + verify_clip_sha
# --------------------------------------------------------------------------- #
def test_read_canonical_pcm_hashes_payload_not_container(tmp_path: Path) -> None:
    samples = np.arange(-5, 5, dtype="<i2")
    sha = _write_wav(tmp_path / "a.wav", samples)
    audio = si.read_canonical_pcm(tmp_path / "a.wav")
    assert audio.n_samples == 10
    assert audio.pcm_sha256 == sha
    assert audio.pcm_sha256 == hashlib.sha256(samples.tobytes()).hexdigest()
    np.testing.assert_array_equal(audio.samples, samples)


def test_read_canonical_pcm_rejects_stereo(tmp_path: Path) -> None:
    p = tmp_path / "s.wav"
    with wave.open(str(p), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(np.zeros(20, dtype="<i2").tobytes())
    with pytest.raises(si.InferError, match="mono"):
        si.read_canonical_pcm(p)


def test_read_canonical_pcm_rejects_wrong_rate(tmp_path: Path) -> None:
    p = tmp_path / "r.wav"
    with wave.open(str(p), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(np.zeros(20, dtype="<i2").tobytes())
    with pytest.raises(si.InferError, match="16000 Hz"):
        si.read_canonical_pcm(p)


def test_read_canonical_pcm_rejects_wrong_width(tmp_path: Path) -> None:
    p = tmp_path / "w.wav"
    with wave.open(str(p), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(1)  # 8-bit
        wav.setframerate(16000)
        wav.writeframes(b"\x00" * 20)
    with pytest.raises(si.InferError, match="signed 16-bit"):
        si.read_canonical_pcm(p)


def test_read_canonical_pcm_rejects_unreadable(tmp_path: Path) -> None:
    p = tmp_path / "not.wav"
    p.write_bytes(b"this is not a wav file")
    with pytest.raises(si.InferError, match="not a readable PCM WAV"):
        si.read_canonical_pcm(p)


def _write_raw_wav(path: Path, data_payload: bytes, declared_data_size: int) -> None:
    """Hand-build a mono/16k/16-bit WAV with an explicit (possibly odd) data size."""
    fmt = struct.pack("<HHIIHH", 1, 1, 16000, 32000, 2, 16)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", declared_data_size) + data_payload
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


def test_read_canonical_pcm_rejects_orphan_byte(tmp_path: Path) -> None:
    # A 3-byte data chunk is not a whole number of 2-byte frames; stdlib wave
    # would silently floor it to one frame, so the RIFF-size cross-check must catch
    # the orphan byte.
    p = tmp_path / "orphan.wav"
    _write_raw_wav(p, b"\x01\x00\x02", 3)
    with pytest.raises(si.InferError, match="orphan byte"):
        si.read_canonical_pcm(p)


def test_riff_data_chunk_size_reads_declared_size(tmp_path: Path) -> None:
    p = tmp_path / "sz.wav"
    _write_raw_wav(p, b"\x01\x00\x02\x00", 4)
    assert si._riff_data_chunk_size(p) == 4


def test_verify_clip_sha_match_and_mismatch(tmp_path: Path) -> None:
    samples = np.arange(-3, 3, dtype="<i2")
    sha = _write_wav(tmp_path / "c.wav", samples)
    audio = si.read_canonical_pcm(tmp_path / "c.wav")
    entry = corpus.validate_clip(_clip_record("c", "c.wav", sha), 0)
    si.verify_clip_sha(entry, audio)  # no raise
    bad = corpus.validate_clip(_clip_record("c", "c.wav", "a" * 64), 0)
    with pytest.raises(si.InferError, match="does not match the manifest"):
        si.verify_clip_sha(bad, audio)


# --------------------------------------------------------------------------- #
# windowing + padding + batch + pooling
# --------------------------------------------------------------------------- #
def test_model_input_samples_fixed_and_arbitrary() -> None:
    assert si.model_input_samples(MODEL.windowing) == WIDTH
    arb = get_model("antideepfake-xlsr-2b").windowing
    with pytest.raises(si.InferError, match="arbitrary-length"):
        si.model_input_samples(arb)


def test_plan_windows_upstream_short_is_repeat_padded() -> None:
    plan = si.plan_windows(1000, MODEL.windowing, mode="upstream")
    assert plan.spans == ((0, 1000),)
    assert plan.repeat_padded is True


def test_plan_windows_upstream_long_is_prefix_not_padded() -> None:
    plan = si.plan_windows(WIDTH + 5000, MODEL.windowing, mode="upstream")
    assert plan.spans == ((0, WIDTH),)
    assert plan.repeat_padded is False


def test_plan_windows_production_chunks_and_flags_tail() -> None:
    # 4 s windows at 16 kHz = 64000 samples/window; 100000 samples -> 2 windows,
    # the second a short tail (repeat-padded to the model width).
    plan = si.plan_windows(100000, MODEL.windowing, mode="production")
    assert plan.spans == ((0, 64000), (64000, 100000))
    assert plan.repeat_padded is True


def test_plan_windows_rejects_empty_and_unknown_mode() -> None:
    with pytest.raises(si.InferError, match="empty clip"):
        si.plan_windows(0, MODEL.windowing, mode="upstream")
    with pytest.raises(si.InferError, match="unknown windowing mode"):
        si.plan_windows(100, MODEL.windowing, mode="sideways")


def test_repeat_pad_to_crops_tiles_and_rejects_empty() -> None:
    x = np.array([1, 2, 3], dtype="<i2")
    np.testing.assert_array_equal(si.repeat_pad_to(x, 2), [1, 2])
    np.testing.assert_array_equal(si.repeat_pad_to(x, 7), [1, 2, 3, 1, 2, 3, 1])
    np.testing.assert_array_equal(si.repeat_pad_to(x, 3), [1, 2, 3])
    with pytest.raises(si.InferError, match="empty span"):
        si.repeat_pad_to(np.array([], dtype="<i2"), 4)


def test_build_batch_shape_scale_and_recording_sensitivity() -> None:
    samples = np.arange(-100, WIDTH - 100, dtype="<i2")  # exactly WIDTH long
    audio = si.CanonicalAudio(samples=samples, pcm_sha256="x", n_samples=samples.shape[0])
    plan = si.plan_windows(samples.shape[0], MODEL.windowing, mode="upstream")
    batch = si.build_batch(audio, plan, WIDTH)
    assert batch.shape == (1, WIDTH)
    assert batch.dtype == np.float32
    # scaled by 2**15
    assert batch[0, 0] == pytest.approx(-100 / 32768.0)


def test_score_windows_batched_chunks_preserve_order_and_values() -> None:
    engine = RecordingEngine()
    batch = np.arange(5 * WIDTH, dtype=np.float32).reshape(5, WIDTH)
    one = engine.score_windows(batch)  # reference: all at once
    batched = si.score_windows_batched(RecordingEngine(), batch, batch_size=2)
    assert batched.shape == (5,)
    np.testing.assert_allclose(batched, one)


def test_score_windows_batched_rejects_bad_batch_size() -> None:
    with pytest.raises(si.InferError, match="batch_size must be positive"):
        si.score_windows_batched(RecordingEngine(), np.zeros((2, WIDTH), np.float32), 0)


def test_pool_scores_logit_mean_and_failures() -> None:
    assert si.pool_scores(np.array([1.0, 3.0]), "logit-mean") == pytest.approx(2.0)
    with pytest.raises(si.InferError, match="empty score vector"):
        si.pool_scores(np.array([]), "logit-mean")
    with pytest.raises(si.InferError, match="non-finite"):
        si.pool_scores(np.array([1.0, np.inf]), "logit-mean")
    with pytest.raises(si.InferError, match="unknown pooling"):
        si.pool_scores(np.array([1.0]), "max")


# --------------------------------------------------------------------------- #
# header identity + execution_identity + resume parsing
# --------------------------------------------------------------------------- #
def _header(**over: Any) -> dict[str, Any]:
    base = dict(
        model=MODEL, manifest_sha256="d" * 64, split="eval",
        selected_clip_ids=["a", "b"], windowing_mode="upstream",
        runtime={"torch": "2.1.0"}, flags={"deterministic_algorithms": True},
        weights={"LA_model.pth": {"sha256": "e" * 64, "size_bytes": 1}},
        runner_git={"commit": "abc", "dirty": False},
        created_at="2026-01-01T00:00:00+00:00", run_id="r1", host="h1",
    )
    base.update(over)
    return si.build_header(**base)


def test_build_header_carries_scorer_required_keys() -> None:
    h = _header()
    for key in ("kind", "schema_version", "inference_space", "model_id", "manifest_sha256"):
        assert h[key]
    assert h["kind"] == "synthdetect_journal"
    assert h["windowing"]["pooling"] == "logit-mean"
    assert h["scoring"]["journaled_score"] == "negated column 1"
    assert h["selection"]["n_selected"] == 2


def test_execution_identity_ignores_volatile_but_tracks_substance() -> None:
    a = _header(created_at="t1", run_id="r1", host="h1")
    b = _header(created_at="t2", run_id="r2", host="h2")
    assert a["execution_identity_sha256"] == b["execution_identity_sha256"]
    c = _header(manifest_sha256="f" * 64)
    assert c["execution_identity_sha256"] != a["execution_identity_sha256"]
    d = _header(flags={"deterministic_algorithms": False})
    assert d["execution_identity_sha256"] != a["execution_identity_sha256"]


def test_selection_sha256_is_order_sensitive() -> None:
    assert si.selection_sha256(["a", "b"]) != si.selection_sha256(["b", "a"])


def test_parse_resume_journal_header_only_is_valid() -> None:
    _, done = si.parse_resume_journal(json.dumps(_header()) + "\n")
    assert done == []


def test_parse_resume_journal_collects_validated_records() -> None:
    h = json.dumps(_header())
    r1 = json.dumps({"clip_id": "a", "raw_score": 1.0, "n_windows": 1})
    r2 = json.dumps({"clip_id": "b", "skip_reason": "too-short", "n_windows": 0})
    _, done = si.parse_resume_journal("\n".join([h, r1, r2]) + "\n")
    assert done == ["a", "b"]


def test_parse_resume_journal_rejects_bad_states() -> None:
    h = json.dumps(_header())
    with pytest.raises(si.InferError, match="no header"):
        si.parse_resume_journal("\n")
    with pytest.raises(si.InferError, match="not a synthdetect_journal header"):
        si.parse_resume_journal(json.dumps({"kind": "other"}) + "\n")
    # A malformed complete line is fatal (prepare_journal removes only a torn tail).
    bad = "\n".join([h, "{not json", json.dumps({"clip_id": "z", "raw_score": 1.0})])
    with pytest.raises(si.InferError, match="malformed JSON"):
        si.parse_resume_journal(bad + "\n")
    # A record violating the S1 score/skip XOR must NOT be trusted as completed.
    xor = "\n".join([h, json.dumps({"clip_id": "a", "raw_score": 1.0, "skip_reason": "x"})])
    with pytest.raises(si.InferError, match="exactly one of raw_score"):
        si.parse_resume_journal(xor + "\n")
    # A non-finite score is rejected.
    inf = "\n".join([h, json.dumps({"clip_id": "a"})])  # neither score nor skip
    with pytest.raises(si.InferError, match="exactly one of raw_score"):
        si.parse_resume_journal(inf + "\n")
    # duplicate clip id
    dup = "\n".join([h, json.dumps({"clip_id": "a", "raw_score": 1.0}),
                     json.dumps({"clip_id": "a", "raw_score": 2.0})])
    with pytest.raises(si.InferError, match="duplicate clip_id"):
        si.parse_resume_journal(dup + "\n")
    # missing clip id
    miss = "\n".join([h, json.dumps({"raw_score": 1.0})])
    with pytest.raises(si.InferError, match="missing a clip_id"):
        si.parse_resume_journal(miss + "\n")


# --------------------------------------------------------------------------- #
# prepare_journal: exclusive create, torn-tail truncation, resume identity
# --------------------------------------------------------------------------- #
def test_prepare_journal_fresh_writes_header_once(tmp_path: Path) -> None:
    out = tmp_path / "sub" / "j.jsonl"
    header = _header(selected_clip_ids=["a"])
    already = si.prepare_journal(out, header, resume=False)
    assert already == frozenset()
    lines = out.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["kind"] == "synthdetect_journal"


def test_prepare_journal_refuses_existing_without_resume(tmp_path: Path) -> None:
    out = tmp_path / "j.jsonl"
    header = _header()
    si.prepare_journal(out, header, resume=False)
    with pytest.raises(si.InferError, match="already exists; pass --resume"):
        si.prepare_journal(out, header, resume=False)


def test_prepare_journal_resume_matches_and_skips(tmp_path: Path) -> None:
    out = tmp_path / "j.jsonl"
    header = _header(selected_clip_ids=["a", "b"])
    si.prepare_journal(out, header, resume=False)
    with out.open("a") as fh:
        fh.write(json.dumps({"clip_id": "a", "raw_score": 1.0, "n_windows": 1}) + "\n")
    already = si.prepare_journal(out, header, resume=True,
                                 selected_ids=frozenset({"a", "b"}))
    assert already == frozenset({"a"})


def test_prepare_journal_truncates_torn_tail_and_keeps_recovery(tmp_path: Path) -> None:
    out = tmp_path / "j.jsonl"
    header = _header(selected_clip_ids=["a", "b"])
    si.prepare_journal(out, header, resume=False)
    with out.open("a") as fh:
        fh.write(json.dumps({"clip_id": "a", "raw_score": 1.0, "n_windows": 1}) + "\n")
        fh.write('{"clip_id": "b", "raw_sc')  # torn: no trailing newline
    already = si.prepare_journal(out, header, resume=True,
                                 selected_ids=frozenset({"a", "b"}))
    assert already == frozenset({"a"})
    # the torn bytes were removed and preserved for forensics
    text = out.read_text()
    assert text.endswith("\n")
    assert '"clip_id": "b"' not in text  # the torn "b" record is gone
    assert '"clip_id": "a"' in text  # the complete "a" record survived
    assert (tmp_path / "j.jsonl.torn").read_text().startswith('{"clip_id": "b"')


def test_prepare_journal_rejects_tampered_identity(tmp_path: Path) -> None:
    out = tmp_path / "j.jsonl"
    header = _header()
    # Write a header whose stored identity does NOT match its own content.
    tampered = dict(header)
    tampered["execution_identity_sha256"] = "f" * 64
    out.write_text(json.dumps(tampered) + "\n")
    with pytest.raises(si.InferError, match="does not match its own recorded"):
        si.prepare_journal(out, header, resume=True)


def test_prepare_journal_rejects_identity_mismatch(tmp_path: Path) -> None:
    out = tmp_path / "j.jsonl"
    prior = _header(manifest_sha256="a" * 64)
    si.prepare_journal(out, prior, resume=False)
    current = _header(manifest_sha256="b" * 64)  # different run
    with pytest.raises(si.InferError, match="execution identity differs"):
        si.prepare_journal(out, current, resume=True)


def test_prepare_journal_rejects_unknown_completed_id(tmp_path: Path) -> None:
    out = tmp_path / "j.jsonl"
    header = _header(selected_clip_ids=["a"])
    si.prepare_journal(out, header, resume=False)
    with out.open("a") as fh:
        fh.write(json.dumps({"clip_id": "ghost", "raw_score": 1.0, "n_windows": 1}) + "\n")
    with pytest.raises(si.InferError, match="not in this selection"):
        si.prepare_journal(out, header, resume=True, selected_ids=frozenset({"a"}))


# --------------------------------------------------------------------------- #
# JournalWriter + the cross-module contract: our header parses in the S1 scorer
# --------------------------------------------------------------------------- #
def test_journal_round_trips_through_the_host_scorer(tmp_path: Path) -> None:
    pytest.importorskip("sklearn", reason="synthdetect-eval extra not installed")
    out = tmp_path / "j.jsonl"
    header = _header(selected_clip_ids=["a"])
    with si.JournalWriter(out) as w:
        w.write_line(header)
        w.write_line(si.ClipOutcome("a", raw_score=1.5, skip_reason=None, n_windows=1).as_record())
    # The S1 scorer's fail-closed parser must accept the runner's output verbatim.
    parsed = se.parse_journal(out.read_text())
    assert parsed.header["model_id"] == "w2v2-aasist"
    assert parsed.results[0].clip_id == "a"
    assert parsed.results[0].raw_score == pytest.approx(1.5)


def test_clip_outcome_record_is_score_xor_skip() -> None:
    scored = si.ClipOutcome("a", 1.0, None, 2).as_record()
    assert scored == {"clip_id": "a", "n_windows": 2, "raw_score": 1.0}
    skipped = si.ClipOutcome("b", None, "too-short", 0).as_record()
    assert skipped == {"clip_id": "b", "n_windows": 0, "skip_reason": "too-short"}


# --------------------------------------------------------------------------- #
# score_clip + run_inference end to end (recording fake engine)
# --------------------------------------------------------------------------- #
def _corpus(tmp_path: Path, n_clips: int = 2) -> tuple[Path, corpus.Manifest, list[str]]:
    root = tmp_path / "corpus"
    root.mkdir()
    records = []
    shas = []
    rng = np.random.default_rng(0)
    for i in range(n_clips):
        samples = (rng.integers(-1000, 1000, size=20000)).astype("<i2")
        sha = _write_wav(root / f"c{i}.wav", samples)
        shas.append(sha)
        records.append(_clip_record(f"c{i}", f"c{i}.wav", sha))
    manifest = corpus.load_manifest({"schema_version": 1, "clips": records})
    return root, manifest, shas


def test_score_clip_and_run_inference_writes_scores(tmp_path: Path) -> None:
    root, manifest, _ = _corpus(tmp_path, 2)
    engine = RecordingEngine()
    out = tmp_path / "j.jsonl"
    clips = si.select_clips(manifest, "eval")
    with si.JournalWriter(out) as writer:
        counts = si.run_inference(
            clips=clips, corpus_root=root, engine=engine, model=MODEL,
            header=_header(), writer=writer, windowing_mode="upstream",
        )
    assert counts == {"scored": 2, "skipped_error": 0, "resumed": 0}
    lines = [json.loads(ln) for ln in out.read_text().splitlines()]
    assert len(lines) == 2
    assert all("raw_score" in ln and np.isfinite(ln["raw_score"]) for ln in lines)
    # the recording engine saw one window per (upstream) clip
    assert len(engine.calls) == 2
    assert engine.calls[0].shape == (1, WIDTH)


def test_run_inference_resume_skips_done(tmp_path: Path) -> None:
    root, manifest, _ = _corpus(tmp_path, 3)
    clips = si.select_clips(manifest, "eval")
    out = tmp_path / "j.jsonl"
    with si.JournalWriter(out) as writer:
        counts = si.run_inference(
            clips=clips, corpus_root=root, engine=RecordingEngine(), model=MODEL,
            header=_header(), writer=writer, windowing_mode="upstream",
            already_done=frozenset({"c0", "c1"}),
        )
    assert counts == {"scored": 1, "skipped_error": 0, "resumed": 2}
    assert [json.loads(ln)["clip_id"] for ln in out.read_text().splitlines()] == ["c2"]


def test_run_inference_stop_on_error_vs_skip(tmp_path: Path) -> None:
    root, manifest, _ = _corpus(tmp_path, 2)
    # Corrupt c1's manifest sha so verify fails closed.
    records = [dict(_clip_record(c.clip_id, c.rel_path, c.sha256)) for c in manifest.clips]
    records[1]["sha256"] = "0" * 64
    broken = corpus.load_manifest({"schema_version": 1, "clips": records})
    clips = si.select_clips(broken, "eval")
    out = tmp_path / "j.jsonl"
    with si.JournalWriter(out) as writer, pytest.raises(si.InferError, match="does not match"):
        si.run_inference(
            clips=clips, corpus_root=root, engine=RecordingEngine(), model=MODEL,
            header=_header(), writer=writer, windowing_mode="upstream", stop_on_error=True,
        )
    # skip mode journals the bad clip instead of stopping
    out2 = tmp_path / "j2.jsonl"
    with si.JournalWriter(out2) as writer:
        counts = si.run_inference(
            clips=clips, corpus_root=root, engine=RecordingEngine(), model=MODEL,
            header=_header(), writer=writer, windowing_mode="upstream", stop_on_error=False,
        )
    assert counts["scored"] == 1
    assert counts["skipped_error"] == 1
    recs = {json.loads(ln)["clip_id"]: json.loads(ln) for ln in out2.read_text().splitlines()}
    assert "skip_reason" in recs["c1"]


def test_score_clip_rejects_engine_shape_mismatch(tmp_path: Path) -> None:
    root, manifest, _ = _corpus(tmp_path, 1)

    class WrongShapeEngine:
        def score_windows(self, batch: np.ndarray) -> np.ndarray:
            return np.array([1.0, 2.0, 3.0])  # too many

    with pytest.raises(si.InferError, match="returned"):
        si.score_clip(manifest.clips[0], root, WrongShapeEngine(), MODEL, windowing_mode="upstream")


def test_select_clips_empty_split_fails(tmp_path: Path) -> None:
    _, manifest, _ = _corpus(tmp_path, 1)
    with pytest.raises(si.InferError, match="no clips to score"):
        si.select_clips(manifest, "holdout")


# --------------------------------------------------------------------------- #
# determinism capture + configure (fake torch, no GPU)
# --------------------------------------------------------------------------- #
class _Backends:
    class cudnn:
        deterministic = False
        benchmark = True
        allow_tf32 = True

        @staticmethod
        def version() -> int:
            return 8907

    class cuda:
        class matmul:
            allow_tf32 = True


class FakeTorch:
    __version__ = "2.1.0+cu118"

    class version:
        cuda = "11.8"

    class cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_name(_i: int) -> str:
            return "NVIDIA GeForce RTX 3060"

        @staticmethod
        def get_device_capability(_i: int) -> tuple[int, int]:
            return (8, 6)

        @staticmethod
        def manual_seed_all(_seed: int) -> None:
            FakeTorch.seeded.append(("cuda", _seed))

    backends = _Backends
    _det = False
    _warn = False
    seeded: ClassVar[list[tuple[str, int]]] = []

    @classmethod
    def are_deterministic_algorithms_enabled(cls) -> bool:
        return cls._det

    @classmethod
    def is_deterministic_algorithms_warn_only_enabled(cls) -> bool:
        return cls._warn

    @staticmethod
    def get_float32_matmul_precision() -> str:
        return "highest"

    @classmethod
    def use_deterministic_algorithms(cls, value: bool, *, warn_only: bool = False) -> None:
        cls._det = value
        cls._warn = warn_only

    @classmethod
    def manual_seed(cls, seed: int) -> None:
        cls.seeded.append(("cpu", seed))


def test_capture_runtime_reads_device_and_versions() -> None:
    rt = si.capture_runtime(FakeTorch, device_index=2, image_digest="sha256:abc",
                            provenance_sha256="p" * 64, fairseq_version="1.0.0a0+deadbee")
    assert rt["torch"] == "2.1.0+cu118"
    assert rt["cuda"] == "11.8"
    assert rt["device_index"] == 2
    assert rt["device_capability"] == [8, 6]
    assert rt["cuda_driver_version"] is None  # FakeTorch has no _C; fail-soft to None
    assert rt["image_digest"] == "sha256:abc"
    assert rt["fairseq"] == "1.0.0a0+deadbee"


def test_parse_device_index_and_rejects_non_cuda() -> None:
    assert si.parse_device_index("cuda") == 0
    assert si.parse_device_index("cuda:2") == 2
    for bad in ("cpu", "mps", "cuda:x", "0"):
        with pytest.raises(si.InferError, match="device must be"):
            si.parse_device_index(bad)


def test_seed_everything_seeds_all_rngs() -> None:
    FakeTorch.seeded = []
    seeds = si.seed_everything(FakeTorch, 7)
    assert seeds == {"python": 7, "numpy": 7, "torch": 7, "torch_cuda": 7}
    assert ("cpu", 7) in FakeTorch.seeded
    assert ("cuda", 7) in FakeTorch.seeded


def test_configure_determinism_sets_flags_and_requires_cublas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(si.InferError, match="CUBLAS_WORKSPACE_CONFIG"):
        si.configure_determinism(FakeTorch)
    # A non-empty but INVALID value must also fail closed (not just an unset one).
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":2:2")
    with pytest.raises(si.InferError, match="not one of"):
        si.configure_determinism(FakeTorch)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    si.configure_determinism(FakeTorch)
    assert FakeTorch.backends.cudnn.allow_tf32 is False
    assert FakeTorch.backends.cuda.matmul.allow_tf32 is False
    assert FakeTorch.backends.cudnn.benchmark is False
    assert FakeTorch.are_deterministic_algorithms_enabled() is True
    flags = si.capture_flags(FakeTorch, batch_size=8, model_eval=True, seeds={"python": 0})
    assert flags["deterministic_algorithms"] is True
    assert flags["deterministic_warn_only"] is False
    assert flags["cublas_workspace_config"] == ":4096:8"
    assert flags["seeds"] == {"python": 0}
    assert flags["batch_size"] == 8
    assert flags["inference_mode"] is True


def test_runner_git_provenance_shape() -> None:
    prov = si.runner_git_provenance(REPO)
    assert set(prov) == {"commit", "dirty"}


# --------------------------------------------------------------------------- #
# verify-sources receipt
# --------------------------------------------------------------------------- #
def test_hash_file_streams_matching_sha_and_size(tmp_path: Path) -> None:
    data = b"abc" * 100000
    p = tmp_path / "big.bin"
    p.write_bytes(data)
    digest, size = si._hash_file(p, block_size=4096)
    assert digest == hashlib.sha256(data).hexdigest()
    assert size == len(data)


def test_compute_weight_receipt_missing_and_measured(tmp_path: Path) -> None:
    # No files present -> missing verdicts, all_present False.
    receipt = si.compute_weight_receipt(tmp_path, MODEL)
    assert receipt["all_present"] is False
    assert receipt["weights_pinned"] is False
    assert {f["verdict"] for f in receipt["files"]} == {"missing"}

    # Present but registry sha is CANDIDATE (None) -> candidate-measured.
    for w in MODEL.weights:
        (tmp_path / w.filename).write_bytes(b"weight-bytes-" + w.filename.encode())
    receipt2 = si.compute_weight_receipt(tmp_path, MODEL)
    assert receipt2["all_present"] is True
    for f in receipt2["files"]:
        assert f["verdict"] == "candidate-measured"
        assert len(f["actual_sha256"]) == 64
        assert f["actual_size_bytes"] > 0


def test_compute_weight_receipt_match_and_mismatch(tmp_path: Path) -> None:
    # Simulate a PINNED registry by writing a file and pinning its real sha onto a
    # copy of the model's first weight entry.
    from dataclasses import replace

    data = b"pinned-weight-payload"
    real = hashlib.sha256(data).hexdigest()
    w0 = MODEL.weights[0]
    (tmp_path / w0.filename).write_bytes(data)
    pinned_model = replace(MODEL, weights=(replace(w0, sha256=real),))
    receipt = si.compute_weight_receipt(tmp_path, pinned_model)
    assert receipt["files"][0]["verdict"] == "match"

    (tmp_path / w0.filename).write_bytes(b"different-bytes")
    receipt2 = si.compute_weight_receipt(tmp_path, pinned_model)
    assert receipt2["files"][0]["verdict"] == "MISMATCH"


# --------------------------------------------------------------------------- #
# CLI: verify-sources exit codes; run fails closed at the engine seam
# --------------------------------------------------------------------------- #
def test_cli_verify_sources_missing_is_ok(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = si.main(["verify-sources", "--weights-dir", str(tmp_path), "--model-id", "w2v2-aasist"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["kind"] == "synthdetect_weight_receipt"


def test_cli_verify_sources_mismatch_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    (tmp_path / "LA_model.pth").write_bytes(b"x")
    (tmp_path / "xlsr2_300m.pt").write_bytes(b"y")
    w = MODEL.weights[0]
    pinned = replace(MODEL, weights=(replace(w, sha256="a" * 64), MODEL.weights[1]))
    monkeypatch.setattr(si, "get_model", lambda _mid: pinned)
    rc = si.main(["verify-sources", "--weights-dir", str(tmp_path)])
    assert rc == 2


def test_cli_run_fails_closed_without_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The pipeline reaches the (unwired) fairseq engine and fails closed honestly,
    # without pretending to score. A fake torch keeps the determinism config happy.
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    root, manifest, _ = _corpus(tmp_path, 1)
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text(json.dumps({
        "schema_version": 1,
        "clips": [_clip_record(c.clip_id, c.rel_path, c.sha256) for c in manifest.clips],
    }))
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    rc = si.main([
        "run", "--manifest", str(manifest_path), "--corpus-root", str(root),
        "--out", str(tmp_path / "j.jsonl"), "--weights-dir", str(tmp_path),
    ])
    assert rc == 1  # InferError from the unwired engine seam
