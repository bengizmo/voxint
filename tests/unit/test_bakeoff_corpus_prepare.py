"""Unit tests for the whisper-bakeoff corpus prepare tool's pure core.

Covers the parsers/selection/serialization that were also exercised against
real AMI + TED bytes in ``tools/prepare_bakeoff_corpus.py verify-sources``
(2026-08-16): STM (incl. ignore-masks), AMI NXT words, AMI meetings channel
map, deterministic selection, and float-free time canonicalization. Synthetic
fixtures mirror the real formats. Network/soundfile paths are not touched here.
"""

from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]


def _load_tool():
    path = REPO / "tools" / "prepare_bakeoff_corpus.py"
    spec = importlib.util.spec_from_file_location("prepare_bakeoff_corpus", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: the module defines dataclasses, whose creation
    # resolves cls.__module__ via sys.modules (as tools/generate_parity_corpus
    # does for its dynamically loaded module).
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


prep = _load_tool()
src = prep.src


class TestMicroseconds:
    def test_exact_two_dp(self) -> None:
        assert prep._us("5.57") == 5_570_000
        assert prep._us("12.5") == 12_500_000
        assert prep._us("0.00") == 0
        assert prep._us(0) == 0

    def test_no_float_drift(self) -> None:
        # 0.07 * 1e6 is 70000.00000000001 as a float; Decimal must land exact.
        assert prep._us("0.07") == 70_000


class TestParseStm:
    STM = "\n".join(
        [
            ";; a header comment",
            "AlGore_2009 1 AlGore_2009 0.00 5.00 <o,f0,male> hello world",
            "AlGore_2009 1 inter_segment_gap 5.00 8.00 <o,,unknown> ignore_time_segment_in_scoring",
            "AlGore_2009 1 AlGore_2009 8.00 12.50 <o,f0,male> more speech here",
            "",
        ]
    )

    def test_segments_and_ignore_masks(self) -> None:
        segs = prep.parse_stm(self.STM)
        assert len(segs) == 3
        assert [s.ignore for s in segs] == [False, True, False]
        assert segs[0].start_us == 0 and segs[0].end_us == 5_000_000
        assert segs[0].text == "hello world"
        # ignore rows carry no scoreable text
        assert segs[1].text == ""
        assert segs[2].start_us == 8_000_000 and segs[2].end_us == 12_500_000

    def test_reference_hash_is_deterministic(self) -> None:
        segs = prep.parse_stm(self.STM)
        a = prep.sha256_hex(prep.canonical_reference_bytes(prep.stm_reference_payload(segs)))
        b = prep.sha256_hex(prep.canonical_reference_bytes(prep.stm_reference_payload(segs)))
        assert a == b and len(a) == 64


class TestParseNxtWords:
    WORDS = (
        b'<nite:root xmlns:nite="http://nite.sourceforge.net/" nite:id="ES.A.words">'
        b'<w nite:id="w0" starttime="5.57" endtime="5.94">Okay</w>'
        b'<w nite:id="w1" starttime="5.94" endtime="5.94" punc="true">.</w>'
        b'<w nite:id="w2" starttime="11.09" endtime="11.25">Does</w>'
        b'<w nite:id="w3">notimes</w>'
        b'<vocalsound nite:id="v0" starttime="1.0" endtime="2.0" type="laugh"/>'
        b"</nite:root>"
    )

    def test_only_timed_words(self) -> None:
        words = prep.parse_nxt_words(self.WORDS)
        # w3 (no times) and the vocalsound are skipped; punctuation w1 kept.
        assert [w.text for w in words] == ["Okay", ".", "Does"]
        assert words[0].start_us == 5_570_000 and words[0].end_us == 5_940_000
        assert words[2].start_us == 11_090_000


class TestParseMeetings:
    XML = (
        b'<meetings xmlns:nite="http://nite.sourceforge.net/">'
        b'<meeting observation="ES2002a">'
        b'<speaker nxt_agent="A" channel="0" global_name="FEE005"/>'
        b'<speaker nxt_agent="B" channel="1"/>'
        b"</meeting>"
        b'<meeting observation="IS1007d">'
        b'<speaker nxt_agent="A" channel="5"/>'
        b"</meeting>"
        b"</meetings>"
    )

    def test_channel_map_is_not_positional(self) -> None:
        mapping = prep.parse_meetings(self.XML)
        assert mapping["ES2002a"] == {"A": 0, "B": 1}
        # IS1007d proves agent A need not be channel 0.
        assert mapping["IS1007d"] == {"A": 5}


class TestSelection:
    def test_deterministic_and_stable(self) -> None:
        ids = [f"talk_{i}" for i in range(19)]
        first = prep.hash_rank(ids)
        assert prep.hash_rank(list(reversed(ids))) == first  # order-independent
        assert prep.select(ids, 15) == first[:15]
        # A superset's top-N is a stable extension, not a reshuffle of the head.
        assert prep.select(ids, 5) == prep.select(ids, 15)[:5]

    def test_count_clamped_to_available(self) -> None:
        assert len(prep.select(["a", "b"], 15)) == 2


class TestSourcesSanity:
    def test_spdx_and_granularity(self) -> None:
        assert src.TEDLIUM3["license_spdx"] == "CC-BY-NC-ND-3.0"
        assert src.AMI["license_spdx"] == "CC-BY-4.0"
        assert src.TEDLIUM3["ts_granularity"] == "segment"
        assert src.AMI["ts_granularity"] == "word"
        assert src.TEDLIUM3["commit_transcripts"] is False  # NC-ND
        assert src.AMI["commit_transcripts"] is True

    def test_boundary_eligibility_rule(self) -> None:
        assert src.boundary_gate_eligible("word") is True
        assert src.boundary_gate_eligible("segment") is False
        assert src.boundary_gate_eligible("none") is False

    def test_pinned_archives_have_hashes_and_sizes(self) -> None:
        for meta in src.TEDLIUM3["archives"].values():
            assert len(meta["sha256"]) == 64
            assert meta["size_bytes"] > 0
        assert len(src.AMI["annotations_sha256"]) == 64

    def test_window_policy_and_eligibility_floor(self) -> None:
        # AMI and TED share the same fixed content-independent window.
        assert src.AMI["slice_offset_s"] == src.TEDLIUM3["slice_offset_s"] == 120.0
        assert src.AMI["slice_window_s"] == src.TEDLIUM3["slice_window_s"] == 240.0
        assert src.AMI_MIN_WINDOW_WORDS == src.AMI["min_window_words"] == 50


# --------------------------------------------------------------------------
# Canonical audio: quantize / write / read (byte-reproducible, little-endian)
# --------------------------------------------------------------------------
class TestCanonicalAudio:
    def test_quantize_is_pinned_and_clips(self) -> None:
        audio = np.array([0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0], dtype=np.float64)
        pcm = prep.quantize_int16(audio)
        assert pcm.dtype == np.dtype("<i2")
        # Clipped to the representable range; +2.0 saturates to 32767, -2.0 to -32768.
        assert pcm[1] == 32767 and pcm[5] == 32767
        assert pcm[2] == -32768 and pcm[6] == -32768
        assert pcm[0] == 0 and pcm[3] == 16384 and pcm[4] == -16384

    def test_write_read_round_trip_and_deterministic_bytes(self, tmp_path: Path) -> None:
        samples = np.array([0, 1, -1, 16384, -16384, 32767, -32768], dtype="<i2")
        a = prep.write_canonical_wav(tmp_path / "a.wav", samples)
        b = prep.write_canonical_wav(tmp_path / "b.wav", samples)
        assert a == b and len(a) == 64  # identical sha, two writes
        assert (tmp_path / "a.wav").read_bytes() == (tmp_path / "b.wav").read_bytes()
        back = prep.read_canonical_wav(tmp_path / "a.wav")
        assert np.array_equal(back, samples)

    def test_bytes_are_little_endian(self, tmp_path: Path) -> None:
        prep.write_canonical_wav(tmp_path / "le.wav", np.array([258], dtype="<i2"))
        # 258 == 0x0102 → little-endian data bytes are 0x02, 0x01.
        assert (tmp_path / "le.wav").read_bytes()[44:46] == b"\x02\x01"


def _canonical_header(data_size: int, *, channels: int = 1, rate: int = 16000,
                      bits: int = 16, fmt: int = 1, data_tag: bytes = b"data") -> bytes:
    block_align = channels * bits // 8
    byte_rate = rate * block_align
    return (
        b"RIFF" + struct.pack("<I", 36 + data_size) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, fmt, channels, rate, byte_rate, block_align, bits)
        + data_tag + struct.pack("<I", data_size)
    )


class TestWavHeader:
    def test_parse_and_validate_canonical(self) -> None:
        info = prep.parse_wav_header(_canonical_header(7_680_000))
        prep.validate_canonical_riff(info)  # no raise
        assert info["sample_rate"] == 16000 and info["channels"] == 1 and info["bits"] == 16

    def test_reject_non_canonical(self) -> None:
        with pytest.raises(SystemExit):
            prep.validate_canonical_riff(prep.parse_wav_header(_canonical_header(100, channels=2)))
        with pytest.raises(SystemExit):
            prep.validate_canonical_riff(prep.parse_wav_header(_canonical_header(100, rate=44100)))
        with pytest.raises(SystemExit):
            prep.validate_canonical_riff(prep.parse_wav_header(_canonical_header(100, fmt=3)))
        # A non-'data' chunk at offset 36 (e.g. a LIST/fact chunk) must abort, since
        # the AMI Range path assumes the data chunk begins at byte 44.
        with pytest.raises(SystemExit):
            prep.parse_wav_header(_canonical_header(100, data_tag=b"LIST"))
        with pytest.raises(SystemExit):
            prep.parse_wav_header(b"RIFF" + b"\x00" * 8)  # too short


# --------------------------------------------------------------------------
# Reference windowing: word gold containment + STM boundary→ignore clipping
# --------------------------------------------------------------------------
class TestWordsInWindow:
    def test_containment_and_rebase(self) -> None:
        offset_us, window_us = 120_000_000, 240_000_000
        words = [
            prep.NxtWord(119_000_000, 121_000_000, "straddle_start"),  # crosses start → drop
            prep.NxtWord(130_000_000, 130_500_000, "inside"),          # fully inside → keep
            prep.NxtWord(359_000_000, 361_000_000, "straddle_end"),    # crosses end → drop
            prep.NxtWord(500_000_000, 500_500_000, "after"),           # outside → drop
        ]
        kept = prep.words_in_window(words, offset_us, window_us)
        assert [w.text for w in kept] == ["inside"]
        # rebased to window-relative µs.
        assert kept[0].start_us == 10_000_000 and kept[0].end_us == 10_500_000


class TestClipStmToWindow:
    def test_boundary_becomes_ignore_and_masks_preserved(self) -> None:
        offset_us, window_us = 120_000_000, 240_000_000
        segs = [
            prep.StmSegment(100_000_000, 130_000_000, "crosses start", False),
            prep.StmSegment(130_000_000, 140_000_000, "fully inside", False),
            prep.StmSegment(135_000_000, 145_000_000, "", True),  # existing ignore mask
            prep.StmSegment(350_000_000, 400_000_000, "crosses end", False),
            prep.StmSegment(500_000_000, 510_000_000, "outside", False),
        ]
        out = prep.clip_stm_to_window(segs, offset_us, window_us)
        assert [(s.text, s.ignore) for s in out] == [
            ("", True),               # crossed start → clipped ignore, text dropped
            ("fully inside", False),  # kept verbatim
            ("", True),               # mask preserved, clipped
            ("", True),               # crossed end → clipped ignore
        ]
        # boundary-crossing segments are clipped to the window and rebased.
        assert out[0].start_us == 0 and out[0].end_us == 10_000_000
        assert out[3].start_us == 230_000_000 and out[3].end_us == 240_000_000


# --------------------------------------------------------------------------
# Synthetic strata determinism (seeded → byte-identical on regeneration)
# --------------------------------------------------------------------------
class TestSyntheticDeterminism:
    def test_silence_pure_zero_and_near_silence(self) -> None:
        assert np.array_equal(prep.gen_silence(0, 5.0, False), np.zeros(80_000, dtype="<i2"))
        near = prep.gen_silence(2, 1.0, True)
        assert near.dtype == np.dtype("<i2") and len(near) == 16_000
        assert np.abs(near).max() <= 2  # noise floor, not speech
        assert np.array_equal(near, prep.gen_silence(2, 1.0, True))  # reproducible

    def test_bait_is_reproducible_and_bounded(self) -> None:
        for i, (_name, kind, _seconds) in enumerate(prep.BAIT_SPECS):
            a = prep.gen_bait(i, kind, 1.0)
            b = prep.gen_bait(i, kind, 1.0)
            assert np.array_equal(a, b) and a.dtype == np.dtype("<i2")
            assert len(a) == 16_000


# --------------------------------------------------------------------------
# Manifest schema invariants (pure — mirrors the committed manifest contract)
# --------------------------------------------------------------------------
def _valid_manifest() -> dict:
    sha = "a" * 64
    files: list[dict] = []
    for i in range(src.STRATA_TARGETS["ami_ihm"]):
        files.append({
            "dataset": "ami_ihm", "upstream_id": f"M{i}.Headset-0", "sha256": sha,
            "duration_s": 240.0, "strata": ["ami_ihm"], "license_spdx": "CC-BY-4.0",
            "transcript_sha256": sha, "ts_granularity": "word",
            "acquire": {"kind": "ami_range", "meeting": f"M{i}", "agent": "A",
                        "channel": 0, "offset_s": 120.0, "window_s": 240.0},
            "gold_file": f"gold/ami/M{i}.A.words.json",
        })
    for i in range(src.STRATA_TARGETS["tedlium3"]):
        files.append({
            "dataset": "tedlium3", "upstream_id": f"Talk_{i}", "sha256": sha,
            "duration_s": 240.0, "strata": ["tedlium3"], "license_spdx": "CC-BY-NC-ND-3.0",
            "transcript_sha256": sha, "ts_granularity": "segment",
            "acquire": {"kind": "ted_window", "archive": "x", "sph": "a.sph",
                        "stm": "a.stm", "offset_s": 120.0, "window_s": 240.0},
        })
    synth = [("synthetic_silence", 5, None), ("synthetic_bait", 5, None),
             ("synthetic_short_clean", 5, "hello world")]
    for stratum, count, text in synth:
        for i in range(count):
            files.append({
                "dataset": "synthetic", "upstream_id": f"{stratum}_{i}", "sha256": sha,
                "duration_s": 5.0, "strata": [stratum], "license_spdx": "CC0-1.0",
                "transcript_sha256": sha if text else None, "ts_granularity": "none",
                "acquire": {"kind": "committed", "path": f"synthetic/{stratum}_{i}.wav"},
                "text": text,
            })
    return {
        "schema_version": 1,
        "selection": {"seed": src.SELECTION_SEED, "version": src.SELECTION_VERSION},
        "strata_targets": dict(src.STRATA_TARGETS),
        "provenance": {},
        "files": files,
    }


class TestManifestSchema:
    def test_valid_manifest_passes(self) -> None:
        prep.validate_manifest_schema(_valid_manifest())  # no raise

    def test_wrong_schema_version_rejected(self) -> None:
        m = _valid_manifest()
        m["schema_version"] = 2
        with pytest.raises(SystemExit):
            prep.validate_manifest_schema(m)

    def test_selection_drift_rejected(self) -> None:
        m = _valid_manifest()
        m["selection"]["seed"] = "tampered"
        with pytest.raises(SystemExit):
            prep.validate_manifest_schema(m)

    def test_strata_shortfall_rejected(self) -> None:
        m = _valid_manifest()
        m["files"] = m["files"][:-1]  # one short_clean missing
        with pytest.raises(SystemExit):
            prep.validate_manifest_schema(m)

    def test_ted_transcript_must_not_be_committed(self) -> None:
        m = _valid_manifest()
        ted = next(e for e in m["files"] if e["dataset"] == "tedlium3")
        ted["gold_file"] = "gold/ted/leak.json"  # NC-ND violation
        with pytest.raises(SystemExit):
            prep.validate_manifest_schema(m)

    def test_ami_requires_gold_file(self) -> None:
        m = _valid_manifest()
        ami = next(e for e in m["files"] if e["dataset"] == "ami_ihm")
        del ami["gold_file"]
        with pytest.raises(SystemExit):
            prep.validate_manifest_schema(m)

    def test_synthetic_text_transcript_must_agree(self) -> None:
        m = _valid_manifest()
        sc = next(e for e in m["files"] if e["strata"] == ["synthetic_short_clean"])
        sc["transcript_sha256"] = None  # text present but hash dropped
        with pytest.raises(SystemExit):
            prep.validate_manifest_schema(m)

    def test_bad_sha_rejected(self) -> None:
        m = _valid_manifest()
        m["files"][0]["sha256"] = "nothex"
        with pytest.raises(SystemExit):
            prep.validate_manifest_schema(m)
