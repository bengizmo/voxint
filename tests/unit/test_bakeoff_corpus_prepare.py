"""Unit tests for the whisper-bakeoff corpus prepare tool's pure core.

Covers the parsers/selection/serialization that were also exercised against
real AMI + TED bytes in ``tools/prepare_bakeoff_corpus.py verify-sources``
(2026-08-16): STM (incl. ignore-masks), AMI NXT words, AMI meetings channel
map, deterministic selection, and float-free time canonicalization. Synthetic
fixtures mirror the real formats. Network/soundfile paths are not touched here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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
