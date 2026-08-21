"""Unit tests for the AMI WER reference freeze tool's pure core (issue #97).

Exercises UEM parsing, midpoint UEM cropping, deterministic chronological merge
across speakers, and the end-to-end per-meeting build, with synthetic NXT XML /
UEM fixtures that mirror the real AMI formats. No real corpus bytes and no
network: the tool's AMI word parsing is reused from ``prepare_bakeoff_corpus``
(covered against real bytes there), so here we pin the freeze-tool logic layered
on top of it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load_tool():
    path = REPO / "tools" / "build_ami_wer_reference.py"
    spec = importlib.util.spec_from_file_location("build_ami_wer_reference", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


tool = _load_tool()


def _words_xml(recording: str, agent: str, rows: list[tuple[str, str, str]]) -> str:
    """Build a minimal NXT ``*.words.xml`` body from (start, end, token) rows."""
    body = [
        '<?xml version="1.0" encoding="ISO-8859-1" standalone="yes"?>',
        f'<nite:root nite:id="{recording}.{agent}.words" '
        'xmlns:nite="http://nite.sourceforge.net/">',
    ]
    for i, (start, end, token) in enumerate(rows):
        body.append(
            f'   <w nite:id="{recording}.{agent}.words{i}" '
            f'starttime="{start}" endtime="{end}">{token}</w>'
        )
    body.append("</nite:root>")
    return "\n".join(body)


class TestParseUem:
    def test_single_region(self) -> None:
        regions = tool.parse_uem("EN2002c 1 5.00 600.00\n", "EN2002c")
        assert regions == [tool.UemRegion(start_us=5_000_000, end_us=600_000_000)]

    def test_filters_other_recordings(self) -> None:
        text = "OTHER 1 0.00 9.00\nEN2002c 1 5.00 10.00\n"
        regions = tool.parse_uem(text, "EN2002c")
        assert regions == [tool.UemRegion(start_us=5_000_000, end_us=10_000_000)]

    def test_no_region_for_recording_raises(self) -> None:
        with pytest.raises(tool.BuildError):
            tool.parse_uem("OTHER 1 0.00 9.00\n", "EN2002c")

    def test_non_positive_region_raises(self) -> None:
        with pytest.raises(tool.BuildError):
            tool.parse_uem("EN2002c 1 10.00 10.00\n", "EN2002c")

    def test_comment_and_blank_lines_ignored(self) -> None:
        text = ";; header\n\nEN2002c 1 1.00 2.00\n"
        assert tool.parse_uem(text, "EN2002c") == [
            tool.UemRegion(start_us=1_000_000, end_us=2_000_000)
        ]


class TestMidpointCrop:
    def _w(self, start_us: int, end_us: int) -> tool.TimedWord:
        return tool.TimedWord(start_us=start_us, end_us=end_us, text="x", speaker="A")

    def test_midpoint_inside_kept(self) -> None:
        region = [tool.UemRegion(start_us=0, end_us=1_000_000)]
        assert tool.in_any_region(self._w(400_000, 600_000), region)

    def test_word_straddling_start_dropped_when_midpoint_before(self) -> None:
        # word 0.0-0.4 with midpoint 0.2 is below a region starting at 0.5
        region = [tool.UemRegion(start_us=500_000, end_us=1_000_000)]
        assert not tool.in_any_region(self._w(0, 400_000), region)

    def test_midpoint_at_region_end_excluded(self) -> None:
        # end is exclusive: midpoint exactly at end_us is out
        region = [tool.UemRegion(start_us=0, end_us=1_000_000)]
        assert not tool.in_any_region(self._w(1_000_000, 1_000_000), region)


class TestMergeChronologically:
    def test_orders_across_speakers_by_start(self) -> None:
        words = [
            tool.TimedWord(30, 40, "third", "A"),
            tool.TimedWord(10, 20, "first", "B"),
            tool.TimedWord(20, 30, "second", "A"),
        ]
        assert [w.text for w in tool.merge_chronologically(words)] == [
            "first",
            "second",
            "third",
        ]

    def test_deterministic_tiebreak_on_equal_times(self) -> None:
        # same interval, two speakers: tie-break by speaker then text
        words = [
            tool.TimedWord(10, 20, "beta", "B"),
            tool.TimedWord(10, 20, "alpha", "A"),
        ]
        merged = tool.merge_chronologically(words)
        assert [(w.speaker, w.text) for w in merged] == [("A", "alpha"), ("B", "beta")]


class TestBuildReference:
    def _layout(self, tmp_path: Path) -> Path:
        ami = tmp_path / "ami"
        words = ami / "annotations" / "words"
        uems = ami / "AMI-diarization-setup-main" / "uems" / "test"
        words.mkdir(parents=True)
        uems.mkdir(parents=True)
        (words / "EN2002c.A.words.xml").write_text(
            _words_xml("EN2002c", "A", [("1.00", "1.20", "hello"), ("3.00", "3.20", "late")])
        )
        (words / "EN2002c.B.words.xml").write_text(
            _words_xml("EN2002c", "B", [("2.00", "2.20", "world")])
        )
        # UEM crops out the 3.0s word from speaker A.
        (uems / "EN2002c.uem").write_text("EN2002c 1 0.50 2.50\n")
        return ami

    def test_merges_crops_and_hashes(self, tmp_path: Path) -> None:
        ami = self._layout(tmp_path)
        text, record, cpwer_streams = tool.build_reference(ami, "EN2002c", "test")
        assert text == "hello world"  # 'late' at 3.0s cropped by the UEM
        assert record["word_count"] == 2
        assert record["words_dropped_outside_uem"] == 1
        assert record["speakers"] == ["A", "B"]
        assert record["evaluated_duration_s"] == pytest.approx(2.0)
        assert len(record["text_sha256"]) == 64
        assert len(record["canonical_words_sha256"]) == 64
        # cpWER streams: per-speaker, occurrence-partition of the merged words.
        assert cpwer_streams == {"speaker:A": ["hello"], "speaker:B": ["world"]}
        assert record["cpwer_speaker_word_counts"] == {"speaker:A": 1, "speaker:B": 1}
        assert sum(record["cpwer_speaker_word_counts"].values()) == record["word_count"]

    def test_deterministic_shas(self, tmp_path: Path) -> None:
        ami = self._layout(tmp_path)
        _, a, sa = tool.build_reference(ami, "EN2002c", "test")
        _, b, sb = tool.build_reference(ami, "EN2002c", "test")
        assert a["text_sha256"] == b["text_sha256"]
        assert a["canonical_words_sha256"] == b["canonical_words_sha256"]
        assert sa == sb

    def test_missing_uem_raises(self, tmp_path: Path) -> None:
        ami = self._layout(tmp_path)
        with pytest.raises(tool.BuildError):
            tool.build_reference(ami, "EN2002c", "dev")  # wrong split → no UEM


class TestSubsetSelection:
    def test_ami_ids_from_subset(self, tmp_path: Path) -> None:
        subset = tmp_path / "scoring_subset.json"
        subset.write_text(
            '{"files": ['
            '{"corpus": "ami", "id": "EN2002c", "split": "test"},'
            '{"corpus": "voxconverse", "id": "uicid", "split": "test"},'
            '{"corpus": "ami", "id": "IS1008a", "split": "dev"}'
            "]}"
        )
        assert tool.ami_ids_from_subset(subset) == [
            ("EN2002c", "test"),
            ("IS1008a", "dev"),
        ]

    def test_no_ami_entries_raises(self, tmp_path: Path) -> None:
        subset = tmp_path / "s.json"
        subset.write_text('{"files": [{"corpus": "voxconverse", "id": "x", "split": "test"}]}')
        with pytest.raises(tool.BuildError):
            tool.ami_ids_from_subset(subset)
