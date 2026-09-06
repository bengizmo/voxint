from xml.etree import ElementTree as ET

import pytest

from voxint.harness.ami_recurrence import (
    MeetingSpeaker,
    SpeakerAppearance,
    base_session_id,
    build_speaker_index,
    check_kill_criterion,
    count_genuine_pairs,
    count_impostor_pairs,
    cross_session_speakers,
    parse_meetings_full,
)

MEETINGS_XML = b"""<?xml version="1.0" encoding="ISO-8859-1"?>
<nite:root xmlns:nite="http://nite.sourceforge.net/">
  <meeting observation="ES2002a" type="scenario">
    <speaker nxt_agent="A" channel="0" global_name="FEE005" role="PM"/>
    <speaker nxt_agent="B" channel="1" global_name="MEE018" role="ID"/>
  </meeting>
  <meeting observation="ES2002b" type="scenario">
    <speaker nxt_agent="A" channel="0" global_name="FEE005" role="PM"/>
    <speaker nxt_agent="B" channel="1" global_name="MEE018" role="ID"/>
  </meeting>
  <meeting observation="IS1003a" type="scenario">
    <speaker nxt_agent="A" channel="0" global_name="FEE005" role="ME"/>
    <speaker nxt_agent="B" channel="1" global_name="MIO034" role="PM"/>
  </meeting>
</nite:root>
"""


def test_base_session_id_strips_scenario_suffix() -> None:
    assert base_session_id("ES2002a") == "ES2002"
    assert base_session_id("ES2002b") == "ES2002"
    assert base_session_id("EN2001a") == "EN2001"
    assert base_session_id("IB4001") == "IB4001"


def test_parse_meetings_full_returns_speaker_metadata() -> None:
    meetings = parse_meetings_full(MEETINGS_XML)

    assert meetings == {
        "ES2002a": {
            "A": MeetingSpeaker("A", 0, "FEE005"),
            "B": MeetingSpeaker("B", 1, "MEE018"),
        },
        "ES2002b": {
            "A": MeetingSpeaker("A", 0, "FEE005"),
            "B": MeetingSpeaker("B", 1, "MEE018"),
        },
        "IS1003a": {
            "A": MeetingSpeaker("A", 0, "FEE005"),
            "B": MeetingSpeaker("B", 1, "MIO034"),
        },
    }


def test_build_speaker_index_builds_reverse_index() -> None:
    meetings = parse_meetings_full(MEETINGS_XML)
    meetings["IB4001"] = {
        "A": MeetingSpeaker("A", 2, None),
    }

    index = build_speaker_index(meetings)

    assert index == {
        "FEE005": [
            SpeakerAppearance("ES2002a", "A", 0, "ES2002"),
            SpeakerAppearance("ES2002b", "A", 0, "ES2002"),
            SpeakerAppearance("IS1003a", "A", 0, "IS1003"),
        ],
        "MEE018": [
            SpeakerAppearance("ES2002a", "B", 1, "ES2002"),
            SpeakerAppearance("ES2002b", "B", 1, "ES2002"),
        ],
        "MIO034": [
            SpeakerAppearance("IS1003a", "B", 1, "IS1003"),
        ],
    }
    assert None not in index


def test_cross_session_speakers_requires_distinct_base_sessions() -> None:
    index = build_speaker_index(parse_meetings_full(MEETINGS_XML))

    cross_session = cross_session_speakers(index)

    assert cross_session == {"FEE005": index["FEE005"]}


def test_count_genuine_pairs_for_two_sessions() -> None:
    index = build_speaker_index(parse_meetings_full(MEETINGS_XML))

    assert count_genuine_pairs(index) == 1


def test_count_genuine_pairs_for_three_sessions() -> None:
    index = {
        "FEE005": [
            SpeakerAppearance("ES2002a", "A", 0, "ES2002"),
            SpeakerAppearance("IS1003a", "A", 0, "IS1003"),
            SpeakerAppearance("TS3001a", "B", 1, "TS3001"),
        ]
    }

    assert count_genuine_pairs(index) == 3


def test_count_impostor_pairs_for_speakers_cooccurring_in_session() -> None:
    recurring_speakers = {
        "FEE005": [
            SpeakerAppearance("ES2002a", "A", 0, "ES2002"),
            SpeakerAppearance("IS1003a", "A", 0, "IS1003"),
        ],
        "MIO034": [
            SpeakerAppearance("IS1003a", "B", 1, "IS1003"),
            SpeakerAppearance("TS3001a", "B", 1, "TS3001"),
        ],
    }

    assert count_impostor_pairs(recurring_speakers) == 1


def test_check_kill_criterion_passes_baseline_but_fails_calibration() -> None:
    meetings = {
        "ES2002a": {
            "A": MeetingSpeaker("A", 0, "FEE005"),
            "B": MeetingSpeaker("B", 1, "MEE018"),
        },
        "IS1003a": {
            "A": MeetingSpeaker("A", 0, "FEE005"),
            "B": MeetingSpeaker("B", 1, "MEE018"),
        },
    }

    report = check_kill_criterion(
        meetings,
        min_speakers=2,
        min_genuine_pairs=2,
        min_calibration_clusters=3,
    )

    assert report.n_meetings == 2
    assert report.n_base_sessions == 2
    assert report.n_participants == 2
    assert report.n_cross_session_speakers == 2
    assert report.n_genuine_pairs == 2
    assert report.n_impostor_pairs == 2
    assert report.baseline_viable is True
    assert report.calibration_viable is False
    assert set(report.speakers) == {"FEE005", "MEE018"}


def test_check_kill_criterion_tiny_corpus_fails_both_gates() -> None:
    meetings = {
        "ES2002a": {
            "A": MeetingSpeaker("A", 0, "FEE005"),
        }
    }

    report = check_kill_criterion(
        meetings,
        min_speakers=1,
        min_genuine_pairs=1,
        min_calibration_clusters=1,
    )

    assert report.n_meetings == 1
    assert report.n_base_sessions == 1
    assert report.n_participants == 1
    assert report.n_cross_session_speakers == 0
    assert report.n_genuine_pairs == 0
    assert report.n_impostor_pairs == 0
    assert report.baseline_viable is False
    assert report.calibration_viable is False
    assert report.speakers == {}


def test_parse_meetings_full_omits_meeting_with_no_speakers() -> None:
    xml = b"""<root>
      <meeting observation="ES2002a" type="scenario" />
    </root>"""

    assert parse_meetings_full(xml) == {}


def test_parse_meetings_full_preserves_speaker_without_global_name() -> None:
    xml = b"""<root>
      <meeting observation="ES2002a">
        <speaker nxt_agent="A" channel="0" />
      </meeting>
    </root>"""

    meetings = parse_meetings_full(xml)

    assert meetings == {
        "ES2002a": {"A": MeetingSpeaker("A", 0, None)},
    }
    assert build_speaker_index(meetings) == {}


def test_parse_meetings_full_rejects_empty_xml() -> None:
    with pytest.raises(ET.ParseError):
        parse_meetings_full(b"")
