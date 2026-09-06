"""AMI cross-session speaker recurrence analysis (#113 item 4).

Pure, DB-free. Parses AMI ``meetings.xml`` global participant identities to
determine whether the corpus has enough cross-session speaker recurrence for
a defensible speaker-attribution baseline.

AMI scenario suffixes (a/b/c/d) are four meetings from one day-recording
session, not independent sessions. This module derives ``base_session_id``
from the meeting-series prefix so same-day recordings are grouped correctly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import combinations
from xml.etree import ElementTree as ET


@dataclass(frozen=True)
class MeetingSpeaker:
    """One speaker slot in one AMI meeting."""

    nxt_agent: str
    channel: int
    global_name: str | None


@dataclass(frozen=True)
class SpeakerAppearance:
    """A participant's appearance in a specific meeting."""

    meeting_id: str
    nxt_agent: str
    channel: int
    base_session_id: str


@dataclass(frozen=True)
class RecurrenceReport:
    """Result of the cross-session recurrence analysis."""

    n_meetings: int
    n_base_sessions: int
    n_participants: int
    n_cross_session_speakers: int
    n_genuine_pairs: int
    n_impostor_pairs: int
    baseline_viable: bool
    calibration_viable: bool
    speakers: dict[str, list[SpeakerAppearance]] = field(repr=False)


# ---- AMI meeting-ID conventions ------------------------------------------

_SCENARIO_SUFFIX = re.compile(r"[a-d]$")


def base_session_id(meeting_id: str) -> str:
    """Derive the base session from an AMI meeting ID.

    AMI scenario meetings (e.g. ES2002a, ES2002b, ES2002c, ES2002d) share
    the same participants on the same day. The base session groups them so
    enrollment/test disjointness can be enforced at the session level.

    Non-scenario meetings (e.g. EN2001a is standalone) still get their
    suffix stripped -- conservatively treating them as part of a series
    is safe (it only restricts the split, never loosens it).
    """
    return _SCENARIO_SUFFIX.sub("", meeting_id)


# ---- XML parsing ----------------------------------------------------------


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_meetings_full(
    xml_bytes: bytes,
) -> dict[str, dict[str, MeetingSpeaker]]:
    """Parse AMI ``meetings.xml`` with full speaker metadata.

    Returns ``{meeting_id: {nxt_agent: MeetingSpeaker}}``.

    The ``global_name`` attribute on ``<speaker>`` elements links
    participants across meetings. NXT agent letters (A/B/C/D) are
    meeting-local channel/annotation labels, not identity.
    """
    root = ET.fromstring(xml_bytes)
    mapping: dict[str, dict[str, MeetingSpeaker]] = {}
    for meeting in root.iter():
        if _localname(meeting.tag) != "meeting":
            continue
        observation = meeting.get("observation")
        if not observation:
            continue
        agents: dict[str, MeetingSpeaker] = {}
        for speaker in meeting:
            if _localname(speaker.tag) != "speaker":
                continue
            agent = speaker.get("nxt_agent")
            channel_str = speaker.get("channel")
            if agent is None or channel_str is None:
                continue
            global_name = speaker.get("global_name")
            agents[agent] = MeetingSpeaker(
                nxt_agent=agent,
                channel=int(channel_str),
                global_name=global_name,
            )
        if agents:
            mapping[observation] = agents
    return mapping


# ---- Recurrence analysis --------------------------------------------------


def build_speaker_index(
    meetings: dict[str, dict[str, MeetingSpeaker]],
) -> dict[str, list[SpeakerAppearance]]:
    """Index participants by global name across all meetings.

    Speakers without a ``global_name`` are excluded (they cannot be linked
    across meetings).
    """
    index: dict[str, list[SpeakerAppearance]] = {}
    for meeting_id, agents in sorted(meetings.items()):
        session = base_session_id(meeting_id)
        for agent_id, spk in sorted(agents.items()):
            if spk.global_name is None:
                continue
            appearance = SpeakerAppearance(
                meeting_id=meeting_id,
                nxt_agent=agent_id,
                channel=spk.channel,
                base_session_id=session,
            )
            index.setdefault(spk.global_name, []).append(appearance)
    return index


def cross_session_speakers(
    index: dict[str, list[SpeakerAppearance]],
) -> dict[str, list[SpeakerAppearance]]:
    """Filter to participants appearing in 2+ distinct base sessions."""
    return {
        name: appearances
        for name, appearances in index.items()
        if len({a.base_session_id for a in appearances}) >= 2
    }


def count_genuine_pairs(
    index: dict[str, list[SpeakerAppearance]],
) -> int:
    """Count cross-session genuine pairs (same speaker, different session).

    A genuine pair is two appearances of the same speaker in distinct base
    sessions. Within-session pairs are excluded (they share acoustic
    conditions and provide no discrimination).
    """
    total = 0
    for appearances in index.values():
        sessions = {a.base_session_id for a in appearances}
        if len(sessions) >= 2:
            total += len(list(combinations(sessions, 2)))
    return total


def count_impostor_pairs(
    cross_session: dict[str, list[SpeakerAppearance]],
) -> int:
    """Count cross-session impostor pairs (different speakers, same session).

    An impostor pair is two different cross-session speakers co-occurring in
    the same base session. These are the natural negative trials for FAR.
    """
    session_to_speakers: dict[str, set[str]] = {}
    for name, appearances in cross_session.items():
        for a in appearances:
            session_to_speakers.setdefault(a.base_session_id, set()).add(name)
    total = 0
    for speakers in session_to_speakers.values():
        if len(speakers) >= 2:
            total += len(list(combinations(speakers, 2)))
    return total


def check_kill_criterion(
    meetings: dict[str, dict[str, MeetingSpeaker]],
    *,
    min_speakers: int = 8,
    min_genuine_pairs: int = 50,
    min_calibration_clusters: int = 50,
) -> RecurrenceReport:
    """Run the full recurrence viability analysis.

    Two gates:

    - **Baseline viable**: enough cross-session speakers and genuine pairs
      to run the harness and yield a rough operating point. AMI is expected
      to pass this.
    - **Calibration viable**: enough independent speaker clusters for
      threshold certification (``MIN_INDEPENDENT_CLUSTERS`` in
      ``calibration.py``). AMI is expected to *fail* this -- calibration
      uses production adjudication data instead.
    """
    index = build_speaker_index(meetings)
    cross = cross_session_speakers(index)
    genuine = count_genuine_pairs(cross)
    impostor = count_impostor_pairs(cross)

    all_sessions = {
        base_session_id(mid) for mid in meetings
    }

    return RecurrenceReport(
        n_meetings=len(meetings),
        n_base_sessions=len(all_sessions),
        n_participants=len(index),
        n_cross_session_speakers=len(cross),
        n_genuine_pairs=genuine,
        n_impostor_pairs=impostor,
        baseline_viable=(
            len(cross) >= min_speakers and genuine >= min_genuine_pairs
        ),
        calibration_viable=len(cross) >= min_calibration_clusters,
        speakers=cross,
    )
