"""Pure read-time attribution helpers shared by the transcript view and exports.

``parse_transcript_text`` (variant selection), ``display_name`` /
``segment_speaker`` (the speaker string for a segment), and
``winning_attribution`` (most-specific-wins precedence over one walk step) are
all pure functions of their arguments — no ``Session``, no I/O. The DB-coupled
walks that feed them (``attributed_transcript`` / ``attributed_intervals`` /
``walk_attributions``) are exercised in the integration suite; this pins the
pure logic directly. The sibling selector ``effective_text`` is covered in
``test_effective_text.py``.

Fakes are duck-typed ``SimpleNamespace`` objects, matching ``test_effective_text``:
each function reads only a few attributes, named in the helper it is built with.
"""

import uuid
from types import SimpleNamespace

import pytest

from voxint.adjudication.attribution import AttributionScope, winning_attribution
from voxint.adjudication.resolver import Resolution
from voxint.adjudication.transcript import (
    TranscriptText,
    display_name,
    parse_transcript_text,
    segment_speaker,
)


# --------------------------------------------------------------------------- #
# parse_transcript_text
# --------------------------------------------------------------------------- #
def test_parse_transcript_text_blank_or_absent_is_corrected() -> None:
    # A blank/absent value means the operator-effective default view.
    assert parse_transcript_text(None) is TranscriptText.CORRECTED
    assert parse_transcript_text("") is TranscriptText.CORRECTED


def test_parse_transcript_text_named_variants() -> None:
    assert parse_transcript_text("corrected") is TranscriptText.CORRECTED
    assert parse_transcript_text("enhanced") is TranscriptText.ENHANCED
    assert parse_transcript_text("raw") is TranscriptText.RAW


def test_parse_transcript_text_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown transcript text 'sideways'"):
        parse_transcript_text("sideways")


# --------------------------------------------------------------------------- #
# display_name
# --------------------------------------------------------------------------- #
def _seg(label: str | None) -> SimpleNamespace:
    # display_name / segment_speaker read only .diarization_label.
    return SimpleNamespace(diarization_label=label)


def _state(resolution: Resolution, speaker_name: str | None) -> SimpleNamespace:
    # display_name reads only .resolution and .speaker_name off a LabelState.
    return SimpleNamespace(resolution=resolution, speaker_name=speaker_name)


def test_display_name_no_state_uses_raw_label() -> None:
    assert display_name(None, _seg("SPEAKER_00")) == "SPEAKER_00"


def test_display_name_no_state_no_label_is_placeholder() -> None:
    assert display_name(None, _seg(None)) == "(no speaker)"


def test_display_name_human_assign_shows_speaker_name() -> None:
    state = _state(Resolution.HUMAN_ASSIGN, "Alice")
    assert display_name(state, _seg("SPEAKER_00")) == "Alice"


def test_display_name_grounded_cosine_shows_speaker_name() -> None:
    state = _state(Resolution.GROUNDED_COSINE, "Bob")
    assert display_name(state, _seg("SPEAKER_01")) == "Bob"


def test_display_name_assign_without_name_falls_back_to_label() -> None:
    # A grounded/assigned state with no name still shows the local label.
    state = _state(Resolution.HUMAN_ASSIGN, None)
    assert display_name(state, _seg("SPEAKER_00")) == "SPEAKER_00"


def test_display_name_exclude_annotates_label() -> None:
    state = _state(Resolution.HUMAN_EXCLUDE, None)
    assert display_name(state, _seg("SPEAKER_00")) == "(excluded) SPEAKER_00"


def test_display_name_unknown_annotates_label() -> None:
    state = _state(Resolution.HUMAN_UNKNOWN, None)
    assert display_name(state, _seg("SPEAKER_00")) == "Unknown (SPEAKER_00)"


def test_display_name_unresolved_falls_back_to_label() -> None:
    state = _state(Resolution.UNRESOLVED, None)
    assert display_name(state, _seg("SPEAKER_00")) == "SPEAKER_00"


# --------------------------------------------------------------------------- #
# segment_speaker
# --------------------------------------------------------------------------- #
def test_segment_speaker_uses_override_name() -> None:
    override = SimpleNamespace(speaker_name="Carol")
    assert segment_speaker(override, _seg("SPEAKER_00")) == "Carol"


def test_segment_speaker_without_name_falls_back_to_label() -> None:
    override = SimpleNamespace(speaker_name=None)
    assert segment_speaker(override, _seg("SPEAKER_00")) == "SPEAKER_00"


def test_segment_speaker_without_name_or_label_is_placeholder() -> None:
    override = SimpleNamespace(speaker_name=None)
    assert segment_speaker(override, _seg(None)) == "(no speaker)"


# --------------------------------------------------------------------------- #
# winning_attribution (word-range > whole-segment > label)
# --------------------------------------------------------------------------- #
def _overlay(speaker_id: uuid.UUID, speaker_name: str | None) -> SimpleNamespace:
    # winning_attribution reads .speaker_id / .speaker_name off an override.
    return SimpleNamespace(speaker_id=speaker_id, speaker_name=speaker_name)


def _label_state(
    resolution: Resolution, speaker_id: uuid.UUID, speaker_name: str | None
) -> SimpleNamespace:
    return SimpleNamespace(resolution=resolution, speaker_id=speaker_id, speaker_name=speaker_name)


def _emission(
    *,
    range_override: object = None,
    seg_override: object = None,
    label_state: object = None,
) -> SimpleNamespace:
    # winning_attribution reads only these three overlays off an Emission.
    return SimpleNamespace(
        range_override=range_override,
        seg_override=seg_override,
        label_state=label_state,
    )


def test_winning_attribution_word_range_override_wins() -> None:
    rid, sid, lid = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    em = _emission(
        range_override=_overlay(rid, "Range"),
        seg_override=_overlay(sid, "Seg"),
        label_state=_label_state(Resolution.GROUNDED_COSINE, lid, "Label"),
    )
    assert winning_attribution(em) == (
        AttributionScope.WORD_RANGE,
        Resolution.HUMAN_ASSIGN,
        rid,
        "Range",
    )


def test_winning_attribution_segment_override_beats_label() -> None:
    sid, lid = uuid.uuid4(), uuid.uuid4()
    em = _emission(
        seg_override=_overlay(sid, "Seg"),
        label_state=_label_state(Resolution.GROUNDED_COSINE, lid, "Label"),
    )
    assert winning_attribution(em) == (
        AttributionScope.SEGMENT,
        Resolution.HUMAN_ASSIGN,
        sid,
        "Seg",
    )


def test_winning_attribution_no_state_is_unresolved() -> None:
    assert winning_attribution(_emission()) == (
        AttributionScope.LABEL,
        Resolution.UNRESOLVED,
        None,
        None,
    )


def test_winning_attribution_label_human_assign() -> None:
    lid = uuid.uuid4()
    em = _emission(label_state=_label_state(Resolution.HUMAN_ASSIGN, lid, "Label"))
    assert winning_attribution(em) == (
        AttributionScope.LABEL,
        Resolution.HUMAN_ASSIGN,
        lid,
        "Label",
    )


def test_winning_attribution_label_grounded_cosine() -> None:
    lid = uuid.uuid4()
    em = _emission(label_state=_label_state(Resolution.GROUNDED_COSINE, lid, "Label"))
    assert winning_attribution(em) == (
        AttributionScope.LABEL,
        Resolution.GROUNDED_COSINE,
        lid,
        "Label",
    )


def test_winning_attribution_label_exclude_carries_no_speaker() -> None:
    # A non-attributing label ruling reports the resolution but no speaker.
    em = _emission(label_state=_label_state(Resolution.HUMAN_EXCLUDE, uuid.uuid4(), "Label"))
    assert winning_attribution(em) == (
        AttributionScope.LABEL,
        Resolution.HUMAN_EXCLUDE,
        None,
        None,
    )
