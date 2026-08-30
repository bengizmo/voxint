"""Table-driven tests for the pure name-mention extraction inventory (#38).

Transcript cases deliberately use lowercase, punctuation-poor text: ASR
``raw_text`` has no reliable casing, so every guard must hold without it.
"""

import pytest

from voxint.enrichment.producers.name_patterns import (
    Attribution,
    MetadataRef,
    SegmentRef,
    extract_from_metadata,
    extract_from_segment,
    normalize_name,
    person_shaped,
)


def _segment(text: str, *, label: str | None = "SPEAKER_00", suspect: bool = False):  # type: ignore[no-untyped-def]
    return extract_from_segment(
        text,
        segment_index=0,
        diarization_label=label,
        start_seconds=12.5,
        suspect=suspect,
    )


# ---------------------------------------------------------------------------
# normalize_name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("jane doe", "Jane Doe"),
        ("JANE DOE", "Jane Doe"),
        ("Jane Doe", "Jane Doe"),
        ("McLovin Smith", "McLovin Smith"),  # mixed case preserved
        ("dr jane doe", "Jane Doe"),
        ("Dr. Jane Doe", "Jane Doe"),
        ("john smith jr", "John Smith"),
        ("john smith jr.", "John Smith"),
        ("josé garcía", "José García"),
        ("shaun o'brien", "Shaun O'Brien"),
        ("mary-anne lee", "Mary-Anne Lee"),
        ('"jane doe"', "Jane Doe"),
        ("  jane   doe  ", "Jane Doe"),
    ],
)
def test_normalize_name_accepts(raw: str, expected: str) -> None:
    assert normalize_name(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "j",  # too short
        "jane4doe",  # digit
        "jane.com",  # url-ish
        "www.jane.example",
        "@janedoe",
        "dr",  # nothing left after honorific strip
        "jane doe podcast",  # org word
        "acme inc",  # org word
        "everybody",  # generic
        "monday",  # weekday
        "me",  # pronoun-generic; also too short after checks
        "one two three four five",  # > 4 tokens
    ],
)
def test_normalize_name_rejects(raw: str) -> None:
    assert normalize_name(raw) is None


def test_person_shaped_requires_two_to_four_plain_tokens() -> None:
    assert person_shaped("Jane Doe") == "Jane Doe"
    assert person_shaped("Veritasium") is None  # single token
    assert person_shaped("Jane Doe Podcast") is None  # org word
    assert person_shaped("Jane Doe | Live") is None  # not just a name
    assert person_shaped("Jane Doe 2024") is None  # digits


# ---------------------------------------------------------------------------
# Transcript SELF patterns
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "pattern_id", "name"),
    [
        ("hi everyone my name is jane doe and welcome", "self_my_name_is", "Jane Doe"),
        ("my name's bob smith", "self_my_name_is", "Bob Smith"),
        ("i'm jane doe from the daily brief", "self_i_am", "Jane Doe"),
        ("i am carlos rivera and this week we talk pumps", "self_i_am", "Carlos Rivera"),
        ("hey this is jane doe", "self_this_is", "Jane Doe"),
        ("this is sam king coming to you live", "self_this_is", "Sam King"),
        ("jane doe here, thanks for tuning in", "self_speaking", "Jane Doe"),
        ("sam king speaking", "self_speaking", "Sam King"),
    ],
)
def test_self_patterns_match(text: str, pattern_id: str, name: str) -> None:
    mentions = [m for m in _segment(text) if m.pattern_id == pattern_id]
    assert [m.name for m in mentions] == [name]
    assert mentions[0].attribution is Attribution.SELF


@pytest.mark.parametrize(
    "text",
    [
        "i'm sure this will work",  # predicate complement
        "i'm really excited about this",
        "i'm not jane by the way",  # negation via predicate stoplist
        "my name isn't jane",  # negated trigger never matches
        "i'm from boston originally",  # capture-stopword leaves nothing
        "i'm going to show you something",
        "he said my name is jane",  # reported speech
        "she says i'm jane all the time",  # quotative window
        "this is the best show ever",  # non-person "this is"
        "and now this is acme podcast",  # org-shaped
        "well this is awkward",  # predicate complement
        "i'm here with you today",  # "here" predicate stopword
    ],
)
def test_self_patterns_guarded(text: str) -> None:
    assert [m for m in _segment(text) if m.attribution is Attribution.SELF] == []


def test_mid_sentence_this_is_does_not_match() -> None:
    # self_this_is is anchored to the segment start.
    assert _segment("as i was saying this is jane doe") == []


def test_coordinated_intro_splits_names() -> None:
    # "and" terminates the first capture; the second clause is not start-anchored.
    mentions = _segment("i'm jane doe and this is bob smith")
    assert [(m.pattern_id, m.name) for m in mentions] == [("self_i_am", "Jane Doe")]


def test_runaway_capture_terminated_by_stopwords() -> None:
    mentions = _segment("my name is jane and today we talk about heat pumps")
    assert [m.name for m in mentions] == ["Jane"]


def test_ambiguous_single_token_flagged() -> None:
    (mention,) = _segment("my name is will")
    assert mention.name == "Will"
    assert mention.ambiguous is True
    (full,) = _segment("my name is will turner")
    assert full.ambiguous is False


def test_suspect_flag_rides_on_source() -> None:
    (mention,) = _segment("my name is jane doe", suspect=True)
    assert isinstance(mention.source, SegmentRef)
    assert mention.source.suspect is True
    assert mention.source.diarization_label == "SPEAKER_00"
    assert mention.source.start_seconds == 12.5


# ---------------------------------------------------------------------------
# Transcript OTHER patterns
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "pattern_id", "name"),
    [
        ("please welcome jane doe", "other_welcome", "Jane Doe"),
        ("welcome to the show, sam king", "other_welcome", "Sam King"),
        ("we're joined by dr maria lopez", "other_joined_by", "Maria Lopez"),
        ("we are joined today by ken burns", "other_joined_by", "Ken Burns"),
        ("i'm here with jane doe", "other_joined_by", "Jane Doe"),
        ("joining us today is bob smith", "other_joined_by", "Bob Smith"),
        ("thanks for joining us jane", "other_thanks_joining", "Jane"),
        ("thank you, maria", "other_vocative_thanks", "Maria"),
    ],
)
def test_other_patterns_match(text: str, pattern_id: str, name: str) -> None:
    mentions = [m for m in _segment(text) if m.pattern_id == pattern_id]
    assert [m.name for m in mentions] == [name]
    assert mentions[0].attribution is Attribution.OTHER


@pytest.mark.parametrize(
    "text",
    [
        "welcome back everybody",  # generic target
        "welcome to my channel",  # capture stopword
        "thank you so much for listening",  # no vocative comma
        "this episode is sponsored by jane doe promo code jane10",  # sponsor window
        "welcome jane doe brought to you by acme",  # sponsor window
    ],
)
def test_other_patterns_guarded(text: str) -> None:
    assert [m for m in _segment(text) if m.attribution is Attribution.OTHER] == []


def test_i_m_here_with_yields_other_not_self() -> None:
    mentions = _segment("i'm here with jane doe")
    assert [(m.pattern_id, m.attribution) for m in mentions] == [
        ("other_joined_by", Attribution.OTHER)
    ]


# ---------------------------------------------------------------------------
# Metadata patterns
# ---------------------------------------------------------------------------


def _metadata(  # type: ignore[no-untyped-def]
    *,
    title: str | None = None,
    description: str | None = None,
    channel: str | None = None,
    uploader: str | None = None,
    tags: tuple[str, ...] = (),
):
    return extract_from_metadata(
        title=title, description=description, channel=channel, uploader=uploader, tags=tags
    )


def test_title_interview_with_dedupes_overlapping_with() -> None:
    # Both title_interview_with and title_with cover the same span; the
    # stronger pattern wins and no fake corroboration remains.
    mentions = _metadata(title="Interview with Jane Doe")
    assert [(m.pattern_id, m.name, m.reliability) for m in mentions] == [
        ("title_interview_with", "Jane Doe", 0.85)
    ]


@pytest.mark.parametrize(
    ("title", "pattern_id", "name"),
    [
        ("Heat Pump Myths ft. Jane Doe", "title_ft", "Jane Doe"),
        ("Deep Dive featuring Bob Smith", "title_ft", "Bob Smith"),
        ("Refrigerant Basics with Jane Doe", "title_with", "Jane Doe"),
        ("Jane Doe | Ep 12", "title_guest_sep", "Jane Doe"),
        ("Compressor Talk: Maria Lopez", "title_guest_sep", "Maria Lopez"),
    ],
)
def test_title_patterns_match(title: str, pattern_id: str, name: str) -> None:
    mentions = [m for m in _metadata(title=title) if m.pattern_id == pattern_id]
    assert [m.name for m in mentions] == [name]
    assert mentions[0].attribution is Attribution.METADATA
    assert mentions[0].source == MetadataRef(field="title")


def test_title_with_must_be_trailing() -> None:
    assert [
        m for m in _metadata(title="With Jane Doe we explore ducts") if m.pattern_id == "title_with"
    ] == []


@pytest.mark.parametrize(
    ("description", "pattern_id", "name"),
    [
        ("A show hosted by Jane Doe about pumps.", "desc_hosted_by", "Jane Doe"),
        ("Host: Bob Smith", "desc_hosted_by", "Bob Smith"),
        ("Your host Maria Lopez digs in.", "desc_hosted_by", "Maria Lopez"),
        ("Guest: Ken Burns", "desc_guest", "Ken Burns"),
        ("This week we're in conversation with Jane Doe.", "desc_guest", "Jane Doe"),
    ],
)
def test_description_patterns_match(description: str, pattern_id: str, name: str) -> None:
    mentions = [m for m in _metadata(description=description) if m.pattern_id == pattern_id]
    assert [m.name for m in mentions] == [name]


def test_channel_as_host_person_shaped_only() -> None:
    assert [(m.pattern_id, m.name) for m in _metadata(channel="Jane Doe")] == [
        ("channel_as_host", "Jane Doe")
    ]
    assert _metadata(channel="Jane Doe Podcast") == []
    assert _metadata(channel="Veritasium") == []
    (uploader_mention,) = _metadata(uploader="Bob Smith")
    assert uploader_mention.source == MetadataRef(field="uploader")


def test_tag_person_records_item_index() -> None:
    mentions = _metadata(tags=("repair tips", "jane doe", "bob"))
    assert [(m.name, m.source) for m in mentions] == [
        ("Jane Doe", MetadataRef(field="tags", item_index=1))
    ]


def test_sponsor_window_guards_description() -> None:
    text = "Use code JANE10 for a discount. This episode featuring Jane Doe."
    assert [m for m in _metadata(description=text) if m.name == "Jane Doe"] == []


def test_empty_inputs_yield_nothing() -> None:
    assert _metadata() == []
    assert _segment("") == []
    assert _segment("   ") == []


# ---------------------------------------------------------------------------
# _snippet word alignment (#318)
# ---------------------------------------------------------------------------

from voxint.enrichment.producers.name_patterns import (  # noqa: E402
    _SNIPPET_ALIGN_CHARS,
    SNIPPET_CONTEXT_CHARS,
    _snippet,
)


def test_snippet_edges_expand_to_word_boundaries() -> None:
    # 11-char tokens; the fixed +/-60 window lands mid-token on both sides,
    # so both edges must walk outward to whitespace.
    text = "abcdefghij " * 40
    start = 300  # mid-token (300 % 11 == 3)
    result = _snippet(text, start, start + 4)
    tokens = result.split(" ")
    assert tokens[0] == "abcdefghij"
    assert tokens[-1] == "abcdefghij"


def test_snippet_keeps_punctuation_attached_to_tokens() -> None:
    # Punctuation is not whitespace: an edge landing inside "chen," keeps the
    # whole token, comma included.
    filler = "abcdefghij " * 10
    text = filler + "sarah chen, host of the show " + filler
    idx = text.index("host")
    result = _snippet(text, idx, idx + 4)
    assert "sarah chen," in result
    assert "hen," not in result.split(" ", 1)[0] or result.split(" ", 1)[0] == "chen,"


def test_snippet_long_unbroken_token_cut_at_cap() -> None:
    text = "a" * 500
    start, end = 250, 254
    result = _snippet(text, start, end)
    # No whitespace to find: both sides stop at the alignment cap.
    assert len(result) == 2 * (SNIPPET_CONTEXT_CHARS + _SNIPPET_ALIGN_CHARS) + (
        end - start
    )


def test_snippet_at_text_start_and_end() -> None:
    text = "jane doe hosts the show"
    assert _snippet(text, 0, 8) == text  # window covers everything
    assert _snippet(text, len(text) - 4, len(text)) == text


def test_snippet_collapses_whitespace() -> None:
    text = "jane\n\tdoe   hosts the show"
    assert _snippet(text, 0, 8) == "jane doe hosts the show"
