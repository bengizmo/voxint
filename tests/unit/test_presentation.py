"""Pure display helpers for the operator console (issue #56) — no DB."""

from datetime import UTC, datetime, timedelta

import pytest

from voxint.api.presentation import (
    format_age,
    format_duration,
    friendly_media_label,
    humanize_stage,
    humanize_status,
)

# ---- friendly_media_label ---------------------------------------------------


def test_friendly_label_prefers_title() -> None:
    assert friendly_media_label("City Council 2026", "/media/a1b2.mp3") == "City Council 2026"


def test_friendly_label_strips_whitespace_title() -> None:
    # A whitespace-only title must NOT render blank — it falls through to the path.
    assert friendly_media_label("  Padded Talk  ", "/x/y.mp3") == "Padded Talk"
    assert friendly_media_label("   ", "/media/interview.wav") == "interview.wav"


def test_friendly_label_falls_back_to_basename_keeping_extension() -> None:
    assert friendly_media_label(None, "/srv/media/2026/interview.mp3") == "interview.mp3"


def test_friendly_label_percent_decodes_only_the_basename() -> None:
    assert friendly_media_label(None, "/media/My%20Council%20Talk.mp3") == "My Council Talk.mp3"


def test_friendly_label_uuid_source_path_is_the_honest_basename() -> None:
    # A pre-#36 URL run with a uuid path and no title: show the uuid, don't invent.
    label = friendly_media_label(None, "/store/3f2a9c14-0b7e-4d21-9a1a-abc123def456")
    assert label == "3f2a9c14-0b7e-4d21-9a1a-abc123def456"


def test_friendly_label_trailing_slash_uses_last_real_segment() -> None:
    # A trailing slash is trimmed, so the last real segment still names the row.
    assert friendly_media_label(None, "/media/folder/") == "folder"


def test_friendly_label_root_only_path_never_empties() -> None:
    # Pathological all-separator input still yields a non-empty string.
    assert friendly_media_label(None, "/") == "/"


def test_friendly_label_handles_windows_separators() -> None:
    assert friendly_media_label(None, "C:\\recordings\\board.mp3") == "board.mp3"


def test_friendly_label_collapses_control_chars() -> None:
    assert friendly_media_label(None, "/m/we%0Aird%09name.mp3") == "we ird name.mp3"


def test_friendly_label_cleans_bidi_and_zero_width_in_title() -> None:
    # The title is the most externally-controlled string (a fetched URL title);
    # a bidi override (U+202E) or zero-width char must be stripped, not rendered.
    rlo, zwsp, bom = chr(0x202E), chr(0x200B), chr(0xFEFF)
    # Each invisible run collapses to a single space (never glues or vanishes
    # silently), and a trailing one is stripped.
    assert friendly_media_label(f"Talk{rlo}evil{zwsp}", "/x/y.mp3") == "Talk evil"
    # A newline in the title collapses to a single space, not a broken line.
    assert friendly_media_label("line1\nline2", "/x/y.mp3") == "line1 line2"
    # A title that is only invisibles falls through to the path.
    assert friendly_media_label(f"{zwsp}{bom}", "/x/real.mp3") == "real.mp3"


# ---- format_duration --------------------------------------------------------


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (None, "—"),
        (-5.0, "—"),
        (float("nan"), "—"),
        (float("inf"), "—"),
        (0.0, "0:00"),
        (9.0, "0:09"),
        (65.0, "1:05"),
        (600.0, "10:00"),
        (3599.0, "59:59"),
        (3600.0, "1:00:00"),
        (3661.0, "1:01:01"),
        (7325.4, "2:02:05"),
    ],
)
def test_format_duration(seconds: float | None, expected: str) -> None:
    assert format_duration(seconds) == expected


# ---- format_age -------------------------------------------------------------


def _ago(**kw: float) -> tuple[datetime, datetime]:
    now = datetime(2026, 8, 17, 12, 0, 0, tzinfo=UTC)
    return now - timedelta(**kw), now


@pytest.mark.parametrize(
    ("delta", "expected"),
    [
        ({"seconds": 5}, "just now"),
        ({"seconds": 59}, "just now"),
        ({"minutes": 1}, "1 minute ago"),
        ({"minutes": 3}, "3 minutes ago"),
        ({"hours": 1}, "1 hour ago"),
        ({"hours": 5}, "5 hours ago"),
        ({"days": 1}, "1 day ago"),
        ({"days": 6}, "6 days ago"),
        ({"days": 8}, "1 week ago"),
        ({"days": 20}, "2 weeks ago"),
        ({"days": 45}, "1 month ago"),
        ({"days": 200}, "6 months ago"),
        ({"days": 400}, "1 year ago"),
    ],
)
def test_format_age_buckets(delta: dict[str, float], expected: str) -> None:
    created_at, now = _ago(**delta)
    assert format_age(created_at, now=now) == expected


@pytest.mark.parametrize(
    ("delta", "expected"),
    [
        ({"seconds": 60}, "1 minute ago"),
        ({"minutes": 60}, "1 hour ago"),
        ({"hours": 24}, "1 day ago"),
        ({"days": 7}, "1 week ago"),
        ({"days": 30}, "1 month ago"),
        ({"days": 365}, "1 year ago"),
    ],
)
def test_format_age_bucket_boundaries(delta: dict[str, float], expected: str) -> None:
    # Lock the exact edge where each unit rolls over to the next.
    created_at, now = _ago(**delta)
    assert format_age(created_at, now=now) == expected


def test_format_age_future_is_just_now() -> None:
    # Clock skew: a future timestamp never renders a negative age.
    now = datetime(2026, 8, 17, 12, 0, 0, tzinfo=UTC)
    assert format_age(now + timedelta(minutes=5), now=now) == "just now"


def test_format_age_normalizes_naive_operands() -> None:
    # A naive created_at (or now) must not raise TypeError and 500 the listing;
    # it is treated as UTC.
    aware_now = datetime(2026, 8, 17, 12, 0, 0, tzinfo=UTC)
    naive_created = datetime(2026, 8, 17, 9, 0, 0)  # deliberately tz-naive
    assert format_age(naive_created, now=aware_now) == "3 hours ago"
    # A naive `now` is normalized too (both operands guarded).
    naive_now = datetime(2026, 8, 17, 12, 0, 0)
    assert format_age(datetime(2026, 8, 17, 10, 0, 0, tzinfo=UTC), now=naive_now) == (
        "2 hours ago"
    )


# ---- humanize_stage / humanize_status ---------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("acquire", "Acquire"),
        ("transcribe", "Transcribe"),
        ("diarize_embed", "Diarize & embed"),
        ("enhance_match", "Enhance & match"),
        ("finalize", "Finalize"),
    ],
)
def test_humanize_stage(value: str, expected: str) -> None:
    assert humanize_stage(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("queued", "Queued"),
        ("completed", "Completed"),
        ("failed", "Failed"),
        ("awaiting_adjudication", "Awaiting adjudication"),
    ],
)
def test_humanize_status(value: str, expected: str) -> None:
    assert humanize_status(value) == expected


def test_humanized_label_differs_from_raw_css_class_key() -> None:
    # Guards the load-bearing invariant: the humanized text is display-only and
    # never equals the raw enum a template still uses in class="pill {value}".
    for value in ("awaiting_adjudication", "diarize_embed"):
        assert humanize_status(value) != value
        assert humanize_stage(value) != value


def test_humanize_unknown_value_degrades_gracefully() -> None:
    # A future enum value renders acceptably with no code change here.
    assert humanize_stage("new_future_stage") == "New future stage"


def test_humanize_empty_string_is_returned_verbatim() -> None:
    # Degenerate input never raises — it round-trips to itself.
    assert humanize_stage("") == ""
    assert humanize_status("") == ""


def test_title_from_snapshot_reads_and_cleans() -> None:
    from voxint.api.presentation import title_from_snapshot

    assert title_from_snapshot({"title": "Plain"}) == "Plain"
    # Same cleaning as friendly_media_label: padding stripped, zero-width and
    # bidi controls removed (a tampered snapshot must not be the one console
    # title path that skips the strip).
    assert title_from_snapshot({"title": "  padded  "}) == "padded"
    assert title_from_snapshot({"title": "a​b‮c"}) == "a b c"


@pytest.mark.parametrize(
    "snapshot",
    [None, [], "str", 42, {}, {"title": None}, {"title": 7}, {"title": "   "},
     {"title": "​"}],
)
def test_title_from_snapshot_tolerates_tampered_snapshots(snapshot: object) -> None:
    from voxint.api.presentation import title_from_snapshot

    assert title_from_snapshot(snapshot) is None
