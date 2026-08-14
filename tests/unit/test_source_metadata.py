"""Unit tests for the info-JSON sanitizer (media/source_metadata.py).

The load-bearing test is the hostile-document contract: an info-JSON carrying
every secret-bearing key yt-dlp can emit must yield a snapshot in which none of
those keys or values survive — allowlisting IS the redaction story for
persisted metadata.
"""

import dataclasses
import json
from datetime import date
from typing import Any

import pytest

from voxint.media.source_metadata import (
    MAX_DESCRIPTION_CHARS,
    MAX_INFO_JSON_BYTES,
    MAX_TAG_CHARS,
    MAX_TAGS,
    MAX_TEXT_CHARS,
    RAW_ALLOWLIST,
    SNAPSHOT_SCHEMA_VERSION,
    SourceMetadata,
    SourceMetadataError,
    extract,
)


def _payload(overrides: "dict[str, Any] | None" = None) -> bytes:
    info: dict[str, Any] = {
        "id": "abc123",
        "title": "Field Notes Episode 42",
        "uploader": "Example Uploader",
        "uploader_url": "https://example.com/@uploader",
        "channel": "Example Channel",
        "channel_url": "https://example.com/channel/UC123",
        "description": "A conversation about microphones.\nWith timestamps.",
        "upload_date": "20260214",
        "duration": 3641.5,
        "tags": ["interviews", "acoustics"],
        "webpage_url": "https://example.com/watch?v=abc123",
        "extractor": "example",
        "extractor_key": "Example",
        "_version": {"version": "2026.07.04", "repository": "yt-dlp/yt-dlp"},
    }
    if overrides:
        info.update(overrides)
    return json.dumps(info).encode()


class TestHappyPath:
    def test_normalized_fields(self) -> None:
        meta = extract(_payload())
        assert meta.title == "Field Notes Episode 42"
        assert meta.uploader == "Example Uploader"
        assert meta.uploader_url == "https://example.com/@uploader"
        assert meta.channel == "Example Channel"
        assert meta.channel_url == "https://example.com/channel/UC123"
        assert meta.description == "A conversation about microphones.\nWith timestamps."
        assert meta.upload_date == date(2026, 2, 14)
        assert meta.duration_seconds == pytest.approx(3641.5)
        assert meta.tags == ("interviews", "acoustics")
        assert meta.canonical_url == "https://example.com/watch?v=abc123"
        assert meta.extractor == "example"
        assert meta.extractor_version == "2026.07.04"
        assert meta.raw_schema_version == SNAPSHOT_SCHEMA_VERSION

    def test_raw_subset_is_allowlisted_and_json_serializable(self) -> None:
        meta = extract(_payload())
        assert set(meta.raw) <= set(RAW_ALLOWLIST)
        assert meta.raw["id"] == "abc123"
        assert meta.raw["extractor_key"] == "Example"
        json.dumps(meta.raw)  # must round-trip into JSONB without surprises

    def test_missing_fields_degrade_to_absent(self) -> None:
        meta = extract(b"{}")
        assert meta == SourceMetadata()
        assert meta.raw == {}


class TestSecretsNeverSurvive:
    """The hostile-document contract (pin: allowlisting is the redaction)."""

    SECRET_MARKERS = (
        "SIGNED-TOKEN",
        "session-cookie-value",
        "fragment-sig",
        "manifest-sig",
        "sub-sig",
        "thumb-sig",
        "Bearer topsecret",
    )

    def _hostile(self) -> bytes:
        return _payload(
            {
                "url": "https://cdn.example.com/video.mp4?token=SIGNED-TOKEN",
                "formats": [
                    {
                        "url": "https://cdn.example.com/f1.mp4?token=SIGNED-TOKEN",
                        "http_headers": {"Cookie": "session-cookie-value"},
                    }
                ],
                "requested_formats": [
                    {"url": "https://cdn.example.com/f2.mp4?token=SIGNED-TOKEN"}
                ],
                "fragments": [{"url": "https://cdn.example.com/frag?sig=fragment-sig"}],
                "manifest_url": "https://cdn.example.com/m.m3u8?sig=manifest-sig",
                "http_headers": {
                    "Cookie": "session-cookie-value",
                    "Authorization": "Bearer topsecret",
                },
                "cookies": "session-cookie-value",
                "subtitles": {
                    "en": [{"url": "https://cdn.example.com/s.vtt?sig=sub-sig"}]
                },
                "automatic_captions": {
                    "en": [{"url": "https://cdn.example.com/a.vtt?sig=sub-sig"}]
                },
                "thumbnails": [
                    {"url": "https://cdn.example.com/t.jpg?sig=thumb-sig"}
                ],
            }
        )

    def test_no_secret_marker_survives_anywhere(self) -> None:
        meta = extract(self._hostile())
        flat = json.dumps(dataclasses.asdict(meta), default=str)
        for marker in self.SECRET_MARKERS:
            assert marker not in flat, f"secret marker {marker!r} leaked into snapshot"

    def test_secret_bearing_keys_never_enter_raw(self) -> None:
        meta = extract(self._hostile())
        for key in (
            "url",
            "formats",
            "requested_formats",
            "fragments",
            "manifest_url",
            "http_headers",
            "cookies",
            "subtitles",
            "automatic_captions",
            "thumbnails",
        ):
            assert key not in meta.raw
            assert key not in RAW_ALLOWLIST

    def test_extra_secrets_removed_verbatim_from_retained_text(self) -> None:
        payload = _payload(
            {"description": "yt-dlp used proxy http://198.51.100.9:3128 for this"}
        )
        meta = extract(payload, extra_secrets=("http://198.51.100.9:3128",))
        assert meta.description is not None
        assert "198.51.100.9" not in meta.description
        assert "<redacted>" in meta.description

    def test_credentialed_or_nonhttp_urls_dropped_not_mangled(self) -> None:
        meta = extract(
            _payload(
                {
                    "webpage_url": "https://user:pass@example.com/watch?v=1",
                    "channel_url": "ftp://example.com/channel",
                    "uploader_url": "javascript:alert(1)",
                }
            )
        )
        assert meta.canonical_url is None
        assert meta.channel_url is None
        assert meta.uploader_url is None


class TestTotality:
    """Field-level junk degrades to absent; only unusable documents raise."""

    def test_oversized_document_refused(self) -> None:
        blob = b'{"title": "' + b"x" * MAX_INFO_JSON_BYTES + b'"}'
        with pytest.raises(SourceMetadataError):
            extract(blob)

    @pytest.mark.parametrize("blob", [b"not json", b"[1, 2]", b'"a string"', b"null"])
    def test_unusable_documents_refused(self, blob: bytes) -> None:
        with pytest.raises(SourceMetadataError):
            extract(blob)

    def test_wrong_typed_fields_dropped(self) -> None:
        meta = extract(
            _payload(
                {
                    "title": {"nested": "dict"},
                    "uploader": 42,
                    "description": None,
                    "upload_date": "not-a-date",
                    "duration": "3641",
                    "tags": "not-a-list",
                    "_version": "not-a-dict",
                }
            )
        )
        assert meta.title is None
        assert meta.uploader is None
        assert meta.description is None
        assert meta.upload_date is None
        assert meta.duration_seconds is None
        assert meta.tags == ()
        assert meta.extractor_version is None

    @pytest.mark.parametrize("bad", [-1, float("nan"), float("inf"), True])
    def test_bad_durations_dropped(self, bad: object) -> None:
        raw = json.dumps({"duration": bad}, default=str).encode()
        meta = extract(raw)
        assert meta.duration_seconds is None

    def test_nan_and_inf_never_enter_raw(self) -> None:
        # json.dumps writes bare NaN/Infinity tokens by default; python's loads
        # accepts them, so the sanitizer must drop them before JSONB would balk.
        blob = b'{"view_count": NaN, "like_count": Infinity, "duration": 5}'
        meta = extract(blob)
        assert "view_count" not in meta.raw
        assert "like_count" not in meta.raw
        assert meta.raw["duration"] == 5

    def test_text_and_tag_bounds_enforced(self) -> None:
        meta = extract(
            _payload(
                {
                    "title": "t" * (MAX_TEXT_CHARS + 500),
                    "description": "d" * (MAX_DESCRIPTION_CHARS + 500),
                    "tags": ["x" * (MAX_TAG_CHARS + 50)] + ["ok"] * (MAX_TAGS + 50),
                }
            )
        )
        assert meta.title is not None and len(meta.title) == MAX_TEXT_CHARS
        assert meta.description is not None
        assert len(meta.description) == MAX_DESCRIPTION_CHARS
        assert len(meta.tags) == MAX_TAGS
        assert len(meta.tags[0]) == MAX_TAG_CHARS

    def test_control_characters_stripped_newlines_kept(self) -> None:
        meta = extract(_payload({"title": "a\x00b\x1bc", "description": "l1\nl2\tl3"}))
        assert meta.title == "abc"
        assert meta.description == "l1\nl2\tl3"

    def test_whitespace_only_strings_absent(self) -> None:
        meta = extract(_payload({"title": "   ", "uploader": "\n\t"}))
        assert meta.title is None
        assert meta.uploader is None
