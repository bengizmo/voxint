"""Unit tests for the pure YAML sidecar layer (issue #104).

Everything here runs without a database or Docker: ``parse_sidecar`` is pure,
``read_sidecar``/``find_sidecar`` touch only ``tmp_path``. The adversarial
cases (aliases, cycles, duplicate keys, non-finite numbers, byte caps) pin the
hold-not-crash contract: every malformed input becomes a ``SidecarError`` with
a plain-language message, never an escaped exception.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from voxint.ingest.sidecar import (
    MAX_NOTES_CHARS,
    MAX_SIDECAR_BYTES,
    MAX_SPEAKER_CHARS,
    MAX_SPEAKERS,
    MAX_TITLE_CHARS,
    Sidecar,
    SidecarError,
    find_sidecar,
    parse_sidecar,
    read_sidecar,
)

# --- parse_sidecar: happy paths -----------------------------------------------


def test_parse_full_sidecar() -> None:
    sc = parse_sidecar(
        """
        title: Interview with Jane Doe
        speakers:
          - Jane Doe
          - John Smith
        domain_pack: hvac
        notes: |
          Recorded at the spring conference.
          Second line.
        """,
        source_name="interview.wav.yaml",
    )
    assert sc.title == "Interview with Jane Doe"
    assert sc.speakers == ("Jane Doe", "John Smith")
    assert sc.domain_pack == "hvac"
    assert sc.notes == "Recorded at the spring conference.\nSecond line."
    assert sc.ignored_keys == ()
    assert sc.raw["title"] == "Interview with Jane Doe"


@pytest.mark.parametrize(
    ("text", "field", "expected"),
    [
        ("title: Just a title", "title", "Just a title"),
        ("speakers: [Solo Speaker]", "speakers", ("Solo Speaker",)),
        ("domain_pack: general", "domain_pack", "general"),
        ("notes: one line", "notes", "one line"),
    ],
)
def test_parse_each_field_alone(text: str, field: str, expected: object) -> None:
    sc = parse_sidecar(text, source_name="x.yaml")
    assert getattr(sc, field) == expected


def test_empty_mapping_is_valid() -> None:
    sc = parse_sidecar("{}", source_name="x.yaml")
    assert sc == Sidecar(
        title=None, speakers=(), domain_pack=None, notes=None, raw={}, ignored_keys=()
    )


def test_blank_document_is_valid_empty() -> None:
    # A blank or comment-only file parses to a None root; that is a stub, not an
    # error — nothing is applied and nothing is held.
    for text in ("", "   \n", "# reference sidecar, nothing for Voxint yet\n"):
        sc = parse_sidecar(text, source_name="x.yaml")
        assert sc.raw == {}
        assert sc.title is None


def test_unknown_keys_only_is_valid_and_preserved() -> None:
    sc = parse_sidecar(
        """
        content_item_id: 12345
        source_type: rss_feed
        url: https://example.com/episode-1
        published: 2026-01-15
        provenance:
          media_filename: episode-1.mp3
        """,
        source_name="episode-1.mp3.yaml",
    )
    assert sc.title is None and sc.speakers == ()
    assert sc.ignored_keys == (
        "content_item_id",
        "provenance",
        "published",
        "source_type",
        "url",
    )
    # Preserved verbatim (dates normalized to ISO strings for JSON storage).
    assert sc.raw["content_item_id"] == 12345
    assert sc.raw["published"] == "2026-01-15"
    assert sc.raw["provenance"] == {"media_filename": "episode-1.mp3"}


def test_known_and_unknown_keys_mix() -> None:
    sc = parse_sidecar(
        "title: T\ndescription: long scraped text\nduration_seconds: 90\n",
        source_name="x.yaml",
    )
    assert sc.title == "T"
    assert sc.ignored_keys == ("description", "duration_seconds")
    # description is deliberately NOT applied to notes.
    assert sc.notes is None


def test_whitespace_is_stripped() -> None:
    sc = parse_sidecar(
        'title: "  padded  "\nspeakers: ["  Jane  "]\nnotes: "  n  "\n',
        source_name="x.yaml",
    )
    assert sc.title == "padded"
    assert sc.speakers == ("Jane",)
    assert sc.notes == "n"


# --- parse_sidecar: document-level errors -------------------------------------


def test_invalid_yaml_is_error() -> None:
    with pytest.raises(SidecarError, match="not valid YAML"):
        parse_sidecar("title: [unclosed", source_name="x.yaml")


@pytest.mark.parametrize("text", ["- a\n- b", "just a string", "42"])
def test_non_mapping_root_is_error(text: str) -> None:
    with pytest.raises(SidecarError, match="mapping"):
        parse_sidecar(text, source_name="x.yaml")


def test_duplicate_keys_are_error() -> None:
    # PyYAML's stock loader silently keeps the last duplicate; here that would
    # quietly drop operator data, so duplicates are rejected loudly.
    with pytest.raises(SidecarError, match=r"not valid YAML|duplicate"):
        parse_sidecar("title: A\ntitle: B\n", source_name="x.yaml")


def test_error_messages_name_the_file() -> None:
    with pytest.raises(SidecarError, match=r"clip\.wav\.yaml"):
        parse_sidecar("title: [unclosed", source_name="clip.wav.yaml")


# --- parse_sidecar: known-key validation --------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "title: 42",
        "title: [a, b]",
        "title:",  # explicit key with no value
        "title: 2026-01-01",  # YAML date, not text
        "speakers: not-a-list",
        "speakers: [42]",
        "speakers: [[nested]]",
        "domain_pack: [x]",
        "domain_pack:",
        "notes: {a: b}",
    ],
    ids=lambda t: t.replace("\n", ";"),
)
def test_bad_known_key_types_are_errors(text: str) -> None:
    with pytest.raises(SidecarError):
        parse_sidecar(text, source_name="x.yaml")


@pytest.mark.parametrize(
    "text",
    ['title: "   "', 'speakers: ["  "]', 'domain_pack: ""', 'notes: "  "'],
)
def test_blank_known_values_are_errors(text: str) -> None:
    with pytest.raises(SidecarError, match="empty"):
        parse_sidecar(text, source_name="x.yaml")


def test_multiline_title_is_error() -> None:
    # \n must be a YAML escape: a literal line break inside a double-quoted
    # scalar folds to a space and would never reach the validator.
    with pytest.raises(SidecarError, match="single line"):
        parse_sidecar('title: "a\\nb"', source_name="x.yaml")


def test_multiline_speaker_is_error() -> None:
    with pytest.raises(SidecarError, match="single line"):
        parse_sidecar('speakers: ["a\\nb"]', source_name="x.yaml")


def test_multiline_notes_are_fine_but_other_controls_rejected() -> None:
    assert parse_sidecar('notes: "a\\nb"', source_name="x.yaml").notes == "a\nb"
    with pytest.raises(SidecarError, match="non-printing"):
        parse_sidecar('notes: "a\\x00b"', source_name="x.yaml")


@pytest.mark.parametrize(
    "text",
    [
        'title: "a\\u200bb"',  # zero-width space (Cf)
        'speakers: ["a\\u202eb"]',  # bidi override (Cf)
        'domain_pack: "a\\tb"',  # tab (Cc) in a single-line field
    ],
)
def test_non_printing_in_single_line_fields_is_error(text: str) -> None:
    with pytest.raises(SidecarError, match="non-printing"):
        parse_sidecar(text, source_name="x.yaml")


def test_bounds_title() -> None:
    ok = "t" * MAX_TITLE_CHARS
    assert parse_sidecar(f"title: {ok}", source_name="x.yaml").title == ok
    with pytest.raises(SidecarError, match=str(MAX_TITLE_CHARS)):
        parse_sidecar(f"title: {'t' * (MAX_TITLE_CHARS + 1)}", source_name="x.yaml")


def test_bounds_speakers_count() -> None:
    ok = yaml.safe_dump({"speakers": [f"s{i}" for i in range(MAX_SPEAKERS)]})
    assert len(parse_sidecar(ok, source_name="x.yaml").speakers) == MAX_SPEAKERS
    over = yaml.safe_dump({"speakers": [f"s{i}" for i in range(MAX_SPEAKERS + 1)]})
    with pytest.raises(SidecarError, match=str(MAX_SPEAKERS)):
        parse_sidecar(over, source_name="x.yaml")


def test_bounds_speaker_chars() -> None:
    with pytest.raises(SidecarError, match=str(MAX_SPEAKER_CHARS)):
        parse_sidecar(
            yaml.safe_dump({"speakers": ["s" * (MAX_SPEAKER_CHARS + 1)]}),
            source_name="x.yaml",
        )


def test_bounds_notes_chars() -> None:
    with pytest.raises(SidecarError, match=str(MAX_NOTES_CHARS)):
        parse_sidecar(
            yaml.safe_dump({"notes": "n" * (MAX_NOTES_CHARS + 1)}),
            source_name="x.yaml",
        )


# --- parse_sidecar: adversarial snapshot normalization -------------------------


def test_dates_and_datetimes_become_iso_strings() -> None:
    sc = parse_sidecar(
        "published: 2026-01-15\nacquired: 2026-01-15 10:30:00\n",
        source_name="x.yaml",
    )
    assert sc.raw["published"] == "2026-01-15"
    assert sc.raw["acquired"].startswith("2026-01-15T10:30:00")


def test_non_string_keys_are_stringified() -> None:
    sc = parse_sidecar("1: one\n2.5: half\n", source_name="x.yaml")
    assert sc.raw == {"1": "one", "2.5": "half"}
    assert sc.ignored_keys == ("1", "2.5")


def test_bool_int_key_collision_is_rejected_at_parse() -> None:
    # YAML `1:` and `true:` collide in a Python dict (True == 1), so the
    # strict loader reports them as duplicates rather than silently dropping
    # one.
    with pytest.raises(SidecarError, match="not valid YAML"):
        parse_sidecar("1: one\ntrue: flag\n", source_name="x.yaml")


def test_stringified_key_collision_is_error() -> None:
    with pytest.raises(SidecarError):
        parse_sidecar('1: int\n"1": str\n', source_name="x.yaml")


def test_non_finite_numbers_are_error() -> None:
    with pytest.raises(SidecarError):
        parse_sidecar("value: .nan", source_name="x.yaml")
    with pytest.raises(SidecarError):
        parse_sidecar("value: .inf", source_name="x.yaml")


def test_recursive_alias_is_error_not_hang() -> None:
    # &a [*a] builds a self-referential list; the sanitizer must detect the
    # cycle rather than recurse forever.
    with pytest.raises(SidecarError):
        parse_sidecar("extra: &a [*a]", source_name="x.yaml")


def test_deep_nesting_is_error() -> None:
    depth = 200
    text = "extra: " + "[" * depth + "]" * depth
    with pytest.raises(SidecarError):
        parse_sidecar(text, source_name="x.yaml")


def test_alias_expansion_bomb_is_bounded() -> None:
    # A billion-laughs-style bomb: 9 levels of 10x alias fan-out stays under
    # the input cap but expands to ~1e9 nodes; the node budget must stop it.
    lines = ["l0: &l0 [x, x, x, x, x, x, x, x, x, x]"]
    for level in range(1, 9):
        prev = f"l{level - 1}"
        refs = ", ".join([f"*{prev}"] * 10)
        lines.append(f"l{level}: &l{level} [{refs}]")
    with pytest.raises(SidecarError):
        parse_sidecar("\n".join(lines), source_name="x.yaml")


def test_binary_scalar_is_stringified_not_crash() -> None:
    sc = parse_sidecar("blob: !!binary aGVsbG8=", source_name="x.yaml")
    assert isinstance(sc.raw["blob"], str)


def test_raw_is_json_serializable() -> None:
    import json

    sc = parse_sidecar(
        "title: T\npublished: 2026-01-15\nnested: {a: [1, 2.5, true, null]}\n",
        source_name="x.yaml",
    )
    json.dumps(sc.raw)  # must not raise


# --- read_sidecar ---------------------------------------------------------------


def test_read_sidecar_ok(tmp_path: Path) -> None:
    p = tmp_path / "clip.wav.yaml"
    p.write_text("title: From disk\n", encoding="utf-8")
    assert read_sidecar(p).title == "From disk"


def test_read_sidecar_missing_is_error(tmp_path: Path) -> None:
    with pytest.raises(SidecarError, match="could not be read"):
        read_sidecar(tmp_path / "absent.yaml")


def test_read_sidecar_too_large_is_error(tmp_path: Path) -> None:
    p = tmp_path / "big.yaml"
    p.write_bytes(b"# " + b"x" * MAX_SIDECAR_BYTES)
    with pytest.raises(SidecarError, match=r"too large|larger"):
        read_sidecar(p)


def test_read_sidecar_invalid_utf8_is_error(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_bytes(b"title: \xff\xfe\n")
    with pytest.raises(SidecarError, match="UTF-8"):
        read_sidecar(p)


def test_read_sidecar_refuses_symlink(tmp_path: Path) -> None:
    target = tmp_path / "real.yaml"
    target.write_text("title: T\n", encoding="utf-8")
    link = tmp_path / "clip.wav.yaml"
    link.symlink_to(target)
    with pytest.raises(SidecarError):
        read_sidecar(link)


def test_read_sidecar_refuses_directory(tmp_path: Path) -> None:
    d = tmp_path / "dir.yaml"
    d.mkdir()
    with pytest.raises(SidecarError):
        read_sidecar(d)


# --- find_sidecar ---------------------------------------------------------------


def test_find_none(tmp_path: Path) -> None:
    media = tmp_path / "clip.wav"
    media.touch()
    assert find_sidecar(media) is None


def test_find_full_name_form(tmp_path: Path) -> None:
    media = tmp_path / "clip.wav"
    media.touch()
    sidecar = tmp_path / "clip.wav.yaml"
    sidecar.write_text("title: T\n", encoding="utf-8")
    assert find_sidecar(media) == sidecar


def test_find_stem_form(tmp_path: Path) -> None:
    media = tmp_path / "clip.wav"
    media.touch()
    sidecar = tmp_path / "clip.yaml"
    sidecar.write_text("title: T\n", encoding="utf-8")
    assert find_sidecar(media) == sidecar


def test_full_name_wins_over_stem(tmp_path: Path) -> None:
    media = tmp_path / "clip.wav"
    media.touch()
    (tmp_path / "clip.yaml").write_text("title: stem\n", encoding="utf-8")
    full = tmp_path / "clip.wav.yaml"
    full.write_text("title: full\n", encoding="utf-8")
    assert find_sidecar(media) == full


def test_full_name_never_falls_back_even_when_symlink(tmp_path: Path) -> None:
    # A present-but-unusable full-name sidecar is returned (and later held by
    # read_sidecar) rather than silently falling back to the stem form.
    media = tmp_path / "clip.wav"
    media.touch()
    (tmp_path / "clip.yaml").write_text("title: stem\n", encoding="utf-8")
    full = tmp_path / "clip.wav.yaml"
    full.symlink_to(tmp_path / "clip.yaml")
    assert find_sidecar(media) == full


def test_stem_ambiguity_with_other_media_is_error(tmp_path: Path) -> None:
    media = tmp_path / "clip.wav"
    media.touch()
    (tmp_path / "clip.mp4").touch()  # same stem, another media suffix
    (tmp_path / "clip.yaml").write_text("title: T\n", encoding="utf-8")
    with pytest.raises(SidecarError, match=r"clip\.wav\.yaml"):
        # The error names the fix: rename to the full form.
        find_sidecar(media)


def test_stem_ambiguity_suffix_case_insensitive(tmp_path: Path) -> None:
    media = tmp_path / "clip.wav"
    media.touch()
    (tmp_path / "clip.MP4").touch()
    (tmp_path / "clip.yaml").write_text("title: T\n", encoding="utf-8")
    with pytest.raises(SidecarError):
        find_sidecar(media)


def test_stem_non_media_sibling_is_not_ambiguous(tmp_path: Path) -> None:
    media = tmp_path / "clip.wav"
    media.touch()
    (tmp_path / "clip.txt").touch()  # not a media suffix — irrelevant
    sidecar = tmp_path / "clip.yaml"
    sidecar.write_text("title: T\n", encoding="utf-8")
    assert find_sidecar(media) == sidecar


def test_stem_different_stem_sibling_is_not_ambiguous(tmp_path: Path) -> None:
    media = tmp_path / "clip.wav"
    media.touch()
    (tmp_path / "other.mp4").touch()
    sidecar = tmp_path / "clip.yaml"
    sidecar.write_text("title: T\n", encoding="utf-8")
    assert find_sidecar(media) == sidecar


def test_yml_extension_is_not_recognized(tmp_path: Path) -> None:
    media = tmp_path / "clip.wav"
    media.touch()
    (tmp_path / "clip.wav.yml").write_text("title: T\n", encoding="utf-8")
    (tmp_path / "clip.yml").write_text("title: T\n", encoding="utf-8")
    assert find_sidecar(media) is None


def test_find_unreadable_directory_is_error(tmp_path: Path) -> None:
    if os.geteuid() == 0:
        pytest.skip("permission checks are bypassed as root")
    sub = tmp_path / "locked"
    sub.mkdir()
    media = sub / "clip.wav"
    media.touch()
    (sub / "clip.yaml").write_text("title: T\n", encoding="utf-8")
    sub.chmod(0o000)
    try:
        with pytest.raises(SidecarError):
            find_sidecar(media)
    finally:
        sub.chmod(0o755)
