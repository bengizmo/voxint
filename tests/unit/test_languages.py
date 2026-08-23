"""The pinned whisper language code -> display name map (issue #124)."""

from voxint.api.languages import LANGUAGE_NAMES, language_label


def test_known_code_gets_name_and_code() -> None:
    assert language_label("es") == "Spanish (es)"
    assert language_label("en") == "English (en)"
    assert language_label("haw") == "Hawaiian (haw)"


def test_unknown_code_falls_back_to_raw_code() -> None:
    # Honest fallback: a future model emitting a code the map predates renders
    # as the code itself, never a guessed name.
    assert language_label("xx") == "xx"


def test_map_covers_faster_whisper_codes() -> None:
    # Pinned to faster-whisper 1.2.1's tokenizer._LANGUAGE_CODES (100 codes,
    # including yue which the pinned large-v2 predates but the engine defines).
    assert len(LANGUAGE_NAMES) == 100
    assert "yue" in LANGUAGE_NAMES
    # Every entry is a non-empty display name for a lowercase code.
    for code, name in LANGUAGE_NAMES.items():
        assert code == code.lower() and code
        assert name and name[0].isupper()
