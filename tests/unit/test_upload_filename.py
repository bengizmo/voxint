"""Filename policy for browser uploads — the first line of path-traversal defence.

These are pure-function checks (no DB, no filesystem); the route and the
uuid-namespaced parent dir are the second and third lines, exercised in
tests/integration/test_submit_api.py.
"""

import pytest

from voxint.ingest.service import UploadValidationError, sanitize_upload_filename


@pytest.mark.parametrize(
    "name",
    [
        "",  # empty
        "   ",  # whitespace-only collapses to empty
        ".",  # current dir
        "..",  # parent dir
        "a/b.wav",  # forward slash
        "..\\b.wav",  # backslash (windows traversal)
        "/etc/passwd",  # absolute
        "a\x00b.wav",  # NUL
        "a\x01b.wav",  # control char
        "a\x7fb.wav",  # DEL
        "x" * 201 + ".wav",  # over one path component's byte budget
    ],
)
def test_rejects_unsafe_names(name: str) -> None:
    with pytest.raises(UploadValidationError):
        sanitize_upload_filename(name)


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("podcast ep 1.wav", "podcast ep 1.wav"),
        ("  trimmed.wav  ", "trimmed.wav"),  # surrounding whitespace stripped
        (".hidden.wav", ".hidden.wav"),  # a leading dot is not '.'/'..'
        ("naïve-über.m4a", "naïve-über.m4a"),  # non-ASCII kept (bytes under budget)
    ],
)
def test_accepts_and_normalizes_safe_names(given: str, expected: str) -> None:
    assert sanitize_upload_filename(given) == expected
