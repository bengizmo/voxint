"""Unit tests for the purge executor."""

from __future__ import annotations

import uuid
from pathlib import Path

from voxint.media.purge import _confined_path


class TestConfinedPath:
    def test_relative_inside_root(self, tmp_path: Path) -> None:
        child = tmp_path / "sub" / "file.wav"
        child.parent.mkdir(parents=True)
        child.touch()
        result = _confined_path(tmp_path, "sub/file.wav")
        assert result is not None
        assert result == child.resolve()

    def test_dotdot_escape_returns_none(self, tmp_path: Path) -> None:
        result = _confined_path(tmp_path, "../escape.wav")
        assert result is None

    def test_absolute_path_returns_none(self, tmp_path: Path) -> None:
        result = _confined_path(tmp_path, "/etc/passwd")
        assert result is None

    def test_symlink_escape_returns_none(self, tmp_path: Path) -> None:
        outside = tmp_path.parent / f"outside_{uuid.uuid4().hex[:8]}"
        outside.mkdir()
        try:
            link = tmp_path / "link"
            link.symlink_to(outside)
            result = _confined_path(tmp_path, "link/file.wav")
            assert result is None
        finally:
            outside.rmdir()
