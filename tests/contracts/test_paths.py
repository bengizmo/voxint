"""MEDIA_ROOT containment tests, run against every service's copy.

The paths module is duplicated across the three services on purpose (each image
is self-contained); the byte-identity test keeps the copies from drifting.
"""

from pathlib import Path
from types import ModuleType

import pytest

from tests.contracts.conftest import SERVICES_DIR, load_service_module

SERVICES = ["whisper", "pyannote", "titanet"]


@pytest.mark.parametrize("dup", ["paths.py", "errors.py"])
def test_duplicated_modules_are_identical(dup: str) -> None:
    contents = {
        svc: (SERVICES_DIR / svc / "app" / dup).read_bytes() for svc in SERVICES
    }
    assert len(set(contents.values())) == 1, f"{dup} copies have drifted: {sorted(contents)}"


@pytest.fixture(params=SERVICES)
def paths_mod(request: pytest.FixtureRequest) -> ModuleType:
    return load_service_module(request.param, "paths")


@pytest.fixture
def media_root(tmp_path: Path) -> Path:
    root = tmp_path / "media"
    root.mkdir()
    (root / "items").mkdir()
    (root / "items" / "audio.wav").write_bytes(b"RIFF")
    return root


def test_valid_relative_path_resolves(paths_mod, media_root: Path) -> None:
    resolved = paths_mod.resolve_media_path(media_root, "items/audio.wav")
    assert resolved == media_root / "items" / "audio.wav"


def test_absolute_path_rejected(paths_mod, media_root: Path) -> None:
    with pytest.raises(paths_mod.PathViolation):
        paths_mod.resolve_media_path(media_root, str(media_root / "items" / "audio.wav"))


def test_traversal_rejected(paths_mod, media_root: Path, tmp_path: Path) -> None:
    (tmp_path / "secret.wav").write_bytes(b"RIFF")
    with pytest.raises(paths_mod.PathViolation):
        paths_mod.resolve_media_path(media_root, "../secret.wav")


def test_symlink_escape_rejected(paths_mod, media_root: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"RIFF")
    (media_root / "link.wav").symlink_to(outside)
    with pytest.raises(paths_mod.PathViolation):
        paths_mod.resolve_media_path(media_root, "link.wav")


def test_missing_file_is_not_found(paths_mod, media_root: Path) -> None:
    with pytest.raises(paths_mod.PathNotFound):
        paths_mod.resolve_media_path(media_root, "items/nope.wav")


def test_directory_rejected(paths_mod, media_root: Path) -> None:
    with pytest.raises(paths_mod.PathViolation):
        paths_mod.resolve_media_path(media_root, "items")


def test_empty_path_rejected(paths_mod, media_root: Path) -> None:
    with pytest.raises(paths_mod.PathViolation):
        paths_mod.resolve_media_path(media_root, "  ")
