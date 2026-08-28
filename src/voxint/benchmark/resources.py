"""Load the bundled benchmark corpus shipped as package data.

The WAV files and metadata live under ``voxint/benchmark/assets/`` and are
packaged into the wheel and Docker image.  Everything is read through
:mod:`importlib.resources`, so the assets work identically from a source
checkout, an installed wheel, and the container.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

_PACKAGE = "voxint.benchmark"
_ASSETS_DIR = "assets"


def _load_json(name: str) -> dict[str, Any] | list[Any]:
    resource = files(_PACKAGE).joinpath(_ASSETS_DIR).joinpath(name)
    result: dict[str, Any] | list[Any] = json.loads(
        resource.read_text(encoding="utf-8")
    )
    return result


def load_manifest() -> dict[str, Any]:
    """The benchmark corpus manifest (file list, references, corpus version)."""
    data = _load_json("manifest.json")
    assert isinstance(data, dict)
    return data


def load_provenance() -> dict[str, Any]:
    """Upstream sources, licenses, and generation details."""
    data = _load_json("provenance.json")
    assert isinstance(data, dict)
    return data


def corpus_file_ids() -> list[str]:
    """Return the ordered list of corpus file IDs from the manifest."""
    manifest = load_manifest()
    return [f["id"] for f in manifest["files"]]


@contextmanager
def corpus_wav_path(file_id: str) -> Iterator[Path]:
    """Yield a real filesystem path to a corpus WAV, extracting if packaged."""
    manifest = load_manifest()
    entry = next((f for f in manifest["files"] if f["id"] == file_id), None)
    if entry is None:
        raise ValueError(f"Unknown corpus file ID: {file_id!r}")
    resource = files(_PACKAGE).joinpath(_ASSETS_DIR).joinpath(entry["filename"])
    with as_file(resource) as path:
        yield path


def corpus_wav_bytes(file_id: str) -> bytes:
    """Return the raw bytes of a corpus WAV for copying into media_root."""
    manifest = load_manifest()
    entry = next((f for f in manifest["files"] if f["id"] == file_id), None)
    if entry is None:
        raise ValueError(f"Unknown corpus file ID: {file_id!r}")
    resource = files(_PACKAGE).joinpath(_ASSETS_DIR).joinpath(entry["filename"])
    return resource.read_bytes()
