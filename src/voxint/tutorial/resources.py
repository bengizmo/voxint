"""Load the bundled guided-tutorial assets shipped as package data.

The WAV and its fixtures live under ``voxint/tutorial/assets/`` and are packaged
into the wheel and Docker image (hatchling ships non-Python files under the
package dir). Everything is read through :mod:`importlib.resources`, so the seed
works identically from a source checkout, an installed wheel, and the container —
none of which is guaranteed to expose the assets as plain filesystem paths.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

_PACKAGE = "voxint.tutorial"
_ASSETS_DIR = "assets"
WAV_NAME = "sample-3speaker.wav"


def _load_json(name: str) -> dict[str, Any]:
    resource = files(_PACKAGE).joinpath(_ASSETS_DIR).joinpath(name)
    data: dict[str, Any] = json.loads(resource.read_text(encoding="utf-8"))
    return data


def load_layout() -> dict[str, Any]:
    """The authored speaker layout (labels, voices, per-utterance timings + text)."""
    return _load_json("utterance.json")


def load_expected_transcript() -> dict[str, Any]:
    """The attributed transcript the seeded run reproduces (test ground truth)."""
    return _load_json("expected-transcript.json")


def load_provenance() -> dict[str, Any]:
    """Tool versions, per-voice synthesis args, and the WAV SHA-256."""
    return _load_json("provenance.json")


def load_sample_wav_bytes() -> bytes:
    """The bundled 16 kHz mono PCM WAV as raw bytes (for copying into media_root)."""
    return files(_PACKAGE).joinpath(_ASSETS_DIR).joinpath(WAV_NAME).read_bytes()


@contextmanager
def sample_wav_path() -> Iterator[Path]:
    """Yield a real filesystem path to the bundled WAV, extracting it if packaged.

    Use only when a caller genuinely needs a path (e.g. ffprobe); the seed copies
    bytes directly via :func:`load_sample_wav_bytes`.
    """
    resource = files(_PACKAGE).joinpath(_ASSETS_DIR).joinpath(WAV_NAME)
    with as_file(resource) as path:
        yield path
