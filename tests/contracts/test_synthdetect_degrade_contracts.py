"""Contract tests for the synthdetect degrade executor (#144, M1 S5 PR-2b).

Pins the degrade executor's cross-seam contracts:

- ``resolve_clip_path`` exact-once resolution against two roots.
- Container image pin format (``image@sha256:64hex``).
- Raw framing constants agree between ``build_recipe_argv`` and the executor.
"""

from __future__ import annotations

import hashlib
import struct
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import synthdetect_corpus as corpus  # noqa: E402


def _make_pcm_payload(n_samples: int = 16000) -> bytes:
    return struct.pack(
        f"<{n_samples}h",
        *(int(32767 * (0.5 if i % 2 == 0 else -0.5)) for i in range(n_samples)),
    )


def _clip_entry(**over: dict) -> corpus.ClipEntry:
    payload = _make_pcm_payload()
    sha = hashlib.sha256(payload).hexdigest()
    raw: dict = {
        "clip_id": "test-clip",
        "rel_path": "test/test-clip.wav",
        "sha256": sha,
        "duration_s": 1.0,
        "label": "bona_fide",
        "language": "en",
        "license_spdx": "CC-BY-4.0",
        "stratum": "bona_fide|organic|meetingroom",
        "source": "ami",
        "speaker_id": "spk",
        "split": "calibration",
        "generator": None,
        "degradation": None,
        "parent_clip_id": None,
        "acquire": None,
    }
    raw.update(over)
    return corpus.load_manifest({"schema_version": 1, "clips": [raw]}).clips[0]


class TestResolveClipPathContract:
    """resolve_clip_path must resolve each clip in exactly one root (exact-once)."""

    def test_parent_in_parent_root_child_in_degrade_root(
        self, tmp_path: Path
    ) -> None:
        parent_root = tmp_path / "parent"
        degrade_root = tmp_path / "degrade"
        (parent_root / "test" / "turn").mkdir(parents=True)
        (parent_root / "test" / "turn" / "p.wav").write_bytes(b"\x00")
        (degrade_root / "test" / "turn" / "degraded").mkdir(parents=True)
        (degrade_root / "test" / "turn" / "degraded" / "c.wav").write_bytes(b"\x00")

        parent_clip = _clip_entry(
            clip_id="p", rel_path="test/turn/p.wav"
        )
        # Build a combined manifest with both parent and child to get a valid child
        parent_dict = {
            "clip_id": "p", "rel_path": "test/turn/p.wav",
            "sha256": parent_clip.sha256, "duration_s": 1.0,
            "label": "bona_fide", "language": "en",
            "license_spdx": "CC-BY-4.0",
            "stratum": "bona_fide|organic|meetingroom",
            "source": "ami", "speaker_id": "spk",
            "split": "calibration", "generator": None,
            "degradation": None, "parent_clip_id": None, "acquire": None,
        }
        child_dict = {
            **parent_dict,
            "clip_id": "c", "rel_path": "test/turn/degraded/c.wav",
            "degradation": "mp3-cbr48-v1", "parent_clip_id": "p",
            "stratum": "bona_fide|organic|meetingroom|mp3-cbr48-v1",
        }
        combined = corpus.load_manifest(
            {"schema_version": 1, "clips": [parent_dict, child_dict]}
        )
        child_clip = next(c for c in combined.clips if c.clip_id == "c")

        roots = (degrade_root, parent_root)
        assert corpus.resolve_clip_path(parent_clip, roots=roots).parent == (
            parent_root / "test" / "turn"
        )
        assert corpus.resolve_clip_path(child_clip, roots=roots).parent == (
            degrade_root / "test" / "turn" / "degraded"
        )

    def test_same_path_in_both_roots_rejected(self, tmp_path: Path) -> None:
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        for root in (root_a, root_b):
            (root / "test").mkdir(parents=True)
            (root / "test" / "clip.wav").write_bytes(b"\x00")
        clip = _clip_entry(clip_id="dup", rel_path="test/clip.wav")
        with pytest.raises(corpus.CorpusError, match="multiple roots"):
            corpus.resolve_clip_path(clip, roots=(root_a, root_b))

    def test_missing_in_all_roots_rejected(self, tmp_path: Path) -> None:
        clip = _clip_entry(clip_id="gone", rel_path="test/gone.wav")
        with pytest.raises(corpus.CorpusError, match="not found"):
            corpus.resolve_clip_path(clip, roots=(tmp_path,))


class TestContainerImagePinFormat:
    """Container image pin must be ``<repo>@sha256:<64hex>``."""

    @pytest.mark.parametrize("good", [
        "repo/image@sha256:" + "a" * 64,
        "registry.io/org/image@sha256:" + "0" * 64,
    ])
    def test_valid(self, good: str) -> None:
        assert corpus._CONTAINER_IMAGE_RE.match(good)

    @pytest.mark.parametrize("bad", [
        "image:latest",
        "image@sha256:short",
        "@sha256:" + "a" * 64,
    ])
    def test_invalid(self, bad: str) -> None:
        assert not corpus._CONTAINER_IMAGE_RE.match(bad)


class TestRawFramingIdentity:
    """The raw-framing constants used by ``build_recipe_argv`` must agree with
    the executor's expectations: raw s16le in, raw s16le out."""

    def test_raw_input_framing_is_canonical_pcm(self) -> None:
        assert corpus._RAW_INPUT_FRAMING == ("-f", "s16le", "-ar", "16000", "-ac", "1")

    def test_canonical_output_is_pcm_s16le(self) -> None:
        assert corpus._CANONICAL_OUTPUT == (
            "-f", "s16le", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le"
        )

    def test_block_align_matches_s16le(self) -> None:
        assert corpus._BLOCK_ALIGN == 2
