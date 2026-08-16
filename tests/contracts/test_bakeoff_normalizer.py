"""Contract: the whisper-bakeoff text normalizer is a frozen denominator.

Two layers, per the codex-reviewed Slice 1 design:

* ``TestVendorIntegrity`` — ALWAYS runs (pure stdlib, imports nothing from the
  vendored package). It proves the byte-identical upstream files match the
  recorded digests and that the version wiring has a single source of truth. If
  a vendored file drifts by one byte, this fails.
* ``TestNormalizerBehavior`` — runs where the ``parity`` extra is installed
  (SKIP otherwise; the deps are pure-python but pinned there). It locks the
  *observable* normalization via golden input→output vectors — the belt to the
  digest-check's braces, and the thing that would catch a runtime (Python /
  Unicode / dependency) shift the digests alone cannot see.

See ``tests/parity/bakeoff/normalize.py`` and the pre-registered gate in
``docs/gpu-contracts.md``.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import ClassVar

import pytest

REPO = Path(__file__).resolve().parents[2]
BAKEOFF = REPO / "tests" / "parity" / "bakeoff"
VENDOR = BAKEOFF / "_vendor"
PROVENANCE = VENDOR / "provenance.json"
NORMALIZE_PY = BAKEOFF / "normalize.py"


class TestVendorIntegrity:
    """Byte-for-byte binding of the vendored normalizer to its provenance."""

    def test_provenance_shape(self) -> None:
        prov = json.loads(PROVENANCE.read_text())
        assert prov["upstream_repo"] == "openai/whisper"
        assert re.fullmatch(r"[0-9a-f]{40}", prov["upstream_commit"]), (
            "upstream_commit must be a full 40-hex git sha"
        )
        assert prov["license"] == "MIT"
        assert prov["verbatim"] is True

    def test_files_match_digests_exactly(self) -> None:
        prov = json.loads(PROVENANCE.read_text())
        recorded = prov["files"]
        # No extra, unrecorded files hiding in the vendored package, and no
        # recorded file missing — the set must match exactly.
        pkg = VENDOR / "openai_whisper_normalizers"
        on_disk = {
            f"openai_whisper_normalizers/{p.name}"
            for p in pkg.iterdir()
            if p.is_file() and p.name != "__pycache__"
        }
        assert on_disk == set(recorded), (
            f"vendored file set drifted from provenance: {on_disk ^ set(recorded)}"
        )
        for rel, meta in recorded.items():
            blob = (VENDOR / rel).read_bytes()
            assert hashlib.sha256(blob).hexdigest() == meta["sha256"], (
                f"{rel} sha256 mismatch — re-vendor deliberately, do not hand-edit"
            )
            assert len(blob) == meta["size_bytes"], f"{rel} size mismatch"

    def test_license_present_and_referenced(self) -> None:
        prov = json.loads(PROVENANCE.read_text())
        license_rel = f"openai_whisper_normalizers/{prov['license_file']}"
        assert license_rel in prov["files"], "license file must be hashed too"
        assert "MIT License" in (VENDOR / license_rel).read_text()

    def test_version_wiring_single_source_of_truth(self) -> None:
        # The wrapper builds NORMALIZER_VERSION from provenance's upstream_commit
        # rather than repeating it — assert that wiring without importing the
        # module (which needs the parity deps). If someone hardcodes the commit
        # in normalize.py, this guard should notice.
        src = NORMALIZE_PY.read_text()
        assert 'WRAPPER_REVISION = "voxint-wrapper-v1"' in src
        assert 'upstream_commit()' in src
        assert "openai-whisper@{" in src, (
            "NORMALIZER_VERSION should be composed from the provenance commit"
        )


class TestNormalizerBehavior:
    """Golden input→output vectors — the observable frozen denominator."""

    # Verified against the vendored EnglishTextNormalizer at the pinned commit
    # (Python 3.12, Unicode 15.0.0). Each row exercises a deliberate-but-
    # aggressive transform the gate depends on being STABLE, not "correct":
    GOLDEN: ClassVar[list[tuple[str, str]]] = [
        # number-word folding (EnglishNumberNormalizer / more_itertools path)
        ("Five twenty four crates.", "524 crates"),
        ("It was nineteen ninety nine.", "it was 1999"),
        ("THIRTY-TWO oranges and 406 apples.", "32 oranges and 406 apples"),
        # contraction + possessive + title expansion
        ("Don't you think it's Dr. Smith's?", "do not you think it is doctor smith is"),
        # British → American spelling map (english.json)
        ("The colour of the theatre organisation.", "the color of the theater organization"),
        # currency + ordinals preserved
        ("He paid $3.50 for it.", "he paid $3.50 for it"),
        ("That is the 1st, 2nd, and 3rd time.", "that is the 1st 2nd and 3rd time"),
        # filler removal
        ("Um, so, uh, well you know.", "so well you know"),
        # diacritic stripping (unicode / basic normalizer)
        ("Café naïve résumé.", "cafe naive resume"),
        # edge whitespace is stripped by the wrapper (WRAPPER_REVISION behavior)
        ("   leading and trailing   ", "leading and trailing"),
        ("", ""),
    ]

    def test_golden_vectors(self) -> None:
        pytest.importorskip("more_itertools", reason="bakeoff parity extra required")
        pytest.importorskip("regex", reason="bakeoff parity extra required")
        from tests.parity.bakeoff.normalize import normalize_text

        for raw, expected in self.GOLDEN:
            assert normalize_text(raw) == expected, f"normalized {raw!r}"

    def test_version_and_fingerprint(self) -> None:
        pytest.importorskip("more_itertools", reason="bakeoff parity extra required")
        pytest.importorskip("regex", reason="bakeoff parity extra required")
        from tests.parity.bakeoff import normalize

        prov = json.loads(PROVENANCE.read_text())
        expected = f"openai-whisper@{prov['upstream_commit']}/voxint-wrapper-v1"
        assert expected == normalize.NORMALIZER_VERSION
        fp = normalize.runtime_fingerprint()
        assert set(fp) == {"python", "unicodedata_unidata_version", "normalizer_version"}
        assert fp["normalizer_version"] == expected
