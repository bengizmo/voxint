"""Unit tests for benchmark corpus asset integrity and the resources loader."""

from __future__ import annotations

import hashlib

import pytest

from voxint.benchmark.resources import (
    corpus_file_ids,
    corpus_wav_bytes,
    corpus_wav_path,
    load_manifest,
    load_provenance,
)


class TestManifest:
    def test_loads(self) -> None:
        manifest = load_manifest()
        assert "corpus_version" in manifest
        assert "scorer_protocol" in manifest
        assert "files" in manifest

    def test_corpus_version_positive_int(self) -> None:
        manifest = load_manifest()
        assert isinstance(manifest["corpus_version"], int)
        assert manifest["corpus_version"] >= 1

    def test_file_count(self) -> None:
        manifest = load_manifest()
        assert len(manifest["files"]) == 12

    def test_required_fields(self) -> None:
        manifest = load_manifest()
        required = {"id", "filename", "sha256", "duration_s", "num_speakers",
                     "category", "reference_transcript", "source", "license_spdx"}
        for entry in manifest["files"]:
            assert required.issubset(entry.keys()), f"Missing fields in {entry['id']}"

    def test_categories(self) -> None:
        manifest = load_manifest()
        categories = {f["category"] for f in manifest["files"]}
        assert categories == {"speech", "silence", "bait"}

    def test_speech_files_have_transcripts(self) -> None:
        manifest = load_manifest()
        for f in manifest["files"]:
            if f["category"] == "speech":
                assert f["reference_transcript"] is not None, f"{f['id']} missing transcript"
                assert len(f["reference_transcript"]) > 0

    def test_nonspech_files_have_null_transcripts(self) -> None:
        manifest = load_manifest()
        for f in manifest["files"]:
            if f["category"] in ("silence", "bait"):
                assert f["reference_transcript"] is None, f"{f['id']} should have null transcript"

    def test_unique_ids(self) -> None:
        manifest = load_manifest()
        ids = [f["id"] for f in manifest["files"]]
        assert len(ids) == len(set(ids))

    def test_licenses(self) -> None:
        manifest = load_manifest()
        for f in manifest["files"]:
            assert f["license_spdx"] in ("CC-BY-4.0", "CC0-1.0")


class TestProvenance:
    def test_loads(self) -> None:
        provenance = load_provenance()
        assert "corpus_version" in provenance
        assert "sources" in provenance

    def test_version_matches_manifest(self) -> None:
        manifest = load_manifest()
        provenance = load_provenance()
        assert manifest["corpus_version"] == provenance["corpus_version"]


class TestCorpusFileIds:
    def test_returns_12_ids(self) -> None:
        ids = corpus_file_ids()
        assert len(ids) == 12

    def test_all_strings(self) -> None:
        for fid in corpus_file_ids():
            assert isinstance(fid, str)


class TestCorpusWavPath:
    def test_valid_id(self) -> None:
        with corpus_wav_path("libri_01_spk1089") as p:
            assert p.exists()
            assert p.suffix == ".wav"

    def test_invalid_id_raises(self) -> None:
        with (
            pytest.raises(ValueError, match="Unknown corpus file ID"),
            corpus_wav_path("nonexistent"),
        ):
            pass


class TestCorpusWavBytes:
    def test_valid_id(self) -> None:
        data = corpus_wav_bytes("libri_01_spk1089")
        assert len(data) > 0
        assert data[:4] == b"RIFF"

    def test_invalid_id_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown corpus file ID"):
            corpus_wav_bytes("nonexistent")


class TestAssetIntegrity:
    """Verify every WAV file matches its manifest sha256."""

    def test_all_sha256s(self) -> None:
        manifest = load_manifest()
        for entry in manifest["files"]:
            data = corpus_wav_bytes(entry["id"])
            actual = hashlib.sha256(data).hexdigest()
            assert actual == entry["sha256"], (
                f"SHA256 mismatch for {entry['id']}: "
                f"expected {entry['sha256']}, got {actual}"
            )
