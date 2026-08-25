"""Unit tests for the ASVspoof 2021 DF importer emission verb (#144, S3).

Covers the audio-dependent half: archive verification, safe extraction of the
native tree, the canonical-PCM transcode, and the v2 imported-benchmark manifest
plus receipt. The fail-closed paths are exercised heavily because the whole point
of the verb is a cryptographic one-to-one chain from a pinned archive byte to a
scored canonical PCM; a manifest that validates while pointing a trial id at the
wrong audio is the failure a paired Gate-2 comparison could not catch.

The tests that touch real audio build tiny FLAC fixtures with ffmpeg and are
skipped when ffmpeg/ffprobe are absent (CI installs them); the archive-, schema-,
and serialization-level tests run everywhere.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import synthdetect_df_import as di  # noqa: E402
from synthdetect_corpus import load_manifest  # noqa: E402
from synthdetect_infer import CanonicalAudio  # noqa: E402

_HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #
def _row(
    *,
    trial_id: str,
    speaker: str = "LA_0001",
    codec: str = "nocodec",
    source: str = "asvspoof",
    attack: str = "A07",
    label: str = "spoof",
    split: str = "eval",
    vocoder: str = "traditional_vocoder",
) -> str:
    """One 13-column official trial row (trailing task/team/gender columns = dash)."""
    tail = ["notrim", split, vocoder, "-", "-", "-", "-"]
    return " ".join([speaker, trial_id, codec, source, attack, label, *tail])


def _record(**kwargs: str) -> di.TrialRecord:
    """A TrialRecord parsed from one row, so field mapping matches production."""
    (rec,) = di.parse_trial_metadata(_row(**kwargs))
    return rec


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _keys_tar(path: Path, rows: list[str]) -> str:
    """Write a keys tar.gz holding the metadata member; return its sha256."""
    meta = ("\n".join(rows) + "\n").encode("utf-8")
    with tarfile.open(path, "w:gz") as tf:
        info = tarfile.TarInfo(di.KEYS_METADATA_MEMBER)
        info.size = len(meta)
        tf.addfile(info, io.BytesIO(meta))
    return di._sha256_file(path)


def _make_flac(path: Path, *, freq: int, dur: float = 0.1) -> None:
    """Synthesize a 16 kHz mono 16-bit FLAC via ffmpeg."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i",
            f"sine=frequency={freq}:sample_rate=16000:duration={dur}",
            "-ac", "1", "-sample_fmt", "s16", "-c:a", "flac", str(path),
        ],
        check=True,
    )


def _audio_tar(path: Path, flac_dir: Path, trial_ids: list[str]) -> str:
    """Pack the given flacs under ASVspoof2021_DF_eval/flac/; return the tar sha256."""
    with tarfile.open(path, "w:gz") as tf:
        for tid in trial_ids:
            src = flac_dir / f"{tid}.flac"
            tf.add(src, arcname=f"{di.NATIVE_TREE_ROOT}/{di.NATIVE_FLAC_SUBDIR}/{tid}.flac")
    return di._sha256_file(path)


def _tar_with_members(path: Path, members: list[tuple[tarfile.TarInfo, bytes | None]]) -> str:
    """Write a tar.gz from crafted (TarInfo, payload) pairs; return its sha256."""
    with tarfile.open(path, "w:gz") as tf:
        for info, payload in members:
            tf.addfile(info, io.BytesIO(payload) if payload is not None else None)
    return di._sha256_file(path)


# --------------------------------------------------------------------------- #
# verify_archive_sha
# --------------------------------------------------------------------------- #
def test_parse_rejects_unsafe_trial_id() -> None:
    """A trial id that is not a safe token is rejected before it becomes a path."""
    with pytest.raises(di.DfImportError, match="not a safe token"):
        di.parse_trial_metadata(_row(trial_id="DF_E_1/../escape"))


# --------------------------------------------------------------------------- #
# _ffprobe_source — monkeypatched ffprobe JSON (no real audio needed)
# --------------------------------------------------------------------------- #
def _fake_ffprobe(
    monkeypatch: pytest.MonkeyPatch, *, returncode: int = 0, stdout: str = ""
) -> None:
    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr="probe error")

    monkeypatch.setattr(di.subprocess, "run", fake_run)


def _stream(**over: object) -> str:
    s = {"codec_name": "flac", "sample_rate": "16000", "channels": 1, "sample_fmt": "s16",
         "bits_per_raw_sample": "16"}
    s.update(over)
    return json.dumps({"streams": [s]})


def test_ffprobe_rejects_missing_bit_depth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An 8/12-bit FLAC decodes to s16; absent bits_per_raw_sample must fail closed."""
    s = json.loads(_stream())
    del s["streams"][0]["bits_per_raw_sample"]
    _fake_ffprobe(monkeypatch, stdout=json.dumps(s))
    with pytest.raises(di.DfImportError, match="bits_per_raw_sample must be 16"):
        di._ffprobe_source(tmp_path / "x.flac")


def test_ffprobe_rejects_zero_streams(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_ffprobe(monkeypatch, stdout=json.dumps({"streams": []}))
    with pytest.raises(di.DfImportError, match="exactly 1 audio stream"):
        di._ffprobe_source(tmp_path / "x.flac")


def test_ffprobe_rejects_probe_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_ffprobe(monkeypatch, returncode=1, stdout="")
    with pytest.raises(di.DfImportError, match="ffprobe failed"):
        di._ffprobe_source(tmp_path / "x.flac")


def test_ffprobe_rejects_unparseable_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_ffprobe(monkeypatch, stdout="not json")
    with pytest.raises(di.DfImportError, match="unparseable"):
        di._ffprobe_source(tmp_path / "x.flac")


def test_verify_archive_sha_matches(tmp_path: Path) -> None:
    p = tmp_path / "DF-keys-full.tar.gz"
    sha = _keys_tar(p, [_row(trial_id="DF_E_1")])
    assert di.verify_archive_sha(p, {"DF-keys-full.tar.gz": sha}) == sha


def test_verify_archive_sha_rejects_mismatch(tmp_path: Path) -> None:
    p = tmp_path / "DF-keys-full.tar.gz"
    _keys_tar(p, [_row(trial_id="DF_E_1")])
    with pytest.raises(di.DfImportError, match="does not match the pinned"):
        di.verify_archive_sha(p, {"DF-keys-full.tar.gz": "0" * 64})


def test_verify_archive_sha_rejects_unpinned_name(tmp_path: Path) -> None:
    p = tmp_path / "surprise.tar.gz"
    sha = _keys_tar(p, [_row(trial_id="DF_E_1")])
    with pytest.raises(di.DfImportError, match="no pinned sha256"):
        di.verify_archive_sha(p, {"DF-keys-full.tar.gz": sha})


# --------------------------------------------------------------------------- #
# _read_keys_metadata
# --------------------------------------------------------------------------- #
def test_read_keys_metadata_returns_member_bytes(tmp_path: Path) -> None:
    p = tmp_path / "DF-keys-full.tar.gz"
    sha = _keys_tar(p, [_row(trial_id="DF_E_1"), _row(trial_id="DF_E_2")])
    data, got_sha = di._read_keys_metadata(p, {"DF-keys-full.tar.gz": sha})
    assert got_sha == sha
    assert b"DF_E_1" in data and b"DF_E_2" in data


def test_read_keys_metadata_verifies_sha_from_same_open(tmp_path: Path) -> None:
    p = tmp_path / "DF-keys-full.tar.gz"
    _keys_tar(p, [_row(trial_id="DF_E_1")])
    with pytest.raises(di.DfImportError, match="does not match the pinned"):
        di._read_keys_metadata(p, {"DF-keys-full.tar.gz": "0" * 64})


def test_read_keys_metadata_rejects_missing_member(tmp_path: Path) -> None:
    p = tmp_path / "DF-keys-full.tar.gz"
    payload = b"nope\n"
    info = tarfile.TarInfo("keys/DF/CM/other.txt")
    info.size = len(payload)
    sha = _tar_with_members(p, [(info, payload)])
    with pytest.raises(di.DfImportError, match="missing member"):
        di._read_keys_metadata(p, {"DF-keys-full.tar.gz": sha})


# --------------------------------------------------------------------------- #
# _safe_extract
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAVE_FFMPEG, reason="ffmpeg/ffprobe not installed")
def test_safe_extract_merges_split_parts(tmp_path: Path) -> None:
    flacs = tmp_path / "flacs"
    for i, tid in enumerate(["DF_E_1", "DF_E_2", "DF_E_3"]):
        _make_flac(flacs / f"{tid}.flac", freq=300 + i * 20)
    a0 = tmp_path / "part00.tar.gz"
    a1 = tmp_path / "part01.tar.gz"
    s0 = _audio_tar(a0, flacs, ["DF_E_1", "DF_E_2"])
    s1 = _audio_tar(a1, flacs, ["DF_E_3"])
    dest = tmp_path / "native"
    dest.mkdir()
    shas = di._safe_extract([a0, a1], dest, {a0.name: s0, a1.name: s1})
    assert shas == {a0.name: s0, a1.name: s1}
    flac_dir = dest / di.NATIVE_TREE_ROOT / di.NATIVE_FLAC_SUBDIR
    assert {p.name for p in flac_dir.glob("*.flac")} == {
        "DF_E_1.flac", "DF_E_2.flac", "DF_E_3.flac",
    }


def _regfile(name: str, payload: bytes) -> tuple[tarfile.TarInfo, bytes]:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    return info, payload


def _one_archive(path: Path, members: list[tuple[tarfile.TarInfo, bytes | None]]) -> dict[str, str]:
    """Build a one-off archive and return the {name: sha} provenance for it."""
    return {path.name: _tar_with_members(path, members)}


def test_safe_extract_rejects_unpinned_archive(tmp_path: Path) -> None:
    p = tmp_path / "part00.tar.gz"
    _tar_with_members(p, [_regfile(f"{di.NATIVE_TREE_ROOT}/x", b"x")])
    with pytest.raises(di.DfImportError, match="no pinned sha256"):
        di._safe_extract([p], tmp_path / "native", {})


def test_safe_extract_rejects_absolute_path(tmp_path: Path) -> None:
    p = tmp_path / "evil.tar.gz"
    prov = _one_archive(p, [_regfile("/etc/passwd", b"x")])
    with pytest.raises(di.DfImportError, match="absolute path"):
        di._safe_extract([p], tmp_path / "native", prov)


def test_safe_extract_rejects_traversal(tmp_path: Path) -> None:
    p = tmp_path / "evil.tar.gz"
    prov = _one_archive(p, [_regfile("ASVspoof2021_DF_eval/../escape", b"x")])
    with pytest.raises(di.DfImportError, match="traverses"):
        di._safe_extract([p], tmp_path / "native", prov)


def test_safe_extract_rejects_member_outside_root(tmp_path: Path) -> None:
    p = tmp_path / "evil.tar.gz"
    prov = _one_archive(p, [_regfile("some_other_tree/x.flac", b"x")])
    with pytest.raises(di.DfImportError, match="outside"):
        di._safe_extract([p], tmp_path / "native", prov)


def test_safe_extract_rejects_symlink(tmp_path: Path) -> None:
    p = tmp_path / "evil.tar.gz"
    link = tarfile.TarInfo(f"{di.NATIVE_TREE_ROOT}/link")
    link.type = tarfile.SYMTYPE
    link.linkname = "/etc/passwd"
    prov = _one_archive(p, [(link, None)])
    with pytest.raises(di.DfImportError, match="is a link"):
        di._safe_extract([p], tmp_path / "native", prov)


def test_safe_extract_rejects_hardlink(tmp_path: Path) -> None:
    p = tmp_path / "evil.tar.gz"
    link = tarfile.TarInfo(f"{di.NATIVE_TREE_ROOT}/hard")
    link.type = tarfile.LNKTYPE
    link.linkname = f"{di.NATIVE_TREE_ROOT}/flac/DF_E_1.flac"
    prov = _one_archive(p, [(link, None)])
    with pytest.raises(di.DfImportError, match="is a link"):
        di._safe_extract([p], tmp_path / "native", prov)


def test_safe_extract_rejects_device(tmp_path: Path) -> None:
    p = tmp_path / "evil.tar.gz"
    dev = tarfile.TarInfo(f"{di.NATIVE_TREE_ROOT}/dev")
    dev.type = tarfile.CHRTYPE
    dev.devmajor, dev.devminor = 1, 3
    prov = _one_archive(p, [(dev, None)])
    with pytest.raises(di.DfImportError, match="special device"):
        di._safe_extract([p], tmp_path / "native", prov)


def test_safe_extract_rejects_duplicate_flac_across_parts(tmp_path: Path) -> None:
    name = f"{di.NATIVE_TREE_ROOT}/{di.NATIVE_FLAC_SUBDIR}/DF_E_1.flac"
    a0 = tmp_path / "part00.tar.gz"
    a1 = tmp_path / "part01.tar.gz"
    s0 = _tar_with_members(a0, [_regfile(name, b"one")])
    s1 = _tar_with_members(a1, [_regfile(name, b"two")])
    with pytest.raises(di.DfImportError, match="duplicate member"):
        di._safe_extract([a0, a1], tmp_path / "native", {a0.name: s0, a1.name: s1})


def test_safe_extract_rejects_normalized_duplicate(tmp_path: Path) -> None:
    """A ``/./`` spelling must not slip a second payload past the duplicate check."""
    base = f"{di.NATIVE_TREE_ROOT}/{di.NATIVE_FLAC_SUBDIR}"
    a0 = tmp_path / "part00.tar.gz"
    a1 = tmp_path / "part01.tar.gz"
    s0 = _tar_with_members(a0, [_regfile(f"{base}/DF_E_1.flac", b"one")])
    s1 = _tar_with_members(a1, [_regfile(f"{base}/./DF_E_1.flac", b"two")])
    with pytest.raises(di.DfImportError, match="duplicate member"):
        di._safe_extract([a0, a1], tmp_path / "native", {a0.name: s0, a1.name: s1})


# --------------------------------------------------------------------------- #
# build_clip
# --------------------------------------------------------------------------- #
def _audio(sha_seed: str = "a", n: int = 1600) -> CanonicalAudio:
    return CanonicalAudio(
        samples=np.zeros(1, dtype=np.int16),
        pcm_sha256=hashlib.sha256(sha_seed.encode()).hexdigest(),
        n_samples=n,
    )


def test_build_clip_spoof_carries_attack_and_vocoder() -> None:
    rec = _record(
        trial_id="DF_E_1", label="spoof", codec="low_mp3", attack="A09", vocoder="unknown"
    )
    clip = di.build_clip(rec, _audio(), rel_path="canonical/DF_E_1.wav")
    assert clip["label"] == "spoof"
    assert clip["stratum"] == "spoof|low_mp3"
    prov = clip["imported_provenance"]
    assert prov["attack_system"] == "A09"
    assert prov["vocoder_family"] == "unknown"  # a real official family, not a placeholder
    assert prov["official_trial_id"] == "DF_E_1"
    assert clip["duration_s"] == 1600 / 16000


def test_build_clip_bonafide_maps_label_and_nulls_attack() -> None:
    rec = _record(
        trial_id="DF_E_2", label="bonafide", codec="nocodec", attack="-", vocoder="bonafide"
    )
    clip = di.build_clip(rec, _audio(), rel_path="canonical/DF_E_2.wav")
    assert clip["label"] == "bona_fide"  # schema vocabulary, underscore
    assert clip["stratum"] == "bona_fide|nocodec"
    prov = clip["imported_provenance"]
    assert prov["attack_system"] is None
    assert prov["vocoder_family"] == "bonafide"


def test_build_clip_rejects_zero_frames() -> None:
    rec = _record(trial_id="DF_E_1")
    with pytest.raises(di.DfImportError, match="no frames"):
        di.build_clip(rec, _audio(n=0), rel_path="canonical/DF_E_1.wav")


def test_build_clip_output_validates_as_v2_clip() -> None:
    from synthdetect_corpus import CORPUS_KIND_IMPORTED, validate_clip

    rec = _record(trial_id="DF_E_1", label="spoof", codec="high_ogg", attack="A11")
    clip = di.build_clip(rec, _audio(), rel_path="canonical/DF_E_1.wav")
    entry = validate_clip(clip, 0, corpus_kind=CORPUS_KIND_IMPORTED)
    assert entry.clip_id == "DF_E_1"
    assert entry.imported_provenance is not None
    assert entry.generator is None


# --------------------------------------------------------------------------- #
# serialization
# --------------------------------------------------------------------------- #
def _two_clips() -> list[dict[str, object]]:
    a = di.build_clip(
        _record(trial_id="DF_E_1", label="spoof", attack="A07"),
        _audio("x"),
        rel_path="canonical/DF_E_1.wav",
    )
    b = di.build_clip(
        _record(trial_id="DF_E_2", label="bonafide", attack="-", vocoder="bonafide"),
        _audio("y"),
        rel_path="canonical/DF_E_2.wav",
    )
    return [a, b]


def test_serialize_manifest_is_deterministic_and_loads() -> None:
    clips = _two_clips()
    b1 = di.serialize_manifest(clips, benchmark="asvspoof2021_df")
    b2 = di.serialize_manifest(clips, benchmark="asvspoof2021_df")
    assert b1 == b2  # byte-stable: the runner hashes these exact bytes
    assert b1.endswith(b"\n")
    manifest = load_manifest(json.loads(b1.decode("utf-8")))
    assert manifest.schema_version == 2
    assert manifest.corpus_kind == "imported_benchmark"
    assert manifest.benchmark == "asvspoof2021_df"
    assert [c.clip_id for c in manifest.clips] == ["DF_E_1", "DF_E_2"]


def test_serialize_manifest_preserves_given_clip_order() -> None:
    clips = list(reversed(_two_clips()))
    manifest = load_manifest(json.loads(di.serialize_manifest(clips, benchmark="b").decode()))
    assert [c.clip_id for c in manifest.clips] == ["DF_E_2", "DF_E_1"]


def test_serialize_trial_list_emits_raw_rows_lf_terminated() -> None:
    r1 = _record(trial_id="DF_E_1")
    r2 = _record(trial_id="DF_E_2")
    out = di.serialize_trial_list({"DF_E_1": r1, "DF_E_2": r2}, ["DF_E_1", "DF_E_2"])
    assert out == (r1.raw + "\n" + r2.raw + "\n").encode("utf-8")


# --------------------------------------------------------------------------- #
# emit_subset — full end to end (ffmpeg)
# --------------------------------------------------------------------------- #
def _emit_fixture(tmp_path: Path) -> tuple[Path, list[Path], dict[str, str], list[str]]:
    """Build a keys archive + one audio archive covering 20 eval trials.

    Ten bona fide and ten spoof trials share the ``nocodec`` codec, so the
    seeded 10 % selection keeps exactly one per stratum (two clips total).
    """
    rows: list[str] = []
    trial_ids: list[str] = []
    flacs = tmp_path / "flacs"
    for i in range(10):
        tid = f"DF_E_{2000000 + i}"
        rows.append(_row(trial_id=tid, speaker=f"SPK{i:03d}", label="bonafide",
                         attack="-", vocoder="bonafide"))
        trial_ids.append(tid)
    for i in range(10, 20):
        tid = f"DF_E_{2000000 + i}"
        rows.append(_row(trial_id=tid, speaker=f"SPK{i:03d}", label="spoof",
                         attack="A07", vocoder="traditional_vocoder"))
        trial_ids.append(tid)
    for i, tid in enumerate(trial_ids):
        _make_flac(flacs / f"{tid}.flac", freq=200 + i * 15)

    keys = tmp_path / "DF-keys-full.tar.gz"
    keys_sha = _keys_tar(keys, rows)
    audio = tmp_path / "ASVspoof2021_DF_eval_part00.tar.gz"
    audio_sha = _audio_tar(audio, flacs, trial_ids)
    prov = {keys.name: keys_sha, audio.name: audio_sha}
    return keys, [audio], prov, trial_ids


@pytest.mark.skipif(not _HAVE_FFMPEG, reason="ffmpeg/ffprobe not installed")
def test_emit_subset_end_to_end(tmp_path: Path) -> None:
    keys, audios, prov, _ = _emit_fixture(tmp_path)
    out_dir = tmp_path / "corpus"
    native_root = tmp_path / "native"
    result = di.emit_subset(
        keys_archive=keys, audio_archives=audios,
        native_root=native_root, out_dir=out_dir, expected_sha256=prov,
    )
    assert result.n_selected == 2

    # Manifest loads, is v2, ordered, and both labels are represented.
    manifest_bytes = (out_dir / "manifest.json").read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == result.manifest_sha256
    manifest = load_manifest(json.loads(manifest_bytes.decode("utf-8")))
    assert manifest.corpus_kind == "imported_benchmark"
    clip_ids = [c.clip_id for c in manifest.clips]
    assert clip_ids == sorted(clip_ids)
    assert {c.label for c in manifest.clips} == {"bona_fide", "spoof"}

    # Native tree is preserved and every canonical wav re-reads to the manifest sha.
    flac_dir = native_root / di.NATIVE_TREE_ROOT / di.NATIVE_FLAC_SUBDIR
    assert len(list(flac_dir.glob("*.flac"))) == 20
    from synthdetect_infer import read_canonical_pcm

    for clip in manifest.clips:
        audio = read_canonical_pcm(out_dir / clip.rel_path)
        assert audio.pcm_sha256 == clip.sha256
        assert round(clip.duration_s * 16000) == audio.n_samples


@pytest.mark.skipif(not _HAVE_FFMPEG, reason="ffmpeg/ffprobe not installed")
def test_emit_subset_binds_each_trial_to_its_own_audio(tmp_path: Path) -> None:
    """The anti-cross-wiring invariant: clip sha == transcode of THAT trial's flac."""
    keys, audios, prov, _ = _emit_fixture(tmp_path)
    out_dir = tmp_path / "corpus"
    native_root = tmp_path / "native"
    di.emit_subset(
        keys_archive=keys, audio_archives=audios,
        native_root=native_root, out_dir=out_dir, expected_sha256=prov,
    )
    manifest = load_manifest(json.loads((out_dir / "manifest.json").read_bytes().decode()))
    flac_dir = native_root / di.NATIVE_TREE_ROOT / di.NATIVE_FLAC_SUBDIR
    for clip in manifest.clips:
        native_flac = flac_dir / f"{clip.clip_id}.flac"
        tmp_wav = tmp_path / f"recheck_{clip.clip_id}.wav"
        di._transcode_to_canonical(native_flac, tmp_wav)
        from synthdetect_infer import read_canonical_pcm

        assert read_canonical_pcm(tmp_wav).pcm_sha256 == clip.sha256

    # The per-trial receipt binds native sha -> canonical sha for each clip.
    receipts = [
        json.loads(line)
        for line in (out_dir / "clip_receipt.jsonl").read_text().splitlines()
    ]
    assert len(receipts) == 2
    for r in receipts:
        native_flac = flac_dir / f"{r['official_trial_id']}.flac"
        assert r["native_flac_sha256"] == di._sha256_file(native_flac)


@pytest.mark.skipif(not _HAVE_FFMPEG, reason="ffmpeg/ffprobe not installed")
def test_emit_subset_receipt_records_provenance(tmp_path: Path) -> None:
    keys, audios, prov, _ = _emit_fixture(tmp_path)
    out_dir = tmp_path / "corpus"
    di.emit_subset(
        keys_archive=keys, audio_archives=audios,
        native_root=tmp_path / "native", out_dir=out_dir, expected_sha256=prov,
    )
    receipt = json.loads((out_dir / "selection_receipt.json").read_bytes().decode())
    assert receipt["canonicalization_id"] == "pcm-s16le-mono-16000-v1"
    assert receipt["benchmark"] == "asvspoof2021_df"
    assert receipt["n_selected"] == 2
    assert receipt["audio_archive_sha256"] == {audios[0].name: prov[audios[0].name]}
    assert "ffmpeg_version" in receipt and receipt["ffmpeg_version"].startswith("ffmpeg")


def test_emit_subset_refuses_existing_out_dir(tmp_path: Path) -> None:
    keys = tmp_path / "DF-keys-full.tar.gz"
    keys_sha = _keys_tar(keys, [_row(trial_id="DF_E_1")])
    out_dir = tmp_path / "corpus"
    out_dir.mkdir()
    (out_dir / "stale.txt").write_text("x")
    with pytest.raises(di.DfImportError, match="existing path"):
        di.emit_subset(
            keys_archive=keys, audio_archives=[],
            native_root=tmp_path / "native", out_dir=out_dir,
            expected_sha256={keys.name: keys_sha},
        )


def test_emit_subset_refuses_existing_empty_out_dir(tmp_path: Path) -> None:
    """Even an EMPTY pre-existing destination is refused (atomic-publish guard)."""
    keys = tmp_path / "DF-keys-full.tar.gz"
    keys_sha = _keys_tar(keys, [_row(trial_id="DF_E_1")])
    out_dir = tmp_path / "corpus"
    out_dir.mkdir()  # empty
    with pytest.raises(di.DfImportError, match="existing path"):
        di.emit_subset(
            keys_archive=keys, audio_archives=[],
            native_root=tmp_path / "native", out_dir=out_dir,
            expected_sha256={keys.name: keys_sha},
        )


def test_emit_subset_refuses_same_native_and_out_dir(tmp_path: Path) -> None:
    keys = tmp_path / "DF-keys-full.tar.gz"
    keys_sha = _keys_tar(keys, [_row(trial_id="DF_E_1")])
    same = tmp_path / "shared"
    with pytest.raises(di.DfImportError, match="different paths"):
        di.emit_subset(
            keys_archive=keys, audio_archives=[],
            native_root=same, out_dir=same,
            expected_sha256={keys.name: keys_sha},
        )


@pytest.mark.skipif(not _HAVE_FFMPEG, reason="ffmpeg/ffprobe not installed")
def test_emit_subset_leaves_nothing_on_failure(tmp_path: Path) -> None:
    """A selected trial with no native flac fails closed and publishes nothing."""
    keys, _, prov, _ = _emit_fixture(tmp_path)
    # An audio archive that contains the tree dirs but zero flacs: every selected
    # trial's flac is missing, so emission raises after extraction.
    empty_audio = tmp_path / "ASVspoof2021_DF_eval_part00.tar.gz"
    info = tarfile.TarInfo(f"{di.NATIVE_TREE_ROOT}/{di.NATIVE_FLAC_SUBDIR}/")
    info.type = tarfile.DIRTYPE
    prov[empty_audio.name] = _tar_with_members(empty_audio, [(info, None)])
    out_dir = tmp_path / "corpus"
    native_root = tmp_path / "native"
    with pytest.raises(di.DfImportError, match="native flac missing"):
        di.emit_subset(
            keys_archive=keys, audio_archives=[empty_audio],
            native_root=native_root, out_dir=out_dir, expected_sha256=prov,
        )
    assert not out_dir.exists()
    assert not native_root.exists()
    assert not di._staging_for(out_dir).exists()
    assert not di._staging_for(native_root).exists()


@pytest.mark.skipif(not _HAVE_FFMPEG, reason="ffmpeg/ffprobe not installed")
def test_emit_subset_rolls_back_when_second_publish_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the corpus publishes but the native publish fails, neither root survives."""
    keys, audios, prov, _ = _emit_fixture(tmp_path)
    out_dir = tmp_path / "corpus"
    native_root = tmp_path / "native"
    real_replace = os.replace

    def flaky_replace(src: object, dst: object, *a: object, **k: object) -> None:
        if Path(dst) == native_root:  # the second of the two publishing renames
            raise OSError("simulated failure publishing the native tree")
        real_replace(src, dst)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "replace", flaky_replace)
    with pytest.raises(OSError, match="simulated failure"):
        di.emit_subset(
            keys_archive=keys, audio_archives=audios,
            native_root=native_root, out_dir=out_dir, expected_sha256=prov,
        )
    assert not out_dir.exists()  # rolled back after the native publish failed
    assert not native_root.exists()
    assert not di._staging_for(out_dir).exists()
    assert not di._staging_for(native_root).exists()


# --------------------------------------------------------------------------- #
# _ffprobe_source — the "no silent resample" fail-closed gates
# --------------------------------------------------------------------------- #
def _make_audio(path: Path, *, rate: int, channels: int, sample_fmt: str, codec: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate={rate}:duration=0.1",
            "-ac", str(channels), "-sample_fmt", sample_fmt, "-c:a", codec, str(path),
        ],
        check=True,
    )


@pytest.mark.skipif(not _HAVE_FFMPEG, reason="ffmpeg/ffprobe not installed")
def test_ffprobe_accepts_canonical_source(tmp_path: Path) -> None:
    src = tmp_path / "ok.flac"
    _make_audio(src, rate=16000, channels=1, sample_fmt="s16", codec="flac")
    di._ffprobe_source(src)  # no raise


@pytest.mark.skipif(not _HAVE_FFMPEG, reason="ffmpeg/ffprobe not installed")
def test_ffprobe_rejects_non_16k(tmp_path: Path) -> None:
    src = tmp_path / "hi.flac"
    _make_audio(src, rate=48000, channels=1, sample_fmt="s16", codec="flac")
    with pytest.raises(di.DfImportError, match="16000 Hz"):
        di._ffprobe_source(src)


@pytest.mark.skipif(not _HAVE_FFMPEG, reason="ffmpeg/ffprobe not installed")
def test_ffprobe_rejects_stereo(tmp_path: Path) -> None:
    src = tmp_path / "stereo.flac"
    _make_audio(src, rate=16000, channels=2, sample_fmt="s16", codec="flac")
    with pytest.raises(di.DfImportError, match="mono"):
        di._ffprobe_source(src)


@pytest.mark.skipif(not _HAVE_FFMPEG, reason="ffmpeg/ffprobe not installed")
def test_ffprobe_rejects_non_s16_depth(tmp_path: Path) -> None:
    """A wider-than-16-bit FLAC decodes to a non-s16 sample_fmt and is rejected."""
    src = tmp_path / "deep.flac"
    _make_audio(src, rate=16000, channels=1, sample_fmt="s32", codec="flac")
    with pytest.raises(di.DfImportError, match="s16"):
        di._ffprobe_source(src)


@pytest.mark.skipif(not _HAVE_FFMPEG, reason="ffmpeg/ffprobe not installed")
def test_ffprobe_rejects_non_flac(tmp_path: Path) -> None:
    src = tmp_path / "plain.wav"
    _make_audio(src, rate=16000, channels=1, sample_fmt="s16", codec="pcm_s16le")
    with pytest.raises(di.DfImportError, match="flac"):
        di._ffprobe_source(src)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_cli_select_writes_artifacts(tmp_path: Path) -> None:
    rows = [_row(trial_id=f"DF_E_{2000000 + i}", speaker=f"SPK{i:03d}") for i in range(10)]
    keys = tmp_path / "DF-keys-full.tar.gz"
    keys_sha = _keys_tar(keys, rows)
    out_dir = tmp_path / "sel"
    monkey = {"DF-keys-full.tar.gz": keys_sha}
    orig = di.OFFICIAL_ARCHIVE_SHA256
    di.OFFICIAL_ARCHIVE_SHA256 = monkey  # inject fixture provenance for the CLI path
    try:
        rc = di.main(["select", "--keys-archive", str(keys), "--out-dir", str(out_dir)])
    finally:
        di.OFFICIAL_ARCHIVE_SHA256 = orig
    assert rc == 0
    assert (out_dir / "trial_list.txt").is_file()
    assert (out_dir / "trial_ids.txt").is_file()
    receipt = json.loads((out_dir / "selection_receipt.json").read_bytes().decode())
    assert receipt["n_selected"] == 1  # round(10/10) within the single stratum


@pytest.mark.skipif(not _HAVE_FFMPEG, reason="ffmpeg/ffprobe not installed")
def test_cli_emit_writes_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    keys, audios, prov, _ = _emit_fixture(tmp_path)
    monkeypatch.setattr(di, "OFFICIAL_ARCHIVE_SHA256", prov)
    out_dir = tmp_path / "corpus"
    native_root = tmp_path / "native"
    rc = di.main(
        [
            "emit", "--keys-archive", str(keys),
            "--audio-archive", str(audios[0]),
            "--native-root", str(native_root), "--out-dir", str(out_dir),
        ]
    )
    assert rc == 0
    assert (out_dir / "manifest.json").is_file()
    assert (native_root / di.NATIVE_TREE_ROOT / di.NATIVE_FLAC_SUBDIR).is_dir()


def test_cli_returns_2_on_import_error(tmp_path: Path) -> None:
    """An unpinned archive fails closed and the CLI reports a non-zero status."""
    keys = tmp_path / "surprise.tar.gz"  # a name absent from the pinned provenance
    _keys_tar(keys, [_row(trial_id="DF_E_1")])
    rc = di.main(["select", "--keys-archive", str(keys), "--out-dir", str(tmp_path / "sel")])
    assert rc == 2
