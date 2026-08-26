"""Degradation-chain tests for synthdetect S5 (issue #144).

Freezes the pure, audio-free `degrade` layer before any ffmpeg runs: the closed
recipe vocabulary, canonical chain serialization, the exact ffmpeg argv the
executor will run, degraded-child derivation with lineage inheritance, and the
hardened lineage invariants (unknown recipe, cycle, inheritance mismatch). No
ffmpeg is invoked; the builders return argument lists.

PR-2b executor tests use a fake ``docker`` executable (a shell script that
copies input to output with a known transform, placed on PATH in the test)
instead of a real ffmpeg container, so the unit suite stays CI-portable and
fast. Real-codec acceptance is the Phase 3 maintainer gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import struct
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import synthdetect_corpus as corpus  # noqa: E402
import synthdetect_sources as sources  # noqa: E402

_SHA = "b" * 64


# --------------------------------------------------------------------------- #
# Recipe registry
# --------------------------------------------------------------------------- #
def test_recipe_registry_integrity() -> None:
    for recipe_id, recipe in sources.DEGRADATION_RECIPES.items():
        assert recipe.recipe_id == recipe_id
        assert recipe.family in sources.DEGRADATION_FAMILIES
        assert recipe.encode_args
        assert recipe.lossy == bool(recipe.intermediate_format)


def test_get_recipe_unknown_rejected() -> None:
    with pytest.raises(sources.SourcesError, match="unknown degradation recipe"):
        sources.get_recipe("nope-v1")


def _recipe(**over: Any) -> sources.DegradationRecipe:
    base: dict[str, Any] = {
        "recipe_id": "x-v1",
        "family": "codec",
        "implementation": "libmp3lame",
        "lossy": True,
        "encode_args": ("-c:a", "libmp3lame"),
        "intermediate_format": "mp3",
    }
    base.update(over)
    return sources.DegradationRecipe(**base)


def test_validate_recipes_key_id_mismatch() -> None:
    with pytest.raises(sources.SourcesError, match="!= recipe_id"):
        sources._validate_recipes({"x-v1": _recipe(recipe_id="y-v1")})


@pytest.mark.parametrize(
    "recipe,match",
    [
        (_recipe(recipe_id="X_v1"), "lowercase"),  # unsafe id (uppercase + underscore)
        (_recipe(family="noise"), "unknown family"),
        (_recipe(encode_args=()), "empty encode_args"),
        (_recipe(lossy=True, intermediate_format=""), "must name an intermediate_format"),
        (_recipe(lossy=False, intermediate_format="mp3"), "must not carry an intermediate_format"),
    ],
)
def test_validate_recipes_fail_closed(recipe: sources.DegradationRecipe, match: str) -> None:
    # Key by the recipe's own id so the key/id-match rail is not what trips first.
    with pytest.raises(sources.SourcesError, match=match):
        sources._validate_recipes({recipe.recipe_id: recipe})


def test_validate_recipes_empty_implementation_rejected() -> None:
    with pytest.raises(sources.SourcesError, match="empty implementation"):
        sources._validate_recipes({"x-v1": _recipe(implementation="  ")})


def test_noise_family_is_deferred() -> None:
    # Additive-noise is intentionally absent from the pure recipe set (its SNR mix
    # needs a measured parent RMS). Assert no noise recipe leaked in.
    assert "noise" not in sources.DEGRADATION_FAMILIES
    assert all(r.family != "noise" for r in sources.DEGRADATION_RECIPES.values())


# --------------------------------------------------------------------------- #
# Chain serialization
# --------------------------------------------------------------------------- #
def test_serialize_and_parse_roundtrip() -> None:
    chain = corpus.serialize_chain(["speed-atempo-0p90-v1", "mp3-cbr48-v1"])
    assert chain == "speed-atempo-0p90-v1|mp3-cbr48-v1"
    assert corpus.parse_chain(chain) == ("speed-atempo-0p90-v1", "mp3-cbr48-v1")


def test_serialize_chain_order_significant() -> None:
    a = corpus.serialize_chain(["speed-atempo-0p90-v1", "mp3-cbr48-v1"])
    b = corpus.serialize_chain(["mp3-cbr48-v1", "speed-atempo-0p90-v1"])
    assert a != b


def test_serialize_chain_unknown_rejected() -> None:
    with pytest.raises(corpus.CorpusError, match="unknown recipe"):
        corpus.serialize_chain(["mp3-cbr48-v1", "bogus-v1"])


def test_serialize_chain_empty_rejected() -> None:
    with pytest.raises(corpus.CorpusError, match="at least one recipe"):
        corpus.serialize_chain([])


@pytest.mark.parametrize("bad", ["", "  ", "mp3-cbr48-v1|", "|mp3-cbr48-v1", "a||b"])
def test_parse_chain_malformed_rejected(bad: str) -> None:
    with pytest.raises(corpus.CorpusError):
        corpus.parse_chain(bad)


# --------------------------------------------------------------------------- #
# ffmpeg argv builders (exact)
# --------------------------------------------------------------------------- #
_PREFIX = ("ffmpeg", "-nostdin", "-y", "-threads", "1", "-filter_threads", "1")
_RAW_IN = ("-f", "s16le", "-ar", "16000", "-ac", "1")
_CANON_OUT = ("-f", "s16le", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le")


def test_build_recipe_argv_lossy_two_passes() -> None:
    recipe = sources.get_recipe("mp3-cbr48-v1")
    cmds = corpus.build_recipe_argv(
        recipe, in_path="in.s16le", out_path="out.wav", intermediate_path="t.mp3"
    )
    assert cmds == (
        (
            *_PREFIX, *_RAW_IN, "-i", "in.s16le",
            "-c:a", "libmp3lame", "-b:a", "48k", "-ar", "16000", "-ac", "1",
            "-threads", "1", "-f", "mp3", "t.mp3",
        ),
        (
            *_PREFIX, "-i", "t.mp3", "-threads", "1", *_CANON_OUT, "out.wav",
        ),
    )


def test_build_recipe_argv_speed_single_pass() -> None:
    recipe = sources.get_recipe("speed-atempo-0p90-v1")
    cmds = corpus.build_recipe_argv(recipe, in_path="in.s16le", out_path="out.wav")
    assert cmds == (
        (
            *_PREFIX, *_RAW_IN, "-i", "in.s16le",
            "-filter:a", "atempo=0.90",
            "-threads", "1", *_CANON_OUT, "out.wav",
        ),
    )


@pytest.mark.parametrize(
    "recipe_id,encode_args,intermediate_format",
    [
        ("opus-voip-cbr16-f20-v1",
         ("-c:a", "libopus", "-b:a", "16k", "-vbr", "off", "-application", "voip",
          "-frame_duration", "20", "-ar", "16000", "-ac", "1"), "opus"),
        ("aac-lc-cbr48-v1",
         ("-c:a", "aac", "-b:a", "48k", "-profile:a", "aac_low", "-ar", "16000", "-ac", "1"),
         "adts"),
        ("g711-mulaw-8k-v1", ("-c:a", "pcm_mulaw", "-ar", "8000", "-ac", "1"), "wav"),
        ("amr-nb-122-v1",
         ("-c:a", "libopencore_amrnb", "-b:a", "12.2k", "-ar", "8000", "-ac", "1"), "amr"),
    ],
)
def test_build_recipe_argv_lossy_goldens(
    recipe_id: str, encode_args: tuple[str, ...], intermediate_format: str
) -> None:
    # Full argv is golden for every lossy recipe so PR-2 cannot drift the encode
    # options, the -threads determinism pin, or the raw framing without a red test.
    recipe = sources.get_recipe(recipe_id)
    mid = f"mid.{intermediate_format}"
    cmds = corpus.build_recipe_argv(
        recipe, in_path="in.s16le", out_path="out.wav", intermediate_path=mid
    )
    assert cmds == (
        (
            *_PREFIX, *_RAW_IN, "-i", "in.s16le", *encode_args,
            "-threads", "1", "-f", intermediate_format, mid,
        ),
        (
            *_PREFIX, "-i", mid, "-threads", "1", *_CANON_OUT, "out.wav",
        ),
    )


def test_build_recipe_argv_lossy_needs_intermediate() -> None:
    recipe = sources.get_recipe("mp3-cbr48-v1")
    with pytest.raises(corpus.CorpusError, match="needs an intermediate_path"):
        corpus.build_recipe_argv(recipe, in_path="in.s16le", out_path="out.wav")


@pytest.mark.parametrize("bad", ["-in.s16le", "-out.wav"])
def test_build_recipe_argv_rejects_option_like_path(bad: str) -> None:
    # A path starting with '-' would be parsed by ffmpeg as an option, not a file.
    recipe = sources.get_recipe("speed-atempo-0p90-v1")
    in_path = bad if bad.startswith("-in") else "in.s16le"
    out_path = bad if bad.startswith("-out") else "out.wav"
    with pytest.raises(corpus.CorpusError, match="must not start with"):
        corpus.build_recipe_argv(recipe, in_path=in_path, out_path=out_path)


def test_build_recipe_argv_output_threads_pin_encoder() -> None:
    # -threads 1 must appear on the OUTPUT side (after -i) on every pass, not only in
    # the prefix, so the encoder itself is pinned to one thread.
    for recipe in sources.DEGRADATION_RECIPES.values():
        cmds = corpus.build_recipe_argv(
            recipe, in_path="in.s16le", out_path="out.wav", intermediate_path="mid.x"
        )
        for cmd in cmds:
            after_i = cmd.index("-threads", cmd.index("-i"))
            assert cmd[after_i : after_i + 2] == ("-threads", "1")


def test_build_recipe_argv_raw_input_framing_every_recipe() -> None:
    # Every recipe frames its raw input as -f s16le -ar 16000 -ac 1 before -i, so
    # ffmpeg never guesses the headerless PCM layout.
    for recipe in sources.DEGRADATION_RECIPES.values():
        cmds = corpus.build_recipe_argv(
            recipe, in_path="in.s16le", out_path="out.wav", intermediate_path="mid.x"
        )
        first = cmds[0]
        i = first.index("-i")
        assert first[i - 6 : i] == ("-f", "s16le", "-ar", "16000", "-ac", "1")


# --------------------------------------------------------------------------- #
# Degraded-child derivation + lineage
# --------------------------------------------------------------------------- #
def _parent_clip(**over: Any) -> corpus.ClipEntry:
    raw: dict[str, Any] = {
        "clip_id": "ami-m1-A-turn-0-64600",
        "rel_path": "ami/turn/ami-m1-A-turn-0-64600.wav",
        "sha256": _SHA,
        "duration_s": 4.0,
        "label": "bona_fide",
        "language": "en",
        "license_spdx": "CC-BY-4.0",
        "stratum": "bona_fide|organic|meetingroom",
        "source": "ami",
        "speaker_id": "ami-m1-A",
        "split": "calibration",
        "generator": None,
        "degradation": None,
        "parent_clip_id": None,
        "acquire": None,
    }
    raw.update(over)
    return corpus.load_manifest({"schema_version": 1, "clips": [raw]}).clips[0]


def test_derive_degraded_record_inherits_and_derives() -> None:
    parent = _parent_clip()
    child = corpus.derive_degraded_record(parent, ["mp3-cbr48-v1"])
    assert child.parent_clip_id == parent.clip_id
    assert child.degradation == "mp3-cbr48-v1"
    assert child.clip_id == "ami-m1-A-turn-0-64600-mp3-cbr48-v1"
    assert child.rel_path == "ami/turn/degraded/ami-m1-A-turn-0-64600-mp3-cbr48-v1.wav"
    assert child.stratum == "bona_fide|organic|meetingroom|mp3-cbr48-v1"
    # Inheritance:
    for field in ("label", "speaker_id", "language", "license_spdx", "split", "source"):
        assert getattr(child, field) == getattr(parent, field)


def test_derive_degraded_record_chain_slug() -> None:
    parent = _parent_clip()
    child = corpus.derive_degraded_record(parent, ["speed-atempo-0p90-v1", "mp3-cbr48-v1"])
    assert child.degradation == "speed-atempo-0p90-v1|mp3-cbr48-v1"
    assert "|" not in child.clip_id
    assert child.clip_id.endswith("-speed-atempo-0p90-v1-mp3-cbr48-v1")


def test_derive_degraded_record_unknown_recipe_rejected() -> None:
    with pytest.raises(corpus.CorpusError, match="unknown recipe"):
        corpus.derive_degraded_record(_parent_clip(), ["not-real-v1"])


def test_parent_and_child_finalize_roundtrip() -> None:
    # A turn parent + its degraded child validate together through finalize.
    turns = corpus.parse_rttm(
        "SPEAKER m1 1 0.0 5.0 <NA> <NA> A <NA> <NA>"
    )
    plan = corpus.build_plan(corpus.ORGANIC_SOURCES["ami"], {"m1": turns})
    parent_record = plan.turn_clips[0]
    measured = {parent_record.clip_id: (_SHA, parent_record.interval.n_samples)}
    parent_manifest = corpus.finalize_manifest([parent_record], measured)
    parent = parent_manifest.clips[0]

    child = corpus.derive_degraded_record(parent, ["mp3-cbr48-v1"])
    both_measured = {
        parent.clip_id: (_SHA, 80000),
        child.clip_id: (hashlib.sha256(child.clip_id.encode()).hexdigest(), 80000),
    }
    # finalize with the parent IngestRecord + the child DegradedRecord.
    manifest = corpus.finalize_manifest([parent_record, child], both_measured)
    assert len(manifest.clips) == 2
    degraded = [c for c in manifest.clips if c.degradation is not None]
    assert len(degraded) == 1
    assert degraded[0].parent_clip_id == parent.clip_id


def _raw(clip_id: str, **over: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "clip_id": clip_id,
        "rel_path": f"ami/{clip_id}.wav",
        "sha256": _SHA,
        "duration_s": 4.0,
        "label": "bona_fide",
        "language": "en",
        "license_spdx": "CC-BY-4.0",
        "stratum": "bona_fide|organic|meetingroom",
        "source": "ami",
        "speaker_id": "spk",
        "split": None,
        "generator": None,
        "degradation": None,
        "parent_clip_id": None,
        "acquire": None,
    }
    raw.update(over)
    return raw


def test_lineage_inheritance_mismatch_rejected() -> None:
    parent = _raw("p1")
    child = _raw(
        "c1", degradation="mp3-cbr48-v1", parent_clip_id="p1", speaker_id="other"
    )
    with pytest.raises(corpus.CorpusError, match="does not match its parent"):
        corpus.load_manifest({"schema_version": 1, "clips": [parent, child]})


def test_lineage_cycle_rejected() -> None:
    # A two-node parent cycle c1<->c2 must be caught (self-parent alone is not enough).
    a = _raw("c1", degradation="mp3-cbr48-v1", parent_clip_id="c2")
    b = _raw("c2", degradation="mp3-cbr48-v1", parent_clip_id="c1")
    with pytest.raises(corpus.CorpusError, match="cycle"):
        corpus.load_manifest({"schema_version": 1, "clips": [a, b]})


def test_lineage_three_node_cycle_rejected() -> None:
    # A longer loop a->b->c->a must also be caught, not just two-node cycles.
    a = _raw("c1", degradation="mp3-cbr48-v1", parent_clip_id="c3")
    b = _raw("c2", degradation="mp3-cbr48-v1", parent_clip_id="c1")
    c = _raw("c3", degradation="mp3-cbr48-v1", parent_clip_id="c2")
    with pytest.raises(corpus.CorpusError, match="cycle"):
        corpus.load_manifest({"schema_version": 1, "clips": [a, b, c]})


def test_lineage_source_mismatch_rejected() -> None:
    # A degraded child is an audio transform of its parent, so it cannot change
    # source (which would re-attribute the clip's domain and strata).
    parent = _raw("p1")
    child = _raw("c1", degradation="mp3-cbr48-v1", parent_clip_id="p1", source="voxconverse")
    with pytest.raises(corpus.CorpusError, match="does not match its parent"):
        corpus.load_manifest({"schema_version": 1, "clips": [parent, child]})


# --------------------------------------------------------------------------- #
# PR-2b executor tests
# --------------------------------------------------------------------------- #


def _make_pcm_payload(n_samples: int = 16000) -> bytes:
    """Generate a deterministic PCM payload (sine wave, s16le)."""
    return struct.pack(
        f"<{n_samples}h",
        *(int(32767 * (0.5 if i % 2 == 0 else -0.5)) for i in range(n_samples)),
    )


def _write_parent_corpus(
    parent_root: Path,
    n_clips: int = 1,
    *,
    split: str | None = "calibration",
    source: str = "ami",
) -> tuple[list[dict[str, Any]], bytes]:
    """Create a minimal parent corpus with canonical WAVs and manifest.

    Returns the raw clip dicts and the manifest bytes.
    """
    payload = _make_pcm_payload()
    sha, count = corpus.payload_sha_and_count(payload)
    clips = []
    for idx in range(n_clips):
        clip_id = f"{source}-m1-A-turn-{idx}-{count}"
        rel_path = f"{source}/turn/{clip_id}.wav"
        wav_path = parent_root / rel_path
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        corpus.write_canonical_wav(wav_path, payload)
        clips.append({
            "clip_id": clip_id,
            "rel_path": rel_path,
            "sha256": sha,
            "duration_s": corpus._duration_s_from_samples(count),
            "label": "bona_fide",
            "language": "en",
            "license_spdx": "CC-BY-4.0",
            "stratum": "bona_fide|organic|meetingroom",
            "source": source,
            "speaker_id": f"{source}-m1-A",
            "split": split,
            "generator": None,
            "degradation": None,
            "parent_clip_id": None,
            "acquire": None,
        })
    manifest_obj = {"schema_version": 1, "clips": clips}
    corpus.load_manifest(manifest_obj)
    manifest_bytes = (
        json.dumps(manifest_obj, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    (parent_root / "manifest.json").write_bytes(manifest_bytes)
    return clips, manifest_bytes


_FAKE_DOCKER = textwrap.dedent("""\
    #!/bin/bash
    # Fake docker that copies input.raw to the output path (identity transform).
    # Understands the minimal contract: finds -v <host>:/work:rw, finds -i <in>,
    # and the last arg is the output path. Copies the input to the output.
    workdir=""
    infile=""
    outfile=""
    prev=""
    for arg in "$@"; do
        case "$prev" in
            -v) workdir="${arg%%:*}" ;;
            -i) infile="$arg" ;;
        esac
        prev="$arg"
    done
    outfile="$arg"
    # Resolve /work/ paths to host paths
    host_in="${infile/\\/work/$workdir}"
    host_out="${outfile/\\/work/$workdir}"
    mkdir -p "$(dirname "$host_out")"
    cp "$host_in" "$host_out"
    exit 0
""")


@pytest.fixture()
def fake_docker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Install a fake ``docker`` on PATH that performs an identity copy."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(_FAKE_DOCKER)
    docker.chmod(docker.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ.get('PATH', '')}")
    return docker


_FAKE_IMAGE = "test/ffmpeg@sha256:" + "a" * 64


# -- _run_containerized_ffmpeg --------------------------------------------- #

class TestRunContainerizedFfmpeg:
    def test_strips_ffmpeg_prefix(self, tmp_path: Path, fake_docker: Path) -> None:
        workdir = tmp_path / "work"
        workdir.mkdir()
        (workdir / "in.raw").write_bytes(b"\x00" * 100)
        argv = ("ffmpeg", "-nostdin", "-y", "-f", "s16le", "-i", "/work/in.raw",
                "-f", "s16le", "/work/out.raw")
        corpus._run_containerized_ffmpeg(
            argv, workdir=workdir, container_image=_FAKE_IMAGE
        )
        assert (workdir / "out.raw").exists()
        assert (workdir / "out.raw").read_bytes() == b"\x00" * 100

    def test_rejects_non_ffmpeg_prefix(self, tmp_path: Path) -> None:
        with pytest.raises(corpus.CorpusError, match="must start with 'ffmpeg'"):
            corpus._run_containerized_ffmpeg(
                ("notffmpeg", "-i", "/work/in.raw", "/work/out.raw"),
                workdir=tmp_path, container_image=_FAKE_IMAGE,
            )

    def test_rejects_non_work_output(self, tmp_path: Path) -> None:
        with pytest.raises(corpus.CorpusError, match="must be under /work/"):
            corpus._run_containerized_ffmpeg(
                ("ffmpeg", "-i", "/work/in.raw", "/tmp/out.raw"),
                workdir=tmp_path, container_image=_FAKE_IMAGE,
            )


# -- _degrade_one_clip ----------------------------------------------------- #

class TestDegradeOneClip:
    def test_identity_transform(self, tmp_path: Path, fake_docker: Path) -> None:
        parent_root = tmp_path / "parent"
        parent_root.mkdir()
        clips, _ = _write_parent_corpus(parent_root)
        parent_manifest = corpus.load_manifest({"schema_version": 1, "clips": clips})
        parent_clip = parent_manifest.clips[0]
        child_record = corpus.derive_degraded_record(
            parent_clip, ["speed-atempo-0p90-v1"]
        )
        staging = tmp_path / "staging"
        staging.mkdir()
        sha, count = corpus._degrade_one_clip(
            parent_clip, child_record, ("speed-atempo-0p90-v1",),
            parent_root=parent_root, staging=staging, container_image=_FAKE_IMAGE,
        )
        assert corpus._is_sha256(sha)
        assert count == 16000
        child_wav = staging / child_record.rel_path
        assert child_wav.is_file()
        roundtrip = corpus.read_canonical_wav_payload(child_wav)
        assert corpus.payload_sha_and_count(roundtrip) == (sha, count)

    def test_chain_feeds_stages(self, tmp_path: Path, fake_docker: Path) -> None:
        """Two-recipe chain: stage-0 output becomes stage-1 input."""
        parent_root = tmp_path / "parent"
        parent_root.mkdir()
        clips, _ = _write_parent_corpus(parent_root)
        parent_manifest = corpus.load_manifest({"schema_version": 1, "clips": clips})
        parent_clip = parent_manifest.clips[0]
        chain = ("speed-atempo-0p90-v1", "speed-atempo-0p90-v1")
        child_record = corpus.derive_degraded_record(parent_clip, list(chain))
        staging = tmp_path / "staging"
        staging.mkdir()
        _sha, count = corpus._degrade_one_clip(
            parent_clip, child_record, chain,
            parent_root=parent_root, staging=staging, container_image=_FAKE_IMAGE,
        )
        assert count == 16000
        assert (staging / child_record.rel_path).is_file()

    def test_parent_sha_mismatch_rejected(
        self, tmp_path: Path, fake_docker: Path
    ) -> None:
        parent_root = tmp_path / "parent"
        parent_root.mkdir()
        clips, _ = _write_parent_corpus(parent_root)
        clips[0]["sha256"] = "c" * 64
        manifest_bytes = (
            json.dumps({"schema_version": 1, "clips": clips}, indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8")
        (parent_root / "manifest.json").write_bytes(manifest_bytes)
        parent_manifest = corpus.load_manifest({"schema_version": 1, "clips": clips})
        parent_clip = parent_manifest.clips[0]
        child_record = corpus.derive_degraded_record(
            parent_clip, ["speed-atempo-0p90-v1"]
        )
        staging = tmp_path / "staging"
        staging.mkdir()
        with pytest.raises(corpus.CorpusError, match="does not match manifest"):
            corpus._degrade_one_clip(
                parent_clip, child_record, ("speed-atempo-0p90-v1",),
                parent_root=parent_root, staging=staging, container_image=_FAKE_IMAGE,
            )

    def test_length_sanity_too_short_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Fake docker that outputs a very short file
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        docker = bin_dir / "docker"
        docker.write_text(textwrap.dedent("""\
            #!/bin/bash
            prev=""
            workdir=""
            for arg in "$@"; do
                case "$prev" in -v) workdir="${arg%%:*}" ;; esac
                prev="$arg"
            done
            outfile="$arg"
            host_out="${outfile/\\/work/$workdir}"
            mkdir -p "$(dirname "$host_out")"
            # Output 2 samples (4 bytes) — way below 50% of 16000
            printf '\\x00\\x00\\x00\\x00' > "$host_out"
            exit 0
        """))
        docker.chmod(docker.stat().st_mode | stat.S_IEXEC)
        monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ.get('PATH', '')}")

        parent_root = tmp_path / "parent"
        parent_root.mkdir()
        clips, _ = _write_parent_corpus(parent_root)
        parent_manifest = corpus.load_manifest({"schema_version": 1, "clips": clips})
        parent_clip = parent_manifest.clips[0]
        child_record = corpus.derive_degraded_record(
            parent_clip, ["speed-atempo-0p90-v1"]
        )
        staging = tmp_path / "staging"
        staging.mkdir()
        with pytest.raises(corpus.CorpusError, match="below"):
            corpus._degrade_one_clip(
                parent_clip, child_record, ("speed-atempo-0p90-v1",),
                parent_root=parent_root, staging=staging, container_image=_FAKE_IMAGE,
            )


# -- _assemble_combined_manifest ------------------------------------------- #

class TestAssembleCombinedManifest:
    def test_combined_validates(self, tmp_path: Path) -> None:
        parent_root = tmp_path / "parent"
        parent_root.mkdir()
        clips, _ = _write_parent_corpus(parent_root, n_clips=2)
        parent_manifest = corpus.load_manifest({"schema_version": 1, "clips": clips})
        child_records = [
            corpus.derive_degraded_record(c, ["mp3-cbr48-v1"])
            for c in parent_manifest.clips
        ]
        measured = {}
        for r in child_records:
            sha = hashlib.sha256(r.clip_id.encode()).hexdigest()
            measured[r.clip_id] = (sha, 16000)
        manifest, combined = corpus._assemble_combined_manifest(
            clips, child_records, measured
        )
        assert len(manifest.clips) == 4
        assert len(combined) == 4

    def test_orphan_measurement_rejected(self) -> None:
        clips = [_raw("p1")]
        parent_manifest = corpus.load_manifest({"schema_version": 1, "clips": clips})
        child = corpus.derive_degraded_record(
            parent_manifest.clips[0], ["mp3-cbr48-v1"]
        )
        measured = {
            child.clip_id: ("a" * 64, 16000),
            "ghost": ("b" * 64, 16000),
        }
        with pytest.raises(corpus.CorpusError, match="keyset mismatch"):
            corpus._assemble_combined_manifest(clips, [child], measured)

    def test_missing_measurement_rejected(self) -> None:
        clips = [_raw("p1")]
        parent_manifest = corpus.load_manifest({"schema_version": 1, "clips": clips})
        child = corpus.derive_degraded_record(
            parent_manifest.clips[0], ["mp3-cbr48-v1"]
        )
        with pytest.raises(corpus.CorpusError, match="keyset mismatch"):
            corpus._assemble_combined_manifest(clips, [child], {})

    def test_bad_sha_rejected(self) -> None:
        clips = [_raw("p1")]
        parent_manifest = corpus.load_manifest({"schema_version": 1, "clips": clips})
        child = corpus.derive_degraded_record(
            parent_manifest.clips[0], ["mp3-cbr48-v1"]
        )
        measured = {child.clip_id: ("NOT_A_SHA", 16000)}
        with pytest.raises(corpus.CorpusError, match="64 lowercase hex"):
            corpus._assemble_combined_manifest(clips, [child], measured)

    def test_zero_count_rejected(self) -> None:
        clips = [_raw("p1")]
        parent_manifest = corpus.load_manifest({"schema_version": 1, "clips": clips})
        child = corpus.derive_degraded_record(
            parent_manifest.clips[0], ["mp3-cbr48-v1"]
        )
        measured = {child.clip_id: ("a" * 64, 0)}
        with pytest.raises(corpus.CorpusError, match="positive int"):
            corpus._assemble_combined_manifest(clips, [child], measured)


# -- resolve_clip_path ----------------------------------------------------- #

class TestResolveClipPath:
    def test_resolves_in_correct_root(self, tmp_path: Path) -> None:
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        (root_b / "ami" / "turn").mkdir(parents=True)
        (root_b / "ami" / "turn" / "clip.wav").write_bytes(b"\x00")
        clip = _parent_clip(rel_path="ami/turn/clip.wav")
        resolved = corpus.resolve_clip_path(clip, roots=(root_a, root_b))
        assert resolved == root_b / "ami" / "turn" / "clip.wav"

    def test_zero_roots_rejected(self, tmp_path: Path) -> None:
        clip = _parent_clip(rel_path="ami/turn/clip.wav")
        with pytest.raises(corpus.CorpusError, match="not found in any root"):
            corpus.resolve_clip_path(clip, roots=(tmp_path,))

    def test_multiple_roots_rejected(self, tmp_path: Path) -> None:
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        for root in (root_a, root_b):
            (root / "ami" / "turn").mkdir(parents=True)
            (root / "ami" / "turn" / "clip.wav").write_bytes(b"\x00")
        clip = _parent_clip(rel_path="ami/turn/clip.wav")
        with pytest.raises(corpus.CorpusError, match="multiple roots"):
            corpus.resolve_clip_path(clip, roots=(root_a, root_b))


# -- materialize_degrade (integration with fake docker) -------------------- #

class TestMaterializeDegrade:
    def test_full_pipeline(self, tmp_path: Path, fake_docker: Path) -> None:
        parent_root = tmp_path / "parent"
        parent_root.mkdir()
        _write_parent_corpus(parent_root, n_clips=2)
        degrade_root = tmp_path / "degrade"
        result = corpus.materialize_degrade(
            parent_root=parent_root,
            corpus_root=degrade_root,
            container_image=_FAKE_IMAGE,
            recipe_ids=("speed-atempo-0p90-v1",),
        )
        assert result.children == 2
        assert result.parents_reaudited == 2
        assert corpus._is_sha256(result.combined_manifest_sha256)
        assert corpus._is_sha256(result.parent_manifest_sha256)
        assert (degrade_root / "manifest.json").is_file()
        assert (degrade_root / "clip_receipt.jsonl").is_file()
        assert (degrade_root / "degrade_receipt.json").is_file()
        manifest = corpus.load_manifest(
            json.loads((degrade_root / "manifest.json").read_text())
        )
        assert len(manifest.clips) == 4
        parents = [c for c in manifest.clips if c.parent_clip_id is None]
        children = [c for c in manifest.clips if c.parent_clip_id is not None]
        assert len(parents) == 2
        assert len(children) == 2
        for child in children:
            child_wav = degrade_root / child.rel_path
            assert child_wav.is_file()
            payload = corpus.read_canonical_wav_payload(child_wav)
            sha, _count = corpus.payload_sha_and_count(payload)
            assert sha == child.sha256

    def test_split_filter(self, tmp_path: Path, fake_docker: Path) -> None:
        parent_root = tmp_path / "parent"
        parent_root.mkdir()
        _write_parent_corpus(parent_root, n_clips=1, split="eval")
        degrade_root = tmp_path / "degrade"
        with pytest.raises(corpus.CorpusError, match="no eligible"):
            corpus.materialize_degrade(
                parent_root=parent_root,
                corpus_root=degrade_root,
                container_image=_FAKE_IMAGE,
                recipe_ids=("speed-atempo-0p90-v1",),
                split_filter="calibration",
            )

    def test_populated_root_rejected(self, tmp_path: Path) -> None:
        parent_root = tmp_path / "parent"
        parent_root.mkdir()
        _write_parent_corpus(parent_root)
        degrade_root = tmp_path / "degrade"
        degrade_root.mkdir()
        (degrade_root / "leftover").write_text("x")
        with pytest.raises(corpus.CorpusError, match="already populated"):
            corpus.materialize_degrade(
                parent_root=parent_root,
                corpus_root=degrade_root,
                container_image=_FAKE_IMAGE,
                recipe_ids=("speed-atempo-0p90-v1",),
            )

    def test_bad_container_image_rejected(self, tmp_path: Path) -> None:
        parent_root = tmp_path / "parent"
        parent_root.mkdir()
        _write_parent_corpus(parent_root)
        with pytest.raises(corpus.CorpusError, match="sha256"):
            corpus.materialize_degrade(
                parent_root=parent_root,
                corpus_root=tmp_path / "degrade",
                container_image="notpinned:latest",
                recipe_ids=("speed-atempo-0p90-v1",),
            )

    def test_v2_manifest_rejected(self, tmp_path: Path) -> None:
        parent_root = tmp_path / "parent"
        parent_root.mkdir()
        v2_manifest = {
            "schema_version": 2,
            "corpus_kind": "imported_benchmark",
            "benchmark": "asvspoof2021-df",
            "clips": [{
                "clip_id": "x",
                "rel_path": "canonical/x.wav",
                "sha256": "a" * 64,
                "duration_s": 1.0,
                "label": "bona_fide",
                "language": "und",
                "license_spdx": "LicenseRef-ASVspoof2021-DF",
                "stratum": "bona_fide|nocodec",
                "source": "asvspoof2021-df",
                "speaker_id": "s1",
                "split": "eval",
                "imported_provenance": {
                    "official_trial_id": "x",
                    "source_dataset": "asvspoof",
                    "codec_condition": "nocodec",
                    "official_split": "eval",
                    "attack_system": None,
                    "vocoder_family": "bonafide",
                },
            }],
        }
        (parent_root / "manifest.json").write_text(
            json.dumps(v2_manifest, indent=2, sort_keys=True) + "\n"
        )
        with pytest.raises(corpus.CorpusError, match="v1"):
            corpus.materialize_degrade(
                parent_root=parent_root,
                corpus_root=tmp_path / "degrade",
                container_image=_FAKE_IMAGE,
                recipe_ids=("speed-atempo-0p90-v1",),
            )

    def test_parent_with_degraded_entries_rejected(
        self, tmp_path: Path
    ) -> None:
        parent_root = tmp_path / "parent"
        parent_root.mkdir()
        clips, _ = _write_parent_corpus(parent_root)
        payload = _make_pcm_payload()
        sha, count = corpus.payload_sha_and_count(payload)
        child_dict = {
            "clip_id": clips[0]["clip_id"] + "-mp3-cbr48-v1",
            "rel_path": "ami/turn/degraded/" + clips[0]["clip_id"] + "-mp3-cbr48-v1.wav",
            "sha256": sha,
            "duration_s": corpus._duration_s_from_samples(count),
            "label": "bona_fide",
            "language": "en",
            "license_spdx": "CC-BY-4.0",
            "stratum": "bona_fide|organic|meetingroom|mp3-cbr48-v1",
            "source": "ami",
            "speaker_id": "ami-m1-A",
            "split": "calibration",
            "degradation": "mp3-cbr48-v1",
            "parent_clip_id": clips[0]["clip_id"],
            "acquire": None,
            "generator": None,
        }
        child_wav = parent_root / child_dict["rel_path"]
        child_wav.parent.mkdir(parents=True, exist_ok=True)
        corpus.write_canonical_wav(child_wav, payload)
        all_clips = [*clips, child_dict]
        manifest_bytes = (
            json.dumps(
                {"schema_version": 1, "clips": all_clips},
                indent=2, sort_keys=True,
            ) + "\n"
        ).encode("utf-8")
        (parent_root / "manifest.json").write_bytes(manifest_bytes)
        with pytest.raises(corpus.CorpusError, match="degraded entries"):
            corpus.materialize_degrade(
                parent_root=parent_root,
                corpus_root=tmp_path / "degrade",
                container_image=_FAKE_IMAGE,
                recipe_ids=("speed-atempo-0p90-v1",),
            )

    def test_reaudit_sha_mismatch_rejected(
        self, tmp_path: Path, fake_docker: Path
    ) -> None:
        parent_root = tmp_path / "parent"
        parent_root.mkdir()
        clips, _ = _write_parent_corpus(parent_root)
        # Tamper the manifest sha without touching the WAV
        clips[0]["sha256"] = "c" * 64
        (parent_root / "manifest.json").write_bytes(
            (json.dumps({"schema_version": 1, "clips": clips},
                        indent=2, sort_keys=True) + "\n").encode()
        )
        with pytest.raises(corpus.CorpusError, match="re-audit failed"):
            corpus.materialize_degrade(
                parent_root=parent_root,
                corpus_root=tmp_path / "degrade",
                container_image=_FAKE_IMAGE,
                recipe_ids=("speed-atempo-0p90-v1",),
            )

    def test_cleanup_on_failure(self, tmp_path: Path, fake_docker: Path) -> None:
        parent_root = tmp_path / "parent"
        parent_root.mkdir()
        _write_parent_corpus(parent_root)
        degrade_root = tmp_path / "degrade"
        # Sabotage: remove the parent WAV so the re-audit reads it but then
        # the degrade step would fail (but re-audit catches sha mismatch first).
        # Instead, use an invalid container image to fail preflight.
        with pytest.raises(corpus.CorpusError):
            corpus.materialize_degrade(
                parent_root=parent_root,
                corpus_root=degrade_root,
                container_image="bad:latest",
                recipe_ids=("speed-atempo-0p90-v1",),
            )
        assert not degrade_root.exists()

    def test_grandchild_skip(self, tmp_path: Path, fake_docker: Path) -> None:
        """cmd_degrade dry-run skips already-degraded clips (no grandchildren)."""
        parent_root = tmp_path / "parent"
        parent_root.mkdir()
        _write_parent_corpus(parent_root)
        manifest_path = parent_root / "manifest.json"
        rc = corpus.cmd_degrade(
            argparse.Namespace(
                manifest=str(manifest_path),
                recipe=["speed-atempo-0p90-v1"],
                split=None,
                corpus_root=None,
                parent_root=None,
                container_image=None,
            )
        )
        assert rc == 0


# -- CLI dry-run vs execution equivalence ---------------------------------- #

class TestCmdDegradeExecution:
    def test_execution_mode_requires_flags(self, capsys: Any) -> None:
        rc = corpus.cmd_degrade(
            argparse.Namespace(
                manifest="/dev/null",
                recipe=["speed-atempo-0p90-v1"],
                split=None,
                corpus_root="/tmp/out",
                parent_root=None,
                container_image=None,
            )
        )
        assert rc == 2
        assert "requires" in capsys.readouterr().err

    def test_execution_mode_produces_result(
        self, tmp_path: Path, fake_docker: Path, capsys: Any
    ) -> None:
        parent_root = tmp_path / "parent"
        parent_root.mkdir()
        _write_parent_corpus(parent_root)
        degrade_root = tmp_path / "degrade"
        rc = corpus.cmd_degrade(
            argparse.Namespace(
                manifest=str(parent_root / "manifest.json"),
                recipe=["speed-atempo-0p90-v1"],
                split=None,
                corpus_root=str(degrade_root),
                parent_root=str(parent_root),
                container_image=_FAKE_IMAGE,
            )
        )
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert out["children"] == 1
        assert corpus._is_sha256(out["combined_manifest_sha256"])


# -- Container image pin format -------------------------------------------- #

class TestContainerImagePin:
    @pytest.mark.parametrize("good", [
        "repo/image@sha256:" + "a" * 64,
        "registry.io/org/image@sha256:" + "0" * 64,
        "jrottenberg/ffmpeg@sha256:" + "f" * 64,
    ])
    def test_valid_pins_accepted(self, good: str) -> None:
        assert corpus._CONTAINER_IMAGE_RE.match(good)

    @pytest.mark.parametrize("bad", [
        "image:latest",
        "image@sha256:short",
        "image@sha256:" + "g" * 64,
        "@sha256:" + "a" * 64,
    ])
    def test_invalid_pins_rejected(self, bad: str) -> None:
        assert not corpus._CONTAINER_IMAGE_RE.match(bad)
