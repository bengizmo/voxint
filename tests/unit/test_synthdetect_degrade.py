"""Degradation-chain tests for synthdetect S5 (issue #144).

Freezes the pure, audio-free `degrade` layer before any ffmpeg runs: the closed
recipe vocabulary, canonical chain serialization, the exact ffmpeg argv the
executor will run, degraded-child derivation with lineage inheritance, and the
hardened lineage invariants (unknown recipe, cycle, inheritance mismatch). No
ffmpeg is invoked; the builders return argument lists.
"""

from __future__ import annotations

import hashlib
import sys
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
