#!/usr/bin/env python3
"""Manifest schema v1 + seeded split assignment for the synthdetect corpus (#144).

The reference corpus lives on maintainer storage and is NEVER committed (the
corpus root is always a CLI arg). What IS reviewable and testable in CI is the
manifest that describes it: one record per clip binding a content sha to its
label, provenance, and split. THIS module is the schema, its fail-closed
validation, and the deterministic split assignment -- everything the S2+
acquisition verbs (generate / prepare / synthesize / degrade) will stand on,
frozen and unit-covered before any audio exists.

Two properties the split assignment guarantees, both load-bearing for an honest
generalization estimate:

* **Speaker-disjoint.** A speaker's clips never straddle two splits: the split
  is assigned per speaker by a seeded hash-rank, so a voice the calibrator saw
  cannot leak into the eval set and flatter the numbers.
* **Unseen-generator eval.** At least one generator is forced eval-only (all its
  clips in ``eval``, never calibration/holdout), so the eval set measures
  generalization to a synthesis system the operating point was never tuned on.

Nothing here reads audio or touches a service; it validates JSON-shaped records
and computes a pure, seeded assignment.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import wave
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from synthdetect_sources import (  # noqa: E402
    DEGRADATION_RECIPES,
    SELECTION_SEED,
    DegradationRecipe,
)

MANIFEST_SCHEMA_VERSION = 1

# Schema v2 is a single imported-benchmark variant (issue #144, S3): a corpus of
# real audio acquired from an external evaluation benchmark, NOT one we synthesize.
# It is deliberately NOT a general benchmark ontology; a second benchmark earns its
# own review. v1 validation is unchanged. See docs/gpu-contracts.md, the S3
# reproduction pre-registration, for why an imported eval cannot carry synthesis
# generator provenance and must record officially-absent fields as null.
IMPORTED_MANIFEST_SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = (MANIFEST_SCHEMA_VERSION, IMPORTED_MANIFEST_SCHEMA_VERSION)

# The corpus kind. v1 manifests are implicitly ``synthesis`` (clips we generate,
# each spoof carrying reproducible generator provenance); a v2 manifest MUST declare
# ``imported_benchmark`` and name the ``benchmark`` it was imported from.
CORPUS_KIND_SYNTHESIS = "synthesis"
CORPUS_KIND_IMPORTED = "imported_benchmark"

# Honest provenance: an officially-absent field is JSON null, never a placeholder.
# These tokens are the placeholders that masquerade as data; a v2 provenance string
# equal to any of them (case-insensitively) is rejected as fabricated.
_PROVENANCE_SENTINELS = frozenset(
    {"", "-", "n/a", "na", "none", "null", "unknown", "unspecified", "tbd", "todo"}
)

# A clip is bona fide (genuine human speech) or spoof (synthetic/converted). The
# two-class labelling is the detector's target; ``stratum`` carries the finer
# grouping (codec chain, generator family, domain) the per-stratum scorer uses.
LABELS = ("bona_fide", "spoof")

# The three splits. ``calibration`` fits the Platt policy + threshold; ``eval``
# measures it; ``holdout`` is opened exactly once, after every runtime and
# calibration choice is frozen (holdout discipline -- see docs/gpu-contracts.md).
SPLITS = ("calibration", "eval", "holdout")

# A clip_id / speaker_id becomes a manifest key and (for clip_id) a path stem, so
# it must be a plain token: no separators, traversal, or whitespace.
_SAFE_ID_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")


class CorpusError(Exception):
    """A manifest-integrity problem (fail closed; never repair)."""


def _is_safe_id(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value[0] not in "._-"
        and all(c in _SAFE_ID_CHARS for c in value)
    )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GeneratorProvenance:
    """How a synthetic clip was produced -- required for every spoof clip.

    The fields make a synthesis reproducible and let the eval measure
    per-generator behaviour: ``name``/``version`` identify the system,
    ``checkpoint_sha`` pins its weights, ``voice``/``seed``/``text_source`` pin
    the exact utterance. A bona fide clip carries no generator (``None``).
    """

    name: str
    version: str
    checkpoint_sha: str | None
    voice: str
    seed: str
    text_source: str


@dataclass(frozen=True)
class ImportedBenchmarkProvenance:
    """The official provenance of one imported-benchmark clip (v2 only).

    Records only what the source benchmark actually publishes for the clip: its
    official trial id, the source dataset and codec condition, the official split
    the row belongs to, the vocoder family (always published: a real family, the
    literal ``unknown``, or ``bonafide`` for genuine speech), and the attack system
    (published for spoof clips, ``None`` for bona fide). An unset field is ``None``,
    never coerced to a placeholder.
    """

    official_trial_id: str
    source_dataset: str
    codec_condition: str
    official_split: str
    vocoder_family: str
    attack_system: str | None


@dataclass(frozen=True)
class ClipEntry:
    """One validated manifest clip.

    ``split`` is None until :func:`assign_splits` stamps it (the acquisition
    verbs build the manifest first, then assign). ``generator`` is set iff the
    label is ``spoof``; ``degradation``/``parent_clip_id`` are set for a clip
    derived from another (a codec/degradation chain applied to ``parent_clip_id``).

    In a v2 imported-benchmark manifest, ``generator`` is always None and
    ``imported_provenance`` carries the official provenance instead; in a v1
    synthesis manifest ``imported_provenance`` is None. The scoring path
    (``clip_id``/``rel_path``/``sha256``/``duration_s``/``label``/``stratum``/
    ``split``) is identical for both, so the runner is variant-agnostic.
    """

    clip_id: str
    rel_path: str
    sha256: str
    duration_s: float
    label: str
    language: str
    license_spdx: str
    stratum: str
    source: str
    speaker_id: str
    split: str | None
    generator: GeneratorProvenance | None
    degradation: str | None
    parent_clip_id: str | None
    acquire: str | None
    imported_provenance: ImportedBenchmarkProvenance | None = None


@dataclass(frozen=True)
class Manifest:
    """A validated corpus manifest: schema version, corpus kind + the clip records.

    ``corpus_kind`` is ``synthesis`` for v1 and ``imported_benchmark`` for v2;
    ``benchmark`` names the source benchmark for a v2 manifest and is None for v1.
    """

    schema_version: int
    clips: tuple[ClipEntry, ...]
    corpus_kind: str = CORPUS_KIND_SYNTHESIS
    benchmark: str | None = None


def _validate_generator(raw: Any, where: str) -> GeneratorProvenance:
    if not isinstance(raw, dict):
        raise CorpusError(f"{where}: generator must be an object")
    allowed = {"name", "version", "checkpoint_sha", "voice", "seed", "text_source"}
    extra = set(raw) - allowed
    if extra:
        raise CorpusError(f"{where}: generator has unexpected keys {sorted(extra)}")
    for key in ("name", "version", "voice", "seed", "text_source"):
        val = raw.get(key)
        if not isinstance(val, str) or not val.strip():
            raise CorpusError(f"{where}: generator.{key} must be a non-empty string")
    checkpoint_sha = raw.get("checkpoint_sha")
    # A generator checkpoint sha is optional (CANDIDATE), but if present it must
    # be a real digest, never a placeholder string.
    if checkpoint_sha is not None and not _is_sha256(checkpoint_sha):
        raise CorpusError(f"{where}: generator.checkpoint_sha must be 64 lowercase hex or null")
    return GeneratorProvenance(
        name=raw["name"],
        version=raw["version"],
        checkpoint_sha=checkpoint_sha,
        voice=raw["voice"],
        seed=raw["seed"],
        text_source=raw["text_source"],
    )


def _reject_sentinel(value: str, where: str) -> None:
    """Raise if a provenance string is a placeholder masquerading as data."""
    if value.strip().lower() in _PROVENANCE_SENTINELS:
        raise CorpusError(
            f"{where}: {value!r} is a placeholder; an officially-absent field "
            "must be null, not a sentinel string"
        )


def _validate_imported_provenance(
    raw: Any, where: str, *, label: str
) -> ImportedBenchmarkProvenance:
    """Validate a v2 imported-benchmark provenance block (fail closed).

    ``official_trial_id``/``source_dataset``/``codec_condition``/``official_split``/
    ``vocoder_family`` are always required non-empty strings; the four identity
    fields must not be sentinels. ``vocoder_family`` is exempt from the sentinel
    scan because the official metadata uses the literal ``unknown`` as a real
    vocoder family (and ``bonafide`` for genuine speech). ``attack_system`` is the
    one label-coupled field: an attack id for a spoof clip, and null for a bona
    fide clip. A field the benchmark leaves unset is present-and-null, not omitted.
    """
    if not isinstance(raw, dict):
        raise CorpusError(f"{where}: imported_provenance must be an object")
    allowed = {
        "official_trial_id", "source_dataset", "codec_condition",
        "official_split", "attack_system", "vocoder_family",
    }
    extra = set(raw) - allowed
    if extra:
        raise CorpusError(f"{where}: imported_provenance has unexpected keys {sorted(extra)}")
    # Honest provenance records absence as an explicit null, so an officially-absent
    # attack must be present-and-null rather than omitted. Require the full field
    # set; a missing key is not the same assertion as a null one.
    missing = allowed - set(raw)
    if missing:
        raise CorpusError(f"{where}: imported_provenance is missing keys {sorted(missing)}")

    for key in ("official_trial_id", "source_dataset", "codec_condition", "official_split"):
        val = raw.get(key)
        if not isinstance(val, str) or not val.strip():
            raise CorpusError(f"{where}: imported_provenance.{key} must be a non-empty string")
        _reject_sentinel(val, f"{where}: imported_provenance.{key}")

    # The vocoder family is always officially supplied (a real family, the literal
    # ``unknown``, or ``bonafide`` for genuine speech); required, never sentinel-scanned.
    vocoder = raw.get("vocoder_family")
    if not isinstance(vocoder, str) or not vocoder.strip():
        raise CorpusError(f"{where}: imported_provenance.vocoder_family must be a non-empty string")

    # The attack system is the one label-coupled field: present for a spoof clip,
    # null for a bona fide clip. Presence is real data, absence is an explicit null.
    attack = raw.get("attack_system")
    if attack is not None and (not isinstance(attack, str) or not attack.strip()):
        raise CorpusError(
            f"{where}: imported_provenance.attack_system must be a non-empty string or null"
        )
    if label == "spoof" and attack is None:
        raise CorpusError(f"{where}: a spoof imported clip must name its attack_system")
    if label != "spoof" and attack is not None:
        raise CorpusError(f"{where}: a bona_fide imported clip must not carry an attack_system")

    return ImportedBenchmarkProvenance(
        official_trial_id=raw["official_trial_id"],
        source_dataset=raw["source_dataset"],
        codec_condition=raw["codec_condition"],
        official_split=raw["official_split"],
        attack_system=attack,
        vocoder_family=vocoder,
    )


def validate_clip(raw: Any, index: int, *, corpus_kind: str = CORPUS_KIND_SYNTHESIS) -> ClipEntry:
    """Validate one raw clip record into a :class:`ClipEntry` (fail closed).

    Enforces the required-field set and types, the label/split vocabularies, a
    real content sha, and a positive duration for both corpus kinds. For a v1
    ``synthesis`` clip it enforces the label<->generator coupling (a spoof clip
    MUST carry generator provenance; a bona fide clip MUST NOT) and the
    degradation/``parent_clip_id`` derivation. For a v2 ``imported_benchmark``
    clip it forbids the synthesis keys, requires an ``imported_provenance`` block,
    and binds that block to the clip (official trial id equals the clip id, the
    clip is eval-only, and the stratum matches the official label and codec).
    """
    where = f"clip {index}"
    if not isinstance(raw, dict):
        raise CorpusError(f"{where}: must be an object")
    if corpus_kind not in (CORPUS_KIND_SYNTHESIS, CORPUS_KIND_IMPORTED):
        raise CorpusError(f"{where}: unknown corpus_kind {corpus_kind!r}")
    imported = corpus_kind == CORPUS_KIND_IMPORTED
    common = {
        "clip_id", "rel_path", "sha256", "duration_s", "label", "language",
        "license_spdx", "stratum", "source", "speaker_id", "split",
    }
    # An imported-benchmark clip carries official provenance and never a synthesis
    # generator or a degradation chain; a synthesis clip is the reverse.
    if imported:
        allowed = common | {"imported_provenance"}
    else:
        allowed = common | {"generator", "degradation", "parent_clip_id", "acquire"}
    extra = set(raw) - allowed
    if extra:
        raise CorpusError(f"{where}: unexpected keys {sorted(extra)}")

    clip_id = raw.get("clip_id")
    if not _is_safe_id(clip_id):
        raise CorpusError(f"{where}: clip_id {clip_id!r} is not a safe token")
    assert isinstance(clip_id, str)  # _is_safe_id guarantees this; narrows for mypy
    where = f"clip {clip_id!r}"

    for key in ("rel_path", "language", "license_spdx", "stratum", "source"):
        val = raw.get(key)
        if not isinstance(val, str) or not val.strip():
            raise CorpusError(f"{where}: {key} must be a non-empty string")

    speaker_id = raw.get("speaker_id")
    if not _is_safe_id(speaker_id):
        raise CorpusError(f"{where}: speaker_id {speaker_id!r} is not a safe token")
    assert isinstance(speaker_id, str)  # narrows for mypy (guaranteed by _is_safe_id)

    if not _is_sha256(raw.get("sha256")):
        raise CorpusError(f"{where}: sha256 must be 64 lowercase hex chars")

    duration_s = raw.get("duration_s")
    # json.loads parses NaN/Infinity by default, and `nan <= 0` is False, so a
    # non-finite duration would slip past a bare positivity check in a fail-closed
    # module. Require a finite positive number.
    if (
        not isinstance(duration_s, (int, float))
        or isinstance(duration_s, bool)
        or not math.isfinite(duration_s)
        or duration_s <= 0
    ):
        raise CorpusError(f"{where}: duration_s must be a finite number > 0, got {duration_s!r}")

    label = raw.get("label")
    if label not in LABELS:
        raise CorpusError(f"{where}: label must be one of {LABELS}, got {label!r}")

    split = raw.get("split")
    if split is not None and split not in SPLITS:
        raise CorpusError(f"{where}: split must be null or one of {SPLITS}, got {split!r}")

    # A relative path must stay relative (never absolute or traversing) -- the
    # corpus root is supplied at read time and a clip must resolve under it.
    rel_path = raw["rel_path"]
    if rel_path.startswith("/") or ".." in Path(rel_path).parts:
        raise CorpusError(f"{where}: rel_path {rel_path!r} must be relative and not traverse")

    generator: GeneratorProvenance | None = None
    imported_provenance: ImportedBenchmarkProvenance | None = None
    degradation: str | None = None
    parent_clip_id: str | None = None
    acquire: str | None = None

    if imported:
        # v2: official provenance replaces the synthesis generator entirely.
        prov_raw = raw.get("imported_provenance")
        if prov_raw is None:
            raise CorpusError(f"{where}: an imported clip must carry imported_provenance")
        imported_provenance = _validate_imported_provenance(prov_raw, where, label=label)
        # Bind the provenance to this clip's scoring identity: the block must
        # describe THIS trial, the clip must be eval-only with the official split
        # agreeing, and the stratum must match the official label and codec. Each
        # field validates in isolation above; these checks stop a manifest from
        # attaching one trial's audio to another trial's provenance or grouping.
        if imported_provenance.official_trial_id != clip_id:
            raise CorpusError(
                f"{where}: imported_provenance.official_trial_id "
                f"{imported_provenance.official_trial_id!r} must equal clip_id {clip_id!r}"
            )
        if split != "eval":
            raise CorpusError(f"{where}: an imported clip must have split 'eval', got {split!r}")
        if imported_provenance.official_split != "eval":
            raise CorpusError(
                f"{where}: imported_provenance.official_split must be 'eval', "
                f"got {imported_provenance.official_split!r}"
            )
        expected_stratum = f"{label}|{imported_provenance.codec_condition}"
        if raw["stratum"] != expected_stratum:
            raise CorpusError(
                f"{where}: stratum {raw['stratum']!r} must be {expected_stratum!r} "
                "(label and official codec condition)"
            )
    else:
        generator_raw = raw.get("generator")
        if label == "spoof":
            if generator_raw is None:
                raise CorpusError(f"{where}: a spoof clip must carry generator provenance")
            generator = _validate_generator(generator_raw, where)
        else:
            if generator_raw is not None:
                raise CorpusError(f"{where}: a bona_fide clip must not carry a generator")

        degradation = raw.get("degradation")
        if degradation is not None:
            if not isinstance(degradation, str) or not degradation.strip():
                raise CorpusError(f"{where}: degradation must be null or a non-empty string")
            # A degradation string must name a chain of known recipe ids, so a
            # manifest can never reference a phantom transform (S5 lineage rail).
            try:
                parse_chain(degradation)
            except CorpusError as exc:
                raise CorpusError(f"{where}: {exc}") from None
        parent_clip_id = raw.get("parent_clip_id")
        if parent_clip_id is not None and not _is_safe_id(parent_clip_id):
            raise CorpusError(f"{where}: parent_clip_id must be null or a safe token")
        if degradation is not None and parent_clip_id is None:
            raise CorpusError(f"{where}: a degraded clip must name its parent_clip_id")
        # The coupling is bidirectional: a parent pointer without a degradation label
        # is a dangling derivation, and a clip can never be its own parent.
        if parent_clip_id is not None and degradation is None:
            raise CorpusError(f"{where}: parent_clip_id set without a degradation label")
        if parent_clip_id is not None and parent_clip_id == clip_id:
            raise CorpusError(f"{where}: a clip cannot be its own parent")

        acquire = raw.get("acquire")
        if acquire is not None and (not isinstance(acquire, str) or not acquire.strip()):
            raise CorpusError(f"{where}: acquire must be null or a non-empty string")

    return ClipEntry(
        clip_id=clip_id,
        rel_path=rel_path,
        sha256=raw["sha256"],
        duration_s=float(duration_s),
        label=label,
        language=raw["language"],
        license_spdx=raw["license_spdx"],
        stratum=raw["stratum"],
        source=raw["source"],
        speaker_id=speaker_id,
        split=split,
        generator=generator,
        degradation=degradation,
        parent_clip_id=parent_clip_id,
        acquire=acquire,
        imported_provenance=imported_provenance,
    )


def load_manifest(obj: Any) -> Manifest:
    """Validate a whole manifest object into a :class:`Manifest` (fail closed).

    Requires a supported schema version, a non-empty ``clips`` array, unique
    ``clip_id`` values, and that every ``parent_clip_id`` refers to a clip that
    exists in the manifest (a degradation chain can never dangle). A v2 manifest
    additionally declares ``corpus_kind: imported_benchmark`` and names its
    ``benchmark``; a v1 manifest carries neither and is validated exactly as before.
    """
    if not isinstance(obj, dict):
        raise CorpusError("manifest must be an object")
    version = obj.get("schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise CorpusError(
            f"manifest schema_version must be one of {SUPPORTED_SCHEMA_VERSIONS}, "
            f"got {version!r}"
        )
    imported = version == IMPORTED_MANIFEST_SCHEMA_VERSION

    # Fail closed on unexpected top-level keys, version-aware.
    top_allowed = {"schema_version", "clips"}
    if imported:
        top_allowed |= {"corpus_kind", "benchmark"}
    top_extra = set(obj) - top_allowed
    if top_extra:
        raise CorpusError(f"manifest has unexpected top-level keys {sorted(top_extra)}")

    if imported:
        corpus_kind = obj.get("corpus_kind")
        if corpus_kind != CORPUS_KIND_IMPORTED:
            raise CorpusError(
                f"a v2 manifest must declare corpus_kind {CORPUS_KIND_IMPORTED!r}, "
                f"got {corpus_kind!r}"
            )
        benchmark = obj.get("benchmark")
        if not isinstance(benchmark, str) or not benchmark.strip():
            raise CorpusError("a v2 manifest must name a non-empty benchmark")
        _reject_sentinel(benchmark, "manifest benchmark")
    else:
        corpus_kind = CORPUS_KIND_SYNTHESIS
        benchmark = None

    raw_clips = obj.get("clips")
    if not isinstance(raw_clips, list) or not raw_clips:
        raise CorpusError("manifest 'clips' must be a non-empty array")
    clips = tuple(validate_clip(raw, i, corpus_kind=corpus_kind) for i, raw in enumerate(raw_clips))
    ids = [c.clip_id for c in clips]
    if len(set(ids)) != len(ids):
        dupes = sorted({cid for cid in ids if ids.count(cid) > 1})
        raise CorpusError(f"manifest has duplicate clip_id(s): {dupes}")
    known = set(ids)
    for c in clips:
        if c.parent_clip_id is not None and c.parent_clip_id not in known:
            raise CorpusError(
                f"clip {c.clip_id!r}: parent_clip_id {c.parent_clip_id!r} is not in the manifest"
            )
    by_id = {c.clip_id: c for c in clips}
    # Lineage inheritance (S5): a degraded child is the SAME source, speaker, label,
    # language, split, and license as its parent -- it is an audio-domain transform,
    # not a new sample. A mismatch means a manifest attached a child to the wrong
    # parent or mislabelled it, which the per-clip validator cannot see.
    for c in clips:
        if c.parent_clip_id is None:
            continue
        parent = by_id[c.parent_clip_id]
        for field in ("label", "speaker_id", "language", "split", "license_spdx", "source"):
            if getattr(c, field) != getattr(parent, field):
                raise CorpusError(
                    f"clip {c.clip_id!r}: {field} {getattr(c, field)!r} does not match its "
                    f"parent {parent.clip_id!r} ({getattr(parent, field)!r})"
                )
    # No parent cycle: a degradation chain must terminate at a root, so lineage is a
    # forest. self-parent is rejected per-clip; walk the chain to catch longer loops.
    for c in clips:
        seen_ids = {c.clip_id}
        node = c
        while node.parent_clip_id is not None:
            if node.parent_clip_id in seen_ids:
                raise CorpusError(
                    f"clip {c.clip_id!r}: parent chain forms a cycle at {node.parent_clip_id!r}"
                )
            seen_ids.add(node.parent_clip_id)
            node = by_id[node.parent_clip_id]
    # Speaker-disjointness is guaranteed by assign_splits, but a hand-edited
    # manifest can stamp one speaker's clips into two splits and still validate;
    # scoring would then trust a leaked split. Enforce it as a load-time invariant
    # for clips that already carry a split.
    split_of_speaker: dict[str, str] = {}
    for c in clips:
        if c.split is None:
            continue
        prior = split_of_speaker.get(c.speaker_id)
        if prior is not None and prior != c.split:
            raise CorpusError(
                f"speaker {c.speaker_id!r} straddles splits {prior!r} and {c.split!r} "
                "(splits must be speaker-disjoint)"
            )
        split_of_speaker[c.speaker_id] = c.split
    return Manifest(
        schema_version=version, clips=clips, corpus_kind=corpus_kind, benchmark=benchmark
    )


# --------------------------------------------------------------------------- #
# Seeded, speaker-disjoint split assignment
# --------------------------------------------------------------------------- #
def _rank_key(seed: str, speaker_id: str) -> int:
    """A deterministic 256-bit rank key for a speaker (seed-domain-separated)."""
    digest = hashlib.sha256(f"{seed}\x00{speaker_id}".encode()).hexdigest()
    return int(digest, 16)


def _partition_free_speakers(
    speaker_ids: set[str] | frozenset[str],
    *,
    forced_eval: set[str] | frozenset[str],
    calibration_fraction: float,
    holdout_fraction: float,
    seed: str,
) -> dict[str, str]:
    """Seeded speaker -> split partition (the shared core of split assignment).

    ``forced_eval`` speakers are pinned to ``eval``; the rest are ranked by
    :func:`_rank_key` and cut into ``calibration`` (first
    ``calibration_fraction``), ``holdout`` (last ``holdout_fraction``), and
    ``eval`` (the middle). Callers own their own fraction validation.
    """
    free = sorted(
        set(speaker_ids) - set(forced_eval), key=lambda sp: (_rank_key(seed, sp), sp)
    )
    n = len(free)
    n_cal = int(n * calibration_fraction)
    n_hold = min(int(n * holdout_fraction), n - n_cal)
    out: dict[str, str] = dict.fromkeys(forced_eval, "eval")
    for rank, sp in enumerate(free):
        if rank < n_cal:
            out[sp] = "calibration"
        elif rank >= n - n_hold:
            out[sp] = "holdout"
        else:
            out[sp] = "eval"
    return out


def assign_splits(
    clips: tuple[ClipEntry, ...] | list[ClipEntry],
    *,
    eval_only_generators: frozenset[str] | set[str],
    calibration_fraction: float = 0.5,
    holdout_fraction: float = 0.2,
    seed: str = SELECTION_SEED,
) -> dict[str, str]:
    """Assign each clip a split, seeded and speaker-disjoint. Returns clip_id->split.

    The algorithm, in order:

    1. Any speaker with a clip from an ``eval_only_generators`` generator is
       pinned to ``eval`` -- the unseen-generator guarantee. Naming a generator
       that never appears in the corpus is a mistake, not a no-op, and raises.
    2. The remaining speakers are ranked by :func:`_rank_key` (a seeded hash, so
       the partition is deterministic and reshuffles only when the seed changes)
       and partitioned by cumulative fraction into ``calibration`` (the first
       ``calibration_fraction``), ``holdout`` (the last ``holdout_fraction``),
       and ``eval`` (the middle). A speaker maps wholly to one split.

    ``calibration_fraction + holdout_fraction`` must leave a non-negative eval
    share; a corpus with no clips is rejected. The forced-eval speakers are
    counted OUTSIDE the fractional partition (they are not genuine random draws),
    so the fractions describe the freely-assignable speakers.
    """
    clips = list(clips)
    if not clips:
        raise CorpusError("assign_splits: no clips to assign")
    if not math.isfinite(calibration_fraction) or not math.isfinite(holdout_fraction):
        raise CorpusError("assign_splits: fractions must be finite")
    if calibration_fraction < 0 or holdout_fraction < 0:
        raise CorpusError("assign_splits: fractions must be non-negative")
    if calibration_fraction + holdout_fraction > 1.0 + 1e-9:
        raise CorpusError(
            f"assign_splits: calibration+holdout fractions exceed 1.0 "
            f"({calibration_fraction}+{holdout_fraction})"
        )

    speakers_of_generator: dict[str, set[str]] = {}
    speaker_ids: set[str] = set()
    for c in clips:
        speaker_ids.add(c.speaker_id)
        if c.generator is not None:
            speakers_of_generator.setdefault(c.generator.name, set()).add(c.speaker_id)

    forced_eval_speakers: set[str] = set()
    for gen_name in eval_only_generators:
        if gen_name not in speakers_of_generator:
            raise CorpusError(
                f"assign_splits: eval-only generator {gen_name!r} has no clips in the corpus"
            )
        forced_eval_speakers |= speakers_of_generator[gen_name]

    split_of_speaker = _partition_free_speakers(
        speaker_ids,
        forced_eval=forced_eval_speakers,
        calibration_fraction=calibration_fraction,
        holdout_fraction=holdout_fraction,
        seed=seed,
    )
    return {c.clip_id: split_of_speaker[c.speaker_id] for c in clips}


def split_summary(assignment: dict[str, str]) -> dict[str, int]:
    """Count clips per split (a report/CLI convenience over an assignment)."""
    counts = dict.fromkeys(SPLITS, 0)
    for split in assignment.values():
        counts[split] += 1
    return counts


# --------------------------------------------------------------------------- #
# Organic corpus planning (S5, pure): RTTM -> cleaned turns + session segments
# -> a materialization PLAN. No audio is read here; the plan records what to
# extract. The real sha256/duration only exist after the executor (PR-2) runs,
# so this layer NEVER builds a final manifest -- see finalize_manifest.
# --------------------------------------------------------------------------- #
CANONICAL_SAMPLE_RATE = 16000
# The AASIST fixed model input width; a session segment must reach this to score
# one full production window (docs/gpu-contracts.md, the windowing policy).
MODEL_WIDTH_SAMPLES = 64600

# Pinned corpus-construction floors. Pre-registered in docs/gpu-contracts.md so a
# change is a visible, deliberate diff, never drift. Second-domain thresholds are
# Decimal so gap/overlap comparisons carry no binary-float noise; length floors are
# enforced in the SAMPLE domain (after floor/ceil) because the contract is written
# in samples -- a span whose second-length rounds just under the threshold can still
# be a full 64,600-sample window.
CLIP_MERGE_GAP_S = Decimal("0.3")  # merge same-speaker turns/words within this gap into one clip
SESSION_MERGE_GAP_S = Decimal("1.0")  # production windowing merge_gap_s: the segment-view merge
OVERLAP_FLOOR_S = Decimal("0.1")  # ignore other-speaker overlap regions shorter than this
TURN_MIN_S = Decimal("1.0")  # a cleaned turn sub-span must reach this
TURN_MIN_SAMPLES = int(TURN_MIN_S * CANONICAL_SAMPLE_RATE)  # 16000: turn length floor, in samples
SEGMENT_MIN_SAMPLES = MODEL_WIDTH_SAMPLES  # 64600 = 4.0375s: a segment must reach one full window

PLAN_SCHEMA_VERSION = 1


def _duration_s_from_samples(sample_count: int) -> float:
    """The one canonical clip duration rule: measured sample count / 16000.

    Duration is derived only from the measured PCM sample count (never the plan),
    so the manifest ``duration_s`` and the numerics doctrine have a single source.
    """
    return sample_count / CANONICAL_SAMPLE_RATE


# --------------------------------------------------------------------------- #
# Canonical WAV I/O (S5 PR-2a executor primitives, numpy-free).
#
# The manifest identity is the sha256 of the canonical PCM ``data``-chunk payload
# only (never the container). ``synthdetect_infer.read_canonical_pcm`` computes the
# same digest with numpy for the scoring path; these helpers stay numpy-free so the
# pure corpus module carries no heavy dependency, and MUST agree with it byte for
# byte on what "the payload" is (the raw ``data``-chunk bytes). CANONICAL_CHANNELS /
# CANONICAL_SAMPLE_WIDTH / CANONICALIZATION_ID mirror the infer constants; a contract
# test pins that they stay equal so the two readers can never drift apart.
# --------------------------------------------------------------------------- #
CANONICAL_CHANNELS = 1
CANONICAL_SAMPLE_WIDTH = 2  # signed 16-bit little-endian
CANONICALIZATION_ID = "pcm-s16le-mono-16000-v1"
_BLOCK_ALIGN = CANONICAL_CHANNELS * CANONICAL_SAMPLE_WIDTH


def _riff_data_chunk_size_from_bytes(raw: bytes, label: str) -> int:
    """Return the declared byte size of the WAV ``data`` chunk in ``raw`` (fail closed).

    The stdlib ``wave`` reader silently floors an odd-sized ``data`` chunk to whole
    frames, so an orphan trailing byte (a truncated or non-canonical payload) would
    slip past a frame-count check. Walking the RIFF chunks directly lets the caller
    reject a payload whose declared size is not a whole number of frames.
    """
    if len(raw) < 12 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise CorpusError(f"{label}: not a RIFF/WAVE file")
    pos = 12
    while True:
        if pos + 8 > len(raw):
            raise CorpusError(f"{label}: no data chunk found")
        chunk_id = raw[pos : pos + 4]
        size = int.from_bytes(raw[pos + 4 : pos + 8], "little")
        if chunk_id == b"data":
            return size
        pos += 8 + size + (size & 1)  # skip payload plus its RIFF pad byte


def _riff_data_chunk_size(path: Path) -> int:
    """Path wrapper over :func:`_riff_data_chunk_size_from_bytes` (reads once)."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CorpusError(f"{path}: cannot read: {exc}") from exc
    return _riff_data_chunk_size_from_bytes(raw, str(path))


def canonical_payload_from_bytes(raw: bytes, *, label: str) -> bytes:
    """Validate ``raw`` as a canonical-PCM WAV and return its ``data``-chunk payload.

    Operates on a single in-memory buffer (no file reopen), so a caller that already
    read and hashed the bytes slices exactly the bytes it pinned. Fails closed on any
    non-canonical property (compressed, not mono, not 16 kHz, not signed 16-bit) and
    on a data chunk that is not a whole number of frames or is truncated.
    """
    try:
        with wave.open(io.BytesIO(raw), "rb") as wav:
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            rate = wav.getframerate()
            comp = wav.getcomptype()
            n_frames = wav.getnframes()
            frames = wav.readframes(n_frames)
    except (wave.Error, OSError, EOFError) as exc:
        raise CorpusError(f"{label}: not a readable PCM WAV: {exc}") from exc
    if comp != "NONE":
        raise CorpusError(
            f"{label}: compressed WAV ({comp!r}); canonical audio must be uncompressed"
        )
    if channels != CANONICAL_CHANNELS:
        raise CorpusError(f"{label}: {channels} channels; canonical audio must be mono")
    if rate != CANONICAL_SAMPLE_RATE:
        raise CorpusError(f"{label}: {rate} Hz; canonical audio must be {CANONICAL_SAMPLE_RATE} Hz")
    if width != CANONICAL_SAMPLE_WIDTH:
        raise CorpusError(
            f"{label}: {width * 8}-bit; canonical audio must be signed 16-bit "
            f"({CANONICALIZATION_ID})"
        )
    declared = _riff_data_chunk_size_from_bytes(raw, label)
    if declared % _BLOCK_ALIGN != 0:
        raise CorpusError(
            f"{label}: data chunk is {declared} bytes, not a whole number of "
            f"{_BLOCK_ALIGN}-byte frames (orphan byte)"
        )
    if len(frames) != declared or len(frames) != n_frames * _BLOCK_ALIGN:
        raise CorpusError(
            f"{label}: data payload is truncated ({len(frames)} bytes read, {declared} declared)"
        )
    return frames


def read_canonical_wav_payload(path: Path) -> bytes:
    """Read a canonical-PCM WAV once and return its raw ``data``-chunk payload bytes.

    The returned bytes are exactly what ``sha256`` is taken over, so a clip sliced
    from this payload and re-read here yields the manifest identity. Numpy-free
    sibling of ``synthdetect_infer.read_canonical_pcm``; both hash the same
    ``data``-chunk bytes.
    """
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CorpusError(f"{path}: cannot read: {exc}") from exc
    return canonical_payload_from_bytes(raw, label=str(path))


def payload_sha_and_count(payload: bytes) -> tuple[str, int]:
    """The measured ``(pcm_sha256, sample_count)`` of a canonical payload (fail closed).

    ``sample_count`` is ``len(payload) // 2``; a payload that is not a whole number of
    16-bit frames is a corrupt clip, not something to round.
    """
    if len(payload) % _BLOCK_ALIGN != 0:
        raise CorpusError(
            f"payload is {len(payload)} bytes, not a whole number of {_BLOCK_ALIGN}-byte frames"
        )
    if not payload:
        raise CorpusError("payload is empty")
    return hashlib.sha256(payload).hexdigest(), len(payload) // _BLOCK_ALIGN


def write_canonical_wav(path: Path, payload: bytes) -> None:
    """Write ``payload`` as a deterministic canonical-PCM WAV (mono/16 kHz/s16le).

    The stdlib ``wave`` writer emits a fixed RIFF/``fmt ``/``data`` layout with no
    metadata, so the file is byte-stable for a given payload and the ``data``-chunk
    bytes equal ``payload`` exactly (so ``read_canonical_wav_payload`` and
    ``read_canonical_pcm`` recover the same digest). Fails closed on a non-frame-
    aligned payload.
    """
    if len(payload) % _BLOCK_ALIGN != 0:
        raise CorpusError(
            f"payload is {len(payload)} bytes, not a whole number of {_BLOCK_ALIGN}-byte frames"
        )
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(CANONICAL_CHANNELS)
        wav.setsampwidth(CANONICAL_SAMPLE_WIDTH)
        wav.setframerate(CANONICAL_SAMPLE_RATE)
        wav.writeframes(payload)


@dataclass(frozen=True)
class OrganicSource:
    """A known organic (real-speech) corpus source: its domain, language, license.

    Data, not logic: the two staged CC-BY-4.0 diarization corpora. ``language``
    is the honest per-source default (``und`` where the corpus does not publish a
    per-clip language). A source not in :data:`ORGANIC_SOURCES` is a caller error.
    """

    source_id: str
    domain: str
    language: str
    license_spdx: str


ORGANIC_SOURCES: dict[str, OrganicSource] = {
    "ami": OrganicSource("ami", "meetingroom", "en", "CC-BY-4.0"),
    "voxconverse": OrganicSource("voxconverse", "webvideo", "und", "CC-BY-4.0"),
}


@dataclass(frozen=True)
class RttmTurn:
    """One speaker-active interval parsed from an RTTM row (seconds).

    Times are :class:`~decimal.Decimal`, parsed from the RTTM text, so the pinned
    floor/ceil sample rule (:func:`to_sample_interval`) is applied in exact decimal
    arithmetic. Binary ``float`` would make ``0.1 * 16000`` evaluate to
    ``1600.0000000000002`` and ceil to ``1601``, silently breaking the
    byte-reproducible-from-RTTM-times contract for ordinary decimals.
    """

    recording: str
    speaker_label: str
    start_s: Decimal
    dur_s: Decimal

    @property
    def end_s(self) -> Decimal:
        return self.start_s + self.dur_s


@dataclass(frozen=True)
class SampleInterval:
    """A half-open source sample interval ``[start_sample, end_sample)`` to slice."""

    start_sample: int
    end_sample: int

    @property
    def n_samples(self) -> int:
        return self.end_sample - self.start_sample


@dataclass(frozen=True)
class IngestRecord:
    """One planned clip: everything known BEFORE audio exists.

    Carries no content sha or measured duration (those are produced by the
    executor and supplied to :func:`finalize_manifest`). ``kind`` is ``turn`` (a
    per-speaker clip for strata/degradation/calibration) or ``segment`` (a merged
    same-speaker run, the production-windowing validation unit).
    """

    clip_id: str
    rel_path: str
    source: str
    recording: str
    speaker_id: str
    label: str
    language: str
    license_spdx: str
    stratum: str
    interval: SampleInterval
    split: str | None
    acquire: dict[str, Any]
    kind: str

    def clip_dict(self, pcm_sha256: str, sample_count: int) -> dict[str, Any]:
        """Build the v1 manifest clip record from this plan entry + measured facts."""
        return {
            "clip_id": self.clip_id,
            "rel_path": self.rel_path,
            "sha256": pcm_sha256,
            "duration_s": _duration_s_from_samples(sample_count),
            "label": self.label,
            "language": self.language,
            "license_spdx": self.license_spdx,
            "stratum": self.stratum,
            "source": self.source,
            "speaker_id": self.speaker_id,
            "split": self.split,
            "acquire": json.dumps(self.acquire, sort_keys=True),
        }


@dataclass(frozen=True)
class DegradedRecord:
    """One planned degraded child clip, derived from an already-materialized parent.

    A degraded child is not sliced from a source recording; it is a deterministic
    ffmpeg transform of its parent's canonical PCM, so it carries no source
    interval. It inherits the parent's label, speaker, language, license, and split
    (the hardened lineage invariant) and records the canonical ``degradation`` chain
    string plus the ``parent_clip_id``. Its content sha and duration, like a turn
    clip's, exist only after the executor runs.
    """

    clip_id: str
    rel_path: str
    parent_clip_id: str
    degradation: str
    source: str
    speaker_id: str
    label: str
    language: str
    license_spdx: str
    stratum: str
    split: str | None

    def clip_dict(self, pcm_sha256: str, sample_count: int) -> dict[str, Any]:
        """Build the v1 manifest clip record for a degraded child + measured facts."""
        return {
            "clip_id": self.clip_id,
            "rel_path": self.rel_path,
            "sha256": pcm_sha256,
            "duration_s": _duration_s_from_samples(sample_count),
            "label": self.label,
            "language": self.language,
            "license_spdx": self.license_spdx,
            "stratum": self.stratum,
            "source": self.source,
            "speaker_id": self.speaker_id,
            "split": self.split,
            "degradation": self.degradation,
            "parent_clip_id": self.parent_clip_id,
        }


@dataclass(frozen=True)
class MaterializationPlan:
    """The audio-free plan: which clips + segments to extract, and their splits."""

    schema_version: int
    source: str
    turn_clips: tuple[IngestRecord, ...]
    segments: tuple[IngestRecord, ...]


def parse_rttm(text: str, *, recording: str | None = None) -> tuple[RttmTurn, ...]:
    """Parse RTTM ``SPEAKER`` rows into turns (fail closed on any malformed row).

    Requires the ``SPEAKER <rec> <chan> <start> <dur> <NA> <NA> <label> ...`` shape
    (at least the eight fields through the label) with a finite ``start >= 0``,
    ``dur > 0``, and a real speaker label. If ``recording`` is given, every row's
    recording id must match. Start/dur are read as exact :class:`~decimal.Decimal`
    values, not binary ``float``, so the pinned sample rule stays byte-reproducible.
    """
    turns: list[RttmTurn] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        fields = stripped.split()
        if len(fields) < 9 or fields[0] != "SPEAKER":
            raise CorpusError(f"rttm line {lineno}: not a SPEAKER row: {stripped!r}")
        rec = fields[1]
        if recording is not None and rec != recording:
            raise CorpusError(
                f"rttm line {lineno}: recording {rec!r} does not match expected {recording!r}"
            )
        try:
            start = Decimal(fields[3])
            dur = Decimal(fields[4])
        except InvalidOperation:
            raise CorpusError(f"rttm line {lineno}: non-numeric start/dur") from None
        if not start.is_finite() or not dur.is_finite() or start < 0 or dur <= 0:
            raise CorpusError(
                f"rttm line {lineno}: need finite start >= 0 and dur > 0, got {start}, {dur}"
            )
        label = fields[7]
        if not label or label == "<NA>":
            raise CorpusError(f"rttm line {lineno}: missing speaker label")
        turns.append(RttmTurn(recording=rec, speaker_label=label, start_s=start, dur_s=dur))
    if not turns:
        raise CorpusError("rttm has no SPEAKER rows")
    return tuple(turns)


def _as_decimal(value: Decimal | float | int | str) -> Decimal:
    """Normalize a time to Decimal, going through ``str`` for floats.

    ``Decimal(0.1)`` captures the binary-float value (``0.1000...0055``); routing a
    float through ``str`` recovers the shortest decimal the caller meant (``0.1``).
    Internal callers already pass Decimal (from :func:`parse_rttm`); this only
    protects a direct float caller from the IEEE-754 boundary bug.
    """
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def to_sample_interval(
    start_s: Decimal | float | int | str, end_s: Decimal | float | int | str
) -> SampleInterval:
    """Convert a second span to a pinned half-open sample interval.

    ``start_sample = floor(start_s * 16000)``, ``end_sample = ceil(end_s * 16000)``
    -- pinned so the corpus is byte-reproducible from the RTTM times. The arithmetic
    is exact decimal (not binary float), so ``0.1`` maps to sample ``1600`` rather
    than ``1601``.
    """
    start = _as_decimal(start_s)
    end = _as_decimal(end_s)
    if not start.is_finite() or not end.is_finite() or start < 0 or end <= start:
        raise CorpusError(f"to_sample_interval: need 0 <= start < end, got {start_s}, {end_s}")
    return SampleInterval(
        start_sample=math.floor(start * CANONICAL_SAMPLE_RATE),
        end_sample=math.ceil(end * CANONICAL_SAMPLE_RATE),
    )


def namespaced_speaker(source_id: str, recording: str, label: str) -> str:
    """A conservative, recording-scoped speaker id: ``{source}-{recording}-{label}``.

    Recording-scoped so a recording-local or mislabeled RTTM label can never cause
    cross-split leakage (docs/gpu-contracts.md, the S5 pre-registration). Must be a
    safe token.
    """
    speaker_id = f"{source_id}-{recording}-{label}"
    if not _is_safe_id(speaker_id):
        raise CorpusError(
            f"speaker id {speaker_id!r} (from {source_id}/{recording}/{label}) is not a safe token"
        )
    return speaker_id


def _merge_intervals(
    intervals: list[tuple[Decimal, Decimal]], gap_s: Decimal
) -> list[tuple[Decimal, Decimal]]:
    """Sort and merge intervals whose gap is ``< gap_s`` (overlaps always merge)."""
    merged: list[tuple[Decimal, Decimal]] = []
    for start, end in sorted(intervals):
        if merged and start - merged[-1][1] < gap_s:
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _coalesce_intervals(
    intervals: list[tuple[Decimal, Decimal]],
) -> list[tuple[Decimal, Decimal]]:
    """Union overlapping or touching intervals into maximal continuous regions.

    Used to fold the OTHER-speaker rows into continuous other-speech regions before
    the overlap floor is applied. Word-level RTTMs emit a run of adjacent short rows;
    testing the floor per atomic row would let a long continuous stretch of other
    speech (each 80 ms word below the 0.1 s floor) survive inside a nominally
    single-speaker turn. The floor is meant to ignore a brief boundary graze, not a
    sustained region assembled from many short rows.
    """
    merged: list[tuple[Decimal, Decimal]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _subtract_overlaps(
    span: tuple[Decimal, Decimal],
    others: list[tuple[Decimal, Decimal]],
    *,
    floor_s: Decimal,
) -> list[tuple[Decimal, Decimal]]:
    """Return the sub-spans of ``span`` not covered by any OTHER-speaker interval.

    ``others`` must already be coalesced into continuous regions
    (:func:`_coalesce_intervals`). A region is only cut when its intersection with
    ``span`` is at least ``floor_s`` (sub-``floor_s`` grazes are ignored so a
    boundary touch does not fragment a turn).
    """
    span_start, span_end = span
    active: list[tuple[Decimal, Decimal]] = []
    for other_start, other_end in others:
        lo = max(span_start, other_start)
        hi = min(span_end, other_end)
        if hi - lo >= floor_s:
            active.append((lo, hi))
    result: list[tuple[Decimal, Decimal]] = []
    cursor = span_start
    for cut_start, cut_end in sorted(active):
        if cut_start > cursor:
            result.append((cursor, cut_start))
        cursor = max(cursor, cut_end)
    if cursor < span_end:
        result.append((cursor, span_end))
    return result


def _clean_spans(
    turns: tuple[RttmTurn, ...],
    *,
    merge_gap_s: Decimal,
    overlap_floor_s: Decimal,
) -> dict[str, list[tuple[Decimal, Decimal]]]:
    """Per speaker: merge same-speaker turns, then drop other-speaker overlap.

    Returns ``label -> cleaned single-speaker spans (seconds)``. A label with no
    surviving span is omitted. The minimum-length floor is NOT applied here; it is
    enforced in the sample domain once spans are converted (see
    :func:`_kept_intervals`), because the contract is written in samples.
    """
    by_speaker: dict[str, list[tuple[Decimal, Decimal]]] = {}
    for turn in turns:
        by_speaker.setdefault(turn.speaker_label, []).append((turn.start_s, turn.end_s))
    cleaned: dict[str, list[tuple[Decimal, Decimal]]] = {}
    for label, intervals in by_speaker.items():
        merged = _merge_intervals(intervals, merge_gap_s)
        others = _coalesce_intervals(
            [
                iv
                for other_label, other_ivs in by_speaker.items()
                if other_label != label
                for iv in other_ivs
            ]
        )
        spans: list[tuple[Decimal, Decimal]] = []
        for span in merged:
            spans.extend(_subtract_overlaps(span, others, floor_s=overlap_floor_s))
        if spans:
            cleaned[label] = spans
    return cleaned


# A planned span keyed by (recording, rttm_label, speaker_id) -> the converted
# sample interval that met the length floor, plus its exact source seconds (kept
# only as documentary provenance; the sample interval is the authoritative cut).
_PlannedSpan = tuple[SampleInterval, Decimal, Decimal]


def _kept_intervals(
    cleaned_by_label: dict[str, list[tuple[Decimal, Decimal]]],
    source: OrganicSource,
    recording: str,
    *,
    min_samples: int,
) -> dict[tuple[str, str, str], list[_PlannedSpan]]:
    """Convert cleaned second-spans to sample intervals, keeping those >= min_samples.

    The length floor is applied here, in the SAMPLE domain, so a span whose
    continuous second-length rounds just under the threshold but whose floor/ceil
    interval reaches ``min_samples`` is kept (and vice versa). A label with no
    surviving interval is omitted, so it never enters the split partition.
    """
    kept: dict[tuple[str, str, str], list[_PlannedSpan]] = {}
    for label, spans in cleaned_by_label.items():
        speaker_id = namespaced_speaker(source.source_id, recording, label)
        surviving: list[_PlannedSpan] = []
        for start_s, end_s in spans:
            interval = to_sample_interval(start_s, end_s)
            if interval.n_samples >= min_samples:
                surviving.append((interval, start_s, end_s))
        if surviving:
            kept[(recording, label, speaker_id)] = surviving
    return kept


def _records_from_intervals(
    source: OrganicSource,
    kept_by_speaker: dict[tuple[str, str, str], list[_PlannedSpan]],
    *,
    kind: str,
    split_of_speaker: dict[str, str],
) -> tuple[IngestRecord, ...]:
    """Build stable, split-stamped :class:`IngestRecord`s from kept sample intervals."""
    records: list[IngestRecord] = []
    for (recording, label, speaker_id), spans in sorted(kept_by_speaker.items()):
        for interval, start_s, end_s in spans:
            clip_id = f"{speaker_id}-{kind}-{interval.start_sample}-{interval.end_sample}"
            records.append(
                IngestRecord(
                    clip_id=clip_id,
                    rel_path=f"{source.source_id}/{kind}/{clip_id}.wav",
                    source=source.source_id,
                    recording=recording,
                    speaker_id=speaker_id,
                    label="bona_fide",
                    language=source.language,
                    license_spdx=source.license_spdx,
                    stratum=f"bona_fide|organic|{source.domain}",
                    interval=interval,
                    split=split_of_speaker[speaker_id],
                    acquire={
                        "source_file": f"{recording}.wav",
                        "recording": recording,
                        "rttm_label": label,
                        # The sample offsets are authoritative; seconds are documentary
                        # provenance only (float, so the plan is JSON-serializable). An
                        # executor MUST slice on start_sample/end_sample, never seconds.
                        "start_sample": interval.start_sample,
                        "end_sample": interval.end_sample,
                        "start_s": float(start_s),
                        "end_s": float(end_s),
                        "kind": kind,
                    },
                    kind=kind,
                )
            )
    return tuple(records)


def build_plan(
    source: OrganicSource,
    recordings: dict[str, tuple[RttmTurn, ...]],
    *,
    calibration_fraction: float = 0.5,
    holdout_fraction: float = 0.2,
    seed: str = SELECTION_SEED,
) -> MaterializationPlan:
    """Plan an organic source: cleaned turn clips + merged session segments.

    Splits are assigned once, across every recording of the source, so a speaker
    never straddles two splits (speaker-disjoint by construction). Turn clips feed
    strata/degradation/calibration; session segments (merged same-speaker runs of
    at least one full window) feed production-windowing validation.
    """
    if not recordings:
        raise CorpusError("build_plan: no recordings")
    if not math.isfinite(calibration_fraction) or not math.isfinite(holdout_fraction):
        raise CorpusError("build_plan: fractions must be finite")
    if calibration_fraction < 0 or holdout_fraction < 0:
        raise CorpusError("build_plan: fractions must be non-negative")
    if calibration_fraction + holdout_fraction > 1.0 + 1e-9:
        raise CorpusError("build_plan: calibration + holdout fractions exceed 1.0")
    turn_kept: dict[tuple[str, str, str], list[_PlannedSpan]] = {}
    segment_kept: dict[tuple[str, str, str], list[_PlannedSpan]] = {}
    for recording, turns in recordings.items():
        # The dict key is the authoritative recording id (it drives the speaker
        # namespace); a row that declares a different recording is a caller error.
        for turn in turns:
            if turn.recording != recording:
                raise CorpusError(
                    f"build_plan: recording key {recording!r} has a turn for {turn.recording!r}"
                )
        clean_turns = _clean_spans(
            turns, merge_gap_s=CLIP_MERGE_GAP_S, overlap_floor_s=OVERLAP_FLOOR_S
        )
        clean_segments = _clean_spans(
            turns, merge_gap_s=SESSION_MERGE_GAP_S, overlap_floor_s=OVERLAP_FLOOR_S
        )
        turn_kept.update(
            _kept_intervals(clean_turns, source, recording, min_samples=TURN_MIN_SAMPLES)
        )
        segment_kept.update(
            _kept_intervals(clean_segments, source, recording, min_samples=SEGMENT_MIN_SAMPLES)
        )
    # Only speakers with at least one surviving clip enter the split partition, so a
    # speaker whose every span fell below the floor never shifts the fractional cuts.
    speaker_ids = {sp for (_, _, sp) in turn_kept} | {sp for (_, _, sp) in segment_kept}
    if not speaker_ids:
        raise CorpusError("build_plan: no clips survived cleaning (every span below the floor)")
    split_of_speaker = _partition_free_speakers(
        speaker_ids,
        forced_eval=frozenset(),
        calibration_fraction=calibration_fraction,
        holdout_fraction=holdout_fraction,
        seed=seed,
    )
    return MaterializationPlan(
        schema_version=PLAN_SCHEMA_VERSION,
        source=source.source_id,
        turn_clips=_records_from_intervals(
            source, turn_kept, kind="turn", split_of_speaker=split_of_speaker
        ),
        segments=_records_from_intervals(
            source, segment_kept, kind="segment", split_of_speaker=split_of_speaker
        ),
    )


def _record_to_dict(record: IngestRecord) -> dict[str, Any]:
    return {
        "clip_id": record.clip_id,
        "rel_path": record.rel_path,
        "source": record.source,
        "recording": record.recording,
        "speaker_id": record.speaker_id,
        "label": record.label,
        "language": record.language,
        "license_spdx": record.license_spdx,
        "stratum": record.stratum,
        "start_sample": record.interval.start_sample,
        "end_sample": record.interval.end_sample,
        "split": record.split,
        "acquire": record.acquire,
        "kind": record.kind,
    }


def plan_to_dict(plan: MaterializationPlan) -> dict[str, Any]:
    """A stable, JSON-serializable view of a plan (for the CLI and golden tests)."""
    return {
        "schema_version": plan.schema_version,
        "source": plan.source,
        "turn_clips": [_record_to_dict(r) for r in plan.turn_clips],
        "segments": [_record_to_dict(r) for r in plan.segments],
    }


def finalize_manifest(
    records: tuple[IngestRecord | DegradedRecord, ...] | list[IngestRecord | DegradedRecord],
    measured: dict[str, tuple[str, int]],
) -> Manifest:
    """Build and validate a v1 manifest from a plan plus the executor's measurements.

    ``records`` are turn/segment :class:`IngestRecord`s and/or :class:`DegradedRecord`
    children. ``measured`` maps ``clip_id -> (pcm_sha256, sample_count)``, the two
    facts that exist only after audio is materialized. ``duration_s`` is derived
    from the sample count (``sample_count / 16000``). Every record must have a
    measurement; the resulting manifest is validated by :func:`load_manifest`
    (fail closed, so lineage inheritance and cycles are caught here too).
    """
    if not records:
        raise CorpusError("finalize_manifest: no records")
    # Fail closed on measured/record drift in BOTH directions: a record with no
    # measurement (below) and a measurement for a clip not in the record list. A
    # dropped record would silently vanish from the manifest otherwise.
    record_ids = {record.clip_id for record in records}
    orphan_facts = sorted(set(measured) - record_ids)
    if orphan_facts:
        raise CorpusError(
            f"finalize_manifest: measured facts for clips not in the record list: {orphan_facts}"
        )
    clips: list[dict[str, Any]] = []
    for record in records:
        fact = measured.get(record.clip_id)
        if fact is None:
            raise CorpusError(f"finalize_manifest: no measured facts for clip {record.clip_id!r}")
        if not isinstance(fact, tuple) or len(fact) != 2:
            raise CorpusError(
                f"finalize_manifest: measured fact for {record.clip_id!r} must be "
                "a (pcm_sha256, sample_count) tuple"
            )
        pcm_sha256, sample_count = fact
        if not _is_sha256(pcm_sha256):
            raise CorpusError(
                f"finalize_manifest: clip {record.clip_id!r} sha256 must be 64 lowercase hex"
            )
        if not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count <= 0:
            raise CorpusError(
                f"finalize_manifest: clip {record.clip_id!r} sample_count must be a positive int"
            )
        clips.append(record.clip_dict(pcm_sha256, sample_count))
    return load_manifest({"schema_version": MANIFEST_SCHEMA_VERSION, "clips": clips})


# --------------------------------------------------------------------------- #
# Degradation chains (S5, pure): recipe-id chain strings + ffmpeg argv builders
# + degraded-child derivation. The recipe VOCABULARY is data in
# synthdetect_sources.py; this is the deterministic command + lineage logic.
# --------------------------------------------------------------------------- #
CHAIN_SEPARATOR = "|"
# The pinned raw-PCM I/O framing every ffmpeg pass uses: canonical-PCM in, and
# canonical-PCM out (decode/re-canonicalize, never a WAV-header identity). Kept as
# data so the builder and the pre-registration agree byte-for-byte.
_RAW_INPUT_FRAMING = ("-f", "s16le", "-ar", "16000", "-ac", "1")
_CANONICAL_OUTPUT = ("-f", "s16le", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le")
# Determinism pinning. In ffmpeg's option model `-threads` is per-stream: before
# `-i` it binds the input decoder, before an output it binds that output's encoder.
# The encoder is the hash-relevant pass, so `-threads 1` MUST also appear on the
# output side -- a prefix-only `-threads 1` leaves a multi-threaded encoder
# unpinned. `-filter_threads 1` (global) pins the filter graph (the atempo speed
# pass). Together they hold the intermediate bitstream and the final canonical PCM
# byte-stable across runs on the pinned realization toolchain.
_FFMPEG_PREFIX = ("ffmpeg", "-nostdin", "-y", "-threads", "1", "-filter_threads", "1")
_OUTPUT_THREADS = ("-threads", "1")


def parse_chain(chain: str) -> tuple[str, ...]:
    """Split a canonical degradation-chain string into its ordered recipe ids.

    Fail closed: the chain must be non-empty and every segment must be a recipe id
    known to the registry. Order is significant and preserved.
    """
    if not isinstance(chain, str) or not chain.strip():
        raise CorpusError("degradation chain must be a non-empty string")
    parts = chain.split(CHAIN_SEPARATOR)
    if any(not part for part in parts):
        raise CorpusError(f"degradation chain {chain!r} has an empty segment")
    for part in parts:
        if part not in DEGRADATION_RECIPES:
            raise CorpusError(
                f"degradation chain {chain!r} names unknown recipe {part!r}; "
                f"known: {sorted(DEGRADATION_RECIPES)}"
            )
    return tuple(parts)


def serialize_chain(recipe_ids: tuple[str, ...] | list[str]) -> str:
    """Serialize ordered recipe ids into the one canonical chain string.

    Every id must be known; the result is the ``|``-joined sequence, so the same
    ordered recipes always yield the same manifest ``degradation`` identity.
    """
    ids = tuple(recipe_ids)
    if not ids:
        raise CorpusError("serialize_chain: at least one recipe id is required")
    for recipe_id in ids:
        if recipe_id not in DEGRADATION_RECIPES:
            raise CorpusError(
                f"serialize_chain: unknown recipe {recipe_id!r}; "
                f"known: {sorted(DEGRADATION_RECIPES)}"
            )
    return CHAIN_SEPARATOR.join(ids)


def build_recipe_argv(
    recipe: DegradationRecipe,
    *,
    in_path: str,
    out_path: str,
    intermediate_path: str | None = None,
) -> tuple[tuple[str, ...], ...]:
    """Build the exact ffmpeg command(s) for one recipe (pure, no execution).

    A lossy recipe returns two commands: an encode pass (canonical PCM ->
    ``intermediate_format`` bitstream, needing an ``intermediate_path``) and a
    decode pass (bitstream -> canonical PCM). A non-lossy recipe returns one pass
    straight to canonical PCM. Every raw input is framed with the pinned
    ``-f s16le -ar 16000 -ac 1`` and every output re-canonicalizes to ``pcm_s16le``;
    ``-threads 1`` is emitted on both the input (decoder) and the output (encoder)
    side, with ``-filter_threads 1`` global, so every pass is deterministic.
    """
    for path in (in_path, out_path):
        if path.startswith("-"):
            raise CorpusError(
                f"path {path!r} must not start with '-' (ffmpeg reads it as an option)"
            )
    if recipe.lossy:
        if not intermediate_path:
            raise CorpusError(
                f"recipe {recipe.recipe_id!r} is lossy and needs an intermediate_path"
            )
        if intermediate_path.startswith("-"):
            raise CorpusError(
                f"path {intermediate_path!r} must not start with '-' "
                "(ffmpeg reads it as an option)"
            )
        encode = (
            *_FFMPEG_PREFIX,
            *_RAW_INPUT_FRAMING,
            "-i", in_path,
            *recipe.encode_args,
            *_OUTPUT_THREADS,
            "-f", recipe.intermediate_format,
            intermediate_path,
        )
        decode = (
            *_FFMPEG_PREFIX,
            "-i", intermediate_path,
            *_OUTPUT_THREADS,
            *_CANONICAL_OUTPUT,
            out_path,
        )
        return (encode, decode)
    single = (
        *_FFMPEG_PREFIX,
        *_RAW_INPUT_FRAMING,
        "-i", in_path,
        *recipe.encode_args,
        *_OUTPUT_THREADS,
        *_CANONICAL_OUTPUT,
        out_path,
    )
    return (single,)


def _chain_slug(chain: str) -> str:
    """The chain string as a safe clip-id suffix (``|`` -> ``-``)."""
    return chain.replace(CHAIN_SEPARATOR, "-")


def derive_degraded_record(
    parent: ClipEntry,
    recipe_ids: tuple[str, ...] | list[str],
) -> DegradedRecord:
    """Plan a degraded child of an already-materialized parent clip (pure).

    The child inherits the parent's label, speaker, language, license, and split
    (lineage inheritance); its stratum extends the parent's with the chain; its
    clip id and path are derived deterministically from the parent id and the
    chain. A parent that is itself degraded may be chained further. Only bona fide
    or spoof parents that carry no generator constraint are handled here (organic
    bona fide today); a spoof synthesis parent keeps its own provenance and is out
    of scope for S5.
    """
    chain = serialize_chain(recipe_ids)
    slug = _chain_slug(chain)
    clip_id = f"{parent.clip_id}-{slug}"
    if not _is_safe_id(clip_id):
        raise CorpusError(f"derived clip_id {clip_id!r} is not a safe token")
    # Keep the child beside the parent under a degraded/ subtree, path-safe.
    rel_dir = str(Path(parent.rel_path).parent)
    rel_path = f"{rel_dir}/degraded/{clip_id}.wav" if rel_dir != "." else f"degraded/{clip_id}.wav"
    return DegradedRecord(
        clip_id=clip_id,
        rel_path=rel_path,
        parent_clip_id=parent.clip_id,
        degradation=chain,
        source=parent.source,
        speaker_id=parent.speaker_id,
        label=parent.label,
        language=parent.language,
        license_spdx=parent.license_spdx,
        stratum=f"{parent.stratum}|{chain}",
        split=parent.split,
    )


# --------------------------------------------------------------------------- #
# Prepare executor (S5 PR-2a): materialize bona fide clips by slicing the staged
# source recordings at the plan's integer sample offsets. ffmpeg-free: the staged
# sources are already canonical PCM (16 kHz mono s16le), so materialization is
# deterministic byte slicing of a pin-verified payload, never a codec pass. Inputs
# (audio dir, corpus root, acquisition manifest) are CLI arguments; nothing here is
# committed. See docs/plans/2026-08-26-14-21_synthdetect-s5-pr2-executor.md.
# --------------------------------------------------------------------------- #
_HASH_CHUNK = 1 << 20


@dataclass(frozen=True)
class AcquisitionPin:
    """A pinned staged input file: its relative path, sha256, and byte size.

    ``sha256`` is the digest of the WHOLE staged file (the artifact on disk), which
    is deliberately distinct from a manifest clip's ``sha256`` (the canonical PCM
    ``data``-chunk payload digest). The pin verifies the acquired bytes; the manifest
    identity is the decoded PCM.
    """

    rel_path: str
    sha256: str
    size: int


def _safe_rel_path(rel_path: str, *, where: str) -> str:
    """Validate a relative path stays inside its root (fail closed on traversal)."""
    if not isinstance(rel_path, str) or not rel_path:
        raise CorpusError(f"{where}: rel_path must be a non-empty string")
    pure = Path(rel_path)
    if pure.is_absolute() or rel_path.startswith("-") or ".." in pure.parts:
        raise CorpusError(f"{where}: rel_path {rel_path!r} must be relative and contain no '..'")
    return rel_path


def load_acquisition_manifest(
    obj: Any,
) -> tuple[str, dict[str, AcquisitionPin], tuple[AcquisitionPin, ...]]:
    """Parse and validate an acquisition manifest (pins for the staged inputs).

    Shape: ``{"source": id, "recordings": {rec_id: {rel_path, sha256, size}},
    "rttms": [{rel_path, sha256, size}, ...]}``. Recordings are keyed by the logical
    recording id the plan uses; RTTMs are a content-addressed list (matched by sha,
    so a pin is path-independent). Fails closed on a bad shape, a non-hex sha, a
    non-positive size, an unsafe rel_path, or a duplicate rttm sha.
    """
    if not isinstance(obj, dict):
        raise CorpusError("acquisition manifest must be a JSON object")
    source = obj.get("source")
    if not isinstance(source, str) or not source:
        raise CorpusError("acquisition manifest: 'source' must be a non-empty string")
    raw_recordings = obj.get("recordings")
    if not isinstance(raw_recordings, dict) or not raw_recordings:
        raise CorpusError("acquisition manifest: 'recordings' must be a non-empty object")
    recordings: dict[str, AcquisitionPin] = {}
    seen_rel: dict[str, str] = {}
    for rec_id, entry in raw_recordings.items():
        pin = _acquisition_pin(entry, where=f"recording {rec_id!r}")
        if pin.rel_path in seen_rel:
            raise CorpusError(
                f"acquisition manifest: recordings {seen_rel[pin.rel_path]!r} and {rec_id!r} "
                f"share rel_path {pin.rel_path!r}"
            )
        seen_rel[pin.rel_path] = rec_id
        recordings[rec_id] = pin
    raw_rttms = obj.get("rttms")
    if not isinstance(raw_rttms, list) or not raw_rttms:
        raise CorpusError("acquisition manifest: 'rttms' must be a non-empty list")
    rttms: list[AcquisitionPin] = [
        _acquisition_pin(entry, where=f"rttm #{i}") for i, entry in enumerate(raw_rttms)
    ]
    seen: set[str] = set()
    for pin in rttms:
        if pin.sha256 in seen:
            raise CorpusError(f"acquisition manifest: duplicate rttm sha256 {pin.sha256}")
        seen.add(pin.sha256)
    return source, recordings, tuple(rttms)


def _acquisition_pin(entry: Any, *, where: str) -> AcquisitionPin:
    if not isinstance(entry, dict):
        raise CorpusError(f"{where}: pin must be an object")
    rel_path = _safe_rel_path(entry.get("rel_path", ""), where=where)
    sha256 = entry.get("sha256")
    if not isinstance(sha256, str) or not _is_sha256(sha256):
        raise CorpusError(f"{where}: sha256 must be 64 lowercase hex")
    size = entry.get("size")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise CorpusError(f"{where}: size must be a positive int")
    return AcquisitionPin(rel_path=rel_path, sha256=sha256, size=size)


def _hash_file(path: Path) -> tuple[str, int]:
    """Stream a file once, returning ``(sha256, size)`` (open-once, no reopen race)."""
    hasher = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(_HASH_CHUNK), b""):
                hasher.update(chunk)
                size += len(chunk)
    except OSError as exc:
        raise CorpusError(f"{path}: cannot read: {exc}") from exc
    return hasher.hexdigest(), size


def verify_and_read_rttms(
    rttm_paths: list[str] | tuple[str, ...],
    rttm_pins: tuple[AcquisitionPin, ...],
) -> dict[str, tuple[RttmTurn, ...]]:
    """Read each RTTM once, verify it against a pin by content sha, then parse it.

    Every RTTM file must match exactly one pin (by sha256 and size), and every pin
    must be matched (exact keyset coverage), so the plan is built only from pinned
    bytes. Turns are grouped by the recording id each row declares.
    """
    by_sha = {pin.sha256: pin for pin in rttm_pins}
    matched: set[str] = set()
    recordings: dict[str, list[RttmTurn]] = {}
    for rttm_path in rttm_paths:
        path = Path(rttm_path)
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise CorpusError(f"{path}: cannot read: {exc}") from exc
        sha = hashlib.sha256(data).hexdigest()
        pin = by_sha.get(sha)
        if pin is None:
            raise CorpusError(f"{path}: sha256 {sha} is not pinned in the acquisition manifest")
        if pin.size != len(data):
            raise CorpusError(f"{path}: size {len(data)} does not match pinned size {pin.size}")
        if sha in matched:
            raise CorpusError(f"{path}: rttm sha256 {sha} supplied twice")
        matched.add(sha)
        for turn in parse_rttm(data.decode("utf-8")):
            recordings.setdefault(turn.recording, []).append(turn)
    missing = sorted(set(by_sha) - matched)
    if missing:
        raise CorpusError(f"acquisition manifest names rttm pins never supplied: {missing}")
    return {rec: tuple(turns) for rec, turns in recordings.items()}


@dataclass(frozen=True)
class PrepareResult:
    """Summary of a completed prepare materialization (host-path-free for stdout)."""

    source: str
    turn_clips: int
    segments: int
    recordings: int
    manifest_sha256: str


def _read_verified_recording(
    audio_dir: Path, rec_id: str, recordings: dict[str, AcquisitionPin]
) -> bytes:
    """Resolve, pin-verify, and decode one source recording's canonical payload.

    Reads the file exactly once: the bytes that are hashed for the pin are the bytes
    the payload is decoded from (no verify-then-reopen race). Errors name the logical
    ``rel_path``, not the resolved host path, to keep captured output clean-room.
    """
    pin = recordings.get(rec_id)
    if pin is None:
        raise CorpusError(f"recording {rec_id!r} is not pinned in the acquisition manifest")
    try:
        raw = (audio_dir / pin.rel_path).read_bytes()
    except OSError as exc:
        raise CorpusError(f"recording {pin.rel_path!r}: cannot read: {exc.strerror}") from exc
    sha = hashlib.sha256(raw).hexdigest()
    if sha != pin.sha256:
        raise CorpusError(f"recording {pin.rel_path!r}: sha256 {sha} does not match pinned")
    if len(raw) != pin.size:
        raise CorpusError(
            f"recording {pin.rel_path!r}: size {len(raw)} does not match pinned {pin.size}"
        )
    return canonical_payload_from_bytes(raw, label=f"recording {pin.rel_path!r}")


def _materialize_record(
    record: IngestRecord, payload: bytes, staging: Path
) -> tuple[str, int]:
    """Slice one record from its recording payload, write it, return measured facts."""
    n_samples = len(payload) // _BLOCK_ALIGN
    start, end = record.interval.start_sample, record.interval.end_sample
    if not 0 <= start < end <= n_samples:
        raise CorpusError(
            f"clip {record.clip_id!r}: interval [{start}, {end}) out of range for "
            f"{n_samples}-sample recording (plan drift)"
        )
    clip = payload[start * _BLOCK_ALIGN : end * _BLOCK_ALIGN]
    rel = _safe_rel_path(record.rel_path, where=f"clip {record.clip_id!r}")
    out_path = staging / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_canonical_wav(out_path, clip)
    # Prove the on-disk file is canonical and recovers the exact sliced bytes.
    if read_canonical_wav_payload(out_path) != clip:
        raise CorpusError(f"clip {record.clip_id!r}: written WAV did not round-trip to the slice")
    sha, count = payload_sha_and_count(clip)
    if count != end - start:
        raise CorpusError(
            f"clip {record.clip_id!r}: measured {count} samples, planned {end - start}"
        )
    return sha, count


def materialize_prepare(
    source: OrganicSource,
    recordings_turns: dict[str, tuple[RttmTurn, ...]],
    *,
    corpus_root: Path,
    audio_dir: Path,
    recordings: dict[str, AcquisitionPin],
) -> PrepareResult:
    """Materialize an organic source into a corpus root, atomically, from pins.

    Builds the same plan as dry-run, reads each pin-verified source recording once,
    slices every turn and segment clip by sample offset, writes canonical WAVs,
    finalizes and validates the v1 manifest, and publishes the whole tree by atomic
    rename. Fails closed on any pin mismatch, plan drift, or a populated destination;
    nothing is published on failure.
    """
    if corpus_root.exists() and any(corpus_root.iterdir()):
        raise CorpusError(f"corpus root {corpus_root} is already populated; refusing to overwrite")
    # Exact recording-pin coverage both ways (mirrors the RTTM pin contract): every
    # recording the RTTMs declare has a pin, and every pin is a declared recording,
    # so a surplus or wrong-source pin cannot pass unnoticed.
    declared = set(recordings_turns)
    pinned = set(recordings)
    if declared != pinned:
        missing = sorted(declared - pinned)
        surplus = sorted(pinned - declared)
        raise CorpusError(
            f"recording pin coverage mismatch: missing={missing} surplus={surplus}"
        )
    plan = build_plan(source, recordings_turns)
    all_records = (*plan.turn_clips, *plan.segments)
    needed = sorted({r.recording for r in all_records})
    corpus_root.parent.mkdir(parents=True, exist_ok=True)
    measured: dict[str, tuple[str, int]] = {}
    clip_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    # mkdtemp + explicit cleanup (not TemporaryDirectory), so a cleanup error after a
    # successful os.replace can never turn a published corpus into a CLI failure.
    tmp = tempfile.mkdtemp(dir=corpus_root.parent, prefix=".prepare-")
    try:
        staging = Path(tmp) / "corpus"
        staging.mkdir()
        for rec_id in needed:
            payload = _read_verified_recording(audio_dir, rec_id, recordings)
            rec_sha, rec_count = payload_sha_and_count(payload)
            source_rows.append(
                {"recording": rec_id, "payload_sha256": rec_sha, "n_samples": rec_count}
            )
            for record in all_records:
                if record.recording != rec_id:
                    continue
                sha, count = _materialize_record(record, payload, staging)
                measured[record.clip_id] = (sha, count)
                clip_rows.append(
                    {
                        "clip_id": record.clip_id,
                        "recording": record.recording,
                        "start_sample": record.interval.start_sample,
                        "end_sample": record.interval.end_sample,
                        "sha256": sha,
                        "n_samples": count,
                    }
                )
        manifest = finalize_manifest(all_records, measured)
        manifest_sha = _write_prepare_artifacts(
            staging, plan, manifest, all_records, measured, source_rows, clip_rows
        )
        os.replace(staging, corpus_root)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return PrepareResult(
        source=source.source_id,
        turn_clips=len(plan.turn_clips),
        segments=len(plan.segments),
        recordings=len(needed),
        manifest_sha256=manifest_sha,
    )


def _write_prepare_artifacts(
    staging: Path,
    plan: MaterializationPlan,
    manifest: Manifest,
    all_records: tuple[IngestRecord | DegradedRecord, ...],
    measured: dict[str, tuple[str, int]],
    source_rows: list[dict[str, Any]],
    clip_rows: list[dict[str, Any]],
) -> str:
    """Write the manifest and the (sorted, host-path-free) prepare receipts.

    The manifest.json is serialized from the exact clip dicts ``finalize_manifest``
    validated (``record.clip_dict(sha, count)``), so the written manifest is the one
    that passed ``load_manifest``, byte for byte. Returns the sha256 of the written
    manifest bytes (a whole-corpus fingerprint for audit).
    """
    clips = sorted(
        (record.clip_dict(*measured[record.clip_id]) for record in all_records),
        key=lambda c: c["clip_id"],
    )
    manifest_bytes = (
        json.dumps(
            {"schema_version": manifest.schema_version, "clips": clips}, indent=2, sort_keys=True
        )
        + "\n"
    ).encode("utf-8")
    (staging / "manifest.json").write_bytes(manifest_bytes)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    clip_rows_sorted = sorted(clip_rows, key=lambda r: r["clip_id"])
    (staging / "clip_receipt.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in clip_rows_sorted),
        encoding="utf-8",
    )
    (staging / "prepare_receipt.json").write_text(
        json.dumps(
            {
                "source": plan.source,
                "canonicalization_id": CANONICALIZATION_ID,
                "plan_schema_version": plan.schema_version,
                "manifest_schema_version": manifest.schema_version,
                "manifest_sha256": manifest_sha,
                "turn_clips": len(plan.turn_clips),
                "segments": len(plan.segments),
                "recordings": sorted(source_rows, key=lambda r: r["recording"]),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_sha


# --------------------------------------------------------------------------- #
# Degrade executor (S5 PR-2b): materialize degraded children of already-
# materialized bona fide clips by running the frozen build_recipe_argv ffmpeg
# round trips inside a digest-pinned container. The output is a separate
# immutable degrade root containing child WAVs plus a combined parent+child
# manifest validated through load_manifest. See the PR-2b plan for the B1/B2/B5
# blocker resolutions.
# --------------------------------------------------------------------------- #
_CONTAINER_IMAGE_RE = re.compile(r"^[a-zA-Z0-9._/-]+@sha256:[0-9a-f]{64}$")
_DEGRADE_LENGTH_MIN = 0.5
_DEGRADE_LENGTH_MAX = 2.0
_DEGRADE_SUBPROCESS_TIMEOUT = 300
_DEGRADE_STDERR_CAP = 4096


@dataclass(frozen=True)
class DegradeResult:
    """Summary of a completed degrade materialization (host-path-free for stdout)."""

    parent_manifest_sha256: str
    combined_manifest_sha256: str
    container_image: str
    children: int
    parents_reaudited: int


def _run_containerized_ffmpeg(
    argv: tuple[str, ...],
    *,
    workdir: Path,
    container_image: str,
    timeout: int = _DEGRADE_SUBPROCESS_TIMEOUT,
) -> None:
    """Run one ffmpeg command inside the digest-pinned container.

    Strips the ``"ffmpeg"`` prefix from ``argv`` (since ``--entrypoint ffmpeg`` is
    set explicitly) and bind-mounts ``workdir`` at ``/work``. ``--network none``
    isolates the container. After the run, verifies the output file (the last
    element of argv) exists in workdir as a non-empty regular file (not a symlink).
    """
    if not argv or argv[0] != "ffmpeg":
        raise CorpusError(f"container ffmpeg: argv must start with 'ffmpeg', got {argv[:1]!r}")
    ff_args = argv[1:]
    out_container_path = argv[-1]
    if not out_container_path.startswith("/work/"):
        raise CorpusError(
            f"container ffmpeg: output path {out_container_path!r} must be under /work/"
        )
    out_host_path = workdir / out_container_path.removeprefix("/work/")
    cmd = [
        "docker", "run", "--rm", "--network", "none",
        "--entrypoint", "ffmpeg",
        "-v", f"{workdir}:/work:rw",
        container_image,
        *ff_args,
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise CorpusError(
            f"container ffmpeg timed out after {timeout}s"
        ) from None
    if proc.returncode != 0:
        stderr_tail = proc.stderr[-_DEGRADE_STDERR_CAP:].decode("utf-8", errors="replace")
        raise CorpusError(f"container ffmpeg failed (rc={proc.returncode}): {stderr_tail}")
    if not out_host_path.exists():
        raise CorpusError("container ffmpeg: output file missing after run")
    if out_host_path.is_symlink():
        raise CorpusError("container ffmpeg: output file is a symlink")
    if out_host_path.stat().st_size == 0:
        raise CorpusError("container ffmpeg: output file is empty")


def _degrade_one_clip(
    parent_clip: ClipEntry,
    child_record: DegradedRecord,
    recipe_chain: tuple[str, ...],
    *,
    parent_root: Path,
    staging: Path,
    container_image: str,
) -> tuple[str, int]:
    """Execute the full recipe chain for one degraded child and return measured facts.

    B1 resolution: reads the parent WAV's data-chunk payload (headerless s16le),
    writes it as a raw file, pipes it through the recipe chain (each stage's output
    feeds the next), wraps the final raw output in a canonical WAV, and proves the
    round-trip. Returns ``(pcm_sha256, sample_count)``.
    """
    parent_wav = parent_root / parent_clip.rel_path
    parent_payload = read_canonical_wav_payload(parent_wav)
    parent_sha, parent_count = payload_sha_and_count(parent_payload)
    if parent_sha != parent_clip.sha256:
        raise CorpusError(
            f"clip {parent_clip.clip_id!r}: parent WAV sha256 {parent_sha} "
            f"does not match manifest {parent_clip.sha256}"
        )

    clip_workdir = staging / "_work" / child_record.clip_id
    clip_workdir.mkdir(parents=True, exist_ok=True)
    (clip_workdir / "input.raw").write_bytes(parent_payload)

    current_input = f"/work/_work/{child_record.clip_id}/input.raw"
    for i, recipe_id in enumerate(recipe_chain):
        recipe = DEGRADATION_RECIPES[recipe_id]
        stage_out = f"/work/_work/{child_record.clip_id}/stage-{i}.raw"
        if recipe.lossy:
            fmt = recipe.intermediate_format
            intermediate = f"/work/_work/{child_record.clip_id}/intermediate-{i}.{fmt}"
        else:
            intermediate = None
        cmds = build_recipe_argv(
            recipe,
            in_path=current_input,
            out_path=stage_out,
            intermediate_path=intermediate,
        )
        for cmd in cmds:
            _run_containerized_ffmpeg(cmd, workdir=staging, container_image=container_image)
        current_input = stage_out

    final_host = staging / current_input.removeprefix("/work/")
    raw_output = final_host.read_bytes()
    if len(raw_output) % _BLOCK_ALIGN != 0:
        raise CorpusError(
            f"clip {child_record.clip_id!r}: raw output is {len(raw_output)} bytes, "
            f"not a whole number of {_BLOCK_ALIGN}-byte frames"
        )
    if not raw_output:
        raise CorpusError(f"clip {child_record.clip_id!r}: raw output is empty")

    child_count = len(raw_output) // _BLOCK_ALIGN
    if child_count < parent_count * _DEGRADE_LENGTH_MIN:
        raise CorpusError(
            f"clip {child_record.clip_id!r}: output {child_count} samples is below "
            f"{_DEGRADE_LENGTH_MIN:.0%} of parent {parent_count}"
        )
    if child_count > parent_count * _DEGRADE_LENGTH_MAX:
        raise CorpusError(
            f"clip {child_record.clip_id!r}: output {child_count} samples exceeds "
            f"{_DEGRADE_LENGTH_MAX:.0%} of parent {parent_count}"
        )

    child_rel = _safe_rel_path(child_record.rel_path, where=f"clip {child_record.clip_id!r}")
    child_wav_path = staging / child_rel
    child_wav_path.parent.mkdir(parents=True, exist_ok=True)
    write_canonical_wav(child_wav_path, raw_output)
    roundtrip = read_canonical_wav_payload(child_wav_path)
    if roundtrip != raw_output:
        raise CorpusError(
            f"clip {child_record.clip_id!r}: written WAV did not round-trip to the raw output"
        )
    sha, count = payload_sha_and_count(raw_output)
    return sha, count


def _assemble_combined_manifest(
    parent_clip_dicts: list[dict[str, Any]],
    child_records: list[DegradedRecord],
    measured: dict[str, tuple[str, int]],
) -> tuple[Manifest, list[dict[str, Any]]]:
    """Build and validate a combined parent+child manifest (B2 resolution).

    Reproduces ``finalize_manifest``'s measurement-coverage rails for the children
    while accepting parent clip dicts (already validated, already measured) as-is.
    Returns the validated :class:`Manifest` plus the combined clip dicts in
    deterministic order for serialization.
    """
    if not child_records:
        raise CorpusError("_assemble_combined_manifest: no child records")
    child_ids = {r.clip_id for r in child_records}
    measured_ids = set(measured)
    if child_ids != measured_ids:
        missing = sorted(child_ids - measured_ids)
        orphan = sorted(measured_ids - child_ids)
        raise CorpusError(
            f"_assemble_combined_manifest: child/measurement keyset mismatch "
            f"missing={missing} orphan={orphan}"
        )
    child_dicts: list[dict[str, Any]] = []
    for record in child_records:
        sha, count = measured[record.clip_id]
        if not _is_sha256(sha):
            raise CorpusError(
                f"_assemble_combined_manifest: clip {record.clip_id!r} sha256 "
                f"must be 64 lowercase hex"
            )
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise CorpusError(
                f"_assemble_combined_manifest: clip {record.clip_id!r} sample_count "
                f"must be a positive int"
            )
        child_dicts.append(record.clip_dict(sha, count))
    combined = parent_clip_dicts + child_dicts
    manifest = load_manifest({"schema_version": MANIFEST_SCHEMA_VERSION, "clips": combined})
    return manifest, combined


def resolve_clip_path(clip: ClipEntry, *, roots: tuple[Path, ...]) -> Path:
    """Resolve a clip's ``rel_path`` against an ordered list of roots (exact-once).

    Returns the resolved path. Fails closed if the clip resolves in zero or
    multiple roots (the same rel_path must exist as a regular file in exactly one).
    """
    found: list[Path] = []
    for root in roots:
        candidate = root / clip.rel_path
        if candidate.is_file() and not candidate.is_symlink():
            found.append(candidate)
    if len(found) == 0:
        raise CorpusError(
            f"clip {clip.clip_id!r}: rel_path {clip.rel_path!r} not found in any root"
        )
    if len(found) > 1:
        raise CorpusError(
            f"clip {clip.clip_id!r}: rel_path {clip.rel_path!r} found in multiple roots"
        )
    return found[0]


def _write_degrade_artifacts(
    staging: Path,
    manifest: Manifest,
    combined_clip_dicts: list[dict[str, Any]],
    child_records: list[DegradedRecord],
    measured: dict[str, tuple[str, int]],
    parent_manifest_sha: str,
    parent_reaudit: list[dict[str, Any]],
    container_image: str,
    toolchain_info: dict[str, str],
) -> str:
    """Write manifest.json, clip_receipt.jsonl, and degrade_receipt.json.

    Returns the sha256 of the written manifest bytes.
    """
    clips_sorted = sorted(combined_clip_dicts, key=lambda c: c["clip_id"])
    manifest_bytes = (
        json.dumps(
            {"schema_version": manifest.schema_version, "clips": clips_sorted},
            indent=2, sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    (staging / "manifest.json").write_bytes(manifest_bytes)
    combined_manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()

    child_rows = sorted(
        (
            {
                "clip_id": r.clip_id,
                "parent_clip_id": r.parent_clip_id,
                "degradation": r.degradation,
                "sha256": measured[r.clip_id][0],
                "n_samples": measured[r.clip_id][1],
            }
            for r in child_records
        ),
        key=lambda row: str(row["clip_id"]),
    )
    (staging / "clip_receipt.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in child_rows),
        encoding="utf-8",
    )
    (staging / "degrade_receipt.json").write_text(
        json.dumps(
            {
                "canonicalization_id": CANONICALIZATION_ID,
                "children": len(child_records),
                "combined_manifest_sha256": combined_manifest_sha,
                "container_image": container_image,
                "manifest_schema_version": manifest.schema_version,
                "parent_manifest_sha256": parent_manifest_sha,
                "parent_reaudit": sorted(parent_reaudit, key=lambda r: r["clip_id"]),
                "parents_reaudited": len(parent_reaudit),
                "toolchain": toolchain_info,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return combined_manifest_sha


def materialize_degrade(
    *,
    parent_root: Path,
    corpus_root: Path,
    container_image: str,
    recipe_ids: tuple[str, ...],
    split_filter: str | None = None,
    toolchain_info: dict[str, str] | None = None,
) -> DegradeResult:
    """Materialize degraded children of a parent bona fide corpus, atomically.

    Reads and re-audits the parent manifest, derives degraded children via
    ``derive_degraded_record``, executes the recipe chain in the digest-pinned
    container, and publishes a separate immutable degrade root with a combined
    parent+child manifest.
    """
    # -- preflight ---------------------------------------------------------- #
    if not _CONTAINER_IMAGE_RE.match(container_image):
        raise CorpusError(
            f"container image must be <repo>@sha256:<64hex>, got {container_image!r}"
        )
    if corpus_root.exists() and any(corpus_root.iterdir()):
        raise CorpusError(
            f"degrade root {corpus_root} is already populated; refusing to overwrite"
        )
    parent_manifest_path = parent_root / "manifest.json"
    if not parent_manifest_path.is_file():
        raise CorpusError("parent root missing manifest.json")
    parent_manifest_bytes = parent_manifest_path.read_bytes()
    parent_manifest_sha = hashlib.sha256(parent_manifest_bytes).hexdigest()
    parent_obj = json.loads(parent_manifest_bytes)
    parent_manifest = load_manifest(parent_obj)
    if parent_manifest.schema_version != MANIFEST_SCHEMA_VERSION:
        raise CorpusError(
            f"degrade requires a v{MANIFEST_SCHEMA_VERSION} manifest, "
            f"got v{parent_manifest.schema_version}"
        )
    if parent_manifest.corpus_kind != CORPUS_KIND_SYNTHESIS:
        raise CorpusError(
            f"degrade requires corpus_kind {CORPUS_KIND_SYNTHESIS!r}, "
            f"got {parent_manifest.corpus_kind!r}"
        )
    has_degraded = any(c.parent_clip_id is not None for c in parent_manifest.clips)
    if has_degraded:
        raise CorpusError("parent manifest already contains degraded entries")
    non_bonafide = [c.clip_id for c in parent_manifest.clips if c.label != "bona_fide"]
    if non_bonafide:
        raise CorpusError(
            f"degrade requires all parent clips to be bona_fide; "
            f"found {len(non_bonafide)} non-bona_fide"
        )
    for clip in parent_manifest.clips:
        clip_path = parent_root / clip.rel_path
        if not clip_path.is_file() or clip_path.is_symlink():
            raise CorpusError(
                f"parent clip {clip.clip_id!r}: {clip.rel_path!r} is not a regular file"
            )

    # -- derive children ---------------------------------------------------- #
    eligible = [
        c for c in parent_manifest.clips
        if c.parent_clip_id is None
        and (split_filter is None or c.split == split_filter)
    ]
    if not eligible:
        raise CorpusError(
            f"no eligible parent clips (split={split_filter!r})"
        )
    child_records: list[DegradedRecord] = []
    for parent_clip in eligible:
        child_records.append(derive_degraded_record(parent_clip, recipe_ids))

    # collision check: child clip_ids and rel_paths must not collide with parents
    parent_ids = {c.clip_id for c in parent_manifest.clips}
    parent_paths = {c.rel_path for c in parent_manifest.clips}
    child_ids = {r.clip_id for r in child_records}
    child_paths = {r.rel_path for r in child_records}
    id_collision = child_ids & parent_ids
    if id_collision:
        raise CorpusError(f"child clip_id collides with parent: {sorted(id_collision)}")
    path_collision = child_paths & parent_paths
    if path_collision:
        raise CorpusError(f"child rel_path collides with parent: {sorted(path_collision)}")
    if len(child_ids) != len(child_records):
        raise CorpusError("duplicate child clip_ids in derivation")
    if len(child_paths) != len(child_records):
        raise CorpusError("duplicate child rel_paths in derivation")

    # -- re-audit parents + execute children -------------------------------- #
    corpus_root.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.mkdtemp(dir=corpus_root.parent, prefix=".degrade-")
    try:
        staging = Path(tmp) / "corpus"
        staging.mkdir()

        parent_reaudit: list[dict[str, Any]] = []
        for clip in parent_manifest.clips:
            payload = read_canonical_wav_payload(parent_root / clip.rel_path)
            sha, count = payload_sha_and_count(payload)
            if sha != clip.sha256:
                raise CorpusError(
                    f"parent re-audit failed for {clip.clip_id!r}: "
                    f"sha256 {sha} != manifest {clip.sha256}"
                )
            expected_duration = _duration_s_from_samples(count)
            if clip.duration_s != expected_duration:
                raise CorpusError(
                    f"parent re-audit failed for {clip.clip_id!r}: "
                    f"duration_s {clip.duration_s} != computed {expected_duration} "
                    f"(sample_count={count})"
                )
            parent_reaudit.append(
                {"clip_id": clip.clip_id, "sha256": sha, "n_samples": count, "ok": True}
            )

        measured: dict[str, tuple[str, int]] = {}
        recipe_chain = recipe_ids
        for child_record in child_records:
            parent_clip = next(
                c for c in parent_manifest.clips if c.clip_id == child_record.parent_clip_id
            )
            sha, count = _degrade_one_clip(
                parent_clip,
                child_record,
                recipe_chain,
                parent_root=parent_root,
                staging=staging,
                container_image=container_image,
            )
            measured[child_record.clip_id] = (sha, count)

        parent_clip_dicts = [
            {
                "clip_id": c.clip_id,
                "rel_path": c.rel_path,
                "sha256": c.sha256,
                "duration_s": c.duration_s,
                "label": c.label,
                "language": c.language,
                "license_spdx": c.license_spdx,
                "stratum": c.stratum,
                "source": c.source,
                "speaker_id": c.speaker_id,
                "split": c.split,
                "degradation": c.degradation,
                "parent_clip_id": c.parent_clip_id,
                "acquire": c.acquire,
            }
            for c in parent_manifest.clips
        ]
        manifest, combined_dicts = _assemble_combined_manifest(
            parent_clip_dicts, child_records, measured
        )
        combined_sha = _write_degrade_artifacts(
            staging, manifest, combined_dicts, child_records, measured,
            parent_manifest_sha, parent_reaudit, container_image,
            toolchain_info or {},
        )

        # Clean up workdir before publish (not part of the artifact)
        work_dir = staging / "_work"
        if work_dir.exists():
            shutil.rmtree(work_dir)

        os.replace(staging, corpus_root)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return DegradeResult(
        parent_manifest_sha256=parent_manifest_sha,
        combined_manifest_sha256=combined_sha,
        container_image=container_image,
        children=len(child_records),
        parents_reaudited=len(parent_reaudit),
    )


# --------------------------------------------------------------------------- #
# CLI: validate a manifest file (schema + integrity, no audio)
# --------------------------------------------------------------------------- #
def cmd_validate(args: argparse.Namespace) -> int:
    try:
        obj = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        manifest = load_manifest(obj)
    except (OSError, json.JSONDecodeError, CorpusError) as exc:
        sys.stderr.write(f"manifest invalid: {exc}\n")
        return 2
    assigned = {c.clip_id: c.split for c in manifest.clips if c.split is not None}
    summary = split_summary(assigned) if assigned else {}
    sys.stdout.write(
        json.dumps(
            {
                "ok": True,
                "schema_version": manifest.schema_version,
                "corpus_kind": manifest.corpus_kind,
                "benchmark": manifest.benchmark,
                "clips": len(manifest.clips),
                "labels": {
                    label: sum(1 for c in manifest.clips if c.label == label) for label in LABELS
                },
                "split_summary": summary,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    """Plan an organic source from its RTTMs, or materialize it under a corpus root.

    Without ``--corpus-root`` this is the dry-run: read the RTTMs, group by recording,
    build the plan, print it as JSON, touch no audio. With ``--corpus-root`` it
    executes (PR-2a): the RTTMs and source recordings are pin-verified against the
    ``--acquisition-manifest``, each recording is sliced at the plan's sample offsets
    into canonical WAV clips, the v1 manifest is finalized, and the tree is published
    atomically. The executed plan is identical to the dry-run plan.
    """
    source = ORGANIC_SOURCES.get(args.source)
    if source is None:
        sys.stderr.write(f"unknown source {args.source!r}; known: {sorted(ORGANIC_SOURCES)}\n")
        return 2
    if args.corpus_root is None:
        recordings: dict[str, list[RttmTurn]] = {}
        try:
            for rttm_path in args.rttm:
                for turn in parse_rttm(Path(rttm_path).read_text(encoding="utf-8")):
                    recordings.setdefault(turn.recording, []).append(turn)
            plan = build_plan(source, {rec: tuple(turns) for rec, turns in recordings.items()})
        except (OSError, CorpusError) as exc:
            sys.stderr.write(f"prepare failed: {exc}\n")
            return 2
        sys.stdout.write(json.dumps(plan_to_dict(plan), indent=2, sort_keys=True) + "\n")
        return 0
    if args.audio_dir is None or args.acquisition_manifest is None:
        sys.stderr.write(
            "prepare --corpus-root requires --audio-dir and --acquisition-manifest\n"
        )
        return 2
    try:
        acq = json.loads(Path(args.acquisition_manifest).read_text(encoding="utf-8"))
        acq_source, recording_pins, rttm_pins = load_acquisition_manifest(acq)
        if acq_source != source.source_id:
            raise CorpusError(
                f"acquisition manifest source {acq_source!r} does not match "
                f"--source {source.source_id!r}"
            )
        recordings_turns = verify_and_read_rttms(args.rttm, rttm_pins)
        result = materialize_prepare(
            source,
            recordings_turns,
            corpus_root=Path(args.corpus_root),
            audio_dir=Path(args.audio_dir),
            recordings=recording_pins,
        )
    except (OSError, json.JSONDecodeError, CorpusError) as exc:
        sys.stderr.write(f"prepare failed: {exc}\n")
        return 2
    sys.stdout.write(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n")
    return 0


def cmd_degrade(args: argparse.Namespace) -> int:
    """Plan degraded children (dry-run), or materialize them with --corpus-root.

    Without ``--corpus-root`` this is the unchanged dry-run: loads a parent
    manifest, derives degraded children per non-degraded parent clip, prints the
    plan as JSON. With ``--corpus-root`` it executes: re-audits the parent corpus,
    runs the recipe chain in the pinned container, and publishes a separate
    immutable degrade root with a combined parent+child manifest.
    """
    if args.corpus_root is None:
        try:
            obj = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
            manifest = load_manifest(obj)
            chain = serialize_chain(args.recipe)
            children = [
                asdict(derive_degraded_record(clip, args.recipe))
                for clip in manifest.clips
                if (args.split is None or clip.split == args.split)
                and clip.parent_clip_id is None
            ]
        except (OSError, json.JSONDecodeError, CorpusError) as exc:
            sys.stderr.write(f"degrade failed: {exc}\n")
            return 2
        if not children:
            sys.stderr.write(
                f"degrade: no non-degraded parent clips in split {args.split!r}\n"
            )
            return 2
        sys.stdout.write(
            json.dumps({"chain": chain, "children": children}, indent=2, sort_keys=True) + "\n"
        )
        return 0
    # execution mode
    if args.parent_root is None or args.container_image is None:
        sys.stderr.write(
            "degrade --corpus-root requires --parent-root and --container-image\n"
        )
        return 2
    try:
        result = materialize_degrade(
            parent_root=Path(args.parent_root),
            corpus_root=Path(args.corpus_root),
            container_image=args.container_image,
            recipe_ids=tuple(args.recipe),
            split_filter=args.split,
        )
    except (OSError, json.JSONDecodeError, CorpusError) as exc:
        sys.stderr.write(f"degrade failed: {exc}\n")
        return 2
    sys.stdout.write(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="synthdetect corpus manifest tools (#144)")
    sub = parser.add_subparsers(dest="command", required=True)
    p_validate = sub.add_parser("validate", help="validate a manifest file (schema + integrity)")
    p_validate.add_argument("--manifest", required=True, help="path to the manifest JSON")
    p_validate.set_defaults(func=cmd_validate)

    p_prepare = sub.add_parser(
        "prepare",
        help="plan an organic source from its RTTMs (dry-run), or materialize it "
        "with --corpus-root",
    )
    p_prepare.add_argument(
        "--source", required=True, help=f"organic source id ({sorted(ORGANIC_SOURCES)})"
    )
    p_prepare.add_argument("--rttm", required=True, nargs="+", help="one or more RTTM files")
    p_prepare.add_argument(
        "--corpus-root",
        default=None,
        help="materialize into this (must-be-empty) root instead of printing the plan",
    )
    p_prepare.add_argument(
        "--audio-dir",
        default=None,
        help="directory holding the source recordings (required with --corpus-root)",
    )
    p_prepare.add_argument(
        "--acquisition-manifest",
        default=None,
        help="JSON pin file for the source recordings + RTTMs (required with --corpus-root)",
    )
    p_prepare.set_defaults(func=cmd_prepare)

    p_degrade = sub.add_parser(
        "degrade",
        help="plan degraded children of a parent manifest (dry-run), or materialize "
        "them with --corpus-root",
    )
    p_degrade.add_argument("--manifest", required=True, help="path to the parent manifest JSON")
    p_degrade.add_argument(
        "--recipe",
        required=True,
        nargs="+",
        help="ordered degradation recipe id(s) forming a chain",
    )
    p_degrade.add_argument(
        "--split",
        default=None,
        choices=SPLITS,
        help="restrict to parents in this split (default: all; PR-3 uses calibration)",
    )
    p_degrade.add_argument(
        "--corpus-root",
        default=None,
        help="degrade output root (must be empty/nonexistent); triggers execution mode",
    )
    p_degrade.add_argument(
        "--parent-root",
        default=None,
        help="parent corpus root containing manifest.json (required with --corpus-root)",
    )
    p_degrade.add_argument(
        "--container-image",
        default=None,
        help="digest-pinned ffmpeg container (image@sha256:...; required with --corpus-root)",
    )
    p_degrade.set_defaults(func=cmd_degrade)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
