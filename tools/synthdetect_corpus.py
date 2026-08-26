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
import json
import math
import sys
from dataclasses import asdict, dataclass
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
    # Lineage inheritance (S5): a degraded child is the SAME speaker, label,
    # language, split, and license as its parent -- it is an audio-domain transform,
    # not a new sample. A mismatch means a manifest attached a child to the wrong
    # parent or mislabelled it, which the per-clip validator cannot see.
    for c in clips:
        if c.parent_clip_id is None:
            continue
        parent = by_id[c.parent_clip_id]
        for field in ("label", "speaker_id", "language", "split", "license_spdx"):
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
BYTES_PER_SAMPLE = 2  # pcm-s16le-mono-16000-v1: signed 16-bit little-endian
# The AASIST fixed model input width; a session segment must reach this to score
# one full production window (docs/gpu-contracts.md, the windowing policy).
MODEL_WIDTH_SAMPLES = 64600

# Pinned corpus-construction floors (seconds). Pre-registered in
# docs/gpu-contracts.md so a change is a visible, deliberate diff, never drift.
CLIP_MERGE_GAP_S = 0.3  # merge same-speaker turns/words within this gap into one turn clip
SESSION_MERGE_GAP_S = 1.0  # production windowing merge_gap_s: merge turns for the segment view
OVERLAP_FLOOR_S = 0.1  # ignore other-speaker overlaps shorter than this when cleaning a turn
TURN_MIN_S = 1.0  # drop a cleaned turn sub-span shorter than this
SEGMENT_MIN_S = MODEL_WIDTH_SAMPLES / CANONICAL_SAMPLE_RATE  # 4.0375s: one full window

PLAN_SCHEMA_VERSION = 1


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
    """One speaker-active interval parsed from an RTTM row (seconds)."""

    recording: str
    speaker_label: str
    start_s: float
    dur_s: float

    @property
    def end_s(self) -> float:
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
            "duration_s": sample_count / CANONICAL_SAMPLE_RATE,
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
            "duration_s": sample_count / CANONICAL_SAMPLE_RATE,
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

    Requires the RTTM 10-field ``SPEAKER <rec> <chan> <start> <dur> <NA> <NA>
    <label> ...`` shape with a finite ``start >= 0``, ``dur > 0``, and a real
    speaker label. If ``recording`` is given, every row's recording id must match.
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
            start = float(fields[3])
            dur = float(fields[4])
        except ValueError:
            raise CorpusError(f"rttm line {lineno}: non-numeric start/dur") from None
        if not math.isfinite(start) or not math.isfinite(dur) or start < 0 or dur <= 0:
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


def to_sample_interval(start_s: float, end_s: float) -> SampleInterval:
    """Convert a second span to a pinned half-open sample interval.

    ``start_sample = floor(start_s * 16000)``, ``end_sample = ceil(end_s * 16000)``
    -- pinned so the corpus is byte-reproducible from the RTTM times.
    """
    if (
        not math.isfinite(start_s)
        or not math.isfinite(end_s)
        or start_s < 0
        or end_s <= start_s
    ):
        raise CorpusError(f"to_sample_interval: need 0 <= start < end, got {start_s}, {end_s}")
    return SampleInterval(
        start_sample=math.floor(start_s * CANONICAL_SAMPLE_RATE),
        end_sample=math.ceil(end_s * CANONICAL_SAMPLE_RATE),
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
    intervals: list[tuple[float, float]], gap_s: float
) -> list[tuple[float, float]]:
    """Sort and merge intervals whose gap is ``< gap_s`` (overlaps always merge)."""
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if merged and start - merged[-1][1] < gap_s:
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _subtract_overlaps(
    span: tuple[float, float],
    others: list[tuple[float, float]],
    *,
    floor_s: float,
) -> list[tuple[float, float]]:
    """Return the sub-spans of ``span`` not covered by any OTHER-speaker interval.

    An other-speaker interval is only treated as overlapping when its intersection
    with ``span`` is at least ``floor_s`` (sub-``floor_s`` grazes are ignored so a
    boundary touch does not fragment a turn).
    """
    span_start, span_end = span
    active: list[tuple[float, float]] = []
    for other_start, other_end in others:
        lo = max(span_start, other_start)
        hi = min(span_end, other_end)
        if hi - lo >= floor_s:
            active.append((lo, hi))
    result: list[tuple[float, float]] = []
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
    merge_gap_s: float,
    min_s: float,
    overlap_floor_s: float,
) -> dict[str, list[tuple[float, float]]]:
    """Per speaker: merge same-speaker turns, drop other-speaker overlap, floor.

    Returns ``label -> cleaned single-speaker spans (seconds)``. A label with no
    surviving span is omitted.
    """
    by_speaker: dict[str, list[tuple[float, float]]] = {}
    for turn in turns:
        by_speaker.setdefault(turn.speaker_label, []).append((turn.start_s, turn.end_s))
    cleaned: dict[str, list[tuple[float, float]]] = {}
    for label, intervals in by_speaker.items():
        merged = _merge_intervals(intervals, merge_gap_s)
        others = [
            iv
            for other_label, other_ivs in by_speaker.items()
            if other_label != label
            for iv in other_ivs
        ]
        spans: list[tuple[float, float]] = []
        for span in merged:
            for sub in _subtract_overlaps(span, others, floor_s=overlap_floor_s):
                if sub[1] - sub[0] >= min_s:
                    spans.append(sub)
        if spans:
            cleaned[label] = spans
    return cleaned


def _records_from_spans(
    source: OrganicSource,
    spans_by_speaker: dict[tuple[str, str, str], list[tuple[float, float]]],
    *,
    kind: str,
    split_of_speaker: dict[str, str],
) -> tuple[IngestRecord, ...]:
    """Build stable, split-stamped :class:`IngestRecord`s from cleaned spans."""
    records: list[IngestRecord] = []
    for (recording, label, speaker_id), spans in sorted(spans_by_speaker.items()):
        for start_s, end_s in spans:
            interval = to_sample_interval(start_s, end_s)
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
                        "start_sample": interval.start_sample,
                        "end_sample": interval.end_sample,
                        "start_s": round(start_s, 6),
                        "end_s": round(end_s, 6),
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
    turn_spans: dict[tuple[str, str, str], list[tuple[float, float]]] = {}
    segment_spans: dict[tuple[str, str, str], list[tuple[float, float]]] = {}
    speaker_ids: set[str] = set()
    for recording, turns in recordings.items():
        clean_turns = _clean_spans(
            turns, merge_gap_s=CLIP_MERGE_GAP_S, min_s=TURN_MIN_S, overlap_floor_s=OVERLAP_FLOOR_S
        )
        clean_segments = _clean_spans(
            turns,
            merge_gap_s=SESSION_MERGE_GAP_S,
            min_s=SEGMENT_MIN_S,
            overlap_floor_s=OVERLAP_FLOOR_S,
        )
        for label, spans in clean_turns.items():
            speaker_id = namespaced_speaker(source.source_id, recording, label)
            speaker_ids.add(speaker_id)
            turn_spans[(recording, label, speaker_id)] = spans
        for label, spans in clean_segments.items():
            speaker_id = namespaced_speaker(source.source_id, recording, label)
            speaker_ids.add(speaker_id)
            segment_spans[(recording, label, speaker_id)] = spans
    if not math.isfinite(calibration_fraction) or not math.isfinite(holdout_fraction):
        raise CorpusError("build_plan: fractions must be finite")
    if calibration_fraction < 0 or holdout_fraction < 0:
        raise CorpusError("build_plan: fractions must be non-negative")
    if calibration_fraction + holdout_fraction > 1.0 + 1e-9:
        raise CorpusError("build_plan: calibration + holdout fractions exceed 1.0")
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
        turn_clips=_records_from_spans(
            source, turn_spans, kind="turn", split_of_speaker=split_of_speaker
        ),
        segments=_records_from_spans(
            source, segment_spans, kind="segment", split_of_speaker=split_of_speaker
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
    clips: list[dict[str, Any]] = []
    for record in records:
        fact = measured.get(record.clip_id)
        if fact is None:
            raise CorpusError(f"finalize_manifest: no measured facts for clip {record.clip_id!r}")
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
_FFMPEG_PREFIX = ("ffmpeg", "-nostdin", "-y", "-threads", "1")


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
    ``-f s16le -ar 16000 -ac 1`` and every output re-canonicalizes to
    ``pcm_s16le``; ``-threads 1`` keeps encoding deterministic.
    """
    if recipe.lossy:
        if not intermediate_path:
            raise CorpusError(
                f"recipe {recipe.recipe_id!r} is lossy and needs an intermediate_path"
            )
        encode = (
            *_FFMPEG_PREFIX,
            *_RAW_INPUT_FRAMING,
            "-i", in_path,
            *recipe.encode_args,
            "-f", recipe.intermediate_format,
            intermediate_path,
        )
        decode = (
            *_FFMPEG_PREFIX,
            "-i", intermediate_path,
            *_CANONICAL_OUTPUT,
            out_path,
        )
        return (encode, decode)
    single = (
        *_FFMPEG_PREFIX,
        *_RAW_INPUT_FRAMING,
        "-i", in_path,
        *recipe.encode_args,
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
    """Plan an organic source from its RTTMs (dry-run: emit a plan, extract nothing).

    Reads one or more RTTM files, groups their rows by the recording id each row
    declares, builds the materialization plan, and writes it as JSON. It touches no
    audio; the executor (a later slice) materializes what the plan describes.
    """
    source = ORGANIC_SOURCES.get(args.source)
    if source is None:
        sys.stderr.write(f"unknown source {args.source!r}; known: {sorted(ORGANIC_SOURCES)}\n")
        return 2
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


def cmd_degrade(args: argparse.Namespace) -> int:
    """Plan degraded children of a parent manifest for a recipe chain (dry-run).

    Loads a parent manifest, optionally restricts to one split, derives a degraded
    child per parent clip for the given recipe chain, and writes the child plan as
    JSON. It builds no audio; the argv the executor will run is unit-covered.
    """
    try:
        obj = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        manifest = load_manifest(obj)
        chain = serialize_chain(args.recipe)
        children = [
            asdict(derive_degraded_record(clip, args.recipe))
            for clip in manifest.clips
            if args.split is None or clip.split == args.split
        ]
    except (OSError, json.JSONDecodeError, CorpusError) as exc:
        sys.stderr.write(f"degrade failed: {exc}\n")
        return 2
    if not children:
        sys.stderr.write(f"degrade: no parent clips in split {args.split!r}\n")
        return 2
    sys.stdout.write(
        json.dumps({"chain": chain, "children": children}, indent=2, sort_keys=True) + "\n"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="synthdetect corpus manifest tools (#144)")
    sub = parser.add_subparsers(dest="command", required=True)
    p_validate = sub.add_parser("validate", help="validate a manifest file (schema + integrity)")
    p_validate.add_argument("--manifest", required=True, help="path to the manifest JSON")
    p_validate.set_defaults(func=cmd_validate)

    p_prepare = sub.add_parser(
        "prepare", help="plan an organic source from its RTTMs (dry-run, no audio)"
    )
    p_prepare.add_argument(
        "--source", required=True, help=f"organic source id ({sorted(ORGANIC_SOURCES)})"
    )
    p_prepare.add_argument("--rttm", required=True, nargs="+", help="one or more RTTM files")
    p_prepare.set_defaults(func=cmd_prepare)

    p_degrade = sub.add_parser(
        "degrade", help="plan degraded children of a parent manifest (dry-run, no audio)"
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
    p_degrade.set_defaults(func=cmd_degrade)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
