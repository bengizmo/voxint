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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from synthdetect_sources import SELECTION_SEED  # noqa: E402

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
        if degradation is not None and (
            not isinstance(degradation, str) or not degradation.strip()
        ):
            raise CorpusError(f"{where}: degradation must be null or a non-empty string")
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

    free_speakers = sorted(
        speaker_ids - forced_eval_speakers,
        key=lambda sp: (_rank_key(seed, sp), sp),
    )
    n = len(free_speakers)
    n_cal = int(n * calibration_fraction)
    n_hold = int(n * holdout_fraction)
    # Clamp so calibration + holdout never overrun the ranked list (rounding).
    n_hold = min(n_hold, n - n_cal)

    split_of_speaker: dict[str, str] = dict.fromkeys(forced_eval_speakers, "eval")
    for rank, sp in enumerate(free_speakers):
        if rank < n_cal:
            split_of_speaker[sp] = "calibration"
        elif rank >= n - n_hold:
            split_of_speaker[sp] = "holdout"
        else:
            split_of_speaker[sp] = "eval"

    return {c.clip_id: split_of_speaker[c.speaker_id] for c in clips}


def split_summary(assignment: dict[str, str]) -> dict[str, int]:
    """Count clips per split (a report/CLI convenience over an assignment)."""
    counts = dict.fromkeys(SPLITS, 0)
    for split in assignment.values():
        counts[split] += 1
    return counts


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="synthdetect corpus manifest tools (#144)")
    sub = parser.add_subparsers(dest="command", required=True)
    p_validate = sub.add_parser("validate", help="validate a manifest file (schema + integrity)")
    p_validate.add_argument("--manifest", required=True, help="path to the manifest JSON")
    p_validate.set_defaults(func=cmd_validate)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
