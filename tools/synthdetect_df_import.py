#!/usr/bin/env python3
"""Narrow importer for the official ASVspoof 2021 DF eval benchmark (#144, S3).

This is a deliberately benchmark-specific, one-off tool, NOT a general corpus
framework. Its whole surface is: verify operator-supplied official archives and
keys by pinned sha, preserve an untouched native tree for the Gate-1 upstream
runner, emit the seeded subset (trial list + receipt + cohort hash), and emit a
canonical ``pcm-s16le-mono-16000-v1`` view for the Gate-2 paired compare. No TTS
synthesis, no degradation, no acquisition machinery lands here.

THIS module holds the pure, audio-free core: parsing the official
``trial_metadata.txt`` and the frozen, seeded, stratified subset selection over
official trial IDs. Everything here is deterministic and unit-covered against a
fixture before any audio is touched, exactly as the S3 pre-registration
(docs/gpu-contracts.md) requires the selection to be reproducible and auditable.

The selection rule (frozen):

* The subset is drawn from the official ``eval`` split ONLY. The ``progress`` and
  ``hidden`` partitions are not part of the scored DF cohort and are excluded.
* Trials are stratified by ``label x codec condition`` (both always present in the
  official metadata). The speaker-disjoint split assigner in ``synthdetect_corpus``
  is a training-split tool and is deliberately NOT used on an official eval set.
* Within each stratum, trials are ranked by a seeded hash of the official trial id
  (seed-domain-separated, the same construction ``synthdetect_corpus`` uses for its
  split ranks) and the lowest-ranked ``round(n / 10)`` are selected. Rank ties break
  on the trial id. The fraction is exact integer round-half-up, so the rule carries
  no float non-determinism.
* The cohort is emitted in one canonical order (trial id ascending) that the trial
  list, the cohort hash, and the canonical manifest all share, so the selection
  identity is order-stable across every downstream artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from synthdetect_corpus import (  # noqa: E402
    CORPUS_KIND_IMPORTED,
    IMPORTED_MANIFEST_SCHEMA_VERSION,
    load_manifest,
)
from synthdetect_infer import (  # noqa: E402
    CANONICAL_SAMPLE_RATE,
    CANONICALIZATION_ID,
    CanonicalAudio,
    read_canonical_pcm,
)
from synthdetect_sources import SELECTION_SEED  # noqa: E402

# The official DF trial_metadata.txt is whitespace-separated with a fixed column
# count. We pin the columns we consume by index; the trailing task/team/gender
# columns are preserved verbatim in the receipt rather than interpreted here.
TRIAL_METADATA_COLUMNS = 13
_COL_SPEAKER = 0
_COL_TRIAL_ID = 1
_COL_CODEC = 2
_COL_SOURCE = 3
_COL_ATTACK = 4
_COL_LABEL = 5
_COL_SPLIT = 7
_COL_VOCODER = 8

# The official label and split vocabularies (as written in the metadata) and the
# canonical scored partition. In the official DF metadata the attack-system column
# is the single dash for every bona fide trial and a real id for every spoof trial,
# while the vocoder-family column is ALWAYS populated (the literal ``bonafide`` for
# genuine speech, and for spoof a family that legitimately includes the token
# ``unknown``). So the honest, data-faithful coupling is "attack id present iff
# spoof", and the vocoder column is never treated as a placeholder.
OFFICIAL_LABELS = ("bonafide", "spoof")
OFFICIAL_SPLITS = ("eval", "progress", "hidden")
SCORED_SPLIT = "eval"
ABSENT = "-"

# The seeded 10% subset, as integer round-half-up (no floats): k = round(n/10).
SUBSET_FRACTION_NUM = 1
SUBSET_FRACTION_DEN = 10


class DfImportError(Exception):
    """A DF-import integrity problem (fail closed; never repair or substitute)."""


@dataclass(frozen=True)
class TrialRecord:
    """One official DF trial row, only the fields the importer consumes.

    ``attack_system`` and ``vocoder_family`` are ``None`` when the official row
    writes the absent-value dash; they are never coerced to a placeholder string.
    ``raw`` preserves the exact official line so the receipt can carry the source
    representation losslessly.
    """

    speaker_id: str
    trial_id: str
    codec: str
    source: str
    attack_system: str | None
    label: str
    split: str
    vocoder_family: str | None
    raw: str


def _absent_or(value: str) -> str | None:
    """Map the official absent-value dash to ``None``, else pass the token through."""
    return None if value == ABSENT else value


def parse_trial_metadata(text: str) -> tuple[TrialRecord, ...]:
    """Parse the official ``trial_metadata.txt`` into validated records (fail closed).

    Every non-empty line must have exactly ``TRIAL_METADATA_COLUMNS`` whitespace
    separated fields, a known label, and a known split; the label must agree with
    the attack-system column (a spoof row carries an attack id, a bona fide row
    carries the absent dash). Anything else raises rather than being skipped, so a
    truncated or reformatted keys file can never silently shrink the cohort, and an
    unknown split can never be dropped without a word. Trial ids must be unique.

    Field integrity beyond structure is the job of the pinned keys-archive sha, not
    a per-token placeholder scan: the official metadata legitimately uses tokens
    that look like placeholders (``unknown`` is a real vocoder family; ``bonafide``
    is the vocoder-column value for genuine speech), so a sentinel scan here would
    reject real official rows.
    """
    records: list[TrialRecord] = []
    seen: set[str] = set()
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != TRIAL_METADATA_COLUMNS:
            raise DfImportError(
                f"trial_metadata line {lineno}: expected {TRIAL_METADATA_COLUMNS} "
                f"fields, got {len(fields)}"
            )
        label = fields[_COL_LABEL]
        if label not in OFFICIAL_LABELS:
            raise DfImportError(
                f"trial_metadata line {lineno}: label must be one of "
                f"{OFFICIAL_LABELS}, got {label!r}"
            )
        split = fields[_COL_SPLIT]
        if split not in OFFICIAL_SPLITS:
            # An unknown split would otherwise be silently dropped by the eval
            # filter, shrinking the cohort without a word. Fail closed instead.
            raise DfImportError(
                f"trial_metadata line {lineno}: split must be one of "
                f"{OFFICIAL_SPLITS}, got {split!r}"
            )
        trial_id = fields[_COL_TRIAL_ID]
        if trial_id in seen:
            raise DfImportError(f"trial_metadata line {lineno}: duplicate trial id {trial_id!r}")
        seen.add(trial_id)

        attack_system = _absent_or(fields[_COL_ATTACK])
        vocoder_family = _absent_or(fields[_COL_VOCODER])
        # The one clean official invariant: an attack id is present for exactly the
        # spoof trials. The vocoder column is always populated and is recorded as-is.
        if label == "spoof" and attack_system is None:
            raise DfImportError(
                f"trial_metadata line {lineno}: spoof trial {trial_id!r} "
                "must carry an attack_system"
            )
        if label == "bonafide" and attack_system is not None:
            raise DfImportError(
                f"trial_metadata line {lineno}: bona fide trial {trial_id!r} "
                "must not carry an attack_system"
            )

        records.append(
            TrialRecord(
                speaker_id=fields[_COL_SPEAKER],
                trial_id=trial_id,
                codec=fields[_COL_CODEC],
                source=fields[_COL_SOURCE],
                attack_system=attack_system,
                label=label,
                split=split,
                vocoder_family=vocoder_family,
                raw=line,
            )
        )
    if not records:
        raise DfImportError("trial_metadata is empty")
    return tuple(records)


def stratum_key(record: TrialRecord) -> str:
    """The stratification key: official ``label`` and ``codec`` condition."""
    return f"{record.label}|{record.codec}"


def _rank_key(seed: str, trial_id: str) -> int:
    """A deterministic 256-bit rank for a trial id (seed-domain-separated).

    The construction matches ``synthdetect_corpus._rank_key`` so the two seeded
    selections read as one family, though this one ranks official trial ids and
    never the speaker ids the training splitter uses.
    """
    digest = hashlib.sha256(f"{seed}\x00{trial_id}".encode()).hexdigest()
    return int(digest, 16)


def _round_half_up(n: int, num: int, den: int) -> int:
    """Integer round-half-up of ``n * num / den`` (no float arithmetic)."""
    return (2 * n * num + den) // (2 * den)


@dataclass(frozen=True)
class StratumSelection:
    """The per-stratum selection outcome (for the receipt)."""

    stratum: str
    n_total: int
    n_selected: int


@dataclass(frozen=True)
class SubsetSelection:
    """A frozen, reproducible subset selection over official DF trial ids."""

    seed: str
    fraction_num: int
    fraction_den: int
    trial_ids: tuple[str, ...]  # canonical order: trial id ascending
    strata: tuple[StratumSelection, ...]
    cohort_hash: str

    @property
    def n_selected(self) -> int:
        return len(self.trial_ids)


def select_subset(
    records: tuple[TrialRecord, ...] | list[TrialRecord],
    *,
    seed: str = SELECTION_SEED,
    fraction_num: int = SUBSET_FRACTION_NUM,
    fraction_den: int = SUBSET_FRACTION_DEN,
) -> SubsetSelection:
    """Select the seeded, stratified subset from the official ``eval`` trials.

    Filters to the scored ``eval`` split, groups by ``label x codec``, and within
    each stratum keeps the lowest seeded-hash-ranked ``round(n * num / den)`` trial
    ids. The result is emitted in one canonical order (trial id ascending) and the
    cohort hash is bound to that order, so the trial list, the manifest, and both
    runners agree on the exact cohort and its identity.
    """
    if fraction_den <= 0 or fraction_num <= 0 or fraction_num > fraction_den:
        raise DfImportError(
            f"selection fraction {fraction_num}/{fraction_den} must be in (0, 1]"
        )
    eval_records = [r for r in records if r.split == SCORED_SPLIT]
    if not eval_records:
        raise DfImportError(f"no records in the scored split {SCORED_SPLIT!r}")

    by_stratum: dict[str, list[TrialRecord]] = {}
    for r in eval_records:
        by_stratum.setdefault(stratum_key(r), []).append(r)

    selected: list[str] = []
    strata: list[StratumSelection] = []
    for stratum in sorted(by_stratum):
        members = by_stratum[stratum]
        # Rank by seeded hash, ties broken on the trial id for total determinism.
        ranked = sorted(members, key=lambda r: (_rank_key(seed, r.trial_id), r.trial_id))
        k = _round_half_up(len(members), fraction_num, fraction_den)
        chosen = ranked[:k]
        selected.extend(r.trial_id for r in chosen)
        strata.append(
            StratumSelection(stratum=stratum, n_total=len(members), n_selected=len(chosen))
        )

    trial_ids = tuple(sorted(selected))
    if not trial_ids:
        # Every stratum rounded to zero (all strata smaller than the reciprocal of
        # the fraction). A vacuous cohort is a mistake, not a valid selection.
        raise DfImportError(
            "selection produced an empty cohort; the eval strata are too small "
            f"for fraction {fraction_num}/{fraction_den}"
        )
    cohort_hash = hashlib.sha256(("\n".join(trial_ids) + "\n").encode()).hexdigest()
    return SubsetSelection(
        seed=seed,
        fraction_num=fraction_num,
        fraction_den=fraction_den,
        trial_ids=trial_ids,
        strata=tuple(strata),
        cohort_hash=cohort_hash,
    )


# --------------------------------------------------------------------------- #
# Emission: verified archives -> native tree + canonical v2 manifest + receipt
#
# The audio-dependent half of the importer (S3, docs/gpu-contracts.md, "Two
# corpus views, never conflated"). It verifies the operator's official archives
# by pinned sha, materialises an untouched native FLAC tree from those verified
# bytes (never a tree it is merely handed), transcodes the seeded subset to the
# canonical ``pcm-s16le-mono-16000-v1`` view, and emits a v2 imported-benchmark
# manifest whose per-clip sha is the PCM payload only. Every step fails closed;
# nothing is published until the whole run succeeds.
#
# The load-bearing invariant is a one-to-one chain from a pinned archive byte to
# a scored canonical PCM: pinned archive -> exact native FLAC member (never a
# basename search) -> its sha -> the canonical WAV decoded from THAT file -> the
# manifest clip. A schema-valid manifest that pointed a trial id at another
# trial's audio would let a paired Gate-2 comparison agree perfectly on the
# wrong audio, so the emitter binds identity cryptographically and re-audits the
# published bytes before returning.
# --------------------------------------------------------------------------- #

# The imported benchmark's identity and the honest provenance the schema records
# for every clip. The audio is ASVspoof 2021 DF eval (16 kHz / 16-bit mono FLAC,
# per its README): a multi-source deepfake eval whose per-clip language is not
# published (``und``), released under the Open Database License (the contents
# carry the DbCL-1.0, incorporated by reference into ODbL-1.0).
DF_BENCHMARK = "asvspoof2021_df"
DF_LICENSE_SPDX = "ODbL-1.0"
DF_LANGUAGE = "und"

# The official archive layout: one top directory holding ``flac/<trial_id>.flac``
# plus the upstream trial list, README, and license.
NATIVE_TREE_ROOT = "ASVspoof2021_DF_eval"
NATIVE_FLAC_SUBDIR = "flac"
FLAC_SUFFIX = ".flac"
# Where the keys archive stores the official metadata the scorer reads.
KEYS_METADATA_MEMBER = "keys/DF/CM/trial_metadata.txt"

# The canonical view lives under this subdir of the emitted corpus root; the
# manifest ``rel_path`` is ``canonical/<trial_id>.wav``.
CANONICAL_SUBDIR = "canonical"
CANONICAL_SUFFIX = ".wav"

# The official schema label vocabulary differs from the metadata's: the corpus
# schema writes ``bona_fide`` where the official metadata writes ``bonafide``.
# The mapping is explicit so a silent vocabulary drift cannot slip a clip past
# the v2 ``stratum == f"{label}|{codec}"`` binding.
_OFFICIAL_TO_SCHEMA_LABEL = {"bonafide": "bona_fide", "spoof": "spoof"}

# Pinned sha256 digests of the public official archives (reproducibility pins,
# not secrets). Keys: the asvspoof.org DF keys+metadata archive. Audio: the
# Zenodo record 4835108 eval tarballs. A digest is listed only once verified;
# the emitter refuses any archive whose basename is absent here, so an
# unpinned byte-stream can never enter the native tree. The remaining audio
# parts are pinned as each is cross-checked at the S3 acceptance run.
OFFICIAL_ARCHIVE_SHA256: dict[str, str] = {
    "DF-keys-full.tar.gz": (
        "426f93e1cdaf507bf36c355c00c0567137ba1fe8bc177b17281eeae6f4d870a6"
    ),
    "ASVspoof2021_DF_eval_part00.tar.gz": (
        "99273ef077604afa8f79f25070755bf99f3524f9fc55397e59ab0c00661165ea"
    ),
}

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"


@dataclass(frozen=True)
class ClipReceipt:
    """The per-trial identity chain, one row per selected clip (canonical order).

    Binds the pinned-archive-derived native FLAC to the exact canonical PCM the
    manifest scores, so the archive -> native -> canonical -> manifest path is
    auditable byte for byte after the fact.
    """

    official_trial_id: str
    native_rel_path: str
    native_flac_sha256: str
    canonical_rel_path: str
    canonical_pcm_sha256: str
    n_samples: int


@dataclass(frozen=True)
class EmitResult:
    """Summary of a completed emission (the receipt is written to disk too)."""

    out_dir: Path
    native_root: Path
    n_selected: int
    cohort_hash: str
    manifest_sha256: str


def _sha256_file(path: Path) -> str:
    """Stream a file's sha256 without loading it whole."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_archive_sha(path: Path, expected: Mapping[str, str]) -> str:
    """Verify one archive's sha256 against its pinned digest (fail closed).

    The archive is matched by basename; an archive whose basename is not pinned,
    or whose bytes do not match the pin, is rejected rather than trusted.
    """
    name = path.name
    pinned = expected.get(name)
    if pinned is None:
        raise DfImportError(
            f"{path}: no pinned sha256 for archive {name!r}; refusing an unpinned archive"
        )
    actual = _sha256_file(path)
    if actual != pinned:
        raise DfImportError(
            f"{path}: sha256 {actual} does not match the pinned {pinned}"
        )
    return actual


def _read_keys_metadata(keys_archive: Path) -> bytes:
    """Read the official ``trial_metadata.txt`` bytes from the keys archive.

    Reads the single named member's content in memory (no extraction to disk),
    failing closed if the member is missing or is not a regular file.
    """
    try:
        with tarfile.open(keys_archive, "r:gz") as tf:
            try:
                member = tf.getmember(KEYS_METADATA_MEMBER)
            except KeyError as exc:
                raise DfImportError(
                    f"{keys_archive}: missing member {KEYS_METADATA_MEMBER!r}"
                ) from exc
            if not member.isfile():
                raise DfImportError(
                    f"{keys_archive}: {KEYS_METADATA_MEMBER!r} is not a regular file"
                )
            handle = tf.extractfile(member)
            if handle is None:
                raise DfImportError(
                    f"{keys_archive}: cannot read {KEYS_METADATA_MEMBER!r}"
                )
            with handle:
                return handle.read()
    except tarfile.TarError as exc:
        raise DfImportError(f"{keys_archive}: not a readable tar.gz: {exc}") from exc


@dataclass(frozen=True)
class _Selection:
    records_by_id: dict[str, TrialRecord]
    selection: SubsetSelection
    keys_metadata_sha256: str
    keys_archive_sha256: str


def _prepare_selection(
    keys_archive: Path, *, seed: str, expected: Mapping[str, str]
) -> _Selection:
    """Verify the keys archive, parse the official metadata, and select the subset."""
    keys_sha = verify_archive_sha(keys_archive, expected)
    data = _read_keys_metadata(keys_archive)
    records = parse_trial_metadata(data.decode("utf-8"))
    selection = select_subset(records, seed=seed)
    return _Selection(
        records_by_id={r.trial_id: r for r in records},
        selection=selection,
        keys_metadata_sha256=hashlib.sha256(data).hexdigest(),
        keys_archive_sha256=keys_sha,
    )


def _reject_unsafe_member(member: tarfile.TarInfo, archive: Path) -> None:
    """Reject any archive member that is not a plain regular file or directory."""
    name = member.name
    if name.startswith("/") or os.path.isabs(name):
        raise DfImportError(f"{archive}: member {name!r} is an absolute path")
    if ".." in Path(name).parts:
        raise DfImportError(f"{archive}: member {name!r} traverses out of the tree")
    if member.issym() or member.islnk():
        raise DfImportError(f"{archive}: member {name!r} is a link")
    if member.ischr() or member.isblk() or member.isfifo() or member.isdev():
        raise DfImportError(f"{archive}: member {name!r} is a special device")
    if not (member.isfile() or member.isdir()):
        raise DfImportError(f"{archive}: member {name!r} is not a file or directory")
    top = Path(name).parts[0] if Path(name).parts else ""
    if top != NATIVE_TREE_ROOT:
        raise DfImportError(
            f"{archive}: member {name!r} is outside {NATIVE_TREE_ROOT!r}"
        )


def _safe_extract(archives: Sequence[Path], dest_root: Path) -> None:
    """Extract the archives into a fresh ``dest_root``, merging the split parts.

    Rejects absolute/traversing paths, links, and device members, restricts
    every member to the official ``NATIVE_TREE_ROOT``, and refuses a duplicate
    file member across parts (a repeated FLAC is corruption and the exact
    failure the identity chain must not tolerate). Directory entries legitimately
    repeat across parts and are exempt from the duplicate check.
    """
    seen_files: set[str] = set()
    for archive in archives:
        try:
            with tarfile.open(archive, "r:gz") as tf:
                for member in tf:
                    _reject_unsafe_member(member, archive)
                    if member.isfile():
                        if member.name in seen_files:
                            raise DfImportError(
                                f"{archive}: duplicate member {member.name!r} across archives"
                            )
                        seen_files.add(member.name)
                    tf.extract(member, path=dest_root, filter="data")
        except tarfile.TarError as exc:
            raise DfImportError(f"{archive}: extraction failed: {exc}") from exc


def _tool_version(tool: str) -> str:
    """Return the first line of ``tool -version`` (fail closed if absent)."""
    if shutil.which(tool) is None:
        raise DfImportError(f"{tool} is not on PATH; it is required to emit the canonical view")
    proc = subprocess.run(
        [tool, "-version"], capture_output=True, text=True, check=False
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        raise DfImportError(f"{tool} -version failed: {proc.stderr.strip()}")
    return proc.stdout.splitlines()[0].strip()


def _ffprobe_source(path: Path) -> None:
    """Assert the source FLAC is exactly one 16 kHz / mono / 16-bit stream.

    The decoded ``sample_fmt`` is the reliable depth gate (a 24-bit FLAC decodes
    to ``s32``); ``bits_per_raw_sample`` cross-checks it when ffprobe supplies
    it, and its absence never weakens the check. Validating only the transcoded
    output would miss a 24->16 truncation, so the source is probed directly.
    """
    proc = subprocess.run(
        [
            FFPROBE, "-v", "error", "-select_streams", "a",
            "-show_entries",
            "stream=codec_name,sample_rate,channels,bits_per_raw_sample,sample_fmt",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise DfImportError(f"{path}: ffprobe failed: {proc.stderr.strip()}")
    try:
        streams = json.loads(proc.stdout).get("streams", [])
    except json.JSONDecodeError as exc:
        raise DfImportError(f"{path}: unparseable ffprobe output: {exc}") from exc
    if len(streams) != 1:
        raise DfImportError(f"{path}: expected exactly 1 audio stream, got {len(streams)}")
    s = streams[0]
    if s.get("codec_name") != "flac":
        raise DfImportError(f"{path}: source must be flac, got {s.get('codec_name')!r}")
    if str(s.get("sample_rate")) != str(CANONICAL_SAMPLE_RATE):
        raise DfImportError(
            f"{path}: source must be {CANONICAL_SAMPLE_RATE} Hz, got {s.get('sample_rate')!r}"
        )
    if s.get("channels") != 1:
        raise DfImportError(f"{path}: source must be mono, got {s.get('channels')!r} channels")
    if s.get("sample_fmt") != "s16":
        raise DfImportError(
            f"{path}: source must decode to s16, got sample_fmt {s.get('sample_fmt')!r}"
        )
    braw = s.get("bits_per_raw_sample")
    if braw is not None and str(braw) != "16":
        raise DfImportError(f"{path}: source bits_per_raw_sample must be 16, got {braw!r}")


def _transcode_to_canonical(src: Path, dst: Path) -> None:
    """Transcode a source FLAC to a canonical PCM WAV (no resample, no filters).

    ``-map 0:a:0`` selects the one audio stream, ``-c:a pcm_s16le`` with no
    ``-ar``/``-ac`` never resamples or down-mixes (a non-conforming source is
    caught by :func:`_ffprobe_source` before this and by ``read_canonical_pcm``
    after), and ``+bitexact`` / ``-map_metadata -1`` keep the output free of
    encoder and timestamp noise.
    """
    proc = subprocess.run(
        [
            FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error",
            "-fflags", "+bitexact", "-flags", "+bitexact",
            "-i", str(src), "-map", "0:a:0", "-vn", "-sn", "-dn",
            "-map_metadata", "-1", "-c:a", "pcm_s16le", "-f", "wav", str(dst),
        ],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise DfImportError(f"{src}: ffmpeg transcode failed: {proc.stderr.strip()}")


def build_clip(record: TrialRecord, audio: CanonicalAudio, *, rel_path: str) -> dict[str, object]:
    """Build one v2 imported-benchmark clip dict from a selected trial + its audio.

    ``audio`` is a ``synthdetect_infer.CanonicalAudio``; the clip's sha is its
    PCM payload digest and ``duration_s`` is derived from the decoded frame
    count. The provenance is rebuilt from THIS record (never reparsed from a
    second source), the official ``bonafide`` label is mapped to the schema's
    ``bona_fide``, and the stratum is bound to the schema label so the manifest
    satisfies the v2 ``stratum == f"{label}|{codec}"`` clip binding.
    """
    label = _OFFICIAL_TO_SCHEMA_LABEL.get(record.label)
    if label is None:  # the parser guarantees an official label; defensive
        raise DfImportError(f"{record.trial_id}: unmapped official label {record.label!r}")
    if record.vocoder_family is None:  # official vocoder column is always populated
        raise DfImportError(f"{record.trial_id}: missing vocoder_family")
    n_samples = int(audio.n_samples)  # plain int; no numpy scalar into the manifest
    if n_samples <= 0:
        raise DfImportError(f"{record.trial_id}: canonical audio has no frames")
    return {
        "clip_id": record.trial_id,
        "rel_path": rel_path,
        "sha256": audio.pcm_sha256,
        "duration_s": n_samples / CANONICAL_SAMPLE_RATE,
        "label": label,
        "language": DF_LANGUAGE,
        "license_spdx": DF_LICENSE_SPDX,
        "stratum": f"{label}|{record.codec}",
        "source": DF_BENCHMARK,
        "speaker_id": record.speaker_id,
        "split": "eval",
        "imported_provenance": {
            "official_trial_id": record.trial_id,
            "source_dataset": DF_BENCHMARK,
            "codec_condition": record.codec,
            "official_split": "eval",
            "vocoder_family": record.vocoder_family,
            "attack_system": record.attack_system,
        },
    }


def serialize_manifest(clips: Sequence[dict[str, object]], *, benchmark: str) -> bytes:
    """Serialise a v2 imported-benchmark manifest to deterministic file bytes.

    The runner hashes these exact on-disk bytes as ``manifest_sha256`` and both
    Gate-2 journals must stamp the same value, so the serialisation is fixed:
    ``sort_keys`` orders object keys, ``allow_nan=False`` forbids non-finite
    floats, and the clips array is written in the order given (the caller emits
    it in canonical trial-id order). Written with ``write_bytes`` so no newline
    translation can perturb the digest.
    """
    obj = {
        "schema_version": IMPORTED_MANIFEST_SCHEMA_VERSION,
        "corpus_kind": CORPUS_KIND_IMPORTED,
        "benchmark": benchmark,
        "clips": list(clips),
    }
    return (json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def serialize_trial_list(
    records_by_id: Mapping[str, TrialRecord], trial_ids: Sequence[str]
) -> bytes:
    """Serialise the selected official metadata rows, LF-terminated, in ``trial_ids`` order.

    Emits each trial's exact official line (``TrialRecord.raw``) so the trial
    list is directly consumable by the unmodified upstream Gate-1 runner.
    """
    return ("\n".join(records_by_id[t].raw for t in trial_ids) + "\n").encode("utf-8")


def _staging_for(target: Path) -> Path:
    return target.parent / (target.name + ".staging")


def _require_absent_or_empty(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise DfImportError(f"{path}: refusing to publish into a non-empty destination")


def _serialize_clip_receipts(receipts: Sequence[ClipReceipt]) -> bytes:
    """Serialise the per-trial identity chain as canonical-order JSONL."""
    lines = [
        json.dumps(
            {
                "official_trial_id": r.official_trial_id,
                "native_rel_path": r.native_rel_path,
                "native_flac_sha256": r.native_flac_sha256,
                "canonical_rel_path": r.canonical_rel_path,
                "canonical_pcm_sha256": r.canonical_pcm_sha256,
                "n_samples": r.n_samples,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        for r in receipts
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _build_selection_receipt(
    sel: _Selection, *, benchmark: str, trial_list_sha: str, trial_ids_sha: str
) -> dict[str, object]:
    """The reproducible selection receipt (no timestamps, no host paths)."""
    selection = sel.selection
    return {
        "benchmark": benchmark,
        "seed": selection.seed,
        "fraction": [selection.fraction_num, selection.fraction_den],
        "scored_split": SCORED_SPLIT,
        "n_total_eval": sum(s.n_total for s in selection.strata),
        "n_selected": selection.n_selected,
        "cohort_hash": selection.cohort_hash,
        "strata": [
            {"stratum": s.stratum, "n_total": s.n_total, "n_selected": s.n_selected}
            for s in selection.strata
        ],
        "keys_archive_sha256": sel.keys_archive_sha256,
        "keys_metadata_sha256": sel.keys_metadata_sha256,
        "trial_list_sha256": trial_list_sha,
        "trial_ids_sha256": trial_ids_sha,
    }


def _audit_corpus(root: Path, selection: SubsetSelection, manifest_bytes: bytes) -> None:
    """Re-read the emitted corpus and prove it before it is published (fail closed).

    Reparses the manifest bytes through ``load_manifest``, confirms the clip set
    and order equal the cohort, that the canonical WAV set equals the cohort, and
    that every published WAV re-reads to the manifest's PCM sha and frame-derived
    duration. This closes the loop from emitted bytes back to on-disk audio.
    """
    published = (root / "manifest.json").read_bytes()
    if published != manifest_bytes:
        raise DfImportError("staged manifest bytes differ from the emitted bytes")
    manifest = load_manifest(json.loads(published.decode("utf-8")))
    clip_ids = [c.clip_id for c in manifest.clips]
    if clip_ids != list(selection.trial_ids):
        raise DfImportError("emitted manifest clip order does not match the selection")
    wav_stems = {p.stem for p in (root / CANONICAL_SUBDIR).glob(f"*{CANONICAL_SUFFIX}")}
    if wav_stems != set(selection.trial_ids):
        raise DfImportError("emitted canonical WAV set does not equal the cohort")
    for clip in manifest.clips:
        audio = read_canonical_pcm(root / clip.rel_path)
        if audio.pcm_sha256 != clip.sha256:
            raise DfImportError(f"{clip.clip_id}: emitted PCM sha does not match the manifest")
        if round(clip.duration_s * CANONICAL_SAMPLE_RATE) != audio.n_samples:
            raise DfImportError(f"{clip.clip_id}: manifest duration does not match the frame count")


def emit_subset(
    *,
    keys_archive: Path,
    audio_archives: Sequence[Path],
    native_root: Path,
    out_dir: Path,
    seed: str = SELECTION_SEED,
    benchmark: str = DF_BENCHMARK,
    expected_sha256: Mapping[str, str] | None = None,
) -> EmitResult:
    """Emit the DF subset: native tree + canonical v2 manifest + receipt.

    Verifies the keys and audio archives by pinned sha, selects the seeded
    subset, materialises the native FLAC tree from the verified archives, and
    transcodes each selected trial's exact native FLAC to the canonical PCM view.
    Every artifact is staged and audited before anything is published; on any
    failure the staging is removed and neither the native tree nor the corpus
    appears, so a partial or unverified corpus can never be left behind.
    """
    if expected_sha256 is None:
        expected_sha256 = OFFICIAL_ARCHIVE_SHA256
    ffmpeg_version = _tool_version(FFMPEG)
    ffprobe_version = _tool_version(FFPROBE)
    _require_absent_or_empty(native_root)
    _require_absent_or_empty(out_dir)

    sel = _prepare_selection(keys_archive, seed=seed, expected=expected_sha256)
    selection = sel.selection
    audio_sha = {p.name: verify_archive_sha(p, expected_sha256) for p in audio_archives}

    native_staging = _staging_for(native_root)
    out_staging = _staging_for(out_dir)
    for staging in (native_staging, out_staging):
        if staging.exists():
            shutil.rmtree(staging)
    native_staging.mkdir(parents=True)
    canonical_dir = out_staging / CANONICAL_SUBDIR
    canonical_dir.mkdir(parents=True)

    try:
        _safe_extract(audio_archives, native_staging)
        flac_dir = native_staging / NATIVE_TREE_ROOT / NATIVE_FLAC_SUBDIR

        clips: list[dict[str, object]] = []
        clip_receipts: list[ClipReceipt] = []
        for trial_id in selection.trial_ids:
            record = sel.records_by_id[trial_id]
            native_flac = flac_dir / f"{trial_id}{FLAC_SUFFIX}"
            if not native_flac.is_file():
                raise DfImportError(f"{trial_id}: native flac missing at {native_flac}")
            _ffprobe_source(native_flac)
            native_sha = _sha256_file(native_flac)

            rel_path = f"{CANONICAL_SUBDIR}/{trial_id}{CANONICAL_SUFFIX}"
            tmp_wav = canonical_dir / f".{trial_id}{CANONICAL_SUFFIX}.tmp"
            final_wav = canonical_dir / f"{trial_id}{CANONICAL_SUFFIX}"
            _transcode_to_canonical(native_flac, tmp_wav)
            audio = read_canonical_pcm(tmp_wav)
            if audio.n_samples <= 0:
                raise DfImportError(f"{trial_id}: canonical audio has no frames")
            os.replace(tmp_wav, final_wav)

            clips.append(build_clip(record, audio, rel_path=rel_path))
            clip_receipts.append(
                ClipReceipt(
                    official_trial_id=trial_id,
                    native_rel_path=(
                        f"{NATIVE_TREE_ROOT}/{NATIVE_FLAC_SUBDIR}/{trial_id}{FLAC_SUFFIX}"
                    ),
                    native_flac_sha256=native_sha,
                    canonical_rel_path=rel_path,
                    canonical_pcm_sha256=audio.pcm_sha256,
                    n_samples=int(audio.n_samples),
                )
            )

        if [c["clip_id"] for c in clips] != list(selection.trial_ids):
            raise DfImportError("emitted clip order does not match the selection")

        manifest_bytes = serialize_manifest(clips, benchmark=benchmark)
        load_manifest(json.loads(manifest_bytes.decode("utf-8")))  # validate before publish
        manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        trial_list_bytes = serialize_trial_list(sel.records_by_id, selection.trial_ids)
        trial_ids_bytes = ("\n".join(selection.trial_ids) + "\n").encode("utf-8")
        clip_receipt_bytes = _serialize_clip_receipts(clip_receipts)

        (out_staging / "manifest.json").write_bytes(manifest_bytes)
        (out_staging / "trial_list.txt").write_bytes(trial_list_bytes)
        (out_staging / "trial_ids.txt").write_bytes(trial_ids_bytes)
        (out_staging / "clip_receipt.jsonl").write_bytes(clip_receipt_bytes)

        receipt = _build_selection_receipt(
            sel,
            benchmark=benchmark,
            trial_list_sha=hashlib.sha256(trial_list_bytes).hexdigest(),
            trial_ids_sha=hashlib.sha256(trial_ids_bytes).hexdigest(),
        )
        receipt.update(
            {
                "canonicalization_id": CANONICALIZATION_ID,
                "audio_archive_sha256": dict(sorted(audio_sha.items())),
                "manifest_sha256": manifest_sha,
                "clip_receipt_sha256": hashlib.sha256(clip_receipt_bytes).hexdigest(),
                "ffmpeg_version": ffmpeg_version,
                "ffprobe_version": ffprobe_version,
            }
        )
        (out_staging / "selection_receipt.json").write_bytes(
            (json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
        )

        _audit_corpus(out_staging, selection, manifest_bytes)

        os.replace(native_staging, native_root)
        os.replace(out_staging, out_dir)
    except BaseException:
        for staging in (native_staging, out_staging):
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
        raise

    return EmitResult(
        out_dir=out_dir,
        native_root=native_root,
        n_selected=selection.n_selected,
        cohort_hash=selection.cohort_hash,
        manifest_sha256=manifest_sha,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def cmd_select(args: argparse.Namespace) -> int:
    """Audio-free: verify the keys archive and emit the trial list + receipt."""
    out_dir = Path(args.out_dir)
    _require_absent_or_empty(out_dir)
    sel = _prepare_selection(
        Path(args.keys_archive), seed=args.seed, expected=OFFICIAL_ARCHIVE_SHA256
    )
    selection = sel.selection
    trial_list_bytes = serialize_trial_list(sel.records_by_id, selection.trial_ids)
    trial_ids_bytes = ("\n".join(selection.trial_ids) + "\n").encode("utf-8")
    receipt = _build_selection_receipt(
        sel,
        benchmark=args.benchmark,
        trial_list_sha=hashlib.sha256(trial_list_bytes).hexdigest(),
        trial_ids_sha=hashlib.sha256(trial_ids_bytes).hexdigest(),
    )

    out_staging = _staging_for(out_dir)
    if out_staging.exists():
        shutil.rmtree(out_staging)
    out_staging.mkdir(parents=True)
    try:
        (out_staging / "trial_list.txt").write_bytes(trial_list_bytes)
        (out_staging / "trial_ids.txt").write_bytes(trial_ids_bytes)
        (out_staging / "selection_receipt.json").write_bytes(
            (json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
        )
        os.replace(out_staging, out_dir)
    except BaseException:
        shutil.rmtree(out_staging, ignore_errors=True)
        raise

    sys.stdout.write(
        json.dumps(
            {
                "ok": True,
                "n_selected": selection.n_selected,
                "cohort_hash": selection.cohort_hash,
                "out_dir": str(out_dir),
            },
            sort_keys=True,
        )
        + "\n"
    )
    return 0


def cmd_emit(args: argparse.Namespace) -> int:
    """Full verb: verify archives and emit the native tree + canonical manifest."""
    result = emit_subset(
        keys_archive=Path(args.keys_archive),
        audio_archives=[Path(p) for p in args.audio_archive],
        native_root=Path(args.native_root),
        out_dir=Path(args.out_dir),
        seed=args.seed,
        benchmark=args.benchmark,
    )
    sys.stdout.write(
        json.dumps(
            {
                "ok": True,
                "n_selected": result.n_selected,
                "cohort_hash": result.cohort_hash,
                "manifest_sha256": result.manifest_sha256,
                "out_dir": str(result.out_dir),
                "native_root": str(result.native_root),
            },
            sort_keys=True,
        )
        + "\n"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="ASVspoof 2021 DF benchmark importer (issue #144, S3)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_sel = sub.add_parser(
        "select", help="verify keys + emit the seeded trial list and receipt (audio-free)"
    )
    p_sel.add_argument("--keys-archive", required=True)
    p_sel.add_argument("--out-dir", required=True)
    p_sel.add_argument("--seed", default=SELECTION_SEED)
    p_sel.add_argument("--benchmark", default=DF_BENCHMARK)
    p_sel.set_defaults(func=cmd_select)

    p_emit = sub.add_parser(
        "emit", help="verify archives + emit native tree, canonical manifest, and receipt"
    )
    p_emit.add_argument("--keys-archive", required=True)
    p_emit.add_argument(
        "--audio-archive", required=True, action="append",
        help="repeatable: one per official DF eval audio part",
    )
    p_emit.add_argument("--native-root", required=True)
    p_emit.add_argument("--out-dir", required=True)
    p_emit.add_argument("--seed", default=SELECTION_SEED)
    p_emit.add_argument("--benchmark", default=DF_BENCHMARK)
    p_emit.set_defaults(func=cmd_emit)

    args = parser.parse_args(argv)
    try:
        result: int = args.func(args)
    except DfImportError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    return result


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
