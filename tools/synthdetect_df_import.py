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

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
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

# The official label vocabulary (as written in the metadata) and the canonical
# scored partition. A sentinel single dash marks an officially-absent value.
OFFICIAL_LABELS = ("bonafide", "spoof")
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
    separated fields and a known label; anything else raises rather than being
    skipped, so a truncated or reformatted keys file can never silently shrink the
    cohort. Trial ids must be unique.
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
        trial_id = fields[_COL_TRIAL_ID]
        if trial_id in seen:
            raise DfImportError(f"trial_metadata line {lineno}: duplicate trial id {trial_id!r}")
        seen.add(trial_id)
        records.append(
            TrialRecord(
                speaker_id=fields[_COL_SPEAKER],
                trial_id=trial_id,
                codec=fields[_COL_CODEC],
                source=fields[_COL_SOURCE],
                attack_system=_absent_or(fields[_COL_ATTACK]),
                label=label,
                split=fields[_COL_SPLIT],
                vocoder_family=_absent_or(fields[_COL_VOCODER]),
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
    cohort_hash = hashlib.sha256(("\n".join(trial_ids) + "\n").encode()).hexdigest()
    return SubsetSelection(
        seed=seed,
        fraction_num=fraction_num,
        fraction_den=fraction_den,
        trial_ids=trial_ids,
        strata=tuple(strata),
        cohort_hash=cohort_hash,
    )
