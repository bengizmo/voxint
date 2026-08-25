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
