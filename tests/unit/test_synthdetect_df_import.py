"""Unit tests for the ASVspoof 2021 DF importer core (#144, S3).

Covers the audio-free surface: parsing the official ``trial_metadata.txt`` and
the frozen, seeded, stratified subset selection. The selection rule is pinned
here so a change to the seed, the stratification, or the rounding is caught as a
deliberate, reviewed edit rather than a silent drift of the cohort.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import synthdetect_df_import as di  # noqa: E402
from synthdetect_sources import SELECTION_SEED  # noqa: E402


def _row(
    *,
    speaker: str = "LA_0001",
    trial_id: str,
    codec: str = "nocodec",
    source: str = "asvspoof",
    attack: str = "A07",
    label: str = "spoof",
    split: str = "eval",
    vocoder: str = "traditional_vocoder",
) -> str:
    """Build one 13-column official trial row (trailing task/team/gender = dash)."""
    tail = ["notrim", split, vocoder, "-", "-", "-", "-"]
    return " ".join([speaker, trial_id, codec, source, attack, label, *tail])


# --------------------------------------------------------------------------- #
# parse_trial_metadata
# --------------------------------------------------------------------------- #
def test_parse_happy_path_maps_pinned_columns() -> None:
    text = "\n".join(
        [
            _row(trial_id="DF_E_1", label="spoof", codec="low_mp3", attack="A09"),
            _row(trial_id="DF_E_2", label="bonafide", attack="-", vocoder="-", source="vcc2020"),
        ]
    )
    recs = di.parse_trial_metadata(text)
    assert len(recs) == 2
    a, b = recs
    assert (a.trial_id, a.label, a.codec, a.split) == ("DF_E_1", "spoof", "low_mp3", "eval")
    assert a.attack_system == "A09"
    # An officially-absent value becomes None, never a sentinel string.
    assert b.attack_system is None
    assert b.vocoder_family is None
    assert b.source == "vcc2020"


def test_parse_preserves_raw_official_line() -> None:
    line = _row(trial_id="DF_E_1")
    (rec,) = di.parse_trial_metadata(line)
    assert rec.raw == line


def test_parse_skips_blank_lines() -> None:
    text = f"\n{_row(trial_id='DF_E_1')}\n\n{_row(trial_id='DF_E_2')}\n"
    assert len(di.parse_trial_metadata(text)) == 2


def test_parse_rejects_wrong_column_count() -> None:
    with pytest.raises(di.DfImportError, match="expected 13 fields, got 5"):
        di.parse_trial_metadata("LA_0001 DF_E_1 nocodec asvspoof A07")


def test_parse_rejects_unknown_label() -> None:
    with pytest.raises(di.DfImportError, match="label must be one of"):
        di.parse_trial_metadata(_row(trial_id="DF_E_1", label="maybe"))


def test_parse_rejects_duplicate_trial_id() -> None:
    text = "\n".join([_row(trial_id="DF_E_1"), _row(trial_id="DF_E_1")])
    with pytest.raises(di.DfImportError, match="duplicate trial id"):
        di.parse_trial_metadata(text)


def test_parse_rejects_empty() -> None:
    with pytest.raises(di.DfImportError, match="empty"):
        di.parse_trial_metadata("\n  \n")


# --------------------------------------------------------------------------- #
# stratum_key
# --------------------------------------------------------------------------- #
def test_stratum_key_is_label_by_codec() -> None:
    (rec,) = di.parse_trial_metadata(_row(trial_id="DF_E_1", label="spoof", codec="high_ogg"))
    assert di.stratum_key(rec) == "spoof|high_ogg"


# --------------------------------------------------------------------------- #
# select_subset
# --------------------------------------------------------------------------- #
def _corpus(n_per_stratum: int) -> tuple:
    rows = []
    for label in ("bonafide", "spoof"):
        for codec in ("nocodec", "low_mp3"):
            for i in range(n_per_stratum):
                rows.append(
                    _row(
                        trial_id=f"DF_E_{label}_{codec}_{i:04d}",
                        label=label,
                        codec=codec,
                        attack="A07" if label == "spoof" else "-",
                        vocoder="traditional_vocoder" if label == "spoof" else "-",
                    )
                )
    return di.parse_trial_metadata("\n".join(rows))


def test_select_subset_keeps_ten_percent_per_stratum() -> None:
    recs = _corpus(100)  # 4 strata x 100
    sel = di.select_subset(recs)
    assert sel.n_selected == 40  # round(100/10) x 4
    assert {s.stratum for s in sel.strata} == {
        "bonafide|nocodec",
        "bonafide|low_mp3",
        "spoof|nocodec",
        "spoof|low_mp3",
    }
    for s in sel.strata:
        assert s.n_total == 100
        assert s.n_selected == 10


def test_select_subset_is_eval_only() -> None:
    eval_rows = [_row(trial_id=f"DF_E_{i}", split="eval") for i in range(100)]
    other = [_row(trial_id=f"DF_P_{i}", split="progress") for i in range(100)]
    other += [_row(trial_id=f"DF_H_{i}", split="hidden") for i in range(100)]
    recs = di.parse_trial_metadata("\n".join(eval_rows + other))
    sel = di.select_subset(recs)
    # Only the single eval stratum contributes; progress/hidden are excluded.
    assert sel.n_selected == 10
    assert all(tid.startswith("DF_E_") for tid in sel.trial_ids)


def test_select_subset_rejects_no_eval_records() -> None:
    recs = di.parse_trial_metadata(_row(trial_id="DF_P_1", split="progress"))
    with pytest.raises(di.DfImportError, match="no records in the scored split"):
        di.select_subset(recs)


def test_select_subset_is_deterministic_and_canonically_ordered() -> None:
    recs = _corpus(50)
    a = di.select_subset(recs)
    b = di.select_subset(recs)
    assert a.trial_ids == b.trial_ids
    assert a.cohort_hash == b.cohort_hash
    # Canonical order is trial id ascending, and the hash binds that order.
    assert list(a.trial_ids) == sorted(a.trial_ids)
    expected = hashlib.sha256(("\n".join(a.trial_ids) + "\n").encode()).hexdigest()
    assert a.cohort_hash == expected


def test_select_subset_changes_with_seed() -> None:
    recs = _corpus(200)
    a = di.select_subset(recs, seed=SELECTION_SEED)
    b = di.select_subset(recs, seed="a-different-seed")
    # Same count, different members: the draw is seed-controlled.
    assert a.n_selected == b.n_selected
    assert set(a.trial_ids) != set(b.trial_ids)


def test_select_subset_selection_is_a_subset_of_ranked_prefix() -> None:
    # The chosen trials in a stratum are exactly the lowest-ranked round(n/10).
    recs = _corpus(30)
    sel = di.select_subset(recs)
    stratum = [r for r in recs if di.stratum_key(r) == "spoof|nocodec"]
    ranked = sorted(stratum, key=lambda r: (di._rank_key(SELECTION_SEED, r.trial_id), r.trial_id))
    k = di._round_half_up(len(stratum), 1, 10)
    expected = {r.trial_id for r in ranked[:k]}
    got = {tid for tid in sel.trial_ids if tid.startswith("DF_E_spoof_nocodec_")}
    assert got == expected


@pytest.mark.parametrize(
    "n,expected",
    [(100, 10), (104, 10), (105, 11), (95, 10), (9, 1), (4, 0), (5, 1)],
)
def test_round_half_up_matches_expected(n: int, expected: int) -> None:
    assert di._round_half_up(n, 1, 10) == expected


@pytest.mark.parametrize("num,den", [(0, 10), (11, 10), (1, 0), (-1, 10)])
def test_select_subset_rejects_bad_fraction(num: int, den: int) -> None:
    recs = _corpus(10)
    with pytest.raises(di.DfImportError, match="fraction"):
        di.select_subset(recs, fraction_num=num, fraction_den=den)
