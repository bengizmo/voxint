"""Hand-computed cpWER goldens + the meeteval token contract (issue #97 commit 3).

cpWER (concatenated minimum-permutation WER) is scored with ``meeteval``: each
speaker stream is whisper-normalized, then meeteval assigns hypothesis speakers
to reference speakers (Hungarian) and pools integer edit counts. These goldens
are computed BY HAND so a meeteval or normalizer bump that moved a number would
fail loudly, and they pin the traps the design review flagged: the single-speaker
anchor (cpWER == plain WER), the label-swap that proves the assignment runs, the
raw ``unassigned_words`` gate, the empty/oversized guards, and the contract that
meeteval leaves the normalized token lists intact.

Needs the scoring stack (pyannote + jiwer + meeteval):
``uv run --isolated --extra parity --extra eval-quality``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("meeteval", reason="eval-quality extra not installed")

REPO = Path(__file__).resolve().parents[2]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


eq = _load("eval_quality", "tools/eval_quality.py")


def _cp(ref: dict, hyp: dict, rid: str = "r") -> dict:
    return eq.score_cpwer([(rid, ref, hyp)])


# --------------------------------------------------------------------------- #
# Anchor: one speaker, cpWER == plain WER on the identical tokens
# --------------------------------------------------------------------------- #
def test_single_speaker_cpwer_equals_plain_wer() -> None:
    eq._load_scoring()
    ref_words = ["the", "quick", "brown", "fox", "jumps"]
    hyp_words = ["the", "quick", "brown", "dog", "jumps"]  # one substitution
    cp = _cp({"speaker:A": ref_words}, {"speaker:A": hyp_words})
    wer = eq.score_wer([("r", " ".join(ref_words), " ".join(hyp_words))])
    assert (cp["substitutions"], cp["deletions"], cp["insertions"], cp["length"]) == (
        wer["substitutions"], wer["deletions"], wer["insertions"], wer["ref_words"]
    )
    assert cp["pooled_cpwer"] == wer["pooled_wer"] == pytest.approx(1 / 5)
    assert cp["unassigned_words"] == 0


# --------------------------------------------------------------------------- #
# The Hungarian assignment actually runs (naive pairing would be all wrong)
# --------------------------------------------------------------------------- #
def test_two_speaker_label_swap_scores_zero() -> None:
    ref = {"speaker:A": ["hello", "world"], "speaker:B": ["foo", "bar"]}
    hyp = {"speaker:1": ["foo", "bar"], "speaker:0": ["hello", "world"]}
    cp = _cp(ref, hyp)
    assert cp["pooled_cpwer"] == 0.0
    assert (cp["substitutions"], cp["deletions"], cp["insertions"]) == (0, 0, 0)
    assert cp["scored_speaker"] == 2


def test_orientation_deletions_not_insertions() -> None:
    # Reference longer than hypothesis: the missing words are DELETIONS. If the
    # ref/hyp arguments were swapped they would read as insertions instead, so
    # this pins the plumbing direction.
    cp = _cp({"speaker:A": ["a", "b", "c", "d"]}, {"speaker:A": ["a", "b"]})
    assert (cp["substitutions"], cp["deletions"], cp["insertions"]) == (0, 2, 0)


# --------------------------------------------------------------------------- #
# Speaker-count mismatches
# --------------------------------------------------------------------------- #
def test_extra_hypothesis_stream_is_insertions() -> None:
    # Plain words only: the frozen Whisper normalizer merges spelled-out numbers
    # ("one two" -> "12"), so number words would not be word-for-word countable.
    cp = _cp({"speaker:A": ["cat", "dog"]}, {"speaker:A": ["cat", "dog"], "speaker:B": ["fish"]})
    assert (cp["substitutions"], cp["deletions"], cp["insertions"]) == (0, 0, 1)
    assert cp["falarm_speaker"] == 1


def test_missing_hypothesis_stream_is_deletions() -> None:
    ref = {"speaker:A": ["cat", "dog"], "speaker:B": ["fish", "bird"]}
    cp = _cp(ref, {"speaker:A": ["cat", "dog"]})
    assert (cp["substitutions"], cp["deletions"], cp["insertions"]) == (0, 2, 0)
    assert cp["missed_speaker"] == 1


def test_anonymous_stream_is_assignable_to_a_real_speaker() -> None:
    # The null-label stream participates in the assignment like any other, so a
    # perfect but anonymous hypothesis scores 0 (it is NOT forced to insertions).
    ref = {"speaker:A": ["alpha", "beta"], "speaker:B": ["gamma", "delta"]}
    hyp = {"speaker:A": ["alpha", "beta"], eq.CPWER_UNASSIGNED_KEY: ["gamma", "delta"]}
    cp = _cp(ref, hyp)
    assert cp["pooled_cpwer"] == 0.0
    # But the raw unassigned count is still reported for the gate.
    assert cp["unassigned_words"] == 2


# --------------------------------------------------------------------------- #
# unassigned_words gate counts RAW tokens (normalization could hide a regression)
# --------------------------------------------------------------------------- #
def test_unassigned_words_counts_raw_pre_normalization() -> None:
    # A punctuation-only unassigned token normalizes away, but the RAW count must
    # still flag it (a lost diarization label is a regression regardless).
    ref = {"speaker:A": ["real", "words", "here"]}
    hyp = {"speaker:A": ["real", "words", "here"], eq.CPWER_UNASSIGNED_KEY: ["%%%"]}
    cp = _cp(ref, hyp)
    assert cp["unassigned_words"] == 1  # raw count, even though it normalizes to nothing


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #
def test_empty_reference_stream_is_rejected() -> None:
    with pytest.raises(eq.EvalError):
        _cp({"speaker:A": ["   "]}, {"speaker:A": ["x"]})


def test_silent_hypothesis_is_all_deletions() -> None:
    # A recording with no ASR output is an empty hypothesis stream dict; meeteval
    # scores it as all-deletions, cpWER 1.0 (never an error, never a None rate).
    cp = _cp({"speaker:A": ["a", "b", "c"]}, {})
    assert (cp["substitutions"], cp["deletions"], cp["insertions"]) == (0, 3, 0)
    assert cp["pooled_cpwer"] == 1.0
    assert cp["unassigned_words"] == 0


def test_over_twenty_speakers_raises_actionable_error() -> None:
    ref = {f"speaker:s{i}": ["w"] for i in range(21)}
    hyp = {f"speaker:h{i}": ["w"] for i in range(21)}
    with pytest.raises(eq.EvalError, match="exceeds 20"):
        _cp(ref, hyp)


def test_empty_cohort_is_rejected() -> None:
    with pytest.raises(eq.EvalError, match="empty cpWER cohort"):
        eq.score_cpwer([])


# --------------------------------------------------------------------------- #
# Pooling: micro-average over recordings (combine_error_rates sums counts)
# --------------------------------------------------------------------------- #
def test_pooled_counts_are_a_true_micro_average() -> None:
    items = [
        ("r1", {"speaker:A": ["a", "b", "c"]}, {"speaker:A": ["a", "x", "c"]}),   # 1 sub / 3
        ("r2", {"speaker:A": ["d", "e"]}, {"speaker:A": ["d", "e"]}),             # 0 / 2
    ]
    cp = eq.score_cpwer(items)
    assert cp["substitutions"] == 1 and cp["deletions"] == 0 and cp["insertions"] == 0
    assert cp["length"] == 5  # 3 + 2 pooled
    assert cp["pooled_cpwer"] == pytest.approx(1 / 5)
    # Integer identity: pooled errors == S + D + I.
    assert cp["substitutions"] + cp["deletions"] + cp["insertions"] == 1


# --------------------------------------------------------------------------- #
# The meeteval token contract: it leaves normalized tokens intact
# --------------------------------------------------------------------------- #
def test_meeteval_preserves_normalized_token_count() -> None:
    # After the frozen normalizer + .split(), every token is whitespace-free, so
    # meeteval's whitespace re-tokenization is a no-op: the reported length equals
    # the number of normalized tokens, and none of them carries internal
    # whitespace or is empty.
    eq._load_scoring()
    ref_words = ["hello", "world", "this", "is", "a", "test"]
    tokens = eq._normalize_cpwer_stream(ref_words)
    assert tokens and all(t and " " not in t for t in tokens)
    cp = _cp({"speaker:A": ref_words}, {"speaker:A": ref_words})
    assert cp["length"] == len(tokens)
    assert cp["pooled_cpwer"] == 0.0


# --------------------------------------------------------------------------- #
# cmd_score enforces the unassigned_words == 0 gate end to end
# --------------------------------------------------------------------------- #
def _cpwer_json(path: Path, rid: str, streams: dict) -> Path:
    path.write_text(json.dumps({"recording_id": rid, "streams": streams}))
    return path


def test_cmd_score_rejects_unassigned_words(tmp_path: Path) -> None:
    # A manifest whose cpWER hypothesis carries an unassigned word must fail
    # closed (score refuses to publish a fabricated improvement).
    ref = _cpwer_json(tmp_path / "ref.json", "r", {"speaker:A": ["hello", "world"]})
    hyp = _cpwer_json(
        tmp_path / "hyp.json", "r",
        {"speaker:A": ["hello"], eq.CPWER_UNASSIGNED_KEY: ["world"]},
    )
    rttm_ref = tmp_path / "ref.rttm"
    rttm_ref.write_text("SPEAKER r 1 0.000 1.000 <NA> <NA> A <NA> <NA>\n")
    rttm_hyp = tmp_path / "hyp.rttm"
    rttm_hyp.write_text("SPEAKER r 1 0.000 1.000 <NA> <NA> s <NA> <NA>\n")
    manifest = {
        "diarization": [
            {"recording_id": "r", "reference_rttm": str(rttm_ref),
             "hypothesis_rttm": str(rttm_hyp), "uem": None},
        ],
        "cpwer": [{"recording_id": "r", "reference_json": str(ref), "hypothesis_json": str(hyp)}],
    }
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(manifest))
    args = types.SimpleNamespace(manifest=str(mpath), out=None)
    with pytest.raises(eq.EvalError, match="unassigned words"):
        eq.cmd_score(args)
