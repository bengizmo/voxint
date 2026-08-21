"""Occurrence-partition conservation for the cpWER hypothesis render (issue #97).

``eval_run.ami_hypothesis_renders`` builds the plain-WER text AND the per-label
cpWER streams from ONE cropped word list, so the two can never disagree about the
hypothesis. These pure tests (no worker, no meeteval) freeze the conservation
contract: every UEM-kept ``(segment_index, word_index)`` enters the plain stream
AND exactly one cpWER stream, null labels land in the anonymous bucket, and the
collision-free key encoding keeps a model label from ever spelling the sentinel.

``eval_run`` imports only the bakeoff ``_us`` helper (no pyannote/jiwer), so this
runs in the default dev lane.
"""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load_tool():
    path = REPO / "tools" / "eval_run.py"
    spec = importlib.util.spec_from_file_location("eval_run", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


er = _load_tool()

# One wide UEM region covering every fixture word (crop is exercised separately).
UEM = [(0, 100_000_000)]


def _w(start: float, end: float, word: str) -> dict:
    return {"start": start, "end": end, "word": word}


def test_interleaved_labels_partition_by_encoded_key() -> None:
    segments = [
        (0, "SPEAKER_00", [_w(0.0, 0.5, "hello"), _w(0.5, 1.0, "there")]),
        (1, "SPEAKER_01", [_w(1.0, 1.5, "general")]),
        (2, "SPEAKER_00", [_w(1.5, 2.0, "kenobi")]),
    ]
    text, streams = er.ami_hypothesis_renders(segments, UEM)
    assert text == "hello there general kenobi"
    assert streams == {
        "speaker:SPEAKER_00": ["hello", "there", "kenobi"],
        "speaker:SPEAKER_01": ["general"],
    }
    # Conservation: the streams hold exactly the plain-text words (as a multiset).
    assert Counter(w for v in streams.values() for w in v) == Counter(text.split())


def test_null_label_lands_in_the_anonymous_bucket() -> None:
    segments = [
        (0, None, [_w(0.0, 0.5, "orphan")]),
        (1, "SPEAKER_00", [_w(0.5, 1.0, "owned")]),
    ]
    _text, streams = er.ami_hypothesis_renders(segments, UEM)
    assert streams == {
        er.CPWER_UNASSIGNED_KEY: ["orphan"],
        "speaker:SPEAKER_00": ["owned"],
    }


def test_all_null_labels_collapse_to_one_anonymous_stream() -> None:
    segments = [
        (0, None, [_w(0.0, 0.5, "a")]),
        (1, None, [_w(0.5, 1.0, "b")]),
    ]
    text, streams = er.ami_hypothesis_renders(segments, UEM)
    assert text == "a b"
    assert streams == {er.CPWER_UNASSIGNED_KEY: ["a", "b"]}


def test_repeated_words_are_conserved_not_deduplicated() -> None:
    # A Counter/occurrence partition must survive repeated tokens: "yeah yeah
    # yeah" is three distinct occurrences, not one.
    words = [_w(0.0, 0.3, "yeah"), _w(0.3, 0.6, "yeah"), _w(0.6, 0.9, "yeah")]
    segments = [(0, "SPEAKER_00", words)]
    text, streams = er.ami_hypothesis_renders(segments, UEM)
    assert text == "yeah yeah yeah"
    assert streams == {"speaker:SPEAKER_00": ["yeah", "yeah", "yeah"]}
    assert sum(len(v) for v in streams.values()) == len(text.split())


def test_empty_segment_contributes_nothing() -> None:
    segments = [
        (0, "SPEAKER_00", []),
        (1, "SPEAKER_00", [_w(0.0, 0.5, "solo")]),
    ]
    text, streams = er.ami_hypothesis_renders(segments, UEM)
    assert text == "solo"
    assert streams == {"speaker:SPEAKER_00": ["solo"]}


def test_uem_crop_drops_word_from_both_renders() -> None:
    # The second word's midpoint (5.0s) is outside the 0..4s UEM; it must vanish
    # from BOTH the plain text and the cpWER streams (never one and not the other).
    segments = [(0, "SPEAKER_00", [_w(0.0, 2.0, "in"), _w(4.5, 5.5, "out")])]
    text, streams = er.ami_hypothesis_renders(segments, [(0, 4_000_000)])
    assert text == "in"
    assert streams == {"speaker:SPEAKER_00": ["in"]}


def test_duplicate_segment_index_is_rejected() -> None:
    # Two labelled segments sharing a segment_index would make the occurrence id
    # (segment_index, word_index) non-unique; the render must refuse it.
    segments = [
        (0, "SPEAKER_00", [_w(0.0, 0.5, "a")]),
        (0, "SPEAKER_01", [_w(0.5, 1.0, "b")]),
    ]
    with pytest.raises(er.RunError, match="duplicate"):
        er.ami_hypothesis_renders(segments, UEM)


def test_no_uem_regions_is_rejected() -> None:
    with pytest.raises(er.RunError, match="no UEM regions"):
        er.ami_hypothesis_renders([(0, "SPEAKER_00", [_w(0.0, 0.5, "x")])], [])


def test_encoded_key_cannot_collide_with_the_sentinel() -> None:
    # A model that emitted the literal label "unassigned:" must NOT collapse into
    # the anonymous bucket: the speaker: prefix keeps the namespaces disjoint.
    segments = [
        (0, "unassigned:", [_w(0.0, 0.5, "labelled")]),
        (1, None, [_w(0.5, 1.0, "anonymous")]),
    ]
    _text, streams = er.ami_hypothesis_renders(segments, UEM)
    assert streams == {
        "speaker:unassigned:": ["labelled"],
        er.CPWER_UNASSIGNED_KEY: ["anonymous"],
    }
    assert er.CPWER_UNASSIGNED_KEY != "speaker:unassigned:"


def test_plain_text_matches_ami_hypothesis_text_wrapper() -> None:
    # The back-compat wrapper (labels dropped) must render the identical plain
    # text the shared pass produces, proving there is one crop implementation.
    words = [[_w(0.0, 0.5, "one")], [_w(0.5, 1.0, "two")]]
    labeled = [(i, "SPEAKER_00", w) for i, w in enumerate(words)]
    text, _ = er.ami_hypothesis_renders(labeled, UEM)
    assert text == er.ami_hypothesis_text(words, UEM)
