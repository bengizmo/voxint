"""Torch-free service logic: whisper's repetition detector, pyannote's post-processing.

Not wire-schema tests, but they pin the behaviors the contract documents
(suspect tagging semantics, post-processing operation order, overlap_seconds).
"""

import numpy as np
import pytest

from tests.contracts.conftest import load_service_module

detector = load_service_module("whisper", "transcription")
postprocess = load_service_module("pyannote", "postprocess")
embedding = load_service_module("titanet", "embedding")


class TestRepetitionDetector:
    def test_canonical_hallucination_loop_fires(self) -> None:
        fired, score, span = detector.detect_repetition("Class " * 12)
        assert fired
        assert score > 0.9
        assert span is not None and span.startswith("class class")

    def test_normal_text_does_not_fire(self) -> None:
        fired, score, span = detector.detect_repetition(
            "hello and welcome to the show, thanks for having me today"
        )
        assert not fired
        assert score == 0.0
        assert span is None

    def test_short_natural_repetition_not_density_flagged(self) -> None:
        # Under density_min_tokens, only the run-length rule may fire; three
        # repeats stay under min_repeats=4.
        fired, _, _ = detector.detect_repetition("check check check this")
        assert not fired

    def test_ngram_run_fires(self) -> None:
        fired, _, span = detector.detect_repetition("thank you " * 6)
        assert fired
        assert span is not None

    def test_empty_input(self) -> None:
        assert detector.detect_repetition("") == (False, 0.0, None)

    def test_annotation_record_shape(self) -> None:
        record, flagged = detector.build_segment_annotation(
            start_seconds=0.0,
            end_seconds=2.0,
            text="Class " * 12,
            confidence=0.5,
            enabled=True,
        )
        assert flagged
        assert record["suspect"] is True
        assert record["suspect_score"] is not None
        assert record["text"] == "Class " * 12  # verbatim, never dropped

    def test_annotation_disabled_never_flags(self) -> None:
        record, flagged = detector.build_segment_annotation(
            start_seconds=0.0,
            end_seconds=2.0,
            text="Class " * 12,
            confidence=0.5,
            enabled=False,
        )
        assert not flagged
        assert record["suspect"] is False


class TestTurnPostprocess:
    def test_operation_order_drop_then_merge(self) -> None:
        raw = [
            {"start_seconds": 0.0, "end_seconds": 4.0, "label": "SPEAKER_00"},
            # 0.2s turn: dropped by the min filter BEFORE merging, so it can't
            # bridge the two SPEAKER_00 turns on its own.
            {"start_seconds": 4.1, "end_seconds": 4.3, "label": "SPEAKER_01"},
            # gap 4.0 → 4.5 is 0.5 ≤ min_duration_off 0.6 → merges.
            {"start_seconds": 4.5, "end_seconds": 8.0, "label": "SPEAKER_00"},
        ]
        turns, speakers = postprocess.process_turns(
            raw, min_turn_seconds=0.5, min_duration_off=0.6
        )
        assert len(turns) == 1
        assert turns[0]["start_seconds"] == 0.0
        assert turns[0]["end_seconds"] == 8.0
        assert speakers == [
            {"label": "SPEAKER_00", "total_seconds": 8.0, "num_turns": 1}
        ]

    def test_gap_wider_than_min_duration_off_not_merged(self) -> None:
        raw = [
            {"start_seconds": 0.0, "end_seconds": 4.0, "label": "SPEAKER_00"},
            {"start_seconds": 5.0, "end_seconds": 8.0, "label": "SPEAKER_00"},
        ]
        turns, _ = postprocess.process_turns(raw, min_turn_seconds=0.5, min_duration_off=0.6)
        assert len(turns) == 2

    def test_overlap_seconds_precise(self) -> None:
        raw = [
            {"start_seconds": 0.0, "end_seconds": 10.0, "label": "SPEAKER_00"},
            {"start_seconds": 9.5, "end_seconds": 12.0, "label": "SPEAKER_01"},
        ]
        turns, _ = postprocess.process_turns(raw, min_turn_seconds=0.5, min_duration_off=0.6)
        long_turn = next(t for t in turns if t["label"] == "SPEAKER_00")
        assert long_turn["overlap"] is True
        # A grazing 0.5s overlap on a 10s turn is measurable, not disqualifying.
        assert long_turn["overlap_seconds"] == pytest.approx(0.5)

    def test_same_speaker_never_overlaps_itself(self) -> None:
        raw = [
            {"start_seconds": 0.0, "end_seconds": 5.0, "label": "SPEAKER_00"},
            {"start_seconds": 4.0, "end_seconds": 9.0, "label": "SPEAKER_00"},
        ]
        turns, _ = postprocess.process_turns(raw, min_turn_seconds=0.5, min_duration_off=0.6)
        assert all(not t["overlap"] for t in turns)

    def test_summaries_describe_returned_turns_most_talkative_first(self) -> None:
        raw = [
            {"start_seconds": 0.0, "end_seconds": 2.0, "label": "SPEAKER_01"},
            {"start_seconds": 3.0, "end_seconds": 10.0, "label": "SPEAKER_00"},
        ]
        _, speakers = postprocess.process_turns(raw, min_turn_seconds=0.5, min_duration_off=0.6)
        assert [s["label"] for s in speakers] == ["SPEAKER_00", "SPEAKER_01"]


class TestSnrGate:
    def test_pure_silence_scores_zero(self) -> None:
        assert embedding.calculate_snr_db(np.zeros(32000, dtype=np.float32)) == 0.0

    def test_near_silence_scores_zero(self) -> None:
        audio = np.full(32000, 1e-8, dtype=np.float32)
        assert embedding.calculate_snr_db(audio) == 0.0

    def test_speechlike_audio_beats_the_gate(self) -> None:
        rng = np.random.default_rng(seed=7)
        # Loud bursts over a quiet-but-nonzero floor, like speech over room tone.
        floor = rng.normal(0.0, 0.001, 64000).astype(np.float32)
        bursts = np.zeros(64000, dtype=np.float32)
        bursts[8000:24000] = rng.normal(0.0, 0.5, 16000).astype(np.float32)
        snr = embedding.calculate_snr_db(floor + bursts)
        assert snr > 5.0
