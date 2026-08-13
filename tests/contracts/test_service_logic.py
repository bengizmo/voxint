"""Torch-free service logic: whisper's repetition detector, pyannote's post-processing.

Not wire-schema tests, but they pin the behaviors the contract documents
(suspect tagging semantics, post-processing operation order, overlap_seconds).
"""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tests.contracts.conftest import load_service_module

detector = load_service_module("whisper", "transcription")
postprocess = load_service_module("pyannote", "postprocess")
embedding = load_service_module("titanet", "embedding")
preprocess = load_service_module("titanet", "preprocess")


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
        turns, speakers = postprocess.process_turns(raw, min_turn_seconds=0.5, min_duration_off=0.6)
        assert len(turns) == 1
        assert turns[0]["start_seconds"] == 0.0
        assert turns[0]["end_seconds"] == 8.0
        assert speakers == [{"label": "SPEAKER_00", "total_seconds": 8.0, "num_turns": 1}]

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
        assert preprocess.calculate_snr_db(np.zeros(32000, dtype=np.float32)) == 0.0

    def test_near_silence_scores_zero(self) -> None:
        audio = np.full(32000, 1e-8, dtype=np.float32)
        assert preprocess.calculate_snr_db(audio) == 0.0

    def test_speechlike_audio_beats_the_gate(self) -> None:
        rng = np.random.default_rng(seed=7)
        # Loud bursts over a quiet-but-nonzero floor, like speech over room tone.
        floor = rng.normal(0.0, 0.001, 64000).astype(np.float32)
        bursts = np.zeros(64000, dtype=np.float32)
        bursts[8000:24000] = rng.normal(0.0, 0.5, 16000).astype(np.float32)
        snr = preprocess.calculate_snr_db(floor + bursts)
        assert snr > 5.0


class TestWindowSampleBounds:
    """Space-definition step 1: truncating slice math, end clamped to media."""

    def test_truncating_not_rounding(self) -> None:
        # 0.9999s x 16000 = 15998.4 → truncates to 15998, never rounds up.
        assert preprocess.window_sample_bounds(0.9999, 2.0, 16000, 100000) == (15998, 32000)

    def test_end_clamped_to_media_length(self) -> None:
        assert preprocess.window_sample_bounds(0.0, 10.0, 16000, 48000) == (0, 48000)

    def test_matches_previous_inline_math(self) -> None:
        # Refactor gate: identical to the pre-extraction inline expressions.
        for start_s, end_s, sr, total in [
            (12.1, 18.9, 16000, 10**7),
            (0.0, 0.5, 16000, 8000),
            (3.333, 4.444, 16000, 70000),
        ]:
            expected = (int(start_s * sr), min(int(end_s * sr), total))
            assert preprocess.window_sample_bounds(start_s, end_s, sr, total) == expected


class TestL2Normalize:
    def test_unit_norm_and_previous_inline_math(self) -> None:
        rng = np.random.default_rng(seed=3)
        vec = rng.normal(0.0, 1.0, 192)
        out = preprocess.l2_normalize(vec)
        assert np.allclose(out, vec / (np.linalg.norm(vec) + 1e-8))
        assert abs(float(np.linalg.norm(out)) - 1.0) < 1e-6


class TestResolveDeviceName:
    """Honest /healthz device reporting: rocm must never report as cuda."""

    def test_non_cuda_passthrough_without_torch(self) -> None:
        assert embedding.resolve_device_name("cpu") == "cpu"
        assert embedding.resolve_device_name("mps") == "mps"

    def test_cuda_stays_cuda_without_hip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_torch = SimpleNamespace(version=SimpleNamespace(hip=None))
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        assert embedding.resolve_device_name("cuda") == "cuda"

    def test_cuda_reports_rocm_when_hip_build(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_torch = SimpleNamespace(version=SimpleNamespace(hip="6.2.41133"))
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        assert embedding.resolve_device_name("cuda") == "rocm"


class TestEngineFactory:
    """EMBED_ENGINE selects the engine and fails fast on unknown values."""

    def test_unknown_engine_fails_fast(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EMBED_ENGINE", "tensorrt")
        with pytest.raises(ValueError, match="tensorrt"):
            embedding.create_embedder()

    def test_default_is_nemo(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from tests.contracts.conftest import service_package

        monkeypatch.delenv("EMBED_ENGINE", raising=False)
        with service_package("titanet"):
            embedder = embedding.create_embedder()
        assert embedder.engine == "nemo"
        assert embedder.runtime == "torch"

    def test_onnx_engine_selected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from tests.contracts.conftest import service_package

        monkeypatch.setenv("EMBED_ENGINE", "onnx")
        with service_package("titanet"):
            embedder = embedding.create_embedder()
        assert embedder.engine == "onnxruntime"
        assert embedder.runtime == "onnxruntime"


class TestMelConstantsPinnedToCheckpoint:
    """app/mel.py constants must match the checkpoint's dumped preprocessor
    config (tests/parity/fixtures/onnx/preprocessor-config.json) — the mel
    front-end is part of the titanet-large-v1 space definition."""

    def test_constants_match_dumped_config(self) -> None:
        import json
        from pathlib import Path

        mel = load_service_module("titanet", "mel")
        cfg_path = (
            Path(__file__).resolve().parents[1]
            / "parity"
            / "fixtures"
            / "onnx"
            / "preprocessor-config.json"
        )
        cfg = json.loads(cfg_path.read_text())
        assert cfg["sample_rate"] == mel.SAMPLE_RATE
        assert int(cfg["window_size"] * cfg["sample_rate"]) == mel.WIN_LENGTH
        assert int(cfg["window_stride"] * cfg["sample_rate"]) == mel.HOP_LENGTH
        assert cfg["n_fft"] == mel.N_FFT
        assert cfg["features"] == mel.N_MELS
        assert cfg["window"] == "hann"
        assert cfg["normalize"] == "per_feature"
        assert cfg["frame_splicing"] == 1  # mel.py implements splicing == 1 only

    def test_frame_count_formula(self) -> None:
        mel = load_service_module("titanet", "mel")
        # floor(L / hop) + 1; output is exactly the valid frames — NEVER padded
        # to NeMo's pad_to (the exported graph has no conv masking, so padding
        # leaks into convolutions; measured 0.988 vs 0.999999 cosine on 1 s).
        audio = np.zeros(16000, dtype=np.float32)
        out = mel.mel_spectrogram(audio)
        valid = mel.num_valid_frames(16000)
        assert valid == 101
        assert out.shape == (mel.N_MELS, valid)


class TestMelEdgeCases:
    """Fail-loud preconditions for the mel front-end (part of the space
    definition's non-NeMo runtime requirements)."""

    def test_non_mono_rejected(self) -> None:
        mel = load_service_module("titanet", "mel")
        with pytest.raises(ValueError, match="mono"):
            mel.mel_spectrogram(np.zeros((2, 16000), dtype=np.float32))

    def test_short_input_rejected_not_nan(self) -> None:
        # < N_FFT samples would NaN the unbiased per-feature std (seq_len==1)
        # or break reflect padding; the module must raise, never emit NaN.
        mel = load_service_module("titanet", "mel")
        with pytest.raises(ValueError, match="samples"):
            mel.mel_spectrogram(np.zeros(511, dtype=np.float32))

    def test_frame_count_table(self) -> None:
        mel = load_service_module("titanet", "mel")
        # floor(L / hop) + 1 across the range the service can produce.
        for samples, frames in [(512, 4), (16000, 101), (16001, 101), (16160, 102), (64000, 401)]:
            assert mel.num_valid_frames(samples) == frames, samples
            out = mel.mel_spectrogram(np.zeros(samples, dtype=np.float32))
            assert out.shape == (mel.N_MELS, frames), samples

    def test_dither_pinned_but_not_applied(self) -> None:
        # The checkpoint config carries dither=1e-5; NeMo applies it only in
        # training mode, so eval-mode extraction omits it (provenance records
        # the verified source lines). Pin the config value so a checkpoint
        # swap can't silently change what "omitted" means.
        import json
        from pathlib import Path

        cfg = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "parity"
                / "fixtures"
                / "onnx"
                / "preprocessor-config.json"
            ).read_text()
        )
        assert cfg["dither"] == pytest.approx(1e-5)
