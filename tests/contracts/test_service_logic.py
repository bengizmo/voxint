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
diarizer = load_service_module("pyannote", "diarizer")
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


# The services carry their own copy by design (separate images, no shared
# package); the contract is the same, so every copy is pinned here. whisper's
# copy additionally falls through to a HIP-runtime probe (tested below).
@pytest.mark.parametrize(
    "module", [embedding, diarizer, detector], ids=["titanet", "pyannote", "whisper"]
)
class TestResolveDeviceName:
    """Honest /healthz device reporting: rocm must never report as cuda."""

    def test_non_cuda_passthrough_without_torch(self, module: object) -> None:
        assert module.resolve_device_name("cpu") == "cpu"
        assert module.resolve_device_name("mps") == "mps"

    def test_cuda_stays_cuda_without_hip(
        self, module: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_torch = SimpleNamespace(version=SimpleNamespace(hip=None))
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        assert module.resolve_device_name("cuda") == "cuda"

    def test_cuda_reports_rocm_when_hip_build(
        self, module: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_torch = SimpleNamespace(version=SimpleNamespace(hip="6.2.41133"))
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        assert module.resolve_device_name("cuda") == "rocm"


class TestWhisperHipRuntimeProbe:
    """whisper's torch-free rocm detection: the -rocm image ships no torch,
    so the ONLY honest-device signal is the HIP runtime library the loaded
    CT2 extension maps (gpu-contracts.md device honesty requirement)."""

    def _drop_torch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # sys.modules[name] = None makes `import torch` raise ImportError.
        monkeypatch.setitem(sys.modules, "torch", None)

    def test_reports_rocm_when_hip_mapped_and_no_torch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._drop_torch(monkeypatch)
        monkeypatch.setattr(detector, "_hip_runtime_loaded", lambda: True)
        assert detector.resolve_device_name("cuda") == "rocm"

    def test_stays_cuda_when_no_hip_and_no_torch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._drop_torch(monkeypatch)
        monkeypatch.setattr(detector, "_hip_runtime_loaded", lambda: False)
        assert detector.resolve_device_name("cuda") == "cuda"

    def test_probe_matches_hip_library_in_binary_maps(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: object
    ) -> None:
        import builtins
        import io

        real_open = builtins.open

        def fake_open(path: object, *args: object, **kwargs: object) -> object:
            if path == "/proc/self/maps":
                return io.BytesIO(
                    b"7f0 r-xp /opt/rocm-7.0.2/lib/libamdhip64.so.7\n"
                )
            return real_open(path, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "open", fake_open)
        assert detector._hip_runtime_loaded() is True

    def test_probe_never_raises_on_unreadable_maps(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins

        def raise_oserror(*args: object, **kwargs: object) -> object:
            raise OSError("no /proc here")

        monkeypatch.setattr(builtins, "open", raise_oserror)
        assert detector._hip_runtime_loaded() is False
        # And through the public entry point it degrades to the raw label.
        self._drop_torch(monkeypatch)
        assert detector.resolve_device_name("cuda") == "cuda"


class TestWhisperFlavorPinParity:
    def test_rocm_requirements_mirror_shared_pins(self) -> None:
        # Same philosophy as the titanet cpu/cuda pin mirror: a one-sided
        # bump of a shared package silently forks behavior across flavors.
        # numpy and torch are DELIBERATELY exempt: the rocm image is py3.12
        # (numpy 1.26.4 vs 1.24.3) and torch-free (documented in
        # requirements.rocm.txt).
        import re

        from tests.contracts.conftest import REPO_ROOT

        def pins(name: str) -> dict[str, str]:
            text = (REPO_ROOT / "services" / "whisper" / name).read_text()
            return dict(re.findall(r"^([A-Za-z0-9_\[\]-]+)==([0-9.]+)$", text, re.MULTILINE))

        base = pins("requirements.txt")
        rocm = pins("requirements.rocm.txt")
        for pkg in ("fastapi", "uvicorn[standard]", "pydantic", "faster-whisper", "soundfile"):
            assert pkg in base and pkg in rocm, f"{pkg} pin missing from a flavor file"
            assert base[pkg] == rocm[pkg], (
                f"{pkg}: base {base[pkg]} != rocm {rocm[pkg]} — bump both or neither"
            )
        assert "torch" not in rocm, "the rocm image is torch-free by design"


class TestDiarizerModelResolution:
    """The offline-by-default model resolution that closes issue #24: explicit
    DIARIZER_MODEL_NAME > existing vendored config > HF repo id fallback —
    with /healthz always reporting the canonical pipeline identity unless an
    explicit override changes it."""

    CANONICAL = "pyannote/speaker-diarization-3.1"

    def test_explicit_model_name_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DIARIZER_MODEL_NAME", "someorg/custom-pipeline")
        monkeypatch.setenv("VOXINT_VENDORED_PIPELINE", "/nonexistent/config.yaml")
        d = diarizer.Diarizer()
        assert d.model_name == "someorg/custom-pipeline"
        assert d.model_source == "someorg/custom-pipeline"
        assert d.model_is_local is False

    def test_vendored_config_is_default_and_keeps_canonical_identity(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: object
    ) -> None:
        vendored = tmp_path / "config.yaml"  # type: ignore[operator]
        vendored.write_text("version: 3.1.0\n")
        monkeypatch.delenv("DIARIZER_MODEL_NAME", raising=False)
        monkeypatch.setenv("VOXINT_VENDORED_PIPELINE", str(vendored))
        d = diarizer.Diarizer()
        assert d.model_name == self.CANONICAL  # /healthz identity contract
        assert d.model_source == str(vendored)
        assert d.model_is_local is True

    def test_unset_vendored_and_missing_default_falls_back_to_hf(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Bare-host venv runs: no env, no /app/vendored — the online HF path
        # is the only option and identity stays canonical.
        monkeypatch.delenv("DIARIZER_MODEL_NAME", raising=False)
        monkeypatch.delenv("VOXINT_VENDORED_PIPELINE", raising=False)
        d = diarizer.Diarizer()
        assert d.model_name == self.CANONICAL
        assert d.model_source == self.CANONICAL
        assert d.model_is_local is False

    def test_explicitly_configured_missing_vendored_path_fails_fast(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A typo'd/broken explicit vendored path must not silently degrade to
        # a gated network fetch.
        monkeypatch.delenv("DIARIZER_MODEL_NAME", raising=False)
        monkeypatch.setenv("VOXINT_VENDORED_PIPELINE", "/nonexistent/config.yaml")
        with pytest.raises(RuntimeError, match="does not exist"):
            diarizer.Diarizer()


class TestDeviceCascade:
    """pyannote's cuda → mps → cpu selection, every candidate probe-gated."""

    @staticmethod
    def _fake_torch(*, cuda: bool, mps: bool | None) -> SimpleNamespace:
        # mps=None models a torch build without the mps backend attribute.
        backends = SimpleNamespace()
        if mps is not None:
            backends.mps = SimpleNamespace(is_available=lambda: mps)
        return SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: cuda),
            backends=backends,
        )

    def test_cuda_wins_when_probe_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(
            sys.modules, "torch", self._fake_torch(cuda=True, mps=False)
        )
        monkeypatch.setattr(diarizer, "probe_device", lambda name: True)
        assert diarizer.select_device() == "cuda"

    def test_probe_failure_falls_through_to_cpu(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(
            sys.modules, "torch", self._fake_torch(cuda=True, mps=True)
        )
        probed: list[str] = []

        def failing_probe(name: str) -> bool:
            probed.append(name)
            return False

        monkeypatch.setattr(diarizer, "probe_device", failing_probe)
        assert diarizer.select_device() == "cpu"
        assert probed == ["cuda", "mps"]

    def test_mps_selected_without_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(
            sys.modules, "torch", self._fake_torch(cuda=False, mps=True)
        )
        monkeypatch.setattr(diarizer, "probe_device", lambda name: name == "mps")
        assert diarizer.select_device() == "mps"

    def test_cpu_floor_never_probed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(
            sys.modules, "torch", self._fake_torch(cuda=False, mps=False)
        )

        def boom(name: str) -> bool:
            raise AssertionError("cpu floor must not be probed")

        monkeypatch.setattr(diarizer, "probe_device", boom)
        assert diarizer.select_device() == "cpu"

    def test_torch_without_mps_backend_attribute(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(
            sys.modules, "torch", self._fake_torch(cuda=False, mps=None)
        )
        monkeypatch.setattr(diarizer, "probe_device", lambda name: True)
        assert diarizer.select_device() == "cpu"


class _ProbeTensor:
    """numpy-backed stand-in for torch tensors inside probe_device.

    ``to_behavior`` injects the device-transfer outcome under test: identity
    (healthy device), corruption (silent-wrong-output backend), or raising
    (transfer failure).
    """

    def __init__(self, arr: "np.ndarray", to_behavior: object) -> None:
        self.arr = arr
        self._to = to_behavior

    def __matmul__(self, other: "_ProbeTensor") -> "_ProbeTensor":
        return _ProbeTensor(self.arr @ other.arr, self._to)

    def to(self, device_name: str) -> "_ProbeTensor":
        return self._to(self)  # type: ignore[operator]

    def cpu(self) -> "_ProbeTensor":
        return self


class TestProbeDevice:
    """The probe itself (not the cascade): must return False — never raise —
    for wrong-output, NaN, and crashing backends, and True for a healthy one."""

    @staticmethod
    def _install_fake_torch(
        monkeypatch: pytest.MonkeyPatch, to_behavior: object
    ) -> None:
        base = np.random.default_rng(0).normal(size=(64, 64))

        def randn(*shape: int, generator: object = None) -> _ProbeTensor:
            return _ProbeTensor(base.copy(), to_behavior)

        fake = SimpleNamespace(
            Generator=lambda: SimpleNamespace(manual_seed=lambda seed: None),
            randn=randn,
            allclose=lambda a, b, rtol=1e-5, atol=1e-8: bool(
                np.allclose(a.arr, b.arr, rtol=rtol, atol=atol)
            ),
        )
        monkeypatch.setitem(sys.modules, "torch", fake)

    def test_healthy_device_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._install_fake_torch(monkeypatch, lambda t: t)
        assert diarizer.probe_device("mps") is True

    def test_silently_wrong_output_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The historical MPS failure mode: no exception, wrong numbers.
        self._install_fake_torch(
            monkeypatch, lambda t: _ProbeTensor(t.arr + 1.0, t._to)
        )
        assert diarizer.probe_device("mps") is False

    def test_nan_output_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._install_fake_torch(
            monkeypatch, lambda t: _ProbeTensor(t.arr * float("nan"), t._to)
        )
        assert diarizer.probe_device("mps") is False

    def test_transfer_crash_returns_false_not_raise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(t: _ProbeTensor) -> _ProbeTensor:
            raise RuntimeError("device out of memory")

        self._install_fake_torch(monkeypatch, boom)
        assert diarizer.probe_device("cuda") is False


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
        # mel_spectrogram() lazily imports librosa (mel.py:_filterbank); librosa
        # ships only in the optional `parity` extra, so skip cleanly rather than
        # hard-fail on a `uv sync --extra dev` checkout. Still runs in the parity
        # lane (`uv run --extra dev --extra parity pytest`).
        pytest.importorskip("librosa")
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
        # See test_frame_count_formula: mel_spectrogram() needs librosa (parity
        # extra); skip cleanly when it is absent rather than erroring.
        pytest.importorskip("librosa")
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


class TestCpuImageProvenance:
    """The CPU image must bake exactly the artifact + pins the parity verdict
    was measured against — drift here is silent embedding-space drift."""

    def test_dockerfile_onnx_sha_matches_provenance(self) -> None:
        import json
        import re

        from tests.contracts.conftest import REPO_ROOT

        dockerfile = (REPO_ROOT / "services" / "titanet" / "Dockerfile.cpu").read_text()
        match = re.search(r"ARG TITANET_ONNX_SHA256=([0-9a-f]{64})", dockerfile)
        assert match is not None, "Dockerfile.cpu lost its TITANET_ONNX_SHA256 default"
        provenance = json.loads(
            (
                REPO_ROOT / "tests" / "parity" / "fixtures" / "onnx" / "provenance.json"
            ).read_text()
        )
        assert match.group(1) == provenance["onnx_sha256"]

    @pytest.mark.parametrize("dockerfile_name", ["Dockerfile", "Dockerfile.cpu"])
    def test_pyannote_dockerfile_shas_match_provenance(self, dockerfile_name: str) -> None:
        import json
        import re

        from tests.contracts.conftest import REPO_ROOT

        pyannote_dir = REPO_ROOT / "services" / "pyannote"
        dockerfile = (pyannote_dir / dockerfile_name).read_text()
        provenance = json.loads((pyannote_dir / "models" / "provenance.json").read_text())
        for arg, filename in (
            ("SEGMENTATION_SHA256", "segmentation-3.0.bin"),
            ("WESPEAKER_SHA256", "wespeaker-voxceleb-resnet34-LM.bin"),
        ):
            match = re.search(rf"ARG {arg}=([0-9a-f]{{64}})", dockerfile)
            assert match is not None, f"{dockerfile_name} lost its {arg} default"
            assert match.group(1) == provenance["files"][filename]["sha256"], (
                f"{dockerfile_name} {arg} drifted from models/provenance.json"
            )

    def test_pyannote_vendored_config_references_baked_paths(self) -> None:
        # The vendored pipeline config must reference exactly the paths the
        # Dockerfiles COPY the checkpoints to — and the embedding path must
        # contain "pyannote", or pyannote.audio 3.1.1's substring dispatch
        # routes it to the (uninstalled) ONNX WeSpeaker loader.
        from tests.contracts.conftest import REPO_ROOT

        config = (
            REPO_ROOT / "services" / "pyannote" / "models" / "config.vendored.yaml"
        ).read_text()
        embedding = "/app/vendored/pyannote/wespeaker-voxceleb-resnet34-LM.bin"
        assert f"embedding: {embedding}" in config
        assert "segmentation: /app/vendored/pyannote/segmentation-3.0.bin" in config
        assert "pyannote" in embedding

    @pytest.mark.parametrize("dockerfile_name", ["Dockerfile", "Dockerfile.cpu"])
    def test_pyannote_dockerfiles_bake_the_vendored_tree(self, dockerfile_name: str) -> None:
        # The sha ARGs alone don't guarantee the checkpoints land where the
        # vendored config points: a one-sided edit of a COPY destination
        # builds green and fails only at container boot. Pin the full wiring.
        from tests.contracts.conftest import REPO_ROOT

        dockerfile = (REPO_ROOT / "services" / "pyannote" / dockerfile_name).read_text()
        for line in (
            "models/config.vendored.yaml /app/vendored/config.yaml",
            "models/provenance.json /app/vendored/provenance.json",
            "models/segmentation-3.0.bin /app/vendored/pyannote/segmentation-3.0.bin",
            "models/wespeaker-voxceleb-resnet34-LM.bin"
            " /app/vendored/pyannote/wespeaker-voxceleb-resnet34-LM.bin",
            "ENV VOXINT_VENDORED_PIPELINE=/app/vendored/config.yaml",
        ):
            assert line in dockerfile, f"{dockerfile_name} lost: {line}"

    def test_pyannote_vendored_config_params_match_provenance(self) -> None:
        # The sha gates cover the checkpoints; the pipeline hyperparameters in
        # the vendored config are pinned here against the values recorded from
        # the upstream config at its pinned revision — a silent edit to
        # clustering/segmentation params must fail a contract, not ship.
        import json

        import yaml

        from tests.contracts.conftest import REPO_ROOT

        models_dir = REPO_ROOT / "services" / "pyannote" / "models"
        cfg = yaml.safe_load((models_dir / "config.vendored.yaml").read_text())
        pinned = json.loads((models_dir / "provenance.json").read_text())["pipeline_params"]

        assert str(cfg["version"]) == pinned["version"]
        assert cfg["pipeline"]["name"] == pinned["pipeline_name"]
        pp = cfg["pipeline"]["params"]
        assert pp["clustering"] == pinned["clustering"]
        assert pp["embedding_batch_size"] == pinned["embedding_batch_size"]
        assert pp["embedding_exclude_overlap"] == pinned["embedding_exclude_overlap"]
        assert pp["segmentation_batch_size"] == pinned["segmentation_batch_size"]
        clustering = cfg["params"]["clustering"]
        assert clustering["method"] == pinned["clustering_method"]
        assert clustering["min_cluster_size"] == pinned["clustering_min_cluster_size"]
        assert clustering["threshold"] == pytest.approx(
            pinned["clustering_threshold"], abs=0.0
        )
        assert cfg["params"]["segmentation"]["min_duration_off"] == pytest.approx(
            pinned["segmentation_min_duration_off"], abs=0.0
        )

    def test_cpu_requirements_mirror_cuda_preprocess_pins(self) -> None:
        from tests.contracts.conftest import REPO_ROOT

        def pins(path: str) -> dict[str, str]:
            out: dict[str, str] = {}
            for line in (REPO_ROOT / "services" / "titanet" / path).read_text().splitlines():
                line = line.split("#", 1)[0].strip()
                if "==" in line:
                    name, version = line.split("==", 1)
                    out[name.strip().lower()] = version.strip()
            return out

        cuda, cpu = pins("requirements.txt"), pins("requirements.cpu.txt")
        # The full preprocessing chain that DEFINES titanet-large-v1, plus its
        # numerics substrate — every one must match the CUDA image exactly.
        for pkg in ("numpy", "librosa", "soundfile", "scipy", "numba",
                    "pyloudnorm", "noisereduce"):
            assert cpu[pkg] == cuda[pkg], f"{pkg}: cpu {cpu[pkg]} != cuda {cuda[pkg]}"

    def test_cpu_onnxruntime_matches_parity_harness(self) -> None:
        # The binding invariant is image == HARNESS (what the parity verdict
        # actually measured), pinned in three places: the image requirements,
        # the pyproject parity extra, and the uv.lock resolution. NOT the
        # export-time ORT in provenance.json — that records graph fidelity
        # at export and may lag the runtime.
        import re

        from tests.contracts.conftest import REPO_ROOT

        reqs = (REPO_ROOT / "services" / "titanet" / "requirements.cpu.txt").read_text()
        image_match = re.search(r"^onnxruntime==([0-9.]+)$", reqs, re.MULTILINE)
        assert image_match is not None, "requirements.cpu.txt lost its onnxruntime pin"
        image_pin = image_match.group(1)

        pyproject = (REPO_ROOT / "pyproject.toml").read_text()
        assert f'"onnxruntime=={image_pin}"' in pyproject, (
            f"pyproject parity extra does not pin onnxruntime=={image_pin}"
        )

        lock = (REPO_ROOT / "uv.lock").read_text()
        lock_match = re.search(
            r'name = "onnxruntime"\nversion = "([0-9.]+)"', lock
        )
        assert lock_match is not None and lock_match.group(1) == image_pin, (
            f"uv.lock resolves onnxruntime {lock_match and lock_match.group(1)}, "
            f"image pins {image_pin}"
        )

    def test_compose_default_pins_identical(self) -> None:
        # release-process.md step 1 requires an ATOMIC pin bump across all
        # compose files; this makes it machine-checked. A partial bump ships
        # a mixed-version stack — e.g. an app that predates COMPUTE_TIER and
        # silently ignores it.
        import re

        from tests.contracts.conftest import REPO_ROOT

        pins: dict[str, set[str]] = {}
        for name in ("compose.yaml", "compose.gpu.yaml", "compose.cpu.yaml", "compose.rocm.yaml"):
            found = set(
                re.findall(r"VOXINT_IMAGE_TAG:-([0-9][0-9.]*)", (REPO_ROOT / name).read_text())
            )
            assert len(found) == 1, f"{name} has inconsistent internal pins: {found}"
            pins[name] = found
        assert len(set().union(*pins.values())) == 1, f"compose pin skew: {pins}"
        # The pins must also match the package version: a release commit that
        # bumps pyproject but not the compose defaults ships pulls of tags
        # that do not exist yet (bit the 0.5.0 draft: 0.4.1-rocm never existed).
        pyproject = (REPO_ROOT / "pyproject.toml").read_text()
        version = re.search(r'^version = "([0-9.]+)"', pyproject, re.MULTILINE)
        assert version is not None
        assert set().union(*pins.values()) == {version.group(1)}, (
            f"compose pins {pins} disagree with pyproject version {version.group(1)}"
        )
        # The runtime constant and the .env.example override comment are part
        # of the same atomic bump: `voxint --version` and /healthz report
        # __version__, and users who uncomment the documented VOXINT_IMAGE_TAG
        # would silently run the previous release (bit the 0.5.1 draft: both
        # still said 0.5.0 after pyproject/compose were bumped).
        import voxint

        assert voxint.__version__ == version.group(1), (
            f"voxint.__version__ {voxint.__version__} disagrees with "
            f"pyproject version {version.group(1)}"
        )
        env_example = (REPO_ROOT / ".env.example").read_text()
        env_pin = re.search(r"VOXINT_IMAGE_TAG=([0-9][0-9.]*)", env_example)
        assert env_pin is not None, ".env.example lost its VOXINT_IMAGE_TAG line"
        assert env_pin.group(1) == version.group(1), (
            f".env.example VOXINT_IMAGE_TAG {env_pin.group(1)} disagrees with "
            f"pyproject version {version.group(1)}"
        )

    def test_long_running_services_have_restart_policy(self) -> None:
        # Issue #23: a transient model-service crash left the stack down
        # because nothing carried a restart policy. Every long-running service
        # in every overlay must self-heal; only the one-shot migrate job may
        # (and must) opt out — restarting it after exit 0 would loop alembic.
        import yaml

        from tests.contracts.conftest import REPO_ROOT

        base = yaml.safe_load((REPO_ROOT / "compose.yaml").read_text())["services"]
        for name in ("compose.yaml", "compose.gpu.yaml", "compose.cpu.yaml", "compose.rocm.yaml"):
            services = yaml.safe_load((REPO_ROOT / name).read_text())["services"]
            for svc_name, svc in services.items():
                # Overlay entries are partial overrides that compose merges
                # onto the base file — a missing key there inherits the base
                # service's policy.
                restart = svc.get("restart", base.get(svc_name, {}).get("restart"))
                expected = "no" if svc_name == "migrate" else "unless-stopped"
                assert restart == expected, (
                    f"{name}:{svc_name} effective restart policy is {restart!r}, "
                    f"expected {expected!r}"
                )

    def test_torch_pins_match_across_flavors(self) -> None:
        # A one-sided torch bump silently changes cross-flavor numerics; the
        # CUDA and CPU images must agree on the base torch version per
        # service (local build tags like +cu118 stripped).
        import re

        from tests.contracts.conftest import REPO_ROOT

        def torch_base(path: str) -> str:
            text = (REPO_ROOT / path).read_text()
            match = re.search(r"torch==([0-9.]+)", text)
            assert match is not None, f"{path} has no torch pin"
            return match.group(1)

        assert torch_base("services/whisper/Dockerfile") == torch_base(
            "services/whisper/Dockerfile.cpu"
        )
        assert torch_base("services/pyannote/Dockerfile") == torch_base(
            "services/pyannote/Dockerfile.cpu"
        )
