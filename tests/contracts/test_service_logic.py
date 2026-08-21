"""Torch-free service logic: whisper's repetition detector, pyannote's post-processing.

Not wire-schema tests, but they pin the behaviors the contract documents
(suspect tagging semantics, post-processing operation order, overlap_seconds).
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.contracts.conftest import load_service_module

detector = load_service_module("whisper", "transcription")
whisper_backends = load_service_module("whisper", "backends")
whisper_startup = load_service_module("whisper", "whisper_startup")
whisper_ct2_legacy = load_service_module("whisper", "backends.ct2_legacy")
whisper_ct2 = load_service_module("whisper", "backends.ct2")
postprocess = load_service_module("pyannote", "postprocess")
diarizer = load_service_module("pyannote", "diarizer")
embedding = load_service_module("titanet", "embedding")
preprocess = load_service_module("titanet", "preprocess")
engine_onnx = load_service_module("titanet", "engine_onnx")


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


def _fake_backend(*, device: str) -> object:
    """A loaded backend stub exposing exactly what ``decode_identity`` reads.

    The device is the canonical post-load label (the real backends settle it
    via ``resolve_device_name`` inside ``load_model``), so a CUDA and a ROCm
    deployment reach ``decode_identity`` with everything else identical.
    """
    return SimpleNamespace(
        kind="shared_windows",
        engine="faster-whisper",
        engine_version="1.2.1",
        runtime="ctranslate2",
        runtime_version="4.8.1",
        device=device,
        model_name="large-v2",
        compute_type="int8",
        batch_size=16,
    )


class TestDecodeIdentityDevice:
    """The decode identity must hash the canonical device: a CUDA box and a
    ROCm box are otherwise byte-identical (same faster-whisper/ctranslate2
    versions, same model/compute/batch/VAD), so without device they collide."""

    def test_cuda_and_rocm_hash_differently(self) -> None:
        from tests.contracts.conftest import service_package

        cuda = detector.WhisperTranscriber(backend=_fake_backend(device="cuda"))
        rocm = detector.WhisperTranscriber(backend=_fake_backend(device="rocm"))
        # decode_identity lazily imports app.backends.vad_plan at call time.
        with service_package("whisper"):
            cuda_hash = cuda.decode_identity()["decode_config_hash"]
            rocm_hash = rocm.decode_identity()["decode_config_hash"]
        assert cuda_hash != rocm_hash

    def test_identity_is_stable_and_cached(self) -> None:
        from tests.contracts.conftest import service_package

        transcriber = detector.WhisperTranscriber(backend=_fake_backend(device="rocm"))
        with service_package("whisper"):
            first = transcriber.decode_identity()
            second = transcriber.decode_identity()
        # Same object returned (cached), and the digest is deterministic.
        assert first is second
        assert first["decode_config_hash"] == second["decode_config_hash"]


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

    def test_metal_requirements_mirror_shared_pins(self) -> None:
        # The metal venv is py3.11 like the CUDA/CPU images, so unlike rocm
        # it gets NO numpy exemption — the whole preprocessing substrate must
        # match. Torch-free like rocm; ctranslate2 must be exact-pinned (the
        # macOS arm64 wheel comes from PyPI and would otherwise float inside
        # faster-whisper's >=4.0,<5 range with nothing else pinning it).
        import re

        from tests.contracts.conftest import REPO_ROOT

        def pins(name: str) -> dict[str, str]:
            text = (REPO_ROOT / "services" / "whisper" / name).read_text()
            return dict(re.findall(r"^([A-Za-z0-9_\[\]-]+)==([0-9.]+)$", text, re.MULTILINE))

        base = pins("requirements.txt")
        metal = pins("requirements.metal.txt")
        for pkg in (
            "fastapi",
            "uvicorn[standard]",
            "pydantic",
            "faster-whisper",
            "soundfile",
            "numpy",
        ):
            assert pkg in base and pkg in metal, f"{pkg} pin missing from a flavor file"
            assert base[pkg] == metal[pkg], (
                f"{pkg}: base {base[pkg]} != metal {metal[pkg]} — bump both or neither"
            )
        assert "torch" not in metal, "the metal whisper venv is torch-free by design"
        assert "ctranslate2" in metal, (
            "requirements.metal.txt lost its exact ctranslate2 pin — the macOS "
            "wheel would float inside faster-whisper's range"
        )


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

    def test_revision_pins_an_overridden_hf_pipeline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # DIARIZER_REVISION on an HF repo-id override is carried as reproducible
        # provenance and surfaces in /healthz as model_revision (configurable
        # models A3).
        monkeypatch.setenv("DIARIZER_MODEL_NAME", "someorg/custom-pipeline")
        monkeypatch.setenv("DIARIZER_REVISION", "a" * 40)
        monkeypatch.setenv("VOXINT_VENDORED_PIPELINE", "/nonexistent/config.yaml")
        d = diarizer.Diarizer()
        assert d.model_is_local is False
        assert d.model_revision == "a" * 40

    def test_revision_ignored_for_vendored_local_source(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: object
    ) -> None:
        # The vendored config is itself the pin; a stray DIARIZER_REVISION must
        # not pretend to pin it, so model_revision stays null (healthz reports
        # null, matching the per-attempt provenance probe).
        vendored = tmp_path / "config.yaml"  # type: ignore[operator]
        vendored.write_text("version: 3.1.0\n")
        monkeypatch.delenv("DIARIZER_MODEL_NAME", raising=False)
        monkeypatch.setenv("VOXINT_VENDORED_PIPELINE", str(vendored))
        monkeypatch.setenv("DIARIZER_REVISION", "b" * 40)
        d = diarizer.Diarizer()
        assert d.model_is_local is True
        assert d.model_revision is None

    def test_no_revision_leaves_model_revision_null(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DIARIZER_MODEL_NAME", "someorg/custom-pipeline")
        monkeypatch.delenv("DIARIZER_REVISION", raising=False)
        monkeypatch.setenv("VOXINT_VENDORED_PIPELINE", "/nonexistent/config.yaml")
        d = diarizer.Diarizer()
        assert d.model_revision is None


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


class TestDiarizerFromPretrainedAdaptive:
    """The version-adaptive pipeline load (configurable models A3). pyannote 4.x
    takes revision=/token=; 3.1.1 (our pin) has neither and pins via the
    "repo@revision" string with use_auth_token=. The default (no revision) load
    must stay exactly the pre-existing no-revision path."""

    class _FakePipeline:
        """Records from_pretrained calls; ``reject`` simulates an older pyannote
        that raises TypeError on unexpected kwargs (token=/revision=)."""

        def __init__(self, reject: tuple[str, ...] = ()) -> None:
            self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
            self._reject = reject

        def from_pretrained(self, *args: object, **kwargs: object) -> object:
            if any(k in self._reject for k in kwargs):
                raise TypeError(
                    f"from_pretrained() got an unexpected keyword argument "
                    f"{next(k for k in kwargs if k in self._reject)!r}"
                )
            self.calls.append((args, kwargs))
            return object()

    def test_v4_uses_revision_and_token(self) -> None:
        fake = self._FakePipeline()
        diarizer._from_pretrained_adaptive(fake, "org/pipe", "c" * 40, "tok")
        (args, kwargs) = fake.calls[-1]
        assert args == ("org/pipe",)
        assert kwargs == {"revision": "c" * 40, "token": "tok"}

    def test_v31_falls_back_to_repo_at_revision_and_use_auth_token(self) -> None:
        # 3.1.1 rejects both revision= and token=; the pin must move into the
        # "repo@revision" string and auth into use_auth_token=.
        fake = self._FakePipeline(reject=("revision", "token"))
        diarizer._from_pretrained_adaptive(fake, "org/pipe", "c" * 40, "tok")
        (args, kwargs) = fake.calls[-1]
        assert args == ("org/pipe@" + "c" * 40,)
        assert kwargs == {"use_auth_token": "tok"}

    def test_default_no_revision_v4(self) -> None:
        fake = self._FakePipeline()
        diarizer._from_pretrained_adaptive(fake, "org/pipe", None, "tok")
        (args, kwargs) = fake.calls[-1]
        assert args == ("org/pipe",)
        assert kwargs == {"token": "tok"}

    def test_default_no_revision_v31_unchanged(self) -> None:
        # The validated/vendored default path: no revision, 3.1.1 fallback lands
        # on the plain source + use_auth_token, byte-identical to before A3.
        fake = self._FakePipeline(reject=("token",))
        diarizer._from_pretrained_adaptive(fake, "org/pipe", None, None)
        (args, kwargs) = fake.calls[-1]
        assert args == ("org/pipe",)
        assert kwargs == {"use_auth_token": None}

    def test_unrelated_typeerror_propagates(self) -> None:
        # A TypeError that is not about the auth/revision kwargs is a real bug,
        # not a version skew, and must not be swallowed by the fallback.
        class Boom:
            def from_pretrained(self, *a: object, **k: object) -> object:
                raise TypeError("something else entirely")

        with pytest.raises(TypeError, match="something else"):
            diarizer._from_pretrained_adaptive(Boom(), "org/pipe", "d" * 40, "tok")


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

    # ---- DIARIZER_DEVICE forcing (no silent fallback, plan decision 6) ----

    def test_forced_mps_probed_and_returned(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Case-insensitive on purpose: an operator writing MPS must not get
        # a confusing "not one of" error.
        monkeypatch.setenv("DIARIZER_DEVICE", "MPS")
        monkeypatch.setitem(
            sys.modules, "torch", self._fake_torch(cuda=False, mps=True)
        )
        probed: list[str] = []

        def probe(name: str) -> bool:
            probed.append(name)
            return True

        monkeypatch.setattr(diarizer, "probe_device", probe)
        assert diarizer.select_device() == "mps"
        assert probed == ["mps"]

    def test_forced_mps_without_backend_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DIARIZER_DEVICE", "mps")
        monkeypatch.setitem(
            sys.modules, "torch", self._fake_torch(cuda=True, mps=False)
        )

        def boom(name: str) -> bool:
            raise AssertionError("unavailable backend must fail before the probe")

        monkeypatch.setattr(diarizer, "probe_device", boom)
        with pytest.raises(RuntimeError, match="mps backend"):
            diarizer.select_device()

    def test_forced_cuda_without_backend_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DIARIZER_DEVICE", "cuda")
        monkeypatch.setitem(
            sys.modules, "torch", self._fake_torch(cuda=False, mps=True)
        )
        with pytest.raises(RuntimeError, match="cuda backend"):
            diarizer.select_device()

    def test_forced_mps_failing_probe_raises_not_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # THE case the forcing exists for: a backend that is "available" but
        # computes garbage must abort the service, not demote to CPU while an
        # A/B measurement believes it ran on MPS.
        monkeypatch.setenv("DIARIZER_DEVICE", "mps")
        monkeypatch.setitem(
            sys.modules, "torch", self._fake_torch(cuda=False, mps=True)
        )
        monkeypatch.setattr(diarizer, "probe_device", lambda name: False)
        with pytest.raises(RuntimeError, match="sanity probe"):
            diarizer.select_device()

    def test_forced_cpu_never_probed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DIARIZER_DEVICE", "cpu")
        monkeypatch.setitem(
            sys.modules, "torch", self._fake_torch(cuda=True, mps=True)
        )

        def boom(name: str) -> bool:
            raise AssertionError("forced cpu must not be probed")

        monkeypatch.setattr(diarizer, "probe_device", boom)
        assert diarizer.select_device() == "cpu"

    def test_unknown_forced_value_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DIARIZER_DEVICE", "tpu")
        with pytest.raises(RuntimeError, match="auto\\|cuda\\|mps\\|cpu"):
            diarizer.select_device()

    def test_blank_env_means_auto(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Compose-style ${DIARIZER_DEVICE:-} passes "" through — that must
        # run the normal cascade, not error.
        monkeypatch.setenv("DIARIZER_DEVICE", "")
        monkeypatch.setitem(
            sys.modules, "torch", self._fake_torch(cuda=False, mps=False)
        )
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


class TestWhisperEngineRegistry:
    """WHISPER_ENGINE selects the backend, lazy-imports it, and fails closed on
    unknown values — mirrors titanet's EMBED_ENGINE factory (#33 Slice 2a)."""

    def _make(self, **_kw: object) -> object:
        from tests.contracts.conftest import service_package

        with service_package("whisper"):
            return whisper_backends.create_transcriber(
                model_name="large-v2", device="cpu", compute_type="int8", batch_size=4
            )

    def test_default_is_ct2_legacy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("WHISPER_ENGINE", raising=False)
        transcriber = self._make()
        # Identity unchanged from the pre-seam shipped path (healthz contract).
        assert transcriber._backend.kind == "legacy_file"  # type: ignore[attr-defined]
        assert transcriber.engine == "faster-whisper"  # type: ignore[attr-defined]
        assert transcriber.runtime == "ctranslate2"  # type: ignore[attr-defined]
        assert transcriber.model_name == "large-v2"  # type: ignore[attr-defined]

    def test_ct2_legacy_selected_explicitly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WHISPER_ENGINE", "ct2-legacy")
        transcriber = self._make()
        assert transcriber._backend.kind == "legacy_file"  # type: ignore[attr-defined]

    def test_unknown_engine_fails_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A truly unknown engine raises rather than silently running on CPU.
        monkeypatch.setenv("WHISPER_ENGINE", "mlx")
        from tests.contracts.conftest import service_package

        with service_package("whisper"), pytest.raises(ValueError, match="mlx"):
            whisper_backends.create_transcriber(
                model_name="large-v2", device="cpu", compute_type="int8", batch_size=4
            )

    def test_ct2_shared_resolves_to_shared_windows_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # ct2 (shared-VAD) resolves the shared_windows backend (Slice 2b): a
        # real decode engine exposing decode_windows + transcribe_raw, with the
        # same /healthz identity as the legacy path (only the decode differs).
        monkeypatch.setenv("WHISPER_ENGINE", "ct2")
        transcriber = self._make()
        assert transcriber._backend.kind == "shared_windows"  # type: ignore[attr-defined]
        assert transcriber.engine == "faster-whisper"  # type: ignore[attr-defined]
        assert transcriber.runtime == "ctranslate2"  # type: ignore[attr-defined]
        assert transcriber.model_name == "large-v2"  # type: ignore[attr-defined]
        assert callable(transcriber._backend.decode_windows)  # type: ignore[attr-defined]
        assert callable(transcriber._backend.transcribe_raw)  # type: ignore[attr-defined]


class TestWhisperStartupResolution:
    """The fail-closed whisper model-selection truth table (configurable models
    A2). The validated large-v2 default keeps the baked, offline path untouched;
    an alternate model must be explicitly and fully gated or the service refuses
    to start. The resolver is pure over an env mapping — no model, no container."""

    BAKED = "f0fe81560cb8b68660e564f55dd99207059c092e"
    ALT = "0123456789abcdef0123456789abcdef01234567"

    def _resolve(self, **env: str) -> object:
        return whisper_startup.resolve_whisper_startup(env)

    def test_unset_model_is_the_baked_default(self) -> None:
        d = self._resolve()
        assert d.model_name == "large-v2"
        assert d.is_override is False
        assert d.env_overrides == {}
        assert d.warning is None

    @pytest.mark.parametrize("model", ["large-v2", "Systran/faster-whisper-large-v2"])
    def test_default_model_forms_change_nothing(self, model: str) -> None:
        # Both accepted spellings of the validated default resolve to the baked,
        # offline path with no env overrides — even if a stray ALLOW_DOWNLOAD is
        # present, the default is never an override.
        d = self._resolve(WHISPER_MODEL=model, WHISPER_ALLOW_DOWNLOAD="1")
        assert d.is_override is False
        assert d.env_overrides == {}

    def test_blank_model_fails_closed(self) -> None:
        with pytest.raises(whisper_startup.WhisperStartupError, match="empty"):
            self._resolve(WHISPER_MODEL="   ")

    def test_default_path_restores_baked_revision_when_env_blank(self) -> None:
        # The compose overlays forward ${WHISPER_REVISION:-}, so an operator who
        # does not override gets WHISPER_REVISION="" — which shadows the image's
        # baked revision and would make the offline load resolve "main" and fail.
        # On the default path the resolver restores the baked revision so the
        # shipped path stays offline-clean.
        d = self._resolve(
            WHISPER_MODEL="large-v2", WHISPER_REVISION="", WHISPER_BAKED_REVISION=self.BAKED
        )
        assert d.is_override is False
        assert d.env_overrides == {"WHISPER_REVISION": self.BAKED}

    def test_default_path_leaves_operator_revision_untouched(self) -> None:
        # A non-empty WHISPER_REVISION on the default path is already correct;
        # the resolver must not clobber it with the baked reference.
        d = self._resolve(
            WHISPER_MODEL="large-v2", WHISPER_REVISION=self.ALT, WHISPER_BAKED_REVISION=self.BAKED
        )
        assert d.env_overrides == {}

    @pytest.mark.parametrize("model", ["/models/local", "./rel", "~/m", "a/b/c"])
    def test_path_like_model_rejected(self, model: str) -> None:
        with pytest.raises(whisper_startup.WhisperStartupError, match=r"repo id|path"):
            self._resolve(WHISPER_MODEL=model, WHISPER_ALLOW_DOWNLOAD="1",
                          WHISPER_REVISION=self.ALT)

    def test_alternate_without_allow_download_fails_closed(self) -> None:
        with pytest.raises(
            whisper_startup.WhisperStartupError, match="WHISPER_ALLOW_DOWNLOAD=1"
        ):
            self._resolve(WHISPER_MODEL="openai/whisper-large-v3", WHISPER_REVISION=self.ALT)

    @pytest.mark.parametrize(
        "revision",
        ["", "main", "v3", "abc123", "F0FE81560CB8B68660E564F55DD99207059C092E"],
    )
    def test_alternate_requires_full_lowercase_sha(self, revision: str) -> None:
        with pytest.raises(whisper_startup.WhisperStartupError, match="40-character"):
            self._resolve(
                WHISPER_MODEL="openai/whisper-large-v3",
                WHISPER_ALLOW_DOWNLOAD="1",
                WHISPER_REVISION=revision,
            )

    def test_alternate_still_carrying_baked_revision_fails_closed(self) -> None:
        # The image bakes WHISPER_REVISION to the large-v2 SHA (a valid 40-char
        # value); an operator who sets an alternate model but forgets to change
        # the revision must fail closed, not silently fetch the wrong snapshot.
        with pytest.raises(whisper_startup.WhisperStartupError, match="baked"):
            self._resolve(
                WHISPER_MODEL="openai/whisper-large-v3",
                WHISPER_ALLOW_DOWNLOAD="1",
                WHISPER_REVISION=self.BAKED,
                WHISPER_BAKED_REVISION=self.BAKED,
            )

    def test_fully_gated_alternate_enables_fetch_to_separate_cache(self) -> None:
        d = self._resolve(
            WHISPER_MODEL="openai/whisper-large-v3",
            WHISPER_ALLOW_DOWNLOAD="1",
            WHISPER_REVISION=self.ALT,
            WHISPER_BAKED_REVISION=self.BAKED,
        )
        assert d.is_override is True
        assert d.model_name == "openai/whisper-large-v3"
        # Downloads land in the SEPARATE cache, never over the baked root.
        assert d.env_overrides["WHISPER_DOWNLOAD_ROOT"] == whisper_startup.ALT_CACHE_ROOT
        assert whisper_startup.ALT_CACHE_ROOT != "/app/.cache/whisper"
        # ALLOW_DOWNLOAD=1 is the explicit authority that turns offline off.
        assert d.env_overrides["HF_HUB_OFFLINE"] == "0"
        assert d.warning is not None and "unvalidated" in d.warning

    def test_apply_mutates_environ_and_returns_decision(self) -> None:
        env: dict[str, str] = {
            "WHISPER_MODEL": "openai/whisper-large-v3",
            "WHISPER_ALLOW_DOWNLOAD": "1",
            "WHISPER_REVISION": self.ALT,
            "WHISPER_BAKED_REVISION": self.BAKED,
        }
        d = whisper_startup.apply_whisper_startup(env)
        assert d.is_override is True
        assert env["WHISPER_DOWNLOAD_ROOT"] == whisper_startup.ALT_CACHE_ROOT
        assert env["HF_HUB_OFFLINE"] == "0"

    def test_apply_default_leaves_offline_untouched(self) -> None:
        # The baked default must never have its offline flag flipped by the applier.
        env = {"HF_HUB_OFFLINE": "1"}
        whisper_startup.apply_whisper_startup(env)
        assert env["HF_HUB_OFFLINE"] == "1"
        assert "WHISPER_DOWNLOAD_ROOT" not in env


class TestCt2DeviceVerify:
    """ct2-legacy fails closed on a device CTranslate2 cannot run — the whisper
    analogue of pyannote's probe_device (#33 Slice 2a)."""

    def test_shipped_devices_pass(self) -> None:
        # No-op assertion for the shipped cpu/cuda strings (no model load).
        whisper_ct2_legacy.Ct2LegacyBackend(device="cpu").verify_device()
        whisper_ct2_legacy.Ct2LegacyBackend(device="cuda").verify_device()

    def test_unsupported_device_fails_closed(self) -> None:
        with pytest.raises(RuntimeError, match="mps"):
            whisper_ct2_legacy.Ct2LegacyBackend(device="mps").verify_device()

    def test_verify_uses_requested_device_not_relabelled(self) -> None:
        # verify_device must read the raw requested device, so a cuda->rocm
        # relabel at load can never mask an originally-unsupported request.
        backend = whisper_ct2_legacy.Ct2LegacyBackend(device="cuda")
        backend.device = "rocm"  # simulate post-load relabelling
        backend.verify_device()  # still passes: requested device was cuda

    def test_ct2_shared_shipped_devices_pass(self) -> None:
        # The shared-window ct2 backend applies the same fail-closed device
        # policy as ct2-legacy (no-op for the shipped cpu/cuda strings).
        whisper_ct2.Ct2Backend(device="cpu").verify_device()
        whisper_ct2.Ct2Backend(device="cuda").verify_device()

    def test_ct2_shared_unsupported_device_fails_closed(self) -> None:
        with pytest.raises(RuntimeError, match="mps"):
            whisper_ct2.Ct2Backend(device="mps").verify_device()


def _fake_ort(available: list[str], session_providers: list[str]) -> SimpleNamespace:
    """Stub onnxruntime: a session that reports session_providers regardless
    of what was requested — exactly ORT's silent-degradation behavior."""

    class _FakeSession:
        def __init__(self, path: str, providers: list[str] | None = None) -> None:
            self._providers = list(session_providers)

        def get_providers(self) -> list[str]:
            return list(self._providers)

        def get_inputs(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(name="audio_signal"), SimpleNamespace(name="length")]

        def get_outputs(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(name="embs")]

    return SimpleNamespace(
        __version__="0.0-test",
        get_available_providers=lambda: list(available),
        InferenceSession=_FakeSession,
    )


class TestOrtProviderSelection:
    """TITANET_ORT_PROVIDERS: priority-ordered EP selection with no silent
    fallback — a requested-but-unavailable EP fails BEFORE session
    construction, and a session that quietly degraded (ORT drops an EP it
    cannot initialize with only a log line) fails AFTER, so healthz can never
    report a device that is not actually in use."""

    def test_parse_priority_order_and_blank_entries(self) -> None:
        raw = " CoreMLExecutionProvider , CPUExecutionProvider ,,"
        assert engine_onnx.parse_ort_providers(raw) == [
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ]

    def test_parse_blank_means_unset(self) -> None:
        # A compose-style ${TITANET_ORT_PROVIDERS:-} passes "" through; that
        # must behave like unset (caller falls back to the default), not crash.
        assert engine_onnx.parse_ort_providers("") == []
        assert engine_onnx.parse_ort_providers(" , ") == []

    def test_default_is_exactly_the_measured_cpu_chain(self) -> None:
        # Shipped images set no TITANET_ORT_PROVIDERS — the default must stay
        # exactly what the committed parity verdict measured.
        assert engine_onnx.DEFAULT_ORT_PROVIDERS == ["CPUExecutionProvider"]

    def test_unavailable_provider_fails_before_construction(self) -> None:
        with pytest.raises(RuntimeError, match="CoreMLExecutionProvider"):
            engine_onnx.validate_requested_providers(
                ["CoreMLExecutionProvider"], ["CPUExecutionProvider"]
            )

    def test_available_subset_passes(self) -> None:
        engine_onnx.validate_requested_providers(
            ["CPUExecutionProvider"],
            ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        )

    def test_silent_degradation_fails_at_load(self) -> None:
        with pytest.raises(RuntimeError, match="silently degraded"):
            engine_onnx.assert_session_honors_providers(
                ["CoreMLExecutionProvider", "CPUExecutionProvider"],
                ["CPUExecutionProvider"],
            )

    def test_appended_cpu_fallback_is_tolerated(self) -> None:
        # ORT always appends the CPU EP as final fallback; requesting CoreML
        # alone must accept ["CoreML", "CPU"] back (prefix rule, not equality).
        engine_onnx.assert_session_honors_providers(
            ["CoreMLExecutionProvider"],
            ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        )

    def test_device_name_mapping(self) -> None:
        assert engine_onnx.provider_device_name("CPUExecutionProvider") == "cpu"
        assert engine_onnx.provider_device_name("CoreMLExecutionProvider") == "metal"
        assert engine_onnx.provider_device_name("CUDAExecutionProvider") == "cuda"
        assert engine_onnx.provider_device_name("ROCMExecutionProvider") == "rocm"

    def test_load_model_refuses_unavailable_provider(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Wiring, not just helpers: load_model must consult availability
        # before constructing a session (the fake would accept anything).
        onnx = tmp_path / "graph.onnx"
        onnx.write_bytes(b"")
        monkeypatch.setenv("TITANET_ONNX_PATH", str(onnx))
        monkeypatch.setenv("TITANET_ORT_PROVIDERS", "CoreMLExecutionProvider")
        monkeypatch.setitem(
            sys.modules,
            "onnxruntime",
            _fake_ort(["CPUExecutionProvider"], ["CPUExecutionProvider"]),
        )
        embedder = engine_onnx.OnnxEmbedder()
        with pytest.raises(RuntimeError, match="only provides"):
            embedder.load_model()
        assert embedder.model_loaded is False

    def test_load_model_detects_silent_degradation(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        onnx = tmp_path / "graph.onnx"
        onnx.write_bytes(b"")
        monkeypatch.setenv("TITANET_ONNX_PATH", str(onnx))
        monkeypatch.setenv("TITANET_ORT_PROVIDERS", "CoreMLExecutionProvider")
        monkeypatch.setitem(
            sys.modules,
            "onnxruntime",
            _fake_ort(
                ["CoreMLExecutionProvider", "CPUExecutionProvider"],
                ["CPUExecutionProvider"],  # session dropped CoreML anyway
            ),
        )
        embedder = engine_onnx.OnnxEmbedder()
        with pytest.raises(RuntimeError, match="silently degraded"):
            embedder.load_model()
        assert embedder.model_loaded is False

    def test_load_model_default_reports_cpu(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Unset env → the shipped-image behavior, bit-for-bit: CPU EP only.
        onnx = tmp_path / "graph.onnx"
        onnx.write_bytes(b"")
        monkeypatch.setenv("TITANET_ONNX_PATH", str(onnx))
        monkeypatch.delenv("TITANET_ORT_PROVIDERS", raising=False)
        monkeypatch.setitem(
            sys.modules,
            "onnxruntime",
            _fake_ort(["CPUExecutionProvider"], ["CPUExecutionProvider"]),
        )
        embedder = engine_onnx.OnnxEmbedder()
        embedder.load_model()
        assert embedder.device_name == "cpu"
        assert embedder.model_loaded is True


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
        # This frame-count arithmetic is pure (no librosa) — always runs.
        valid = mel.num_valid_frames(16000)
        assert valid == 101
        # mel_spectrogram() lazily imports librosa (mel.py:_filterbank), which
        # ships only in the optional `parity` extra — skip just the spectrogram
        # shape check on a `uv sync --extra dev` checkout (it still runs in the
        # parity lane: `uv run --extra dev --extra parity pytest`).
        pytest.importorskip("librosa")
        out = mel.mel_spectrogram(np.zeros(16000, dtype=np.float32))
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
        table = [(512, 4), (16000, 101), (16001, 101), (16160, 102), (64000, 401)]
        # Frame-count arithmetic is pure (no librosa) — always runs.
        for samples, frames in table:
            assert mel.num_valid_frames(samples) == frames, samples
        # The spectrogram shape check needs librosa (parity extra); skip only that
        # part on a librosa-less dev lane — it still runs under `--extra parity`.
        pytest.importorskip("librosa")
        for samples, frames in table:
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

    def test_llm_gguf_sha_matches_provenance(self) -> None:
        # The bundled-LLM image bakes a vendored, sha-pinned Qwen GGUF (#67).
        # Drift between the Dockerfile's build-time hash gate and the committed
        # provenance would let a differently-quantized weight ship silently and
        # invalidate the #66 qualification the serving profile was measured on.
        import json
        import re

        from tests.contracts.conftest import REPO_ROOT

        llm_dir = REPO_ROOT / "services" / "llama-cpp"
        dockerfile = (llm_dir / "Dockerfile").read_text()
        match = re.search(r"ARG QWEN_GGUF_SHA256=([0-9a-f]{64})", dockerfile)
        assert match is not None, "Dockerfile lost its QWEN_GGUF_SHA256 default"
        provenance = json.loads((llm_dir / "provenance.json").read_text())
        assert match.group(1) == provenance["sha256"], (
            "services/llama-cpp/Dockerfile QWEN_GGUF_SHA256 drifted from provenance.json"
        )

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
        # silently ignores it. Globbed so a new overlay is checked the day it
        # appears: files with zero pins are exempt (rewiring-only overlays
        # like compose.metal.yaml and the :dev build overlays carry no
        # published-image pins by design), but the image-bearing files
        # must each keep exactly one — a glob that stopped finding those
        # would make this test vacuously green.
        import re

        from tests.contracts.conftest import REPO_ROOT

        image_bearing = {
            "compose.yaml", "compose.gpu.yaml", "compose.cpu.yaml", "compose.rocm.yaml",
            "compose.llm.yaml",
        }
        pins: dict[str, set[str]] = {}
        seen = set()
        for path in sorted(REPO_ROOT.glob("compose*.yaml")):
            seen.add(path.name)
            found = set(re.findall(r"VOXINT_IMAGE_TAG:-([0-9][0-9.]*)", path.read_text()))
            if path.name in image_bearing:
                assert len(found) == 1, (
                    f"{path.name} must carry exactly one VOXINT_IMAGE_TAG pin, found: {found}"
                )
            elif not found:
                continue
            assert len(found) == 1, f"{path.name} has inconsistent internal pins: {found}"
            pins[path.name] = found
        assert image_bearing <= seen, f"image-bearing compose files missing: {image_bearing - seen}"
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
        for name in (
            "compose.yaml",
            "compose.gpu.yaml",
            "compose.cpu.yaml",
            "compose.rocm.yaml",
            "compose.metal.yaml",
            "compose.ytdlp-egress.yaml",
            "compose.llm.yaml",
        ):
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

    def test_metal_overlay_is_rewiring_only(self) -> None:
        # The metal tier's model services run natively on the host — the
        # overlay's ONLY job is pointing api/worker at them and stamping the
        # tier. An image, volume, or port sneaking in here would silently
        # turn "bare-metal tier" back into a container deployment (or expose
        # a listener) without any gate noticing.
        import yaml

        from tests.contracts.conftest import REPO_ROOT

        doc = yaml.safe_load((REPO_ROOT / "compose.metal.yaml").read_text())
        assert set(doc) == {"services"}, f"unexpected top-level keys: {set(doc) - {'services'}}"
        services = doc["services"]
        # BOTH core callers must be rewired: a missing one would resolve the
        # base compose.yaml URLs and call services that do not exist.
        assert set(services) == {"api", "worker"}, f"unexpected services: {set(services)}"
        expected_urls = {
            "ASR_URL": "http://host.docker.internal:8022",
            "DIARIZER_URL": "http://host.docker.internal:8024",
            "EMBEDDER_URL": "http://host.docker.internal:8021",
        }
        for svc_name, svc in services.items():
            assert set(svc) <= {"environment", "extra_hosts"}, (
                f"{svc_name} carries non-rewiring keys: {set(svc) - {'environment', 'extra_hosts'}}"
            )
            env = svc["environment"]
            assert env.get("COMPUTE_TIER") == "metal", f"{svc_name} missing COMPUTE_TIER: metal"
            for key, url in expected_urls.items():
                assert env.get(key) == url, (
                    f"{svc_name} {key} is {env.get(key)!r}, expected {url!r}"
                )
            # host-gateway keeps host.docker.internal resolving on engines
            # that don't provide the name natively (Docker Desktop does).
            assert "host.docker.internal:host-gateway" in svc.get("extra_hosts", []), (
                f"{svc_name} missing the host-gateway extra_hosts mapping"
            )

    def test_ytdlp_egress_overlay_wiring(self) -> None:
        # Issue #16: the opt-in restricted-egress overlay must (a) route the
        # worker's yt-dlp egress through the filtering proxy via the always-passed
        # YTDLP_PROXY, (b) keep the worker on its normal network so DB/Redis/model
        # services still resolve, (c) run the proxy on an INTERNAL worker-facing
        # network plus a routable one, and (d) run it least-privilege. A drift in
        # any of these silently turns the hardening off or breaks the worker.
        import yaml

        from tests.contracts.conftest import REPO_ROOT

        doc = yaml.safe_load((REPO_ROOT / "compose.ytdlp-egress.yaml").read_text())
        assert set(doc) == {"services", "networks"}, (
            f"unexpected top-level keys: {set(doc) - {'services', 'networks'}}"
        )
        worker = doc["services"]["worker"]
        # (a) + (b): worker points yt-dlp at the proxy AND keeps the default network.
        assert worker["environment"]["YTDLP_PROXY"] == "http://ytdlp-egress-proxy:3128"
        assert set(worker["networks"]) == {"default", "ytdlp_egress"}, (
            f"worker networks must keep 'default' + add 'ytdlp_egress', got {worker['networks']}"
        )
        proxy = doc["services"]["ytdlp-egress-proxy"]
        # (c): the proxy bridges the internal client net to a routable one, and it
        # runs the egress-proxy module from the SAME app image (one pin).
        assert set(proxy["networks"]) == {"ytdlp_egress", "ytdlp_public"}
        assert proxy["image"].startswith("ghcr.io/bengizmo/voxint:")
        assert proxy["command"] == [
            "python", "-m", "voxint.media.egress_proxy", "--listen", "0.0.0.0:3128"
        ]
        # (d): least privilege — no writes, no new privileges, no extra caps.
        assert proxy["read_only"] is True
        assert proxy["cap_drop"] == ["ALL"]
        assert "no-new-privileges:true" in proxy["security_opt"]
        # The worker-facing network has no external route; the proxy is the only
        # way out. (An accidental non-internal here would defeat the isolation.)
        assert doc["networks"]["ytdlp_egress"].get("internal") is True
        assert doc["networks"].get("ytdlp_public") in ({}, None), (
            "ytdlp_public must be a plain routable network (no 'internal: true')"
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
        # The metal venv joins the same parity set: its torch/torchaudio must
        # track Dockerfile.cpu exactly (the MPS spike measured 2.5.0; a
        # one-sided bump would fork numerics between the container and native
        # deployments of the same service).
        assert torch_base("services/pyannote/requirements.metal.txt") == torch_base(
            "services/pyannote/Dockerfile.cpu"
        )

        def torchaudio_base(path: str) -> str:
            text = (REPO_ROOT / path).read_text()
            match = re.search(r"torchaudio==([0-9.]+)", text)
            assert match is not None, f"{path} has no torchaudio pin"
            return match.group(1)

        assert torchaudio_base("services/pyannote/requirements.metal.txt") == torchaudio_base(
            "services/pyannote/Dockerfile.cpu"
        )

    def test_pyannote_metal_requirements_include_shared_stack(self) -> None:
        # The metal flavor must stay a THIN additive layer: the shared stack
        # comes from `-r requirements.txt` (single source, drift impossible),
        # and setuptools must stay inside the images' >=70,<81 window —
        # pyannote.database imports pkg_resources, removed in setuptools 81
        # (uv venvs ship no setuptools, so losing the pin breaks startup).
        from tests.contracts.conftest import REPO_ROOT

        text = (
            REPO_ROOT / "services" / "pyannote" / "requirements.metal.txt"
        ).read_text()
        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        assert "-r requirements.txt" in lines, (
            "requirements.metal.txt must include the shared stack via "
            "-r requirements.txt, not fork a copy of it"
        )
        assert "setuptools>=70,<81" in lines, (
            "requirements.metal.txt lost the setuptools<81 window that keeps "
            "pkg_resources importable for pyannote.database"
        )
        # Additive-only: nothing beyond the include, the torch pair, and
        # setuptools — anything else belongs in requirements.txt.
        extras = set(lines) - {
            "-r requirements.txt",
            "setuptools>=70,<81",
        }
        assert all(
            line.startswith(("torch==", "torchaudio==")) for line in extras
        ), f"unexpected additions in requirements.metal.txt: {sorted(extras)}"


class TestCudaTitanetImageProvenance:
    """The CUDA titanet image downloads the TitaNet-Large .nemo at build time
    (from_pretrained, no revision arg), so a build-time sha256 gate is the only
    thing standing between a re-published upstream checkpoint and silent
    embedding-space drift on the one weights lane CI cannot parity-gate (no GPU
    runner). Pin the sha ARG to provenance, keep the gate wired, and bind the
    runtime load offline so it cannot re-fetch a different snapshot at startup."""

    @staticmethod
    def _dockerfile() -> str:
        from tests.contracts.conftest import REPO_ROOT

        # The CUDA Dockerfile (not Dockerfile.cpu) is the one that bakes .nemo.
        return (REPO_ROOT / "services" / "titanet" / "Dockerfile").read_text()

    def test_dockerfile_nemo_sha_matches_provenance(self) -> None:
        import json
        import re

        from tests.contracts.conftest import REPO_ROOT

        # Anchor to an active ARG instruction so a commented-out line can't pass.
        match = re.search(
            r"(?m)^ARG TITANET_NEMO_SHA256=([0-9a-f]{64})$", self._dockerfile()
        )
        assert match is not None, (
            "services/titanet/Dockerfile lost its active TITANET_NEMO_SHA256 default"
        )
        provenance = json.loads(
            (
                REPO_ROOT / "tests" / "parity" / "fixtures" / "onnx" / "provenance.json"
            ).read_text()
        )
        assert match.group(1) == provenance["nemo_checkpoint_sha256"], (
            "services/titanet/Dockerfile TITANET_NEMO_SHA256 drifted from "
            "provenance.json nemo_checkpoint_sha256"
        )

    def test_dockerfile_verifies_the_baked_nemo_against_the_arg(self) -> None:
        # The pin is only meaningful if a build step actually consumes it: guard
        # against deleting the sha256sum gate while leaving the ARG default.
        dockerfile = self._dockerfile()
        assert "sha256sum -c" in dockerfile, (
            "services/titanet/Dockerfile lost the sha256sum -c weight-integrity gate"
        )
        assert "${TITANET_NEMO_SHA256}" in dockerfile, (
            "services/titanet/Dockerfile no longer feeds TITANET_NEMO_SHA256 into "
            "the checksum gate"
        )

    def test_runtime_is_offline_bound(self) -> None:
        # engine_nemo.py calls from_pretrained again at startup; without an
        # offline bind an online host could re-resolve a re-published checkpoint
        # that never passed the build-time gate, reopening the drift path.
        assert "ENV HF_HUB_OFFLINE=1" in self._dockerfile(), (
            "services/titanet/Dockerfile lost HF_HUB_OFFLINE=1; the runtime "
            "from_pretrained could re-download an unverified checkpoint"
        )


class TestWhisperOfflineStartup:
    """Weights are baked/pre-downloaded, so no whisper deployment may phone
    home at startup (issue #30): an unadvertised HF hub revision check would
    stall or fail on air-gapped hosts and could re-download a different
    revision than the one the numerics were measured against."""

    WHISPER_DOCKERFILES = ("Dockerfile", "Dockerfile.cpu", "Dockerfile.rocm")

    def test_all_whisper_deployments_pin_the_same_hf_revision(self) -> None:
        # The bake, the runtime WHISPER_REVISION env, and the metal
        # launcher's pre-download must all agree — a one-sided sha edit is
        # silent revision drift between deployment flavors.
        import re

        from tests.contracts.conftest import REPO_ROOT

        shas: set[str] = set()
        for name in self.WHISPER_DOCKERFILES:
            text = (REPO_ROOT / "services" / "whisper" / name).read_text()
            found = re.findall(r"ARG WHISPER_HF_REVISION=([0-9a-f]{40})", text)
            assert found, f"whisper {name} lost its WHISPER_HF_REVISION default"
            shas.update(found)
        launcher = (REPO_ROOT / "scripts" / "metal" / "voxint-metal.sh").read_text()
        match = re.search(r"^WHISPER_HF_REVISION=([0-9a-f]{40})$", launcher, re.M)
        assert match is not None, "metal launcher lost its WHISPER_HF_REVISION pin"
        shas.add(match.group(1))
        assert len(shas) == 1, f"WHISPER_HF_REVISION drifted across deployments: {shas}"

    @pytest.mark.parametrize("dockerfile_name", ("Dockerfile", "Dockerfile.cpu", "Dockerfile.rocm"))
    def test_whisper_images_are_offline_clean(self, dockerfile_name: str) -> None:
        # HF_HUB_OFFLINE forbids the hub call; WHISPER_REVISION pins the load
        # to the baked snapshot (a bare sha resolves offline with no ref
        # lookup — revision-less loads resolve "main", which HF_HUB_OFFLINE
        # can only satisfy if the bake wrote refs/main).
        from tests.contracts.conftest import REPO_ROOT

        text = (REPO_ROOT / "services" / "whisper" / dockerfile_name).read_text()
        assert "ENV HF_HUB_OFFLINE=1" in text, f"{dockerfile_name} lost HF_HUB_OFFLINE=1"
        assert "ENV WHISPER_REVISION=${WHISPER_HF_REVISION}" in text, (
            f"{dockerfile_name} runtime stage lost the WHISPER_REVISION pin"
        )
        # The startup resolver rejects an alternate model that still carries the
        # baked SHA; that guard needs the baked reference in the image, sourced
        # from the same ARG so it can never drift from WHISPER_REVISION.
        assert "ENV WHISPER_BAKED_REVISION=${WHISPER_HF_REVISION}" in text, (
            f"{dockerfile_name} lost the WHISPER_BAKED_REVISION resolver reference"
        )
