"""Metal-tier diarization gate: pyannote on MPS vs forced-CPU vs the CUDA reference.

The first pyannote parity module: the metal tier moves diarization onto the
Apple GPU (torch-MPS), and per plan decision 3 there is NO committed metal
reference oracle — MPS is not run-to-run or cross-chip stable enough to be
one. Instead this gate runs the REAL service ``Diarizer`` (same code path as
production, service-default hyperparameters) on the committed 3-speaker
parity corpus and measures:

* **Decision level** — speaker-count equality across forced-MPS, forced-CPU,
  and the committed CUDA reference (``references/cuda/diarize.json``), plus a
  clustering-threshold sweep bracketing the service default: the knife-edge
  test — MPS and CPU must flip speaker counts at the SAME thresholds.
* **Boundary level** — turn-count equality and per-boundary drift vs the CUDA
  reference within a pre-registered smoke bound.
* **Mapping level** — frame-level label agreement (greedy overlap mapping)
  between MPS/CPU/reference, and an MPS repeat run for run-to-run stability.

Bounds below are PRE-REGISTERED smoke bounds (the Phase 0 spike measured
MPS ≡ CPU exactly on its corpus: identical DER/speaker counts/turns, 3x
repeats bit-stable). The post-measurement pass (plan slice 9) ratchets them
from recorded per-chip numbers; loosening any of them afterwards is a
numerics decision, not a test fix.

Runs only on maintainer Apple Silicon hardware (Gate M in
docs/release-process.md): every prerequisite below is a plain SKIP elsewhere
— ``VOXINT_PARITY_REQUIRED`` is deliberately never applied to metal lanes,
because no shared CI has this hardware. Requires pyannote.audio + torch in
the running interpreter (use the metal pyannote venv:
``uv pip install --python "$VOXINT_METAL_HOME/venvs/pyannote/bin/python" pytest``
then run pytest from that venv) and the sha-verified vendored checkpoints
(``voxint-metal.sh setup`` downloads them).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
CORPUS_DIR = REPO / "tests" / "parity" / "fixtures" / "corpus"
CUDA_DIARIZE_JSON = (
    REPO / "tests" / "parity" / "fixtures" / "references" / "cuda" / "diarize.json"
)
VENDORED_CONFIG_SRC = REPO / "services" / "pyannote" / "models" / "config.vendored.yaml"
PYANNOTE_PROVENANCE = REPO / "services" / "pyannote" / "models" / "provenance.json"
CHECKPOINT_NAMES = ("segmentation-3.0.bin", "wespeaker-voxceleb-resnet34-LM.bin")

# Service-default request parameters — the CUDA reference was recorded through
# /v1/diarize with exactly these (schema defaults).
DIARIZE_KWARGS = {"min_speakers": 1, "max_speakers": 10, "min_turn_seconds": 0.5}

# --- Pre-registered smoke bounds (ratcheted post-measurement, plan slice 9).
TURN_BOUNDARY_TOLERANCE_S = 0.25  # per matched boundary vs the CUDA reference
FRAME_AGREEMENT_VS_REFERENCE = 0.95  # greedy-mapped label agreement
FRAME_AGREEMENT_MPS_VS_CPU = 0.99  # spike measured exact agreement
FRAME_AGREEMENT_REPEAT = 0.999  # spike measured bit-stable repeats
# Brackets the service-default clustering threshold (0.55): counts must agree
# between devices at every point, including wherever the flip happens.
THRESHOLD_SWEEP = (0.50, 0.60)


def _checkpoint_dir() -> Path | None:
    """Sha-verified vendored checkpoints, wherever this machine has them:
    the repo models dir (release workflow layout) or the metal launcher's
    home. Both paths keep the load-bearing "pyannote" substring."""
    prov = json.loads(PYANNOTE_PROVENANCE.read_text())["files"]
    metal_home = Path(
        os.getenv("VOXINT_METAL_HOME", str(Path.home() / ".voxint-metal"))
    )
    candidates = (
        REPO / "services" / "pyannote" / "models",
        metal_home / "models" / "pyannote" / "vendored" / "pyannote",
    )
    for cand in candidates:
        if not all((cand / name).is_file() for name in CHECKPOINT_NAMES):
            continue
        if all(
            hashlib.sha256((cand / name).read_bytes()).hexdigest()
            == prov[name]["sha256"]
            for name in CHECKPOINT_NAMES
        ):
            return cand
    return None


_IS_APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"

pytestmark = [
    pytest.mark.skipif(
        not _IS_APPLE_SILICON, reason="metal gate runs on Apple Silicon macOS only"
    ),
    pytest.mark.skipif(
        not CUDA_DIARIZE_JSON.exists(), reason="CUDA diarize reference missing"
    ),
]

if _IS_APPLE_SILICON:
    torch = pytest.importorskip("torch", reason="pyannote metal venv required")
    pytest.importorskip("pyannote.audio", reason="pyannote metal venv required")
    pytestmark.append(
        pytest.mark.skipif(
            not torch.backends.mps.is_available(),
            reason="torch MPS backend unavailable",
        )
    )
    _CKPT_DIR = _checkpoint_dir()
    pytestmark.append(
        pytest.mark.skipif(
            _CKPT_DIR is None,
            reason="sha-verified vendored checkpoints not found — run "
            "scripts/metal/voxint-metal.sh setup",
        )
    )
else:  # pragma: no cover - non-mac collection path
    _CKPT_DIR = None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="session")
def local_config(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The committed vendored config with ONLY the two checkpoint paths
    repointed at this machine's sha-verified copies (same generation rule as
    voxint-metal.sh; the 'pyannote' path substring stays load-bearing)."""
    assert _CKPT_DIR is not None
    text = VENDORED_CONFIG_SRC.read_text().replace(
        "/app/vendored/pyannote/", f"{_CKPT_DIR}/"
    )
    assert "pyannote" in str(_CKPT_DIR), "checkpoint dir lost the pyannote substring"
    out = tmp_path_factory.mktemp("vendored") / "config.yaml"
    out.write_text(text)
    return out


@pytest.fixture(scope="session")
def cuda_reference() -> dict[str, Any]:
    ref = json.loads(CUDA_DIARIZE_JSON.read_text())
    bound = ref["meta"]["corpus_files_sha256"]["diarize-3speaker.wav"]
    actual = _sha256(CORPUS_DIR / "diarize-3speaker.wav")
    assert actual == bound, "corpus wav does not match the committed CUDA reference"
    assert ref["meta"]["service_healthz"]["device"] == "cuda"
    return ref["response"]


def _run_diarize(
    local_config: Path, device: str, threshold: float | None = None
) -> dict[str, Any]:
    """One diarization through the real service Diarizer with a FORCED device
    (slice 4 semantics: broken/absent backend errors instead of degrading)."""
    from tests.contracts.conftest import service_package

    # Pin EVERY PYANNOTE_* knob, not just the threshold: an ambient
    # PYANNOTE_SEGMENTATION_BATCH_SIZE (etc.) in the maintainer's shell would
    # silently shift what this lane measures away from the service defaults.
    ambient_pyannote = [k for k in os.environ if k.startswith("PYANNOTE_")]
    saved = {
        k: os.environ.get(k)
        for k in (
            "VOXINT_VENDORED_PIPELINE",
            "DIARIZER_DEVICE",
            "DIARIZER_MODEL_NAME",
            *ambient_pyannote,
            "PYANNOTE_CLUSTERING_THRESHOLD",
        )
    }
    for k in ambient_pyannote:
        os.environ.pop(k, None)
    os.environ["VOXINT_VENDORED_PIPELINE"] = str(local_config)
    os.environ["DIARIZER_DEVICE"] = device
    os.environ.pop("DIARIZER_MODEL_NAME", None)
    if threshold is None:
        os.environ.pop("PYANNOTE_CLUSTERING_THRESHOLD", None)
    else:
        os.environ["PYANNOTE_CLUSTERING_THRESHOLD"] = str(threshold)
    try:
        with service_package("pyannote"):
            from app.diarizer import Diarizer

            diarizer = Diarizer()
            diarizer.load_model()
            assert diarizer.device_name == ("mps" if device == "mps" else device)
            return diarizer.diarize(
                str(CORPUS_DIR / "diarize-3speaker.wav"), **DIARIZE_KWARGS
            )
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.fixture(scope="session")
def mps_result(local_config: Path) -> dict[str, Any]:
    return _run_diarize(local_config, "mps")


@pytest.fixture(scope="session")
def cpu_result(local_config: Path) -> dict[str, Any]:
    return _run_diarize(local_config, "cpu")


def _by_label(turns: list[dict[str, Any]]) -> dict[str, list[tuple[float, float]]]:
    out: dict[str, list[tuple[float, float]]] = {}
    for t in turns:
        out.setdefault(t["label"], []).append(
            (t["start_seconds"], t["end_seconds"])
        )
    return out


def _overlap(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    total = 0.0
    for sa, ea in a:
        for sb, eb in b:
            total += max(0.0, min(ea, eb) - max(sa, sb))
    return total


def frame_agreement(
    turns_a: list[dict[str, Any]], turns_b: list[dict[str, Any]]
) -> float:
    """Greedy label-mapped speech-time agreement in [0, 1].

    Labels are arbitrary per run (SPEAKER_00 on MPS need not be SPEAKER_00 on
    CPU): map label pairs greedily by descending overlapping seconds, then
    score matched overlap against the larger side's total speech time. A DER
    replacement is NOT intended — this is the smoke-level mapping-agreement
    metric the plan's slice 8 calls for; ground-truth DER lives in the
    per-chip verdict report.
    """
    by_a, by_b = _by_label(turns_a), _by_label(turns_b)
    pairs = sorted(
        (
            (_overlap(ia, ib), la, lb)
            for la, ia in by_a.items()
            for lb, ib in by_b.items()
        ),
        reverse=True,
    )
    used_a: set[str] = set()
    used_b: set[str] = set()
    matched = 0.0
    for seconds, la, lb in pairs:
        if la in used_a or lb in used_b:
            continue
        used_a.add(la)
        used_b.add(lb)
        matched += seconds

    def total(by: dict[str, list[tuple[float, float]]]) -> float:
        return sum(e - s for spans in by.values() for s, e in spans)

    # Fail closed on empty/zero-duration diarizations: two empty outputs
    # "agreeing" is a vacuous pass, not evidence.
    denom = max(total(by_a), total(by_b))
    return matched / denom if denom else 0.0


class TestDecisionLevel:
    def test_speaker_counts_agree_everywhere(
        self,
        mps_result: dict[str, Any],
        cpu_result: dict[str, Any],
        cuda_reference: dict[str, Any],
    ) -> None:
        assert mps_result["num_speakers"] == cpu_result["num_speakers"], (
            "forced-MPS and forced-CPU disagree on speaker count — "
            "decision-level MPS drift"
        )
        assert mps_result["num_speakers"] == cuda_reference["num_speakers"], (
            "MPS speaker count differs from the committed CUDA reference"
        )

    @pytest.mark.parametrize("threshold", THRESHOLD_SWEEP)
    def test_threshold_knife_edge_agrees_between_devices(
        self, local_config: Path, threshold: float
    ) -> None:
        # The clustering threshold is where small numeric drift becomes a
        # DIFFERENT ANSWER (wrong speaker count). Sweeping a bracket around
        # the service default (0.55, covered by the main runs) checks that
        # MPS and CPU sit on the same side of every edge.
        mps = _run_diarize(local_config, "mps", threshold=threshold)
        cpu = _run_diarize(local_config, "cpu", threshold=threshold)
        assert mps["num_speakers"] == cpu["num_speakers"], (
            f"threshold {threshold}: MPS found {mps['num_speakers']} speakers, "
            f"CPU {cpu['num_speakers']} — knife-edge divergence"
        )


class TestBoundaryLevel:
    def test_turns_match_reference_within_smoke_bound(
        self, mps_result: dict[str, Any], cuda_reference: dict[str, Any]
    ) -> None:
        got = sorted(mps_result["turns"], key=lambda t: t["start_seconds"])
        ref = sorted(cuda_reference["turns"], key=lambda t: t["start_seconds"])
        assert len(got) == len(ref), (
            f"MPS produced {len(got)} turns, CUDA reference has {len(ref)} "
            "(pre-registered smoke bound — investigate before loosening)"
        )
        for g, r in zip(got, ref, strict=True):
            for key in ("start_seconds", "end_seconds"):
                drift = abs(g[key] - r[key])
                assert drift <= TURN_BOUNDARY_TOLERANCE_S, (
                    f"turn boundary drift {drift:.3f}s vs reference exceeds "
                    f"{TURN_BOUNDARY_TOLERANCE_S}s ({key} of turn at "
                    f"{r['start_seconds']:.2f}s)"
                )


class TestMappingLevel:
    def test_mps_vs_cpu_agreement(
        self, mps_result: dict[str, Any], cpu_result: dict[str, Any]
    ) -> None:
        score = frame_agreement(mps_result["turns"], cpu_result["turns"])
        print(f"\nmps-vs-cpu mapping agreement: {score:.6f}")
        assert score >= FRAME_AGREEMENT_MPS_VS_CPU

    def test_mps_vs_reference_agreement(
        self, mps_result: dict[str, Any], cuda_reference: dict[str, Any]
    ) -> None:
        score = frame_agreement(mps_result["turns"], cuda_reference["turns"])
        print(f"\nmps-vs-cuda mapping agreement: {score:.6f}")
        assert score >= FRAME_AGREEMENT_VS_REFERENCE

    def test_mps_repeat_stability(
        self, local_config: Path, mps_result: dict[str, Any]
    ) -> None:
        repeat = _run_diarize(local_config, "mps")
        assert repeat["num_speakers"] == mps_result["num_speakers"]
        score = frame_agreement(mps_result["turns"], repeat["turns"])
        identical = repeat["turns"] == mps_result["turns"]
        print(f"\nmps repeat agreement: {score:.6f} (bitwise-identical: {identical})")
        assert score >= FRAME_AGREEMENT_REPEAT
