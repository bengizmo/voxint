"""Contract tests for the synthdetect eval container (#144, M1 S2).

CPU-only, no torch/fairseq/GPU. These bind the eval-container spec
(``services/synthdetect/Dockerfile.eval``, ``requirements.eval.txt``,
``provenance.eval.json``) to the pins-as-data registry, the runner's constants,
and the host scorer, so a drift between them is caught at test time instead of
at a GPU build in S2b. They mirror the titanet/pyannote/MiniLM provenance gates
in ``tests/contracts/test_service_logic.py`` and ``test_text_embedding_deps.py``.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SVC = REPO / "services" / "synthdetect"
DOCKERFILE = SVC / "Dockerfile.eval"
REQUIREMENTS = SVC / "requirements.eval.txt"
PROVENANCE = SVC / "provenance.eval.json"

sys.path.insert(0, str(REPO / "tools"))

import synthdetect_eval as se  # noqa: E402
import synthdetect_infer as si  # noqa: E402
from synthdetect_sources import default_model  # noqa: E402


def _provenance() -> dict:
    return json.loads(PROVENANCE.read_text())


def _dockerfile() -> str:
    return DOCKERFILE.read_text()


def test_eval_container_files_exist() -> None:
    for path in (DOCKERFILE, REQUIREMENTS, PROVENANCE):
        assert path.is_file(), f"missing eval-container file {path}"


def test_journal_schema_version_matches_scorer() -> None:
    # The runner writes the journal the host scorer reads; a schema drift would
    # make every produced journal unscoreable. Pin the two constants together.
    assert si.JOURNAL_SCHEMA_VERSION == se.JOURNAL_SCHEMA_VERSION


def test_provenance_identity_matches_registry_default() -> None:
    prov = _provenance()
    model = default_model()
    assert prov["for_model"] == model.model_id
    assert prov["inference_space"] == model.inference_space
    assert prov["reference_runtime"] == "fairseq"


def test_provenance_canonicalization_id_matches_runner() -> None:
    # The corpus is canonicalized to exactly this id at acquisition and the runner
    # refuses anything else; provenance must name the same one.
    assert _provenance()["canonicalization_id"] == si.CANONICALIZATION_ID


def test_provenance_weight_files_match_registry() -> None:
    prov = _provenance()
    model = default_model()
    assert set(prov["weights"]["files"]) == {w.filename for w in model.weights}
    for w in model.weights:
        assert prov["weights"]["files"][w.filename]["license"] == w.license_spdx


def test_candidate_state_is_coherent() -> None:
    # In S2a nothing is frozen yet: the registry weights are CANDIDATE, so
    # provenance must not claim a qualified/pinned state or carry frozen shas.
    prov = _provenance()
    model = default_model()
    assert model.weights_pinned() is False
    assert prov["weights"]["qualification_state"] == "candidate"
    for f in prov["weights"]["files"].values():
        assert f["sha256"] is None
        assert f["size_bytes"] is None
    assert prov["runtime"]["fairseq_commit"] is None
    assert prov["base_image_digest"] is None
    assert prov["model_repo"]["commit"] is None


def test_scoring_semantics_match_the_runner_header() -> None:
    # The provenance's declared polarity/column must match what build_header pins,
    # so the negated-column-1 convention is stated identically in both places.
    prov = _provenance()["scoring_semantics"]
    model = default_model()
    header = si.build_header(
        model=model, manifest_sha256="d" * 64, split=None,
        selected_clip_ids=["a"], windowing_mode="upstream",
        runtime={}, flags={}, weights={}, runner_git={},
        created_at="t", run_id="r", host="h",
    )
    assert prov["output_column"] == header["scoring"]["output_column"] == 1
    assert header["scoring"]["polarity"] == "higher-is-more-synthetic"


def test_dockerfile_requires_fairseq_commit_arg_without_default() -> None:
    # FAIRSEQ_COMMIT must be an ARG with NO default so an unpinned runtime can
    # never be built by accident; the build asserts it is non-empty.
    df = _dockerfile()
    assert re.search(r"^ARG FAIRSEQ_COMMIT\s*$", df, flags=re.MULTILINE), (
        "Dockerfile.eval must declare `ARG FAIRSEQ_COMMIT` with no default"
    )
    assert 'test -n "${FAIRSEQ_COMMIT}"' in df


def test_dockerfile_base_image_matches_provenance() -> None:
    df = _dockerfile()
    prov = _provenance()
    match = re.search(r"^FROM (\S+)", df, flags=re.MULTILINE)
    assert match is not None
    assert match.group(1) == prov["base_image"]


def test_dockerfile_bakes_determinism_env_matching_provenance() -> None:
    df = _dockerfile()
    prov = _provenance()["determinism_env"]
    for key, value in (("CUBLAS_WORKSPACE_CONFIG", prov["CUBLAS_WORKSPACE_CONFIG"]),
                       ("PYTHONHASHSEED", prov["PYTHONHASHSEED"])):
        assert re.search(rf"^ENV {key}={re.escape(value)}\s*$", df, flags=re.MULTILINE), (
            f"Dockerfile.eval must set ENV {key}={value} (deterministic runtime)"
        )


def test_dockerfile_entrypoint_is_the_runner() -> None:
    df = _dockerfile()
    assert "tools/synthdetect_infer.py" in df
    assert re.search(r'^ENTRYPOINT \[.*synthdetect_infer\.py.*\]', df, flags=re.MULTILINE)
    # No server surface: the eval container runs a one-shot CLI, so it declares
    # no EXPOSE/HEALTHCHECK directive (match the directive at line start, not the
    # words where they appear in an explanatory comment).
    assert not re.search(r"^EXPOSE\b", df, flags=re.MULTILINE)
    assert not re.search(r"^HEALTHCHECK\b", df, flags=re.MULTILINE)


def _requirement_pin(name: str) -> str | None:
    for line in REQUIREMENTS.read_text().splitlines():
        line = line.strip()
        m = re.match(rf"^{re.escape(name)}==(\S+)$", line)
        if m:
            return m.group(1)
    return None


def test_requirements_pins_mirror_provenance_runtime() -> None:
    # The reviewable pin set (requirements.eval.txt) and the recorded runtime
    # (provenance.eval.json) must agree, so neither drifts silently.
    prov = _provenance()["runtime"]
    for req_name, prov_key in (
        ("numpy", "numpy"),
        ("omegaconf", "omegaconf"),
        ("hydra-core", "hydra_core"),
        ("soundfile", "soundfile"),
        ("soxr", "soxr"),
        ("scipy", "scipy"),
    ):
        assert _requirement_pin(req_name) == prov[prov_key], (
            f"requirements.eval.txt {req_name} drifted from provenance.eval.json runtime.{prov_key}"
        )


def test_dockerfile_torch_pin_matches_provenance() -> None:
    df = _dockerfile()
    prov = _provenance()["runtime"]
    # torch/torchaudio come from the cu118 index in the Dockerfile, not the
    # requirements file; bind their pins to provenance directly.
    assert f"torch=={prov['torch']}" in df
    assert f"torchaudio=={prov['torchaudio']}" in df
