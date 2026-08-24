"""Contract tests for the synthdetect eval container (#144, M1 S2).

CPU-only, no torch/fairseq/GPU. These bind the eval-container spec
(``services/synthdetect/Dockerfile.eval``, ``requirements.eval.txt``,
``provenance.eval.json``) to the pins-as-data registry, the runner's constants,
and the host scorer, so a drift between them is caught at test time instead of
at a GPU build in S2b. They mirror the titanet/pyannote/MiniLM provenance gates
in ``tests/contracts/test_service_logic.py`` and ``test_text_embedding_deps.py``.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SVC = REPO / "services" / "synthdetect"
DOCKERFILE = SVC / "Dockerfile.eval"
REQUIREMENTS = SVC / "requirements.eval.txt"
PROVENANCE = SVC / "provenance.eval.json"
VENDOR = REPO / "tools" / "synthdetect_vendor"
VENDOR_MODEL = VENDOR / "ssl_antispoofing_model.py"
VENDOR_PROVENANCE = VENDOR / "provenance.json"

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


def test_pinned_state_is_coherent() -> None:
    # S2b (2026-08-24) froze the weights from real bytes: the registry is pinned,
    # provenance carries the frozen shas/sizes, the model-repo commit, the fairseq
    # commit, and the base-image digest together, and they agree with the registry.
    # Both pinned_unqualified (frozen shas only) and qualified (GPU determinism +
    # smoke evidence) are coherent pinned states here; the qualified promotion is
    # bound to its evidence separately by test_qualified_state_binds_its_evidence.
    prov = _provenance()
    model = default_model()
    assert model.weights_pinned() is True
    assert prov["weights"]["qualification_state"] in ("pinned_unqualified", "qualified")
    for w in model.weights:
        f = prov["weights"]["files"][w.filename]
        assert f["sha256"] is not None and f["sha256"] == w.sha256
        assert f["size_bytes"] is not None and f["size_bytes"] == w.size_bytes
    assert prov["runtime"]["fairseq_commit"] is not None
    assert prov["base_image_digest"] is not None
    assert prov["model_repo"]["commit"] is not None
    assert prov["model_repo"]["commit"] == model.commit


def test_vendored_model_bytes_match_provenance() -> None:
    # The vendored ssl_antispoofing_model.py IS the numerics-defining model
    # definition (loaded by file path at eval time), so a silent edit would change
    # inference numerics while every pin still looked frozen. Both provenance files
    # claim a contract test binds this hash; this is that test. Byte-identity to the
    # pinned upstream commit is the integrity check.
    vp = json.loads(VENDOR_PROVENANCE.read_text())["files"]["ssl_antispoofing_model.py"]
    digest = hashlib.sha256(VENDOR_MODEL.read_bytes()).hexdigest()
    assert digest == vp["sha256"], (
        f"vendored ssl_antispoofing_model.py sha256 {digest} drifted from the pinned "
        f"{vp['sha256']}; a change to the numerics-defining model must re-freeze provenance"
    )


def test_vendored_model_commit_matches_both_registries() -> None:
    # The vendored file's upstream_commit must equal the model-repo commit recorded
    # in BOTH the pins-as-data registry and provenance.eval.json, so the three
    # never drift apart.
    vp = json.loads(VENDOR_PROVENANCE.read_text())["files"]["ssl_antispoofing_model.py"]
    model = default_model()
    prov = _provenance()
    assert vp["upstream_commit"] == model.commit
    assert vp["upstream_commit"] == prov["model_repo"]["commit"]


def test_qualified_state_binds_its_evidence() -> None:
    # 'qualified' is a determinism + smoke claim, not merely frozen shas (that is
    # pinned_unqualified). When provenance declares qualified, the dated evidence
    # reports it rests on must be listed and present; otherwise the only other
    # allowed pinned state is pinned_unqualified. This stops a silent promotion to
    # qualified (or a downgrade that keeps the qualified label) from passing CI.
    prov = _provenance()
    state = prov["weights"]["qualification_state"]
    assert state in ("pinned_unqualified", "qualified")
    if state == "qualified":
        reports = prov["weights"].get("qualification_evidence", {}).get("reports", [])
        assert reports, "qualified state must list weights.qualification_evidence.reports"
        for rel in reports:
            assert (REPO / rel).is_file(), f"qualified evidence report missing: {rel}"


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


def test_dockerfile_pins_frozen_fairseq_commit() -> None:
    # S2b froze FAIRSEQ_COMMIT: the ARG carries the pinned commit as its default
    # (its value is bound to provenance by
    # test_frozen_digests_bind_the_dockerfile_when_present), and the non-empty
    # guard remains so an explicitly-blank override still fails the build, keeping
    # an unpinned runtime unbuildable.
    df = _dockerfile()
    commit = _provenance()["runtime"]["fairseq_commit"]
    assert commit is not None
    assert re.search(rf"^ARG FAIRSEQ_COMMIT={re.escape(commit)}\s*$", df, flags=re.MULTILINE), (
        "Dockerfile.eval must default `ARG FAIRSEQ_COMMIT` to the frozen commit"
    )
    assert 'test -n "${FAIRSEQ_COMMIT}"' in df


def test_dockerfile_base_image_matches_provenance() -> None:
    df = _dockerfile()
    prov = _provenance()
    match = re.search(r"^FROM (\S+)", df, flags=re.MULTILINE)
    assert match is not None
    ref = match.group(1)
    # After the S2b freeze FROM carries an @sha256 digest; the tag portion must
    # still be the recorded base_image, and the full ref its base_image@digest.
    assert ref.split("@", 1)[0] == prov["base_image"]
    digest = prov.get("base_image_digest")
    if digest is not None:
        assert ref == f"{prov['base_image']}@{digest}"


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


def _pins(text: str) -> dict[str, str]:
    """Every ``name==version`` pin in a file, keyed by name (case-insensitive)."""
    out: dict[str, str] = {}
    for m in re.finditer(r"([A-Za-z0-9_.-]+)==([0-9][0-9A-Za-z.+-]*)", text):
        out.setdefault(m.group(1).lower(), m.group(2))
    return out


def test_every_requirements_pin_is_recorded_in_provenance() -> None:
    # EVERY pin in requirements.eval.txt (not a hand-picked subset) must appear in
    # provenance.eval.json runtime with the same version, so no dependency drifts
    # silently between the reviewable pin set and the recorded runtime.
    prov = {k.lower(): v for k, v in _provenance()["runtime"].items() if isinstance(v, str)}
    for name, version in _pins(REQUIREMENTS.read_text()).items():
        assert prov.get(name) == version, (
            f"requirements.eval.txt {name}=={version} not recorded in provenance.eval.json runtime"
        )


def test_every_dockerfile_package_pin_is_recorded_in_provenance() -> None:
    # Same for the pins the Dockerfile installs directly (torch/torchaudio from the
    # cu118 index, cython, and the fairseq runtime companions): each must match
    # provenance. pip/setuptools/wheel are build tooling, not runtime, so they are
    # exempted here.
    prov = {k.lower(): v for k, v in _provenance()["runtime"].items() if isinstance(v, str)}
    exempt = {"pip", "setuptools", "wheel"}
    for name, version in _pins(_dockerfile()).items():
        if name in exempt:
            continue
        assert prov.get(name) == version, (
            f"Dockerfile.eval {name}=={version} not recorded in provenance.eval.json runtime"
        )


def test_runner_is_py310_safe() -> None:
    # The eval image is nvidia/cuda:*-ubuntu22.04 (system Python 3.10). A 3.11+-only
    # construct in the runner or its container-time imports crash-loops the runner
    # at execution: S2b's GPU smoke first failed because `_utcnow` used
    # `from datetime import UTC` (3.11+). The host test runner is newer, so importing
    # the module never catches it; guard the source statically. When a new 3.11+-only
    # name bites, add it here. (The scorer synthdetect_eval.py runs on the host, not
    # in the container, so it is not covered.)
    banned = {
        "from datetime import UTC": "datetime.UTC is 3.11+; use timezone.utc",
        "datetime.UTC": "datetime.UTC is 3.11+; use timezone.utc",
    }
    runtime_sources = [
        REPO / "tools" / "synthdetect_infer.py",
        REPO / "tools" / "synthdetect_corpus.py",
        REPO / "tools" / "synthdetect_sources.py",
        REPO / "tools" / "synthdetect_vendor" / "ssl_antispoofing_model.py",
    ]
    offenders = [
        f"{path.relative_to(REPO)}: {needle!r} ({reason})"
        for path in runtime_sources
        for needle, reason in banned.items()
        if needle in path.read_text()
    ]
    assert not offenders, (
        "synthdetect eval-container code must stay Python 3.10-compatible:\n"
        + "\n".join(offenders)
    )


def test_frozen_digests_bind_the_dockerfile_when_present() -> None:
    # In S2a the base-image digest and fairseq commit are CANDIDATE (null) and the
    # Dockerfile carries a mutable tag + a no-default ARG. When S2b freezes them,
    # this binds the frozen values into the Dockerfile so the image cannot claim
    # provenance for different bytes: a pinned digest must appear in FROM, and a
    # pinned fairseq commit must become the ARG default.
    df = _dockerfile()
    prov = _provenance()
    digest = prov["base_image_digest"]
    if digest is not None:
        assert re.search(rf"^FROM \S+@{re.escape(digest)}", df, flags=re.MULTILINE), (
            "base_image_digest is frozen but the Dockerfile FROM does not pin it"
        )
    commit = prov["runtime"]["fairseq_commit"]
    if commit is not None:
        assert re.search(rf"^ARG FAIRSEQ_COMMIT={re.escape(commit)}\s*$", df, flags=re.MULTILINE), (
            "fairseq_commit is frozen but the Dockerfile ARG default does not pin it"
        )
