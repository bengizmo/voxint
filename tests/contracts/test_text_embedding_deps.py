"""Dependency-closure + constant contracts for the text embedder (issue #121).

Transcript semantic search runs ``paraphrase-multilingual-MiniLM-L12-v2`` as a
vendored ONNX graph. The load-bearing decision is that it does this WITHOUT
``torch`` / ``transformers`` / ``sentence-transformers``: the app ships
``onnxruntime`` + the Rust ``tokenizers`` lib only, and the "NO torch/TF/CUDA"
boundary stays intact. The reference vectors for the equivalence gate are
generated once in a throwaway dev-only ``sentence-transformers`` environment
that is NOT part of this project's lock, so none of those heavy packages may
appear anywhere in ``uv.lock`` (not in the default closure, an extra, or dev).

These are pure file/constant checks (no DB); the measured ONNX-vs-reference
equivalence lives in ``tests/parity/test_text_embedding.py``.
"""

import json
import re
import tomllib

from tests.contracts.conftest import REPO_ROOT
from voxint.db.models import TEXT_EMBEDDING_DIM
from voxint.embeddings.onnx_embedder import EMBEDDING_SPACE, MAX_SEQUENCE_TOKENS

# The embedder must ship in the DEFAULT install (main deps), or semantic search
# would silently be an opt-in extra.
REQUIRED_MAIN_DEPS = ("onnxruntime", "tokenizers")

# None of these may resolve anywhere in the lock — a single one implies the
# heavy ML stack leaked into an install path it must never enter.
FORBIDDEN_LOCK_PACKAGES = (
    "torch",
    "transformers",
    "sentence-transformers",
    "tensorflow",
    "triton",
)
FORBIDDEN_LOCK_PREFIXES = ("nvidia-", "cuda-")


def _main_dependencies() -> list[str]:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    return list(data["project"]["dependencies"])


def _dist_name(spec: str) -> str:
    return re.split(r"[<>=!~ \[]", spec, maxsplit=1)[0].strip().lower()


def test_embedder_runtime_ships_in_the_default_closure() -> None:
    names = {_dist_name(spec) for spec in _main_dependencies()}
    for dep in REQUIRED_MAIN_DEPS:
        assert dep in names, (
            f"{dep!r} must be a MAIN dependency so the embedder ships by default"
        )


def test_no_torch_stack_anywhere_in_the_lock() -> None:
    lock = (REPO_ROOT / "uv.lock").read_text()
    locked = set(re.findall(r'^name = "([^"]+)"', lock, flags=re.MULTILINE))
    for pkg in FORBIDDEN_LOCK_PACKAGES:
        assert pkg not in locked, (
            f"{pkg!r} resolved in uv.lock — the embedder must stay torch-free;"
            " reference vectors are generated in a throwaway dev-only env, never"
            " a locked dependency"
        )
    leaked_prefixed = sorted(
        name
        for name in locked
        for prefix in FORBIDDEN_LOCK_PREFIXES
        if name.startswith(prefix)
    )
    assert not leaked_prefixed, f"GPU/CUDA packages leaked into uv.lock: {leaked_prefixed}"


def _provenance() -> dict:
    return json.loads(
        (REPO_ROOT / "src" / "voxint" / "embeddings" / "models" / "provenance.json").read_text()
    )


def test_embedding_space_and_dim_constants() -> None:
    # The stored vector width and the equivalence contract both depend on these;
    # a change is a new embedding space (a visible re-index), never silent drift.
    # Bound to provenance, not bare literals, so the module constants and the
    # committed provenance can never diverge unnoticed.
    provenance = _provenance()
    assert TEXT_EMBEDDING_DIM == 384 == provenance["embedding_dim"]
    assert MAX_SEQUENCE_TOKENS == 128 == provenance["max_sequence_tokens"]
    assert EMBEDDING_SPACE, "embedding space id must be non-empty"
    assert provenance["embedding_space"] == EMBEDDING_SPACE


def test_app_dockerfile_minilm_shas_match_provenance() -> None:
    # The app image bakes the vendored MiniLM weights behind a build-time
    # sha256 gate (issue #121). Drift between the Dockerfile's ARG defaults and
    # the committed provenance would let a differently-exported graph ship
    # silently and invalidate the measured equivalence gate the embedding space
    # was pinned against. Mirrors the titanet/pyannote/gguf provenance gates in
    # tests/contracts/test_service_logic.py.
    dockerfile = (REPO_ROOT / "Dockerfile").read_text()
    provenance = _provenance()
    for arg, filename in (
        ("MINILM_ONNX_SHA256", "model.onnx"),
        ("MINILM_TOKENIZER_SHA256", "tokenizer.json"),
    ):
        # Anchored + exactly-one: a commented-out or stale duplicate ARG must not
        # satisfy the gate in place of the effective default.
        matches = re.findall(rf"^ARG {arg}=([0-9a-f]{{64}})$", dockerfile, flags=re.MULTILINE)
        assert len(matches) == 1, (
            f"Dockerfile must declare {arg} exactly once (found {len(matches)})"
        )
        assert matches[0] == provenance["files"][filename]["sha256"], (
            f"Dockerfile {arg} drifted from src/voxint/embeddings/models/provenance.json"
        )


def test_release_workflow_minilm_asset_matches_provenance() -> None:
    # The release workflow's MINILM_ONNX_RELEASE and provenance's asset_release
    # name the same immutable GitHub release; a refresh that bumps one but not
    # the other would fail late at `gh release download` in CI. Pin them here so
    # the drift is caught at test time instead.
    release_yml = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text()
    match = re.search(r"^\s*MINILM_ONNX_RELEASE:\s*(\S+)\s*$", release_yml, flags=re.MULTILINE)
    assert match is not None, "release.yml lost its MINILM_ONNX_RELEASE env"
    assert match.group(1) == _provenance()["asset_release"], (
        "release.yml MINILM_ONNX_RELEASE drifted from provenance.json asset_release"
    )
