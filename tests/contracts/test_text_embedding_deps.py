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


def test_embedding_space_and_dim_constants() -> None:
    # The stored vector width and the equivalence contract both depend on these;
    # a change is a new embedding space (a visible re-index), never silent drift.
    assert TEXT_EMBEDDING_DIM == 384
    assert EMBEDDING_SPACE, "embedding space id must be non-empty"
    assert MAX_SEQUENCE_TOKENS == 128
