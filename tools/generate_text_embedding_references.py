"""Regenerate the semantic-search equivalence reference vectors (issue #121).

MAINTAINER, DEV-ONLY. Run this in a throwaway ``sentence-transformers``
environment; ``torch`` / ``transformers`` / ``sentence-transformers`` MUST NEVER
enter the app dependency closure. The committed output
(``tests/parity/fixtures/text-embedding/references.json``) is the ground truth
the app's ONNX embedder is measured against by
``tests/parity/test_text_embedding.py`` (per-vector cosine >= 0.9999).

Run it exactly like the one-off used to seed the fixtures, with an ephemeral
overlay env that never touches the project lock::

    uv run --no-project --with "sentence-transformers" --with "numpy<2" \
        python tools/generate_text_embedding_references.py

Regenerate ONLY when the pinned upstream revision or the embedding_space id
changes (that is a visible re-index, not a silent fix). The reference vectors
are stored UN-normalized exactly as sentence-transformers returns them; the
test compares by cosine, which is invariant to the app's L2 normalization.
"""

from __future__ import annotations

import json
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "parity" / "fixtures" / "text-embedding"
FIXTURES_JSON = FIXTURES / "fixtures.json"
REFERENCES_JSON = FIXTURES / "references.json"


def main() -> None:
    from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

    spec = json.loads(FIXTURES_JSON.read_text(encoding="utf-8"))
    items = spec["items"]
    model = SentenceTransformer(spec["model"], revision=spec["upstream_revision"])
    vectors = model.encode(
        [item["text"] for item in items],
        normalize_embeddings=False,
        convert_to_numpy=True,
    )

    references = {
        "generated_by": "tools/generate_text_embedding_references.py",
        "model": spec["model"],
        "upstream_revision": spec["upstream_revision"],
        "embedding_space": spec["embedding_space"],
        "sentence_transformers_version": _st_version(),
        "normalized": False,
        "vectors": {
            item["id"]: [round(float(x), 7) for x in vec]
            for item, vec in zip(items, vectors, strict=True)
        },
    }
    REFERENCES_JSON.write_text(
        json.dumps(references, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(items)} reference vectors -> {REFERENCES_JSON}")


def _st_version() -> str:
    try:
        import sentence_transformers

        return str(sentence_transformers.__version__)
    except Exception:  # pragma: no cover - best-effort provenance
        return "unknown"


if __name__ == "__main__":
    main()
