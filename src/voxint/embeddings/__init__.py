"""Transcript text embedding for the semantic-search spine (issue #121).

An additive subsystem: it reads finished, resolved transcript text and produces
384-dim MiniLM sentence embeddings for cross-corpus semantic retrieval. It runs
the model as a vendored ONNX graph via ``onnxruntime`` and never pulls
``torch`` / ``transformers`` / ``sentence-transformers`` into the app
dependency closure. It does not touch ASR / diarization / TitaNet, so it does
not trip the numerics parity gate.
"""

from voxint.embeddings.onnx_embedder import (
    EMBEDDING_SPACE,
    MAX_SEQUENCE_TOKENS,
    TextEmbedder,
    get_text_embedder,
)

__all__ = [
    "EMBEDDING_SPACE",
    "MAX_SEQUENCE_TOKENS",
    "TextEmbedder",
    "get_text_embedder",
]
