"""In-process MiniLM text embedder — the semantic-search runtime (issue #121).

Runs ``sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`` as a
vendored ONNX graph via ``onnxruntime``, reproducing the sentence-transformer
contract in numpy:

    tokenize (max 128 tokens, special tokens, dynamic right-pad)
      -> ONNX BERT backbone forward (``last_hidden_state``)
      -> attention-mask MEAN pool
      -> L2 normalize

The shipped ``onnx/model.onnx`` is the BERT backbone only (its single output is
``last_hidden_state``; pooling is NOT baked in), so pooling and normalization
happen here. The mandatory equivalence contract test
(``tests/parity/test_text_embedding.py``) measures per-vector cosine >= 0.9999
against reference vectors generated once in a dev-only ``sentence-transformers``
environment; that test — not this reasoning — is what guards against silent
tokenizer / mask / pooling / normalize drift.

``torch`` / ``transformers`` / ``sentence-transformers`` MUST NEVER enter the
app dependency closure. Only ``onnxruntime`` + the Rust ``tokenizers`` lib +
numpy are used here.

Weights are vendored, sha-pinned, and not in git (weights doctrine): the app
image bakes ``model.onnx`` + ``tokenizer.json`` under ``/app/models/minilm``;
``src/voxint/embeddings/models/provenance.json`` records per-file sha256 +
upstream revision + license. ``VOXINT_MINILM_ONNX_PATH`` /
``VOXINT_MINILM_TOKENIZER_PATH`` override the baked paths (the maintainer dev +
equivalence gate point them at ``vendor/minilm``).
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from voxint.db.models import TEXT_EMBEDDING_DIM

if TYPE_CHECKING:  # pragma: no cover - typing only
    import onnxruntime as ort
    from tokenizers import Tokenizer

# The embedding-space id encodes model revision + graph variant + tokenizer +
# pooling + max-len. A model or pooling change = a NEW space id = a visible
# re-index, never silent drift. Stored on every row and filtered on at query
# time; cosine is only ever valid within one space.
EMBEDDING_SPACE = "minilm-multi-l12-onnx-fp32-mean-v1"

# The sentence-transformer sequence contract for this model
# (sentence_bert_config.json: max_seq_length=128). The tokenizer's own config
# advertises 512 — that is the backbone position limit, NOT the ST contract.
# Embedding past 128 tokens would diverge from every reference vector.
MAX_SEQUENCE_TOKENS = 128

# Baked image paths; overridable for dev / tests.
_DEFAULT_ONNX_PATH = "/app/models/minilm/model.onnx"
_DEFAULT_TOKENIZER_PATH = "/app/models/minilm/tokenizer.json"


def _onnx_path() -> str:
    return os.getenv("VOXINT_MINILM_ONNX_PATH", _DEFAULT_ONNX_PATH)


def _tokenizer_path() -> str:
    return os.getenv("VOXINT_MINILM_TOKENIZER_PATH", _DEFAULT_TOKENIZER_PATH)


def minilm_artifacts_available() -> bool:
    """True when both vendored MiniLM files are present at the resolved paths.

    A cheap file-existence probe (no onnxruntime/tokenizer load) so callers can
    decide whether embedding is possible before committing to it. The finalize
    hook uses it to skip enqueueing a doomed job on an install where the
    ``minilm-onnx-v1`` asset was never fetched (native, no-Docker), and the
    native ``doctor`` uses it as a preflight check. This is advisory: a file
    can vanish or be corrupt between the probe and the load, so the embedder
    still validates at construction and the job lane still fails honestly.
    """
    return Path(_onnx_path()).exists() and Path(_tokenizer_path()).exists()


class TextEmbedder:
    """A loaded ONNX session + tokenizer that turns text into unit vectors.

    Prefer the process-wide :func:`get_text_embedder` singleton; construct this
    directly only in tests. Loading is done eagerly in ``__init__`` so a missing
    or corrupt weight file fails at construction, not on the first embed.
    """

    def __init__(
        self, onnx_path: str | None = None, tokenizer_path: str | None = None
    ) -> None:
        # Imported lazily so importing this module (e.g. for EMBEDDING_SPACE)
        # never forces onnxruntime/tokenizers to load.
        import onnxruntime as ort
        from tokenizers import Tokenizer

        resolved_onnx = onnx_path or _onnx_path()
        resolved_tok = tokenizer_path or _tokenizer_path()
        if not Path(resolved_onnx).exists():
            raise FileNotFoundError(
                f"MiniLM ONNX weights not found: {resolved_onnx} "
                "(set VOXINT_MINILM_ONNX_PATH, or fetch the minilm-onnx-v1 asset)"
            )
        if not Path(resolved_tok).exists():
            raise FileNotFoundError(
                f"MiniLM tokenizer not found: {resolved_tok} "
                "(set VOXINT_MINILM_TOKENIZER_PATH, or fetch the minilm-onnx-v1 asset)"
            )

        self.onnx_path = resolved_onnx
        self.tokenizer_path = resolved_tok

        self._tokenizer: Tokenizer = Tokenizer.from_file(resolved_tok)
        # Deterministic truncation to the ST contract + dynamic right-pad so a
        # batch is one rectangular tensor. Padding positions are masked out of
        # the mean pool (attention_mask=0), so the pad token value never affects
        # a pooled vector; we still pin the tokenizer's own pad token for a
        # well-formed tensor.
        self._tokenizer.enable_truncation(max_length=MAX_SEQUENCE_TOKENS)
        pad_id = self._tokenizer.token_to_id("<pad>")
        if pad_id is None:
            pad_id = 0
        self._tokenizer.enable_padding(
            pad_id=pad_id, pad_token="<pad>", direction="right"
        )

        # A SECOND tokenizer with truncation and padding explicitly OFF, purely
        # to count a string's true token length for chunking. tokenizer.json
        # itself bakes truncation.max_length=128, so a counter that inherited it
        # would report every over-long paragraph as exactly 128 and silently
        # defeat the splitter — chunking must see the real length to know a
        # paragraph needs splitting at all.
        self._counter: Tokenizer = Tokenizer.from_file(resolved_tok)
        self._counter.no_truncation()
        self._counter.no_padding()

        # CPU EP only: this runs on the API/worker host beside Postgres, not a
        # GPU service. Single-threaded intra-op keeps it a good neighbour; the
        # workload is tiny (a run's paragraphs) and latency-insensitive.
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.getenv("VOXINT_MINILM_ORT_THREADS", "1"))
        self._session: ort.InferenceSession = ort.InferenceSession(
            resolved_onnx, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in self._session.get_inputs()}

    def count_tokens(self, text: str) -> int:
        """True encoded length (special tokens included, NOT truncated).

        The chunker's token budget is measured with this so an over-long
        paragraph is actually split; the embedding tokenizer truncates and would
        report a flat 128.
        """
        return len(self._counter.encode(text).ids)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of strings into L2-normalized 384-dim float32 vectors.

        Returns an ``(len(texts), 384)`` array. An empty input returns an empty
        ``(0, 384)`` array. Padded-batch and single-item inference agree (the
        equivalence gate tests both).
        """
        if not texts:
            return np.zeros((0, TEXT_EMBEDDING_DIM), dtype=np.float32)

        encodings = self._tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

        feeds: dict[str, np.ndarray] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        # This BERT graph takes token_type_ids; single sentences are all zeros.
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.array(
                [e.type_ids for e in encodings], dtype=np.int64
            )

        outputs = self._session.run(["last_hidden_state"], feeds)
        last_hidden_state = np.asarray(outputs[0])

        pooled = _mean_pool(last_hidden_state, attention_mask)
        normalized = _l2_normalize(pooled)

        if normalized.shape[1] != TEXT_EMBEDDING_DIM:
            raise ValueError(
                f"embedder produced dim {normalized.shape[1]}, "
                f"expected {TEXT_EMBEDDING_DIM}"
            )
        if not np.all(np.isfinite(normalized)):
            raise ValueError("embedder produced non-finite values")
        return normalized.astype(np.float32, copy=False)


def _mean_pool(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Attention-mask mean pool: mean of token vectors over real tokens only.

    Mirrors sentence-transformers ``Pooling(pooling_mode_mean_tokens=True)``.
    """
    mask = attention_mask.astype(np.float32)[:, :, None]  # (n, seq, 1)
    summed = np.sum(last_hidden_state.astype(np.float32) * mask, axis=1)  # (n, dim)
    counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)  # (n, 1)
    return summed / counts


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return np.asarray(vectors / norms, dtype=np.float32)


_singleton_lock = threading.Lock()
_singleton: TextEmbedder | None = None


def get_text_embedder() -> TextEmbedder:
    """Return the process-wide embedder, loading it once (lazy singleton).

    Shared by the worker producer AND the API query path: load the ONNX session
    and tokenizer once per process, never twice, and never round-trip through
    Celery to embed a query.
    """
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = TextEmbedder()
    return _singleton


def reset_text_embedder() -> None:
    """Drop the cached singleton (tests only)."""
    global _singleton
    with _singleton_lock:
        _singleton = None
