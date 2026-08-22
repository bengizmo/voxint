"""Turn a resolved transcript into embeddable chunks (issue #121).

The chunk grain is the reading paragraph (``paragraphize_transcript``): one
speaker's consecutive lines merged, which is also the unit the read mode and
Markdown export use, so a semantic hit lines up with what the operator sees. A
paragraph longer than the model's 128-token contract is split deterministically
into token-bounded sub-chunks with a modest word overlap — long monologues must
never be silently truncated (the tail is exactly the passage someone is
searching for).

Pure and DB-free: it takes already-resolved paragraphs plus a token counter and
returns chunk rows. The producer supplies the counter (the real tokenizer) and
the DB-derived paragraphs; tests supply a fake counter and hand-built
paragraphs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from voxint.embeddings.onnx_embedder import MAX_SEQUENCE_TOKENS

# Token budget for a single chunk, INCLUDING the tokenizer's special tokens
# (the embedder truncates the encoded sequence to MAX_SEQUENCE_TOKENS, specials
# included, so the counter must too). Overlap carries continuity across a split.
DEFAULT_MAX_TOKENS = MAX_SEQUENCE_TOKENS
DEFAULT_OVERLAP_TOKENS = 16

TokenCounter = Callable[[str], int]


@dataclass(frozen=True)
class ParagraphInput:
    """A resolved reading paragraph plus the rendering its text came from."""

    speaker: str
    start_seconds: float
    end_seconds: float
    text: str
    # Dominant rendering of the constituent segments: corrected / enhanced / raw.
    text_rendering: str


@dataclass(frozen=True)
class TranscriptChunk:
    """One embeddable chunk, ready to become a ``SegmentEmbedding`` row."""

    chunk_index: int
    start_seconds: float
    end_seconds: float
    speaker_label: str | None
    text_rendering: str
    text: str


def chunk_transcript(
    paragraphs: Sequence[ParagraphInput],
    count_tokens: TokenCounter,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[TranscriptChunk]:
    """Chunk resolved paragraphs into token-bounded, contiguously-indexed rows.

    ``chunk_index`` is a run-wide monotonic 0-based counter (never per
    paragraph), so a split paragraph's pieces keep transcript order across the
    whole run. Blank paragraphs are dropped (nothing to embed). Every returned
    chunk satisfies ``count_tokens(text) <= max_tokens`` except the pathological
    case of a single whitespace-free token longer than the budget, which is
    emitted whole (the embedder truncates it) rather than dropped.
    """
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be < max_tokens")

    chunks: list[TranscriptChunk] = []
    index = 0
    for para in paragraphs:
        for piece in _split_to_budget(
            para.text, count_tokens, max_tokens, overlap_tokens
        ):
            chunks.append(
                TranscriptChunk(
                    chunk_index=index,
                    start_seconds=para.start_seconds,
                    end_seconds=para.end_seconds,
                    speaker_label=para.speaker or None,
                    text_rendering=para.text_rendering,
                    text=piece,
                )
            )
            index += 1
    return chunks


def _split_to_budget(
    text: str,
    count_tokens: TokenCounter,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """Split one paragraph into <=max_tokens word-runs with word overlap.

    Greedy word packing: fill a chunk until the next word would blow the budget,
    flush it, then seed the next chunk with a deterministic tail of the flushed
    words (~overlap_tokens) so meaning that straddles the cut is embedded on both
    sides. Deterministic given the same text + counter.
    """
    stripped = text.strip()
    if not stripped:
        return []
    if count_tokens(stripped) <= max_tokens:
        return [stripped]

    words = stripped.split()
    chunks: list[str] = []
    current: list[str] = []

    for word in words:
        candidate = [*current, word]
        if count_tokens(" ".join(candidate)) <= max_tokens:
            current = candidate
            continue
        if current:
            chunks.append(" ".join(current))
            current = [*_overlap_tail(current, count_tokens, overlap_tokens), word]
            # A single word plus overlap can itself exceed the budget; if so,
            # drop the overlap so the word still lands in a valid chunk.
            if count_tokens(" ".join(current)) > max_tokens:
                current = [word]
        else:
            # A lone word longer than the whole budget (no whitespace to split
            # on). Emit it whole — the embedder truncates it; dropping it would
            # be the silent-truncation bug this function exists to prevent.
            chunks.append(word)
            current = []

    if current:
        chunks.append(" ".join(current))
    return chunks


def _overlap_tail(
    words: list[str], count_tokens: TokenCounter, overlap_tokens: int
) -> list[str]:
    """The longest suffix of ``words`` whose token count is <= overlap_tokens."""
    if overlap_tokens <= 0:
        return []
    tail: list[str] = []
    for word in reversed(words):
        candidate = [word, *tail]
        if count_tokens(" ".join(candidate)) > overlap_tokens:
            break
        tail = candidate
    return tail
