"""HTTP client for the titanet service's v1 embedding contract."""

import math
from pathlib import Path
from typing import Any

from voxint.clients._http import ServiceHttpClient
from voxint.clients.base import EmbeddingEntry, EmbeddingResult
from voxint.clients.errors import ProtocolError
from voxint.db.models import EMBEDDING_DIM

# Contract cap on windows per request; larger jobs are batched transparently.
MAX_WINDOWS_PER_REQUEST = 512


class HttpEmbedderClient(ServiceHttpClient):
    def embed(
        self, audio_path: Path, windows: tuple[tuple[float, float], ...]
    ) -> EmbeddingResult:
        # The contract rejects an empty windows list; calling with none is a
        # caller bug, not something to paper over with a fabricated response.
        if not windows:
            raise ValueError("embed() requires at least one window")
        rel = self.relative_path(audio_path)
        entries: list[EmbeddingEntry] = []
        space: str | None = None
        for offset in range(0, len(windows), MAX_WINDOWS_PER_REQUEST):
            batch = windows[offset : offset + MAX_WINDOWS_PER_REQUEST]
            body = self.post_json(
                "/v1/embed",
                {
                    "path": rel,
                    "windows": [
                        {"start_seconds": start, "end_seconds": end} for start, end in batch
                    ],
                },
            )
            batch_space, batch_entries = _parse_embed_response(body, expected=len(batch))
            if space is None:
                space = batch_space
            elif batch_space != space:
                raise ProtocolError(
                    f"embedding_space changed across batches: {space!r} != {batch_space!r}"
                )
            entries.extend(batch_entries)
        if space is None:  # unreachable: len(windows) >= 1 → at least one batch ran
            raise ProtocolError("no embed batches produced an embedding_space")
        return EmbeddingResult(embedding_space=space, entries=tuple(entries))


def _parse_embed_response(
    body: dict[str, Any], *, expected: int
) -> tuple[str, list[EmbeddingEntry]]:
    try:
        space = body["embedding_space"]
        if not isinstance(space, str) or not space:
            raise ProtocolError("embedding_space must be a non-empty string")
        results = body["results"]
        if len(results) != expected:
            raise ProtocolError(
                f"expected {expected} window results, got {len(results)}"
            )
        entries: list[EmbeddingEntry] = []
        for row in results:
            entries.append(_parse_window_result(row))
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError(f"malformed embed response: {exc!r}") from exc
    return space, entries


def _parse_window_result(row: dict[str, Any]) -> EmbeddingEntry:
    vector = row["embedding"]
    skip_reason = row.get("skip_reason")
    snr_db = row.get("snr_db")
    if (vector is None) == (skip_reason is None):
        raise ProtocolError("exactly one of embedding / skip_reason must be set per window")
    if skip_reason is not None and skip_reason not in ("too_short", "low_snr"):
        raise ProtocolError(f"unknown skip_reason {skip_reason!r}")
    if snr_db is not None:
        if isinstance(snr_db, bool) or not isinstance(snr_db, int | float):
            raise ProtocolError("snr_db must be a number or null")
        snr_db = float(snr_db)
        if not math.isfinite(snr_db):
            raise ProtocolError("snr_db must be finite")
    embedding: tuple[float, ...] | None = None
    if vector is not None:
        if len(vector) != EMBEDDING_DIM:
            raise ProtocolError(
                f"embedding has {len(vector)} dimensions, expected {EMBEDDING_DIM}"
            )
        embedding = tuple(float(x) for x in vector)
        if not all(math.isfinite(x) for x in embedding):
            raise ProtocolError("embedding values must be finite")
    return EmbeddingEntry(embedding=embedding, snr_db=snr_db, skip_reason=skip_reason)
