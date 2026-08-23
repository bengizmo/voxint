"""HTTP client for the whisper service's v1 transcription contract."""

import math
from pathlib import Path
from typing import Any

from voxint.clients._http import ServiceHttpClient, finite_interval
from voxint.clients.base import (
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionWord,
)
from voxint.clients.errors import ProtocolError


def _parse_unit_interval(raw: Any, field: str) -> float | None:
    """A missing/null value stays None; anything present must be a finite
    number in [0, 1]. We validate rather than clamp so a malformed value is a
    loud ProtocolError, never a silently-massaged score."""
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ProtocolError(f"{field} must be a number or null")
    value = float(raw)
    if not math.isfinite(value) or not (0.0 <= value <= 1.0):
        raise ProtocolError(f"{field} must be finite in [0, 1]")
    return value


def _parse_confidence(raw: Any) -> float | None:
    """Segment/word confidence: exp(avg_logprob), already clamped by the
    whisper service."""
    return _parse_unit_interval(raw, "segment confidence")


class HttpASRClient(ServiceHttpClient):
    def transcribe(
        self, audio_path: Path, initial_prompt: str | None = None
    ) -> TranscriptionResult:
        # language: null opts into the contract's documented auto-detect path
        # (#124). Sent explicitly because the v1 OMITTED-field default is "en" —
        # that default is contract for other callers and stays untouched.
        payload: dict[str, Any] = {
            "path": self.relative_path(audio_path),
            "language": None,
        }
        # Only send initial_prompt when non-empty: the whisper contract treats a
        # missing key as "no bias", and an empty string would be a wasted field.
        if initial_prompt:
            payload["initial_prompt"] = initial_prompt
        body = self.post_json("/v1/transcribe", payload)
        try:
            segments = []
            for seg in body["segments"]:
                start, end = finite_interval(
                    seg["start_seconds"], seg["end_seconds"], zero_length_ok=True
                )
                text, suspect = seg["text"], seg.get("suspect", False)
                if not isinstance(text, str) or not isinstance(suspect, bool):
                    raise ProtocolError("segment text/suspect have wrong types")
                confidence = _parse_confidence(seg.get("confidence"))
                segments.append(
                    TranscriptionSegment(
                        start_seconds=start,
                        end_seconds=end,
                        text=text,
                        suspect=suspect,
                        confidence=confidence,
                    )
                )
            language = body["language"]
            if language is not None and not isinstance(language, str):
                raise ProtocolError("language must be a string or null")
            # Absent key = an older service that predates the field (#124);
            # present-but-malformed is loud, mirroring segment confidence.
            language_probability = _parse_unit_interval(
                body.get("language_probability"), "language_probability"
            )
            # Word timings (word_timestamps=True) are a flat, run-level list in
            # the v1 contract. A service or fake that OMITS the key predates #59;
            # treat that as "no word data". A present-but-null value is not
            # back-compat — it's a malformed current response, so it must be loud.
            words = _parse_words(body["words"]) if "words" in body else ()
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError(f"malformed transcribe response: {exc!r}") from exc
        return TranscriptionResult(
            segments=tuple(segments),
            language=language,
            language_probability=language_probability,
            words=words,
        )


def _parse_words(raw: Any) -> tuple[TranscriptionWord, ...]:
    """Parse the response's flat ``words`` list, validating each as strictly as
    segments. The caller only reaches here when the ``words`` key is present, so
    anything but a list (``null`` included — the v1 contract types it
    ``list[Word]``) is a loud ProtocolError, never silently dropped."""
    if not isinstance(raw, list):
        raise ProtocolError("words must be a list")
    words: list[TranscriptionWord] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ProtocolError("each word must be an object")
        start, end = finite_interval(
            entry["start_seconds"], entry["end_seconds"], zero_length_ok=True
        )
        token = entry["word"]
        # Empty tokens would render as zero-width, unclickable split targets;
        # whisper never emits them. Leading/trailing spaces ARE kept — faster-
        # whisper attaches them to word boundaries and they matter for text.
        if not isinstance(token, str) or token == "":
            raise ProtocolError("word text must be a non-empty string")
        words.append(
            TranscriptionWord(
                start_seconds=start,
                end_seconds=end,
                word=token,
                confidence=_parse_confidence(entry.get("confidence")),
            )
        )
    return tuple(words)
