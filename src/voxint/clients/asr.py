"""HTTP client for the whisper service's v1 transcription contract."""

from pathlib import Path
from typing import Any

from voxint.clients._http import ServiceHttpClient, finite_interval
from voxint.clients.base import TranscriptionResult, TranscriptionSegment
from voxint.clients.errors import ProtocolError


class HttpASRClient(ServiceHttpClient):
    def transcribe(
        self, audio_path: Path, initial_prompt: str | None = None
    ) -> TranscriptionResult:
        payload: dict[str, Any] = {"path": self.relative_path(audio_path)}
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
                segments.append(
                    TranscriptionSegment(
                        start_seconds=start, end_seconds=end, text=text, suspect=suspect
                    )
                )
            language = body["language"]
            if language is not None and not isinstance(language, str):
                raise ProtocolError("language must be a string or null")
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError(f"malformed transcribe response: {exc!r}") from exc
        return TranscriptionResult(segments=tuple(segments), language=language)
