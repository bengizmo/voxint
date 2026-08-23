"""HTTP client for the pyannote service's v1 diarization contract."""

import math
from pathlib import Path

from voxint.clients._http import ServiceHttpClient, finite_interval
from voxint.clients.base import DiarizationResult, DiarizationTurn
from voxint.clients.errors import ProtocolError


class HttpDiarizerClient(ServiceHttpClient):
    def diarize(
        self,
        audio_path: Path,
        *,
        max_speakers: int | None = None,
        num_speakers: int | None = None,
    ) -> DiarizationResult:
        payload: dict[str, object] = {"path": self.relative_path(audio_path)}
        # The pyannote request model has no exact-count field: an exact count is
        # expressed by pinning both bounds (min == max forces that many
        # clusters). A bound sets only max_speakers and leaves min at the
        # service default. num_speakers wins when both are supplied.
        if num_speakers is not None:
            payload["min_speakers"] = num_speakers
            payload["max_speakers"] = num_speakers
        elif max_speakers is not None:
            payload["max_speakers"] = max_speakers
        body = self.post_json("/v1/diarize", payload)
        try:
            turns = []
            for turn in body["turns"]:
                start, end = finite_interval(turn["start_seconds"], turn["end_seconds"])
                label = turn["label"]
                overlap = turn.get("overlap", False)
                overlap_seconds = turn.get("overlap_seconds", 0.0)
                if not isinstance(label, str) or not label:
                    raise ProtocolError("turn label must be a non-empty string")
                if not isinstance(overlap, bool):
                    raise ProtocolError("turn overlap must be a boolean")
                if (
                    isinstance(overlap_seconds, bool)
                    or not isinstance(overlap_seconds, int | float)
                    or not math.isfinite(float(overlap_seconds))
                    or float(overlap_seconds) < 0
                ):
                    raise ProtocolError("turn overlap_seconds must be finite and >= 0")
                turns.append(
                    DiarizationTurn(
                        start_seconds=start,
                        end_seconds=end,
                        label=label,
                        overlap=overlap,
                        overlap_seconds=float(overlap_seconds),
                    )
                )
            reported = body.get("num_speakers")
            if reported is not None and (
                isinstance(reported, bool) or not isinstance(reported, int)
            ):
                raise ProtocolError("num_speakers must be an integer")
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError(f"malformed diarize response: {exc!r}") from exc
        return DiarizationResult(turns=tuple(turns), num_speakers=reported)
