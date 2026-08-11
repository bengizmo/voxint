"""Gated media serving for the review UI.

A file is served only if it (a) is referenced by the run's DB rows, (b)
resolves to a regular file inside ``media_root`` (symlink escapes rejected
after resolution), and (c) carries a decodable audio stream per ffprobe —
validated once per (path, size, mtime) and cached, with a hard subprocess
timeout so a hung probe can't wedge a request thread.

Range support is the single-range subset browsers actually use: one
``bytes=`` range per request (open-ended and suffix forms included), 206/416
semantics, and Accept-Ranges advertised. Multipart ranges are ignored — the
whole file is returned with 200, which HTTP permits.
"""

import os
import stat as stat_module
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from voxint.media.normalize import NormalizationError, probe_audio


class MediaNotServableError(Exception):
    """The path is missing, escapes the media root, or is not decodable audio."""


@dataclass(frozen=True)
class ByteRange:
    start: int
    end: int  # inclusive, per RFC 9110

    @property
    def length(self) -> int:
        return self.end - self.start + 1


class RangeNotSatisfiableError(Exception):
    """The Range header parsed but selects nothing inside the file."""

    def __init__(self, size: int) -> None:
        super().__init__(f"unsatisfiable range for size {size}")
        self.size = size


def parse_range(header: str | None, size: int) -> ByteRange | None:
    """RFC 9110 single byte range; None = serve the whole file with 200.

    Malformed headers and multipart ranges are treated as absent (the RFC
    allows ignoring Range); a well-formed range beyond EOF raises 416.
    """
    if header is None or size == 0:
        return None
    header = header.strip()
    if not header.startswith("bytes="):
        return None
    spec = header[len("bytes=") :]
    if "," in spec:  # multipart — legal to ignore
        return None
    start_s, sep, end_s = spec.partition("-")
    if not sep:
        return None
    start_s, end_s = start_s.strip(), end_s.strip()
    try:
        if start_s == "":
            # suffix form: last N bytes
            suffix = int(end_s)
            if suffix <= 0:
                raise RangeNotSatisfiableError(size)
            start = max(0, size - suffix)
            return ByteRange(start, size - 1)
        start = int(start_s)
        if start >= size:
            raise RangeNotSatisfiableError(size)
        end = min(int(end_s), size - 1) if end_s else size - 1
    except ValueError:
        return None  # malformed — ignore the header
    if end < start:
        return None
    return ByteRange(start, end)


class MediaGate:
    """Path confinement + cached ffprobe validation.

    Everything — the size the response advertises, the probe, and the bytes
    streamed — comes from ONE file descriptor, opened after confinement
    checks. Probing goes through ``/proc/self/fd`` (v1 targets one Linux
    machine), so a path replaced on disk mid-request can never smuggle
    unprobed content into a response.
    """

    def __init__(
        self, media_root: Path, *, ffprobe_bin: str = "ffprobe", timeout_seconds: float = 30.0
    ) -> None:
        self._root = media_root.resolve()
        self._ffprobe_bin = ffprobe_bin
        self._timeout = timeout_seconds
        # (resolved path) -> (size, mtime_ns) of the last successful probe.
        self._validated: dict[Path, tuple[int, int]] = {}
        self._lock = threading.Lock()

    def open_for_serving(self, path: Path) -> tuple[BinaryIO, int]:
        """Validate and return an open (file object, size). Caller closes it.

        Raises MediaNotServableError; never returns a handle that has not
        been probed (or fingerprint-matched against a previous probe).
        """
        resolved = path.resolve()
        if not resolved.is_relative_to(self._root):
            raise MediaNotServableError(f"{path} escapes the media root")
        if not resolved.is_file():
            raise MediaNotServableError(f"{path} is not a regular file")
        try:
            fh: BinaryIO = resolved.open("rb")
        except OSError as exc:
            raise MediaNotServableError(f"cannot open {path}: {exc}") from exc
        try:
            stat = os.fstat(fh.fileno())
            if not stat_module.S_ISREG(stat.st_mode):
                raise MediaNotServableError(f"{path} is not a regular file")
            fingerprint = (stat.st_size, stat.st_mtime_ns)
            with self._lock:
                cached = self._validated.get(resolved) == fingerprint
            if not cached:
                # The parent's pid, not /proc/self: ffprobe runs as a child
                # with close_fds, so "self" would be the child's empty table.
                fd_path = Path(f"/proc/{os.getpid()}/fd/{fh.fileno()}")
                try:
                    probe_audio(
                        fd_path, ffprobe_bin=self._ffprobe_bin, timeout_seconds=self._timeout
                    )
                except NormalizationError as exc:
                    raise MediaNotServableError(str(exc)) from exc
                except subprocess.TimeoutExpired as exc:
                    raise MediaNotServableError(f"ffprobe timed out on {path}") from exc
                with self._lock:
                    self._validated[resolved] = fingerprint
            fh.seek(0)
            return fh, stat.st_size
        except BaseException:
            fh.close()
            raise
