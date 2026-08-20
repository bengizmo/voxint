"""The media-file suffix allowlist shared by the scan and sidecar pairing.

One policy, two consumers: the setup wizard / watch-folder scan uses it to
decide what counts as an ingestable recording (``api.setup_wizard``), and
sidecar pairing uses the SAME set for its stem-ambiguity check
(``ingest.sidecar``), so "what is media" can never silently diverge between
them. A convenience filter, matched case-insensitively on the lowercased
suffix; ffprobe is NOT run per candidate (the PREPARE stage validates the
actual media when the run executes).
"""

from __future__ import annotations

MEDIA_SUFFIXES = frozenset(
    {
        ".wav", ".mp3", ".m4a", ".m4v", ".flac", ".ogg", ".oga", ".opus", ".aac",
        ".wma", ".aiff", ".aif", ".alac", ".mp4", ".mkv", ".mov", ".webm", ".avi",
        ".mpeg", ".mpg", ".ts", ".3gp",
    }
)
