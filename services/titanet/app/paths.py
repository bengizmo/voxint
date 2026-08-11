"""MEDIA_ROOT-relative path resolution — torch-free by design.

Every request names its audio file relative to the shared media volume.
Absolute paths and traversal outside MEDIA_ROOT are contract violations (400);
a missing file is 404. Duplicated verbatim across the three GPU services on
purpose — each image is self-contained.
"""

from pathlib import Path


class PathViolation(ValueError):
    """Request path breaks the MEDIA_ROOT containment contract (HTTP 400)."""


class PathNotFound(ValueError):
    """Request path is inside MEDIA_ROOT but no file exists there (HTTP 404)."""


def resolve_media_path(media_root: Path, raw_path: str) -> Path:
    if not raw_path or raw_path.strip() == "":
        raise PathViolation("Empty path")
    if raw_path.startswith("/"):
        raise PathViolation("Absolute paths are not accepted; use MEDIA_ROOT-relative paths")

    resolved = (media_root / raw_path).resolve()
    if not resolved.is_relative_to(media_root.resolve()):
        raise PathViolation("Path escapes MEDIA_ROOT")
    if not resolved.exists():
        raise PathNotFound(f"Audio file not found: {raw_path}")
    if not resolved.is_file():
        raise PathViolation(f"Not a file: {raw_path}")
    return resolved
