"""Pure helpers for the relational media-folder layer (issue #153).

Console 2.0 P2a moved registered folders from the ``app_settings`` list into
first-class :class:`~voxint.db.models.MediaFolder` rows (ADR 0002). This module
holds the path logic shared by the migration backfill and the folder-registration
write paths:

- deepest-ancestor folder membership, component-boundary matched so ``audio/pod``
  never matches a file under ``audio/podcasts``;
- overlap detection (one registration nested inside another), which the write
  paths refuse.

Everything here is pure: paths in, findings out. The filesystem and database
reads are done by the caller and passed in, so the logic is testable without
either.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import PurePosixPath


def deepest_ancestor(source_path: str, folders: Sequence[str]) -> str | None:
    """The deepest folder in ``folders`` that contains ``source_path``.

    Matches on path components (``PurePosixPath``), so ``audio/pod`` is not an
    ancestor of ``audio/podcasts/ep.wav``; a folder equal to the file's own
    directory or any ancestor matches. Deeper (longer) folders win. ``None`` when
    no folder contains the path. The registered media root (``"."``) is an
    ancestor of every relative path and the shallowest possible match.
    """
    src = PurePosixPath(source_path)
    best: str | None = None
    best_depth = -1
    for folder in folders:
        folder_path = PurePosixPath(folder)
        if src == folder_path or folder_path in src.parents:
            depth = len(folder_path.parts)
            if depth > best_depth:
                best_depth = depth
                best = folder
    return best


def is_ancestor(ancestor: str, descendant: str) -> bool:
    """True when ``ancestor`` is a proper component-boundary ancestor of
    ``descendant`` (not equal). ``"."`` is an ancestor of every other folder."""
    if ancestor == descendant:
        return False
    return PurePosixPath(ancestor) in PurePosixPath(descendant).parents


def overlapping_registration(candidate: str, existing: Sequence[str]) -> str | None:
    """The first folder in ``existing`` that nests with ``candidate`` (either
    direction), or ``None`` when ``candidate`` overlaps nothing.

    The write-path guard: a new registration is refused when it is an ancestor or
    a descendant of an already-registered folder, because a file could then belong
    to two folders and membership (ADR 0002) would be ambiguous. An exact
    duplicate is caught separately by the ``UNIQUE(path)`` constraint.
    """
    for other in existing:
        if candidate == other:
            continue
        if is_ancestor(candidate, other) or is_ancestor(other, candidate):
            return other
    return None


def nested_pairs(folders: Sequence[str]) -> list[tuple[str, str]]:
    """All (ancestor, descendant) pairs among ``folders``, ancestor first."""
    pairs: list[tuple[str, str]] = []
    for outer in folders:
        for inner in folders:
            if is_ancestor(outer, inner):
                pairs.append((outer, inner))
    return pairs
