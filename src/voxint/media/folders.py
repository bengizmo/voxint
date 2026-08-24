"""Pure helpers for the relational media-folder layer (issue #153).

Console 2.0 P2a moves registered folders from the ``app_settings`` list into
first-class :class:`~voxint.db.models.MediaFolder` rows (ADR 0002). This module
holds the path logic shared by the migration backfill, the folder-registration
write paths, and the ``voxint media folders preflight`` command:

- deepest-ancestor folder membership, component-boundary matched so ``audio/pod``
  never matches a file under ``audio/podcasts``;
- overlap detection (one registration nested inside another), which the write
  paths refuse and the preflight reports;
- the preflight report itself, which surfaces the ambiguities an operator must
  reconcile before the migration cuts over.

Everything here is pure: paths in, findings out. The filesystem and database
reads (missing directories, the media corpus) are done by the caller and passed
in, so the logic is testable without either.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
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


def folder_pack_name(
    source_path: str, folder_domain_packs: Mapping[str, str]
) -> str | None:
    """The pre-#153 resolution: the pack of the deepest pack-MAPPED ancestor.

    Mirrors ``domain_packs.registry.resolve_folder_pack_name`` (which only
    considers folders that carry a pack). Kept here so the preflight can compare
    the old effective pack against the new folder-relation pack.
    """
    mapped = deepest_ancestor(source_path, list(folder_domain_packs))
    return folder_domain_packs.get(mapped) if mapped is not None else None


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


@dataclass(frozen=True)
class PackDivergence:
    """A media file whose effective domain pack would change under the cutover."""

    source_path: str
    old_pack: str | None
    new_pack: str | None


@dataclass
class PreflightReport:
    """What an operator must reconcile before the folder migration cuts over."""

    nested: list[tuple[str, str]] = field(default_factory=list)
    missing_dirs: list[str] = field(default_factory=list)
    orphan_pack_keys: list[str] = field(default_factory=list)
    pack_divergences: list[PackDivergence] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when nothing blocks the cutover.

        Missing directories are reported but do not block: a registered folder
        whose bytes were removed is a benign staleness the migration tolerates
        (it simply matches no media). Nesting, orphan pack keys, and any pack
        divergence are the blocking ambiguities — the migration aborts on the
        divergences, and nesting/orphans are their structural cause.
        """
        return not (self.nested or self.orphan_pack_keys or self.pack_divergences)


def build_preflight_report(
    *,
    folders: Sequence[str],
    folder_domain_packs: Mapping[str, str],
    source_paths: Sequence[str],
    missing_dirs: Sequence[str] = (),
) -> PreflightReport:
    """Build the folder-migration preflight report from already-gathered data.

    ``folders`` and ``folder_domain_packs`` are the legacy ``app_settings``
    registrations; ``source_paths`` is every ``media_items.source_path`` (so the
    exact rows the migration will reclassify); ``missing_dirs`` is the subset of
    ``folders`` the caller found absent under MEDIA_ROOT. ``pack_divergences``
    reproduces the migration's abort condition: for each media file, the old
    longest-ancestor pack versus the pack of the deepest registered folder it
    would now belong to.
    """
    orphans = sorted(k for k in folder_domain_packs if k not in set(folders))
    divergences: list[PackDivergence] = []
    for source_path in source_paths:
        old_pack = folder_pack_name(source_path, folder_domain_packs)
        folder = deepest_ancestor(source_path, folders)
        new_pack = folder_domain_packs.get(folder) if folder is not None else None
        if old_pack != new_pack:
            divergences.append(PackDivergence(source_path, old_pack, new_pack))
    return PreflightReport(
        nested=nested_pairs(folders),
        missing_dirs=list(missing_dirs),
        orphan_pack_keys=orphans,
        pack_divergences=divergences,
    )
