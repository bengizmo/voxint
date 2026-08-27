"""Resolve domain packs by name from the configured sources (issue #11).

Per-run/per-folder selection stores a pack *name* (on ``media_folders.domain_pack``
or as an explicit CLI arg) and freezes the resolved pack's content onto the run.
This module is the name → :class:`DomainPack` resolver those seams share. Three
sources, in precedence order when names collide (identical content is idempotent;
a genuine clash is a config error, raised loudly):

1. the bundled ``generic`` pack (always available, zero config);
2. the configured default pack (``DOMAIN_PACK_PATH``), if set;
3. every direct child folder of ``DOMAIN_PACKS_DIR`` that holds a manifest.

Kept separate from :mod:`voxint.domain_packs.base` (the pure dataclass, imported
widely) because resolution reads :class:`Settings` and touches the filesystem.
"""

import logging
from collections.abc import Mapping
from typing import Any

from voxint.config import Settings
from voxint.domain_packs.base import DomainPack, DomainPackError, load_default

logger = logging.getLogger(__name__)


def default_domain_pack(settings: Settings) -> DomainPack:
    """The pack for an unmapped folder or a legacy (pre-#11) run.

    ``DOMAIN_PACK_PATH`` when configured, else the bundled ``generic`` pack —
    exactly the pre-#11 global behavior, so with no per-folder mapping nothing
    changes.
    """
    if settings.domain_pack_path is not None:
        return DomainPack.load(settings.domain_pack_path)
    return load_default()


def available_domain_packs(settings: Settings) -> dict[str, DomainPack]:
    """All resolvable packs keyed by manifest ``name``.

    Backs the Settings folder→pack picker and validates a mapping before it is
    saved. Two packs sharing a name with *different* content is a config error
    (which pack did the operator mean?); the same pack reached by two sources
    (e.g. ``DOMAIN_PACK_PATH`` also sits under ``DOMAIN_PACKS_DIR``) is fine.
    """
    packs: dict[str, DomainPack] = {}

    def _add(pack: DomainPack) -> None:
        existing = packs.get(pack.name)
        if existing is not None and existing.to_mapping() != pack.to_mapping():
            raise DomainPackError(
                f"two different domain packs both claim the name {pack.name!r}; "
                "give each pack a unique 'name' in its manifest.yaml"
            )
        packs[pack.name] = pack

    _add(load_default())
    if settings.domain_pack_path is not None:
        _add(DomainPack.load(settings.domain_pack_path))
    if settings.domain_packs_dir is not None:
        packs_dir = settings.domain_packs_dir
        if not packs_dir.is_dir():
            raise DomainPackError(
                f"DOMAIN_PACKS_DIR is set to {packs_dir}, which is not a directory"
            )
        for child in sorted(packs_dir.iterdir()):
            if (child / "manifest.yaml").is_file():
                _add(DomainPack.load(child))
    return packs


def resolve_domain_pack_by_name(name: str, settings: Settings) -> DomainPack:
    """Look up a pack by name, raising :class:`DomainPackError` if unknown.

    The caller (submission stamping, per-run restore) must never silently fall
    back to generic — an unknown name is a config error the operator must see.
    """
    packs = available_domain_packs(settings)
    try:
        return packs[name]
    except KeyError:
        raise DomainPackError(
            f"unknown domain pack {name!r}; available packs: {sorted(packs)}"
        ) from None


def resolve_run_domain_pack(
    *,
    settings: Settings,
    explicit_name: str | None = None,
) -> dict[str, Any]:
    """Resolve the pack for a NEW run and return its frozen snapshot mapping.

    If ``explicit_name`` is given, that pack is resolved by name (raising
    :class:`DomainPackError` if it does not exist). Otherwise the configured
    default pack is used. Per-folder pack selection is handled by the caller
    (reading ``MediaFolder.domain_pack``) and passed through as
    ``explicit_name`` when set.
    """
    if explicit_name is not None:
        return resolve_domain_pack_by_name(explicit_name, settings).to_mapping()
    return default_domain_pack(settings).to_mapping()


def domain_pack_from_snapshot(
    snapshot: Mapping[str, Any] | None, settings: Settings
) -> DomainPack:
    """Reconstruct a run's frozen pack (issue #11) from its ``domain_pack`` column.

    Shared by the pipeline worker and the enrichment producers, so both read the
    SAME pack a run was transcribed with. ``None`` = a legacy run created before
    per-run selection: resolve the current default pack (its pre-#11 behavior). A
    present-but-corrupt snapshot degrades to the default with a warning rather than
    propagating — in the pipeline it runs before ``execute_run``'s failure handling,
    so raising would strand the run QUEUED for the recovery sweep to re-publish
    forever (a poison loop), mirroring how the LLM-client build degrades a malformed
    base URL. In practice the snapshot is our own validated round-trip, so this only
    fires on out-of-band DB tampering.
    """
    if snapshot is None:
        return default_domain_pack(settings)
    try:
        return DomainPack.from_mapping(snapshot)
    except DomainPackError as exc:
        logger.warning(
            "run's domain_pack snapshot is unreadable; proceeding with the default "
            "pack for this run: %s",
            exc,
        )
        return default_domain_pack(settings)
