"""Domain pack seam.

A domain pack supplies the domain-specific knowledge the pipeline consumes —
ASR vocabulary hints, name seeds, and LLM prompt fragments — as a folder with a
``manifest.yaml``. The bundled ``generic`` pack (neutral meeting/podcast) is the
default; deployments point ``DOMAIN_PACK_PATH`` at their own, or drop several
named packs under ``DOMAIN_PACKS_DIR`` and select per run/folder (issue #11).

A pack's selection is frozen onto its run as a JSON snapshot at submit time
(``pipeline_runs.domain_pack``), so a run — and the enrichment that reads it
hours later — always sees the exact vocabulary, seeds, and fragments it was
transcribed with, even if the manifest on disk is later edited or removed.
:meth:`DomainPack.to_mapping` / :meth:`DomainPack.from_mapping` are that
snapshot's round-trip; ``from_mapping`` is strict because a corrupt persisted
snapshot is a deterministic run/enrichment error, never a silent fallback.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from voxint.domain_packs.corrections import CorrectionRule


class DomainPackError(Exception):
    """A domain pack could not be loaded, resolved, or reconstructed.

    Deterministic operator/config error (a missing manifest, an unknown pack
    name, a corrupt persisted snapshot), never transient — callers surface it
    loudly rather than substituting the generic pack, which would produce
    plausible-but-inconsistent output.
    """


def dedup_order_preserving(items: Iterable[str]) -> tuple[str, ...]:
    """First occurrence wins; blank/whitespace-only entries dropped.

    The one canonicalization for an effective vocabulary list, shared by the
    worker's live layering (:mod:`voxint.pipeline.stages.context`) and the
    submit-time freeze (:mod:`voxint.ingest.service`). Keeping a single
    definition is load-bearing for issue #153: the v2 snapshot freezes the
    already-canonicalized effective vocabulary, so it must be byte-identical to
    what the v1 worker computed live — ``D(pack + D(app)) == D(pack + app)``
    holds only because both sides run THIS function.
    """
    seen: dict[str, None] = {}
    for item in items:
        stripped = item.strip()
        if stripped and stripped not in seen:
            seen[stripped] = None
    return tuple(seen)


def _str_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    """Coerce a manifest/snapshot list field to ``tuple[str, ...]``, strictly."""
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise DomainPackError(
            f"domain pack {field_name!r} must be a list, got {type(value).__name__}"
        )
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise DomainPackError(f"domain pack {field_name!r} entries must be strings")
        out.append(item)
    return tuple(out)


def _str_map(value: Any, field_name: str) -> dict[str, str]:
    """Coerce a manifest/snapshot mapping field to ``dict[str, str]``, strictly."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise DomainPackError(f"domain pack {field_name!r} must be a mapping")
    out: dict[str, str] = {}
    for key, val in value.items():
        if not isinstance(key, str) or not isinstance(val, str):
            raise DomainPackError(f"domain pack {field_name!r} must map strings to strings")
        out[key] = val
    return out


@dataclass(frozen=True)
class DomainPack:
    name: str
    description: str = ""
    vocabulary: tuple[str, ...] = ()
    name_seeds: tuple[str, ...] = ()
    prompt_fragments: dict[str, str] = field(default_factory=dict)
    corrections: tuple[CorrectionRule, ...] = ()

    @classmethod
    def load(cls, pack_dir: Path) -> DomainPack:
        """Load a pack from ``<pack_dir>/manifest.yaml`` (raises on a bad manifest)."""
        manifest_path = pack_dir / "manifest.yaml"
        try:
            raw = yaml.safe_load(manifest_path.read_text())
        except FileNotFoundError as exc:
            raise DomainPackError(f"no domain pack manifest at {manifest_path}") from exc
        except yaml.YAMLError as exc:
            raise DomainPackError(
                f"malformed domain pack manifest at {manifest_path}: {exc}"
            ) from exc
        if not isinstance(raw, Mapping):
            raise DomainPackError(f"domain pack manifest at {manifest_path} is not a mapping")
        return cls.from_mapping(raw)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> DomainPack:
        """Reconstruct a pack from a manifest/snapshot mapping (strict).

        Shared by :meth:`load` (manifest.yaml) and the per-run snapshot restore
        (``pipeline_runs.domain_pack``). A missing or non-string ``name`` — the
        one required field — is a hard error; every other field defaults empty.
        A non-mapping input (a tampered snapshot holding a JSON array or scalar)
        is rejected as :class:`DomainPackError` here too, so the single
        validation point never leaks a raw ``AttributeError`` past the tolerant
        snapshot-restore path.
        """
        if not isinstance(data, Mapping):
            raise DomainPackError(
                f"domain pack must be a mapping, got {type(data).__name__}"
            )
        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            raise DomainPackError("domain pack 'name' is required and must be a non-empty string")
        description = data.get("description", "")
        if not isinstance(description, str):
            raise DomainPackError("domain pack 'description' must be a string")
        # Local import breaks the module cycle: corrections.py imports
        # DomainPackError from this module at top level, so this module must not
        # import corrections at top level. By call time `base` is fully loaded.
        from voxint.domain_packs.corrections import parse_corrections

        return cls(
            name=name,
            description=description,
            vocabulary=_str_tuple(data.get("vocabulary"), "vocabulary"),
            name_seeds=_str_tuple(data.get("name_seeds"), "name_seeds"),
            prompt_fragments=_str_map(data.get("prompt_fragments"), "prompt_fragments"),
            corrections=parse_corrections(data.get("corrections")),
        )

    def to_mapping(self) -> dict[str, Any]:
        """Serialize to a JSON-safe mapping for the per-run snapshot column.

        Round-trips through :meth:`from_mapping` losslessly. Tuples become lists
        (JSON has no tuple), and ``prompt_fragments`` is copied so the persisted
        value never aliases the live dict.
        """
        return {
            "name": self.name,
            "description": self.description,
            "vocabulary": list(self.vocabulary),
            "name_seeds": list(self.name_seeds),
            "prompt_fragments": dict(self.prompt_fragments),
            "corrections": [rule.to_mapping() for rule in self.corrections],
        }


def load_default() -> DomainPack:
    return DomainPack.load(Path(__file__).parent / "generic")


def union_pack_name_seeds(
    pack_mapping: Mapping[str, Any], seeds: Sequence[str]
) -> dict[str, Any]:
    """Union extra operator-supplied name seeds onto a resolved pack snapshot.

    The sidecar-ingest sibling of ``union_pack_corrections`` (issue #104):
    appends ``seeds`` AFTER the pack's own ``name_seeds`` (pack entries keep
    priority), skipping exact-string duplicates, and returns a NEW snapshot
    mapping — the input is never mutated. The combined list is re-validated
    through the same strict coercion the snapshot restore uses, so a
    malformed entry surfaces at submit-time freeze, never downstream.
    """
    merged = dict(pack_mapping)
    existing = _str_tuple(pack_mapping.get("name_seeds"), "name_seeds")
    combined = list(existing)
    seen = set(existing)
    for seed in _str_tuple(list(seeds), "name_seeds"):
        if seed not in seen:
            seen.add(seed)
            combined.append(seed)
    merged["name_seeds"] = combined
    return merged
