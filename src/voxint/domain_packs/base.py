"""Domain pack seam.

A domain pack supplies the domain-specific knowledge the pipeline consumes —
ASR vocabulary hints, name seeds, and LLM prompt fragments — as a folder with a
``manifest.yaml``. The bundled ``generic`` pack (neutral meeting/podcast) is the
default; deployments point ``DOMAIN_PACK_PATH`` at their own.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DomainPack:
    name: str
    description: str = ""
    vocabulary: tuple[str, ...] = ()
    name_seeds: tuple[str, ...] = ()
    prompt_fragments: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, pack_dir: Path) -> "DomainPack":
        manifest = yaml.safe_load((pack_dir / "manifest.yaml").read_text())
        return cls(
            name=manifest["name"],
            description=manifest.get("description", ""),
            vocabulary=tuple(manifest.get("vocabulary", [])),
            name_seeds=tuple(manifest.get("name_seeds", [])),
            prompt_fragments=dict(manifest.get("prompt_fragments", {})),
        )


def load_default() -> DomainPack:
    return DomainPack.load(Path(__file__).parent / "generic")
