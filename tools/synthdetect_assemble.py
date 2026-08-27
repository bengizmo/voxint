#!/usr/bin/env python3
"""Assemble a v3 composite manifest from S5 bona fide + S6 spoof + ASVspoof components."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from synthdetect_corpus import (  # noqa: E402
    COMPOSITE_MANIFEST_SCHEMA_VERSION,
    CORPUS_KIND_COMPOSITE,
    CORPUS_KIND_IMPORTED,
    CORPUS_KIND_SYNTHESIS,
    load_manifest,
)

GENERATORS_SEEN = frozenset({"piper", "chatterbox"})
GENERATORS_UNSEEN = frozenset({"elevenlabs", "google"})


@dataclass(frozen=True)
class ComponentSpec:
    component_id: str
    manifest_path: Path
    audio_root: Path
    corpus_kind: str
    subdir: str
    benchmark: str | None = None


def _manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _extract_parent_clip_id(clip_id: str) -> str | None:
    """Extract the bona fide parent clip_id from a TTS spoof clip_id.

    Convention: ``{parent_clip_id}--{generator_name}``.
    """
    if "--" not in clip_id:
        return None
    return clip_id.rsplit("--", 1)[0]


def _transform_synthesis_clip(
    raw: dict[str, Any],
    component_id: str,
    subdir: str,
    bonafide_ids: set[str],
) -> dict[str, Any]:
    """Transform a v1 synthesis clip dict into a v3 composite clip dict."""
    clip = dict(raw)
    clip["provenance_kind"] = CORPUS_KIND_SYNTHESIS
    clip["component_id"] = component_id
    clip["rel_path"] = f"{subdir}/{raw['rel_path']}"
    if raw["label"] == "spoof":
        parent_id = _extract_parent_clip_id(raw["clip_id"])
        if parent_id and parent_id in bonafide_ids:
            clip["partition_group_id"] = parent_id
        else:
            clip["partition_group_id"] = None
    elif raw["label"] == "bona_fide":
        clip["partition_group_id"] = raw["clip_id"]
    else:
        clip["partition_group_id"] = None
    return clip


def _transform_imported_clip(
    raw: dict[str, Any],
    component_id: str,
    subdir: str,
) -> dict[str, Any]:
    """Transform a v2 imported-benchmark clip dict into a v3 composite clip dict."""
    clip = dict(raw)
    clip["provenance_kind"] = CORPUS_KIND_IMPORTED
    clip["component_id"] = component_id
    clip["rel_path"] = f"{subdir}/{raw['rel_path']}"
    clip["partition_group_id"] = None
    return clip


def _hardlink_clips(
    manifest_clips: list[dict[str, Any]],
    audio_root: Path,
    composite_root: Path,
    original_rel_paths: list[str],
) -> int:
    """Hardlink clip audio files into the composite directory. Returns count."""
    count = 0
    for clip, orig_rel in zip(manifest_clips, original_rel_paths, strict=True):
        src = audio_root / orig_rel
        dst = composite_root / clip["rel_path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst.unlink()
        os.link(src, dst)
        count += 1
    return count


def assemble(specs: list[ComponentSpec], out_dir: Path, *, dry_run: bool = False) -> Path:
    """Assemble the v3 composite manifest and corpus directory."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all bona fide clip IDs first (needed for partition_group_id)
    bonafide_ids: set[str] = set()
    for spec in specs:
        if spec.corpus_kind != CORPUS_KIND_SYNTHESIS:
            continue
        raw_manifest = json.loads(spec.manifest_path.read_bytes())
        for clip in raw_manifest["clips"]:
            if clip["label"] == "bona_fide":
                bonafide_ids.add(clip["clip_id"])

    components: list[dict[str, Any]] = []
    all_clips: list[dict[str, Any]] = []
    assembly_components: list[dict[str, Any]] = []

    for spec in specs:
        manifest_bytes = spec.manifest_path.read_bytes()
        manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        raw_manifest = json.loads(manifest_bytes)
        raw_clips = raw_manifest["clips"]

        component_entry: dict[str, Any] = {
            "component_id": spec.component_id,
            "corpus_kind": spec.corpus_kind,
            "manifest_sha256": manifest_sha,
            "clip_count": len(raw_clips),
        }
        if spec.benchmark:
            component_entry["benchmark"] = spec.benchmark
        components.append(component_entry)

        original_rel_paths: list[str] = []
        component_clips: list[dict[str, Any]] = []

        for raw in raw_clips:
            original_rel_paths.append(raw["rel_path"])
            if spec.corpus_kind == CORPUS_KIND_SYNTHESIS:
                clip = _transform_synthesis_clip(
                    raw, spec.component_id, spec.subdir, bonafide_ids,
                )
            else:
                clip = _transform_imported_clip(raw, spec.component_id, spec.subdir)
            component_clips.append(clip)

        if not dry_run:
            linked = _hardlink_clips(component_clips, spec.audio_root, out_dir, original_rel_paths)
            print(f"  {spec.component_id}: {linked} clips hardlinked")

        all_clips.extend(component_clips)
        assembly_components.append({
            "component_id": spec.component_id,
            "manifest_path": str(spec.manifest_path),
            "manifest_sha256": manifest_sha,
            "clip_count": len(raw_clips),
            "audio_root": str(spec.audio_root),
            "subdir": spec.subdir,
        })

    all_clips.sort(key=lambda c: c["clip_id"])

    composite = {
        "schema_version": COMPOSITE_MANIFEST_SCHEMA_VERSION,
        "corpus_kind": CORPUS_KIND_COMPOSITE,
        "components": sorted(components, key=lambda c: c["component_id"]),
        "clips": all_clips,
    }

    manifest_bytes_out = (json.dumps(composite, indent=2, sort_keys=True) + "\n").encode("utf-8")
    manifest_sha_out = hashlib.sha256(manifest_bytes_out).hexdigest()

    # Validate the composite manifest through load_manifest (fail-closed)
    loaded = load_manifest(json.loads(manifest_bytes_out))
    assert loaded.components is not None
    print(f"  Composite manifest validated: {len(loaded.clips)} clips, "
          f"{len(loaded.components)} components")

    manifest_path = out_dir / "manifest.json"
    if not dry_run:
        manifest_path.write_bytes(manifest_bytes_out)
        print(f"  Written: {manifest_path}")

        receipt = {
            "composite_manifest_sha256": manifest_sha_out,
            "total_clips": len(all_clips),
            "components": assembly_components,
            "bonafide_count": sum(1 for c in all_clips if c["label"] == "bona_fide"),
            "spoof_count": sum(1 for c in all_clips if c["label"] == "spoof"),
        }
        receipt_path = out_dir / "assembly_receipt.json"
        receipt_path.write_bytes(
            (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
        )
        print(f"  Written: {receipt_path}")

    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s5-dir", type=Path, required=True,
                        help="S5 bona fide corpus root (contains ami-corpus/, vc-corpus/)")
    parser.add_argument("--s6-dir", type=Path, required=True,
                        help="S6 spoof output root (contains ami/, vc/ with generator subdirs)")
    parser.add_argument("--asvspoof-dir", type=Path, required=True,
                        help="ASVspoof DF corpus root (contains manifest.json + canonical/)")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Output directory for composite corpus")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate only, don't write files or hardlink")
    args = parser.parse_args()

    specs: list[ComponentSpec] = []

    # S5 bona fide components
    for domain, subdir_name in [("ami", "ami-corpus"), ("vc", "vc-corpus")]:
        corpus_dir = args.s5_dir / subdir_name
        specs.append(ComponentSpec(
            component_id=f"organic-bonafide-{domain}",
            manifest_path=corpus_dir / "manifest.json",
            audio_root=corpus_dir,
            corpus_kind=CORPUS_KIND_SYNTHESIS,
            subdir=f"organic-bonafide/{domain}",
        ))

    # S6 TTS spoof components
    # rel_path in spoof manifests already includes the generator subdir
    # (e.g. "piper/clip.wav"), so audio_root is the domain dir, not generator dir
    for generator in ["piper", "chatterbox", "elevenlabs", "google"]:
        for domain in ["ami", "vc"]:
            gen_dir = args.s6_dir / domain / generator
            specs.append(ComponentSpec(
                component_id=f"tts-{generator}-{domain}",
                manifest_path=gen_dir / "manifest.json",
                audio_root=args.s6_dir / domain,
                corpus_kind=CORPUS_KIND_SYNTHESIS,
                subdir=f"tts-{generator}/{domain}",
            ))

    # ASVspoof 2021 DF benchmark anchor
    specs.append(ComponentSpec(
        component_id="asvspoof-df",
        manifest_path=args.asvspoof_dir / "manifest.json",
        audio_root=args.asvspoof_dir,
        corpus_kind=CORPUS_KIND_IMPORTED,
        subdir="asvspoof-df",
        benchmark="asvspoof2021_df",
    ))

    # Verify all manifest files exist before starting
    for spec in specs:
        if not spec.manifest_path.is_file():
            print(f"ERROR: manifest not found: {spec.manifest_path}", file=sys.stderr)
            sys.exit(1)

    print(f"Assembling composite from {len(specs)} components into {args.out_dir}")
    assemble(specs, args.out_dir, dry_run=args.dry_run)
    print("Done.")


if __name__ == "__main__":
    main()
