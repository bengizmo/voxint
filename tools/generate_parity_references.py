#!/usr/bin/env python3
"""Produce CUDA reference outputs for the parity gate (run on NVIDIA hardware).

Feeds the committed corpus (``tests/parity/fixtures/corpus/``, built by
``tools/generate_parity_corpus.py``) through *running* voxint GPU services and
records their responses as the reference fixtures every alternative backend
(CPU, ROCm, ONNX, Metal) is measured against — and as the standing NVIDIA
regression gate re-run before each tier release.

Usage (services running with MEDIA_ROOT mounted at the corpus directory):

    docker run -d --gpus all -p 8022:8022 -v $PWD/tests/parity/fixtures/corpus:/data/media:ro \\
        ghcr.io/bengizmo/voxint-whisper:<tag>
    ... (pyannote on 8024 — weights vendored, no token; titanet on 8021) ...
    python tools/generate_parity_references.py --tag <image tag>

Writes ``tests/parity/fixtures/references/cuda/{embed,transcribe,diarize}.json``.
Each file embeds the corpus checksums and the service's /healthz identity so a
reference can never be silently paired with the wrong corpus or engine. By
default services must report ``device: cuda`` (--allow-device overrides for
experiments; the committed references must stay CUDA).

The script also cross-checks the corpus's expected skip_reason/snr_db per
window against the live service and fails on any mismatch — validating the
locally-computed expectations against the real implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
CORPUS_DIR = REPO / "tests" / "parity" / "fixtures" / "corpus"
REFS_DIR = REPO / "tests" / "parity" / "fixtures" / "references" / "cuda"

SNR_REPORT_TOLERANCE_DB = 0.05


def _http_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 3600.0) -> Any:
    if payload is None:
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _healthz(base_url: str, allow_device: str | None) -> dict[str, Any]:
    health = _http_json(f"{base_url}/healthz")
    if health.get("status") != "ok":
        raise SystemExit(f"{base_url}/healthz not ok: {health}")
    device = health.get("device")
    if device != "cuda" and device != allow_device:
        raise SystemExit(
            f"{base_url} reports device={device!r}; committed references must be CUDA "
            "(use --allow-device only for uncommitted experiments)"
        )
    if not health.get("engine"):
        # Pre-Phase-0 images (<= 0.3.0) predate the engine/runtime health
        # fields; their stacks are unambiguous (see docs/gpu-contracts.md) but
        # the gap should be visible in the run log and the recorded meta.
        print(
            f"WARNING: {base_url} does not report engine/runtime identity "
            "(pre-engine-field image); meta.service_healthz will lack it",
            file=sys.stderr,
        )
    return health


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asr-url", default="http://127.0.0.1:8022")
    parser.add_argument("--diarizer-url", default="http://127.0.0.1:8024")
    parser.add_argument("--embedder-url", default="http://127.0.0.1:8021")
    parser.add_argument("--tag", required=True, help="image tag the services run, e.g. 0.3.0")
    parser.add_argument(
        "--image-digest", default=None,
        help="optional immutable image digest (sha256:...) recorded next to --tag",
    )
    parser.add_argument("--allow-device", default=None, help="accept a non-cuda device value")
    parser.add_argument(
        "--out-dir", type=Path, default=REFS_DIR,
        help="output directory (default: the committed CUDA reference dir)",
    )
    parser.add_argument(
        "--only", choices=["embed", "transcribe", "diarize"], action="append",
        help="restrict to specific references (repeatable; default: all)",
    )
    args = parser.parse_args()
    which = set(args.only or ["embed", "transcribe", "diarize"])
    out_dir: Path = args.out_dir
    if args.allow_device and out_dir.resolve() == REFS_DIR.resolve():
        # The escape hatch must never overwrite the committed CUDA baseline.
        raise SystemExit(
            "--allow-device writes non-CUDA output; pass --out-dir pointing "
            f"somewhere other than {REFS_DIR}"
        )

    provenance = json.loads((CORPUS_DIR / "provenance.json").read_text())
    manifest = json.loads((CORPUS_DIR / "manifest.json").read_text())

    for wav, recorded in provenance["wav_sha256"].items():
        actual = hashlib.sha256((CORPUS_DIR / wav).read_bytes()).hexdigest()
        if actual != recorded:
            raise SystemExit(f"{wav} sha256 mismatch — regenerate or restore the corpus first")

    # Bind the SEMANTIC corpus inputs too, not just the audio: window bounds,
    # expected gates, pair labels, and clip layouts all change test meaning
    # without touching a WAV.
    corpus_files_sha256 = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(CORPUS_DIR.iterdir())
        if p.suffix in {".wav", ".json"}
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    def meta(health: dict[str, Any]) -> dict[str, Any]:
        return {
            "generated_on": date.today().isoformat(),
            "image_tag": args.tag,
            "image_digest": args.image_digest,
            "hardware": "maintainer NVIDIA hardware",
            "corpus_sha256": provenance["wav_sha256"],
            "corpus_files_sha256": corpus_files_sha256,
            "service_healthz": health,
        }

    if "embed" in which:
        health = _healthz(args.embedder_url, args.allow_device)
        windows_doc = json.loads((CORPUS_DIR / "embed-windows.json").read_text())
        windows = windows_doc["windows"]
        response = _http_json(
            f"{args.embedder_url}/v1/embed",
            {
                "path": windows_doc["wav"],
                "windows": [
                    {"start_seconds": w["start_seconds"], "end_seconds": w["end_seconds"]}
                    for w in windows
                ],
            },
        )
        results = response["results"]
        if len(results) != len(windows):
            raise SystemExit(f"embed returned {len(results)} results for {len(windows)} windows")
        mismatches = []
        for w, r in zip(windows, results, strict=True):
            if r["skip_reason"] != w["expected_skip_reason"]:
                mismatches.append(
                    f"{w['id']}: skip {r['skip_reason']!r} != expected "
                    f"{w['expected_skip_reason']!r}"
                )
            elif w["expected_snr_db"] is not None and (
                r["snr_db"] is None
                or abs(r["snr_db"] - w["expected_snr_db"]) > SNR_REPORT_TOLERANCE_DB
            ):
                mismatches.append(
                    f"{w['id']}: snr {r['snr_db']} != expected {w['expected_snr_db']}"
                )
        if mismatches:
            for m in mismatches:
                print(f"MISMATCH {m}", file=sys.stderr)
            raise SystemExit(f"{len(mismatches)} corpus-expectation mismatches — investigate")
        (out_dir / "embed.json").write_text(
            json.dumps(
                {"meta": meta(health), "embedding_space": response["embedding_space"],
                 "windows": [
                     {"id": w["id"], **r} for w, r in zip(windows, results, strict=True)
                 ]},
                indent=2,
            ) + "\n"
        )
        n_ok = sum(1 for r in results if r["embedding"] is not None)
        print(f"embed.json: {n_ok}/{len(results)} embedded, space={response['embedding_space']}")

    if "transcribe" in which:
        health = _healthz(args.asr_url, args.allow_device)
        variants = {}
        for vad in (True, False):
            variants[f"vad_{str(vad).lower()}"] = _http_json(
                f"{args.asr_url}/v1/transcribe",
                {"path": manifest["transcribe"]["wav"], "language": "en", "vad_filter": vad},
            )
        (out_dir / "transcribe.json").write_text(
            json.dumps({"meta": meta(health), "variants": variants}, indent=2) + "\n"
        )
        seg = len(variants["vad_true"]["segments"])
        print(f"transcribe.json: {seg} segments (vad_true), "
              f"confidence {variants['vad_true']['confidence']:.3f}")

    if "diarize" in which:
        health = _healthz(args.diarizer_url, args.allow_device)
        response = _http_json(
            f"{args.diarizer_url}/v1/diarize",
            {"path": manifest["diarize"]["wav"], "min_speakers": 1, "max_speakers": 10},
        )
        (out_dir / "diarize.json").write_text(
            json.dumps({"meta": meta(health), "response": response}, indent=2) + "\n"
        )
        print(f"diarize.json: {response['num_speakers']} speakers, "
              f"{len(response['turns'])} turns")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
