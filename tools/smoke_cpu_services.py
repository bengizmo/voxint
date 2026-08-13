#!/usr/bin/env python3
"""Release smoke for the CPU model-service images — stdlib only.

Run by release.yml's smoke-cpu job (against per-arch DIGEST images, before
any tag exists) and equally usable against locally built images:

    python3 tools/smoke_cpu_services.py \
        --fixtures tests/parity/fixtures \
        --whisper http://localhost:8022 \
        --titanet http://localhost:8021 \
        [--pyannote http://localhost:8024]

The services must have the corpus directory mounted as their MEDIA_ROOT.

This is deliberately stronger than a liveness check:
- healthz must report the CPU-tier identity fields (device=cpu; titanet
  engine=onnxruntime) — a mislabeled or wrong-flavor image fails here.
- whisper must actually transcribe the corpus clip (segments > 0), so the
  CT2 path runs end to end.
- titanet must return a real 192-dim, unit-norm embedding for a corpus
  window that clears the quality gates AND land within cosine 0.999 of the
  committed CUDA reference — the ONNX graph provably executed on the SHIPPED
  numerical stack, and the embedding space provably held. (A skip_reason
  response would mean inference never ran; that is a failure at smoke level.)
- pyannote (optional, needs HF-gated weights) must diarize the 3-speaker
  clip to exactly 3 speakers.
"""

import argparse
import json
import math
import sys
import urllib.request

COSINE_FLOOR = 0.999  # vector-level parity gate is 0.9995; smoke keeps margin
SNR_TOLERANCE_DB = 0.5
EMBED_WINDOW_ID = "clean_utt_00"  # clean-category window; clears all gates


def _request(url: str, payload: dict | None = None, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"} if payload is not None else {},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def fail(msg: str) -> None:
    print(f"SMOKE FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def check_healthz(base: str, service: str, expect: dict[str, object]) -> None:
    health = _request(f"{base}/healthz")
    for key, want in expect.items():
        got = health.get(key)
        if got != want:
            fail(f"{service} /healthz {key}={got!r}, expected {want!r} ({health})")
    print(f"ok: {service} healthz identity {expect}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", required=True, help="tests/parity/fixtures dir")
    parser.add_argument("--whisper", help="whisper base URL")
    parser.add_argument("--titanet", help="titanet base URL")
    parser.add_argument("--pyannote", help="pyannote base URL (optional; HF-gated)")
    args = parser.parse_args()

    if args.whisper:
        check_healthz(
            args.whisper,
            "whisper",
            {"device": "cpu", "engine": "faster-whisper", "model_loaded": True},
        )
        result = _request(
            f"{args.whisper}/v1/transcribe",
            {"path": "transcribe-short.wav", "language": "en"},
        )
        if not result.get("segments"):
            fail(f"whisper returned no segments: {result}")
        print(f"ok: whisper transcribed {len(result['segments'])} segments")

    if args.titanet:
        check_healthz(
            args.titanet,
            "titanet",
            {"device": "cpu", "engine": "onnxruntime", "model_loaded": True},
        )
        with open(f"{args.fixtures}/corpus/embed-windows.json") as fh:
            windows = json.load(fh)["windows"]
        with open(f"{args.fixtures}/references/cuda/embed.json") as fh:
            references = json.load(fh)["windows"]
        spec = next(w for w in windows if w["id"] == EMBED_WINDOW_ID)
        reference = next(r for r in references if r["id"] == EMBED_WINDOW_ID)
        result = _request(
            f"{args.titanet}/v1/embed",
            {
                "path": "embed-corpus.wav",
                "windows": [
                    {
                        "start_seconds": spec["start_seconds"],
                        "end_seconds": spec["end_seconds"],
                    }
                ],
            },
        )
        window = result["results"][0]
        embedding = window.get("embedding")
        if embedding is None:
            # skip_reason here means inference never ran — that IS the failure
            # this check exists to catch (a smoke that "passes" without ever
            # executing the ONNX graph proves nothing).
            fail(f"titanet skipped the clean reference window: {window}")
        if len(embedding) != 192:
            fail(f"titanet embedding has {len(embedding)} dims, expected 192")
        norm = math.sqrt(sum(x * x for x in embedding))
        if abs(norm - 1.0) > 1e-3:
            fail(f"titanet embedding norm {norm} not ~1.0")
        ref_vec = reference["embedding"]
        cosine = sum(a * b for a, b in zip(embedding, ref_vec, strict=True))  # both unit-norm
        if cosine < COSINE_FLOOR:
            fail(
                f"titanet cosine {cosine:.6f} vs CUDA reference below {COSINE_FLOOR} "
                "— embedding space drift on the shipped image"
            )
        snr = window.get("snr_db")
        if snr is None or abs(snr - reference["snr_db"]) > SNR_TOLERANCE_DB:
            fail(f"titanet snr_db {snr} vs reference {reference['snr_db']}")
        print(f"ok: titanet embedding cosine {cosine:.6f} vs CUDA reference")

    if args.pyannote:
        check_healthz(
            args.pyannote, "pyannote", {"device": "cpu", "model_loaded": True}
        )
        result = _request(
            f"{args.pyannote}/v1/diarize",
            {"path": "diarize-3speaker.wav", "min_speakers": 1, "max_speakers": 5},
        )
        if result.get("num_speakers") != 3:
            fail(f"pyannote found {result.get('num_speakers')} speakers, expected 3")
        print(f"ok: pyannote diarized 3 speakers, {len(result['turns'])} turns")

    print("SMOKE PASS")


if __name__ == "__main__":
    main()
