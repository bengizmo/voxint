#!/usr/bin/env python3
"""Freeze the v1 **CT2-CPU** whisper baseline for the Metal bakeoff (issue #33).

This is the load-bearing numerics oracle of #33: the frozen CT2-CPU transcript
that the Metal candidate (mlx-whisper, then whisper.cpp / CT2-MPS) is measured
against for WER/CER, boundary drift, zero-insertion, and confidence conformance
(``docs/gpu-contracts.md`` → "Whisper Metal ASR engine (#33)"). It MUST be
captured from the **UNMODIFIED** ``services/whisper/app/transcription.py`` —
before the Slice-2 VAD-lift refactor — or the candidate is later measured
against an accidental new CT2 segmentation instead of the shipped path.

Unlike the Metal *reference generator* (``tools/generate_parity_references.py``,
which REFUSES to commit MPS/CoreML output — those are per-chip evidence, never a
committed oracle), CT2-on-CPU int8 is deterministic run-to-run, so its output is
a legitimate committed oracle. It is NOT asserted byte-identical *across*
machines: CTranslate2 selects its ISA/BLAS backend dynamically and the arm64
build differs from x86 (docs/gpu-contracts.md whisper row). The full runtime
identity is therefore pinned in ``meta`` and every committed entry is bound to
its corpus ``sha256``; a cross-machine re-run is a *measured* re-verification,
not a diff.

What it records, per manifest entry, for BOTH decode paths the frozen engine
exposes (``vad_true`` = production path via ``BatchedInferencePipeline`` +
Silero; ``vad_false`` = raw ``model.transcribe``) — the pre-registered gate
scores CT2 self-parity at both:

  * transcript, per-segment {start, end, text, confidence, suspect*}, per-word
    {start, end, word, confidence} (the gate promises word-timestamp drift),
    overall confidence, suspect_segment_count, duration, language.

Licensing split (enforced by ``tests/contracts/test_bakeoff_baseline.py``):

  * **AMI** (CC-BY-4.0) + **synthetic** (CC0) → committed to
    ``tests/parity/fixtures/references/ct2-cpu-metal/transcribe.json``. Synthetic
    is required by the zero-insertion gate ("≤ baseline + 0" needs a baseline).
  * **TED-LIUM 3** (CC-BY-NC-ND-3.0) → metrics-only, written to an UNCOMMITTED
    path: per-variant ``hypothesis_sha256`` + counts + confidence, never text,
    tokens, or word strings.

Every committed entry is captured across two warm passes; any text / segment /
word / timestamp difference between passes fails the run closed (CT2-CPU must be
deterministic here). Confidence float drift is recorded, not averaged.

Maintainer run (native services already up via ``scripts/metal/voxint-metal.sh``
with the whisper env's ``BATCH_SIZE=4`` — mirroring the CPU image, the pinned
oracle batch size):

    python tools/prepare_bakeoff_corpus.py prepare      # ensure work-dir WAVs exist
    python tools/generate_bakeoff_baseline.py           # capture + write

The tool stages the 45 WAVs as hardlinks inside the live service ``MEDIA_ROOT``
(real dir entries — the service rejects symlink escapes), sha-verifies each
against the manifest, then removes the staging dir on exit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
MANIFEST_DEFAULT = REPO / "tests" / "parity" / "fixtures" / "bakeoff" / "manifest.json"
BAKEOFF_DIR = MANIFEST_DEFAULT.parent
OUT_DIR_DEFAULT = REPO / "tests" / "parity" / "fixtures" / "references" / "ct2-cpu-metal"
TRANSCRIPTION_PY = REPO / "services" / "whisper" / "app" / "transcription.py"
WORK_DIR_DEFAULT = Path.home() / ".voxint-metal" / "bakeoff" / "work"
WHISPER_VENV_PY_DEFAULT = Path.home() / ".voxint-metal" / "venvs" / "whisper" / "bin" / "python"
WEIGHTS_MANIFEST_DEFAULT = (
    Path.home() / ".voxint-metal" / "models" / "whisper" / ".voxint-manifest.sha256"
)

# The pinned oracle batch size: mirror the CPU image (Dockerfile.cpu), NOT the
# GPU/ROCm app default of 16. Verified against the running service via the
# rendered launchd plist below; recorded in meta.decode_config.
ORACLE_BATCH_SIZE = 4
EXPECTED_DEVICE = "cpu"
EXPECTED_ENGINE = "faster-whisper"

# The frozen decode config — pinned in transcription.py, reproduced here for the
# recorded oracle identity (Slice-2 will expose a decode_config_hash on
# /healthz; until then this is a declared, contract-anchored claim).
FROZEN_DECODE_CONFIG: dict[str, Any] = {
    "temperature": 0.0,
    "condition_on_previous_text": False,
    "compression_ratio_threshold": 2.4,
    "no_speech_threshold": 0.6,
    "hallucination_silence_threshold": 2.0,
    "log_prob_threshold": -1.0,
    "word_timestamps": True,
    "compute_type": "int8",
    "cpu_threads": 4,
    "num_workers": 1,
    "batch_size": ORACLE_BATCH_SIZE,
    "language": "en",
}

# Run-to-run timestamp tolerance (seconds) and confidence tolerance for the
# determinism gate. CT2-CPU should be exactly reproducible; these bound
# unexpected float wobble without silently averaging it away.
TS_TOL_S = 1e-3
CONF_TOL = 1e-4


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


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def _verify_frozen_transcription_py() -> dict[str, Any]:
    """Fail closed unless transcription.py is committed and unmodified.

    The oracle is only valid if captured from the shipped decode path.
    """
    dirty = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", str(TRANSCRIPTION_PY)],
        cwd=REPO,
    ).returncode
    if dirty != 0:
        raise SystemExit(
            f"{TRANSCRIPTION_PY.relative_to(REPO)} has uncommitted changes — the "
            "CT2 baseline must come from the UNMODIFIED shipped decode path. "
            "Commit/stash the seam refactor first."
        )
    rel = TRANSCRIPTION_PY.relative_to(REPO)
    return {
        "git_head": _git("rev-parse", "HEAD"),
        "transcription_py_blob_sha": _git("rev-parse", f"HEAD:{rel}"),
        "transcription_py_content_sha256": _sha256_file(TRANSCRIPTION_PY),
        "working_tree_clean_for_transcription_py": True,
    }


def _verify_running_batch_size() -> str:
    """Best-effort proof the LIVE service runs the oracle batch size.

    /healthz does not expose batch_size (Slice-2 adds decode_config_hash), so we
    read the rendered launchd plist that started the service. If it is present it
    is authoritative for this host; if absent we warn rather than block (the
    value is contract-tested in the launcher).
    """
    plist = Path.home() / ".voxint-metal" / "run" / "com.voxint.metal.whisper.plist"
    if not plist.exists():
        print(
            f"WARNING: {plist} not found — cannot verify the live service batch "
            f"size; recording the pinned oracle value {ORACLE_BATCH_SIZE} on trust",
            file=sys.stderr,
        )
        return "unverified (plist absent)"
    text = plist.read_text()
    marker = f"<key>BATCH_SIZE</key><string>{ORACLE_BATCH_SIZE}</string>"
    if marker not in text:
        raise SystemExit(
            f"live whisper plist {plist} does not pin BATCH_SIZE={ORACLE_BATCH_SIZE}; "
            "re-run scripts/metal/voxint-metal.sh up (the oracle batch size drifted)"
        )
    return f"verified via {plist.name}"


def _runtime_identity(healthz: dict[str, Any], venv_python: Path) -> dict[str, Any]:
    """Pin the full transitive runtime CT2 output depends on."""
    ident: dict[str, Any] = {
        "engine": healthz.get("engine"),
        "engine_version": healthz.get("engine_version"),
        "runtime": healthz.get("runtime"),
        "runtime_version": healthz.get("runtime_version"),
        "pyav_version": None,
        "onnxruntime_version": None,
        "ffmpeg_version": None,
    }
    if venv_python.exists():
        try:
            out = subprocess.check_output(
                [
                    str(venv_python),
                    "-c",
                    "import av,onnxruntime;print(av.__version__);print(onnxruntime.__version__)",
                ],
                text=True,
            ).split()
            ident["pyav_version"], ident["onnxruntime_version"] = out[0], out[1]
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"WARNING: could not read pyav/onnxruntime versions: {exc}", file=sys.stderr)
    else:
        print(f"WARNING: whisper venv python {venv_python} absent; runtime versions partial",
              file=sys.stderr)
    try:
        first = subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]
        ident["ffmpeg_version"] = first.replace("ffmpeg version ", "").split(" ", 1)[0]
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"WARNING: could not read ffmpeg version: {exc}", file=sys.stderr)
    return ident


def _model_identity(weights_manifest: Path) -> dict[str, Any]:
    ident: dict[str, Any] = {"name": "large-v2", "revision": None, "weights_manifest_sha256": None}
    if weights_manifest.exists():
        text = weights_manifest.read_text()
        for line in text.splitlines():
            if line.startswith("# revision:"):
                ident["revision"] = line.split(":", 1)[1].strip()
                break
        ident["weights_manifest_sha256"] = _sha256_file(weights_manifest)
    else:
        print(f"WARNING: weights manifest {weights_manifest} absent; model revision unrecorded",
              file=sys.stderr)
    return ident


def _entry_wav_source(entry: dict[str, Any], work_dir: Path) -> Path:
    """Map a manifest entry to its prepared WAV on disk."""
    acq = entry["acquire"]
    kind = acq["kind"]
    if kind == "ami_range":
        return work_dir / "ami" / f"{acq['meeting']}.{acq['agent']}.wav"
    if kind == "ted_window":
        return work_dir / "ted" / f"{entry['upstream_id']}.wav"
    if kind == "committed":
        return BAKEOFF_DIR / str(acq["path"])
    raise SystemExit(f"unknown acquire.kind {kind!r} for {entry['upstream_id']}")


def _stage_corpus(
    manifest: dict[str, Any], work_dir: Path, media_root: Path
) -> tuple[Path, dict[str, str]]:
    """Hardlink every WAV into a fresh dir inside MEDIA_ROOT; sha-verify each.

    Returns (staging_dir, {upstream_id: media-root-relative path}). Hardlinks are
    real directory entries (not symlinks), so the service's resolve_media_path
    (.resolve() + is_relative_to) accepts them, and the bytes — hence the
    sha256 — are shared with the source.
    """
    staging = media_root / "bakeoff-capture"
    if staging.exists():
        for p in sorted(staging.rglob("*"), reverse=True):
            p.unlink() if p.is_file() else p.rmdir()
        staging.rmdir()
    rel_paths: dict[str, str] = {}
    for entry in manifest["files"]:
        src = _entry_wav_source(entry, work_dir)
        if not src.exists():
            raise SystemExit(
                f"missing prepared WAV {src} for {entry['upstream_id']} — run "
                "`python tools/prepare_bakeoff_corpus.py prepare` first"
            )
        actual = _sha256_file(src)
        if actual != entry["sha256"]:
            raise SystemExit(
                f"{src} sha256 {actual} != manifest {entry['sha256']} for "
                f"{entry['upstream_id']} — corpus is not the frozen one"
            )
        rel = Path("bakeoff-capture") / entry["dataset"] / f"{entry['upstream_id']}.wav"
        dst = media_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst.unlink()
        os.link(src, dst)
        rel_paths[entry["upstream_id"]] = str(rel)
    return staging, rel_paths


def _cleanup_staging(staging: Path) -> None:
    if not staging.exists():
        return
    for p in sorted(staging.rglob("*"), reverse=True):
        p.unlink() if p.is_file() else p.rmdir()
    staging.rmdir()


def _canonical_response(resp: dict[str, Any]) -> dict[str, Any]:
    """The recorded projection of one transcribe response."""
    return {
        "language": resp["language"],
        "duration_seconds": resp["duration_seconds"],
        "confidence": resp["confidence"],
        "suspect_segment_count": resp.get("suspect_segment_count", 0),
        "transcript": resp["transcript"],
        "segments": resp["segments"],
        "words": resp["words"],
    }


def _compare_passes(a: dict[str, Any], b: dict[str, Any], label: str) -> tuple[float, float]:
    """Fail closed on any text/count/timestamp divergence; return max drift."""
    if a["transcript"] != b["transcript"]:
        raise SystemExit(f"{label}: transcript differs between warm passes (non-deterministic)")
    if len(a["segments"]) != len(b["segments"]):
        raise SystemExit(f"{label}: segment count differs between passes")
    if len(a["words"]) != len(b["words"]):
        raise SystemExit(f"{label}: word count differs between passes")
    max_ts = 0.0
    max_conf = 0.0
    for s1, s2 in zip(a["segments"], b["segments"], strict=True):
        if s1["text"] != s2["text"]:
            raise SystemExit(f"{label}: segment text differs between passes")
        for k in ("start_seconds", "end_seconds"):
            d = abs((s1[k] or 0.0) - (s2[k] or 0.0))
            max_ts = max(max_ts, d)
            if d > TS_TOL_S:
                raise SystemExit(f"{label}: segment {k} drift {d:.4g}s > {TS_TOL_S}s")
        c1, c2 = s1.get("confidence"), s2.get("confidence")
        if c1 is not None and c2 is not None:
            max_conf = max(max_conf, abs(c1 - c2))
    for w1, w2 in zip(a["words"], b["words"], strict=True):
        if w1["word"] != w2["word"]:
            raise SystemExit(f"{label}: word text differs between passes")
        for k in ("start_seconds", "end_seconds"):
            d = abs((w1[k] or 0.0) - (w2[k] or 0.0))
            max_ts = max(max_ts, d)
            if d > TS_TOL_S:
                raise SystemExit(f"{label}: word {k} drift {d:.4g}s > {TS_TOL_S}s")
    return max_ts, max_conf


def _transcribe_twice(
    asr_url: str, path: str, vad: bool, label: str
) -> tuple[dict[str, Any], float, float]:
    payload: dict[str, Any] = {"path": path, "language": "en"}
    if not vad:
        payload["vad_filter"] = False
    r1 = _canonical_response(_http_json(f"{asr_url}/v1/transcribe", payload))
    r2 = _canonical_response(_http_json(f"{asr_url}/v1/transcribe", payload))
    max_ts, max_conf = _compare_passes(r1, r2, label)
    r1["request"] = payload
    return r1, max_ts, max_conf


def _atomic_write(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--asr-url", default="http://127.0.0.1:8022")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DEFAULT)
    parser.add_argument("--work-dir", type=Path, default=WORK_DIR_DEFAULT)
    parser.add_argument(
        "--media-root", type=Path, default=None,
        help="live service MEDIA_ROOT to stage into (default: $MEDIA_ROOT or repo media/)",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    parser.add_argument(
        "--ted-metrics-out", type=Path, default=WORK_DIR_DEFAULT / "ted-baseline-metrics.json",
        help="UNCOMMITTED TED metrics path (hypothesis_sha256 only, never text)",
    )
    parser.add_argument("--whisper-venv-python", type=Path, default=WHISPER_VENV_PY_DEFAULT)
    parser.add_argument("--weights-manifest", type=Path, default=WEIGHTS_MANIFEST_DEFAULT)
    args = parser.parse_args()

    media_root = args.media_root or Path(os.environ.get("MEDIA_ROOT", REPO / "media"))
    media_root = media_root.resolve()
    if not media_root.is_dir():
        raise SystemExit(
            f"MEDIA_ROOT {media_root} is not a directory (start services / pass --media-root)"
        )

    manifest = json.loads(args.manifest.read_text())
    manifest_sha = _sha256_file(args.manifest)

    # ---- fail-closed preflight -------------------------------------------
    code_identity = _verify_frozen_transcription_py()
    health = _http_json(f"{args.asr_url}/healthz")
    if health.get("status") != "ok" or not health.get("model_loaded"):
        raise SystemExit(f"whisper /healthz not ready: {health}")
    if health.get("device") != EXPECTED_DEVICE:
        raise SystemExit(f"whisper device={health.get('device')!r}, expected {EXPECTED_DEVICE!r}")
    if health.get("engine") != EXPECTED_ENGINE:
        raise SystemExit(f"whisper engine={health.get('engine')!r}, expected {EXPECTED_ENGINE!r}")
    batch_size_proof = _verify_running_batch_size()

    meta: dict[str, Any] = {
        "generated_on": date.today().isoformat(),
        "tier": "ct2-cpu-metal",
        "note": (
            "Frozen v1 CT2-CPU whisper baseline (issue #33). Committed oracle for "
            "AMI+synthetic; deterministic run-to-run on this runtime, NOT asserted "
            "byte-identical across machines (CT2 selects ISA/BLAS dynamically). "
            "A cross-machine re-run is a measured re-verification."
        ),
        "code": code_identity,
        "decode_config": FROZEN_DECODE_CONFIG,
        "batch_size_proof": batch_size_proof,
        "service_healthz": health,
        "runtime": _runtime_identity(health, args.whisper_venv_python),
        "model": _model_identity(args.weights_manifest),
        "host": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "macos_version": platform.mac_ver()[0],
            "python_version": platform.python_version(),
        },
        "corpus": {"manifest_sha256": manifest_sha, "num_entries": len(manifest["files"])},
        "determinism": {"warm_passes": 2, "ts_tol_s": TS_TOL_S, "conf_tol": CONF_TOL},
    }

    # ---- stage + capture --------------------------------------------------
    staging, rel_paths = _stage_corpus(manifest, args.work_dir, media_root)
    committed: dict[str, Any] = {}
    ted: dict[str, Any] = {}
    max_ts_drift = 0.0
    max_conf_drift = 0.0
    try:
        for entry in manifest["files"]:
            uid = entry["upstream_id"]
            dataset = entry["dataset"]
            path = rel_paths[uid]
            key = f"{dataset}/{uid}"
            variants: dict[str, Any] = {}
            for vad, vname in ((True, "vad_true"), (False, "vad_false")):
                resp, mts, mcf = _transcribe_twice(args.asr_url, path, vad, f"{key} {vname}")
                max_ts_drift = max(max_ts_drift, mts)
                max_conf_drift = max(max_conf_drift, mcf)
                variants[vname] = resp
            if dataset == "tedlium3":
                # NC-ND: metrics only, never text/tokens/word strings.
                ted[key] = {
                    "dataset": dataset,
                    "upstream_id": uid,
                    "audio_sha256": entry["sha256"],
                    "variants": {
                        vn: {
                            "hypothesis_sha256": _sha256_text(v["transcript"]),
                            "num_segments": len(v["segments"]),
                            "num_words": len(v["words"]),
                            "confidence": v["confidence"],
                            "duration_seconds": v["duration_seconds"],
                            "language": v["language"],
                        }
                        for vn, v in variants.items()
                    },
                }
            else:
                committed[key] = {
                    "dataset": dataset,
                    "upstream_id": uid,
                    "audio_sha256": entry["sha256"],
                    "gold_transcript_sha256": entry.get("transcript_sha256"),
                    "duration_s_manifest": entry["duration_s"],
                    "variants": variants,
                }
            print(f"  captured {key}: "
                  f"vad_true {len(variants['vad_true']['segments'])} seg / "
                  f"vad_false {len(variants['vad_false']['segments'])} seg", flush=True)
    finally:
        _cleanup_staging(staging)

    meta["determinism"]["max_timestamp_drift_s"] = max_ts_drift
    meta["determinism"]["max_confidence_drift"] = max_conf_drift

    _atomic_write(args.out_dir / "transcribe.json", {"meta": meta, "entries": committed})
    _atomic_write(args.ted_metrics_out, {"meta": meta, "entries": ted})

    print(
        f"\ntranscribe.json: {len(committed)} committed entries (AMI+synthetic); "
        f"TED metrics: {len(ted)} entries → {args.ted_metrics_out}\n"
        f"max ts drift {max_ts_drift:.4g}s, max conf drift {max_conf_drift:.4g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
