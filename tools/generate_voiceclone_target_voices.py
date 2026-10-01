#!/usr/bin/env -S uv run --script
# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "qwen-tts==0.1.1",
#   "transformers==4.57.3",
#   "accelerate==1.12.0",
#   "torch==2.8.0",
#   "torchaudio==2.8.0",
#   "numpy==2.2.6",
#   "librosa==0.11.0",
#   "huggingface-hub==0.36.0",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# torchaudio = { index = "pytorch-cu128" }
# ///
"""Generate the voiceclone target-voice candidates (#664). Maintainer tool.

Runs Qwen3-TTS-12Hz-1.7B-VoiceDesign on one CUDA GPU in bf16 and turns every
(voice, seed) in ``tools/voiceclone_target_voices.spec.json`` into a candidate:
a 24 kHz mono PCM16 WAV post-processed to the Chatterbox VC reference format,
plus a JSON record. Voices come from the spec's attribute-only text
descriptions; no recording of anyone is used.

    uv run --script tools/generate_voiceclone_target_voices.py \\
        --out-dir candidates/ --tool-git-sha "$(git rev-parse HEAD)"

Its dependencies live only in the PEP 723 header above and the sibling
``.lock`` file (``uv lock --script``); they never enter Voxint's
``pyproject.toml`` or ``uv.lock``. It is Linux x86_64 + CUDA only.

Fail-closed steps:

* the HF snapshot is downloaded at the pinned revision and every weight file's
  sha256 must match the spec before the model loads; then ``HF_HUB_OFFLINE=1``;
* dtype is bf16 (fp16 gives NaNs on this model), attention is ``sdpa``
  (flash-attn has a NaN report upstream);
* a candidate that comes out non-finite, silent or shorter than 8.0 s after
  trimming gets a rejection record, never padding.

The WAV sha256 is the artifact. Seed, sampling parameters, device and versions
are provenance, not a reproducibility promise: sampling on another GPU or
driver can give different audio.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import voiceclone_target_voice_core as core

DEFAULT_SPEC = Path(__file__).resolve().parent / "voiceclone_target_voices.spec.json"
RECORD_SCHEMA_VERSION = 1
ATTN_IMPLEMENTATION = "sdpa"
VERSION_PACKAGES = (
    "qwen-tts",
    "transformers",
    "accelerate",
    "torch",
    "torchaudio",
    "numpy",
    "librosa",
    "huggingface-hub",
)


@dataclass(frozen=True)
class Job:
    voice_id: str
    seed: int
    description: str

    @property
    def candidate_id(self) -> str:
        return f"{self.voice_id}-s{self.seed}"


def plan_jobs(
    spec: Mapping[str, Any],
    voices: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
) -> list[Job]:
    """Every (voice, seed) in spec order, optionally narrowed (the spike runs
    one voice and one seed). Unknown filters fail closed."""
    known_voices = {v["id"] for v in spec["voices"]}
    known_seeds = {s for v in spec["voices"] for s in v["seeds"]}
    for vid in voices or ():
        if vid not in known_voices:
            raise core.SpecError(f"unknown voice {vid!r}")
    for seed in seeds or ():
        if seed not in known_seeds:
            raise core.SpecError(f"seed {seed} is not in the spec")
    jobs = [
        Job(v["id"], s, v["description"])
        for v in spec["voices"]
        if not voices or v["id"] in voices
        for s in v["seeds"]
        if not seeds or s in seeds
    ]
    if not jobs:
        raise core.SpecError("the voice/seed filter selects nothing")
    return jobs


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_weights(snapshot: Path, expected: Mapping[str, str]) -> dict[str, str]:
    """Every pinned weight file must exist in the snapshot with its sha256."""
    for name, digest in expected.items():
        path = snapshot / name
        if not path.is_file():
            raise core.SpecError(f"weight file {name} is missing from {snapshot}")
        actual = file_sha256(path)
        if actual != digest:
            raise core.SpecError(f"weight file {name}: sha256 {actual} != pinned {digest}")
    return dict(expected)


def median_f0(pcm: np.ndarray, sample_rate: int, fmin: float, fmax: float) -> float | None:
    """Median F0 over voiced frames (librosa pyin), or None if nothing is voiced."""
    import librosa  # type: ignore[import-not-found]  # generator env only

    f0, voiced, _ = librosa.pyin(
        pcm.astype(np.float32) / core.PCM16_FULL_SCALE, fmin=fmin, fmax=fmax, sr=sample_rate
    )
    values = f0[voiced & np.isfinite(f0)]
    return float(np.median(values)) if values.size else None


def write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def build_record(
    job: Job,
    spec: Mapping[str, Any],
    *,
    wav_name: str | None,
    wav_bytes: bytes | None,
    raw: Mapping[str, Any],
    f0_hz: float | None,
    rejection: str | None,
    environment: Mapping[str, Any],
    tool: Mapping[str, Any],
) -> dict[str, Any]:
    """One candidate's provenance record. Format fields come from the written
    WAV bytes, never from the parameters that were meant to produce them."""
    record: dict[str, Any] = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "candidate_id": job.candidate_id,
        "voice_id": job.voice_id,
        "seed": job.seed,
        "description": job.description,
        "text": spec["text"],
        "language": spec["generator"]["language"],
        "sampling": dict(spec["sampling"]),
        "dsp": dict(spec["dsp"]),
        "generator": dict(spec["generator"]),
        "raw_output": dict(raw),
        "environment": dict(environment),
        "tool": dict(tool),
        "status": "rejected" if rejection else "candidate",
    }
    if rejection:
        record["rejection"] = rejection
        return record
    if wav_name is None or wav_bytes is None:
        raise ValueError("an accepted candidate needs its WAV")
    fmt = core.read_wav_format(wav_bytes)
    problems = core.format_problems(fmt, core.Dsp.from_spec(spec))
    if problems:
        raise ValueError(f"{job.candidate_id}: written WAV misses the format: {problems}")
    record.update(
        wav=wav_name,
        wav_sha256=core.sha256_hex(wav_bytes),
        format=dataclasses.asdict(fmt),
        median_f0_hz=f0_hz,
    )
    return record


def sha256sums_lines(records: Sequence[Mapping[str, Any]]) -> str:
    """``sha256sum -c`` input for the accepted candidates' WAVs."""
    rows = sorted((r["wav"], r["wav_sha256"]) for r in records if r["status"] == "candidate")
    return "".join(f"{digest}  {name}\n" for name, digest in rows)


# --- GPU path (torch / qwen-tts imported lazily) ----------------------------------


def _versions() -> dict[str, str]:
    from importlib.metadata import version

    return {name: version(name) for name in VERSION_PACKAGES}


def environment_record(dtype: str) -> dict[str, Any]:
    import torch  # type: ignore[import-not-found]  # generator env only

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this tool never falls back to CPU or MPS")
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader", "-i", "0"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    name, driver = (part.strip() for part in smi.split(",", 1))
    return {
        "device": "cuda:0",
        "gpu": torch.cuda.get_device_name(0),
        "gpu_nvidia_smi": name,
        "driver_version": driver,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "dtype": dtype,
        "attn_implementation": ATTN_IMPLEMENTATION,
        "python": platform.python_version(),
        "versions": _versions(),
    }


def load_model(spec: Mapping[str, Any], cache_dir: Path) -> tuple[Any, Path]:
    import torch
    from huggingface_hub import snapshot_download

    gen = spec["generator"]
    snapshot = Path(
        snapshot_download(gen["model_id"], revision=gen["hf_revision"], cache_dir=str(cache_dir))
    )
    verify_weights(snapshot, gen["weights_sha256"])
    os.environ["HF_HUB_OFFLINE"] = "1"
    from qwen_tts import Qwen3TTSModel  # type: ignore[import-not-found]

    model = Qwen3TTSModel.from_pretrained(
        str(snapshot),
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation=ATTN_IMPLEMENTATION,
    )
    return model, snapshot


def generate_one(model: Any, job: Job, spec: Mapping[str, Any]) -> tuple[np.ndarray, int, float]:
    import torch

    torch.manual_seed(job.seed)
    torch.cuda.manual_seed_all(job.seed)
    start = time.monotonic()
    with torch.inference_mode():
        wavs, sample_rate = model.generate_voice_design(
            text=spec["text"],
            instruct=job.description,
            language=spec["generator"]["language"],
            non_streaming_mode=True,
            **spec["sampling"],
        )
    torch.cuda.synchronize()
    return np.asarray(wavs[0], dtype=np.float64), int(sample_rate), time.monotonic() - start


def run(args: argparse.Namespace) -> int:
    spec = core.load_spec(args.spec)
    dsp = core.Dsp.from_spec(spec)
    jobs = plan_jobs(spec, args.voice, args.seed)
    out: Path = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    environment = environment_record(spec["generator"]["dtype"])
    model, _snapshot = load_model(spec, args.cache_dir)
    tool = {
        "name": "tools/generate_voiceclone_target_voices.py",
        "git_sha": args.tool_git_sha,
        "spec_sha256": file_sha256(args.spec),
    }
    analysis = spec["analysis"]
    records = []
    for job in jobs:
        audio, sample_rate, seconds = generate_one(model, job, spec)
        raw = {
            "sample_rate": sample_rate,
            "n_samples": int(audio.size),
            "peak": float(np.max(np.abs(audio))) if audio.size else 0.0,
            "finite": bool(np.all(np.isfinite(audio))),
            "generation_seconds": round(seconds, 3),
            "generated_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        }
        try:
            pcm = core.postprocess(audio, sample_rate, dsp)
        except core.CandidateRejected as exc:
            record = build_record(
                job,
                spec,
                wav_name=None,
                wav_bytes=None,
                raw=raw,
                f0_hz=None,
                rejection=str(exc),
                environment=environment,
                tool=tool,
            )
        else:
            wav_name = f"{job.candidate_id}.wav"
            data = core.encode_wav(pcm, dsp.sample_rate)
            write_atomic(out / wav_name, data)
            f0 = median_f0(pcm, dsp.sample_rate, analysis["f0_fmin_hz"], analysis["f0_fmax_hz"])
            record = build_record(
                job,
                spec,
                wav_name=wav_name,
                wav_bytes=data,
                raw=raw,
                f0_hz=f0,
                rejection=None,
                environment=environment,
                tool=tool,
            )
        write_atomic(
            out / f"{job.candidate_id}.json",
            (json.dumps(record, indent=2) + "\n").encode("utf-8"),
        )
        records.append(record)
        outcome = record.get("wav_sha256", record.get("rejection"))
        print(f"{job.candidate_id}: {record['status']} {outcome}")
    write_atomic(out / "candidates.sha256", sha256sums_lines(records).encode("utf-8"))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate voiceclone target-voice candidates.")
    p.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, default=Path("hf-cache"))
    p.add_argument("--tool-git-sha", required=True, help="commit of this tool and its spec")
    p.add_argument("--voice", action="append", help="limit to this voice id (repeatable)")
    p.add_argument("--seed", type=int, action="append", help="limit to this seed (repeatable)")
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
