#!/usr/bin/env python3
"""Maintainer-only TTS spoof generation from S5 bona fide turn clips."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
import wave
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from synthdetect_corpus import (  # noqa: E402
    CANONICAL_SAMPLE_RATE,
    MANIFEST_SCHEMA_VERSION,
    ClipEntry,
    CorpusError,
    Manifest,
    load_manifest,
    payload_sha_and_count,
    read_canonical_wav_payload,
    resolve_clip_path,
    validate_clip,
    write_canonical_wav,
)
from synthdetect_sources import SELECTION_SEED  # noqa: E402

WHISPER_MODEL = "Systran/faster-whisper-large-v2"
TEXT_SOURCE = "whisper-large-v2-transcript"
DEFAULT_WHISPER_URL = "http://127.0.0.1:8025/transcribe/process"
ELEVENLABS_MODEL = "eleven_multilingual_v2"
ELEVENLABS_OUTPUT_FORMAT = "pcm_16000"
CHATTERBOX_REPO_ID = "ResembleAI/chatterbox"
CHATTERBOX_WEIGHT_FILES = (
    "ve.safetensors",
    "t3_cfg.safetensors",
    "s3gen.safetensors",
    "conds.pt",
)
GOOGLE_TTS_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"
GOOGLE_TTS_DEFAULT_VOICE = "en-US-Neural2-D"
GENERATORS = ("piper", "chatterbox", "elevenlabs", "google")
LICENSES = {
    "piper": "MIT",
    "chatterbox": "MIT",
    "elevenlabs": "LicenseRef-ElevenLabs-TOS",
    "google": "LicenseRef-Google-Cloud-TOS",
}


class TtsGenerateError(Exception):
    """A fail-closed generation, input, or provenance error."""


@dataclass(frozen=True)
class GeneratorIdentity:
    name: str
    version: str
    checkpoint_sha: str | None
    voice: str
    seed: str | None


@dataclass(frozen=True)
class GenerateResult:
    generator: str
    eligible: int
    generated: int
    resumed: int
    manifest_sha256: str
    out_dir: str


@dataclass(frozen=True)
class SynthesizedAudio:
    samples: Any
    sample_rate: int


Synthesize = Callable[[ClipEntry, Path, str], SynthesizedAudio | bytes]


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TtsGenerateError(f"{path}: cannot read valid JSON: {exc}") from exc


def _write_json_atomic(path: Path, obj: Any) -> bytes:
    raw = (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_bytes(raw)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return raw


def _package_version(distribution: str) -> str:
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise TtsGenerateError(
            f"required distribution {distribution!r} is not installed in this venv"
        ) from exc
    if not version.strip():
        raise TtsGenerateError(f"distribution {distribution!r} has an empty version")
    return version


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise TtsGenerateError(f"{path}: cannot hash model weights: {exc}") from exc
    return digest.hexdigest()


def _chatterbox_checkpoint_sha() -> tuple[str, dict[str, str]]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise TtsGenerateError("chatterbox requires huggingface-hub") from exc
    files: dict[str, str] = {}
    bundle = hashlib.sha256()
    for filename in CHATTERBOX_WEIGHT_FILES:
        try:
            path = Path(hf_hub_download(repo_id=CHATTERBOX_REPO_ID, filename=filename))
        except Exception as exc:
            raise TtsGenerateError(
                f"cannot resolve Chatterbox checkpoint file {filename!r}: {exc}"
            ) from exc
        file_sha = _sha256_file(path)
        files[filename] = file_sha
        bundle.update(filename.encode("utf-8") + b"\0" + bytes.fromhex(file_sha))
    return bundle.hexdigest(), files


def _eligible_clips(
    manifest: Manifest, *, eval_only: bool = False
) -> tuple[ClipEntry, ...]:
    if manifest.schema_version != MANIFEST_SCHEMA_VERSION:
        raise TtsGenerateError(
            f"input must be a v{MANIFEST_SCHEMA_VERSION} synthesis manifest, "
            f"got v{manifest.schema_version}"
        )
    candidates: list[ClipEntry] = []
    for clip in manifest.clips:
        if clip.label != "bona_fide" or clip.acquire is None:
            continue
        try:
            acquire = json.loads(clip.acquire)
        except (json.JSONDecodeError, TypeError) as exc:
            raise TtsGenerateError(
                f"clip {clip.clip_id!r}: invalid acquire JSON: {exc}"
            ) from exc
        if acquire.get("kind") != "turn":
            continue
        if eval_only and clip.split != "eval":
            continue
        candidates.append(clip)
    eligible = tuple(sorted(candidates, key=lambda clip: clip.clip_id))
    if not eligible:
        raise TtsGenerateError(
            "manifest has no eligible bona_fide clips acquired as turns"
            + (" in the eval split" if eval_only else "")
        )
    return eligible


def _verify_parent(clip: ClipEntry, audio_root: Path) -> Path:
    path = resolve_clip_path(clip, roots=(audio_root,))
    payload = read_canonical_wav_payload(path)
    sha, count = payload_sha_and_count(payload)
    if sha != clip.sha256:
        raise TtsGenerateError(
            f"clip {clip.clip_id!r}: PCM sha256 {sha} does not match manifest {clip.sha256}"
        )
    duration = count / CANONICAL_SAMPLE_RATE
    if not math.isclose(duration, clip.duration_s, rel_tol=0.0, abs_tol=1e-12):
        raise TtsGenerateError(
            f"clip {clip.clip_id!r}: measured duration {duration} does not match "
            f"manifest {clip.duration_s}"
        )
    return path


def _multipart_transcription(
    path: Path, url: str, timeout_s: float, *, max_retries: int = 5
) -> str | None:
    try:
        wav_bytes = path.read_bytes()
    except OSError as exc:
        raise TtsGenerateError(f"{path}: cannot read for transcription: {exc}") from exc
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        boundary = f"----synthdetect-{uuid.uuid4().hex}"
        parts: list[bytes] = [
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"audio_file\"; "
            f"filename=\"{path.name}\"\r\nContent-Type: audio/wav\r\n\r\n".encode(),
            wav_bytes,
            f"\r\n--{boundary}--\r\n".encode("ascii"),
        ]
        request = urllib.request.Request(
            url,
            data=b"".join(parts),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                body = response.read()
                parsed = json.loads(body)
                text: str = parsed.get("transcript", "").strip()
            return text or None
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < max_retries:
                delay = min(2 ** (attempt + 1), 60)
                sys.stderr.write(f"429 on {path.name}, retry in {delay}s\n")
                time.sleep(delay)
                last_exc = exc
                continue
            raise TtsGenerateError(
                f"Whisper transcription failed for {path}: {exc}"
            ) from exc
        except (
            urllib.error.URLError, UnicodeDecodeError, TimeoutError, json.JSONDecodeError
        ) as exc:
            raise TtsGenerateError(
                f"Whisper transcription failed for {path}: {exc}"
            ) from exc
    raise TtsGenerateError(
        f"Whisper transcription failed for {path} after {max_retries} retries: {last_exc}"
    )


def _load_transcripts(path: Path, eligible_ids: set[str]) -> dict[str, str]:
    if not path.exists():
        return {}
    transcripts: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise TtsGenerateError(f"{path}: cannot read transcript cache: {exc}") from exc
    for number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TtsGenerateError(f"{path}:{number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict) or set(row) != {"clip_id", "text"}:
            raise TtsGenerateError(f"{path}:{number}: expected only clip_id and text")
        clip_id, text = row["clip_id"], row["text"]
        if not isinstance(clip_id, str):
            raise TtsGenerateError(f"{path}:{number}: clip_id must be a string")
        if clip_id not in eligible_ids:
            continue
        if not isinstance(text, str) or not text.strip():
            raise TtsGenerateError(f"{path}:{number}: text must be non-empty")
        if clip_id in transcripts:
            raise TtsGenerateError(f"{path}:{number}: duplicate clip_id {clip_id!r}")
        transcripts[clip_id] = text.strip()
    return transcripts


def _write_transcripts(path: Path, transcripts: dict[str, str]) -> None:
    raw = "".join(
        json.dumps({"clip_id": clip_id, "text": transcripts[clip_id]}, sort_keys=True) + "\n"
        for clip_id in sorted(transcripts)
    ).encode("utf-8")
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_bytes(raw)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _load_skipped(path: Path) -> set[str]:
    if not path.exists():
        return set()
    skipped: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise TtsGenerateError(f"{path}: cannot read skip cache: {exc}") from exc
    for number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TtsGenerateError(f"{path}:{number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict) or "clip_id" not in row:
            raise TtsGenerateError(f"{path}:{number}: expected object with clip_id")
        clip_id = row["clip_id"]
        if not isinstance(clip_id, str):
            raise TtsGenerateError(f"{path}:{number}: clip_id must be a string")
        skipped.add(clip_id)
    return skipped


def _append_skipped(path: Path, clip_id: str, reason: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"clip_id": clip_id, "reason": reason}, sort_keys=True) + "\n")


def transcribe_eligible(
    eligible: tuple[ClipEntry, ...], audio_root: Path, out_dir: Path, url: str, timeout_s: float
) -> tuple[dict[str, str], dict[str, Path]]:
    paths = {clip.clip_id: _verify_parent(clip, audio_root) for clip in eligible}
    cache_path = out_dir / "transcripts.jsonl"
    skipped_path = out_dir / "transcripts_skipped.jsonl"
    transcripts = _load_transcripts(cache_path, set(paths))
    skipped = _load_skipped(skipped_path)
    for clip in eligible:
        if clip.clip_id in transcripts or clip.clip_id in skipped:
            continue
        text = _multipart_transcription(paths[clip.clip_id], url, timeout_s)
        if text is None:
            _append_skipped(skipped_path, clip.clip_id, "empty-transcript")
            skipped.add(clip.clip_id)
            sys.stderr.write(f"SKIP {clip.clip_id}: empty Whisper transcript\n")
            continue
        transcripts[clip.clip_id] = text
        _write_transcripts(cache_path, transcripts)
    return transcripts, paths


def _read_pcm_wav(path: Path) -> SynthesizedAudio:
    try:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            rate = wav.getframerate()
            compression = wav.getcomptype()
            frames = wav.readframes(wav.getnframes())
    except (OSError, wave.Error, EOFError) as exc:
        raise TtsGenerateError(f"{path}: generator did not produce a readable WAV: {exc}") from exc
    if channels != 1 or width != 2 or compression != "NONE" or rate <= 0 or not frames:
        raise TtsGenerateError(
            f"{path}: expected non-empty mono s16 PCM WAV, got "
            f"channels={channels}, width={width}, rate={rate}, compression={compression}"
        )
    try:
        import numpy as np
    except ImportError as exc:
        raise TtsGenerateError("audio conversion requires numpy") from exc
    return SynthesizedAudio(np.frombuffer(frames, dtype="<i2").copy(), rate)


def _canonical_payload(audio: SynthesizedAudio) -> bytes:
    try:
        import numpy as np
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise TtsGenerateError("audio conversion requires numpy and scipy") from exc
    samples = np.asarray(audio.samples)
    if samples.ndim == 2 and 1 in samples.shape:
        samples = samples.reshape(-1)
    if samples.ndim != 1 or samples.size == 0:
        raise TtsGenerateError(f"generator returned invalid audio shape {samples.shape}")
    if audio.sample_rate <= 0:
        raise TtsGenerateError(f"generator returned invalid sample rate {audio.sample_rate}")
    if np.issubdtype(samples.dtype, np.integer):
        info = np.iinfo(samples.dtype)
        scale = float(max(abs(info.min), info.max + 1))
        floating = samples.astype(np.float64) / scale
    else:
        floating = samples.astype(np.float64)
    if not np.isfinite(floating).all():
        raise TtsGenerateError("generator returned non-finite audio samples")
    if audio.sample_rate != CANONICAL_SAMPLE_RATE:
        divisor = math.gcd(audio.sample_rate, CANONICAL_SAMPLE_RATE)
        floating = resample_poly(
            floating,
            CANONICAL_SAMPLE_RATE // divisor,
            audio.sample_rate // divisor,
        )
    pcm = np.rint(np.clip(floating, -1.0, 32767.0 / 32768.0) * 32768.0).astype("<i2")
    payload = pcm.tobytes()
    payload_sha_and_count(payload)
    return payload


def _piper_synthesizer(model_path: Path, executable: str) -> Synthesize:
    def synthesize(_clip: ClipEntry, _parent_path: Path, text: str) -> SynthesizedAudio:
        descriptor, tmp_name = tempfile.mkstemp(prefix="synthdetect-piper-", suffix=".wav")
        os.close(descriptor)
        tmp = Path(tmp_name)
        argv = [executable, "--model", str(model_path), "--output_file", str(tmp)]
        try:
            proc = subprocess.run(
                argv, input=text + "\n", text=True, capture_output=True, timeout=300,
            )
            if proc.returncode != 0:
                raise TtsGenerateError(
                    f"piper exited {proc.returncode}: {proc.stderr.strip() or proc.stdout.strip()}"
                )
            return _read_pcm_wav(tmp)
        except subprocess.TimeoutExpired as exc:
            raise TtsGenerateError(
                f"piper timed out after 300s on {_clip.clip_id!r}"
            ) from exc
        except OSError as exc:
            raise TtsGenerateError(f"cannot execute piper: {exc}") from exc
        finally:
            tmp.unlink(missing_ok=True)

    return synthesize


def _chatterbox_synthesizer(seed: str) -> Synthesize:
    try:
        import perth
        import torch

        # Disable Chatterbox watermarking; must precede the chatterbox import.
        perth.PerthImplicitWatermarker = perth.DummyWatermarker

        chatterbox_module = importlib.import_module("chatterbox.tts")
        chatterbox_tts: Any = chatterbox_module.ChatterboxTTS
    except ImportError as exc:
        raise TtsGenerateError("chatterbox generation requires torch and chatterbox-tts") from exc
    model: Any = None

    def synthesize(clip: ClipEntry, parent_path: Path, text: str) -> SynthesizedAudio:
        nonlocal model
        if model is None:
            if not torch.cuda.is_available():
                raise TtsGenerateError("chatterbox requires an available CUDA device")
            try:
                model = chatterbox_tts.from_pretrained(device="cuda")
            except Exception as exc:
                raise TtsGenerateError(f"cannot load Chatterbox checkpoint: {exc}") from exc
        digest = hashlib.sha256(f"{seed}\0{clip.clip_id}".encode()).digest()
        clip_seed = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
        torch.manual_seed(clip_seed)
        torch.cuda.manual_seed_all(clip_seed)
        try:
            wav = model.generate(text, audio_prompt_path=str(parent_path))
        except Exception as exc:
            raise TtsGenerateError(
                f"Chatterbox synthesis failed for {clip.clip_id!r}: {exc}"
            ) from exc
        samples = wav.detach().cpu().numpy() if hasattr(wav, "detach") else wav
        return SynthesizedAudio(samples, int(model.sr))

    return synthesize


def _elevenlabs_synthesizer(api_key: str, voice_id: str, model_id: str) -> Synthesize:
    try:
        elevenlabs_module = importlib.import_module("elevenlabs")
        elevenlabs_client: Any = elevenlabs_module.ElevenLabs
    except ImportError as exc:
        raise TtsGenerateError("ElevenLabs generation requires the elevenlabs package") from exc
    client = elevenlabs_client(api_key=api_key)

    def synthesize(_clip: ClipEntry, _parent_path: Path, text: str) -> bytes:
        try:
            chunks: Iterable[bytes] = client.text_to_speech.convert(
                voice_id=voice_id,
                output_format=ELEVENLABS_OUTPUT_FORMAT,
                text=text,
                model_id=model_id,
            )
            payload = b"".join(chunks)
        except Exception as exc:
            raise TtsGenerateError(f"ElevenLabs synthesis failed: {exc}") from exc
        payload_sha_and_count(payload)
        return payload

    return synthesize


def _google_adc_token_and_project() -> tuple[str, str | None]:
    try:
        import google.auth
        import google.auth.transport.requests
    except ImportError as exc:
        raise TtsGenerateError(
            "google ADC requires google-auth: pip install google-auth"
        ) from exc
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    credentials, _ = google.auth.default(scopes=scopes)
    credentials.refresh(google.auth.transport.requests.Request())
    token: str | None = credentials.token
    if not token:
        raise TtsGenerateError(
            "ADC returned no access token; run: "
            "gcloud auth application-default login"
        )
    quota_project: str | None = getattr(credentials, "quota_project_id", None)
    return token, quota_project


def _google_synthesizer(
    voice_name: str, *, api_key: str | None = None
) -> Synthesize:
    token_box: list[str] = []
    project_box: list[str | None] = []

    def _url() -> str:
        if api_key:
            return f"{GOOGLE_TTS_URL}?key={api_key}"
        return GOOGLE_TTS_URL

    def _refresh_adc() -> None:
        token_box.clear()
        project_box.clear()
        token, project = _google_adc_token_and_project()
        token_box.append(token)
        project_box.append(project)

    def _headers() -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if not api_key:
            if not token_box:
                _refresh_adc()
            headers["Authorization"] = f"Bearer {token_box[0]}"
            if project_box and project_box[0]:
                headers["x-goog-user-project"] = project_box[0]
        return headers

    def synthesize(
        _clip: ClipEntry, _parent_path: Path, text: str
    ) -> bytes:
        body = json.dumps({
            "input": {"text": text},
            "voice": {
                "languageCode": voice_name[:5],
                "name": voice_name,
            },
            "audioConfig": {
                "audioEncoding": "LINEAR16",
                "sampleRateHertz": CANONICAL_SAMPLE_RATE,
            },
        }).encode("utf-8")
        last_exc: Exception | None = None
        for attempt in range(6):
            request = urllib.request.Request(
                _url(), data=body, headers=_headers(), method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=60) as resp:
                    parsed = json.loads(resp.read())
                audio_b64: str = parsed["audioContent"]
                wav_bytes = base64.b64decode(audio_b64)
                break
            except urllib.error.HTTPError as exc:
                if exc.code == 401 and not api_key:
                    _refresh_adc()
                    last_exc = exc
                    continue
                if exc.code == 429 and attempt < 5:
                    delay = min(2 ** (attempt + 1), 60)
                    time.sleep(delay)
                    last_exc = exc
                    continue
                raise TtsGenerateError(
                    f"Google TTS failed: {exc}"
                ) from exc
            except (
                urllib.error.URLError, json.JSONDecodeError
            ) as exc:
                raise TtsGenerateError(
                    f"Google TTS failed: {exc}"
                ) from exc
        else:
            raise TtsGenerateError(
                f"Google TTS failed after retries: {last_exc}"
            )
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
            if (
                wav.getnchannels() != 1
                or wav.getsampwidth() != 2
                or wav.getframerate() != CANONICAL_SAMPLE_RATE
            ):
                raise TtsGenerateError(
                    f"Google TTS returned unexpected format: "
                    f"ch={wav.getnchannels()} width={wav.getsampwidth()} "
                    f"rate={wav.getframerate()}"
                )
            payload = wav.readframes(wav.getnframes())
        payload_sha_and_count(payload)
        return payload

    return synthesize


def _domain(parent: ClipEntry) -> str:
    parts = parent.stratum.split("|")
    if len(parts) < 2 or not parts[-1]:
        raise TtsGenerateError(
            f"clip {parent.clip_id!r}: cannot derive domain from stratum {parent.stratum!r}"
        )
    return parts[-1]


def _clip_dict(
    parent: ClipEntry, identity: GeneratorIdentity, sha: str, count: int
) -> dict[str, Any]:
    clip_id = f"{parent.clip_id}--{identity.name}"
    raw = {
        "clip_id": clip_id,
        "rel_path": f"{identity.name}/{clip_id}.wav",
        "sha256": sha,
        "duration_s": count / CANONICAL_SAMPLE_RATE,
        "label": "spoof",
        "language": parent.language,
        "license_spdx": LICENSES[identity.name],
        "stratum": f"spoof|tts|{identity.name}|{_domain(parent)}",
        "source": parent.source,
        "speaker_id": f"{parent.speaker_id}--{identity.name}--{identity.voice}",
        "split": parent.split,
        "generator": {
            "name": identity.name,
            "version": identity.version,
            "checkpoint_sha": identity.checkpoint_sha,
            "voice": identity.voice,
            "seed": identity.seed,
            "text_source": TEXT_SOURCE,
        },
        "degradation": None,
        "parent_clip_id": None,
        "acquire": None,
    }
    validate_clip(raw, 0)
    return raw


def _load_partial_manifest(path: Path, identity: GeneratorIdentity) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    raw = _read_json(path)
    manifest = load_manifest(raw)
    if manifest.schema_version != MANIFEST_SCHEMA_VERSION:
        raise TtsGenerateError(f"{path}: partial manifest is not v1")
    rows = raw["clips"]
    result: dict[str, dict[str, Any]] = {}
    for clip, row in zip(manifest.clips, rows, strict=True):
        if clip.generator is None or asdict(clip.generator) != {
            "name": identity.name,
            "version": identity.version,
            "checkpoint_sha": identity.checkpoint_sha,
            "voice": identity.voice,
            "seed": identity.seed,
            "text_source": TEXT_SOURCE,
        }:
            raise TtsGenerateError(
                f"{path}: clip {clip.clip_id!r} has different generator identity"
            )
        result[clip.clip_id] = row
    return result


def _write_manifest(path: Path, rows: dict[str, dict[str, Any]]) -> bytes:
    obj = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "clips": sorted(rows.values(), key=lambda row: row["clip_id"]),
    }
    load_manifest(obj)
    return _write_json_atomic(path, obj)


def _identity_and_synthesizer(
    args: argparse.Namespace,
) -> tuple[GeneratorIdentity, Synthesize, dict[str, str]]:
    if args.generator == "piper":
        if args.piper_model is None:
            raise TtsGenerateError("piper requires --piper-model")
        model = Path(args.piper_model)
        if not model.is_file():
            raise TtsGenerateError(f"Piper model is not a file: {model}")
        voice = args.voice or model.stem
        identity = GeneratorIdentity(
            "piper", _package_version("piper-tts"), _sha256_file(model), voice, args.seed
        )
        return identity, _piper_synthesizer(model, args.piper_bin), {}
    if args.generator == "chatterbox":
        checkpoint_sha, checkpoint_files = _chatterbox_checkpoint_sha()
        identity = GeneratorIdentity(
            "chatterbox",
            _package_version("chatterbox-tts"),
            checkpoint_sha,
            args.voice or "parent-voice-clone",
            args.seed,
        )
        return identity, _chatterbox_synthesizer(args.seed), checkpoint_files
    if args.generator == "elevenlabs":
        api_key = args.elevenlabs_api_key or os.environ.get("ELEVENLABS_API_KEY")
        voice = args.elevenlabs_voice_id or args.voice
        if not api_key:
            raise TtsGenerateError(
                "elevenlabs requires --elevenlabs-api-key or ELEVENLABS_API_KEY"
            )
        if not voice:
            raise TtsGenerateError("elevenlabs requires --elevenlabs-voice-id or --voice")
        identity = GeneratorIdentity(
            "elevenlabs", _package_version("elevenlabs"), None, voice, None
        )
        return identity, _elevenlabs_synthesizer(api_key, voice, args.elevenlabs_model), {}
    voice = args.google_voice or args.voice or GOOGLE_TTS_DEFAULT_VOICE
    api_key = args.google_api_key or os.environ.get("GOOGLE_TTS_API_KEY")
    identity = GeneratorIdentity("google", "cloud-tts-v1", None, voice, None)
    return identity, _google_synthesizer(voice, api_key=api_key), {}


def generate(args: argparse.Namespace) -> GenerateResult:
    manifest_path = Path(args.manifest)
    manifest = load_manifest(_read_json(manifest_path))
    eligible = _eligible_clips(manifest, eval_only=args.eval_only)
    audio_root = Path(args.audio_root)
    if not audio_root.is_dir():
        raise TtsGenerateError(f"audio root is not a directory: {audio_root}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    transcripts, parent_paths = transcribe_eligible(
        eligible, audio_root, out_dir, args.whisper_url, args.request_timeout
    )
    identity, synthesize, checkpoint_files = _identity_and_synthesizer(args)
    generator_dir = out_dir / identity.name
    generator_dir.mkdir(exist_ok=True)
    manifest_out = generator_dir / "manifest.json"
    rows = _load_partial_manifest(manifest_out, identity)
    parent_ids = {clip.clip_id for clip in eligible}
    expected_ids = {f"{clip_id}--{identity.name}" for clip_id in parent_ids}
    unexpected = set(rows) - expected_ids
    if unexpected:
        raise TtsGenerateError(
            f"{manifest_out}: clips not derived from eligible parents: {sorted(unexpected)}"
        )
    generated = 0
    resumed = 0
    skipped = 0
    for parent in eligible:
        if parent.clip_id not in transcripts:
            skipped += 1
            continue
        clip_id = f"{parent.clip_id}--{identity.name}"
        output_path = generator_dir / f"{clip_id}.wav"
        prior = rows.get(clip_id)
        if prior is not None and output_path.is_file():
            payload = read_canonical_wav_payload(output_path)
            sha, _count = payload_sha_and_count(payload)
            if sha != prior["sha256"]:
                raise TtsGenerateError(
                    f"{output_path}: PCM sha256 does not match partial manifest"
                )
            resumed += 1
            continue
        synthesized = synthesize(parent, parent_paths[parent.clip_id], transcripts[parent.clip_id])
        payload = synthesized if isinstance(synthesized, bytes) else _canonical_payload(synthesized)
        sha, count = payload_sha_and_count(payload)
        row = _clip_dict(parent, identity, sha, count)
        tmp_wav = generator_dir / f".{clip_id}.{uuid.uuid4().hex}.tmp.wav"
        try:
            write_canonical_wav(tmp_wav, payload)
            reread = read_canonical_wav_payload(tmp_wav)
            if hashlib.sha256(reread).hexdigest() != sha:
                raise TtsGenerateError(f"{tmp_wav}: canonical WAV re-audit failed")
            os.replace(tmp_wav, output_path)
        finally:
            tmp_wav.unlink(missing_ok=True)
        rows[clip_id] = row
        manifest_bytes = _write_manifest(manifest_out, rows)
        generated += 1
    manifest_bytes = _write_manifest(manifest_out, rows)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    receipt = {
        "generator": asdict(identity),
        "canonical_sample_rate": CANONICAL_SAMPLE_RATE,
        "eligible": len(eligible),
        "generated": generated,
        "resumed": resumed,
        "skipped_no_transcript": skipped,
        "total": len(rows),
        "manifest_sha256": manifest_sha,
        "selection_seed": SELECTION_SEED,
        "checkpoint_files": checkpoint_files,
    }
    _write_json_atomic(generator_dir / "generate_receipt.json", receipt)
    return GenerateResult(
        identity.name, len(eligible), generated, resumed, manifest_sha, str(generator_dir)
    )


def cmd_generate(args: argparse.Namespace) -> int:
    result = generate(args)
    sys.stdout.write(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="maintainer-only S6 TTS spoof generator")
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("generate", help="transcribe eligible S5 turns and synthesize TTS")
    command.add_argument("--manifest", required=True)
    command.add_argument("--audio-root", required=True)
    command.add_argument("--out-dir", required=True)
    command.add_argument("--generator", required=True, choices=GENERATORS)
    command.add_argument("--whisper-url", default=DEFAULT_WHISPER_URL)
    command.add_argument("--request-timeout", type=float, default=300.0)
    command.add_argument("--seed", default=SELECTION_SEED)
    command.add_argument("--voice")
    command.add_argument("--piper-model")
    command.add_argument("--piper-bin", default="piper")
    command.add_argument("--elevenlabs-api-key")
    command.add_argument("--elevenlabs-voice-id")
    command.add_argument("--elevenlabs-model", default=ELEVENLABS_MODEL)
    command.add_argument("--google-api-key", help="API key (fallback if no ADC)")
    command.add_argument("--google-voice", default=GOOGLE_TTS_DEFAULT_VOICE)
    command.add_argument(
        "--eval-only",
        action="store_true",
        help="restrict to eval-split parents (for unseen generators)",
    )
    command.set_defaults(func=cmd_generate)
    args = parser.parse_args(argv)
    if not math.isfinite(args.request_timeout) or args.request_timeout <= 0:
        parser.error("--request-timeout must be a finite number > 0")
    try:
        return int(args.func(args))
    except (OSError, ValueError, CorpusError, TtsGenerateError, NotImplementedError) as exc:
        sys.stderr.write(f"generate failed: {exc}\n")
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
