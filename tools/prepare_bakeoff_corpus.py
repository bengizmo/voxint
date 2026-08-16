#!/usr/bin/env python3
"""Acquire + verify the whisper Metal ASR bakeoff corpus (issue #33).

The pre-registered gate (``docs/gpu-contracts.md``) benchmarks a candidate ASR
engine against the shipped CT2-CPU engine over a fixed, deterministically
selected corpus. This tool obtains that corpus from upstream and pins it, so the
gate is reproducible. **Audio is never committed** (repo size + TED's NC-ND
license); TED transcripts are never committed either. The committed artifacts
are the ``manifest.json`` and — for AMI (CC-BY-4.0) only — the word-aligned gold
and (later) the frozen CT2 baseline.

Sources are pinned in ``tests/parity/bakeoff/corpus_sources.py``; see that file
for provenance. Deps: **stdlib only** plus ``soundfile`` (already in the
``parity`` extra) to read TED's SPHERE files. Never ``extractall()`` — only the
exact members we need are extracted.

Modes
-----
``verify-sources``
    Cheap integrity pass: download the AMI annotations zip and the TED legacy
    ``dev.tar.gz``, verify their pinned sha256/size, extract one sample from
    each, and exercise the parsers on real bytes. HEAD-checks one AMI headset
    WAV. Downloads to ``--work-dir`` (default: a scratch dir), never the repo.

``generate``
    Maintainer run (network; Apple Silicon not required): fetch everything,
    prepare canonical 16 kHz mono PCM, extract references, select the
    pre-registered strata, and write a CANDIDATE ``manifest.json`` for review.
    NOTE: not yet run end-to-end — see the Slice-1 session prompt.

``prepare``
    Normal/gate run: read the committed ``manifest.json`` and re-fetch + verify
    every file, failing closed on any mismatch. NOTE: pairs with a committed
    manifest, which ``generate`` produces first.

Run from the repo root: ``python tools/prepare_bakeoff_corpus.py verify-sources``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import wave
import zipfile
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tests.parity.bakeoff import corpus_sources as src  # noqa: E402

# TED/AMI times are seconds with 2 dp; canonicalize to integer microseconds so a
# reference hash never depends on float formatting.
US = 1_000_000

# Canonical prepared-audio format: 16 kHz mono signed-16-bit little-endian PCM,
# with a deterministic 44-byte RIFF header (stdlib ``wave`` emits exactly this).
# The manifest's per-file ``sha256`` is taken over these exact bytes, so a
# re-encode that changes a single sample stops a ``prepare`` run.
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # pcm_s16le
CHANNELS = 1
WAV_HEADER_BYTES = 44
BYTES_PER_FRAME = SAMPLE_WIDTH * CHANNELS  # 2
BYTES_PER_SECOND = SAMPLE_RATE * BYTES_PER_FRAME  # 32000

# Deterministic seed base for the synthetic strata (silence/bait). Bumping it is
# a visible, deliberate reshuffle recorded in the manifest provenance.
SYNTH_SEED = 20260816

# Committed-fixture destinations (relative to the repo root).
BAKEOFF_FIXTURES = REPO / "tests" / "parity" / "fixtures" / "bakeoff"
AMI_GOLD_DIR = BAKEOFF_FIXTURES / "gold" / "ami"
SYNTHETIC_DIR = BAKEOFF_FIXTURES / "synthetic"


def _us(seconds: str | float) -> int:
    """Seconds (as string or float) → integer microseconds, exactly."""
    return int((Decimal(str(seconds)) * US).to_integral_value())


# --------------------------------------------------------------------------
# Reference data models
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class StmSegment:
    """One TED-LIUM STM line. ``ignore`` marks scoring-excluded regions."""

    start_us: int
    end_us: int
    text: str
    ignore: bool


@dataclass(frozen=True)
class NxtWord:
    """One AMI word-aligned reference token."""

    start_us: int
    end_us: int
    text: str


# --------------------------------------------------------------------------
# Parsers (pure — unit-tested against synthetic + real samples)
# --------------------------------------------------------------------------
def parse_stm(text: str) -> list[StmSegment]:
    """Parse a TED-LIUM STM file.

    Line layout: ``filename channel speaker start end <label> transcript...``.
    A transcript of exactly ``ignore_time_segment_in_scoring`` marks a masked
    region (speech that must be excluded from WER, not counted as insertions).
    """
    segments: list[StmSegment] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(";;"):
            continue
        parts = line.split(maxsplit=6)
        if len(parts) < 6:
            continue
        start, end = parts[3], parts[4]
        transcript = parts[6].strip() if len(parts) == 7 else ""
        ignore = transcript == "ignore_time_segment_in_scoring"
        segments.append(
            StmSegment(
                start_us=_us(start),
                end_us=_us(end),
                text="" if ignore else transcript,
                ignore=ignore,
            )
        )
    return segments


# XML safety note: stdlib ElementTree is used deliberately (no defusedxml dep,
# per the project's stdlib-only tooling policy). Every XML input here is read
# from an archive whose sha256 is verified against a pinned hash BEFORE parsing
# (download()/verify_file()), so the bytes are trusted upstream artifacts, not
# untrusted network input — the XXE / entity-expansion vectors do not apply.
def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_nxt_words(xml_bytes: bytes) -> list[NxtWord]:
    """Parse an AMI NXT ``*.words.xml`` file into timed word tokens.

    Word elements are ``<w>`` with ``starttime``/``endtime`` attributes.
    Elements without both times (truncations, some punctuation) are skipped —
    they cannot enter the word-boundary gate.
    """
    root = ET.fromstring(xml_bytes)
    words: list[NxtWord] = []
    for el in root.iter():
        if _localname(el.tag) != "w":
            continue
        start = el.get("starttime")
        end = el.get("endtime")
        token = (el.text or "").strip()
        if start is None or end is None or not token:
            continue
        words.append(NxtWord(start_us=_us(start), end_us=_us(end), text=token))
    return words


def parse_meetings(xml_bytes: bytes) -> dict[str, dict[str, int]]:
    """Parse AMI ``meetings.xml`` → ``{meeting: {agent: headset_channel}}``.

    Never assume agent A/B/C/D maps to channel 0/1/2/3; this mapping is the
    authority (AMI documents per-meeting exceptions).
    """
    root = ET.fromstring(xml_bytes)
    mapping: dict[str, dict[str, int]] = {}
    for meeting in root.iter():
        if _localname(meeting.tag) != "meeting":
            continue
        observation = meeting.get("observation")
        if not observation:
            continue
        agents: dict[str, int] = {}
        for speaker in meeting:
            if _localname(speaker.tag) != "speaker":
                continue
            agent = speaker.get("nxt_agent")
            channel = speaker.get("channel")
            if agent is None or channel is None:
                continue
            agents[agent] = int(channel)
        if agents:
            mapping[observation] = agents
    return mapping


# --------------------------------------------------------------------------
# Deterministic selection
# --------------------------------------------------------------------------
def hash_rank(ids: list[str]) -> list[str]:
    """Deterministically order ``ids`` by a seeded hash (pre-registration-safe).

    Stable across runs and machines; independent of upstream listing order.
    """

    def key(item: str) -> str:
        payload = f"{src.SELECTION_SEED}:{src.SELECTION_VERSION}:{item}"
        return hashlib.sha256(payload.encode()).hexdigest()

    return sorted(ids, key=key)


def select(ids: list[str], count: int) -> list[str]:
    """Top-``count`` deterministically ranked ids (fewer if not enough exist)."""
    return hash_rank(ids)[:count]


# --------------------------------------------------------------------------
# Canonical serialization (for transcript_sha256)
# --------------------------------------------------------------------------
def canonical_reference_bytes(payload: dict[str, Any]) -> bytes:
    """Deterministic bytes for hashing a reference (integer times, sorted keys)."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def stm_reference_payload(segments: list[StmSegment]) -> dict[str, Any]:
    return {
        "kind": "tedlium3-stm",
        "segments": [
            {"start_us": s.start_us, "end_us": s.end_us, "text": s.text, "ignore": s.ignore}
            for s in segments
        ],
    }


def nxt_reference_payload(words: list[NxtWord]) -> dict[str, Any]:
    return {
        "kind": "ami-nxt-words",
        "words": [
            {"start_us": w.start_us, "end_us": w.end_us, "text": w.text} for w in words
        ],
    }


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------
# Network + archive I/O (thin wrappers)
# --------------------------------------------------------------------------
def download(url: str, dest: Path, *, expected_sha256: str, expected_size: int | None) -> Path:
    """Download ``url`` to ``dest`` (cached) and verify sha256 (+ size). Fail closed."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        tmp = dest.with_suffix(dest.suffix + ".part")
        req = urllib.request.Request(url, headers={"User-Agent": "voxint-bakeoff/1"})
        with urllib.request.urlopen(req) as resp, tmp.open("wb") as out:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
        tmp.rename(dest)
    verify_file(dest, expected_sha256=expected_sha256, expected_size=expected_size)
    return dest


def verify_file(path: Path, *, expected_sha256: str, expected_size: int | None) -> None:
    size = path.stat().st_size
    if expected_size is not None and size != expected_size:
        raise SystemExit(f"{path.name}: size {size} != expected {expected_size}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise SystemExit(f"{path.name}: sha256 {digest} != expected {expected_sha256}")


def read_zip_member(zip_path: Path, member: str) -> bytes:
    with zipfile.ZipFile(zip_path) as zf:
        return zf.read(member)


def list_zip_members(zip_path: Path) -> list[str]:
    with zipfile.ZipFile(zip_path) as zf:
        return zf.namelist()


def read_tar_member(tar_path: Path, member: str) -> bytes:
    with tarfile.open(tar_path, "r:*") as tf:
        fh = tf.extractfile(member)
        if fh is None:
            raise SystemExit(f"{tar_path.name}: member not found: {member}")
        return fh.read()


def list_tar_members(tar_path: Path) -> list[str]:
    with tarfile.open(tar_path, "r:*") as tf:
        return tf.getnames()


def wav_info(data: bytes) -> dict[str, int]:
    """Sample rate / channels / frames of a RIFF WAV, from bytes."""
    with wave.open(io.BytesIO(data), "rb") as w:
        return {
            "sample_rate": w.getframerate(),
            "channels": w.getnchannels(),
            "frames": w.getnframes(),
            "sampwidth": w.getsampwidth(),
        }


def sphere_info(data: bytes) -> dict[str, Any]:
    """Sample rate / channels / frames of a NIST SPHERE file, via soundfile."""
    import soundfile as sf  # parity extra

    with sf.SoundFile(io.BytesIO(data)) as f:
        return {"sample_rate": f.samplerate, "channels": f.channels, "frames": len(f)}


def head_ok(url: str) -> tuple[bool, dict[str, str]]:
    """HEAD a URL; return (ok, {status, content_type, content_length})."""
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "voxint-bakeoff/1"})
    try:
        with urllib.request.urlopen(req) as resp:
            return True, {
                "status": str(resp.status),
                "content_type": resp.headers.get("Content-Type", ""),
                "content_length": resp.headers.get("Content-Length", ""),
            }
    except Exception as exc:  # pragma: no cover - network dependent
        return False, {"error": str(exc)}


# --------------------------------------------------------------------------
# Canonical audio: write / quantize / hash (deterministic, platform-independent)
# --------------------------------------------------------------------------
def quantize_int16(audio: np.ndarray) -> np.ndarray:
    """Float ``[-1, 1)`` → int16 the way the WAV will store it.

    Pinned for byte-reproducibility (codex Q5): clip to the representable range,
    scale by 32768, round half-to-even (``np.rint``), no dither. Mirrors
    ``tools/generate_parity_corpus.py``'s quantizer.
    """
    clipped = np.clip(audio.astype(np.float64), -1.0, 1.0 - 1.0 / 32768.0)
    return np.rint(clipped * 32768.0).astype("<i2")


def write_canonical_wav(path: Path, samples_int16: np.ndarray) -> str:
    """Write ``samples_int16`` as canonical 16 kHz mono s16le PCM; return sha256.

    Serializes explicit little-endian (``<i2``) so the bytes never depend on host
    endianness, and lets stdlib ``wave`` emit the fixed 44-byte header.
    """
    pcm = np.ascontiguousarray(samples_int16, dtype="<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(CHANNELS)
        out.setsampwidth(SAMPLE_WIDTH)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(pcm.tobytes())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_canonical_wav(path: Path) -> np.ndarray:
    """Read a canonical 16 kHz mono s16le WAV back to an int16 array."""
    with wave.open(str(path), "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (
            SAMPLE_RATE, CHANNELS, SAMPLE_WIDTH
        ):
            raise SystemExit(f"{path.name}: not canonical 16 kHz mono s16le")
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2")


# --------------------------------------------------------------------------
# RIFF header validation + HTTP Range fetch (AMI windowing)
# --------------------------------------------------------------------------
def parse_wav_header(header: bytes) -> dict[str, int]:
    """Parse a *canonical* 44-byte RIFF/WAVE PCM header. Fail closed on anything else.

    Deliberately strict: the AMI Range path computes byte offsets assuming the
    data chunk starts at byte 44, so a non-canonical layout (extra chunks, float
    format, stereo) must abort rather than silently read the wrong bytes (codex Q4).
    """
    if len(header) < WAV_HEADER_BYTES:
        raise SystemExit(f"WAV header too short: {len(header)} < {WAV_HEADER_BYTES}")
    if header[0:4] != b"RIFF" or header[8:12] != b"WAVE":
        raise SystemExit("not a RIFF/WAVE stream")
    if header[12:16] != b"fmt ":
        raise SystemExit("first subchunk is not 'fmt '")
    (fmt_size,) = struct.unpack_from("<I", header, 16)
    audio_format, channels, sample_rate, _byte_rate, _block_align, bits = struct.unpack_from(
        "<HHIIHH", header, 20
    )
    if header[36:40] != b"data":
        raise SystemExit("'data' chunk not at offset 36 (non-canonical header)")
    (data_size,) = struct.unpack_from("<I", header, 40)
    return {
        "fmt_size": fmt_size,
        "audio_format": audio_format,
        "channels": channels,
        "sample_rate": sample_rate,
        "bits": bits,
        "data_size": data_size,
    }


def validate_canonical_riff(info: dict[str, int]) -> None:
    """Assert a parsed header is PCM / mono / 16 kHz / 16-bit with a 16-byte fmt."""
    checks = {
        "fmt_size": (info["fmt_size"], 16),
        "audio_format": (info["audio_format"], 1),  # 1 = PCM
        "channels": (info["channels"], CHANNELS),
        "sample_rate": (info["sample_rate"], SAMPLE_RATE),
        "bits": (info["bits"], SAMPLE_WIDTH * 8),
    }
    bad = {k: got for k, (got, want) in checks.items() if got != want}
    if bad:
        raise SystemExit(f"non-canonical RIFF header: {bad}")


def range_get(url: str, start: int, length: int) -> bytes:
    """Fetch exactly ``length`` bytes at ``start`` via an HTTP Range request.

    Requires a ``206 Partial Content`` with an exact ``Content-Range`` and the
    right byte count — a server that ignores Range (returns 200 + the whole file)
    fails closed rather than corrupting the window.
    """
    end = start + length - 1
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "voxint-bakeoff/1", "Range": f"bytes={start}-{end}"},
    )
    with urllib.request.urlopen(req) as resp:
        if resp.status != 206:
            raise SystemExit(f"{url}: expected 206 Partial Content, got {resp.status}")
        content_range = resp.headers.get("Content-Range", "")
        if not content_range.startswith(f"bytes {start}-{end}/"):
            raise SystemExit(
                f"{url}: Content-Range {content_range!r} != expected 'bytes {start}-{end}/…'"
            )
        data: bytes = resp.read()
    if len(data) != length:
        raise SystemExit(f"{url}: got {len(data)} bytes, expected {length}")
    return data


def frames(seconds: float) -> int:
    """Whole audio frames in ``seconds`` (integer; windows are integer-frame)."""
    return round(seconds * SAMPLE_RATE)


def ami_wav_duration_s(url: str) -> float | None:
    """Duration of a canonical AMI headset WAV from its Content-Length (HEAD only)."""
    ok, info = head_ok(url)
    if not ok:
        return None
    content_length = info.get("content_length")
    if not content_length:
        return None
    return (int(content_length) - WAV_HEADER_BYTES) / BYTES_PER_SECOND


def fetch_ami_window(url: str, offset_s: float, window_s: float) -> np.ndarray:
    """Range-fetch a validated ``window_s`` window at ``offset_s`` → int16 samples."""
    info = parse_wav_header(range_get(url, 0, WAV_HEADER_BYTES))
    validate_canonical_riff(info)
    start_byte = WAV_HEADER_BYTES + frames(offset_s) * BYTES_PER_FRAME
    length = frames(window_s) * BYTES_PER_FRAME
    if start_byte + length > WAV_HEADER_BYTES + info["data_size"]:
        raise SystemExit(f"{url}: window [{offset_s}, {offset_s + window_s}) exceeds data")
    return np.frombuffer(range_get(url, start_byte, length), dtype="<i2")


# --------------------------------------------------------------------------
# Reference windowing (word gold + STM clipping)
# --------------------------------------------------------------------------
def words_in_window(words: list[NxtWord], offset_us: int, window_us: int) -> list[NxtWord]:
    """Words wholly inside ``[offset, offset+window)``, rebased to window-relative µs.

    Only fully-contained words enter the boundary gate — a word straddling an edge
    has no clean start/end within the clipped audio.
    """
    end_us = offset_us + window_us
    return [
        NxtWord(w.start_us - offset_us, w.end_us - offset_us, w.text)
        for w in words
        if w.start_us >= offset_us and w.end_us <= end_us
    ]


def clip_stm_to_window(
    segments: list[StmSegment], offset_us: int, window_us: int
) -> list[StmSegment]:
    """Clip STM segments to the window, rebased to window-relative µs.

    A segment fully inside keeps its text; one crossing either edge — or already an
    ``ignore`` mask — becomes a clipped ``ignore`` region (codex Q1: partial audio
    must never carry full text, and masked speech stays excluded from WER).
    """
    win_end = offset_us + window_us
    out: list[StmSegment] = []
    for s in segments:
        lo = max(s.start_us, offset_us)
        hi = min(s.end_us, win_end)
        if lo >= hi:
            continue  # no overlap with the window
        crosses = s.start_us < offset_us or s.end_us > win_end
        ignore = s.ignore or crosses
        out.append(
            StmSegment(
                start_us=lo - offset_us,
                end_us=hi - offset_us,
                text="" if ignore else s.text,
                ignore=ignore,
            )
        )
    return out


# --------------------------------------------------------------------------
# verify-sources (the cheap integrity pass — run this session)
# --------------------------------------------------------------------------
def cmd_verify_sources(work_dir: Path) -> int:
    ok = True
    print(f"work dir: {work_dir}")

    # --- AMI annotations zip ---------------------------------------------
    print("\n[AMI] annotations v1.6.2")
    ann = download(
        src.AMI["annotations_url"],
        work_dir / "ami_public_manual_1.6.2.zip",
        expected_sha256=src.AMI["annotations_sha256"],
        expected_size=None,
    )
    print(f"  sha256 OK ({ann.stat().st_size} bytes)")
    meetings_xml = read_zip_member(ann, src.AMI["meetings_member"])
    mapping = parse_meetings(meetings_xml)
    print(f"  meetings.xml: {len(mapping)} meetings parsed")
    sample_meeting = next(iter(sorted(mapping)))
    print(f"  e.g. {sample_meeting}: agent→channel {mapping[sample_meeting]}")
    agent = next(iter(sorted(mapping[sample_meeting])))
    words_member = src.AMI["words_member"].format(meeting=sample_meeting, agent=agent)
    try:
        words_xml = read_zip_member(ann, words_member)
        words = parse_nxt_words(words_xml)
        print(f"  {words_member}: {len(words)} timed words; first: "
              f"{[(w.text, w.start_us, w.end_us) for w in words[:3]]}")
    except KeyError:
        print(f"  WARNING: {words_member} not in zip (channel map still valid)")

    # --- AMI headset WAV URL (HEAD only — no download) -------------------
    channel = mapping[sample_meeting][agent]
    wav_url = src.AMI["audio_url_template"].format(meeting=sample_meeting, channel=channel)
    reachable, info = head_ok(wav_url)
    print(f"  headset WAV HEAD {sample_meeting}.Headset-{channel}.wav: "
          f"{'OK' if reachable else 'UNREACHABLE'} {info}")
    if not reachable:
        ok = False

    # --- TED-LIUM 3 legacy dev.tar.gz -----------------------------------
    print("\n[TED-LIUM 3] legacy dev.tar.gz")
    dev_rel = "TEDLIUM_release3/legacy/dev.tar.gz"
    dev_meta = src.TEDLIUM3["archives"][dev_rel]
    dev = download(
        src.TEDLIUM3["url_template"].format(path=dev_rel),
        work_dir / "tedlium3-dev.tar.gz",
        expected_sha256=dev_meta["sha256"],
        expected_size=dev_meta["size_bytes"],
    )
    print(f"  sha256 + size OK ({dev.stat().st_size} bytes)")
    members = list_tar_members(dev)
    stm_members = sorted(m for m in members if m.endswith(".stm"))
    sph_members = sorted(m for m in members if m.endswith(".sph"))
    print(f"  archive: {len(stm_members)} .stm, {len(sph_members)} .sph")
    if stm_members:
        stm_text = read_tar_member(dev, stm_members[0]).decode("utf-8", "replace")
        segs = parse_stm(stm_text)
        ignored = sum(1 for s in segs if s.ignore)
        ref_sha = sha256_hex(canonical_reference_bytes(stm_reference_payload(segs)))
        print(f"  {stm_members[0]}: {len(segs)} segments ({ignored} ignore-masks); "
              f"ref sha256 {ref_sha[:12]}…")
    if sph_members:
        sph_bytes = read_tar_member(dev, sph_members[0])
        try:
            print(f"  {sph_members[0]}: SPHERE {sphere_info(sph_bytes)}")
        except Exception as exc:  # pragma: no cover
            print(f"  WARNING: could not read SPHERE via soundfile: {exc}")
            ok = False

    print("\nverify-sources:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# --------------------------------------------------------------------------
# Synthetic strata (CC0, committed): silence / hallucination-bait / short-clean
# --------------------------------------------------------------------------
# Silence: pure digital zero + seeded near-silence noise-floor. Zero-insertion
# gate expects an ABSOLUTE 0 chars on these.
SILENCE_SPECS: list[tuple[str, float, bool]] = [
    ("silence_00", 5.0, False),   # pure digital zero
    ("silence_01", 10.0, False),  # pure digital zero
    ("silence_02", 8.0, True),    # near-silence noise floor
    ("silence_03", 3.0, True),
    ("silence_04", 15.0, True),
]

# Hallucination-bait: energy but NO speech — a well-behaved ASR emits nothing.
# Each texture is seeded/deterministic; amplitudes are moderate (no clipping).
BAIT_SPECS: list[tuple[str, str, float]] = [
    ("bait_00_applause", "applause", 8.0),
    ("bait_01_breath", "breath", 6.0),
    ("bait_02_tone", "tone", 7.0),
    ("bait_03_clicks", "clicks", 5.0),
    ("bait_04_pink", "pink", 10.0),
]

# Short-clean read-speech sanity floor (original CC0 sentences, espeak-ng voice).
SHORT_CLEAN_VOICE = "en-us"
SHORT_CLEAN_SPEED = 160
SHORT_CLEAN_PITCH = 50
SHORT_CLEAN_TEXTS: list[str] = [
    "The lighthouse keeper recorded the tide at dawn and again at dusk.",
    "Seven copper kettles hung above the wide kitchen hearth.",
    "A quiet river wound between the orchard and the old stone wall.",
    "The cartographer folded the map and traced the northern road twice.",
    "Bright lanterns swung along the pier as the last ferry departed.",
]


def _normalize_peak(signal: np.ndarray, peak: float) -> np.ndarray:
    """Scale a float signal to a fixed peak amplitude (safe on all-zero input)."""
    hi = float(np.max(np.abs(signal))) if signal.size else 0.0
    return signal * (peak / hi) if hi > 0 else signal


def gen_silence(index: int, seconds: float, near_silence: bool) -> np.ndarray:
    """True digital zero, or a seeded ±2-LSB noise floor (deterministic)."""
    n = frames(seconds)
    if not near_silence:
        return np.zeros(n, dtype="<i2")
    rng = np.random.default_rng(SYNTH_SEED + index)
    return rng.integers(-2, 3, size=n, endpoint=False).astype("<i2")


def gen_bait(index: int, kind: str, seconds: float) -> np.ndarray:
    """Seeded non-speech texture (int16). Deterministic per (kind, index)."""
    n = frames(seconds)
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    rng = np.random.default_rng(SYNTH_SEED + 100 + index)
    if kind == "applause":
        env = np.zeros(n)
        spikes = rng.random(n) < 0.02
        env[spikes] = rng.random(int(spikes.sum()))
        decay = np.exp(-np.arange(800) / 200.0)
        env = np.convolve(env, decay)[:n]
        sig = rng.normal(0.0, 1.0, n) * env
    elif kind == "breath":
        low = np.convolve(rng.normal(0.0, 1.0, n), np.ones(400) / 400.0, mode="same")
        sig = low * (0.5 + 0.5 * np.sin(2 * np.pi * 0.3 * t))
    elif kind == "tone":
        chord = sum(np.sin(2 * np.pi * f * t) for f in (220.0, 277.18, 329.63)) / 3.0
        sig = chord * (0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t))
    elif kind == "clicks":
        period = max(1, int(0.25 * SAMPLE_RATE))
        impulses = np.zeros(n)
        impulses[::period] = 1.0
        sig = np.convolve(impulses, np.exp(-np.arange(200) / 40.0))[:n]
    elif kind == "pink":
        brown = np.cumsum(rng.normal(0.0, 1.0, n))
        sig = brown - np.convolve(brown, np.ones(2000) / 2000.0, mode="same")
    else:  # pragma: no cover - guarded by BAIT_SPECS
        raise SystemExit(f"unknown bait kind: {kind}")
    return quantize_int16(_normalize_peak(np.asarray(sig, dtype=np.float64), 0.3))


def _binary_version(binary: str, *args: str) -> str:
    """First non-empty line of ``binary <args>`` output (for provenance)."""
    out = subprocess.run([binary, *args], capture_output=True, text=True, check=True)
    for line in (out.stdout + out.stderr).splitlines():
        if line.strip():
            return line.strip()
    return "unknown"


def synthesize_speech(text: str) -> np.ndarray:
    """espeak-ng → ffmpeg → canonical 16 kHz mono int16 samples for ``text``."""
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw.wav"
        canon = Path(tmp) / "canon.wav"
        subprocess.run(
            ["espeak-ng", "-v", SHORT_CLEAN_VOICE, "-s", str(SHORT_CLEAN_SPEED),
             "-p", str(SHORT_CLEAN_PITCH), "-w", str(raw), text],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(raw),
             "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS), "-c:a", "pcm_s16le", str(canon)],
            check=True, capture_output=True,
        )
        return read_canonical_wav(canon)


def text_reference_payload(text: str) -> dict[str, Any]:
    """Canonical payload for a synthetic short-clean transcript (CC0, committed)."""
    return {"kind": "text-clean", "text": text}


# --------------------------------------------------------------------------
# Manifest assembly
# --------------------------------------------------------------------------
def manifest_entry(
    *,
    dataset: str,
    upstream_id: str,
    sha256: str,
    duration_s: float,
    strata: list[str],
    license_spdx: str,
    transcript_sha256: str | None,
    ts_granularity: str,
    acquire: dict[str, Any],
    gold_file: str | None = None,
    text: str | None = None,
) -> dict[str, Any]:
    """One manifest ``files[]`` entry (the approved fields + acquisition provenance)."""
    entry: dict[str, Any] = {
        "dataset": dataset,
        "upstream_id": upstream_id,
        "sha256": sha256,
        "duration_s": round(duration_s, 3),
        "strata": strata,
        "license_spdx": license_spdx,
        "transcript_sha256": transcript_sha256,
        "ts_granularity": ts_granularity,
        "acquire": acquire,
    }
    if gold_file is not None:
        entry["gold_file"] = gold_file
    if text is not None:
        entry["text"] = text
    return entry


def validate_strata_counts(files: list[dict[str, Any]]) -> None:
    """Fail closed unless every stratum hit its exact pre-registered target."""
    counts: dict[str, int] = {}
    for entry in files:
        for stratum in entry["strata"]:
            counts[stratum] = counts.get(stratum, 0) + 1
    mismatches = {
        stratum: (counts.get(stratum, 0), target)
        for stratum, target in src.STRATA_TARGETS.items()
        if counts.get(stratum, 0) != target
    }
    if mismatches:
        raise SystemExit(f"strata count mismatch (got, want): {mismatches}")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


# Per-dataset manifest invariants (license / timestamp granularity / acquire kind /
# whether derived references may be committed). These encode the licensing doctrine:
# AMI (CC-BY) commits word gold; TED (NC-ND) commits only a transcript hash; synthetic
# (CC0) commits the audio itself.
_DATASET_RULES: dict[str, dict[str, Any]] = {
    "ami_ihm": {"license_spdx": "CC-BY-4.0", "ts_granularity": "word",
                "acquire_kind": "ami_range", "gold_file": True, "transcript": True},
    "tedlium3": {"license_spdx": "CC-BY-NC-ND-3.0", "ts_granularity": "segment",
                 "acquire_kind": "ted_window", "gold_file": False, "transcript": True},
    "synthetic": {"license_spdx": "CC0-1.0", "ts_granularity": "none",
                  "acquire_kind": "committed", "gold_file": False, "transcript": None},
}

_REQUIRED_ENTRY_KEYS = {
    "dataset", "upstream_id", "sha256", "duration_s", "strata",
    "license_spdx", "transcript_sha256", "ts_granularity", "acquire",
}


def validate_manifest_schema(manifest: dict[str, Any]) -> None:
    """Fail closed unless ``manifest`` satisfies every pre-registered invariant.

    Pure (no I/O): usable both as a unit test over synthetic manifests and as the
    contract check over the committed ``manifest.json``.
    """
    if manifest.get("schema_version") != 1:
        raise SystemExit(f"schema_version must be 1, got {manifest.get('schema_version')!r}")
    selection = manifest.get("selection", {})
    if (selection.get("seed"), selection.get("version")) != (
        src.SELECTION_SEED, src.SELECTION_VERSION
    ):
        raise SystemExit("manifest selection seed/version drifted from corpus_sources")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise SystemExit("manifest.files must be a non-empty list")

    for entry in files:
        missing = _REQUIRED_ENTRY_KEYS - entry.keys()
        if missing:
            raise SystemExit(f"{entry.get('upstream_id')}: missing keys {missing}")
        dataset = entry["dataset"]
        rule = _DATASET_RULES.get(dataset)
        if rule is None:
            raise SystemExit(f"{entry['upstream_id']}: unknown dataset {dataset!r}")
        if not _is_sha256(entry["sha256"]):
            raise SystemExit(f"{entry['upstream_id']}: sha256 not 64 hex chars")
        if not (isinstance(entry["duration_s"], (int, float)) and entry["duration_s"] > 0):
            raise SystemExit(f"{entry['upstream_id']}: duration_s must be positive")
        if entry["license_spdx"] != rule["license_spdx"]:
            raise SystemExit(f"{entry['upstream_id']}: license {entry['license_spdx']!r}")
        if entry["ts_granularity"] != rule["ts_granularity"]:
            raise SystemExit(f"{entry['upstream_id']}: ts_granularity {entry['ts_granularity']!r}")
        if entry["ts_granularity"] not in src.TS_GRANULARITIES:
            raise SystemExit(f"{entry['upstream_id']}: bad ts_granularity")
        if entry["acquire"].get("kind") != rule["acquire_kind"]:
            raise SystemExit(f"{entry['upstream_id']}: acquire.kind != {rule['acquire_kind']}")
        if rule["gold_file"] and "gold_file" not in entry:
            raise SystemExit(f"{entry['upstream_id']}: {dataset} must commit gold_file")
        # NC-ND doctrine: TED must never commit a transcript or its text.
        if dataset == "tedlium3" and ("gold_file" in entry or "text" in entry):
            raise SystemExit(f"{entry['upstream_id']}: TED transcript must not be committed")
        # transcript_sha256 presence rule: True → required; None → forbidden;
        # synthetic (None sentinel) → depends on whether the stratum carries text.
        has_transcript = entry["transcript_sha256"] is not None
        if rule["transcript"] is True and not has_transcript:
            raise SystemExit(f"{entry['upstream_id']}: transcript_sha256 required")
        if has_transcript and not _is_sha256(entry["transcript_sha256"]):
            raise SystemExit(f"{entry['upstream_id']}: transcript_sha256 not 64 hex chars")
        if dataset == "synthetic":
            has_text = entry.get("text") is not None
            if has_text != has_transcript:
                raise SystemExit(f"{entry['upstream_id']}: text/transcript_sha256 disagree")

    validate_strata_counts(files)


def tool_provenance() -> dict[str, str]:
    """Record the exact tool versions a ``prepare`` re-synthesis must reproduce."""
    return {
        "python": ".".join(str(v) for v in sys.version_info[:3]),
        "numpy": np.__version__,
        "espeak_ng": _binary_version("espeak-ng", "--version"),
        "ffmpeg": _binary_version("ffmpeg", "-version"),
    }


# --------------------------------------------------------------------------
# generate: fetch/synthesize every stratum, write a CANDIDATE manifest for review
# --------------------------------------------------------------------------
def _generate_ami(work_dir: Path, ann_zip: Path) -> list[dict[str, Any]]:
    offset_s = float(src.AMI["slice_offset_s"])
    window_s = float(src.AMI["slice_window_s"])
    offset_us, window_us = _us(offset_s), _us(window_s)
    target = src.STRATA_TARGETS["ami_ihm"]
    min_words = src.AMI_MIN_WINDOW_WORDS

    meetings = parse_meetings(read_zip_member(ann_zip, src.AMI["meetings_member"]))
    audio_dir = work_dir / "ami"
    entries: list[dict[str, Any]] = []
    print(f"\n[AMI] selecting {target} meetings (>= {min_words} words in window)")
    for meeting in hash_rank(sorted(meetings)):
        if len(entries) >= target:
            break
        agent = sorted(meetings[meeting])[0]  # sorted-first mapped agent (Q3)
        channel = meetings[meeting][agent]
        wav_url = src.AMI["audio_url_template"].format(meeting=meeting, channel=channel)
        duration = ami_wav_duration_s(wav_url)
        if duration is None or duration < offset_s + window_s:
            continue
        words_member = src.AMI["words_member"].format(meeting=meeting, agent=agent)
        try:
            all_words = parse_nxt_words(read_zip_member(ann_zip, words_member))
        except KeyError:
            continue
        win_words = words_in_window(all_words, offset_us, window_us)
        if len(win_words) < min_words:
            continue

        samples = fetch_ami_window(wav_url, offset_s, window_s)
        sha = write_canonical_wav(audio_dir / f"{meeting}.{agent}.wav", samples)
        payload = nxt_reference_payload(win_words)
        transcript_sha = sha256_hex(canonical_reference_bytes(payload))
        gold_rel = f"gold/ami/{meeting}.{agent}.words.json"
        gold_path = BAKEOFF_FIXTURES / gold_rel
        gold_path.parent.mkdir(parents=True, exist_ok=True)
        gold_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        entries.append(manifest_entry(
            dataset=src.AMI["dataset"],
            upstream_id=f"{meeting}.Headset-{channel}",
            sha256=sha,
            duration_s=window_s,
            strata=["ami_ihm"],
            license_spdx=src.AMI["license_spdx"],
            transcript_sha256=transcript_sha,
            ts_granularity="word",
            acquire={"kind": "ami_range", "meeting": meeting, "agent": agent,
                     "channel": channel, "offset_s": offset_s, "window_s": window_s},
            gold_file=gold_rel,
        ))
        print(f"  {meeting}.{agent} (ch{channel}): {len(win_words)} words → {sha[:12]}…")
    if len(entries) < target:
        raise SystemExit(f"AMI: only {len(entries)}/{target} eligible meetings found")
    return entries


def _decode_sph_window(sph_bytes: bytes, offset_s: float, window_s: float) -> np.ndarray | None:
    import soundfile as sf  # parity extra

    with sf.SoundFile(io.BytesIO(sph_bytes)) as f:
        if f.samplerate != SAMPLE_RATE or f.channels != CHANNELS:
            raise SystemExit(f"TED SPHERE not 16 kHz mono: {f.samplerate} Hz / {f.channels} ch")
        start, count = frames(offset_s), frames(window_s)
        if start + count > len(f):
            return None
        f.seek(start)
        data = f.read(count, dtype="int16", always_2d=False)
    return np.asarray(data, dtype="<i2")


def _generate_ted(work_dir: Path) -> list[dict[str, Any]]:
    offset_s = float(src.TEDLIUM3["slice_offset_s"])
    window_s = float(src.TEDLIUM3["slice_window_s"])
    offset_us, window_us = _us(offset_s), _us(window_s)
    target = src.STRATA_TARGETS["tedlium3"]

    talks: dict[str, dict[str, Any]] = {}
    for rel, meta in src.TEDLIUM3["archives"].items():
        tar_path = download(
            src.TEDLIUM3["url_template"].format(path=rel),
            work_dir / f"tedlium3-{rel.split('/')[-1]}",
            expected_sha256=meta["sha256"], expected_size=meta["size_bytes"],
        )
        for member in list_tar_members(tar_path):
            if member.endswith(".sph"):
                talk = Path(member).stem
                talks[talk] = {"rel": rel, "tar": tar_path, "sph": member,
                               "stm": member[:-4] + ".stm"}

    audio_dir = work_dir / "ted"
    entries: list[dict[str, Any]] = []
    print(f"\n[TED-LIUM 3] selecting {target} windowed talks")
    for talk in hash_rank(sorted(talks)):
        if len(entries) >= target:
            break
        info = talks[talk]
        window = _decode_sph_window(read_tar_member(info["tar"], info["sph"]), offset_s, window_s)
        if window is None:  # talk shorter than the window
            continue
        sha = write_canonical_wav(audio_dir / f"{talk}.wav", window)
        stm_text = read_tar_member(info["tar"], info["stm"]).decode("utf-8", "replace")
        segs = clip_stm_to_window(parse_stm(stm_text), offset_us, window_us)
        transcript_sha = sha256_hex(canonical_reference_bytes(stm_reference_payload(segs)))
        entries.append(manifest_entry(
            dataset=src.TEDLIUM3["dataset"],
            upstream_id=talk,
            sha256=sha,
            duration_s=window_s,
            strata=["tedlium3"],
            license_spdx=src.TEDLIUM3["license_spdx"],
            transcript_sha256=transcript_sha,  # transcript itself NEVER committed
            ts_granularity="segment",
            acquire={"kind": "ted_window", "archive": info["rel"], "sph": info["sph"],
                     "stm": info["stm"], "offset_s": offset_s, "window_s": window_s},
        ))
        print(f"  {talk}: {len(segs)} clipped segments → {sha[:12]}…")
    if len(entries) < target:
        raise SystemExit(f"TED: only {len(entries)}/{target} talks held the window")
    return entries


def _synthetic_entry(name: str, stratum: str, seconds: float, sha: str,
                     text: str | None) -> dict[str, Any]:
    transcript_sha = (
        sha256_hex(canonical_reference_bytes(text_reference_payload(text)))
        if text is not None else None
    )
    return manifest_entry(
        dataset="synthetic",
        upstream_id=name,
        sha256=sha,
        duration_s=seconds,
        strata=[stratum],
        license_spdx="CC0-1.0",
        transcript_sha256=transcript_sha,
        ts_granularity="none",
        acquire={"kind": "committed", "path": f"synthetic/{name}.wav"},
        text=text,
    )


def _generate_synthetic() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    print("\n[synthetic] silence / bait / short-clean (committed CC0)")
    for i, (name, seconds, near) in enumerate(SILENCE_SPECS):
        sha = write_canonical_wav(SYNTHETIC_DIR / f"{name}.wav", gen_silence(i, seconds, near))
        entries.append(_synthetic_entry(name, "synthetic_silence", seconds, sha, None))
    for i, (name, kind, seconds) in enumerate(BAIT_SPECS):
        sha = write_canonical_wav(SYNTHETIC_DIR / f"{name}.wav", gen_bait(i, kind, seconds))
        entries.append(_synthetic_entry(name, "synthetic_bait", seconds, sha, None))
    for i, text in enumerate(SHORT_CLEAN_TEXTS):
        name = f"short_clean_{i:02d}"
        samples = synthesize_speech(text)
        seconds = len(samples) / SAMPLE_RATE
        sha = write_canonical_wav(SYNTHETIC_DIR / f"{name}.wav", samples)
        entries.append(_synthetic_entry(name, "synthetic_short_clean", seconds, sha, text))
    for entry in entries:
        print(f"  {entry['upstream_id']}: {entry['duration_s']}s → {entry['sha256'][:12]}…")
    return entries


def cmd_generate(work_dir: Path, out_manifest: Path) -> int:
    ann_zip = download(
        src.AMI["annotations_url"],
        work_dir / "ami_public_manual_1.6.2.zip",
        expected_sha256=src.AMI["annotations_sha256"], expected_size=None,
    )
    files = _generate_ami(work_dir, ann_zip) + _generate_ted(work_dir) + _generate_synthetic()
    manifest = {
        "schema_version": 1,
        "note": "CANDIDATE manifest — review before committing (generate mode)",
        "selection": {"seed": src.SELECTION_SEED, "version": src.SELECTION_VERSION},
        "strata_targets": dict(src.STRATA_TARGETS),
        "provenance": tool_provenance(),
        "files": files,
    }
    validate_manifest_schema(manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"\ngenerate: wrote {len(files)} files → {out_manifest}")
    return 0


# --------------------------------------------------------------------------
# prepare: reconstruct + verify every file against the committed manifest
# --------------------------------------------------------------------------
def _prepare_ami(entry: dict[str, Any], work_dir: Path) -> None:
    acq = entry["acquire"]
    wav_url = src.AMI["audio_url_template"].format(meeting=acq["meeting"], channel=acq["channel"])
    samples = fetch_ami_window(wav_url, acq["offset_s"], acq["window_s"])
    sha = write_canonical_wav(
        work_dir / "ami" / f"{acq['meeting']}.{acq['agent']}.wav", samples
    )
    if sha != entry["sha256"]:
        raise SystemExit(f"AMI {entry['upstream_id']}: audio sha256 mismatch")
    gold_payload = json.loads((BAKEOFF_FIXTURES / entry["gold_file"]).read_text())
    if sha256_hex(canonical_reference_bytes(gold_payload)) != entry["transcript_sha256"]:
        raise SystemExit(f"AMI {entry['upstream_id']}: committed gold ≠ transcript_sha256")


def _prepare_ted(entry: dict[str, Any], work_dir: Path, tars: dict[str, Path]) -> None:
    acq = entry["acquire"]
    tar_path = tars[acq["archive"]]
    window = _decode_sph_window(
        read_tar_member(tar_path, acq["sph"]), acq["offset_s"], acq["window_s"]
    )
    if window is None:
        raise SystemExit(f"TED {entry['upstream_id']}: window no longer fits")
    sha = write_canonical_wav(work_dir / "ted" / f"{entry['upstream_id']}.wav", window)
    if sha != entry["sha256"]:
        raise SystemExit(f"TED {entry['upstream_id']}: audio sha256 mismatch")
    stm_text = read_tar_member(tar_path, acq["stm"]).decode("utf-8", "replace")
    segs = clip_stm_to_window(parse_stm(stm_text), _us(acq["offset_s"]), _us(acq["window_s"]))
    recomputed = sha256_hex(canonical_reference_bytes(stm_reference_payload(segs)))
    if recomputed != entry["transcript_sha256"]:
        raise SystemExit(f"TED {entry['upstream_id']}: transcript_sha256 mismatch")


def _prepare_committed(entry: dict[str, Any]) -> None:
    path = BAKEOFF_FIXTURES / entry["acquire"]["path"]
    if not path.exists():
        raise SystemExit(f"synthetic {entry['upstream_id']}: missing committed {path}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise SystemExit(f"synthetic {entry['upstream_id']}: committed audio sha256 mismatch")
    text = entry.get("text")
    if text is not None:
        expected = sha256_hex(canonical_reference_bytes(text_reference_payload(text)))
        if expected != entry["transcript_sha256"]:
            raise SystemExit(f"synthetic {entry['upstream_id']}: text ≠ transcript_sha256")


def cmd_prepare(manifest_path: Path, work_dir: Path) -> int:
    manifest = json.loads(manifest_path.read_text())
    validate_manifest_schema(manifest)
    files = manifest["files"]

    # Pre-fetch TED archives once (cached), verifying their pinned shas.
    ted_tars: dict[str, Path] = {}
    if any(e["acquire"]["kind"] == "ted_window" for e in files):
        for rel, meta in src.TEDLIUM3["archives"].items():
            ted_tars[rel] = download(
                src.TEDLIUM3["url_template"].format(path=rel),
                work_dir / f"tedlium3-{rel.split('/')[-1]}",
                expected_sha256=meta["sha256"], expected_size=meta["size_bytes"],
            )

    for entry in files:
        kind = entry["acquire"]["kind"]
        if kind == "ami_range":
            _prepare_ami(entry, work_dir)
        elif kind == "ted_window":
            _prepare_ted(entry, work_dir, ted_tars)
        elif kind == "committed":
            _prepare_committed(entry)
        else:  # pragma: no cover - guarded by generate
            raise SystemExit(f"unknown acquire kind: {kind}")
        print(f"  ok: {entry['dataset']}/{entry['upstream_id']}")

    print(f"\nprepare: verified {len(files)} files against {manifest_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    default_work = Path.home() / ".voxint-metal" / "bakeoff" / "work"

    p_verify = sub.add_parser("verify-sources", help="cheap pin + parser integrity pass")
    p_verify.add_argument("--work-dir", type=Path, default=default_work)

    p_gen = sub.add_parser("generate", help="fetch all + write candidate manifest")
    p_gen.add_argument("--work-dir", type=Path, default=default_work)
    p_gen.add_argument(
        "--out", type=Path,
        default=REPO / "tests" / "parity" / "fixtures" / "bakeoff" / "manifest.json",
    )

    p_prep = sub.add_parser("prepare", help="verify committed manifest, fail closed")
    p_prep.add_argument("--work-dir", type=Path, default=default_work)
    p_prep.add_argument(
        "--manifest", type=Path,
        default=REPO / "tests" / "parity" / "fixtures" / "bakeoff" / "manifest.json",
    )

    args = parser.parse_args()
    if args.mode == "verify-sources":
        return cmd_verify_sources(args.work_dir)
    if args.mode == "generate":
        return cmd_generate(args.work_dir, args.out)
    if args.mode == "prepare":
        return cmd_prepare(args.manifest, args.work_dir)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
