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
import sys
import tarfile
import urllib.request
import wave
import zipfile
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tests.parity.bakeoff import corpus_sources as src  # noqa: E402

# TED/AMI times are seconds with 2 dp; canonicalize to integer microseconds so a
# reference hash never depends on float formatting.
US = 1_000_000


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
# generate / prepare (built on the verified core — first run pending)
# --------------------------------------------------------------------------
@dataclass
class ManifestEntry:
    dataset: str
    upstream_id: str
    sha256: str
    duration_s: float
    strata: list[str]
    license_spdx: str
    transcript_sha256: str | None
    ts_granularity: str
    extra: dict[str, Any] = field(default_factory=dict)


def cmd_generate(work_dir: Path, out_manifest: Path) -> int:
    raise SystemExit(
        "generate mode is not yet run end-to-end — see the Slice-1 bakeoff "
        "session prompt. It reuses the verify-sources core (download/verify, "
        "parse_stm, parse_nxt_words, parse_meetings, select) to fetch all "
        "strata, prepare canonical 16 kHz mono PCM, and write a candidate "
        "manifest for review."
    )


def cmd_prepare(manifest_path: Path, work_dir: Path) -> int:
    raise SystemExit(
        "prepare mode pairs with a committed manifest.json, which generate "
        "produces first — see the Slice-1 bakeoff session prompt."
    )


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
