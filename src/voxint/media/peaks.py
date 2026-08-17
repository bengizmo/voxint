"""Waveform peaks: amplitude envelope of the normalized WAV (issue #57).

The review console draws a who-spoke-when strip; its amplitude backdrop comes
from a fixed-bucket max-abs envelope of the run's 16 kHz mono pcm_s16le WAV
(the prepare-stage invariant). Peaks are computed lazily on the first
``GET /media/{run_id}/peaks`` and cached as a derived artifact:
``artifacts/{run_id}/peaks.json`` plus an ``audio_artifacts`` row with
``kind = 'waveform_peaks'``.

Cache-validity contract (why row presence alone is NOT trusted): prepare
atomically replaces ``normalized.wav`` *before* its DB transaction commits, so
a crash in that window can strand a peaks row describing the previous bytes.
The row's ``meta.source_fingerprint`` therefore records the ``{size,
mtime_ns}`` of the WAV the peaks were computed from; while the WAV is live the
route fstat-verifies it on every cache hit and recomputes on mismatch. Once the
WAV is formally reclaimed there is nothing to verify against — the cache is
served as-is so a static waveform can still render (derived evidence, like the
transcript).

Everything here fails closed: a WAV that is stereo, non-16-bit, compressed, at
the wrong rate, zero-length, or shorter than its header claims raises
``PeaksError`` and nothing is cached.
"""

from __future__ import annotations

import json
import math
import os
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from voxint.db.models import ArtifactKind, AudioArtifact
from voxint.media.normalize import TARGET_SAMPLE_RATE

# Fixed bucket count, not samples-per-peak: caps the payload (~14 KB) for any
# duration, and an overview strip gains nothing from finer resolution than its
# on-screen pixel width. Module constant, deliberately not a setting.
PEAK_BUCKETS = 2000
PEAKS_VERSION = 1

# Run-scoped, beside prepare's normalized.wav (same directory lifecycle).
_PEAKS_TEMPLATE = "artifacts/{run_id}/peaks.json"


def peaks_relative(run_id: uuid.UUID) -> str:
    """The run's peaks-cache path, relative to ``media_root`` (one place)."""
    return _PEAKS_TEMPLATE.format(run_id=run_id)

# ~0.5-1 MiB of int16 frames per read keeps memory bounded on 60+ min files.
_TARGET_CHUNK_FRAMES = 262_144


class PeaksError(Exception):
    """The WAV cannot yield trustworthy peaks — fail closed, cache nothing."""


@dataclass(frozen=True)
class PeaksPayload:
    """Envelope + its full coordinate system (the response IS the cache file)."""

    duration_seconds: float
    sample_rate: int
    frame_count: int
    samples_per_bucket: int
    peaks: list[float]  # max(|sample|)/32768 per bucket, 0..1, 3 dp

    def to_json_bytes(self) -> bytes:
        """Exact bytes written to peaks.json AND returned by the route."""
        return json.dumps(
            {
                "version": PEAKS_VERSION,
                "duration": self.duration_seconds,
                "sampleRate": self.sample_rate,
                "frameCount": self.frame_count,
                "samplesPerBucket": self.samples_per_bucket,
                "peaks": self.peaks,
            },
            separators=(",", ":"),
        ).encode("utf-8")


def compute_peaks(fh: BinaryIO, *, buckets: int = PEAK_BUCKETS) -> PeaksPayload:
    """Reduce a pcm_s16le mono WAV to a fixed-bucket max-abs envelope.

    Reads through ``wave`` in whole-bucket chunks so memory stays bounded
    regardless of duration. Every format deviation from the prepare-stage
    invariant raises :class:`PeaksError` — peaks for audio we don't understand
    would be an actively misleading picture.
    """
    try:
        with wave.open(fh, "rb") as reader:
            return _reduce(reader, buckets)
    except (wave.Error, EOFError, OSError) as exc:
        # OSError too: a disk read failure mid-reduce stays fail-closed (route
        # answers 404, nothing cached) rather than escaping as an opaque 500.
        raise PeaksError(f"unreadable WAV: {exc}") from exc


def _reduce(reader: wave.Wave_read, buckets: int) -> PeaksPayload:
    if reader.getcomptype() != "NONE":
        raise PeaksError(f"compressed WAV ({reader.getcomptype()}) unsupported")
    if reader.getnchannels() != 1:
        raise PeaksError(f"expected mono, got {reader.getnchannels()} channels")
    if reader.getsampwidth() != 2:
        raise PeaksError(f"expected 16-bit samples, got {reader.getsampwidth() * 8}-bit")
    rate = reader.getframerate()
    if rate != TARGET_SAMPLE_RATE:
        raise PeaksError(f"expected {TARGET_SAMPLE_RATE} Hz, got {rate}")
    nframes = reader.getnframes()
    if nframes <= 0:
        raise PeaksError("WAV contains no frames")

    spb = max(1, math.ceil(nframes / buckets))
    # Whole buckets per read: the reshape below never straddles a boundary.
    chunk_frames = spb * max(1, _TARGET_CHUNK_FRAMES // spb)

    peaks: list[float] = []
    carry = np.empty(0, dtype=np.int32)
    frames_read = 0
    while True:
        data = reader.readframes(chunk_frames)
        if not data:
            break
        if len(data) % 2:
            raise PeaksError("WAV data ends mid-sample")
        # int32 BEFORE abs: np.abs(int16 -32768) overflows back to -32768.
        block = np.abs(np.frombuffer(data, dtype="<i2").astype(np.int32))
        frames_read += block.size
        if carry.size:
            block = np.concatenate([carry, block])
        full = (block.size // spb) * spb
        if full:
            peaks.extend(
                (block[:full].reshape(-1, spb).max(axis=1) / 32768.0).round(3).tolist()
            )
        carry = block[full:]
    if carry.size:
        peaks.append(round(float(carry.max()) / 32768.0, 3))

    # wave returns short reads (no exception) on a truncated data chunk —
    # a header/data mismatch means we cannot trust the time axis at all.
    if frames_read != nframes:
        raise PeaksError(
            f"WAV truncated: header claims {nframes} frames, read {frames_read}"
        )

    duration = frames_read / rate
    if not (math.isfinite(duration) and duration > 0):
        raise PeaksError(f"non-finite duration {duration!r}")
    return PeaksPayload(
        duration_seconds=round(duration, 3),
        sample_rate=rate,
        frame_count=frames_read,
        samples_per_bucket=spb,
        peaks=peaks,
    )


@dataclass(frozen=True)
class SourceFingerprint:
    """fstat identity of the WAV the peaks were computed from."""

    size: int
    mtime_ns: int

    @classmethod
    def of_descriptor(cls, fh: BinaryIO) -> SourceFingerprint:
        stat = os.fstat(fh.fileno())
        return cls(size=stat.st_size, mtime_ns=stat.st_mtime_ns)

    @classmethod
    def of_path(cls, path: Path) -> SourceFingerprint | None:
        """Fingerprint of a live file, or None if it cannot be statted."""
        try:
            stat = path.stat()
        except OSError:
            return None
        return cls(size=stat.st_size, mtime_ns=stat.st_mtime_ns)

    def to_meta(self) -> dict[str, int]:
        return {"size": self.size, "mtime_ns": self.mtime_ns}

    @classmethod
    def from_meta(cls, meta: object) -> SourceFingerprint | None:
        if not isinstance(meta, dict):
            return None
        size, mtime_ns = meta.get("size"), meta.get("mtime_ns")
        if isinstance(size, int) and isinstance(mtime_ns, int):
            return cls(size=size, mtime_ns=mtime_ns)
        return None


@dataclass(frozen=True)
class CachedPeaks:
    """A readable cached envelope plus what's needed to trust/serve it."""

    body: bytes
    artifact_id: uuid.UUID
    source_fingerprint: SourceFingerprint | None


def peaks_artifact_row(session: Session, run_id: uuid.UUID) -> AudioArtifact | None:
    return session.execute(
        select(AudioArtifact)
        .where(
            AudioArtifact.pipeline_run_id == run_id,
            AudioArtifact.kind == ArtifactKind.WAVEFORM_PEAKS.value,
        )
        .limit(1)
    ).scalar_one_or_none()


def load_cached_peaks(
    session: Session, run_id: uuid.UUID, media_root: Path
) -> CachedPeaks | None:
    """Read the cached envelope, or None on any miss (no row, bad path, unreadable).

    Deliberately NOT MediaGate.open_for_serving — its ffprobe validation would
    (correctly) reject JSON. Confinement is re-checked here with the same
    resolve()/is_relative_to idiom because the path crosses a trust boundary:
    DB row -> filesystem.
    """
    row = peaks_artifact_row(session, run_id)
    if row is None:
        return None
    resolved = (media_root / row.path).resolve()
    if not resolved.is_relative_to(media_root.resolve()) or not resolved.is_file():
        return None
    try:
        body = resolved.read_bytes()
    except OSError:
        return None
    meta = row.meta or {}
    return CachedPeaks(
        body=body,
        artifact_id=row.id,
        source_fingerprint=SourceFingerprint.from_meta(meta.get("source_fingerprint")),
    )


def store_peaks(
    session: Session,
    run_id: uuid.UUID,
    media_root: Path,
    payload: PeaksPayload,
    fingerprint: SourceFingerprint,
) -> uuid.UUID:
    """Publish the envelope atomically and return the CANONICAL row id.

    The route serializes this per run with a transaction advisory lock, so for a
    given run only one compute+publish runs at a time; file bytes, the row's
    fingerprint, and the row-UUID ETag therefore always correspond. File: unique
    temp sibling + ``os.replace``. Row: any stale-fingerprint survivor is DELETED
    first — never refreshed in place, because the row UUID doubles as the strong
    ETag and changed bytes must mint a new one — then ``INSERT … ON CONFLICT DO
    NOTHING`` against the one-per-run partial unique index, then reselect (a
    belt-and-braces guard should the lock ever be bypassed).
    """
    relative = peaks_relative(run_id)
    target = media_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_bytes(payload.to_json_bytes())
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)

    meta = {
        "version": PEAKS_VERSION,
        "buckets": len(payload.peaks),
        "duration_seconds": payload.duration_seconds,
        "source_fingerprint": fingerprint.to_meta(),
    }
    existing = peaks_artifact_row(session, run_id)
    if existing is not None:
        stale = (
            SourceFingerprint.from_meta((existing.meta or {}).get("source_fingerprint"))
            != fingerprint
        )
        if not stale:
            return existing.id
        session.delete(existing)
        session.flush()
    session.execute(
        pg_insert(AudioArtifact)
        .values(
            id=uuid.uuid4(),
            pipeline_run_id=run_id,
            kind=ArtifactKind.WAVEFORM_PEAKS.value,
            path=relative,
            meta=meta,
        )
        # No explicit conflict target: the partial unique index needs its WHERE
        # predicate to be named as a target, and "any conflict" is what we mean.
        .on_conflict_do_nothing()
    )
    row = peaks_artifact_row(session, run_id)
    if row is None:  # pragma: no cover - the insert or a concurrent one exists
        raise RuntimeError(f"run {run_id}: waveform_peaks row vanished after insert")
    return row.id
