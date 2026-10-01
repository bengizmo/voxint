#!/usr/bin/env python3
"""TitaNet collision screen for the voiceclone target-voice pack (#664).

Runs in the Voxint dev venv with no extra dependencies. Every audio file is
consumed by sha256 and refused on mismatch. Audio reaches the embedder the way
the pipeline delivers it: ``voxint.media.normalize.normalize_to_wav`` (the
prepare stage's ffmpeg transcode to 16 kHz mono PCM16) writes a sha-named copy
under the media root, and the titanet service's ``/v1/embed`` embeds it as one
window. Run the service with the shipped ONNX engine (``EMBED_ENGINE=onnx``,
``titanet-onnx-v1``) and ``MEDIA_ROOT`` pointed at the pool root.

Subcommands:

``build-pool``
    Write the committed pool manifest: dataset pins plus each clip's id, path,
    condition and sha256. Pool audio is never committed or released.
``embed-pool``
    Embed every pool clip; write per-clip vectors to a local (uncommitted) file.
``screen``
    Embed the generator's candidates and write the screening report: maxima
    against speaker centroids and single clips, headroom to the 0.60 floor, null
    percentiles and the pairwise matrix.
``select``
    Read a listen-pass shortlist, choose one finalist per slot by maximising the
    minimum pairwise distance, and add the choice to the report.

The screen is a collision check against a named pool in Voxint's own embedding
space. It is not proof that a voice resembles nobody, and its report says so.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import wave
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import voiceclone_target_voice_core as core

POOL_SCHEMA_VERSION = 1
EMBEDDINGS_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
EXPECTED_SPACE = "titanet-large-v2"
EXPECTED_ENGINE = "onnxruntime"
TITANET_ONNX_RELEASE = "titanet-onnx-v1"
ONNX_PROVENANCE = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "parity"
    / "fixtures"
    / "onnx"
    / "provenance.json"
)
NORMALIZED_DIR = "norm16k"
SCREEN_LIMITS = (
    "A collision check against the named pool in Voxint's titanet-large-v2 space. "
    "It cannot screen the generator's undisclosed training data, and it is not "
    "proof that a voice resembles no real person."
)

# Embeds one MEDIA_ROOT-relative 16 kHz file as a single window.
# Returns the unit vector, or a skip reason string.
EmbedFn = Callable[[str, float], "np.ndarray | str"]
# Writes a 16 kHz mono PCM16 copy of src at dest; returns its duration in seconds.
NormalizeFn = Callable[[Path, Path], float]


class PoolError(ValueError):
    """The pool or its inputs are inconsistent; the tool fails closed."""


@dataclass(frozen=True)
class PoolClip:
    speaker: str
    clip_id: str
    path: str  # relative to the pool root
    sha256: str
    condition: str
    source: dict[str, Any]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_sha(path: Path, expected: str) -> None:
    actual = file_sha256(path)
    if actual != expected:
        raise PoolError(f"{path}: sha256 {actual} != expected {expected}")


# --- pool manifest --------------------------------------------------------------


def gate1_clips(manifest: Mapping[str, Any], prefix: str = "gate1") -> list[PoolClip]:
    """All 12 clips per Gate 1 speaker: the four references and the clean and
    phone copies of each held-out window."""
    clips: list[PoolClip] = []
    for spk in manifest["speakers"]:
        key = spk["speaker_key"]
        base = {"corpus": spk["corpus"], "source_label": spk["source_label"]}
        for ref in spk["refs"]:
            clips.append(
                PoolClip(
                    key,
                    ref["ref_id"],
                    f"{prefix}/{ref['path']}",
                    ref["sha256"],
                    ref["condition"],
                    {**base, **spk["ref_window"], "length_s": ref["length_s"]},
                )
            )
        for held in spk["heldout"]:
            window = {
                **base,
                "recording": held["recording"],
                "start_s": held["start_s"],
                "end_s": held["end_s"],
            }
            hid = held["heldout_id"]
            clips.append(
                PoolClip(key, hid, f"{prefix}/{held['path']}", held["sha256"], "clean", window)
            )
            clips.append(
                PoolClip(
                    key,
                    f"{hid}_phone",
                    f"{prefix}/{held['phone_path']}",
                    held["phone_sha256"],
                    "phone",
                    window,
                )
            )
    return clips


def wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def libritts_clips(
    subset_root: Path,
    pool_root: Path,
    per_speaker: int,
    min_seconds: float,
) -> tuple[list[PoolClip], list[str]]:
    """Up to ``per_speaker`` utterances per speaker: the first, by utterance id,
    that last at least ``min_seconds``. ``subset_root`` is ``.../test-clean``.

    Returns the clips and the speakers with no qualifying utterance (LibriTTS-R
    filtered some test-clean speakers down to one or two files)."""
    by_speaker: dict[str, list[Path]] = {}
    for wav in sorted(subset_root.glob("*/*/*.wav")):
        by_speaker.setdefault(wav.parts[-3], []).append(wav)
    if not by_speaker:
        raise PoolError(f"no LibriTTS-R utterances under {subset_root}")
    clips: list[PoolClip] = []
    excluded: list[str] = []
    for speaker in sorted(by_speaker, key=int):
        ordered = sorted(by_speaker[speaker], key=lambda p: p.stem)
        chosen = [w for w in ordered if wav_seconds(w) >= min_seconds][:per_speaker]
        if not chosen:
            excluded.append(f"libritts-r-{speaker}")
        for wav in chosen:
            clips.append(
                PoolClip(
                    f"libritts-r-{speaker}",
                    wav.stem,
                    wav.relative_to(pool_root).as_posix(),
                    file_sha256(wav),
                    "clean",
                    {"corpus": "libritts-r", "subset": "test-clean", "utterance": wav.stem},
                )
            )
    return clips, excluded


def build_pool_manifest(
    clips: Sequence[PoolClip], datasets: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    ids = [c.clip_id for c in clips]
    if len(set(ids)) != len(ids):
        raise PoolError("clip ids are not unique")
    return {
        "schema_version": POOL_SCHEMA_VERSION,
        "note": "Pool audio is never committed or released; clips are pinned by sha256.",
        "datasets": list(datasets),
        "speakers": len({c.speaker for c in clips}),
        "clips": [asdict(c) for c in clips],
    }


def load_pool_manifest(path: Path) -> list[PoolClip]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != POOL_SCHEMA_VERSION:
        raise PoolError(f"{path}: unsupported pool schema_version")
    return [PoolClip(**c) for c in data["clips"]]


# --- embedding -------------------------------------------------------------------


def normalized_copy(
    pool_root: Path, src_rel: str, sha256: str, normalize: NormalizeFn
) -> tuple[str, float]:
    """Verify ``src_rel`` by sha, then return the MEDIA_ROOT-relative path and
    duration of its 16 kHz copy (named by the source sha, made once)."""
    src = pool_root / src_rel
    verify_sha(src, sha256)
    rel = f"{NORMALIZED_DIR}/{sha256}.wav"
    dest = pool_root / rel
    duration = wav_seconds(dest) if dest.exists() else normalize(src, dest)
    return rel, duration


def embed_pool(
    clips: Sequence[PoolClip], pool_root: Path, embed: EmbedFn, normalize: NormalizeFn
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for clip in clips:
        rel, duration = normalized_copy(pool_root, clip.path, clip.sha256, normalize)
        result = embed(rel, duration)
        record: dict[str, Any] = {
            "speaker": clip.speaker,
            "clip_id": clip.clip_id,
            "sha256": clip.sha256,
            "condition": clip.condition,
        }
        if isinstance(result, str):
            record["skip_reason"] = result
        else:
            record["vector"] = [float(x) for x in result]
        out.append(record)
    return out


def pool_vectors(
    records: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, list[tuple[str, np.ndarray]]], list[str]]:
    """Group embedded clips by speaker; return the pool and the skipped clip ids."""
    pool: dict[str, list[tuple[str, np.ndarray]]] = {}
    skipped: list[str] = []
    for rec in records:
        if "vector" in rec:
            pool.setdefault(rec["speaker"], []).append(
                (rec["clip_id"], np.asarray(rec["vector"], dtype=np.float64))
            )
        else:
            skipped.append(rec["clip_id"])
    return pool, skipped


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "n": int(values.size),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


# --- screening + selection ---------------------------------------------------------


def load_candidates(candidates_dir: Path) -> list[dict[str, Any]]:
    """The generator's accepted candidates (``*.json`` records), sorted by id.

    Records with ``status: rejected`` (non-finite, silent or too short; they
    have no WAV) are skipped. Any other status fails closed.
    """
    records = []
    ids = []
    for path in sorted(candidates_dir.glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        status = rec.get("status")
        if status not in ("candidate", "rejected"):
            raise PoolError(f"{path}: record status {status!r} is not candidate/rejected")
        ids.append(rec.get("candidate_id"))
        if status == "rejected":
            continue
        for key in ("candidate_id", "voice_id", "wav", "wav_sha256"):
            if key not in rec:
                raise PoolError(f"{path}: candidate record lacks {key!r}")
        records.append(rec)
    if len(set(ids)) != len(ids):
        raise PoolError("candidate ids are not unique")
    if not records:
        raise PoolError(f"no accepted candidate records in {candidates_dir}")
    return records


def screen_report(
    candidates: Sequence[Mapping[str, Any]],
    candidate_vectors: Mapping[str, np.ndarray],
    pool_records: Sequence[Mapping[str, Any]],
    rules: core.ScreenRules,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    pool, skipped = pool_vectors(pool_records)
    if len(pool) < 2:
        raise PoolError("the pool needs at least two embedded speakers")
    centroids = {s: core.centroid([v for _, v in clips]) for s, clips in pool.items()}
    pairs = core.cross_speaker_cosines(centroids)
    nearest = core.nearest_neighbour_cosines(centroids)
    results: dict[str, Any] = {}
    for cand in candidates:
        cid = cand["candidate_id"]
        res = core.screen_candidate(candidate_vectors[cid], pool, centroids, rules)
        entry = res.as_dict()
        entry.pop("centroid_cosines")
        entry.update(
            voice_id=cand["voice_id"],
            wav_sha256=cand["wav_sha256"],
            median_f0_hz=cand.get("median_f0_hz"),
            null_percentile_nearest_neighbour=core.percentile_rank(
                res.max_centroid_cosine, nearest
            ),
            null_percentile_all_pairs=core.percentile_rank(res.max_centroid_cosine, pairs),
        )
        results[cid] = entry
    passing = {cid: candidate_vectors[cid] for cid, r in results.items() if not r["rejected"]}
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "limits": SCREEN_LIMITS,
        **provenance,
        "rules": asdict(rules),
        "centroid_policy": "equal-weight unit mean of each speaker's unit clip vectors",
        "pool_summary": {
            "speakers": len(pool),
            "clips_embedded": sum(len(v) for v in pool.values()),
            "clips_skipped": sorted(skipped),
        },
        "null": {
            "nearest_neighbour": _summary(nearest),
            "all_pairs": _summary(pairs),
        },
        "candidates": results,
        "pairwise_passing": core.pairwise_matrix(passing) if passing else {},
    }


def select_into_report(
    report: dict[str, Any],
    shortlist: Mapping[str, Sequence[str]],
    candidate_vectors: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Add the finalist choice to ``report``. Every shortlisted id must be a
    screened, non-rejected candidate of the slot it is listed under."""
    cands = report["candidates"]
    for slot, ids in shortlist.items():
        for cid in ids:
            if cid not in cands:
                raise PoolError(f"shortlisted {cid} was not screened")
            if cands[cid]["rejected"]:
                raise PoolError(f"shortlisted {cid} was rejected by the screen")
            if cands[cid]["voice_id"] != slot:
                raise PoolError(
                    f"shortlisted {cid} belongs to {cands[cid]['voice_id']}, not {slot}"
                )
    finalists = core.select_finalists(shortlist, candidate_vectors)
    chosen = {cid: candidate_vectors[cid] for cid in finalists.values()}
    out = dict(report)
    out["selection"] = {
        "shortlist": {k: list(v) for k, v in sorted(shortlist.items())},
        "rule": "maximise the minimum pairwise distance (minimise the largest cosine)",
        "finalists": finalists,
        "finalist_pairwise": core.pairwise_matrix(chosen),
        "finalist_median_f0_hz": {s: cands[c]["median_f0_hz"] for s, c in finalists.items()},
        "swap_candidates": sorted(c for c in finalists.values() if cands[c]["swap_candidate"]),
    }
    return out


# --- I/O adapters (live service + ffmpeg) -----------------------------------------


def titanet_health(base_url: str) -> dict[str, Any]:
    import httpx

    health = httpx.get(f"{base_url}/healthz", timeout=30).raise_for_status().json()
    if health.get("embedding_space") != EXPECTED_SPACE:
        raise PoolError(f"titanet space {health.get('embedding_space')!r} != {EXPECTED_SPACE}")
    if health.get("engine") != EXPECTED_ENGINE:
        raise PoolError(f"titanet engine {health.get('engine')!r} != {EXPECTED_ENGINE}")
    keys = ("model", "engine", "engine_version", "runtime", "runtime_version", "embedding_space")
    return {k: health.get(k) for k in keys} | {"service_version": health.get("version")}


def onnx_graph_identity(graph: Path, provenance: Path = ONNX_PROVENANCE) -> dict[str, str]:
    """The ONNX graph the service loads must be the shipped ``titanet-onnx-v1``
    asset: its sha256 must equal the committed provenance."""
    expected = json.loads(provenance.read_text(encoding="utf-8"))["onnx_sha256"]
    actual = file_sha256(graph)
    if actual != expected:
        raise PoolError(f"{graph}: sha256 {actual} is not the {TITANET_ONNX_RELEASE} graph")
    return {"release": TITANET_ONNX_RELEASE, "onnx_sha256": actual}


def http_embedder(base_url: str) -> EmbedFn:
    import httpx

    client = httpx.Client(base_url=base_url, timeout=600)

    def embed(rel: str, duration: float) -> np.ndarray | str:
        body = {"path": rel, "windows": [{"start_seconds": 0.0, "end_seconds": duration}]}
        resp = client.post("/v1/embed", json=body).raise_for_status().json()
        if resp["embedding_space"] != EXPECTED_SPACE:
            raise PoolError(f"embed returned space {resp['embedding_space']!r}")
        (result,) = resp["results"]
        if result["embedding"] is None:
            return str(result["skip_reason"])
        return np.asarray(result["embedding"], dtype=np.float64)

    return embed


def ffmpeg_normalizer() -> NormalizeFn:
    from voxint.media.normalize import normalize_to_wav

    def normalize(src: Path, dest: Path) -> float:
        return normalize_to_wav(src, dest).duration_seconds

    return normalize


def normalizer_provenance() -> dict[str, str]:
    out = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True)
    return {
        "resampler": "voxint.media.normalize.normalize_to_wav (ffmpeg, 16 kHz mono pcm_s16le)",
        "ffmpeg_version": out.stdout.splitlines()[0].strip(),
    }


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _load_embeddings(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != EMBEDDINGS_SCHEMA_VERSION:
        raise PoolError(f"{path}: unsupported embeddings schema_version")
    return dict(data)


# --- CLI ---------------------------------------------------------------------------


def cmd_build_pool(args: argparse.Namespace) -> None:
    root: Path = args.pool_root
    gate1_manifest_path = root / "gate1" / "manifest.json"
    gate1 = json.loads(gate1_manifest_path.read_text(encoding="utf-8"))
    src = gate1["sources"]
    libri, libri_excluded = libritts_clips(
        root / args.libritts_subset, root, args.per_speaker, args.min_seconds
    )
    clips = gate1_clips(gate1) + libri
    for clip in clips:
        verify_sha(root / clip.path, clip.sha256)
    datasets = [
        {
            "name": "voiceclone-gate1-refset",
            "description": "The 15 AMI and VoxConverse speakers of the V1 Gate 1 refset "
            "(12 clips each: 10 s and 3 s references, four held-out windows, each "
            "clean and phone-band).",
            "license": "CC-BY-4.0 (AMI Meeting Corpus; VoxConverse)",
            "refset_manifest_sha256": file_sha256(gate1_manifest_path),
            "ami_words_zip_sha256": src["ami_words_zip_sha256"],
            "ami_meetings_xml_sha256": src["meetings_xml_sha256"],
            "voxconverse_repo_sha": src["voxconverse_repo_sha"],
            "voxconverse_rttm_sha256": src["voxconverse_rttm_sha256"],
        },
        {
            "name": "LibriTTS-R test-clean",
            "url": "https://www.openslr.org/resources/141/test_clean.tar.gz",
            "archive_sha256": args.libritts_archive_sha256,
            "license": "CC-BY-4.0",
            "excluded_speakers": libri_excluded,
            "selection": f"up to {args.per_speaker} utterances per speaker, the first by "
            "utterance id "
            f"lasting at least {args.min_seconds} s",
        },
    ]
    _write_json(args.out, build_pool_manifest(clips, datasets))
    print(f"wrote {args.out}: {len(clips)} clips, {len({c.speaker for c in clips})} speakers")


def cmd_embed_pool(args: argparse.Namespace) -> None:
    clips = load_pool_manifest(args.manifest)
    graph = onnx_graph_identity(args.onnx_graph)
    health = titanet_health(args.titanet_url)
    records = embed_pool(
        clips, args.pool_root, http_embedder(args.titanet_url), ffmpeg_normalizer()
    )
    _write_json(
        args.out,
        {
            "schema_version": EMBEDDINGS_SCHEMA_VERSION,
            "pool_manifest_sha256": file_sha256(args.manifest),
            "embedder": health,
            "onnx_graph": graph,
            "normalizer": normalizer_provenance(),
            "clips": records,
        },
    )
    skipped = sum(1 for r in records if "skip_reason" in r)
    print(f"wrote {args.out}: {len(records)} clips, {skipped} skipped")


def cmd_screen(args: argparse.Namespace) -> None:
    spec = core.load_spec(args.spec)
    rules = core.ScreenRules.from_spec(spec)
    emb = _load_embeddings(args.pool_embeddings)
    if emb["pool_manifest_sha256"] != file_sha256(args.manifest):
        raise PoolError("pool embeddings were made from a different pool manifest")
    if onnx_graph_identity(args.onnx_graph) != emb["onnx_graph"]:
        raise PoolError("the ONNX graph changed since the pool was embedded")
    health = titanet_health(args.titanet_url)
    if health != emb["embedder"]:
        raise PoolError(
            f"titanet changed since the pool was embedded: {health} != {emb['embedder']}"
        )
    candidates = load_candidates(args.candidates_dir)
    embed = http_embedder(args.titanet_url)
    normalize = ffmpeg_normalizer()
    vectors: dict[str, np.ndarray] = {}
    for cand in candidates:
        src_rel = (
            (args.candidates_dir / cand["wav"]).resolve().relative_to(args.pool_root.resolve())
        )
        rel, duration = normalized_copy(
            args.pool_root, src_rel.as_posix(), cand["wav_sha256"], normalize
        )
        result = embed(rel, duration)
        if isinstance(result, str):
            raise PoolError(f"candidate {cand['candidate_id']} was skipped by titanet: {result}")
        vectors[cand["candidate_id"]] = result
    provenance = {
        "embedder": health,
        "onnx_graph": emb["onnx_graph"],
        "normalizer": emb["normalizer"],
        "pool_manifest_sha256": emb["pool_manifest_sha256"],
        "pool_manifest": args.manifest.name,
    }
    report = screen_report(candidates, vectors, emb["clips"], rules, provenance)
    _write_json(args.out, report)
    _write_json(
        args.out.with_name(args.out.stem + ".vectors.json"),
        {cid: [float(x) for x in v] for cid, v in vectors.items()},
    )
    rejected = sorted(c for c, r in report["candidates"].items() if r["rejected"])
    print(f"wrote {args.out}: {len(candidates)} candidates, rejected {rejected or 'none'}")


def cmd_select(args: argparse.Namespace) -> None:
    report = json.loads(args.report.read_text(encoding="utf-8"))
    vectors_path = args.report.with_name(args.report.stem + ".vectors.json")
    raw = json.loads(vectors_path.read_text(encoding="utf-8"))
    vectors = {k: np.asarray(v, dtype=np.float64) for k, v in raw.items()}
    shortlist = json.loads(args.shortlist.read_text(encoding="utf-8"))
    out = select_into_report(report, shortlist, vectors)
    _write_json(args.report, out)
    print(f"finalists: {out['selection']['finalists']}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-pool")
    b.add_argument("--pool-root", type=Path, required=True)
    b.add_argument("--libritts-subset", default="libritts_r/LibriTTS_R/test-clean")
    b.add_argument("--libritts-archive-sha256", required=True)
    b.add_argument("--per-speaker", type=int, default=8)
    b.add_argument("--min-seconds", type=float, default=3.0)
    b.add_argument("--out", type=Path, required=True)
    e = sub.add_parser("embed-pool")
    e.add_argument("--manifest", type=Path, required=True)
    e.add_argument("--pool-root", type=Path, required=True)
    e.add_argument("--titanet-url", default="http://127.0.0.1:8021")
    e.add_argument("--onnx-graph", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("screen")
    s.add_argument("--spec", type=Path, required=True)
    s.add_argument("--manifest", type=Path, required=True)
    s.add_argument("--pool-embeddings", type=Path, required=True)
    s.add_argument("--pool-root", type=Path, required=True)
    s.add_argument("--candidates-dir", type=Path, required=True)
    s.add_argument("--titanet-url", default="http://127.0.0.1:8021")
    s.add_argument("--onnx-graph", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    sel = sub.add_parser("select")
    sel.add_argument("--report", type=Path, required=True)
    sel.add_argument("--shortlist", type=Path, required=True)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    handlers = {
        "build-pool": cmd_build_pool,
        "embed-pool": cmd_embed_pool,
        "screen": cmd_screen,
        "select": cmd_select,
    }
    handlers[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
