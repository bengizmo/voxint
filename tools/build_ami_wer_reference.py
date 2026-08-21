#!/usr/bin/env python3
"""Freeze the AMI WER reference text for the eval-quality harness (issue #97).

The eval-quality harness (``tools/eval_quality.py``) scores pooled ASR word
error rate on AMI. AMI ships its transcripts as per-speaker, word-aligned NXT
XML (``words/{meeting}.{agent}.words.xml``), not as a ready reference string, so
this maintainer step does the one error-prone extraction once and freezes the
result: for each meeting in the scoring subset it merges every speaker's timed
words into one chronological stream, crops to the official scoring region (UEM),
and writes the raw merged text plus an integrity manifest beside the ground
truth. The harness then loads that frozen text and never re-parses the XML.

What this tool deliberately does NOT do: normalize. Per the numerics doctrine
(``tests/parity/bakeoff/normalize.py``), WER normalization is applied to raw
reference and raw hypothesis together at scoring time; storing a pre-normalized
reference would let a normalizer bump silently desync the two sides. So the
frozen artifact is the raw merged stream, and the harness records which
normalizer scored it. Absolute AMI WER will look high (interleaving overlapped
speech into one stream plus normalization offsets is a constant condition); the
harness gates on deltas, not the absolute number.

AMI word parsing is reused verbatim from ``prepare_bakeoff_corpus`` so the
codebase keeps a single AMI XML parser — two parsers would be a silent-drift
generator, exactly what the doctrine forbids.

Run (off-repo data; corpus root + out dir are always CLI args, never hardcoded):

    uv run --isolated tools/build_ami_wer_reference.py \\
        --ami-root  /path/to/groundtruth/ami \\
        --subset    /path/to/groundtruth/scoring_subset.json \\
        --out-dir   /path/to/groundtruth/ami/wer_reference

``--ami-root`` must contain ``annotations/words/`` (the NXT word XML) and
``AMI-diarization-setup-main/uems/{dev,test}/`` (the official UEMs).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

# Single-source-of-truth AMI parser + hashing helpers (issue #33 bakeoff tool).
import prepare_bakeoff_corpus as bake  # noqa: E402

# The frozen artifact is intended for this normalizer; the harness applies it.
# Recorded as provenance metadata only — importing the constant is cheap and
# does not pull the normalizer runtime (that stays a scoring-time concern).
#
# Schema 2 (issue #97 commit 3) adds the per-speaker cpWER reference:
# ``{id}.cpwer_reference.json`` holds the SAME cropped ``scored`` words the
# merged ``.words.txt`` is built from, bucketed by speaker in chronological order,
# so the harness can score concatenated minimum-permutation WER (cpWER) against
# the same ground truth. Both artifacts derive from ONE cropped word list (no
# second parse), and the per-speaker grouping is checked to conserve exactly the
# merged words (``canonical_words_sha256`` identity), so the two can never drift.
SCHEMA_VERSION = 2

# Reserved cpWER stream-label prefixes (mirrors eval_quality's collision-free
# encoding): a real speaker becomes ``speaker:<agent>`` so a model-emitted
# hypothesis label can never collide with the anonymous ``unassigned:`` bucket.
# The reference has no anonymous stream, but it uses the same ``speaker:`` prefix
# so reference and hypothesis stream keys share one namespace.
SPEAKER_LABEL_PREFIX = "speaker:"


def _canonical_json_bytes(payload: Any) -> bytes:
    """Deterministic JSON bytes: sorted keys, compact, UTF-8, no NaN/Inf.

    The single canonical encoder shared by every artifact this tool writes, so a
    hash taken here is reproducible by the harness that reads it back.
    """
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


@dataclass(frozen=True)
class TimedWord:
    """One AMI reference word tagged with its speaker (agent letter)."""

    start_us: int
    end_us: int
    text: str
    speaker: str

    @property
    def mid_us(self) -> int:
        return (self.start_us + self.end_us) // 2


@dataclass(frozen=True)
class UemRegion:
    """A scored interval from a UEM file, in integer microseconds."""

    start_us: int
    end_us: int


class BuildError(Exception):
    """A user-facing input problem (missing file, malformed UEM)."""


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def parse_uem(text: str, recording_id: str) -> list[UemRegion]:
    """Parse a NIST UEM file, keeping only rows for ``recording_id``.

    Row layout: ``<uri> <channel> <start> <end>``. Times are seconds; they are
    canonicalized to integer microseconds (``bake._us``) so a region boundary
    never depends on float formatting. Rows for other recordings are ignored so
    a per-split UEM (many recordings in one file) also works.
    """
    regions: list[UemRegion] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith(";;"):
            continue
        parts = line.split()
        if len(parts) < 4:
            raise BuildError(f"UEM line {lineno}: expected 4 fields, got {parts!r}")
        uri = parts[0]
        if uri != recording_id:
            continue
        start_us = bake._us(parts[2])
        end_us = bake._us(parts[3])
        if end_us <= start_us:
            raise BuildError(f"UEM line {lineno}: non-positive region {start_us}..{end_us}")
        regions.append(UemRegion(start_us=start_us, end_us=end_us))
    if not regions:
        raise BuildError(f"UEM has no region for recording {recording_id!r}")
    return regions


def load_speaker_words(words_dir: Path, recording_id: str) -> list[TimedWord]:
    """Load every ``{recording_id}.{agent}.words.xml`` file into tagged words.

    Reuses ``bake.parse_nxt_words`` (which filters to ``<w>`` elements with both
    times and a non-empty token) so extraction matches the bakeoff exactly.
    """
    paths = sorted(words_dir.glob(f"{recording_id}.*.words.xml"))
    if not paths:
        raise BuildError(f"no word XML for {recording_id!r} under {words_dir}")
    words: list[TimedWord] = []
    for path in paths:
        # filename is "<id>.<agent>.words.xml"; the agent letter is the speaker.
        speaker = path.name.split(".")[1]
        for nxt in bake.parse_nxt_words(path.read_bytes()):
            words.append(
                TimedWord(
                    start_us=nxt.start_us,
                    end_us=nxt.end_us,
                    text=nxt.text,
                    speaker=speaker,
                )
            )
    return words


# --------------------------------------------------------------------------- #
# Merge + crop
# --------------------------------------------------------------------------- #
def in_any_region(word: TimedWord, regions: list[UemRegion]) -> bool:
    """A word is scored if its MIDPOINT falls in a UEM region.

    Midpoint (not overlap) keeps a word straddling a region edge from being
    counted twice against disjoint regions, and matches how short lexical tokens
    are conventionally assigned to a scoring interval.
    """
    return any(r.start_us <= word.mid_us < r.end_us for r in regions)


def merge_chronologically(words: list[TimedWord]) -> list[TimedWord]:
    """Deterministic chronological merge across speakers.

    Sort by (start, end, speaker, text) so overlapping speech interleaves the
    same way on every machine. This interleaving is why absolute AMI WER is high
    — an accepted, constant condition the harness gates deltas against.
    """
    return sorted(words, key=lambda w: (w.start_us, w.end_us, w.speaker, w.text))


def group_cpwer_streams(merged: list[TimedWord]) -> dict[str, list[str]]:
    """Bucket the merged words by speaker into cpWER reference streams.

    Iterating the globally-sorted ``merged`` list and appending per speaker keeps
    each speaker's words in chronological order (a stable subsequence) WITHOUT a
    second sort, and derives the per-speaker streams from the exact same word
    objects the merged ``.words.txt`` is built from, so the two artifacts cannot
    drift. Keys use the collision-free ``speaker:<agent>`` encoding the harness
    reads back (see :data:`SPEAKER_LABEL_PREFIX`). A speaker whose words all fell
    outside the UEM is simply absent from ``merged`` and so is omitted here (no
    empty streams, which meeteval's cpWER would reject as a 0-length denominator).
    """
    streams: dict[str, list[str]] = {}
    for w in merged:
        streams.setdefault(f"{SPEAKER_LABEL_PREFIX}{w.speaker}", []).append(w.text)
    return streams


def build_reference(
    ami_root: Path, recording_id: str, split: str
) -> tuple[str, dict[str, Any], dict[str, list[str]]]:
    """Return ``(raw_merged_text, integrity_record, cpwer_streams)`` for a meeting.

    ``cpwer_streams`` is the per-speaker cpWER reference (raw words, un-normalized;
    the harness normalizes each stream at scoring time exactly as it does the
    plain-WER stream). It is an occurrence-partition of the merged words: every
    merged word enters exactly one speaker stream, checked here so the merged
    ``.words.txt`` and the per-speaker JSON provably score the same ground truth.
    """
    words_dir = ami_root / "annotations" / "words"
    uem_path = (
        ami_root / "AMI-diarization-setup-main" / "uems" / split / f"{recording_id}.uem"
    )
    if not uem_path.is_file():
        raise BuildError(f"missing UEM {uem_path}")

    regions = parse_uem(uem_path.read_text(encoding="utf-8"), recording_id)
    all_words = load_speaker_words(words_dir, recording_id)
    scored = [w for w in all_words if in_any_region(w, regions)]
    merged = merge_chronologically(scored)

    text = " ".join(w.text for w in merged)
    text_bytes = text.encode("utf-8")
    # Order/format-independent integrity hash: integer-us word tuples, canonical
    # JSON (mirrors prepare_bakeoff_corpus's transcript_sha256 convention).
    canonical = bake.canonical_reference_bytes(
        {
            "kind": "ami-wer-reference",
            "recording_id": recording_id,
            "words": [
                {"start_us": w.start_us, "end_us": w.end_us, "text": w.text}
                for w in merged
            ],
        }
    )

    cpwer_streams = group_cpwer_streams(merged)
    cpwer_counts = {key: len(words) for key, words in cpwer_streams.items()}
    # Occurrence-partition conservation: the per-speaker streams together hold
    # exactly the merged words, no more, no fewer. A mismatch means the grouping
    # dropped or duplicated a word, which would silently move cpWER.
    grouped_total = sum(cpwer_counts.values())
    if grouped_total != len(merged):
        raise BuildError(
            f"{recording_id}: cpWER streams hold {grouped_total} words but the merged "
            f"reference has {len(merged)} (grouping is not an occurrence-partition)"
        )

    record = {
        "recording_id": recording_id,
        "split": split,
        "uem_regions_us": [[r.start_us, r.end_us] for r in regions],
        "evaluated_duration_s": sum(r.end_us - r.start_us for r in regions) / 1_000_000,
        "speakers": sorted({w.speaker for w in merged}),
        "word_count": len(merged),
        "words_dropped_outside_uem": len(all_words) - len(scored),
        "text_sha256": bake.sha256_hex(text_bytes),
        "canonical_words_sha256": bake.sha256_hex(canonical),
        "text_bytes": len(text_bytes),
        "cpwer_speaker_word_counts": cpwer_counts,
    }
    return text, record, cpwer_streams


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def ami_ids_from_subset(subset_path: Path) -> list[tuple[str, str]]:
    """Return ``(recording_id, split)`` for every AMI entry in the subset."""
    payload = json.loads(subset_path.read_text(encoding="utf-8"))
    out: list[tuple[str, str]] = []
    for entry in payload.get("files", []):
        if entry.get("corpus") != "ami":
            continue
        rec_id = entry["id"]
        split = entry["split"]
        out.append((rec_id, split))
    if not out:
        raise BuildError(f"{subset_path}: no AMI entries")
    return out


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError):  # pragma: no cover - env dependent
        return "unknown"


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def run(args: argparse.Namespace) -> int:
    ami_root = Path(args.ami_root)
    out_dir = Path(args.out_dir)
    if args.ids:
        subset_payload = json.loads(Path(args.subset).read_text(encoding="utf-8"))
        split_by_id = {e["id"]: e["split"] for e in subset_payload.get("files", [])}
        try:
            targets = [(rid, split_by_id[rid]) for rid in args.ids]
        except KeyError as exc:
            raise BuildError(f"id {exc} not in subset {args.subset}") from exc
    else:
        targets = ami_ids_from_subset(Path(args.subset))

    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for rec_id, split in targets:
        text, record, cpwer_streams = build_reference(ami_root, rec_id, split)
        _write_atomic(out_dir / f"{rec_id}.words.txt", text + "\n")
        cpwer_reference = {
            "schema_version": SCHEMA_VERSION,
            "kind": "ami-cpwer-reference",
            "recording_id": rec_id,
            "split": split,
            "streams": cpwer_streams,
            "speaker_word_counts": record["cpwer_speaker_word_counts"],
            "word_count": record["word_count"],
            # Cross-link to the merged reference: the harness binds the bytes of
            # THIS file in the cohort, and this hash ties it back to the same
            # cropped word set the .words.txt was built from.
            "canonical_words_sha256": record["canonical_words_sha256"],
        }
        # Canonical bytes + a trailing newline, written atomically like the sibling
        # .words.txt so a crash never leaves a half-written reference.
        _write_atomic(
            out_dir / f"{rec_id}.cpwer_reference.json",
            _canonical_json_bytes(cpwer_reference).decode("utf-8") + "\n",
        )
        records.append(record)
        preview = text[:160].replace("\n", " ")
        print(
            f"{rec_id} [{split}]: {record['word_count']} words, "
            f"{record['evaluated_duration_s']:.1f}s, "
            f"dropped {record['words_dropped_outside_uem']} outside UEM, "
            f"sha {record['text_sha256'][:12]}",
            file=sys.stderr,
        )
        print(f"    preview: {preview!r}", file=sys.stderr)

    provenance = {
        "schema_version": SCHEMA_VERSION,
        "kind": "ami-wer-reference",
        "source": "ami_public_manual_1.6.2 NXT word annotations",
        "builder_git_sha": _git_sha(),
        "parser": "prepare_bakeoff_corpus.parse_nxt_words",
        "crop": "midpoint-in-UEM",
        "merge": "chronological (start,end,speaker,text)",
        "normalization": "none (harness normalizes raw ref+hyp together at scoring time)",
        "references": sorted(records, key=lambda r: r["recording_id"]),
    }
    _write_atomic(
        out_dir / "provenance.json",
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
    )
    print(f"wrote {len(records)} references + provenance.json to {out_dir}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ami-root", required=True, help="AMI ground-truth root")
    parser.add_argument("--subset", required=True, help="scoring_subset.json")
    parser.add_argument("--out-dir", required=True, help="output dir (off-repo)")
    parser.add_argument(
        "--ids", nargs="*", help="explicit recording ids (default: all AMI in subset)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except BuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
