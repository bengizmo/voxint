#!/usr/bin/env python3
"""Pure contracts for the eval-quality ``run`` step (issue #97, commit 1 of 2).

The ``run`` step drives the live pipeline over the ground-truth subset, then
emits the relabelled hypothesis RTTM/text that ``eval_quality.py score``
consumes. That live driver (submit -> poll -> read DB -> export) needs an idle
worker and lands in commit 2. THIS module is everything the driver stands on
that is testable WITHOUT a worker, so the schemas and their invariants are
frozen and unit-covered before any submission code exists:

* subset + path resolution (which recording maps to which audio/reference/UEM),
  with ambiguous-VoxConverse rejection and filename-safe recording ids;
* the AMI WER hypothesis-text builder, cropped by word MIDPOINT against the AMI
  UEM with the SAME integer-microsecond rule as the frozen reference
  (``build_ami_wer_reference.py``), so hypothesis and reference stay aligned;
* the ``pipeline_environment`` identity schema (model-weight + code + runtime +
  decode identity) that binds a metrics JSON to exactly what produced it;
* the cohort descriptor + hash that lets ``report`` reject a mislabelled
  "zero-change" set (different ids, audio, references, or pipeline identity);
* the artifact journal + its pure resume decision (write-ahead submission
  intent, per-attempt records, explicit resume/retry), so an interrupted batch
  never duplicates a run nor infers state from a stray file.

Nothing here imports the ORM, opens a socket, or touches a live service; the
driver in commit 2 imports these helpers and adds only the IO. Design reviewed
by a codex planner consult (Q: preserve ASR word order, not timestamp-sort;
include audio in the cohort identity; stream-verify WAV length, do not trust the
header nframes).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import tempfile
import wave
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
_TOOLS = REPO / "tools"
if str(_TOOLS) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_TOOLS))

# Single source of truth for the seconds -> integer-microsecond conversion and
# the byte-hash helper (issue #33 bakeoff tool). The AMI WER hypothesis MUST be
# cropped with the exact same ``_us`` the frozen reference used, or a word that
# the reference kept (or dropped) at a UEM boundary would fall on the other side
# in the hypothesis and desync the two streams.
import prepare_bakeoff_corpus as bake  # noqa: E402

JOURNAL_SCHEMA_VERSION = 1
PIPELINE_ENVIRONMENT_SCHEMA_VERSION = 1
COHORT_SCHEMA_VERSION = 1

CORPORA = ("ami", "voxconverse")
SPLITS = ("dev", "test")

# A recording id becomes a path component and a run label, so it must be a plain
# filename token: no separators, no traversal, no dotfiles. This is the
# path-containment guard codex flagged (an id like ``../../etc`` must never
# resolve a path outside the corpus root).
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# Synology drops a metadata directory named ``@eaDir`` beside real files on the
# NAS that holds the corpora; it must never be mistaken for an audio candidate.
_SYNOLOGY_META = "@eaDir"

# The live DB status vocabulary is ``voxint.db.models.RunStatus`` (stored
# lowercase): queued, running, awaiting_adjudication, completed, failed,
# cancelled. There is NO ``pending``/``submitted`` state — an earlier draft
# invented those, which would have made the driver blind-copy a status the DB
# never emits. The driver must map a polled DB status through ``map_db_status``
# (fail-closed on anything unknown) rather than trust an arbitrary string.
RUN_STATUSES = frozenset(
    {"queued", "running", "awaiting_adjudication", "completed", "failed", "cancelled"}
)

# Journal-only write-ahead states the DB never emits: recorded BEFORE the run
# uuid is known (``submitting``) or when a submit's outcome is unknown
# (``submission_unknown``). Kept disjoint from the DB vocabulary so a mapping
# bug can't disguise one as the other.
JOURNAL_ONLY_STATES = frozenset({"submitting", "submission_unknown"})
UNKNOWN_SUBMISSION = "submission_unknown"

COMPLETED = "completed"
# A run in ``awaiting_adjudication`` is NOT terminal: Voxint can resume it to
# ``running``, so the journal must be able to re-poll it rather than treat it as
# a stop. ``submitting`` is the write-ahead state whose recorded run is polled
# once its uuid lands (else re-submitted idempotently on resume).
RESUMABLE_STATES = frozenset({"queued", "running", "awaiting_adjudication", "submitting"})
FAILURE_STATES = frozenset({"failed", "cancelled"})

# Which artifact shas a completed run MUST carry before ``skip_done`` is safe,
# per corpus (AMI also produces a WER hypothesis text; VoxConverse does not).
# A completed journal item missing any of these is treated as incomplete
# (STOP), never silently skipped into a fake zero-change pass.
REQUIRED_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "ami": ("hypothesis_rttm_sha256", "wer_text_sha256"),
    "voxconverse": ("hypothesis_rttm_sha256",),
}


def map_db_status(db_status: Any) -> str:
    """Map a polled DB ``RunStatus`` to a journal status, failing closed.

    The DB and journal share the live status vocabulary, so this is an identity
    map WITH validation: an unknown/misspelled/None status raises rather than
    being copied verbatim into the journal, where it would later fall through
    ``plan_resume`` to an opaque STOP. The write-ahead journal-only states
    (``submitting``/``submission_unknown``) are never produced by the DB and are
    rejected here on purpose.
    """
    if db_status not in RUN_STATUSES:
        raise RunError(f"unrecognized DB run status {db_status!r} (not a RunStatus value)")
    return str(db_status)


class RunError(Exception):
    """A user-facing input problem (bad subset, ambiguous path, stale journal)."""


# --------------------------------------------------------------------------- #
# Canonical hashing (domain-separated; stable across dict order)
# --------------------------------------------------------------------------- #
def _canonical_bytes(payload: Any) -> bytes:
    """Deterministic JSON bytes: sorted keys, compact, UTF-8, no NaN/Inf.

    ``allow_nan=False`` refuses a non-finite float rather than emit the
    JSON-invalid ``NaN`` token, so a bad number fails loudly instead of
    producing a hash that other JSON parsers reject.
    """
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- #
# 1. AMI WER hypothesis text (pure; midpoint-cropped, ASR order preserved)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HypWord:
    """One ASR hypothesis word with its timing, in integer microseconds."""

    start_us: int
    end_us: int
    text: str

    @property
    def mid_us(self) -> int:
        return (self.start_us + self.end_us) // 2


def _coerce_word(raw: dict[str, Any]) -> HypWord:
    """Validate one stored word object ``{start,end,word,confidence}`` -> HypWord.

    ``start``/``end`` are float seconds converted with the SAME ``bake._us`` the
    reference used (Decimal, not float, so a boundary never depends on binary
    float formatting). The token keeps its stored spacing here; the builder
    strips it. A missing key or a non-numeric time is a precondition failure to
    surface, never to paper over.
    """
    try:
        start = raw["start"]
        end = raw["end"]
        token = raw["word"]
    except (KeyError, TypeError) as exc:
        raise RunError(f"malformed hypothesis word {raw!r}: missing key {exc}") from exc
    if not isinstance(token, str):
        raise RunError(f"hypothesis word token is not a string: {token!r}")
    try:
        start_us = bake._us(start)
        end_us = bake._us(end)
    except (ValueError, TypeError, ArithmeticError) as exc:
        raise RunError(f"hypothesis word has non-numeric timing {raw!r}: {exc}") from exc
    if end_us < start_us:
        raise RunError(f"hypothesis word end<start: {raw!r}")
    return HypWord(start_us=start_us, end_us=end_us, text=token)


def ami_hypothesis_text(
    segments_words: list[list[dict[str, Any]]], uem_regions_us: list[tuple[int, int]]
) -> str:
    """Build the UEM-cropped AMI WER hypothesis from stored per-word timings.

    ``segments_words`` is the run's transcript, one inner list of word objects
    per ``TranscriptSegment`` ORDERED by ``segment_index`` (the caller in commit
    2 supplies that order from the DB). Words are flattened in provider order and
    NOT re-sorted: this is a single ASR stream whose order is the decoder's, and
    timestamp-sorting could reorder tied or jittered tokens and move WER. (The
    reference sorts only because it merges independent per-speaker streams.)

    A word is kept when its MIDPOINT lies in some UEM region, half-open
    ``start_us <= mid_us < end_us`` — identical to the reference's
    ``in_any_region``. The kept tokens are stripped of their stored leading space
    and joined with single spaces, matching the reference's ``" ".join``. An
    empty result is legitimate (a recording whose speech is all outside the UEM),
    not an error.
    """
    if not uem_regions_us:
        raise RunError("ami_hypothesis_text: no UEM regions (AMI WER is always UEM-cropped)")
    kept: list[str] = []
    for segment in segments_words:
        for raw in segment:
            word = _coerce_word(raw)
            if any(start <= word.mid_us < end for start, end in uem_regions_us):
                stripped = word.text.strip()
                if stripped:
                    kept.append(stripped)
    return " ".join(kept)


# --------------------------------------------------------------------------- #
# 2. Subset validation + selection
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SubsetItem:
    """One validated scoring-subset entry."""

    corpus: str
    split: str
    recording_id: str
    num_speakers: int
    extent_s: float


def _validate_item(raw: dict[str, Any], index: int) -> SubsetItem:
    where = f"subset item {index}"
    try:
        corpus = raw["corpus"]
        split = raw["split"]
        rec_id = raw["id"]
        num_speakers = raw["num_speakers"]
        extent_s = raw["extent_s"]
    except (KeyError, TypeError) as exc:
        raise RunError(f"{where}: missing field {exc}") from exc
    if corpus not in CORPORA:
        raise RunError(f"{where}: corpus must be one of {CORPORA}, got {corpus!r}")
    if split not in SPLITS:
        raise RunError(f"{where}: split must be one of {SPLITS}, got {split!r}")
    if not isinstance(rec_id, str) or not _SAFE_ID.match(rec_id):
        raise RunError(f"{where}: id {rec_id!r} is not a filename-safe token")
    if not isinstance(num_speakers, int) or isinstance(num_speakers, bool) or num_speakers < 1:
        raise RunError(f"{where}: num_speakers must be an int >= 1, got {num_speakers!r}")
    if not isinstance(extent_s, (int, float)) or isinstance(extent_s, bool) or extent_s <= 0:
        raise RunError(f"{where}: extent_s must be > 0, got {extent_s!r}")
    return SubsetItem(corpus, split, rec_id, int(num_speakers), float(extent_s))


def load_subset(entries: list[dict[str, Any]], corpus: str) -> list[SubsetItem]:
    """Validate a scoring-subset list and return the items for ONE corpus.

    ``run`` is single-corpus per invocation (the journal and cohort carry a
    scalar corpus), so a mixed subset is filtered to ``corpus`` here. Duplicate
    ids WITHIN the selected corpus are rejected; the known-bad recording is
    excluded by leaving it out of the subset metadata, never by a hardcoded id.
    """
    if corpus not in CORPORA:
        raise RunError(f"corpus must be one of {CORPORA}, got {corpus!r}")
    items = [_validate_item(raw, i) for i, raw in enumerate(entries)]
    selected = [it for it in items if it.corpus == corpus]
    seen: set[str] = set()
    for it in selected:
        if it.recording_id in seen:
            raise RunError(f"duplicate recording id {it.recording_id!r} in corpus {corpus!r}")
        seen.add(it.recording_id)
    if not selected:
        raise RunError(f"subset has no items for corpus {corpus!r}")
    return selected


def select_only(items: list[SubsetItem], only: list[str] | None) -> list[SubsetItem]:
    """Restrict to an explicit ``--only ID,ID`` list; unknown/duplicate ids error.

    A repeated id (``--only A,A``) is rejected rather than deduplicated: it would
    otherwise expand to two SUBMIT decisions for one recording and stage/submit
    it twice, which the whole idempotency design exists to prevent.
    """
    if not only:
        return items
    dupes = sorted({rec for rec in only if only.count(rec) > 1})
    if dupes:
        raise RunError(f"--only names duplicate ids: {dupes}")
    by_id = {it.recording_id: it for it in items}
    unknown = [rec for rec in only if rec not in by_id]
    if unknown:
        raise RunError(f"--only names ids not in the subset for this corpus: {sorted(unknown)}")
    return [by_id[rec] for rec in only]


# --------------------------------------------------------------------------- #
# 3. Path resolution (exactly one existing, unambiguous path per role)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ResolvedItem:
    """One recording's resolved input paths. ``uem``/``wer_reference`` are None
    when the corpus has none (VoxConverse has no UEM and no reference transcript)."""

    recording_id: str
    corpus: str
    split: str
    audio: Path
    reference_rttm: Path
    uem: Path | None
    wer_reference: Path | None


def _require_within(root: Path, candidate: Path, what: str) -> Path:
    """Resolve ``candidate`` and require it stays under ``root`` (traversal guard)."""
    root_resolved = root.resolve()
    resolved = candidate.resolve()
    if root_resolved != resolved and root_resolved not in resolved.parents:
        raise RunError(f"{what} resolves outside the corpus root: {resolved}")
    return candidate


def _require_file(path: Path, what: str) -> Path:
    if not path.is_file():
        raise RunError(f"missing {what}: {path}")
    return path


def _unambiguous_audio(candidates: list[Path], recording_id: str) -> Path:
    """Exactly one existing audio file, else reject (never guess)."""
    present = [
        p for p in candidates if p.is_file() and _SYNOLOGY_META not in p.parts
    ]
    if not present:
        raise RunError(f"no audio found for {recording_id!r} among {candidates}")
    if len(present) > 1:
        raise RunError(f"ambiguous audio for {recording_id!r}: {present}")
    return present[0]


def resolve_ami(root: Path, item: SubsetItem) -> ResolvedItem:
    audio = _require_within(
        root, root / "ami" / "audio" / f"{item.recording_id}.Mix-Headset.wav", "AMI audio"
    )
    ref = (
        root
        / "ami"
        / "AMI-diarization-setup-main"
        / "only_words"
        / "rttms"
        / item.split
        / f"{item.recording_id}.rttm"
    )
    uem = (
        root
        / "ami"
        / "AMI-diarization-setup-main"
        / "uems"
        / item.split
        / f"{item.recording_id}.uem"
    )
    wer_ref = root / "ami" / "wer_reference" / f"{item.recording_id}.words.txt"
    return ResolvedItem(
        recording_id=item.recording_id,
        corpus=item.corpus,
        split=item.split,
        audio=_require_file(_unambiguous_audio([audio], item.recording_id), "AMI audio"),
        reference_rttm=_require_file(
            _require_within(root, ref, "AMI reference RTTM"), "AMI reference RTTM"
        ),
        uem=_require_file(_require_within(root, uem, "AMI UEM"), "AMI UEM"),
        wer_reference=_require_file(
            _require_within(root, wer_ref, "AMI WER reference"), "AMI WER reference"
        ),
    )


def resolve_voxconverse(root: Path, item: SubsetItem) -> ResolvedItem:
    # The two splits nest audio differently; resolve BOTH candidate nestings and
    # reject if a recording id somehow exists under more than one (an ambiguous
    # match must fail, never silently pick a split).
    vc = root / "voxconverse"
    candidates = [
        vc / "audio_test" / "voxconverse_test_wav" / f"{item.recording_id}.wav",
        vc / "audio_dev" / "audio" / f"{item.recording_id}.wav",
    ]
    for c in candidates:
        _require_within(root, c, "VoxConverse audio")
    audio = _unambiguous_audio(candidates, item.recording_id)
    ref = vc / "voxconverse-master" / item.split / f"{item.recording_id}.rttm"
    return ResolvedItem(
        recording_id=item.recording_id,
        corpus=item.corpus,
        split=item.split,
        audio=audio,
        reference_rttm=_require_file(
            _require_within(root, ref, "VoxConverse reference RTTM"), "VoxConverse reference RTTM"
        ),
        uem=None,
        wer_reference=None,
    )


def resolve_item(root: Path, item: SubsetItem) -> ResolvedItem:
    if item.corpus == "ami":
        return resolve_ami(root, item)
    return resolve_voxconverse(root, item)


# --------------------------------------------------------------------------- #
# 4. WAV preflight (stream-verify length; do NOT trust the header nframes)
# --------------------------------------------------------------------------- #
def measure_wav_seconds(path: Path) -> float:
    """Decoded duration of a PCM WAV, verified by streaming to EOF.

    The header's ``nframes`` is a claim, not a guarantee: a truncated file can
    declare a full length its ``data`` chunk does not contain. So the frames are
    actually read and counted; if the readable count is short of the header the
    file is truncated and the error says so (catching truncation without pulling
    ffmpeg into the harness). Both corpora are PCM WAV (verified on disk).
    """
    try:
        with wave.open(str(path), "rb") as wav:
            rate = wav.getframerate()
            declared = wav.getnframes()
            if rate <= 0:
                raise RunError(f"{path}: non-positive sample rate {rate}")
            read = 0
            chunk = 1 << 20
            while True:
                frames = wav.readframes(chunk)
                if not frames:
                    break
                read += len(frames) // (wav.getsampwidth() * wav.getnchannels())
    except wave.Error as exc:
        raise RunError(f"{path}: not a readable PCM WAV: {exc}") from exc
    except OSError as exc:
        raise RunError(f"{path}: {exc.strerror or exc}") from exc
    if read < declared:
        raise RunError(f"{path}: truncated WAV, header claims {declared} frames, read {read}")
    return read / rate


def rttm_max_end_seconds(text: str) -> float:
    """Latest end time (start+duration) over an RTTM's ``SPEAKER`` rows.

    A pure text scan (no pyannote), so the ``run`` lane can preflight that a
    reference does not extend past the decoded audio without pulling the scoring
    stack in. Malformed rows raise rather than being silently skipped.
    """
    latest = 0.0
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith(";;"):
            continue
        parts = line.split()
        if parts[0] != "SPEAKER":
            continue
        if len(parts) < 9:
            raise RunError(f"RTTM line {lineno}: expected >=9 fields, got {len(parts)}")
        try:
            start = float(parts[3])
            duration = float(parts[4])
        except ValueError as exc:
            raise RunError(f"RTTM line {lineno}: bad start/duration: {exc}") from exc
        latest = max(latest, start + duration)
    return latest


def uem_max_end_seconds(text: str, recording_id: str) -> float | None:
    """Latest UEM region end for ``recording_id`` (None if the UEM has no rows).

    Pure text scan matching ``eval_quality.parse_uem``'s column layout, for the
    same preflight; ``None`` lets the caller pass it straight to
    :func:`check_duration` for a corpus/recording with no UEM.
    """
    latest: float | None = None
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith(";;"):
            continue
        parts = line.split()
        if len(parts) < 4:
            raise RunError(f"UEM line {lineno}: expected 4 fields, got {parts!r}")
        if parts[0] != recording_id:
            continue
        try:
            end = float(parts[3])
        except ValueError as exc:
            raise RunError(f"UEM line {lineno}: bad end: {exc}") from exc
        latest = end if latest is None else max(latest, end)
    return latest


def check_duration(
    measured_s: float,
    extent_s: float,
    reference_max_end_s: float,
    uem_max_end_s: float | None,
    tol_s: float,
) -> list[str]:
    """Return human-readable preflight problems (empty list == OK).

    The bounds are MAX-END times (the last moment any interval ends), not summed
    durations: a summed UEM length cannot detect a region that runs past the end
    of the audio, whereas ``uem_max_end_s > measured_s`` does. Each check is a
    warning string so the caller can log them all rather than stopping at the
    first.

    ``extent_s`` is the subset's reference ANNOTATION extent (the last annotated
    speech end from ``rttm_stats.json``), NOT the audio duration. VoxConverse
    clips are trimmed to roughly their annotated region so the two coincide, but
    AMI Mix-Headset audio legitimately runs past the last annotated word. So the
    length guard is one-directional: only a decoded file SHORTER than its own
    extent (a truncated or wrong download) is flagged; audio longer than the
    extent is expected and fine. A truncated file is also caught by the
    reference/UEM out-of-bounds checks below.
    """
    problems: list[str] = []
    if extent_s - measured_s > tol_s:
        problems.append(
            f"decoded duration {measured_s:.3f}s is shorter than subset extent "
            f"{extent_s:.3f}s by more than {tol_s:.3f}s (truncated or wrong file?)"
        )
    if reference_max_end_s > measured_s + tol_s:
        problems.append(
            f"reference extends to {reference_max_end_s:.3f}s past decoded audio {measured_s:.3f}s"
        )
    if uem_max_end_s is not None and uem_max_end_s > measured_s + tol_s:
        problems.append(
            f"UEM extends to {uem_max_end_s:.3f}s past decoded audio {measured_s:.3f}s"
        )
    return problems


# --------------------------------------------------------------------------- #
# 5. pipeline_environment identity (strict, versioned; no timestamp)
# --------------------------------------------------------------------------- #
# The identity of what produced a hypothesis, so two snapshots taken before and
# after a batch can be compared for EQUALITY (a mid-batch model/driver change
# invalidates the run). Timestamps live in the journal, never here, so equality
# is over identity only. Every field is required, typed, AND type-checked; an
# ellipsis-style open dict, a string where a bool belongs, or a dict where a
# scalar digest belongs are all rejected (codex: an under-specified OR
# mistyped env silently weakens the cohort identity a hash is supposed to bind).
_STR = "str"
_BOOL = "bool"
_INT_GE1 = "int_ge1"

_PIPELINE_ENV_SPEC: dict[str, dict[str, str]] = {
    "code": {"git_sha": _STR, "image_digest": _STR},
    "model_weights": {
        "whisper_ct2_dir_sha256": _STR,
        "pyannote_pipeline_sha256": _STR,
        "titanet_sha256": _STR,
    },
    "gpu": {"name": _STR, "driver": _STR, "cuda": _STR},
    "runtime": {"ctranslate2": _STR, "torch": _STR, "pyannote_audio": _STR},
    "decode": {"beam_size": _INT_GE1, "batch_size": _INT_GE1, "word_timestamps": _BOOL},
    "flags": {"tf32": _BOOL, "deterministic": _BOOL},
}
# Retained (derived) so callers/tests that referenced the group->keys shape keep
# working; the spec above is the single source of truth.
_PIPELINE_ENV_FIELDS: dict[str, tuple[str, ...]] = {
    group: tuple(spec) for group, spec in _PIPELINE_ENV_SPEC.items()
}


def _check_env_scalar(where: str, kind: str, value: Any) -> None:
    """Type/range-check one pipeline_environment scalar (raises on mismatch).

    ``bool`` is a subclass of ``int`` in Python, so an int field explicitly
    rejects a bool (``True`` is not a beam size) and a bool field explicitly
    requires ``bool`` (a ``0``/``1`` int is not accepted as a flag).
    """
    if value is None:
        raise RunError(f"{where} must not be null")
    if kind == _STR:
        if not isinstance(value, str) or not value.strip():
            raise RunError(f"{where} must be a non-empty string, got {value!r}")
    elif kind == _BOOL:
        if not isinstance(value, bool):
            raise RunError(f"{where} must be a bool, got {value!r}")
    elif kind == _INT_GE1:
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise RunError(f"{where} must be an int >= 1, got {value!r}")
    else:  # pragma: no cover - guards a spec typo
        raise RunError(f"{where}: unknown field kind {kind!r}")


def validate_pipeline_environment(env: dict[str, Any]) -> dict[str, Any]:
    """Validate the strict pipeline-environment identity dict (raises on any gap).

    Returns the same dict on success so it can be used inline. Requires the exact
    top-level groups and their required keys, each of the DECLARED type; a
    missing group/key, a null value, an unexpected key, or a wrongly-typed value
    is an error (an identity with holes OR wrong types is worse than none).
    """
    if not isinstance(env, dict):
        raise RunError("pipeline_environment must be an object")
    if env.get("schema_version") != PIPELINE_ENVIRONMENT_SCHEMA_VERSION:
        raise RunError(
            f"pipeline_environment schema_version must be {PIPELINE_ENVIRONMENT_SCHEMA_VERSION}"
        )
    allowed_top = {"schema_version", *_PIPELINE_ENV_SPEC}
    extra_top = set(env) - allowed_top
    if extra_top:
        raise RunError(f"pipeline_environment has unexpected keys: {sorted(extra_top)}")
    for group, spec in _PIPELINE_ENV_SPEC.items():
        block = env.get(group)
        if not isinstance(block, dict):
            raise RunError(f"pipeline_environment.{group} must be an object")
        missing = [k for k in spec if k not in block]
        if missing:
            raise RunError(f"pipeline_environment.{group} missing keys: {missing}")
        extra = set(block) - set(spec)
        if extra:
            raise RunError(f"pipeline_environment.{group} has unexpected keys: {sorted(extra)}")
        for key, kind in spec.items():
            _check_env_scalar(f"pipeline_environment.{group}.{key}", kind, block[key])
    return env


def pipeline_environment_hash(env: dict[str, Any]) -> str:
    """Stable identity hash of a validated pipeline_environment."""
    return _sha256_hex(_canonical_bytes(validate_pipeline_environment(env)))


# --------------------------------------------------------------------------- #
# 6. Cohort descriptor + hash (domain-separated array of per-item records)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CohortInput:
    """One recording's scoring inputs by role, each bound to bytes.

    ``role`` names the input (``audio``/``reference_rttm``/``uem``/
    ``wer_reference``) so the hash cannot confuse two files of equal bytes, and a
    role a corpus lacks is an EXPLICIT null record (not an omission), so a
    VoxConverse cohort (no UEM/WER) is distinguishable from an AMI one whose
    files merely failed to load.
    """

    recording_id: str
    role: str
    byte_len: int | None
    sha256: str | None


def cohort_descriptor(
    corpus: str,
    split_by_id: dict[str, str],
    inputs: list[CohortInput],
    pipeline_env_hash: str,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Build the canonical cohort descriptor (the object the hash is taken over).

    A cohort is one repeatable experiment: same corpus, same recordings, same
    input BYTES (audio included, because audio drives the hypothesis), same
    pipeline identity, same scoring protocol. Two ``run`` passes that share this
    descriptor are a legitimate zero-change pair; anything else is not, and
    ``report`` must refuse to call them a noise floor.
    """
    records = sorted(
        (
            {
                "id": ci.recording_id,
                "split": split_by_id[ci.recording_id],
                "role": ci.role,
                "byte_len": ci.byte_len,
                "sha256": ci.sha256,
            }
            for ci in inputs
        ),
        key=lambda r: (r["id"], r["role"]),
    )
    return {
        "schema_version": COHORT_SCHEMA_VERSION,
        "corpus": corpus,
        "recording_ids": sorted(split_by_id),
        "inputs": records,
        "pipeline_environment_hash": pipeline_env_hash,
        "protocol": protocol,
    }


def cohort_sha256(descriptor: dict[str, Any]) -> str:
    """Stable hash of a cohort descriptor (sorted, compact, finite-only)."""
    if descriptor.get("schema_version") != COHORT_SCHEMA_VERSION:
        raise RunError(f"cohort descriptor schema_version must be {COHORT_SCHEMA_VERSION}")
    return _sha256_hex(_canonical_bytes(descriptor))


# --------------------------------------------------------------------------- #
# 7. Artifact journal + pure resume decision (write-ahead, per-attempt records)
# --------------------------------------------------------------------------- #
# Actions the resume planner returns for each selected recording.
ACTION_SUBMIT = "submit"          # never submitted (or an unknown outcome to reconcile)
ACTION_POLL = "poll"              # submitted/running; re-poll the recorded run
ACTION_SKIP_DONE = "skip_done"    # completed with verified artifacts; nothing to do
ACTION_RETRY = "retry"            # a failure the caller asked to retry
ACTION_STOP = "stop"              # a failure/unknown outcome that must halt


@dataclass(frozen=True)
class ResumeDecision:
    recording_id: str
    action: str
    reason: str


def _fsync_dir(directory: Path) -> None:
    """fsync a directory so a rename within it is durable (best-effort).

    Directory fsync is not portable to every filesystem; a platform that cannot
    open a directory for fsync (some networked/foreign mounts) is tolerated
    rather than failing the write, since the temp-file fsync + atomic replace
    already give crash-consistency on the common case.
    """
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def write_json_atomic(path: Path, payload: Any) -> None:
    """Durably, atomically write ``payload`` as canonical JSON to ``path``.

    Writes a temp file in the SAME directory (so ``os.replace`` is a same-device
    atomic rename), flushes + ``fsync``s the temp file's data to disk BEFORE the
    rename, replaces, then ``fsync``s the parent directory so the rename itself
    survives a crash. Without the fsyncs ``os.replace`` is atomic but not durable
    (the rename can be lost on power loss even though the bytes look written).
    """
    path = Path(path)
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    data = _canonical_bytes(payload)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(directory))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
    _fsync_dir(directory)


@contextlib.contextmanager
def out_dir_lock(out_dir: Path) -> Iterator[Path]:
    """Hold an EXCLUSIVE, non-blocking lock on an out-dir for a run.

    Two drivers sharing one ``--out-dir`` would interleave journal writes and
    corrupt the write-ahead invariant, so a run takes an ``flock`` on a
    ``.eval_run.lock`` file in the out-dir and refuses to start (rather than
    block indefinitely) if another process already holds it. The lock file is
    left in place (its presence is harmless; the lock is advisory and released
    with the fd), and the fd is always closed on exit.
    """
    import fcntl

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_path = out_dir / ".eval_run.lock"
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RunError(
                f"another eval run holds the out-dir lock {lock_path} "
                f"(is a second driver using the same --out-dir?): {exc}"
            ) from exc
        try:
            yield lock_path
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def new_journal(corpus: str, cohort_hash: str, pipeline_env: dict[str, Any]) -> dict[str, Any]:
    """An empty journal bound to one corpus + cohort + pipeline identity."""
    return {
        "schema_version": JOURNAL_SCHEMA_VERSION,
        "kind": "eval_run_journal",
        "corpus": corpus,
        "cohort_sha256": cohort_hash,
        "pipeline_environment": validate_pipeline_environment(pipeline_env),
        "items": {},
    }


def validate_journal(journal: dict[str, Any], *, corpus: str, cohort_hash: str) -> None:
    """Reject a journal that does not belong to this run (fail closed).

    A resume must load a journal for the SAME corpus and cohort; a mismatch means
    the inputs, audio, or pipeline changed, so continuing would silently mix two
    experiments. This is the guard that makes ``--resume`` safe.

    """
    if journal.get("schema_version") != JOURNAL_SCHEMA_VERSION:
        raise RunError("journal schema_version mismatch")
    if journal.get("kind") != "eval_run_journal":
        raise RunError("not an eval_run_journal")
    if journal.get("corpus") != corpus:
        raise RunError(f"journal corpus {journal.get('corpus')!r} != requested {corpus!r}")
    if journal.get("cohort_sha256") != cohort_hash:
        raise RunError(
            "journal cohort_sha256 mismatch: inputs/audio/pipeline changed since last run"
        )


def plan_resume(
    journal: dict[str, Any],
    selected_ids: list[str],
    *,
    resume: bool,
    retry_failed: bool,
) -> list[ResumeDecision]:
    """Decide, per selected recording, what the driver should do — purely.

    Never infers state from a file on disk: the journal item's recorded status is
    the only truth. A recording with no item, or in a resumable state without the
    ``--resume`` flag, is a fresh submit; a completed item is skipped; a failure
    is retried only under ``--retry-failed``, else it stops the batch. The
    write-ahead ``submitting`` / ``submission_unknown`` states resolve to a
    submit that the driver must make idempotent by its recorded submission key.
    """
    corpus = journal.get("corpus")
    required = REQUIRED_ARTIFACTS.get(corpus)
    if required is None:
        raise RunError(f"journal corpus {corpus!r} has no required-artifact contract")
    items = journal.get("items", {})
    decisions: list[ResumeDecision] = []
    for rec in selected_ids:

        def add(action: str, reason: str, _rec: str = rec) -> None:
            decisions.append(ResumeDecision(_rec, action, reason))

        item = items.get(rec)
        if item is None:
            add(ACTION_SUBMIT, "not yet submitted")
            continue
        status = item.get("status")
        if status == COMPLETED:
            artifacts = item.get("artifacts") or {}
            missing = [k for k in required if not artifacts.get(k)]
            if not missing:
                add(ACTION_SKIP_DONE, "completed with all required artifacts")
            else:
                add(ACTION_STOP, f"completed but missing artifacts: {missing}")
            continue
        if status == UNKNOWN_SUBMISSION:
            add(ACTION_SUBMIT, "submission outcome unknown; reconcile by submission key")
            continue
        if status in FAILURE_STATES:
            if retry_failed:
                add(ACTION_RETRY, f"{status}; --retry-failed set")
            else:
                add(ACTION_STOP, f"{status}; --retry-failed not set")
            continue
        if status in RESUMABLE_STATES:
            if resume and item.get("run_uuid"):
                add(ACTION_POLL, f"{status}; poll recorded run")
            elif resume:
                add(ACTION_SUBMIT, f"{status} without a run_uuid; re-submit")
            else:
                add(ACTION_STOP, f"{status} in journal but --resume not set")
            continue
        add(ACTION_STOP, f"unrecognized status {status!r}")
    return decisions
