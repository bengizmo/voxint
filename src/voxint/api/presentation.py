"""Pure display helpers for the operator-facing console (issue #56).

The console's data comes from :mod:`voxint.api.runs_query`,
:mod:`voxint.api.stats_query`, and :mod:`voxint.adjudication.resolver`; this
module holds the small, dependency-free formatting functions that turn that data
into text a non-technical operator can read. Every function is a pure transform
over primitives — no DB, no HTTP, no clock of its own — so the whole module
unit-tests without a database (like ``tests/unit/test_runs_cursor.py``).

.. versionchanged:: 0.29
   Added :func:`humanize_error` (#244).

Two rules the callers rely on:

- ``now`` is always injected (never read from the wall clock here) so relative
  times are deterministic under test, mirroring
  :func:`voxint.api.stats_query.parse_since`.
- The ``humanize_*`` helpers produce **display text only**. Their input is a raw
  enum ``value`` that also keys a CSS class and the machine-facing
  Prometheus/JSON/CLI outputs; templates must keep the raw value in ``class=``
  and only swap the visible label, and the machine renderers must not call these
  at all.
"""

import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import unquote

# Collapses any run of whitespace or non-printing character a display string can
# carry into a single space, so a name stays a single clean, non-spoofable line.
# Covers C0/DEL/C1 controls, zero-width and directionality marks, the bidi
# override/isolate ranges, and the BOM — the title of a fetched URL is the most
# externally-controlled string in the console, and a bidi override there would
# let a hostile title reorder how a filename reads.
_CONTROL_RUN = re.compile(
    "[\\s"  # any Unicode whitespace (newline, tab, NBSP, separators)
    "\\x00-\\x1f\\x7f-\\x9f"  # C0 controls, DEL, C1 controls
    "\\u200b-\\u200f"  # zero-width space/NJ/J + LRM/RLM
    "\\u202a-\\u202e"  # bidi embeddings + overrides (display-spoofing vector)
    "\\u2066-\\u2069"  # bidi isolates
    "\\ufeff"  # BOM / zero-width no-break space
    "]+"
)


def _clean_basename(source_path: str) -> str:
    """The last path segment of ``source_path``, percent-decoded and de-noised.

    Splits on both separators (a Windows-style path can reach a POSIX host),
    falls back to the whole value when there is no trailing segment (a path
    ending in a slash), and never returns an empty string.
    """
    segment = re.split(r"[\\/]", source_path.rstrip("\\/"))[-1]
    decoded = unquote(segment) if segment else source_path
    cleaned = _CONTROL_RUN.sub(" ", decoded).strip()
    return cleaned or source_path.strip() or source_path


def title_from_snapshot(snapshot: object) -> str | None:
    """The ``title`` of a run's frozen sidecar snapshot, read tolerantly.

    ``pipeline_runs.sidecar`` (issue #104) stores the whole sidecar mapping;
    the title was validated at submit, but this reader stays tamper-tolerant
    (an out-of-band-edited snapshot yields ``None``, never a crash): only a
    mapping with a non-blank string ``title`` produces a value. Display-only —
    enrichment keeps reading the scraped source-metadata title.
    """
    if not isinstance(snapshot, dict):
        return None
    title = snapshot.get("title")
    if isinstance(title, str):
        # Same cleaning as friendly_media_label: a tampered snapshot must not
        # be the one console title path that skips the bidi/zero-width strip.
        cleaned = _CONTROL_RUN.sub(" ", title).strip()
        if cleaned:
            return cleaned
    return None


def friendly_media_label(title: str | None, source_path: str) -> str:
    """A human name for a recording: the source title, else a cleaned filename.

    Prefers the acquisition-metadata ``title`` (issue #36) when it carries actual
    text — ``.strip()`` guards a whitespace-only title that would otherwise render
    blank (a latent bug in the pre-#56 ``runs.html`` ``{% if it.title %}``
    fallback). Otherwise it derives a readable name from ``source_path``'s
    basename (percent-decoded, extension kept — a non-technical operator
    recognizes ``.mp3``). For a pre-#36 URL run whose ``source_path`` is a bare
    uuid with no title, the honest result is that uuid basename; the caller keeps
    the raw path visible as ground truth rather than this inventing an origin.
    Never truncates — the template does that in CSS so copy/paste stays intact.
    """
    if title is not None:
        # Clean the title the same way as a basename: it is the most externally
        # controlled string in the console (a fetched URL's title), so strip the
        # bidi/zero-width family before it reaches the DOM, not just whitespace.
        cleaned = _CONTROL_RUN.sub(" ", title).strip()
        if cleaned:
            return cleaned
    return _clean_basename(source_path)


def format_duration(seconds: float | None) -> str:
    """A recording length as ``H:MM:SS`` / ``M:SS``; ``"—"`` when unknown.

    Unknown (``None``) is the honest render for media that was never probed;
    a negative or non-finite value (the DB check-constraint forbids negatives,
    but Postgres accepts ``NaN`` in a ``float`` column, and ``NaN < 0`` is
    ``False``, so guard it explicitly) is treated as unknown rather than shown
    as a nonsense clock or crashing ``int()`` on ``NaN``/``inf``.
    """
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "—"
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def format_size(size_bytes: int | None) -> str:
    """A file size as a coarse ``B`` / ``KB`` / ``MB`` / ``GB``; ``"—"`` unknown.

    Binary units (1024) with the conventional ``KB``/``MB`` labels the way file
    managers show them; one decimal place above bytes, so a listing reads
    "412 MB", not "412.37 MB". Unknown (``None``, media that was never probed)
    and a negative size (the DB check-constraint forbids it, but guard anyway
    rather than render a nonsense value) both collapse to the em dash.
    """
    if size_bytes is None or size_bytes < 0:
        return "—"
    if size_bytes < 1024:
        return f"{size_bytes} B"
    value = float(size_bytes)
    for unit in ("KB", "MB", "GB"):
        value /= 1024
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
    return f"{value:.1f} GB"  # pragma: no cover - loop always returns at GB


def format_age(created_at: datetime, *, now: datetime) -> str:
    """A coarse, human relative age ("3 hours ago") for a timestamp.

    Deliberately coarse — a non-technical operator wants "how stale is this",
    not a precise interval — and single-unit (never "1 hour 3 minutes"). ``now``
    is injected so the output is deterministic under test. A ``created_at`` in
    the (clock-skewed) future collapses to "just now" rather than a negative age.
    Absolute wall-clock time stays available to the caller via a ``title=``
    tooltip; this is the at-a-glance label. Both operands are expected tz-aware
    (the DB columns are ``TIMESTAMPTZ`` and the route injects ``now`` as UTC); a
    naive value is normalized to UTC rather than raising a ``TypeError`` that
    would 500 the whole listing.
    """
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    delta = now - created_at
    seconds = delta.total_seconds()
    if seconds < 60:
        return "just now"
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    days = hours // 24
    if days < 7:
        return f"{days} day{'s' if days != 1 else ''} ago"
    weeks = days // 7
    if days < 30:
        return f"{weeks} week{'s' if weeks != 1 else ''} ago"
    months = days // 30
    if days < 365:
        return f"{months} month{'s' if months != 1 else ''} ago"
    years = days // 365
    return f"{years} year{'s' if years != 1 else ''} ago"


def format_clock_time(at: datetime, *, now: datetime) -> str:
    """A mono-friendly timestamp: ``just now`` within 60 s, else ``HH:MM``.

    Today's events show the wall-clock time; older events fall back to
    ``format_age``. Both operands should be tz-aware UTC.
    """
    if at.tzinfo is None:
        at = at.replace(tzinfo=UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    delta = now - at
    if delta.total_seconds() < 60:
        return "just now"
    if at.date() == now.date():
        return at.strftime("%H:%M")
    return format_age(at, now=now)


def format_compact_duration(seconds: float | None) -> str:
    """A run wall-clock as ``XmYYs`` or ``XhYYm``; ``...`` when unknown."""
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "…"
    total = int(seconds)
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{secs:02d}s"


def _humanize_enum(value: str) -> str:
    """``diarize_embed`` → ``Diarize embed``: split on ``_``, sentence-case.

    Only the first word is capitalized (sentence case, not Title Case) so a
    multi-word label reads as prose, not a header.
    """
    words = value.replace("_", " ").split()
    if not words:
        return value
    return " ".join([words[0].capitalize(), *words[1:]])


# Stage identifiers whose plain de-underscoring does not read well. Anything not
# listed falls through to the generic sentence-case above, so a new Stage enum
# value renders acceptably without a code change here.
_STAGE_LABELS = {
    "diarize_embed": "Diarize & embed",
    "enhance_match": "Enhance & match",
}


def humanize_stage(value: str) -> str:
    """A pipeline stage identifier as an operator-readable label. Display only.

    ``value`` is the raw ``Stage`` enum string, which also keys a CSS class and
    the Prometheus/JSON/CLI outputs — callers keep the raw value there and use
    this for the visible text alone.
    """
    return _STAGE_LABELS.get(value, _humanize_enum(value))


def humanize_status(value: str) -> str:
    """A run-status identifier as an operator-readable label. Display only.

    Same display-only contract as :func:`humanize_stage`: the raw ``RunStatus``
    value stays in ``class=`` and the machine outputs; this is the label text.
    """
    return _humanize_enum(value)


@dataclass(frozen=True)
class NormalizedError:
    """A pipeline error rendered for a non-technical operator.

    ``label`` is the plain-language one-liner, ``hint`` is an optional recovery
    suggestion, and ``raw`` is the original string (for a details fold).
    """

    label: str
    hint: str | None
    raw: str


_ERROR_PATTERNS: list[tuple[re.Pattern[str], str, str | None]] = [
    (re.compile(r"^interrupted: lease expired", re.IGNORECASE),
     "worker timed out",
     "Try again. If it keeps timing out, check that the model service is running."),
    (re.compile(r"^interrupted: worker died", re.IGNORECASE),
     "worker restarted mid-stage", "The run can be retried safely."),
    (re.compile(r"downloaded file is empty", re.IGNORECASE),
     "downloaded file was empty", "Check that the source file exists and is not zero-length."),
    (re.compile(r"AcquisitionError|URL acquisition", re.IGNORECASE),
     "download failed", "Check that the URL is reachable and try again."),
    (re.compile(r"FileNotFoundError|No such file", re.IGNORECASE),
     "file not found", "Check that the source file still exists at the expected path."),
    (re.compile(r"ConnectionError|connect.*refused", re.IGNORECASE),
     "service unreachable", "A model service is down. Check that all containers are running."),
    (re.compile(
        r"(?:empty|zero[- ]duration).*audio|audio.*(?:empty|zero[- ]duration)",
        re.IGNORECASE,
    ), "audio track was empty", "The file has no usable audio. Try a different recording."),
    (re.compile(r"paused before commit", re.IGNORECASE),
     "paused", None),
    (re.compile(r"cancelled before commit", re.IGNORECASE),
     "cancelled", None),
    (re.compile(r"StageDeferError|active operation", re.IGNORECASE),
     "waiting on another operation", "Retry after the other operation finishes."),
    (re.compile(r"CUDA out of memory|OutOfMemoryError", re.IGNORECASE),
     "GPU ran out of memory", "Try a shorter recording or restart the model service."),
    (re.compile(r"torch\.cuda.*error|CUBLAS_STATUS|CUDNN", re.IGNORECASE),
     "GPU error", "Restart the model service and retry."),
]


def normalize_error(error: str | None) -> NormalizedError | None:
    """Raw pipeline error text as a structured plain-language result.

    Returns ``None`` for blank/missing input. The ``label`` field is the same
    string ``humanize_error`` would return; ``hint`` adds a recovery suggestion
    for known patterns; ``raw`` preserves the original for a details fold.

    .. versionadded:: 0.32
    """
    if error is None:
        return None
    for pattern, label, hint in _ERROR_PATTERNS:
        if pattern.search(error):
            return NormalizedError(label=label, hint=hint, raw=error)
    lines = error.strip().splitlines()
    if not lines:
        return None
    cleaned = lines[-1].strip()
    if not cleaned:
        return None
    if len(cleaned) > 80:
        cleaned = cleaned[:77] + "…"
    return NormalizedError(label=cleaned, hint=None, raw=error)


def humanize_error(error: str | None) -> str | None:
    """Raw pipeline error text as a plain-language label. Display only.

    Thin wrapper around :func:`normalize_error` that returns just the label
    string, for templates that only need the one-liner.
    """
    result = normalize_error(error)
    return result.label if result is not None else None


def confidence_band(score: float | None) -> str:
    """A 0-1 score as a plain-language band: likely / possible / low / unknown."""
    if score is None or not math.isfinite(score):
        return "unknown"
    if score >= 0.8:
        return "likely"
    if score >= 0.5:
        return "possible"
    return "low"


def folder_label(source_path: str) -> str | None:
    """The parent directory of a source path as a short folder name.

    Returns ``None`` for top-level files (no directory component). A
    non-technical operator sees "interviews" rather than
    ``/data/media/interviews/episode-42.mp3``.
    """
    last_sep = max(source_path.rfind("/"), source_path.rfind("\\"))
    if last_sep <= 0:
        return None
    parent = source_path[:last_sep]
    basename_sep = max(parent.rfind("/"), parent.rfind("\\"))
    name = parent[basename_sep + 1:] if basename_sep >= 0 else parent
    return name or None
