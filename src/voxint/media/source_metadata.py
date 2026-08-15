"""Sanitize a yt-dlp info-JSON into the bounded snapshot Voxint persists.

Pure, DB-free, network-free (a sibling of :mod:`voxint.media.redaction`):
the ACQUIRE stage hands this module the raw ``source.info.json`` yt-dlp wrote
next to the download, and gets back a frozen :class:`SourceMetadata` holding
the normalized display fields plus a bounded, allowlisted ``raw`` subset —
never the original document.

Safety model is **structural allowlisting, not scrubbing**: an info-JSON's
secret-bearing keys (``formats``/``requested_formats``/``fragments`` and their
signed transport URLs, ``http_headers`` with cookies, ``cookies``,
``manifest_url``, ``subtitles``/``automatic_captions``/``thumbnails`` URL
lists…) are simply never copied, so nothing signed or credential-bearing can
persist at rest. :func:`voxint.media.redaction.redact` stays the *error
message* boundary and is deliberately NOT applied to retained content fields —
structurally rewriting a description's legitimate URLs would corrupt operator
context; the caller instead passes its known secret literals (proxy, cookies
path) as ``extra_secrets`` and those are removed verbatim from every retained
string.

Extraction is **total and fail-closed** like the redactor: the info-JSON is
untrusted remote content, so junk types are dropped (never raised), every
string/list is capped, and the only refusals are an oversized or non-object
document (:class:`SourceMetadataError`, which the ACQUIRE stage treats as
"no metadata" — capture is best-effort and never fails an acquisition).

``SNAPSHOT_SCHEMA_VERSION`` stamps every persisted snapshot (DB row and
replay sidecar) and is bumped whenever the allowlist or bounds change, so a
future consumer can tell which extraction contract produced a row.
"""

import json
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime

# Bump when the allowlist, bounds, or normalization below change shape.
SNAPSHOT_SCHEMA_VERSION = 1

# Version of the replay-sidecar envelope (the file ACQUIRE publishes beside the
# media so a crash between publish and DB commit can be repaired without a
# re-download). Independent of SNAPSHOT_SCHEMA_VERSION, which versions the
# snapshot *content*; the filename carries this version so a future v2 reader
# never has to sniff.
SIDECAR_SCHEMA_VERSION = 1

# Refuse to parse a document larger than this — a hostile or pathological
# info-JSON (huge formats array, embedded subtitle tracks) must not be able to
# balloon worker memory. Internal invariant, not a configurable.
MAX_INFO_JSON_BYTES = 8 * 1024 * 1024

# Per-field bounds. Content caps are hygiene (keep rows legible and small),
# not overflow protection — columns are unbounded TEXT.
MAX_DESCRIPTION_CHARS = 16_000
MAX_TEXT_CHARS = 1_000
MAX_TAGS = 64
MAX_TAG_CHARS = 100

# Keys copied into the ``raw`` subset (beyond the normalized columns). Scalars
# and flat string-lists only — anything absent from this tuple never persists,
# which is the entire redaction story for the snapshot. Notably excluded:
# formats / requested_formats / requested_downloads / fragments / url /
# manifest_url / http_headers / cookies / thumbnails / subtitles /
# automatic_captions / comments / entries / chapters / heatmap — all can carry
# signed URLs, cookies, or unbounded payloads.
RAW_ALLOWLIST: tuple[str, ...] = (
    "id",
    "display_id",
    "title",
    "uploader",
    "uploader_id",
    "channel",
    "channel_id",
    "upload_date",
    "timestamp",
    "release_timestamp",
    "duration",
    "view_count",
    "like_count",
    "live_status",
    "availability",
    "age_limit",
    "language",
    "license",
    "categories",
    "tags",
    "series",
    "season",
    "episode",
    "extractor",
    "extractor_key",
    "webpage_url",
)

# RAW_ALLOWLIST keys whose value is a URL: cleaned with the structural URL
# policy (_clean_http_url) instead of the prose cleaner, so `raw` can never
# hold a credential-bearing or scheme-smuggled URL the normalized columns
# would refuse.
_RAW_URL_KEYS = frozenset({"webpage_url"})


class SourceMetadataError(Exception):
    """The info-JSON could not be used at all (oversized, unparseable, or not
    a JSON object). The ACQUIRE stage treats this as "no metadata captured" —
    metadata is context, and its absence never fails an acquisition."""


@dataclass(frozen=True)
class SourceMetadata:
    """The sanitized, bounded snapshot ACQUIRE persists (see module docstring)."""

    title: str | None = None
    uploader: str | None = None
    uploader_url: str | None = None
    channel: str | None = None
    channel_url: str | None = None
    description: str | None = None
    upload_date: date | None = None
    duration_seconds: float | None = None
    tags: tuple[str, ...] = ()
    canonical_url: str | None = None
    extractor: str | None = None
    extractor_version: str | None = None
    raw: dict[str, object] = field(default_factory=dict)
    raw_schema_version: int = SNAPSHOT_SCHEMA_VERSION


def _clean_text(
    value: object, *, max_chars: int, extra_secrets: tuple[str, ...]
) -> str | None:
    """Coerce an untrusted scalar to bounded, control-character-free text.

    Non-string junk (dicts, lists, numbers where prose is expected) is dropped
    — extraction is total, never raising on shape surprises. Known secret
    literals are removed verbatim BEFORE truncation so a cut cannot leave an
    unrecognizable secret suffix; newlines/tabs survive (descriptions are
    multi-line) but other control characters do not.
    """
    if not isinstance(value, str):
        return None
    for secret in extra_secrets:
        if secret:
            value = value.replace(secret, "<redacted>")
    value = "".join(
        ch for ch in value if ch in "\n\t" or not unicodedata.category(ch).startswith("C")
    )
    value = value.strip()
    if not value:
        return None
    return value[:max_chars]


def _clean_http_url(value: object, extra_secrets: tuple[str, ...]) -> str | None:
    """Keep only an absolute http(s) URL with no embedded credentials.

    These fields (``webpage_url``, ``channel_url``, ``uploader_url``) are
    *retained for display*, so they must be structurally safe rather than
    redacted-to-a-stub: refuse userinfo (``user:pass@``), non-http schemes,
    and any surviving control/whitespace character (``_clean_text`` keeps
    ``\\n``/``\\t`` for prose, but a URL destined for an href must be one
    unbroken token) outright instead of keeping a mangled remnant.
    """
    text = _clean_text(value, max_chars=MAX_TEXT_CHARS, extra_secrets=extra_secrets)
    if text is None:
        return None
    if any(ch.isspace() or ord(ch) < 0x20 for ch in text):
        return None
    lowered = text.lower()
    if not (lowered.startswith("https://") or lowered.startswith("http://")):
        return None
    authority = text.split("://", 1)[1].split("/", 1)[0]
    if "@" in authority:
        return None
    return text


def _clean_upload_date(value: object) -> date | None:
    """Parse yt-dlp's ``YYYYMMDD`` upload_date string; None when unparseable."""
    if not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value.strip(), "%Y%m%d").date()
    except ValueError:
        return None


def _clean_duration(value: object) -> float | None:
    """Non-negative finite number, else None (bools are not durations)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        seconds = float(value)
    except OverflowError:  # a JSON integer too large for a float is junk
        return None
    if seconds != seconds or seconds in (float("inf"), float("-inf")) or seconds < 0:
        return None
    return seconds


def _clean_tags(value: object, extra_secrets: tuple[str, ...]) -> tuple[str, ...]:
    """Bounded flat list of bounded strings; junk elements dropped."""
    if not isinstance(value, list):
        return ()
    cleaned: list[str] = []
    for item in value:
        text = _clean_text(item, max_chars=MAX_TAG_CHARS, extra_secrets=extra_secrets)
        if text is not None:
            cleaned.append(text)
        if len(cleaned) >= MAX_TAGS:
            break
    return tuple(cleaned)


def extract(
    info_json_bytes: bytes, *, extra_secrets: tuple[str, ...] = ()
) -> SourceMetadata:
    """Build the sanitized snapshot from raw ``source.info.json`` bytes.

    Raises :class:`SourceMetadataError` only for a document that cannot be
    used at all (over :data:`MAX_INFO_JSON_BYTES`, malformed JSON, or a
    non-object top level). Every field-level surprise inside a valid object
    degrades to an absent field instead — the info-JSON is untrusted remote
    content and extraction must be total.

    ``extra_secrets`` are the caller's known secret literals (configured proxy
    string, cookies path) removed verbatim from every retained string — the
    same contract as :func:`voxint.media.redaction.redact`.
    """
    if len(info_json_bytes) > MAX_INFO_JSON_BYTES:
        raise SourceMetadataError(
            f"info-JSON is {len(info_json_bytes)} bytes,"
            f" over the {MAX_INFO_JSON_BYTES}-byte limit"
        )
    try:
        info = json.loads(info_json_bytes)
    except (ValueError, UnicodeDecodeError) as exc:
        raise SourceMetadataError(f"info-JSON is not valid JSON: {exc}") from None
    except RecursionError:
        # Pathologically nested but syntactically valid JSON: totality demands
        # this surfaces as the module's own error, never an ACQUIRE failure.
        raise SourceMetadataError("info-JSON is nested too deeply") from None
    if not isinstance(info, dict):
        raise SourceMetadataError("info-JSON top level is not an object")

    def clean(key: str, max_chars: int = MAX_TEXT_CHARS) -> str | None:
        return _clean_text(info.get(key), max_chars=max_chars, extra_secrets=extra_secrets)

    raw: dict[str, object] = {}
    for key in RAW_ALLOWLIST:
        if key not in info:
            continue
        value = info[key]
        if isinstance(value, (bool, int, float)):
            if isinstance(value, float) and (
                value != value or value in (float("inf"), float("-inf"))
            ):
                continue  # NaN/inf would not survive JSONB round-tripping honestly
            raw[key] = value
        elif isinstance(value, str):
            # URL-valued keys get the structural URL policy, not the prose
            # cleaner: a `https://user:token@host?sig=…` webpage_url passed
            # through _clean_text would persist credentials into `raw` that
            # the normalized canonical_url correctly refuses.
            cleaned = (
                _clean_http_url(value, extra_secrets)
                if key in _RAW_URL_KEYS
                else _clean_text(
                    value, max_chars=MAX_TEXT_CHARS, extra_secrets=extra_secrets
                )
            )
            if cleaned is not None:
                raw[key] = cleaned
        elif isinstance(value, list):
            tags = _clean_tags(value, extra_secrets)
            if tags:
                raw[key] = list(tags)
        # dicts / None / anything else: dropped wholesale — the allowlist names
        # only scalar/string-list keys, so deeper structure is a shape surprise
        # from untrusted content, not data to preserve.

    # _version is yt-dlp's build stamp: {"version": "2026.07.04", ...}. yt-dlp
    # has no separately versioned extractors, so the build version IS the
    # extractor version we can honestly record.
    version_obj = info.get("_version")
    extractor_version = (
        _clean_text(
            version_obj.get("version"),
            max_chars=MAX_TEXT_CHARS,
            extra_secrets=extra_secrets,
        )
        if isinstance(version_obj, dict)
        else None
    )

    return SourceMetadata(
        title=clean("title"),
        uploader=clean("uploader"),
        uploader_url=_clean_http_url(info.get("uploader_url"), extra_secrets),
        channel=clean("channel"),
        channel_url=_clean_http_url(info.get("channel_url"), extra_secrets),
        description=clean("description", MAX_DESCRIPTION_CHARS),
        upload_date=_clean_upload_date(info.get("upload_date")),
        duration_seconds=_clean_duration(info.get("duration")),
        tags=_clean_tags(info.get("tags"), extra_secrets),
        canonical_url=_clean_http_url(info.get("webpage_url"), extra_secrets),
        extractor=clean("extractor"),
        extractor_version=extractor_version,
        raw=raw,
    )


def sidecar_filename(media_sha256: str) -> str:
    """The hash-addressed replay-sidecar name for a published media file.

    Embedding the media's sha256 binds the sidecar to exactly the bytes it
    describes: overlapping ACQUIRE attempts that downloaded *different* bytes
    (upstream changed between them) publish different sidecar names, so a
    replay repair can never associate one attempt's metadata with another
    attempt's media — it only ever loads the sidecar matching the
    authoritative file's hash.
    """
    return f"source.{media_sha256}.metadata.v{SIDECAR_SCHEMA_VERSION}.json"


def to_sidecar_bytes(
    meta: SourceMetadata, *, media_sha256: str, acquired_at: datetime
) -> bytes:
    """Serialize an already-sanitized snapshot for the replay sidecar.

    Deterministic compact JSON (sorted keys) so identical inputs produce
    identical bytes. Only ever fed a :class:`SourceMetadata` this module built,
    so the content is sanitized by construction — the raw info-JSON never
    persists outside the attempt directory.
    """
    payload = {
        "sidecar_schema_version": SIDECAR_SCHEMA_VERSION,
        "media_sha256": media_sha256,
        "acquired_at": acquired_at.isoformat(),
        "snapshot": {
            "title": meta.title,
            "uploader": meta.uploader,
            "uploader_url": meta.uploader_url,
            "channel": meta.channel,
            "channel_url": meta.channel_url,
            "description": meta.description,
            "upload_date": meta.upload_date.isoformat() if meta.upload_date else None,
            "duration_seconds": meta.duration_seconds,
            "tags": list(meta.tags),
            "canonical_url": meta.canonical_url,
            "extractor": meta.extractor,
            "extractor_version": meta.extractor_version,
            "raw": meta.raw,
            "raw_schema_version": meta.raw_schema_version,
        },
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _expect_str_or_none(value: object, field_name: str) -> str | None:
    if value is None or isinstance(value, str):
        return value
    raise SourceMetadataError(f"sidecar field {field_name!r} has a non-string value")


def load_sidecar(
    data: bytes, *, expected_media_sha256: str
) -> "tuple[SourceMetadata, datetime]":
    """Parse a replay sidecar back into ``(snapshot, acquired_at)``.

    Strict, unlike :func:`extract`: the sidecar is Voxint's own sanitized
    artifact, so a shape surprise means corruption or version skew and raises
    :class:`SourceMetadataError` (the caller treats it as "no metadata", never
    a failed acquisition). A ``media_sha256`` mismatch is refused outright —
    a sidecar must never describe bytes other than the authoritative file.
    """
    try:
        payload = json.loads(data)
    except (ValueError, UnicodeDecodeError) as exc:
        raise SourceMetadataError(f"sidecar is not valid JSON: {exc}") from None
    except RecursionError:
        raise SourceMetadataError("sidecar is nested too deeply") from None
    if not isinstance(payload, dict):
        raise SourceMetadataError("sidecar top level is not an object")
    if payload.get("sidecar_schema_version") != SIDECAR_SCHEMA_VERSION:
        raise SourceMetadataError("sidecar schema version mismatch")
    if payload.get("media_sha256") != expected_media_sha256:
        raise SourceMetadataError("sidecar media hash does not match the published file")
    acquired_raw = payload.get("acquired_at")
    if not isinstance(acquired_raw, str):
        raise SourceMetadataError("sidecar acquired_at missing")
    try:
        acquired_at = datetime.fromisoformat(acquired_raw)
    except ValueError:
        raise SourceMetadataError("sidecar acquired_at is not ISO-8601") from None
    if acquired_at.tzinfo is None:
        raise SourceMetadataError("sidecar acquired_at is not timezone-aware")
    snapshot = payload.get("snapshot")
    if not isinstance(snapshot, dict):
        raise SourceMetadataError("sidecar snapshot missing")

    upload_date_raw = snapshot.get("upload_date")
    upload_date: date | None = None
    if upload_date_raw is not None:
        if not isinstance(upload_date_raw, str):
            raise SourceMetadataError("sidecar upload_date has a non-string value")
        try:
            upload_date = date.fromisoformat(upload_date_raw)
        except ValueError:
            raise SourceMetadataError("sidecar upload_date is not ISO-8601") from None
    duration = snapshot.get("duration_seconds")
    if duration is not None and (
        isinstance(duration, bool) or not isinstance(duration, (int, float))
    ):
        raise SourceMetadataError("sidecar duration_seconds has a non-numeric value")
    tags_raw = snapshot.get("tags", [])
    if not isinstance(tags_raw, list) or any(
        not isinstance(tag, str) for tag in tags_raw
    ):
        raise SourceMetadataError("sidecar tags is not a list of strings")
    raw = snapshot.get("raw", {})
    if not isinstance(raw, dict):
        raise SourceMetadataError("sidecar raw is not an object")
    raw_schema_version = snapshot.get("raw_schema_version")
    if not isinstance(raw_schema_version, int) or isinstance(raw_schema_version, bool):
        raise SourceMetadataError("sidecar raw_schema_version missing")

    meta = SourceMetadata(
        title=_expect_str_or_none(snapshot.get("title"), "title"),
        uploader=_expect_str_or_none(snapshot.get("uploader"), "uploader"),
        uploader_url=_expect_str_or_none(snapshot.get("uploader_url"), "uploader_url"),
        channel=_expect_str_or_none(snapshot.get("channel"), "channel"),
        channel_url=_expect_str_or_none(snapshot.get("channel_url"), "channel_url"),
        description=_expect_str_or_none(snapshot.get("description"), "description"),
        upload_date=upload_date,
        duration_seconds=_clean_duration(duration),
        tags=tuple(tags_raw),
        canonical_url=_expect_str_or_none(snapshot.get("canonical_url"), "canonical_url"),
        extractor=_expect_str_or_none(snapshot.get("extractor"), "extractor"),
        extractor_version=_expect_str_or_none(
            snapshot.get("extractor_version"), "extractor_version"
        ),
        raw=raw,
        raw_schema_version=raw_schema_version,
    )
    return meta, acquired_at
