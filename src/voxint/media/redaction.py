"""Error-text hygiene for values persisted to the ledger and surfaced in logs.

Two pure, DB-free, network-free primitives applied at the boundaries where
operator-facing error text is *produced* (the yt-dlp subprocess wrapper) or
*stored* (the pipeline engine's error-persistence path):

- :func:`redact` scrubs anything credential-ish that untrusted remote content —
  a yt-dlp ``stderr`` tail — can carry into an error message. A full
  ``http(s)://`` URL can hold a signed/expiring query param, a path signature,
  or embedded ``user:pass@`` credentials, so it is reduced to
  ``scheme://host/<redacted>`` (host kept as the one useful diagnostic — *which*
  service failed — since the host itself is not a secret). The *value* of a
  credential-bearing yt-dlp flag (``--proxy``, ``--cookies``, a password) is
  dropped entirely. It errs toward redacting too much rather than leaking a
  token.
- :func:`cap_length` bounds an error string's length before it is written to
  ``StageRun.error`` / ``PipelineRun.error`` so a pathological diagnostic can
  never bloat the ledger.

The redaction is deliberately conservative and structural, not a classifier: it
does NOT try to tell a bot-block from a transient failure (the pipeline parks
such a run FAILED for a manual Requeue regardless). Freeform secrets that a
tool might print outside a URL or a known flag (a bare signature line, a cookie
path echoed in prose) are out of scope here — the injection points Voxint
*controls* (the URL and the proxy/cookie flags) are covered; broader coverage
needs the yt-dlp lockdown in slice 6g.
"""

import re
from urllib.parse import urlsplit

_REDACTED = "<redacted>"
_REDACTED_URL = "<redacted-url>"

# Bound on any error string persisted to a ledger error column. The columns are
# unbounded Postgres TEXT, so this is hygiene (keep a diagnostic legible and the
# row small), not overflow protection.
MAX_STORED_ERROR_CHARS = 4000
_TRUNCATION_MARKER = "… [truncated]"

# A run of non-space characters starting at an http(s) scheme. URL structure
# guarantees every credential-bearing part (userinfo, path, query, fragment)
# sits AFTER the authority, so keeping only scheme+host and dropping the rest
# cannot leak a signed token hiding inside the bare host.
_URL_RE = re.compile(r"(?i)\bhttps?://\S+")

# Trailing punctuation the greedy \S+ may have swallowed onto an authority that
# has no path/query delimiter (e.g. "https://host," or "https://host)."). The
# authority itself is host[:port] with optional [IPv6]; strip from the first
# character that cannot belong to one so we never keep glued-on junk.
_HOST_TAIL_RE = re.compile(r"[^A-Za-z0-9._:\[\]-].*$", re.DOTALL)

# yt-dlp flags whose FOLLOWING token is a secret (a proxy string, a cookie-file
# path, a password). Redact the value but keep the flag so the diagnostic still
# says *what* was configured, just not the secret. Matched before URL redaction
# so a flag whose value is itself a URL loses its host too (more conservative).
_SECRET_FLAG_RE = re.compile(
    r"(?i)(--(?:proxy|cookies|cookies-from-browser|username|password|"
    r"video-password|ap-username|ap-password)[=\s]+)(\S+)"
)


def _redact_url(match: "re.Match[str]") -> str:
    parts = urlsplit(match.group(0))
    host = parts.netloc.rpartition("@")[2]  # drop any user:pass@ userinfo
    host = _HOST_TAIL_RE.sub("", host)
    if not host:
        return _REDACTED_URL
    return f"{parts.scheme}://{host}/{_REDACTED}"


def redact(text: str) -> str:
    """Return ``text`` with URLs reduced to ``scheme://host/<redacted>`` and the
    values of credential-bearing yt-dlp flags replaced with ``<redacted>``.

    Pure and idempotent-friendly: a string with nothing to redact is returned
    unchanged. Apply it BEFORE truncating an untrusted blob — truncating first
    could split a URL and leave a schemeless ``?token=...`` fragment this would
    no longer recognise.
    """
    scrubbed = _SECRET_FLAG_RE.sub(lambda m: f"{m.group(1)}{_REDACTED}", text)
    return _URL_RE.sub(_redact_url, scrubbed)


def cap_length(text: str, max_len: int = MAX_STORED_ERROR_CHARS) -> str:
    """Return ``text`` unchanged if within ``max_len``, else truncated to at most
    ``max_len`` characters with a trailing truncation marker."""
    if len(text) <= max_len:
        return text
    keep = max_len - len(_TRUNCATION_MARKER)
    if keep <= 0:
        return _TRUNCATION_MARKER[:max_len]
    return text[:keep] + _TRUNCATION_MARKER
