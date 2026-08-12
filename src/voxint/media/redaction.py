"""URL hygiene for values persisted to the ledger, logged, or shown to the operator.

Three pure, DB-free, network-free primitives applied at the boundaries where
operator-facing text is *produced* (the yt-dlp subprocess wrapper), *stored*
(the pipeline engine's error-persistence path), or *displayed* (a run's URL
provenance in the console):

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
- :func:`provenance_host` reduces a stored ``MediaItem.source_url`` to its bare
  host for display, so the console can show *where* a URL run came from without
  ever rendering the raw URL (whose path/query could carry a signed token). It
  shares :func:`redact`'s URL-parsing core so display and error text can never
  disagree on what the "host" of a URL is.

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
# cannot leak a signed token hiding inside the bare host. No leading boundary:
# a URL glued to preceding text ("...forhttps://host/x?token=...") must still be
# caught — anchoring on the literal scheme keeps the real scheme in the output.
# (Non-http schemes — e.g. a "socks5://user:pass@" proxy echoed in prose — are
# not matched here; they cannot reach stderr until slice 6g wires proxy args.)
_URL_RE = re.compile(r"(?i)https?://\S+")

# Trailing punctuation the greedy \S+ may have swallowed onto an authority that
# has no path/query delimiter (e.g. "https://host," or "https://host)."). The
# authority is host[:port] with optional [IPv6]; strip from the first character
# that cannot belong to one so we never keep glued-on junk.
_HOST_TAIL_RE = re.compile(r"[^A-Za-z0-9._:\[\]-].*$", re.DOTALL)

# yt-dlp flags whose FOLLOWING token is a secret (a proxy string, a cookie-file
# path, a password). Redact the value but keep the flag so the diagnostic still
# says *what* was configured, just not the secret. A value may be a shell-quoted
# string with spaces (a cookie path like '/Application Support/x.txt'), so a
# quoted run is consumed whole before the unquoted \S+ fallback. Matched before
# URL redaction so a flag whose value is itself a URL loses its host too.
_SECRET_FLAG_RE = re.compile(
    r"(?i)(--(?:proxy|cookies|cookies-from-browser|username|password|"
    r"video-password|ap-username|ap-password)[=\s]+)"
    r"(\"[^\"]*\"|'[^']*'|\S+)"
)


def _split_authority(url: str) -> "tuple[str, str, int | None] | None":
    """Parse ``url`` into ``(scheme, bare_host, port)``, or ``None`` fail-closed.

    ``bare_host`` is the hostname with userinfo, port, and IPv6 brackets already
    stripped by the stdlib parser, then trimmed of any junk the greedy URL regex
    glued on (``_HOST_TAIL_RE``); an IPv6 literal is returned *unbracketed* so
    each caller can re-bracket it as it needs. Returns ``None`` — never raises —
    for any authority the parser rejects (a malformed IPv6 literal, a non-numeric
    or out-of-range port) or that yields no scheme/host: the shared fail-closed
    core both :func:`_redact_url` (untrusted stderr) and :func:`provenance_host`
    (a stored URL) rely on to never leak arbitrary text.
    """
    try:
        parts = urlsplit(url)
        scheme = parts.scheme
        host = parts.hostname  # drops userinfo + port + IPv6 brackets, lowercased
        port = parts.port  # raises ValueError on a non-numeric/out-of-range port
    except ValueError:
        return None
    if not scheme or not host:
        return None
    host = _HOST_TAIL_RE.sub("", host)
    if not host:
        return None
    return scheme, host, port


def _redact_url(match: "re.Match[str]") -> str:
    """Reduce one matched URL to ``scheme://host[:port]/<redacted>``.

    TOTAL and fail-closed: any authority :func:`_split_authority` rejects (a
    malformed IPv6 literal, a non-numeric port) or that yields no host collapses
    to ``<redacted-url>`` rather than raising or preserving arbitrary text —
    stderr is untrusted, so the callback must never throw out of ``re.sub``. The
    port is *kept* here (it is a useful, non-secret diagnostic of which endpoint
    failed); the display-side :func:`provenance_host` drops it.
    """
    parsed = _split_authority(match.group(0))
    if parsed is None:
        return _REDACTED_URL
    scheme, host, port = parsed
    if ":" in host:  # IPv6 literal needs its brackets back before a port
        host = f"[{host}]"
    authority = f"{host}:{port}" if port is not None else host
    return f"{scheme}://{authority}/{_REDACTED}"


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
        # max_len too small to fit the marker; a non-positive cap yields "".
        return _TRUNCATION_MARKER[:max_len] if max_len > 0 else ""
    return text[:keep] + _TRUNCATION_MARKER


def provenance_host(source_url: str | None) -> str | None:
    """Reduce a stored ``source_url`` to its bare host for display, or ``None``.

    Returns just the host — ``cdn.example.com`` for a DNS/IPv4 authority,
    ``[2001:db8::1]`` for an IPv6 literal (re-bracketed so it reads as a host) —
    with the **port omitted**: "provenance" answers *where from*, not *which
    endpoint*, so the authority's port is noise here (unlike :func:`_redact_url`,
    which keeps it as a failure diagnostic). Everything after the authority — the
    path, query, and fragment that can carry a signed token — is dropped, so this
    is the only value a template should ever see for a URL run; the raw
    ``source_url`` must never reach the view.

    Fail-closed via the shared :func:`_split_authority` core: ``None`` in (a run
    with no ``source_url``), a malformed URL, a bad port, or a missing host all
    return ``None`` — never the raw string — which the template renders as a
    neutral placeholder. A stored ``source_url`` has already passed
    ``validate_ingest_url`` (absolute http/https), so in practice only ``None``
    is hit; the structural fail-closed guards a future non-validated caller.
    """
    if source_url is None:
        return None
    parsed = _split_authority(source_url)
    if parsed is None:
        return None
    _scheme, host, _port = parsed
    if ":" in host:  # IPv6 literal → re-bracket so it displays as a host
        host = f"[{host}]"
    return host
