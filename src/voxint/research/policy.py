"""Research-side URL gate: the shared netcheck string gate plus fetcher extras.

Thin by design — the address policy and the base string gate live in
:mod:`voxint.media.netcheck` (ONE place to audit); this module layers only the
rules that exist because ``read_url`` actually CONNECTS (the ingest path hands
its URL to yt-dlp, which re-parses it independently, so these would buy ingest
nothing):

- the fragment is stripped (never sent on the wire; keeping it would make the
  logical URL and the request URL diverge),
- a ``%`` in the host is refused (percent-encoded authorities and IPv6 zone
  identifiers are parser-differential surface with no legitimate use here),
- a DNS hostname is IDNA-canonicalized to ASCII exactly once — the SAME string
  is then used for resolution, the Host header, and TLS SNI, so the address
  that was vetted is provably the identity that is verified.
"""

from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from voxint.media.netcheck import HttpUrl, UrlPolicyError, parse_http_url

__all__ = ["ResearchUrl", "UrlPolicyError", "gate_research_url"]


@dataclass(frozen=True)
class ResearchUrl:
    """A vetted, fetchable URL: the fragment-free logical URL + its parts.

    ``ascii_host`` is the IDNA-encoded hostname (or the IP literal verbatim):
    the one string used for DNS resolution, the Host header, and SNI.
    ``authority`` is what the Host header carries (ascii host, bracketed if
    IPv6, plus any explicit port).
    """

    url: str
    scheme: str
    ascii_host: str
    port: int | None
    is_ip_literal: bool

    @property
    def authority(self) -> str:
        host = self.ascii_host
        if ":" in host:  # an IPv6 literal — the Host header needs brackets
            host = f"[{host}]"
        if self.port is not None:
            return f"{host}:{self.port}"
        return host


def _strip_fragment(url: str) -> str:
    parts = urlsplit(url)
    if not parts.fragment:
        return url
    return urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))


def gate_research_url(url: str) -> ResearchUrl:
    """Apply the shared string gate plus the research-only rules.

    Raises :class:`UrlPolicyError` (netcheck's own error — one vocabulary) on
    refusal; messages never carry the URL.
    """
    parsed: HttpUrl = parse_http_url(url)
    if "%" in parsed.host:
        # Percent-encoded authority bytes or an IPv6 zone-id ("fe80::1%eth0")
        # — both are parser-differential/scope tricks, never a public target.
        raise UrlPolicyError("URL host contains a percent sign")
    if parsed.ip is not None:
        ascii_host = parsed.host
    else:
        try:
            ascii_host = parsed.host.encode("idna").decode("ascii")
        except UnicodeError:
            # An un-encodable (or embedded-dot trick like "a。b") hostname is
            # refused outright — resolving one string and fetching another is
            # exactly the ambiguity this gate exists to close.
            raise UrlPolicyError("URL host is not IDNA-encodable") from None
    return ResearchUrl(
        url=_strip_fragment(parsed.url),
        scheme=parsed.scheme,
        ascii_host=ascii_host,
        port=parsed.port,
        is_ip_literal=parsed.ip is not None,
    )
