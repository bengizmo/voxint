"""LLM destination policy: permit LAN inference, reject metadata destinations.

This is intentionally separate from ingest's public-address-only policy. Checks
resolve DNS again at client construction, but do not pin httpx's later resolution.
"""

import ipaddress
import socket
from urllib.parse import urlsplit

from voxint.config import get_settings, llm_endpoint_explicitly_set

# Metadata endpoints outside the usual link-local ranges (Alibaba and AWS IPv6).
_METADATA_ADDRESSES = frozenset(
    ipaddress.ip_address(value) for value in ("100.100.100.200", "fd00:ec2::254")
)
_NAT64 = ipaddress.ip_network("64:ff9b::/96")
_IPV4_COMPAT = ipaddress.ip_network("::/96")


def _blocked_reason(
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address, block_loopback: bool
) -> str | None:
    if ip in _METADATA_ADDRESSES:
        return "cloud metadata"
    if ip.is_unspecified:
        return "unspecified address"
    if ip.is_link_local:
        return "link-local"
    if block_loopback and ip.is_loopback:
        return "loopback in multi-user mode"
    if isinstance(ip, ipaddress.IPv6Address):
        embedded = ip.ipv4_mapped or ip.sixtofour
        if ip.teredo is not None:
            embedded = ip.teredo[1]
        if ip in _NAT64 or (ip in _IPV4_COMPAT and int(ip) not in (0, 1)):
            embedded = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
        if embedded is not None:
            return _blocked_reason(embedded, block_loopback)
    return None


def validate_llm_destination(url: str, *, multi_user: bool | None = None) -> None:
    """Reject malformed URLs, DNS failures, or any blocked destination answer.

    RFC1918 addresses remain allowed. Loopback is allowed for single-operator
    deployments, and blocked for changed endpoints in multi-user deployments.
    Errors deliberately omit the URL, which may contain credentials.
    """
    try:
        parts = urlsplit(url)
        host = parts.hostname
        port = parts.port
    except ValueError:
        raise ValueError("LLM base URL is malformed") from None
    if parts.scheme not in ("http", "https") or not host:
        raise ValueError("LLM base URL must be an absolute http/https URL")
    if parts.username is not None or parts.password is not None:
        raise ValueError("LLM base URL must not embed credentials")
    if multi_user is None:
        multi_user = get_settings().voxint_multi_user
    block_loopback = multi_user and llm_endpoint_explicitly_set(url)
    try:
        addresses = [ipaddress.ip_address(host)]
    except ValueError:
        try:
            infos = socket.getaddrinfo(
                host,
                port or (443 if parts.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            )
        except (OSError, UnicodeError):
            raise ValueError("LLM destination could not be resolved") from None
        if not infos:
            raise ValueError("LLM destination resolved to no addresses") from None
        try:
            addresses = [ipaddress.ip_address(info[4][0]) for info in infos]
        except ValueError:
            raise ValueError("LLM destination resolved to an invalid address") from None
    for ip in addresses:
        reason = _blocked_reason(ip, block_loopback)
        if reason:
            raise ValueError(f"LLM destination is blocked: {reason}")
