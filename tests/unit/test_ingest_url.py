"""URL policy for yt-dlp ingestion — the first (string-level) SSRF guard.

These are pure-function checks (no DB, no network); worker-side DNS
re-resolution + yt-dlp lockdown are the authoritative second line, landing in
slice 6g. ``validate_ingest_url`` gates row creation in ``submit_url`` (exercised
against Postgres in tests/integration/test_ingest_service.py).
"""

import pytest

from voxint.ingest.service import UrlValidationError, validate_ingest_url


@pytest.mark.parametrize(
    "url",
    [
        "",  # empty
        "   ",  # whitespace-only collapses to empty
        "example.com/watch",  # no scheme
        "//example.com/watch",  # scheme-relative, no scheme
        "ftp://example.com/f.mp3",  # non-http(s) scheme
        "file:///etc/passwd",  # file scheme
        "data:text/plain,hi",  # data scheme
        "http:///just/a/path",  # no host
        "http://user:pass@example.com/v",  # embedded credentials
        "http://user@example.com/v",  # embedded username only
        "http://exa mple.com/v",  # internal whitespace (space)
        "http://example.com/a\tb",  # internal tab
        "http://example.com/a\nb",  # internal newline
        "http://example.com/\x00",  # NUL
        "http://example.com/a\x1fb",  # control char
        "http://example.com:99999/v",  # out-of-range port
        "http://example.com:abc/v",  # non-numeric port
        "http://[::1/v",  # malformed IPv6 literal
        "https://" + "a" * 2100 + ".com/v",  # over the length ceiling
        # Non-public hosts (loopback / private / link-local / unspecified) —
        # rejected as literals now; names are re-resolved worker-side (6g). The
        # IETF TEST-NET documentation ranges below (192.0.2/24, 198.51.100/24,
        # 203.0.113/24) are non-global too, so they exercise the same is_global
        # rejection as RFC1918 (10/8, 192.168/16, 172.16/12) while keeping the
        # fixtures free of real internal-network literals.
        "http://localhost/v",
        "http://api.localhost/v",
        "http://127.0.0.1/v",
        "http://198.51.100.5/v",  # stands in for RFC1918 10/8
        "http://203.0.113.1/v",  # stands in for RFC1918 192.168/16
        "http://192.0.2.1/v",  # stands in for RFC1918 172.16/12
        "http://169.254.169.254/latest/meta-data",  # cloud metadata SSRF
        "http://0.0.0.0/v",
        "http://[::1]/v",  # IPv6 loopback
        "http://[fe80::1]/v",  # IPv6 link-local
        "http://[fc00::1]/v",  # IPv6 unique-local (private)
    ],
)
def test_rejects_unsafe_urls(url: str) -> None:
    with pytest.raises(UrlValidationError):
        validate_ingest_url(url)


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        (
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        ),
        # surrounding whitespace (a pasted trailing newline) is stripped
        ("  https://example.com/podcast.mp3\n", "https://example.com/podcast.mp3"),
        ("http://example.com/v", "http://example.com/v"),  # http is allowed
        ("https://8.8.8.8/v", "https://8.8.8.8/v"),  # a globally-routable IP literal
        ("https://example.com:8443/media", "https://example.com:8443/media"),  # valid port
        (
            "https://sub.example.co.uk/a/b?c=d&e=f#frag",
            "https://sub.example.co.uk/a/b?c=d&e=f#frag",
        ),
    ],
)
def test_accepts_and_trims_public_urls(given: str, expected: str) -> None:
    assert validate_ingest_url(given) == expected


def test_error_message_never_echoes_the_url() -> None:
    """A signed/secret query string must not leak into a 422 body or the logs."""
    secret = "https://cdn.example.com/media?token=SUPERSECRETSIGNATURE"  # public host, ok scheme
    # A private host with the same secret query — validation fails; the message
    # must not contain the secret (the host below is private → rejected).
    with pytest.raises(UrlValidationError) as exc:
        validate_ingest_url("http://192.0.2.9/media?token=SUPERSECRETSIGNATURE")
    assert "SUPERSECRETSIGNATURE" not in str(exc.value)
    # And a well-formed public URL with a token round-trips unchanged.
    assert validate_ingest_url(secret) == secret
