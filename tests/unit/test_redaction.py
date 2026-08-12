"""Unit tests for the error-text redaction/length-cap primitives.

The redaction property is asserted STRUCTURALLY: a unique sentinel token is
planted in the secret position and the assertion is that it is ABSENT from the
output (a substring like ``"token"`` could false-pass on unrelated text). All
synthetic data is neutral — ``example.com`` and the IETF documentation ranges
(TEST-NET-1/2/3) — never a private/internal host.
"""

import pytest

from voxint.media.redaction import (
    MAX_STORED_ERROR_CHARS,
    cap_length,
    provenance_host,
    redact,
)

_SENTINEL = "planted-marker-do-not-match"


def test_signed_query_params_are_dropped() -> None:
    text = f"ERROR: HTTP 403 for https://cdn.example.com/media.mp3?token={_SENTINEL}&sig=abcd"
    out = redact(text)
    assert _SENTINEL not in out
    assert "token=" not in out and "sig=" not in out
    assert "https://cdn.example.com/<redacted>" in out
    # The failure context around the URL survives.
    assert "HTTP 403" in out


def test_signed_path_segment_is_dropped() -> None:
    out = redact(f"fetching https://example.com/{_SENTINEL}/manifest.m3u8")
    assert _SENTINEL not in out
    assert "https://example.com/<redacted>" in out


def test_embedded_userinfo_is_dropped() -> None:
    out = redact(f"auth https://alice:{_SENTINEL}@example.com/feed rejected")
    assert _SENTINEL not in out
    assert "alice" not in out
    assert "https://example.com/<redacted>" in out


def test_host_is_preserved_as_a_diagnostic() -> None:
    # The host is not a secret and tells the operator WHICH service failed.
    out = redact("https://podcasts.example.org/ep1?e=1699999999&h=deadbeef")
    assert "podcasts.example.org" in out
    assert "deadbeef" not in out and "e=1699999999" not in out


def test_ip_literal_host_url() -> None:
    out = redact(f"https://203.0.113.7/stream?key={_SENTINEL}")
    assert _SENTINEL not in out
    assert "https://203.0.113.7/<redacted>" in out


def test_port_is_kept_on_host() -> None:
    out = redact(f"http://198.51.100.9:8443/dl?sig={_SENTINEL}")
    assert _SENTINEL not in out
    assert "http://198.51.100.9:8443/<redacted>" in out


def test_multiple_urls_all_redacted() -> None:
    text = (
        f"redirect https://a.example.com/x?t={_SENTINEL} -> "
        f"https://b.example.net/y?t={_SENTINEL}"
    )
    out = redact(text)
    assert _SENTINEL not in out
    assert "https://a.example.com/<redacted>" in out
    assert "https://b.example.net/<redacted>" in out


def test_proxy_flag_value_is_redacted() -> None:
    out = redact(f"argv: --proxy http://bob:{_SENTINEL}@203.0.113.5:3128 --no-config")
    assert _SENTINEL not in out
    assert "--proxy <redacted>" in out
    # An unrelated flag after the redacted value is untouched.
    assert "--no-config" in out


def test_proxy_flag_equals_form_is_redacted() -> None:
    out = redact(f"--proxy=socks5://{_SENTINEL}@203.0.113.5:1080")
    assert _SENTINEL not in out
    assert "--proxy=<redacted>" in out


def test_cookies_flag_path_is_redacted() -> None:
    out = redact(f"--cookies /secrets/{_SENTINEL}.txt")
    assert _SENTINEL not in out
    assert "--cookies <redacted>" in out


def test_cookies_from_browser_value_is_redacted() -> None:
    out = redact(f"--cookies-from-browser chrome:{_SENTINEL}")
    assert _SENTINEL not in out
    assert "--cookies-from-browser <redacted>" in out


def test_password_flags_are_redacted() -> None:
    out = redact(f"--username carol --password {_SENTINEL} --video-password {_SENTINEL}")
    assert _SENTINEL not in out
    assert "carol" not in out  # --username value is a credential too
    assert out.count("<redacted>") == 3


def test_text_without_secrets_is_unchanged() -> None:
    text = "download command failed (exit 1): ERROR: Unable to extract player response"
    assert redact(text) == text


def test_trailing_punctuation_is_not_kept_in_host() -> None:
    # A URL glued to following punctuation with no path/query delimiter must not
    # keep the junk on the authority.
    out = redact("see (https://example.com), then retry")
    assert "https://example.com/<redacted>" in out
    assert "example.com)" not in out


def test_scheme_only_token_is_fully_redacted() -> None:
    # Degenerate authority-less token still cannot pass through raw.
    out = redact("weird https://@ token")
    assert "<redacted-url>" in out


def test_url_glued_to_preceding_text_is_still_redacted() -> None:
    # No leading word boundary: a URL abutting prose with no separator must not
    # slip through with its signed query intact.
    out = redact(f"ERROR:retryhttps://cdn.example.com/a?token={_SENTINEL}")
    assert _SENTINEL not in out
    assert "token=" not in out


def test_ipv6_host_with_port_is_kept_bracketed() -> None:
    out = redact(f"https://[2001:db8::1]:8443/dl?sig={_SENTINEL}")
    assert _SENTINEL not in out
    assert "https://[2001:db8::1]:8443/<redacted>" in out


def test_non_numeric_port_fails_closed() -> None:
    # A bogus "port" that is really leaked text must not be preserved as a port.
    out = redact(f"https://example.com:{_SENTINEL}/path")
    assert _SENTINEL not in out
    assert "<redacted-url>" in out


def test_malformed_ipv6_authority_does_not_raise_and_is_redacted() -> None:
    # An unterminated IPv6 literal makes urlsplit raise; the callback must fail
    # closed to <redacted-url> instead of throwing out of re.sub.
    out = redact(f"https://[2001:db8::{_SENTINEL}/x")
    assert _SENTINEL not in out
    assert "<redacted-url>" in out


def test_host_starting_with_invalid_char_fails_closed() -> None:
    # A host whose first character cannot belong to an authority is stripped to
    # empty and must collapse to <redacted-url>, never leak the query.
    out = redact(f"https://~weird/x?token={_SENTINEL}")
    assert _SENTINEL not in out
    assert "<redacted-url>" in out


def test_quoted_flag_value_with_spaces_is_fully_redacted() -> None:
    # A cookie path with spaces is one shell-quoted token; the whole quoted run
    # must be redacted, not just up to the first space.
    out = redact(f"--cookies '/secrets/My Cookies/{_SENTINEL}.txt'")
    assert _SENTINEL not in out
    assert "--cookies <redacted>" in out


# --- slice 6g: socks-scheme widening + extra_secrets verbatim scrub -----------


def test_socks_proxy_url_in_prose_is_redacted() -> None:
    # A socks5:// proxy string echoed as prose (no --proxy flag in front, so
    # _SECRET_FLAG_RE misses it) must still lose its userinfo — _URL_RE now matches
    # socks schemes, not only http(s).
    out = redact(f"Unable to connect to proxy socks5://bob:{_SENTINEL}@203.0.113.5:1080")
    assert _SENTINEL not in out
    assert "bob" not in out
    assert "socks5://203.0.113.5:1080/<redacted>" in out


def test_socks4_and_socks5h_schemes_are_redacted() -> None:
    for scheme in ("socks4", "socks5h"):
        out = redact(f"proxy {scheme}://{_SENTINEL}@198.51.100.9:1080 failed")
        assert _SENTINEL not in out
        assert f"{scheme}://198.51.100.9:1080/<redacted>" in out


def test_extra_secrets_scrub_a_prose_cookie_path() -> None:
    # A cookies path echoed WITHOUT the --cookies flag (yt-dlp's "[Errno 2]" prose)
    # is a bare path no scheme/flag regex can catch — extra_secrets removes it.
    path = f"/secrets/{_SENTINEL}/cookies.txt"
    out = redact(f"Could not load cookies [Errno 2]: '{path}'", extra_secrets=(path,))
    assert _SENTINEL not in out
    assert path not in out
    assert "<redacted>" in out


def test_extra_secrets_scrub_a_prose_proxy_string() -> None:
    proxy = f"socks5://user:{_SENTINEL}@203.0.113.5:1080"
    out = redact(f"connecting via {proxy} timed out", extra_secrets=(proxy,))
    assert _SENTINEL not in out


def test_extra_secrets_empty_values_are_skipped() -> None:
    # An unset proxy/cookies ("") must not splice the redaction marker into text.
    text = "download command failed (exit 1): ERROR: Forbidden"
    assert redact(text, extra_secrets=("", "")) == text


def test_redact_without_extra_secrets_is_unchanged_behaviour() -> None:
    out = redact(f"https://cdn.example.com/a?token={_SENTINEL}")
    assert _SENTINEL not in out
    assert "https://cdn.example.com/<redacted>" in out


def test_cap_length_passes_through_within_limit() -> None:
    text = "short error"
    assert cap_length(text, max_len=100) == text


def test_cap_length_at_exact_limit_is_unchanged() -> None:
    text = "x" * 50
    assert cap_length(text, max_len=50) == text


def test_cap_length_truncates_over_limit_with_marker() -> None:
    text = "y" * 500
    out = cap_length(text, max_len=100)
    assert len(out) <= 100
    assert out.endswith("[truncated]")
    assert out.startswith("y")


def test_cap_length_tiny_limit_stays_bounded() -> None:
    out = cap_length("abcdefghij", max_len=3)
    assert len(out) <= 3


def test_cap_length_non_positive_limit_yields_empty() -> None:
    # A zero or negative cap has no room for any content or marker.
    assert cap_length("abcdef", max_len=0) == ""
    assert cap_length("abcdef", max_len=-5) == ""


def test_cap_length_default_bound() -> None:
    text = "z" * (MAX_STORED_ERROR_CHARS + 1000)
    out = cap_length(text)
    assert len(out) <= MAX_STORED_ERROR_CHARS
    assert out.endswith("[truncated]")


# --- provenance_host: display-only host extraction ----------------------------
# The display analogue of redact(): reduce a stored source_url to a bare host so
# the console shows *where* a URL run came from, never the raw URL (whose query
# could carry a signed token). Bare host = no port, no path/query/fragment.


def test_provenance_host_returns_bare_host() -> None:
    assert provenance_host("https://www.youtube.com/watch?v=abc") == "www.youtube.com"


def test_provenance_host_drops_signed_query_and_credentials() -> None:
    # A planted token in the query and userinfo must be absent from the host.
    url = f"https://user:{_SENTINEL}@cdn.example.com/media.mp3?sig={_SENTINEL}"
    out = provenance_host(url)
    assert out == "cdn.example.com"
    assert out is not None and _SENTINEL not in out


def test_provenance_host_omits_the_port() -> None:
    # Unlike _redact_url (which keeps the port as a failure diagnostic),
    # provenance is "where from", so the port is noise and is dropped.
    assert provenance_host("https://example.com:8443/dl") == "example.com"


def test_provenance_host_ipv4_literal() -> None:
    assert provenance_host("https://203.0.113.7/dl") == "203.0.113.7"


def test_provenance_host_ipv6_literal_is_rebracketed_without_port() -> None:
    assert provenance_host("https://[2001:db8::1]:8443/dl") == "[2001:db8::1]"


def test_provenance_host_none_input_returns_none() -> None:
    # A local/uploaded run has no source_url; the template renders None as "—".
    assert provenance_host(None) is None


@pytest.mark.parametrize(
    "garbage",
    [
        "",  # empty
        "not-a-url",  # no scheme/host
        "https://",  # scheme but no host
        "https://example.com:notaport/dl",  # non-numeric port fails closed
        "https://[2001:db8::/dl",  # malformed IPv6 literal fails closed
        "ftp://example.com/x",  # returns the host regardless of scheme…
    ],
)
def test_provenance_host_fails_closed_to_none_or_safe_host(garbage: str) -> None:
    out = provenance_host(garbage)
    # Fail-closed contract: it never returns the raw string, and never leaks a
    # path/query. It is either None or a plain host token with no separators.
    assert out != garbage
    if out is not None:
        assert "/" not in out and "?" not in out and " " not in out
