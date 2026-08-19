"""Stateless, action-bound, time-limited CSRF token mint/verify (pure functions)."""

from voxint.api.csrf import (
    _CSRF_FUTURE_SKEW_SECONDS,
    _CSRF_TTL_SECONDS,
    CSRF_CLAIM,
    CSRF_FETCH,
    CSRF_REQUEUE,
    CSRF_SETUP,
    CSRF_SUBMIT,
    mint_csrf_token,
    verify_csrf_token,
)

_CSRF_KEY = "csrf-signing-key-for-tests"  # low-entropy, not a real secret
_NOW = 1_700_000_000.0  # fixed clock for deterministic expiry tests


def test_minted_token_verifies_for_its_action() -> None:
    for action in (CSRF_SUBMIT, CSRF_FETCH, CSRF_REQUEUE, CSRF_CLAIM, CSRF_SETUP):
        token = mint_csrf_token(_CSRF_KEY, action)
        assert verify_csrf_token(_CSRF_KEY, action, token) is True


def test_token_is_bound_to_its_action() -> None:
    # A token minted for /submit is NOT valid on /fetch or /requeue.
    token = mint_csrf_token(_CSRF_KEY, CSRF_SUBMIT)
    assert verify_csrf_token(_CSRF_KEY, CSRF_FETCH, token) is False
    assert verify_csrf_token(_CSRF_KEY, CSRF_REQUEUE, token) is False


def test_setup_token_is_bound_to_its_action() -> None:
    # The wizard's token is not interchangeable with the other mutation forms
    # (and theirs are not valid on the wizard's POSTs).
    setup_token = mint_csrf_token(_CSRF_KEY, CSRF_SETUP)
    assert verify_csrf_token(_CSRF_KEY, CSRF_SETUP, setup_token) is True
    assert verify_csrf_token(_CSRF_KEY, CSRF_SUBMIT, setup_token) is False
    submit_token = mint_csrf_token(_CSRF_KEY, CSRF_SUBMIT)
    assert verify_csrf_token(_CSRF_KEY, CSRF_SETUP, submit_token) is False


def test_wrong_secret_fails() -> None:
    token = mint_csrf_token(_CSRF_KEY, CSRF_FETCH)
    assert verify_csrf_token("a-different-key", CSRF_FETCH, token) is False


def test_missing_or_malformed_token_fails() -> None:
    # None/empty, wrong part-count (incl. a legacy two-part token), empty parts.
    for bad in (
        None,
        "",
        "no-separator",
        "..",
        "nonce.ts.",
        "nonce..mac",
        ".ts.mac",
        "nonce.mac",  # legacy two-part shape → refused
        "nonce.notanint.mac",  # non-integer ts
    ):
        assert verify_csrf_token(_CSRF_KEY, CSRF_FETCH, bad) is False


def test_legacy_two_part_token_is_refused() -> None:
    # A token in the pre-D2 nonce.mac format (open form from before the upgrade)
    # has the wrong shape and is refused → the operator refreshes for a fresh one.
    import hmac
    from hashlib import sha256

    nonce = "legacy-nonce"
    legacy_mac = hmac.new(
        _CSRF_KEY.encode(), f"{CSRF_SUBMIT}.{nonce}".encode(), sha256
    ).hexdigest()
    assert verify_csrf_token(_CSRF_KEY, CSRF_SUBMIT, f"{nonce}.{legacy_mac}") is False


def test_tampered_mac_fails() -> None:
    token = mint_csrf_token(_CSRF_KEY, CSRF_SUBMIT, now=_NOW)
    nonce, ts, mac = token.split(".")
    assert (
        verify_csrf_token(_CSRF_KEY, CSRF_SUBMIT, f"{nonce}.{ts}.{'0' * len(mac)}", now=_NOW)
        is False
    )


def test_tampered_nonce_fails() -> None:
    token = mint_csrf_token(_CSRF_KEY, CSRF_SUBMIT, now=_NOW)
    nonce, ts, mac = token.split(".")
    assert verify_csrf_token(_CSRF_KEY, CSRF_SUBMIT, f"{nonce}x.{ts}.{mac}", now=_NOW) is False


def test_backdated_ts_fails() -> None:
    # The ts is signed, so shifting it invalidates the mac — a captured token
    # cannot be re-stamped to dodge expiry.
    token = mint_csrf_token(_CSRF_KEY, CSRF_SUBMIT, now=_NOW)
    nonce, ts, mac = token.split(".")
    forged = f"{nonce}.{int(ts) + 10}.{mac}"
    assert verify_csrf_token(_CSRF_KEY, CSRF_SUBMIT, forged, now=_NOW) is False


def test_fresh_nonce_each_mint() -> None:
    a = mint_csrf_token(_CSRF_KEY, CSRF_FETCH)
    b = mint_csrf_token(_CSRF_KEY, CSRF_FETCH)
    assert a != b  # a fresh random nonce each render
    assert verify_csrf_token(_CSRF_KEY, CSRF_FETCH, a)
    assert verify_csrf_token(_CSRF_KEY, CSRF_FETCH, b)


def test_token_verifies_up_to_the_ttl_boundary() -> None:
    token = mint_csrf_token(_CSRF_KEY, CSRF_SUBMIT, now=_NOW)
    # Valid at mint, mid-window, and exactly at the TTL edge.
    assert verify_csrf_token(_CSRF_KEY, CSRF_SUBMIT, token, now=_NOW) is True
    assert verify_csrf_token(_CSRF_KEY, CSRF_SUBMIT, token, now=_NOW + 3600) is True
    assert verify_csrf_token(
        _CSRF_KEY, CSRF_SUBMIT, token, now=_NOW + _CSRF_TTL_SECONDS
    ) is True


def test_expired_token_fails() -> None:
    token = mint_csrf_token(_CSRF_KEY, CSRF_SUBMIT, now=_NOW)
    assert verify_csrf_token(
        _CSRF_KEY, CSRF_SUBMIT, token, now=_NOW + _CSRF_TTL_SECONDS + 1
    ) is False


def test_future_dated_token_fails_but_tolerates_small_skew() -> None:
    # A token minted slightly ahead of the verifier's clock still verifies (skew
    # allowance); one implausibly far in the future is refused.
    token = mint_csrf_token(_CSRF_KEY, CSRF_SUBMIT, now=_NOW)
    assert verify_csrf_token(
        _CSRF_KEY, CSRF_SUBMIT, token, now=_NOW - _CSRF_FUTURE_SKEW_SECONDS
    ) is True
    assert verify_csrf_token(
        _CSRF_KEY, CSRF_SUBMIT, token, now=_NOW - _CSRF_FUTURE_SKEW_SECONDS - 1
    ) is False
