"""Stateless, action-bound CSRF token mint/verify (pure functions)."""

from voxint.api.csrf import (
    CSRF_FETCH,
    CSRF_REQUEUE,
    CSRF_SUBMIT,
    mint_csrf_token,
    verify_csrf_token,
)

_CSRF_KEY = "csrf-signing-key-for-tests"  # low-entropy, not a real secret


def test_minted_token_verifies_for_its_action() -> None:
    for action in (CSRF_SUBMIT, CSRF_FETCH, CSRF_REQUEUE):
        token = mint_csrf_token(_CSRF_KEY, action)
        assert verify_csrf_token(_CSRF_KEY, action, token) is True


def test_token_is_bound_to_its_action() -> None:
    # A token minted for /submit is NOT valid on /fetch or /requeue.
    token = mint_csrf_token(_CSRF_KEY, CSRF_SUBMIT)
    assert verify_csrf_token(_CSRF_KEY, CSRF_FETCH, token) is False
    assert verify_csrf_token(_CSRF_KEY, CSRF_REQUEUE, token) is False


def test_wrong_secret_fails() -> None:
    token = mint_csrf_token(_CSRF_KEY, CSRF_FETCH)
    assert verify_csrf_token("a-different-key", CSRF_FETCH, token) is False


def test_missing_or_malformed_token_fails() -> None:
    for bad in (None, "", "no-separator", ".", "nonce.", ".mac", "nonce.wrongmac"):
        assert verify_csrf_token(_CSRF_KEY, CSRF_FETCH, bad) is False


def test_tampered_mac_fails() -> None:
    token = mint_csrf_token(_CSRF_KEY, CSRF_SUBMIT)
    nonce, _, mac = token.partition(".")
    assert verify_csrf_token(_CSRF_KEY, CSRF_SUBMIT, f"{nonce}.{'0' * len(mac)}") is False


def test_tampered_nonce_fails() -> None:
    token = mint_csrf_token(_CSRF_KEY, CSRF_SUBMIT)
    nonce, _, mac = token.partition(".")
    assert verify_csrf_token(_CSRF_KEY, CSRF_SUBMIT, f"{nonce}x.{mac}") is False


def test_fresh_nonce_each_mint() -> None:
    a = mint_csrf_token(_CSRF_KEY, CSRF_FETCH)
    b = mint_csrf_token(_CSRF_KEY, CSRF_FETCH)
    assert a != b  # a fresh random nonce each render
    assert verify_csrf_token(_CSRF_KEY, CSRF_FETCH, a)
    assert verify_csrf_token(_CSRF_KEY, CSRF_FETCH, b)
