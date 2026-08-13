"""Stateless, action-bound CSRF tokens for the mutation forms.

The review console has no sessions — auth is HTTP Basic — so CSRF is defended
with a stateless *signed* token embedded in each mutating form and verified on
POST, rather than a server-stored synchronizer token or a cookie. HTTP Basic
credentials are cached and auto-attached by the browser to cross-site requests,
so CSRF is a real (if low-severity, on single-operator localhost) vector.

A token is ``nonce.HMAC_SHA256(secret, "action.nonce")``: unforgeable without the
app's ``csrf_secret``, and **bound to a specific action** so a token minted for
one form cannot be replayed against another route (a cheap, useful property — the
three mutation routes each mint and verify under their own action string). The
attacker cannot read a same-origin rendered token, so they cannot supply a valid
one; a missing/malformed/mis-signed token is refused *before* any DB write.

The secret is independent of ``voxint_password`` on purpose: a human-memorable
Basic password would make every rendered token a fast offline password-guessing
oracle. ``create_app`` uses ``Settings.csrf_secret`` when set (persistent, shared
by every worker) and otherwise mints a random per-process one (zero-config, but
open forms break on restart / across workers).

Not defended here: this is not an XSS mitigation (script on the page can read the
token), and a leaked token stays valid until the secret rotates (fresh nonces do
not make a token single-use — acceptable for this console). Verification is
constant-time (``hmac.compare_digest``).
"""

import hmac
import secrets
from hashlib import sha256

# token_urlsafe / hexdigest never emit ".", so it cleanly separates the two parts.
_SEP = "."
_NONCE_BYTES = 16

# Stable per-route action strings. A form mints its token under one of these and
# the matching POST route verifies under the same one, so a token is not
# interchangeable between the three mutation forms.
CSRF_SUBMIT = "submit"
CSRF_FETCH = "fetch"
CSRF_REQUEUE = "requeue"
CSRF_CLAIM = "claim"
# The first-run setup wizard (issue #3). One action for the whole flow: its
# POST steps land in slice 4, and a token minted on one wizard step being valid
# on another is a harmless same-operator replay (the wizard is a single guided
# flow, not independent mutation surfaces). Split per-step only if a step ever
# needs to reject another step's token.
CSRF_SETUP = "setup"


def _sign(secret: str, action: str, nonce: str) -> str:
    message = f"{action}{_SEP}{nonce}".encode()
    return hmac.new(secret.encode(), message, sha256).hexdigest()


def mint_csrf_token(secret: str, action: str) -> str:
    """Mint a fresh CSRF token binding a random nonce to ``action``."""
    nonce = secrets.token_urlsafe(_NONCE_BYTES)
    return f"{nonce}{_SEP}{_sign(secret, action, nonce)}"


def verify_csrf_token(secret: str, action: str, token: str | None) -> bool:
    """Return True iff ``token`` is a well-formed, correctly-signed token for
    ``action``. False for None/empty/malformed/mis-signed — the caller maps that
    to a 403 before any state change. Constant-time on the signature compare."""
    if not token:
        return False
    nonce, sep, mac = token.partition(_SEP)
    if not sep or not nonce or not mac:
        return False
    return hmac.compare_digest(mac, _sign(secret, action, nonce))
