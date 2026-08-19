"""Stateless, action-bound CSRF tokens for the mutation forms.

The review console has no sessions — auth is HTTP Basic — so CSRF is defended
with a stateless *signed* token embedded in each mutating form and verified on
POST, rather than a server-stored synchronizer token or a cookie. HTTP Basic
credentials are cached and auto-attached by the browser to cross-site requests,
so CSRF is a real (if low-severity, on single-operator localhost) vector.

A token is ``nonce.ts.HMAC_SHA256(secret, "action.nonce.ts")``: unforgeable
without the app's ``csrf_secret``, **bound to a specific action** so a token
minted for one form cannot be replayed against another route (a cheap, useful
property — each mutation route mints and verifies under its own action string),
and **stamped with a mint time** so it expires (finding D2). The attacker cannot
read a same-origin rendered token, so they cannot supply a valid one; a
missing/malformed/mis-signed/expired token is refused *before* any DB write.

The secret is independent of ``voxint_password`` on purpose: a human-memorable
Basic password would make every rendered token a fast offline password-guessing
oracle. ``create_app`` uses ``Settings.csrf_secret`` when set (persistent, shared
by every worker) and otherwise mints a random per-process one (zero-config, but
open forms break on restart / across workers).

Expiry (finding D2): the signed ``ts`` bounds a leaked token's usefulness to
``_CSRF_TTL_SECONDS`` after mint (24h — generous for a console form, so an
operator who leaves a page open still submits) rather than "forever until the
secret rotates". A small future-skew allowance tolerates minor clock jitter. This
is *not* single-use — replay inside the TTL is still possible — but it removes the
unbounded-lifetime window. The ``ts`` is signed, so it cannot be back-dated. A
legacy two-part token (minted before this change) has the wrong shape and is
refused with a harmless 403 → the operator refreshes to mint a fresh one.

Not defended here: this is not an XSS mitigation (script on the page can read the
token). Verification is constant-time (``hmac.compare_digest``).
"""

import hmac
import secrets
import time
from hashlib import sha256

# token_urlsafe (base64url) / decimal ts / hexdigest never emit ".", so it cleanly
# separates the three parts.
_SEP = "."
_NONCE_BYTES = 16
# A minted token verifies for 24h after its stamped mint time (finding D2). Long
# enough that an operator who leaves a form open still submits; short enough that a
# captured token is not usable indefinitely. Fixed, not a Settings knob — no
# operator needs to tune this, and a knob would be bloat.
_CSRF_TTL_SECONDS = 24 * 60 * 60
# Tolerate a token whose ts is a little ahead of our clock (minor skew between
# mint and verify), but reject one implausibly far in the future.
_CSRF_FUTURE_SKEW_SECONDS = 60

# Stable per-route action strings. A form mints its token under one of these and
# the matching POST route verifies under the same one, so a token is not
# interchangeable between mutation forms.
CSRF_SUBMIT = "submit"
CSRF_FETCH = "fetch"
CSRF_REQUEUE = "requeue"
# Run cancellation (issue #5). Its own action — a distinct pipeline-state
# mutation on the run detail page, never interchangeable with requeue/notes.
CSRF_CANCEL = "cancel"
CSRF_CLAIM = "claim"
# The first-run setup wizard (issue #3). One action for the whole flow: its
# POST steps land in slice 4, and a token minted on one wizard step being valid
# on another is a harmless same-operator replay (the wizard is a single guided
# flow, not independent mutation surfaces). Split per-step only if a step ever
# needs to reject another step's token.
CSRF_SETUP = "setup"
# The Settings page's tutorial actions (issue #3, slice 6): mark-complete and
# replay. One action for both — they are same-operator, same-surface mutations on
# a single row, so a token minted for one being valid on the other is a harmless
# replay (mirrors CSRF_SETUP's single-flow rationale).
CSRF_SETTINGS = "settings"
# Speaker roster curation (issue #7). Per-action tokens — unlike the wizard and
# settings surfaces these are independent mutations with different blast radii
# (a rename token must not be replayable as a merge), so each form mints and
# verifies under its own action.
# Per-run operator notes (issue #36): a single dedicated action for the run
# detail page's notes form — an independent mutation surface, so it never
# shares a token with requeue/claim on the same page.
CSRF_NOTES = "notes"
CSRF_ROSTER_RENAME = "roster-rename"
CSRF_ROSTER_MERGE = "roster-merge"
CSRF_ROSTER_ARCHIVE = "roster-archive"
CSRF_ROSTER_RESTORE = "roster-restore"
CSRF_ROSTER_EMBEDDING_DELETE = "roster-embedding-delete"
# Web-research jobs and profile-draft review (issue #40). Per-action tokens:
# starting a job (egress + LLM spend), cancelling one, and ruling on a draft
# are independent mutations with different blast radii.
CSRF_RESEARCH_START = "research-start"
CSRF_RESEARCH_CANCEL = "research-cancel"
CSRF_PROFILE_DECISION = "profile-decision"
# Run-level asset generation (issue #41): starting jobs (LLM spend) and
# cancelling one are independent mutations with different blast radii.
CSRF_ASSETS_GENERATE = "assets-generate"
CSRF_ASSETS_CANCEL = "assets-cancel"
# Run soft-archive + derived-media deletion (issue #5, slice 2). Per-action
# tokens — hiding a run (reversible), un-hiding it, and irreversibly deleting its
# derived audio files have very different blast radii and must never share a token.
CSRF_RUN_ARCHIVE = "run-archive"
CSRF_RUN_UNARCHIVE = "run-unarchive"
CSRF_RUN_MEDIA_DELETE = "run-media-delete"


def _sign(secret: str, action: str, nonce: str, ts: int) -> str:
    message = f"{action}{_SEP}{nonce}{_SEP}{ts}".encode()
    return hmac.new(secret.encode(), message, sha256).hexdigest()


def mint_csrf_token(secret: str, action: str, *, now: float | None = None) -> str:
    """Mint a fresh CSRF token binding a random nonce and the mint time to
    ``action``. ``now`` is injectable for tests; production uses wall-clock."""
    nonce = secrets.token_urlsafe(_NONCE_BYTES)
    ts = int(time.time() if now is None else now)
    return f"{nonce}{_SEP}{ts}{_SEP}{_sign(secret, action, nonce, ts)}"


def verify_csrf_token(
    secret: str, action: str, token: str | None, *, now: float | None = None
) -> bool:
    """Return True iff ``token`` is a well-formed, correctly-signed, unexpired token
    for ``action``. False for None/empty/malformed/mis-signed/expired/future-dated —
    the caller maps that to a 403 before any state change. A legacy two-part token
    is malformed here and refused. Constant-time on the signature compare; the
    ts/expiry checks run only on an already-authenticated signature, so they leak no
    timing about the secret. ``now`` is injectable for tests."""
    if not token:
        return False
    parts = token.split(_SEP)
    if len(parts) != 3:
        return False
    nonce, ts_raw, mac = parts
    if not nonce or not ts_raw or not mac:
        return False
    try:
        ts = int(ts_raw)
    except ValueError:
        return False
    if not hmac.compare_digest(mac, _sign(secret, action, nonce, ts)):
        return False
    current = time.time() if now is None else now
    # In-window: not expired (older than the TTL) and not implausibly future-dated
    # beyond the skew allowance (a forged/replayed clock).
    return current - _CSRF_TTL_SECONDS <= ts <= current + _CSRF_FUTURE_SKEW_SECONDS
