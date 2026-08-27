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
oracle. ``create_app`` uses ``Settings.csrf_secret`` when set (explicit,
operator-managed) and otherwise auto-generates one and persists it to
``media_root/.csrf_secret`` with 0600 permissions on first run, so open forms
survive restarts and multiple workers share the same key without any
configuration.

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

import contextlib
import hmac
import logging
import os
import secrets
import time
from hashlib import sha256
from pathlib import Path

logger = logging.getLogger(__name__)

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
# Manual speaker-profile field edits on the Console 2.0 profile page (#159) —
# a distinct act from ruling on a research draft, so a distinct token.
CSRF_SPEAKER_PROFILE_EDIT = "speaker-profile-edit"
# Run-level asset generation (issue #41): starting jobs (LLM spend) and
# cancelling one are independent mutations with different blast radii.
CSRF_ASSETS_GENERATE = "assets-generate"
CSRF_ASSETS_CANCEL = "assets-cancel"
# Transcript translation (issue #133): starting a job (LLM spend) and
# cancelling one are independent mutations with different blast radii.
CSRF_TRANSLATION_GENERATE = "translation-generate"
CSRF_TRANSLATION_CANCEL = "translation-cancel"
# Run soft-archive + derived-media deletion (issue #5, slice 2). Per-action
# tokens — hiding a run (reversible), un-hiding it, and irreversibly deleting its
# derived audio files have very different blast radii and must never share a token.
CSRF_RUN_ARCHIVE = "run-archive"
CSRF_RUN_UNARCHIVE = "run-unarchive"
CSRF_RUN_MEDIA_DELETE = "run-media-delete"
# Operator annotation tags (issue #86): global, run-less tag CRUD (create /
# rename / recolour / archive). Its own action — tag writes have no run or claim
# context, so they are CSRF-gated like run notes and never share a token with a
# run-scoped mutation.
CSRF_ANNOTATION_TAGS = "annotation-tags"
# Attributed audio-clip extraction (issue #88): POST a highlight -> a cached WAV
# clip. Its own action — clip generation is claim-less like the tag writes and
# the pull-quote export, so it never shares a token with a run-scoped mutation.
# Replay is harmless (extraction is idempotent + content-addressed), but the
# token still refuses a forged cross-site POST before any file is written.
CSRF_CLIP_EXTRACT = "clip-extract"
# Plugin mutating routes (issue #138). One shared action for every builtin
# plugin's POST surface: the capped PluginRouteDeps bundle exposes a single
# uniform ``verify_csrf(request)`` (token carried in the ``X-CSRF-Token`` header)
# rather than the core routes' per-form (action, token) pair, so a plugin never
# reaches into the app's per-surface CSRF constants. Same-operator, same-origin
# replay across a plugin's own forms is harmless (mirrors CSRF_SETTINGS).
CSRF_PLUGIN = "plugin"
# Projects (issue #153, Console 2.0 P2b). Per-action tokens: creating a project
# and assigning a folder to one are independent mutations with different blast
# radii (a create token must not be replayable to move a folder), so each form
# mints and verifies under its own action.
CSRF_PROJECT_CREATE = "project-create"
CSRF_PROJECT_ASSIGN = "project-assign"
# Project-scoped config editors (issue #153, P2a precedence freeze). The
# vocabulary and corrections overrides are independent mutations under their own
# per-action tokens; each also carries an "inherit" reset (write NULL) under the
# same action as its save.
CSRF_PROJECT_VOCAB = "project-vocabulary"
CSRF_PROJECT_CORRECTIONS = "project-corrections"
# Media library ingest (issue #154, Console 2.0 P2b). Upload and URL fetch move
# onto /media with their own action tokens, distinct from the legacy /submit and
# /fetch forms' CSRF_SUBMIT/CSRF_FETCH so a token minted on one surface is not
# valid on the other.
CSRF_MEDIA_SUBMIT = "media-submit"
CSRF_MEDIA_FETCH = "media-fetch"
# Media library organization (issue #154, Console 2.0 P2b). Per-action tokens:
# bulk-assigning a settings folder over a selection and registering/unregistering
# a folder are independent mutations with different blast radii (an assign token
# must not be replayable to unregister a folder), so each mints and verifies under
# its own action, and neither is interchangeable with the ingest tokens above.
CSRF_MEDIA_ASSIGN = "media-assign"
CSRF_MEDIA_FOLDERS = "media-folders"
# Media library bulk re-run (issue #154, Console 2.0 P2b). The two-step re-run has
# its own action tokens: the advisory preview (``media-rerun``, no mutation) and
# the atomic confirm that mints the runs (``media-rerun-confirm``). Splitting them
# means a token minted for the preview can never be replayed to drive the actual
# dispatch, and neither is interchangeable with assign or the ingest tokens.
CSRF_MEDIA_RERUN = "media-rerun"
CSRF_MEDIA_RERUN_CONFIRM = "media-rerun-confirm"
# Media library bulk archive/unarchive (issue #154, Console 2.0 P2b). Archiving a
# selection's latest run (reversible, hides it from the active library) and
# restoring it are independent mutations under their own per-action tokens, and
# neither is interchangeable with assign, re-run, or the ingest tokens above.
CSRF_MEDIA_ARCHIVE = "media-archive"
CSRF_MEDIA_UNARCHIVE = "media-unarchive"
CSRF_MEDIA_TRASH = "media-trash"
CSRF_MEDIA_RESTORE = "media-restore"
CSRF_MEDIA_EMPTY_TRASH = "media-empty-trash"


_CSRF_SECRET_FILENAME = ".csrf_secret"


def load_or_create_csrf_secret(media_root: Path) -> str:
    """Load a persistent CSRF secret from the data directory, generating one
    on first run.

    Publication is atomic: the secret is written to a PID-specific temp file,
    then hard-linked to the canonical path. ``os.link()`` fails with EEXIST if
    the target already exists, so exactly one concurrent starter wins and every
    other process reads the winner's fully-written secret. The operator can
    rotate by deleting the file and restarting.
    """
    secret_path = media_root / _CSRF_SECRET_FILENAME

    try:
        content = secret_path.read_text().strip()
        if len(content) >= 16:
            return content
    except (FileNotFoundError, OSError):
        pass

    # A short/empty file (corrupt or partial) blocks exclusive creation;
    # remove it so the atomic publish below can proceed.
    with contextlib.suppress(FileNotFoundError, OSError):
        secret_path.unlink()

    candidate = secrets.token_urlsafe(32)
    try:
        media_root.mkdir(parents=True, exist_ok=True)
        tmp = secret_path.with_suffix(f".{os.getpid()}.tmp")
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, (candidate + "\n").encode())
        finally:
            os.close(fd)
        try:
            os.link(str(tmp), str(secret_path))
            logger.info("generated persistent CSRF secret at %s", secret_path)
        except (FileExistsError, OSError):
            pass
        with contextlib.suppress(OSError):
            os.unlink(str(tmp))
    except OSError:
        logger.warning(
            "could not persist CSRF secret to %s; using an in-memory secret "
            "(open forms will break on restart)",
            secret_path,
            exc_info=True,
        )
        return candidate

    # Canonical read: all processes converge on the file's content.
    try:
        content = secret_path.read_text().strip()
        if len(content) >= 16:
            return content
    except OSError:
        pass

    return candidate


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
