"""yt-dlp download boundary for the ACQUIRE stage.

This is the *only* place Voxint shells out to yt-dlp. It is a pure "download
into a temp dir" unit: the ACQUIRE stage owns the DB writes and the atomic
publish to ``source_path`` (see ``pipeline/stages/acquire.py``), so this module
never touches the database and only ever produces files under a caller-provided
directory. Keeping the two apart is what lets the stage inject a FAKE downloader
in tests and exercise every publish/idempotency invariant with no network.

yt-dlp is invoked as a **subprocess** (an argument vector, never a shell), not
imported, so importing this module stays stdlib-only and cheap — the read path
can import the stage-context graph without dragging in the yt-dlp library.

The subprocess runs in its **own process group** (``start_new_session=True``)
so a hung download that spawned children (ffmpeg muxers, fragment fetchers) is
killed as a group on wall-clock timeout — ``--socket-timeout`` alone cannot cap
a download that keeps trickling bytes just under the socket deadline.

All terminal failures raise :class:`AcquisitionError`, which is **not** a
``ServiceError``, so the worker's retry classifier treats it as deterministic:
the run parks FAILED @ acquire for a manual Requeue rather than auto-retrying a
bot-block forever. Every message this module builds from the subprocess is
passed through :func:`voxint.media.redaction.redact` at the raise site, so a
signed source URL (or, once slice 6g wires them in, a proxy string / cookie
path) that yt-dlp echoes into stderr is scrubbed before it can reach the error
message, the ledger, or the logs — the exception is born clean.
"""

import contextlib
import ctypes
import os
import signal
import subprocess
from collections.abc import Callable
from pathlib import Path

from voxint.media.redaction import redact

# A downloader materializes the media for ``url`` into ``dest_dir`` (an existing,
# attempt-unique directory) and returns nothing; the stage discovers the produced
# file, enforces the authoritative size cap, hashes it, and publishes it. It is
# handed ``max_bytes`` so it can pass an early ``--max-filesize`` hint, but the
# stage re-checks the produced file, so the hint is advisory. Raises
# :class:`AcquisitionError` on any terminal failure.
Downloader = Callable[[str, Path, int], None]

# yt-dlp stderr can run long; keep only the tail in the surfaced error.
_STDERR_LIMIT = 2000
# Grace for a SIGTERM'd process group to exit before we SIGKILL it.
_KILL_GRACE_SECONDS = 5.0

# PR_SET_PDEATHSIG (Linux): ask the kernel to SIGKILL the download if THIS worker
# process dies (an OOM-kill or crash) while a download is in flight, so a killed
# worker cannot orphan a yt-dlp process that keeps downloading into an abandoned
# temp dir. The parent-death signal survives execve for a non-privileged binary,
# so it stays in effect on the yt-dlp process itself. libc is loaded once here,
# before any fork, so the post-fork hook does no dlopen in that fragile context.
_PR_SET_PDEATHSIG = 1
try:
    _libc: "ctypes.CDLL | None" = ctypes.CDLL("libc.so.6", use_errno=True)
except OSError:  # pragma: no cover - non-glibc platform; the group-kill still applies
    _libc = None


def _isolate_child() -> None:  # pragma: no cover - runs post-fork, pre-exec in the child
    """Child-side setup between fork and exec: lead a new session/process group so
    a timeout can group-kill the whole download tree, and request SIGKILL-on-
    parent-death so a worker crash can't leave the download orphaned. Deliberately
    minimal — no Python-level locks, no dlopen (libc is cached at import)."""
    os.setsid()
    if _libc is not None:
        _libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL)


class AcquisitionError(Exception):
    """A URL acquisition failed terminally (download error, timeout, or the
    output was missing/over-size/not exactly one file).

    Deterministic for a given input, so the pipeline's failure lane owns it —
    NOT the retry path. It deliberately does not subclass ``ServiceError`` so
    the worker never auto-retries it; the operator Requeues instead. The
    subprocess wrapper raises it with a redacted message (see the module
    docstring), so the URL/stderr it carries is already scrubbed.
    """


def _kill_process_group(
    proc: "subprocess.Popen[str]", *, grace_seconds: float = _KILL_GRACE_SECONDS
) -> None:
    """SIGTERM then SIGKILL the child's whole process group on timeout.

    The child leads its own session/group (see :func:`_isolate_child`), so the
    process-group id equals its pid — signal ``proc.pid`` directly rather than
    calling ``os.getpgid``, which FAILS once the group *leader* exits even while
    descendants (a fragment fetcher, a muxer) are still alive, silently leaving
    them running. Escalate to SIGKILL after the grace period regardless of whether
    the leader itself has reaped, so a descendant that ignored SIGTERM cannot
    outlive us. A failing teardown signal is benign and must never escape to mask
    the redacted ``AcquisitionError``: ``ProcessLookupError`` (ESRCH, an
    already-empty group) on every platform, and on macOS/BSD also
    ``PermissionError`` (EPERM) — which ``killpg`` returns instead of ESRCH once
    the leader has been reaped and the survivor is a zombie reparented to launchd.
    Both are suppressed on both signals.
    """
    pgid = proc.pid
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, signal.SIGTERM)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=grace_seconds)
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, signal.SIGKILL)


def run_download_command(
    argv: list[str], *, timeout_seconds: float, extra_secrets: "tuple[str, ...]" = ()
) -> None:
    """Run a download argument vector under a hard wall-clock timeout.

    Spawns ``argv`` in its own process group, waits up to ``timeout_seconds``,
    and on expiry terminates the entire group (``terminate → kill``) so a stalled
    download and any children it spawned cannot outlive the lease. A non-zero
    exit or a failure to even launch the binary is a terminal
    :class:`AcquisitionError`. The output directory is inspected by the caller,
    not here.

    ``extra_secrets`` are caller-known secret literals (the configured
    ``--proxy`` value and ``--cookies`` path) scrubbed verbatim from any surfaced
    message, so a proxy string or cookie path yt-dlp echoes into stderr as *prose*
    — with neither an http(s) scheme nor our flag in front of it, which the
    structural :func:`redact` cannot otherwise catch — is removed too.
    """
    try:
        proc = subprocess.Popen(
            argv,
            # stdout is written to disk via -o; we never read it, so discard it
            # rather than buffer an unbounded stream in memory.
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            # New session/group + SIGKILL-on-worker-death; see _isolate_child.
            preexec_fn=_isolate_child,
        )
    except OSError as exc:  # missing binary, permission denied
        # argv[0] is the yt-dlp binary and the OSError names only it (never the
        # full argv), so chaining it leaks nothing; redact the message anyway so
        # every message this module builds from a subprocess is scrubbed at one
        # boundary. `from exc` is safe here precisely because the cause names
        # argv[0], unlike the timeout cause below which carries the whole argv.
        raise AcquisitionError(
            redact(f"failed to execute {argv[0]}: {exc}", extra_secrets=extra_secrets)
        ) from exc
    try:
        _, stderr = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        # Reap the (now-terminated) group so no zombie/pipe is left behind.
        with contextlib.suppress(Exception):
            proc.communicate(timeout=_KILL_GRACE_SECONDS)
        # `from None`, NOT `from exc`: subprocess.TimeoutExpired.__str__ embeds
        # the ENTIRE argv — including the source URL as its last token — so
        # chaining it would leak the signed URL (and, once 6g wires them in, the
        # proxy/cookie args) into any rendered traceback or Celery task log, even
        # though this message is clean. Suppress the cause so no sink sees argv.
        raise AcquisitionError(
            f"acquisition exceeded the {timeout_seconds:g}s wall-clock timeout"
        ) from None
    if proc.returncode != 0:
        # Redact BEFORE slicing: yt-dlp echoes the full source URL (signed query
        # params, embedded creds) into stderr. Slicing first could keep a
        # schemeless "?token=..." tail the redactor no longer recognises, so the
        # whole blob is scrubbed and only then trimmed to the tail limit.
        tail = redact(stderr or "", extra_secrets=extra_secrets)[-_STDERR_LIMIT:]
        raise AcquisitionError(
            f"download command failed (exit {proc.returncode}): {tail}"
        )


def build_ytdlp_downloader(
    *,
    timeout_seconds: float,
    socket_timeout_seconds: float,
    ytdlp_bin: str = "yt-dlp",
    proxy: str = "",
    cookies_file: "Path | None" = None,
) -> Downloader:
    """Build the production yt-dlp downloader bound to the given timeouts.

    The returned callable downloads ``bestaudio/best`` (PREPARE re-encodes to
    16 kHz mono WAV, so no extraction/transcode here) into ``dest_dir`` under a
    hard wall-clock timeout, keeping yt-dlp's own retries small and explicit.
    ``--no-playlist``/``--max-downloads 1`` keep a single item; the *authoritative*
    single-output guard is the stage's "exactly one file" check.

    yt-dlp lockdown (verified against yt-dlp 2026.07.04):
    - ``--no-config`` — never honour an ambient user config in a worker.
    - ``--no-plugin-dirs`` — clear ALL plugin directories, defaults included, so a
      dropped-in local/remote plugin cannot execute.
    - ``--no-exec`` — neutralise any ``--exec`` post-processor command (we never
      set one; belt-and-suspenders against a future/injected one).
    - ``file:`` URLs need no flag: yt-dlp disables them by default (we never pass
      ``--enable-file-urls``), so a ``file:`` input/redirect is already refused.

    Egress config: ``--proxy`` is passed ALWAYS — with the configured ``proxy`` or
    an empty string, which yt-dlp reads as an explicit *direct* connection. That is
    deliberate: omitting ``--proxy`` lets yt-dlp fall back to an ambient
    ``HTTP(S)_PROXY`` / ``ALL_PROXY`` in the worker env, silently rerouting egress
    (and its DNS) around the intended path, so egress is pinned to exactly what is
    configured. ``cookies_file`` → ``--cookies`` is appended only when set. A
    non-empty proxy and the cookies path are credentials: their literal values go
    to ``run_download_command`` as ``extra_secrets`` and are scrubbed verbatim from
    any surfaced error, even in prose yt-dlp emits.

    Residual (NOT closed by a userland flag, documented in docs/architecture.md):
    yt-dlp re-resolves the host independently and its generic extractor follows
    redirects, so a redirect / extractor-constructed URL to a private address is
    beyond this process's reach — that needs network policy.
    """

    def download(url: str, dest_dir: Path, max_bytes: int) -> None:
        argv = [
            ytdlp_bin,
            "--no-config",
            "--no-plugin-dirs",
            "--no-exec",
            "--no-playlist",
            "--max-downloads",
            "1",
            "--format",
            "bestaudio/best",
            "--max-filesize",
            str(max_bytes),
            "--socket-timeout",
            str(socket_timeout_seconds),
            "--retries",
            "2",
            "--fragment-retries",
            "2",
            "--extractor-retries",
            "1",
        ]
        # Egress options ride among the flags, before the ``--`` URL terminator.
        # --proxy is ALWAYS passed (empty string = explicit direct) so an ambient
        # HTTP(S)_PROXY in the worker env can never silently reroute egress; only a
        # non-empty value is a credential to scrub.
        secret_values: list[str] = []
        argv += ["--proxy", proxy]
        if proxy:
            secret_values.append(proxy)
        if cookies_file is not None:
            cookies_path = str(cookies_file)
            argv += ["--cookies", cookies_path]
            secret_values.append(cookies_path)
        argv += [
            "--no-progress",
            "--output",
            str(dest_dir / "source.%(ext)s"),
            "--",
            url,
        ]
        run_download_command(
            argv, timeout_seconds=timeout_seconds, extra_secrets=tuple(secret_values)
        )

    return download
