"""Read-only preflight checks behind ``voxint doctor``.

Each check answers one question — is this dependency reachable? — and returns a
:class:`CheckResult` rather than raising, so the command can report every
dependency's state in one pass and pick a single exit code at the end.

Two deliberate rules:

* **Hard vs advisory.** Postgres, Redis, and the three model services are hard —
  the pipeline cannot run without them, so a failure sets exit 1. The Hugging
  Face token and the LLM endpoint are advisory: the default install needs no HF
  token (weights are vendored) and enhancement is best-effort, so their state is
  reported but never changes the exit code.
* **No secrets in output.** ``database_url``/``redis_url``/``llm_api_key`` are
  credentials. A detail string therefore never echoes a URL, a token, or a raw
  exception (a DSN can ride inside ``str(exc)``) — only the exception *type* and
  a static phrase. Callers print these details verbatim.
"""

import contextlib
import os
from dataclasses import dataclass
from typing import Literal

import httpx
from sqlalchemy import Engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from voxint.api.health_probe import probe_services
from voxint.app_settings import (
    get_app_settings,
    resolve_effective_llm_api_key,
    resolve_effective_llm_endpoint,
)
from voxint.config import Settings

HF_WHOAMI_URL = "https://huggingface.co/api/whoami-v2"


@dataclass(frozen=True)
class CheckResult:
    """One dependency's outcome. ``hard`` failures drive a nonzero exit code."""

    name: str
    ok: bool
    hard: bool
    detail: str  # plain-language; never a credential, URL, or raw exception text


def _safe(exc: Exception) -> str:
    """The exception's type name only — ``str(exc)`` can embed a DSN/password."""
    return type(exc).__name__


def check_database(engine: Engine) -> CheckResult:
    """Hard: a bare ``SELECT 1`` proves the connection and credentials work."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:  # any driver/transport failure is a down dependency
        return CheckResult("postgres", False, True, f"connection failed ({_safe(exc)})")
    return CheckResult("postgres", True, True, "connected")


def check_redis(url: str, *, client: object | None = None) -> CheckResult:
    """Hard: the broker/result backend must answer PING.

    ``client`` is injectable for tests (any object with ``ping()``); otherwise a
    short-timeout client is built from ``url`` and closed here.
    """
    import redis

    # Construction is inside the try: redis.from_url raises ValueError on a
    # malformed DSN, and that must normalize to a hard FAIL, not escape as a
    # traceback (which would also violate the no-raw-exceptions rule).
    r: object | None = None
    try:
        r = client if client is not None else redis.Redis.from_url(
            url, socket_connect_timeout=3.0, socket_timeout=3.0
        )
        r.ping()  # type: ignore[attr-defined]
    except Exception as exc:
        return CheckResult("redis", False, True, f"unreachable ({_safe(exc)})")
    finally:
        if client is None and r is not None:
            with contextlib.suppress(Exception):
                r.close()  # type: ignore[attr-defined]
    return CheckResult("redis", True, True, "reachable")


def check_models(settings: Settings, *, client: httpx.Client | None = None) -> list[CheckResult]:
    """Hard: probe each model service's ``/healthz`` (reuses the wizard's probe).

    A ready service reports its compute device (cpu/cuda/rocm) — surface it so the
    operator can confirm the GPU tier actually took.
    """
    results = []
    for h in probe_services(settings, client=client):
        detail = f"ready ({h.device})" if h.up and h.device else (h.detail)
        results.append(CheckResult(h.name, h.up, True, detail))
    return results


def check_hf_token(token: str | None, *, client: httpx.Client) -> CheckResult:
    """Advisory: validate ``HF_TOKEN`` via whoami. Absent is fine (vendored weights)."""
    if not token:
        return CheckResult(
            "hugging face token", True, False, "not set (default install needs none)"
        )
    try:
        resp = client.get(HF_WHOAMI_URL, headers={"Authorization": f"Bearer {token}"})
    except httpx.HTTPError as exc:
        return CheckResult("hugging face token", False, False, f"check failed ({_safe(exc)})")
    if resp.status_code == 200:
        try:
            body = resp.json()
        except ValueError:
            body = None
        # A valid-JSON but non-object body (a captive portal/proxy answering 200
        # with `[]` or `"ok"`) must not crash .get — guard the type explicitly.
        name = body.get("name") if isinstance(body, dict) else None
        return CheckResult(
            "hugging face token", True, False, f"valid ({name})" if name else "valid"
        )
    if resp.status_code == 401:
        return CheckResult("hugging face token", False, False, "rejected (401)")
    return CheckResult("hugging face token", False, False, f"HTTP {resp.status_code}")


def check_llm(
    *, enabled: bool, base_url: str, api_key: str, client: httpx.Client
) -> CheckResult | None:
    """Advisory: reachability of the LLM endpoint, only when enhancement is on.

    Takes the EFFECTIVE LLM configuration (issue #10): ``enabled`` is the row's
    enablement over env, and ``base_url``/``api_key`` are row-wins-over-env resolved
    by :func:`run_diagnostics` — so ``doctor`` reflects a UI-stored key/endpoint, not
    just env. Returns ``None`` when enhancement is off — an unconfigured, unused LLM
    is not a fault. Any HTTP answer (even 401/404) proves the host is reachable; only
    a transport failure is a miss. The base URL and key are never printed.
    """
    if not enabled:
        return None
    url = base_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        resp = client.get(url, headers=headers)
    except httpx.InvalidURL:
        # InvalidURL is NOT an httpx.HTTPError; a malformed llm_base_url would
        # otherwise escape this advisory check and abort the whole doctor run.
        return CheckResult("llm endpoint", False, False, "invalid url")
    except httpx.HTTPError as exc:
        return CheckResult("llm endpoint", False, False, f"unreachable ({_safe(exc)})")
    return CheckResult("llm endpoint", True, False, f"reachable (HTTP {resp.status_code})")


def run_diagnostics(
    settings: Settings,
    engine: Engine,
    *,
    hf_token: str | None = None,
    http_client: httpx.Client,
    redis_client: object | None = None,
    include_hf_token: bool = True,
) -> list[CheckResult]:
    """Run every check in dependency order and collect the results.

    ``hf_token`` defaults to reading ``HF_TOKEN`` from the environment (it is not a
    Setting); pass it explicitly in tests. The caller owns ``engine``/``http_client``.

    ``include_hf_token=False`` drops the Hugging Face check entirely — this is both a
    display choice and a network one: ``check_hf_token`` makes a live ``whoami`` GET to
    huggingface.co, so skipping it keeps the caller (the setup wizard's SERVICES step,
    issue #61) from making an external internet call it has no reason to — the default
    install runs on vendored weights and needs no HF token.
    """
    token = hf_token if hf_token is not None else os.environ.get("HF_TOKEN")
    results = [check_database(engine), check_redis(settings.redis_url, client=redis_client)]
    results.extend(check_models(settings, client=http_client))
    if include_hf_token:
        results.append(check_hf_token(token, client=http_client))
    # Resolve the effective LLM config from the app_settings row (issue #10) in a
    # short read-only session on the caller's engine: a UI-stored key/endpoint and
    # the row's enablement win over env, so doctor reports what a run would actually
    # use. Attributes are read INSIDE the session (a detached row can't lazy-load).
    # If the row can't be read — the DB is down (which check_database already
    # reports) or unmigrated — fall back to env rather than crash this advisory
    # check; that is robustness for a diagnostic, not a fallback masking a fault.
    llm_enabled = settings.llm_enabled
    base_url, api_key = settings.llm_base_url, settings.llm_api_key.strip()
    with contextlib.suppress(SQLAlchemyError), Session(engine) as session:
        row = get_app_settings(session)
        llm_enabled = row.llm_enabled if row is not None else settings.llm_enabled
        base_url, _model = resolve_effective_llm_endpoint(row, settings)
        api_key = resolve_effective_llm_api_key(row, settings)
    llm = check_llm(
        enabled=llm_enabled, base_url=base_url, api_key=api_key, client=http_client
    )
    if llm is not None:
        results.append(llm)
    return results


def exit_code(results: list[CheckResult]) -> int:
    """0 when every hard check passed; 1 when any hard dependency is down."""
    return 1 if any(r.hard and not r.ok for r in results) else 0


def check_state(result: CheckResult) -> Literal["ready", "failed", "unverified"]:
    """Collapse a result's ``(ok, hard)`` into one of three honest display states.

    Mirrors the CLI's ``ok``/``FAIL``/``warn`` tagging (``cli.py`` ``_doctor``) for the
    setup wizard (issue #61): ``ready`` (passed), ``failed`` (a hard dependency is
    down — the pipeline cannot run), or ``unverified`` (an advisory check that did not
    pass — reported honestly rather than as healthy, never a false all-good).
    """
    if result.ok:
        return "ready"
    return "failed" if result.hard else "unverified"
