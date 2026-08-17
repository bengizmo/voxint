#!/usr/bin/env python3
"""Canonical driver for the docker-free **native** install + usage E2E lane.

Companion to ``tools/e2e_browser_lifecycle.py``, but deliberately NARROWER: the
native launcher ``scripts/native/voxint-native.sh`` already owns the whole
lifecycle (setup/up/down/backup/restore/doctor), so this tool only adds the two
pieces the launcher cannot: an **HTTP driver** (onboard → submit → poll a real run
through the running api/worker) and a **read-back DB verifier** that proves the
launchd-supervised pipeline really did all six stages with real TitaNet
embeddings. The ``voxint-native-e2e`` skill drives the launcher subcommands via
Bash and calls into here for the HTTP/DB steps.

Three properties make this lane different from the browser one:

* **Live-DB safety by construction.** It runs against the launcher's *live*
  ``voxint`` database — not a disposable ``voxint_e2e``. So this tool has **no
  DDL/drop/migrate path at all** (the browser tool's ``_reset_schema_and_migrate``
  / ``_drop_database`` machinery is simply absent) and ``verify`` is SELECT-only.
  It must NOT reuse ``assert_disposable_db`` — that guard would *reject* the
  ``voxint`` name.
* **Secret hygiene.** ``state.env`` holds *generated* secrets (``DB_PASSWORD``,
  ``CSRF_SECRET``). Every subcommand reads ``state.env`` itself and composes the
  DSN / mints CSRF tokens INTERNALLY — never on argv, never echoed. Only
  non-secret resolved values (base URL, ports, api user) are printed.
* **Anti-flake.** DB assertions are shape / presence / identity only (``> 0``,
  space id, dim), never exact transcript text or exact counts — a real Metal run
  is not bit-deterministic. Distinct ``SMOKE/ONBOARD/SUBMIT/POLL/VERIFY`` prefixes
  name the boundary that broke.

Subcommands (run under ``uv`` — this imports ``voxint``):

    uv run python tools/native_e2e_lifecycle.py env
    uv run python tools/native_e2e_lifecycle.py smoke
    uv run python tools/native_e2e_lifecycle.py onboard
    uv run python tools/native_e2e_lifecycle.py submit  --file media/diarize-3speaker.wav
    uv run python tools/native_e2e_lifecycle.py poll     --run-id <uuid>
    uv run python tools/native_e2e_lifecycle.py verify   --run-id <uuid>
    uv run python tools/native_e2e_lifecycle.py drive    --file media/diarize-3speaker.wav

Cleanup is the launcher's job (``down``) and the install is throwaway
(``rm -rf ~/.voxint-native``); this tool never mutates the database.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from voxint.api.csrf import CSRF_SETUP, CSRF_SUBMIT, mint_csrf_token
from voxint.db.models import (
    EMBEDDING_DIM,
    DiarizationTurn,
    PipelineRun,
    RunStatus,
    Speaker,
    SpeakerEmbedding,
    Stage,
    StageRun,
    StageStatus,
    TranscriptSegment,
)

DEFAULT_STATE_FILE = Path.home() / ".voxint-native" / "state.env"
DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "voxint"
    / "api"
    / "static"
    / "app"
    / ".vite"
    / "manifest.json"
)
# The six keys voxint-native.sh's write_state_env persists (ports + secrets).
REQUIRED_STATE_KEYS = (
    "PG_PORT",
    "REDIS_PORT",
    "API_PORT",
    "DB_PASSWORD",
    "VOXINT_PASSWORD",
    "CSRF_SECRET",
)
# Not persisted in state.env — recomputed from the launcher's defaults.
DEFAULT_DB_USER = "voxint"
DEFAULT_DB_NAME = "voxint"
# The launcher sets only VOXINT_PASSWORD; the api user falls back to the app default.
DEFAULT_API_USER = "admin"
LOOPBACK = "127.0.0.1"
STATIC_APP_PREFIX = "/static/app/"
TITANET_SPACE = "titanet-large-v1"
POLL_INTERVAL_DEFAULT = 5.0
POLL_TIMEOUT_DEFAULT = 600.0

_KEY_VALUE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_RUN_PATH_RE = re.compile(r"/runs/([0-9a-fA-F-]{36})")


def fail(msg: str) -> None:
    print(f"NATIVE LIFECYCLE FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


class LaneError(Exception):
    """A boundary failure in one of the HTTP steps; the CLI maps it to ``fail``."""


# --------------------------------------------------------------------------- #
# Pure helpers (no I/O) — unit-tested for parity with the launcher.
# --------------------------------------------------------------------------- #
def _clean_value(raw: str) -> str:
    """Mirror voxint-native.sh ``env_value_from_file`` (lines 234-249): strip ONE
    trailing CR, then surrounding blanks (space/tab only), then ONE matched pair
    of surrounding quotes. Order matters so ``  "x"  `` → ``x``."""
    if raw.endswith("\r"):
        raw = raw[:-1]
    raw = raw.strip(" \t")
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ("'", '"'):
        raw = raw[1:-1]
    return raw


def parse_state_env(text: str) -> dict[str, str]:
    """Parse ``state.env`` with the launcher's dotenv semantics.

    ``KEY=value`` lines only (comments and blanks ignored); **last assignment
    wins** (dict insertion order over the file); values cleaned by ``_clean_value``.
    A value containing ``=`` keeps everything after the first ``=``.
    """
    values: dict[str, str] = {}
    for line in text.splitlines():
        match = _KEY_VALUE_RE.match(line)
        if match:
            values[match.group(1)] = _clean_value(match.group(2))
    return values


@dataclass(frozen=True)
class NativeConfig:
    """Resolved, non-secret-composing view of a running native stack.

    Ports + secrets come from ``state.env``; ``db_user``/``db_name``/``api_user``
    are the launcher's non-persisted defaults. The credentialed DSN and the CSRF
    secret are composed on demand and never printed.
    """

    pg_port: str
    redis_port: str
    api_port: str
    db_password: str
    voxint_password: str
    csrf_secret: str
    db_user: str = DEFAULT_DB_USER
    db_name: str = DEFAULT_DB_NAME
    api_user: str = DEFAULT_API_USER

    @classmethod
    def from_state_text(cls, text: str) -> NativeConfig:
        values = parse_state_env(text)
        missing = [key for key in REQUIRED_STATE_KEYS if not values.get(key)]
        if missing:
            raise ValueError(
                f"state.env is missing required key(s): {', '.join(missing)} "
                "(has 'voxint-native.sh setup' run?)"
            )
        return cls(
            pg_port=values["PG_PORT"],
            redis_port=values["REDIS_PORT"],
            api_port=values["API_PORT"],
            db_password=values["DB_PASSWORD"],
            voxint_password=values["VOXINT_PASSWORD"],
            csrf_secret=values["CSRF_SECRET"],
        )

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.db_user}:{self.db_password}"
            f"@{LOOPBACK}:{self.pg_port}/{self.db_name}"
        )

    @property
    def redis_url(self) -> str:
        return f"redis://{LOOPBACK}:{self.redis_port}/0"

    @property
    def base_url(self) -> str:
        return f"http://{LOOPBACK}:{self.api_port}"

    @property
    def auth(self) -> tuple[str, str]:
        return (self.api_user, self.voxint_password)


def manifest_bundles(manifest: dict[str, Any]) -> list[str]:
    """Every hashed emitted asset (``file`` + any ``css`` entries) as a served
    ``/static/app/…`` URL. Sorted + de-duplicated. Raises on an empty/invalid
    manifest so a missing build fails the smoke gate loudly."""
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    urls: list[str] = []
    seen: set[str] = set()
    for record in manifest.values():
        if not isinstance(record, dict):
            continue
        rels: list[str] = []
        file = record.get("file")
        if isinstance(file, str):
            rels.append(file)
        css = record.get("css")
        if isinstance(css, list):
            rels.extend(item for item in css if isinstance(item, str))
        for rel in rels:
            url = STATIC_APP_PREFIX + rel
            if url not in seen:
                seen.add(url)
                urls.append(url)
    if not urls:
        raise ValueError("no bundles found in the Vite manifest (frontend not built?)")
    return sorted(urls)


def entry_url(manifest: dict[str, Any], entry_name: str) -> str | None:
    """The served URL for a logical entry (``main``, ``tailwind``), keyed by the
    manifest source path's stem — mirrors ``app._load_asset_manifest``."""
    for src, record in manifest.items():
        if isinstance(record, dict) and Path(src).stem == entry_name:
            file = record.get("file")
            if isinstance(file, str):
                return STATIC_APP_PREFIX + file
    return None


def run_id_from_location(location: str) -> str | None:
    """Extract the run uuid from a ``Location: /runs/{uuid}[?...]`` header."""
    match = _RUN_PATH_RE.search(urlsplit(location).path)
    if not match:
        return None
    try:
        return str(uuid.UUID(match.group(1)))
    except ValueError:
        return None


def check_run_invariants(session: Session, run_id: uuid.UUID) -> list[str]:
    """Read-back verifier: the durable state a real native run must leave behind.

    Pure (session only, no exit) so it is unit/integration-testable and the CLI
    owns the fail-closed exit. An empty list means every invariant held. Assertions
    are shape/presence/identity only — never exact transcript text or counts.
    """
    problems: list[str] = []

    run = session.get(PipelineRun, run_id)
    if run is None:
        return [f"run {run_id}: not found in pipeline_runs"]
    if run.status != RunStatus.COMPLETED.value:
        problems.append(
            f"run {run_id}: status={run.status!r}, expected {RunStatus.COMPLETED.value!r}"
        )

    # (2) Every one of the six stages reached a completed attempt (a failed earlier
    # attempt is allowed — assert per-stage, not per-row).
    completed_stages = set(
        session.execute(
            select(StageRun.stage).where(
                StageRun.pipeline_run_id == run_id,
                StageRun.status == StageStatus.COMPLETED.value,
            )
        ).scalars()
    )
    for stage in Stage:
        if stage.value not in completed_stages:
            problems.append(f"stage_runs: stage {stage.value!r} has no completed attempt")

    # (3) ASR left non-empty transcript text.
    n_text = session.execute(
        select(func.count())
        .select_from(TranscriptSegment)
        .where(
            TranscriptSegment.pipeline_run_id == run_id,
            func.length(func.trim(TranscriptSegment.raw_text)) > 0,
        )
    ).scalar_one()
    if n_text == 0:
        problems.append(
            "transcript_segments: no rows with non-empty raw_text (ASR produced nothing)"
        )

    # (4) Real TitaNet embeddings on the diarization turns.
    turns = list(
        session.execute(
            select(DiarizationTurn).where(DiarizationTurn.pipeline_run_id == run_id)
        ).scalars()
    )
    if not turns:
        problems.append("diarization_turns: no rows for run (diarization produced nothing)")
    else:
        embedded = [turn for turn in turns if turn.embedding is not None]
        if not embedded:
            problems.append("diarization_turns: no embedded turn (all skipped/unembedded)")
        for turn in embedded:
            if turn.embedding_space != TITANET_SPACE:
                problems.append(
                    f"diarization_turn {turn.turn_index}: embedding_space="
                    f"{turn.embedding_space!r}, expected {TITANET_SPACE!r}"
                )
            dim = len(turn.embedding)
            if dim != EMBEDDING_DIM:
                problems.append(
                    f"diarization_turn {turn.turn_index}: embedding dim {dim}, "
                    f"expected {EMBEDDING_DIM}"
                )

    # (5) No operator-enrollment artifacts on an automated, fresh-install run.
    n_speakers = session.execute(select(func.count()).select_from(Speaker)).scalar_one()
    if n_speakers:
        problems.append(
            f"speakers: {n_speakers} rows on a fresh install "
            "(enrollment centroids are not pipeline output)"
        )
    n_speaker_embeddings = session.execute(
        select(func.count()).select_from(SpeakerEmbedding)
    ).scalar_one()
    if n_speaker_embeddings:
        problems.append(
            f"speaker_embeddings: {n_speaker_embeddings} rows on a fresh install "
            "(enrollment centroids are not pipeline output)"
        )

    return problems


# --------------------------------------------------------------------------- #
# HTTP step helpers (raise LaneError; cmd_* + drive map to fail).
# --------------------------------------------------------------------------- #
def _onboard(client: httpx.Client, cfg: NativeConfig) -> None:
    token = mint_csrf_token(cfg.csrf_secret, CSRF_SETUP)
    response = client.post("/setup/finish", auth=cfg.auth, data={"csrf_token": token})
    # 303 on success; complete_onboarding is idempotent so an already-onboarded
    # install returns 303 again (never a 4xx). Redirects are not followed.
    if response.status_code != 303:
        raise LaneError(
            f"ONBOARD: /setup/finish returned {response.status_code}, expected 303 "
            f"(body: {response.text[:200]!r})"
        )


def _submit(client: httpx.Client, cfg: NativeConfig, media_bytes: bytes, name: str) -> str:
    token = mint_csrf_token(cfg.csrf_secret, CSRF_SUBMIT)
    response = client.post(
        "/submit",
        auth=cfg.auth,
        files={"file": (name, media_bytes, "application/octet-stream")},
        data={"submission_id": uuid.uuid4().hex, "csrf_token": token},
    )
    if response.status_code != 303:
        raise LaneError(
            f"SUBMIT: /submit returned {response.status_code}, expected 303 "
            f"(body: {response.text[:300]!r})"
        )
    location = response.headers.get("location", "")
    if not location:
        raise LaneError("SUBMIT: 303 without a Location header")
    if "enqueue=deferred" in location:
        raise LaneError(
            f"SUBMIT: run enqueue deferred ({location}) -- the Celery broker/worker is "
            "not up; the run would never progress"
        )
    run_id = run_id_from_location(location)
    if run_id is None:
        raise LaneError(f"SUBMIT: could not parse a run id from Location {location!r}")
    return run_id


def _poll(
    client: httpx.Client, cfg: NativeConfig, run_id: str, interval: float, timeout: float
) -> None:
    deadline = time.monotonic() + timeout
    last: str | None = None
    while True:
        try:
            response = client.get(f"/runs/{run_id}/export.json", auth=cfg.auth)
        except httpx.HTTPError as exc:
            raise LaneError(f"POLL: request to export.json failed: {exc}") from exc
        if response.status_code != 200:
            raise LaneError(f"POLL: export.json returned {response.status_code}, expected 200")
        last = (response.json().get("run") or {}).get("status")
        if last == RunStatus.COMPLETED.value:
            return
        if last in (RunStatus.FAILED.value, RunStatus.CANCELLED.value):
            raise LaneError(f"POLL: run reached terminal {last!r} (expected completed)")
        if time.monotonic() >= deadline:
            raise LaneError(f"POLL: timed out after {timeout:.0f}s (last status {last!r})")
        time.sleep(interval)


def _read_media(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise LaneError(f"SUBMIT: cannot read media file {path}: {exc}") from exc


def _validate_uuid(value: str) -> str:
    try:
        return str(uuid.UUID(value))
    except ValueError:
        fail(f"--run-id is not a valid UUID: {value!r}")
        raise  # unreachable; keeps type checkers happy


def _load_config(args: argparse.Namespace) -> NativeConfig:
    state_file = Path(args.state_file)
    try:
        text = state_file.read_text()
    except OSError as exc:
        fail(f"cannot read state file {state_file}: {exc} (has 'voxint-native.sh setup' run?)")
        raise  # unreachable
    try:
        return NativeConfig.from_state_text(text)
    except ValueError as exc:
        fail(str(exc))
        raise  # unreachable


# --------------------------------------------------------------------------- #
# Subcommands.
# --------------------------------------------------------------------------- #
def cmd_env(args: argparse.Namespace) -> None:
    cfg = _load_config(args)
    # Non-secret resolved values only — the skill captures BASE_URL for healthz.
    print(f"BASE_URL={cfg.base_url}")
    print(f"API_PORT={cfg.api_port}")
    print(f"API_USER={cfg.api_user}")
    print("ENV PASS")


def cmd_smoke(args: argparse.Namespace) -> None:
    cfg = _load_config(args)
    manifest_path = Path(args.manifest)
    try:
        manifest = json.loads(manifest_path.read_text())
    except OSError as exc:
        fail(f"SMOKE: cannot read manifest {manifest_path}: {exc} (frontend not staged?)")
    except json.JSONDecodeError as exc:
        fail(f"SMOKE: manifest {manifest_path} is not valid JSON: {exc}")
    try:
        bundles = manifest_bundles(manifest)
    except ValueError as exc:
        fail(f"SMOKE: {exc}")

    with httpx.Client(base_url=cfg.base_url, timeout=15.0) as client:
        health = client.get("/healthz")
        if health.status_code != 200:
            fail(f"SMOKE: /healthz returned {health.status_code}, expected 200")

        # /setup is auth-gated but NOT onboarding-gated, and it extends base.html
        # (which embeds the main loader + tailwind), so it catches template↔manifest
        # drift without needing a run or the model tier.
        setup = client.get("/setup", auth=cfg.auth)
        if setup.status_code != 200:
            fail(f"SMOKE: /setup returned {setup.status_code}, expected 200")
        for name in ("main", "tailwind"):
            url = entry_url(manifest, name)
            if url and url not in setup.text:
                fail(f"SMOKE: /setup HTML does not reference the {name} bundle {url} (drift)")

        for url in bundles:
            asset = client.get(url, auth=cfg.auth)
            if asset.status_code != 200:
                fail(f"SMOKE: bundle {url} returned {asset.status_code}, expected 200")
            if not asset.content:
                fail(f"SMOKE: bundle {url} served empty content")

    print(f"ok: /healthz + /setup wiring + {len(bundles)} bundles all 200")
    print("SMOKE PASS")


def cmd_onboard(args: argparse.Namespace) -> None:
    cfg = _load_config(args)
    try:
        with httpx.Client(base_url=cfg.base_url, timeout=30.0) as client:
            _onboard(client, cfg)
    except LaneError as exc:
        fail(str(exc))
    print("ONBOARD PASS")


def cmd_submit(args: argparse.Namespace) -> None:
    cfg = _load_config(args)
    media = Path(args.file)
    try:
        with httpx.Client(base_url=cfg.base_url, timeout=60.0) as client:
            run_id = _submit(client, cfg, _read_media(media), media.name)
    except LaneError as exc:
        fail(str(exc))
    print(f"RUN_ID={run_id}")
    print("SUBMIT PASS")


def cmd_poll(args: argparse.Namespace) -> None:
    cfg = _load_config(args)
    run_id = _validate_uuid(args.run_id)
    try:
        with httpx.Client(base_url=cfg.base_url, timeout=30.0) as client:
            _poll(client, cfg, run_id, args.interval, args.timeout)
    except LaneError as exc:
        fail(str(exc))
    print(f"ok: run {run_id} reached '{RunStatus.COMPLETED.value}'")
    print("POLL PASS")


def cmd_verify(args: argparse.Namespace) -> None:
    cfg = _load_config(args)
    run_id = _validate_uuid(args.run_id)
    engine = create_engine(cfg.database_url)  # SELECT-only; no DDL path exists here.
    factory = sessionmaker(engine, expire_on_commit=False)
    try:
        with factory() as session:
            problems = check_run_invariants(session, uuid.UUID(run_id))
    finally:
        engine.dispose()
    if problems:
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        fail(f"VERIFY: {len(problems)} invariant problem(s) for run {run_id}")
    print(f"ok: run {run_id} satisfies every native-run invariant")
    print("VERIFY PASS")


def cmd_drive(args: argparse.Namespace) -> None:
    cfg = _load_config(args)
    media = Path(args.file)
    try:
        with httpx.Client(base_url=cfg.base_url, timeout=60.0) as client:
            _onboard(client, cfg)
            run_id = _submit(client, cfg, _read_media(media), media.name)
            print(f"RUN_ID={run_id}")
            _poll(client, cfg, run_id, args.interval, args.timeout)
    except LaneError as exc:
        fail(str(exc))
    print("DRIVE PASS")


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_state(p: argparse.ArgumentParser) -> None:
        p.add_argument("--state-file", default=str(DEFAULT_STATE_FILE))

    def add_poll(p: argparse.ArgumentParser) -> None:
        p.add_argument("--interval", type=float, default=POLL_INTERVAL_DEFAULT)
        p.add_argument("--timeout", type=float, default=POLL_TIMEOUT_DEFAULT)

    p_env = sub.add_parser("env", help="print non-secret resolved BASE_URL/API_PORT/API_USER")
    add_state(p_env)
    p_env.set_defaults(func=cmd_env)

    p_smoke = sub.add_parser("smoke", help="healthz + /setup wiring + every manifest bundle 200")
    add_state(p_smoke)
    p_smoke.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p_smoke.set_defaults(func=cmd_smoke)

    p_onboard = sub.add_parser("onboard", help="POST /setup/finish (idempotent)")
    add_state(p_onboard)
    p_onboard.set_defaults(func=cmd_onboard)

    p_submit = sub.add_parser("submit", help="POST /submit a media file; prints RUN_ID=")
    add_state(p_submit)
    p_submit.add_argument("--file", required=True, help="media file to upload")
    p_submit.set_defaults(func=cmd_submit)

    p_poll = sub.add_parser("poll", help="poll export.json until the run is completed")
    add_state(p_poll)
    add_poll(p_poll)
    p_poll.add_argument("--run-id", required=True)
    p_poll.set_defaults(func=cmd_poll)

    p_verify = sub.add_parser("verify", help="SELECT-only durable-state invariant check")
    add_state(p_verify)
    p_verify.add_argument("--run-id", required=True)
    p_verify.set_defaults(func=cmd_verify)

    p_drive = sub.add_parser("drive", help="onboard -> submit -> poll in one call")
    add_state(p_drive)
    add_poll(p_drive)
    p_drive.add_argument("--file", required=True, help="media file to upload")
    p_drive.set_defaults(func=cmd_drive)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
