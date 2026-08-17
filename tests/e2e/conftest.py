"""End-to-end fixtures: the *real* pipeline against *real* model services.

This is a maintainer-run, opt-in gate — never public CI (see
``docs/release-process.md``). Unlike ``tests/integration`` (a real DB, faked
model I/O) and ``tests/parity`` (numerics against committed references), these
tests submit audio and drive faster-whisper + pyannote + TitaNet running in
their containers, asserting the persistence invariants the whole pipeline is
supposed to leave behind.

Gate semantics (deliberately asymmetric — codex review):

* ``VOXINT_E2E`` **unset** → the whole directory is *skipped* at collection, so
  a bare ``pytest`` run stays green on any developer machine.
* ``VOXINT_E2E=1`` → an explicit E2E run. A missing prerequisite (test DB,
  model-service health, wrong device identity) is a **failure, not a skip** — an
  operator who asked for the real gate must never get a green board because the
  suite quietly skipped itself.

Prerequisites when enabled:

* ``VOXINT_TEST_DATABASE_URL`` points at a **disposable** database (its schema is
  dropped and rebuilt from the alembic chain, and every table is truncated
  between tests — never aim it at live data).
* The three model services answer ``/healthz`` with the expected identity and no
  device fallback. Host-specific bring-up (compose overlays, CPU caps, the AMD
  render gid) lives outside this repo; this file only *asserts* the contract.
* ``MEDIA_ROOT`` (via ``Settings``) is the host directory the model-service
  containers mount at ``/data/media``. Fixtures stage audio there so a container
  reading a path relative to *its* mount sees the same bytes.
"""

from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from voxint.config import Settings
from voxint.db.models import Base
from voxint.pipeline.stages.context import StageContext, build_stage_context

# Enabled by a truthy VOXINT_E2E. Deliberately treat unset / empty / "0" / "false"
# as OFF: an explicit ``VOXINT_E2E=0`` reads as "off" to any operator, and turning
# the destructive lane ON for it would be the opposite of the fail-safe intent.
E2E_ENABLED = os.environ.get("VOXINT_E2E", "").strip().lower() not in ("", "0", "false", "no")
TEST_DB_URL = os.environ.get("VOXINT_TEST_DATABASE_URL")
REPO_ROOT = Path(__file__).resolve().parents[2]
TUTORIAL_WAV = REPO_ROOT / "src" / "voxint" / "tutorial" / "assets" / "sample-3speaker.wav"

# The identity each model service must report at /healthz for the real-pipeline
# lane. ``device`` is load-bearing: a container that silently fell back to CPU
# (whisper) or CUDA (a mis-tagged image) would still "work" but stops being the
# thing we mean to gate — abort on any drift, never skip.
EXPECTED_SERVICES: dict[str, dict[str, str]] = {
    "asr_url": {"service": "whisper", "device": "rocm", "model": "large-v2"},
    "diarizer_url": {
        "service": "pyannote",
        "device": "cpu",
        "model": "pyannote/speaker-diarization-3.1",
    },
    "embedder_url": {"service": "titanet", "device": "cpu"},
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    # pytestmark in a conftest does NOT propagate to sibling modules, and this
    # hook sees the whole session — mark only this directory's items. Skip the
    # E2E lane unless it was explicitly requested; when it IS requested, the
    # fixtures below FAIL (not skip) on any missing prerequisite.
    if E2E_ENABLED:
        return
    here = Path(__file__).parent
    skip = pytest.mark.skip(reason="VOXINT_E2E not set (opt-in maintainer gate)")
    for item in items:
        if here in Path(str(item.fspath)).parents:
            item.add_marker(skip)


def _healthz(base_url: str) -> dict[str, object]:
    resp = httpx.get(f"{base_url.rstrip('/')}/healthz", timeout=10.0)
    resp.raise_for_status()
    data = resp.json()
    assert isinstance(data, dict)
    return data


def assert_service_identity(base_url: str, expected: dict[str, str]) -> None:
    """Fail hard unless the service at ``base_url`` reports the expected identity.

    Called by the ``model_services`` fixture; raises ``pytest.fail`` (a failure,
    not a skip) so an enabled E2E run cannot go green against the wrong image,
    an unloaded model, or a device fallback.
    """
    try:
        health = _healthz(base_url)
    except (httpx.HTTPError, ValueError) as exc:  # unreachable / non-JSON
        pytest.fail(
            f"VOXINT_E2E=1 but model service {expected['service']} at {base_url} "
            f"is not answering /healthz: {exc}"
        )
    if health.get("model_loaded") is not True:
        pytest.fail(f"{expected['service']} at {base_url}: model_loaded is not True: {health}")
    if health.get("contract_version") != "v1":
        pytest.fail(
            f"{expected['service']} at {base_url}: contract_version "
            f"{health.get('contract_version')!r} != 'v1': {health}"
        )
    for key, want in expected.items():
        got = health.get(key)
        if got != want:
            pytest.fail(
                f"{expected['service']} at {base_url}: {key}={got!r} != {want!r} "
                f"(device fallback or wrong image?): {health}"
            )


def _assert_disposable_db(url: str) -> None:
    """Refuse to run against anything that is not unmistakably a throwaway DB.

    The ``engine`` fixture opens with ``DROP SCHEMA public CASCADE`` — pointed at
    the live ``voxint`` database (a typo in ``VOXINT_TEST_DATABASE_URL`` is all it
    takes) that silently destroys the operator's data before a single test runs.
    Fail closed: the database name must carry an explicit disposable marker
    (``test`` or ``e2e``). The live database is named ``voxint`` and is rejected.
    """
    db_name = urlsplit(url).path.lstrip("/").lower()
    if not db_name or ("test" not in db_name and "e2e" not in db_name):
        pytest.fail(
            "VOXINT_TEST_DATABASE_URL must name a DISPOSABLE database whose name "
            f"contains 'test' or 'e2e' (its schema is dropped and rebuilt); got "
            f"{db_name!r}. Refusing to run destructive setup against a database "
            "that could be live data."
        )


@pytest.fixture(scope="session")
def settings() -> Settings:
    """Application settings for the E2E lane, pinned to the disposable test DB.

    Fails (not skips) when the test DB URL is absent or not clearly disposable: an
    explicit E2E run with no isolated database is a misconfiguration, and we must
    never silently fall back to ``Settings``' default live DSN nor drop the live
    schema.
    """
    if TEST_DB_URL is None:
        pytest.fail(
            "VOXINT_E2E=1 requires VOXINT_TEST_DATABASE_URL pointing at a "
            "disposable database (its schema is dropped and rebuilt)."
        )
    # Read the operator's REAL config (env / .env) BEFORE we override DATABASE_URL,
    # and refuse if the test URL is the live one — the sharpest guard against a
    # copy-pasted DSN, on top of the disposable-name heuristic below.
    if str(Settings().database_url) == TEST_DB_URL:
        pytest.fail(
            "VOXINT_TEST_DATABASE_URL equals the live DATABASE_URL; the E2E engine "
            "drops and rebuilds its schema. Point it at a disposable database."
        )
    _assert_disposable_db(TEST_DB_URL)
    # Settings reads DATABASE_URL from the environment over any .env value; pin it
    # to the disposable DB so the in-process pipeline never touches live data. This
    # process-wide mutation is not restored (mirrors tests/integration/conftest.py);
    # it only matters for a combined full-suite run and always points at the test DB.
    os.environ["DATABASE_URL"] = TEST_DB_URL
    return Settings()


@pytest.fixture(scope="session")
def model_services(settings: Settings) -> None:
    """Assert all three model services are up with the expected identity."""
    assert_service_identity(settings.asr_url, EXPECTED_SERVICES["asr_url"])
    assert_service_identity(settings.diarizer_url, EXPECTED_SERVICES["diarizer_url"])
    assert_service_identity(settings.embedder_url, EXPECTED_SERVICES["embedder_url"])


@pytest.fixture(scope="session")
def stage_context(settings: Settings, model_services: None) -> Iterator[StageContext]:
    """One process-cached stage context with real HTTP clients, closed at teardown.

    This mirrors the worker, which builds the transport clients once and reuses
    them across runs (``build_stage_context`` docstring), and — unlike a
    per-run context — owns the clients' connection pools so they are closed
    instead of leaking until the interpreter exits. ``llm`` stays ``None`` (the
    pipeline lane runs no enrichment).
    """
    ctx = build_stage_context(settings)
    yield ctx
    for client in (ctx.asr, ctx.diarizer, ctx.embedder):
        close = getattr(client, "close", None)
        if callable(close):
            close()


@pytest.fixture(scope="session")
def engine(settings: Settings) -> Iterator[Engine]:
    """A fresh schema on the disposable test DB, migrated by the alembic chain."""
    assert TEST_DB_URL is not None  # guarded by the settings fixture
    eng = create_engine(str(settings.database_url))
    with eng.connect() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
        conn.commit()
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "alembic"))
    command.upgrade(cfg, "head")
    yield eng
    eng.dispose()


@pytest.fixture()
def session_factory(engine: Engine) -> Iterator[sessionmaker[Session]]:
    yield sessionmaker(engine, expire_on_commit=False)
    with engine.connect() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(text(f'TRUNCATE TABLE "{table.name}" CASCADE'))
        conn.commit()


@dataclass(frozen=True)
class LLMConfig:
    """The resolved, verified LLM endpoint for the real-enrichment lane.

    ``resolved_identity`` is the concrete backend the alias routed to at gate
    time (an endpoint's ``root``/served-model field). A LiteLLM/vLLM alias can
    be silently repointed at different weights; recording what it resolved to
    turns that kind of reroute into a named signal instead of an invisible
    flake in the summary text.
    """

    base_url: str
    model: str
    resolved_identity: str


@pytest.fixture(scope="session")
def llm_config(settings: Settings) -> LLMConfig:
    """Gate + resolve the real LLM endpoint for the enrichment lane.

    Asymmetric like ``model_services``, but with an extra *unconfigured* rung —
    the LLM lane is an OPTIONAL sub-lane of the E2E gate. An operator may run the
    real-pipeline lane (Phase 1) without wiring an LLM, so:

    * LLM not enabled / no model configured → **skip** these tests (unconfigured);
      never fail — the operator did not ask for the LLM lane.
    * enabled AND configured, but the endpoint is unreachable or the alias does
      not resolve → **fail** (a configured-but-broken lane must not go green,
      exactly as a device fallback fails the model-service gate).

    Enablement is read from ``Settings`` (env / .env): the lane is "configured"
    when ``LLM_ENABLED`` and ``ENRICHMENT_RUN_ASSETS_ENABLED`` are true and a
    model alias is set. The endpoint URL / model / key live in the operator's
    environment (gitignored ``internal/``), never in this committed file.
    """
    if not (settings.llm_enabled and settings.enrichment_run_assets_enabled and settings.llm_model):
        pytest.skip(
            "LLM enrichment lane not configured — set LLM_ENABLED=true, "
            "ENRICHMENT_RUN_ASSETS_ENABLED=true, LLM_BASE_URL, LLM_MODEL, LLM_API_KEY "
            "to exercise it (optional sub-lane of the E2E gate)."
        )
    base_url = settings.llm_base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {settings.llm_api_key}"} if settings.llm_api_key else {}
    try:
        resp = httpx.get(f"{base_url}/models", headers=headers, timeout=15.0)
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, ValueError) as exc:  # unreachable / non-JSON
        pytest.fail(
            f"VOXINT_E2E=1 and the LLM lane is configured, but its endpoint "
            f"{base_url} is not answering /models: {exc}"
        )
    entries = data.get("data") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        pytest.fail(f"LLM endpoint {base_url}/models did not return a model list: {data!r:.200}")
    match = next(
        (e for e in entries if isinstance(e, dict) and e.get("id") == settings.llm_model), None
    )
    if match is None:
        available = sorted(str(e.get("id")) for e in entries if isinstance(e, dict))
        pytest.fail(
            f"LLM alias {settings.llm_model!r} does not resolve at {base_url}; "
            f"available: {available}"
        )
    resolved = str(match.get("root") or match.get("id"))
    return LLMConfig(
        base_url=settings.llm_base_url, model=settings.llm_model, resolved_identity=resolved
    )


@pytest.fixture()
def stage_media(settings: Settings) -> Iterator[Callable[[], str]]:
    """Return a factory that stages the tutorial WAV under the media root.

    Each call copies ``sample-3speaker.wav`` to a unique path *relative to*
    ``MEDIA_ROOT`` and returns that relative path (what ``MediaItem.source_path``
    stores). The model-service containers mount the same host directory at
    ``/data/media``, so a path they resolve against their mount points at these
    bytes. A unique name per call lets one test submit several runs without
    colliding on ``source_path``. All staged files are removed afterwards.
    """
    if not TUTORIAL_WAV.is_file():
        pytest.fail(f"tutorial fixture WAV missing: {TUTORIAL_WAV}")
    media_root = Path(settings.media_root)
    staged: list[Path] = []

    def _stage() -> str:
        rel = Path("e2e") / f"{uuid.uuid4().hex}.wav"
        dest = media_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(TUTORIAL_WAV, dest)
        staged.append(dest)
        return str(rel)

    yield _stage

    for dest in staged:
        dest.unlink(missing_ok=True)
    # Leave no empty `e2e/` behind in the operator's media root. (A crash mid-run
    # can still orphan a staged WAV or an `artifacts/<run_id>/` tree — the `e2e/`
    # prefix and per-run-id dirs keep those findable.)
    staging_dir = media_root / "e2e"
    if staging_dir.is_dir() and not any(staging_dir.iterdir()):
        staging_dir.rmdir()
