"""Real-wire acceptance matrix for the in-UI LLM API key (issue #10).

The unit tests in ``test_llm_key_resolver.py`` pin the *precedence* resolver; the
integration tests in ``test_llm_key_ui.py`` pin the *routes*. This module closes
the loop end to end: for **every** of the five sites that build an LLM client, it
drives the real code path and asserts the **actual outbound HTTP request** carries
``Authorization: Bearer <effective-key>`` — proving

  1. a UI-stored row key is used with env ``LLM_API_KEY`` unset (row wins),
  2. a *replacement* row key takes effect on the **next** run/job with **no
     process restart** (the effective key is resolved live, never cached), and
  3. *removing* the stored key (row → NULL) falls back to ``Bearer <env-key>`` live.

Genuine wire, not a construction stub: each site's own ``HttpLLMClient`` symbol is
swapped for a factory that builds a *real* :class:`~voxint.clients.llm.HttpLLMClient`
over an ``httpx.MockTransport``. Because the Bearer header is emitted per-request
from the ``api_key`` arg (``clients/llm.py`` builds ``self._headers`` at construction
and passes them on every POST), the captured request header is exactly what a live
endpoint would receive. The endpoint answer is canned so the caller completes; a few
sites may still fail *downstream* of the request under the canned body — that is
irrelevant here, the assertion is on the header that already went out on the wire.

The fifth site (``diagnostics.check_llm``) takes an ``httpx.Client`` directly, so it
needs no patching — the MockTransport is injected straight in.

Secret-absence is swept alongside (``caplog`` + doctor output). ``repr(AppSettings)``
and the GET/POST HTML are covered in ``test_llm_key_resolver.py`` / ``test_llm_key_ui.py``;
run/transcript exports render only transcript lines (``voxint.export`` never reads
``app_settings``), so they are structurally incapable of carrying the key.
"""

import json
import uuid
from pathlib import Path

import httpx
import pytest
from sqlalchemy.orm import Session, sessionmaker

import voxint.enrichment.asset_jobs as asset_mod
import voxint.enrichment.producers.names_llm as names_mod
import voxint.enrichment.research_jobs as research_mod
import voxint.pipeline.stages.context as context_mod
from tests.fakes import FakeASR, FakeDiarizer, FakeEmbedder, FakeLLM
from tests.integration.test_web_researcher import research_settings, seed_speaker
from tests.unit.test_research_agent import (
    PAGE_TEXT,
    PUBLIC_A,
    SEARCH_RESULT,
    FakeProvider,
    conclude,
    page_factory,
    resolver_map,
)
from voxint.app_settings import (
    SINGLETON_ID,
    get_app_settings,
    get_or_create,
    resolve_effective_llm_api_key,
)
from voxint.clients.base import EnhancementRequestSegment
from voxint.clients.llm import HttpLLMClient
from voxint.config import Settings
from voxint.db.models import (
    AppSettings,
    MediaItem,
    PipelineRun,
    RunAssetKind,
    RunStatus,
    TranscriptSegment,
)
from voxint.diagnostics import check_llm
from voxint.enrichment.asset_jobs import (
    _settings_from_snapshot,
    create_jobs,
)
from voxint.enrichment.asset_jobs import execute_job as asset_execute_job
from voxint.enrichment.producers.names_llm import run_llm_name_producer
from voxint.enrichment.research_jobs import create_job as research_create_job
from voxint.enrichment.research_jobs import execute_job as research_execute_job
from voxint.pipeline.stages.context import (
    StageContext,
    apply_run_preferences,
    resolve_run_preferences,
)

# ---------------------------------------------------------------------------
# The three key-states shared by every site: (row_key, env_key, expected_bearer).
# State 2 mutates the SAME row in the SAME process to prove "no restart"; state 3
# clears the row so env wins. env is unset in states 1-2 so a Bearer match there
# can ONLY come from the stored row value.
# ---------------------------------------------------------------------------
SENTINEL = "sk-SEKRIT-must-never-be-logged"

STATES: tuple[tuple[str | None, str, str], ...] = (
    ("sk-row-stored", "", "Bearer sk-row-stored"),  # 1: stored wins, env unset
    ("sk-row-replaced", "", "Bearer sk-row-replaced"),  # 2: replacement, live
    (None, "sk-env-fallback", "Bearer sk-env-fallback"),  # 3: removal -> env
)


def _completion(content: object) -> httpx.Response:
    """An OpenAI-compatible chat completion whose message content is ``content``."""
    body = content if isinstance(content, str) else json.dumps(content)
    return httpx.Response(200, json={"choices": [{"message": {"content": body}}]})


def _capturing_factory(captured: list[str | None], content: object) -> type[HttpLLMClient]:
    """A drop-in ``HttpLLMClient`` **subclass** that records the outbound
    ``Authorization`` header on a real MockTransport-backed client.

    Returned as a *class*, not a factory function, because some sites
    (``names_llm``) do ``isinstance(llm, HttpLLMClient)`` against the patched
    symbol to decide ownership/close — a bare function would break that check.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.headers.get("authorization"))
        return _completion(content)

    class _CapturingClient(HttpLLMClient):
        def __init__(
            self,
            base_url: str,
            model: str,
            api_key: str,
            timeout: float,
            client: object = None,
            *,
            sampling: object = None,
            disable_thinking: bool = False,
        ) -> None:
            # Mirror the real HttpLLMClient signature: run-asset routing (#67) now
            # always passes sampling= (None on the BYO path), and the reasoning-off
            # switch (LLM_DISABLE_THINKING) passes disable_thinking=, so the stand-in
            # must accept and forward both.
            http = httpx.Client(base_url=base_url, transport=httpx.MockTransport(handler))
            super().__init__(
                base_url,
                model,
                api_key,
                timeout,
                client=http,
                sampling=sampling,  # type: ignore[arg-type]
                disable_thinking=disable_thinking,
            )

    return _CapturingClient


def _make_settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "llm_enabled": True,
        "llm_base_url": "https://env.example/v1",
        "llm_model": "env-model",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[arg-type]


def _seed_app_settings(session_factory: sessionmaker[Session], *, key: str | None) -> None:
    """Set the singleton row's LLM key (and a row endpoint) for a DB-driven site."""
    with session_factory() as session:
        row = get_or_create(session, llm_enabled_default=True)
        row.llm_enabled = True
        row.llm_api_key = key
        row.llm_base_url = "https://row.example/v1"
        row.llm_model = "row-model"
        session.commit()


# ============================================================ site 1: enhance


def test_enhance_wire_key_precedence_live() -> None:
    """``apply_run_preferences`` -> enhancement client carries the effective key.

    Reproduces exactly what ``run_pipeline`` does: resolve prefs + effective key
    from the row, then build the per-run client. Pure (no DB) because both
    resolvers take a row object directly.
    """
    monkeypatch = pytest.MonkeyPatch()
    try:
        for row_key, env_key, expected in STATES:
            captured: list[str | None] = []
            monkeypatch.setattr(
                context_mod,
                "HttpLLMClient",
                _capturing_factory(captured, {"segments": [{"index": 0, "text": "Hi."}]}),
            )
            settings = _make_settings(llm_api_key=env_key)
            row = AppSettings(id=1, llm_enabled=True, llm_api_key=row_key)
            prefs = resolve_run_preferences(row, settings)
            key = resolve_effective_llm_api_key(row, settings)
            base = StageContext(
                asr=FakeASR(),
                diarizer=FakeDiarizer(),
                embedder=FakeEmbedder(),
                llm=FakeLLM(),
                media_root=Path("/data/media"),
                enhancement_context="",
                vocabulary=(),
            )
            ctx = apply_run_preferences(base, settings, prefs, base.domain_pack, llm_api_key=key)
            assert isinstance(ctx.llm, HttpLLMClient)
            ctx.llm.enhance_segments(
                (EnhancementRequestSegment(segment_index=0, text="hi", diarization_label="S0"),),
                "",
            )
            ctx.llm.close()
            assert captured == [expected]
    finally:
        monkeypatch.undo()


# ======================================================= site 2: asset_jobs


def _seed_completed_run(session_factory: sessionmaker[Session]) -> uuid.UUID:
    with session_factory() as session:
        media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
        session.add(media)
        session.flush()
        run = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
        session.add(run)
        session.flush()
        session.add(
            TranscriptSegment(
                pipeline_run_id=run.id,
                segment_index=0,
                start_seconds=0.0,
                end_seconds=5.0,
                raw_text="Hello, I am Joanne from Acme Corp.",
                diarization_label="S0",
            )
        )
        session.commit()
        return run.id


def test_asset_job_wire_key_precedence_live(
    session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``asset_jobs.execute_job`` resolves the effective key LIVE from the row at
    execution and carries it on the wire — a fresh run per state proves each pick
    up happens with no restart."""
    for row_key, env_key, expected in STATES:
        captured: list[str | None] = []
        monkeypatch.setattr(
            asset_mod,
            "HttpLLMClient",
            _capturing_factory(captured, {"summary": "An abstract."}),
        )
        env_settings = _make_settings(enrichment_run_assets_enabled=True, llm_api_key=env_key)
        _seed_app_settings(session_factory, key=row_key)
        run_id = _seed_completed_run(session_factory)
        with session_factory() as session:
            created, _ = create_jobs(
                session,
                pipeline_run_id=run_id,
                kinds=(RunAssetKind.SUMMARY,),
                settings=env_settings,
            )
            session.commit()
            job_id = created[0].id
        asset_execute_job(session_factory, job_id, settings=env_settings)
        assert captured and captured[0] == expected


def test_asset_create_jobs_never_snapshots_the_key(
    session_factory: sessionmaker[Session],
) -> None:
    """The enqueue snapshot MUST carry the row endpoint but NEVER the key; the
    executor re-sources base_url/model from that snapshot (no double-source)."""
    settings = _make_settings(enrichment_run_assets_enabled=True, llm_api_key="sk-env-endpoint")
    _seed_app_settings(session_factory, key=SENTINEL)
    run_id = _seed_completed_run(session_factory)
    with session_factory() as session:
        created, _ = create_jobs(
            session,
            pipeline_run_id=run_id,
            kinds=(RunAssetKind.SUMMARY,),
            settings=settings,
        )
        session.commit()
        config = dict(created[0].config)
    # endpoint IS snapshotted from the row ...
    assert config["base_url"] == "https://row.example/v1"
    assert config["model"] == "row-model"
    # ... but neither the stored key nor the env key is anywhere in the snapshot.
    blob = json.dumps(config)
    assert SENTINEL not in blob
    assert "sk-env-endpoint" not in blob
    assert not any("key" in str(k).lower() for k in config)
    # the executor reconstructs the SAME endpoint from the snapshot, no double-source.
    exec_settings = _settings_from_snapshot(settings, config)
    assert exec_settings.llm_base_url == "https://row.example/v1"
    assert exec_settings.llm_model == "row-model"


# ======================================================= site 3: research_jobs


def test_research_job_wire_key_precedence_live(
    session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``research_jobs.execute_job`` builds its client with the live row-resolved
    key; the first agent turn puts it on the wire."""
    speaker_id = seed_speaker(session_factory)
    for row_key, env_key, expected in STATES:
        captured: list[str | None] = []
        monkeypatch.setattr(
            research_mod,
            "HttpLLMClient",
            _capturing_factory(captured, conclude()),
        )
        settings = research_settings(llm_api_key=env_key)
        _seed_app_settings(session_factory, key=row_key)
        with session_factory() as session:
            job = research_create_job(session, speaker_id=speaker_id, settings=settings)
            job_id = job.id
            session.commit()
        research_execute_job(
            session_factory,
            job_id,
            settings=settings,
            search_provider=FakeProvider([SEARCH_RESULT]),
            read_client_factory=page_factory(PAGE_TEXT, []),
            read_resolver=resolver_map({"example.com": [PUBLIC_A]}),
        )
        assert captured and captured[0] == expected


# ========================================================== site 4: names_llm


def _seed_run_with_segments(session_factory: sessionmaker[Session]) -> uuid.UUID:
    with session_factory() as session:
        media = MediaItem(source_path=f"incoming/names/{uuid.uuid4()}.wav")
        session.add(media)
        session.flush()
        run = PipelineRun(media_item_id=media.id)
        session.add(run)
        session.flush()
        session.add(
            TranscriptSegment(
                pipeline_run_id=run.id,
                segment_index=0,
                start_seconds=0.0,
                end_seconds=5.0,
                raw_text="well jane doe checking in as always",
                diarization_label="S0",
            )
        )
        session.commit()
        return run.id


def test_names_producer_wire_key_precedence_live(
    session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``run_llm_name_producer`` resolves the effective key live from the row and
    carries it on the wire (a fresh run per state avoids idempotent short-circuit)."""
    for row_key, env_key, expected in STATES:
        captured: list[str | None] = []
        monkeypatch.setattr(
            names_mod,
            "HttpLLMClient",
            _capturing_factory(captured, {"segments": [{"index": 0, "text": "x"}]}),
        )
        env_settings = _make_settings(enrichment_names_llm_enabled=True, llm_api_key=env_key)
        _seed_app_settings(session_factory, key=row_key)
        run_id = _seed_run_with_segments(session_factory)
        with session_factory() as session:
            run_llm_name_producer(session, run_id=run_id, settings=env_settings, client=None)
            session.commit()
        assert captured and captured[0] == expected


# ========================================================== site 5: check_llm


def test_check_llm_wire_carries_key_and_omits_when_absent() -> None:
    """The doctor's LLM reachability probe emits ``Bearer <key>`` when a key is
    resolved and NO ``Authorization`` header when the effective key is empty."""
    captured: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.headers.get("authorization"))
        return httpx.Response(200, json={"data": []})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = check_llm(
        enabled=True,
        configured=True,
        base_url="https://row.example/v1",
        api_key="sk-effective",
        client=client,
    )
    assert result is not None and result.ok
    assert captured == ["Bearer sk-effective"]

    captured.clear()
    check_llm(
        enabled=True, configured=True, base_url="https://row.example/v1", api_key="", client=client
    )
    assert captured == [None]  # empty key -> no Authorization header on the wire


# ==================================================== secret-absence sweep


def test_wire_drive_never_logs_the_key(
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A sentinel stored key that actually reaches the wire never lands in logs."""
    captured: list[str | None] = []
    monkeypatch.setattr(
        asset_mod,
        "HttpLLMClient",
        _capturing_factory(captured, {"summary": "An abstract."}),
    )
    settings = _make_settings(enrichment_run_assets_enabled=True)
    _seed_app_settings(session_factory, key=SENTINEL)
    run_id = _seed_completed_run(session_factory)
    with caplog.at_level("DEBUG"):
        with session_factory() as session:
            created, _ = create_jobs(
                session,
                pipeline_run_id=run_id,
                kinds=(RunAssetKind.SUMMARY,),
                settings=settings,
            )
            session.commit()
            job_id = created[0].id
        asset_execute_job(session_factory, job_id, settings=settings)
    assert captured and captured[0] == f"Bearer {SENTINEL}"  # it truly went on the wire
    assert SENTINEL not in caplog.text  # ... and truly stayed out of the logs


def test_check_llm_result_never_contains_the_key() -> None:
    """Doctor output for the LLM probe reports reachability only — never the key."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": []})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = check_llm(
        enabled=True,
        configured=True,
        base_url="https://row.example/v1",
        api_key=SENTINEL,
        client=client,
    )
    assert result is not None
    assert SENTINEL not in (result.detail or "")
    assert SENTINEL not in result.name


def test_singleton_id_is_one() -> None:
    """Guard the module-level constant the seed helper relies on."""
    assert SINGLETON_ID == 1
    assert get_app_settings.__module__ == "voxint.app_settings"
