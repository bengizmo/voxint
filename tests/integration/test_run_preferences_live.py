"""Live per-run preferences: an app_settings edit changes the next run's
effective vocabulary + LLM wiring with NO worker restart (slice 2).

This is the codex-gated proof for the "live" refactor. A single process-cached
base context is built once; between two runs the app_settings row is mutated in
the DB, and each run re-reads + re-applies it. The shared FakeASR records the
initial_prompt it was handed, so we can assert the vocabulary that actually
reached ASR changed — without ever rebuilding the base context.
"""

import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy.orm import Session, sessionmaker

from tests.fakes import FakeASR, FakeDiarizer, FakeEmbedder, FakeLLM
from voxint import app_settings as store
from voxint.config import Settings
from voxint.db.models import ArtifactKind, AudioArtifact, MediaItem, PipelineRun, RunStatus, Stage
from voxint.domain_packs.base import DomainPack
from voxint.pipeline.stages import transcribe
from voxint.pipeline.stages.context import (
    StageContext,
    apply_run_preferences,
    resolve_run_preferences,
)


def _seed_run(session_factory: sessionmaker[Session]) -> uuid.UUID:
    """A run with a preprocessed-audio artifact so the transcribe stage can run.

    FakeASR ignores the audio path, so the file need not exist — only the
    artifact row must, since transcribe locates its input through it.
    """
    with session_factory() as session:
        media = MediaItem(source_path="incoming/live.wav")
        session.add(media)
        session.flush()
        run = PipelineRun(media_item_id=media.id)
        session.add(run)
        session.flush()
        run_id = run.id
        session.add(
            AudioArtifact(
                pipeline_run_id=run_id,
                kind=ArtifactKind.PREPROCESSED_AUDIO.value,
                path="runs/live/normalized.wav",
            )
        )
        session.commit()
    return run_id


def test_app_settings_edit_changes_next_run_without_rebuild(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    settings = Settings(_env_file=None, llm_enabled=False, llm_api_key="sk-test")
    # Built ONCE — stands in for the process-cached _runtime() base context.
    pack = DomainPack(
        name="live-test",
        vocabulary=("Packword",),
        prompt_fragments={"enhancement_context": "PACK"},
    )
    base_ctx = StageContext(
        asr=FakeASR(),
        diarizer=FakeDiarizer(),
        embedder=FakeEmbedder(),
        llm=FakeLLM(),
        media_root=tmp_path,
        domain_pack=pack,
        enhancement_context="PACK",
        vocabulary=("Packword",),
    )
    asr = base_ctx.asr  # the shared transport client dataclasses.replace preserves
    run_id = _seed_run(session_factory)

    # --- config A: user vocab "Foobar", LLM enabled ---
    with session_factory() as session:
        row = store.get_or_create(session, llm_enabled_default=False)
        row.vocabulary = ["Foobar"]
        row.llm_enabled = True
        row.llm_model = "model-a"
        row.llm_base_url = "https://a.example/v1"
        session.commit()

    with session_factory() as session:
        row = store.get_app_settings(session)
        prefs = resolve_run_preferences(row, settings)
        key = store.resolve_effective_llm_api_key(row, settings)
        ctx_a = apply_run_preferences(base_ctx, settings, prefs, pack, llm_api_key=key)
        transcribe.run(ctx_a, session, run_id)
        session.commit()

    assert asr.last_initial_prompt is not None
    assert "Foobar" in asr.last_initial_prompt  # user vocab reached ASR
    assert "Packword" in asr.last_initial_prompt  # pack vocab still unioned
    assert ctx_a.llm is not None  # enabled + env key present

    # --- config B: user vocab "Bazqux", LLM disabled — same base_ctx, no rebuild ---
    with session_factory() as session:
        row = store.get_or_create(session, llm_enabled_default=False)
        row.vocabulary = ["Bazqux"]
        row.llm_enabled = False
        row.llm_model = None
        session.commit()

    with session_factory() as session:
        row = store.get_app_settings(session)
        prefs = resolve_run_preferences(row, settings)
        key = store.resolve_effective_llm_api_key(row, settings)
        ctx_b = apply_run_preferences(base_ctx, settings, prefs, pack, llm_api_key=key)
        transcribe.run(ctx_b, session, run_id)
        session.commit()

    assert asr.last_initial_prompt is not None
    assert "Bazqux" in asr.last_initial_prompt  # the edit took effect...
    assert "Foobar" not in asr.last_initial_prompt  # ...with no _runtime rebuild
    assert "Packword" in asr.last_initial_prompt  # pack vocab persists
    assert ctx_b.llm is None  # disabled by the edit

    # The provenance stamp tracks the re-run: the persisted prompt equals what the
    # SECOND decode saw, so a requeue with an edited glossary overwrites the first
    # run's stamp rather than leaving stale provenance (issue #123).
    with session_factory() as session:
        stored = session.get(PipelineRun, run_id)
        assert stored is not None
        assert stored.initial_prompt == asr.last_initial_prompt


def test_run_pipeline_reapplies_settings_each_invocation(
    session_factory: sessionmaker[Session], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run_pipeline glue (not just the resolve/apply seam) re-reads the row
    every invocation while reusing ONE cached base context — the real no-restart
    path. _runtime() is stubbed to a fixed base; execute_run captures the applied
    StageContext so we can assert its vocabulary tracks the DB edit.
    """
    from voxint.worker import tasks as worker_tasks

    base_ctx = StageContext(
        asr=FakeASR(),
        diarizer=FakeDiarizer(),
        embedder=FakeEmbedder(),
        llm=None,
        media_root=tmp_path,
    )
    # Stand in for the process-cached _runtime() singleton — built ONCE, reused.
    monkeypatch.setattr(worker_tasks, "_runtime", lambda: (session_factory, base_ctx))

    # The run carries a frozen pack snapshot with "Packword" (issue #11): run_pipeline
    # resolves the pack from the RUN, not the cached base, and unions it with the
    # user vocab from the (edited) app_settings row on each invocation.
    with session_factory() as session:
        media = MediaItem(source_path="incoming/reapply.wav")
        session.add(media)
        session.flush()
        run = PipelineRun(
            media_item_id=media.id,
            domain_pack=DomainPack(name="p", vocabulary=("Packword",)).to_mapping(),
        )
        session.add(run)
        session.flush()
        seeded_run_id = run.id
        session.commit()

    captured: list[StageContext] = []

    def fake_execute_run(
        factory: object,
        run_id: uuid.UUID,
        stage_fns: dict[Stage, object],
        *,
        settings: object = None,
        stages: object = None,
    ) -> object:
        # partial(transcribe.run, ctx) — arg 0 is the applied StageContext.
        captured.append(stage_fns[Stage.TRANSCRIBE].args[0])  # type: ignore[attr-defined]
        return SimpleNamespace(status=RunStatus.COMPLETED)

    monkeypatch.setattr(worker_tasks, "execute_run", fake_execute_run)

    run_id = seeded_run_id  # a real run so run_pipeline can read its pack snapshot

    with session_factory() as session:
        store.get_or_create(session, llm_enabled_default=False).vocabulary = ["Foobar"]
        session.commit()
    worker_tasks.run_pipeline.apply(args=[str(run_id)]).get()

    with session_factory() as session:
        store.get_or_create(session, llm_enabled_default=False).vocabulary = ["Bazqux"]
        session.commit()
    worker_tasks.run_pipeline.apply(args=[str(run_id)]).get()

    assert captured[0].vocabulary == ("Packword", "Foobar")
    assert captured[1].vocabulary == ("Packword", "Bazqux")  # re-read, no rebuild


def test_transcribe_records_applied_initial_prompt(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    """The transcribe stage stamps the run with the exact bounded prompt whisper
    decoded with (issue #123) — the operator's glossary is captured nowhere else, so
    this closes the provenance gap the pack snapshot leaves. The persisted value must
    equal what FakeASR was handed, term-for-term."""
    settings = Settings(_env_file=None, llm_enabled=False)
    pack = DomainPack(name="prov-test", vocabulary=("Packword",))
    base_ctx = StageContext(
        asr=FakeASR(),
        diarizer=FakeDiarizer(),
        embedder=FakeEmbedder(),
        llm=None,
        media_root=tmp_path,
        domain_pack=pack,
        vocabulary=("Packword",),
    )
    asr = base_ctx.asr
    run_id = _seed_run(session_factory)

    with session_factory() as session:
        row = store.get_or_create(session, llm_enabled_default=False)
        row.vocabulary = ["Foobar"]
        session.commit()

    with session_factory() as session:
        row = store.get_app_settings(session)
        prefs = resolve_run_preferences(row, settings)
        ctx = apply_run_preferences(base_ctx, settings, prefs, pack, llm_api_key=None)
        transcribe.run(ctx, session, run_id)
        session.commit()

    with session_factory() as session:
        stored = session.get(PipelineRun, run_id)
        assert stored is not None
        # Exactly what whisper saw, and the union order is preserved.
        assert stored.initial_prompt == asr.last_initial_prompt == "Packword, Foobar"


def test_transcribe_records_null_prompt_when_no_vocabulary(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    """A run with no effective vocabulary renders an empty prompt (None), and the
    stage records that as SQL NULL — never an empty string masquerading as a hint."""
    base_ctx = StageContext(
        asr=FakeASR(),
        diarizer=FakeDiarizer(),
        embedder=FakeEmbedder(),
        llm=None,
        media_root=tmp_path,
        vocabulary=(),
    )
    asr = base_ctx.asr
    run_id = _seed_run(session_factory)

    with session_factory() as session:
        transcribe.run(base_ctx, session, run_id)
        session.commit()

    assert asr.last_initial_prompt is None
    with session_factory() as session:
        stored = session.get(PipelineRun, run_id)
        assert stored is not None
        assert stored.initial_prompt is None


def test_transcribe_leaves_prompt_null_when_decode_fails(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    """A failed decode records no provenance: the stamp lands only AFTER a successful
    transcribe, so a run that never decoded leaves the column NULL — the card must not
    claim whisper saw a prompt the service never received (issue #123)."""

    class RaisingASR:
        def transcribe(self, *args: object, **kwargs: object) -> object:
            raise RuntimeError("asr unreachable")

    base_ctx = StageContext(
        asr=RaisingASR(),  # type: ignore[arg-type]
        diarizer=FakeDiarizer(),
        embedder=FakeEmbedder(),
        llm=None,
        media_root=tmp_path,
        vocabulary=("Packword",),
    )
    run_id = _seed_run(session_factory)

    with session_factory() as session:
        with pytest.raises(RuntimeError, match="asr unreachable"):
            transcribe.run(base_ctx, session, run_id)
        session.rollback()

    with session_factory() as session:
        stored = session.get(PipelineRun, run_id)
        assert stored is not None
        assert stored.initial_prompt is None


def test_run_pipeline_closes_per_run_llm_client(
    session_factory: sessionmaker[Session], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The per-run HttpLLMClient (built when the row enables LLM and a key is set)
    is closed after the run, so a long-lived worker doesn't leak a pool per run."""
    from voxint.clients.llm import HttpLLMClient
    from voxint.worker import tasks as worker_tasks

    base_ctx = StageContext(
        asr=FakeASR(),
        diarizer=FakeDiarizer(),
        embedder=FakeEmbedder(),
        llm=None,
        media_root=tmp_path,
    )
    monkeypatch.setattr(worker_tasks, "_runtime", lambda: (session_factory, base_ctx))
    monkeypatch.setattr(
        worker_tasks,
        "execute_run",
        lambda factory, run_id, stage_fns, *, settings=None, stages=None: SimpleNamespace(
            status=RunStatus.COMPLETED
        ),
    )
    # Env supplies the (never-stored) API key so the enabled row actually builds a client.
    monkeypatch.setattr(
        worker_tasks, "get_settings", lambda: Settings(_env_file=None, llm_api_key="sk-test")
    )
    closes: list[int] = []
    real_close = HttpLLMClient.close
    monkeypatch.setattr(
        HttpLLMClient, "close", lambda self: (closes.append(1), real_close(self))[1]
    )

    with session_factory() as session:
        row = store.get_or_create(session, llm_enabled_default=False)
        row.llm_enabled = True
        row.llm_model = "m"
        session.commit()

    worker_tasks.run_pipeline.apply(args=[str(uuid.uuid4())]).get()
    assert closes == [1]  # exactly one per-run client, closed exactly once
