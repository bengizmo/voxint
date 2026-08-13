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

from sqlalchemy.orm import Session, sessionmaker

from tests.fakes import FakeASR, FakeDiarizer, FakeEmbedder, FakeLLM
from voxint import app_settings as store
from voxint.config import Settings
from voxint.db.models import ArtifactKind, AudioArtifact, MediaItem, PipelineRun
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
    base_ctx = StageContext(
        asr=FakeASR(),
        diarizer=FakeDiarizer(),
        embedder=FakeEmbedder(),
        llm=FakeLLM(),
        media_root=tmp_path,
        enhancement_context="PACK",
        vocabulary=("Packword",),
    )
    asr = base_ctx.asr  # the shared transport client dataclasses.replace preserves
    run_id = _seed_run(session_factory)

    # --- config A: user vocab "Foobar", LLM enabled ---
    with session_factory() as session:
        row = store.get_or_create(session)
        row.vocabulary = ["Foobar"]
        row.llm_enabled = True
        row.llm_model = "model-a"
        row.llm_base_url = "https://a.example/v1"
        session.commit()

    with session_factory() as session:
        prefs = resolve_run_preferences(store.get_app_settings(session), settings)
        ctx_a = apply_run_preferences(base_ctx, settings, prefs)
        transcribe.run(ctx_a, session, run_id)
        session.commit()

    assert asr.last_initial_prompt is not None
    assert "Foobar" in asr.last_initial_prompt  # user vocab reached ASR
    assert "Packword" in asr.last_initial_prompt  # pack vocab still unioned
    assert ctx_a.llm is not None  # enabled + env key present

    # --- config B: user vocab "Bazqux", LLM disabled — same base_ctx, no rebuild ---
    with session_factory() as session:
        row = store.get_or_create(session)
        row.vocabulary = ["Bazqux"]
        row.llm_enabled = False
        row.llm_model = None
        session.commit()

    with session_factory() as session:
        prefs = resolve_run_preferences(store.get_app_settings(session), settings)
        ctx_b = apply_run_preferences(base_ctx, settings, prefs)
        transcribe.run(ctx_b, session, run_id)
        session.commit()

    assert asr.last_initial_prompt is not None
    assert "Bazqux" in asr.last_initial_prompt  # the edit took effect...
    assert "Foobar" not in asr.last_initial_prompt  # ...with no _runtime rebuild
    assert "Packword" in asr.last_initial_prompt  # pack vocab persists
    assert ctx_b.llm is None  # disabled by the edit
