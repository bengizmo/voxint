"""Migration 0016 (app_settings.llm_api_key), up/down + parity (issue #10).

Mirrors the 0014 migration test: real alembic up/down against the shared test
database (head restored in teardown), the new column asserted present/absent,
an existing populated singleton **preserved** across the upgrade (new column
NULL), a key round-trip, ORM/DDL parity, and a single alembic head.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import Engine, inspect, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.types import TypeEngine

from voxint.db.models import AppSettings

REPO_ROOT = Path(__file__).resolve().parents[2]


def _pg_type(type_: TypeEngine[object]) -> str:
    compiled = type_.compile(dialect=postgresql.dialect())
    return "DOUBLE PRECISION" if compiled == "FLOAT" else compiled


@pytest.fixture()
def alembic_cfg(engine: Engine) -> Iterator[Config]:
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "alembic"))
    try:
        yield cfg
    finally:
        with engine.connect() as conn:
            conn.execute(text("DROP SCHEMA public CASCADE"))
            conn.execute(text("CREATE SCHEMA public"))
            conn.commit()
        command.upgrade(cfg, "head")


def test_migration_0016_roundtrip_preserves_singleton(
    alembic_cfg: Config, engine: Engine
) -> None:
    command.downgrade(alembic_cfg, "0015")
    cols = {c["name"] for c in inspect(engine).get_columns("app_settings")}
    assert "llm_api_key" not in cols

    # A populated singleton written under 0015 must survive the upgrade with the new
    # column defaulting to NULL (env fallback), not be dropped or reset.
    with engine.connect() as conn:
        conn.execute(
            text(
                "INSERT INTO app_settings (id, onboarding_complete, llm_enabled,"
                " llm_model) VALUES (1, true, true, 'row-model')"
            )
        )
        conn.commit()

    command.upgrade(alembic_cfg, "0016")
    reflected = {c["name"]: c for c in inspect(engine).get_columns("app_settings")}
    assert "llm_api_key" in reflected
    assert reflected["llm_api_key"]["nullable"] is True
    assert _pg_type(reflected["llm_api_key"]["type"]) == "TEXT"

    with engine.connect() as conn:
        onboarded, enabled, model, key = conn.execute(
            text(
                "SELECT onboarding_complete, llm_enabled, llm_model, llm_api_key"
                " FROM app_settings WHERE id = 1"
            )
        ).one()
        assert onboarded is True and enabled is True and model == "row-model"
        assert key is None  # new column defaults NULL on the preserved row

        # Key round-trips.
        conn.execute(text("UPDATE app_settings SET llm_api_key = 'sk-round-trip' WHERE id = 1"))
        conn.commit()
        assert (
            conn.execute(text("SELECT llm_api_key FROM app_settings WHERE id = 1")).scalar_one()
            == "sk-round-trip"
        )

    command.downgrade(alembic_cfg, "0015")
    cols = {c["name"] for c in inspect(engine).get_columns("app_settings")}
    assert "llm_api_key" not in cols


def test_app_settings_model_matches_migrated_schema(engine: Engine) -> None:
    reflected = {
        col["name"]: _pg_type(col["type"])
        for col in inspect(engine).get_columns("app_settings")
    }
    model = {col.name: _pg_type(col.type) for col in AppSettings.__table__.columns}
    assert reflected == model


def test_single_alembic_head() -> None:
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "alembic"))
    heads = ScriptDirectory.from_config(cfg).get_heads()
    # One linear head, no branches. Bump this to the newest revision whenever a
    # migration is added (0017 = domain pack selection #11 (released in 0.15.0);
    # 0018 = adjudication segment scope, issue #54; 0019 = transcript segment
    # confidence, 0020 = segment_review_states, issues #53/#58; 0021 =
    # waveform_peaks artifact kind, issue #57; 0022 = transcript_segments word
    # timings JSONB, issue #59; 0023 = segment_split_boundaries, issue #59 slice 2;
    # 0024 = app_settings feature-flag columns, issue #74; 0025 =
    # adjudication_decisions word-range scope, issue #59 slice 3; 0026 =
    # app_settings watch-folder columns, issue #60; 0027 =
    # app_settings.llm_bundled_enabled, issue #67; 0028 = transcript_segments
    # correction trace + version, issue #82; 0029 = app_settings.corrections
    # operator-authored rules, issue #84; 0030 = pipeline_runs.sidecar frozen
    # YAML sidecar snapshot, issue #104; 0031 = operator annotation layer
    # (annotation_tags + transcript_annotations + annotation_tag_links), #86;
    # 0032 = match_candidates observational match-decision evidence, issue #113;
    # 0033 = pipeline_runs diarization speaker-count hint, issue #128; 0034 =
    # pipeline_runs.initial_prompt provenance, issue #123; 0035 = segment_embeddings
    # + embedding_jobs semantic-search spine, issue #121; 0036 = app_settings
    # semantic-index feature flags, issue #121; 0037 = pipeline_runs detected
    # language + detection score, issue #124; 0038 = transcript translation
    # settings columns + translation_jobs/run_translations, issue #133; 0039 =
    # audio_clip artifact kind + audio_artifacts.idempotency_key clip cache key,
    # issue #88; 0040 = projects + media_folders + media location split, issue
    # #153; 0041 = speaker_profiles current per-field profile with provenance,
    # issue #159; 0042 = activity_events console activity outbox, issue #162;
    # 0043 = widen activity_events.kind for speaker_identified, issue #162;
    # 0044 = media_operations journal + media_operation_files, issue #155;
    # 0045 = widen file_kind CHECK for chunk + transcript_export, issue #155;
    # 0046 = drop legacy app_settings.media_folders / folder_domain_packs, #177).
    # 0047 = translation idempotency_key + replay_digest + immutability trigger, ADR 0008.
    # 0048 = benchmark_runs + benchmark_items tables.
    # 0049 = widen adjudication_decisions CHECK constraints for auto_enroll, #275.
    # 0050 = synthdetect_jobs + synthdetect_scores + AppSettings columns (#145).
    # 0051 = multi-user auth: users + auth_sessions tables, user_id FK on decisions (#9).
    # 0052 = profile_review_decisions.user_id attribution (#308).
    # 0053 = widen pipeline_runs status CHECK for operator-initiated PAUSED.
    # 0054 = API keys for the public REST API (#340).
    assert list(heads) == ["0054"]
