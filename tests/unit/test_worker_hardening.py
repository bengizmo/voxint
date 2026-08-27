import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

from celery.exceptions import OperationalError

from voxint.config import Settings
from voxint.pipeline.stages.context import parse_config_resolution_version


def _broker_down(args: tuple[str], **kwargs: object) -> None:
    raise OperationalError("broker down")


def test_autogenerate_run_assets_commits_before_broker_failure(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from voxint.worker import tasks

    factory = MagicMock()
    session = factory.return_value.__enter__.return_value
    job_id = uuid.uuid4()
    monkeypatch.setattr(tasks.app_settings, "get_app_settings", lambda session: None)
    monkeypatch.setattr(
        tasks.app_settings,
        "resolve_effective_enrichment_run_assets_autogenerate",
        lambda row, settings: True,
    )
    monkeypatch.setattr(tasks.asset_jobs, "run_asset_gates_open", lambda settings, row: True)
    monkeypatch.setattr(
        tasks.asset_jobs,
        "kinds_needing_generation",
        lambda session, run_id: (object(),),
    )
    monkeypatch.setattr(
        tasks.asset_jobs,
        "create_jobs",
        lambda *args, **kwargs: ([SimpleNamespace(id=job_id)], []),
    )
    monkeypatch.setattr(tasks.generate_run_asset, "apply_async", _broker_down)

    tasks._autogenerate_run_assets(factory, uuid.uuid4(), Settings(_env_file=None))

    session.commit.assert_called_once_with()


def test_autogenerate_translation_commits_before_broker_failure(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from voxint.worker import tasks

    factory = MagicMock()
    session = factory.return_value.__enter__.return_value
    session.get.return_value = SimpleNamespace(detected_language="en")
    job_id = uuid.uuid4()
    monkeypatch.setattr(tasks.app_settings, "get_app_settings", lambda session: None)
    monkeypatch.setattr(
        tasks.app_settings,
        "resolve_effective_translation_autogenerate",
        lambda row, settings: True,
    )
    monkeypatch.setattr(
        tasks.app_settings,
        "resolve_effective_translation_target_language",
        lambda row, settings: "es",
    )
    monkeypatch.setattr(tasks.translation_jobs, "normalized_language", lambda value: value)
    monkeypatch.setattr(
        tasks.translation_jobs, "translation_gates_open", lambda settings, row: True
    )
    monkeypatch.setattr(
        tasks.translation_jobs, "translation_needed", lambda session, run_id, target: True
    )
    monkeypatch.setattr(
        tasks.translation_jobs,
        "create_job",
        lambda *args, **kwargs: (SimpleNamespace(id=job_id), False),
    )
    monkeypatch.setattr(tasks.translate_run, "apply_async", _broker_down)

    tasks._autogenerate_translation(factory, uuid.uuid4(), Settings(_env_file=None))

    session.commit.assert_called_once_with()


def test_autogenerate_embeddings_commits_before_broker_failure(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from voxint.worker import tasks

    factory = MagicMock()
    session = factory.return_value.__enter__.return_value
    job_id = uuid.uuid4()
    monkeypatch.setattr(tasks.app_settings, "get_app_settings", lambda session: None)
    monkeypatch.setattr(
        tasks.app_settings,
        "resolve_effective_semantic_index_autogenerate",
        lambda row, settings: True,
    )
    monkeypatch.setattr(tasks.embedding_jobs, "embedding_gates_open", lambda settings, row: True)
    monkeypatch.setattr(tasks, "minilm_artifacts_available", lambda: True)
    monkeypatch.setattr(
        tasks.embedding_jobs,
        "create_jobs",
        lambda *args, **kwargs: (SimpleNamespace(id=job_id), False),
    )
    monkeypatch.setattr(tasks.generate_segment_embeddings, "apply_async", _broker_down)

    tasks._autogenerate_segment_embeddings(factory, uuid.uuid4(), Settings(_env_file=None))

    session.commit.assert_called_once_with()


def test_parse_config_resolution_version_defaults_to_live_union() -> None:
    assert parse_config_resolution_version(None) == 1
    assert parse_config_resolution_version({}) == 1
    assert parse_config_resolution_version({"config_resolution_version": None}) == 1
    assert parse_config_resolution_version({"config_resolution_version": "malformed"}) == 1
    assert parse_config_resolution_version({"config_resolution_version": float("inf")}) == 1


def test_parse_config_resolution_version_accepts_valid_snapshot() -> None:
    assert parse_config_resolution_version({"config_resolution_version": 2}) == 2
