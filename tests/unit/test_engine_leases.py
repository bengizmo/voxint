import pytest
from pydantic import ValidationError

from voxint.config import Settings
from voxint.db.models import STAGE_ORDER, Stage
from voxint.pipeline import engine


def test_default_stage_leases_gives_acquire_and_diarize_dedicated_budgets(
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    settings = Settings(
        _env_file=None,
        stage_lease_seconds=1000,
        diarize_embed_lease_seconds=2000,
        acquire_lease_seconds=400,
        acquire_timeout_seconds=50.0,  # + 300 margin stays below the 400 lease
    )
    monkeypatch.setattr(engine, "get_settings", lambda: settings)

    leases = engine.default_stage_leases()

    # Every stage in the pipeline order has a lease, and only ACQUIRE and
    # DIARIZE_EMBED deviate from the shared stage_lease_seconds.
    assert set(leases) == set(STAGE_ORDER)
    assert leases[Stage.ACQUIRE] == 400
    assert leases[Stage.DIARIZE_EMBED] == 2000
    for stage in (Stage.PREPARE, Stage.TRANSCRIBE, Stage.ENHANCE_MATCH, Stage.FINALIZE):
        assert leases[stage] == 1000


def test_visibility_floor_matches_engine_lease_topology(
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    # config._celery_visibility_covers_all_leases replicates the six-stage lease
    # sum INLINE (config cannot import engine.default_stage_leases without a
    # cycle). This ties the two hand-maintained formulas together: if a stage is
    # ever added or reclassified in default_stage_leases() without the matching
    # edit to the inline sum, the engine sum and the config floor diverge and one
    # of these assertions fails loudly instead of silently under-provisioning the
    # acks-late horizon. Distinct lease values make the sum unambiguous.
    leases = dict(
        stage_lease_seconds=1000,
        diarize_embed_lease_seconds=2000,
        acquire_lease_seconds=500,
        acquire_timeout_seconds=100.0,  # + 300 margin < 500 lease
    )
    base = Settings(_env_file=None, celery_visibility_timeout_seconds=10_000, **leases)
    monkeypatch.setattr(engine, "get_settings", lambda: base)
    engine_sum = sum(engine.default_stage_leases().values())

    # Visibility exactly at the engine's per-stage sum is accepted (floor is >=).
    accepted = Settings(
        _env_file=None, celery_visibility_timeout_seconds=engine_sum, **leases
    )
    assert accepted.celery_visibility_timeout_seconds == engine_sum
    # One second below the engine's sum is rejected by the config validator.
    with pytest.raises(ValidationError, match="stage leases"):
        Settings(_env_file=None, celery_visibility_timeout_seconds=engine_sum - 1, **leases)
