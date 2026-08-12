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
