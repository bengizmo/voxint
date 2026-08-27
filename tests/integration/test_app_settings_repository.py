"""Repository for the app_settings singleton: get / create idempotency + onboarding."""

import threading

from sqlalchemy.orm import Session, sessionmaker

from voxint import app_settings as store
from voxint.db.models import MediaItem, PipelineRun


def test_get_app_settings_absent_is_none(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        assert store.get_app_settings(session) is None
        assert store.is_onboarded(session) is False


def test_get_or_create_defaults_and_is_idempotent(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        first = store.get_or_create(session, llm_enabled_default=False)
        assert first.id == store.SINGLETON_ID
        assert first.onboarding_complete is False
        assert first.vocabulary == []
        assert first.llm_enabled is False
        assert first.llm_base_url is None
        # a second call in the same session returns the same row, not a new one
        assert store.get_or_create(session, llm_enabled_default=False) is first
        session.commit()

    # persisted, and still exactly one row across a fresh session
    with session_factory() as session:
        again = store.get_or_create(session, llm_enabled_default=False)
        assert again.id == store.SINGLETON_ID
        assert again.onboarding_complete is False


def test_get_or_create_seeds_llm_enabled_from_default(
    session_factory: sessionmaker[Session],
) -> None:
    """A newly-created row takes ``llm_enabled`` from ``llm_enabled_default``.

    This is the deferred-finding-1 fix: ``resolve_run_preferences`` reads
    ``llm_enabled`` HARD from the row once one exists, so the first write (whatever
    its reason) must carry the env's enablement, not the model default False.
    """
    with session_factory() as session:
        row = store.get_or_create(session, llm_enabled_default=True)
        assert row.llm_enabled is True
        session.commit()
    # An existing row is returned unchanged — the default only applies at insert.
    with session_factory() as session:
        again = store.get_or_create(session, llm_enabled_default=False)
        assert again.llm_enabled is True


def test_complete_onboarding_flips_the_flag(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        assert store.is_onboarded(session) is False
        store.complete_onboarding(session, llm_enabled_default=False)
        session.commit()

    with session_factory() as session:
        assert store.is_onboarded(session) is True
        # idempotent: completing again keeps a single onboarded row
        store.complete_onboarding(session, llm_enabled_default=False)
        session.commit()
        assert store.is_onboarded(session) is True


def test_deleting_tutorial_run_sets_fk_null(
    session_factory: sessionmaker[Session],
) -> None:
    """Deleting the referenced run must NULL tutorial_run_id (ON DELETE SET NULL).

    Exercises the DB-level FK rule end to end: the singleton survives, the FK is
    cleared — it must not cascade-delete the settings row or raise.
    """
    with session_factory() as session:
        media = MediaItem(source_path="/media/tutorial.wav")
        session.add(media)
        session.flush()
        run = PipelineRun(media_item_id=media.id)
        session.add(run)
        session.flush()
        run_id = run.id
        settings = store.get_or_create(session, llm_enabled_default=False)
        settings.tutorial_run_id = run_id
        session.commit()
        assert settings.tutorial_run_id == run_id

    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        session.delete(run)
        session.commit()
        settings = store.get_app_settings(session)
        assert settings is not None
        assert settings.tutorial_run_id is None


def test_get_or_create_is_race_safe(
    session_factory: sessionmaker[Session],
) -> None:
    """Two writers that both observe an empty table still converge on one row.

    Forces the SAVEPOINT/IntegrityError adoption branch: a barrier makes both
    threads read None before either inserts, so one wins the UNIQUE(id) and the
    loser must re-read and adopt it — both end on id == 1, neither errors.
    """
    barrier = threading.Barrier(2)
    errors: list[Exception] = []
    results: list[int] = []
    lock = threading.Lock()

    def writer() -> None:
        try:
            with session_factory() as session:
                # observe the empty table, then release both threads together
                assert store.get_app_settings(session) is None
                barrier.wait(timeout=10)
                row = store.get_or_create(session, llm_enabled_default=False)
                session.commit()
                with lock:
                    results.append(row.id)
        except Exception as exc:
            # Capture rather than raise in a thread; surfaced via the assert below.
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=writer) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert not errors, errors
    assert results == [store.SINGLETON_ID, store.SINGLETON_ID]
