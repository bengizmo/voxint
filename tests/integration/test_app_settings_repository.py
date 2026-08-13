"""Repository for the app_settings singleton: get / create idempotency + onboarding."""

from sqlalchemy.orm import Session, sessionmaker

from voxint import app_settings as store


def test_get_app_settings_absent_is_none(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        assert store.get_app_settings(session) is None
        assert store.is_onboarded(session) is False


def test_get_or_create_defaults_and_is_idempotent(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        first = store.get_or_create(session)
        assert first.id == store.SINGLETON_ID
        assert first.onboarding_complete is False
        assert first.media_folders == []
        assert first.vocabulary == []
        assert first.llm_enabled is False
        assert first.llm_base_url is None
        # a second call in the same session returns the same row, not a new one
        assert store.get_or_create(session) is first
        session.commit()

    # persisted, and still exactly one row across a fresh session
    with session_factory() as session:
        again = store.get_or_create(session)
        assert again.id == store.SINGLETON_ID
        assert again.onboarding_complete is False


def test_complete_onboarding_flips_the_flag(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        assert store.is_onboarded(session) is False
        store.complete_onboarding(session)
        session.commit()

    with session_factory() as session:
        assert store.is_onboarded(session) is True
        # idempotent: completing again keeps a single onboarded row
        store.complete_onboarding(session)
        session.commit()
        assert store.is_onboarded(session) is True
