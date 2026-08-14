"""The /speakers roster page end to end: list, rename, merge, archive/restore,
embedding removal — htmx fragments, plain-form redirects, and CSRF refusal.

Real Postgres (migrated), real templates.
"""

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from tests.integration.test_speaker_enrollment import add_turn, make_completed_run, unit
from tests.integration.test_speaker_roster import enroll
from voxint.api.app import create_app
from voxint.api.csrf import (
    CSRF_ROSTER_ARCHIVE,
    CSRF_ROSTER_EMBEDDING_DELETE,
    CSRF_ROSTER_MERGE,
    CSRF_ROSTER_RENAME,
    CSRF_ROSTER_RESTORE,
    mint_csrf_token,
)
from voxint.config import Settings
from voxint.db.models import Speaker, SpeakerEmbedding
from voxint.speakers.roster import archive_speaker

CREDS = ("reviewer", "s3cret")
_CSRF_KEY = "roster-api-test-csrf-key"  # low-entropy; a known secret lets tests mint


def token(action: str) -> str:
    return mint_csrf_token(_CSRF_KEY, action)


@pytest.fixture()
def client(session_factory: sessionmaker[Session]) -> TestClient:
    settings = Settings(
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        csrf_secret=_CSRF_KEY,
    )
    test_client = TestClient(
        create_app(settings=settings, session_factory=session_factory)
    )
    test_client.auth = CREDS
    seed_onboarded(session_factory)
    return test_client


def seed_roster(session_factory: sessionmaker[Session]) -> tuple[uuid.UUID, uuid.UUID]:
    """Two enrolled speakers (Alice, Bob), committed."""
    with session_factory() as session:
        run_id = make_completed_run(session)
        add_turn(session, run_id, 0, "S0", vector=unit(0))
        add_turn(session, run_id, 1, "S0", vector=unit(0))
        add_turn(session, run_id, 2, "S1", vector=unit(1))
        add_turn(session, run_id, 3, "S1", vector=unit(1))
        session.commit()
        alice = enroll(session, run_id, "S0", "Alice")
        bob = enroll(session, run_id, "S1", "Bob")
        session.commit()
    return alice, bob


def test_page_lists_roster_with_evidence_and_confirms(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    seed_roster(session_factory)
    page = client.get("/speakers")
    assert page.status_code == 200
    assert "Alice" in page.text and "Bob" in page.text
    assert "1 enrollment" in page.text
    assert "titanet-large-v1" in page.text
    # The signature voiceprint strip renders from real centroid data.
    assert 'class="voiceprint"' in page.text
    # Destructive actions carry hx-confirm (first use in the console).
    assert "hx-confirm" in page.text
    # Empty inactive section is not rendered.
    assert "Former speakers" not in page.text


def test_rename_htmx_and_collision(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    alice, _ = seed_roster(session_factory)
    ok = client.post(
        f"/speakers/{alice}/rename",
        data={"display_name": "Alice Verified", "csrf_token": token(CSRF_ROSTER_RENAME)},
        headers={"HX-Request": "true"},
    )
    assert ok.status_code == 200
    assert "Alice Verified" in ok.text
    # Collision with another speaker's name: inline operator error, no change.
    clash = client.post(
        f"/speakers/{alice}/rename",
        data={"display_name": "Bob", "csrf_token": token(CSRF_ROSTER_RENAME)},
        headers={"HX-Request": "true"},
    )
    assert clash.status_code == 200
    assert "already exists" in clash.text
    with session_factory() as session:
        assert session.get(Speaker, alice).display_name == "Alice Verified"  # type: ignore[union-attr]


def test_rename_plain_form_redirects(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    alice, _ = seed_roster(session_factory)
    resp = client.post(
        f"/speakers/{alice}/rename",
        data={"display_name": "Alice P.", "csrf_token": token(CSRF_ROSTER_RENAME)},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/speakers"


def test_merge_moves_source_to_inactive(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    alice, bob = seed_roster(session_factory)
    resp = client.post(
        f"/speakers/{bob}/merge",
        data={"target_id": str(alice), "csrf_token": token(CSRF_ROSTER_MERGE)},
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200
    assert "Former speakers" in resp.text
    assert "merged into Alice" in resp.text
    with session_factory() as session:
        assert {
            row
            for row in session.execute(
                select(SpeakerEmbedding.speaker_id).distinct()
            ).scalars()
        } == {alice}


def test_archive_and_restore_roundtrip(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    _, bob = seed_roster(session_factory)
    archived = client.post(
        f"/speakers/{bob}/archive",
        data={"csrf_token": token(CSRF_ROSTER_ARCHIVE)},
        headers={"HX-Request": "true"},
    )
    assert archived.status_code == 200
    assert "archived" in archived.text and "Restore" in archived.text
    restored = client.post(
        f"/speakers/{bob}/restore",
        data={"csrf_token": token(CSRF_ROSTER_RESTORE)},
        headers={"HX-Request": "true"},
    )
    assert restored.status_code == 200
    assert "Former speakers" not in restored.text
    with session_factory() as session:
        assert session.get(Speaker, bob).deleted_at is None  # type: ignore[union-attr]


def test_embedding_delete_and_last_one_warning(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    _, bob = seed_roster(session_factory)
    page = client.get("/speakers")
    # Each speaker has exactly one enrollment, so the honest last-one warning shows.
    assert "unmatchable" in page.text
    with session_factory() as session:
        embedding_id = session.execute(
            select(SpeakerEmbedding.id).where(SpeakerEmbedding.speaker_id == bob)
        ).scalar_one()
    resp = client.post(
        f"/speakers/{bob}/embeddings/{embedding_id}/delete",
        data={"csrf_token": token(CSRF_ROSTER_EMBEDDING_DELETE)},
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200
    with session_factory() as session:
        assert session.get(SpeakerEmbedding, embedding_id) is None


def test_csrf_is_verified_before_any_write(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    alice, bob = seed_roster(session_factory)
    # Missing token → 403.
    assert (
        client.post(f"/speakers/{alice}/rename", data={"display_name": "X"}).status_code
        == 403
    )
    # A token minted for ANOTHER roster action is refused: per-action binding.
    cross = client.post(
        f"/speakers/{bob}/archive",
        data={"csrf_token": token(CSRF_ROSTER_RENAME)},
    )
    assert cross.status_code == 403
    with session_factory() as session:
        assert session.get(Speaker, alice).display_name == "Alice"  # type: ignore[union-attr]
        assert session.get(Speaker, bob).deleted_at is None  # type: ignore[union-attr]


def test_not_found_and_stale_target(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    alice, bob = seed_roster(session_factory)
    assert (
        client.post(
            f"/speakers/{uuid.uuid4()}/rename",
            data={"display_name": "Ghost", "csrf_token": token(CSRF_ROSTER_RENAME)},
        ).status_code
        == 404
    )
    # Stale merge form: the target was archived meanwhile → inline conflict text.
    with session_factory() as session:
        archive_speaker(session, alice)
        session.commit()
    resp = client.post(
        f"/speakers/{bob}/merge",
        data={"target_id": str(alice), "csrf_token": token(CSRF_ROSTER_MERGE)},
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200
    assert "no longer an active" in resp.text
