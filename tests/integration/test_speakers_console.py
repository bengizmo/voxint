"""The Console 2.0 speakers overview (issue #159) end to end.

/speakers is a LIVE page, so ``console_speakers_enabled`` branches content
rather than gating access: off must keep the legacy roster exactly as
shipped; on renders the new overview with resolver-backed numbers. These pin
the flag matrix, the sort allowlist's honest degrade, sort/view
cross-preservation (toggles and post-action re-renders), both views, the
empty state, and the verified badge / tier chip wiring.
"""

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.adjudication.ledger import record_decision
from voxint.api.app import create_app
from voxint.config import Settings
from voxint.db.models import (
    EMBEDDING_DIM,
    Decision,
    DiarizationTurn,
    MediaItem,
    PipelineRun,
    RunStatus,
    Speaker,
    SpeakerAssignment,
    TranscriptSegment,
)

CREDS = ("reviewer", "s3cret")
SPACE = "titanet-large-v1"
BASE = datetime(2026, 1, 1, tzinfo=UTC)


def _make_client(
    session_factory: sessionmaker[Session],
    tmp_path: Path,
    *,
    speakers_enabled: bool,
) -> TestClient:
    settings = Settings(
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        media_root=tmp_path,
        console_speakers_enabled=speakers_enabled,
    )
    client = TestClient(create_app(settings=settings, session_factory=session_factory))
    client.auth = CREDS
    seed_onboarded(session_factory)
    return client


def _seed_speaker_with_activity(
    session: Session, name: str, *, minutes_rank: int, human: bool
) -> uuid.UUID:
    """One speaker attributed in one completed run; more segments = more rank."""
    speaker = Speaker(display_name=name)
    session.add(speaker)
    session.flush()
    media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
    session.add(media)
    session.flush()
    media.created_at = BASE + timedelta(days=minutes_rank)
    run = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
    session.add(run)
    session.flush()
    for i in range(minutes_rank):
        vector = [0.0] * EMBEDDING_DIM
        vector[i % EMBEDDING_DIM] = 1.0
        session.add(
            DiarizationTurn(
                pipeline_run_id=run.id,
                turn_index=i,
                start_seconds=float(i * 10),
                end_seconds=float(i * 10 + 8),
                label="S0",
                embedding=vector,
                embedding_space=SPACE,
            )
        )
        session.add(
            TranscriptSegment(
                pipeline_run_id=run.id,
                segment_index=i,
                start_seconds=float(i * 10),
                end_seconds=float(i * 10 + 8),
                raw_text="hello there",
                diarization_label="S0",
            )
        )
    if human:
        record_decision(
            session,
            pipeline_run_id=run.id,
            diarization_label="S0",
            decision=Decision.ASSIGN,
            operator="op",
            idempotency_key=f"k-{uuid.uuid4()}",
            speaker_id=speaker.id,
        )
    else:
        session.add(
            SpeakerAssignment(
                pipeline_run_id=run.id,
                diarization_label="S0",
                speaker_id=speaker.id,
                method="cosine",
                confidence=0.9,
                grounded=True,
            )
        )
    session.flush()
    return speaker.id


def test_flag_off_renders_legacy_roster(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    client = _make_client(session_factory, tmp_path, speakers_enabled=False)
    with session_factory() as session:
        _seed_speaker_with_activity(session, "Alice", minutes_rank=1, human=True)
        session.commit()
    page = client.get("/speakers")
    assert page.status_code == 200
    # Legacy markers present, new-kit markers absent.
    assert "roster-card" in page.text
    assert 'class="lib-toolbar"' not in page.text
    assert 'class="view-toggle"' not in page.text


def test_flag_on_renders_overview_with_numbers(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    client = _make_client(session_factory, tmp_path, speakers_enabled=True)
    with session_factory() as session:
        _seed_speaker_with_activity(session, "Alice", minutes_rank=2, human=True)
        _seed_speaker_with_activity(session, "Bob", minutes_rank=1, human=False)
        session.commit()
    page = client.get("/speakers")
    assert page.status_code == 200
    assert 'class="lib-toolbar"' in page.text
    assert "2 speakers" in page.text
    assert "1 verified" in page.text
    # Alice (human assign) carries the badge; Bob (grounded, no diagnostics
    # row) shows the honest unavailable state, never "weak".
    assert "Verified" in page.text
    assert "match details unavailable" in page.text
    assert "weak voice match" not in page.text
    # Default sort = minutes: Alice (2 segments) before Bob (1).
    assert page.text.index("Alice") < page.text.index("Bob")


def test_sorts_apply_and_unknown_degrades(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    client = _make_client(session_factory, tmp_path, speakers_enabled=True)
    with session_factory() as session:
        _seed_speaker_with_activity(session, "Zed", minutes_rank=3, human=False)
        _seed_speaker_with_activity(session, "Amy", minutes_rank=1, human=False)
        session.commit()
    by_name = client.get("/speakers", params={"sort": "name"})
    assert by_name.text.index("Amy") < by_name.text.index("Zed")
    by_minutes = client.get("/speakers", params={"sort": "minutes"})
    assert by_minutes.text.index("Zed") < by_minutes.text.index("Amy")
    degraded = client.get("/speakers", params={"sort": "nope", "view": "bogus"})
    assert degraded.status_code == 200
    # Degrades to the defaults: minutes ordering, cards view.
    assert degraded.text.index("Zed") < degraded.text.index("Amy")
    assert 'class="lib-cards"' in degraded.text


def test_view_toggle_and_cross_preservation(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    client = _make_client(session_factory, tmp_path, speakers_enabled=True)
    with session_factory() as session:
        _seed_speaker_with_activity(session, "Alice", minutes_rank=1, human=True)
        session.commit()
    table = client.get("/speakers", params={"sort": "name", "view": "table"})
    assert 'class="lib-table"' in table.text
    # Each toggle's links carry the other control's current value.
    assert "/speakers?sort=name&view=cards" in table.text  # view links keep sort
    assert "/speakers?sort=minutes&view=table" in table.text  # sort links keep view
    cards = client.get("/speakers", params={"sort": "name", "view": "cards"})
    assert 'class="lib-cards"' in cards.text


def _csrf(client: TestClient, marker: str) -> str:
    """Scrape a minted token out of the rendered page (test_projects idiom)."""
    import re

    page = client.get("/speakers")
    fields = re.findall(r'name="csrf_token" value="([^"]+)"', page.text)
    assert fields, "no csrf token rendered"
    return fields[0]


def test_actions_preserve_sort_view_and_rerender_overview(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    client = _make_client(session_factory, tmp_path, speakers_enabled=True)
    with session_factory() as session:
        speaker_id = _seed_speaker_with_activity(
            session, "Alice", minutes_rank=1, human=True
        )
        session.commit()
    token = _csrf(client, "rename")
    # Plain POST: 303 back to the page with sort/view preserved.
    plain = client.post(
        f"/speakers/{speaker_id}/rename?sort=name&view=table",
        data={"display_name": "Alicia", "csrf_token": token},
        follow_redirects=False,
    )
    assert plain.status_code == 303
    assert plain.headers["location"] == "/speakers?sort=name&view=table"
    # htmx POST: the overview fragment, not the legacy roster fragment.
    fragment = client.post(
        f"/speakers/{speaker_id}/rename?sort=name&view=table",
        data={"display_name": "Alicia B", "csrf_token": token},
        headers={"HX-Request": "true"},
    )
    assert fragment.status_code == 200
    assert 'class="lib-toolbar"' in fragment.text
    assert "Alicia B" in fragment.text
    assert "roster-card" not in fragment.text
    # An operator refusal re-renders the overview inline (duplicate name 409-free).
    with session_factory() as session:
        _seed_speaker_with_activity(session, "Taken", minutes_rank=1, human=True)
        session.commit()
    refused = client.post(
        f"/speakers/{speaker_id}/rename?sort=name&view=table",
        data={"display_name": "Taken", "csrf_token": token},
        headers={"HX-Request": "true"},
    )
    assert refused.status_code == 200
    assert 'class="error"' in refused.text


def test_flag_off_post_paths_unchanged(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    client = _make_client(session_factory, tmp_path, speakers_enabled=False)
    with session_factory() as session:
        speaker_id = _seed_speaker_with_activity(
            session, "Alice", minutes_rank=1, human=True
        )
        session.commit()
    token = _csrf(client, "rename")
    plain = client.post(
        f"/speakers/{speaker_id}/rename",
        data={"display_name": "Alicia", "csrf_token": token},
        follow_redirects=False,
    )
    assert plain.status_code == 303
    assert plain.headers["location"] == "/speakers"
    fragment = client.post(
        f"/speakers/{speaker_id}/rename",
        data={"display_name": "Alicia B", "csrf_token": token},
        headers={"HX-Request": "true"},
    )
    assert "roster-card" in fragment.text
    assert 'class="lib-toolbar"' not in fragment.text


def test_empty_state_and_restore_section(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    client = _make_client(session_factory, tmp_path, speakers_enabled=True)
    empty = client.get("/speakers")
    assert empty.status_code == 200
    assert "No speakers yet" in empty.text
    with session_factory() as session:
        speaker = Speaker(display_name="Gone")
        session.add(speaker)
        session.flush()
        speaker.deleted_at = BASE
        session.commit()
    page = client.get("/speakers")
    assert "Former speakers (1)" in page.text
    assert "Restore" in page.text
