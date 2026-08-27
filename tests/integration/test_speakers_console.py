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
from sqlalchemy import select
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
    assert "2 people" in page.text
    assert "1 verified by you" in page.text
    # Alice (human assign) carries the verified chip; Bob (grounded, no
    # diagnostics row) shows "needs you", never "weak".
    assert "verified" in page.text
    assert "needs you" in page.text
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
    assert 'class="grid-table' in table.text
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


# ---- Profile page (#159): dark routes, stats, edits, OOB refresh -----------


def _seed_candidate(
    session: Session, speaker_id: uuid.UUID, field: str, value: str
) -> uuid.UUID:
    """One proposed research draft for the speaker (test_speaker_profile idiom)."""
    from voxint.db.models import EnrichmentCandidate, EnrichmentProducerRun

    now = datetime.now(UTC)
    run = EnrichmentProducerRun(
        producer="test-producer",
        producer_version="1",
        target_kind="speaker",
        speaker_id=speaker_id,
        covered_fields=[field],
        generation=1,
        outcome="found",
        idempotency_key=f"prod-{uuid.uuid4()}",
        started_at=now,
        completed_at=now,
    )
    session.add(run)
    session.flush()
    cand = EnrichmentCandidate(
        producer_run_id=run.id,
        target_kind="speaker",
        speaker_id=speaker_id,
        field=field,
        value=value,
    )
    session.add(cand)
    session.flush()
    return cand.id


def _profile_edit_token(page_text: str) -> str:
    """The CSRF token inside a profile-panel edit form (it is the hidden input
    immediately followed by the ``field`` input)."""
    import re

    match = re.search(
        r'name="csrf_token" value="([^"]+)">\s*<input type="hidden" name="field"',
        page_text,
    )
    assert match, "no profile edit form on the page"
    return match.group(1)


def _decision_token(page_text: str) -> str:
    """The CSRF token inside a research decision form (followed by the nonce)."""
    import re

    match = re.search(
        r'name="csrf_token" value="([^"]+)">\s*<input type="hidden" name="nonce"',
        page_text,
    )
    assert match, "no decision form on the page"
    return match.group(1)


def test_profile_routes_dark_when_flag_off(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    client = _make_client(session_factory, tmp_path, speakers_enabled=False)
    with session_factory() as session:
        speaker_id = _seed_speaker_with_activity(
            session, "Alice", minutes_rank=1, human=True
        )
        session.commit()
    assert client.get(f"/speakers/{speaker_id}").status_code == 404
    assert (
        client.post(
            f"/speakers/{speaker_id}/profile", data={"field": "bio", "value": "x"}
        ).status_code
        == 404
    )


def test_profile_page_renders_stats_research_and_recordings(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    client = _make_client(session_factory, tmp_path, speakers_enabled=True)
    with session_factory() as session:
        speaker_id = _seed_speaker_with_activity(
            session, "Alice", minutes_rank=2, human=True
        )
        session.commit()
    page = client.get(f"/speakers/{speaker_id}")
    assert page.status_code == 200
    assert "Alice" in page.text
    assert "verified" in page.text
    # Stats tiles: recordings, segments, first/last heard.
    assert "FIRST HEARD" in page.text
    assert "SEGMENTS" in page.text
    # Profile panel with edit forms, research block, recordings table.
    assert 'id="profile-panel"' in page.text
    assert "not set" in page.text
    assert f'id="research-{speaker_id}"' in page.text
    assert "/runs/" in page.text  # recordings drill through to the run page
    assert "verified" in page.text  # the human-assign chip on the appearance


def test_profile_tombstone_redirects_and_archived_reads_only(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    from voxint.speakers.roster import merge_speakers as do_merge

    client = _make_client(session_factory, tmp_path, speakers_enabled=True)
    with session_factory() as session:
        source = _seed_speaker_with_activity(session, "Dupe", minutes_rank=1, human=True)
        target = _seed_speaker_with_activity(session, "Keep", minutes_rank=1, human=True)
        do_merge(session, source, target)
        archived = _seed_speaker_with_activity(
            session, "Old voice", minutes_rank=1, human=True
        )
        session.commit()
    moved = client.get(f"/speakers/{source}", follow_redirects=False)
    assert moved.status_code == 303
    assert moved.headers["location"] == f"/speakers/{target}"
    with session_factory() as session:
        session.get(Speaker, archived).deleted_at = BASE
        session.commit()
    page = client.get(f"/speakers/{archived}")
    assert page.status_code == 200
    assert "archived" in page.text
    assert 'name="field"' not in page.text  # no edit forms
    assert f'id="research-{archived}"' not in page.text  # research off
    refused = client.post(
        f"/speakers/{archived}/profile",
        data={"field": "bio", "value": "x", "csrf_token": "bogus"},
    )
    assert refused.status_code == 403  # CSRF refuses before the archived check


def test_profile_manual_edit_set_clear_and_refusals(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    client = _make_client(session_factory, tmp_path, speakers_enabled=True)
    with session_factory() as session:
        speaker_id = _seed_speaker_with_activity(
            session, "Alice", minutes_rank=1, human=True
        )
        session.commit()
    token = _profile_edit_token(client.get(f"/speakers/{speaker_id}").text)
    # Plain POST: 303 back to the page; the value persists with manual provenance.
    plain = client.post(
        f"/speakers/{speaker_id}/profile",
        data={"field": "bio", "value": "Ornithologist.", "csrf_token": token},
        follow_redirects=False,
    )
    assert plain.status_code == 303
    assert plain.headers["location"] == f"/speakers/{speaker_id}"
    page = client.get(f"/speakers/{speaker_id}")
    assert "Ornithologist." in page.text
    assert "entered by hand" in page.text
    # htmx POST: the refreshed panel fragment only.
    fragment = client.post(
        f"/speakers/{speaker_id}/profile",
        data={"field": "affiliation", "value": "Birds Inc", "csrf_token": token},
        headers={"HX-Request": "true"},
    )
    assert fragment.status_code == 200
    assert 'id="profile-panel"' in fragment.text
    assert "Birds Inc" in fragment.text
    assert "<h1>" not in fragment.text
    # Clear removes the field.
    cleared = client.post(
        f"/speakers/{speaker_id}/profile",
        data={"field": "bio", "action": "clear", "csrf_token": token},
        headers={"HX-Request": "true"},
    )
    assert "Ornithologist." not in cleared.text
    # Missing CSRF: uniform 403 before any write.
    assert (
        client.post(
            f"/speakers/{speaker_id}/profile", data={"field": "bio", "value": "x"}
        ).status_code
        == 403
    )
    # Unknown field: operator error rendered inline, nothing stored.
    bad = client.post(
        f"/speakers/{speaker_id}/profile",
        data={"field": "name", "value": "x", "csrf_token": token},
        headers={"HX-Request": "true"},
    )
    assert bad.status_code == 200
    assert 'class="error"' in bad.text


def test_profile_decision_refreshes_panel_out_of_band(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    client = _make_client(session_factory, tmp_path, speakers_enabled=True)
    with session_factory() as session:
        speaker_id = _seed_speaker_with_activity(
            session, "Alice", minutes_rank=1, human=True
        )
        cand = _seed_candidate(session, speaker_id, "bio", "Wrote the book on birds.")
        reject_me = _seed_candidate(session, speaker_id, "affiliation", "Birds Inc")
        session.commit()
    page = client.get(f"/speakers/{speaker_id}")
    assert "Wrote the book on birds." in page.text  # proposed draft visible
    assert "?page=profile" in page.text  # research URLs carry the page context
    token = _decision_token(page.text)
    # htmx accept from the profile page: research fragment PLUS the profile
    # panel out-of-band, already showing the materialized value.
    resp = client.post(
        f"/speakers/{speaker_id}/research/candidates/{cand}/decision?page=profile",
        data={"csrf_token": token, "nonce": uuid.uuid4().hex, "verdict": "accept"},
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200
    assert f'id="research-{speaker_id}"' in resp.text
    assert 'hx-swap-oob="true"' in resp.text
    assert 'id="profile-panel"' in resp.text
    assert "accepted claim" in resp.text
    assert "Wrote the book on birds." in resp.text
    # No-JS decision from the profile page: a full-page 303, never a bare fragment.
    nojs = client.post(
        f"/speakers/{speaker_id}/research/candidates/{reject_me}/decision?page=profile",
        data={"csrf_token": token, "nonce": uuid.uuid4().hex, "verdict": "reject"},
        follow_redirects=False,
    )
    assert nojs.status_code == 303
    assert nojs.headers["location"] == f"/speakers/{speaker_id}"
    # Off the profile page the decision response stays the plain fragment.
    third = _seed_candidate_committed(session_factory, speaker_id)
    fragment = client.post(
        f"/speakers/{speaker_id}/research/candidates/{third}/decision",
        data={"csrf_token": token, "nonce": uuid.uuid4().hex, "verdict": "reject"},
        headers={"HX-Request": "true"},
    )
    assert fragment.status_code == 200
    assert 'hx-swap-oob="true"' not in fragment.text


def _seed_candidate_committed(
    session_factory: sessionmaker[Session], speaker_id: uuid.UUID
) -> uuid.UUID:
    with session_factory() as session:
        cand = _seed_candidate(session, speaker_id, "link", "https://example.org/x")
        session.commit()
    return cand


def test_overview_names_link_to_profiles(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    client = _make_client(session_factory, tmp_path, speakers_enabled=True)
    with session_factory() as session:
        speaker_id = _seed_speaker_with_activity(
            session, "Alice", minutes_rank=1, human=True
        )
        session.commit()
    page = client.get("/speakers")
    assert f'href="/speakers/{speaker_id}"' in page.text


# ---- Overview reminders (#159): pending names + unverified high-activity ---


def _seed_name_candidate(
    session: Session, run_id: uuid.UUID, label: str, value: str
) -> uuid.UUID:
    """One proposed speaker-name draft on a run's diarization label."""
    from voxint.db.models import EnrichmentCandidate, EnrichmentProducerRun

    now = datetime.now(UTC)
    producer = EnrichmentProducerRun(
        producer="test-producer",
        producer_version="1",
        target_kind="run_label",
        pipeline_run_id=run_id,
        diarization_label=label,
        covered_fields=["name"],
        generation=1,
        outcome="found",
        idempotency_key=f"prod-{uuid.uuid4()}",
        started_at=now,
        completed_at=now,
    )
    session.add(producer)
    session.flush()
    cand = EnrichmentCandidate(
        producer_run_id=producer.id,
        target_kind="run_label",
        pipeline_run_id=run_id,
        diarization_label=label,
        field="name",
        value=value,
    )
    session.add(cand)
    session.flush()
    return cand.id


def test_reminders_pending_names_and_unverified_high_activity(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    client = _make_client(session_factory, tmp_path, speakers_enabled=True)
    with session_factory() as session:
        # 40 segments x 8s = 320s > the 5-minute floor, no human ruling.
        _seed_speaker_with_activity(
            session, "Loud stranger", minutes_rank=40, human=False
        )
        # Below the floor: never a reminder.
        _seed_speaker_with_activity(session, "Quiet voice", minutes_rank=1, human=False)
        # Verified: never a reminder however active.
        _seed_speaker_with_activity(session, "Known voice", minutes_rank=40, human=True)
        run_id = session.execute(
            select(PipelineRun.id).limit(1)
        ).scalar_one()
        _seed_name_candidate(session, run_id, "S0", "Dr. Example")
        _seed_name_candidate(session, run_id, "S1", "Someone Else")
        session.commit()
    page = client.get("/speakers")
    assert page.status_code == 200
    assert "2 name suggestions to review" in page.text
    assert "1 voice waiting for a name" in page.text
    assert "TO DO" in page.text


def test_reminders_absent_when_no_work_waits(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    client = _make_client(session_factory, tmp_path, speakers_enabled=True)
    with session_factory() as session:
        _seed_speaker_with_activity(session, "Alice", minutes_rank=1, human=True)
        session.commit()
    page = client.get("/speakers")
    assert "TO DO" not in page.text


def test_archived_speaker_refuses_research_mutations(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    """Archived = read-only at the mutation boundary too (#159 review): a
    stale form must not start research or decide drafts for an archived
    speaker, whatever the page promised before archival."""
    client = _make_client(session_factory, tmp_path, speakers_enabled=True)
    with session_factory() as session:
        speaker_id = _seed_speaker_with_activity(
            session, "Shelved", minutes_rank=1, human=True
        )
        cand = _seed_candidate(session, speaker_id, "bio", "A draft.")
        session.commit()
    page = client.get(f"/speakers/{speaker_id}")
    decision_token = _decision_token(page.text)
    with session_factory() as session:
        session.get(Speaker, speaker_id).deleted_at = BASE
        session.commit()
    refused = client.post(
        f"/speakers/{speaker_id}/research/candidates/{cand}/decision",
        data={
            "csrf_token": decision_token,
            "nonce": uuid.uuid4().hex,
            "verdict": "accept",
        },
    )
    assert refused.status_code == 409
    # No decision was recorded and nothing materialized.
    from voxint.db.models import ProfileReviewDecision, SpeakerProfile

    with session_factory() as session:
        assert session.query(ProfileReviewDecision).count() == 0
        assert session.query(SpeakerProfile).count() == 0


def test_flag_off_legacy_research_stays_single_id(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    """Alias-aware draft gathering is a Console 2.0 behavior: with the flag
    off, the legacy roster renders exactly the drafts it always did (a merged
    source's drafts do NOT appear), while the flag-on profile page shows them."""
    from voxint.speakers.roster import merge_speakers as do_merge

    with session_factory() as session:
        source = _seed_speaker_with_activity(session, "Src", minutes_rank=1, human=True)
        target = _seed_speaker_with_activity(session, "Tgt", minutes_rank=1, human=True)
        _seed_candidate(session, source, "bio", "Draft recorded under the source.")
        do_merge(session, source, target)
        session.commit()
    legacy = _make_client(session_factory, tmp_path, speakers_enabled=False)
    page = legacy.get("/speakers")
    assert page.status_code == 200
    assert "Draft recorded under the source." not in page.text
    modern = _make_client(session_factory, tmp_path, speakers_enabled=True)
    profile = modern.get(f"/speakers/{target}")
    assert "Draft recorded under the source." in profile.text
