"""Inline speaker merge (issue #54, Phase A) end to end.

Real Postgres, real templates. Exercises the preview -> confirm -> apply
contract: server-computed impact, optimistic-concurrency 409, idempotent replay,
run-local semantics (no roster merge_speakers), and the enroll-new path.
"""

import html as html_lib
import re
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from tests.integration.test_review_api import (
    CREDS,
    claim_token,
    seed_run,
)
from voxint.api.app import create_app
from voxint.config import Settings
from voxint.db.models import (
    AdjudicationDecision,
    Speaker,
    SpeakerEmbedding,
)

_CSRF_KEY = "review-api-test-csrf-key"


@pytest.fixture()
def media_root(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture()
def client(session_factory: sessionmaker[Session], media_root: Path) -> TestClient:
    settings = Settings(
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        media_root=media_root,
        review_claim_ttl_seconds=600,
        csrf_secret=_CSRF_KEY,
    )
    test_client = TestClient(create_app(settings=settings, session_factory=session_factory))
    test_client.auth = CREDS
    seed_onboarded(session_factory)
    return test_client


def hidden_fields(body: str) -> dict[str, object]:
    """Every hidden input in a confirm fragment; repeated `labels` become a list."""
    fields: dict[str, object] = {}
    labels: list[str] = []
    for match in re.finditer(r'<input type="hidden" name="(\w+)" value="([^"]*)"', body):
        name, value = match.group(1), html_lib.unescape(match.group(2))
        if name == "labels":
            labels.append(value)
        else:
            fields[name] = value
    if labels:
        fields["labels"] = labels
    return fields


HX = {"HX-Request": "true"}


def test_merge_preview_reports_server_computed_impact(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)
        known_id = session.execute(select(Speaker.id)).scalars().one()

    token = claim_token(client, run_id)
    resp = client.post(
        f"/review/{run_id}/merge/preview",
        data={"token": token, "labels": ["S0", "S1"], "target": str(known_id)},
        headers=HX,
    )
    assert resp.status_code == 200
    # Seed: S0 has 2 turns / 2 segments, S1 has 2 turns / 1 segment.
    flat = " ".join(resp.text.split())
    assert "Affects 4 diarization turns and 3 transcript segments" in flat
    assert "Known Voice" in resp.text
    fields = hidden_fields(resp.text)
    assert fields["labels"] == ["S0", "S1"]
    assert fields["speaker_id"] == str(known_id)
    # The optimistic-concurrency token: S0/S1 have no ledger ruling yet -> null.
    assert fields["expected"] == '{"S0": null, "S1": null}'


def test_merge_applies_run_local_no_roster_merge(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)
        known_id = session.execute(select(Speaker.id)).scalars().one()

    token = claim_token(client, run_id)
    preview = client.post(
        f"/review/{run_id}/merge/preview",
        data={"token": token, "labels": ["S0", "S1"], "target": str(known_id)},
        headers=HX,
    )
    fields = hidden_fields(preview.text)
    fields["nonce"] = uuid.uuid4().hex
    apply = client.post(f"/review/{run_id}/merge", data=fields, headers=HX)
    assert apply.status_code == 200
    # Both labels now attribute to the survivor, in this run.
    assert apply.text.count("assigned: Known Voice") == 2
    export = client.get(f"/review/{run_id}/export.txt").text
    assert "Norma" not in export  # S1's old llm-hint name is gone
    with session_factory() as session:
        # Run-local: exactly two assign rulings, NO roster merge, NO new speaker.
        rows = session.execute(
            select(AdjudicationDecision).where(
                AdjudicationDecision.pipeline_run_id == run_id
            )
        ).scalars().all()
        assert len(rows) == 2
        assert all(r.speaker_id == known_id and r.decision == "assign" for r in rows)
        speakers = session.execute(select(Speaker)).scalars().all()
        assert len(speakers) == 1  # only "Known Voice"; nothing merged or created
        assert speakers[0].merged_into_id is None
        assert session.execute(select(SpeakerEmbedding)).scalars().all() == []


def test_merge_enrolls_new_speaker_and_assigns_all(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)

    token = claim_token(client, run_id)
    preview = client.post(
        f"/review/{run_id}/merge/preview",
        data={"token": token, "labels": ["S0", "S1"], "target": "new", "new_name": "Merged Person"},
        headers=HX,
    )
    assert preview.status_code == 200
    assert "Merged Person" in preview.text
    fields = hidden_fields(preview.text)
    assert fields.get("display_name") == "Merged Person"
    fields["nonce"] = uuid.uuid4().hex
    apply = client.post(f"/review/{run_id}/merge", data=fields, headers=HX)
    assert apply.status_code == 200
    assert apply.text.count("assigned: Merged Person") == 2
    with session_factory() as session:
        person = session.execute(
            select(Speaker).where(Speaker.display_name == "Merged Person")
        ).scalars().one()
        rows = session.execute(
            select(AdjudicationDecision).where(
                AdjudicationDecision.pipeline_run_id == run_id
            )
        ).scalars().all()
        assert len(rows) == 2
        assert all(r.speaker_id == person.id for r in rows)
        # Exactly one enrollment centroid, minted from the primary label only.
        embeddings = session.execute(select(SpeakerEmbedding)).scalars().all()
        assert len(embeddings) == 1
        assert embeddings[0].speaker_id == person.id


def test_merge_rejects_stale_preview_with_409(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)
        known_id = session.execute(select(Speaker.id)).scalars().one()

    token = claim_token(client, run_id)
    preview = client.post(
        f"/review/{run_id}/merge/preview",
        data={"token": token, "labels": ["S0", "S1"], "target": str(known_id)},
        headers=HX,
    )
    fields = hidden_fields(preview.text)
    # A ruling lands on S1 AFTER the operator previewed — the confirm is now stale.
    client.post(
        f"/review/{run_id}/labels/S1/decision",
        data={"token": token, "nonce": uuid.uuid4().hex, "action": "exclude"},
        headers=HX,
    )
    fields["nonce"] = uuid.uuid4().hex
    stale = client.post(f"/review/{run_id}/merge", data=fields, headers=HX)
    assert stale.status_code == 409
    with session_factory() as session:
        # The stale merge wrote nothing; S1 is still the exclude it drifted to.
        rows = session.execute(
            select(AdjudicationDecision).where(
                AdjudicationDecision.pipeline_run_id == run_id
            )
        ).scalars().all()
        assert len(rows) == 1
        assert rows[0].diarization_label == "S1" and rows[0].decision == "exclude"


def test_merge_replay_is_idempotent(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)
        known_id = session.execute(select(Speaker.id)).scalars().one()

    token = claim_token(client, run_id)
    preview = client.post(
        f"/review/{run_id}/merge/preview",
        data={"token": token, "labels": ["S0", "S1"], "target": str(known_id)},
        headers=HX,
    )
    fields = hidden_fields(preview.text)
    fields["nonce"] = uuid.uuid4().hex
    first = client.post(f"/review/{run_id}/merge", data=fields, headers=HX)
    assert first.status_code == 200
    # Replaying the exact confirm (same nonce + now-stale expected) must NOT 409:
    # the replay guard sees the child keys already exist and returns the original
    # outcome instead of re-checking the drifted expected-state.
    replay = client.post(f"/review/{run_id}/merge", data=fields, headers=HX)
    assert replay.status_code == 200
    with session_factory() as session:
        count = session.execute(
            select(func.count())
            .select_from(AdjudicationDecision)
            .where(AdjudicationDecision.pipeline_run_id == run_id)
        ).scalar_one()
        assert count == 2  # no duplicate rulings from the replay


def test_merge_shows_distinct_roster_note(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)
        known_id = session.execute(select(Speaker.id)).scalars().one()

    token = claim_token(client, run_id)
    # Give S1 its own distinct roster identity, so the two labels map to two people.
    client.post(
        f"/review/{run_id}/labels/S1/enroll",
        data={"token": token, "nonce": uuid.uuid4().hex, "display_name": "Second Person"},
        headers=HX,
    )
    # And pin S0 to Known Voice with a human ruling.
    client.post(
        f"/review/{run_id}/labels/S0/decision",
        data={
            "token": token,
            "nonce": uuid.uuid4().hex,
            "action": "assign",
            "speaker_id": str(known_id),
        },
        headers=HX,
    )
    preview = client.post(
        f"/review/{run_id}/merge/preview",
        data={"token": token, "labels": ["S0", "S1"], "target": str(known_id)},
        headers=HX,
    )
    assert preview.status_code == 200
    assert "does" in preview.text and "not" in preview.text
    assert "/speakers" in preview.text  # routes the global act to its reviewed home


def test_merge_apply_rejects_archived_survivor(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    import datetime as _dt

    with session_factory() as session:
        run_id = seed_run(session, media_root)
        known_id = session.execute(select(Speaker.id)).scalars().one()

    token = claim_token(client, run_id)
    preview = client.post(
        f"/review/{run_id}/merge/preview",
        data={"token": token, "labels": ["S0", "S1"], "target": str(known_id)},
        headers=HX,
    )
    fields = hidden_fields(preview.text)
    fields["nonce"] = uuid.uuid4().hex
    # The survivor is archived AFTER the preview — the confirm must not resurrect it.
    with session_factory() as session:
        speaker = session.get(Speaker, known_id)
        assert speaker is not None
        speaker.deleted_at = _dt.datetime.now(tz=_dt.UTC)
        session.commit()
    resp = client.post(f"/review/{run_id}/merge", data=fields, headers=HX)
    assert resp.status_code == 400
    assert "active roster identity" in resp.text
    with session_factory() as session:
        assert session.execute(
            select(func.count())
            .select_from(AdjudicationDecision)
            .where(AdjudicationDecision.pipeline_run_id == run_id)
        ).scalar_one() == 0


def test_merge_enroll_new_name_collision_is_400(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)  # seeds an active "Known Voice"

    token = claim_token(client, run_id)
    preview = client.post(
        f"/review/{run_id}/merge/preview",
        data={"token": token, "labels": ["S0", "S1"], "target": "new", "new_name": "Known Voice"},
        headers=HX,
    )
    assert preview.status_code == 200  # preview never enrolls
    fields = hidden_fields(preview.text)
    fields["nonce"] = uuid.uuid4().hex
    resp = client.post(f"/review/{run_id}/merge", data=fields, headers=HX)
    assert resp.status_code == 400
    assert "already exists" in resp.text


def test_merge_validation_and_auth(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)
        known_id = session.execute(select(Speaker.id)).scalars().one()

    token = claim_token(client, run_id)
    # Fewer than two labels.
    assert client.post(
        f"/review/{run_id}/merge/preview",
        data={"token": token, "labels": ["S0"], "target": str(known_id)},
        headers=HX,
    ).status_code == 400
    # Unknown label.
    assert client.post(
        f"/review/{run_id}/merge/preview",
        data={"token": token, "labels": ["S0", "NOPE"], "target": str(known_id)},
        headers=HX,
    ).status_code == 400
    # Enroll-new with an empty name.
    assert client.post(
        f"/review/{run_id}/merge/preview",
        data={"token": token, "labels": ["S0", "S1"], "target": "new", "new_name": "  "},
        headers=HX,
    ).status_code == 400
    # A stale claim token is refused before any read.
    fresh = claim_token(client, run_id)
    assert fresh != token
    assert client.post(
        f"/review/{run_id}/merge/preview",
        data={"token": token, "labels": ["S0", "S1"], "target": str(known_id)},
        headers=HX,
    ).status_code == 409
