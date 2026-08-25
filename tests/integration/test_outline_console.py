"""Navigable outline (#87) end to end over the real app + Postgres.

Asserts the shared transcript island props carry an honest ``outline`` block
(absent / present / asset-stale / gated) and that the review transcript's JS-off
fallback lists grounded entities as inert text with no jump link. The
group/dedup/order/drop table itself is unit-tested in tests/unit/test_outline.py.
"""

from __future__ import annotations

import html
import json
import re
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy import update
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from tests.integration.test_run_assets import seed_run
from voxint.api.csrf import CSRF_CLAIM, mint_csrf_token
from voxint.config import Settings
from voxint.db.models import RunAssetKind, TranscriptSegment
from voxint.enrichment.run_assets import load_source, record_asset

CREDS = ("reviewer", "s3cret")
_CSRF_KEY = "outline-console-test-csrf-key"
NOW = datetime.now(tz=UTC)

# "Acme Corp" occupies [24, 33) in seed_run's default text; the writer validates
# segment[start:end] == quote verbatim, so the offsets must be exact.
VALID_MENTIONS: dict[str, Any] = {
    "mentions": [
        {
            "surface": "Acme Corp",
            "kind": "organization",
            "occurrences": [
                {"segment_index": 0, "quote": "Acme Corp", "start_char": 24, "end_char": 33}
            ],
        }
    ],
    "diagnostics": {"dropped_unlocatable": 0, "dropped_out_of_run": 0},
}


def _build_client(session_factory: sessionmaker[Session], *, gates_open: bool = True) -> TestClient:
    overrides: dict[str, object] = (
        {"llm_enabled": True, "enrichment_run_assets_enabled": True} if gates_open else {}
    )
    settings = Settings(
        _env_file=None,
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        csrf_secret=_CSRF_KEY,
        **overrides,  # type: ignore[arg-type]
    )
    from voxint.api.app import create_app

    client = TestClient(create_app(settings=settings, session_factory=session_factory))
    client.auth = CREDS
    seed_onboarded(session_factory, llm_enabled=gates_open)
    return client


def _island_props(body: str) -> dict[str, Any]:
    match = re.search(r"data-props='([^']*)'", body)
    assert match is not None, "island mount node missing"
    return json.loads(html.unescape(match.group(1)))


def _outline(client: TestClient, run_id: uuid.UUID) -> dict[str, Any]:
    body = client.get(f"/runs/{run_id}/transcript").text
    return _island_props(body)["outline"]


def _seed_mentions(
    session: Session, run_id: uuid.UUID, *, key: str = "m1", schema_version: int = 1
) -> None:
    record_asset(
        session,
        source=load_source(session, run_id),
        kind=RunAssetKind.ENTITY_MENTIONS,
        payload=VALID_MENTIONS,
        payload_schema_version=schema_version,
        producer="run_assets.llm",
        producer_version="1",
        model="test-model",
        idempotency_key=key,
        started_at=NOW,
        completed_at=NOW + timedelta(seconds=1),
    )
    session.commit()


def _claim_token(client: TestClient, run_id: uuid.UUID) -> str:
    resp = client.post(
        f"/review/{run_id}/claim",
        data={"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_CLAIM)},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    return resp.headers["location"].split("token=")[1]


def test_outline_absent_when_no_asset(session_factory: sessionmaker[Session]) -> None:
    client = _build_client(session_factory)
    with session_factory() as session:
        run_id = seed_run(session)
    outline = _outline(client, run_id)
    assert outline["available"] is False
    assert outline["gated"] is False
    assert outline["mentions"] == []


def test_outline_gated_off(session_factory: sessionmaker[Session]) -> None:
    client = _build_client(session_factory, gates_open=False)
    with session_factory() as session:
        run_id = seed_run(session)
    outline = _outline(client, run_id)
    assert outline["available"] is False
    assert outline["gated"] is True


def test_outline_unknown_schema_version_is_unavailable_not_empty(
    session_factory: sessionmaker[Session],
) -> None:
    # A mentions asset written under a schema version this reader does not know is
    # left unavailable, never parsed through v1 keys into a fabricated "none found".
    client = _build_client(session_factory)
    with session_factory() as session:
        run_id = seed_run(session)
        _seed_mentions(session, run_id, schema_version=2)
    outline = _outline(client, run_id)
    assert outline["available"] is False
    assert outline["mentions"] == []


def test_outline_present_resolves_target(session_factory: sessionmaker[Session]) -> None:
    client = _build_client(session_factory)
    with session_factory() as session:
        run_id = seed_run(session)
        _seed_mentions(session, run_id)
    outline = _outline(client, run_id)
    assert outline["available"] is True
    assert outline["assetStale"] is False
    assert len(outline["mentions"]) == 1
    mention = outline["mentions"][0]
    assert mention["surface"] == "Acme Corp"
    assert mention["kind"] == "organization"
    occ = mention["occurrences"][0]
    assert occ["segmentIndex"] == 0
    assert occ["startSeconds"] == 0.0


def test_outline_asset_stale_after_source_edit(
    session_factory: sessionmaker[Session],
) -> None:
    client = _build_client(session_factory)
    with session_factory() as session:
        run_id = seed_run(session)
        _seed_mentions(session, run_id)
    # Mutate the transcript source after generation: the stored hash no longer
    # matches, so the outline is asset-stale. The segment still exists, so the
    # target still resolves and stays seekable.
    with session_factory() as session:
        session.execute(
            update(TranscriptSegment)
            .where(TranscriptSegment.pipeline_run_id == run_id)
            .values(raw_text="Hello, I am Joanne from Globex Corp.")
        )
        session.commit()
    outline = _outline(client, run_id)
    assert outline["available"] is True
    assert outline["assetStale"] is True
    assert len(outline["mentions"]) == 1


def test_review_fallback_lists_grounded_entities_inert(
    session_factory: sessionmaker[Session],
) -> None:
    client = _build_client(session_factory)
    with session_factory() as session:
        run_id = seed_run(session)
        _seed_mentions(session, run_id)
    token = _claim_token(client, run_id)
    body = client.get(f"/review/{run_id}/transcript", params={"token": token}).text
    # The fallback names the panel and the grounded entity...
    assert 'aria-label="Outline"' in body
    assert "Acme Corp" in body
    # ...states the honest no-JS limitation...
    assert "Jumping to a moment needs JavaScript" in body
    # ...and carries the entity as data, never a seek control (no jump button/link
    # in the fallback outline section).
    outline_section = body.split('aria-label="Outline"', 1)[1].split("</section>", 1)[0]
    assert "outline-jump" not in outline_section
    assert "<button" not in outline_section


def test_readonly_transcript_ships_outline_props(
    session_factory: sessionmaker[Session],
) -> None:
    # The read-only run transcript reuses the SAME shared island props, so a
    # present outline rides to the transcript-player island exactly as it does to
    # the review-stepper — the client renders the OutlinePanel over it (issue #87
    # follow-up). _outline() already reads /runs/{id}/transcript's props.
    client = _build_client(session_factory)
    with session_factory() as session:
        run_id = seed_run(session)
        _seed_mentions(session, run_id)
    outline = _outline(client, run_id)
    assert outline["available"] is True
    assert len(outline["mentions"]) == 1
    assert outline["mentions"][0]["surface"] == "Acme Corp"


def test_readonly_transcript_fallback_lists_grounded_entities_inert(
    session_factory: sessionmaker[Session],
) -> None:
    # The read-only surface renders the SAME inert JS-off outline fallback the
    # review workbench does (shared macro): grounded entities as text, the honest
    # no-JS limitation, and no seek control in the outline section.
    client = _build_client(session_factory)
    with session_factory() as session:
        run_id = seed_run(session)
        _seed_mentions(session, run_id)
    body = client.get(f"/runs/{run_id}/transcript").text
    assert 'aria-label="Outline"' in body
    assert "Acme Corp" in body
    assert "Jumping to a moment needs JavaScript" in body
    outline_section = body.split('aria-label="Outline"', 1)[1].split("</section>", 1)[0]
    assert "outline-jump" not in outline_section
    assert "<button" not in outline_section


def test_readonly_transcript_outline_hidden_when_gated_off(
    session_factory: sessionmaker[Session],
) -> None:
    # Gated off with no asset: the read-only fallback stays hidden (the macro's
    # gated-off guard) rather than nagging a persistent panel onto the page.
    client = _build_client(session_factory, gates_open=False)
    with session_factory() as session:
        run_id = seed_run(session)
    body = client.get(f"/runs/{run_id}/transcript").text
    assert 'aria-label="Outline"' not in body
