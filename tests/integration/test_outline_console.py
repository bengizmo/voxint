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
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.csrf import CSRF_CLAIM, mint_csrf_token
from voxint.config import Settings
from voxint.db.models import RunAssetKind
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



# Tests for outline island props on the interactive transcript page were removed
# in issue #158 (outline data now delivered via the media editor island).



# Tests for retired-page outline rendering (workbench, interactive transcript
# stepper) were removed in issue #158; the outline is now served by the media
# editor island.



# test_readonly_transcript_outline_hidden_when_gated_off was removed in
# issue #158 (interactive transcript is retired; gating tested via editor).
