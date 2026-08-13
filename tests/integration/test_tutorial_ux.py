"""The guided-tutorial UX (issue #3, slice 6) end to end against real Postgres.

Server-rendered ``?tutorial=<step>`` banners are injected into the EXISTING
run/review/workbench pages ONLY for the real seeded tutorial run and only on the
step's bound page; a spoofed param, the wrong page, or a non-tutorial run shows
nothing. Claiming the tutorial run continues the walkthrough on RUN IDENTITY (any
claim control, not a hidden form field), the adjudicate→export link preserves the
claim token, htmx label swaps never clobber the banner, completion is an explicit
idempotent CSRF-guarded POST, and replay clears completion without touching prior
rulings. The setup finish launches the tutorial only after onboarding commits.
"""

import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.adjudication.resolver import adjudication_queue, label_states
from voxint.api.app import create_app
from voxint.api.csrf import CSRF_CLAIM, CSRF_SETTINGS, CSRF_SETUP, mint_csrf_token
from voxint.app_settings import get_app_settings
from voxint.config import Settings
from voxint.db.models import MediaItem, PipelineRun, RunStatus
from voxint.db.session import session_scope
from voxint.tutorial.seed import seed_tutorial_run

CREDS = ("reviewer", "s3cret")
_CSRF_KEY = "tutorial-ux-test-csrf-key"  # low-entropy; a known secret lets tests mint

# Labels of the committed three-speaker sample (see tutorial/assets/utterance.json).
GROUNDED_LABEL = "SPEAKER_00"  # grounded cosine → Jordan Rivera (Tutorial)
HEARD_LABEL = "SPEAKER_01"  # unresolved, heard name "Priya"
UNRESOLVED_LABEL = "SPEAKER_02"  # purely unresolved

# The rendered walkthrough banner carries this aria-label — a marker that appears
# only in the fragment partial, never in the base stylesheet or the settings-page
# "Tutorial complete" celebration, so its presence/absence is an exact banner test.
BANNER = 'aria-label="Guided tutorial"'


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        media_root=tmp_path,
        csrf_secret=_CSRF_KEY,
        review_claim_ttl_seconds=600,
    )


def _client(
    session_factory: sessionmaker[Session], settings: Settings, *, onboarded: bool
) -> TestClient:
    client = TestClient(create_app(settings=settings, session_factory=session_factory))
    client.auth = CREDS
    if onboarded:
        seed_onboarded(session_factory)
    return client


@pytest.fixture()
def client(
    session_factory: sessionmaker[Session], settings: Settings
) -> TestClient:
    return _client(session_factory, settings, onboarded=True)


@pytest.fixture()
def tutorial_run_id(
    session_factory: sessionmaker[Session], settings: Settings
) -> uuid.UUID:
    with session_scope(session_factory) as session:
        return seed_tutorial_run(
            session, media_root=settings.media_root, settings=settings
        )


def _plain_completed_run(session_factory: sessionmaker[Session]) -> uuid.UUID:
    """A minimal COMPLETED run that is NOT the tutorial run."""
    with session_scope(session_factory) as session:
        media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav", media_type="audio/wav")
        session.add(media)
        session.flush()
        run = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
        session.add(run)
        session.flush()
        return run.id


def _claim(client: TestClient, run_id: uuid.UUID) -> str:
    """Claim a run the ordinary way (no tutorial-specific form field).

    Tutorial continuation is keyed on run identity in the claim route, so an
    ordinary claim of the tutorial run continues the walkthrough — that IS the
    behaviour under test, not a special form field.
    """
    resp = client.post(
        f"/review/{run_id}/claim",
        data={"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_CLAIM)},
        follow_redirects=False,
    )
    assert resp.status_code == 303, resp.text
    return resp.headers["location"]


def _token(location: str) -> str:
    return location.split("token=")[1].split("&")[0]


def _decide(client: TestClient, run_id: uuid.UUID, label: str, token: str, action: str) -> None:
    resp = client.post(
        f"/review/{run_id}/labels/{label}/decision",
        data={"token": token, "nonce": uuid.uuid4().hex, "action": action},
        follow_redirects=False,
    )
    assert resp.status_code in (200, 303), resp.text


# ---------------------------------------------------------------- banner gating


def test_run_step_banner_renders_on_tutorial_run(
    client: TestClient, tutorial_run_id: uuid.UUID
) -> None:
    resp = client.get(f"/runs/{tutorial_run_id}?tutorial=run")
    assert resp.status_code == 200
    assert BANNER in resp.text
    assert "Your tutorial run" in resp.text
    assert "step 1 of 4" in resp.text
    # Next step points at the review console in tutorial mode.
    assert "/review?tutorial=review" in resp.text


def test_no_banner_without_param(
    client: TestClient, tutorial_run_id: uuid.UUID
) -> None:
    resp = client.get(f"/runs/{tutorial_run_id}")
    assert resp.status_code == 200
    assert BANNER not in resp.text


def test_no_banner_on_non_tutorial_run_even_with_param(
    client: TestClient,
    tutorial_run_id: uuid.UUID,
    session_factory: sessionmaker[Session],
) -> None:
    other = _plain_completed_run(session_factory)
    resp = client.get(f"/runs/{other}?tutorial=run")
    assert resp.status_code == 200
    assert BANNER not in resp.text


def test_wrong_step_for_page_renders_no_banner(
    client: TestClient, tutorial_run_id: uuid.UUID
) -> None:
    # adjudicate is bound to the workbench, not the run-detail page.
    resp = client.get(f"/runs/{tutorial_run_id}?tutorial=adjudicate")
    assert resp.status_code == 200
    assert BANNER not in resp.text


def test_unknown_param_is_ignored_not_422(
    client: TestClient, tutorial_run_id: uuid.UUID
) -> None:
    resp = client.get(f"/runs/{tutorial_run_id}?tutorial=bogus")
    assert resp.status_code == 200
    assert BANNER not in resp.text


def test_done_step_on_run_page_renders_no_banner(
    client: TestClient, tutorial_run_id: uuid.UUID
) -> None:
    # DONE binds to the Settings page; on a run page it is a clean page-mismatch
    # (no banner), never a KeyError on the step→page map.
    resp = client.get(f"/runs/{tutorial_run_id}?tutorial=done")
    assert resp.status_code == 200
    assert BANNER not in resp.text


def test_review_step_banner_offers_tutorial_claim(
    client: TestClient, tutorial_run_id: uuid.UUID
) -> None:
    resp = client.get("/review?tutorial=review")
    assert resp.status_code == 200
    assert BANNER in resp.text
    assert "Claim the tutorial run" in resp.text
    # The banner's own claim form targets the exact tutorial run; continuation into
    # adjudicate mode is decided by the claim route on run identity (asserted in
    # test_claim_of_tutorial_run_lands_on_adjudicate), not a hidden form field.
    assert f'action="/review/{tutorial_run_id}/claim"' in resp.text


def test_review_banner_present_even_when_run_resolved(
    client: TestClient,
    tutorial_run_id: uuid.UUID,
    session_factory: sessionmaker[Session],
) -> None:
    # Resolve every label, so the run leaves the queue; the review-step banner must
    # still offer to claim it (it does not depend on a queue row existing).
    location = _claim(client, tutorial_run_id)
    token = _token(location)
    _decide(client, tutorial_run_id, HEARD_LABEL, token, "exclude")
    _decide(client, tutorial_run_id, UNRESOLVED_LABEL, token, "exclude")
    with session_scope(session_factory) as session:
        assert not any(e.run_id == tutorial_run_id for e in adjudication_queue(session))
    resp = client.get("/review?tutorial=review")
    assert BANNER in resp.text
    assert f'action="/review/{tutorial_run_id}/claim"' in resp.text


# ------------------------------------------------------ claim continuity + token


def test_claim_of_tutorial_run_lands_on_adjudicate(
    client: TestClient, tutorial_run_id: uuid.UUID
) -> None:
    # ANY ordinary claim of the (active) tutorial run continues the walkthrough —
    # no special form field — so a user can't fall out by using the queue's own
    # Review button instead of the banner's button.
    location = _claim(client, tutorial_run_id)
    assert "token=" in location
    assert "tutorial=adjudicate" in location


def test_claim_of_completed_tutorial_has_no_suffix(
    client: TestClient, tutorial_run_id: uuid.UUID
) -> None:
    # Once the tutorial is completed, the run claims normally (no banner) — the
    # continuation is gated on the walkthrough still being active.
    client.post(
        "/settings/tutorial/complete",
        data={"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_SETTINGS)},
        follow_redirects=False,
    )
    location = _claim(client, tutorial_run_id)
    assert "token=" in location
    assert "tutorial=" not in location


def test_claim_of_non_tutorial_run_has_no_suffix(
    client: TestClient,
    tutorial_run_id: uuid.UUID,
    session_factory: sessionmaker[Session],
) -> None:
    other = _plain_completed_run(session_factory)
    location = _claim(client, other)
    # The marker is only appended for the REAL tutorial run — never an arbitrary one.
    assert "tutorial=" not in location


def test_adjudicate_banner_next_link_carries_token(
    client: TestClient, tutorial_run_id: uuid.UUID
) -> None:
    token = _token(_claim(client, tutorial_run_id))
    resp = client.get(f"/review/{tutorial_run_id}?token={token}&tutorial=adjudicate")
    assert resp.status_code == 200
    assert BANNER in resp.text
    assert "Attribute the three voices" in resp.text
    # The onward link keeps the claim token so the export-step page stays writable.
    assert f"token={token}" in resp.text
    assert "tutorial=export" in resp.text


def test_adjudicate_banner_without_token_offers_claim(
    client: TestClient, tutorial_run_id: uuid.UUID
) -> None:
    resp = client.get(f"/review/{tutorial_run_id}?tutorial=adjudicate")
    assert resp.status_code == 200
    assert BANNER in resp.text
    # No live claim → degrade to a claim form (which the route continues on run
    # identity), not a dead next-link that would land on a read-only workbench.
    assert f'action="/review/{tutorial_run_id}/claim"' in resp.text


def test_export_banner_has_export_link_and_finish(
    client: TestClient, tutorial_run_id: uuid.UUID
) -> None:
    token = _token(_claim(client, tutorial_run_id))
    resp = client.get(f"/review/{tutorial_run_id}?token={token}&tutorial=export")
    assert resp.status_code == 200
    assert BANNER in resp.text
    assert f'href="/review/{tutorial_run_id}/export.txt"' in resp.text
    assert 'action="/settings/tutorial/complete"' in resp.text
    # The claim token must NOT leak into the plaintext export URL.
    assert f"export.txt?token={token}" not in resp.text


# ------------------------------------------------------------------ htmx safety


def test_htmx_decision_fragment_excludes_banner(
    client: TestClient, tutorial_run_id: uuid.UUID
) -> None:
    token = _token(_claim(client, tutorial_run_id))
    resp = client.post(
        f"/review/{tutorial_run_id}/labels/{UNRESOLVED_LABEL}/decision",
        data={"token": token, "nonce": uuid.uuid4().hex, "action": "exclude"},
        headers={"HX-Request": "true"},
        follow_redirects=False,
    )
    assert resp.status_code == 200
    # The swap replaces only #labels; the fragment must not carry the banner (which
    # lives above the body and therefore survives the swap untouched).
    assert BANNER not in resp.text
    assert "label-card" in resp.text  # sanity: it IS the labels fragment


# --------------------------------------------------------------- settings states


def test_settings_unseeded_shows_seed_hint(client: TestClient) -> None:
    resp = client.get("/settings")
    assert resp.status_code == 200
    assert "voxint tutorial seed" in resp.text


def test_settings_seeded_offers_start(
    client: TestClient, tutorial_run_id: uuid.UUID
) -> None:
    resp = client.get("/settings")
    assert resp.status_code == 200
    assert "Start the guided tutorial" in resp.text
    assert f"/runs/{tutorial_run_id}?tutorial=run" in resp.text


def test_settings_completed_offers_replay(
    client: TestClient, tutorial_run_id: uuid.UUID
) -> None:
    resp = client.post(
        "/settings/tutorial/complete",
        data={"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_SETTINGS)},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    page = client.get("/settings")
    assert "Replay tutorial" in page.text


def test_settings_done_spoof_hidden_when_incomplete(
    client: TestClient, tutorial_run_id: uuid.UUID
) -> None:
    # A bookmarked/spoofed ?tutorial=done must NOT show the completion celebration
    # while the tutorial is seeded-but-incomplete — otherwise the page would
    # simultaneously claim completion and offer "Start the guided tutorial".
    resp = client.get("/settings?tutorial=done")
    assert resp.status_code == 200
    assert "You finished the tutorial" not in resp.text
    assert "Start the guided tutorial" in resp.text


def test_settings_done_spoof_hidden_when_unseeded(client: TestClient) -> None:
    resp = client.get("/settings?tutorial=done")
    assert resp.status_code == 200
    assert "You finished the tutorial" not in resp.text
    assert "voxint tutorial seed" in resp.text


# --------------------------------------------------------- completion + replay


def test_complete_sets_timestamp_and_redirects_to_done(
    client: TestClient,
    tutorial_run_id: uuid.UUID,
    session_factory: sessionmaker[Session],
) -> None:
    resp = client.post(
        "/settings/tutorial/complete",
        data={"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_SETTINGS)},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/settings?tutorial=done"
    with session_scope(session_factory) as session:
        row = get_app_settings(session)
        assert row is not None
        assert row.tutorial_completed_at is not None
    # The done celebration renders on the redirect target.
    assert "You finished the tutorial" in client.get("/settings?tutorial=done").text


def test_complete_is_idempotent(
    client: TestClient,
    tutorial_run_id: uuid.UUID,
    session_factory: sessionmaker[Session],
) -> None:
    token = {"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_SETTINGS)}
    client.post("/settings/tutorial/complete", data=token, follow_redirects=False)
    with session_scope(session_factory) as session:
        first = get_app_settings(session).tutorial_completed_at  # type: ignore[union-attr]
    client.post("/settings/tutorial/complete", data=token, follow_redirects=False)
    with session_scope(session_factory) as session:
        second = get_app_settings(session).tutorial_completed_at  # type: ignore[union-attr]
    # A repost preserves the original completion time rather than rewriting it.
    assert first == second


def test_complete_requires_csrf(
    client: TestClient, tutorial_run_id: uuid.UUID
) -> None:
    resp = client.post(
        "/settings/tutorial/complete", data={"csrf_token": "bad"}, follow_redirects=False
    )
    assert resp.status_code == 403


def test_complete_409_when_unseeded(client: TestClient) -> None:
    resp = client.post(
        "/settings/tutorial/complete",
        data={"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_SETTINGS)},
        follow_redirects=False,
    )
    assert resp.status_code == 409


def test_replay_clears_completion_preserves_rulings(
    client: TestClient,
    tutorial_run_id: uuid.UUID,
    session_factory: sessionmaker[Session],
) -> None:
    # Make a ruling, complete, then replay: completion clears, the ruling remains.
    token = _token(_claim(client, tutorial_run_id))
    _decide(client, tutorial_run_id, HEARD_LABEL, token, "exclude")
    client.post(
        "/settings/tutorial/complete",
        data={"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_SETTINGS)},
        follow_redirects=False,
    )
    resp = client.post(
        "/settings/tutorial/replay",
        data={"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_SETTINGS)},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == f"/runs/{tutorial_run_id}?tutorial=run"
    with session_scope(session_factory) as session:
        row = get_app_settings(session)
        assert row is not None
        assert row.tutorial_completed_at is None  # cleared
        # The exclude ruling is preserved — replay is non-destructive.
        states = {s.label: s for s in label_states(session, tutorial_run_id)}
        assert states[HEARD_LABEL].resolution.value == "human_exclude"


def test_replay_requires_csrf(
    client: TestClient, tutorial_run_id: uuid.UUID
) -> None:
    resp = client.post(
        "/settings/tutorial/replay", data={"csrf_token": "bad"}, follow_redirects=False
    )
    assert resp.status_code == 403


def test_replay_409_when_unseeded(client: TestClient) -> None:
    resp = client.post(
        "/settings/tutorial/replay",
        data={"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_SETTINGS)},
        follow_redirects=False,
    )
    assert resp.status_code == 409


# ----------------------------------------------------- launch after onboarding


def test_finish_launches_tutorial_when_seeded(
    session_factory: sessionmaker[Session],
    settings: Settings,
    tutorial_run_id: uuid.UUID,
) -> None:
    client = _client(session_factory, settings, onboarded=False)
    resp = client.post(
        "/setup/finish",
        data={"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_SETUP)},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == f"/runs/{tutorial_run_id}?tutorial=run"


def test_finish_falls_back_to_review_when_unseeded(
    session_factory: sessionmaker[Session], settings: Settings
) -> None:
    client = _client(session_factory, settings, onboarded=False)
    resp = client.post(
        "/setup/finish",
        data={"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_SETUP)},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/review"


# ------------------------------------------------------------- full walkthrough


def test_full_walkthrough_resolves_and_leaves_queue(
    client: TestClient,
    tutorial_run_id: uuid.UUID,
    session_factory: sessionmaker[Session],
) -> None:
    # run → review (claim, continue) → adjudicate the two unresolved voices → the
    # run is fully resolved and drops out of the adjudication queue → finish.
    location = _claim(client, tutorial_run_id)
    assert "tutorial=adjudicate" in location
    token = _token(location)
    _decide(client, tutorial_run_id, HEARD_LABEL, token, "exclude")
    _decide(client, tutorial_run_id, UNRESOLVED_LABEL, token, "exclude")
    with session_scope(session_factory) as session:
        assert not any(e.run_id == tutorial_run_id for e in adjudication_queue(session))
    finish = client.post(
        "/settings/tutorial/complete",
        data={"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_SETTINGS)},
        follow_redirects=False,
    )
    assert finish.status_code == 303
    assert finish.headers["location"] == "/settings?tutorial=done"
