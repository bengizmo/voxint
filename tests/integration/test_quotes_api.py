"""The quote board API end to end (issue #338, Phase 6).

Pins the wiring the pure CRUD module cannot see: CSRF on every state change,
the run -> media -> folder -> project resolution (and its honest 422 when the
chain is broken), the duplicate 409 off the DB unique constraint, the CSV
export response, the quote-board props on the project detail page, and the
reachable ondelete behavior (project and segment CASCADE; run teardown is
never blocked by quotes — the run_id CASCADE itself is unreachable while
segments exist, since ``transcript_segments.pipeline_run_id`` has no
ondelete).
"""

import html
import json
import re
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from tests.integration.test_runs_api import make_run
from voxint.api.app import create_app
from voxint.api.csrf import CSRF_QUOTE_MANAGE, CSRF_QUOTE_SAVE, mint_csrf_token
from voxint.config import Settings
from voxint.db.models import (
    MAX_QUOTE_NOTE_CHARS,
    MediaFolder,
    MediaItem,
    PipelineRun,
    Project,
    SavedQuote,
    TranscriptSegment,
)

CREDS = ("reviewer", "s3cret")


@pytest.fixture()
def client(session_factory: sessionmaker[Session], tmp_path: object) -> TestClient:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        media_root=tmp_path,  # type: ignore[arg-type]
        console_projects_enabled=True,
    )
    test_client = TestClient(create_app(settings=settings, session_factory=session_factory))
    test_client.auth = CREDS
    seed_onboarded(session_factory)
    return test_client


def _seed_project_run(
    session_factory: sessionmaker[Session],
    *,
    in_project: bool = True,
) -> tuple[uuid.UUID, uuid.UUID, uuid.UUID | None]:
    """One run with one segment; optionally inside a project folder.

    Returns (run_id, segment_id, project_id or None).
    """
    with session_factory() as session:
        run_id = make_run(
            session,
            segments=[("S0", "the council debated the waterfront rezoning", None)],
        )
        project_id: uuid.UUID | None = None
        if in_project:
            project = Project(name=f"Project {uuid.uuid4()}")
            session.add(project)
            session.flush()
            folder = MediaFolder(path=f"folder-{uuid.uuid4()}", project_id=project.id)
            session.add(folder)
            session.flush()
            run = session.get(PipelineRun, run_id)
            assert run is not None
            media = session.get(MediaItem, run.media_item_id)
            assert media is not None
            media.media_folder_id = folder.id
            project_id = project.id
        segment_id = session.execute(
            select(TranscriptSegment.id).where(
                TranscriptSegment.pipeline_run_id == run_id
            )
        ).scalar_one()
        session.commit()
    return run_id, segment_id, project_id


def _save_form(
    run_id: uuid.UUID,
    segment_id: uuid.UUID,
    client: TestClient,
    **overrides: object,
) -> dict[str, object]:
    form: dict[str, object] = {
        "segment_id": str(segment_id),
        "run_id": str(run_id),
        "search_query": "rezoning",
        "left_context": "the council debated the",
        "hit": "waterfront rezoning",
        "right_context": "for months",
        "media_title": "Council meeting",
        "start_seconds": 12.5,
        "csrf_token": mint_csrf_token(client.app.state.csrf_secret, CSRF_QUOTE_SAVE),
    }
    form.update(overrides)
    return form


def _manage_token(client: TestClient) -> str:
    return mint_csrf_token(client.app.state.csrf_secret, CSRF_QUOTE_MANAGE)


def _quotes(session_factory: sessionmaker[Session]) -> list[SavedQuote]:
    with session_factory() as session:
        return list(session.execute(select(SavedQuote)).scalars())


# ---- save -------------------------------------------------------------------


def test_save_persists_and_resolves_the_project(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, segment_id, project_id = _seed_project_run(session_factory)
    resp = client.post("/quotes", data=_save_form(run_id, segment_id, client))
    assert resp.status_code == 201
    body = resp.json()
    assert body["ok"] is True
    assert body["project_id"] == str(project_id)
    rows = _quotes(session_factory)
    assert len(rows) == 1
    quote = rows[0]
    assert quote.project_id == project_id
    assert quote.run_id == run_id
    assert quote.segment_id == segment_id
    assert quote.hit == "waterfront rezoning"
    assert quote.operator == CREDS[0]


def test_save_duplicate_triple_is_409(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, segment_id, _ = _seed_project_run(session_factory)
    assert (
        client.post("/quotes", data=_save_form(run_id, segment_id, client)).status_code
        == 201
    )
    resp = client.post("/quotes", data=_save_form(run_id, segment_id, client))
    assert resp.status_code == 409
    assert "already saved" in resp.json()["error"]
    assert len(_quotes(session_factory)) == 1
    # A different query on the same segment is a distinct quote, not a dupe.
    resp = client.post(
        "/quotes", data=_save_form(run_id, segment_id, client, search_query="council")
    )
    assert resp.status_code == 201
    assert len(_quotes(session_factory)) == 2


def test_save_outside_a_project_is_an_honest_422(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, segment_id, _ = _seed_project_run(session_factory, in_project=False)
    resp = client.post("/quotes", data=_save_form(run_id, segment_id, client))
    assert resp.status_code == 422
    assert "not assigned to a project" in resp.json()["error"]
    assert _quotes(session_factory) == []


def test_save_segment_from_another_run_is_422(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, _, _ = _seed_project_run(session_factory)
    _, other_segment, _ = _seed_project_run(session_factory)
    resp = client.post("/quotes", data=_save_form(run_id, other_segment, client))
    assert resp.status_code == 422
    assert "does not belong" in resp.json()["error"]
    assert _quotes(session_factory) == []


def test_save_unknown_segment_is_422(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, _, _ = _seed_project_run(session_factory)
    resp = client.post("/quotes", data=_save_form(run_id, uuid.uuid4(), client))
    assert resp.status_code == 422
    assert "re-processed" in resp.json()["error"]


def test_save_overlong_note_is_422(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, segment_id, _ = _seed_project_run(session_factory)
    resp = client.post(
        "/quotes",
        data=_save_form(
            run_id, segment_id, client, note="x" * (MAX_QUOTE_NOTE_CHARS + 1)
        ),
    )
    assert resp.status_code == 422
    assert _quotes(session_factory) == []


def test_save_without_csrf_is_403(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, segment_id, _ = _seed_project_run(session_factory)
    form = _save_form(run_id, segment_id, client)
    del form["csrf_token"]
    assert client.post("/quotes", data=form).status_code == 403
    assert _quotes(session_factory) == []


def test_manage_endpoints_without_csrf_are_403(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, segment_id, _ = _seed_project_run(session_factory)
    quote_id = client.post(
        "/quotes", data=_save_form(run_id, segment_id, client)
    ).json()["id"]
    # Missing token, and a token minted for the WRONG action, both refuse.
    wrong_action = mint_csrf_token(client.app.state.csrf_secret, CSRF_QUOTE_SAVE)
    assert client.request("DELETE", f"/quotes/{quote_id}").status_code == 403
    assert (
        client.request(
            "DELETE", f"/quotes/{quote_id}", data={"csrf_token": wrong_action}
        ).status_code
        == 403
    )
    assert client.patch(f"/quotes/{quote_id}", data={"note": "x"}).status_code == 403
    assert (
        client.patch(
            f"/quotes/{quote_id}",
            data={"note": "x", "csrf_token": wrong_action},
        ).status_code
        == 403
    )
    rows = _quotes(session_factory)
    assert len(rows) == 1
    assert rows[0].note is None


# ---- delete + note ----------------------------------------------------------


def test_delete_removes_the_row(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, segment_id, _ = _seed_project_run(session_factory)
    quote_id = client.post(
        "/quotes", data=_save_form(run_id, segment_id, client)
    ).json()["id"]
    resp = client.request(
        "DELETE", f"/quotes/{quote_id}", data={"csrf_token": _manage_token(client)}
    )
    assert resp.status_code == 200
    assert _quotes(session_factory) == []
    # A second delete reports the truth.
    resp = client.request(
        "DELETE", f"/quotes/{quote_id}", data={"csrf_token": _manage_token(client)}
    )
    assert resp.status_code == 404


def test_patch_note_sets_and_clears(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, segment_id, _ = _seed_project_run(session_factory)
    quote_id = client.post(
        "/quotes", data=_save_form(run_id, segment_id, client)
    ).json()["id"]
    resp = client.patch(
        f"/quotes/{quote_id}",
        data={"note": "  key evidence  ", "csrf_token": _manage_token(client)},
    )
    assert resp.status_code == 200
    assert resp.json()["note"] == "key evidence"
    resp = client.patch(
        f"/quotes/{quote_id}", data={"csrf_token": _manage_token(client)}
    )
    assert resp.status_code == 200
    assert resp.json()["note"] is None
    assert _quotes(session_factory)[0].note is None


def test_patch_unknown_quote_is_404(client: TestClient) -> None:
    resp = client.patch(
        f"/quotes/{uuid.uuid4()}",
        data={"note": "x", "csrf_token": _manage_token(client)},
    )
    assert resp.status_code == 404


# ---- CSV export -------------------------------------------------------------


def test_csv_export_contains_the_quote(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, segment_id, project_id = _seed_project_run(session_factory)
    assert (
        client.post(
            "/quotes",
            data=_save_form(run_id, segment_id, client, note="=SUM(A1)"),
        ).status_code
        == 201
    )
    resp = client.get(f"/projects/{project_id}/quotes/csv")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    assert "attachment" in resp.headers["content-disposition"]
    lines = resp.text.splitlines()
    assert lines[0].startswith("search_query,")
    assert "waterfront rezoning" in lines[1]
    # Formula-injection guard: the note ships neutralized.
    assert "'=SUM(A1)" in lines[1]


def test_csv_export_unknown_project_is_404(client: TestClient) -> None:
    assert client.get(f"/projects/{uuid.uuid4()}/quotes/csv").status_code == 404


# ---- project page read path -------------------------------------------------


def test_project_page_renders_the_quote_board_props(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, segment_id, project_id = _seed_project_run(session_factory)
    assert (
        client.post("/quotes", data=_save_form(run_id, segment_id, client)).status_code
        == 201
    )
    resp = client.get(f"/projects/{project_id}")
    assert resp.status_code == 200
    match = re.search(
        r'data-island="quote-board"\s+data-props="([^"]+)"', resp.text
    )
    assert match, "the quote-board island must render on the project page"
    # The template writes tojson|replace('"', '&quot;') and Jinja autoescape
    # then escapes the ampersands again, so the attribute needs two passes.
    props = json.loads(html.unescape(html.unescape(match.group(1))))
    assert props["total"] == 1
    assert props["projectId"] == str(project_id)
    assert props["csrfToken"], "the board needs a manage token to mutate"
    (quote,) = props["quotes"]
    assert quote["hit"] == "waterfront rezoning"
    assert quote["run_id"] == str(run_id)


# ---- ondelete behavior ------------------------------------------------------


def _saved_quote_count(session_factory: sessionmaker[Session]) -> int:
    return len(_quotes(session_factory))


def test_project_delete_cascades_to_quotes(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, segment_id, project_id = _seed_project_run(session_factory)
    assert (
        client.post("/quotes", data=_save_form(run_id, segment_id, client)).status_code
        == 201
    )
    with session_factory() as session:
        project = session.get(Project, project_id)
        assert project is not None
        session.delete(project)
        session.commit()
    assert _saved_quote_count(session_factory) == 0


def test_run_delete_is_not_blocked_by_quotes(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    # transcript_segments.pipeline_run_id has no ondelete, so a run row can
    # only go after its segments; the point pinned here is that saved_quotes
    # never blocks that teardown and leaves no orphans behind.
    run_id, segment_id, _ = _seed_project_run(session_factory)
    assert (
        client.post("/quotes", data=_save_form(run_id, segment_id, client)).status_code
        == 201
    )
    with session_factory() as session:
        segment = session.get(TranscriptSegment, segment_id)
        assert segment is not None
        session.delete(segment)
        run = session.get(PipelineRun, run_id)
        assert run is not None
        session.delete(run)
        session.commit()
    assert _saved_quote_count(session_factory) == 0
    with session_factory() as session:
        assert session.get(PipelineRun, run_id) is None


def test_segment_delete_cascades_to_quotes(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    # Re-transcription deletes and re-inserts segments; the quote follows the
    # segment out (documented CASCADE, product intent still open in #338).
    run_id, segment_id, _ = _seed_project_run(session_factory)
    assert (
        client.post("/quotes", data=_save_form(run_id, segment_id, client)).status_code
        == 201
    )
    with session_factory() as session:
        segment = session.get(TranscriptSegment, segment_id)
        assert segment is not None
        session.delete(segment)
        session.commit()
    assert _saved_quote_count(session_factory) == 0
