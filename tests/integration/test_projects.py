"""The projects pages and their query (Console 2.0 P2b, #153) end to end.

Pins the wiring the pure query cannot see: the area flag gate (404 until
``console_projects_enabled``, auth first), project creation (validation, the
unique-name 409, CSRF), the folder-assign flow and its supersede note, and the
derived-speaker precedence (a human adjudication over a grounded cosine match),
which reuses ``resolver.label_states``.
"""

import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.api.csrf import (
    CSRF_PROJECT_ASSIGN,
    CSRF_PROJECT_CORRECTIONS,
    CSRF_PROJECT_VOCAB,
    mint_csrf_token,
)
from voxint.api.projects_query import project_detail
from voxint.config import Settings
from voxint.db.models import (
    AdjudicationDecision,
    DiarizationTurn,
    MediaFolder,
    MediaItem,
    PipelineRun,
    Project,
    RunStatus,
    Speaker,
    SpeakerAssignment,
)

CREDS = ("reviewer", "s3cret")


def _make_client(
    session_factory: sessionmaker[Session],
    tmp_path: Path,
    *,
    projects_enabled: bool,
) -> TestClient:
    settings = Settings(
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        media_root=tmp_path,
        console_projects_enabled=projects_enabled,
    )
    client = TestClient(create_app(settings=settings, session_factory=session_factory))
    client.auth = CREDS
    seed_onboarded(session_factory)
    return client


@pytest.fixture()
def client(session_factory: sessionmaker[Session], tmp_path: Path) -> TestClient:
    return _make_client(session_factory, tmp_path, projects_enabled=True)


def _csrf(client: TestClient, path: str) -> str:
    """The csrf_token minted into the form at ``path`` (also exercises the mint)."""
    html = client.get(path).text
    match = re.search(r'name="csrf_token" value="([^"]+)"', html)
    assert match, f"no csrf_token field at {path}"
    return match.group(1)


def _assign_token(client: TestClient) -> str:
    """A valid folder-assign token, minted directly.

    The assign form (and its csrf field) renders only when a project has an
    assignable folder, so the refusal cases have no form to scrape; mint the
    token under the same action the route verifies.
    """
    return mint_csrf_token(client.app.state.csrf_secret, CSRF_PROJECT_ASSIGN)


def _make_project(session: Session, name: str = "Election coverage") -> Project:
    project = Project(name=name)
    session.add(project)
    session.flush()
    return project


# ---- the area flag gate -----------------------------------------------------


def test_projects_404s_when_flag_off(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    client = _make_client(session_factory, tmp_path, projects_enabled=False)
    assert client.get("/projects").status_code == 404
    assert client.get(f"/projects/{uuid.uuid4()}").status_code == 404


def test_projects_requires_auth_before_the_gate(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    client = _make_client(session_factory, tmp_path, projects_enabled=False)
    client.auth = None
    assert client.get("/projects").status_code == 401


# ---- list + create ----------------------------------------------------------


def test_empty_list_states_it_honestly(client: TestClient) -> None:
    resp = client.get("/projects")
    assert resp.status_code == 200
    assert "No projects yet" in resp.text


def test_create_project_redirects_and_persists(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    token = _csrf(client, "/projects")
    resp = client.post(
        "/projects",
        data={"name": "  Election coverage  ", "description": "2026 cycle", "csrf_token": token},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"].startswith("/projects/")
    with session_factory() as session:
        rows = session.query(Project).all()
    assert len(rows) == 1
    assert rows[0].name == "Election coverage"  # trimmed
    assert rows[0].description == "2026 cycle"


def test_create_project_rejects_blank_name(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    token = _csrf(client, "/projects")
    resp = client.post(
        "/projects", data={"name": "   ", "csrf_token": token}
    )
    assert resp.status_code == 400
    assert "name is required" in resp.text.lower()
    with session_factory() as session:
        assert session.query(Project).count() == 0


def test_create_project_duplicate_name_is_a_friendly_409(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        _make_project(session, "Town halls")
        session.commit()
    token = _csrf(client, "/projects")
    resp = client.post("/projects", data={"name": "Town halls", "csrf_token": token})
    assert resp.status_code == 409
    assert "already exists" in resp.text
    with session_factory() as session:
        assert session.query(Project).count() == 1


def test_create_project_without_csrf_is_403(client: TestClient) -> None:
    resp = client.post("/projects", data={"name": "No token"})
    assert resp.status_code == 403


# ---- detail + assign --------------------------------------------------------


def test_detail_404s_for_a_missing_project(client: TestClient) -> None:
    assert client.get(f"/projects/{uuid.uuid4()}").status_code == 404


def test_detail_lists_member_folders(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        project = _make_project(session)
        folder = MediaFolder(
            path="interviews", domain_pack="interview", project_id=project.id
        )
        session.add(folder)
        session.flush()
        session.add(MediaItem(source_path="interviews/a.wav", media_folder_id=folder.id))
        pid = project.id
        session.commit()
    resp = client.get(f"/projects/{pid}")
    assert resp.status_code == 200
    assert "interviews" in resp.text
    assert "interview" in resp.text  # the pack name


def test_assign_folder_moves_it_and_warns_about_the_pack(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        project = _make_project(session)
        folder = MediaFolder(path="courtroom", domain_pack="legal")  # unassigned
        session.add(folder)
        session.flush()
        pid, fid = project.id, folder.id
        session.commit()

    token = _assign_token(client)
    resp = client.post(
        f"/projects/{pid}/folders",
        data={"folder_id": str(fid), "csrf_token": token},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == f"/projects/{pid}?assigned={fid}"
    with session_factory() as session:
        assert session.get(MediaFolder, fid).project_id == pid

    # The confirmation banner names the pack precedence (no project config yet).
    followed = client.get(resp.headers["location"])
    assert "Assigned" in followed.text
    assert "legal" in followed.text
    assert "keeps applying until this project sets its own" in followed.text


def test_assign_already_assigned_folder_is_refused(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        owner = _make_project(session, "Owner")
        other = _make_project(session, "Other")
        folder = MediaFolder(path="taken", project_id=owner.id)
        session.add(folder)
        session.flush()
        other_id, fid = other.id, folder.id
        session.commit()
    token = _assign_token(client)
    resp = client.post(
        f"/projects/{other_id}/folders",
        data={"folder_id": str(fid), "csrf_token": token},
    )
    assert resp.status_code == 400
    assert "already assigned" in resp.text
    with session_factory() as session:
        # Unmoved.
        assert session.get(MediaFolder, fid).project_id != other_id


def test_assign_nonexistent_folder_is_refused(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        project = _make_project(session)
        pid = project.id
        session.commit()
    token = _assign_token(client)
    resp = client.post(
        f"/projects/{pid}/folders",
        data={"folder_id": str(uuid.uuid4()), "csrf_token": token},
    )
    assert resp.status_code == 400
    assert "no longer exists" in resp.text


def test_assign_garbage_folder_id_is_refused(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        pid = _make_project(session).id
        session.commit()
    token = _assign_token(client)
    resp = client.post(
        f"/projects/{pid}/folders",
        data={"folder_id": "not-a-uuid", "csrf_token": token},
    )
    assert resp.status_code == 400


def test_assign_to_missing_project_is_404(client: TestClient) -> None:
    token = _assign_token(client)
    resp = client.post(
        f"/projects/{uuid.uuid4()}/folders",
        data={"folder_id": str(uuid.uuid4()), "csrf_token": token},
    )
    assert resp.status_code == 404


# ---- derived speakers (the load-bearing precedence) -------------------------


def _seed_run_with_label(
    session: Session, project: Project, *, folder_path: str, label: str
) -> PipelineRun:
    folder = MediaFolder(path=folder_path, project_id=project.id)
    session.add(folder)
    session.flush()
    media = MediaItem(source_path=f"{folder_path}/rec.wav", media_folder_id=folder.id)
    session.add(media)
    session.flush()
    run = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
    session.add(run)
    session.flush()
    session.add(
        DiarizationTurn(
            pipeline_run_id=run.id,
            turn_index=0,
            start_seconds=0.0,
            end_seconds=5.0,
            label=label,
            # A turn carries an embedding XOR a skip_reason; the resolver only
            # needs the label to exist, so mark it skipped rather than synthesize
            # a vector.
            skip_reason="test-seed",
        )
    )
    return run


def test_derived_speaker_from_grounded_cosine(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        speaker = Speaker(display_name="Ada")
        session.add(speaker)
        project = _make_project(session)
        session.flush()
        run = _seed_run_with_label(
            session, project, folder_path="calls", label="SPEAKER_00"
        )
        session.add(
            SpeakerAssignment(
                pipeline_run_id=run.id,
                diarization_label="SPEAKER_00",
                speaker_id=speaker.id,
                method="cosine",
                confidence=0.9,
                grounded=True,
            )
        )
        pid, sid = project.id, speaker.id
        session.commit()

        detail = project_detail(session, pid)
    assert detail is not None
    assert [(s.id, s.name, s.run_count) for s in detail.speakers] == [(sid, "Ada", 1)]


def test_human_decision_overrides_grounded_cosine(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        machine = Speaker(display_name="Machine guess")
        human = Speaker(display_name="Human truth")
        session.add_all([machine, human])
        project = _make_project(session)
        session.flush()
        run = _seed_run_with_label(
            session, project, folder_path="calls", label="SPEAKER_00"
        )
        # A grounded cosine proposes `machine`...
        session.add(
            SpeakerAssignment(
                pipeline_run_id=run.id,
                diarization_label="SPEAKER_00",
                speaker_id=machine.id,
                method="cosine",
                confidence=0.9,
                grounded=True,
            )
        )
        # ...but the operator ruled it is `human` (label scope).
        session.add(
            AdjudicationDecision(
                pipeline_run_id=run.id,
                diarization_label="SPEAKER_00",
                decision="assign",
                speaker_id=human.id,
                operator="reviewer",
                idempotency_key=str(uuid.uuid4()),
            )
        )
        pid, human_id = project.id, human.id
        session.commit()

        detail = project_detail(session, pid)
    assert detail is not None
    # The human decision wins over the grounded cosine.
    assert [(s.id, s.name) for s in detail.speakers] == [(human_id, "Human truth")]


def test_unresolved_labels_contribute_no_speaker(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        project = _make_project(session)
        session.flush()
        # A completed run with a label but no assignment or decision: the label
        # resolves as UNRESOLVED (no speaker), so it must not inflate the roster.
        _seed_run_with_label(
            session, project, folder_path="calls", label="SPEAKER_00"
        )
        pid = project.id
        session.commit()
        detail = project_detail(session, pid)
    assert detail is not None
    assert detail.speakers == []


def test_archived_and_unfinished_runs_contribute_no_speakers(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        speaker = Speaker(display_name="Ghost")
        session.add(speaker)
        project = _make_project(session)
        session.flush()
        # A completed-but-archived run and a still-queued run: neither counts.
        archived = _seed_run_with_label(
            session, project, folder_path="a", label="SPEAKER_00"
        )
        archived.status = RunStatus.COMPLETED.value
        archived.archived_at = datetime.now(UTC)
        queued = _seed_run_with_label(
            session, project, folder_path="b", label="SPEAKER_00"
        )
        queued.status = RunStatus.QUEUED.value
        for run in (archived, queued):
            session.add(
                SpeakerAssignment(
                    pipeline_run_id=run.id,
                    diarization_label="SPEAKER_00",
                    speaker_id=speaker.id,
                    method="cosine",
                    confidence=0.9,
                    grounded=True,
                )
            )
        pid = project.id
        session.commit()
        detail = project_detail(session, pid)
    assert detail is not None
    assert detail.speakers == []


# ---- project config editors (issue #153, P2a precedence freeze) -------------


def _vocab_token(client: TestClient) -> str:
    return mint_csrf_token(client.app.state.csrf_secret, CSRF_PROJECT_VOCAB)


def _corr_token(client: TestClient) -> str:
    return mint_csrf_token(client.app.state.csrf_secret, CSRF_PROJECT_CORRECTIONS)


def test_project_detail_renders_config_editors(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        pid = _make_project(session).id
        session.commit()
    html = client.get(f"/projects/{pid}").text
    assert "Project vocabulary" in html
    assert "Project corrections" in html
    assert 'data-island="corrections-editor"' in html


def test_set_project_vocabulary_persists_override(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        pid = _make_project(session).id
        session.commit()
    resp = client.post(
        f"/projects/{pid}/vocabulary",
        data={"mode": "set", "vocabulary": "Alpha\nBeta", "csrf_token": _vocab_token(client)},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    with session_factory() as session:
        assert session.get(Project, pid).vocabulary == ["Alpha", "Beta"]


def test_set_project_vocabulary_empty_set_is_explicit_none(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        pid = _make_project(session).id
        session.commit()
    resp = client.post(
        f"/projects/{pid}/vocabulary",
        data={"mode": "set", "vocabulary": "   ", "csrf_token": _vocab_token(client)},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    with session_factory() as session:
        # An explicit empty list — distinct from inherit (NULL) — and it wins.
        assert session.get(Project, pid).vocabulary == []


def test_set_project_vocabulary_inherit_clears_to_null(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        project = _make_project(session)
        project.vocabulary = ["Was", "Set"]
        pid = project.id
        session.commit()
    resp = client.post(
        f"/projects/{pid}/vocabulary",
        data={"mode": "inherit", "vocabulary": "ignored", "csrf_token": _vocab_token(client)},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    with session_factory() as session:
        assert session.get(Project, pid).vocabulary is None


def test_set_project_vocabulary_rejects_overlong_term(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        pid = _make_project(session).id
        session.commit()
    resp = client.post(
        f"/projects/{pid}/vocabulary",
        data={"mode": "set", "vocabulary": "x" * 121, "csrf_token": _vocab_token(client)},
    )
    assert resp.status_code == 422
    with session_factory() as session:
        assert session.get(Project, pid).vocabulary is None  # nothing written


def test_set_project_vocabulary_rejects_unknown_mode(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    # An unknown mode must NOT fall through to "set" (which could replace inherited
    # config with an empty list). Refuse with 422 and write nothing.
    with session_factory() as session:
        pid = _make_project(session).id
        session.commit()
    resp = client.post(
        f"/projects/{pid}/vocabulary",
        data={"mode": "inheritt", "vocabulary": "Alpha", "csrf_token": _vocab_token(client)},
    )
    assert resp.status_code == 422
    with session_factory() as session:
        assert session.get(Project, pid).vocabulary is None


def test_set_project_vocabulary_rejected_keeps_set_radio_checked(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    # A rejected "set" re-renders with the "set" radio checked, not reset to the
    # stored "inherit" state, so the operator's attempted mode is not lost.
    with session_factory() as session:
        pid = _make_project(session).id
        session.commit()
    resp = client.post(
        f"/projects/{pid}/vocabulary",
        data={"mode": "set", "vocabulary": "x" * 121, "csrf_token": _vocab_token(client)},
    )
    assert resp.status_code == 422
    assert re.search(r'value="set"[^>]*\schecked', resp.text)
    assert not re.search(r'value="inherit"[^>]*\schecked', resp.text)


def test_set_project_corrections_rejects_unknown_mode(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        pid = _make_project(session).id
        session.commit()
    resp = client.post(
        f"/projects/{pid}/corrections",
        data={"mode": "bogus", "rules": "[]", "csrf_token": _corr_token(client)},
        headers={"accept": "application/json"},
    )
    assert resp.status_code == 422
    assert resp.json()["ok"] is False
    with session_factory() as session:
        assert session.get(Project, pid).corrections is None


def test_project_detail_supersede_copy_is_per_field(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    # A project overriding ONLY vocabulary must not claim it supersedes the folder
    # pack's corrections too (ADR 0002 per-field resolution). The folder-table note
    # names just the overridden field.
    with session_factory() as session:
        project = _make_project(session)
        project.vocabulary = ["Alpha"]  # corrections still inherit
        folder = MediaFolder(
            path="interviews", domain_pack="interview", project_id=project.id
        )
        session.add(folder)
        pid = project.id
        session.commit()
    html = client.get(f"/projects/{pid}").text
    assert "vocabulary superseded by this project" in html
    assert "vocabulary and corrections superseded" not in html


def test_project_vocabulary_requires_csrf(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        pid = _make_project(session).id
        session.commit()
    resp = client.post(
        f"/projects/{pid}/vocabulary",
        data={"mode": "set", "vocabulary": "Alpha", "csrf_token": "forged"},
    )
    assert resp.status_code == 403
    with session_factory() as session:
        assert session.get(Project, pid).vocabulary is None


def test_set_project_corrections_island_persists(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        pid = _make_project(session).id
        session.commit()
    rules = json.dumps(
        [
            {
                "id": "a",
                "match": "foo",
                "replace": "bar",
                "case_sensitive": False,
                "whole_word": False,
            }
        ]
    )
    resp = client.post(
        f"/projects/{pid}/corrections",
        data={"rules": rules, "csrf_token": _corr_token(client)},
        headers={"accept": "application/json"},
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    with session_factory() as session:
        stored = session.get(Project, pid).corrections
        assert stored is not None
        assert [rule["match"] for rule in stored] == ["foo"]


def test_set_project_corrections_inherit_clears_to_null(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        project = _make_project(session)
        project.corrections = [
            {
                "id": "a",
                "match": "foo",
                "replace": "bar",
                "case_sensitive": False,
                "whole_word": False,
            }
        ]
        pid = project.id
        session.commit()
    resp = client.post(
        f"/projects/{pid}/corrections",
        data={"mode": "inherit", "csrf_token": _corr_token(client)},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    with session_factory() as session:
        assert session.get(Project, pid).corrections is None


def test_set_project_corrections_empty_set_is_explicit_none(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        pid = _make_project(session).id
        session.commit()
    resp = client.post(
        f"/projects/{pid}/corrections",
        data={"rules": "[]", "csrf_token": _corr_token(client)},
        headers={"accept": "application/json"},
    )
    assert resp.status_code == 200
    assert resp.json()["corrections"] == []
    with session_factory() as session:
        # Explicitly none (an empty array), distinct from inherit (NULL).
        assert session.get(Project, pid).corrections == []


def test_set_project_corrections_invalid_rule_is_422(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        pid = _make_project(session).id
        session.commit()
    rules = json.dumps([{"id": "a", "match": "", "replace": "x"}])  # empty match
    resp = client.post(
        f"/projects/{pid}/corrections",
        data={"rules": rules, "csrf_token": _corr_token(client)},
        headers={"accept": "application/json"},
    )
    assert resp.status_code == 422
    assert resp.json()["ok"] is False
    with session_factory() as session:
        assert session.get(Project, pid).corrections is None  # nothing written


def test_project_corrections_requires_csrf(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        pid = _make_project(session).id
        session.commit()
    resp = client.post(
        f"/projects/{pid}/corrections",
        data={"rules": "[]", "csrf_token": "forged"},
        headers={"accept": "application/json"},
    )
    assert resp.status_code == 403
