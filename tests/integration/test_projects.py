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
from sqlalchemy import select as sa_select
from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.api.csrf import (
    CSRF_PROJECT_ARCHIVE,
    CSRF_PROJECT_ASSIGN,
    CSRF_PROJECT_CORRECTIONS,
    CSRF_PROJECT_CREATE,
    CSRF_PROJECT_DELETE,
    CSRF_PROJECT_LEARNING,
    CSRF_PROJECT_RENAME,
    CSRF_PROJECT_RESTORE,
    CSRF_PROJECT_UNLINK,
    CSRF_PROJECT_VOCAB,
    mint_csrf_token,
)
from voxint.api.projects_query import project_detail
from voxint.config import Settings
from voxint.db.models import (
    AdjudicationDecision,
    CorpusAnalysisArtifact,
    DiarizationTurn,
    LearnedCorrection,
    LearnedCorrectionEvidence,
    MediaFolder,
    MediaItem,
    PipelineRun,
    Project,
    RunStatus,
    SavedQuote,
    Speaker,
    SpeakerAssignment,
    TranscriptSegment,
)
from voxint.projects.lifecycle import (
    ProjectNotArchivedError,
    ProjectNotFoundError,
    delete_project,
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
    assert "VOCABULARY" in html
    assert "AUTO-CORRECT RULES" in html
    assert 'data-island="corrections-editor"' in html


def test_corrections_props_inheriting_true_when_null(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    """Issue #176: the island must know it is inheriting so it cannot
    silently convert NULL into an explicit empty list on save."""
    with session_factory() as session:
        project = _make_project(session)
        project.corrections = None
        pid = project.id
        session.commit()
    html = client.get(f"/projects/{pid}").text
    assert '"inheriting": true' in html


def test_corrections_props_inheriting_false_when_explicit(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        project = _make_project(session)
        project.corrections = []
        pid = project.id
        session.commit()
    html = client.get(f"/projects/{pid}").text
    assert '"inheriting": false' in html


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


# ---- #247 Project rename ---------------------------------------------------


def test_rename_project_success(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        pid = _make_project(session).id
        session.commit()
    token = mint_csrf_token(client.app.state.csrf_secret, CSRF_PROJECT_RENAME)
    resp = client.post(
        f"/projects/{pid}/rename",
        data={"name": "New Name", "csrf_token": token},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    with session_factory() as session:
        project = session.get(Project, pid)
        assert project is not None
        assert project.name == "New Name"


def test_rename_project_empty_name(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        pid = _make_project(session).id
        session.commit()
    token = mint_csrf_token(client.app.state.csrf_secret, CSRF_PROJECT_RENAME)
    resp = client.post(
        f"/projects/{pid}/rename",
        data={"name": "  ", "csrf_token": token},
    )
    assert resp.status_code == 400
    assert "cannot be empty" in resp.text


def test_rename_project_duplicate(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        _make_project(session, name="Taken")
        pid = _make_project(session, name="Mine").id
        session.commit()
    token = mint_csrf_token(client.app.state.csrf_secret, CSRF_PROJECT_RENAME)
    resp = client.post(
        f"/projects/{pid}/rename",
        data={"name": "Taken", "csrf_token": token},
    )
    assert resp.status_code == 409
    assert "already exists" in resp.text


# ---- #247 Folder unlink ---------------------------------------------------


def test_unlink_folder_success(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        project = _make_project(session)
        folder = MediaFolder(path="incoming/test", project_id=project.id)
        session.add(folder)
        session.commit()
        pid, fid = project.id, folder.id
    token = mint_csrf_token(client.app.state.csrf_secret, CSRF_PROJECT_UNLINK)
    resp = client.post(
        f"/projects/{pid}/folders/{fid}/unlink",
        data={"csrf_token": token},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    with session_factory() as session:
        folder = session.get(MediaFolder, fid)
        assert folder is not None
        assert folder.project_id is None


def test_unlink_folder_not_found(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        pid = _make_project(session).id
        session.commit()
    token = mint_csrf_token(client.app.state.csrf_secret, CSRF_PROJECT_UNLINK)
    resp = client.post(
        f"/projects/{pid}/folders/{uuid.uuid4()}/unlink",
        data={"csrf_token": token},
    )
    assert resp.status_code == 404


def test_rename_project_requires_csrf(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        pid = _make_project(session).id
        session.commit()
    resp = client.post(
        f"/projects/{pid}/rename",
        data={"name": "X", "csrf_token": "forged"},
    )
    assert resp.status_code == 403


def test_unlink_folder_requires_csrf(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        project = _make_project(session)
        folder = MediaFolder(path="incoming/csrf-test", project_id=project.id)
        session.add(folder)
        session.commit()
        pid, fid = project.id, folder.id
    resp = client.post(
        f"/projects/{pid}/folders/{fid}/unlink",
        data={"csrf_token": "forged"},
    )
    assert resp.status_code == 403


# ---- learned corrections (#476) ---------------------------------------------


def _learning_token(client: TestClient) -> str:
    return mint_csrf_token(client.app.state.csrf_secret, CSRF_PROJECT_LEARNING)


def test_learning_toggle_on_off(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        project = _make_project(session)
        project.corrections = []
        session.commit()
        pid = project.id
    token = _learning_token(client)
    resp = client.post(
        f"/projects/{pid}/learning",
        data={"enabled": "on", "csrf_token": token},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    with session_factory() as session:
        assert session.get(Project, pid).learn_corrections is True
    resp = client.post(
        f"/projects/{pid}/learning",
        data={"enabled": "off", "csrf_token": token},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    with session_factory() as session:
        assert session.get(Project, pid).learn_corrections is False


def test_learning_toggle_refused_while_inheriting(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        project = _make_project(session)
        session.commit()
        pid = project.id
    token = _learning_token(client)
    resp = client.post(
        f"/projects/{pid}/learning",
        data={"enabled": "on", "csrf_token": token},
    )
    assert resp.status_code == 422
    assert "Set corrections for this project first" in resp.text


def test_inherit_reset_clears_toggle(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        project = _make_project(session)
        project.corrections = []
        project.learn_corrections = True
        session.commit()
        pid = project.id
    corrections_token = mint_csrf_token(
        client.app.state.csrf_secret, CSRF_PROJECT_CORRECTIONS
    )
    client.post(
        f"/projects/{pid}/corrections",
        data={"mode": "inherit", "csrf_token": corrections_token},
        follow_redirects=False,
    )
    with session_factory() as session:
        p = session.get(Project, pid)
        assert p.corrections is None
        assert p.learn_corrections is False


def test_accept_appends_validated_rule(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        project = _make_project(session)
        project.corrections = []
        session.commit()
        lc = LearnedCorrection(
            project_id=project.id,
            match="seer",
            replace="SEER",
            status="suggested",
        )
        session.add(lc)
        session.commit()
        pid, lcid = project.id, lc.id
    token = _learning_token(client)
    resp = client.post(
        f"/projects/{pid}/suggestions/{lcid}",
        data={"decision": "accept", "csrf_token": token},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    with session_factory() as session:
        p = session.get(Project, pid)
        assert len(p.corrections) == 1
        assert p.corrections[0]["match"] == "seer"
        assert p.corrections[0]["replace"] == "SEER"
        assert p.corrections[0]["case_sensitive"] is True
        assert p.corrections[0]["whole_word"] is True
        lc = session.get(LearnedCorrection, lcid)
        assert lc.status == "accepted"
        assert lc.accepted_rule_id == p.corrections[0]["id"]


def test_dismiss_deletes_row(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        project = _make_project(session)
        project.corrections = []
        session.commit()
        lc = LearnedCorrection(
            project_id=project.id,
            match="seer",
            replace="SEER",
            status="suggested",
        )
        session.add(lc)
        session.commit()
        pid, lcid = project.id, lc.id
    token = _learning_token(client)
    resp = client.post(
        f"/projects/{pid}/suggestions/{lcid}",
        data={"decision": "dismiss", "csrf_token": token},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    with session_factory() as session:
        assert session.get(LearnedCorrection, lcid) is None


def test_corrections_save_prunes_orphaned_accepted(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        project = _make_project(session)
        project.corrections = [
            {"id": "rule-1", "match": "seer", "replace": "SEER",
             "case_sensitive": True, "whole_word": True}
        ]
        session.commit()
        lc = LearnedCorrection(
            project_id=project.id,
            match="seer",
            replace="SEER",
            status="accepted",
            accepted_rule_id="rule-1",
        )
        session.add(lc)
        session.commit()
        pid, lcid = project.id, lc.id
    corrections_token = mint_csrf_token(
        client.app.state.csrf_secret, CSRF_PROJECT_CORRECTIONS
    )
    resp = client.post(
        f"/projects/{pid}/corrections",
        data={
            "mode": "set",
            "rules": json.dumps([]),
            "csrf_token": corrections_token,
        },
        headers={"accept": "application/json"},
    )
    assert resp.status_code == 200
    with session_factory() as session:
        assert session.get(LearnedCorrection, lcid) is None


def test_learning_toggle_requires_csrf(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        project = _make_project(session)
        project.corrections = []
        session.commit()
        pid = project.id
    resp = client.post(
        f"/projects/{pid}/learning",
        data={"enabled": "on", "csrf_token": "forged"},
    )
    assert resp.status_code == 403


def test_suggestion_requires_csrf(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        project = _make_project(session)
        project.corrections = []
        session.commit()
        lc = LearnedCorrection(
            project_id=project.id,
            match="seer",
            replace="SEER",
            status="suggested",
        )
        session.add(lc)
        session.commit()
        pid, lcid = project.id, lc.id
    resp = client.post(
        f"/projects/{pid}/suggestions/{lcid}",
        data={"decision": "accept", "csrf_token": "forged"},
    )
    assert resp.status_code == 403


def test_detail_read_model_has_learning_fields(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        project = _make_project(session)
        project.corrections = []
        project.learn_corrections = True
        session.commit()
        pid = project.id
    with session_factory() as session:
        detail = project_detail(session, pid)
        assert detail.learn_corrections is True
        assert detail.suggestions == []
        assert detail.learned_counts == {}


# ---- archive / restore (#477) ------------------------------------------------


def test_overflow_archive_form_scrapes_with_csrf(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    """Active detail page has a '...' menu with an archive form + CSRF token."""
    with session_factory() as session:
        project = _make_project(session)
        pid = project.id
        session.commit()
    page = client.get(f"/projects/{pid}")
    match = re.search(
        rf'action="(/projects/{pid}/archive)"[^>]*>\s*'
        r'<input type="hidden" name="csrf_token" value="([^"]+)"',
        page.text,
        re.S,
    )
    assert match, "overflow menu should carry an archive form with a CSRF token"
    assert "active" in page.text.lower()


def test_archive_wrong_csrf_is_403(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        project = _make_project(session)
        pid = project.id
        session.commit()
    resp = client.post(
        f"/projects/{pid}/archive",
        data={"csrf_token": "wrong"},
        follow_redirects=False,
    )
    assert resp.status_code == 403
    with session_factory() as session:
        assert session.get(Project, pid).archived_at is None


def test_restore_wrong_csrf_is_403(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        project = _make_project(session)
        project.archived_at = datetime.now(UTC)
        pid = project.id
        session.commit()
    resp = client.post(
        f"/projects/{pid}/restore",
        data={"csrf_token": "wrong"},
        follow_redirects=False,
    )
    assert resp.status_code == 403
    with session_factory() as session:
        assert session.get(Project, pid).archived_at is not None


def test_archive_round_trip(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    """Archive then restore: detail page flips chips and controls."""
    with session_factory() as session:
        project = _make_project(session)
        pid = project.id
        session.commit()

    # Archive
    token = mint_csrf_token(client.app.state.csrf_secret, CSRF_PROJECT_ARCHIVE)
    resp = client.post(
        f"/projects/{pid}/archive",
        data={"csrf_token": token},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    with session_factory() as session:
        assert session.get(Project, pid).archived_at is not None

    # Detail page shows archived state
    page = client.get(f"/projects/{pid}")
    assert "archived" in page.text.lower()
    assert "This project is archived" in page.text
    # No rename dropdown
    assert 'action="/projects/' + str(pid) + '/rename"' not in page.text
    # Has restore form
    assert f'/projects/{pid}/restore' in page.text

    # Restore
    token = mint_csrf_token(client.app.state.csrf_secret, CSRF_PROJECT_RESTORE)
    resp = client.post(
        f"/projects/{pid}/restore",
        data={"csrf_token": token},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    with session_factory() as session:
        assert session.get(Project, pid).archived_at is None

    # Back to active
    page = client.get(f"/projects/{pid}")
    assert "This project is archived" not in page.text
    assert f'/projects/{pid}/rename' in page.text


def test_archive_idempotent(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        project = _make_project(session)
        project.archived_at = datetime.now(UTC)
        pid = project.id
        session.commit()
    token = mint_csrf_token(client.app.state.csrf_secret, CSRF_PROJECT_ARCHIVE)
    resp = client.post(
        f"/projects/{pid}/archive",
        data={"csrf_token": token},
        follow_redirects=False,
    )
    assert resp.status_code == 303


def test_restore_idempotent(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        project = _make_project(session)
        pid = project.id
        session.commit()
    token = mint_csrf_token(client.app.state.csrf_secret, CSRF_PROJECT_RESTORE)
    resp = client.post(
        f"/projects/{pid}/restore",
        data={"csrf_token": token},
        follow_redirects=False,
    )
    assert resp.status_code == 303


def test_archive_missing_project_is_404(client: TestClient) -> None:
    token = mint_csrf_token(client.app.state.csrf_secret, CSRF_PROJECT_ARCHIVE)
    resp = client.post(
        f"/projects/{uuid.uuid4()}/archive",
        data={"csrf_token": token},
        follow_redirects=False,
    )
    assert resp.status_code == 404


def test_restore_missing_project_is_404(client: TestClient) -> None:
    token = mint_csrf_token(client.app.state.csrf_secret, CSRF_PROJECT_RESTORE)
    resp = client.post(
        f"/projects/{uuid.uuid4()}/restore",
        data={"csrf_token": token},
        follow_redirects=False,
    )
    assert resp.status_code == 404


@pytest.mark.parametrize(
    "route_suffix,data_extra",
    [
        ("/rename", {"name": "New name"}),
        ("/vocabulary", {"mode": "set", "vocabulary": "term"}),
        ("/vocabulary", {"mode": "inherit"}),
        ("/corrections", {"mode": "set", "rules": "[]"}),
        ("/corrections", {"mode": "inherit"}),
        ("/learning", {"enabled": "on"}),
        ("/learning", {"enabled": "off"}),
    ],
    ids=[
        "rename",
        "vocabulary-set",
        "vocabulary-inherit",
        "corrections-set",
        "corrections-inherit",
        "learning-on",
        "learning-off",
    ],
)
def test_mutation_routes_409_on_archived(
    client: TestClient,
    session_factory: sessionmaker[Session],
    route_suffix: str,
    data_extra: dict,
) -> None:
    """Every config mutation route refuses with 409 on an archived project."""
    with session_factory() as session:
        project = _make_project(session)
        project.archived_at = datetime.now(UTC)
        # Set corrections so learning-on doesn't hit the "set corrections first" branch
        project.corrections = []
        pid = project.id
        session.commit()

    # Determine which CSRF action this route needs
    csrf_map = {
        "/rename": CSRF_PROJECT_RENAME,
        "/vocabulary": CSRF_PROJECT_VOCAB,
        "/corrections": CSRF_PROJECT_CORRECTIONS,
        "/learning": CSRF_PROJECT_LEARNING,
    }
    action = csrf_map[route_suffix]
    token = mint_csrf_token(client.app.state.csrf_secret, action)

    resp = client.post(
        f"/projects/{pid}{route_suffix}",
        data={"csrf_token": token, **data_extra},
        follow_redirects=False,
    )
    assert resp.status_code == 409, f"{route_suffix} should be 409 on archived"

    # Verify no state changed
    with session_factory() as session:
        proj = session.get(Project, pid)
        assert proj.archived_at is not None
        assert proj.name == "Election coverage"


def test_assign_folder_409_on_archived(
    client: TestClient, session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_factory() as session:
        project = _make_project(session)
        project.archived_at = datetime.now(UTC)
        pid = project.id
        folder = MediaFolder(path="test/audio")
        session.add(folder)
        session.flush()
        fid = folder.id
        session.commit()
    token = mint_csrf_token(client.app.state.csrf_secret, CSRF_PROJECT_ASSIGN)
    resp = client.post(
        f"/projects/{pid}/folders",
        data={"csrf_token": token, "folder_id": str(fid)},
        follow_redirects=False,
    )
    assert resp.status_code == 409
    with session_factory() as session:
        assert session.get(MediaFolder, fid).project_id is None


def test_unlink_folder_allowed_on_archived(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    """Unlink is the one mutation allowed on archived projects."""
    with session_factory() as session:
        project = _make_project(session)
        project.archived_at = datetime.now(UTC)
        pid = project.id
        folder = MediaFolder(path="test/audio", project_id=pid)
        session.add(folder)
        session.flush()
        fid = folder.id
        session.commit()
    token = mint_csrf_token(client.app.state.csrf_secret, CSRF_PROJECT_UNLINK)
    resp = client.post(
        f"/projects/{pid}/folders/{fid}/unlink",
        data={"csrf_token": token},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    with session_factory() as session:
        assert session.get(MediaFolder, fid).project_id is None


def test_create_project_with_archived_name_gives_guidance(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        project = _make_project(session, name="Archived One")
        project.archived_at = datetime.now(UTC)
        session.commit()
    token = _csrf(client, "/projects")
    resp = client.post(
        "/projects",
        data={"name": "Archived One", "csrf_token": token},
        follow_redirects=False,
    )
    assert resp.status_code == 409
    assert "Restore it instead" in resp.text


def test_list_page_partitions_active_and_archived(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        _make_project(session, name="Active A")
        _make_project(session, name="Active B")
        archived = _make_project(session, name="Old project")
        archived.archived_at = datetime.now(UTC)
        session.commit()
    page = client.get("/projects")
    assert "2 projects, 1 archived" in page.text
    assert "Archived projects (1)" in page.text.replace("ARCHIVED PROJECTS", "Archived projects")
    # The restore form is inside the archived section
    assert "/restore" in page.text


def test_archived_detail_hides_controls(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    """An archived project with populated config hides edit controls."""
    with session_factory() as session:
        project = _make_project(session)
        project.vocabulary = ["term1", "term2"]
        project.corrections = [{"id": "r1", "match": "foo", "replace": "bar"}]
        project.learn_corrections = True
        project.archived_at = datetime.now(UTC)
        pid = project.id
        folder = MediaFolder(path="test/audio", project_id=pid, domain_pack="legal")
        session.add(folder)
        session.commit()

    page = client.get(f"/projects/{pid}")
    text = page.text

    # Chip
    assert "archived" in text.lower()
    # Banner
    assert "This project is archived" in text
    # No rename
    assert f'/projects/{pid}/rename' not in text
    # No "+ add term"
    assert "+ add term" not in text
    # No edit vocabulary details
    assert "Edit vocabulary" not in text
    # Inactive helper text
    assert "Inactive while archived" in text
    # No learning toggle form
    assert "Learn from my edits" not in text
    # Corrections editor island not mounted (no data-island="corrections-editor")
    assert 'data-island="corrections-editor"' not in text
    # Static rules still shown
    assert "foo" in text and "bar" in text
    # No reset button
    assert "Reset to inherited" not in text
    # No "+ link folder"
    assert "+ link folder" not in text
    # No assign form
    assert "Assign a folder" not in text
    # Supersede note suppressed
    assert "superseded by this project" not in text
    # Unlink stays
    assert "unlink" in text
    # Restore available
    assert f'/projects/{pid}/restore' in text


@pytest.mark.parametrize("decision", ["accept", "dismiss"])
def test_suggestion_409_on_archived(
    client: TestClient,
    session_factory: sessionmaker[Session],
    decision: str,
) -> None:
    """Accept and dismiss refuse with 409 on an archived project."""
    with session_factory() as session:
        project = _make_project(session)
        project.corrections = [{"id": "r1", "match": "foo", "replace": "bar"}]
        project.learn_corrections = True
        project.archived_at = datetime.now(UTC)
        pid = project.id
        suggestion = LearnedCorrection(
            project_id=pid, match="baz", replace="qux", status="suggested"
        )
        session.add(suggestion)
        session.flush()
        sid = suggestion.id
        session.commit()
    token = mint_csrf_token(client.app.state.csrf_secret, CSRF_PROJECT_LEARNING)
    resp = client.post(
        f"/projects/{pid}/suggestions/{sid}",
        data={"csrf_token": token, "decision": decision},
        follow_redirects=False,
    )
    assert resp.status_code == 409
    with session_factory() as session:
        assert session.get(LearnedCorrection, sid) is not None
        assert session.get(LearnedCorrection, sid).status == "suggested"


# ---- hard delete (#488) -----------------------------------------------------


def test_delete_project_cascade(session_factory: sessionmaker[Session]) -> None:
    """Delete project-owned data while preserving media and unrelated scopes."""
    with session_factory() as session:
        target = _make_project(session, "Target")
        target.archived_at = datetime.now(UTC)
        control = _make_project(session, "Control")
        target_id, control_id = target.id, control.id
        target_folder = MediaFolder(
            path="test/delete/target", project_id=target_id, domain_pack="legal", watch=False
        )
        control_folder = MediaFolder(path="test/delete/control", project_id=control_id)
        correction = LearnedCorrection(
            project_id=target_id, match="foo", replace="bar", status="suggested"
        )
        control_correction = LearnedCorrection(
            project_id=control_id, match="control", replace="untouched", status="suggested"
        )
        session.add_all([target_folder, control_folder, correction, control_correction])
        session.flush()
        folder_id, control_folder_id = target_folder.id, control_folder.id
        correction_id, control_correction_id = correction.id, control_correction.id
        item = MediaItem(source_path="test/delete/target/audio.wav", media_folder_id=folder_id)
        session.add(item)
        session.flush()
        item_id = item.id
        run = PipelineRun(media_item_id=item_id, status="completed")
        session.add(run)
        session.flush()
        run_id = run.id
        segment = TranscriptSegment(
            pipeline_run_id=run_id,
            segment_index=0,
            start_seconds=0,
            end_seconds=1,
            raw_text="foo",
        )
        session.add(segment)
        session.flush()
        segment_id = segment.id
        session.add(
            LearnedCorrectionEvidence(learned_correction_id=correction_id, segment_id=segment_id)
        )
        session.add(
            SavedQuote(
                project_id=target_id,
                segment_id=segment_id,
                run_id=run_id,
                search_query="foo",
                left_context="",
                hit="foo",
                right_context="",
                media_title="Target audio",
                start_seconds=0,
                operator="reviewer",
            )
        )
        for kind in ("project_insights", "temporal_trends"):
            session.add(
                CorpusAnalysisArtifact(
                    scope_kind="project", scope_id=target_id, artifact_kind=kind,
                    generation=1, source_hash="a" * 64, payload={},
                )
            )
        control_artifact = CorpusAnalysisArtifact(
            scope_kind="project", scope_id=control_id, artifact_kind="project_insights",
            generation=1, source_hash="b" * 64, payload={},
        )
        corpus_artifact = CorpusAnalysisArtifact(
            scope_kind="corpus", scope_id=None, artifact_kind="term_stats",
            generation=1, source_hash="c" * 64, payload={},
        )
        session.add_all([control_artifact, corpus_artifact])
        session.flush()
        control_artifact_id, corpus_artifact_id = control_artifact.id, corpus_artifact.id
        session.commit()

    with session_factory() as session:
        delete_project(session, target_id)
        session.commit()

    with session_factory() as session:
        assert session.get(Project, target_id) is None
        assert session.query(LearnedCorrection).filter_by(project_id=target_id).all() == []
        # Evidence is owned by the correction, with no direct project FK.
        assert session.query(LearnedCorrectionEvidence).filter_by(
            learned_correction_id=correction_id
        ).all() == []
        assert session.query(SavedQuote).filter_by(project_id=target_id).all() == []
        assert session.query(CorpusAnalysisArtifact).filter_by(
            scope_kind="project", scope_id=target_id
        ).all() == []
        folder = session.get(MediaFolder, folder_id)
        assert folder is not None
        assert folder.project_id is None
        assert folder.path == "test/delete/target"
        assert folder.domain_pack == "legal"
        assert folder.watch is False
        assert session.get(MediaItem, item_id).media_folder_id == folder_id
        assert session.get(PipelineRun, run_id).media_item_id == item_id
        assert session.get(PipelineRun, run_id).status == "completed"
        assert session.get(TranscriptSegment, segment_id).pipeline_run_id == run_id
        assert session.get(TranscriptSegment, segment_id).raw_text == "foo"

        control = session.get(Project, control_id)
        assert control is not None
        assert control.name == "Control"
        assert control.archived_at is None
        control_folder = session.get(MediaFolder, control_folder_id)
        assert control_folder.project_id == control_id
        assert control_folder.path == "test/delete/control"
        control_correction = session.get(LearnedCorrection, control_correction_id)
        assert control_correction.project_id == control_id
        assert control_correction.match == "control"
        assert control_correction.replace == "untouched"
        assert control_correction.status == "suggested"
        control_artifact = session.get(CorpusAnalysisArtifact, control_artifact_id)
        assert control_artifact.scope_kind == "project"
        assert control_artifact.scope_id == control_id
        assert control_artifact.artifact_kind == "project_insights"
        assert control_artifact.generation == 1
        assert control_artifact.source_hash == "b" * 64
        assert control_artifact.payload == {}
        corpus_artifact = session.get(CorpusAnalysisArtifact, corpus_artifact_id)
        assert corpus_artifact.scope_kind == "corpus"
        assert corpus_artifact.scope_id is None
        assert corpus_artifact.artifact_kind == "term_stats"
        assert corpus_artifact.generation == 1
        assert corpus_artifact.source_hash == "c" * 64
        assert corpus_artifact.payload == {}

        folder.project_id = control_id
        session.commit()

    with session_factory() as session:
        assert session.get(MediaFolder, folder_id).project_id == control_id


def test_delete_active_project_raises(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        pid = _make_project(session, "Active").id
        session.commit()

    with session_factory() as session:
        with pytest.raises(ProjectNotArchivedError):
            delete_project(session, pid)
        session.commit()

    with session_factory() as session:
        project = session.get(Project, pid)
        assert project is not None
        assert project.archived_at is None


def test_delete_missing_project_raises(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session, pytest.raises(ProjectNotFoundError):
        delete_project(session, uuid.uuid4())


def test_delete_project_double_delete(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        project = _make_project(session, "Delete twice")
        project.archived_at = datetime.now(UTC)
        pid = project.id
        session.commit()
        delete_project(session, pid)
        session.commit()

    with session_factory() as session, pytest.raises(ProjectNotFoundError):
        delete_project(session, pid)


def test_delete_project_name_reuse(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        project = _make_project(session, "Reusable Name")
        project.corrections = [{"id": "r1", "match": "foo", "replace": "bar"}]
        project.vocabulary = ["old vocabulary"]
        project.archived_at = datetime.now(UTC)
        pid = project.id
        session.commit()
        delete_project(session, pid)
        session.commit()

    with session_factory() as session:
        project = _make_project(session, "Reusable Name")
        new_id = project.id
        assert new_id != pid
        assert project.corrections is None
        assert project.vocabulary is None
        assert project.learned_corrections == []
        session.commit()

    with session_factory() as session:
        assert session.get(Project, pid) is None
        project = session.get(Project, new_id)
        assert project.name == "Reusable Name"
        assert project.corrections is None
        assert project.vocabulary is None
        assert project.learned_corrections == []


def test_project_fk_inventory(session_factory: sessionmaker[Session]) -> None:
    """Every FK into projects.id must have an explicit lifecycle decision."""
    with session_factory() as session:
        rows = session.execute(text("""
            SELECT fk.table_name, source.column_name, rc.delete_rule
            FROM information_schema.referential_constraints AS rc
            JOIN information_schema.table_constraints AS fk
              ON fk.constraint_catalog = rc.constraint_catalog
             AND fk.constraint_schema = rc.constraint_schema
             AND fk.constraint_name = rc.constraint_name
            JOIN information_schema.constraint_column_usage AS target
              ON target.constraint_catalog = rc.unique_constraint_catalog
             AND target.constraint_schema = rc.unique_constraint_schema
             AND target.constraint_name = rc.unique_constraint_name
            JOIN information_schema.key_column_usage AS source
              ON source.constraint_catalog = fk.constraint_catalog
             AND source.constraint_schema = fk.constraint_schema
             AND source.constraint_name = fk.constraint_name
             AND source.table_schema = fk.table_schema
             AND source.table_name = fk.table_name
            WHERE fk.constraint_type = 'FOREIGN KEY'
              AND target.table_schema = current_schema()
              AND target.table_name = 'projects'
              AND target.column_name = 'id'
        """)).all()
    assert len(rows) == 3
    assert {tuple(row) for row in rows} == {
        ("learned_corrections", "project_id", "CASCADE"),
        ("saved_quotes", "project_id", "CASCADE"),
        ("media_folders", "project_id", "SET NULL"),
    }


# ---- hard delete routes (#488, slice 3) --------------------------------------


def test_delete_confirm_page_shows_counts(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        project = _make_project(session, "Confirm counts")
        project.archived_at = datetime.now(UTC)
        pid = project.id
        for i in range(3):
            session.add(
                LearnedCorrection(
                    project_id=pid, match=f"m{i}", replace=f"r{i}", status="suggested"
                )
            )
        folder = MediaFolder(path="test/delete-confirm/f1", project_id=pid)
        session.add(folder)
        session.flush()
        item = MediaItem(source_path="test/delete-confirm/a.wav", media_folder_id=folder.id)
        session.add(item)
        session.flush()
        run = PipelineRun(media_item_id=item.id, status="completed")
        session.add(run)
        session.flush()
        seg = TranscriptSegment(
            pipeline_run_id=run.id, segment_index=0,
            start_seconds=0, end_seconds=1, raw_text="x",
        )
        session.add(seg)
        session.flush()
        for i in range(2):
            session.add(
                SavedQuote(
                    project_id=pid, segment_id=seg.id, run_id=run.id,
                    search_query=f"q{i}", left_context="", hit="x",
                    right_context="", media_title="a", start_seconds=0,
                    operator="reviewer",
                )
            )
        session.commit()
    page = client.get(f"/projects/{pid}/delete")
    assert page.status_code == 200
    assert "Permanently delete project" in page.text
    assert "Confirm counts" in page.text
    assert "3 learned corrections" in page.text
    assert "2 saved quotes" in page.text
    assert "1 folder" in page.text
    assert "Cancel" in page.text


def test_delete_confirm_active_redirects(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        pid = _make_project(session, "Still active").id
        session.commit()
    resp = client.get(f"/projects/{pid}/delete", follow_redirects=False)
    assert resp.status_code == 303
    assert f"/projects/{pid}" in resp.headers["location"]


def test_delete_confirm_missing_is_404(client: TestClient) -> None:
    resp = client.get(f"/projects/{uuid.uuid4()}/delete")
    assert resp.status_code == 404


def test_delete_confirm_empty_project(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        project = _make_project(session, "Empty archived")
        project.archived_at = datetime.now(UTC)
        pid = project.id
        session.commit()
    page = client.get(f"/projects/{pid}/delete")
    assert page.status_code == 200
    assert "cannot be undone" in page.text
    assert "learned correction" not in page.text
    assert "saved quote" not in page.text


def test_delete_post_invalid_csrf_is_403(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        project = _make_project(session, "Bad csrf")
        project.archived_at = datetime.now(UTC)
        pid = project.id
        session.commit()
    resp = client.post(
        f"/projects/{pid}/delete",
        data={"csrf_token": "wrong"},
        follow_redirects=False,
    )
    assert resp.status_code == 403
    with session_factory() as session:
        assert session.get(Project, pid) is not None


def test_delete_post_missing_csrf_is_403(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        project = _make_project(session, "No csrf")
        project.archived_at = datetime.now(UTC)
        pid = project.id
        session.commit()
    resp = client.post(f"/projects/{pid}/delete", follow_redirects=False)
    assert resp.status_code == 403
    with session_factory() as session:
        assert session.get(Project, pid) is not None


def test_delete_post_cross_action_csrf_is_403(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        project = _make_project(session, "Cross csrf")
        project.archived_at = datetime.now(UTC)
        pid = project.id
        session.commit()
    archive_token = mint_csrf_token(client.app.state.csrf_secret, CSRF_PROJECT_ARCHIVE)
    resp = client.post(
        f"/projects/{pid}/delete",
        data={"csrf_token": archive_token},
        follow_redirects=False,
    )
    assert resp.status_code == 403
    with session_factory() as session:
        assert session.get(Project, pid) is not None


def test_delete_post_success_redirects(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        project = _make_project(session, "Delete me")
        project.archived_at = datetime.now(UTC)
        pid = project.id
        session.commit()
    token = mint_csrf_token(client.app.state.csrf_secret, CSRF_PROJECT_DELETE)
    resp = client.post(
        f"/projects/{pid}/delete",
        data={"csrf_token": token},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/projects"
    with session_factory() as session:
        assert session.get(Project, pid) is None


def test_delete_post_active_is_409(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        pid = _make_project(session, "Not archived yet").id
        session.commit()
    token = mint_csrf_token(client.app.state.csrf_secret, CSRF_PROJECT_DELETE)
    resp = client.post(
        f"/projects/{pid}/delete",
        data={"csrf_token": token},
        follow_redirects=False,
    )
    assert resp.status_code == 409
    with session_factory() as session:
        assert session.get(Project, pid) is not None


def test_delete_post_double_submit_is_404(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        project = _make_project(session, "Double submit")
        project.archived_at = datetime.now(UTC)
        pid = project.id
        session.commit()
    token = mint_csrf_token(client.app.state.csrf_secret, CSRF_PROJECT_DELETE)
    resp1 = client.post(
        f"/projects/{pid}/delete",
        data={"csrf_token": token},
        follow_redirects=False,
    )
    assert resp1.status_code == 303
    token2 = mint_csrf_token(client.app.state.csrf_secret, CSRF_PROJECT_DELETE)
    resp2 = client.post(
        f"/projects/{pid}/delete",
        data={"csrf_token": token2},
        follow_redirects=False,
    )
    assert resp2.status_code == 404


def test_archived_detail_shows_delete_link(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        project = _make_project(session, "Has delete link")
        project.archived_at = datetime.now(UTC)
        pid = project.id
        session.commit()
    page = client.get(f"/projects/{pid}")
    assert f"/projects/{pid}/delete" in page.text
    assert "Delete permanently" in page.text


def test_active_detail_hides_delete_link(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        pid = _make_project(session, "No delete link").id
        session.commit()
    page = client.get(f"/projects/{pid}")
    assert "Delete permanently" not in page.text


def test_delete_name_reuse_via_routes(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    create_token = mint_csrf_token(client.app.state.csrf_secret, CSRF_PROJECT_CREATE)
    resp = client.post(
        "/projects",
        data={"csrf_token": create_token, "name": "Reuse via route"},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    with session_factory() as session:
        project = session.execute(
            sa_select(Project).where(Project.name == "Reuse via route")
        ).scalar_one()
        project.archived_at = datetime.now(UTC)
        pid = project.id
        session.commit()
    delete_token = mint_csrf_token(client.app.state.csrf_secret, CSRF_PROJECT_DELETE)
    resp = client.post(
        f"/projects/{pid}/delete",
        data={"csrf_token": delete_token},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    create_token2 = mint_csrf_token(client.app.state.csrf_secret, CSRF_PROJECT_CREATE)
    resp = client.post(
        "/projects",
        data={"csrf_token": create_token2, "name": "Reuse via route"},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    with session_factory() as session:
        new_project = session.execute(
            sa_select(Project).where(Project.name == "Reuse via route")
        ).scalar_one()
        assert new_project.id != pid
        assert new_project.corrections is None
        assert new_project.vocabulary is None
