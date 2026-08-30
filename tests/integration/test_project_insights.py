"""Project overview insights against the migrated PostgreSQL schema (#336)."""

import re
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.api.project_insights import compute_project_insights
from voxint.api.projects_query import project_detail
from voxint.config import Settings
from voxint.db.models import (
    AdjudicationDecision,
    DiarizationTurn,
    MediaFolder,
    MediaItem,
    PipelineRun,
    Project,
    RunAssetKind,
    RunEnrichmentAsset,
    RunStatus,
    Speaker,
)

CREDS = ("reviewer", "s3cret")
SOURCE_HASH = "a" * 64


@pytest.fixture()
def client(session_factory: sessionmaker[Session], tmp_path: Path) -> TestClient:
    settings = Settings(
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        media_root=tmp_path,
        console_projects_enabled=True,
    )
    test_client = TestClient(
        create_app(settings=settings, session_factory=session_factory)
    )
    test_client.auth = CREDS
    seed_onboarded(session_factory)
    return test_client


def _project_and_folder(
    session: Session, *, name: str = "Project insights", path: str = "project"
) -> tuple[Project, MediaFolder]:
    project = Project(name=name)
    session.add(project)
    session.flush()
    folder = MediaFolder(path=path, project_id=project.id)
    session.add(folder)
    session.flush()
    return project, folder


def _media(
    session: Session,
    folder: MediaFolder,
    path: str,
    *,
    created_at: datetime | None = None,
) -> MediaItem:
    item = MediaItem(
        source_path=path,
        media_folder_id=folder.id,
        created_at=created_at or datetime.now(UTC),
    )
    session.add(item)
    session.flush()
    return item


def _run(
    session: Session,
    media: MediaItem,
    *,
    status: RunStatus = RunStatus.COMPLETED,
    created_at: datetime | None = None,
    archived: bool = False,
) -> PipelineRun:
    run = PipelineRun(
        media_item_id=media.id,
        status=status.value,
        created_at=created_at or datetime.now(UTC),
        archived_at=datetime.now(UTC) if archived else None,
    )
    session.add(run)
    session.flush()
    return run


def _asset(
    session: Session,
    run: PipelineRun,
    kind: RunAssetKind,
    payload: dict[str, object],
    *,
    generation: int = 1,
) -> RunEnrichmentAsset:
    now = datetime.now(UTC)
    asset = RunEnrichmentAsset(
        pipeline_run_id=run.id,
        asset_kind=kind.value,
        generation=generation,
        payload=payload,
        payload_schema_version=1,
        producer="integration-test",
        producer_version="1",
        model="test-model",
        source_content_hash=SOURCE_HASH,
        idempotency_key=str(uuid.uuid4()),
        started_at=now,
        completed_at=now,
    )
    session.add(asset)
    session.flush()
    return asset


def _entity_payload(*surfaces: str) -> dict[str, object]:
    return {
        "mentions": [
            {"surface": surface, "kind": "organization", "occurrences": [{}]}
            for surface in surfaces
        ]
    }


def _assign_speaker(
    session: Session,
    run: PipelineRun,
    speaker: Speaker,
    *,
    label: str,
    turn_index: int = 0,
) -> None:
    session.add(
        DiarizationTurn(
            pipeline_run_id=run.id,
            turn_index=turn_index,
            start_seconds=0.0,
            end_seconds=5.0,
            label=label,
            skip_reason="test-seed",
        )
    )
    session.add(
        AdjudicationDecision(
            pipeline_run_id=run.id,
            diarization_label=label,
            decision="assign",
            speaker_id=speaker.id,
            operator="reviewer",
            idempotency_key=str(uuid.uuid4()),
        )
    )


def test_compute_uses_only_current_assets_on_canonical_project_runs(
    session_factory: sessionmaker[Session],
) -> None:
    now = datetime.now(UTC)
    with session_factory() as session:
        project, folder = _project_and_folder(session)
        first = _media(session, folder, "project/first.wav")
        second = _media(session, folder, "project/second.wav")
        old_run = _run(session, first, created_at=now - timedelta(days=2))
        current_run = _run(session, first, created_at=now - timedelta(days=1))
        _run(session, second, created_at=now)

        _asset(
            session,
            old_run,
            RunAssetKind.ENTITY_MENTIONS,
            _entity_payload("StaleCorp"),
        )
        superseded = _asset(
            session,
            current_run,
            RunAssetKind.ENTITY_MENTIONS,
            _entity_payload("SupersededCorp"),
            generation=1,
        )
        current = _asset(
            session,
            current_run,
            RunAssetKind.ENTITY_MENTIONS,
            _entity_payload("Acme Corp"),
            generation=2,
        )
        superseded.superseded_by_asset_id = current.id
        _asset(
            session,
            current_run,
            RunAssetKind.TOPICS,
            {"topics": [{"label": "Heat pumps", "confidence": 0.9}]},
        )

        _, outside_folder = _project_and_folder(
            session, name="Outside", path="outside"
        )
        outside_run = _run(
            session, _media(session, outside_folder, "outside/leak.wav")
        )
        _asset(
            session,
            outside_run,
            RunAssetKind.ENTITY_MENTIONS,
            _entity_payload("OutsideCorp"),
        )
        archived_run = _run(
            session,
            _media(session, folder, "project/archived.wav"),
            archived=True,
        )
        _asset(
            session,
            archived_run,
            RunAssetKind.ENTITY_MENTIONS,
            _entity_payload("ArchivedCorp"),
        )
        queued_run = _run(
            session,
            _media(session, folder, "project/queued.wav"),
            status=RunStatus.QUEUED,
        )
        _asset(
            session,
            queued_run,
            RunAssetKind.ENTITY_MENTIONS,
            _entity_payload("QueuedCorp"),
        )
        session.commit()

        insights = compute_project_insights(session, project.id)

    assert [entity.label for entity in insights.entities["organization"]] == [
        "Acme Corp"
    ]
    assert [topic.label for topic in insights.topics] == ["Heat pumps"]
    assert insights.coverage.entity_runs == 1
    assert insights.coverage.topic_runs == 1
    assert insights.coverage.total_runs == 2


def test_detail_page_renders_links_coverage_and_escaped_values(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    now = datetime.now(UTC)
    with session_factory() as session:
        project, folder = _project_and_folder(session)
        enriched = _media(session, folder, "project/enriched.wav", created_at=now)
        hostile = _media(
            session,
            folder,
            "project/<script>alert(1)</script>.wav",
            created_at=now + timedelta(seconds=1),
        )
        enriched_run = _run(session, enriched)
        hostile_run = _run(session, hostile)
        _asset(
            session,
            enriched_run,
            RunAssetKind.ENTITY_MENTIONS,
            _entity_payload("Acme Corp", "<b>Evil</b>"),
        )
        speaker = Speaker(display_name="Presenter")
        session.add(speaker)
        session.flush()
        _assign_speaker(session, hostile_run, speaker, label="SPEAKER_00")
        project_id = project.id
        session.commit()

    response = client.get(f"/projects/{project_id}")

    assert response.status_code == 200
    assert "Acme Corp" in response.text
    assert f"/explore?project={project_id}&amp;q=Acme%20Corp" in response.text
    assert "Entity enrichment covers 1 of 2 recordings" in response.text
    assert "<b>Evil</b>" not in response.text
    assert "&lt;b&gt;Evil&lt;/b&gt;" in response.text
    assert "<script>alert(1)</script>.wav" not in response.text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;.wav" in response.text


def test_detail_page_empty_when_no_completed_recordings(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        project, _ = _project_and_folder(session)
        project_id = project.id
        session.commit()

    response = client.get(f"/projects/{project_id}")

    assert response.status_code == 200
    assert "recordings finish processing" in response.text


def test_detail_page_explains_when_run_assets_are_absent(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        project, folder = _project_and_folder(session)
        _run(session, _media(session, folder, "project/recording.wav"))
        project_id = project.id
        session.commit()

    response = client.get(f"/projects/{project_id}")

    assert response.status_code == 200
    assert "hasn't produced topics or entity mentions for this project yet" in response.text
    assert 'href="/settings#features"' in response.text


def test_detail_page_distinguishes_assets_with_empty_results(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    # The writer can legitimately record an entity asset whose model reply
    # offered no mentions at all (an authoritative "no entities"); an empty
    # topics payload is writer-unreachable, so none is seeded.
    with session_factory() as session:
        project, folder = _project_and_folder(session)
        run = _run(session, _media(session, folder, "project/recording.wav"))
        _asset(
            session,
            run,
            RunAssetKind.ENTITY_MENTIONS,
            {
                "mentions": [],
                "diagnostics": {"dropped_unlocatable": 0, "dropped_out_of_run": 0},
            },
        )
        project_id = project.id
        session.commit()

    response = client.get(f"/projects/{project_id}")

    assert response.status_code == 200
    assert "found no topics" in response.text
    assert "entity mentions to report" in response.text


def test_entity_bar_widths_never_exceed_the_track(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    """Rows rank by recording count, but bar widths scale to the group's max
    occurrence count — a lower-ranked entity with more occurrences must not
    overflow the track."""
    now = datetime.now(UTC)
    with session_factory() as session:
        project, folder = _project_and_folder(session)
        first_run = _run(
            session, _media(session, folder, "project/first.wav", created_at=now)
        )
        second_run = _run(
            session,
            _media(
                session,
                folder,
                "project/second.wav",
                created_at=now + timedelta(seconds=1),
            ),
        )
        # "Frequent Corp" leads by run_count (2 runs, 1 occurrence each);
        # "Loud Corp" has one run but five occurrences.
        _asset(
            session, first_run, RunAssetKind.ENTITY_MENTIONS, _entity_payload("Frequent Corp")
        )
        _asset(
            session,
            second_run,
            RunAssetKind.ENTITY_MENTIONS,
            {
                "mentions": [
                    {"surface": "Frequent Corp", "kind": "organization", "occurrences": [{}]},
                    {
                        "surface": "Loud Corp",
                        "kind": "organization",
                        "occurrences": [{}, {}, {}, {}, {}],
                    },
                ]
            },
        )
        project_id = project.id
        session.commit()

    response = client.get(f"/projects/{project_id}")

    assert response.status_code == 200
    widths = [
        int(match)
        for match in re.findall(r"width: (\d+)%", response.text)
    ]
    assert widths, "expected entity bars to render"
    assert max(widths) == 100
    assert all(width <= 100 for width in widths)


def test_speaker_only_in_historical_rerun_is_not_listed(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    """Membership follows canonical runs: a speaker assigned only in an older,
    superseded run of a media item is absent from the roster and the matrix
    (the newest completed run is the operative transcript)."""
    now = datetime.now(UTC)
    with session_factory() as session:
        project, folder = _project_and_folder(session)
        media = _media(session, folder, "project/recording.wav")
        historical = _run(session, media, created_at=now - timedelta(days=1))
        canonical = _run(session, media, created_at=now)
        ghost = Speaker(display_name="Ghost")
        current = Speaker(display_name="Current")
        session.add_all([ghost, current])
        session.flush()
        _assign_speaker(session, historical, ghost, label="SPEAKER_00")
        _assign_speaker(session, canonical, current, label="SPEAKER_00")
        project_id = project.id
        session.commit()

        detail = project_detail(session, project_id)

    assert detail is not None
    assert [speaker.name for speaker in detail.speakers] == ["Current"]
    assert [row.name for row in detail.matrix.rows] == ["Current"]


def test_matrix_is_canonical_newest_first_and_counts_recordings_once(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    now = datetime.now(UTC)
    with session_factory() as session:
        project, folder = _project_and_folder(session)
        older_media = _media(
            session,
            folder,
            "project/Older Recording.wav",
            created_at=now - timedelta(days=2),
        )
        newer_media = _media(
            session,
            folder,
            "project/Newest Recording.wav",
            created_at=now - timedelta(days=1),
        )
        historical = _run(
            session, older_media, created_at=now - timedelta(days=4)
        )
        canonical_older = _run(
            session, older_media, created_at=now - timedelta(days=3)
        )
        canonical_newer = _run(
            session, newer_media, created_at=now - timedelta(days=1)
        )
        alice = Speaker(display_name="Alice")
        bob = Speaker(display_name="Bob")
        session.add_all([alice, bob])
        session.flush()
        _assign_speaker(session, historical, alice, label="SPEAKER_00")
        _assign_speaker(session, canonical_older, alice, label="SPEAKER_00")
        _assign_speaker(session, canonical_newer, bob, label="SPEAKER_01")
        project_id, alice_id, bob_id = project.id, alice.id, bob.id
        session.commit()

        detail = project_detail(session, project_id)

    assert detail is not None
    assert [column.title for column in detail.matrix.columns] == [
        "project/Newest Recording.wav",
        "project/Older Recording.wav",
    ]
    assert {row.name: row.cells for row in detail.matrix.rows} == {
        "Alice": (False, True),
        "Bob": (True, False),
    }
    assert {speaker.name: speaker.run_count for speaker in detail.speakers} == {
        "Alice": 1,
        "Bob": 1,
    }

    response = client.get(f"/projects/{project_id}")
    assert response.status_code == 200
    assert f'href="/speakers/{alice_id}"' in response.text
    assert f'href="/speakers/{bob_id}"' in response.text
    assert response.text.index("Newest Recording.wav") < response.text.index(
        "Older Recording.wav"
    )
    assert response.text.count(">●<") == 2
